#!/usr/bin/env python
"""Normalize a pyDeltaRCM YAML config into a committed benchmark fixture.

One normalization pass feeds every implementation in the comparison rig
(Python reference, C and Rust ports). The normalizer instantiates a
:class:`pyDeltaRCM.DeltaModel` from a YAML config, then writes a
self-contained fixture directory::

    <out>/
        config.yaml     # verbatim copy of the source config (out_dir excluded)
        config.json     # flat scalar state (user + derived) for native ports
        manifest.json   # per-array dtype/shape/role table + RNG provenance
        state/*.npy     # uncompressed float32/int64 arrays in reference dtype

Arrays are dumped exactly as the model holds them post-init (uncompressed
``.npy``), so native ports can start from byte-identical state without
re-implementing :class:`DeltaModel` init. The free-surface temporaries the
reference computes *inside* ``update()`` are float64; they are internal to
the update loop and are noted in the manifest rather than dumped -- the
fixture carries the float32 state fields and init-derived routing constants.

The RNG stream position at fixture time is "freshly seeded": init consumes
no draws, which the golden eta gates verify end-to-end.

Usage::

    python benchmarks/normalize_fixture.py \
        --config bench/configs/small.yaml --out bench/fixture_default_small

Determinism: two normalizations of the same config produce byte-identical
``state/`` arrays and ``manifest.json`` (``config.json`` ``_meta`` carries
wall-clock provenance and is excluded from byte comparisons).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import pyDeltaRCM
from pyDeltaRCM import DeltaModel

# Attributes excluded from config.json: environment-specific paths and
# lifecycle counters that describe this normalization run, not the model
# physics a port must reproduce.
VOLATILE_SCALARS = frozenset(
    {
        "prefix",
        "prefix_abspath",
        "_checkpoint_folder",
        "_out_dir",
        "default_file",
        "input_file",
        "_is_finalized",
        "_resume_checkpoint",
        "_time",
        "_time_iter",
        "_save_time_since_checkpoint",
        "_save_time_since_data",
        "__pyDeltaRCM_version__",  # recorded in _meta instead
    }
)

# Buffers the update loop zeroes and refills from scratch at the start of
# every water iteration (water_tools.py:61-62). They carry no state between
# updates, so a fixture dump is dead weight -- on the default grid the walk
# buffer alone is ~5 MB of zeros. Ports allocate them as zeros; the manifest
# records their shapes.
SCRATCH_BUFFERS = frozenset(
    {
        "free_surf_flag",
        "free_surf_walk_inds",
    }
)

# Arrays whose bytes constitute the restore-state parity surface (the
# checkpoint-format arrays plus masks and accumulators the update loop
# reads). Everything else in state/ is an init-derived routing constant.
STATE_ROLE_ARRAYS = frozenset(
    {
        "eta",
        "eta0",
        "eta_init",
        "stage",
        "depth",
        "qw",
        "qx",
        "qy",
        "uw",
        "ux",
        "uy",
        "sand_frac",
        "active_layer",
        "qs",
        "cell_type",
        "inlet",
        "Vp_dep_sand",
        "Vp_dep_mud",
        "sfc_visit",
        "sfc_sum",
        "free_surf_flag",
        "free_surf_walk_inds",
    }
)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_scalars(model: DeltaModel) -> dict:
    """Flat dict of the model's scalar state, excluding volatile attributes.

    Keys are the model's own attribute names (including the leading
    underscore) so ports can mirror the reference without translation.
    """
    scalars = {}
    for name, value in sorted(vars(model).items()):
        if name in VOLATILE_SCALARS:
            continue
        if isinstance(value, (bool, str)):
            scalars[name] = value
        elif isinstance(value, int):
            scalars[name] = int(value)
        elif isinstance(value, float):
            scalars[name] = float(value)
        elif isinstance(value, np.integer):
            scalars[name] = int(value)
        elif isinstance(value, np.floating):
            scalars[name] = float(value)
        elif value is None:
            scalars[name] = None
    return scalars


def collect_arrays(model: DeltaModel) -> dict[str, np.ndarray]:
    """Post-init ndarray attributes, name -> array, minus scratch buffers."""
    return {
        name: value
        for name, value in sorted(vars(model).items())
        if isinstance(value, np.ndarray) and name not in SCRATCH_BUFFERS
    }


def init_model_from_config(config_path: Path) -> tuple[DeltaModel, Path]:
    """Init a DeltaModel from ``config_path`` with out_dir in a temp dir.

    The fixture's committed ``config.yaml`` carries no ``out_dir``; the
    normalizer injects a throwaway directory so normalization never writes
    into the repository. Returns the model and the temp dir (caller cleans
    up).
    """
    with open(config_path) as f:
        cfg = yaml_safe_load(f.read())
    if "out_dir" in cfg:
        raise ValueError(
            "fixture configs must not set out_dir; the normalizer and the "
            "reference runner inject throwaway directories"
        )
    tmp_dir = Path(tempfile.mkdtemp(prefix="pydeltarcm_normalize_"))
    cfg["out_dir"] = str(tmp_dir / "out_dir")
    model = DeltaModel(**cfg)
    return model, tmp_dir


def yaml_safe_load(text: str) -> dict:
    """Minimal YAML load without a hard pyyaml dependency at call time."""
    import yaml  # a pyDeltaRCM runtime requirement, already installed

    return yaml.safe_load(text)


def build_manifest(
    model: DeltaModel,
    arrays: dict[str, np.ndarray],
    state_files: dict[str, str],
    source_sha256: str,
    digests: dict[str, str],
) -> dict:
    """Manifest describing every dumped array and the fixture's contract."""
    entries = {}
    for name, arr in arrays.items():
        entry = {
            "file": state_files[name],
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "role": "state" if name in STATE_ROLE_ARRAYS else "init_derived",
            "sha256": digests[name],
        }
        entries[name] = entry
    scratch = {
        name: {
            "dtype": str(getattr(model, name).dtype),
            "shape": list(getattr(model, name).shape),
            "note": (
                "zeroed and refilled at the start of every water iteration "
                "(water_tools.py); excluded from the fixture -- allocate zeros"
            ),
        }
        for name in sorted(SCRATCH_BUFFERS)
    }
    return {
        "format": "pydeltarcm-benchmark-fixture/1",
        "source_config_sha256": source_sha256,
        "arrays": entries,
        "contract": {
            "encoding": "uncompressed .npy (NumPy array format v1.0)",
            "state_fields": "float32 as the reference computes in",
            "index_grids": "int64 (cell_type, inlet, walk constants)",
            "coordinate_vectors": "float64 (xc, yc, X, Y) as held post-init",
            "free_surface_note": (
                "free-surface temporaries inside the reference's update() "
                "are float64; they are internal to the update loop and not "
                "part of the fixture"
            ),
        },
        "excluded_scratch_buffers": scratch,
        "rng": {
            "stream": "NumPy legacy MT19937 (Numba njit-verified bit-identical)",
            "position_at_fixture": "freshly seeded; init consumes no draws",
            "anchor": (
                "seed(0) first uniform == 0.5488135039273248 "
                "(tests/test_shared_tools.py)"
            ),
            "seed_documented_in": "config.json _seed",
        },
    }


def normalize(config_path: Path, out_dir: Path, fixture_name: str) -> dict:
    """Normalize one YAML config into ``out_dir``; return the config.json dict."""
    config_path = Path(config_path).resolve()
    out_dir = Path(out_dir)

    # golden.json is curated (pinned spec values), not derived from a model
    # run -- preserve it across regeneration, which wipes the directory so
    # arrays dropped from the catalog cannot linger as stale fixture bytes.
    golden_path = out_dir / "golden.json"
    golden_bytes = golden_path.read_bytes() if golden_path.is_file() else None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = sha256_file(config_path)
    shutil.copyfile(config_path, out_dir / "config.yaml")

    tmp_dir = None
    try:
        model, tmp_dir = init_model_from_config(config_path)
        arrays = collect_arrays(model)
        scalars = collect_scalars(model)

        state_files = {}
        digests = {}
        for name, arr in arrays.items():
            fname = f"state/{name}.npy"
            np.save(out_dir / fname, arr, allow_pickle=False)
            state_files[name] = fname
            digests[name] = sha256_file(out_dir / fname)

        config_json = {
            "model": "DeltaRCM",
            "fixture": fixture_name,
            **scalars,
            "arrays": state_files,
            "_meta": {
                "normalized_from": config_path.name,
                "source_config_sha256": source_sha256,
                "pyDeltaRCM_version": pyDeltaRCM.__version__,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "normalized_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            },
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_json, f, indent=2, sort_keys=True)
            f.write("\n")

        manifest = build_manifest(
            model, arrays, state_files, source_sha256, digests
        )
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # Restore curated files even when normalization fails mid-run: the
        # directory was wiped up front, so the golden must never depend on
        # the success path for its survival.
        if golden_bytes is not None:
            golden_path.write_bytes(golden_bytes)

    return {
        "config": config_json,
        "manifest": manifest,
        "digests": digests,
        "golden_preserved": golden_bytes is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="source YAML config for the fixture"
    )
    parser.add_argument(
        "--out", required=True, help="fixture directory to write (created)"
    )
    parser.add_argument(
        "--name", required=True, help="fixture name recorded in config.json"
    )
    args = parser.parse_args(argv)

    result = normalize(Path(args.config), Path(args.out), args.name)
    print(f"fixture written: {args.out}")
    print(f"  arrays: {len(result['digests'])}")
    if result["golden_preserved"]:
        print("  golden.json preserved across regeneration")
    for name, digest in sorted(result["digests"].items()):
        print(f"  {name}: {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
