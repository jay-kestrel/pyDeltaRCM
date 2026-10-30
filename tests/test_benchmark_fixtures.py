"""Contract tests for the benchmark fixtures (bench/fixture_*).

Covers what the task asks of the normalizer's output: manifest
completeness, config consistency, golden checkpoint provenance, and
fixture stability (a fresh normalization must be byte-identical to the
committed fixture).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "benchmarks"
FIXTURE_DIR = REPO_ROOT / "bench" / "fixture_default_small"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(FIXTURE_DIR / "manifest.json") as f:
        return json.load(f)


def test_manifest_lists_every_state_file(manifest: dict) -> None:
    """No orphans: every state/*.npy is in the manifest, every manifest
    file exists."""
    committed = {
        str(p.relative_to(FIXTURE_DIR))
        for p in (FIXTURE_DIR / "state").glob("*.npy")
    }
    listed = {entry["file"] for entry in manifest["arrays"].values()}
    assert listed == committed
    for entry in manifest["arrays"].values():
        assert (FIXTURE_DIR / entry["file"]).is_file()


def test_manifest_entries_carry_dtype_shape_role(manifest: dict) -> None:
    for name, entry in manifest["arrays"].items():
        assert "dtype" in entry and "shape" in entry, name
        assert entry["role"] in ("state", "init_derived"), name


def test_scratch_buffers_documented_not_dumped(manifest: dict) -> None:
    """Scratch buffers (zeroed every water iteration) are excluded from
    state/ and recorded with shapes so ports can allocate zeros. Shapes
    pinned per fixture: the small grid runs Np_water=10, stepmax=40."""
    scratch = manifest["excluded_scratch_buffers"]
    assert set(scratch) == {"free_surf_flag", "free_surf_walk_inds"}
    assert scratch["free_surf_flag"]["shape"] == [10]
    assert scratch["free_surf_walk_inds"]["shape"] == [10, 20]
    for name in scratch:
        assert not (FIXTURE_DIR / "state" / f"{name}.npy").exists()


def test_fixture_config_json_covers_yaml_and_derived() -> None:
    """config.json carries every post-init scalar the ports need: each
    user-facing yaml key appears under its model-attribute name (underscore
    prefix) with the same value, alongside derived params the kernels
    consume and the _meta provenance block."""
    with open(FIXTURE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(FIXTURE_DIR / "config.json") as f:
        cfg_json = json.load(f)
    for key, value in cfg.items():
        attr = f"_{key}"
        assert attr in cfg_json, f"yaml key {key} missing from config.json"
        assert cfg_json[attr] == value, key
    for derived in ("u_max", "theta_sand", "size_indices", "_stepmax"):
        assert derived in cfg_json, derived
    assert cfg_json["_meta"]["normalized_from"] == "default_small.yaml"
    assert set(cfg_json["arrays"]) == set(
        json.load(open(FIXTURE_DIR / "manifest.json"))["arrays"]
    )


def test_golden_checkpoints_match_spec() -> None:
    """The committed goldens are exactly the spec's pinned eta values
    (from tests/integration/test_consistent_outputs.py)."""
    with open(FIXTURE_DIR / "golden.json") as f:
        golden = json.load(f)
    assert golden["checkpoints"]["1"] == [-1.0, -0.9152762, -1.0004134, -1.0, -1.0]
    assert golden["checkpoints"]["10"] == [1.7, 0.394864, -0.95006764, -1.0, -1.0]
    assert golden["slice"] == "eta[:5, 4]"


def test_default_fixture_present() -> None:
    """The default-grid fixture exists with the same directory contract."""
    d = REPO_ROOT / "bench" / "fixture_default"
    assert (d / "manifest.json").is_file()
    assert (d / "config.json").is_file()
    with open(d / "manifest.json") as f:
        manifest = json.load(f)
    committed = {
        str(p.relative_to(d)) for p in (d / "state").glob("*.npy")
    }
    listed = {entry["file"] for entry in manifest["arrays"].values()}
    assert listed == committed
    assert manifest["excluded_scratch_buffers"]["free_surf_walk_inds"][
        "shape"
    ] == [2000, 300]


def test_fixture_state_arrays_load_clean(manifest: dict) -> None:
    """Every committed array loads, byte-exact, with recorded dtype/shape."""
    for entry in manifest["arrays"].values():
        arr = np.load(FIXTURE_DIR / entry["file"])
        assert str(arr.dtype) == entry["dtype"], entry["file"]
        assert list(arr.shape) == entry["shape"], entry["file"]


def test_fixture_determinism(tmp_path: Path, manifest: dict) -> None:
    """Regenerating the small fixture reproduces the committed bytes."""
    out = tmp_path / "fixture_check"
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH / "normalize_fixture.py"),
            "--config",
            str(FIXTURE_DIR / "config.yaml"),
            "--out",
            str(out),
            "--name",
            "fixture_check",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    with open(out / "manifest.json") as f:
        fresh = json.load(f)
    fresh_arrays = {
        name: entry["sha256"] for name, entry in fresh["arrays"].items()
    }
    committed_arrays = {
        name: entry["sha256"] for name, entry in manifest["arrays"].items()
    }
    assert fresh_arrays == committed_arrays
