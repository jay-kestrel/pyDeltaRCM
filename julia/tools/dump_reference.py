#!/usr/bin/env python
"""Dump Python-reference water-routing fixtures for the Julia equivalence suite.

Runs the reference (numba) implementation of the water-routing vertical slice
on the committed bench fixture and writes bit-exact input/output pairs the
Julia port is tested against.

How the reference is run
------------------------
The driver loads ``pyDeltaRCM/shared_tools.py`` and ``pyDeltaRCM/water_tools.py``
directly as standalone modules and runs the exact phase sequence of
:meth:`water_tools.route_water` with the fused native phase disabled::

    for iteration in range(itermax):
        init_water_iteration()
        run_water_iteration()
        compute_free_surface()
        finalize_water_iteration(iteration)

This is the code path ``route_water`` itself takes when
``pyDeltaRCM.native_water.can_use_native`` returns ``False`` (the fused Rust
phase is not built in this environment, and ``PYDELTARCM_NATIVE=0`` forces the
same branch). The only behavior skipped is ``route_water``'s no-op logging and
no-op base-class hooks, plus its unconditional import of
``pyDeltaRCM.native_water`` -- importing that module pulls in the package
``__init__``, which requires netCDF4 for NetCDF I/O the numeric path never
touches. The model state is hydrated from the committed
``bench/fixture_default_small`` fixture (post-init arrays + derived scalars),
which is the fork's blessed handoff format: native ports "start from
byte-identical state without re-implementing DeltaModel init". No Python
package source is modified.

Determinism
-----------
The numba MT19937 stream is seeded once with the fixture's ``_seed`` (init
consumes no draws, so this reproduces a freshly constructed DeltaModel) and
the packed RNG state is dumped before/after every update, so the Julia suite
can assert bit-exact RNG trajectory, not just field outputs.

Usage::

    python julia/tools/dump_reference.py \
        --fixture bench/fixture_default_small \
        --out julia/test/fixtures/reference --updates 2
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

os.environ["PYDELTARCM_NATIVE"] = "0"

REPO_ROOT = Path(__file__).resolve().parents[2]

# post-init fields the water phase reads or mutates (fixture names)
STATE_ARRAYS = (
    "eta",
    "stage",
    "depth",
    "qw",
    "qx",
    "qy",
    "qwn",
    "qxn",
    "qyn",
    "uw",
    "ux",
    "uy",
    "cell_type",
    "mod_water_weight",
    "inlet",
    "iwalk_flat",
    "jwalk_flat",
    "ravel_walk_flat",
    "ivec_flat",
    "jvec_flat",
    "distances_flat",
)

# fields dumped after every reference update (Julia compares each bit-exact)
OUTPUT_FIELDS = (
    "qxn",
    "qyn",
    "qwn",
    "qx",
    "qy",
    "qw",
    "stage",
    "depth",
    "uw",
    "ux",
    "uy",
    "free_surf_flag",
    "free_surf_walk_inds",
    "sfc_visit",
    "sfc_sum",
    "Hnew",
)

SCALAR_KEYS = (
    "L",
    "W",
    "L0",
    "CTR",
    "_itermax",
    "_Np_water",
    "_dx",
    "Qp_water",
    "_H_SL",
    "_omega_sfc",
    "_omega_flow",
    "omega_flow_iter",
    "_Nsmooth",
    "_Csmooth",
    "_u0",
    "_S0",
    "_gamma",
    "_theta_water",
    "u_max",
    "dry_depth",
    "qw0",
    "size_indices",
    "_seed",
)

# config.json key -> model attribute name. The water_tools mixin reads some
# attributes without the leading underscore (`stepmax`, `h0`) that config.json
# stores in its underscore form (the DeltaModel exposes both spellings).
SCALAR_ALIASES = {"_stepmax": "stepmax", "_h0": "h0", "_H_SL": "H_SL"}


def load_reference_modules():
    """Load the real shared_tools/water_tools without the package __init__.

    Registers an empty ``pyDeltaRCM`` module whose ``__path__`` points at the
    package directory, so ``import pyDeltaRCM.shared_tools`` resolves the real
    file while the package ``__init__`` (which imports netCDF4 for NetCDF
    output) never executes. The loaded modules are the unmodified sources.
    """
    pkg_dir = REPO_ROOT / "pyDeltaRCM"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    shim = types.ModuleType("pyDeltaRCM")
    shim.__path__ = [str(pkg_dir)]
    sys.modules["pyDeltaRCM"] = shim
    shared_tools = importlib.import_module("pyDeltaRCM.shared_tools")
    water_tools = importlib.import_module("pyDeltaRCM.water_tools")
    return shared_tools, water_tools


def hydrate_model(shared_tools, water_tools, fixture_dir: Path):
    """Build a model object exposing exactly what the water phase methods read."""
    cfg = json.loads((fixture_dir / "config.json").read_text())

    class WaterSlice(water_tools.water_tools):
        """Minimal DeltaModel stand-in for the water phase (no I/O, no hooks)."""

        def __init__(self):
            self.log_info = lambda *args, **kwargs: None
            for key in SCALAR_KEYS:
                setattr(self, key, cfg[key])
            for key, attr in SCALAR_ALIASES.items():
                setattr(self, attr, cfg[key])
            self._time_iter = 0  # volatile scalar: 0 post-init (config omits it)
            for name in STATE_ARRAYS:
                setattr(self, name, np.load(fixture_dir / "state" / f"{name}.npy"))
            # scratch buffers: zeroed and refilled every water iteration; the
            # fixture manifest says ports allocate zeros at the recorded shapes
            self.free_surf_flag = np.zeros((self._Np_water,), dtype=np.int64)
            self.free_surf_walk_inds = np.zeros(
                (self._Np_water, self.size_indices), dtype=np.int64
            )
            self.sfc_visit = np.zeros_like(self.depth)
            self.sfc_sum = np.zeros_like(self.depth)

    return WaterSlice(), cfg


def dump_rng(st, out_dir: Path, name: str) -> None:
    packed = st.pack_rng_state()
    np.save(out_dir / f"{name}_index.npy", np.int64(packed["rng_state_index"]))
    np.save(out_dir / f"{name}_key.npy", packed["rng_state_key"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default=str(REPO_ROOT / "bench" / "fixture_default_small"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "julia" / "test" / "fixtures" / "reference"),
    )
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()

    fixture_dir = Path(args.fixture)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    shared_tools, water_tools = load_reference_modules()
    model, cfg = hydrate_model(shared_tools, water_tools, fixture_dir)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    shared_tools.set_random_seed(int(cfg["_seed"]))

    scalars = {
        "fixture": fixture_dir.name,
        "reference_git_sha": git_sha,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "numba_version": __import__("numba").__version__,
        "rng_name": "MT19937",
        "rng_key_length": 624,
        "native_phase": False,
        "updates": [],
    }

    import numba  # after import so the version probe above stays cheap

    for update in range(1, args.updates + 1):
        if update > 1:
            # mirrors DeltaModel.update(): _time_iter increments per completed
            # update, switching on the _time_iter > 0 branches (omega_sfc
            # blend, flow underrelaxation)
            model._time_iter += 1

        upd_dir = out_root / f"update{update}"
        upd_dir.mkdir(parents=True, exist_ok=True)
        dump_rng(shared_tools, upd_dir, "rng_before")

        for iteration in range(int(model._itermax)):
            model.init_water_iteration()
            np.save(
                upd_dir / f"start_indices_iter{iteration}.npy", model.start_indices
            )
            model.run_water_iteration()
            np.save(
                upd_dir / f"water_weights_iter{iteration}.npy", model.water_weights
            )
            model.compute_free_surface()
            model.finalize_water_iteration(iteration)

        dump_rng(shared_tools, upd_dir, "rng_after")
        for name in OUTPUT_FIELDS:
            np.save(upd_dir / f"{name}.npy", getattr(model, name))
        # isolated _smooth_free_surface I/O pair (deterministic recomputation)
        np.save(
            upd_dir / "Hsmth.npy",
            water_tools._smooth_free_surface(
                model.Hnew, model.cell_type, model._Nsmooth, model._Csmooth
            ),
        )

        entry = {
            "time_iter": int(model._time_iter),
            "walk_columns": int(model.free_surf_walk_inds.shape[1]),
        }
        scalars["updates"].append(entry)
        print(
            f"update {update}: dumped {len(OUTPUT_FIELDS) + 3} arrays "
            f"(time_iter={model._time_iter})"
        )

    (out_root / "scalars.json").write_text(json.dumps(scalars, indent=2) + "\n")

    # compact sanity summary so a degenerate fixture cannot slip through
    print("=== sanity summary ===")
    for name in ("qwn", "stage", "depth", "uw"):
        arr = getattr(model, name)
        print(
            f"  {name}: min={arr.min():.6g} max={arr.max():.6g} "
            f"nonzero={int(np.count_nonzero(arr))}"
        )
    print(f"  free_surf_flag nonzero: {int(np.count_nonzero(model.free_surf_flag))}")
    print(f"  walk steps recorded: {int(np.count_nonzero(model.free_surf_walk_inds))}")
    print(f"  git sha: {git_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
