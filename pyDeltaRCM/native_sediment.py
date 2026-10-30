"""One native call per sediment phase; Python retains hooks and router state."""

import importlib
import importlib.util
import os
import warnings

import numpy as np

from pyDeltaRCM import shared_tools

FLOAT_FIELDS = (
    "eta",
    "stage",
    "depth",
    "uw",
    "ux",
    "uy",
    "Vp_dep_mud",
    "Vp_dep_sand",
    "qw",
    "qx",
    "qy",
    "mod_erosion",
    "mod_sed_weight",
    "mod_stable_weight",
    "ivec_flat",
    "jvec_flat",
    "distances_flat",
)


def can_use_native(router, sand: bool) -> bool:
    """Keep custom router methods, precision, and geometry on the reference path."""
    if os.environ.get("PYDELTARCM_NATIVE", "1") == "0":
        return False
    from pyDeltaRCM.sed_tools import MudRouter, SandRouter

    expected = SandRouter if sand else MudRouter
    for name in (
        "run",
        "_route_one_parcel",
        "_choose_next_location",
        "_deposit_or_erode",
        "_update_fields",
        "_compute_Vp_ero",
        "_limit_Vp_change",
    ) + (("_partition_sediment",) if sand else ()):
        if getattr(getattr(router, name), "__func__", None) is not getattr(
            expected, name
        ):
            return False
    fields = FLOAT_FIELDS + (("qs",) if sand else ())
    if any(getattr(router, name).dtype != np.float32 for name in fields):
        return False
    if not np.array_equal(router.iwalk_flat, [-1, 0, 1, -1, 0, 1, -1, 0, 1]):
        return False
    if not np.array_equal(router.jwalk_flat, [-1, -1, -1, 0, 0, 0, 1, 1, 1]):
        return False
    if importlib.util.find_spec("pyDeltaRCM._native") is None:
        warnings.warn(
            "Rust sediment extension is not installed; using Python. "
            "Build with pip install . or set PYDELTARCM_NATIVE=0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def phase_inputs(router, starts: np.ndarray, sand: bool) -> tuple:
    """Snapshot typed arrays and configuration without changing the router."""
    names = FLOAT_FIELDS + ("cell_type",) + (("qs",) if sand else ())
    arrays = {name: np.ascontiguousarray(getattr(router, name)) for name in names}
    config = {
        "dt": float(router._dt),
        "dx": float(router._dx),
        "volume": float(router.Vp_sed),
        "u_max": float(router.u_max),
        "dry_depth": float(router.dry_depth),
        "beta": float(router._beta),
        "theta": float(router.theta_sed),
        "stepmax": int(router.stepmax),
        "force_deposit": bool(router.force_deposit),
        "sand": sand,
        "qs0": float(router.qs0) if sand else 0.0,
        "u0": float(router._u0) if sand else 1.0,
        "bedload": float(router._f_bedload) if sand else 0.0,
        "erosion_velocity": float(router.U_ero_sand if sand else router.U_ero_mud),
        "deposition_velocity": 0.0 if sand else float(router.U_dep_mud),
        "settling": 0.0 if sand else float(router._lambda),
    }
    return arrays, config, np.ascontiguousarray(starts, dtype=np.int64)


def run_phase(router, starts: np.ndarray, sand: bool) -> None:
    """Publish results only after the complete native phase succeeds."""
    native = importlib.import_module("pyDeltaRCM._native")
    # Use the checkpointed stream for phase seeds; no hidden native RNG state.
    seed = int(shared_tools.get_random_uniform(2**53))
    result = native.sediment_phase(*phase_inputs(router, starts, sand), seed)
    for name, value in result.items():
        current = getattr(router, name, None)
        if isinstance(current, np.ndarray):
            # Reference routers mutate caller arrays, including subclass-held aliases.
            current[...] = value
        else:
            setattr(router, name, value)
