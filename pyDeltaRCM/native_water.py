"""Private coarse-boundary adapter for the optional Rust water phase.

The original methods remain the reference implementation and subclass API.
Only an unmodified phase is fused: a subclass overriding its internal methods
or hooks must still observe the original call sequence.
"""

import importlib
import importlib.util
import os
import warnings

import numpy as np

from pyDeltaRCM import shared_tools
from pyDeltaRCM.hook_tools import hook_tools

FLOAT_FIELDS = ("stage", "eta", "depth", "uw", "ux", "uy", "qxn", "qyn", "qwn")


def can_use_native(model):
    """Select the fused path without bypassing Python customization points."""
    if os.environ.get("PYDELTARCM_NATIVE", "1") == "0":
        return False

    from pyDeltaRCM.water_tools import water_tools

    for name in (
        "init_water_iteration",
        "run_water_iteration",
        "update_Q",
        "check_for_boundary",
        "check_size_of_indices_matrix",
        "compute_free_surface",
        "finalize_free_surface",
        "get_water_weight_array",
    ):
        if getattr(getattr(model, name), "__func__", None) is not getattr(
            water_tools, name
        ):
            return False
    for name in (
        "hook_init_water_iteration",
        "hook_run_water_iteration",
        "hook_compute_free_surface",
    ):
        if getattr(getattr(model, name), "__func__", None) is not getattr(
            hook_tools, name
        ):
            return False
    # Keep custom numerical precision and nonstandard walk geometry on the
    # reference path rather than silently converting the model's semantics.
    weight_fields = (
        "qx",
        "qy",
        "mod_water_weight",
        "ivec_flat",
        "jvec_flat",
        "distances_flat",
    )
    if any(
        getattr(model, name).dtype != np.float32
        for name in FLOAT_FIELDS + weight_fields
    ):
        return False
    if not np.array_equal(model.iwalk_flat, [-1, 0, 1, -1, 0, 1, -1, 0, 1]):
        return False
    if not np.array_equal(model.jwalk_flat, [-1, -1, -1, 0, 0, 0, 1, 1, 1]):
        return False
    if not np.array_equal(
        model.ravel_walk_flat, model.jwalk_flat * model.W + model.iwalk_flat
    ):
        return False
    if importlib.util.find_spec("pyDeltaRCM._native") is None:
        warnings.warn(
            "Rust water extension is not installed; using the Python reference. "
            "Build with pip install . or set PYDELTARCM_NATIVE=0 explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def phase_inputs(model, construct_weights=False):
    """Build the typed phase arguments; also used by differential tests."""
    native = importlib.import_module("pyDeltaRCM._native")
    config = native.WaterConfig(
        model.stepmax,
        model.L0,
        model.CTR,
        model.free_surf_walk_inds.shape[1],
        model._Nsmooth,
        model._dx,
        model.Qp_water,
        model._u0,
        model.h0,
        model._H_SL,
        model._S0,
        model._Csmooth,
    )
    return (
        None
        if construct_weights
        else np.ascontiguousarray(model.water_weights),
        np.ascontiguousarray(model.start_indices, dtype=np.int64),
        np.ascontiguousarray(model.cell_type, dtype=np.int64),
        *(
            np.ascontiguousarray(getattr(model, name))
            for name in FLOAT_FIELDS[:6]
        ),
        config,
    )


def weight_inputs(model):
    """Retain NumPy's power rounding without its per-cell dispatch overhead."""
    factor = (model.depth * model.mod_water_weight) ** model.theta_water
    return (
        np.ascontiguousarray(factor),
        *(
            np.ascontiguousarray(getattr(model, name))
            for name in (
                "qx",
                "qy",
                "ivec_flat",
                "jvec_flat",
                "distances_flat",
            )
        ),
        model.dry_depth,
        model.gamma,
    )


def run_water_phase(model):
    """Route and compute the surface with exactly one native phase call."""
    native = importlib.import_module("pyDeltaRCM._native")
    model.log_info("Computing water weight array", verbosity=2)
    # Draw from the existing checkpointed stream once per iteration. No new
    # persistent RNG state is needed to resume a model exactly from checkpoint.
    seed = int(shared_tools.get_random_uniform(2**53))
    result = native.water_phase(
        *phase_inputs(model, construct_weights=True),
        seed,
        weight_inputs=weight_inputs(model),
    )
    model.water_weights = result["water_weights"]
    for name in ("qxn", "qyn", "qwn", "free_surf_walk_inds", "free_surf_flag"):
        setattr(model, name, result[name])
    model.hook_compute_free_surface()
    for name in ("sfc_visit", "sfc_sum", "Hnew"):
        setattr(model, name, result[name])
    if model._time_iter > 0:
        model.stage = ((1 - model._omega_sfc) * model.stage) + (
            model._omega_sfc * result["Hsmth"]
        )
    model.flooding_correction()
