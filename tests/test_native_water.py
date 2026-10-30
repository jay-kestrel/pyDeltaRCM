"""Deterministic phase contracts and native dispatch compatibility."""

import numpy as np
import pytest

from pyDeltaRCM import DeltaModel, _native, native_water, shared_tools, water_tools


@pytest.fixture
def model(tmp_path):
    instance = DeltaModel(
        out_dir=str(tmp_path / "model"),
        Length=10.0,
        Width=10.0,
        dx=1.0,
        L0_meters=1.0,
        N0_meters=2.0,
        h0=1.0,
        Np_water=20,
        Np_sed=20,
        itermax=1,
        seed=0,
    )
    instance.init_water_iteration()
    instance.get_water_weight_array()
    yield instance
    instance.finalize()


@pytest.mark.parametrize("sea_level", [0.0, 0.25])
@pytest.mark.parametrize("evolved", [False, True])
def test_surface_kernels_equal_reference_on_identical_walks(model, sea_level, evolved):
    if evolved:
        rng = np.random.default_rng(42)
        for name in ("uw", "ux", "uy", "depth"):
            getattr(model, name)[:] = rng.uniform(0.0, 2.0, model.depth.shape)
        model.eta[:] = rng.uniform(-1.0, 1.0, model.eta.shape)
    model._H_SL = sea_level
    result = _native.water_phase(*native_water.phase_inputs(model), 42)
    visits, sums = water_tools._accumulate_free_surface_walks(
        result["free_surf_walk_inds"],
        result["free_surf_flag"],
        model.cell_type,
        model.uw,
        model.ux,
        model.uy,
        model.depth,
        model._dx,
        model._u0,
        model.h0,
        model._H_SL,
        model._S0,
    )
    np.testing.assert_array_equal(result["sfc_visit"], visits)
    np.testing.assert_array_equal(result["sfc_sum"], sums)
    hnew = model.eta + model.depth
    hnew[visits > 0] = sums[visits > 0] / visits[visits > 0]
    hnew = np.maximum(np.maximum(hnew, model._H_SL), model.eta)
    np.testing.assert_array_equal(result["Hnew"], hnew)
    smoothed = water_tools._smooth_free_surface(
        hnew,
        model.cell_type,
        model._Nsmooth,
        model._Csmooth,
    )
    # NumPy uses a pairwise float32 sum; Rust sums sequentially.
    np.testing.assert_allclose(result["Hsmth"], smoothed, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "direction,loop", [(7, False), (7, True), (0, False), (2, False)]
)
def test_forced_walks_and_flux_are_exact(model, monkeypatch, direction, loop):
    # One-hot weights remove RNG differences. A two-cell cycle exercises
    # relocation, loop flags and path growth as well as terminal discharge.
    weights = np.zeros_like(model.water_weights)
    weights[:, :, direction] = 1
    if loop:
        weights[3, :, 7] = 0
        weights[3, :, 1] = 1
    model.water_weights = weights
    model.free_surf_walk_inds = np.zeros((model._Np_water, 2), dtype=np.int64)
    monkeypatch.setattr(model, "get_water_weight_array", lambda: None)
    result = _native.water_phase(*native_water.phase_inputs(model), 101)
    model.run_water_iteration()
    for name in ("qxn", "qyn", "qwn", "free_surf_walk_inds", "free_surf_flag"):
        np.testing.assert_array_equal(result[name], getattr(model, name), err_msg=name)


@pytest.mark.parametrize("count", [1, 2, 20])
def test_duplicate_inlet_parcels_contribute_all_discharge(model, monkeypatch, count):
    """#10 accumulates ghost-node flux for every parcel, not unique starts."""
    model._stepmax = 0  # isolate the inlet injection from subsequent steps
    start = int(model.inlet[0])
    model.start_indices = np.full(count, start, dtype=np.int64)
    model.free_surf_walk_inds = np.zeros((count, 2), dtype=np.int64)
    model.free_surf_flag = np.ones(count, dtype=np.int64)
    monkeypatch.setattr(model, "get_water_weight_array", lambda: None)
    result = _native.water_phase(*native_water.phase_inputs(model), 11)
    model.run_water_iteration()
    for name in ("qxn", "qyn", "qwn"):
        np.testing.assert_array_equal(result[name], getattr(model, name))
    expected = np.float32(count * model.Qp_water / model.dx / 2)
    assert result["qwn"].ravel()[start] == expected


def test_absolute_flux_preserves_numba_arithmetic():
    # Preserve numba/Rust float64 intermediates before each float32 store.
    # Pure NumPy scalar promotion instead produces 0.44999998807907104.
    actual = water_tools._update_absQfield(
        np.array([0.225], dtype=np.float32),
        np.ones(3),
        np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=bool),
        0.15,
        1.0,
    )
    np.testing.assert_array_equal(
        actual, np.array([0.44999995827674866], dtype=np.float32)
    )


def test_native_repeat_is_bit_identical(model):
    args = native_water.phase_inputs(model)
    first = _native.water_phase(*args, 919)
    second = _native.water_phase(*args, 919)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_checkpoint_stream_replays_native_phase(model):
    starts = model.start_indices.copy()
    stage = model.stage.copy()
    state = shared_tools.get_random_state()
    native_water.run_water_phase(model)
    names = ("free_surf_walk_inds", "free_surf_flag", "qxn", "qyn", "qwn", "Hnew")
    expected = {name: getattr(model, name).copy() for name in names}
    expected_state = shared_tools.get_random_state()
    model.stage = stage
    model.init_water_iteration()
    model.start_indices = starts
    shared_tools.set_random_state(state)
    native_water.run_water_phase(model)
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(model, name), value)
    assert shared_tools.get_random_state() == expected_state


def test_dispatch_and_reference_switch(model, monkeypatch):
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    assert native_water.can_use_native(model)
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    assert not native_water.can_use_native(model)


@pytest.mark.parametrize(
    "name",
    [
        "run_water_iteration",
        "update_Q",
        "compute_free_surface",
        "finalize_free_surface",
        "check_for_boundary",
        "check_size_of_indices_matrix",
        "hook_compute_free_surface",
        "hook_run_water_iteration",
        "hook_init_water_iteration",
        "init_water_iteration",
    ],
)
def test_overridden_phase_methods_keep_python_semantics(model, monkeypatch, name):
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    monkeypatch.setattr(model, name, lambda *args: None)
    assert not native_water.can_use_native(model)


@pytest.mark.parametrize("field", native_water.FLOAT_FIELDS)
def test_custom_precision_uses_reference(model, monkeypatch, field):
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    setattr(model, field, getattr(model, field).astype(np.float64))
    assert not native_water.can_use_native(model)


def test_native_rejects_invalid_inputs(model):
    args = list(native_water.phase_inputs(model))
    args[1] = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="start index"):
        _native.water_phase(*args, 0)
    args = list(native_water.phase_inputs(model))
    args[0] = np.zeros((2, 2, 9))
    with pytest.raises(ValueError, match="grid shapes"):
        _native.water_phase(*args, 0)
    args = list(native_water.phase_inputs(model))
    args[0] = np.full_like(args[0], np.nan)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _native.water_phase(*args, 0)
