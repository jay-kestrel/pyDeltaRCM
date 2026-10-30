"""Sediment boundary contracts, including post-P2 parcel fixes."""

import numpy as np
import pytest

from pyDeltaRCM import DeltaModel, _native, native_sediment, shared_tools


@pytest.fixture
def model(tmp_path):
    model = DeltaModel(
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
    yield model
    model.finalize()


def prepare(model, sand, monkeypatch):
    """Initialize a one-cell-wide deterministic downstream channel."""
    model.cell_type[:] = -2
    model.cell_type[:, 4] = 0
    model.cell_type[-1, 4] = -1
    model.stage[:] = 0
    model.eta[:] = -1
    model.depth[:] = 1
    model.qw[:] = model.qx[:] = model.uw[:] = model.ux[:] = 1
    model.qy[:] = model.uy[:] = 0
    model.init_sediment_iteration()
    router = model._sr if sand else model._mr
    router.stepmax = 3
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    args = [
        np.empty(0, dtype=np.int64),
        model.eta,
        model.stage,
        model.depth,
        model.cell_type,
        model.uw,
        model.ux,
        model.uy,
        model.Vp_dep_mud,
        model.Vp_dep_sand,
        model.qw,
        model.qx,
        model.qy,
    ]
    if sand:
        args.append(model.qs)
    args.extend([model.mod_sed_weight, model.mod_stable_weight])
    router.run(*args)
    return router, args


@pytest.mark.parametrize("sand", [False, True])
@pytest.mark.parametrize(
    "regime", ["deposition", "erosion", "overlap", "force", "edge", "dry"]
)
def test_phase_matches_reference_on_forced_paths(model, monkeypatch, sand, regime):
    router, args = prepare(model, sand, monkeypatch)
    router.Vp_sed = 0.2
    router._dt = router._dx = router._u0 = 1.0
    router.qs0 = 0.1 if regime != "erosion" else 10.0
    router._f_bedload = 0.5
    router._beta = 3.0
    router.U_ero_sand = router.U_ero_mud = (
        0.5 if regime in ("erosion", "overlap") else 2.0
    )
    router.U_dep_mud = 2.0 if regime in ("deposition", "overlap") else 0.1
    router._lambda = 1.0
    router.force_deposit = regime == "force"
    if regime == "edge":
        model.cell_type[2, 4] = -1
    if regime == "dry":
        model.depth[:] = 0
        model.qx[:] = 0
        router.stepmax = 1
    args[0] = np.array([4, 4], dtype=np.int64)
    # Reinitialize padding after modifying the channel.
    empty_args = [np.empty(0, dtype=np.int64), *args[1:]]
    router.run(*empty_args)
    arrays, config, starts = native_sediment.phase_inputs(router, args[0], sand)
    before = {name: value.copy() for name, value in arrays.items()}
    native = _native.sediment_phase(arrays, config, starts, 42)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])
    router.run(*args)
    for name, value in native.items():
        np.testing.assert_allclose(
            value, getattr(router, name), rtol=2e-6, atol=2e-7, err_msg=name
        )
    if not sand and regime == "overlap":
        assert native["Vp_dep_mud"].sum() > 0
    if regime == "force":
        assert native["Vp_lost"] == 0
    if regime == "dry":
        assert native["eta"][-1, 4] == model.eta[-1, 4]


@pytest.mark.parametrize("sand", [False, True])
def test_phase_repeats_and_preserves_aliases(model, monkeypatch, sand):
    router, args = prepare(model, sand, monkeypatch)
    args[0] = np.array([4, 4], dtype=np.int64)
    inputs = native_sediment.phase_inputs(router, args[0], sand)
    first = _native.sediment_phase(*inputs, 812)
    second = _native.sediment_phase(*inputs, 812)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    assert native_sediment.can_use_native(router, sand)
    original = router.eta
    state = shared_tools.get_random_state()
    router.run(*args)
    assert original is router.eta
    assert shared_tools.get_random_state() != state


@pytest.mark.parametrize("sand", [False, True])
def test_custom_router_and_precision_keep_reference(model, monkeypatch, sand):
    router, _ = prepare(model, sand, monkeypatch)
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    assert native_sediment.can_use_native(router, sand)
    monkeypatch.setattr(router, "_choose_next_location", lambda *args: (0, 1, 1))
    assert not native_sediment.can_use_native(router, sand)
    monkeypatch.undo()
    router.eta = router.eta.astype(np.float64)
    assert not native_sediment.can_use_native(router, sand)


@pytest.mark.parametrize("bad", ["index", "shape", "nan", "stepmax", "trapped"])
def test_invalid_inputs_fail_without_mutating(model, monkeypatch, bad):
    router, _ = prepare(model, True, monkeypatch)
    arrays, config, starts = native_sediment.phase_inputs(router, np.array([4]), True)
    if bad == "index":
        starts[0] = -1
    elif bad == "shape":
        arrays["eta"] = arrays["eta"][:-1]
    elif bad == "nan":
        arrays["qw"] = np.full_like(arrays["qw"], np.nan)
    elif bad == "stepmax":
        config["stepmax"] = 0
    else:
        arrays["cell_type"] = np.full_like(arrays["cell_type"], -2)
    before = model.eta.copy()
    with pytest.raises(ValueError):
        _native.sediment_phase(arrays, config, starts, 0)
    np.testing.assert_array_equal(model.eta, before)
