# unit tests for pyDeltaRCM/submarine/turbidity_weights.py

import warnings

import numpy as np
import pytest

from pyDeltaRCM.native_water import can_use_native
from pyDeltaRCM.submarine import SubmarineFanModel
from pyDeltaRCM.submarine.turbidity_weights import (
    _get_bed_following_water_weight_array,
)
from pyDeltaRCM.water_tools import _get_water_weight_array

from . import utilities

# neighborhood order of the D8 weights (stock convention):
#   0=NW 1=N 2=NE 3=W 4=self 5=E 6=SW 7=S 8=SE
_NW, _N, _NE, _W, _CTR, _E, _SW, _S, _SE = range(9)


def _d8_walk_constants():
    """Flattened D8 geometry, copied verbatim from init_tools.set_constants."""
    sqrt2, sqrt05 = np.sqrt(2), np.sqrt(0.5)
    distances_flat = np.array(
        [[sqrt2, 1, sqrt2], [1, 1, 1], [sqrt2, 1, sqrt2]], dtype=np.float32
    ).flatten()
    ivec_flat = np.array(
        [[-sqrt05, 0, sqrt05], [-1, 0, 1], [-sqrt05, 0, sqrt05]],
        dtype=np.float32,
    ).flatten()
    jvec_flat = np.array(
        [[-sqrt05, -1, -sqrt05], [0, 0, 0], [sqrt05, 1, sqrt05]],
        dtype=np.float32,
    ).flatten()
    return ivec_flat, jvec_flat, distances_flat


def _synthetic_domain(shape=(5, 5), slope_gradient=0.03):
    """Synthetic fully-submerged domain arrays for the kernel.

    Returns ``(eta, depth, cell_type, qx, qy, mod_water_weight)``: a planar
    bed descending downdip (increasing row) at ``slope_gradient``, a wet
    cell everywhere, and the flow at rest.
    """
    eta = np.zeros(shape, dtype=np.float32)
    eta[:] = (-np.arange(shape[0]) * slope_gradient)[:, np.newaxis]
    depth = np.ones(shape, dtype=np.float32)
    cell_type = np.zeros(shape, dtype=np.int64)
    qx = np.zeros(shape, dtype=np.float32)
    qy = np.zeros(shape, dtype=np.float32)
    mod_water_weight = np.ones(shape, dtype=np.float32)
    return eta, depth, cell_type, qx, qy, mod_water_weight


class TestDowndipWeights:
    """Kernel weights pull parcels toward lower bed elevation."""

    def test_downdip_neighbor_dominates(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain()

        wts = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        w = wts[2, 2]  # mid-domain cell
        # the planar bed descends downdip: S is the modal step, and the
        #   along-dip S step beats the diagonal S-steps (distance weighting)
        assert w[_S] == w.max()
        assert w[_S] > w[_N], "downdip (S) weight must beat updip (N)"
        assert w[_S] > w[_E] and w[_S] > w[_W], "downdip must beat cross-dip"
        assert w[_S] > w[_SE] and w[_S] > w[_SW], "S must beat diagonals"
        assert w[_CTR] == 0
        assert pytest.approx(1 / (1 + np.sqrt(2)), rel=1e-3) == w[_S]

    def test_weights_prefer_downdip_everywhere(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain()

        wts = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        # every wet non-edge cell prefers a downdip step: the combined
        #   SW+S+SE weights (indices 6-8) exceed the combined updip ones
        downdip = wts[1:-1, 1:-1, [_SW, _S, _SE]].sum(axis=2)
        updip = wts[1:-1, 1:-1, [_NW, _N, _NE]].sum(axis=2)
        assert np.all(downdip > updip)

    def test_gamma_blends_bed_gravity_against_inertia(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain()
        # inertia term crossed against the bed slope: qy drives the E
        #   neighbor's inertia weight (weight_int = qx*jvec + qy*ivec)
        qy_crossed = np.ones(eta.shape, dtype=np.float32)

        def _w(gamma):
            return _get_bed_following_water_weight_array(
                depth, eta, mod, cell_type, qx, qy_crossed, ivec_flat,
                jvec_flat, distances_flat, 0.1, gamma, 1.0,
            )[2, 2]

        w_pure_inertia = _w(0.0)
        w_pure_gravity = _w(1.0)
        w_blended_low = _w(0.1)
        w_blended_high = _w(0.9)

        # pure inertia: the crossed flow pulls east, modally -- the D8 unit
        #   vectors give the E-adjacent diagonals half components, which
        #   normalize to E = 0.5, NE = SE = 0.25
        assert w_pure_inertia[_E] == pytest.approx(0.5, rel=1e-3)
        assert w_pure_inertia[_E] == w_pure_inertia.max()
        # pure bed gravity: the planar bed pulls downdip, modally
        assert w_pure_gravity[_S] == w_pure_gravity.max()
        # the blend interpolates monotonically between the two
        assert w_blended_high[_S] > w_blended_low[_S]
        assert w_blended_high[_E] < w_blended_low[_E]

    def test_flat_bed_with_matching_stage_matches_stock_kernel(self) -> None:
        """With stage == eta the kernel IS the stock kernel, bit-for-bit.

        The only change against the stock kernel is which field feeds the
        slope term; when both fields are identical the outputs must be
        identical too.
        """
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        np.random.seed(0)
        eta = np.random.uniform(0, 0.85, (7, 9)).astype(np.float32)
        stage = eta.copy()
        _, depth, cell_type, qx, qy, mod = _synthetic_domain(shape=(7, 9))

        wts_bed = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )
        wts_stock = _get_water_weight_array(
            depth, stage, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        assert np.array_equal(wts_bed, wts_stock)

    def test_bed_low_seen_by_turbidity_kernel_only(self) -> None:
        """Differential proof from the spike: a bed low under a flat surface.

        The stock kernel's slope term reads the stage field, so it cannot
        see bed relief at all (uniform fallback); the turbidity kernel
        pulls exclusively into the bed low.
        """
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain(
            shape=(5, 5), slope_gradient=0.0
        )
        eta[3, 4] = -1.0  # a bed low south-east of the center cell (2, 3)
        stage = np.zeros(eta.shape, dtype=np.float32)  # flat free surface

        wts_bed = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 1.0, 1.0,
        )[2, 3]
        wts_stock = _get_water_weight_array(
            depth, stage, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 1.0, 1.0,
        )[2, 3]

        assert wts_bed[_SE] > 0.99, "bed kernel does not fall into the low"
        assert wts_stock[_SE] < 0.2, "stock kernel unexpectedly sees the low"


class TestZeroWeightFallbacks:
    """The stock zero-weight fallback paths are preserved verbatim."""

    def test_flat_bed_at_rest_uniform_fallback(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain(
            slope_gradient=0.0
        )

        wts = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        w = wts[2, 2]  # no slope signal and no inertia: random walk fallback
        assert w[_CTR] == 0
        assert np.allclose(w[[_NW, _N, _NE, _W, _E, _SW, _S, _SE]], 1 / 8)

    def test_wall_neighbors_excluded_from_fallback(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain(
            slope_gradient=0.0
        )
        cell_type[1, 2] = -2  # the N neighbor of the center cell is a wall

        wts = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        w = wts[2, 2]
        assert w[_N] == 0
        assert np.allclose(
            w[[_NW, _NE, _W, _E, _SW, _S, _SE]], 1 / 7
        )

    def test_dry_neighbors_get_zero_weight(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain()
        depth[1, 2] = 0.05  # the N neighbor is drier than dry_depth = 0.1

        wts = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        w = wts[2, 2]
        assert w[_N] == 0
        assert w[_S] == w.max(), "downdip preference must survive dry N"

    def test_all_dry_falls_back_to_random_walk_into_non_wall(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain(
            slope_gradient=0.0
        )
        depth[:] = 0.05  # everything drier than dry_depth = 0.1

        wts = _get_bed_following_water_weight_array(
            depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
            distances_flat, 0.1, 0.294, 1.0,
        )

        # no wet valid neighbors: stock fallback walks randomly into any
        #   non-wall cell (all 8 neighbors here)
        w = wts[2, 2]
        assert w[_CTR] == 0
        assert np.allclose(w[[_NW, _N, _NE, _W, _E, _SW, _S, _SE]], 1 / 8)

    def test_no_non_wall_neighbor_raises_stock_error(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain(
            slope_gradient=0.0
        )
        cell_type[:] = -2
        cell_type[2, 2] = 0  # a lone non-wall cell surrounded by walls

        with pytest.raises(RuntimeError, match="No non-wall cells"):
            _get_bed_following_water_weight_array(
                depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
                distances_flat, 0.1, 0.294, 1.0,
            )

    def test_nan_input_raises_stock_error(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain(
            slope_gradient=0.0
        )
        eta[2, 2] = np.nan

        with pytest.raises(RuntimeError, match="NaN encountered in input"):
            _get_bed_following_water_weight_array(
                depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
                distances_flat, 0.1, 0.294, 1.0,
            )


class TestGammaDensityRange:
    """No "gamma too large" RuntimeError across the documented range.

    ``gamma_density = coeff_gamma_density * g * slope_gradient * dx
    / u_current**2``; the documented parameter ranges (canyon-to-fan slopes
    1e-3 to 1e-1, the default coefficient 1.0) with test-scale ``dx = 1`` and
    ``u_current = 1`` produce gamma_density in [1e-3, 1]. For gamma in
    [0, 1] the blend coefficients ``gamma`` and ``(1 - gamma)`` are both
    non-negative, so the blended weights can never go negative and the
    stock "gamma too large" error is unreachable.
    """

    gamma_density_range = np.logspace(-3, 0, 25)

    def test_no_runtime_error_across_documented_range(self) -> None:
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()

        flow_fields = {
            "at rest": (np.zeros((5, 5), dtype=np.float32),
                        np.zeros((5, 5), dtype=np.float32)),
            "downdip inertia": (np.ones((5, 5), dtype=np.float32),
                                np.zeros((5, 5), dtype=np.float32)),
            "crossed inertia": (np.zeros((5, 5), dtype=np.float32),
                                np.ones((5, 5), dtype=np.float32)),
        }
        eta, depth, cell_type, _, _, mod = _synthetic_domain()

        for name, (qx, qy) in flow_fields.items():
            for gamma in self.gamma_density_range:
                wts = _get_bed_following_water_weight_array(
                    depth, eta, mod, cell_type, qx, qy, ivec_flat,
                    jvec_flat, distances_flat, 0.1, gamma, 1.0,
                )
                assert np.all(np.isfinite(wts)), (
                    f"nonfinite weights for {name}, gamma={gamma}"
                )
                assert np.allclose(wts.sum(axis=2), 1.0), (
                    f"unnormalized weights for {name}, gamma={gamma}"
                )

    def test_extreme_gamma_density_with_flow_at_rest(self) -> None:
        """Steep-canyon/low-velocity corner (gamma_density >> 1) at rest.

        With no inertia term the blended weight is ``gamma * weight_sfc``,
        non-negative for any gamma, so even extreme values route.
        """
        ivec_flat, jvec_flat, distances_flat = _d8_walk_constants()
        eta, depth, cell_type, qx, qy, mod = _synthetic_domain()

        for gamma in (10.0, 100.0, 1000.0):
            wts = _get_bed_following_water_weight_array(
                depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
                distances_flat, 0.1, gamma, 1.0,
            )
            assert np.all(np.isfinite(wts))
            assert np.allclose(wts.sum(axis=2), 1.0)


def _minimal_config():
    """Tiny-model configuration, mirroring the test_submarine_model fixture."""
    return {
        "Length": 10.0,
        "Width": 10.0,
        "seed": 0,
        "dx": 1.0,
        "L0_meters": 1.0,
        "S0": 0.0002,
        "itermax": 1,
        "Np_water": 10,
        "u0": 1.0,
        "N0_meters": 2.0,
        "h0": 1.0,
        "H_SL": 0.0,
        "SLR": 0.0,
        "Np_sed": 10,
        "f_bedload": 0.5,
        "C0_percent": 0.1,
        "toggle_subsidence": False,
        "subsidence_rate": 0.0,
        "start_subsidence": 50.0,
        "save_dt": 500,
    }


@pytest.fixture(scope="function")
def test_SubmarineFanModel(tmp_path) -> SubmarineFanModel:
    """Tiny SubmarineFanModel built from a YAML file."""
    p = utilities.yaml_from_dict(
        tmp_path, "turbidity_weights.yaml", _minimal_config()
    )
    _delta = SubmarineFanModel(input_file=p)
    return _delta


class TestSubmarineModelWeightSeam:
    """The public get_water_weight_array seam drives the turbidity kernel."""

    def test_overrides_defined_on_subclass(self) -> None:
        assert "get_water_weight_array" in SubmarineFanModel.__dict__
        assert "compute_free_surface" in SubmarineFanModel.__dict__

    def test_weights_prefer_downdip_on_model_domain(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        model.get_water_weight_array()

        assert model.water_weights.shape == (model.L, model.W, 9)
        w = model.water_weights[5, model.CTR]
        assert w[_S] == w.max(), "downdip is not the modal step on the model bed"
        assert w[_S] > w[_N], "downdip (S) weight must beat updip (N)"
        assert w[_CTR] == 0

    def test_compute_free_surface_tracks_bed(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        model.eta[3, 5] += 0.2  # simulate deposition

        model.compute_free_surface()

        np.testing.assert_allclose(model.stage, model.eta + model.h_current)

    def test_current_stays_off_native_fused_path(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        # the native fused water path inlines the stock weighting, so the
        #   subclass must fall back to the Python reference path for the
        #   bed-following kernel to run at all
        assert can_use_native(test_SubmarineFanModel) is False

    def test_update_routes_bed_following_current(self, tmp_path) -> None:
        """One stock update() routes the current downdip through the seam.

        Uses a domain large enough to route: on the tiny 10x10 fixture the
        stock elliptical edge mask marks every ocean cell as outflow
        boundary, so parcels cannot travel (construction-scale fixture
        only). 30x60 keeps an interior, mirroring the spike's 50x100.
        """
        cfg = _minimal_config()
        cfg.update({"Length": 30.0, "Width": 60.0, "N0_meters": 3.0})
        p = utilities.yaml_from_dict(tmp_path, "turbidity_route.yaml", cfg)
        model = SubmarineFanModel(input_file=p)

        model.update()

        for name in ("eta", "stage", "depth", "qx", "qy", "qw", "uw"):
            assert np.all(np.isfinite(getattr(model, name))), (
                f"nonfinite values in {name}"
            )
        # the stock post-update depth invariant holds; depth is only
        #   transiently thinner than h_current where this timestep's
        #   deposition outpaced the pre-sediment stage (the next water
        #   iteration recomputes stage = eta + h_current and restores the
        #   current thickness)
        np.testing.assert_allclose(
            model.depth, np.maximum(model.stage - model.eta, 0)
        )
        # water flux reached the downdip two-thirds of the domain
        assert model.qwn[20:, :].sum() > 0, (
            "no water flux reached the downdip two-thirds of the domain"
        )

    def test_current_thickness_mismatch_warns(self, tmp_path) -> None:
        cfg = _minimal_config()
        cfg["current_thickness"] = 2.0
        p = utilities.yaml_from_dict(tmp_path, "turbidity_weights.yaml", cfg)
        with pytest.warns(UserWarning, match="current_thickness"):
            SubmarineFanModel(input_file=p)

    def test_default_thickness_does_not_warn(self, tmp_path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "turbidity_weights.yaml", _minimal_config()
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SubmarineFanModel(input_file=p)
        assert not [w for w in caught if "current_thickness" in str(w.message)]
