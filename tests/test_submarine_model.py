# unit tests for pyDeltaRCM/submarine/submarine_fan_model.py

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM.submarine import SubmarineFanModel
from pyDeltaRCM.submarine.submarine_fan_model import _submarine_bed_profile

from . import utilities


def _minimal_config():
    """Tiny-model configuration, mirroring the stock test_DeltaModel fixture."""
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
def test_SubmarineFanModel(tmp_path: Path) -> SubmarineFanModel:
    """Tiny SubmarineFanModel built from a YAML file."""
    p = utilities.yaml_from_dict(tmp_path, "submarine_parameters.yaml", _minimal_config())
    _delta = SubmarineFanModel(input_file=p)
    return _delta


class TestSubmarineModelConstruction:

    def test_is_delta_model_subclass(self) -> None:
        assert issubclass(SubmarineFanModel, DeltaModel)

    def test_constructor_from_yaml(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", _minimal_config())
        delta = SubmarineFanModel(input_file=p)
        assert isinstance(delta, SubmarineFanModel)
        assert isinstance(delta, DeltaModel)

    def test_constructor_from_kwargs(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["current_concentration"] = 0.03
        delta = SubmarineFanModel(**cfg)
        assert delta.current_concentration == 0.03
        assert delta.slope_gradient == 0.03  # default applied alongside kwargs

    def test_defaults_applied(self, test_SubmarineFanModel: SubmarineFanModel) -> None:
        assert test_SubmarineFanModel.current_concentration == 0.025
        assert test_SubmarineFanModel.slope_gradient == 0.03
        assert test_SubmarineFanModel.current_thickness is None
        assert test_SubmarineFanModel.coeff_gamma_density == 1.0

    def test_current_thickness_falls_back_to_h0(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        assert test_SubmarineFanModel.current_thickness is None
        assert test_SubmarineFanModel.h_current == test_SubmarineFanModel.h0

    def test_current_thickness_given(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["current_thickness"] = 2.5
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", cfg)
        delta = SubmarineFanModel(input_file=p)
        assert delta.current_thickness == 2.5
        assert delta.h_current == 2.5

    def test_u_current_mirrors_u0(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        assert test_SubmarineFanModel.u_current == test_SubmarineFanModel.u0

    def test_gamma_density_from_defaults(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        _d = test_SubmarineFanModel
        expected = _d.coeff_gamma_density * _d.g * _d.slope_gradient * _d.dx / _d.u_current**2
        assert _d.gamma_density == pytest.approx(expected)

    def test_gamma_density_from_custom_values(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["slope_gradient"] = 0.05
        cfg["coeff_gamma_density"] = 1.5
        cfg["current_thickness"] = 2.0
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", cfg)
        delta = SubmarineFanModel(input_file=p)
        expected = 1.5 * delta.g * 0.05 * delta.dx / delta.u_current**2
        assert delta.gamma_density == pytest.approx(expected)
        assert delta.h_current == 2.0


class TestSubmarineModelValidation:

    def test_type_error_current_concentration(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["current_concentration"] = "high"
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", cfg)
        with pytest.raises(
            TypeError, match=r"Input type was str, but needs to be float, int"
        ):
            SubmarineFanModel(input_file=p)

    def test_type_error_slope_gradient(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["slope_gradient"] = "steep"
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", cfg)
        with pytest.raises(TypeError, match=r"Input type was str, but needs to be float"):
            SubmarineFanModel(input_file=p)

    def test_type_error_current_thickness(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["current_thickness"] = "deep"
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", cfg)
        with pytest.raises(
            TypeError, match=r"Input type was str, but needs to be float, int"
        ):
            SubmarineFanModel(input_file=p)

    def test_type_error_coeff_gamma_density(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["coeff_gamma_density"] = 2
        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", cfg)
        with pytest.raises(TypeError, match=r"Input type was int, but needs to be float"):
            SubmarineFanModel(input_file=p)

    def test_stock_name_collision_raises(self, tmp_path: Path) -> None:
        class CollidingParams(SubmarineFanModel):
            def hook_import_files(self) -> None:
                super().hook_import_files()
                self.subclass_parameters["h0"] = {
                    "type": ["float"], "default": 2.0,
                }

        p = utilities.yaml_from_dict(tmp_path, "submarine.yaml", _minimal_config())
        with pytest.raises(ValueError, match=r"already a default yaml parameter"):
            CollidingParams(input_file=p)


class TestSubmarineBedProfile:
    """Unit tests for the pure bed-profile helper."""

    def test_shape_and_head_datum(self) -> None:
        profile = _submarine_bed_profile(10, 2.0, 0.05, 100.0)
        assert profile.shape == (10,)
        assert profile[0] == 0.0

    def test_descends_at_slope_gradient(self) -> None:
        profile = _submarine_bed_profile(10, 2.0, 0.05, 100.0)
        np.testing.assert_allclose(np.diff(profile), -0.1)

    def test_clamps_at_basin_floor_relief(self) -> None:
        profile = _submarine_bed_profile(100, 1.0, 0.1, 0.3)
        # strictly descending until the floor is reached, flat thereafter
        assert np.all(np.diff(profile[:3]) < 0)
        assert np.all(profile[3:] == -0.3)

    def test_short_domain_never_reaches_floor(self) -> None:
        profile = _submarine_bed_profile(10, 1.0, 0.03, 5.0)
        assert np.all(np.diff(profile) < 0)


class TestSubmarineDomainGeometry:
    """The domain produced by hook_after_create_domain (spec criterion V3)."""

    def test_eta_monotonically_downdip(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        eta = test_SubmarineFanModel.eta
        assert np.all(np.diff(eta, axis=0) <= 0)

    def test_canyon_head_at_inlet_row(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        assert np.all(model.eta[0, :] == 0.0)
        assert np.all(model.eta[0, model.inlet] == model.eta.max())

    def test_stage_trails_bed_at_constant_thickness(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        np.testing.assert_allclose(model.stage, model.eta + model.h_current)
        np.testing.assert_allclose(model.depth, model.h_current)
        assert np.all(model.depth > 0)

    def test_domain_fully_submerged_no_shoreline(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        # former land cells became ocean; there are no walls and no dry cells
        assert not np.any(model.cell_type == -2)
        assert np.all(model.depth > 0)

    def test_inlet_cells_present_at_head(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        assert model.inlet.size == model.N0
        assert np.all(model.cell_type[0, model.inlet] == 1)

    def test_edge_mask_retained(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        # the hook touches cell_type only for land (-2) -> ocean (0); on the
        # tiny fixture domain the stock elliptical mask claims every row
        # below the inlet row
        assert np.all(model.cell_type[1:, :] == -1)
        assert not np.any(model.cell_type[0, :] == -1)

    def test_flow_fields_start_at_rest(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        for field in (model.qx, model.qy, model.qw, model.ux, model.uy, model.uw):
            assert np.all(field == 0)

    def test_bed_trackers_match_initial_bed(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        assert np.all(model.eta0 == model.eta)
        assert np.all(model.eta_init == model.eta)

    def test_stage_floor_never_binds(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        model = test_SubmarineFanModel
        # H_SL is pinned at the basin floor, strictly below the current's
        #   surface, so the stock stage floor max(stage, H_SL) is a no-op
        assert np.all(np.maximum(model.stage, model.H_SL) == model.stage)
        assert model.H_SL == pytest.approx(-model.hb)

    def test_slr_zero_in_v1(
        self, test_SubmarineFanModel: SubmarineFanModel
    ) -> None:
        assert test_SubmarineFanModel.SLR == 0.0

    def test_nonzero_slr_warns_and_zeroes(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["SLR"] = 0.001
        p = utilities.yaml_from_dict(tmp_path, "submarine_slr.yaml", cfg)
        with pytest.warns(UserWarning, match=r"SLR is not simulated"):
            model = SubmarineFanModel(input_file=p)
        assert model.SLR == 0.0

    def test_zero_slr_does_not_warn(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "submarine_slr0.yaml", _minimal_config()
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SubmarineFanModel(input_file=p)
        assert not [w for w in caught if "SLR" in str(w.message)]
        assert model.SLR == 0.0

    def test_large_domain_geometry(self, tmp_path: Path) -> None:
        """Slope reaches a flat basin floor on a longer domain."""
        cfg = _minimal_config()
        cfg.update(
            {
                "Length": 40.0,
                "Width": 20.0,
                "slope_gradient": 0.02,
                "hb": 0.5,
            }
        )
        p = utilities.yaml_from_dict(tmp_path, "submarine_large.yaml", cfg)
        model = SubmarineFanModel(input_file=p)

        # monotonic descent downdip in every column
        assert np.all(np.diff(model.eta, axis=0) <= 0)
        # the slope has flattened at the basin floor (relief = hb = 0.5 m)
        #   well before the downdip end, and the floor is flat
        assert model.eta[24, 0] > -0.5
        assert np.allclose(model.eta[30:, :], -0.5)
        assert np.all(model.eta[30:, :] == model.eta[30, 0])
        # fully wet, no land cells
        assert not np.any(model.cell_type == -2)
        assert np.all(model.depth > 0)
        # stock elliptical edge ring retained: borders and interior edges
        assert np.all(model.cell_type[-1, :] == -1)
        assert np.all(model.cell_type[1:, 0] == -1)
        assert np.all(model.cell_type[1:, -1] == -1)
        assert np.any(model.cell_type[1:, :] == 0)  # interior ocean cells
        # inlet at the canyon head, the highest bed in the domain
        assert np.all(model.cell_type[0, model.inlet] == 1)
        assert np.all(model.eta[0, model.inlet] == model.eta.max())
