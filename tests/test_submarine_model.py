# unit tests for pyDeltaRCM/submarine/submarine_fan_model.py

from pathlib import Path

import pytest

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM.submarine import SubmarineFanModel

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
        "SLR": 0.001,
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
