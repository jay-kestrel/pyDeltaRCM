import pytest

import platform
import os
import logging
import warnings
from pathlib import Path
import glob
import yaml

import unittest.mock as mock

import pyDeltaRCM
from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM.shared_tools import custom_yaml_loader

# from pyDeltaRCM import shared_tools
from pyDeltaRCM import preprocessor

from . import utilities

# from .utilities import test_DeltaModel


class TestPreprocessorSingleJobSetups:

    # test high level python api
    def test_py_hlvl_wo_args(self) -> None:
        with pytest.raises(ValueError):
            _ = preprocessor.Preprocessor()

    def test_py_hlvl_runjobs_simple_param_float(self, tmp_path: Path) -> None:
        # a single parameter float
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"h0": 7.5})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False
        assert pp._config_list[0]["h0"] == 7.5

    def test_py_hlvl_runjobs_simple_param_null0(self, tmp_path: Path) -> None:
        # a parameter that takes null as the default
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"hb": 7.5})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False
        assert pp._config_list[0]["hb"] == 7.5

    def test_py_hlvl_runjobs_simple_param_null1(self, tmp_path: Path) -> None:
        # a parameter that takes null with null
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"hb": None})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False
        assert pp._config_list[0]["hb"] == "None"

    def test_py_hlvl_runjobs_simple_param_yaml(self, tmp_path: Path) -> None:
        yaml_dict = {
            "out_dir": tmp_path / "out_dir",
            "h0": 7.5,
        }
        pp = pyDeltaRCM.Preprocessor(yaml_dict)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False
        assert pp._config_list[0]["h0"] == 7.5

    def test_py_hlvl_tsteps_yml_runjobs_sngle(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"timesteps": 50})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["timesteps"] == 50
        assert not ("time" in pp.config_dict.keys())
        assert not ("time_years" in pp.config_dict.keys())

    def test_py_hlvl_time_yml_runjobs_sngle(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"time": 1000})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["time"] == 1000
        assert not ("timesteps" in pp.config_dict.keys())

    def test_py_hlvl_time_If_yml_runjobs_sngle(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"time": 10000, "If": 0.1})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["time"] == 10000
        assert pp.config_dict["If"] == 0.1
        assert not ("timesteps" in pp.config_dict.keys())

    def test_py_hlvl_timeyears_yml_runjobs_sngle(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"time_years": 3.16880878140e-05}
        )
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["time_years"] == pytest.approx(3.1688087e-05)
        assert not ("If" in pp.config_dict.keys())
        assert not ("time" in pp.config_dict.keys())
        assert not ("timesteps" in pp.config_dict.keys())

    def test_py_hlvl_timeyears_If_yml_runjobs_sngle(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"time_years": 0.00031688087814, "If": 0.1}
        )
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["time_years"] == pytest.approx(0.00031688087814)
        assert pp.config_dict["If"] == 0.1
        assert not ("time" in pp.config_dict.keys())
        assert not ("timesteps" in pp.config_dict.keys())

    def test_py_hlvl_timesteps_and_yaml_argument(self, tmp_path: Path) -> None:
        """
        Test that timesteps can come from cli, but others from yaml
        """
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_eta_figs": True})
        pp = preprocessor.Preprocessor(input_file=p, timesteps=2)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["timesteps"] == 2
        assert pp.config_dict["save_eta_figs"] is True
        assert not ("time" in pp.config_dict.keys())

    def test_py_hlvl_timesteps_yaml_and_cli(self, tmp_path: Path) -> None:
        """
        Test preference for command line argument
        """
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_eta_figs": True, "timesteps": 20}
        )
        pp = preprocessor.Preprocessor(input_file=p, timesteps=13)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["timesteps"] == 13
        assert pp.config_dict["save_eta_figs"] is True
        assert not ("time" in pp.config_dict.keys())

    def test_py_hlvl_timesteps_and_time_yaml_and_cli(self, tmp_path: Path) -> None:
        """
        Test that preference is given to command line arguments

        note that time and timesteps both persist, precedence for these args
        is determined at runtime of run_jobs()
        """
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_eta_figs": True, "timesteps": 20, "time": 1000},
        )
        pp = preprocessor.Preprocessor(input_file=p, timesteps=13, time=13000)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["timesteps"] == 13
        assert pp.config_dict["time"] == 13000
        assert pp.config_dict["save_eta_figs"] is True
        assert "time" in pp.config_dict.keys()
        assert "timesteps" in pp.config_dict.keys()

    def test_py_hlvl_timesteps_not_writted_to_job(self, tmp_path: Path) -> None:
        """
        Test that timesteps is not written to the file

        note that time and timesteps both persist, precedence for these args
        is determined at runtime of run_jobs()
        """
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_eta_figs": True, "timesteps": 20, "time": 1000},
        )
        pp = preprocessor.Preprocessor(input_file=p, timesteps=13, time=13000)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict["timesteps"] == 13
        assert pp.config_dict["time"] == 13000
        assert pp.config_dict["save_eta_figs"] is True
        assert "time" in pp.config_dict.keys()
        assert "timesteps" in pp.config_dict.keys()


class TestPreprocessorMatrixJobsSetups:

    def test_py_hlvl_mtrx_1list_timesteps_arg(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(f, ["f_bedload"], [[0.2, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.2 for j in f_bedload_list]) == 1
        assert sum([j == 0.6 for j in f_bedload_list]) == 1
        assert pp.config_dict["timesteps"] == 3

    def test_py_hlvl_mtrx_1list_tsteps_cfg(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "timesteps", 3)
        utilities.write_matrix_to_file(f, ["f_bedload"], [[0.2, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.2 for j in f_bedload_list]) == 1
        assert sum([j == 0.6 for j in f_bedload_list]) == 1
        assert pp.config_dict["timesteps"] == 3

    def test_py_hlvl_mtrx_smthng_null(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "timesteps", 3)
        utilities.write_matrix_to_file(f, ["hb"], [[None, 2, 7]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 3
        assert pp._is_completed is False

        hb_list = pp._matrix_table[:, 1]

        assert sum([j == "None" for j in hb_list]) == 1
        assert sum([j == 2 for j in hb_list]) == 1
        assert sum([j == 7 for j in hb_list]) == 1

    def test_py_hlvl_mtrx_2list(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(
            f, ["f_bedload", "u0"], [[0.2, 0.5, 0.6], [1.0, 1.5, 2.0]]
        )
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 9
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]
        u0_list = pp._matrix_table[:, 2]

        assert sum([j == 0.2 for j in f_bedload_list]) == 3
        assert sum([j == 0.5 for j in f_bedload_list]) == 3
        assert sum([j == 0.6 for j in f_bedload_list]) == 3
        assert sum([j == 1.0 for j in u0_list]) == 3
        assert sum([j == 1.5 for j in u0_list]) == 3
        assert sum([j == 2.0 for j in u0_list]) == 3

        comb_list = pp._matrix_table[:, 1:].tolist()
        assert [0.2, 2.0] in comb_list
        assert [0.5, 1.0] in comb_list
        assert not ([0.5, 0.2] in comb_list)

    def test_py_hlvl_mtrx_scientificnotation(self, tmp_path: Path) -> None:
        """Test that preprocessor can read and write scinotation"""
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(
            f, ["f_bedload", "SLR"], [[0.2, 0.5, 0.6], [0.00004, 1e-6]]
        )
        f.close()

        # will convert everything to scientific notation
        pp = preprocessor.Preprocessor(input_file=p)

        SLR_list = pp._matrix_table[:, 2]
        assert sum([j == 4e-5 for j in SLR_list]) == 3
        assert sum([j == 0.000001 for j in SLR_list]) == 3

    def test_py_hlvL_mtrx_needs_out_dir(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        # missing out_dir in the config will throw an error
        # => utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f, ["f_bedload"], [[0.2, 0.5, 0.6]])
        f.close()
        with pytest.raises(ValueError, match=r'You must specify "out_dir" in YAML .*'):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_outdir_exists_error(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(f, ["f_bedload"], [[0.2, 0.5, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        # misc assertions
        assert pp._has_matrix is True

        # verify the directory exists
        assert pp._jobs_root == str(tmp_path / "test")
        assert (tmp_path / "test").is_dir()

        # try to create another preprocessor at same directory
        with pytest.raises(FileExistsError, match=r"Job output directory .*"):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_bad_len1(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        # bad configuration is a list of length 1
        utilities.write_matrix_to_file(
            f, ["f_bedload", "u0"], [[0.2], [0.5, 0.6, 1.25]]
        )
        f.close()
        with pytest.warns(
            UserWarning, match=r'Length of matrix key "f_bedload" was 1,'
        ):
            pp = preprocessor.Preprocessor(input_file=p)

        # check job length
        assert pp._has_matrix is True
        assert len(pp.file_list) == 3

    def test_py_hlvl_mtrx_bad_listinlist(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        # bad configuration will lead to error
        utilities.write_matrix_to_file(
            f, ["f_bedload", "u0"], [[0.2, [0.5, 0.6]], [0.5, 1.25]]
        )
        f.close()
        with pytest.raises(
            ValueError, match=r"Depth of matrix expansion must not be > 1"
        ):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_bad_samekey(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "f_bedload", 0.3)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        # bad configuration will lead to error
        utilities.write_matrix_to_file(
            f, ["f_bedload", "u0"], [[0.2, 0.5, 0.6], [0.5, 1.25]]
        )
        f.close()
        with pytest.raises(
            ValueError, match=r"You cannot specify the same key " "in the matrix .*"
        ):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_bad_colon(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        # bad configuration will lead to error
        utilities.write_matrix_to_file(
            f, ["f_bedload", "u0:"], [[0.2, 0.5], [0.5, 1.25]]
        )
        f.close()
        with pytest.raises(
            ValueError, match=r"Colon operator found " "in matrix expansion key."
        ):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_no_out_dir_in_mtrx(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(
            f, ["out_dir", "f_bedload"], [["dir1", "dir2"], [0.2, 0.5, 0.6]]
        )
        f.close()
        with pytest.raises(ValueError, match=r'You cannot specify "out_dir" as .*'):
            _ = preprocessor.Preprocessor(input_file=p, timesteps=3)

    def test_python_highlevelapi_matrix_verbosity(self, tmp_path: Path, capsys) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "verbose", 1)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(
            f, ["f_bedload", "u0"], [[0.2, 0.5, 0.6], [1.5, 2.0]]
        )
        f.close()
        _ = preprocessor.Preprocessor(input_file=p, timesteps=3)

        captd = capsys.readouterr()
        assert "Timestep: 0.0" not in captd.out
        assert "Writing YAML file for job 0" in captd.out
        assert "Writing YAML file for job 1" in captd.out
        assert "Writing YAML file for job 2" in captd.out
        assert "Writing YAML file for job 3" in captd.out
        assert "Writing YAML file for job 4" in captd.out
        assert "Writing YAML file for job 5" in captd.out
        assert "Matrix expansion:" in captd.out
        assert "  dims 2" in captd.out
        assert "  jobs 6" in captd.out


class TestPreprocessorSetJobsSetups:

    def test_py_hlvl_set_two_sets(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_set_to_file(
            f, [{"f_bedload": 0.2, "u0": 1}, {"f_bedload": 0.4, "u0": 2}]
        )
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_set is True
        assert pp._has_matrix is False
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.2 for j in f_bedload_list]) == 1
        assert sum([j == 0.4 for j in f_bedload_list]) == 1
        assert pp.config_dict["timesteps"] == 3

    def test_py_hlvl_set_one_sets(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_set_to_file(f, [{"f_bedload": 0.77, "u0": 1}])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_set is True
        assert pp._has_matrix is False
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.77 for j in f_bedload_list]) == 1
        assert pp.config_dict["timesteps"] == 3

    def test_py_hlvl_set_two_mismatched_sets(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_set_to_file(
            f, [{"f_bedload": 0.2, "u0": 1}, {"h0": 0.4, "u0": 2}]
        )
        f.close()
        with pytest.raises(ValueError, match=r"All keys in all sets *."):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_set_bad_not_list(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        # utilities.write_set_to_file(f, [{'f_bedload': 0.77, 'u0': 1}])
        utilities.write_parameter_to_file(f, "set", "astring!")
        f.close()
        with pytest.raises(TypeError, match=r"Set list must be *."):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_set_bad_colon(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_set_to_file(f, [{"f_bedload:": 0.77, "u0": 1}])
        f.close()
        with pytest.raises(
            ValueError, match=r"Colon operator found " "in matrix expansion key."
        ):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_output_table_correctly(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_set_to_file(
            f, [{"f_bedload": 0.77, "u0": 1, "C0_percent": 0.033}]
        )
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        # now check what the file looks like
        with open(os.path.join(tmp_path, "test", "jobs_parameters.txt"), "r") as tbl:
            Lines = tbl.readlines()

        assert Lines[0] == "job_id, f_bedload, u0, C0_percent\n"


class TestPreprocessorEnsembleJobsSetups:

    def test_py_hlvl_ensemble(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert pp._has_matrix is True
        assert pp._has_ensemble is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2

    def test_py_hlvl_ensemble_1(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 1)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        f.close()
        with pytest.warns(UserWarning, match=r"Ensemble was set to 1. *."):
            pp = preprocessor.Preprocessor(input_file=p)

        # check that keys were reset and config set correctly
        assert pp._has_ensemble is False
        assert pp._has_matrix is False
        assert len(pp.file_list) == 1

    def test_py_hlvl_ensemble_badtype(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2.0)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        f.close()
        with pytest.raises(
            TypeError, match=r"Invalid ensemble type, " "must be an integer."
        ):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_ensemble_double_seeds(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "seed", 1)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        f.close()
        with pytest.raises(
            ValueError,
            match=r"You cannot specify the same key in the "
            "matrix configuration and "  # noqa: E127
            'fixed configuration. Key "seed" was '
            "specified in both.",
        ):
            _ = preprocessor.Preprocessor(input_file=p)


class TestPreprocessorMatrixAndEnsembleJobsSetups:

    def test_py_hlvl_ensemble_with_matrix(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(f, ["f_bedload"], [[0.2, 0.5, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 6

    def test_py_hlvl_ensemble_matrix_seeds(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_matrix_to_file(f, ["seed"], [[1, 2]])
        f.close()
        with pytest.raises(
            ValueError,
            match=r"Random seeds cannot be specified in "
            'the matrix, if an "ensemble" number '  # noqa: E127, E501
            "is specified as well.",
        ):
            _ = preprocessor.Preprocessor(input_file=p, timesteps=3)


class TestPreprocessorSetAndEnsembleJobsSetups:

    def test_1_set_ensemble_3(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "ensemble", 3)
        utilities.write_set_to_file(f, [{"f_bedload": 0.77, "u0": 1}])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_set is True
        assert pp._has_matrix is False
        assert pp._has_ensemble is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 3
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.77 for j in f_bedload_list]) == 3
        assert pp.config_dict["timesteps"] == 3


class TestPreprocessorSetAndMatrixJobsSetups:

    def test_not_possible(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_set_to_file(f, [{"f_bedload": 0.77, "h0": 1}])
        utilities.write_matrix_to_file(f, ["u0"], [[0.2, 0.5, 0.6]])
        f.close()

        with pytest.raises(ValueError, match=r'Cannot use "matrix" and "set" .*'):
            _ = preprocessor.Preprocessor(input_file=p, timesteps=3)


class TestPreprocessorParallelJobsSetups:
    """
    Note that parallel job setup will work on any operating system. This is
    helpful for someone who wants to use the preprocessor to tinker with job
    configurations locally, but will run jobs on a remote Linux cluster/HPC.
    """

    def test_py_hlvl_parallel_works_onejob(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "parallel", True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._has_matrix is False

        assert pp.config_dict["parallel"] is True

    def test_py_hlvl_parallel_boolean_yaml(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "parallel", True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True

        assert pp.config_dict["parallel"] is True

    def test_py_hlvl_parallel_boolean_cli(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, parallel=True)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True

        assert pp.config_dict["parallel"] is True

    def test_py_hlvl_parallel_integer_yaml(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "parallel", 2)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True

        assert pp.config_dict["parallel"] == 2

    def test_py_hlvl_parallel_integer_cli(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, parallel=2)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True
        assert pp._is_completed is False

        assert pp.config_dict["parallel"] == 2


class TestPreprocessorCustomSubclasses:

    def test_subclass_parameter_nowarnings(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_eta_figs": True,
                "timesteps": 2,
                "custom_bool": True,
            },  # custom_bool is not default
        )
        pp = preprocessor.Preprocessor(p)

        class CustomInputs(DeltaModel):
            def __init__(self, input_file=None, **kwargs):
                # inherit base DeltaModel methods
                super().__init__(input_file, **kwargs)

            def hook_import_files(self):
                self.subclass_parameters["custom_bool"] = {
                    "type": "bool",
                    "default": False,
                }

        # patch solver for fast run, no warnings issued
        with mock.patch(
            "pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep"
        ) as ptch:

            # run jobs to initialize deltas
            pp.run_jobs(DeltaModel=CustomInputs)

        # open log and check that True is recorded for custom_bool
        _logs = glob.glob(os.path.join(tmp_path / "out_dir" / "job_000", "*.log"))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], "r") as _logfile:
            _lines = _logfile.readlines()
            _joinedlines = " ".join(_lines)  # collapse to a single string
            assert "`custom_bool`: True" in _joinedlines

    def test_subclass_parameter_warnunused(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_eta_figs": True,
                "timesteps": 2,
                "custom_bool": True,
                "unused_input_yaml": 42,
            },
        )
        pp = preprocessor.Preprocessor(p)

        class CustomInputs(DeltaModel):
            def __init__(self, input_file=None, **kwargs):
                # inherit base DeltaModel methods
                super().__init__(input_file, **kwargs)

            def hook_import_files(self):
                self.subclass_parameters["custom_bool"] = {
                    "type": "bool",
                    "default": False,
                }

        # patch solver for fast run
        with mock.patch(
            "pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep"
        ) as ptch:

            with pytest.warns(
                UserWarning, match=r"One or more .* ['unused_input_yaml']"
            ):
                # run jobs to initialize deltas
                pp.run_jobs(DeltaModel=CustomInputs)

        _logs = glob.glob(os.path.join(tmp_path / "out_dir" / "job_000", "*.log"))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], "r") as _logfile:
            _lines = _logfile.readlines()
            _joinedlines = " ".join(_lines)  # collapse to a single string
            assert "`custom_bool`: True" in _joinedlines


class TestPreprocessorRunJobs:

    def test_dryrun(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp = preprocessor.Preprocessor(p)

        pp._dryrun = True

        # run the method
        pp.run_jobs()

        assert pp._is_completed is False

    def test_run_single_serial_job(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp = preprocessor.Preprocessor(p)

        # patch jobs
        with mock.patch("pyDeltaRCM.preprocessor._SerialJob") as ptch:

            # run the method
            pp.run_jobs()

        assert ptch.call_count == 1
        assert len(pp.job_list) == 1
        assert pp._is_completed is True

    def test_run_single_serial_job_wparams_yaml(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_eta_figs": True, "timesteps": 20}
        )
        pp = preprocessor.Preprocessor(p)

        # patch jobs
        with mock.patch("pyDeltaRCM.preprocessor._SerialJob") as ptch:

            # run the method
            pp.run_jobs()

        assert ptch.call_count == 1
        assert len(pp.job_list) == 1
        assert pp._is_completed is True

        # check that the yaml configs are actually written to the temp directory
        dir_path = Path(tmp_path / "out_dir" / "job_000")
        file_count = sum(1 for item in dir_path.iterdir() if item.is_file())
        assert file_count == 1

    def test_run_single_serial_job_wparams_dict(self, tmp_path: Path) -> None:
        p = {"save_eta_figs": True, "timesteps": 20, "out_dir": tmp_path / "out_dir"}
        pp = preprocessor.Preprocessor(p)

        # patch jobs
        with mock.patch("pyDeltaRCM.preprocessor._SerialJob") as ptch:

            # run the method
            pp.run_jobs()

        assert ptch.call_count == 1
        assert len(pp.job_list) == 1
        assert pp._is_completed is True

        # check that the yaml configs are actually written to the temp directory
        dir_path = Path(tmp_path / "out_dir" / "job_000")
        file_count = sum(1 for item in dir_path.iterdir() if item.is_file())
        assert file_count == 1

    def test_run_single_serial_job_wparams_dict_nooutdir(self, tmp_path: Path) -> None:
        p = {"save_eta_figs": True, "timesteps": 20}
        with pytest.raises(ValueError, match=r'specify "out_dir"'):
            pp = preprocessor.Preprocessor(p)

    def test_run_two_serial_jobs(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp = preprocessor.Preprocessor(p)

        pp._file_list = [*pp.file_list] * 2
        pp._config_list = [*pp.config_list] * 2

        # patch jobs
        with mock.patch("pyDeltaRCM.preprocessor._SerialJob") as ptch:

            # run the method
            pp.run_jobs()

        assert ptch.call_count == 2
        assert len(pp.job_list) == 2
        assert pp._is_completed is True

    def test_serial_batch_bad_job_config_fails_batch(self, tmp_path: Path) -> None:
        # a job whose yaml fails model type validation raises TypeError
        #   during DeltaModel construction; the batch must fail only that
        #   job, still run the remaining jobs, and surface the failure
        #   (and must not mark the preprocessor completed)
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "out_dir": tmp_path / "out_dir",
                "timesteps": 1,
                "matrix": {"Np_water": [10, 2.5]},
            },
        )
        pp = preprocessor.Preprocessor(p)

        with pytest.raises(RuntimeError, match="1 job"):
            pp.run_jobs()

        assert pp._is_completed is False
        # job_000 (valid config) still runs to completion: its log records
        #   successful run and finalize stages
        job0_logs = list(Path(tmp_path / "out_dir" / "job_000").glob("*.log"))
        assert len(job0_logs) == 1
        job0_log = job0_logs[0].read_text()
        assert "job: 0, stage: 1, code: 0" in job0_log
        assert "job: 0, stage: 2, code: 0" in job0_log

    def test_serial_job_releases_log_handler_after_run(
        self, tmp_path: Path
    ) -> None:
        """Test that a serial job releases the model's log handler.

        The job defers the log handler release out of ``finalize`` so its
        stage reporting reaches the log file, and must then release the
        handler itself once the final stage message is written -- exactly
        what a standalone model gets from ``finalize`` -- so batch runs do
        not accumulate open file handles.
        """
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"out_dir": tmp_path / "out_dir", "timesteps": 1},
        )
        pp = preprocessor.Preprocessor(p)
        pp.run_jobs()

        job = pp.job_list[0]
        # the deferred release happened after the final stage message
        assert job.deltamodel._log_fh is None
        _fhs = [
            h
            for h in job.deltamodel.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert _fhs == []

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_parallel_bad_job_config_fails_batch(self, tmp_path: Path) -> None:
        # a job whose yaml fails model type validation must not abort the
        #   batch (nor leak an acquired semaphore unit); the batch runs
        #   the remaining jobs and surfaces the failure
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "out_dir": tmp_path / "out_dir",
                "timesteps": 1,
                "matrix": {"Np_water": [10, 2.5]},
            },
        )
        pp = preprocessor.Preprocessor(p, parallel=True)

        with pytest.raises(RuntimeError, match="1 job"):
            pp.run_jobs()

        assert pp._is_completed is False

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_run_five_parallel_jobs_bool(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp = preprocessor.Preprocessor(p, parallel=True)

        # expand the list to include multiples
        pp._file_list = [*pp.file_list] * 5
        pp._config_list = [*pp.config_list] * 5

        # patch jobs and semaphore to check number of processes called
        with (
            mock.patch("pyDeltaRCM.preprocessor._ParallelJob") as ptch,
            mock.patch("multiprocessing.Semaphore") as sem,
        ):

            # run the method
            pp.run_jobs()

        # assertions
        assert ptch.call_count == 5
        # can only assert sem was called:
        #   number called with will depend on number of cpu cores)
        sem.assert_called()
        assert len(pp.job_list) == 5
        assert pp._is_completed is True

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_run_five_parallel_jobs_int_two(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp = preprocessor.Preprocessor(p, parallel=2)

        # expand the list to include multiples
        pp._file_list = [*pp.file_list] * 5
        pp._config_list = [*pp.config_list] * 5

        # patch jobs and semaphore to check number of processes called
        with (
            mock.patch("pyDeltaRCM.preprocessor._ParallelJob") as ptch,
            mock.patch("multiprocessing.Semaphore") as sem,
        ):

            # run the method
            pp.run_jobs()

        # assertions
        assert ptch.call_count == 5
        sem.assert_called_with(2)
        assert len(pp.job_list) == 5
        assert pp._is_completed is True

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_run_five_parallel_jobs_bad_type_string(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp_str = preprocessor.Preprocessor(p, parallel="string!")
        with (
            mock.patch("pyDeltaRCM.preprocessor._ParallelJob"),
            mock.patch("multiprocessing.Semaphore"),
        ):
            with pytest.raises(ValueError, match=r"Parallel flag *."):
                pp_str.run_jobs()

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_run_five_parallel_jobs_bad_type_float(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp_float = preprocessor.Preprocessor(p, parallel=3.33)
        with (
            mock.patch("pyDeltaRCM.preprocessor._ParallelJob"),
            mock.patch("multiprocessing.Semaphore"),
        ):
            with pytest.raises(ValueError, match=r"Parallel flag *."):
                pp_float.run_jobs()

    @pytest.mark.skipif(
        platform.system() == "Linux", reason="Parallel support only on Linux OS."
    )
    def test_run_parallel_notimplemented_nonlinux(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        pp = preprocessor.Preprocessor(p, parallel=True)

        # patch jobs and semaphore to check number of processes called
        with pytest.raises(NotImplementedError):

            # run the method
            pp.run_jobs()


class TestPreprocessorConsoleRouting:
    """Console-routing contracts for the preprocessor's marked records.

    The preprocessor logs to a namespaced, non-propagating logger with a
    temporary console handler; these tests pin the visible stdout contract
    (spec verification row 5) and that status records flow through the
    logging framework.
    """

    def test_serial_run_jobs_stdout_empty_at_verbose_zero(
        self, tmp_path: Path, capsys
    ) -> None:
        # every preprocessor status message is gated at verbosity >= 1,
        #   and a successful serial batch emits no failure statuses, so
        #   stdout must be empty at verbose=0
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"out_dir": tmp_path / "out_dir", "timesteps": 1, "verbose": 0},
        )
        pp = preprocessor.Preprocessor(p)

        pp.run_jobs()

        assert pp._is_completed is True
        assert capsys.readouterr().out == ""

    def test_build_failure_status_console_visible_once_at_verbose_zero(
        self, tmp_path: Path, capsys
    ) -> None:
        # the job-build failure status line is ungated: it must reach
        #   stdout exactly once even at verbose=0, with unchanged text
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "out_dir": tmp_path / "out_dir",
                "timesteps": 1,
                "verbose": 0,
                "matrix": {"Np_water": [10, 2.5]},
            },
        )
        pp = preprocessor.Preprocessor(p)

        with pytest.raises(RuntimeError, match="1 job"):
            pp.run_jobs()

        out = capsys.readouterr().out
        assert out.count("job: 1, stage: 0, code: 1") == 1

    def test_starting_job_status_console_visible_once_at_verbose_one(
        self, tmp_path: Path, capsys
    ) -> None:
        # the gated "Starting job" status is console-visible at verbosity
        #   >= 1 and appears exactly once per job; the model's own Time
        #   lines may also appear, so only the status text is counted
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"out_dir": tmp_path / "out_dir", "timesteps": 1, "verbose": 1},
        )
        pp = preprocessor.Preprocessor(p)

        pp.run_jobs()

        out = capsys.readouterr().out
        assert out.count("Starting job 0") == 1

    def test_preprocessor_records_flow_through_logging(
        self, tmp_path: Path, capsys
    ) -> None:
        # propagation is disabled by design (library safety), so records
        #   cannot be observed through the root logger or caplog; attach
        #   a capturing handler to the preprocessor logger directly and
        #   assert the status records flow through the logging framework
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"out_dir": tmp_path / "out_dir", "timesteps": 1, "verbose": 1},
        )
        pp = preprocessor.Preprocessor(p)

        records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        capture = _CaptureHandler(level=logging.DEBUG)
        logger = logging.getLogger("pyDeltaRCM.preprocessor")
        logger.addHandler(capture)
        try:
            pp.run_jobs()
        finally:
            logger.removeHandler(capture)

        starting = [r for r in records if r.getMessage() == "Starting job 0"]
        assert len(starting) == 1
        assert starting[0].name == "pyDeltaRCM.preprocessor"
        assert starting[0].levelno == logging.INFO
        # the same record was console-rendered with the bare formatter
        assert "Starting job 0" in capsys.readouterr().out


@mock.patch.multiple(pyDeltaRCM.preprocessor._BaseJob, __abstractmethods__=set())
class TestBaseJob:
    """
    To test the BaseJob, we patch the base job with a filled abstract method
    `.run()`.

    .. note:: This patch is handled at the class level above!!
    """

    def test_wo_timesteps(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        with pytest.raises(ValueError, match=r"You must specify a run duration *."):
            _ = preprocessor._BaseJob(i=0, input_file=p, config_dict={})

    def test_timeargs_precedence_tstepsovertime(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        basej = preprocessor._BaseJob(
            i=0, input_file=p, config_dict={"time": 10000, "timesteps": 2}
        )

        assert isinstance(basej.deltamodel, DeltaModel)
        assert basej._job_end_time == basej.deltamodel._dt * 2
        assert basej._job_end_time != 10000

    def test_timeargs_precedence_tstepsovertimeyears(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        basej = preprocessor._BaseJob(
            i=0, input_file=p, config_dict={"time_years": 10000, "timesteps": 2}
        )

        assert isinstance(basej.deltamodel, DeltaModel)
        assert basej._job_end_time == basej.deltamodel._dt * 2
        assert basej._job_end_time != 10000
        assert basej._job_end_time != 10000 * 365.25

    def test_timeargs_precedence_timeovertimeyears(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        basej = preprocessor._BaseJob(
            i=0, input_file=p, config_dict={"time_years": 10000, "time": 900}
        )

        assert isinstance(basej.deltamodel, DeltaModel)
        assert basej._job_end_time == 900
        assert basej._job_end_time != 10000
        assert basej._job_end_time != 10000 * 365.25


class TestSerialJob:

    def test_run(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        sj = preprocessor._SerialJob(i=0, input_file=p, config_dict={"timesteps": 10})

        # modify the save interval to be twice dt
        sj.deltamodel._save_dt = 2 * sj.deltamodel._dt
        sj.deltamodel._checkpoint_dt = 2 * sj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        sj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        sj.deltamodel.apply_subsidence = mock.MagicMock()
        sj.deltamodel.finalize_timestep = mock.MagicMock()
        sj.deltamodel.log_model_time = mock.MagicMock()
        sj.deltamodel.logger = mock.MagicMock()
        sj.deltamodel.output_data = mock.MagicMock()
        sj.deltamodel.output_checkpoint = mock.MagicMock()
        sj.deltamodel.finalize = mock.MagicMock()

        # run the method (call update 10 times)
        sj.run()

        # assertions
        assert sj.deltamodel._time_iter == 10
        assert sj.deltamodel.output_data.call_count == 10
        assert sj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert sj.deltamodel.apply_subsidence.call_count == 10
        assert sj.deltamodel.finalize_timestep.call_count == 10
        assert sj.deltamodel.log_model_time.call_count == 10
        assert sj.deltamodel.output_checkpoint.call_count == 10
        assert sj.deltamodel.finalize.call_count == 1

        # check the log outputs for successes
        _calls = [
            mock.call("job: 0, stage: 1, code: 0"),
            mock.call("job: 0, stage: 2, code: 0"),
        ]
        sj.deltamodel.logger.info.assert_has_calls(_calls)

    def test_run_error_update(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        sj = preprocessor._SerialJob(i=99, input_file=p, config_dict={"timesteps": 10})

        # modify the save interval to be twice dt
        sj.deltamodel._save_dt = 2 * sj.deltamodel._dt
        sj.deltamodel._checkpoint_dt = 2 * sj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        sj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        sj.deltamodel.apply_subsidence = mock.MagicMock()
        sj.deltamodel.finalize_timestep = mock.MagicMock()
        sj.deltamodel.log_model_time = mock.MagicMock()
        sj.deltamodel.logger = mock.MagicMock()
        sj.deltamodel.output_data = mock.MagicMock()
        sj.deltamodel.output_checkpoint = mock.MagicMock(
            side_effect=RuntimeError("error!")
        )
        sj.deltamodel.finalize = mock.MagicMock()

        # run the method (error on `output_checkpoint`)
        sj.run()

        # assertions
        assert sj.deltamodel._time_iter == 1
        assert sj.deltamodel.solve_water_and_sediment_timestep.call_count == 1
        assert sj.deltamodel.apply_subsidence.call_count == 1
        assert sj.deltamodel.finalize_timestep.call_count == 1
        assert sj.deltamodel.log_model_time.call_count == 1
        assert sj.deltamodel.output_data.call_count == 1
        assert sj.deltamodel.output_checkpoint.call_count == 1
        assert sj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [mock.call("job: 99, stage: 2, code: 0")]
        _error_calls = [mock.call("job: 99, stage: 1, code: 1, msg: error!")]
        sj.deltamodel.logger.info.assert_has_calls(_info_calls)
        sj.deltamodel.logger.error.assert_has_calls(_error_calls)
        sj.deltamodel.logger.exception.assert_called_once()

    def test_run_error_finalize(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        sj = preprocessor._SerialJob(i=0, input_file=p, config_dict={"timesteps": 10})

        # modify the save interval to be twice dt
        sj.deltamodel._save_dt = 2 * sj.deltamodel._dt
        sj.deltamodel._checkpoint_dt = 2 * sj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        sj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        sj.deltamodel.apply_subsidence = mock.MagicMock()
        sj.deltamodel.finalize_timestep = mock.MagicMock()
        sj.deltamodel.log_model_time = mock.MagicMock()
        sj.deltamodel.logger = mock.MagicMock()
        sj.deltamodel.output_data = mock.MagicMock()
        sj.deltamodel.output_checkpoint = mock.MagicMock()
        sj.deltamodel.finalize = mock.MagicMock(side_effect=RuntimeError("error!"))

        # run the method (error on `output_checkpoint`)
        sj.run()

        # assertions
        assert sj.deltamodel._time_iter == 10
        assert sj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert sj.deltamodel.apply_subsidence.call_count == 10
        assert sj.deltamodel.finalize_timestep.call_count == 10
        assert sj.deltamodel.log_model_time.call_count == 10
        assert sj.deltamodel.output_data.call_count == 10
        assert sj.deltamodel.output_checkpoint.call_count == 10
        assert sj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [mock.call("job: 0, stage: 1, code: 0")]
        _error_calls = [mock.call("job: 0, stage: 2, code: 1, msg: error!")]
        sj.deltamodel.logger.info.assert_has_calls(_info_calls)
        sj.deltamodel.logger.error.assert_has_calls(_error_calls)
        sj.deltamodel.logger.exception.assert_called_once()


class TestParallelJob:

    def test_run(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        Q = mock.MagicMock()
        S = mock.MagicMock()

        pj = preprocessor._ParallelJob(
            i=0, queue=Q, sema=S, input_file=p, config_dict={"timesteps": 10}
        )

        # modify the save interval to be twice dt
        pj.deltamodel._save_dt = 2 * pj.deltamodel._dt
        pj.deltamodel._checkpoint_dt = 2 * pj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        pj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        pj.deltamodel.apply_subsidence = mock.MagicMock()
        pj.deltamodel.finalize_timestep = mock.MagicMock()
        pj.deltamodel.log_model_time = mock.MagicMock()
        pj.deltamodel.logger = mock.MagicMock()
        pj.deltamodel.output_data = mock.MagicMock()
        pj.deltamodel.output_checkpoint = mock.MagicMock()
        pj.deltamodel.finalize = mock.MagicMock()

        # mock initialization and their hooks
        pj.deltamodel.init_output_file = mock.MagicMock()
        pj.deltamodel.hook_init_output_file = mock.MagicMock()
        pj.deltamodel.hook_output_data = mock.MagicMock()
        pj.deltamodel.hook_output_checkpoint = mock.MagicMock()

        # run the method (call update 10 times)
        pj.run()

        # assertions
        assert pj.deltamodel._time_iter == 10
        assert pj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert pj.deltamodel.apply_subsidence.call_count == 10
        assert pj.deltamodel.finalize_timestep.call_count == 10
        assert pj.deltamodel.log_model_time.call_count == 10
        assert pj.deltamodel.output_data.call_count == 11  # once in init
        assert pj.deltamodel.output_checkpoint.call_count == 11  # once in init
        assert pj.deltamodel.finalize.call_count == 1
        # hooks/initialization
        assert pj.deltamodel.init_output_file.call_count == 1
        assert pj.deltamodel.hook_init_output_file.call_count == 1
        assert pj.deltamodel.hook_output_data.call_count == 11
        assert pj.deltamodel.hook_output_checkpoint.call_count == 11

        # check the log outputs for successes
        _calls = [
            mock.call({"job": 0, "stage": 0, "code": 0}),
            mock.call({"job": 0, "stage": 1, "code": 0}),
            mock.call({"job": 0, "stage": 2, "code": 0}),
        ]
        Q.put.assert_has_calls(_calls)

    def test_run_error_update(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        Q = mock.MagicMock()
        S = mock.MagicMock()

        pj = preprocessor._ParallelJob(
            i=99, queue=Q, sema=S, input_file=p, config_dict={"timesteps": 10}
        )

        # modify the save interval to be twice dt
        pj.deltamodel._save_dt = 2 * pj.deltamodel._dt
        pj.deltamodel._checkpoint_dt = 2 * pj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        pj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        pj.deltamodel.apply_subsidence = mock.MagicMock()
        pj.deltamodel.finalize_timestep = mock.MagicMock()
        pj.deltamodel.log_model_time = mock.MagicMock(
            side_effect=RuntimeError("error!")
        )
        pj.deltamodel.logger = mock.MagicMock()
        pj.deltamodel.output_data = mock.MagicMock()
        pj.deltamodel.output_checkpoint = mock.MagicMock()
        pj.deltamodel.finalize = mock.MagicMock()

        # run the method (error on `output_checkpoint`)
        pj.run()

        # assertions
        assert pj.deltamodel._time_iter == 1
        assert pj.deltamodel.solve_water_and_sediment_timestep.call_count == 1
        assert pj.deltamodel.apply_subsidence.call_count == 1
        assert pj.deltamodel.finalize_timestep.call_count == 1
        assert pj.deltamodel.log_model_time.call_count == 1
        assert pj.deltamodel.output_data.call_count == 1
        assert pj.deltamodel.output_checkpoint.call_count == 1
        assert pj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [
            mock.call({"job": 99, "stage": 0, "code": 0}),
            mock.call({"job": 99, "stage": 2, "code": 0}),
        ]
        _error_calls = [mock.call({"job": 99, "stage": 1, "code": 1, "msg": "error!"})]
        Q.put.assert_has_calls(_info_calls, any_order=True)
        Q.put.assert_has_calls(_error_calls)
        pj.deltamodel.logger.error.assert_has_calls([mock.call("error!")])
        pj.deltamodel.logger.exception.assert_called_once()

    def test_run_error_finalize(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")

        Q = mock.MagicMock()
        S = mock.MagicMock()

        pj = preprocessor._ParallelJob(
            i=0, queue=Q, sema=S, input_file=p, config_dict={"timesteps": 10}
        )

        # modify the save interval to be twice dt
        pj.deltamodel._save_dt = 2 * pj.deltamodel._dt
        pj.deltamodel._checkpoint_dt = 2 * pj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        pj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        pj.deltamodel.apply_subsidence = mock.MagicMock()
        pj.deltamodel.finalize_timestep = mock.MagicMock()
        pj.deltamodel.log_model_time = mock.MagicMock()
        pj.deltamodel.logger = mock.MagicMock()
        pj.deltamodel.output_data = mock.MagicMock()
        pj.deltamodel.output_checkpoint = mock.MagicMock()
        pj.deltamodel.finalize = mock.MagicMock(side_effect=RuntimeError("error!"))

        # run the method (error on `output_checkpoint`)
        pj.run()

        # assertions
        assert pj.deltamodel._time_iter == 10
        assert pj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert pj.deltamodel.apply_subsidence.call_count == 10
        assert pj.deltamodel.finalize_timestep.call_count == 10
        assert pj.deltamodel.log_model_time.call_count == 10
        assert pj.deltamodel.output_data.call_count == 11  # once in init
        assert pj.deltamodel.output_checkpoint.call_count == 11  # once in init
        assert pj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [
            mock.call({"job": 0, "stage": 0, "code": 0}),
            mock.call({"job": 0, "stage": 1, "code": 0}),
        ]
        _error_calls = [mock.call({"job": 0, "stage": 2, "code": 1, "msg": "error!"})]
        Q.put.assert_has_calls(_info_calls, any_order=True)
        Q.put.assert_has_calls(_error_calls)
        pj.deltamodel.logger.error.assert_has_calls([mock.call("error!")])
        pj.deltamodel.logger.exception.assert_called_once()


class TestWriteYamlConfigToFile:

    def test_write_single_int(self, tmp_path: Path) -> None:
        # set up what to write
        file_path = tmp_path / "output.yml"
        yaml_dict = {"variable": 1}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = " ".join(f.readlines())

        assert _returned == "variable: 1\n"

    def test_write_single_float(self, tmp_path: Path) -> None:
        # set up what to write
        file_path = tmp_path / "output.yml"
        yaml_dict = {"variable": 1.5}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = " ".join(f.readlines())

        assert _returned == "variable: 1.5\n"

    def test_write_single_string(self, tmp_path: Path) -> None:
        # set up what to write
        file_path = tmp_path / "output.yml"
        yaml_dict = {"variable": "a string"}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = " ".join(f.readlines())

        assert _returned == "variable: a string\n"

    def test_write_single_bool(self, tmp_path: Path) -> None:
        # set up what to write
        file_path = tmp_path / "output.yml"
        yaml_dict = {"variable": False}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = " ".join(f.readlines())

        assert _returned == "variable: False\n"

    def test_write_single_null(self, tmp_path: Path) -> None:
        # set up what to write
        file_path = tmp_path / "output.yml"
        yaml_dict = {"variable": None}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = " ".join(f.readlines())

        assert _returned == "variable: null\n"

    def test_write_multiple_one_each(self, tmp_path: Path) -> None:
        # set up what to write
        file_path = tmp_path / "output.yml"
        yaml_dict = {
            "variable1": 1,
            "variable2": 2.2,
            "variable3": "a string",
            "variable4": False,
            "variable5": None,
        }

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = " ".join(f.readlines())
        print(_returned)

        assert _returned == (
            "variable1: 1\n variable2: 2.2\n "
            "variable3: a string\n variable4: False\n "
            "variable5: null\n"
        )


class TestPreprocessorImported:

    def test_Preprocessor_toplevelimport(self) -> None:
        import pyDeltaRCM

        assert "Preprocessor" in dir(pyDeltaRCM)
        assert pyDeltaRCM.Preprocessor is pyDeltaRCM.preprocessor.Preprocessor


class TestScaleRelativeSeaLeveLRiseRate:

    def test_scale_If_1(self) -> None:
        scaled = preprocessor.scale_relative_sea_level_rise_rate(5, If=1)
        assert scaled == 5 / 1000 / 365.25 / 86400

    def test_scale_If_0p1(self) -> None:
        scaled = preprocessor.scale_relative_sea_level_rise_rate(5, If=0.1)
        assert scaled == 5 / 1000 / 365.25 / 86400 * 10


class TestQolUtilityFlags:
    """Tests for the --new, --validate, and --demo utility flags."""

    # --new

    def test_scaffold_writes_file_with_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "starter.yml"
        preprocessor._write_scaffold_config(p)

        assert p.is_file()
        with open(p, mode="r") as f:
            loaded = yaml.load(f, Loader=custom_yaml_loader())
        schema = preprocessor._load_default_schema()

        # every uncommented key is a known schema key or a duration key
        for k in loaded.keys():
            assert k in schema or k == "timesteps"
        # the scaffold is instantly runnable: a duration is set
        assert loaded["timesteps"] == 10
        # null-default keys are commented out, because explicit null values
        # are rejected by the model's type check; all other defaults emitted
        for k, v in schema.items():
            if v["default"] is None:
                assert k not in loaded
            else:
                assert loaded[k] == v["default"]

    def test_scaffold_validates_clean(self, tmp_path: Path) -> None:
        p = tmp_path / "starter.yml"
        preprocessor._write_scaffold_config(p)

        with open(p, mode="r") as f:
            loaded = yaml.load(f, Loader=custom_yaml_loader())
        issues = preprocessor._validate_config_dict(
            loaded, preprocessor._load_default_schema()
        )

        errors = [_issue for _issue in issues if _issue[0] == "error"]
        assert errors == []

    def test_scaffold_refuses_overwrite(self, tmp_path: Path) -> None:
        p = tmp_path / "starter.yml"
        p.write_text("existing: file\n")

        with pytest.raises(FileExistsError):
            preprocessor._write_scaffold_config(p)

    # --validate

    def test_validate_valid_config(self) -> None:
        cfg = {"out_dir": "out", "seed": 0, "timesteps": 5}

        issues = preprocessor._validate_config_dict(
            cfg, preprocessor._load_default_schema()
        )

        assert issues == []

    def test_validate_bad_type(self) -> None:
        cfg = {"Np_sed": "many", "timesteps": 5}

        issues = preprocessor._validate_config_dict(
            cfg, preprocessor._load_default_schema()
        )

        errors = [m for lvl, m in issues if lvl == "error"]
        assert len(errors) == 1
        assert "Np_sed" in errors[0]

    def test_validate_unknown_key(self) -> None:
        cfg = {"banana": 3, "timesteps": 5}

        issues = preprocessor._validate_config_dict(
            cfg, preprocessor._load_default_schema()
        )

        errors = [m for lvl, m in issues if lvl == "error"]
        assert len(errors) == 1
        assert "banana" in errors[0]

    def test_validate_explicit_null_rejected_like_model(self) -> None:
        # mirrors the model: an explicit null raises a TypeError at model
        # initialization, even for keys whose declared types include 'None'
        cfg = {"seed": None, "timesteps": 5}

        issues = preprocessor._validate_config_dict(
            cfg, preprocessor._load_default_schema()
        )

        errors = [m for lvl, m in issues if lvl == "error"]
        assert len(errors) == 1
        assert "seed" in errors[0]

    def test_validate_missing_duration_warns(self) -> None:
        cfg = {"seed": 0}

        issues = preprocessor._validate_config_dict(
            cfg, preprocessor._load_default_schema()
        )

        errors = [m for lvl, m in issues if lvl == "error"]
        warnings_ = [m for lvl, m in issues if lvl == "warning"]
        assert errors == []
        assert len(warnings_) == 1
        assert "duration" in warnings_[0]

    def test_validate_expansion_contents_not_checked(self) -> None:
        # v1 scope: the fixed configuration is validated, expansion
        # internals are only flagged with a warning
        cfg = {"matrix": {"banana": [1, 2]}, "timesteps": 5}

        issues = preprocessor._validate_config_dict(
            cfg, preprocessor._load_default_schema()
        )

        errors = [m for lvl, m in issues if lvl == "error"]
        warnings_ = [m for lvl, m in issues if lvl == "warning"]
        assert errors == []
        assert any("matrix" in w for w in warnings_)

    def test_validate_unknown_type_name_raises(self) -> None:
        # the schema's "type" metadata must resolve through the same
        # allowlist the model uses (see #19's resolver); an unknown name
        # raises a clear error instead of being evaluated
        schema = preprocessor._load_default_schema()
        schema["out_dir"] = dict(schema["out_dir"], type="not_a_real_type")

        with pytest.raises(ValueError, match="Unknown or disallowed type name"):
            preprocessor._validate_config_dict(
                {"out_dir": "out", "timesteps": 5}, schema
            )

    def test_validate_type_metadata_never_executes(self, tmp_path: Path) -> None:
        # a code payload in the schema's "type" field must fail through the
        # allowlist resolver without executing, as the model's own path does
        _marker = tmp_path / "pd3_should_not_exist"
        schema = preprocessor._load_default_schema()
        schema["out_dir"] = dict(
            schema["out_dir"],
            type=f"__import__('os').system('touch {_marker}')",
        )

        with pytest.raises(ValueError, match="Unknown or disallowed type name"):
            preprocessor._validate_config_dict(
                {"out_dir": "out", "timesteps": 5}, schema
            )

        assert not _marker.exists()

    # short-circuit wiring through the CLI parser

    def test_cli_new_flag_short_circuit(self, tmp_path: Path) -> None:
        p = tmp_path / "cli_new.yml"

        with mock.patch("sys.argv", ["pyDeltaRCM", "--new", str(p)]):
            with pytest.raises(SystemExit) as e:
                _ = preprocessor.PreprocessorCLI()

        assert e.value.code == 0
        assert p.is_file()

    def test_cli_validate_flag_short_circuit_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.yml"
        p.write_text("out_dir: somewhere\nseed: 0\ntimesteps: 5\n")

        with mock.patch("sys.argv", ["pyDeltaRCM", "--validate", str(p)]):
            with pytest.raises(SystemExit) as e:
                _ = preprocessor.PreprocessorCLI()

        assert e.value.code == 0

    def test_cli_validate_flag_short_circuit_invalid(self, tmp_path: Path) -> None:
        p = tmp_path / "invalid.yml"
        p.write_text("banana: 3\ntimesteps: 5\n")

        with mock.patch("sys.argv", ["pyDeltaRCM", "--validate", str(p)]):
            with pytest.raises(SystemExit) as e:
                _ = preprocessor.PreprocessorCLI()

        assert e.value.code == 1

    # --demo (the run path itself is covered by the integration tests)

    def test_run_demo_writes_config_and_runs_via_preprocessor(
        self, tmp_path: Path
    ) -> None:
        with mock.patch.object(preprocessor, "Preprocessor", autospec=True) as mock_pp:
            mock_pp.return_value.file_list = [
                tmp_path / "pyDeltaRCM_demo_output" / "job_000" / "job_000.yml"
            ]
            preprocessor._run_demo(tmp_path / "demo")

        assert (tmp_path / "demo" / "pyDeltaRCM_demo.yaml").is_file()
        assert mock_pp.call_args.kwargs["input_file"] == str(
            tmp_path / "demo" / "pyDeltaRCM_demo.yaml"
        )
        mock_pp.return_value.run_jobs.assert_called_once()



class TestDegenerateConfigErrors:
    """Configurations that used to crash with bare ``TypeError`` /
    ``numpy`` errors far from the cause must be rejected with a
    descriptive typed error instead."""

    def test_empty_yaml_file_model(self, tmp_path: Path, monkeypatch) -> None:
        # an empty (or comment-only) YAML file parsed to None and crashed
        #   the direct-model API with a bare TypeError/AttributeError
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "empty.yaml"
        p.write_text("# only a comment\n")
        with pytest.warns(UserWarning, match="[Ee]mpty"):
            _delta = DeltaModel(input_file=p)

    def test_empty_yaml_file_preprocessor(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # an empty configuration plus the CLI duration, with out_dir from
        #   the command line, as a batch driver would pass it
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.warns(UserWarning, match="[Ee]mpty"):
            pp = preprocessor.Preprocessor(
                input_file=p, timesteps=1, out_dir=str(tmp_path / "jobs")
            )
        assert len(pp.file_list) == 1

    def test_empty_matrix_rejected(self, tmp_path: Path) -> None:
        # matrix: {} expanded to np.prod([]) == 1.0 (a float) and crashed
        #   with a numpy TypeError
        p, f = utilities.create_temporary_file(tmp_path, "matrix.yaml")
        f.write("out_dir: " + str(tmp_path / "jobs") + "\n")
        f.write("matrix: {}\n")
        f.close()
        with pytest.raises(ValueError, match="matrix"):
            preprocessor.Preprocessor(input_file=p, timesteps=1)

    def test_scalar_matrix_value_rejected(self, tmp_path: Path) -> None:
        # a scalar matrix value crashed inside len() with a TypeError
        #   pointing at the wrong thing
        p, f = utilities.create_temporary_file(tmp_path, "matrix.yaml")
        f.write("out_dir: " + str(tmp_path / "jobs") + "\n")
        f.write("matrix:\n  f_bedload: 0.2\n")
        f.close()
        with pytest.raises(ValueError, match="f_bedload"):
            preprocessor.Preprocessor(input_file=p, timesteps=1)


class TestRunDurationValidation:
    """If outside (0, 1] and non-positive durations silently produced
    zero-step "successful" runs; they must be rejected."""

    def test_If_zero_rejected(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"time": 86400.0})
        with pytest.raises(ValueError, match=r"\bIf\b"):
            preprocessor._SerialJob(
                i=0, input_file=p, config_dict={"If": 0, "time": 86400.0}
            )

    def test_If_above_one_rejected(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"time": 86400.0})
        with pytest.raises(ValueError, match=r"\bIf\b"):
            preprocessor._SerialJob(
                i=0, input_file=p, config_dict={"If": 1.5, "time": 86400.0}
            )

    def test_negative_timesteps_rejected(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        with pytest.raises(ValueError, match="timesteps"):
            preprocessor._SerialJob(
                i=0, input_file=p, config_dict={"timesteps": -5}
            )

    def test_zero_time_rejected(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        with pytest.raises(ValueError, match="time"):
            preprocessor._SerialJob(i=0, input_file=p, config_dict={"time": 0})

    def test_negative_time_years_rejected(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        with pytest.raises(ValueError, match="time_years"):
            preprocessor._SerialJob(
                i=0, input_file=p, config_dict={"time_years": -0.5}
            )


class TestDurationPrecedenceWarning:
    """When multiple run-duration keys are specified, ``timesteps`` takes
    precedence (checked first in ``_BaseJob.__init__``); the effective
    precedence must be announced instead of applied silently, and the CLI
    help text must match the behavior."""

    def test_timesteps_and_time_both_specified_warns(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"timesteps": 2})
        with pytest.warns(UserWarning, match="timesteps"):
            job = preprocessor._SerialJob(
                i=0, input_file=p, config_dict={"timesteps": 2, "time": 8640000.0}
            )
        assert job._time_type == "timesteps"

    def test_single_duration_key_does_not_warn(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"time": 86400.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            job = preprocessor._SerialJob(
                i=0, input_file=p, config_dict={"time": 86400.0}
            )
        assert job._time_type == "time"


class TestStringResumeCheckpoint:
    """``resume_checkpoint`` accepts a string path, but the preprocessor's
    directory-exists guards used ``is True`` identity checks, so resuming
    with a custom checkpoint folder raised ``FileExistsError`` in exactly
    the scenario where resuming is intended."""

    def test_string_resume_checkpoint_allows_existing_job_dir(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "jobs"
        out.mkdir()
        ckp = tmp_path / "ckp"
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"out_dir": out, "timesteps": 1, "resume_checkpoint": str(ckp)},
        )
        pp = preprocessor.Preprocessor(input_file=p)
        assert len(pp.file_list) == 1

    def test_bool_resume_checkpoint_still_allows_existing_job_dir(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "jobs"
        out.mkdir()
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"out_dir": out, "timesteps": 1, "resume_checkpoint": True},
        )
        pp = preprocessor.Preprocessor(input_file=p)
        assert len(pp.file_list) == 1

    def test_no_resume_still_rejects_existing_job_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "jobs"
        out.mkdir()
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"out_dir": out, "timesteps": 1}
        )
        with pytest.raises(FileExistsError):
            preprocessor.Preprocessor(input_file=p)
