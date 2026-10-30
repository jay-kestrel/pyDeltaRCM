# unit tests for iteration_tools.py

# unit tests for iteration_tools.py

import pytest
from pathlib import Path

import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from . import utilities


class TestSolveWaterAndSedimentTimestep:
    def test_solve_water_and_sediment_timestep_defaults(self, tmp_path: Path) -> None:
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        delta = DeltaModel(input_file=p)

        # mock top-level methods, verify call was made to each
        delta.log_info = mock.MagicMock()
        delta.init_water_iteration = mock.MagicMock()
        delta.run_water_iteration = mock.MagicMock()
        delta.compute_free_surface = mock.MagicMock()
        delta.finalize_water_iteration = mock.MagicMock()
        delta.route_sediment = mock.MagicMock()

        # run the timestep
        delta.solve_water_and_sediment_timestep()

        # assert that methods are called
        assert delta.init_water_iteration.called is True
        assert delta.run_water_iteration.called is True
        assert delta.compute_free_surface.called is True
        assert delta.finalize_water_iteration.called is True
        _calls = [mock.call(0), mock.call(1), mock.call(2)]
        delta.finalize_water_iteration.assert_has_calls(_calls, any_order=False)
        assert delta.finalize_water_iteration.call_count == 3
        assert delta.route_sediment.called is True
        assert delta._is_finalized is False

    def test_solve_water_and_sediment_timestep_itermax_10(self, tmp_path: Path) -> None:
        # create a delta with different itermax
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"itermax": 10})
        delta = DeltaModel(input_file=p)

        # mock top-level methods, verify call was made to each
        delta.log_info = mock.MagicMock()
        delta.init_water_iteration = mock.MagicMock()
        delta.run_water_iteration = mock.MagicMock()
        delta.compute_free_surface = mock.MagicMock()
        delta.finalize_water_iteration = mock.MagicMock()
        delta.route_sediment = mock.MagicMock()

        # run the timestep
        delta.solve_water_and_sediment_timestep()

        # assert that methods are called
        assert delta.init_water_iteration.called is True
        assert delta.run_water_iteration.called is True
        assert delta.compute_free_surface.called is True
        assert delta.finalize_water_iteration.called is True
        _calls = [mock.call(i) for i in range(10)]
        delta.finalize_water_iteration.assert_has_calls(_calls, any_order=False)
        assert delta.finalize_water_iteration.call_count == 10
        assert delta.route_sediment.called is True
        assert delta._is_finalized is False


class TestFinalizeTimestep:
    def test_finalize_timestep(self, tmp_path: Path) -> None:
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"SLR": 0.001})
        delta = DeltaModel(input_file=p)

        # mock the flooding correction and log
        delta.log_info = mock.MagicMock()
        delta.flooding_correction = mock.MagicMock()

        # run the step
        delta.finalize_timestep()

        # assert submethod called once
        delta.flooding_correction.call_count == 1

        # check that sea level rose as expected
        assert delta.H_SL == 25


class TestApplyingSubsidence:
    def test_subsidence_in_update(self, tmp_path: Path) -> None:
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "toggle_subsidence": True,
                "subsidence_rate": 1e-8,
                "start_subsidence": 0,
                "save_eta_grids": True,
                "seed": 0,
            },
        )
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.subsidence_rate == 1e-8
        assert np.all(_delta.sigma[: _delta.L0, :] == 0.0)  # outside the sigma mask
        assert np.all(_delta.sigma[_delta.L0 :, :] == 0.00025)  # inside the sigma mask
        assert np.all(_delta.eta[_delta.L0 :, :] == -_delta.h0)

        _delta.update()
        assert np.all(_delta.eta[_delta.L0 - 1, :25] == 0.0)
        assert np.all(_delta.eta[_delta.L0 :, :] == pytest.approx(-_delta.h0 - 0.00025))
        _delta.output_netcdf.close()

    def test_subsidence_in_update_delayed_start(self, tmp_path: Path) -> None:
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "toggle_subsidence": True,
                "subsidence_rate": 1e-8,
                "start_subsidence": 25000,
                "save_eta_grids": True,
                "seed": 0,
            },
        )
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.subsidence_rate == 1e-8
        assert np.all(_delta.sigma[: _delta.L0, :] == 0.0)  # outside the sigma mask
        assert np.all(_delta.sigma[_delta.L0 :, :] == 0.00025)  # inside the sigma mask
        assert np.all(_delta.eta[_delta.L0 :, :] == -_delta.h0)

        _delta.update()  # no subsidence applied
        assert _delta.time == 25000
        assert np.all(_delta.eta[_delta.L0 - 1, :25] == 0.0)
        assert np.all(_delta.eta[_delta.L0 :, :] == pytest.approx(-_delta.h0))

        _delta.update()
        assert _delta.time == 50000
        assert np.all(_delta.eta[_delta.L0 - 1, :25] == 0.0)
        assert np.all(_delta.eta[_delta.L0 :, :] == pytest.approx(-_delta.h0 - 0.00025))
        _delta.output_netcdf.close()

        _delta.solve_water_and_sediment_timestep.call_count == 2

    def test_subsidence_changed_with_timestep(self, tmp_path: Path) -> None:
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "toggle_subsidence": True,
                "save_eta_grids": True,
                "subsidence_rate": 1e-8,
            },
        )
        _delta = DeltaModel(input_file=p)

        assert _delta.dt == 25000
        assert np.all(_delta.sigma[_delta.L0 :, :] == 0.00025)
        # use the model setter to adjust the timestep
        _delta.time_step = 86400
        assert np.all(_delta.sigma[_delta.L0 :, :] == 0.000864)
        _delta.output_netcdf.close()


class TestOutputCheckpoint:
    def test_save_a_checkpoint_checkpoint_False(self, tmp_path: Path) -> None:
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_checkpoint": False})
        _delta = DeltaModel(input_file=p)

        # mock the actual save checkpoint function to see if it was called
        _delta.save_the_checkpoint = mock.MagicMock()
        _delta.log_info = mock.MagicMock()

        _delta.output_checkpoint()

        # assertions
        assert _delta.save_the_checkpoint.called is False
        assert _delta.log_info.called is False

    def test_save_a_checkpoint_checkpoint_true(self, tmp_path: Path) -> None:
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_checkpoint": True})
        _delta = DeltaModel(input_file=p)

        # force the time to be greater than the checkpoint interval
        _delta._save_time_since_checkpoint = 2 * _delta._checkpoint_dt

        # mock the actual save checkpoint function to see if it was called
        _delta.save_the_checkpoint = mock.MagicMock()
        _delta.log_info = mock.MagicMock()

        # run the output checkpoint func
        _delta.output_checkpoint()

        # assertions
        assert _delta.save_the_checkpoint.call_count == 1
        assert _delta.log_info.call_count == 1

    def test_save_a_checkpoint_checkpoint_true_timewarning(
        self, tmp_path: Path
    ) -> None:
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_checkpoint": True, "checkpoint_dt": 864000000},
        )
        # issue one warning when initializing
        with pytest.warns(UserWarning, match=r"interval are not identical"):
            _delta = DeltaModel(input_file=p)

        # force the time to be greater than the checkpoint interval
        _delta._save_time_since_checkpoint = 2 * _delta._checkpoint_dt

        # mock the actual save checkpoint function to see if it was called
        _delta.save_the_checkpoint = mock.MagicMock()

        # mock the logger for testing in this file
        _delta.logger = mock.MagicMock()

        # issue second warning when outputting
        with pytest.warns(UserWarning, match=r"interval are not identical"):
            # run the output checkpoint func
            _delta.output_checkpoint()

        # assertions
        assert _delta.save_the_checkpoint.call_count == 1
        assert _delta.logger.warning.call_count == 1


class TestSaveGridsAndFigs:
    def test_save_no_figs_no_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_dt": 1})
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        # check nothing created at the start
        _delta.make_figure.call_count == 0
        _delta.save_figure.call_count == 0
        _delta.save_grids.call_count == 0

        # update the delta a few times
        for _t in range(0, 4):
            _delta._time = _t * _delta._dt
            _delta.save_grids_and_figs()

        # check nothing after a number of iterations, greater than dt
        assert _delta._time > _delta.save_dt
        _delta.make_figure.call_count == 0
        _delta.save_figure.call_count == 0
        _delta.save_grids.call_count == 0

    def test_save_one_fig_no_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_eta_figs": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        # check nothing created at the start
        _delta.make_figure.call_count == 0
        _delta.save_figure.call_count == 0
        _delta.save_grids.call_count == 0

        assert len(_delta._save_fig_list) > 0
        assert _delta._save_eta_figs is True

        # update the delta a few times
        for _t in range(0, 5):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        _delta.make_figure.call_count == 5
        _delta.save_figure.call_count == 5
        _delta.save_grids.call_count == 0

    def test_save_one_fig_one_grid(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_dt": 1, "save_eta_grids": True, "save_discharge_figs": True},
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        assert _delta._save_eta_grids is True
        assert _delta._save_metadata is True

        # check for the netcdf file
        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)
        nc_size_before = os.path.getsize(exp_path_nc)
        assert nc_size_before > 0  # saved once already / inited

        # update a couple times, should increase on each save
        for _t in range(0, 5):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        _delta.make_figure.call_count == 5
        _delta.save_figure.call_count == 5
        _delta.save_grids.call_count == 5

    def test_save_all_figures_no_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_dt": 1,
                "save_eta_figs": True,
                "save_discharge_figs": True,
                "save_velocity_figs": True,
                "save_stage_figs": True,
                "save_depth_figs": True,
                "save_sedflux_figs": True,
            },
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert not os.path.isfile(exp_path_nc)

        for _t in range(0, 5):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1
            _delta._time += _delta._dt

        # assertions
        _delta.make_figure.assert_any_call("eta", 0)
        _delta.make_figure.assert_any_call("stage", 0)
        _delta.make_figure.assert_any_call("depth", 0)
        _delta.make_figure.assert_any_call("qw", 0)
        _delta.make_figure.assert_any_call("uw", 0)
        _delta.make_figure.assert_any_call("qs", 0)
        for _i in range(1, 5):
            _delta.make_figure.assert_any_call("eta", _delta._dt * _i)
        _delta.make_figure.call_count == 5 * 6
        _delta.save_figure.call_count == 5 * 6
        _delta.save_grids.call_count == 0

    def test_save_metadata_no_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_metadata": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 3):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert not ("eta" in ds.variables)
        assert ds["auxdata"]["H_SL"].shape[0] == 4  # init + 3
        assert ds["auxdata"]["L0"][:] == 3

    def test_save_metadata_no_grids_legacy(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_dt": 1, "save_metadata": True, "legacy_netcdf": True},
        )
        with pytest.warns(UserWarning, match=r".*netcdf file in legacy schema.*"):
            _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 3):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions LEGACY SUBGROUP SHOULD BE CALLED "meta"
        assert not ("eta" in ds.variables)
        assert ds["meta"]["H_SL"].shape[0] == 4  # init + 3
        assert ds["meta"]["L0"][:] == 3

    def test_save_metadata_and_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_dt": 1,
                "save_metadata": True,
                "save_eta_grids": True,
                "save_velocity_grids": True,
                "f_bedload": 0.25,
            },
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 3):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert "eta" in ds.variables
        assert "velocity" in ds.variables
        assert ds["auxdata"]["H_SL"].shape[0] == 4  # init + 3
        assert ds["auxdata"]["L0"][:] == 3
        assert np.all(ds["auxdata"]["f_bedload"][:] == 0.25)

    def test_save_one_grid_metadata_by_default(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_dt": 1,
                "save_metadata": False,
                "save_eta_grids": True,
                "C0_percent": 0.2,
            },
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        _arr = ds.variables["eta"]
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too
        assert ds.groups["auxdata"]["H_SL"].shape[0] == _arr.shape[0]
        assert np.all(ds.groups["auxdata"]["C0_percent"][:].data == 0.2)
        assert np.all(ds.groups["auxdata"]["f_bedload"][:].data == 0.5)

    def test_save_one_grid_metadata_by_default_legacy(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_dt": 1,
                "save_metadata": False,
                "save_eta_grids": True,
                "C0_percent": 0.2,
                "legacy_netcdf": True,
            },
        )
        with pytest.warns(UserWarning, match=r".*netcdf file in legacy schema.*"):
            _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        _arr = ds.variables["eta"]
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert "meta" in ds.groups  # if any grids, save meta too
        assert ds.groups["meta"]["H_SL"].shape[0] == _arr.shape[0]
        assert np.all(ds.groups["meta"]["C0_percent"][:].data == 0.2)
        assert np.all(ds.groups["meta"]["f_bedload"][:].data == 0.5)


class TestSaveFigure:
    def test_save_figure(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_eta_figs": True}
        )
        _delta = DeltaModel(input_file=p)

        # make a figure
        fig, ax = plt.subplots()
        ax.imshow(np.random.uniform(0, 1, size=(100, 100)))

        # save two figs with different timesteps
        _delta.save_figure(
            fig, directory=_delta.prefix, filename_root="eta_", save_iter=0
        )

        _delta.save_figure(
            fig, directory=_delta.prefix, filename_root="eta_", save_iter=1
        )

        # check for output eta file
        exp_path_png0 = os.path.join(tmp_path / "out_dir", "eta_00000.png")
        assert os.path.isfile(exp_path_png0)
        exp_path_png1 = os.path.join(tmp_path / "out_dir", "eta_00001.png")
        assert os.path.isfile(exp_path_png1)

    def test_save_figure_sequential(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_dt": 1, "save_eta_figs": True, "save_figs_sequential": False},
        )
        _delta = DeltaModel(input_file=p)

        # make a figure
        fig, ax = plt.subplots()
        ax.imshow(np.random.uniform(0, 1, size=(100, 100)))

        # save two figs with different timesteps
        _delta.save_figure(
            fig, directory=_delta.prefix, filename_root="eta_", save_iter=0
        )

        _delta.save_figure(
            fig, directory=_delta.prefix, filename_root="eta_", save_iter=1
        )

        exp_path_png0 = os.path.join(tmp_path / "out_dir", "eta_00000.png")
        exp_path_png0_latest = os.path.join(tmp_path / "out_dir", "eta_latest.png")
        assert not os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png0_latest)


class TestSaveGrids:
    def test_save_eta_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_eta_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids("eta", _delta.eta, _delta._save_iter)
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert _delta.log_info.call_count == 6
        _arr = ds.variables["eta"]
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too

    def test_save_depth_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_depth_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids("depth", _delta.depth, _delta._save_iter)
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert _delta.log_info.call_count == 6
        _arr = ds.variables["depth"]
        assert _arr.shape[1] == _delta.depth.shape[0]
        assert _arr.shape[2] == _delta.depth.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too

    def test_save_velocity_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_velocity_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids("velocity", _delta.uw, _delta._save_iter)
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert _delta.log_info.call_count == 6
        _arr = ds.variables["velocity"]
        assert _arr.shape[1] == _delta.uw.shape[0]
        assert _arr.shape[2] == _delta.uw.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too

    def test_save_stage_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_stage_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids("stage", _delta.stage, _delta._save_iter)
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert _delta.log_info.call_count == 6
        _arr = ds.variables["stage"]
        assert _arr.shape[1] == _delta.stage.shape[0]
        assert _arr.shape[2] == _delta.stage.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too

    def test_save_discharge_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_discharge_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids("discharge", _delta.qw, _delta._save_iter)
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert _delta.log_info.call_count == 6
        _arr = ds.variables["discharge"]
        assert _arr.shape[1] == _delta.qw.shape[0]
        assert _arr.shape[2] == _delta.qw.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too

    def test_save_sedflux_grids(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_sedflux_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.save_grids("sedflux", _delta.qs, _delta._save_iter)
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        assert _delta.log_info.call_count == 6
        _arr = ds.variables["sedflux"]
        assert _arr.shape[1] == _delta.qs.shape[0]
        assert _arr.shape[2] == _delta.qs.shape[1]
        assert "auxdata" in ds.groups  # if any grids, save meta too

    def test_save_grids_exception(self, tmp_path: Path) -> None:
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_dt": 1})
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # no netcdf will be created
        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc) is False

        # trying to save a grid will raise an exception
        with pytest.raises(Exception, match=r"DeltaModel' object .*"):
            _delta.save_grids("sedflux", _delta.qs, _delta._save_iter)

        # assert log was called when exception was raised
        assert _delta.log_info.call_count == 1


class TestCustomTimeVaryingMetaVar:
    def test_dict_format_time_tuple_updates(self, tmp_path: Path) -> None:
        """
        A custom meta variable specified in the documented dict format
        with a proper one-element vardims tuple must be updated on every
        save (the old gate keyed on len("time") == 4 only worked for the
        built-in variables by accident).
        """

        class CustomMetaModel(DeltaModel):
            def hook_init_output_file(self) -> None:
                self._save_var_list["meta"]["custom_tv_var"] = dict(
                    varname="custom_tv_var",
                    varvalue="_time",
                    varunits="seconds",
                    vartype="f8",
                    vardims=(self._netcdf_coords[0],),
                    varlong="custom time-varying variable",
                )

        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_dt": 1, "save_metadata": True}
        )
        _delta = CustomMetaModel(input_file=p)

        # mock the log_info and the actual output routines
        _delta.log_info = mock.MagicMock()
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / "out_dir", "pyDeltaRCM_output.nc")

        for _t in range(0, 3):
            _delta._time += 100
            _delta.save_grids_and_figs()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # the variable must have been written on every save
        assert ds["auxdata"]["custom_tv_var"].shape[0] == 4  # init + 3
        vals = ds["auxdata"]["custom_tv_var"][:]
        assert np.all(np.diff(vals) > 0)
        ds.close()


class TestComputeSandFrac:
    """Direct unit coverage for ``compute_sand_frac``.

    The method updates ``sand_frac`` from the erosion/deposition state of
    the timestep: erosion below the active layer resets to the boundary
    condition, erosion within the active layer adopts the active-layer
    value, and deposition blends the active layer toward the deposit
    mixture. These tests exercise each branch on a tiny model.
    """

    def _make_delta(self, tmp_path: Path, **kwargs) -> DeltaModel:
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_metadata": True, **kwargs}
        )
        return DeltaModel(input_file=p)

    def test_no_erosion_or_deposition_leaves_sand_frac(self, tmp_path: Path) -> None:
        _delta = self._make_delta(tmp_path, seed=0)

        _delta.eta0[:] = -1.0
        _delta.eta[:] = -1.0
        _delta.sand_frac[:] = 0.42
        _delta.active_layer[:] = 0.42
        _delta.Vp_dep_sand[:] = 0.0
        _delta.Vp_dep_mud[:] = 0.0

        _delta.compute_sand_frac()

        assert np.all(_delta.sand_frac == 0.42)
        assert np.all(_delta.active_layer == 0.42)
        _delta.output_netcdf.close()

    def test_erosion_below_active_layer_resets_to_bc(self, tmp_path: Path) -> None:
        _delta = self._make_delta(tmp_path, seed=0, sand_frac_bc=0.2)

        # eta0 near the surface; erosion deep enough to cut below the
        # active layer bottom (eta0 - active layer thickness)
        _delta.eta0[:] = -1.0
        _delta.eta[:] = -1.0 - _delta._active_layer_thickness - 1.0
        _delta.sand_frac[:] = 0.9
        _delta.active_layer[:] = 0.9
        _delta.Vp_dep_sand[:] = 0.0
        _delta.Vp_dep_mud[:] = 0.0

        _delta.compute_sand_frac()

        assert np.all(_delta.sand_frac == 0.2)
        _delta.output_netcdf.close()

    def test_erosion_within_active_layer_adopts_active_layer(
        self, tmp_path: Path
    ) -> None:
        _delta = self._make_delta(tmp_path, seed=0)

        _delta.eta0[:] = -1.0
        # erode less than the active layer thickness
        _delta.eta[:] = -1.0 - 0.5 * _delta._active_layer_thickness
        _delta.sand_frac[:] = 0.9
        _delta.active_layer[:] = 0.55
        _delta.Vp_dep_sand[:] = 0.0
        _delta.Vp_dep_mud[:] = 0.0

        _delta.compute_sand_frac()

        assert np.all(_delta.sand_frac == 0.55)
        _delta.output_netcdf.close()

    def test_mixed_deposition_sets_mixture(self, tmp_path: Path) -> None:
        _delta = self._make_delta(tmp_path, seed=0)

        _delta.eta0[:] = -1.0
        _delta.eta[:] = -1.0  # no erosion
        _delta.sand_frac[:] = 0.1
        _delta.active_layer[:] = 0.1
        _delta.Vp_dep_sand[:] = 0.25
        _delta.Vp_dep_mud[:] = 0.75

        _delta.compute_sand_frac()

        expected = 0.25 / (0.25 + 0.75)
        assert np.all(_delta.sand_frac == pytest.approx(expected))
        assert np.all(_delta.active_layer == pytest.approx(expected))
        _delta.output_netcdf.close()

    def test_mud_only_deposition_above_epsilon_sets_mixture(
        self, tmp_path: Path
    ) -> None:
        _delta = self._make_delta(tmp_path, seed=0)

        _delta.eta0[:] = -1.0
        _delta.eta[:] = -1.0
        _delta.sand_frac[:] = 0.8
        _delta.active_layer[:] = 0.8
        _delta.Vp_dep_sand[:] = 0.0
        _delta.Vp_dep_mud[:] = 0.01  # above the 1e-6 mud-deposition threshold

        _delta.compute_sand_frac()

        expected = 0.0 / (0.0 + 0.01)
        assert np.all(_delta.sand_frac == pytest.approx(expected))
        assert np.all(_delta.active_layer == pytest.approx(expected))
        _delta.output_netcdf.close()

    def test_trace_mud_deposition_below_epsilon_is_ignored(
        self, tmp_path: Path
    ) -> None:
        _delta = self._make_delta(tmp_path, seed=0)

        _delta.eta0[:] = -1.0
        _delta.eta[:] = -1.0
        _delta.sand_frac[:] = 0.8
        _delta.active_layer[:] = 0.8
        _delta.Vp_dep_sand[:] = 0.0
        _delta.Vp_dep_mud[:] = 1e-7  # below the 1e-6 threshold

        _delta.compute_sand_frac()

        assert np.all(_delta.sand_frac == 0.8)
        assert np.all(_delta.active_layer == 0.8)
        _delta.output_netcdf.close()

    def test_erosion_and_deposition_branches_apply_cellwise(
        self, tmp_path: Path
    ) -> None:
        _delta = self._make_delta(tmp_path, seed=0)

        # column 0: deep erosion; column 1: deposition; column 2: untouched
        _delta.eta0[:] = -1.0
        _delta.eta[:, 0] = -1.0 - _delta._active_layer_thickness - 1.0
        _delta.eta[:, 1] = -1.0
        _delta.eta[:, 2] = -1.0
        _delta.sand_frac[:] = 0.9
        _delta.active_layer[:] = 0.9
        _delta.Vp_dep_sand[:, 0] = 0.0
        _delta.Vp_dep_mud[:, 0] = 0.0
        _delta.Vp_dep_sand[:, 1] = 0.5
        _delta.Vp_dep_mud[:, 1] = 0.5

        _delta.compute_sand_frac()

        assert np.all(_delta.sand_frac[:, 0] == _delta._sand_frac_bc)
        assert np.all(_delta.sand_frac[:, 1] == pytest.approx(0.5))
        assert np.all(_delta.sand_frac[:, 2] == 0.9)  # untouched column
        _delta.output_netcdf.close()

