# unit tests for consistent model outputs

import os
import platform
import shutil
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

import unittest.mock as mock

from pyDeltaRCM import DeltaModel
from pyDeltaRCM import preprocessor

from .. import utilities
from pyDeltaRCM.shared_tools import get_random_state, get_random_uniform


def _write_unpickle_marker(path: str) -> None:
    """Create a marker file; called only if a pickled payload executes."""
    with open(path, "w") as f:
        f.write("unpickled")


class _UnpicklePayload:
    """Picklable object whose unpickling creates a marker file.

    Used to test that pickled checkpoint contents are never unpickled:
    if the payload ever runs, the marker file exists.
    """

    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return (_write_unpickle_marker, (self.marker_path,))


@mock.patch(
    "pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep",
    new=utilities.FastIteratingDeltaModel.solve_water_and_sediment_timestep,
)
class TestCheckpointingIntegrations:
    """
    The above patch implements an augmented DeltaModel from `utilities`. In
    this modified DeltaModel, the `solve_water_and_sediment_timestep`
    operations (i.e., the time consuming part of the model) is replaced with
    an updating random field. This guarantees that the random-repeatedness of
    checkpointing is validated, but it is much faster and easier to isolate
    checkpointing-related issues from model issues.
    """

    # Both tests assert exact stream replay after resume. The thread-local
    # NumPy MT19937 generator's state is captured by the checkpoint
    # helpers, so replay holds unconditionally -- numba's no-JIT stream
    # divergence no longer exists on this tree.
    def test_simple_checkpoint(self, tmp_path: Path) -> None:
        """Test checkpoint vs a base run.

        Also, checks resumed model against another checkpoint run.
        """
        # define a yaml for the longer model run
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "save_checkpoint", True)
        base_f.close()
        longModel = DeltaModel(input_file=base_p)

        # run for some number of updates
        for _ in range(0, 50):
            longModel.update()
        longModel.finalize()

        # try defining a new model but plan to load checkpoint from longModel
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        # advance the resumed model until it catch up to longModel
        assert resumeModel.time < longModel.time
        while resumeModel._time < longModel._time:
            resumeModel.update()
        resumeModel.finalize()

        # the longModel and resumeModel should match
        assert longModel.time == resumeModel.time
        assert np.all(longModel.eta == resumeModel.eta)
        assert np.all(longModel.uw == resumeModel.uw)
        assert np.all(longModel.ux == resumeModel.ux)
        assert np.all(longModel.uy == resumeModel.uy)
        assert np.all(longModel.depth == resumeModel.depth)
        assert np.all(longModel.stage == resumeModel.stage)
        assert np.all(longModel.sand_frac == resumeModel.sand_frac)
        assert np.all(longModel.active_layer == resumeModel.active_layer)

        # define another model that loads the checkpoint
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        base_f.close()
        resumeModel2 = DeltaModel(input_file=base_p)

        # advance the resumed model until it catch up to longModel
        while resumeModel2._time < resumeModel._time:
            resumeModel2.update()
        resumeModel2.finalize()

        # the two models that resumed from the checkpoint should be the same
        assert resumeModel2.time == resumeModel.time
        assert np.all(resumeModel2.uw == resumeModel.uw)
        assert np.all(resumeModel2.ux == resumeModel.ux)
        assert np.all(resumeModel2.uy == resumeModel.uy)
        assert np.all(resumeModel2.depth == resumeModel.depth)
        assert np.all(resumeModel2.stage == resumeModel.stage)
        assert np.all(resumeModel2.sand_frac == resumeModel.sand_frac)
        assert np.all(resumeModel2.active_layer == resumeModel.active_layer)

    def test_checkpoint_round_trips_rng_stream(self, tmp_path: Path) -> None:
        """Test save → resume reproduces the exact random state and stream.

        The checkpoint is written with plain arrays (no pickle), and the
        resumed model continues the random stream exactly where the
        checkpoint left off.
        """
        # producer model writes a checkpoint after a few fast updates
        file_name = "rng_round_trip.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "out_dir")
        base_f.close()
        producer = DeltaModel(input_file=base_p)

        for _ in range(3):
            producer.update()
        producer.save_the_checkpoint()
        saved_state = get_random_state()
        producer_draws = [get_random_uniform(1.0) for _ in range(4)]
        producer.finalize()

        # the written checkpoint contains no pickled object arrays:
        # loading it with allow_pickle=False must succeed for every field
        ckp_file = os.path.join(producer.prefix, "checkpoint.npz")
        with np.load(ckp_file, allow_pickle=False) as checkpoint:
            for key in checkpoint.files:
                _ = checkpoint[key]

        # consumer model resumes from the checkpoint folder
        file_name = "rng_round_trip_resume.yaml"
        resume_p, resume_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(resume_f, "out_dir", tmp_path / "out_dir2")
        utilities.write_parameter_to_file(
            resume_f, "resume_checkpoint", str(producer.prefix)
        )
        resume_f.close()
        consumer = DeltaModel(input_file=resume_p)

        # the resumed model continues the exact random stream
        assert get_random_state() == saved_state
        consumer_draws = [get_random_uniform(1.0) for _ in range(4)]
        assert consumer_draws == producer_draws
        consumer.finalize()

    def test_checkpoint_with_pickled_rng_state_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Test a legacy pickled checkpoint raises without executing.

        Checkpoints written before the pickle-free rng format (pyDeltaRCM
        <= 2.3) carry a pickled object array. Loading such a file must
        raise, and the payload embedded in the pickled object must never
        be executed.
        """
        # build a model and write a legitimate checkpoint
        p = utilities.yaml_from_dict(tmp_path, "producer.yaml")
        producer = DeltaModel(input_file=p)
        producer.save_the_checkpoint()
        ckp_file = os.path.join(producer.prefix, "checkpoint.npz")
        assert os.path.isfile(ckp_file)

        # rewrite the checkpoint with the rng state as a pickled object
        # array; unpickling the payload would create the marker file
        marker = tmp_path / "unpickle-marker"
        with np.load(ckp_file, allow_pickle=False) as checkpoint:
            fields = {key: checkpoint[key] for key in checkpoint.files}
        for key in ("rng_name", "rng_state_index", "rng_state_key"):
            fields.pop(key)
        fields["rng_state"] = np.array([_UnpicklePayload(str(marker))], dtype=object)
        np.savez_compressed(ckp_file, **fields)

        # resuming from the tampered checkpoint must fail loudly ...
        resume_p, resume_f = utilities.create_temporary_file(
            tmp_path, "resume_tampered.yaml"
        )
        utilities.write_parameter_to_file(resume_f, "out_dir", tmp_path / "out_dir2")
        utilities.write_parameter_to_file(
            resume_f, "resume_checkpoint", str(producer.prefix)
        )
        resume_f.close()
        with pytest.raises(RuntimeError, match="pickle format"):
            DeltaModel(input_file=resume_p)

        # ... and the embedded payload must never have been executed
        assert not marker.exists()

    def test_checkpoint_nc(self, tmp_path: Path) -> None:
        """Test the netCDF that is written to by the checkpointing."""
        # define a yaml for the base model run
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "save_depth_grids", True)
        utilities.write_parameter_to_file(base_f, "save_discharge_grids", True)
        utilities.write_parameter_to_file(base_f, "save_sandfrac_grids", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", True)
        base_f.close()
        baseModel = DeltaModel(input_file=base_p)

        # run for some base number of steps
        nt_base = 50
        for _ in range(0, 50):
            baseModel.update()

        # force the model run to end immmediately after exporting a checkpoint
        nt_var = 0
        while baseModel._save_time_since_checkpoint != 0:
            baseModel.update()
            nt_var += 1

        # then finalize
        baseModel.finalize()

        # check that the time makes sense
        assert baseModel.time == baseModel._dt * (nt_base + nt_var)

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "save_depth_grids", True)
        utilities.write_parameter_to_file(base_f, "save_discharge_grids", True)
        utilities.write_parameter_to_file(base_f, "save_sandfrac_grids", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", False)
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time == baseModel.time  # same when resumed

        # advance it until output_data has been called again
        nt_resume = 0
        while (resumeModel._save_time_since_data != 0) or (nt_resume < 50):
            resumeModel.update()
            nt_resume += 1
        resumeModel.finalize()

        assert nt_resume > 0
        assert resumeModel.time > baseModel.time

        # assert that output netCDF4 exists
        exp_path_nc = os.path.join(tmp_path / "test", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        # load it into memory and check values in the netCDF4
        output = Dataset(exp_path_nc, "r", allow_pickle=True)
        out_vars = output.variables.keys()

        # check that expected variables are in the file
        assert "x" in out_vars
        assert "y" in out_vars
        assert "seconds" in out_vars
        assert "eta" in out_vars
        assert "depth" in out_vars
        assert "discharge" in out_vars
        assert "sandfrac" in out_vars

        # check attributes of variables
        assert output["seconds"][0].tolist() == 0.0
        assert output["seconds"][-1] == resumeModel.time
        assert output["seconds"][-1].tolist() == resumeModel._dt * (
            nt_base + nt_var + nt_resume
        )
        assert output["eta"][0].shape == resumeModel.eta.shape
        assert output["eta"][-1].shape == resumeModel.eta.shape
        assert output["depth"][-1].shape == resumeModel.eta.shape
        assert output["discharge"][-1].shape == resumeModel.eta.shape
        assert output["sandfrac"][-1].shape == resumeModel.eta.shape
        # check the metadata
        assert output["auxdata"]["L0"][:] == resumeModel.L0
        assert output["auxdata"]["N0"][:] == resumeModel.N0
        assert output["auxdata"]["CTR"][:] == resumeModel.CTR
        assert output["auxdata"]["dx"][:] == resumeModel.dx
        assert output["auxdata"]["h0"][:] == resumeModel.h0
        assert np.all(output["auxdata"]["cell_type"][:] == resumeModel.cell_type)
        assert output["auxdata"]["H_SL"][-1].data == resumeModel.H_SL
        assert output["auxdata"]["f_bedload"][-1].data == resumeModel.f_bedload
        C0_from_file = float(output["auxdata"]["C0_percent"][-1].data)
        assert pytest.approx(C0_from_file) == resumeModel.C0_percent
        assert output["auxdata"]["u0"][-1].data == resumeModel.u0

        # checkpoint interval aligns w/ timestep dt so these should match
        assert output["seconds"][-1].tolist() == resumeModel.time

    def test_checkpoint_nc_legacy(self, tmp_path: Path) -> None:
        """Test the netCDF that is written to by the checkpointing."""
        # define a yaml for the base model run
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "legacy_netcdf", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", True)
        base_f.close()
        with pytest.warns(UserWarning, match=r"Creating output"):
            baseModel = DeltaModel(input_file=base_p)

        # run for some base number of steps
        nt_base = 50
        for _ in range(0, 50):
            baseModel.update()

        # force the model run to end immmediately after exporting a checkpoint
        nt_var = 0
        while baseModel._save_time_since_checkpoint != 0:
            baseModel.update()
            nt_var += 1

        # then finalize
        baseModel.finalize()

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "legacy_netcdf", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", False)
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time == baseModel.time  # same when resumed

        # advance it until output_data has been called again
        nt_resume = 0
        while (resumeModel._save_time_since_data != 0) or (nt_resume < 50):
            resumeModel.update()
            nt_resume += 1
        resumeModel.finalize()

        # assert that output netCDF4 exists
        exp_path_nc = os.path.join(tmp_path / "test", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        # load it into memory and check values in the netCDF4
        output = Dataset(exp_path_nc, "r", allow_pickle=True)
        out_vars = output.variables.keys()

        # check only the things expected to be diff in legacy
        assert "time" in out_vars
        assert "meta" in output.groups.keys()

    def test_checkpoint_diff_dt(self, tmp_path: Path) -> None:
        """Test when checkpoint_dt does not match dt or save_dt."""
        # define a yaml for the base model run
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "save_depth_grids", True)
        utilities.write_parameter_to_file(base_f, "save_discharge_grids", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", True)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        base_f.close()
        baseModel = DeltaModel(input_file=base_p)

        # modify the checkpoint dt to be different than save_dt
        baseModel._checkpoint_dt = baseModel.save_dt * 0.65

        # CHECK for warning about mismatched intervals!
        with pytest.warns(UserWarning, match=r"Grid save interval and checkpoint interval"):
            for _ in range(0, 50):
                baseModel.update()
        baseModel.finalize()

        assert baseModel.time == baseModel._dt * 50
        baseModelSavedTime = baseModel.time - baseModel._save_time_since_checkpoint
        assert baseModelSavedTime > 0

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "save_depth_grids", True)
        utilities.write_parameter_to_file(base_f, "save_discharge_grids", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", False)
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time == baseModelSavedTime

        # advance until some steps and just saved
        nt_resume = 0
        while (resumeModel._save_time_since_data != 0) or (nt_resume < 50):
            resumeModel.update()
            nt_resume += 1
        resumeModel.finalize()

        # assert that output netCDF4 exists
        exp_path_nc = os.path.join(tmp_path / "test", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        # load it into memory and check values in the netCDF4
        output = Dataset(exp_path_nc, "r", allow_pickle=True)
        out_vars = output.variables.keys()
        # check that expected variables are in the file
        assert "x" in out_vars
        assert "y" in out_vars
        assert "seconds" in out_vars
        assert "eta" in out_vars
        assert "depth" in out_vars
        assert "discharge" in out_vars
        # check attributes of variables
        assert output["seconds"][0].tolist() == 0.0
        assert output["seconds"][-1].tolist() == resumeModel.time

    def test_multi_checkpoints(self, tmp_path: Path) -> None:
        """Test using checkpoints multiple times for a given model run."""
        # define a yaml for the base model run
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", True)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        base_f.close()
        baseModel = DeltaModel(input_file=base_p)

        # run base for 2 timesteps
        for _ in range(0, 50):
            baseModel.update()
        baseModel.finalize()

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = "base_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "save_eta_grids", True)
        utilities.write_parameter_to_file(base_f, "save_checkpoint", True)
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time <= baseModel.time

        # advance it more steps
        for _ in range(0, 25):
            resumeModel.update()
        resumeModel.finalize()

        # create another resume model
        resumeModel02 = DeltaModel(input_file=base_p)

        assert resumeModel02.time <= resumeModel.time  # should be same

        # step it some more
        nt_resume02 = 0
        while (resumeModel02._save_time_since_data != 0) or (nt_resume02 < 50):
            resumeModel02.update()
            nt_resume02 += 1

        # assert that output netCDF4 exists
        exp_path_nc = os.path.join(tmp_path / "test", "pyDeltaRCM_output.nc")
        assert os.path.isfile(exp_path_nc)

        # load it into memory and check values in the netCDF4
        output = Dataset(exp_path_nc, "r", allow_pickle=True)
        out_vars = output.variables.keys()
        # check that expected variables are in the file
        assert "x" in out_vars
        assert "y" in out_vars
        assert "seconds" in out_vars
        assert "eta" in out_vars
        # check attributes of variables
        assert output["seconds"][0].tolist() == 0.0
        assert output["seconds"][-1].tolist() == resumeModel02.time

    def test_load_nocheckpoint(self, tmp_path: Path) -> None:
        """Try loading a checkpoint file when one doesn't exist."""
        # define a yaml
        file_name = "trial_run.yaml"
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, "resume_checkpoint", True)
        utilities.write_parameter_to_file(base_f, "out_dir", tmp_path / "test")
        base_f.close()

        # try loading the model yaml despite no checkpoint existing
        with pytest.raises(FileNotFoundError):
            _ = DeltaModel(input_file=base_p)

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_py_hlvl_parallel_checkpoint(self, tmp_path: Path) -> None:
        """Test checkpointing in parallel."""
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "parallel", 2)
        utilities.write_parameter_to_file(f, "save_checkpoint", True)
        utilities.write_parameter_to_file(f, "save_eta_grids", True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=50)
        # assertions for job creation
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        # run the jobs, mocked deltas
        pp.run_jobs()

        # compute the expected final time recorded
        _dt = pp.job_list[1].deltamodel._dt
        _checkpoint_dt = pp.job_list[1].deltamodel._checkpoint_dt
        expected_save_interval = ((_checkpoint_dt // _dt) + 1) * _dt
        expected_last_save_time = (
            (50 * _dt) // expected_save_interval
        ) * expected_save_interval

        # assertions after running jobs
        assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
        assert pp._is_completed is True
        exp_path_nc0 = os.path.join(
            tmp_path / "test", "job_000", "pyDeltaRCM_output.nc"
        )
        exp_path_nc1 = os.path.join(
            tmp_path / "test", "job_001", "pyDeltaRCM_output.nc"
        )
        assert os.path.isfile(exp_path_nc0)
        assert os.path.isfile(exp_path_nc1)
        # check that checkpoint files exist
        exp_path_ckpt0 = os.path.join(tmp_path / "test", "job_000", "checkpoint.npz")
        exp_path_ckpt1 = os.path.join(tmp_path / "test", "job_001", "checkpoint.npz")
        assert os.path.isfile(exp_path_ckpt0)
        assert os.path.isfile(exp_path_ckpt1)
        # load one output files and check values
        out_old = Dataset(exp_path_nc1)
        assert "auxdata" in out_old.groups.keys()
        assert out_old["seconds"][0].tolist() == 0.0
        assert out_old["seconds"][-1].tolist() == expected_last_save_time

        # close netCDF file
        out_old.close()

        # try to resume jobs
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "ensemble", 2)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "test")
        utilities.write_parameter_to_file(f, "parallel", 2)
        utilities.write_parameter_to_file(f, "resume_checkpoint", True)
        utilities.write_parameter_to_file(f, "save_eta_grids", True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=50)
        # assertions for job creation
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        # run the jobs, mocked deltas
        pp.run_jobs()

        # assertions after running jobs
        assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
        assert pp._is_completed is True
        exp_path_nc0 = os.path.join(
            tmp_path / "test", "job_000", "pyDeltaRCM_output.nc"
        )
        exp_path_nc1 = os.path.join(
            tmp_path / "test", "job_001", "pyDeltaRCM_output.nc"
        )
        assert os.path.isfile(exp_path_nc0)
        assert os.path.isfile(exp_path_nc1)
        # check that checkpoint files still exist
        exp_path_ckpt0 = os.path.join(tmp_path / "test", "job_000", "checkpoint.npz")
        exp_path_ckpt1 = os.path.join(tmp_path / "test", "job_001", "checkpoint.npz")
        assert os.path.isfile(exp_path_ckpt0)
        assert os.path.isfile(exp_path_ckpt1)
        # load one output file to check it out
        out_fin = Dataset(exp_path_nc1)
        assert "auxdata" in out_old.groups.keys()
        assert out_fin["seconds"][0].tolist() == 0
        assert out_fin["seconds"][-1].tolist() == expected_last_save_time * 2
        # close netcdf file
        out_fin.close()

    def test_serial_resume_continues_rng_state(self, tmp_path: Path) -> None:
        """Test that a serial resumed run continues the checkpointed RNG state.

        A serial preprocessor job resuming from a checkpoint must continue
        drawing random numbers from the state restored by
        ``load_checkpoint``, not restart the draw sequence from the seed as
        if the run had started from t=0. The parallel job path guards this
        correctly; the serial path must too, or a resumed run silently
        produces a different trajectory than the uninterrupted run.
        """
        # an uninterrupted reference run of 100 timesteps
        p_long = utilities.yaml_from_dict(
            tmp_path,
            "long_run.yaml",
            {"out_dir": tmp_path / "long", "seed": 8},
        )
        pp_long = preprocessor.Preprocessor(input_file=p_long, timesteps=100)
        pp_long.run_jobs()
        long_job = pp_long.job_list[0]
        assert long_job.deltamodel._time == long_job.deltamodel._dt * 100

        # an interrupted run of 50 timesteps, saving a checkpoint
        p_base = utilities.yaml_from_dict(
            tmp_path,
            "base_run.yaml",
            {"out_dir": tmp_path / "base", "seed": 8, "save_checkpoint": True},
        )
        pp_base = preprocessor.Preprocessor(input_file=p_base, timesteps=50)
        pp_base.run_jobs()
        base_job = pp_base.job_list[0]
        ckp_file = os.path.join(base_job.deltamodel.prefix, "checkpoint.npz")
        assert os.path.isfile(ckp_file)

        # the checkpoint was written at the last checkpoint trigger of the
        # interrupted run, so compute how many timesteps the resumed run
        # needs to reach the same model time as the uninterrupted run
        with np.load(ckp_file) as z:
            ckp_time = float(z["time"])
        _dt = base_job.deltamodel._dt
        _rem = long_job.deltamodel._time - ckp_time
        assert _rem > 0 and _rem % _dt == 0
        nt_resume = int(_rem / _dt)

        # a resumed run continuing from the checkpoint
        p_resume = utilities.yaml_from_dict(
            tmp_path,
            "resume_run.yaml",
            {"out_dir": tmp_path / "base", "seed": 8, "resume_checkpoint": True},
        )
        pp_resume = preprocessor.Preprocessor(
            input_file=p_resume, timesteps=nt_resume
        )
        pp_resume.run_jobs()
        resume_job = pp_resume.job_list[0]

        # the resumed run must match the uninterrupted run at the same
        # model time: the checkpointed random state was continued, not
        # replayed from t=0
        assert resume_job.deltamodel._time == long_job.deltamodel._time
        assert np.all(resume_job.deltamodel.eta == long_job.deltamodel.eta)
        assert np.all(resume_job.deltamodel.uw == long_job.deltamodel.uw)
        assert np.all(resume_job.deltamodel.ux == long_job.deltamodel.ux)
        assert np.all(resume_job.deltamodel.uy == long_job.deltamodel.uy)
        assert np.all(resume_job.deltamodel.depth == long_job.deltamodel.depth)
        assert np.all(resume_job.deltamodel.stage == long_job.deltamodel.stage)

    def test_resume_checkpoint_cadence_continues(self, tmp_path: Path) -> None:
        """Test that a resumed run's checkpoint cadence continues on schedule.

        The checkpoint counter (``_save_time_since_checkpoint``) must be
        saved with the checkpoint and restored on resume, so that the
        resumed run's checkpoint stream matches the uninterrupted run.
        Otherwise the counter keeps its initial value of infinity and the
        first update after resume spuriously overwrites the checkpoint
        file, regardless of the ``checkpoint_dt`` phase.
        """
        # base run with a checkpoint interval that does not match save_dt
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True}
        )
        baseModel = DeltaModel(input_file=p)

        # modify the checkpoint dt to be different than save_dt, so
        #   checkpoints trigger mid-cycle
        baseModel._checkpoint_dt = baseModel.save_dt * 0.65

        # run for some number of updates, saving checkpoints as triggered
        with pytest.warns(
            UserWarning, match=r"Grid save interval and checkpoint interval"
        ):
            for _ in range(0, 50):
                baseModel.update()
        baseModel.finalize()

        # the last checkpoint was saved mid-run
        ckp_file = os.path.join(baseModel.prefix, "checkpoint.npz")
        with np.load(ckp_file) as z:
            ckp_time = float(z["time"])
        assert 0 < ckp_time < baseModel.time

        # resume from the checkpoint
        p_resume = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"resume_checkpoint": True}
        )
        resumeModel = DeltaModel(input_file=p_resume)

        # the restored counter is the uninterrupted run's post-save value
        #   (the counter is reset when the checkpoint is saved), not
        #   infinity
        assert resumeModel._save_time_since_checkpoint == 0.0

        # advance one step: the checkpoint must not be overwritten,
        #   because the resumed cadence continues on the uninterrupted
        #   schedule
        resumeModel.update()
        with np.load(ckp_file) as z:
            assert float(z["time"]) == ckp_time
        resumeModel.finalize()

    def test_resume_preserves_fractional_data_cadence(self, tmp_path: Path) -> None:
        """Test that the data output cadence accumulator survives resume.

        ``_save_time_since_data`` accumulates fractional timesteps, so it
        must be restored as a float. Restoring it through ``int()``
        truncation shifts the output cadence of the resumed run relative
        to the uninterrupted run.
        """
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True}
        )
        baseModel = DeltaModel(input_file=p)

        # advance the model with a non-integer timestep, so the data
        #   output accumulator is fractional at checkpoint time
        baseModel._dt = 250.5
        for _ in range(0, 3):
            baseModel.update()

        # force a checkpoint save
        baseModel._save_time_since_checkpoint = float("inf")
        baseModel.output_checkpoint()
        baseModel.finalize()

        # the accumulator value saved to the checkpoint is fractional
        expected = 3 * 250.5
        with np.load(os.path.join(baseModel.prefix, "checkpoint.npz")) as z:
            assert float(z["save_time_since_data"]) == expected

        # resume from the checkpoint
        p_resume = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"resume_checkpoint": True}
        )
        resumeModel = DeltaModel(input_file=p_resume)

        # the float accumulator must survive the round trip without
        #   truncation
        assert resumeModel._save_time_since_data == expected

    def test_midwrite_crash_leaves_previous_checkpoint_loadable(
        self, tmp_path: Path
    ) -> None:
        """Test that a crash mid-checkpoint-write preserves the last checkpoint.

        The checkpoint archive must be written to a temporary file and
        atomically renamed over ``checkpoint.npz``. A process killed
        mid-write (OOM kill, disk full, interrupt) must leave the previous
        checkpoint intact and loadable; writing ``np.savez_compressed``
        directly to the final path truncates the only checkpoint the
        moment the write begins, so the run loses all resumable state
        exactly when it is needed.
        """
        # a model saving checkpoints
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True}
        )
        _delta = DeltaModel(input_file=p)

        # force a first checkpoint save and snapshot its bytes
        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()
        ckp_file = os.path.join(_delta.prefix, "checkpoint.npz")
        assert os.path.isfile(ckp_file)
        with open(ckp_file, "rb") as f:
            first_bytes = f.read()
        with np.load(ckp_file) as z:
            first_time = float(z["time"])

        def _crash_mid_write(file, *args, **kwargs):
            # simulate the underlying write having already flushed partial
            #   bytes to the target path before the process died
            target = file if isinstance(file, str) else file.name
            with open(target, "wb") as f:
                f.write(b"PK\x03\x04" + b"\x00" * 256)
            raise RuntimeError("simulated mid-write crash")

        # a checkpoint write that crashes mid-write
        with mock.patch(
            "pyDeltaRCM.iteration_tools.np.savez_compressed",
            side_effect=_crash_mid_write,
        ):
            _delta._save_time_since_checkpoint = float("inf")
            with pytest.raises(RuntimeError, match="simulated mid-write crash"):
                _delta.output_checkpoint()

        # the previous checkpoint survived byte-for-byte and is loadable
        with open(ckp_file, "rb") as f:
            assert f.read() == first_bytes
        with np.load(ckp_file) as z:
            assert float(z["time"]) == first_time

        # the next (uninterrupted) save succeeds and atomically replaces
        #   the checkpoint; the temporary file is consumed by the rename
        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()
        with np.load(ckp_file) as z:
            assert float(z["time"]) == first_time
        assert not os.path.exists(
            os.path.join(_delta.prefix, "checkpoint.tmp.npz")
        )

        _delta.finalize()


class TestCheckpointingCreatingLoading:
    def test_load_checkpoint_with_netcdf(self, tmp_path: Path) -> None:
        """Test that a run can be resumed when there are outputs."""
        # define a yaml with outputs (defaults will output strata)
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True, "save_eta_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # paths exists
        assert os.path.isfile(os.path.join(_delta.prefix, "pyDeltaRCM_output.nc"))
        assert os.path.isfile(os.path.join(_delta.prefix, "checkpoint.npz"))
        _delta = []  # clear

        # can be resumed
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True, "resume_checkpoint": True}
        )
        _delta = DeltaModel(input_file=p)

        # check that fields match
        assert np.all(_delta.eta == _rand_field)

    def test_create_checkpoint_without_netcdf(self, tmp_path: Path) -> None:
        """Test that a checkpoint can be created when there are no outputs"""
        # define a yaml with NO outputs, but checkpoint
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_checkpoint": True})

        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # should be no file
        assert not os.path.isfile(os.path.join(_delta.prefix, "pyDeltaRCM_output.nc"))

        # can be resumed
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True, "resume_checkpoint": True}
        )
        _delta = DeltaModel(input_file=p)

        # check that fields match
        assert np.all(_delta.eta == _rand_field)

    def test_load_checkpoint_without_netcdf(self, tmp_path: Path) -> None:
        """Test that a run can be resumed when there are outputs."""
        # define a yaml with NO outputs, but checkpoint
        p = utilities.yaml_from_dict(tmp_path, "input.yaml", {"save_checkpoint": True})
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # should be no nc file but should be a checkpoint file
        assert not os.path.isfile(os.path.join(_delta.prefix, "pyDeltaRCM_output.nc"))
        assert os.path.isfile(os.path.join(_delta.prefix, "checkpoint.npz"))

        # now try to resume, will WARN on not finding netcdf
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_checkpoint": True,
                "save_eta_grids": True,
                "resume_checkpoint": True,
            },
        )
        with pytest.warns(UserWarning, match=r"NetCDF4 output *."):
            _delta = DeltaModel(input_file=p)

        # assert that a new output file exists file exists
        assert os.path.isfile(os.path.join(_delta.prefix, "pyDeltaRCM_output.nc"))
        assert os.path.isfile(os.path.join(_delta.prefix, "checkpoint.npz"))

        # check that fields match
        assert np.all(_delta.eta == _rand_field)

    @mock.patch(
        "pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep",
        new=utilities.FastIteratingDeltaModel.solve_water_and_sediment_timestep,
    )
    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_load_ckpt_wo_netcdf_parallel_spinup_to_matrix(
        self, tmp_path: Path
    ) -> None:
        """
        Test that multiple matrix runs can be resumed from a single checkpoint
        file, and take advantage of the preprocessor parallel infrastructure.
        """
        # define a yaml with an output and checkpoint
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True, "save_eta_grids": True}
        )
        baseModel = DeltaModel(input_file=p)

        # run base for 2 timesteps
        for _ in range(0, 50):
            baseModel.update()
        baseModel.finalize()

        # get the number of times the data has been output in the base file
        bmsi = baseModel._save_iter
        assert bmsi > 0

        # check that files exist, and then delete nc
        assert os.path.isfile(os.path.join(baseModel.prefix, "pyDeltaRCM_output.nc"))
        assert os.path.isfile(os.path.join(baseModel.prefix, "checkpoint.npz"))

        # open the file and check dimensions
        exp_nc_path = os.path.join(baseModel.prefix, "pyDeltaRCM_output.nc")
        output = Dataset(exp_nc_path, "r", allow_pickle=True)
        out_vars = output.variables.keys()
        # check that expected variables are in the file
        assert "x" in out_vars
        assert "y" in out_vars
        assert "seconds" in out_vars
        assert "eta" in out_vars
        # check attributes of variables
        assert output["seconds"].shape[0] == bmsi

        ########################

        # set up a matrix of runs
        resume_dict = {
            "save_checkpoint": False,
            "resume_checkpoint": True,
            "save_eta_grids": True,
            "out_dir": os.path.join(tmp_path, "matrix"),
            "parallel": 4,
        }
        _matrix = {"f_bedload": [0.1, 0.2, 0.5, 1]}
        resume_dict["matrix"] = _matrix

        # let the preprocessor write the initial matrix and
        #    create a new output netcdf file
        pp = preprocessor.Preprocessor(resume_dict, timesteps=25)  # 2000

        # now copy the checkpoint
        for j, j_file in enumerate(pp.file_list):
            # get folder
            j_folder = j_file.parent

            # copy the spinup checkpoint to each of the folders
            shutil.copy(
                src=os.path.join(baseModel.prefix, "checkpoint.npz"),
                dst=os.path.join(tmp_path, "matrix", j_folder),
            )

        pp.run_jobs()

        # first job save dimension
        fjsi = pp.job_list[0].deltamodel._save_iter

        # proceed with assertions
        for j, j_job in enumerate(pp.job_list):
            exp_nc_path = os.path.join(
                tmp_path, "matrix", j_folder, "pyDeltaRCM_output.nc"
            )
            assert os.path.isfile(exp_nc_path)

            # check all jobs have same dimensionality
            jsi = j_job.deltamodel._save_iter
            assert jsi == fjsi

            # open the file and check dimensions
            output = Dataset(exp_nc_path, "r", allow_pickle=True)
            out_vars = output.variables.keys()
            # check that expected variables are in the file
            assert "x" in out_vars
            assert "y" in out_vars
            assert "seconds" in out_vars
            assert "eta" in out_vars
            # check attributes of variables

            # this is the critical check, that the dimension of the second
            # netcdf is not expanded to begin at _save_iter from the initial
            # basemodel
            assert output["seconds"].shape[0] <= (bmsi // 2)

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="OS differences regarding netCDF permissions.",
    )
    def test_load_checkpoint_with_open_netcdf(self, tmp_path: Path) -> None:
        """Test what happens if output netCDF file is actually open.

        This is not the same as when the netCDF file is open in another
        process. That situation raises an error for all OS.
        """
        # define a yaml with outputs (defaults will output strata)
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True, "save_eta_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # paths exists
        assert os.path.isfile(os.path.join(_delta.prefix, "pyDeltaRCM_output.nc"))
        assert os.path.isfile(os.path.join(_delta.prefix, "checkpoint.npz"))
        _prefix = _delta.prefix
        _delta = []  # clear

        # open the netCDF file
        _opened = Dataset(
            os.path.join(_prefix, "pyDeltaRCM_output.nc"), "r+", format="NETCDF4"
        )
        assert type(_opened) == Dataset
        assert "eta" in _opened.variables.keys()
        # saved grid is the initial one, before random field was assigned
        assert np.all(_opened.variables["eta"][:].data != _rand_field)

        # can be resumed
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_checkpoint": True,
                "resume_checkpoint": True,
                "save_eta_grids": True,
            },
        )
        _delta = DeltaModel(input_file=p)
        # force save grids/figs
        _delta.save_grids_and_figs()

        # check that fields match
        assert np.all(_delta.eta == _rand_field)
        # assert that old netCDF object is still around and hasn't changed
        assert type(_opened) == Dataset
        assert "eta" in _opened.variables.keys()
        # grid from old netCDF is initial one, before random field was assigned
        assert np.all(_opened.variables["eta"][:].data != _rand_field)

        # clear delta
        _delta = []
        # open the new netCDF file
        _new = Dataset(
            os.path.join(_prefix, "pyDeltaRCM_output.nc"), "r+", format="NETCDF4"
        )
        # first grid should be the OG one
        assert np.all(_opened["eta"][:].data == _new["eta"][0, :, :].data)
        # random field should be saved in the new netCDF file
        # some rounding/truncation happens in the netCDF so we use approx
        assert pytest.approx(_rand_field) == _new["eta"][1, :, :].data

    @pytest.mark.skipif(
        platform.system() != "Windows",
        reason="OS differences regarding netCDF permissions.",
    )
    def test_load_checkpoint_with_open_netcdf_win(self, tmp_path: Path) -> None:
        """Test what happens if output netCDF file is actually open.

        This is not the same as when the netCDF file is open in another
        process. That situation raises an error for all OS.
        """
        # define a yaml with outputs (defaults will output strata)
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True, "save_eta_grids": True}
        )
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # paths exists
        assert os.path.isfile(os.path.join(_delta.prefix, "pyDeltaRCM_output.nc"))
        assert os.path.isfile(os.path.join(_delta.prefix, "checkpoint.npz"))
        _prefix = _delta.prefix
        _delta = []  # clear

        # open the netCDF file
        _opened = Dataset(
            os.path.join(_prefix, "pyDeltaRCM_output.nc"), "r+", format="NETCDF4"
        )
        assert type(_opened) == Dataset
        assert "eta" in _opened.variables.keys()
        # saved grid is the initial one, before random field was assigned
        assert np.all(_opened.variables["eta"][:].data != _rand_field)

        # can be resumed
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_checkpoint": True,
                "resume_checkpoint": True,
                "save_eta_grids": True,
            },
        )
        # raises a permissions error on Windows
        with pytest.raises(PermissionError):
            _ = DeltaModel(input_file=p)

    def test_checkpoint_counter_round_trip(self, tmp_path: Path) -> None:
        """Test that the checkpoint counter survives a save-load round trip.

        ``save_the_checkpoint`` must record ``_save_time_since_checkpoint``
        and ``load_checkpoint`` must restore it exactly, so a resumed run
        continues the checkpoint cadence of the uninterrupted run.
        """
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"save_checkpoint": True}
        )
        _delta = DeltaModel(input_file=p)

        # set the counter to a distinctive value and save the checkpoint
        _delta._save_time_since_checkpoint = 1234.5
        _delta.save_the_checkpoint()
        _delta.finalize()

        # the counter is in the saved checkpoint file
        with np.load(os.path.join(_delta.prefix, "checkpoint.npz")) as z:
            assert float(z["save_time_since_checkpoint"]) == 1234.5

        # resume from the checkpoint
        p_resume = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_checkpoint": True, "resume_checkpoint": True},
        )
        _resume = DeltaModel(input_file=p_resume)

        # the counter survives the save-load round trip exactly
        assert _resume._save_time_since_checkpoint == 1234.5

    def _rewrite_netcdf_group(self, prefix: Path, new_group_name: str) -> None:
        """Rewrite the output netCDF with its group renamed.

        The copy mirrors the dims/vars/groups copy performed by
        ``load_checkpoint``, so the rewritten file remains a valid output
        file for resuming.
        """
        nc_path = os.path.join(prefix, "pyDeltaRCM_output.nc")
        assert os.path.isfile(nc_path)
        tmp_nc = os.path.join(prefix, "rewritten.nc")
        with Dataset(nc_path, "r") as src, Dataset(
            tmp_nc, "w", format="NETCDF4"
        ) as dst:
            # copy attributes
            for name in src.ncattrs():
                dst.setncattr(name, src.getncattr(name))
            # copy dimensions
            for name, dimension in src.dimensions.items():
                if dimension.isunlimited():
                    dst.createDimension(name, None)
                else:
                    dst.createDimension(name, len(dimension))
            # copy root variables
            for name, variable in src.variables.items():
                dst.createVariable(name, variable.datatype, variable.dimensions)
                dst.variables[name][:] = src.variables[name][:]
            # copy groups under the new name; with new_group_name None,
            #   skip group copying entirely, producing a file with no
            #   groups at all
            if new_group_name is not None:
                for name in src.groups.keys():
                    dst.createGroup(new_group_name)
                    for vname, variable in src.groups[name].variables.items():
                        dst.groups[new_group_name].createVariable(
                            vname, variable.datatype, variable.dimensions
                        )
                        dst.groups[new_group_name].variables[vname][:] = src.groups[
                            name
                        ].variables[vname][:]
        os.replace(tmp_nc, nc_path)

    def test_load_checkpoint_subgroup_fallback(self, tmp_path: Path) -> None:
        """Test resuming against an output file without meta/auxdata groups.

        The subgroup name fallback must resolve the first group of the
        file (e.g., for a file written by a subclass with a custom
        subgroup name) instead of raising a TypeError by subscripting the
        ``dict_keys`` view.
        """
        # base run with a netCDF output and a checkpoint
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_checkpoint": True, "save_eta_grids": True},
        )
        _delta = DeltaModel(input_file=p)
        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force a checkpoint
        _delta.finalize()

        # rewrite the output file with a custom group name, so the
        #   resumed model must fall back to the first group
        self._rewrite_netcdf_group(_delta.prefix, "customgroup")

        # resume from the checkpoint
        p_resume = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_checkpoint": False,
                "save_eta_grids": True,
                "resume_checkpoint": True,
            },
        )
        _resume = DeltaModel(input_file=p_resume)

        # the subgroup name falls back to the first group of the file
        assert _resume._subgroup_name == "customgroup"
        _resume.finalize()

    def test_load_checkpoint_no_groups_raises_clear_error(
        self, tmp_path: Path
    ) -> None:
        """Test resuming against an output file with no groups at all.

        The subgroup name fallback must raise a clear error for a file
        with no groups, rather than a TypeError from subscripting the
        ``dict_keys`` view.
        """
        # base run with a netCDF output and a checkpoint
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {"save_checkpoint": True, "save_eta_grids": True},
        )
        _delta = DeltaModel(input_file=p)
        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force a checkpoint
        _delta.finalize()

        # rewrite the output file with no groups at all
        self._rewrite_netcdf_group(_delta.prefix, None)

        # resume from the checkpoint raises a clear error
        p_resume = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "save_checkpoint": False,
                "save_eta_grids": True,
                "resume_checkpoint": True,
            },
        )
        with pytest.raises(RuntimeError, match="no groups"):
            _ = DeltaModel(input_file=p_resume)
