import pytest

import io
import os
import sys
import glob
import logging
from pathlib import Path

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM.init_tools import init_tools
from pyDeltaRCM.shared_tools import ParameterChangedWarning

from . import utilities


class TestLogModelTime:

    def test_records_arbitrary_time_values(self, tmp_path: Path, capsys) -> None:
        """
        This test should create the log, and then print nothing at all.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.logger = mock.MagicMock()

        # change values
        _delta._time = 3.14159
        _delta._time_iter = 42

        _delta.log_model_time()

        # will record whatever value is in _delta
        _calls = [mock.call('Time: 3.1; timestep: 42')]
        _delta.logger.info.assert_has_calls(_calls)

    def test_verbose_printing_0(self, tmp_path: Path, capsys) -> None:
        """
        At verbose 0, the time is recorded in the log file and stdout
        stays empty.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 0)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.log_model_time()

        # stdout is empty because verbose 0
        captd = capsys.readouterr()
        assert captd.out == ''

        # time is logged regardless of verbosity
        _logs = glob.glob(os.path.join(_delta.prefix, '*.log'))
        with open(_logs[0], 'r') as _logfile:
            _lines = ' '.join(_logfile.readlines())
        assert 'Time: 0.0; timestep: 0' in _lines

    def test_verbose_printing_1(self, tmp_path: Path, capsys) -> None:
        """
        At verbose 1, stdout carries exactly the terse Time lines.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 1)
        f.close()
        _delta = DeltaModel(input_file=p)

        # initialization reports the time exactly once, with no other
        # stdout traffic
        captd = capsys.readouterr()
        assert captd.out == 'Time: 0.0; timestep: 0\n'

        # and each timestep report adds exactly one identical line
        _delta.log_model_time()
        captd = capsys.readouterr()
        assert captd.out == 'Time: 0.0; timestep: 0\n'

    def test_verbose_printing_2(self, tmp_path: Path, capsys) -> None:
        """
        At verbose 2, stdout carries exactly the terse Time lines.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 2)
        f.close()
        _delta = DeltaModel(input_file=p)

        captd = capsys.readouterr()
        assert captd.out == 'Time: 0.0; timestep: 0\n'

        _delta.log_model_time()
        captd = capsys.readouterr()
        assert captd.out == 'Time: 0.0; timestep: 0\n'

    def test_verbose_1_console_marked_record(self, tmp_path: Path) -> None:
        """
        At verbose 1, the time record is emitted with the console mark that
        the console handler filters on.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 1)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.logger = mock.MagicMock()
        _delta.log_model_time()

        _delta.logger.info.assert_called_with(
            'Time: 0.0; timestep: 0', extra={'console': True}
        )


class TestLoggerIntegratedDuringInitialization:
    """Test that logger records during instantiation.

    These tests could be set up with a patch on the logger before
    instantiation of the DeltaModel...
    """

    def test_logger_has_initialization_lines(self, tmp_path: Path) -> None:
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 1)
        utilities.write_parameter_to_file(f, 'seed', 10)
        f.close()
        _delta = DeltaModel(input_file=p)

        _logs = glob.glob(os.path.join(_delta.prefix, '*.log'))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], 'r') as _logfile:
            _lines = _logfile.readlines()
            _lines = ' '.join(_lines)  # collapse to a single string
            assert 'Setting model constants' in _lines
            assert 'Random seed is: 10' in _lines
            assert 'Creating model domain' in _lines
            assert 'Initializing output NetCDF4 file' in _lines
            assert 'Model initialization complete' in _lines

            if sys.platform.startswith('linux'):
                assert 'Platform: Linux-' in _lines
            elif sys.platform == 'darwin':
                guess1 = 'Platform: Darwin-' in _lines
                guess2 = 'Platform: macOS-' in _lines
                assert (guess1 | guess2)
            elif sys.platform.startswith('win'):
                assert 'Platform: Windows-' in _lines
            else:
                raise RuntimeError(
                    'Platform type not recognized.')
        assert not os.path.isfile(os.path.join(
            tmp_path, 'out_dir', 'discharge_0.0.png'))
        assert not os.path.isfile(os.path.join(
            tmp_path, 'out_dir', 'eta_0.0.png'))

    def test_logger_random_seed_always_recorded(self, tmp_path: Path) -> None:
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 0)
        # do not set the seed explicitly, let it be set by the model
        f.close()
        delta = DeltaModel(input_file=p)
        _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], 'r') as _logfile:
            _lines = _logfile.readlines()
            _joinedlines = ' '.join(_lines)  # collapse to a single string
            assert 'Random seed is: ' in _joinedlines

            # determine the index of the line
            _idx = ['Random seed is: ' in _l for _l in _lines]
            assert sum(_idx) == 1  # one and only one True in list
            _idx = _idx.index(True)

            # try to covert to int, otherwise fail
            _seed = _lines[_idx].split(':')[-1]  # pull the seed value
            try:
                _intseed = int(_seed)
            except ValueError:
                raise ValueError('Could not convert the seed to int')

            assert _intseed >= 0

    def test_logger_capture_changed_parameters_warnings(self, tmp_path: Path) -> None:
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 1)
        utilities.write_parameter_to_file(f, 'Width', 10001)
        f.close()
        # check that warning is raised
        with pytest.warns(ParameterChangedWarning):
            delta = DeltaModel(input_file=p)
        _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], 'r') as _logfile:
            _lines = _logfile.readlines()
            _joinedlines = ' '.join(_lines)  # collapse to a single string
        # check that warning is logged
        assert "Parameter 'Width' was changed from 10001 to 10000." in _joinedlines


class TestLoggerCoreBehavior:
    """Core logging layer regressions (collision, safety, lifecycle).

    These tests pin the core layer design: loggers namespaced under the
    package logger from a process-local counter, exactly one FileHandler per
    wiring, propagation disabled by default, handler release on finalize,
    and timestep context on log-file records.
    """

    def test_same_second_init_calls_never_stack_handlers(
        self, tmp_path: Path
    ) -> None:
        """Two init_logger calls sharing an out_dir in the same second must
        not attach a second FileHandler to a shared logger."""

        class _StubModel:
            """Exposes only the attributes that init_logger requires."""

            __pyDeltaRCM_version__ = 'test'
            _verbose = 0

            def __init__(self, out_dir: Path) -> None:
                os.makedirs(out_dir, exist_ok=True)
                self.prefix_abspath = str(out_dir)

            def log_info(self, message: str, verbosity: int = 0) -> None:
                if self._verbose >= verbosity:
                    self.logger.info(message)

        stub_a = _StubModel(tmp_path / 'out_dir')
        stub_b = _StubModel(tmp_path / 'out_dir')  # same out_dir, same second

        init_tools.init_logger(stub_a)
        init_tools.init_logger(stub_b)

        # distinct namespaced loggers, never a shared path-named logger
        assert stub_a.logger is not stub_b.logger
        for stub in (stub_a, stub_b):
            assert stub.logger.name.startswith('pyDeltaRCM')
            _fhs = [
                h for h in stub.logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert len(_fhs) == 1

        # each logger's records land in the shared out_dir exactly once
        stub_a.log_info('probe from a', verbosity=0)
        stub_b.log_info('probe from b', verbosity=0)
        _lines = ''
        _out_dir = str(tmp_path / 'out_dir')
        for _log in glob.glob(os.path.join(_out_dir, '*.log')):
            with open(_log, 'r') as _logfile:
                _lines += _logfile.read()
        assert _lines.count('probe from a') == 1
        assert _lines.count('probe from b') == 1

        # re-initialization on a live instance replaces rather than stacks
        _first_name = stub_a.logger.name
        _old_fh = [
            h for h in stub_a.logger.handlers
            if isinstance(h, logging.FileHandler)
        ][0]
        init_tools.init_logger(stub_a)
        assert stub_a.logger.name != _first_name
        _fhs = [
            h for h in stub_a.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(_fhs) == 1
        assert _old_fh.stream is None

    def test_same_second_instances_write_distinct_log_files(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Two instances initialized in the same second must write distinct
        log files: uniqueness comes from the per-instance counter, not from
        clock luck, and each instance's records stay isolated in its file."""

        class _StubModel:
            """Exposes only the attributes that init_logger requires."""

            __pyDeltaRCM_version__ = 'test'
            _verbose = 0

            def __init__(self, out_dir: Path) -> None:
                os.makedirs(out_dir, exist_ok=True)
                self.prefix_abspath = str(out_dir)

            def log_info(self, message: str, verbosity: int = 0) -> None:
                if self._verbose >= verbosity:
                    self.logger.info(message)

        # pin the clock so both initializations land in the same second
        # deterministically, however fast the host executes them
        monkeypatch.setattr(
            "pyDeltaRCM.init_tools.time_lib.strftime",
            lambda *a, **k: "20261030-120000",
        )

        stub_a = _StubModel(tmp_path / 'out_dir')
        stub_b = _StubModel(tmp_path / 'out_dir')  # same out_dir, same second

        init_tools.init_logger(stub_a)
        init_tools.init_logger(stub_b)

        # the two file handlers must target distinct files
        _fh_a = [
            h for h in stub_a.logger.handlers
            if isinstance(h, logging.FileHandler)
        ][0]
        _fh_b = [
            h for h in stub_b.logger.handlers
            if isinstance(h, logging.FileHandler)
        ][0]
        assert _fh_a.baseFilename != _fh_b.baseFilename

        # each instance's records land in its own file and nowhere else
        stub_a.log_info('probe from a', verbosity=0)
        stub_b.log_info('probe from b', verbosity=0)
        with open(_fh_a.baseFilename, 'r') as _fa:
            _lines_a = _fa.read()
        with open(_fh_b.baseFilename, 'r') as _fb:
            _lines_b = _fb.read()
        assert _lines_a.count('probe from a') == 1
        assert 'probe from b' not in _lines_a
        assert _lines_b.count('probe from b') == 1
        assert 'probe from a' not in _lines_b

    def test_import_attaches_only_nullhandler(self) -> None:
        """The package logger carries exactly one NullHandler after import."""
        _pkg_logger = logging.getLogger('pyDeltaRCM')
        assert len(_pkg_logger.handlers) == 1
        assert isinstance(_pkg_logger.handlers[0], logging.NullHandler)

    def test_instance_logger_namespaced_and_not_propagating(
        self, tmp_path: Path
    ) -> None:
        """Instance loggers are pyDeltaRCM-namespaced, never path-named,
        and never propagate to the root logger."""
        p = utilities.yaml_from_dict(
            tmp_path, 'user_parameters.yaml', {'verbose': 0, 'seed': 0})
        _delta = DeltaModel(input_file=p)
        assert _delta.logger.name.startswith('pyDeltaRCM.')
        assert _delta.logger.name != _delta.prefix_abspath
        assert _delta.logger.propagate is False
        _fhs = [
            h for h in _delta.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(_fhs) == 1

    def test_host_configured_root_receives_no_records(
        self, tmp_path: Path
    ) -> None:
        """A host with a configured root handler sees no pyDeltaRCM records."""
        p = utilities.yaml_from_dict(
            tmp_path, 'user_parameters.yaml', {'verbose': 0, 'seed': 0})
        host_stream = io.StringIO()
        root = logging.getLogger()
        _old_level = root.level
        host_handler = logging.StreamHandler(host_stream)
        host_handler.setLevel(logging.INFO)
        root.addHandler(host_handler)
        root.setLevel(logging.INFO)
        try:
            _delta = DeltaModel(input_file=p)
            _delta.update()
            _delta.finalize()
        finally:
            root.removeHandler(host_handler)
            root.setLevel(_old_level)
        assert host_stream.getvalue() == ''

    def test_propagate_escape_hatch_forwards_to_host_root(
        self, tmp_path: Path
    ) -> None:
        """Setting propagate=True forwards records to host root handlers
        exactly once."""
        p = utilities.yaml_from_dict(
            tmp_path, 'user_parameters.yaml', {'verbose': 0, 'seed': 0})
        _delta = DeltaModel(input_file=p)
        host_stream = io.StringIO()
        root = logging.getLogger()
        _old_level = root.level
        host_handler = logging.StreamHandler(host_stream)
        root.addHandler(host_handler)
        root.setLevel(logging.DEBUG)
        try:
            _delta.logger.propagate = True
            _delta.log_info('escape hatch probe', verbosity=0)
        finally:
            _delta.logger.propagate = False
            root.removeHandler(host_handler)
            root.setLevel(_old_level)
        assert host_stream.getvalue().count('escape hatch probe') == 1

    def test_finalize_closes_and_detaches_file_handler(
        self, tmp_path: Path
    ) -> None:
        """finalize() closes and detaches the run's FileHandler."""
        p = utilities.yaml_from_dict(
            tmp_path, 'user_parameters.yaml', {'verbose': 0, 'seed': 0})
        _delta = DeltaModel(input_file=p)
        _fhs = [
            h for h in _delta.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(_fhs) == 1
        _fh = _fhs[0]
        _delta.finalize()
        assert _delta._log_fh is None
        _remaining = [
            h for h in _delta.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert _remaining == []
        assert _fh.stream is None

    def test_records_carry_timestep_context(self, tmp_path: Path) -> None:
        """Log-file records carry the model time/timestep stamp."""
        p = utilities.yaml_from_dict(
            tmp_path, 'user_parameters.yaml', {'verbose': 0, 'seed': 0})
        _delta = DeltaModel(input_file=p)
        _delta.update()
        _delta.log_info('context probe', verbosity=0)
        _delta.finalize()  # flushes and releases the file handler
        _logs = glob.glob(os.path.join(_delta.prefix, '*.log'))
        assert len(_logs) == 1
        with open(_logs[0], 'r') as _logfile:
            _lines = _logfile.readlines()

        _probes = [_l for _l in _lines if 'context probe' in _l]
        assert len(_probes) == 1
        assert (
            f'[t={_delta._time:.1f}, step={_delta._time_iter}]' in _probes[0]
        )

        # initialization records are stamped at t=0 as well
        _init = [
            _l for _l in _lines if 'Output log file initialized' in _l
        ]
        assert len(_init) == 1
        assert '[t=0.0, step=0]' in _init[0]
