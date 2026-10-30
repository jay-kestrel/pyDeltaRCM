import glob
import locale
import os
import platform
import subprocess
from pathlib import Path

import pytest
import yaml

from .. import utilities
import pyDeltaRCM as _pyimportedalias
from pyDeltaRCM.shared_tools import custom_yaml_loader


class TestCommandLineInterfaceDirectly:

    # test the entry points
    def test_entry_point_installed_call(self, tmp_path: Path) -> None:
        """
        test calling the command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'timesteps', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['pyDeltaRCM',
                                 '--config', str(p)])
        exp_path_nc = os.path.join(tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00001.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)

    def test_entry_point_python_main_call(self, tmp_path: Path) -> None:
        """
        test calling the python hook command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'timesteps', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p)])
        exp_path_nc = os.path.join(tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_png = os.path.join(tmp_path / 'test', 'job_000', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00001.png')
        exp_path_png2 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00002.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png)
        assert os.path.isfile(exp_path_png1)
        assert not os.path.isfile(exp_path_png2)

    def test_entry_point_python_main_call_dryrun(self, tmp_path: Path) -> None:
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'timesteps', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p),
                                 '--dryrun'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_png = os.path.join(tmp_path / 'test', 'job_000', 'eta_00000.png')
        assert not os.path.isfile(exp_path_nc)   # does not exist because --dryrun
        assert not os.path.isfile(exp_path_png)  # does not exist because --dryrun

    def test_entry_point_python_main_call_timesteps(self, tmp_path: Path) -> None:
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p),
                                 '--timesteps', '2'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_png = os.path.join(tmp_path / 'test', 'job_000', 'eta_00000.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png)

    def test_error_if_no_timesteps(self, tmp_path: Path) -> None:
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                     '--config', str(p)])

    def test_entry_point_timesteps(self, tmp_path: Path) -> None:
        """
        test calling the command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['pyDeltaRCM',
                                 '--config', str(p), '--timesteps', '2'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00001.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)

    def test_entry_point_time(self, tmp_path: Path) -> None:
        """
        test calling the command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['pyDeltaRCM',
                                 '--config', str(p), '--time', '1000'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'job_000', 'eta_00001.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)

    def test_version_call(self) -> None:
        """
        test calling the command line feature to query the version.
        """
        encoding = locale.getpreferredencoding()
        printed1 = subprocess.run(['pyDeltaRCM', '--version'],
                                  stdout=subprocess.PIPE, encoding=encoding)
        _exp_str1 = 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
        assert printed1.stdout == _exp_str1
        printed2 = subprocess.run(
            ['python', '-m', 'pyDeltaRCM', '--version'],
            stdout=subprocess.PIPE, encoding=encoding)
        _exp_str2 = 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
        assert printed2.stdout == _exp_str2


class TestQolCommandLineFlags:

    def test_qol_new_flag(self, tmp_path: Path) -> None:
        """
        test the --new flag scaffolds a starter configuration file.
        """
        p = tmp_path / 'starter.yml'
        subprocess.check_output(['pyDeltaRCM', '--new', str(p)])
        assert p.is_file()
        with open(p, mode='r') as f:
            loaded = yaml.load(f, Loader=custom_yaml_loader())
        assert loaded['timesteps'] == 10

    def test_qol_new_flag_python_main_call(self, tmp_path: Path) -> None:
        """
        test the --new flag via the python -m entry point.
        """
        p = tmp_path / 'starter.yml'
        subprocess.check_output(['python', '-m', 'pyDeltaRCM', '--new', str(p)])
        assert p.is_file()

    def test_qol_new_flag_refuses_overwrite(self, tmp_path: Path) -> None:
        """
        test the --new flag does not overwrite an existing file.
        """
        p = tmp_path / 'starter.yml'
        p.write_text('existing: file\n')
        _run = subprocess.run(
            ['pyDeltaRCM', '--new', str(p)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding=locale.getpreferredencoding())
        assert _run.returncode == 1
        assert p.read_text() == 'existing: file\n'

    def test_qol_validate_flag_valid_config(self, tmp_path: Path) -> None:
        """
        test the --validate flag passes a valid config and writes nothing.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'timesteps', 2)
        f.close()
        printed = subprocess.check_output(
            ['pyDeltaRCM', '--validate', str(p)],
            stderr=subprocess.STDOUT)
        assert b'OK' in printed
        # validation must not create the output directory
        assert not (tmp_path / 'test').exists()

    def test_qol_validate_flag_invalid_config(self, tmp_path: Path) -> None:
        """
        test the --validate flag reports unknown keys and type errors.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'banana', 3)
        utilities.write_parameter_to_file(f, 'Np_sed', 'many')
        utilities.write_parameter_to_file(f, 'timesteps', 2)
        f.close()
        _run = subprocess.run(
            ['pyDeltaRCM', '--validate', str(p)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding=locale.getpreferredencoding())
        assert _run.returncode == 1
        assert 'banana' in _run.stdout
        assert 'Np_sed' in _run.stdout

    def test_qol_validate_flag_missing_file(self, tmp_path: Path) -> None:
        """
        test the --validate flag reports a missing configuration file.
        """
        _run = subprocess.run(
            ['pyDeltaRCM', '--validate', str(tmp_path / 'nope.yml')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding=locale.getpreferredencoding())
        assert _run.returncode == 1
        assert 'not found' in _run.stdout

    def test_qol_demo_flag(self, tmp_path: Path) -> None:
        """
        test the --demo flag writes and runs a tiny demonstration model.
        """
        env = os.environ.copy()
        env['PYDELTARCM_NATIVE'] = '0'
        demo_dir = tmp_path / 'demo'
        subprocess.check_output(
            ['pyDeltaRCM', '--demo', str(demo_dir)], env=env)
        # the demo configuration was written and run through the standard path
        assert (demo_dir / 'pyDeltaRCM_demo.yaml').is_file()
        exp_path_nc = (demo_dir / 'pyDeltaRCM_demo_output' / 'job_000' /
                       'pyDeltaRCM_output.nc')
        assert exp_path_nc.is_file()
        figs = glob.glob(
            str(demo_dir / 'pyDeltaRCM_demo_output' / 'job_000' / 'eta_*.png'))
        assert len(figs) > 0
