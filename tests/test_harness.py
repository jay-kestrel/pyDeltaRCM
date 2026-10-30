"""Unit tests for the harness package.

Record construction and the runner are exercised against a deterministic
stub model (the ``FastIteratingDeltaModel`` pattern: a tiny grid with the
expensive solve patched out), plus one run on a real ``DeltaModel`` with
the patched solver to pin the attribute surface. No long runs here —
integration tests over the scenario catalog live with the scenarios.
"""

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pyDeltaRCM.model import DeltaModel

from harness import metrics
from harness.run_scenario import duration_steps, main, run_scenario_file
from . import utilities


class StubDeltaModel:
    """Deterministic DeltaModel stand-in with the harness attribute surface.

    Mirrors the model behaviors the harness relies on: ``out_dir`` is read
    from the YAML and created at init (the only always-written outputs are
    the directory and its log), ``eta0`` holds the pre-step bed after each
    update, and ``Vp_dep_*`` are per-cell m³ volumes for the current step.
    """

    __pyDeltaRCM_version__ = "stub-0.0"

    def __init__(self, input_file=None, **kwargs):
        with open(input_file) as f:
            self._config = yaml.safe_load(f)
        out_dir = Path(self._config["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        self.L = 10
        self.W = 10
        self.dt = 6250.0
        self.dVs = 625.0  # == Qs0 * dt, mirroring dt = dVs / Qs0
        self.Qw0 = 100.0
        self.Qs0 = 0.1
        self.Np_water = 50
        self.Np_sed = 50
        self.dry_depth = 0.5
        self.save_iter = 1
        self.seed = 14
        self.eta0 = np.zeros((self.L, self.W), dtype=np.float32)
        self.eta = np.zeros((self.L, self.W), dtype=np.float32)
        self.depth = np.full((self.L, self.W), 2.0, dtype=np.float32)
        self.Vp_dep_sand = np.zeros((self.L, self.W), dtype=np.float32)
        self.Vp_dep_mud = np.zeros((self.L, self.W), dtype=np.float32)
        self._time = 0.0
        self._time_iter = 0
        self.finalized = False

    @property
    def time(self) -> float:
        return self._time

    @property
    def time_iter(self) -> int:
        return self._time_iter

    def update(self) -> None:
        self.eta0 = self.eta.copy()
        self.eta = self.eta + np.float32(0.25)
        self.Vp_dep_sand[:, :5] = np.float32(1.0)  # 50 cells at 1.0 m3
        self.Vp_dep_mud[:, 5:] = np.float32(0.5)  # 50 cells at 0.5 m3
        self._time_iter += 1
        self._time += self.dt

    def finalize(self) -> None:
        self.finalized = True


_STUB_SCENARIO = {
    "Length": 500.0,
    "Width": 500.0,
    "dx": 50.0,
    "seed": 14,
    "timesteps": 3,
    "out_dir": "runs/somewhere",  # the harness must replace this
}

# real-model config: 12x12 cells at dx=1, warning-free snapping (N0 = L0 = 3
# cells, exact multiples of dx) and no netCDF writers; dt derives to 300 s
_REAL_SCENARIO = {
    "Length": 12.0,
    "Width": 12.0,
    "dx": 1.0,
    "N0_meters": 3.0,
    "L0_meters": 3.0,
    "u0": 1.0,
    "h0": 1.0,
    "S0": 0.0002,
    "Np_water": 10,
    "Np_sed": 10,
    "f_bedload": 0.5,
    "C0_percent": 0.1,
    "itermax": 1,
    "seed": 0,
    "save_dt": 10**12,
    "out_dir": "runs/somewhere",
    "timesteps": 2,
}


class _FastScenarioModel(utilities.FastIteratingDeltaModel, DeltaModel):
    """Real DeltaModel with the patched cheap solver from tests/utilities."""


def _write_scenario(directory: Path, name: str, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _assert_schema(record: dict) -> None:
    """Every field of the spec's sample record is present, grouped as such."""
    assert set(record) == {
        "scenario",
        "seed",
        "pyDeltaRCM_version",
        "host",
        "speed",
        "efficiency",
        "workload",
        "outputs",
        "eta_checksum",
    }
    assert set(record["host"]) == {"python", "platform", "cpu_count"}
    assert set(record["speed"]) == {
        "wall_init_s",
        "wall_first_step_s",
        "wall_run_s",
        "wall_total_s",
        "step_ms_median",
        "step_ms_p90",
        "step_ms_series",
    }
    assert set(record["efficiency"]) == {
        "steps_per_s",
        "cell_steps_per_s",
        "model_s_per_wall_s",
    }
    assert set(record["workload"]) == {
        "model_time_end_s",
        "time_iter",
        "dt_s",
        "grid_L",
        "grid_W",
        "cells",
        "parcels_total",
        "Qw0_m3_per_s",
        "Qs0_m3_per_s",
        "water_processed_m3",
        "sediment_processed_m3",
        "cells_changed_total",
        "dep_vol_sand_m3",
        "dep_vol_mud_m3",
        "wet_cells_final",
    }
    assert set(record["outputs"]) == {"save_iter", "out_dir_bytes", "files"}

    for group in ("speed", "efficiency", "workload", "outputs"):
        for value in record[group].values():
            if isinstance(value, float):
                assert np.isfinite(value), f"non-finite {group} value: {value}"


def test_record_schema_completeness(tmp_path):
    scenario = _write_scenario(tmp_path, "00_smoke.yml", _STUB_SCENARIO)
    out_root = tmp_path / "runs"

    record = run_scenario_file(scenario, out_root, model_cls=StubDeltaModel)

    _assert_schema(record)
    on_disk = json.loads((out_root / "00_smoke" / "metrics.json").read_text())
    assert on_disk == record

    workload = record["workload"]
    assert record["scenario"] == "00_smoke"
    assert record["seed"] == 14
    assert workload["time_iter"] == 3
    assert workload["model_time_end_s"] == pytest.approx(3 * 6250.0)
    assert workload["water_processed_m3"] > 0
    assert workload["sediment_processed_m3"] > 0
    assert workload["cells_changed_total"] == 300  # 100 cells x 3 steps
    assert workload["wet_cells_final"] == 100  # depth 2.0 > dry_depth 0.5
    assert record["eta_checksum"] == pytest.approx(75.0)  # 100 cells x 0.25 x 3
    assert record["outputs"] == {"save_iter": 1, "out_dir_bytes": 0, "files": []}


def test_volume_cross_checks(tmp_path):
    scenario = _write_scenario(tmp_path, "00_smoke.yml", _STUB_SCENARIO)

    record = run_scenario_file(scenario, tmp_path / "runs", model_cls=StubDeltaModel)

    workload = record["workload"]
    assert workload["water_processed_m3"] == pytest.approx(
        workload["Qw0_m3_per_s"] * workload["model_time_end_s"]
    )
    assert workload["sediment_processed_m3"] == pytest.approx(
        workload["Qs0_m3_per_s"] * workload["model_time_end_s"]
    )
    assert workload["sediment_processed_m3"] == pytest.approx(
        625.0 * workload["time_iter"]  # dVs * time_iter
    )
    assert workload["dep_vol_sand_m3"] == pytest.approx(50.0 * workload["time_iter"])
    assert workload["dep_vol_mud_m3"] == pytest.approx(25.0 * workload["time_iter"])
    assert workload["parcels_total"] == 100 * workload["time_iter"]


def test_compute_speed_steady_state_excludes_first_step():
    speed = metrics.compute_speed([1000.0, 100.0, 300.0], 1.0, 5.0)

    assert speed["wall_first_step_s"] == 1.0
    assert speed["wall_run_s"] == pytest.approx(1.4)
    assert speed["wall_total_s"] == 5.0
    assert speed["step_ms_median"] == pytest.approx(200.0)
    assert speed["step_ms_p90"] == pytest.approx(280.0)
    assert speed["step_ms_series"] == [1000.0, 100.0, 300.0]


def test_compute_speed_single_step_falls_back_to_full_series():
    speed = metrics.compute_speed([250.0], 0.5, 1.0)

    assert speed["step_ms_median"] == 250.0
    assert speed["step_ms_p90"] == 250.0


def test_compute_efficiency_arithmetic():
    efficiency = metrics.compute_efficiency(14, 1600, 19.68, 87500.0)

    assert efficiency["steps_per_s"] == pytest.approx(14 / 19.68)
    assert efficiency["cell_steps_per_s"] == pytest.approx(1600 * 14 / 19.68)
    assert efficiency["model_s_per_wall_s"] == pytest.approx(87500.0 / 19.68)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_iter": 1, "cells": 100, "wall_run_s": 0.0, "model_time_end_s": 10.0},
        {"time_iter": 1, "cells": 100, "wall_run_s": -1.0, "model_time_end_s": 10.0},
        {"time_iter": 0, "cells": 100, "wall_run_s": 1.0, "model_time_end_s": 10.0},
    ],
)
def test_compute_efficiency_rejects_nonpositive_work(kwargs):
    with pytest.raises(ValueError):
        metrics.compute_efficiency(**kwargs)


def test_compute_outputs_walks_and_sums(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "pyDeltaRCM_run.log").write_text("x" * 100)
    (tmp_path / "sub" / "eta_00000.png").write_text("y" * 50)

    outputs = metrics.compute_outputs(tmp_path, save_iter=2)

    assert outputs == {
        "save_iter": 2,
        "out_dir_bytes": 150,
        "files": ["pyDeltaRCM_run.log", "sub/eta_00000.png"],
    }


def test_compute_outputs_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        metrics.compute_outputs(tmp_path / "absent", save_iter=1)


def _writer_record(scenario: str) -> dict:
    """A fully-populated record for writer tests (no model needed)."""
    speed = metrics.compute_speed([100.0, 200.0], 1.0, 2.0)
    efficiency = metrics.compute_efficiency(2, 100, speed["wall_run_s"], 12500.0)
    model = SimpleNamespace(
        L=10,
        W=10,
        Qw0=100.0,
        Qs0=0.1,
        dt=6250.0,
        Np_water=50,
        Np_sed=50,
        dry_depth=0.5,
        save_iter=1,
        time=12500.0,
        time_iter=2,
        depth=np.full((10, 10), 2.0),
    )
    workload = metrics.compute_workload(
        model,
        cells_changed_total=7,
        dep_vol_sand_m3=1.0,
        dep_vol_mud_m3=2.0,
        model_time_end_s=12500.0,
    )
    return metrics.build_record(
        scenario=scenario,
        seed=14,
        version="2.3.0",
        host=metrics.host_info(),
        speed=speed,
        efficiency=efficiency,
        workload=workload,
        outputs={"save_iter": 1, "out_dir_bytes": 10, "files": ["run.log"]},
        eta_checksum=-1.5,
    )


def test_summary_csv_writer():
    rows = list(
        csv.reader(
            io.StringIO(
                metrics.summary_csv(
                    [_writer_record("00_smoke"), _writer_record("01_classic")]
                )
            )
        )
    )

    assert rows[0][0] == "scenario"
    assert len(rows) == 3
    assert rows[1][0] == "00_smoke"
    assert rows[2][0] == "01_classic"
    checksum_column = rows[0].index("eta_checksum")
    assert float(rows[1][checksum_column]) == -1.5
    water_column = rows[0].index("water_processed_m3")
    assert float(rows[1][water_column]) == pytest.approx(100.0 * 12500.0)


def test_summary_markdown_writer():
    markdown = metrics.summary_markdown([_writer_record("00_smoke")])

    lines = markdown.splitlines()
    assert lines[0] == "# pyDeltaRCM scenario run summary"
    assert lines[2].startswith("| scenario | seed | time_iter |")
    assert lines[3].startswith("| --- |")
    assert len(lines) == 5  # title, blank, header, separator, one row
    assert "| 00_smoke | 14 | 2 |" in lines[4]
    # checksum keeps full float precision in the table
    assert "-1.5" in lines[4]


def test_out_dir_injected_through_temp_yaml(tmp_path):
    scenario = _write_scenario(tmp_path, "00_smoke.yml", _STUB_SCENARIO)
    seen = {}

    class RecordingModel(StubDeltaModel):
        def __init__(self, input_file=None, **kwargs):
            super().__init__(input_file=input_file, **kwargs)
            seen["input_file"] = Path(input_file)

    run_scenario_file(scenario, tmp_path / "runs", model_cls=RecordingModel)

    assert seen["input_file"].name == "scenario.yml"
    assert not seen["input_file"].exists()  # per-run temp workspace cleaned up
    original = yaml.safe_load(scenario.read_text())
    assert original["out_dir"] == "runs/somewhere"  # scenario file untouched
    assert not Path("runs/somewhere").exists()  # nothing created from it


def test_duration_steps_requires_timesteps():
    assert duration_steps({"timesteps": 40}) == 40
    with pytest.raises(ValueError):
        duration_steps({"time": 86400.0})
    with pytest.raises(ValueError):
        duration_steps({"timesteps": 3, "time": 86400.0})
    with pytest.raises(ValueError):
        duration_steps({})
    with pytest.raises(ValueError):
        duration_steps({"timesteps": 0})


def test_cli_main_runs_scenarios_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.run_scenario.DeltaModel", StubDeltaModel)
    first = _write_scenario(tmp_path, "00_smoke.yml", _STUB_SCENARIO)
    second = _write_scenario(
        tmp_path, "06_other.yml", {**_STUB_SCENARIO, "timesteps": 2}
    )
    out_root = tmp_path / "out"

    exit_code = main([str(first), str(second), "--out", str(out_root), "--summary"])

    assert exit_code == 0
    assert (out_root / "00_smoke" / "metrics.json").exists()
    assert (out_root / "06_other" / "metrics.json").exists()
    summary_rows = list(csv.reader((out_root / "summary.csv").open(newline="")))
    assert len(summary_rows) == 3  # header + two runs
    summary_md = (out_root / "summary.md").read_text()
    assert "00_smoke" in summary_md
    assert "06_other" in summary_md


def test_keep_outputs_preserves_model_out_dir(tmp_path):
    class WritingModel(StubDeltaModel):
        def __init__(self, input_file=None, **kwargs):
            super().__init__(input_file=input_file, **kwargs)
            out_dir = Path(self._config["out_dir"])
            (out_dir / "pyDeltaRCM_stub.log").write_text("x" * 100)

    scenario = _write_scenario(tmp_path, "00_smoke.yml", _STUB_SCENARIO)
    out_root = tmp_path / "runs"

    record = run_scenario_file(
        scenario, out_root, keep_outputs=True, model_cls=WritingModel
    )

    assert (out_root / "00_smoke" / "out" / "pyDeltaRCM_stub.log").exists()
    assert record["outputs"]["out_dir_bytes"] == 100
    assert record["outputs"]["files"] == ["pyDeltaRCM_stub.log"]
    # a re-run over the same run_dir replaces the outputs instead of failing
    run_scenario_file(scenario, out_root, keep_outputs=True, model_cls=WritingModel)
    assert (out_root / "00_smoke" / "metrics.json").exists()


def test_without_keep_outputs_temp_outputs_are_removed(tmp_path):
    scenario = _write_scenario(tmp_path, "00_smoke.yml", _STUB_SCENARIO)

    run_scenario_file(scenario, tmp_path / "runs", model_cls=StubDeltaModel)

    assert not (tmp_path / "runs" / "00_smoke" / "out").exists()


def test_run_scenario_file_on_real_model(tmp_path):
    scenario = _write_scenario(tmp_path, "00_smoke.yml", _REAL_SCENARIO)

    record = run_scenario_file(
        scenario, tmp_path / "runs", model_cls=_FastScenarioModel
    )

    _assert_schema(record)
    workload = record["workload"]
    assert workload["time_iter"] == 2
    assert workload["grid_L"] == 12 and workload["grid_W"] == 12
    assert workload["dt_s"] == pytest.approx(300.0)  # dVs / Qs0 for this config
    assert workload["water_processed_m3"] == pytest.approx(
        workload["Qw0_m3_per_s"] * workload["model_time_end_s"]
    )
    assert workload["sediment_processed_m3"] == pytest.approx(
        workload["Qs0_m3_per_s"] * workload["model_time_end_s"]
    )
    assert record["seed"] == 0
    assert record["pyDeltaRCM_version"]
    assert np.isfinite(record["eta_checksum"])
