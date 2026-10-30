"""Tests for benchmarks/run_reference.py (the Python reference runner).

Unit tests exercise the runner's logic against stub models -- no numba, no
model init. One integration test runs the real runner end-to-end on the
committed small fixture and requires it to reproduce the golden eta values.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "benchmarks"
FIXTURE_DIR = REPO_ROOT / "bench" / "fixture_default_small"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_reference", BENCH / "run_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


class StubModel:
    """Minimal stand-in for DeltaModel: no computation, known eta."""

    def __init__(self, eta_value: float = -1.0):
        self.eta = np.full((5, 5), eta_value, dtype=np.float32)

    def update(self) -> None:
        pass

    def finalize(self) -> None:
        pass


GOLDEN = {
    "checkpoints": {
        "1": [-1.0, -0.9152762, -1.0004134, -1.0, -1.0],
        "10": [1.7, 0.394864, -0.95006764, -1.0, -1.0],
    }
}


def test_check_golden_passes_on_match(runner) -> None:
    model = StubModel()
    model.eta[:, 4] = np.array(GOLDEN["checkpoints"]["1"], dtype=np.float32)
    passed, expected, actual = runner.check_golden(model, GOLDEN, 1)
    assert passed
    assert actual == pytest.approx(GOLDEN["checkpoints"]["1"], rel=1e-6)


def test_check_golden_fails_on_mismatch(runner) -> None:
    model = StubModel(eta_value=2.0)
    passed, expected, actual = runner.check_golden(model, GOLDEN, 1)
    assert not passed
    assert expected == GOLDEN["checkpoints"]["1"]
    assert actual == [2.0] * 5


def test_check_golden_within_rtol(runner) -> None:
    """A 1e-7 relative perturbation passes (pytest.approx-equivalent gate)."""
    model = StubModel()
    model.eta[:, 4] = np.array(GOLDEN["checkpoints"]["1"], dtype=np.float32)
    model.eta[:, 4] *= 1.0 + 1e-7
    passed, _, _ = runner.check_golden(model, GOLDEN, 1)
    assert passed


def test_summarize_timing(runner) -> None:
    stats = runner.summarize_timing([1.0, 2.0, 3.0, 4.0])
    assert stats["n_updates"] == 4
    assert stats["median_s"] == 2.5
    assert stats["mean_s"] == 2.5
    assert stats["min_s"] == 1.0
    assert stats["max_s"] == 4.0
    assert stats["total_s"] == 10.0
    assert stats["sd_s"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_smoke_preset_applied(runner) -> None:
    args = runner.parse_args(["--fixture", "x", "--smoke"])
    assert args.updates == 10
    assert args.warmup == 2
    assert args.repeat == 1


def test_smoke_preset_respects_explicit_flags(runner) -> None:
    args = runner.parse_args(
        ["--fixture", "x", "--smoke", "--updates", "3", "--repeat", "2"]
    )
    assert args.updates == 3
    assert args.repeat == 2
    assert args.warmup == 2


def test_measure_cold_process_parses_json_line(runner, monkeypatch) -> None:
    """Warning lines precede the JSON result line; the last JSON wins."""
    payload = {"import_s": 1.0, "init_s": 2.0, "first_update_s": 3.0, "total_s": 6.0}

    class FakeProc:
        returncode = 0
        stdout = "some numba warning\n" + json.dumps(payload) + "\n"
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: FakeProc())
    result = runner.measure_cold_process(Path("/fake"))
    assert result["import_s"] == 1.0
    assert result["first_update_s"] == 3.0
    assert "measurement" in result


def test_measure_cold_process_raises_on_failure(runner, monkeypatch) -> None:
    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: FakeProc())
    with pytest.raises(RuntimeError, match="cold-process measurement failed"):
        runner.measure_cold_process(Path("/fake"))


def test_golden_coverage_guard(runner, monkeypatch, tmp_path) -> None:
    """A checkpoint unreachable within warmup+updates must error, never
    pass silently (this guard exists because a silent pass happened once)."""
    unreachable = {"checkpoints": {"30": [1.0, 2.0, 3.0, 4.0, 5.0]}}
    monkeypatch.setattr(runner, "init_model", lambda d: (StubModel(), tmp_path))
    monkeypatch.setattr(runner, "verify_init_state", lambda m, d: [])
    with pytest.raises(RuntimeError, match="no golden checkpoint is reachable"):
        runner.run_one_repeat(
            tmp_path, unreachable, updates=10, warmup=2
        )


def test_init_state_mismatch_raises(runner, monkeypatch, tmp_path) -> None:
    """Non-byte-identical init state aborts the repeat (staleness detector)."""
    monkeypatch.setattr(runner, "init_model", lambda d: (StubModel(), tmp_path))
    monkeypatch.setattr(
        runner, "verify_init_state", lambda m, d: ["eta: bytes differ"]
    )
    with pytest.raises(RuntimeError, match="not byte-identical"):
        runner.run_one_repeat(tmp_path, None, updates=1, warmup=0)


def test_runner_reproduces_golden_eta(tmp_path: Path) -> None:
    """Integration: the real runner on the committed small fixture, one warm
    repeat, no cold measurement -- golden eta must pass at both checkpoints."""
    out_dir = tmp_path / "results"
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH / "run_reference.py"),
            "--fixture",
            str(FIXTURE_DIR),
            "--name",
            "fixture_default_small",
            "--verify-golden",
            "--repeat",
            "1",
            "--skip-cold",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    results_path = next(out_dir.glob("pydeltarcm_reference_*.json"))
    with open(results_path) as f:
        results = json.load(f)
    assert results["parity"]["init_state_bytes_match"] is True
    golden = results["parity"]["golden_eta"]
    assert set(golden) == {"1", "10"}
    assert all(check["pass"] for check in golden.values())
