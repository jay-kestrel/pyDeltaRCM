"""Sediment acceptance: P2 oracle, numba-free fallback, native replay."""

import json
from pathlib import Path

import numpy as np
from pyDeltaRCM import _native

from benchmarks.sediment_equivalence import (
    CONFIG,
    METRICS,
    SEEDS,
    UPDATES,
    WARMUP,
    equivalence,
    run_ensemble,
    simulate,
)


def observe_native_calls(monkeypatch):
    calls = {"water_phase": 0, "sediment_phase": 0}

    def instrument(name, original):
        def call(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return call

    for name in calls:
        monkeypatch.setattr(
            _native, name, instrument(name, getattr(_native, name))
        )
    return calls


def test_sediment_statistical_equivalence(tmp_path, monkeypatch):
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/baselines/p2_sediment_ensemble.json"
    )
    baseline = json.loads(baseline_path.read_text())
    assert baseline["revision"] == "9d8aca8aaca59260b0a8280847884284415bf8a5"
    assert baseline["backend"] == "0"
    assert baseline["seeds"] == SEEDS
    assert baseline["config"] == CONFIG
    assert baseline["warmup"] == WARMUP
    assert baseline["updates"] == UPDATES
    assert baseline["metrics"] == list(METRICS)
    monkeypatch.setenv("PYDELTARCM_COBOL", "0")
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    calls = observe_native_calls(monkeypatch)
    native = run_ensemble()
    assert (
        calls["water_phase"]
        >= len(SEEDS) * (WARMUP + UPDATES) * CONFIG["itermax"]
    )
    assert calls["sediment_phase"] == len(SEEDS) * (WARMUP + UPDATES) * 2
    native_calls = calls.copy()
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    reference = run_ensemble()
    assert calls == native_calls, "reference ensemble must not invoke Rust"
    report = {
        "native": native,
        "reference": reference,
        "p2_reference": baseline,
        "vs_p2": equivalence(
            np.array(baseline["values"]), np.array(native["values"])
        ),
        "vs_python": equivalence(
            np.array(reference["values"]), np.array(native["values"])
        ),
    }
    (tmp_path / "sediment-equivalence.json").write_text(
        json.dumps(report, indent=2)
    )
    assert all(report["vs_p2"]["passes"]), report["vs_p2"]
    assert all(report["vs_python"]["passes"]), report["vs_python"]


def test_native_sediment_ensemble_replays_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("PYDELTARCM_COBOL", "0")
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1")
    first_metrics, first_bed = simulate(SEEDS[0], tmp_path / "first")
    second_metrics, second_bed = simulate(SEEDS[0], tmp_path / "second")
    np.testing.assert_array_equal(first_metrics, second_metrics)
    np.testing.assert_array_equal(first_bed, second_bed)
