"""Distributional acceptance for independent native and reference RNG streams.

Three seed families, five independent runs per family and backend. Paired
seeds identify comparable experiments, not identical parcel trajectories.
Margins are fixed physical relative-effect limits, not fitted to p-values:
15% eta relief, 15% deposition volume, 10% channel-mouth flux. The entire
90% paired-difference confidence interval must fit inside each margin
(two one-sided equivalence tests at alpha=0.05).
"""

import json

import numpy as np
import pytest
from scipy.stats import t

from pyDeltaRCM import DeltaModel, native_water

SEED_FAMILIES = (0, 1, 42)
RUNS_PER_SEED = 5
UPDATES = 8
MARGINS = np.array([0.15, 0.15, 0.10])
METRICS = ("eta_relief_m", "deposition_m3_per_update", "mouth_flux_m3_per_s")


def simulate(path, seed, native, monkeypatch):
    monkeypatch.setenv("PYDELTARCM_NATIVE", "1" if native else "0")
    model = DeltaModel(
        out_dir=str(path),
        Length=2000.0,
        Width=2000.0,
        dx=50.0,
        Np_water=200,
        Np_sed=200,
        itermax=3,
        seed=int(seed),
        save_dt=10**12,
    )
    assert native_water.can_use_native(model) == native
    deposition = []
    flux = []
    try:
        for _ in range(UPDATES):
            previous = model.eta.copy()
            model.update()
            assert native_water.can_use_native(model) == native
            deposition.append(
                float(np.maximum(model.eta - previous, 0).sum()) * model.dx**2
            )
            # Signed discharge integrated over the first cross-section beyond
            # the inlet, excluding impermeable banks. Not the prescribed row-0
            # boundary flux (which would make this check tautological).
            mouth = model.qx[model.L0, model.cell_type[model.L0] != -2]
            flux.append(float(mouth.sum()) * model.dx)
        relief = float(np.ptp(model.eta[model.cell_type != -2]))
        metrics = np.array([relief, np.mean(deposition), np.mean(flux)])
        fields = {
            name: getattr(model, name).copy()
            for name in ("eta", "stage", "qx", "qy", "depth")
        }
        assert np.isfinite(metrics).all()
        return metrics, fields
    finally:
        model.finalize()


def test_native_reference_statistical_equivalence(tmp_path, monkeypatch):
    results = []
    seeds = []
    for family in SEED_FAMILIES:
        for replicate in range(RUNS_PER_SEED):
            seed = int(np.random.SeedSequence([family, replicate]).generate_state(1)[0])
            seeds.append(seed)
            pair = []
            for backend in (False, True):
                metrics, _ = simulate(
                    tmp_path / f"{family}-{replicate}-{backend}",
                    seed,
                    backend,
                    monkeypatch,
                )
                pair.append(metrics)
            results.append(pair)
    results = np.asarray(results)
    reference, native = results[:, 0], results[:, 1]
    difference = native - reference
    mean_difference = difference.mean(axis=0)
    half_width = (
        t.ppf(0.95, len(results) - 1)
        * difference.std(axis=0, ddof=1)
        / np.sqrt(len(results))
    )
    limits = MARGINS * np.abs(reference.mean(axis=0))
    evidence = {
        "seeds": seeds,
        "metrics": METRICS,
        "relative_margins": MARGINS.tolist(),
        "reference": reference.tolist(),
        "native": native.tolist(),
        "mean_difference": mean_difference.tolist(),
        "ci90_half_width": half_width.tolist(),
        "absolute_limits": limits.tolist(),
    }
    print(json.dumps(evidence, sort_keys=True))
    assert np.all(np.abs(mean_difference) + half_width < limits), evidence


@pytest.mark.parametrize("seed", SEED_FAMILIES)
def test_native_model_repeats_bit_for_bit(tmp_path, monkeypatch, seed):
    first, first_fields = simulate(tmp_path / "first", seed, True, monkeypatch)
    second, second_fields = simulate(tmp_path / "second", seed, True, monkeypatch)
    np.testing.assert_array_equal(first, second)
    for name in first_fields:
        np.testing.assert_array_equal(first_fields[name], second_fields[name])
