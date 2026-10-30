"""Reproducible sediment ensembles, usable unchanged against P2 or P3.

Run from the desired checkout with its installed Python environment:
python /path/to/sediment_equivalence.py --backend reference --output result.json

The 90% paired-difference interval must fit inside predeclared relative
margins: 15% deposition, 20% erosion, 15% RMS bed evolution. These are
physical effect limits, not fitted confidence thresholds. Seed families
identify comparable experiments, not aligned parcel trajectories.
"""

import argparse
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import t

import pyDeltaRCM
from pyDeltaRCM import DeltaModel

METRICS = (
    "deposition_m3_per_update",
    "erosion_m3_per_update",
    "bed_rms_change_m",
)
MARGINS = np.array([0.15, 0.20, 0.15])
SEEDS = [
    int(np.random.SeedSequence([family, replicate]).generate_state(1)[0])
    for family in (0, 1, 42)
    for replicate in range(5)
]
CONFIG = dict(
    Length=1000.0,
    Width=2000.0,
    dx=50.0,
    Np_water=100,
    Np_sed=400,
    itermax=3,
    save_dt=10**12,
)
WARMUP = 20
UPDATES = 5


def bed_metrics(
    changes: np.ndarray, bed_change: np.ndarray, dx: float
) -> np.ndarray:
    """Compute signed-volume components and the spatial RMS bed change."""
    return np.array(
        [
            np.maximum(changes, 0).sum(axis=(1, 2)).mean() * dx**2,
            -np.minimum(changes, 0).sum(axis=(1, 2)).mean() * dx**2,
            np.sqrt(np.mean(bed_change**2)),
        ]
    )


def equivalence(reference: np.ndarray, native: np.ndarray) -> dict:
    """Paired TOST confidence bounds with fixed relative-effect margins."""
    if reference.shape != native.shape or reference.shape != (15, 3):
        raise ValueError("expected fifteen paired runs and three metrics")
    if not np.isfinite(reference).all() or not np.isfinite(native).all():
        raise ValueError("ensemble metrics must be finite")
    difference = native - reference
    mean = difference.mean(axis=0)
    half_width = t.ppf(0.95, 14) * difference.std(axis=0, ddof=1) / np.sqrt(15)
    limits = MARGINS * np.abs(reference.mean(axis=0))
    return dict(
        metrics=METRICS,
        relative_margins=MARGINS.tolist(),
        mean_difference=mean.tolist(),
        ci90_half_width=half_width.tolist(),
        absolute_limits=limits.tolist(),
        passes=(np.abs(mean) + half_width < limits).tolist(),
    )


def simulate(seed: int, directory: Path) -> tuple[np.ndarray, np.ndarray]:
    model = DeltaModel(out_dir=str(directory), seed=seed, **CONFIG)
    try:
        for _ in range(WARMUP):
            model.update()
        initial = model.eta.astype(np.float64).copy()
        previous = initial.copy()
        changes = []
        for _ in range(UPDATES):
            model.update()
            current = model.eta.astype(np.float64)
            changes.append(current - previous)
            previous = current.copy()
        metrics = bed_metrics(np.asarray(changes), current - initial, model.dx)
        return metrics, current.copy()
    finally:
        model.finalize()


def run_ensemble() -> dict:
    values = []
    with tempfile.TemporaryDirectory() as directory:
        for index, seed in enumerate(SEEDS):
            metrics, _ = simulate(seed, Path(directory) / str(index))
            values.append(metrics.tolist())
    return dict(
        seeds=SEEDS,
        config=CONFIG,
        warmup=WARMUP,
        updates=UPDATES,
        metrics=METRICS,
        values=values,
        python=platform.python_version(),
        numpy=np.__version__,
        backend=os.environ.get("PYDELTARCM_NATIVE", "1"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("native", "reference"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PYDELTARCM_NATIVE"] = "1" if args.backend == "native" else "0"
    result = run_ensemble()
    source = Path(pyDeltaRCM.__file__).resolve()
    result["source"] = str(source)
    result["revision"] = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
