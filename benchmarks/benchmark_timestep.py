#!/usr/bin/env python
"""Two-regime per-timestep wall-time benchmark harness for pyDeltaRCM.

Measures the wall time of :meth:`pyDeltaRCM.DeltaModel.update` in the two
flow regimes that matter for performance work:

* ``early_water`` -- a water-dominated run on a small grid, timed shortly
  after initialization, before morphology forms and sediment routing
  becomes expensive.
* ``evolved_sediment`` -- the same configuration, run through a warm-up
  phase until enough morphology has formed (a deposition-coverage
  criterion), then timed for a window in which sediment routing dominates.

Each scenario reports per-update mean, standard deviation, median, min, and
max wall times measured with :func:`time.perf_counter`. Warm-up updates are
discarded from the timing so that numba compilation and initial-model
transients are excluded. Package versions, the RNG seed, and the grid
configuration are recorded alongside the timings so baselines stay
reproducible.

Output is written as JSON (full data, including every per-update time) and
Markdown (summary tables); the Markdown is also echoed to stdout.

Usage::

    python benchmarks/benchmark_timestep.py                     # both scenarios
    python benchmarks/benchmark_timestep.py --smoke             # tiny grid, for CI
    python benchmarks/benchmark_timestep.py --output-dir DIR    # custom output location

This harness is measurement-only: it does not modify any library behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

import pyDeltaRCM
from pyDeltaRCM import DeltaModel
from pyDeltaRCM.native_water import can_use_native

# Cells count as "deposited" once the bed has aggraded at least this much
# above its initial elevation.
MORPHOLOGY_DEPTH_THRESHOLD_M = 0.05

# Grid-output interval high enough that no benchmark run ever triggers a
# netCDF write; checkpoints are disabled outright.
SAVE_DT_NEVER = 10**12

# Full-size scenario grids: 60x120 cells at dx = 50 m.
FULL_GRID = {"Length": 3000.0, "Width": 6000.0, "dx": 50.0}
# Smoke grid for CI: 40x40 cells, fewer parcels, one free-surface iteration.
SMOKE_GRID = {"Length": 2000.0, "Width": 2000.0, "dx": 50.0}

SCENARIOS = ("early_water", "evolved_sediment")

# Scenario-specific config overrides. The evolved scenario runs four times
# the sediment parcels (water forcing unchanged): measured on this grid, the
# sediment phase stays 21-27% of update time at Np_sed = 1000 no matter how
# long the model evolves (probed through 210 updates), while Np_sed = 4000
# makes sediment routing the dominant phase (53% of phase time) — which is
# the regime this scenario must stress.
SCENARIO_CONFIG_OVERRIDES = {
    "evolved_sediment": {"Np_sed": 4000},
}

# Kernel-level benchmark (the COBOL port target): the batched weighted
# parcel pick. Random numbers are supplied by Python (C5); each backend
# consumes identical pre-drawn uniforms, so rows differ only by the
# arithmetic under test. Timing rows are feasibility measurements, never
# gates (port contract).
KERNEL_PICK_DESCRIPTION = (
    "batched weighted parcel pick (shared_tools.random_pick): float32 "
    "cumulative weights + searchsorted-left scan; rows x 9 weights "
    "(D8 window), one pre-drawn uniform(0, row_total) double per row"
)
KERNEL_PICK_WORKLOAD = {
    "cols": 9,
    "dtype": "float32",
    "seed": 0,
    "uniforms": "pre-drawn float64 uniform(0, row_total) (C5)",
}


def build_model_config(smoke: bool) -> dict:
    """Assemble the base DeltaModel kwargs shared by both scenarios.

    All figure/grid/checkpoint output is disabled so the harness measures
    computation rather than I/O; the model still writes into a throwaway
    temporary directory that is removed afterwards.
    """
    parcels = 50 if smoke else 1000
    return {
        **(SMOKE_GRID if smoke else FULL_GRID),
        "Np_water": parcels,
        "Np_sed": parcels,
        "itermax": 1 if smoke else 3,
        "seed": 0,
        "verbose": 0,
        "save_dt": SAVE_DT_NEVER,
        "save_checkpoint": False,
    }


def deposition_coverage(model: DeltaModel, eta_init: np.ndarray) -> float:
    """Fraction of the domain where the bed has aggraded past the threshold."""
    aggraded = model.eta > (eta_init + MORPHOLOGY_DEPTH_THRESHOLD_M)
    return float(np.count_nonzero(aggraded) / aggraded.size)


def run_fixed_warmup(model: DeltaModel, n_updates: int) -> int:
    """Run a fixed number of warm-up updates."""
    for _ in range(n_updates):
        model.update()
    return n_updates


def run_morphology_warmup(
    model: DeltaModel,
    eta_init: np.ndarray,
    coverage_threshold: float,
    max_updates: int,
) -> tuple[int, float]:
    """Update until deposition coverage reaches ``coverage_threshold``.

    Returns the number of warm-up updates run and the coverage reached.
    Stops early at ``max_updates`` so a poorly tuned threshold cannot make
    the harness run forever; the JSON records whether the threshold was met.
    """
    coverage = deposition_coverage(model, eta_init)
    updates = 0
    while coverage < coverage_threshold and updates < max_updates:
        model.update()
        updates += 1
        coverage = deposition_coverage(model, eta_init)
        if updates % 10 == 0:
            print(
                f"[evolved_sediment] warm-up: {updates} updates, "
                f"deposition coverage {coverage:.3f}",
                file=sys.stderr,
                flush=True,
            )
    return updates, coverage


def time_updates(model: DeltaModel, n_updates: int) -> list[float]:
    """Run ``n_updates`` updates, timing each with ``time.perf_counter``."""
    times = []
    for _ in range(n_updates):
        t0 = time.perf_counter()
        model.update()
        times.append(time.perf_counter() - t0)
    return times


def summarize_timing(times: list[float]) -> dict:
    """Reduce per-update times to summary statistics."""
    return {
        "n_updates": len(times),
        "mean_s": statistics.fmean(times),
        "sd_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "total_s": sum(times),
        "per_update_times_s": times,
    }


def time_pick_backend(rows: np.ndarray, uniforms: np.ndarray, repeats: int) -> list[float]:
    """Time full-batch ``pick_rows`` calls through the real dispatch."""
    from pyDeltaRCM import cobol_tools

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cobol_tools.pick_rows(rows, uniforms=uniforms)
        times.append(time.perf_counter() - t0)
    return times


def summarize_picks(times: list[float], k: int) -> dict:
    """Per-pick statistics from full-batch pick timings."""
    mean_batch = statistics.fmean(times)
    return {
        "picks_per_batch": k,
        "batches": len(times),
        "mean_us_per_pick": 1e6 * mean_batch / k,
        "median_us_per_pick": 1e6 * statistics.median(times) / k,
        "picks_per_s": k / mean_batch,
    }


def run_kernel_pick_bench(smoke: bool) -> dict:
    """Time the ported pick routine per backend with backend-tagged rows.

    Asserts exact pick parity between the backends on the workload before
    any timing counts -- a parity failure aborts the harness (integer
    indices must match; there is no tolerance to document). The COBOL row
    is emitted with ``available: false`` when the shared object was not
    requested and is not built; a requested-but-missing build raises.
    """
    from pyDeltaRCM import cobol_tools

    k = 200 if smoke else 4000
    repeats = 5 if smoke else 40
    rng = np.random.default_rng(KERNEL_PICK_WORKLOAD["seed"])
    rows = np.ascontiguousarray(rng.random((k, 9)), dtype=np.float32)
    cum = np.cumsum(rows, axis=1)
    uniforms = np.array(
        [rng.uniform(0, cum[p, -1]) for p in range(k)], dtype=np.float64
    )

    requested = cobol_tools.cobol_active()
    env_backup = os.environ.get("PYDELTARCM_COBOL")
    try:
        os.environ["PYDELTARCM_COBOL"] = "0"
        py_picks = cobol_tools.pick_rows(rows, uniforms=uniforms)
        py_times = time_pick_backend(rows, uniforms, repeats)

        os.environ["PYDELTARCM_COBOL"] = "1"
        try:
            cobol_picks = cobol_tools.pick_rows(rows, uniforms=uniforms)
        except FileNotFoundError:
            if requested:
                raise
            cobol_row = {
                "available": False,
                "reason": "not built: run `make -C cobol` (requires GnuCOBOL 3.x)",
            }
        else:
            if not np.array_equal(py_picks, cobol_picks):
                raise SystemExit(
                    "COBOL pick parity failure on the benchmark workload: "
                    "indices must match exactly (no tolerance applies)"
                )
            cobol_times = time_pick_backend(rows, uniforms, repeats)
            cobol_row = {
                "available": True,
                "implementation": "cobol",
                "gnucobol_version": cobol_tools.gnucobol_version(),
                "single_threaded": True,
                **summarize_picks(cobol_times, k),
            }
    finally:
        if env_backup is None:
            os.environ.pop("PYDELTARCM_COBOL", None)
        else:
            os.environ["PYDELTARCM_COBOL"] = env_backup

    py_row = {
        "available": True,
        "implementation": "numpy",
        "note": (
            "numpy reference formulation of the pick; the in-loop "
            "shared_tools.random_pick runtime function is numba-compiled"
        ),
        **summarize_picks(py_times, k),
    }
    return {
        "description": KERNEL_PICK_DESCRIPTION,
        "workload": {"rows": k, "repeats": repeats, **KERNEL_PICK_WORKLOAD},
        "backends": {"python": py_row, "cobol": cobol_row},
    }


def run_scenario(
    name: str,
    model_config: dict,
    *,
    timed_updates: int,
    warmup_updates: int,
    morphology_threshold: float,
    max_warmup_updates: int,
) -> dict:
    """Run one scenario end-to-end and return its full result record."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        scenario_config = {
            **model_config,
            **SCENARIO_CONFIG_OVERRIDES.get(name, {}),
        }
        model = DeltaModel(**scenario_config, out_dir=tmp_dir)
        eta_init = np.array(model.eta, copy=True)

        if name == "early_water":
            used = run_fixed_warmup(model, warmup_updates)
            warmup = {
                "type": "fixed",
                "updates": used,
                "coverage_at_window_start": deposition_coverage(model, eta_init),
            }
        elif name == "evolved_sediment":
            used, coverage = run_morphology_warmup(
                model, eta_init, morphology_threshold, max_warmup_updates
            )
            warmup = {
                "type": "morphology_coverage",
                "updates": used,
                "coverage_threshold": morphology_threshold,
                "coverage_at_window_start": coverage,
                "threshold_met": coverage >= morphology_threshold,
                "max_updates": max_warmup_updates,
                "depth_threshold_m": MORPHOLOGY_DEPTH_THRESHOLD_M,
            }
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unknown scenario: {name}")

        times = time_updates(model, timed_updates)
        model.finalize()

    rows, cols = (
        int(round(scenario_config["Length"] / scenario_config["dx"])),
        int(round(scenario_config["Width"] / scenario_config["dx"])),
    )
    return {
        "config": {
            "water_backend": "rust" if can_use_native(model) else "python",
            "Length": scenario_config["Length"],
            "Width": scenario_config["Width"],
            "dx": scenario_config["dx"],
            "grid_shape_rows_cols": [rows, cols],
            "Np_water": scenario_config["Np_water"],
            "Np_sed": scenario_config["Np_sed"],
            "itermax": scenario_config["itermax"],
            "seed": scenario_config["seed"],
            "model_timestep_dt_s": float(model.dt),
        },
        "warmup": warmup,
        "timing": summarize_timing(times),
    }


def git_commit() -> str | None:
    """Best-effort git SHA of the checkout the package is imported from.

    Returns ``None`` (recorded as ``null`` in the JSON) when the package is
    not used from a git checkout.
    """
    try:
        package_dir = Path(pyDeltaRCM.__file__).resolve().parent
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(package_dir.parent),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def collect_metadata() -> dict:
    """Versions and platform info recorded alongside every run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pyDeltaRCM_version": pyDeltaRCM.__version__,
        "pyDeltaRCM_git_commit": git_commit(),
        "numpy_version": np.__version__,
        "numba_version": None,
        "scipy_version": scipy.__version__,
    }


def format_markdown(results: dict) -> str:
    """Render the benchmark results as a Markdown summary."""
    meta = results["metadata"]
    lines = [
        "# pyDeltaRCM timestep benchmark",
        "",
        "## Environment",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| timestamp (UTC) | {meta['timestamp_utc']} |",
        f"| platform | {meta['platform']} |",
        f"| python | {meta['python_version']} |",
        f"| pyDeltaRCM | {meta['pyDeltaRCM_version']} "
        f"(commit {meta['pyDeltaRCM_git_commit'] or 'unknown'}) |",
        f"| numpy | {meta['numpy_version']} |",
        f"| numba | {meta['numba_version']} |",
        f"| scipy | {meta['scipy_version']} |",
        "",
        "## Per-update wall time (warm-up discarded)",
        "",
        "| scenario | warm-up updates | timed updates | mean ± sd (s) | "
        "median (s) | min (s) | max (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in results["scenarios"]:
        scen = results["scenarios"][name]
        timing = scen["timing"]
        warmup = scen["warmup"]
        lines.append(
            f"| {name} | {warmup['updates']} | {timing['n_updates']} "
            f"| {timing['mean_s']:.4f} ± {timing['sd_s']:.4f} "
            f"| {timing['median_s']:.4f} | {timing['min_s']:.4f} "
            f"| {timing['max_s']:.4f} |"
        )
    for name in results["scenarios"]:
        scen = results["scenarios"][name]
        cfg = scen["config"]
        warmup = scen["warmup"]
        if warmup["type"] == "morphology_coverage":
            warmup_detail = (
                f", deposition coverage at window start "
                f"{warmup['coverage_at_window_start']:.4f} "
                f"(threshold {warmup['coverage_threshold']}, "
                f"met: {warmup['threshold_met']})"
            )
        else:
            warmup_detail = (
                f", coverage at window start "
                f"{warmup['coverage_at_window_start']:.4f}"
            )
        lines += [
            "",
            f"## Scenario: {name}",
            "",
            "| setting | value |",
            "| --- | --- |",
            f"| grid | {cfg['grid_shape_rows_cols'][0]} x "
            f"{cfg['grid_shape_rows_cols'][1]} cells "
            f"(Length {cfg['Length']} m x Width {cfg['Width']} m, "
            f"dx {cfg['dx']} m) |",
            f"| parcels | Np_water = {cfg['Np_water']}, Np_sed = {cfg['Np_sed']} |",
            f"| itermax | {cfg['itermax']} |",
            f"| water backend | {cfg['water_backend']} |",
            f"| seed | {cfg['seed']} |",
            f"| model timestep (dt) | {cfg['model_timestep_dt_s']:.1f} s |",
            f"| warm-up | {warmup['type']}, {warmup['updates']} updates"
            f"{warmup_detail} |",
        ]
    kernels = results.get("kernels", {})
    if kernels:
        pick = kernels["random_pick"]
        lines += [
            "",
            "## Kernel: random_pick (batched weighted pick)",
            "",
            pick["description"],
            "",
            "Timing rows are feasibility measurements, not gates. "
            "COBOL kernels are single-threaded; parity is asserted "
            "exactly before timings count.",
            "",
            "| backend | mean µs/pick | median µs/pick | picks/s | implementation |",
            "| --- | --- | --- | --- | --- |",
        ]
        for backend_name, row in pick["backends"].items():
            if not row["available"]:
                lines.append(
                    f"| {backend_name} | not built | | | {row['reason']} |"
                )
            else:
                impl = row["implementation"]
                if "gnucobol_version" in row:
                    impl += f" (GnuCOBOL {row['gnucobol_version']})"
                lines.append(
                    f"| {backend_name} | {row['mean_us_per_pick']:.4f} "
                    f"| {row['median_us_per_pick']:.4f} "
                    f"| {row['picks_per_s']:.0f} | {impl} |"
                )
    return "\n".join(lines) + "\n"


def resolve_defaults(args: argparse.Namespace) -> dict:
    """Resolve preset defaults for knobs left at their CLI sentinel values."""
    if args.smoke:
        defaults = {
            "timed_updates": 3,
            "warmup_updates": 2,
            "morphology_threshold": 0.02,
            "max_warmup_updates": 5,
        }
    else:
        defaults = {
            "timed_updates": 20,
            "warmup_updates": 5,
            "morphology_threshold": 0.05,
            "max_warmup_updates": 200,
        }
    return {
        knob: default if getattr(args, knob) is None else getattr(args, knob)
        for knob, default in defaults.items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenarios",
        choices=("both",) + SCENARIOS,
        default="both",
        help="which scenario(s) to run (default: both)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny-grid preset for CI (40x40 grid, 50 parcels, 3 timed updates)",
    )
    parser.add_argument(
        "--timed-updates",
        type=int,
        default=None,
        help="number of timed updates per scenario (default: 20; smoke: 3)",
    )
    parser.add_argument(
        "--warm-up-updates",
        type=int,
        dest="warmup_updates",
        default=None,
        help="fixed warm-up updates for early_water (default: 5; smoke: 2)",
    )
    parser.add_argument(
        "--morphology-threshold",
        type=float,
        default=None,
        help="deposition coverage to reach in evolved_sediment warm-up "
        "(default: 0.05; smoke: 0.02)",
    )
    parser.add_argument(
        "--max-warmup-updates",
        type=int,
        default=None,
        help="cap on evolved_sediment warm-up updates (default: 200; smoke: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="directory for the JSON and Markdown output files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    knobs = resolve_defaults(args)

    scenario_names = list(SCENARIOS)
    if args.scenarios != "both":
        scenario_names = [args.scenarios]

    results = {
        "metadata": collect_metadata(),
        "run_settings": {"smoke": args.smoke, **knobs},
        "scenarios": {},
    }
    for name in scenario_names:
        model_config = build_model_config(args.smoke)
        print(f"[{name}] initializing model...",
              file=sys.stderr, flush=True)
        results["scenarios"][name] = run_scenario(
            name,
            model_config,
            timed_updates=knobs["timed_updates"],
            warmup_updates=knobs["warmup_updates"],
            morphology_threshold=knobs["morphology_threshold"],
            max_warmup_updates=knobs["max_warmup_updates"],
        )

    print("[random_pick] timing the ported pick kernel per backend...",
          file=sys.stderr, flush=True)
    results["kernels"] = {"random_pick": run_kernel_pick_bench(smoke=args.smoke)}

    markdown = format_markdown(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"pydeltarcm_timestep_benchmark_{stamp}.json"
    markdown_path = output_dir / f"pydeltarcm_timestep_benchmark_{stamp}.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    markdown_path.write_text(markdown)

    print(markdown)
    print(f"JSON output:  {json_path}")
    print(f"Markdown output: {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
