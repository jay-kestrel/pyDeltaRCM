"""Scenario runner: drive pyDeltaRCM runs and record per-run metrics.

Usage::

    python -m harness.run_scenario scenarios/00_smoke.yml --out runs/
    python -m harness.run_scenario scenarios/*.yml --out runs/ --summary

Each run writes ``<out>/<scenario-stem>/metrics.json``. ``--keep-outputs``
additionally preserves the model's own output directory (log, netCDF,
figures) as ``<out>/<scenario-stem>/out/``. ``--summary`` writes aggregate
``summary.csv`` / ``summary.md`` tables under ``<out>/``.

Design decisions locked in the spec (art_kOBYprXR):

- The harness owns the update loop so per-step wall time is observable; the
  Preprocessor is not used for timed runs. Scenarios therefore declare
  their duration as ``timesteps``; ``time``/``time_years`` are rejected
  with a clear error instead of re-implementing Preprocessor duration
  scaling.
- Every run goes through a temporary copy of the scenario YAML with a
  per-run temporary ``out_dir`` injected: passing ``out_dir`` as a kwarg is
  silently ignored for directory creation
  (``pyDeltaRCM/init_tools.py:76-99``), so the YAML path is the only
  reliable route.
- Speed separates cost centers — init, first step (numba JIT), and the
  steady-state median are reported individually.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from pyDeltaRCM import DeltaModel

from harness import metrics
from harness.metrics import DeltaModelView

_DURATION_KEYS = ("timesteps", "time", "time_years")


def duration_steps(config: dict) -> int:
    """Number of ``update()`` calls the harness makes for this scenario."""
    present = [key for key in _DURATION_KEYS if key in config]
    if present != ["timesteps"]:
        raise ValueError(
            "Scenario must declare its duration as 'timesteps' "
            f"(found duration keys: {present if present else 'none'}). The "
            "harness drives the update loop itself and does not interpret "
            "'time'/'time_years' the way the Preprocessor does."
        )
    steps = int(config["timesteps"])
    if steps < 1:
        raise ValueError(f"'timesteps' must be >= 1, got {steps}")
    return steps


def write_injected_yaml(config: dict, out_dir: Path, path: Path) -> Path:
    """Write *config* to *path* with a per-run ``out_dir`` injected."""
    injected = dict(config)
    injected["out_dir"] = str(out_dir)
    path.write_text(yaml.safe_dump(injected, sort_keys=False))
    return path


def _run_steps(
    model: DeltaModelView, steps: int
) -> tuple[list[float], int, float, float]:
    """Drive *steps* updates, timing each and accumulating per-step counters.

    Returns the per-step milliseconds, total cells changed, and deposited
    sand/mud volumes (m³). ``eta0`` is refreshed to the pre-step bed at the
    top of each solve, so ``eta - eta0`` after an update is exactly that
    step's morphological change; ``Vp_dep_*`` are per-cell m³ for the just
    completed step (the model zeroes them when each routing pass starts).
    """
    step_ms: list[float] = []
    cells_changed_total = 0
    dep_sand_m3 = 0.0
    dep_mud_m3 = 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        model.update()
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        cells_changed_total += int(np.count_nonzero(model.eta - model.eta0))
        dep_sand_m3 += float(np.sum(model.Vp_dep_sand))
        dep_mud_m3 += float(np.sum(model.Vp_dep_mud))
    return step_ms, cells_changed_total, dep_sand_m3, dep_mud_m3


def run_scenario_file(
    scenario_path: Path,
    out_root: Path,
    keep_outputs: bool = False,
    model_cls: type | None = None,
) -> dict:
    """Run one scenario YAML end to end and return its metrics record.

    Writes ``<out_root>/<scenario-stem>/metrics.json``; with
    *keep_outputs*, copies the model output directory to
    ``<out_root>/<scenario-stem>/out/``. *model_cls* defaults to
    ``DeltaModel`` and is injectable so tests can substitute stub models.
    """
    model_cls = DeltaModel if model_cls is None else model_cls
    scenario_path = Path(scenario_path)
    out_root = Path(out_root)
    config = yaml.safe_load(scenario_path.read_text())
    steps = duration_steps(config)

    run_dir = out_root / scenario_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pydelta_harness_") as tmp:
        workspace = Path(tmp)
        out_dir = workspace / "out"
        injected_yaml = write_injected_yaml(
            config, out_dir, workspace / "scenario.yml"
        )

        t_start = time.perf_counter()
        model = model_cls(input_file=str(injected_yaml))
        wall_init_s = time.perf_counter() - t_start

        step_ms, cells_changed, dep_sand, dep_mud = _run_steps(model, steps)
        model.finalize()
        # covers init + all steps + finalize, so the phase numbers sum under it
        wall_total_s = time.perf_counter() - t_start

        outputs = metrics.compute_outputs(out_dir, save_iter=model.save_iter)
        checksum = metrics.eta_checksum(model.eta)
        model_time_end_s = float(model.time)

        if keep_outputs:
            shutil.copytree(out_dir, run_dir / "out", dirs_exist_ok=True)

    speed = metrics.compute_speed(step_ms, wall_init_s, wall_total_s)
    efficiency = metrics.compute_efficiency(
        time_iter=int(model.time_iter),
        cells=int(model.L * model.W),
        wall_run_s=speed["wall_run_s"],
        model_time_end_s=model_time_end_s,
    )
    workload = metrics.compute_workload(
        model,
        cells_changed_total=cells_changed,
        dep_vol_sand_m3=dep_sand,
        dep_vol_mud_m3=dep_mud,
        model_time_end_s=model_time_end_s,
    )
    record = metrics.build_record(
        scenario=scenario_path.stem,
        seed=model.seed,
        version=str(model.__pyDeltaRCM_version__),
        host=metrics.host_info(),
        speed=speed,
        efficiency=efficiency,
        workload=workload,
        outputs=outputs,
        eta_checksum=checksum,
    )
    metrics.write_metrics_json(record, run_dir / "metrics.json")
    return record


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.run_scenario",
        description="Run pyDeltaRCM scenario YAMLs and record per-run metrics.",
    )
    parser.add_argument(
        "scenarios",
        nargs="+",
        type=Path,
        help="scenario YAML file(s) to run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="root directory for run records (default: runs)",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="preserve each run's model output directory as <out>/<scenario>/out/",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="write aggregate summary.csv and summary.md under --out",
    )
    args = parser.parse_args(argv)

    records = []
    for scenario_path in args.scenarios:
        record = run_scenario_file(
            scenario_path, args.out, keep_outputs=args.keep_outputs
        )
        records.append(record)
        wall_total_s = record["speed"]["wall_total_s"]
        print(
            f"{record['scenario']}: wall total {wall_total_s:.1f} s -> "
            f"{args.out / record['scenario'] / 'metrics.json'}"
        )

    if args.summary:
        csv_path = args.out / "summary.csv"
        md_path = args.out / "summary.md"
        csv_path.write_text(metrics.summary_csv(records))
        md_path.write_text(metrics.summary_markdown(records))
        print(f"wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
