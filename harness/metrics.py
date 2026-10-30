"""Metrics record construction and output writers for the scenario harness.

Every function here is a pure computation over plain data (or over the
model's public read-only attributes), so each recorded number can be tested
in isolation from a live simulation. Writers serialize records to JSON,
CSV, or Markdown. Nothing in this module imports :mod:`pyDeltaRCM`.

The record schema (field names and group structure) mirrors the spec's
sample record exactly: ``scenario`` / ``seed`` / ``pyDeltaRCM_version`` /
``host`` / ``speed`` / ``efficiency`` / ``workload`` / ``outputs`` /
``eta_checksum``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
from pathlib import Path
from typing import Protocol

import numpy as np


class DeltaModelView(Protocol):
    """Read-only attribute surface of ``DeltaModel`` that the harness measures.

    The real ``DeltaModel`` and the test stubs satisfy this structurally;
    properties (``time``, ``time_iter``) count as attributes here.
    """

    L: int
    W: int
    Qw0: float
    Qs0: float
    dVs: float
    dt: float
    Np_water: int
    Np_sed: int
    dry_depth: float
    save_iter: int
    seed: int | None
    time: float
    time_iter: int
    eta: np.ndarray
    eta0: np.ndarray
    depth: np.ndarray
    Vp_dep_sand: np.ndarray
    Vp_dep_mud: np.ndarray

    def update(self) -> None: ...

    def finalize(self) -> None: ...


def host_info() -> dict[str, str | int | None]:
    """Static host description embedded in every record."""
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
    }


def eta_checksum(eta: np.ndarray) -> float:
    """Float64 sum of the bed elevations — determinism/distinctness fingerprint.

    Same-shape float32 beds differ in this sum if and only if they differ in
    values (within float32), and the float64 accumulation is exact enough to
    compare byte-identically across same-seed runs.
    """
    return float(np.sum(eta, dtype=np.float64))


def compute_speed(
    step_ms: list[float], wall_init_s: float, wall_total_s: float
) -> dict:
    """Speed metrics from the per-step wall-time series and phase timers.

    ``wall_run_s`` is the sum of the per-step series, keeping the group
    internally consistent. The first step of a fresh process includes numba
    JIT compilation, so the median/p90 exclude it and summarize steady
    state; a single-step run falls back to the whole series so the fields
    stay finite.
    """
    series = np.asarray(step_ms, dtype=np.float64)
    if series.size == 0:
        raise ValueError("step_ms must contain at least one measurement")
    steady = series[1:] if series.size > 1 else series
    return {
        "wall_init_s": float(wall_init_s),
        "wall_first_step_s": float(series[0]),
        "wall_run_s": float(series.sum() / 1000.0),
        "wall_total_s": float(wall_total_s),
        "step_ms_median": float(np.median(steady)),
        "step_ms_p90": float(np.percentile(steady, 90)),
        "step_ms_series": [float(x) for x in series],
    }


def compute_efficiency(
    time_iter: int, cells: int, wall_run_s: float, model_time_end_s: float
) -> dict:
    """Throughput metrics normalizing the processed work by run wall time."""
    if wall_run_s <= 0:
        raise ValueError(f"wall_run_s must be positive, got {wall_run_s}")
    if time_iter <= 0:
        raise ValueError(f"time_iter must be positive, got {time_iter}")
    return {
        "steps_per_s": float(time_iter / wall_run_s),
        "cell_steps_per_s": float((cells * time_iter) / wall_run_s),
        "model_s_per_wall_s": float(model_time_end_s / wall_run_s),
    }


def compute_workload(
    model: DeltaModelView,
    cells_changed_total: int,
    dep_vol_sand_m3: float,
    dep_vol_mud_m3: float,
    model_time_end_s: float,
) -> dict:
    """Workload metrics: what the run actually processed.

    Water and sediment volumes are the constant inlet rates integrated over
    the model's own end time. Because ``dt = dVs / Qs0``, the sediment
    volume also equals ``dVs * time_iter`` — a cross-check asserted in the
    tests rather than a second recorded field.
    """
    return {
        "model_time_end_s": float(model_time_end_s),
        "time_iter": int(model.time_iter),
        "dt_s": float(model.dt),
        "grid_L": int(model.L),
        "grid_W": int(model.W),
        "cells": int(model.L * model.W),
        "parcels_total": int((model.Np_water + model.Np_sed) * model.time_iter),
        "Qw0_m3_per_s": float(model.Qw0),
        "Qs0_m3_per_s": float(model.Qs0),
        "water_processed_m3": float(model.Qw0 * model_time_end_s),
        "sediment_processed_m3": float(model.Qs0 * model_time_end_s),
        "cells_changed_total": int(cells_changed_total),
        "dep_vol_sand_m3": float(dep_vol_sand_m3),
        "dep_vol_mud_m3": float(dep_vol_mud_m3),
        "wet_cells_final": int(np.count_nonzero(model.depth > model.dry_depth)),
    }


def compute_outputs(out_dir: Path, save_iter: int) -> dict:
    """Byte accounting of everything the model wrote to its output directory."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"model output directory missing: {out_dir}")
    files: list[str] = []
    total_bytes = 0
    for root, _dirs, names in os.walk(out_dir):
        for name in names:
            file_path = Path(root) / name
            files.append(file_path.relative_to(out_dir).as_posix())
            total_bytes += file_path.stat().st_size
    return {
        "save_iter": int(save_iter),
        "out_dir_bytes": total_bytes,
        "files": sorted(files),
    }


def build_record(
    scenario: str,
    seed: int | None,
    version: str,
    host: dict,
    speed: dict,
    efficiency: dict,
    workload: dict,
    outputs: dict,
    eta_checksum: float,
) -> dict:
    """Assemble the full metrics record with the spec's field order."""
    return {
        "scenario": scenario,
        "seed": seed,
        "pyDeltaRCM_version": version,
        "host": host,
        "speed": speed,
        "efficiency": efficiency,
        "workload": workload,
        "outputs": outputs,
        "eta_checksum": eta_checksum,
    }


_SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario",
    "seed",
    "pyDeltaRCM_version",
    "model_time_end_s",
    "time_iter",
    "dt_s",
    "grid_L",
    "grid_W",
    "cells",
    "parcels_total",
    "wall_init_s",
    "wall_first_step_s",
    "wall_run_s",
    "wall_total_s",
    "step_ms_median",
    "step_ms_p90",
    "steps_per_s",
    "cell_steps_per_s",
    "model_s_per_wall_s",
    "Qw0_m3_per_s",
    "Qs0_m3_per_s",
    "water_processed_m3",
    "sediment_processed_m3",
    "cells_changed_total",
    "dep_vol_sand_m3",
    "dep_vol_mud_m3",
    "wet_cells_final",
    "save_iter",
    "out_dir_bytes",
    "eta_checksum",
)

# headline columns only — the per-step series and byte lists stay in metrics.json
_MARKDOWN_COLUMNS: tuple[str, ...] = (
    "scenario",
    "seed",
    "time_iter",
    "model_time_end_s",
    "wall_total_s",
    "step_ms_median",
    "steps_per_s",
    "cell_steps_per_s",
    "model_s_per_wall_s",
    "water_processed_m3",
    "sediment_processed_m3",
    "eta_checksum",
)

_GROUPS = ("speed", "efficiency", "workload", "outputs")


def _summary_value(record: dict, column: str) -> object:
    """Fetch a summary column from its group, or from the record's top level."""
    for group in _GROUPS:
        if column in record[group]:
            return record[group][column]
    return record[column]


def _format_cell(value: object) -> str:
    # repr keeps full float precision so eta_checksum stays comparable in the table
    return repr(value) if isinstance(value, float) else str(value)


def summary_csv(records: list[dict]) -> str:
    """Aggregate records as CSV with one row per run (all summary columns)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_SUMMARY_COLUMNS)
    for record in records:
        writer.writerow(
            [_summary_value(record, column) for column in _SUMMARY_COLUMNS]
        )
    return buffer.getvalue()


def summary_markdown(records: list[dict]) -> str:
    """Aggregate records as a Markdown table of headline columns."""
    header = "| " + " | ".join(_MARKDOWN_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(_MARKDOWN_COLUMNS)) + " |"
    rows = ["# pyDeltaRCM scenario run summary", "", header, separator]
    for record in records:
        cells = [
            _format_cell(_summary_value(record, column))
            for column in _MARKDOWN_COLUMNS
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def metrics_json(record: dict) -> str:
    """Serialize a record to pretty-printed JSON (full float precision)."""
    return json.dumps(record, indent=2)


def write_metrics_json(record: dict, path: Path) -> Path:
    """Write the record's JSON to *path*; returns the path for chaining."""
    path = Path(path)
    path.write_text(metrics_json(record))
    return path
