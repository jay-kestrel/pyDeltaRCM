#!/usr/bin/env python
"""Python reference runner for the pyDeltaRCM comparison rig.

Runs the Python reference on a normalized fixture directory (see
``normalize_fixture.py``) and writes machine-readable timing + parity
results for the project-level comparison report.

Timing methodology (locked by the spec):

* Warm steady-state only for headline numbers. A warm-up window discards
  the numba JIT compilation and model transients before anything is timed.
* Every update is timed individually with ``time.perf_counter`` in-process,
  around :meth:`pyDeltaRCM.DeltaModel.update` only -- I/O stays outside the
  timed region (the model is initialized with all output writing disabled,
  and fixture loading happens before the clock starts).
* ``--repeat R`` repeats the whole init + warm-up + timed-window cycle in
  the same process (JIT pays once); the headline number is the median of
  the per-repeat medians. Default R = 5.
* The numba JIT cold start is measured exactly once, in a fresh subprocess
  (import + init + first update), and reported as its own line item --
  never inside a warm median. ``--skip-cold`` omits it.

Parity: before timing, the runner verifies its post-init state is
byte-identical to the fixture's committed ``state/*.npy`` arrays, proving
the Python reference consumes the same input bytes the native ports will.
With ``--verify-golden``, the seeded golden eta values (pinned by
``tests/integration/test_consistent_outputs.py``) are checked at their
update checkpoints; a mismatch is a hard failure, never a loosened
tolerance.

Usage::

    python benchmarks/run_reference.py \
        --fixture bench/fixture_default_small --updates 10 \
        --warmup 2 --repeat 1 --verify-golden
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Golden comparison tolerance: matches pytest.approx's default relative
# tolerance used by tests/integration/test_consistent_outputs.py.
GOLDEN_RTOL = 1e-6

# Preset for CI smoke runs on the committed small fixture.
SMOKE_PRESET = {
    "updates": 10,
    "warmup": 2,
    "repeat": 1,
}


def load_fixture_config(fixture_dir: Path) -> dict:
    """Load the fixture's YAML config (out_dir injected, never committed)."""
    import yaml

    with open(Path(fixture_dir) / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if "out_dir" in cfg:
        raise ValueError("fixture config.yaml must not set out_dir")
    return cfg


def init_model(fixture_dir: Path) -> tuple[object, Path]:
    """Init a DeltaModel from the fixture config with a throwaway out_dir."""
    from pyDeltaRCM import DeltaModel

    cfg = load_fixture_config(fixture_dir)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pydeltarcm_reference_"))
    cfg["out_dir"] = str(tmp_dir / "out_dir")
    return DeltaModel(**cfg), tmp_dir


def verify_init_state(model: object, fixture_dir: Path) -> list[str]:
    """Byte-compare the model's post-init arrays against the fixture npy files.

    Returns a list of mismatch descriptions (empty when identical). Reads
    the manifest for the authoritative array table.
    """
    with open(Path(fixture_dir) / "manifest.json") as f:
        manifest = json.load(f)
    mismatches = []
    for name, entry in sorted(manifest["arrays"].items()):
        committed = np.load(Path(fixture_dir) / entry["file"])
        live = getattr(model, name)
        if not (
            isinstance(live, np.ndarray)
            and live.dtype == committed.dtype
            and live.shape == committed.shape
            and np.array_equal(live, committed)
        ):
            mismatches.append(
                f"{name}: dtype/shape/bytes differ from {entry['file']}"
            )
    return mismatches


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
    }


def check_golden(
    model: object, golden: dict, checkpoint: int
) -> tuple[bool, list[float], list[float]]:
    """Compare eta[:5, 4] against one golden checkpoint value."""
    expected = np.array(golden["checkpoints"][str(checkpoint)], dtype=np.float64)
    actual = np.asarray(model.eta[:5, 4], dtype=np.float64)
    passed = bool(np.allclose(actual, expected, rtol=GOLDEN_RTOL, atol=0.0))
    return passed, expected.tolist(), actual.tolist()


def run_one_repeat(
    fixture_dir: Path,
    golden: dict | None,
    *,
    updates: int,
    warmup: int,
) -> tuple[dict, list[float], dict]:
    """One init + warm-up + timed-window cycle; returns timing and parity."""
    model, tmp_dir = init_model(fixture_dir)

    def check_golden_at(absolute_update: int) -> dict | None:
        """Golden checkpoints are absolute update counts from fresh init,
        so warm-up updates count too (checks are read-only and sit between
        per-update timers, never inside one)."""
        if golden is None or str(absolute_update) not in golden["checkpoints"]:
            return None
        passed, expected, actual = check_golden(model, golden, absolute_update)
        return {
            "expected": expected,
            "actual": actual,
            "pass": passed,
        }

    try:
        init_mismatches = verify_init_state(model, fixture_dir)
        if init_mismatches:
            raise RuntimeError(
                "post-init state is not byte-identical to the fixture "
                f"({len(init_mismatches)} arrays differ; first: "
                f"{init_mismatches[0]}) -- regenerate the fixture with "
                "normalize_fixture.py"
            )

        golden_checks = {}
        absolute = 0
        for _ in range(warmup):
            model.update()
            absolute += 1
            check = check_golden_at(absolute)
            if check is not None:
                golden_checks[str(absolute)] = check

        times = []
        for _ in range(updates):
            t0 = time.perf_counter()
            model.update()
            times.append(time.perf_counter() - t0)
            absolute += 1
            check = check_golden_at(absolute)
            if check is not None:
                golden_checks[str(absolute)] = check
        if golden is not None and not golden_checks:
            raise RuntimeError(
                "--verify-golden set but no golden checkpoint is reachable "
                f"within warmup {warmup} + updates {updates} (checkpoints: "
                f"{sorted(golden['checkpoints'])}) -- refusing to pass "
                "without verifying"
            )
        model.finalize()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    timing = summarize_timing(times)
    timing["per_update_times_s"] = times
    parity = {"init_state_bytes_match": True, "golden_eta": golden_checks or None}
    return timing, times, parity


def git_commit() -> str | None:
    """Best-effort git SHA of the checkout the package is imported from."""
    try:
        import pyDeltaRCM

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


def collect_metadata(fixture_dir: Path, args: argparse.Namespace) -> dict:
    """Versions, platform, and fixture provenance for results.json."""
    import numba
    import scipy

    import pyDeltaRCM

    with open(Path(fixture_dir) / "manifest.json") as f:
        manifest = json.load(f)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pyDeltaRCM_version": pyDeltaRCM.__version__,
        "pyDeltaRCM_git_commit": git_commit(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "scipy_version": scipy.__version__,
        "fixture": args.fixture,
        "fixture_source_config_sha256": manifest["source_config_sha256"],
        "run_settings": {
            "updates": args.updates,
            "warmup_updates": args.warmup,
            "repeats": args.repeat,
            "verify_golden": args.verify_golden,
            "cold_start_measured": not args.skip_cold,
            "smoke": args.smoke,
        },
    }


def measure_cold_process(fixture_dir: Path) -> dict:
    """Measure JIT cold start in ONE fresh subprocess.

    The child process measures import, model init, and the first update
    (JIT compilation included), then prints a JSON object. This parent
    records the numbers verbatim as a separate line item.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fixture",
        str(fixture_dir),
        "--cold-process-child",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # The child pays the full JIT compile; give it generous headroom.
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cold-process measurement failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )
    # The child prints exactly one JSON line; earlier lines may carry
    # library warnings.
    json_lines = [
        line for line in proc.stdout.splitlines() if line.startswith("{")
    ]
    if not json_lines:
        raise RuntimeError(
            f"cold-process child printed no JSON result:\n{proc.stdout[-2000:]}"
        )
    result = json.loads(json_lines[-1])
    result["measurement"] = (
        "single fresh subprocess: import + DeltaModel init + first update"
    )
    return result


def cold_process_child(fixture_dir: Path) -> dict:
    """Child mode: time import, init, and first update in THIS fresh process."""
    t0 = time.perf_counter()
    import pyDeltaRCM  # noqa: F401 -- the timed import

    t_import = time.perf_counter() - t0

    t1 = time.perf_counter()
    model, tmp_dir = init_model(fixture_dir)
    t_init = time.perf_counter() - t1

    t2 = time.perf_counter()
    model.update()
    t_first_update = time.perf_counter() - t2

    model.finalize()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {
        "import_s": t_import,
        "init_s": t_init,
        "first_update_s": t_first_update,
        "total_s": time.perf_counter() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    fixture_dir = Path(args.fixture)
    if not (fixture_dir / "manifest.json").is_file():
        print(f"error: {fixture_dir} is not a normalized fixture", file=sys.stderr)
        return 2

    if args.cold_process_child:
        result = cold_process_child(fixture_dir)
        print(json.dumps(result))
        return 0

    golden = None
    if args.verify_golden:
        golden_path = fixture_dir / "golden.json"
        if not golden_path.is_file():
            print(
                f"error: --verify-golden requires {golden_path}", file=sys.stderr
            )
            return 2
        with open(golden_path) as f:
            golden = json.load(f)

    results = {
        "metadata": collect_metadata(fixture_dir, args),
        "warm": {
            "updates_per_repeat": args.updates,
            "warmup_updates": args.warmup,
            "repeats": args.repeat,
            "repeat_timings": [],
            "all_per_update_times_s": [],
        },
        "parity": {"init_state_bytes_match": None, "golden_eta": None},
        "jit_cold_start_s": None,
    }

    repeat_medians = []
    for r in range(args.repeat):
        timing, times, parity = run_one_repeat(
            fixture_dir,
            golden,
            updates=args.updates,
            warmup=args.warmup,
        )
        results["warm"]["repeat_timings"].append(timing)
        results["warm"]["all_per_update_times_s"].append(times)
        results["parity"]["init_state_bytes_match"] = parity[
            "init_state_bytes_match"
        ]
        if parity["golden_eta"] is not None:
            results["parity"]["golden_eta"] = parity["golden_eta"]
        repeat_medians.append(timing["median_s"])
        print(
            f"[repeat {r + 1}/{args.repeat}] median per-update "
            f"{timing['median_s']:.6f} s",
            file=sys.stderr,
            flush=True,
        )

    results["warm"]["median_of_repeat_medians_s"] = statistics.median(
        repeat_medians
    )

    if not args.skip_cold:
        results["jit_cold_start_s"] = measure_cold_process(fixture_dir)

    golden_failed = [
        ck
        for ck, check in (results["parity"]["golden_eta"] or {}).items()
        if not check["pass"]
    ]
    if golden_failed:
        print(
            f"error: golden eta mismatch at checkpoint(s) {golden_failed}",
            file=sys.stderr,
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pydeltarcm_reference_{args.name}_{_stamp()}.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"results written (parity FAILED): {out_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pydeltarcm_reference_{args.name}_{_stamp()}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    warm_median = results["warm"]["median_of_repeat_medians_s"]
    print(f"warm median per-update: {warm_median:.6f} s")
    if results["jit_cold_start_s"] is not None:
        cold = results["jit_cold_start_s"]
        print(
            f"jit cold start (separate line item): "
            f"import {cold['import_s']:.3f} s + init {cold['init_s']:.3f} s "
            f"+ first update {cold['first_update_s']:.3f} s "
            f"= {cold['total_s']:.3f} s"
        )
    print(f"results: {out_path}")
    return 0


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fixture", required=True, help="normalized fixture directory"
    )
    parser.add_argument("--name", default="run", help="fixture name for output files")
    parser.add_argument(
        "--updates", type=int, default=10, help="timed updates per repeat"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="untimed warm-up updates (discards numba JIT compilation)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="repeats of init + warm-up + timed window (headline: median of "
        "per-repeat medians)",
    )
    parser.add_argument(
        "--verify-golden",
        action="store_true",
        help="check golden eta checkpoints from the fixture's golden.json",
    )
    parser.add_argument(
        "--skip-cold",
        action="store_true",
        help="skip the fresh-process JIT cold-start measurement",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CI preset: small warm timing budget, one repeat",
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/results", help="results directory"
    )
    parser.add_argument(
        "--cold-process-child",
        dest="cold_process_child",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: spawned by --cold measurement
    )
    args = parser.parse_args(argv)
    if args.smoke:
        for key, value in SMOKE_PRESET.items():
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    return args


if __name__ == "__main__":
    sys.exit(main())
