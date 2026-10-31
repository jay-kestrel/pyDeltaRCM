"""Demonstrate checkpoint/resume continuity.

A model run that is interrupted and then resumed from its checkpoint in a
**fresh interpreter** reproduces, bit for bit, the state trajectory of an
uninterrupted run. This works because the checkpoint file carries the state
of the numba random number generator alongside the model grids and clocks,
so the resumed run continues the random stream exactly where the interrupted
run stopped.

The example executes three model runs:

1. a **reference** run, advanced for ``TOTAL_UPDATES`` updates without
   interruption, hashing its state arrays after every update,
2. an **interrupted** run, advanced for ``STOP_AT_UPDATES`` updates with
   ``save_checkpoint: true``, leaving a checkpoint file behind mid-run, and
3. a **resumed** run, started in a fresh interpreter from that checkpoint
   and advanced until it reaches the reference's total update count.

For every update index, a sha256 hash over the raw bytes of the ``eta``,
``depth``, ``qx`` and ``qy`` arrays is compared between the reference and
the interrupted/resumed trajectory. The example asserts the hashes are
identical at every compared update and prints the comparison table.

The script is its own driver: running it without arguments spawns the
individual phases (as fresh interpreter processes) and prints the table.

Run it standalone with::

    python examples/checkpoint_resume_continuity.py

"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.abspath(os.path.dirname(__file__))
BASE_YAML = os.path.join(HERE, "checkpoint_resume_continuity.yaml")

TOTAL_UPDATES = 8  # updates executed by the uninterrupted reference run
STOP_AT_UPDATES = 4  # updates executed before the interrupted run stops
HASHED_FIELDS = ("eta", "depth", "qx", "qy")

PHASE_RUN = "run"  # reference + interrupted runs (one interpreter)
PHASE_RESUME = "resume"  # resumed run (fresh interpreter)


def field_hashes(model):
    """Per-field sha256 hex digest of the raw bytes of each state array."""
    return {
        field: hashlib.sha256(
            np.ascontiguousarray(getattr(model, field)).tobytes()
        ).hexdigest()
        for field in HASHED_FIELDS
    }


def record_state(record, model):
    """Record the model time and state hashes at the current update."""
    record[model._time_iter] = {
        "time": model.time,
        "hashes": field_hashes(model),
    }


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def run_reference_and_stop(config_run, config_ref):
    """Phase 1: reference run and interrupted run, in one interpreter."""
    import pyDeltaRCM

    # reference run: uninterrupted, hash the state after every update
    reference = pyDeltaRCM.DeltaModel(config_ref)
    reference_hashes = {}
    for _ in range(TOTAL_UPDATES):
        reference.update()
        record_state(reference_hashes, reference)
    reference.finalize()

    # interrupted run: identical configuration and seed, stopped mid-run.
    # the checkpoint file it leaves behind is the resume point for phase 2
    interrupted = pyDeltaRCM.DeltaModel(config_run)
    stop_hashes = {}
    for _ in range(STOP_AT_UPDATES):
        interrupted.update()
        record_state(stop_hashes, interrupted)
    interrupted.finalize()

    run_dir = _config_out_dir(config_run)
    ref_dir = _config_out_dir(config_ref)
    _write_json(os.path.join(run_dir, "stop_hashes.json"), stop_hashes)
    _write_json(os.path.join(ref_dir, "reference_hashes.json"), reference_hashes)


def run_resume(config_run):
    """Phase 2: resume the interrupted run in a fresh interpreter."""
    import pyDeltaRCM

    resumed = pyDeltaRCM.DeltaModel(config_run, resume_checkpoint=True)

    # the restored state is the checkpointed state: hash it as the first
    # point of the resumed trajectory, then run up to the reference's
    # total update count
    resume_hashes = {}
    record_state(resume_hashes, resumed)
    for _ in range(TOTAL_UPDATES - resumed._time_iter):
        resumed.update()
        record_state(resume_hashes, resumed)
    resumed.finalize()

    run_dir = _config_out_dir(config_run)
    _write_json(os.path.join(run_dir, "resume_hashes.json"), resume_hashes)


def _config_out_dir(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)["out_dir"]


def _make_run_config(out_dir, config_path):
    """Write a copy of the base YAML with `out_dir` set to `out_dir`.

    `out_dir` has to be set in the YAML itself: the model reads it from the
    configuration file before keyword arguments are merged, so passing it
    as a keyword argument has no effect on where outputs are written.
    """
    with open(BASE_YAML) as f:
        config = yaml.safe_load(f)
    config["out_dir"] = out_dir
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return config_path


def _spawn_phase(phase, config, ref_config=None):
    """Run one phase in a fresh interpreter, mirroring a user's workflow."""
    cmd = [sys.executable, os.path.abspath(__file__), "--phase", phase,
           "--config", config]
    if ref_config is not None:
        cmd += ["--ref-config", ref_config]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:  # surface the failure, never swallow it
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        result.check_returncode()


def compare_and_report(reference, stop, resume):
    """Assert bit-identity of the resumed trajectory and print the table.

    Compares, per field, the sha256 of the raw state-array bytes at every
    update index shared by the reference and the interrupted/resumed
    trajectories.
    """
    # the interrupted run must match the reference up to the stop ...
    for it, entry in stop.items():
        _assert_identical(reference, it, entry, "interrupted")
    # ... and the resumed run must match it from the checkpoint onward
    for it, entry in resume.items():
        _assert_identical(reference, it, entry, "resumed")

    # merged trajectory: interrupted run up to the stop, resumed run after
    merged = dict(stop)
    merged.update(resume)

    widths = (6, 12, 20, 20)
    header = ("| {:>{w0}} | {:>{w1}} | {:^{w2}} | {:^{w3}} |".format(
        "update", "time (s)", "reference sha256", "interrupted/resumed sha256",
        w0=widths[0], w1=widths[1], w2=widths[2], w3=widths[3]))
    print("|" + ((len(header) - 2) * "-") + "|")
    print(header)
    print("|" + ((len(header) - 2) * "-") + "|")
    for it in sorted(reference, key=int):
        it = str(it)
        entry = merged.get(it)
        if entry is None:
            continue  # reference state beyond the compared window
        ref_digest = _combined(reference[it]["hashes"])[:16]
        res_digest = _combined(entry["hashes"])[:16]
        print("| {:>{w0}} | {:>{w1}.0f} | {:>{w2}} | {:>{w3}} |".format(
            it, entry["time"], ref_digest, res_digest,
            w0=widths[0], w1=widths[1], w2=widths[2], w3=widths[3]))
    print("|" + ((len(header) - 2) * "-") + "|")

    print("")
    compared = sorted(int(k) for k in merged)
    print(
        "sha256 over the raw bytes of eta, depth, qx and qy matched the "
        f"uninterrupted reference at all {len(compared)} compared updates "
        f"({compared[0]}-{compared[-1]}): "
        f"the run resumed from its checkpoint is bit-identical to a run "
        f"that never stopped, including the random number generator state."
    )


def _assert_identical(reference, it, entry, label):
    for field, digest in entry["hashes"].items():
        ref_digest = reference[str(it)]["hashes"][field]
        if digest != ref_digest:
            raise SystemExit(
                f"{label} run diverged from the uninterrupted reference at "
                f"update {it}: sha256({field}) {digest} != {ref_digest}"
            )


def _combined(hashes):
    """Display digest combining the per-field hashes of one state."""
    combined = hashlib.sha256()
    for field in HASHED_FIELDS:
        combined.update(hashes[field].encode())
    return combined.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--phase", choices=[PHASE_RUN, PHASE_RESUME], default=None,
        help="run a single phase (used internally by the driver); without "
             "this flag, the script acts as the driver and spawns both "
             "phases in fresh interpreters",
    )
    parser.add_argument("--config", help="YAML configuration for this phase")
    parser.add_argument(
        "--ref-config", help="YAML configuration for the reference run "
        "(phase '%s' only)" % PHASE_RUN,
    )
    args = parser.parse_args()

    if args.phase == PHASE_RUN:
        run_reference_and_stop(args.config, args.ref_config)
    elif args.phase == PHASE_RESUME:
        run_resume(args.config)
    else:
        # driver: set up isolated output directories, run both phases in
        # fresh interpreters, then compare and report
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "interrupted_run")
            ref_dir = os.path.join(tmp, "reference_run")
            config_run = _make_run_config(
                run_dir, os.path.join(tmp, "config_interrupted.yaml"))
            config_ref = _make_run_config(
                ref_dir, os.path.join(tmp, "config_reference.yaml"))

            _spawn_phase(PHASE_RUN, config_run, ref_config=config_ref)
            _spawn_phase(PHASE_RESUME, config_run)

            reference = _read_json(os.path.join(
                ref_dir, "reference_hashes.json"))
            stop = _read_json(os.path.join(run_dir, "stop_hashes.json"))
            resume = _read_json(os.path.join(run_dir, "resume_hashes.json"))

        compare_and_report(reference, stop, resume)


if __name__ == "__main__":
    main()
