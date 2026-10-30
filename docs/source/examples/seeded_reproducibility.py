"""Verify seeded bit-reproducibility of pyDeltaRCM simulations.

A pyDeltaRCM simulation driven by a fixed ``seed`` is bit-reproducible:
fresh runs of the same configuration produce model states that are
identical, byte for byte. This guarantee is pinned by the integration
test suite (``tests/integration/test_consistent_outputs.py``), but it
previously had no runnable demonstration in the documentation.

This script runs one configuration (``seeded_reproducibility.yaml``,
which sets ``seed: 14``) in four *fresh subprocesses*: three identical
runs (A, B, and C), and a control run that overrides the seed with
``15`` to demonstrate that changing the seed changes the trajectory.

After every model update (the save interval for this example is a
single update), the script computes a SHA-256 digest over the raw bytes
of the state arrays ``eta``, ``depth``, ``qx``, and ``qy``. Identical
digests mean bit-identical state arrays.

The runs execute concurrently, each writing model output to its own
temporary directory that is removed when the script exits. Digests are
platform- and version-dependent: they are stable across runs of the
same environment, and the script fails loudly if the three seeded runs
ever disagree.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

N_UPDATES = 10
N_IDENTICAL_RUNS = 3
CONTROL_SEED = 15
ARRAY_NAMES = ("eta", "depth", "qx", "qy")
PREFIX = 16  # digest characters shown; assertions always compare in full


def sha256_state(model):
    """Combined sha256 over the raw bytes of the model state arrays."""
    digest = hashlib.sha256()
    for name in ARRAY_NAMES:
        digest.update(getattr(model, name).tobytes())
    return digest.hexdigest()


def run_worker(args):
    """Run one fresh model; write ``result.json`` into its output dir."""
    import pyDeltaRCM  # deferred: keeps the parent process lightweight

    kwargs = {"out_dir": args.out_dir}
    if args.seed_override is not None:
        kwargs["seed"] = args.seed_override
    model = pyDeltaRCM.DeltaModel(input_file=args.yaml, **kwargs)

    combined = []
    for _ in range(N_UPDATES):
        model.update()
        combined.append(sha256_state(model))

    result = {
        "seed": int(model.seed),
        "combined": combined,
        "per_array": {
            name: hashlib.sha256(
                getattr(model, name).tobytes()
            ).hexdigest()
            for name in ARRAY_NAMES
        },
    }
    with open(Path(args.out_dir) / "result.json", "w") as fp:
        json.dump(result, fp)


def spawn_run(yaml_path, label, seed_override):
    """Start one fresh worker subprocess with its own output dir."""
    out_dir = tempfile.mkdtemp(prefix=f"pyDeltaRCM-repro-{label}-")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--yaml", str(yaml_path),
        "--out-dir", out_dir,
    ]
    if seed_override is not None:
        cmd += ["--seed-override", str(seed_override)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return label, proc, out_dir


def print_table(header_first, labels, rows):
    """Print an ASCII table of digest prefixes.

    ``rows`` are lists of cells; the first cell of each row goes in the
    first column (``header_first``), the rest align with ``labels``.
    """
    width = max([PREFIX] + [len(label) for label in labels])
    header = (
        f"| {header_first:>8} | "
        + " | ".join(f"{label:>{width}}" for label in labels)
        + " |"
    )
    rule = "|" + "-" * (len(header) - 2) + "|"
    print(rule)
    print(header)
    print(rule)
    for row in rows:
        cells = " | ".join(f"{cell:>{width}}" for cell in row[1:])
        print(f"| {row[0]:>8} | {cells} |")
    print(rule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--yaml")
    parser.add_argument("--out-dir")
    parser.add_argument("--seed-override", type=int, default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    here = Path(__file__).resolve().parent
    yaml_path = here / "seeded_reproducibility.yaml"
    runs = [
        ("A", None),
        ("B", None),
        ("C", None),
        ("control", CONTROL_SEED),
    ]

    tmpdirs = []
    try:
        started = [spawn_run(yaml_path, lab, seed) for lab, seed in runs]
        tmpdirs = [out for _, _, out in started]
        results = []
        for label, proc, out in started:
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"worker {label} failed "
                      f"(exit {proc.returncode})", file=sys.stderr)
                print(stderr, file=sys.stderr)
                sys.exit(1)
            with open(Path(out) / "result.json") as fp:
                results.append((label, json.load(fp)))
    finally:
        for out in tmpdirs:
            shutil.rmtree(out, ignore_errors=True)

    labels = [f"{lab} (seed {res['seed']})" for lab, res in results]
    columns = [res["combined"] for _, res in results]

    arrays = ", ".join(ARRAY_NAMES)
    print(f"Four fresh runs of {yaml_path.name} in parallel subprocesses; "
          f"sha256 of the raw bytes of {arrays} computed after every "
          f"model update (the save interval for this example).")
    print("")
    print_table(
        "interval",
        labels,
        [[str(t + 1)] + [d[t][:PREFIX] for d in columns]
         for t in range(N_UPDATES)],
    )
    print("")

    base = columns[0]
    identical = all(
        columns[i] == base for i in range(1, N_IDENTICAL_RUNS)
    )
    diverged_at = next(
        (t + 1 for t in range(N_UPDATES)
         if columns[-1][t] != base[t]),
        None,
    )

    print("per-array digests at the final save interval:")
    print_table(
        "array",
        labels,
        [[name] + [
            res["per_array"][name][:PREFIX] for _, res in results
        ] for name in ARRAY_NAMES],
    )
    print("")

    print(f"identical runs A-C at every save interval: {identical}")
    print(f"control diverged from run A at interval: {diverged_at}")
    print("")

    if not identical:
        print("FAILURE: the seeded runs produced different states; "
              "bit-reproducibility does not hold in this environment.")
        sys.exit(1)
    if diverged_at is None:
        print("FAILURE: the control run did not diverge; the seed "
              "override had no effect.")
        sys.exit(1)


if __name__ == "__main__":
    main()
