# pyDeltaRCM benchmark harness

Measures per-timestep (`DeltaModel.update`) wall time in the two flow regimes
that matter for performance work:

| scenario | regime | warm-up |
| --- | --- | --- |
| `early_water` | water-dominated: flow routing on the initial flat bed | fixed (default 5 updates) |
| `evolved_sediment` | sediment-dominated: warm-up runs until a deposition-coverage criterion is met, then a timed window where sediment routing is the dominant phase | until coverage threshold (default 5% of cells aggraded ≥ 5 cm), capped at `--max-warmup-updates` |

Both scenarios use the same grid (60×120 cells — `Length: 3000, Width: 6000,
dx: 50`) and shared settings (`itermax = 3`, `seed = 0`), so their per-update
times are directly comparable. They differ in one knob, deliberately: the
evolved scenario runs `Np_sed = 4000` (the early scenario pins
`Np_water = Np_sed = 1000`). Measured on this grid, the sediment phase stays
21–27% of update time at `Np_sed = 1000` no matter how long the model evolves
(probed through 210 updates — flow channelizes and sediment parcel paths
shorten), while `Np_sed = 4000` makes sediment routing the dominant phase
(53% of phase time) with water forcing untouched. The scenario exists to
stress the sediment routers, so that is the configuration recorded in the
benchmark output.

All model output (figures, grids, checkpoints) is disabled; the model writes
only into a throwaway temporary directory.

## Running

```shell
# both scenarios, full size (~3-5 min; numba compiles once per process)
python benchmarks/benchmark_timestep.py

# single scenario
python benchmarks/benchmark_timestep.py --scenarios evolved_sediment

# CI smoke preset: 40x40 grid, 50 parcels, 3 timed updates
python benchmarks/benchmark_timestep.py --smoke
```

Every run writes two timestamped files into `--output-dir`
(default `benchmarks/results/`): one JSON with the full data (including every
per-update time) and one Markdown summary. The Markdown is also echoed to
stdout.

All knobs can be overridden for targeted experiments: `--timed-updates`,
`--warm-up-updates`, `--morphology-threshold`, `--max-warmup-updates`.

## What gets recorded

Each scenario reports per-update mean, standard deviation, median, min, and
max wall time over the timed window (`time.perf_counter`), with warm-up
updates discarded so numba compilation and initial transients never enter the
numbers. Alongside the timings, the output records package versions
(pyDeltaRCM, numpy, numba, scipy), the git commit the package was imported
from, the RNG seed, and the grid configuration.

Port PRs must report before/after numbers from this harness against a
recorded baseline (see the port spec); no speedup claim exists without one.


## Normalized fixtures and the comparison rig

`normalize_fixture.py` and `run_reference.py` implement the shared input
contract for the C and Rust ports (see the port spec, "Interface
contracts"): every implementation — Python reference included — consumes
byte-identical normalized inputs and is gated on the same seeded golden
outputs before its timings count.

### Fixture layout (`bench/fixture_*`)

```
bench/fixture_default_small/     10x10 grid, seed 0, golden-checked
bench/fixture_default/           default production grid (100x200, Np=2000)
├── config.yaml                  source configuration (copied verbatim)
├── config.json                  every post-init scalar as JSON, flat:
│                                user params (_dx, _gamma, ...), derived
│                                params (u_max, theta_sand, size_indices,
│                                _stepmax), the array table, and _meta
├── state/<name>.npy             one uncompressed .npy per post-init model
│                                array, at the dtype the reference holds
│                                (float32 state fields, int64 index grids,
│                                float64 coordinate vectors)
├── manifest.json                dtype/shape/role/sha256 per array, the
│                                encoding contract, RNG provenance, and the
│                                excluded scratch buffers (with shapes)
└── golden.json                  (small fixture only) pinned eta[:5, 4]
                                 checkpoints after 1 and 10 updates, sourced
                                 from tests/integration/test_consistent_outputs.py
```

The npz→npy choice is deliberate: uncompressed `.npy` is a 128-byte header
plus raw values, so the C and Rust ports parse fixtures with zero
dependencies, while the harness loads the same bytes with NumPy.

**Excluded scratch buffers.** `free_surf_flag` and `free_surf_walk_inds`
are zeroed and refilled at the start of every water iteration
(`pyDeltaRCM/water_tools.py`) and carry no state between updates. They are
not committed (on the default grid the walk buffer alone is ~5 MB of
zeros); the manifest records their dtypes and shapes so ports allocate
them as zeros. Everything else in the model's post-init state is committed
and digest-pinned.

### Running the Python reference

```shell
# full methodology on the small fixture: 5 warm repeats, goldens, cold start
python benchmarks/run_reference.py \
    --fixture bench/fixture_default_small --verify-golden

# default-grid workload (timed window of 10 updates, no parity checks)
python benchmarks/run_reference.py --fixture bench/fixture_default

# CI preset: one warm repeat on the small fixture, cold start skipped
python benchmarks/run_reference.py \
    --fixture bench/fixture_default_small --smoke --skip-cold
```

Methodology (matching the two-regime harness above): warm-up updates
discard numba compilation and transients; each update is timed
individually around `DeltaModel.update()` in-process; the headline number
is the median of per-repeat medians (default 5 repeats, each re-initializing
the model so repeats are independent); and the JIT cold start is measured
exactly once in a fresh subprocess (import + init + first update) and
reported as its own line item, never inside a warm median.

Parity is not optional: before timing, the runner byte-compares its
post-init state against the fixture's committed arrays (a stale fixture
fails the run, it is never silently re-normalized), and with
`--verify-golden` the seeded eta checkpoints must match at `rtol=1e-6` —
the same gate the repo's own consistency tests apply. A checkpoint that
cannot be reached within the warm-up + timed window is an error, not a
pass. A parity mismatch is a bug in whatever produced it — RNG stream,
draw order, or operation order — and is never resolved by loosening
tolerance.

### What the ports consume

A port reads `config.json` + `state/*.npy`, allocates the two excluded
scratch buffers as zeros, runs its update loop, and writes the same
fields back out (`.npy` + JSON) plus an internally measured compute
clock. The harness verifies port outputs against `manifest.json` digests
and `golden.json` checkpoints; only then do the port's timings appear
alongside the Python reference's in `results.json`.

Regenerating a fixture after a config or init change:
`python benchmarks/normalize_fixture.py --config benchmarks/configs/default_small.yaml --out bench/fixture_default_small --name fixture_default_small`.
Regeneration wipes the fixture directory (dropped arrays cannot linger)
and preserves the curated `golden.json`. The determinism test
(`tests/test_benchmark_fixtures.py`) re-normalizes and compares digests
against the committed manifest on every run.

## Comparing the water backends

Build the extension first (`python -m pip install -e .`, with Rust installed).
Run the same harness sequentially on an otherwise idle machine:

```shell
PYDELTARCM_NATIVE=0 python benchmarks/benchmark_timestep.py \
    --scenarios early_water --output-dir benchmarks/results/reference
PYDELTARCM_NATIVE=1 python benchmarks/benchmark_timestep.py \
    --scenarios early_water --output-dir benchmarks/results/native
```

On PowerShell, set `$env:PYDELTARCM_NATIVE = '0'` or `'1'` before the respective
command. The harness records the selected `water_backend` alongside the grid
configuration; verify it is `python` and `rust`, respectively. Compare the
reference mean divided by the native mean. Do not include compilation or
warm-up time, or compare runs with different parcel counts. The water-port
gate is a ratio of at least 2 on `early_water`; it is not a promise about the
sediment-dominated scenario.

Native and reference RNG streams differ deliberately. Their scientific
comparison lives in `tests/integration/test_native_water_statistics.py`:
three seed families, five independently derived seeds each, eight updates
per run and backend. The entire paired 90% confidence interval must fit
within the fixed relative margins: 15% for bed relief and deposition per
update, 10% for channel-mouth discharge. This is a finite ensemble test, not
proof of equivalence for every model configuration. Three additional seeded
native repeat tests require bit-identical fields. Run with `pytest -s` to
print seeds, per-run metrics, intervals, and margins.
