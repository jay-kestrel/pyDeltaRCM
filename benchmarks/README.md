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
