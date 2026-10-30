# P2 water-phase validation

The P1 harness measured **3.27× faster updates** for the native water backend
on the early-water scenario on 2026-10-30. These are local Linux measurements,
not a claim about other machines or sediment-dominated workloads.

## Timing evidence

All runs use seed 0, a 60×120 grid, 1000 water and 1000 sediment parcels,
three water iterations, five discarded warm-up updates and 20 timed updates.
The two backend runs were sequential, after the test process had completed.

| Run | Source | Mean ± sample SD (seconds/update) |
| --- | --- | --- |
| Pre-port P1 checkout, `58003912` | `pre_port.json` | 0.5781 ± 0.1054 |
| P2 reference backend, `0803d7f` | `reference.json` | 0.5939 ± 0.0814 |
| P2 native backend, `0803d7f` | `native.json` | 0.1818 ± 0.0088 |

Speedup is reference mean / native mean = 3.2663, exceeding the 2× water-phase
gate. JSON contains every timed sample, configuration, timestamp, package
version and full commit hash. `pre_port.json` was captured before dispatch
was changed; its harness predates the explicit `water_backend` field.
The P2 JSON files identify the selected backend explicitly. These reruns
include develop's routing and checkpoint fixes through `3f98723`; the earlier
pre-port measurement is historical context, not the speedup denominator.
Tests and benchmarks ran at `0803d7fd6d189eb98db5e1a53cd0d25e7ddc94f4`;
the following commit refreshes evidence only.

Commands (with the same Python environment and release-built extension):

```shell
PYDELTARCM_NATIVE=0 python benchmarks/benchmark_timestep.py --scenarios early_water
PYDELTARCM_NATIVE=1 python benchmarks/benchmark_timestep.py --scenarios early_water
python -m pytest tests/integration/test_native_water_statistics.py -s
```

## Scientific evidence

`statistics.json` records 15 seed-paired native/reference experiments from
three seed families × five independent replicates, eight updates each.
The tests require the entire 90% paired-difference confidence interval to
fall within fixed relative-effect margins; they do not infer equivalence
merely from a nonsignificant difference.

| Metric | Native − reference mean | CI90 half-width | Allowed absolute effect |
| --- | --- | --- | --- |
| Bed relief (m) | 0.01018 | 0.22410 | 0.47831 (15%) |
| Deposition (m³/update) | 60.44271 | 355.63887 | 5221.95884 (15%) |
| Channel-mouth discharge (m³/s) | -0.99114 | 8.82859 | 123.60744 (10%) |

All three intervals fit inside their margins. All four integration tests
passed, including three native repeat runs requiring bit-identical fields.
This finite ensemble does not establish equivalence for every configuration.

## Other local checks

The installed-extension suite at the implementation above passed on Linux
with Python 3.13.14: 458 passed, 2 skipped, 1 xfailed, with pytest-mpl enabled
(343.35 seconds). Native contract tests under no-JIT coverage mode: 30 passed,
1 skipped; the skip concerns the existing Numba-only checkpoint RNG helper.
Rust formatting and clippy (`--all-targets -- -D warnings`) passed. New Python
modules pass Ruff; touched legacy modules pass syntax/error lint. CI remains
the authority for the 3-OS × 3-Python matrix.

Backend-only change: no rendered UI or visual workflow is modified, so no
browser screenshot evidence applies.
