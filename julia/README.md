# DeltaRCMJulia — Julia port of the pyDeltaRCM water-routing slice

A dependency-free Julia port of the pyDeltaRCM water-routing vertical slice:
the `shared_tools.py` helper layer plus the `water_tools.py` Numba kernels and
phase methods, driven end-to-end from a dumped model state. The port is
validated **bit-exactly** against reference runs of the Python original.

This is the first track of the Julia-port exploration (`julia/` subproject,
per the approved build spec). It ports only the water phase; sediment
routers, topographic diffusion, and subsidence are later tracks.

## Layout

```
julia/
├── Project.toml            # no dependencies — stdlib only
├── src/
│   ├── DeltaRCMJulia.jl    # module: conventions + exports
│   ├── mt19937.jl          # MT19937 matching NumPy's legacy RandomState
│   ├── npy.jl              # minimal .npy reader (C-order, little-endian)
│   ├── shared_tools.jl     # custom_pad/ravel/unravel, random_pick, get_steps…
│   └── water_tools.jl      # WaterState + water kernels + phase methods
├── test/
│   ├── fixtures.jl         # fixture hydration + bit-exact comparator
│   ├── test_mt19937.jl     # RNG stream parity (pinned vectors)
│   ├── test_shared_tools.jl# helper semantics + translation audits
│   ├── test_water_weights.jl # kernel-level equivalence vs numba dumps
│   ├── test_route_water.jl # end-to-end route_water equivalence, 2 updates
│   └── fixtures/reference/ # committed Python-reference dumps (npy)
└── tools/
    └── dump_reference.py   # regenerates the reference dumps (run on Python side)
```

The committed fixture `bench/fixture_default_small/` (10×10 grid, seed 0,
10 water parcels) is the shared state handoff: the Julia tests hydrate a
`WaterState` from its post-init arrays and run the same update sequence the
Python model ran to produce `test/fixtures/reference/`.

## Running the tests

```sh
julia --project=. test/runtests.jl
```

Requires Julia 1.10+ and nothing else. CI (`.github/workflows/julia.yml`)
downloads the Julia tarball in-script — the org does not allow external
Actions.

## Conventions (the translation contract)

These decisions are fixed; every kernel depends on them.

- **Array layout.** Python state arrays are `(L, W)` (x, y); the Julia port
  stores every field as `(W, L)` — column-major, so `M[j + 1, i + 1]` reads
  Python's `A[i, j]`. The `.npy` reader reverses the axis order so dumps load
  directly.
- **Flat parcel indices stay 0-based.** Parcel position arrays keep Python's
  C-order flat integers (`flat = i*W + j`) with the `0` inactive sentinel;
  they are data, not Julia indices. Where Python's fancy indexing reads
  `cell_type.flat[inds]`, the port computes `(f0 % W, f0 ÷ W)` explicitly —
  the sentinel `f0 == 0` then reads cell `(0, 0)` exactly like the reference.
  Direction code `4` ("stay") pins the sentinel via `_calculate_new_inds`.
- **RNG.** The Python reference consumes Numba's internal MT19937 seeded the
  NumPy-legacy way. The port implements that generator exactly (see
  `mt19937.jl`); first test vectors: seed 0 → `0.5488135039273248`,
  seed 0 limit 0.1 → `0.07151893663724194`. The RNG is an explicit object
  passed through every phase, and its full state can be dumped/restored
  (`pack_rng_state` / `set_rng_state!`), so the Julia port can continue a
  Python run bit-identically. Consumption order per update is part of the
  contract: inlet starts, one draw per active parcel per water step.
- **Dtype fidelity.** State fields are `Float32` (as in the reference);
  `cell_type` and parcel indices are `Int64`. Free-surface accumulators
  (`sfc_visit`, `sfc_sum`) and `water_weights` are `Float64` — numba's
  `np.zeros` allocates float64 there. Numba reduction semantics are
  reproduced: `pysum` accumulates float32 arrays in float32, and the
  rebalancing division stays float32 (`f32 array / f32 scalar → f32 array`)
  before the gamma blend widens to float64.
- **Errors.** The reference raises `RuntimeError` from inside kernels on NaN
  weights and empty walks; the port raises `ErrorException` / `DomainError`
  at the same sites instead of silently falling back.

## Parity status

| Layer | Tier | Status |
|---|---|---|
| MT19937 stream (seed 0/1, state export/import) | A (bit-exact) | passing |
| `random_pick` / start indices (RNG consumption order) | A | passing |
| `_get_water_weight_array` (per-cell kernel over full domain) | A | passing |
| `_accumulate_free_surface_walks`, `_smooth_free_surface` | A | passing |
| `route_water` end-to-end, update 1 (`time_iter = 0`) | A (all 16 dumped fields + post-run RNG state) | passing |
| `route_water` end-to-end, update 2 (`time_iter = 1`, omega_sfc blend) | A | passing |

Bit-exact means `isequal` on every element of every dumped field, including
the RNG state tuple after the update. Regenerate reference dumps with
`julia/tools/dump_reference.py` (needs Python + numpy + numba; see the
script header) when the fixture changes.

## Reference quirks preserved on purpose

- `qwn_div = np.ones((L, W))` is float64 in the reference while everything
  it touches is float32 — the port keeps the float64 divider (double
  rounding on store).
- Discharge augmentation at the inlet accumulates duplicates through a
  float32 store of a float64 sum (per-parcel, not per-unique-cell).
- `int(np.round(...))` half-to-even rounding in loop relocation — Julia
  `round(Int, ...)` matches.
- The upstream dead `whr_agg` line (`iteration_tools.py`) is not part of the
  water slice and is not ported.
