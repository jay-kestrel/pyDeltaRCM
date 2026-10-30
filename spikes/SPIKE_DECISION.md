# Spike decision: bed-following water-weight kernel under numba (spec V1)

**Task:** todo_ASgakB2P · **Spec:** art_dQNJt34w (criterion V1) · **Findings:** art_WXbmVQa7 §3.2, §8
**Code verified at:** `develop` `5800391` — `pyDeltaRCM/` tree identical to the spec-pinned `a107c1d` (the one intervening commit adds only `benchmarks/` and CI config; verified by diff).
**Reproduce:** `python spikes/spike_weight_kernel.py` from a repo checkout (~35 s wall, dominated by one-time numba JIT compilation).
**Result: ALL 13 CHECKS PASSED, exit 0, no numba errors.**

## Verdict

**Both extension paths work under numba. The subclass-override architecture is confirmed; the spec proceeds as designed (Option A). No fallback to parameter/`mod_*` steering is needed, and the guarded-core-branch escalation (Option B) is not triggered.**

| Question the spike exists to answer | Answer |
| --- | --- |
| Can a jitted bed-following weight kernel run at the pinned commit? | Yes — compiled and ran through the stock `update()` loop with zero numba errors. |
| Does the subclass-override path work end-to-end? | Yes — the override dispatched exactly once per `update()` through dynamic dispatch; routing, deposition, and determinism all held on a fully-submerged sloped domain. |
| Can the stock SandRouter/MudRouter jitclasses be driven with substituted weights? | Yes — substituted `mod_sed_weight` arrays measurably steered deposit location through the stock router's own weight computation. |
| Does the documented field-copy contract hold? | Yes — with one precision: the contract is **value consistency** via the copy-back assignments, not object identity (see Contract notes). |

## Path A — subclass override of the water-weight seam

`SpikeFanModel(DeltaModel)` (defined in the spike script, throwaway) overrides:

- `hook_import_files` — registers `slope_gradient` via `subclass_parameters`; the stock validator defaulted it (not present in YAML) exactly like a stock parameter.
- `create_other_variables` — derives `gamma_c = g · slope_gradient · dx / u0²` **after** `super().create_other_variables()`.
- `hook_after_create_domain` — replaces bathymetry with a planar bed descending downdip at 0.03; every cell wet; stock elliptical edge mask kept.
- `get_water_weight_array` — the seam; swaps in the jitted bed-following kernel.
- `compute_free_surface` — no-op relaxation: `stage = eta + h_current`, recompute as the bed evolves.

Observed on a 50×100 domain, `Np_water = Np_sed = 10`, `itermax = 1`, 5 `update()` calls:

- Override dispatched exactly 5 times via dynamic dispatch through the unmodified stock `update()` loop.
- Bed-following weights at a mid-domain cell: `w[S] = 0.357` (modal) vs `w[N] = 0.000`, `w[E] = 0.235`, `w[W] = 0.000` — parcels preferentially step downdip; zero-weight fallback paths preserved (stock `_get_weight_at_cell_water` reused verbatim for the per-cell fallbacks).
- Water flux reached the far downdip basin (rows 30–44); deposition occurred (max `eta` rise 0.507 m) with the **stock** sand/mud routers running on the current's flow fields.
- No NaNs in `eta/stage/depth/qx/qy/qw/uw`; two identical-seed models reproduced final `eta` bitwise.

## Path B — stock routers with substituted weights

- **Field-copy contract** (`sed_tools.py` copy-back blocks after `SandRouter.run` / `MudRouter.run`, flagged by the comment "If you attempt to drop in a replacement SandRouter, you will need to update these fields!!"): after `route_all_sand_parcels()`, model fields `eta`, `depth`, `pad_depth`, `uw`, `qs` are value-identical to the router's; after `route_all_mud_parcels()`, `eta`, `depth`, `pad_depth`, `uw` are (the MudRouter has no `qs` in its `run` signature, and its copy-back omits `qs` accordingly). Padded depth stays synchronized with depth (`pad_depth[1:-1, 1:-1] == depth`). Both routers modified the bed in place.
- **Substituted weights steer the stock router:** with the RNG stream re-seeded between runs (only the weights differ), a 100× `mod_sed_weight` bias on the west half moved the sand-deposit centroid from column 49.40 to 48.40. The stock router's own `_get_weight_at_cell_sediment` honors whatever weight array it is handed — no router replacement is required for steering.
- **Router-slot substitution is mechanically possible:** a user-defined `@jitclass` with the stock `run(...)` signature compiled under numba and wrote through the passed arrays (deposit accumulator + bed elevation updated in the caller's copies).

## Seam correction (matters for implementation)

There is **no `DeltaModel._get_water_weight_array` method** — the name in the task brief and spec diagrams refers to the module-level jitted kernel `_get_water_weight_array` in `water_tools.py`. The overridable seam is the **public method `get_water_weight_array`** on the water mixin, which `run_water_iteration` calls through `self`, and whose stock body's docstring invites reimplementation ("If you are trying to change water weighting behavior, consider reimplementing this method, which calls a custom version of `_get_weight_at_cell_water`"). Overriding a method named `_get_water_weight_array` on the subclass would be dead code. The correct override, verified live in this spike, is:

```python
def get_water_weight_array(self) -> None:
    self.water_weights = _get_bed_following_water_weight_array(
        self.depth, self.eta, self.mod_water_weight, self.cell_type,
        self.qx, self.qy, self.ivec_flat, self.jvec_flat,
        self.distances_flat, self.dry_depth, self.gamma_c,
        self._theta_water,
    )
```

The bed-following kernel substitutes exactly one input: the slope term receives bed elevation `eta` where the stock kernel passes `stage`. The inertia term, `(depth * mod) ** theta` weighting, and every zero-weight fallback are the stock `_get_weight_at_cell_water` code, reused unchanged. Kernel-level differential proof: on identical synthetic arrays with a bed low offset from a flat free surface, the bed kernel gives the low's direction weight 1.000 (exclusive pull) while the stock kernel gives 0.125 (uniform fallback — it cannot see a bed low at all).

## Contract notes for the implementer

1. **The field-copy contract is value-based, not identity-based.** `SandRouter.run` re-creates `pad_depth` internally (`custom_pad(depth)`), so `model.pad_depth is router.pad_depth` is false even though the copy-back assignment synchronizes values. Replacement-router code must copy values back per the comment; tests should assert value equality, not `is`.
2. **`hook_create_other_variables` fires *before* the base `create_other_variables`** (`model.py` init sequence), so `g` and `dx` do not exist inside that hook. Derived constants that depend on them must use the `super()`-then-derive override pattern the spike used (or a later hook).
3. **The stock inlet bed BC stays inert when `h_current == h0`** (`finalize_timestep` shifts inlet `eta` by `stage[0, inlet] − h0 − eta[0, inlet]`, which is zero at constant thickness), and **the sea-level floor never binds when `H_SL` is below the whole domain**. Both make the stock bookkeeping safe for a submerged domain without touching upstream code.
4. **Numba constraints respected:** the kernel is a pure-array function (no Python objects held), compiled under `@njit`, and the spike passes with JIT enabled; the CI `DISABLE_JIT` leg exercises the same code paths un-jitted (same pure-array shape).

## Architecture call

Proceed with the spec's locked decision 2 (pure-subclass architecture, `pyDeltaRCM/submarine/` submodule):

- Water routing: subclass override of the public `get_water_weight_array` with the bed-following jitted kernel (this spike's Path A shape); `compute_free_surface` becomes a stage-tracking no-op relaxation.
- Sediment: stock routers with submarine-calibrated parameters; `mod_*` arrays suffice for spatially-varying closures (Path B shows substituted weights steering the stock router directly). No new router classes required for v1 — consistent with spec locked decision 4.
- Domain/config: `hook_after_create_domain` + `subclass_parameters`, exactly as the spike exercised.

No fallback and no escalation to core edits. Risk R1 (jitclass substitution) and R2 (stage/depth bookkeeping) are both retired for the v1 scope by the observed evidence above.
