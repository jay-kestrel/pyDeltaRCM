"""Feasibility spike: bed-following water-weight kernel under numba (spec V1).

Task todo_ASgakB2P. pyDeltaRCM ``pyDeltaRCM/`` tree is identical between the
spec-pinned commit ``a107c1d`` and ``develop`` at ``5800391`` (verified by
diff; the only intervening change adds ``benchmarks/`` and CI config), so
results at this checkout apply verbatim to the pinned commit.

Retires the single unverified architectural bet of the submarine-fan spec
(``art_dQNJt34w``, criterion V1) before any feature work, by exercising both
extension paths end-to-end:

  (a) a throwaway ``DeltaModel`` subclass overrides the documented
      water-weighting seam and swaps in a jitted bed-following weight kernel,
      run through the stock ``update()`` loop on a fully-submerged domain;
  (b) the stock ``SandRouter``/``MudRouter`` jitclasses are driven directly
      with substituted weights, honoring the field-copy contract documented
      at ``sed_tools.py`` (``route_all_sand_parcels``, "If you attempt to
      drop in a replacement SandRouter, you will need to update these
      fields!!").

Seam correction discovered while pinning this spike to the code (see
``SPIKE_DECISION.md``): there is **no** ``DeltaModel._get_water_weight_array``
method. ``_get_water_weight_array`` is a module-level jitted function in
``water_tools.py``; the overridable seam is the public method
``get_water_weight_array``, which ``run_water_iteration`` calls through
``self`` and whose docstring chain ("consider reimplementing this method,
which calls a custom version of ``_get_weight_at_cell_water``") designates it
as the replacement point. This spike overrides that method.

Run from a repo checkout (no installation needed):

    python spikes/spike_weight_kernel.py

Exit code 0 = every check passed.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numba import float32, njit
from numba.experimental import jitclass

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # allow running the spike from a checkout without installing the package
    sys.path.insert(0, str(_REPO_ROOT))

from pyDeltaRCM import DeltaModel
from pyDeltaRCM.shared_tools import (
    custom_pad,
    get_start_indices,
    get_weight_sfc_int,
    set_random_seed,
)
from pyDeltaRCM.water_tools import (
    _get_water_weight_array,  # stock array kernel: differential baseline
    _get_weight_at_cell_water,  # stock per-cell kernel: reused fallback logic
)


# ---------------------------------------------------------------------------
# Path (a) kernel: bed-following water weights
# ---------------------------------------------------------------------------

@njit
def _get_bed_following_water_weight_array(
    depth,
    eta,
    mod_water_weight,
    cell_type,
    qx,
    qy,
    ivec_flat,
    jvec_flat,
    distances_flat,
    dry_depth,
    gamma,
    theta_water,
):
    """Bed-following replacement for ``water_tools._get_water_weight_array``.

    Structurally identical to the stock kernel, with exactly one physics
    substitution: the slope ("free-surface") term is computed from the *bed*
    elevation ``eta`` instead of the ``stage`` field, so ``weight_sfc``
    becomes a downdip bed-gravity term pulling parcels toward lower bed
    elevation. The inertia term, the ``(depth * mod)**theta`` weighting, and
    every zero-weight fallback are reused unchanged from the stock
    ``_get_weight_at_cell_water``.
    """
    L, W = depth.shape
    pad_eta = custom_pad(eta)
    pad_depth = custom_pad(depth)
    mod_pad_water_weight = custom_pad(mod_water_weight)
    pad_cell_type = custom_pad(cell_type)

    water_weights = np.zeros((L, W, 9))

    for i in range(L):
        for j in range(W):
            eta_nbrs = pad_eta[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
            depth_nbrs = pad_depth[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
            mod_water_weight_nbrs = mod_pad_water_weight[
                i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1
            ]
            ct_nbrs = pad_cell_type[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]

            # the substitution: bed elevation takes the role of stage, so the
            # slope term is a bed-gravity term toward lower bed elevation
            weight_sfc, weight_int = get_weight_sfc_int(
                eta[i, j],
                eta_nbrs.ravel(),
                qx[i, j],
                qy[i, j],
                ivec_flat,
                jvec_flat,
                distances_flat,
            )

            water_weights[i, j] = _get_weight_at_cell_water(
                (i, j),
                weight_sfc,
                weight_int,
                depth_nbrs.ravel(),
                mod_water_weight_nbrs.ravel(),
                ct_nbrs.ravel(),
                dry_depth,
                gamma,
                theta_water,
            )

    return water_weights


# ---------------------------------------------------------------------------
# Path (a) throwaway subclass
# ---------------------------------------------------------------------------

class SpikeFanModel(DeltaModel):
    """Throwaway subclass exercising the extension seams for spec V1.

    Touches documented seams only:

    * ``hook_import_files`` — registers subclass parameters (the sanctioned
      custom-YAML path; stock-name collisions raise ``ValueError``).
    * ``create_other_variables`` — computes density-current derived constants
      *after* the base method. Note: ``hook_create_other_variables`` fires
      *before* the base method, so ``g``/``dx`` (defined by
      ``create_other_variables`` → ``set_constants``) do not exist yet there;
      a ``super()``-then-derive override is the working pattern.
    * ``hook_after_create_domain`` — replaces the planar-topset + flat-basin
      bathymetry with a fully-submerged planar bed descending downdip; every
      cell wet; the stock elliptical edge mask kept as the outflow boundary.
    * ``get_water_weight_array`` — the documented water-weighting seam,
      swapped for the jitted bed-following kernel.
    * ``compute_free_surface`` — nothing to relax: the current's surface
      follows the bed at constant thickness by construction.
    """

    def hook_import_files(self) -> None:
        # subclass parameters are type-checked and defaulted like stock ones
        self.subclass_parameters["slope_gradient"] = {
            "type": ["float"],
            "default": 0.03,
        }
        self._bed_kernel_calls = 0

    def create_other_variables(self) -> None:
        super().create_other_variables()
        # derived constants for the density current, mirroring the pattern
        # create_boundary_conditions uses for the riverine gamma. Thickness
        # equal to h0 also keeps the stock inlet bed BC (finalize_timestep)
        # inert, so no stock bookkeeping drifts during the spike.
        self.h_current = self.h0
        self.gamma_c = self.g * self.slope_gradient * self.dx / self.u0**2

    def hook_after_create_domain(self) -> None:
        # shelf-to-basin bed descending downdip (+y / increasing row), canyon
        # head at the inlet row; stage trails the bed at constant thickness
        self.eta[:] = -self.y * self.slope_gradient * self.dx
        self.stage[:] = self.eta + self.h_current
        self.depth[:] = self.h_current
        # fully submerged: former land cells become ocean; the stock
        # elliptical edge mask (-1) is kept as the outflow boundary
        self.cell_type[self.cell_type == -2] = 0
        # the flow fields describe the current, which starts at rest
        self.qx[:] = 0
        self.qy[:] = 0
        self.qw[:] = 0
        self.ux[:] = 0
        self.uy[:] = 0
        self.uw[:] = 0
        # refresh the bed trackers to the submarine initial condition
        self.eta0[:] = self.eta
        self.eta_init[:] = self.eta

    def get_water_weight_array(self) -> None:
        # the documented seam: "If you are trying to change water weighting
        # behavior, consider reimplementing this method" — swap the stock
        # jitted kernel for the bed-following one, blended by gamma_c
        # (bed-gravity vs inertia)
        self._bed_kernel_calls += 1
        self.water_weights = _get_bed_following_water_weight_array(
            self.depth,
            self.eta,
            self.mod_water_weight,
            self.cell_type,
            self.qx,
            self.qy,
            self.ivec_flat,
            self.jvec_flat,
            self.distances_flat,
            self.dry_depth,
            self.gamma_c,
            self._theta_water,
        )

    def compute_free_surface(self) -> None:
        # no free surface to relax: the current's surface is bed-following by
        # construction; recompute stage as the bed evolves
        self.stage[:] = self.eta + self.h_current


# ---------------------------------------------------------------------------
# model construction (mirrors tests/utilities.py fixture style)
# ---------------------------------------------------------------------------

def _write_config(tmpdir, seed=0):
    """Write a minimal spike YAML config; returns its path."""
    cfg = {
        "out_dir": str(tmpdir),
        "Length": 50.0,
        "Width": 100.0,
        "dx": 1.0,
        "L0_meters": 1.0,
        "N0_meters": 3.0,  # N0 = max(3, ...), avoids the rounding warning
        "S0": 0.0002,
        "itermax": 1,
        "Np_water": 10,
        "Np_sed": 10,
        "u0": 1.0,
        "h0": 1.0,
        "H_SL": -9999.0,  # below the whole domain: the stock floor never binds
        "SLR": 0.0,
        "f_bedload": 0.5,
        "C0_percent": 0.5,
        "toggle_subsidence": False,
        "seed": seed,
    }
    path = os.path.join(str(tmpdir), "spike_parameters.yaml")
    with open(path, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
    return path


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

_RESULTS = []


def _ok(message):
    _RESULTS.append(message)
    print(f"  PASS  {message}")


def check_kernel_differential():
    """Kernel-level differential: bed kernel routes into a bed low, stock does not."""
    print("[1] Kernel differential on synthetic arrays (5x5, no model)")
    # D8 walk constants, copied verbatim from init_tools.set_constants
    sqrt2, sqrt05 = np.sqrt(2), np.sqrt(0.5)
    distances_flat = np.array(
        [[sqrt2, 1, sqrt2], [1, 1, 1], [sqrt2, 1, sqrt2]], dtype=np.float32
    ).flatten()
    ivec_flat = np.array(
        [[-sqrt05, 0, sqrt05], [-1, 0, 1], [-sqrt05, 0, sqrt05]], dtype=np.float32
    ).flatten()
    jvec_flat = np.array(
        [[-sqrt05, -1, -sqrt05], [0, 0, 0], [sqrt05, 1, sqrt05]], dtype=np.float32
    ).flatten()

    shape = (5, 5)
    eta = np.zeros(shape, dtype=np.float32)
    stage = np.zeros(shape, dtype=np.float32)  # flat free surface
    eta[3, 4] = -1.0  # a bed low south-east of the center cell (2, 3)
    depth = np.ones(shape, dtype=np.float32)
    cell_type = np.zeros(shape, dtype=np.int64)
    qx = np.zeros(shape, dtype=np.float32)
    qy = np.zeros(shape, dtype=np.float32)
    mod = np.ones(shape, dtype=np.float32)
    dry_depth, gamma, theta = 0.1, 1.0, 1.0  # pure slope-term comparison

    w_stock = _get_water_weight_array(
        depth, stage, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
        distances_flat, dry_depth, gamma, theta,
    )[2, 3]
    w_bed = _get_bed_following_water_weight_array(
        depth, eta, mod, cell_type, qx, qy, ivec_flat, jvec_flat,
        distances_flat, dry_depth, gamma, theta,
    )[2, 3]

    # SE = neighborhood index 8
    assert w_bed[8] > 0.99, "bed kernel does not fall into the SE bed low"
    assert w_stock[8] < 0.2, "stock kernel unexpectedly prefers the SE low"
    assert w_bed[8] > w_stock[8]
    _ok(f"kernel differential: bed kernel w[SE]={w_bed[8]:.3f} (falls into "
        f"the bed low) vs stock w[SE]={w_stock[8]:.3f} (flat-stage fallback)")


def check_path_a(n_updates=5):
    """Subclass-override path: the seam works end-to-end under numba."""
    print(f"[A] Path A: subclass override of get_water_weight_array "
          f"({n_updates} stock update() calls)")
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        cfg = _write_config(tmp, seed=0)
        model = SpikeFanModel(input_file=cfg)

        # config seam: subclass parameter defaulted (not present in YAML)
        assert model.slope_gradient == 0.03, "subclass parameter default failed"
        # derived constants exist after the super()-then-derive override
        expected_gamma = model.g * 0.03 * model.dx / model.u0**2
        assert np.isclose(model.gamma_c, expected_gamma), "gamma_c mismatch"
        _ok("constructs from YAML; subclass parameter defaulted; "
            "gamma_c = {:.4f}".format(model.gamma_c))

        # domain seam: fully-submerged monotonic shelf-to-basin bed
        center = model.eta[:, model.CTR]
        assert np.all(np.diff(center) < 0), "bed not strictly descending downdip"
        assert np.all(model.depth > 0), "domain has dry cells"
        assert np.all(model.cell_type != -2), "land/wall cells remain"
        assert model.inlet.size == 3, "inlet cells missing"
        assert np.any(model.cell_type == -1), "elliptical edge mask lost"
        _ok("domain: monotonic downdip bed, all wet, inlet present, "
            "edge mask kept")

        # run the seam: kernels compile and route through update()
        for _ in range(n_updates):
            model.update()
        assert model._bed_kernel_calls == n_updates, (
            f"override called {model._bed_kernel_calls} times, "
            f"expected {n_updates}"
        )
        _ok(f"override dispatched {model._bed_kernel_calls} times via "
            "dynamic dispatch through the stock update() loop")

        # weights at a mid-domain cell prefer the downdip neighbor
        # neighborhood order: 0=NW 1=N 2=NE 3=W 4=self 5=E 6=SW 7=S 8=SE
        w = model.water_weights[20, model.CTR]
        assert w[7] > w[1], "downdip (S) weight does not beat updip (N)"
        assert w[7] > w[5] and w[7] > w[3], "downdip weight does not beat cross-dip"
        assert w[7] == w.max(), "downdip is not the modal step"
        _ok(f"weights prefer downdip: w[S]={w[7]:.3f} vs w[N]={w[1]:.3f}, "
            f"w[E]={w[5]:.3f}, w[W]={w[3]:.3f}")

        # water parcels convey flux to the far downstream basin through the seam
        assert model.qwn[30:44, 40:60].max() > 0, \
            "no water flux reached rows 30-44 (downdip conveyance failed)"
        _ok("water flux reaches the far downdip basin (rows 30-44)")

        # sediment routing ran on the bed-following flow without numba errors
        deta = model.eta - model.eta_init
        assert deta.max() > 0, "no deposition occurred"
        assert np.any(deta[1:, :] > 0), "no deposition outside the inlet row"
        _ok(f"deposition occurred (max eta rise {deta.max():.3f} m); "
            "stock routers ran on the current's flow fields")

        # stability: no NaNs anywhere in the state
        for name in ("eta", "stage", "depth", "qx", "qy", "qw", "uw"):
            arr = getattr(model, name)
            assert np.all(np.isfinite(arr)), f"nonfinite values in {name}"
        _ok("no NaNs in eta/stage/depth/flow fields")

        # determinism: an identical-seed model reproduces eta bitwise
        with tempfile.TemporaryDirectory() as tmp2:
            model_b = SpikeFanModel(input_file=_write_config(tmp2, seed=0))
            for _ in range(n_updates):
                model_b.update()
            assert np.array_equal(model.eta, model_b.eta), \
                "identical seeds produced different eta"
        _ok("determinism: identical seeds reproduce eta bitwise")

        print(f"  (path A wall time: {time.time() - t0:.1f} s, "
              f"{n_updates} updates of a {model.L}x{model.W} domain)")


def check_path_b():
    """Stock routers + substituted weights, honoring the field-copy contract."""
    print("[B] Path B: stock SandRouter/MudRouter jitclasses with "
          "substituted weights")
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        cfg = _write_config(tmp, seed=0)
        model = SpikeFanModel(input_file=cfg)
        model.update()  # establish a coherent bed-following flow field

        # dtype contract: model arrays already match the router r_spec
        assert model.eta.dtype == np.float32
        assert model.cell_type.dtype == np.int64
        _ok("model state arrays match router r_spec dtypes (float32/int64)")

        # ---- B1: field-copy contract, model path (sed_tools.py:149-162,
        # "If you attempt to drop in a replacement SandRouter, you will need
        # to update these fields!!") ----
        # The contract is value consistency via the copy-back assignments;
        # the router re-creates pad_depth internally (custom_pad), so object
        # identity is not the invariant — synchronized VALUES are.
        pre_eta = model.eta.copy()
        model.route_all_sand_parcels()  # stock path: router run + copy-back
        for f in ("eta", "depth", "pad_depth", "uw", "qs"):
            model_v = getattr(model, f)
            router_v = np.asarray(getattr(model._sr, f))
            assert np.array_equal(model_v, router_v), \
                f"sand: {f} not consistent between model and router"
        assert np.array_equal(model.depth, model.pad_depth[1:-1, 1:-1]), \
            "sand: padded depth out of sync with depth"
        assert not np.array_equal(model.eta, pre_eta), \
            "sand routing did not modify the bed"
        pre_eta = model.eta.copy()
        model.route_all_mud_parcels()
        for f in ("eta", "depth", "pad_depth", "uw"):
            model_v = getattr(model, f)
            router_v = np.asarray(getattr(model._mr, f))
            assert np.array_equal(model_v, router_v), \
                f"mud: {f} not consistent between model and router"
        assert not np.array_equal(model.eta, pre_eta), \
            "mud routing did not modify the bed"
        _ok("field-copy contract: routers mutate model arrays in place and "
            "the documented copy-back keeps DeltaModel consistent")

        # ---- B2: substituted weights through the stock SandRouter ----
        num_starts = int(model._Np_sed * model._f_bedload)
        inlet_weights = model.get_inlet_weights_sediment(parcel_type="sand")
        start_indices = get_start_indices(model.inlet, inlet_weights, num_starts)

        def sand_run_with(mod_sed_weight):
            """Drive the stock SandRouter on fresh copies of the state."""
            out = dict(
                eta=model.eta.copy(),
                stage=model.stage.copy(),
                depth=model.depth.copy(),
                cell_type=model.cell_type.copy(),
                uw=model.uw.copy(),
                ux=model.ux.copy(),
                uy=model.uy.copy(),
                Vp_dep_mud=np.zeros_like(model.eta),
                Vp_dep_sand=np.zeros_like(model.eta),
                qw=model.qw.copy(),
                qx=model.qx.copy(),
                qy=model.qy.copy(),
                qs=np.zeros_like(model.eta),
                mod_sed_weight=mod_sed_weight,
                mod_stable_weight=model.mod_stable_weight.copy(),
            )
            model._sr.run(
                start_indices,
                out["eta"], out["stage"], out["depth"], out["cell_type"],
                out["uw"], out["ux"], out["uy"],
                out["Vp_dep_mud"], out["Vp_dep_sand"],
                out["qw"], out["qx"], out["qy"], out["qs"],
                out["mod_sed_weight"], out["mod_stable_weight"],
            )
            return out["Vp_dep_sand"], out["eta"]

        def centroid_x(dep):
            mass = dep.sum()
            if mass <= 0:
                return np.nan
            cols = np.arange(dep.shape[1], dtype=np.float64)
            return float((dep.sum(axis=0) * cols).sum() / mass)

        # identical RNG stream for both runs; only the weights differ
        mod_baseline = np.ones_like(model.mod_sed_weight)
        mod_west = np.ones_like(model.mod_sed_weight)
        mod_west[:, : model.W // 2] = 100.0  # amplify weight into the west half

        set_random_seed(0)
        dep_base, _ = sand_run_with(mod_baseline)
        set_random_seed(0)
        dep_west, _ = sand_run_with(mod_west)

        assert dep_base.sum() > 0, "baseline sand routing deposited nothing"
        assert dep_west.sum() > 0, "substituted-weight routing deposited nothing"
        col_base = centroid_x(dep_base)
        col_west = centroid_x(dep_west)
        assert col_west < col_base, (
            f"substituted weights did not pull deposits west "
            f"(baseline centroid col {col_base:.2f}, "
            f"substituted {col_west:.2f})"
        )
        _ok(f"substituted weights steer the stock SandRouter: deposit "
            f"centroid col {col_base:.2f} → {col_west:.2f} under a 100x "
            f"west-half weight bias")

        # ---- B3: a user-defined jitclass can occupy the router slot ----
        probe_spec = [("_dx", float32), ("vp", float32)]

        @jitclass(probe_spec)
        class RouterSlotProbe:
            """Mechanism probe: stock run() signature, writes through arrays."""

            def __init__(self, dx, vp):
                self._dx = dx
                self.vp = vp

            def run(self, start_indices, eta, stage, depth, cell_type, uw, ux,
                    uy, Vp_dep_mud, Vp_dep_sand, qw, qx, qy, qs,
                    mod_sed_weight, mod_stable_weight):
                n = start_indices.shape[0]
                for p in range(n):
                    ind = start_indices[p]
                    Vp_dep_sand.flat[ind] += self.vp
                    eta.flat[ind] += self.vp / (self._dx * self._dx)

        probe = RouterSlotProbe(np.float32(model.dx), np.float32(0.01))
        eta_probe = model.eta.copy()
        vpd_probe = np.zeros_like(model.eta)
        probe.run(start_indices, eta_probe, model.stage.copy(),
                  model.depth.copy(), model.cell_type.copy(),
                  model.uw.copy(), model.ux.copy(), model.uy.copy(),
                  np.zeros_like(model.eta), vpd_probe,
                  model.qw.copy(), model.qx.copy(), model.qy.copy(),
                  np.zeros_like(model.eta),
                  model.mod_sed_weight.copy(),
                  model.mod_stable_weight.copy())
        assert vpd_probe.sum() > 0 and np.any(eta_probe != model.eta), \
            "user-defined router probe did not write through the arrays"
        _ok("user-defined jitclass occupies the router slot with the stock "
            "run() signature and writes through the passed arrays")

        print(f"  (path B wall time: {time.time() - t0:.1f} s)")


def main():
    print("=" * 72)
    print("SPIKE: bed-following weight kernel under numba — spec V1 "
          f"(task todo_ASgakB2P)")
    print("=" * 72)
    check_kernel_differential()
    check_path_a()
    check_path_b()
    print("=" * 72)
    print(f"SPIKE RESULT: ALL {len(_RESULTS)} CHECKS PASSED — "
          "numba-safe bed-following routing confirmed on both paths")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
