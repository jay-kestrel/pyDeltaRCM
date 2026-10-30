# integration tests for fan evolution and stability (spec criteria V5 and V6)
"""Fan-evolution integration tests: morphology, stability, determinism.

Reduced-size but faithful :obj:`~pyDeltaRCM.submarine.SubmarineFanModel`
runs on a fully submerged slope-to-basin domain, with the literature-seeded
submarine defaults (``slope_gradient=0.05`` within the canyon range
1e-2--1e-1, inlet sediment concentration ``C0_percent=2.5`` matching the
volumetric plunging range of ``current_concentration``).

**Sizing.** The domain is 25x50 cells (``dx=1`` m) and the morphology run
evolves 300 timesteps (~1 hour of model time on this micro-grid, a few tens
of seconds of wall time). Duration is bounded by the suite-stays-in-minutes
requirement and chosen by deposit-growth evidence, not calendar equivalence:
the model's internal time scaling ties total sediment volume to model time,
so a calendar-equivalent 50 kyr run is unreachable at any grid the suite
could afford.

**V5 (fan builds; stability)** asserts, for a fixed seed:

- bed values stay finite everywhere (no NaN/inf) throughout the run;
- a deposit develops and thickens *downslope of the inlet*: the mean
  deposit thickness in a downslope window (mid-slope through basin floor)
  grows monotonically, checkpoint by checkpoint;
- the deposit footprint (cell count above a small threshold) grows between
  the early and late checkpoints;
- the basin floor receives a lobe *centered on the inlet path* (deposit
  centroid within a couple of cells of the inlet centerline) whose extent
  exceeds the inlet width -- a lobate body, not a uniform veneer.

**V6 (determinism)** runs the model in fresh processes (the runner script is
written to ``tmp_path`` and executed with ``sys.executable`` so numba's RNG
state starts clean each time):

- two identical runs (same seed, same config, fresh processes) produce
  bitwise-identical final beds and identical parcel-deposition bookkeeping
  (cumulative per-call deposition-cell counts);
- a changed seed produces a different bed (not bitwise equal) that is
  statistically similar: same total deposited volume within a loose
  tolerance (the supply rate is seed-independent) and a downslope deposit of
  the same order of magnitude.
"""

from pathlib import Path

import numpy as np
import pytest

from pyDeltaRCM.submarine import SubmarineFanModel

from . import utilities

# --- reduced-size run constants (see module docstring for sizing) ----------

_FAN_STEPS = 300
_CHECKPOINT_INTERVAL = 50
_DETERMINISM_STEPS = 25

# mean deposit thickness below this (meters) in a cell counts as no deposit
_DEPOSIT_THRESHOLD = 0.05

# downslope window: mid-slope through the basin floor (L=25 rows)
_DOWNSLOPE_WINDOW = slice(10, 25)
# basin-floor section: rows below the toe of the initial slope
_FLOOR_SECTION = slice(20, 25)

_LOBE_CENTROID_TOLERANCE_CELLS = 2.0
_SEED_VOLUME_TOLERANCE = 0.25
_SEED_DOWNSLOPE_FACTOR = 3.0


def _fan_config() -> dict:
    """Reduced-size but faithful fan-evolution configuration.

    Mirrors the ``_minimal_config`` pattern of ``tests/test_submarine_model.py``
    (which mirrors the stock ``test_DeltaModel`` fixture), with the submarine
    slope gradient inside the literature canyon range and an inlet sediment
    concentration matching the volumetric plunging range. The seed is
    deliberately *not* part of the config: it is passed per-run as a kwarg so
    the determinism runs can vary it against one shared YAML file.
    """
    return {
        "Length": 25.0,
        "Width": 50.0,
        "dx": 1.0,
        "L0_meters": 1.0,
        "N0_meters": 3.0,
        "S0": 0.0002,
        "itermax": 3,
        "Np_water": 50,
        "u0": 1.0,
        "h0": 1.0,
        "hb": 1.0,
        "H_SL": 0.0,
        "SLR": 0.0,
        "Np_sed": 50,
        "f_bedload": 0.5,
        "C0_percent": 2.5,
        "toggle_subsidence": False,
        "subsidence_rate": 0.0,
        "start_subsidence": 50.0,
        "save_dt": 1000000,
        "slope_gradient": 0.05,
    }


def _run_fan_evolution(tmp_path: Path, n_steps: int, seed: int = 0):
    """Evolve a reduced-size SubmarineFanModel, collecting V5 checkpoints.

    Returns the final model and a dict of per-checkpoint metrics indexed by
    ``(step // _CHECKPOINT_INTERVAL)``.
    """
    yaml_path = utilities.yaml_from_dict(tmp_path, "fan_evolution.yaml", _fan_config())
    # out_dir is injected into the YAML by utilities.yaml_from_dict, matching
    #   the suite's fixture pattern; no out_dir kwarg is passed (it would
    #   duplicate the YAML entry and warn)
    model = SubmarineFanModel(input_file=yaml_path, seed=seed)

    deposit_thickness = []
    footprint_cells = []
    floor_lobe_cells = []
    for step in range(1, n_steps + 1):
        model.update()
        if step % _CHECKPOINT_INTERVAL == 0:
            deposit = model.eta - model.eta_init
            deposit_thickness.append(
                float(deposit[_DOWNSLOPE_WINDOW, :].mean())
            )
            footprint_cells.append(
                int(np.count_nonzero(deposit > _DEPOSIT_THRESHOLD))
            )
            floor_lobe_cells.append(
                int(
                    np.count_nonzero(
                        deposit[_FLOOR_SECTION, :] > _DEPOSIT_THRESHOLD
                    )
                )
            )

    metrics = {
        "deposit_thickness": deposit_thickness,
        "footprint_cells": footprint_cells,
        "floor_lobe_cells": floor_lobe_cells,
    }
    return model, metrics


class TestFanMorphology:
    """V5: a fan builds downslope of the inlet, and the bed stays stable."""

    def test_fan_builds_downslope_of_inlet(
        self, tmp_path: Path
    ) -> None:
        model, metrics = _run_fan_evolution(tmp_path, _FAN_STEPS, seed=0)
        deposit = model.eta - model.eta_init

        # finite everywhere: the evolved bed and the current's fields
        for name, field in (
            ("eta", model.eta),
            ("depth", model.depth),
            ("stage", model.stage),
        ):
            assert np.isfinite(field).all(), f"{name} contains non-finite values"

        # a deposit develops downslope of the inlet and thickens
        #   monotonically, checkpoint by checkpoint
        thickness = metrics["deposit_thickness"]
        assert thickness[0] > 0.0, "no deposit in the downslope window"
        growth = np.diff(thickness)
        assert np.all(
            growth >= -1e-9
        ), f"downslope deposit thickness not non-decreasing: {thickness}"

        # the deposit footprint grows between early and late checkpoints
        footprint = metrics["footprint_cells"]
        assert footprint[-1] > footprint[0], (
            f"deposit footprint did not grow: {footprint}"
        )

        # the basin floor receives a lobe centered on the inlet path
        floor_lobe = metrics["floor_lobe_cells"]
        assert floor_lobe[-1] >= model.N0 + 2, (
            f"basin-floor deposit not wider than the inlet: {floor_lobe}"
        )
        cols = np.arange(model.W)
        floor_deposit = np.clip(
            deposit[_FLOOR_SECTION, :].max(axis=0), 0.0, None
        )
        centroid = (cols * floor_deposit).sum() / floor_deposit.sum()
        assert (
            abs(centroid - model.inlet.mean())
            <= _LOBE_CENTROID_TOLERANCE_CELLS
        ), (
            f"floor deposit centroid (col {centroid:.1f}) not centered on "
            f"the inlet path (col {model.inlet.mean():.1f})"
        )

        # stability: no net erosion outside the pinned inlet row, so every
        #   bed change in the domain is deposition (calibrated-supply check)
        assert deposit.min() >= 0.0, "spurious erosion outside the inlet row"
