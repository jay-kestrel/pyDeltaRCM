
"""Checkpoint round-trip and RNG-determinism tests for SubmarineFanModel.

Extends the ``tests/integration/test_checkpointing.py`` pattern to
:obj:`~pyDeltaRCM.submarine.SubmarineFanModel`. v1 of the submarine module
adds no new stateful arrays (``h_current``/``gamma_density`` are derived
scalars recomputed from the configuration at construction, and
``water_weights`` is rebuilt on every routing iteration), so the stock
checkpoint save list must suffice. The tests here verify exactly that, by
requiring a checkpoint save -> load -> continue run to reproduce a
no-restart run exactly.

Three angles on the same invariant:

- a real-routing round trip (no solver patch): a reference run, the same
  run interrupted at a manual checkpoint save, and a run resumed from that
  checkpoint must all agree exactly at the common final time. Any model
  state the checkpoint fails to save, or any random draw taken outside the
  shared numba stream, desynchronizes the resumed run and fails these
  assertions;
- the stock fast-solver stream test: a resumed model must continue the
  exact random state saved with the checkpoint, verified immediately after
  construction (before any routing consumes draws);
- a source-level pin: the submarine package draws randomness only through
  ``pyDeltaRCM.shared_tools.get_random_uniform`` -- the numba stream the
  checkpoint machinery saves and restores. Draws taken from
  ``numpy.random`` or the stdlib ``random`` module are invisible to that
  machinery and would silently break resume reproducibility.
"""

import ast
import os
import unittest.mock as mock
from pathlib import Path

import numpy as np

import pyDeltaRCM.submarine as submarine_pkg
from pyDeltaRCM.submarine import SubmarineFanModel
from pyDeltaRCM.shared_tools import get_random_state, get_random_uniform

from .. import utilities


# run sizing mirrors tests/test_fan_evolution.py (faithful 25x50 domain,
# Np=50) but with far fewer timesteps: checkpointing is under test here,
# not morphology, and the round trip evolves the domain twice (reference
# run plus interrupted/resumed run)
_N_TOTAL_STEPS = 20
_N_SPLIT_STEPS = 8

# aliases under which numpy may be addressed in the package sources
_NUMPY_ALIASES = {"np", "numpy"}
# stdlib modules whose import means a draw outside the shared numba stream
_STDLIB_RANDOM_MODULES = {"random"}


def _fan_checkpoint_config() -> dict:
    """Reduced-size fan configuration, mirroring tests/test_fan_evolution.py.

    Same faithful domain and submarine parameters as the V5/V6 suite, with
    the seed fixed in-config (all three runs of the round trip must seed
    identically) and far fewer timesteps: checkpointing is under test
    here, not morphology, and the round trip evolves the domain twice.
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
        "seed": 0,
        "toggle_subsidence": False,
        "subsidence_rate": 0.0,
        "start_subsidence": 50.0,
        "save_dt": 1000000,
        "slope_gradient": 0.05,
    }


def _tiny_config() -> dict:
    """Tiny-model configuration for the fast-solver stream test.

    Mirrors the ``_minimal_config`` pattern of ``tests/test_submarine_model.py``
    (which mirrors the stock ``test_DeltaModel`` fixture).
    """
    return {
        "Length": 10.0,
        "Width": 10.0,
        "seed": 0,
        "dx": 1.0,
        "L0_meters": 1.0,
        "S0": 0.0002,
        "itermax": 1,
        "Np_water": 10,
        "u0": 1.0,
        "N0_meters": 2.0,
        "h0": 1.0,
        "H_SL": 0.0,
        "SLR": 0.0,
        "Np_sed": 10,
        "f_bedload": 0.5,
        "C0_percent": 2.5,
        "toggle_subsidence": False,
        "subsidence_rate": 0.0,
        "start_subsidence": 50.0,
        "save_dt": 500,
    }


class TestSubmarineCheckpointRoundTrip:
    """V8: checkpoint save -> load -> continue reproduces a no-restart run.

    Runs the real routing stack (no solver patch): the bed-following
    weight kernel, the stock sand/mud routers, and the shared numba RNG
    all participate, exactly as in a user-facing run.
    """

    def test_round_trip_reproduces_no_restart_run(self, tmp_path: Path) -> None:
        """A resumed run matches an uninterrupted run at the same time."""
        # no-restart reference run, snapshotted at the split step
        ref_cfg = dict(_fan_checkpoint_config())
        ref_cfg["out_dir"] = tmp_path / "reference"
        ref_yaml = utilities.yaml_from_dict(tmp_path, "reference.yaml", ref_cfg)
        reference = SubmarineFanModel(input_file=ref_yaml)
        for _ in range(_N_SPLIT_STEPS):
            reference.update()
        ref_mid_eta = reference.eta.copy()
        ref_mid_stage = reference.stage.copy()
        ref_mid_depth = reference.depth.copy()
        ref_mid_uw = reference.uw.copy()
        for _ in range(_N_TOTAL_STEPS - _N_SPLIT_STEPS):
            reference.update()
        reference.finalize()

        # interrupted run: identical config and seed, checkpointed at the
        #   split step with the stock save machinery
        prod_cfg = dict(_fan_checkpoint_config())
        prod_cfg["out_dir"] = tmp_path / "producer"
        prod_yaml = utilities.yaml_from_dict(tmp_path, "producer.yaml", prod_cfg)
        producer = SubmarineFanModel(input_file=prod_yaml)
        for _ in range(_N_SPLIT_STEPS):
            producer.update()
        producer.save_the_checkpoint()
        producer.finalize()

        # the interrupted run matches the reference up to the split step
        assert producer._time < reference._time
        assert np.array_equal(producer.eta, ref_mid_eta)
        assert np.array_equal(producer.stage, ref_mid_stage)
        assert np.array_equal(producer.depth, ref_mid_depth)
        assert np.array_equal(producer.uw, ref_mid_uw)

        # resumed run: load the producer's checkpoint, then continue to
        #   the reference's final time
        resume_cfg = dict(_fan_checkpoint_config())
        resume_cfg["out_dir"] = tmp_path / "consumer"
        resume_cfg["resume_checkpoint"] = str(producer.prefix)
        resume_yaml = utilities.yaml_from_dict(tmp_path, "resume.yaml", resume_cfg)
        consumer = SubmarineFanModel(input_file=resume_yaml)

        # the resumed model starts exactly at the checkpoint
        assert consumer._time == producer._time

        # advance the resumed model until it catches up with the reference
        while consumer._time < reference._time:
            consumer.update()
        consumer.finalize()

        # the round trip reproduces the no-restart run exactly
        assert consumer._time == reference._time
        assert np.all(consumer.eta == reference.eta)
        assert np.all(consumer.uw == reference.uw)
        assert np.all(consumer.ux == reference.ux)
        assert np.all(consumer.uy == reference.uy)
        assert np.all(consumer.depth == reference.depth)
        assert np.all(consumer.stage == reference.stage)
        assert np.all(consumer.sand_frac == reference.sand_frac)
        assert np.all(consumer.active_layer == reference.active_layer)

        # non-vacuity: the resumed segment actually moved the bed away
        #   from the checkpointed state, and the run deposited somewhere
        with np.load(
            os.path.join(producer.prefix, "checkpoint.npz"), allow_pickle=False
        ) as ckp:
            ckp_eta = ckp["eta"]
        assert not np.array_equal(consumer.eta, ckp_eta)
        assert not np.array_equal(reference.eta, reference.eta_init)


@mock.patch(
    "pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep",
    new=utilities.FastIteratingDeltaModel.solve_water_and_sediment_timestep,
)
class TestSubmarineCheckpointRNGStream:
    """
    The above patch implements an augmented DeltaModel from `utilities` (see
    the stock checkpointing tests): the time-consuming routing is replaced
    with an updating random field, so the random-repeatedness of
    checkpointing is validated in isolation and fast.
    """

    def test_checkpoint_round_trips_rng_stream(self, tmp_path: Path) -> None:
        """Test a resumed model continues the exact saved random state.

        Mirrors the stock ``test_checkpoint_round_trips_rng_stream`` for
        the SubmarineFanModel: the checkpoint carries the shared numba
        stream as plain arrays (no pickle), and the resumed model draws
        the exact continuation of the saved sequence.
        """
        # producer model writes a checkpoint after a few fast updates
        prod_cfg = dict(_tiny_config())
        prod_cfg["out_dir"] = tmp_path / "producer"
        prod_yaml = utilities.yaml_from_dict(tmp_path, "rng_producer.yaml", prod_cfg)
        producer = SubmarineFanModel(input_file=prod_yaml)

        for _ in range(3):
            producer.update()
        producer.save_the_checkpoint()
        saved_state = get_random_state()
        producer_draws = [get_random_uniform(1.0) for _ in range(4)]
        producer.finalize()

        # the written checkpoint contains no pickled object arrays:
        #   loading it with allow_pickle=False must succeed for every field
        ckp_file = os.path.join(producer.prefix, "checkpoint.npz")
        with np.load(ckp_file, allow_pickle=False) as checkpoint:
            for key in checkpoint.files:
                _ = checkpoint[key]

        # consumer model resumes from the checkpoint folder
        cons_cfg = dict(_tiny_config())
        cons_cfg["out_dir"] = tmp_path / "consumer"
        cons_cfg["resume_checkpoint"] = str(producer.prefix)
        cons_yaml = utilities.yaml_from_dict(tmp_path, "rng_consumer.yaml", cons_cfg)
        consumer = SubmarineFanModel(input_file=cons_yaml)

        # the resumed model continues the exact random stream
        assert get_random_state() == saved_state
        consumer_draws = [get_random_uniform(1.0) for _ in range(4)]
        assert consumer_draws == producer_draws
        consumer.finalize()


class TestSubmarineRNGDiscipline:
    """All randomness in the submarine package flows through the shared RNG.

    Draws taken outside ``shared_tools.get_random_uniform`` -- the numba
    stream that checkpoints save and restore -- are invisible to the
    checkpoint machinery and silently break resume reproducibility. This
    source-level pin makes that requirement explicit for the package.

    The check is AST-level: it flags ``np.random``/``numpy.random``
    attribute accesses, ``import random`` and ``from random import ...``,
    and ``from numpy import random``. It does not chase string-based
    evasion (e.g., ``getattr(np, "random")``).
    """

    def test_no_direct_random_usage_in_submarine_package(self) -> None:
        """Fail if any submarine module draws randomness outside shared_tools."""
        pkg_dir = Path(submarine_pkg.__file__).resolve().parent
        module_files = sorted(pkg_dir.glob("*.py"))
        # the package ships the model, the kernel, and the package init
        assert len(module_files) >= 3

        violations = []
        for path in module_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in _STDLIB_RANDOM_MODULES:
                            violations.append(
                                f"{path.name}:{node.lineno}: import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in _STDLIB_RANDOM_MODULES:
                        violations.append(
                            f"{path.name}:{node.lineno}: from {node.module} import ..."
                        )
                    elif root in _NUMPY_ALIASES and any(
                        alias.name == "random" for alias in node.names
                    ):
                        violations.append(
                            f"{path.name}:{node.lineno}: from {node.module} import random"
                        )
                elif (
                    isinstance(node, ast.Attribute)
                    and node.attr == "random"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in _NUMPY_ALIASES
                ):
                    violations.append(
                        f"{path.name}:{node.lineno}: {node.value.id}.random"
                    )

        assert not violations, (
            "direct random usage found; route all draws through "
            f"pyDeltaRCM.shared_tools.get_random_uniform: {violations}"
        )
