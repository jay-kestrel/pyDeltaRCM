"""Parity and dispatch tests for the COBOL weighted-pick kernel.

The ported routine is :func:`pyDeltaRCM.shared_tools.random_pick` -- the
per-parcel-step weighted direction pick. Parity is exact index equality
(integer output), stronger than the spec's rtol=1e-6 default. The
dispatch tests pin the ``PYDELTARCM_COBOL`` semantics: precedence over
the shared ``PYDELTARCM_NATIVE`` variable, and a missing shared object
raises (contract C2) rather than silently falling back.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numba import config as numba_config

from pyDeltaRCM import cobol_tools
from pyDeltaRCM.shared_tools import (
    get_random_state,
    random_pick,
    set_random_seed,
    set_random_state,
)

COBOL_LIB_BUILT = cobol_tools.library_path().exists()

requires_lib = pytest.mark.skipif(
    not COBOL_LIB_BUILT,
    reason="COBOL kernel not built (make -C cobol, requires GnuCOBOL)",
)


@pytest.fixture(autouse=True)
def _default_dispatch_off(monkeypatch):
    """Default behavior: no COBOL routing unless a test opts in."""
    monkeypatch.delenv("PYDELTARCM_COBOL", raising=False)


def _weight_rows(k, seed, scales=None):
    """Deterministic mixed-magnitude float32 weight rows (9 = D8 window)."""
    rng = np.random.default_rng(seed)
    if scales is None:
        scales = np.array(
            [1e-8, 1.0, 1e8, 0.5, 1e-12, 7.5, 1e-7, 3.3, 1e2], dtype=np.float32
        )
    return (rng.random((k, 9)) * scales).astype(np.float32)


def _uniforms_for(rows, seed):
    """Pre-drawn uniform(0, row_total) doubles for each row (C5 stream)."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(rows, axis=1)
    return np.array(
        [rng.uniform(0, cum[p, -1]) for p in range(rows.shape[0])],
        dtype=np.float64,
    )


def _benchmark_module():
    """Import benchmarks/benchmark_timestep.py as a module."""
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "benchmark_timestep.py"
    spec = importlib.util.spec_from_file_location("benchmark_timestep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReferenceFormulation:
    """The numpy reference backend must match the jitted runtime function."""

    def test_reference_backend_matches_jitted_random_pick(self, monkeypatch):
        """pick_rows (python backend) == shared_tools.random_pick, pick for
        pick, replaying the checkpointed MT19937 state between loops so
        both consume the identical draw sequence."""
        rows = _weight_rows(300, seed=11)
        rows = np.ascontiguousarray(
            rows / rows.sum(axis=1, keepdims=True), dtype=np.float32
        )

        set_random_seed(4242)
        state = get_random_state()
        njit_picks = [random_pick(row) for row in rows]

        set_random_state(state)
        monkeypatch.delenv("PYDELTARCM_COBOL", raising=False)
        wrap_picks = [cobol_tools.pick_rows(row) for row in rows]

        assert njit_picks == wrap_picks

    # This test replays a saved RNG state (get/set_random_state) and asserts
    # identical draws. Those helpers persist the numba RNG mirror, while the
    # no-JIT runtime draws come from numpy's legacy global stream -- the same
    # documented limitation as test_native_water.py's skipif. Mode-aware
    # pack/unpack is the real fix; until then this cannot pass without JIT.
    @pytest.mark.skipif(
        numba_config.DISABLE_JIT,
        reason="Existing checkpoint helpers persist the Numba RNG, not NumPy's no-JIT stream",
    )
    def test_self_drawn_stream_consumes_one_uniform_per_row(self):
        """With uniforms=None the wrapper draws uniform(0, row_total) from
        the checkpointed stream -- the same values, in the same order, as
        the jitted reference (re-validated stream equivalence)."""
        from pyDeltaRCM.shared_tools import get_random_uniform

        rows = np.ascontiguousarray(_weight_rows(40, seed=13) + 0.1, dtype=np.float32)

        set_random_seed(7)
        state = get_random_state()
        expected_draws = [
            get_random_uniform(float(np.cumsum(row)[-1])) for row in rows
        ]

        set_random_state(state)
        _, drawn = _capture_draws(cobol_tools.pick_rows, rows)
        assert drawn == expected_draws


def _capture_draws(func, rows):
    """Run func(rows) while recording the uniform(0, total) draws."""
    from pyDeltaRCM.shared_tools import get_random_uniform

    draws = []

    def recorder(limit):
        value = get_random_uniform(limit)
        draws.append(value)
        return value

    import pyDeltaRCM.cobol_tools as ct

    original = ct.get_random_uniform
    ct.get_random_uniform = recorder
    try:
        result = func(rows)
    finally:
        ct.get_random_uniform = original
    return result, draws


class TestCobolParity:
    """Exact index parity: COBOL kernel vs the numpy reference backend."""

    @requires_lib
    def test_cobol_matches_python_reference(self, monkeypatch):
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        rows = _weight_rows(4000, seed=0)
        uniforms = _uniforms_for(rows, seed=100)

        monkeypatch.setenv("PYDELTARCM_COBOL", "0")
        reference = cobol_tools.pick_rows(rows, uniforms=uniforms)
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        cobol = cobol_tools.pick_rows(rows, uniforms=uniforms)

        assert cobol.dtype == np.int32
        assert np.array_equal(cobol, reference)

    @requires_lib
    def test_cobol_boundary_uniforms(self, monkeypatch):
        """Uniforms at the scan boundaries: 0.0, just below total, and
        interior cutoff values (side='left' equality semantics)."""
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        rows = _weight_rows(1200, seed=3)
        cum = np.cumsum(rows, axis=1)

        uniforms = np.array(
            [
                [0.0, np.nextafter(cum[p, -1], 0.0), cum[p, 4]][p % 3]
                for p in range(1200)
            ],
            dtype=np.float64,
        )
        monkeypatch.setenv("PYDELTARCM_COBOL", "0")
        reference = cobol_tools.pick_rows(rows, uniforms=uniforms)
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        cobol = cobol_tools.pick_rows(rows, uniforms=uniforms)
        assert np.array_equal(cobol, reference)

    @requires_lib
    def test_cobol_single_cell_and_nonfinite_weights(self, monkeypatch):
        """n=1 rows and +inf weights (total=inf passes the >0 gate)."""
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        single = np.array([[0.5], [np.inf], [1e-30]], dtype=np.float32)
        single_rn = np.array([0.25, np.inf, 1e-31], dtype=np.float64)
        inf_row = np.array([[2.0, np.inf, 1.0]], dtype=np.float32)
        inf_rn = np.array([2.5], dtype=np.float64)

        monkeypatch.setenv("PYDELTARCM_COBOL", "0")
        reference = (
            cobol_tools.pick_rows(single, uniforms=single_rn),
            cobol_tools.pick_rows(inf_row, uniforms=inf_rn),
        )
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        cobol = (
            cobol_tools.pick_rows(single, uniforms=single_rn),
            cobol_tools.pick_rows(inf_row, uniforms=inf_rn),
        )
        assert np.array_equal(cobol[0], reference[0])
        assert np.array_equal(cobol[1], reference[1])

    @requires_lib
    def test_cobol_repeated_cutoffs_from_float32_collapse(self, monkeypatch):
        """Mixed scales where float32 accumulation collapses distinct
        cutoffs to equal values (1e8 + 0.5 rounds back to 1e8): the scan
        must pick the FIRST cutoff meeting the uniform, both backends."""
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        # weights: [1e8, 0.5, 1e8, 1e-30, ...] -> cum[0] == cum[2] in f32
        rows = np.tile(
            np.array([1e8, 0.5, 1e8, 1e-30, 1.0, 2.0, 3.0, 4.0, 5.0],
                     dtype=np.float32),
            (6, 1),
        )
        uniforms = np.array(
            [1e8, 1.0000001e8, 0.5, 1.5e8, 1e-20, 2e8], dtype=np.float64
        )

        monkeypatch.setenv("PYDELTARCM_COBOL", "0")
        reference = cobol_tools.pick_rows(rows, uniforms=uniforms)
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        cobol = cobol_tools.pick_rows(rows, uniforms=uniforms)
        assert np.array_equal(cobol, reference)
        # cum[0] == cum[2] (float32 rounding): a uniform at exactly 1e8
        # must land on the first of the equal cutoffs
        assert reference[0] == 0 and reference[2] == 0

    def test_cobol_error_parity_nonpositive_totals(self, monkeypatch):
        """ValueError parity: all-zero, NaN-total, and negative-total rows
        raise the reference's message on both backends."""
        bad = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, np.nan, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.float32,
        )
        uniforms = np.array([0.1, 0.1, 0.1], dtype=np.float64)

        monkeypatch.setenv("PYDELTARCM_COBOL", "0")
        with pytest.raises(ValueError, match="positive total probability"):
            cobol_tools.pick_rows(bad, uniforms=uniforms)

        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        with pytest.raises(ValueError, match="positive total probability"):
            cobol_tools.pick_rows(bad, uniforms=uniforms)

    @requires_lib
    def test_cobol_nan_uniform_index_error_parity(self, monkeypatch):
        """A NaN uniform exceeds no cutoff (unordered compare): both
        backends fall through to the out-of-bounds IndexError."""
        rows = np.array([[0.25, 0.5, 0.25]] * 2, dtype=np.float32)
        uniforms = np.array([np.nan, np.nan], dtype=np.float64)

        monkeypatch.setenv("PYDELTARCM_COBOL", "0")
        with pytest.raises(IndexError):
            cobol_tools.pick_rows(rows, uniforms=uniforms)

        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        with pytest.raises(IndexError):
            cobol_tools.pick_rows(rows, uniforms=uniforms)


class TestDispatch:
    """PYDELTARCM_COBOL semantics: precedence and no silent fallback."""

    def test_cobol_precedes_native(self, monkeypatch):
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
        assert cobol_tools.pick_backend() == "cobol"

    def test_native_only_selects_python(self, monkeypatch):
        monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
        assert cobol_tools.pick_backend() == "python"

    def test_unset_selects_python(self, monkeypatch):
        monkeypatch.delenv("PYDELTARCM_COBOL", raising=False)
        monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
        assert cobol_tools.pick_backend() == "python"

    def test_missing_library_raises(self, monkeypatch):
        """COBOL requested but not built: FileNotFoundError naming the
        build step -- never a silent fallback (contract C2)."""
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        monkeypatch.setattr(cobol_tools, "_lib_cache", None)
        monkeypatch.setattr(
            cobol_tools,
            "library_path",
            lambda: Path("/nonexistent/pydeltarcm_cobol.so"),
        )
        rows = np.array([[0.25, 0.5, 0.25]], dtype=np.float32)
        uniforms = np.array([0.5], dtype=np.float64)
        with pytest.raises(FileNotFoundError, match=r"make -C cobol"):
            cobol_tools.pick_rows(rows, uniforms=uniforms)

    def test_unrequested_missing_library_never_loaded(self, monkeypatch):
        """With routing off, no load is attempted (python path works and
        the adapter cache stays cold)."""
        monkeypatch.delenv("PYDELTARCM_COBOL", raising=False)
        monkeypatch.setattr(cobol_tools, "_lib_cache", None)
        monkeypatch.setattr(
            cobol_tools,
            "library_path",
            lambda: Path("/nonexistent/pydeltarcm_cobol.so"),
        )
        rows = np.array([[0.25, 0.5, 0.25]], dtype=np.float32)
        uniforms = np.array([0.5], dtype=np.float64)
        picks = cobol_tools.pick_rows(rows, uniforms=uniforms)
        assert np.array_equal(picks, np.array([1], dtype=np.int32))
        assert cobol_tools._lib_cache is None


class TestAdapterContract:
    """Dtype discipline, shapes, and input validation at the adapter."""

    @requires_lib
    def test_float64_input_canonicalized_to_float32(self, monkeypatch):
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        row64 = np.array([0.1, 0.7, 0.2, 0.5, 0.3, 0.9, 0.4, 0.6, 0.8])
        row32 = row64.astype(np.float32)
        uniforms = np.array([0.55], dtype=np.float64)
        from_f64 = cobol_tools.pick_rows(row64, uniforms=uniforms)
        from_f32 = cobol_tools.pick_rows(row32, uniforms=uniforms)
        assert from_f64 == from_f32

    @requires_lib
    def test_integer_weights_accepted(self, monkeypatch):
        """Inlet-style ones arrays (int) route through the same float32
        canonicalization."""
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        rows = np.ones((4, 9), dtype=np.int64)
        # cum = [1..9]: 0.0 and 0.99 land on the first cutoff, 8.5 and
        # 9.0 exactly land on the last
        uniforms = np.array([0.0, 0.99, 8.5, 9.0], dtype=np.float64)
        picks = cobol_tools.pick_rows(rows, uniforms=uniforms)
        assert np.array_equal(
            picks, np.array([0, 0, 8, 8], dtype=np.int32)
        )

    def test_single_row_returns_scalar(self):
        picks = cobol_tools.pick_rows(
            np.array([0.25, 0.5, 0.25]), uniforms=np.array([0.5])
        )
        assert isinstance(picks, int) and picks == 1

    def test_empty_batch(self):
        picks = cobol_tools.pick_rows(np.empty((0, 9), dtype=np.float32))
        assert picks.shape == (0,)

    def test_rejects_oversized_rows(self):
        with pytest.raises(ValueError, match="at most 64"):
            cobol_tools.pick_rows(np.ones((2, 65)))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            cobol_tools.pick_rows(np.ones((2, 3, 3)))

    def test_rejects_uniform_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            cobol_tools.pick_rows(
                np.ones((3, 9)), uniforms=np.ones(2)
            )


class TestHarnessKernelRow:
    """The kernel timing section of the landed two-regime harness."""

    @requires_lib
    def test_kernel_pick_bench_record(self, monkeypatch):
        bt = _benchmark_module()
        monkeypatch.delenv("PYDELTARCM_COBOL", raising=False)
        record = bt.run_kernel_pick_bench(smoke=True)

        assert record["workload"]["cols"] == 9
        assert record["workload"]["dtype"] == "float32"

        python = record["backends"]["python"]
        assert python["available"] is True
        assert python["implementation"] == "numpy"
        assert python["mean_us_per_pick"] > 0
        assert python["picks_per_s"] > 0

        cobol = record["backends"]["cobol"]
        assert cobol["available"] is True
        assert cobol["implementation"] == "cobol"
        assert cobol["gnucobol_version"]
        assert cobol["mean_us_per_pick"] > 0

    @requires_lib
    def test_kernel_pick_bench_requested_missing_raises(self, monkeypatch):
        """COBOL requested but not built: the harness aborts (C2) instead
        of emitting a cobol-less row."""
        bt = _benchmark_module()
        monkeypatch.setenv("PYDELTARCM_COBOL", "1")
        monkeypatch.setattr(cobol_tools, "_lib_cache", None)
        monkeypatch.setattr(
            cobol_tools,
            "library_path",
            lambda: Path("/nonexistent/pydeltarcm_cobol.so"),
        )
        with pytest.raises(FileNotFoundError, match=r"make -C cobol"):
            bt.run_kernel_pick_bench(smoke=True)
