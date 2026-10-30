"""Exact post-promotion Python weighting contract, before native integration.

Parity domain: finite float32 grid/vector fields, float64 output, positive
D8 distances, nonnegative depth/modifiers, and feasible non-wall neighbors.
All four domain edges use edge padding. Row zero forbids upstream fallback
choices. The depth-power ufunc stays bulk NumPy to preserve its platform's
float32 power rounding; only the measured per-cell loop moves to Rust.
"""

import numpy as np
import pytest

from pyDeltaRCM import _native, water_tools


def inputs(case):
    rng = np.random.default_rng(42)
    shape = (6, 7)
    depth = rng.uniform(0.2, 2, shape).astype(np.float32)
    stage = rng.uniform(-0.2, 0.5, shape).astype(np.float32)
    modifier = rng.uniform(0.1, 2, shape).astype(np.float32)
    cells = np.zeros(shape, dtype=np.int64)
    qx = rng.uniform(-1, 1, shape).astype(np.float32)
    qy = rng.uniform(-1, 1, shape).astype(np.float32)
    distances = np.array([2**0.5, 1, 2**0.5, 1, 1, 1, 2**0.5, 1, 2**0.5], dtype=np.float32)
    ivec = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float32) / distances
    jvec = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float32) / distances
    if case in ("zero", "dry", "wall", "isolated"):
        stage.fill(0)
        qx.fill(0)
        qy.fill(0)
    if case == "dry":
        depth.fill(0)
    if case == "wall":
        cells[2:4, 2:4] = -2
    if case == "mixed":
        depth[::2, ::2] = 0
        cells[1:3, 1] = -2
    if case == "isolated":
        cells.fill(-2)
        cells[2, 2] = 0
    return [depth, stage, modifier, cells, qx, qy, ivec, jvec, distances, 0.1, 0.0014715, 1.0]


def native_weights(args):
    depth, stage, modifier, cells, qx, qy, ivec, jvec, distances, dry, gamma, theta = args
    return _native.water_weights(
        depth, stage, (depth * modifier) ** theta, cells, qx, qy,
        ivec, jvec, distances, dry, gamma,
    )


@pytest.mark.parametrize("case", ["random", "zero", "dry", "wall", "mixed"])
@pytest.mark.parametrize("theta", [0.0, 1.0, 2.0, 0.7])
def test_weight_array_is_bit_exact(case, theta):
    args = inputs(case)
    args[-1] = theta
    expected = water_tools._get_water_weight_array(*args)
    before = [arg.copy() for arg in args[:9]]
    actual = native_weights(args)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    for arg, saved in zip(args[:9], before):
        np.testing.assert_array_equal(arg, saved)


def test_zero_weight_boundary_fallback_contract():
    args = inputs("dry")
    expected = water_tools._get_water_weight_array(*args)
    np.testing.assert_array_equal(expected[0, 0, :3], [0, 0, 0])
    assert expected[0, 0, 4] == 0
    np.testing.assert_array_equal(expected[0, 0, [3, 5, 6, 7, 8]], np.full(5, np.float32(0.2)))
    np.testing.assert_array_equal(native_weights(args), expected)


def test_isolated_cell_raises_instead_of_inventing_a_direction():
    args = inputs("isolated")
    with pytest.raises(RuntimeError, match="No non-wall"):
        water_tools._get_water_weight_array(*args)
    with pytest.raises(RuntimeError, match="No non-wall"):
        native_weights(args)
