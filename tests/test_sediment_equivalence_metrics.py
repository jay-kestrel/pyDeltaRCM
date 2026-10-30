"""Unit contracts for sediment acceptance metrics and fixed equivalence bands."""

import numpy as np
import pytest

from benchmarks.sediment_equivalence import bed_metrics, equivalence


def test_signed_volumes_and_rms():
    changes = np.array([[[1.0, -2.0]], [[3.0, -4.0]]])
    actual = bed_metrics(changes, np.array([[3.0, 4.0]]), 2.0)
    np.testing.assert_array_equal(actual[:2], [8.0, 12.0])
    assert actual[2] == np.sqrt(12.5)


def test_identical_ensembles_pass_and_large_bias_fails():
    reference = np.ones((15, 3))
    assert all(equivalence(reference, reference)["passes"])
    assert not any(equivalence(reference, reference * 2)["passes"])


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.full((15, 3), np.nan)])
def test_invalid_ensembles_are_rejected(bad):
    with pytest.raises(ValueError):
        equivalence(np.ones((15, 3)), bad)
