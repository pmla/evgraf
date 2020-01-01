import pytest
import numpy as np
from numpy.testing import assert_allclose
from evgraf.utils import standardize_cell


TOL = 1E-10


@pytest.mark.parametrize("seed", range(40))
def test_cell_standardization(seed):
    rng = np.random.RandomState(seed=seed)
    cell = rng.uniform(-1, 1, (3, 3))
    Q, R = standardize_cell(cell)
    assert np.linalg.det(Q) > 0
    assert_allclose(cell, R @ Q.T, atol=TOL)
