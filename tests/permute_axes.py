import pytest
import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms
from evgraf.geometry import permute_axes


TOL = 1E-10


@pytest.mark.parametrize("seed", range(20))
def test_permute_axes(seed):
    rng = np.random.RandomState(seed)
    n = 10
    atoms = Atoms(numbers=[1] * n,
                  scaled_positions=rng.uniform(0, 1, (n, 3)),
                  pbc=rng.randint(0, 2, 3),
                  cell=rng.uniform(-1, 1, (3, 3)))

    permutation = rng.permutation(3)
    permuted = permute_axes(atoms, permutation)
    invperm = np.argsort(permutation)
    original = permute_axes(permuted, invperm)

    assert (original.pbc == atoms.pbc).all()
    assert_allclose(original.cell, atoms.cell, atol=TOL)
    assert_allclose(original.get_positions(), atoms.get_positions(), atol=TOL)
    assert_allclose(atoms.get_positions()[:, permutation],
                    permuted.get_positions(), atol=TOL)

    assert_allclose(atoms.cell.volume, permuted.cell.volume, atol=TOL)
    assert_allclose(atoms.cell.volume, original.cell.volume, atol=TOL)
    assert (permuted.pbc == atoms.pbc[permutation]).all()
