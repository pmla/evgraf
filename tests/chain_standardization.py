import pytest
import numpy as np
from ase.build import nanotube
from evgraf.permute_axes import permute_axes
from evgraf.chains.chain_standardization import standardize_chain


@pytest.mark.parametrize("i", range(3))
def test_standardize_chain(i):
    size = 4
    atoms = nanotube(3, 3, length=size)
    permutation = np.roll(np.arange(3), i)
    permuted = permute_axes(atoms, permutation)

    standardized, axis_permutation = standardize_chain(permuted)
    assert (standardized.pbc == [False, False, True]).all()
    assert (permuted.pbc[axis_permutation] == standardized.pbc).all()
