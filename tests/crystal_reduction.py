import pytest
import numpy as np
from ase.build import bulk, mx2, nanotube
from evgraf.utils import permute_axes
from evgraf import find_crystal_reductions


TOL = 1E-10


def check_components(atoms, result):
    for reduced in result:
        assert (np.bincount(reduced.components) == reduced.factor).all()
        x = atoms.numbers[np.argsort(reduced.components)]
        x = x.reshape((len(atoms) // reduced.factor, reduced.factor))
        assert (x - x[:, 0][:, None] == 0).all()


# 3-dimensional: NaCl
def test_nacl():
    size = 3
    atoms = bulk("NaCl", "rocksalt", a=5.64) * size
    result = find_crystal_reductions(atoms)
    assert len(result) == size
    assert all([reduced.rmsd < TOL for reduced in result])
    factors = [reduced.factor for reduced in result]
    assert tuple(factors) == (3, 9, 27)
    check_components(atoms, result)


# 2-dimensional: MoS2
@pytest.mark.parametrize("i", range(3))
def test_mos2(i):
    size = 4
    atoms = mx2(formula='MoS2', size=(size, size, 1))
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)

    result = find_crystal_reductions(atoms)
    assert len(result) == size
    assert all([reduced.rmsd < TOL for reduced in result])
    factors = [reduced.factor for reduced in result]
    assert tuple(factors) == (2, 4, 8, 16)
    check_components(atoms, result)


# 1-dimensional: carbon nanotube
@pytest.mark.parametrize("i", range(3))
def test_nanotube(i):
    size = 4
    atoms = nanotube(3, 3, length=size)
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)

    result = find_crystal_reductions(atoms)
    factors = [reduced.factor for reduced in result[:2]]
    assert tuple(factors) == (2, 4)
    assert all([reduced.rmsd < TOL for reduced in result[:2]])
    check_components(atoms, result)
