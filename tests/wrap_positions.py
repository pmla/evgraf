import pytest
import numpy as np
from numpy.testing import assert_allclose
from ase.build import bulk, mx2, nanotube
from evgraf.utils import permute_axes, standardize
from evgrafcpp import wrap_positions


TOL = 1E-10


def check_result(atoms, positions, wrapped, translate):
    indices = np.where(atoms.pbc)
    scaled = atoms.cell.scaled_positions(wrapped)

    if translate:
        assert (scaled[:, indices] >= 0).all()
        assert (scaled[:, indices] < 1).all()
        assert_allclose(positions, wrapped, atol=TOL)
    else:
        assert (scaled[:, indices] >= -TOL).all()
        assert (scaled[:, indices] < 1 + TOL).all()


def prepare(seed, i, translate, atoms):
    rng = np.random.RandomState(seed=seed)

    atoms = standardize(atoms).atoms
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)
    if translate:
        atoms.positions += rng.uniform(-1, 1, atoms.positions.shape)
        atoms.wrap(eps=0)

    positions = atoms.get_positions(wrap=False)
    offset = rng.randint(-5, 6, positions.shape)
    for i in range(3):
        if not atoms.pbc[i]:
            offset[:, i] = 0
    scattered = positions.copy() + offset @ atoms.cell

    wrapped = wrap_positions(scattered, atoms.cell, atoms.pbc)
    check_result(atoms, positions, wrapped, translate)


# 3-dimensional: NaCl
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("translate", [False, True])
def test_nacl(seed, translate):
    size = 2
    atoms = bulk("NaCl", "rocksalt", a=5.64) * size
    prepare(seed, 0, translate, atoms)


# 2-dimensional: MoS2
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("i", range(3))
@pytest.mark.parametrize("translate", [False, True])
def test_mos2(seed, i, translate):
    size = 2
    atoms = mx2(formula='MoS2', size=(size, size, 1))
    prepare(seed, i, translate, atoms)


# 1-dimensional: carbon nanotube
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("i", range(3))
@pytest.mark.parametrize("translate", [False, True])
def test_nanotube(seed, i, translate):
    size = 4
    atoms = nanotube(3, 3, length=size)
    prepare(seed, i, translate, atoms)
