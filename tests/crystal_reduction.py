import pytest
import itertools
import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms
from ase.geometry import find_mic
from ase.build import bulk, mx2, nanotube, make_supercell, graphene
from evgraf.utils import permute_axes
from evgraf.crystal_comparator import CrystalComparator
from evgraf import find_crystal_reductions


TOL = 1E-10


def check_components(atoms, result, tol=TOL):
    for reduced in result:
        assert (atoms.pbc == reduced.atoms.pbc).all()
        assert (np.bincount(reduced.components) == reduced.factor).all()

        x = atoms.numbers[np.argsort(reduced.components)]
        x = x.reshape((len(atoms) // reduced.factor, reduced.factor))
        assert (x == x[:, 0][:, None]).all()
        assert (x[:, 0] == reduced.atoms.numbers).all()
        assert(atoms.numbers == reduced.atoms.numbers[reduced.components]).all()

        # check supercell is correct
        supercell = make_supercell(reduced.atoms, np.linalg.inv(reduced.map))
        assert (supercell.pbc == atoms.pbc).all()
        assert_allclose(supercell.cell, atoms.cell, atol=tol)

        # check rmsd is correct
        comparator = CrystalComparator(atoms)
        indices = np.argsort(supercell.numbers, kind='merge')
        supercell = supercell[indices]
        supercell.wrap(eps=0)
        rmsd, permutation = comparator.calculate_rmsd(supercell.get_positions())
        assert_allclose(rmsd, reduced.rmsd, atol=tol)

        # check components are correct
        indices = np.argsort(reduced.components)
        check = atoms[indices]
        components = reduced.components[indices]
        check.set_cell(reduced.atoms.cell, scale_atoms=False)
        check.wrap(eps=0)

        ps = check.get_positions()
        parents = components * reduced.factor
        vmin, _ = find_mic(ps - ps[parents], check.cell, pbc=check.pbc)
        positions = ps[parents] + vmin

        m = len(atoms) // reduced.factor
        meanpos = np.mean(positions.reshape((m, reduced.factor, 3)), axis=1)
        rmsd_check = np.sqrt(np.mean((positions - meanpos[components])**2))
        assert_allclose(reduced.rmsd, rmsd_check, atol=tol)


def randomize(rng, atoms):
    # shuffle the atoms
    indices = np.arange(len(atoms))
    rng.shuffle(indices)
    atoms = atoms[indices]

    # apply a unimodular transformation to the unit cell
    L = np.eye(3, dtype=int)
    indices = np.where(atoms.pbc)[0]
    for i, j in itertools.combinations(indices, 2):
        if i != j:
            assert i < j
            L[i, j] = rng.randint(-2, 3)

    assert_allclose(np.linalg.det(L), 1, atol=TOL)
    atoms.set_cell(L.T @ atoms.cell, scale_atoms=False)
    atoms.wrap(eps=0)
    return atoms


# 3-dimensional: NaCl
@pytest.mark.parametrize("seed", range(2))
def test_nacl(seed):
    size = 3
    atoms = bulk("NaCl", "rocksalt", a=5.64) * size
    rng = np.random.RandomState(seed=seed)
    atoms = randomize(rng, atoms)

    result = find_crystal_reductions(atoms)
    assert len(result) == size + 1
    assert all([reduced.rmsd < TOL for reduced in result])
    factors = [reduced.factor for reduced in result]
    assert tuple(factors) == (1, 3, 9, 27)
    check_components(atoms, result)


@pytest.mark.parametrize("seed", range(2))
def test_nacl_rattled(seed):
    stdev = 0.1
    size = 3
    atoms = bulk("NaCl", "rocksalt", a=5.64) * size
    atoms.rattle(seed=seed, stdev=stdev)

    rng = np.random.RandomState(seed=seed)
    atoms = randomize(rng, atoms)

    result = find_crystal_reductions(atoms)
    check_components(atoms, result, tol=3 * stdev)


# 2-dimensional: MoS2
@pytest.mark.parametrize("i", range(3))
@pytest.mark.parametrize("seed", range(3))
def test_mos2(seed, i):
    size = 4
    atoms = mx2(formula='MoS2', size=(size, size, 1))
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)

    rng = np.random.RandomState(seed=seed)
    atoms = randomize(rng, atoms)

    result = find_crystal_reductions(atoms)
    assert len(result) == size + 1
    assert all([reduced.rmsd < TOL for reduced in result])
    factors = [reduced.factor for reduced in result]
    assert tuple(factors) == (1, 2, 4, 8, 16)
    check_components(atoms, result)


def test_graphene():
    size = (2, 2, 1)
    atoms = graphene(size=size)
    atoms.positions += (-0.1, 0.6, 0)
    atoms.rattle(seed=0, stdev=0.1)
    atoms.wrap()

    result = find_crystal_reductions(atoms)
    assert len(result) > 1


# 1-dimensional: carbon nanotube
@pytest.mark.parametrize("i", range(3))
@pytest.mark.parametrize("seed", range(3))
def test_nanotube(seed, i):
    size = 4
    atoms = nanotube(3, 3, length=size)
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)

    rng = np.random.RandomState(seed=seed)
    atoms = randomize(rng, atoms)

    result = find_crystal_reductions(atoms)[:3]
    factors = [reduced.factor for reduced in result]
    assert tuple(factors) == (1, 2, 4)
    assert all([reduced.rmsd < TOL for reduced in result])
    check_components(atoms, result)


@pytest.mark.parametrize("n", np.arange(1, 16))
def test_line(n):
    positions = np.zeros((n, 3))
    positions[:, 0] = np.arange(n)
    atoms = Atoms(positions=positions, cell=np.diag([n, 0, 0]),
                  pbc=[1, 0, 0], numbers=10 * np.ones(n))

    result = find_crystal_reductions(atoms)
    check_components(atoms, result)
