import pytest
import itertools
import numpy as np
from ase.build import nanotube
from evgraf.utils import permute_axes
from evgraf.chains.chain_alignment import calculate_rmsd_chain


TOL = 1E-10


def randomize(atoms, reflect=[1, 1], rotate=False, seed=None):

    # permute atoms randomly
    rng = np.random.RandomState(seed=seed)
    indices = rng.permutation(len(atoms))
    atoms = atoms[indices]

    # apply a random translation
    positions = atoms.get_positions()
    positions += rng.uniform(0, 10, 3)

    rx, rz = reflect
    positions[:, 0] *= rx
    positions[:, 1] *= rx
    positions[:, 2] *= rz

    if rotate:
        theta = rng.uniform(0, 2 * np.pi)
        sint = np.sin(theta)
        cost = np.cos(theta)
        U = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
        positions = np.dot(positions, U.T)

    atoms.set_positions(positions)
    atoms.wrap(eps=0)
    return atoms


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("reflect", itertools.product([-1, 1], repeat=2))
@pytest.mark.parametrize("i", range(3))
def test_nanotube(i, reflect, seed):
    size = 2
    atoms = nanotube(3, 3, length=size)
    randomized = randomize(atoms, reflect=reflect, rotate=True, seed=seed)

    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)
    randomized = permute_axes(randomized, permutation)

    rmsd = calculate_rmsd_chain(atoms, randomized, ignore_stoichiometry=False,
                                allow_reflection=True, allow_rotation=True)

    assert rmsd < TOL
