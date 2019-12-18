import numpy as np
from evgraf.utils import pbc2pbc, permute_axes


def standardize_chain(atoms):

    # standardize PBCs
    atoms = atoms.copy()
    atoms.set_pbc(pbc2pbc(atoms.pbc))
    assert np.sum(atoms.pbc) == 1

    # standardize periodic axis
    index = np.argmax(atoms.pbc)
    axis_permutation = np.roll([0, 1, 2], 2 - index)
    atoms = permute_axes(atoms, axis_permutation)
    assert (atoms.pbc == [0, 0, 1]).all()

    assert abs(atoms.cell[2, 0]) < 1E-10
    assert abs(atoms.cell[2, 1]) < 1E-10

    # standardize cell
    l = atoms.cell[2, 2]
    atoms.set_cell(l * np.eye(3), scale_atoms=False)
    return atoms, axis_permutation
