import numpy as np
from evgraf.utils import pbc2pbc, permute_axes


TOL = 1E-10


def standardize_layer(atoms):

    # standardize PBCs
    atoms = atoms.copy()
    atoms.set_pbc(pbc2pbc(atoms.pbc))
    assert np.sum(atoms.pbc) == 2

    # standardize periodic axis
    index = np.argmin(atoms.pbc)
    axis_permutation = np.roll([0, 1, 2], 2 - index)
    atoms = permute_axes(atoms, axis_permutation)
    assert (atoms.pbc == [1, 1, 0]).all()

    assert abs(atoms.cell[0, 2]) < TOL
    assert abs(atoms.cell[1, 2]) < TOL
    assert abs(atoms.cell[2, 0]) < TOL
    assert abs(atoms.cell[2, 1]) < TOL

    # standardize cell
    l = np.sqrt(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))
    cell = atoms.cell.copy()
    cell[2, 2] = l
    atoms.set_cell(cell, scale_atoms=False)
    return atoms, axis_permutation
