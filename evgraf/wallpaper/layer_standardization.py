import numpy as np
from evgraf.utils import permute_axes


TOL = 1E-10


def standardize_layer(atoms):

    assert np.sum(atoms.pbc) == 2

    # standardize periodic axis
    index = np.argmin(atoms.pbc)
    axis_permutation = np.roll([0, 1, 2], 2 - index)
    atoms = permute_axes(atoms, axis_permutation)
    assert (atoms.pbc == [1, 1, 0]).all()

    cell = atoms.cell
    assert abs(cell[0, 2]) < TOL
    assert abs(cell[1, 2]) < TOL
    assert abs(cell[2, 0]) < TOL
    assert abs(cell[2, 1]) < TOL

    # standardize cell
    l = np.sqrt(np.linalg.norm(np.cross(cell[0], cell[1])))
    cell[2, 2] = l
    atoms.set_cell(cell, scale_atoms=False)
    return atoms, axis_permutation
