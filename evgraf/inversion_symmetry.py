import itertools
import numpy as np
from collections import namedtuple
from ase import Atoms
from ase.geometry import find_mic
from .crystal_reducer import CrystalReducer, expand_coordinates


def find_inversion_axis(cr):
    n = cr.n
    r = list(range(n))
    dim = sum(cr.atoms.pbc)
    best = (float("inf"), None, None)
    for c in itertools.product(*([r] * dim)):
        rmsd, permutation = cr.get_point(c)
        best = min(best, (rmsd, permutation, c), key=lambda x: x[0])
    rmsd, permutation, c = best
    c = expand_coordinates(c, cr.atoms.pbc)
    return rmsd, permutation, c


def symmetrized_layout(rmsd, atoms, inverted):
    ps = atoms.get_positions()
    v, _ = find_mic(inverted.get_positions() - ps, atoms.cell)
    meanpos = ps + v / 2
    component_rmsd = np.sqrt(np.sum((ps - meanpos)**2) / len(atoms))
    assert abs(rmsd - component_rmsd) < 1E-12

    symmetrized = Atoms(positions=meanpos, numbers=atoms.numbers,
                        cell=atoms.cell, pbc=atoms.pbc)
    symmetrized.set_cell(symmetrized.cell, scale_atoms=False)
    symmetrized.wrap(eps=0)
    return symmetrized


def find_inversion_symmetry(atoms):
    """Finds and quantifies inversion symmetry in a crystal using the
    root-mean-square (RMS) distance.

    The function finds the inversion axis and permutation which minimizes the
    RMS distance from the input structure to the symmetrized structure.

    Parameters:

    atoms: ASE atoms object
        The system to analyze.

    Returns:

    inversion: namedtuple with the following field names:
        rmsd: float
            RMS distance from input structure to reduced structure
        axis: ndarray of shape (3,)
            The inversion axis
        atoms: Atoms object
            The symmetrized structure
        permutation: integer ndarray
            Describes how atoms are paired to create the symmetrized structure
    """
    Inversion = namedtuple('InversionSymmetry', 'rmsd axis atoms permutation')

    n = len(atoms)
    cr = CrystalReducer(atoms, invert=True)
    rmsd, permutation, c = find_inversion_axis(cr)

    axis = (c / n) @ cr.atoms.cell / 2 + cr.barycenter
    perm = cr.zpermutation[permutation][np.argsort(cr.zpermutation)]
    inverted = atoms[perm]
    inverted.positions = -inverted.positions + 2 * axis
    inverted.wrap(eps=0)
    assert (inverted.numbers == atoms.numbers).all()

    symmetrized = symmetrized_layout(rmsd / 2, atoms, inverted)
    return Inversion(rmsd=rmsd / 2, axis=axis, atoms=symmetrized,
                     permutation=perm)
