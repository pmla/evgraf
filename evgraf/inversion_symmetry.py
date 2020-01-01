import itertools
import numpy as np
from collections import namedtuple
from ase import Atoms
from ase.geometry import find_mic
from .crystal_comparator import CrystalComparator


InversionSymmetry = namedtuple('InversionSymmetry',
                               'rmsd axis atoms permutation')


class CrystalInverter:

    def __init__(self, atoms):
        self.n = len(atoms)
        self.comparator = CrystalComparator(atoms, subtract_barycenter=True)

    def get_point(self, c):
        """Calculates the minimum-cost permutation at a desired translation.
        The translation is specified by `c` which describes the coordinates of
        the subgroup element."""
        c = self.comparator.expand_coordinates(c)
        shift = c / self.n @ self.comparator.atoms.cell
        positions = -self.comparator.positions + shift
        return self.comparator.calculate_rmsd(positions)


def find_inversion_axis(inverter):
    n = inverter.n
    r = list(range(n))
    dim = inverter.comparator.dim
    best = (float("inf"), None, None)
    for c in itertools.product(*([r] * dim)):
        rmsd, permutation = inverter.get_point(c)
        best = min(best, (rmsd, permutation, c), key=lambda x: x[0])
    rmsd, permutation, c = best
    c = inverter.comparator.expand_coordinates(c)
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
    n = len(atoms)
    inverter = CrystalInverter(atoms)
    rmsd, permutation, c = find_inversion_axis(inverter)

    comparator = inverter.comparator
    permutation = permutation[np.argsort(comparator.zpermutation)]
    permutation = comparator.zpermutation[permutation]

    axis = (c / n) @ comparator.atoms.cell / 2 + comparator.barycenter
    inverted = atoms[permutation]
    inverted.positions = -inverted.positions + 2 * axis
    inverted.wrap(eps=0)
    assert (inverted.numbers == atoms.numbers).all()

    symmetrized = symmetrized_layout(rmsd / 2, atoms, inverted)
    return InversionSymmetry(rmsd=rmsd / 2, axis=axis, atoms=symmetrized,
                             permutation=permutation)
