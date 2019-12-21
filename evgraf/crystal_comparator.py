import itertools
import numpy as np
from evgrafcpp import calculate_rmsd
from evgraf.utils import standardize


class CrystalComparator:

    def __init__(self, atoms, subtract_barycenter=False):
        std = standardize(atoms, subtract_barycenter)
        self.atoms = std.atoms
        self.invop = std.invop
        self.zpermutation = std.zpermutation
        self.dim = sum(self.atoms.pbc)
        self.positions = self._wrapped_positions(self.atoms.get_positions())

        self.nbr_cells = self._get_neighboring_cells()
        self.offsets = self.nbr_cells @ atoms.cell
        if subtract_barycenter:
            self.barycenter = std.barycenter

    def _wrapped_positions(self, positions):
        scaled_positions = self.atoms.cell.scaled_positions(positions)
        for i in range(3):
            if self.atoms.pbc[i]:
                scaled_positions[:, i] %= 1.0
                scaled_positions[:, i] %= 1.0
        return self.atoms.cell.cartesian_positions(scaled_positions)

    def _get_neighboring_cells(self):
        pbc = self.atoms.pbc.astype(np.int)
        return np.array(list(itertools.product(*[range(-p, p + 1) for p in pbc])))

    def expand_coordinates(self, c):
        """Expands a 1D or 2D coordinate to 3D by filling in zeros where pbc=False.
        For inputs c=(1, 3) and pbc=[True, False, True] this returns
        array([1, 0, 3])"""
        count = 0
        expanded = []
        for i in range(3):
            if self.atoms.pbc[i]:
                expanded.append(c[count])
                count += 1
            else:
                expanded.append(0)
        return np.array(expanded)

    def calculate_rmsd(self, positions):
        positions = self._wrapped_positions(positions)
        return calculate_rmsd(positions, self.positions, self.offsets,
                              self.atoms.numbers.astype(np.int32))
