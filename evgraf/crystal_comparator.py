import itertools
import numpy as np
from evgrafcpp import wrap_positions, calculate_rmsd
from evgraf.utils import standardize


class CrystalComparator:

    def __init__(self, _atoms, subtract_barycenter=False):
        std = standardize(_atoms, subtract_barycenter)
        self.atoms = std.atoms
        self.op = std.op
        self.invop = std.invop
        self.zpermutation = std.zpermutation
        self.dim = sum(self.atoms.pbc)
        self.positions = wrap_positions(self.atoms.get_positions(),
                                        self.atoms.cell, self.atoms.pbc)

        self.nbr_cells = self._get_neighboring_cells()
        self.offsets = self.nbr_cells @ self.atoms.cell
        if subtract_barycenter:
            self.barycenter = std.barycenter

    def _get_neighboring_cells(self):
        pbc = self.atoms.pbc.astype(np.int)
        return np.array(list(itertools.product(*[range(-p, p + 1)
                                                 for p in pbc])))

    def expand_coordinates(self, c):
        """Expands a 1D or 2D coordinate to 3D by filling in zeros where
        pbc=False. For input c=(1, 3) and pbc=[True, False, True] this returns
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
        positions = wrap_positions(positions, self.atoms.cell, self.atoms.pbc)
        return calculate_rmsd(positions, self.positions, self.offsets,
                              self.atoms.numbers.astype(np.int32))
