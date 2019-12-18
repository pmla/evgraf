import math
import functools
import itertools
import numpy as np
from .subgroup_enumeration import (enumerate_subgroup_bases,
                                   get_subgroup_elements)
from evgrafcpp import calculate_rmsd
from evgraf.utils import standardize


def reduce_gcd(x):
    return functools.reduce(math.gcd, x)


def get_neighboring_cells(pbc):
    pbc = pbc.astype(np.int)
    return np.array(list(itertools.product(*[range(-p, p + 1) for p in pbc])))


def expand_coordinates(c, pbc):
    """Expands a 1D or 2D coordinate to 3D by filling in zeros where pbc=False.
    For inputs c=(1, 3) and pbc=[True, False, True] this returns
    array([1, 0, 3])"""
    count = 0
    expanded = []
    for i in range(3):
        if pbc[i]:
            expanded.append(c[count])
            count += 1
        else:
            expanded.append(0)
    return np.array(expanded)


class CrystalReducer:

    # TODO: remove explicit `invert` parameter. use barycenter instead.
    def __init__(self, atoms, invert=False):
        self.invert = invert
        self.distances = {}
        self.permutations = {}
        atoms, invop, barycenter, zpermutation = standardize(atoms,
                                                             self.invert)
        self.atoms = atoms
        self.invop = invop
        self.zpermutation = zpermutation
        self.scaled_positions = atoms.get_scaled_positions()

        self.positions = atoms.get_positions()
        self.nbr_cells = get_neighboring_cells(atoms.pbc)
        self.offsets = self.nbr_cells @ atoms.cell
        if invert:
            self.n = len(atoms)
            self.barycenter = barycenter
        else:
            self.n = reduce_gcd(atoms.symbols.formula.count().values())

    def calculate_rmsd(self, positions):
        return calculate_rmsd(positions, self.positions, self.offsets,
                              self.atoms.numbers.astype(np.int32))

    # TODO: split inversion features into separate module. remove explicit `invert` parameter.
    def get_point(self, c):
        """Calculates the minimum-cost permutation at a desired translation.
        The translation is specified by `c` which describes the coordinates of
        the subgroup element."""
        key = tuple(c)
        if key in self.permutations:
            return self.distances[key], self.permutations[key]

        pbc = self.atoms.pbc
        c = expand_coordinates(c, pbc)

        sign = [1, -1][self.invert]
        transformed = sign * self.scaled_positions + c / self.n
        for i, index in enumerate(c):
            if pbc[i]:
                transformed[:, i] %= 1.0
        positions = self.atoms.cell.cartesian_positions(transformed)

        rmsd, permutation = self.calculate_rmsd(positions)
        if not self.invert:
            self.distances[key] = rmsd
            self.permutations[key] = permutation
        return rmsd, permutation

    def is_consistent(self, H):
        """Callback function which tests whether the minimum-cost permutations
        of the subgroup elements are consistent with each other. H describes
        the subgroup basis."""
        n = self.n
        dims = [n] * sum(self.atoms.pbc)
        num_atoms = len(self.scaled_positions)

        seen = -np.ones((3, num_atoms), dtype=int)
        elements = get_subgroup_elements(dims, H)

        for c1 in elements:
            _, perm1 = self.get_point(c1)
            invperm1 = np.argsort(perm1)
            for i, c2 in enumerate((c1 + H) % n):
                _, perm2 = self.get_point(c2)
                val = perm2[invperm1]
                if seen[i][0] == -1:
                    seen[i] = val
                elif (seen[i] != val).any():
                    return False
        return True

    def find_consistent_reductions(self):
        n = self.n
        dims = [n] * sum(self.atoms.pbc)
        for H in enumerate_subgroup_bases(dims, self.is_consistent,
                                          min_index=1, max_index=n):
            group_index = np.prod(dims // np.diag(H))
            elements = get_subgroup_elements(dims, H)
            distances = np.array([self.distances[tuple(c)] for c in elements])
            permutations = np.array([self.permutations[tuple(c)]
                                     for c in elements])
            rmsd = np.sqrt(np.sum(distances**2) / (2 * group_index))
            yield (rmsd, group_index, H, permutations)
