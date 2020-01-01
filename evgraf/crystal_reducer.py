import math
import functools
import numpy as np
from .crystal_comparator import CrystalComparator
from .subgroup_enumeration import (enumerate_subgroup_bases,
                                   get_subgroup_elements)


def reduce_gcd(x):
    return functools.reduce(math.gcd, x)


class CrystalReducer:

    def __init__(self, atoms):
        self.distances = {}
        self.permutations = {}
        self.n = reduce_gcd(atoms.symbols.formula.count().values())
        self.comparator = CrystalComparator(atoms)

    def get_point(self, c):
        """Calculates the minimum-cost permutation at a desired translation.
        The translation is specified by `c` which describes the coordinates of
        the subgroup element."""
        key = tuple(c)
        if key in self.permutations:
            return self.distances[key], self.permutations[key]

        comparator = self.comparator
        c = comparator.expand_coordinates(c)
        positions = comparator.positions + c @ comparator.atoms.cell / self.n
        rmsd, permutation = self.comparator.calculate_rmsd(positions)

        self.distances[key] = rmsd
        self.permutations[key] = permutation
        return rmsd, permutation

    def is_consistent(self, H):
        """Callback function which tests whether the minimum-cost permutations
        of the subgroup elements are consistent with each other. H describes
        the subgroup basis."""
        n = self.n
        dims = [n] * self.comparator.dim
        num_atoms = len(self.comparator.atoms)

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
        dims = [n] * self.comparator.dim
        for H in enumerate_subgroup_bases(dims, self.is_consistent,
                                          min_index=1, max_index=n):
            group_index = np.prod(dims // np.diag(H))
            elements = get_subgroup_elements(dims, H)
            distances = np.array([self.distances[tuple(c)] for c in elements])
            permutations = np.array([self.permutations[tuple(c)]
                                     for c in elements])
            rmsd = np.sqrt(np.sum(distances**2) / (2 * group_index))
            yield (rmsd, group_index, H, permutations)
