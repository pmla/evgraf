import math
import functools
import itertools
import numpy as np
from typing import List
from scipy.spatial.distance import cdist
from .assignment import linear_sum_assignment
from .subgroup_enumeration import (enumerate_subgroup_bases,
                                   get_subgroup_elements)
from evgrafcpp import bipartite_matching


def reduce_gcd(x):
    return functools.reduce(math.gcd, x)


def concatenate_permutations(permutations: List[np.ndarray]) -> np.ndarray:
    """Takes a list of permutations `[array([0, 1, 2]), array([0, 1])]` and
    returns a consecutive permutation `array([0, 1, 2, 3, 4])`"""
    concatenated = []
    for p in permutations:
        concatenated.extend(p + len(concatenated))
    return np.array(concatenated)


def get_neighboring_cells(pbc):
    pbc = pbc.astype(np.int)
    return np.array(list(itertools.product(*[range(-p, p + 1) for p in pbc])))


def minimum_cost_matching(positions, num_cells, nbr_positions):
    """Finds a minimum-cost matching between atoms in two crystal structures of
    the same element. The squared Euclidean distance is used as the cost
    function. Due to crystal periodicity the cost is calculated as the minimum
    over all neighboring cells."""
    n = len(positions)
    distances = cdist(positions, nbr_positions, 'sqeuclidean')
    distances = np.min(distances.reshape((n, num_cells, n)), axis=1)
    res = linear_sum_assignment(distances)
    return np.sum(distances[res]), res[1]


def calculate_rmsd(positions, zindices, mapped_nbrs, num_cells):
    """Calculates the RMSD between two crystal structures. The cost for each
    element is calculated separately."""
    nrmsdsq = 0
    permutations = []
    for indices, species_nbrs in zip(zindices, mapped_nbrs):
        cost, permutation = minimum_cost_matching(positions[indices],
                                                  num_cells, species_nbrs)
        nrmsdsq += cost
        permutations.append(permutation)

    num_atoms = len(positions)
    rmsd = np.sqrt(nrmsdsq / num_atoms)
    return rmsd, concatenate_permutations(permutations)


def standardize(atoms, subtract_barycenter=False):
    zpermutation = np.argsort(atoms.numbers, kind='merge')
    atoms = atoms[zpermutation]
    atoms.wrap(eps=0)

    barycenter = np.mean(atoms.get_positions(), axis=0)
    if subtract_barycenter:
        atoms.positions -= barycenter

    rcell, op = atoms.cell.minkowski_reduce()
    invop = np.linalg.inv(op)
    atoms.set_cell(rcell, scale_atoms=False)

    atoms.wrap(eps=0)
    return atoms, invop, barycenter, zpermutation


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

    def __init__(self, atoms, invert=False):
        self.invert = invert
        self.distances = {}
        self.permutations = {}
        atoms, invop, barycenter, zpermutation = standardize(atoms,
                                                             self.invert)
        self.atoms = atoms
        self.invop = invop
        self.zpermutation = zpermutation
        self.scaled = atoms.get_scaled_positions()

        positions = atoms.get_positions()
        self.nbr_cells = get_neighboring_cells(atoms.pbc)
        self.zindices = [np.where(atoms.numbers == element)[0]
                         for element in np.unique(atoms.numbers)]
        self.nbr_pos = [np.concatenate([positions[indices] + e @ atoms.cell
                                       for e in self.nbr_cells])
                        for indices in self.zindices]
        if invert:
            self.n = len(atoms)
            self.barycenter = barycenter
        else:
            self.n = reduce_gcd(atoms.symbols.formula.count().values())

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
        transformed = sign * self.scaled + c / self.n
        for i, index in enumerate(c):
            if pbc[i]:
                transformed[:, i] %= 1.0
        positions = self.atoms.cell.cartesian_positions(transformed)

        rmsd, permutation = calculate_rmsd(positions, self.zindices,
                                           self.nbr_pos, len(self.nbr_cells))
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
        num_atoms = len(self.scaled)

        seen = -np.ones((3, num_atoms)).astype(np.int)
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
                                          min_index=2, max_index=n):
            group_index = np.prod(dims // np.diag(H))
            elements = get_subgroup_elements(dims, H)
            distances = np.array([self.distances[tuple(c)] for c in elements])
            permutations = np.array([self.permutations[tuple(c)]
                                     for c in elements])
            rmsd = np.sqrt(np.sum(distances**2) / (2 * group_index))
            yield (rmsd, group_index, H, permutations)
