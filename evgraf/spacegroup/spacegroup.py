import itertools
import numpy as np
from evgraf import find_crystal_reductions
from evgraf.crystal_comparator import CrystalComparator
from .lattice_symmetrization import symmetrize_bravais
from ase.spacegroup import get_bravais_class
from ase.spacegroup import Spacegroup


class SpaceGroup:
    def __init__(self, lattice, symmetries):
        self.lattice = lattice
        self.symmetries = symmetries
        if 'hexagonal' in lattice or 'rhombohedral' in lattice:
            self.num = 3
            self.denom = 3
        else:
            self.num = 2
            self.denom = 2


groups = {}
for i in range(1, 231):
    sg = Spacegroup(i)
    if sg.centrosymmetric:
        signs = [-1, 1]
    else:
        signs = [1]

    symmetries = []
    for s in signs:
        for r, t in zip(sg.rotations, sg.translations):
            symmetries.append((s * r, s * t))

    lattice = get_bravais_class(i).longname
    groups[i] = SpaceGroup(lattice, [symmetries])


def measure(comparator, positions, cell, symmetries, n, denom, acc, best, c):
    offset = comparator.expand_coordinates(c)
    shift = offset / (denom * n) @ cell

    for rotation, translation in symmetries:
        positions = (positions - shift) @ rotation.T + translation @ cell + shift
        rmsd, perm = comparator.calculate_rmsd(positions)
        nrmsdsq = n * rmsd**2
        acc += nrmsdsq
        if acc >= best:
            break

    return acc


def _get_spacegroup_distance(name, atoms, best=float("inf")):
    if name == 'p1':
        return 0
    elif name not in groups:
        raise ValueError("Not a valid space group.")

    n = len(atoms)
    group = groups[name]
    num = group.num
    denom = group.denom

    dsym, sym_atoms = symmetrize_bravais(group.lattice, atoms)
    if dsym >= best:
        return dsym

    comparator = CrystalComparator(sym_atoms, subtract_barycenter=True)
    positions = comparator.positions.copy()
    cell = comparator.invop @ comparator.atoms.cell

    for symmetries in group.symmetries:
        for c in itertools.product(range(num * n), repeat=3):
            acc = measure(comparator, positions, cell, symmetries, n, denom, dsym, best, c)
            best = min(best, acc)
    return best


def get_spacegroup_distance(atoms, name):
    results = []
    reductions = find_crystal_reductions(atoms)
    assert len(reductions)
    best = float("inf")

    for reduced in reductions[::-1]:
        dsym = _get_spacegroup_distance(name, reduced.atoms, best=best)
        results.append(dsym + reduced.rmsd)
        best = min(best, dsym + reduced.rmsd)

    results = np.array(results)
    return np.min(results)


'''
names = ['p1', 'bcc', 'fcc']

def get_distances(atoms):
    results = []
    reductions = find_crystal_reductions(atoms)
    assert len(reductions)
    for reduced in reductions:
        row = []
        for name in names:
            dsym = get_spacegroup_distance(name, reduced.atoms)
            row.append(dsym + reduced.rmsd)
        results.append(row)
    results = np.array(results)
    return np.min(results, axis=0)
'''
