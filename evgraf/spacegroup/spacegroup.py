import itertools
import numpy as np
from ase import Atoms
from ase.geometry import permute_axes
from evgraf import find_crystal_reductions
from evgraf.crystal_comparator import CrystalComparator
from evgraf.utils import calculate_mean_positions
from .lattice_symmetrization import symmetrize_bravais
from .build_spacegroups import spacegroups


class Evaluator:

    def __init__(self, lattice, reduced):
        dsym, sym_atoms, conventional = symmetrize_bravais(lattice,
                                                           reduced.atoms)
        self.n = len(reduced.atoms)
        self.reduced = reduced
        self.dsym = dsym
        self.sym_atoms = sym_atoms
        self.conventional_cell = conventional

    def cost(self, acc=0):
        dsg = np.sqrt(acc**2)
        return self.reduced.rmsd + self.dsym + dsg


def measure(comparator, conventional, symmetries, fractional, shift, yep,
            best):
    n = yep.n
    acc = 0
    permutations = []
    images = []
    for rotation, translation in symmetries:
        transformed = (fractional @ rotation.T + translation) @ conventional
        rmsd, perm = comparator.calculate_rmsd(transformed, translation=shift)
        permutations.append(perm)
        images.append(transformed)

        nrmsdsq = n * rmsd**2
        acc += nrmsdsq
        obj = yep.cost(acc)
        if obj >= best[0]:
            obj = float("inf")
            break

    return acc, np.array(permutations), np.array(images)


def get_spacegroup_distance(settings, yep, best):
    n = yep.n
    sym_atoms = yep.sym_atoms
    conventional_cell = yep.conventional_cell

    fixed = ['cubic', 'hexagonal', 'rhombohedral', 'tetragonal']#, 'monoclinic']
    if any([s in settings[0].lattice for s in fixed]):
        axis_permutations = [[0, 1, 2]]
    else:
        axis_permutations = itertools.permutations(range(3))
        axis_permutations = [list(p) for p in axis_permutations]

    for axis_permutation in axis_permutations:
        permuted_atoms = permute_axes(sym_atoms, axis_permutation)
        conventional = conventional_cell[axis_permutation][:, axis_permutation]
        comparator = CrystalComparator(permuted_atoms, subtract_barycenter=True)
        cell = comparator.invop @ comparator.atoms.cell

        for group in settings:
            k = 2
            if not group.centrosymmetric:
                k = 4

            for c in itertools.product(range(k * n), repeat=3):
                shift = comparator.expand_coordinates(c) @ cell / (k * n)
                fractional = np.linalg.solve(conventional.T,
                                             (comparator.positions - shift).T).T

                acc, rperm, images = measure(comparator, conventional,
                                             group.symmetries, fractional,
                                             shift, yep, best)
                result = (yep.cost(acc), acc, comparator, rperm, images)
                best = min(best, result, key=lambda x:x[0])
    return best


def find_spacegroup_symmetry(atoms, sg_number):
    if sg_number < 0 or sg_number > 230:
        raise ValueError("space group must be in range [0, 230]")
    if sg_number == 0:
        return 0

    reductions = find_crystal_reductions(atoms)
    assert len(reductions)

    settings = spacegroups[sg_number]
    lattice = settings[0].lattice

    yeps = [Evaluator(lattice, reduced) for reduced in reductions]
    indices = np.argsort([yep.cost() for yep in yeps])
    yeps = [yeps[i] for i in indices]

    best = (float("inf"), None, None, None)
    for yep in yeps:
        if yep.cost() > 1E-3: continue
        if yep.cost() < best[0]:
            res = get_spacegroup_distance(settings, yep, best)
            best = min(best, res)
    distance, acc, comparator, permutations, images = best

    # cluster atoms
    cell = comparator.atoms.cell
    pbc = comparator.atoms.pbc
    numbers = []
    positions = []
    for permutation, image in zip(permutations, images):
        inverse = np.argsort(permutation)
        numbers.append(comparator.atoms.numbers[inverse])
        positions.append(image[inverse])
    numbers = np.array(numbers)
    positions = np.array(positions)

    assert len(np.unique(numbers, axis=0)) == 1
    numbers = numbers[0]

    num_atoms = len(comparator.atoms)
    rmsd = np.sqrt(acc / (2 * num_atoms))
    parents = np.arange(num_atoms * len(images)) % num_atoms
    mean_positions = calculate_mean_positions(parents,
                                              positions.reshape((-1, 3)),
                                              cell, pbc, num_atoms, rmsd)
    if mean_positions is None:
        return float("inf"), None

    symmetrized = Atoms(positions=mean_positions, numbers=numbers,
                        cell=cell, pbc=pbc)
    symmetrized.wrap(eps=0)
    return distance, symmetrized
