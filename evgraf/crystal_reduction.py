import numpy as np
from collections import namedtuple
from ase import Atoms
from ase.geometry import find_mic
from ase.geometry.dimensionality.disjoint_set import DisjointSet
from .crystal_reducer import CrystalReducer


Reduced = namedtuple('ReducedCrystal', 'rmsd factor atoms components map')


def assign_atoms_to_clusters(num_atoms, permutations):
    uf = DisjointSet(num_atoms)
    for p in permutations:
        for i, e in enumerate(p):
            uf.merge(i, e)
    return uf.get_components(relabel=True)


def reduction_basis(n, H, pbc):
    dim = sum(pbc)
    # Extend the subgroup basis to 3D (if not already)
    R = np.diag([n, n, n])
    indices = np.where(pbc)[0]
    for i in range(dim):
        for j in range(dim):
            R[indices[i], indices[j]] = H[i, j]
    return R / n


def reduced_layout(reducer, rmsd, group_index, R, permutations, atoms):

    num_atoms = len(reducer.comparator.atoms)
    components = assign_atoms_to_clusters(num_atoms, permutations)
    if num_atoms // group_index != len(np.bincount(components)):
        return None

    if len(np.unique(np.bincount(components))) > 1:
        return None

    # Collect atoms in contracted unit cell
    indices = np.argsort(components)
    collected = reducer.comparator.atoms[indices]
    collected.set_cell(R @ atoms.cell, scale_atoms=False)
    collected.wrap(eps=0)

    clusters = components[indices]
    ps = collected.get_positions()
    parents = clusters * group_index
    vmin, _ = find_mic(ps - ps[parents], collected.cell, pbc=collected.pbc)
    positions = ps[parents] + vmin

    m = num_atoms // group_index
    numbers = collected.numbers.reshape((m, group_index))[:, 0]
    meanpos = np.mean(positions.reshape((m, group_index, 3)), axis=1)
    deltas = positions - meanpos[clusters]
    rmsd_check = np.sqrt(np.sum(deltas**2) / num_atoms)
    if abs(rmsd - rmsd_check) > 1E-12:
        return None

    reduced = Atoms(positions=meanpos, numbers=numbers,
                    cell=collected.cell, pbc=collected.pbc)
    reduced.wrap(eps=0)
    return reduced, components


def find_crystal_reductions(atoms):
    """Finds reductions of a crystal using the root-mean-square (RMS) distance.

    A crystal reduction is defined by a translational symmetry in the input
    structure. Each translational symmetry has an associated cost, which is the
    RMS distance from the input structure to its symmetrized (i.e. reduced)
    structure. The atomic coordinates in the reduced crystal are given by the
    Euclidean average of the those in the input structure (after being wrapped
    into the reduced unit cell).

    If the crystal structure is perfect, the reduced crystal is the textbook
    primitive unit cell and has a RMS distance of zero. As the atomic
    coordinates in the input structure deviate from perfect translational
    symmetry the RMS distance increases correspondingly. Similarly, the RMS
    distance cannot decrease as the reduction factor increases.

    See the tutorial for an example with illustrations.

    Parameters:

    atoms: ASE atoms object
        The system to reduce.

    Returns:

    reduced: list
        List of ReducedCrystal objects for reduction found. A ReducedCrystal
        is a namedtuple with the following field names:

        rmsd: float
            RMS distance from input structure to reduced structure
        factor: integer
            The reduction factor
        atoms: Atoms object
            The reduced structure
        components: integer ndarray
            Describes how atoms in the input structure are combined in the
            reduced structure
        map: ndarray
            Map from input cell to reduced cell
    """
    reducer = CrystalReducer(atoms)
    reductions = reducer.find_consistent_reductions()
    invzperm = np.argsort(reducer.comparator.zpermutation)

    reduced = {}
    for rmsd, group_index, H, permutations in reductions:
        R = reduction_basis(reducer.n, H, atoms.pbc)
        R = reducer.comparator.invop @ R @ reducer.comparator.op
        result = reduced_layout(reducer, rmsd, group_index, R, permutations,
                                atoms)
        if result is not None:
            reduced_atoms, components = result
            key = group_index
            entry = Reduced(rmsd=rmsd, factor=group_index, atoms=reduced_atoms,
                            components=components[invzperm],
                            map=R)
            if key not in reduced:
                reduced[key] = entry
            else:
                reduced[key] = min(reduced[key], entry, key=lambda x: x.rmsd)

    return sorted(reduced.values(), key=lambda x: x.factor)
