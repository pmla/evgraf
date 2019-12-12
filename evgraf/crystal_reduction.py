import numpy as np
from collections import namedtuple
from scipy.spatial.distance import cdist
from ase import Atoms
from ase.geometry.dimensionality.disjoint_set import DisjointSet
from .crystal_reducer import CrystalReducer


def assign_atoms_to_clusters(num_atoms, numbers, permutations):
    uf = DisjointSet(num_atoms)
    for p in permutations:
        for i, e in enumerate(p):
            uf.merge(i, e)

    components = uf.get_components()
    assert (numbers == numbers[components]).all()
    return uf.get_components(relabel=True)


def collect_atoms(n, H, atoms):
    """Collects atoms which belong in the same cluster. H describes the
    subgroup basis."""
    dim = sum(atoms.pbc)
    # Extend the subgroup basis to 3D (if not already)
    R = np.diag([n, n, n])
    indices = np.where(atoms.pbc)[0]
    for i in range(dim):
        for j in range(dim):
            R[indices[i], indices[j]] = H[i, j]

    # Perform an appropriate contraction of the unit cell and wrap the atoms
    atoms.set_cell(R @ atoms.cell / n, scale_atoms=False)
    atoms.wrap(eps=0)
    return atoms


def cluster_component(ps, permutations, shifts, i):
    ds = cdist(ps[i] - shifts, ps[permutations[:, i]])
    indices = np.argmin(ds, axis=0)
    positions = ps[permutations[:, i]] + shifts[indices]

    meanpos = np.mean(positions, axis=0)
    component_rmsd = np.sum((positions - meanpos)**2)
    return meanpos, component_rmsd


def reduced_layout(reducer, rmsd, group_index, H, permutations):

    num_atoms = len(reducer.atoms)
    numbers = reducer.atoms.numbers
    components = assign_atoms_to_clusters(num_atoms, numbers, permutations)
    if num_atoms // group_index != len(np.bincount(components)):
        return None

    if len(np.unique(np.bincount(components))) > 1:
        return None

    collected = collect_atoms(reducer.n, H, reducer.atoms.copy())
    shifts = reducer.nbr_cells @ collected.cell

    data = []
    for c in np.unique(components):
        i = list(components).index(c)
        meanpos, crmsd = cluster_component(collected.get_positions(),
                                           permutations, shifts, i)
        data.append((meanpos, numbers[i], crmsd))
    positions, numbers, crmsd = zip(*data)

    rmsd_check = np.sqrt(sum(crmsd) / num_atoms)
    if abs(rmsd - rmsd_check) > 1E-12:
        return None

    reduced = Atoms(positions=positions, numbers=numbers,
                    cell=collected.cell, pbc=collected.pbc)
    reduced.set_cell(reducer.invop @ reduced.cell, scale_atoms=False)
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
    """
    reducer = CrystalReducer(atoms)
    reductions = reducer.find_consistent_reductions()
    Reduced = namedtuple('ReducedCrystal', 'rmsd factor atoms components')
    invzperm = np.argsort(reducer.zpermutation)

    reduced = {}
    for rmsd, group_index, H, permutations in reductions:
        result = reduced_layout(reducer, rmsd, group_index, H, permutations)
        if result is not None:
            reduced_atoms, components = result
            key = group_index
            entry = Reduced(rmsd=rmsd, factor=group_index, atoms=reduced_atoms,
                            components=components[invzperm])
            if key not in reduced:
                reduced[key] = entry
            else:
                reduced[key] = min(reduced[key], entry, key=lambda x: x.rmsd)

    return sorted(reduced.values(), key=lambda x: x.factor)
