import numpy as np
from evgraf.standardization import standardize
from .chain_standardization import standardize_chain
from .chain_registration import register_chain


def get_multipliers(n1, n2):
    num_chain_steps = min(n1, n2)
    multiplier1 = [1, 1, 1]
    multiplier2 = [1, 1, 1]

    if n1 != n2:
        lcm = np.lcm(n1, n2)
        multiplier1 = [1, 1, lcm // n1]
        multiplier2 = [1, 1, lcm // n2]

    return num_chain_steps, multiplier1, multiplier2


def _calculate_rmsd(atoms1, atoms2, ignore_stoichiometry, allow_rotation,
                    num_chain_steps):

    # put structures into intermediate cell
    l = (atoms1.cell[2, 2] + atoms2.cell[2, 2]) / 2

    imcell = l * np.eye(3)
    atoms1.set_cell(imcell, scale_atoms=True)
    atoms2.set_cell(imcell, scale_atoms=True)
    atoms1.wrap(eps=0)
    atoms2.wrap(eps=0)

    # prepare for registration
    num_atoms = len(atoms1)
    pos1 = atoms1.get_positions(wrap=0)
    pos2 = atoms2.get_positions(wrap=0)

    eindices = [np.where(atoms1.numbers == element)[0]
                for element in np.unique(atoms1.numbers)]

    best = (float("inf"), None, None, None)
    for i in range(num_chain_steps):

        translated_positions = pos1 + [0, 0, i * l / num_atoms]
        translated_positions[:, 2] %= l

        b = (best[0], np.arange(num_atoms), np.eye(2))
        rmsd, perm, u = register_chain(translated_positions, pos2,
                                       eindices, l, b)
        U = np.eye(3)
        U[:2, :2] = u

        trial = (rmsd, perm, U, (0, 0, i))
        best = min(best, trial, key=lambda x: x[0])

    rmsd, perm, U, ijk = best
    return rmsd  # , perm, U, ijk


def calculate_rmsd_chain(atoms1, atoms2, ignore_stoichiometry=False,
                         allow_reflection=False, allow_rotation=False):
    """Calculates the optimal RMSD between two 1D chains.

    atoms1:               The first atoms object.
    atoms2:               The second atoms object.
    ignore_stoichiometry: Whether to use stoichiometry in comparison.
                          Stoichiometries must be identical if 'True'.
    allow_reflection:     Whether to test mirror images.
    allow_rotation:       Whether to test rotations.

    Returns
    =======

    rmsd:                 The root-mean-square distance between atoms in the
                          intermediate cell.
    """

    assert len(atoms1) > 0
    assert len(atoms2) > 0

    # general structural standardization
    std1 = standardize(atoms1, subtract_barycenter=True)
    std2 = standardize(atoms2, subtract_barycenter=True)

    if ignore_stoichiometry:
        std1.atoms.numbers[:] = 1
        std2.atoms.numbers[:] = 1
    else:
        if (std1.atoms.numbers != std2.atoms.numbers).any():
            raise TypeError("stoichiometries must be equal unless "
                            "`ignore_stoichiometry==False`")

    # chain standardization
    atoms1, axis_permutation1 = standardize_chain(std1.atoms)
    atoms2, axis_permutation2 = standardize_chain(std2.atoms)

    num_chain_steps, m1, m2 = get_multipliers(len(atoms1), len(atoms2))
    atoms1 = atoms1 * m1
    atoms2 = atoms2 * m2

    res = _calculate_rmsd(atoms1, atoms2, ignore_stoichiometry,
                          allow_rotation, num_chain_steps)
    if not allow_reflection:
        return res

    atoms1 = atoms1.copy()
    scaled = -atoms1.get_scaled_positions(wrap=0)
    atoms1.set_scaled_positions(scaled)

    fres = _calculate_rmsd(atoms1, atoms2, ignore_stoichiometry,
                           allow_rotation, num_chain_steps)
    return min(res, fres)
