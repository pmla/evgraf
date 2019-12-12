import numpy as np


def permute_axes(atoms, permutation):
    """Permute axes of unit cell and atom positions. Considers only cell and
    atomic positions. Other vector quantities such as momenta are not
    modified."""
    assert (np.sort(permutation) == np.arange(3)).all()

    permuted = atoms.copy()
    scaled = permuted.get_scaled_positions()

    cell = permuted.cell[permutation][:, permutation]
    permuted.set_cell(cell, scale_atoms=False)
    permuted.set_scaled_positions(scaled[:, permutation])
    permuted.set_pbc(permuted.pbc[permutation])
    return permuted
