import numpy as np
from ase.geometry import find_mic


def calculate_mean_positions(parents, positions, cell, pbc, num_atoms, rmsd):
    # collect atoms
    deltas = positions - positions[parents]
    vmin, _ = find_mic(deltas, cell, pbc=pbc)
    roots = np.unique(parents)
    superimposed = vmin + positions[parents]

    # calculate mean positions
    mean_positions = []
    for i in roots:
        indices = np.where(parents == i)[0]
        mean_positions.append(np.mean(superimposed[indices], axis=0))
    mean_positions = np.array(mean_positions)

    # check spatial consistency
    deltas = superimposed - mean_positions[parents]
    rmsd_check = np.sqrt(np.sum(deltas**2) / num_atoms)
    if abs(rmsd - rmsd_check) > 1E-12:
        return None

    return mean_positions

