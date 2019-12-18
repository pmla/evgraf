import numpy as np


def plot_atoms(atoms, other=None):

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))

    cell = atoms.cell[:2, :2]
    path = [(0, 0), cell[0], cell[0] + cell[1], cell[1], (0, 0)]
    xs, ys = zip(*path)
    plt.plot(xs, ys)

    for i, p in enumerate(path[:4]):
        plt.text(p[0], p[1], i)

    xlim = np.max(np.abs(cell[:, 0]))
    ylim = np.max(np.abs(cell[:, 1]))

    xs, ys, _ = atoms.get_positions().T
    plt.scatter(xs, ys)

    if other is not None:
        xs, ys, _ = other.T
        plt.scatter(xs, ys, marker='x')

    lim = max([np.linalg.norm(cell[0]),
               np.linalg.norm(cell[1]),
               np.linalg.norm(cell[0] + cell[1])])
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.show()
