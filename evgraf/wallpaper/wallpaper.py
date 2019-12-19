import itertools
import numpy as np
from evgraf import find_crystal_reductions
from evgraf.utils import standardize
from evgraf.crystal_comparator import CrystalComparator
from .lattice_symmetrization import symmetrize_bravais
from .layer_standardization import standardize_layer


class SymOp:
    def __init__(self, A, b):
        self.A = np.array(A)
        self.b = np.array(b)


identity = SymOp(np.eye(3), [0, 0, 0])
fx =  [identity, SymOp(np.diag([-1, +1, 1]), [  0,   0, 0])]
fy =  [identity, SymOp(np.diag([+1, -1, 1]), [  0,   0, 0])]
gxx = [identity, SymOp(np.diag([-1, +1, 1]), [0.5,   0, 0])]
gyy = [identity, SymOp(np.diag([+1, -1, 1]), [  0, 0.5, 0])]
gxy = [identity, SymOp(np.diag([-1, +1, 1]), [  0, 0.5, 0])]
gyx = [identity, SymOp(np.diag([+1, -1, 1]), [0.5,   0, 0])]
fr  = [identity, SymOp(np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                       [0.5, 0.5, 0])]


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), +np.cos(theta), 0],
                     [            0,              0, 1]])


def rotation(n):
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [SymOp(rotation_matrix(theta), [0, 0, 0]) for theta in thetas]


def product(X, Y):
    return [SymOp(x.A @ y.A, x.b + y.b) for x, y in itertools.product(X, Y)]


class Wallpaper:
    def __init__(self, lattice, symmetries):
        self.lattice = lattice
        self.symmetries = symmetries
        if 'hexagonal' in lattice:
            self.k = 3
            self.denom = 3
        else:
            self.k = 1
            self.denom = 2


groups = {'p1':  Wallpaper('oblique', []),
          'p2':  Wallpaper('oblique', [rotation(2)]),
          'pg':  Wallpaper('rectangular', [gxy, gyx]),
          'pm':  Wallpaper('rectangular', [fx, fy]),
          'cm':  Wallpaper('centred rectangular', [fx, fy]),
          'pgg': Wallpaper('rectangular', [product(gxy, gyx)]),
          'pmg': Wallpaper('rectangular', [product(rotation(2), gyy),
                                           product(rotation(2), gxx)]),
          'pmm': Wallpaper('rectangular', [product(fx, fy)]),
          'cmm': Wallpaper('centred rectangular', [product(fx, fy)]),
          'p4': Wallpaper('square', [rotation(4)]),
          'p4g': Wallpaper('square', [product(rotation(4), fr)]),
          'p4m': Wallpaper('square', [product(rotation(4), fy)]),
          'p3': Wallpaper('hexagonal', [rotation(3)]),
          'p3m1': Wallpaper('hexagonal', [product(rotation(3), fx)]),
          'p31m': Wallpaper('hexagonal', [product(rotation(3), fy)]),
          'p6': Wallpaper('hexagonal', [rotation(6)]),
          'p6m': Wallpaper('hexagonal', [product(rotation(6), fy)]),
         }


def get_wallpaper_distance(name, atoms):
    if name == 'p1':
        return 0

    std = standardize(atoms)
    atoms, axis_permutation = standardize_layer(std.atoms)
    if name not in groups:
        raise ValueError("Not a valid wallpaper group.")

    group = groups[name]
    k = group.k
    denom = group.denom

    n = len(atoms)
    dsym, sym_atoms = symmetrize_bravais(group.lattice, atoms)
    comparator = CrystalComparator(sym_atoms, subtract_barycenter=True)

    scores = np.zeros((k * n, k * n))
    pos = comparator.positions.copy()
    cell = comparator.invop @ comparator.atoms.cell

    best = float("inf")
    for symmetries in group.symmetries:
        for i, j in itertools.product(range(k * n), range(k * n)):
            offset = comparator.expand_coordinates((i, j))
            shift = offset / (denom * n) @ cell
            positions0 = pos - shift

            acc = 0
            for sym in symmetries:
                positions = positions0 @ sym.A.T + sym.b @ cell + shift
                rmsd, perm = comparator.calculate_rmsd(positions)
                nrmsdsq = n * rmsd**2
                acc += nrmsdsq
            scores[i, j] = acc
            best = min(best, acc)

            if 0:
                import matplotlib.pyplot as plt
                plt.imshow(scores, interpolation='none')
                plt.colorbar()
                plt.show()

    return dsym + best


def get_distances(atoms):
    results = []
    reductions = find_crystal_reductions(atoms)
    for reduced in reductions:
        for name in groups:
            dsym = get_wallpaper_distance(name, reduced.atoms)
            results.append((reduced.factor, name, dsym + reduced.rmsd))
    for factor, name, d in results:
        print(factor, name.rjust(4), d)
