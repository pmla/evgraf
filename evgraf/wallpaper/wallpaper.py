import itertools
import numpy as np
from evgraf import find_crystal_reductions
from evgraf.utils import standardize, rotation_matrix
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


def rotation(n):
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [SymOp(rotation_matrix(theta), [0, 0, 0]) for theta in thetas]


def product(X, Y):
    return [SymOp(x.A @ y.A, x.b + y.b) for x, y in itertools.product(X, Y)]


class WallpaperGroup:
    def __init__(self, lattice, symmetries):
        self.lattice = lattice
        self.symmetries = symmetries
        if 'hexagonal' in lattice:
            self.k = 3
            self.denom = 3
        else:
            self.k = 1
            self.denom = 2


groups = {'p1':  WallpaperGroup('oblique', []),
          'p2':  WallpaperGroup('oblique', [rotation(2)]),
          'pg':  WallpaperGroup('rectangular', [gxy, gyx]),
          'pm':  WallpaperGroup('rectangular', [fx, fy]),
          'cm':  WallpaperGroup('centred rectangular', [fx, fy]),
          'pgg': WallpaperGroup('rectangular', [product(gxy, gyx)]),
          'pmg': WallpaperGroup('rectangular', [product(rotation(2), gyy),
                                                product(rotation(2), gxx)]),
          'pmm': WallpaperGroup('rectangular', [product(fx, fy)]),
          'cmm': WallpaperGroup('centred rectangular', [product(fx, fy)]),
          'p4': WallpaperGroup('square', [rotation(4)]),
          'p4g': WallpaperGroup('square', [product(rotation(4), fr)]),
          'p4m': WallpaperGroup('square', [product(rotation(4), fy)]),
          'p3': WallpaperGroup('hexagonal', [rotation(3)]),
          'p3m1': WallpaperGroup('hexagonal', [product(rotation(3), fx)]),
          'p31m': WallpaperGroup('hexagonal', [product(rotation(3), fy)]),
          'p6': WallpaperGroup('hexagonal', [rotation(6)]),
          'p6m': WallpaperGroup('hexagonal', [product(rotation(6), fy)]),
         }


def get_wallpaper_distance(name, atoms):
    if name == 'p1':
        return 0
    elif name not in groups:
        raise ValueError("Not a valid wallpaper group.")

    n = len(atoms)
    std = standardize(atoms)
    atoms, axis_permutation = standardize_layer(std.atoms)
    group = groups[name]
    k = group.k
    denom = group.denom

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

            acc = dsym
            for sym in symmetries:
                positions = positions0 @ sym.A.T + sym.b @ cell + shift
                rmsd, perm = comparator.calculate_rmsd(positions)
                nrmsdsq = n * rmsd**2
                acc += nrmsdsq
                if acc >= best:
                    break
            scores[i, j] = acc
            best = min(best, acc)

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(scores, interpolation='none')
            plt.colorbar()
            plt.savefig("plot_{0}.png".format(name))
            plt.clf()

    return best



names = ['p1', 'p2', 'pg', 'pm', 'cm', 'pgg', 'pmg', 'pmm', 'cmm',
         'p4', 'p4g', 'p4m', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']


def get_distances(atoms):
    results = []
    reductions = find_crystal_reductions(atoms)
    assert len(reductions)
    for reduced in reductions:
        row = []
        for name in names:
            dsym = get_wallpaper_distance(name, reduced.atoms)
            row.append(dsym + reduced.rmsd)
        results.append(row)
    results = np.array(results)
    return np.min(results, axis=0)
