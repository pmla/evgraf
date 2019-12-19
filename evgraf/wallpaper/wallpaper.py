import itertools
import numpy as np
from evgraf import find_crystal_reductions
from evgraf.utils import standardize, plot_atoms
from evgraf.crystal_comparator import CrystalComparator
from .lattice_symmetrization import symmetrize_bravais
from .layer_standardization import standardize_layer


names = ['p1', 'p2', 'pg', 'pm', 'cm', 'pgg', 'pmg', 'pmm', 'cmm',
         'p4', 'p4g', 'p4m', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']


class Symmetries:

    def __init__(self, r=1, f1=False, f2=False, g1=False, g2=False, df1=False, df2=False, cdf=False):
        self.ops = [r, f1, f2, g1, g2, df1, df2, cdf]
        self.nr = r

    def generate_ops(self):
        ops = np.array(self.ops).astype(np.int)
        ops[1:] += 1
        ops = [range(e) for e in ops]
        return list(itertools.product(*ops))[1:]


class WallpaperGroup:

    def __init__(self, bravais_lattice, symmetries=[]):
        self.bravais_lattice = bravais_lattice
        self.symmetries = symmetries
        if 'hexagonal' in bravais_lattice:
            self.k = 3
        else:
            self.k = 1


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), +np.cos(theta), 0],
                     [0, 0, 1]])


def rotation_matrices(n):
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [rotation_matrix(theta) for theta in thetas]


def flip_x(matrices):
    matrices = matrices.copy()
    for e in matrices:
        e[0] *= -1
    return matrices


def flip_y(matrices):
    matrices = matrices.copy()
    for e in matrices:
        e[1] *= -1
    return matrices


def reverse(matrices):
    return [e @ [[0, -1, 0], [-1, 0, 0], [0, 0, 1]] for e in matrices]


affine = {'p1':   [],
          'p2':   [rotation_matrices(2)],
          'pg':   [[(np.diag([-1, 1, 1.]), [0, 0.5, 0])], [(np.diag([1, -1, 1.]), [0.5, 0, 0])]],
          'pm':   [[np.diag([-1, 1, 1.])], [np.diag([1, -1, 1.])]],
          'cm':   [[np.diag([-1, 1, 1.])], [np.diag([1, -1, 1.])]],
          'pgg':  [[(np.diag([+1, +1, 0]), [  0,   0, 0]),
                    (np.diag([+1, -1, 0]), [0.5,   0, 0]),
                    (np.diag([-1, +1, 0]), [  0, 0.5, 0]),
                    (np.diag([-1, -1, 0]), [0.5, 0.5, 0])]],
          'pmg':  [rotation_matrices(2) + [(e, [0, 0.5, 0]) for e in flip_y(rotation_matrices(2))],
                   rotation_matrices(2) + [(e, [0.5, 0, 0]) for e in flip_x(rotation_matrices(2))]],
          'pmm':  [[np.diag([-1, 1, 1.]), np.diag([1, -1, 1.])]],
          'cmm':  [[np.diag([-1, 1, 1.]), np.diag([1, -1, 1.])]],
          'p4':   [rotation_matrices(4)],
          'p4g':  [rotation_matrices(4) + [(e, [0.5, 0.5, 0]) for e in reverse(rotation_matrices(4))]],
          'p4m':  [rotation_matrices(4) + flip_y(rotation_matrices(4))],
          'p3':   [rotation_matrices(3)],
          'p3m1': [rotation_matrices(3) + flip_x(rotation_matrices(3))],
          'p31m': [rotation_matrices(3) + flip_y(rotation_matrices(3))],
          'p6':   [rotation_matrices(6)],
          'p6m':  [rotation_matrices(6) + flip_y(rotation_matrices(6))],
         }


groups = {'p1':   WallpaperGroup('oblique'),
          'p2':   WallpaperGroup('oblique',             [Symmetries(r=2)]),
          'pg':   WallpaperGroup('rectangular',         [Symmetries(g1=True),
                                                         Symmetries(g2=True)]),
          'pm':   WallpaperGroup('rectangular',         [Symmetries(f1=True),
                                                         Symmetries(f2=True)]),
          'cm':   WallpaperGroup('centred rectangular', [Symmetries(df1=True),
                                                         Symmetries(df2=True)]),
          'pgg':  WallpaperGroup('rectangular',         [Symmetries(g1=True, g2=True)]),
          'pmg':  WallpaperGroup('rectangular',         [Symmetries(f2=True, g2=True),
                                                         Symmetries(f1=True, g1=True)]),
          'pmm':  WallpaperGroup('rectangular',         [Symmetries(f1=True, f2=True)]),
          'cmm':  WallpaperGroup('centred rectangular', [Symmetries(df1=True, df2=True)]),
          'p4':   WallpaperGroup('square',              [Symmetries(r=4)]),
          'p4g':  WallpaperGroup('square',              [Symmetries(r=4, cdf=True)]),
          'p4m':  WallpaperGroup('square',              [Symmetries(r=4, f1=True)]),
          'p3':   WallpaperGroup('hexagonal',           [Symmetries(r=3)]),
          'p3m1': WallpaperGroup('hexagonal',           [Symmetries(r=3, df2=True)]),
          'p31m': WallpaperGroup('hexagonal',           [Symmetries(r=3, f1=True)]),
          'p6':   WallpaperGroup('hexagonal',           [Symmetries(r=6)]),
          'p6m':  WallpaperGroup('hexagonal',           [Symmetries(r=6, f1=True)]),
         }


def flip_positions(positions, axis):
    axis = axis / np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0], 0])
    k = - 2 * np.dot(positions, normal)
    return positions + np.outer(k, normal)


def cherry_pick(comparator, positions, c, shift):

    cell = comparator.invop @ comparator.atoms.cell
    if c[1] == 1:
        positions = flip_positions(positions, cell[1])

    if c[2] == 1:
        positions = flip_positions(positions, cell[0])

    if c[3] == 1:
        positions += 0.5 * cell[0]
        positions = flip_positions(positions, cell[0])

    if c[4] == 1:
        positions += 0.5 * cell[1]
        positions = flip_positions(positions, cell[1])

    if c[5] == 1:
        positions = flip_positions(positions, cell[0] + cell[1])

    if c[6] == 1:
        positions = flip_positions(positions, cell[1] - cell[0])

    if c[7] == 1:
        positions += cell[0] / 2
        positions = flip_positions(positions, cell[0] + cell[1])
        positions -= cell[0] / 2

    theta = c[0]
    U = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), +np.cos(theta), 0],
                  [0, 0, 1]])
    positions = positions @ U.T
    positions += shift @ comparator.atoms.cell

    #print(offset, c)
    #plot_atoms(comparator.atoms, positions)
    return comparator.calculate_rmsd(positions)


def get_wallpaper_distance(name, atoms):
    if name == 'p1':
        return 0

    std = standardize(atoms)
    atoms, axis_permutation = standardize_layer(std.atoms)
    if name not in groups:
        raise ValueError("Not a valid wallpaper group.")

    group = groups[name]
    k = group.k

    dsym, sym_atoms = symmetrize_bravais(group.bravais_lattice, atoms)
    comparator = CrystalComparator(sym_atoms, subtract_barycenter=True)
    n = len(atoms)

    scores = np.zeros((k * n, k * n))
    denom = 2
    if group.k >= 3:
        denom = k

    scaled_positions = comparator.atoms.cell.scaled_positions(comparator.positions)
    cell = comparator.invop @ comparator.atoms.cell

    if name in affine:
        best = float("inf")
        for symmetries in affine[name]:
            for i, j in itertools.product(range(k * n), range(k * n)):
                offset = comparator.expand_coordinates((i, j))
                shift = offset / (denom * n)

                positions0 = (scaled_positions - shift) @ comparator.atoms.cell

                acc = 0
                for matrix in symmetries:
                    if isinstance(matrix, tuple):
                        matrix, trans = matrix
                    else:
                        trans = np.zeros(3)

                    positions = positions0 @ matrix.T + trans @ cell
                    positions += shift @ comparator.atoms.cell
                    rmsd, perm = comparator.calculate_rmsd(positions)
                    nrmsdsq = n * rmsd**2
                    #print(best, i, j, op, rmsd)
                    #plot_atoms(comparator.atoms, positions)
                    acc += nrmsdsq
                scores[i, j] = acc
                best = min(best, acc)
    else:
        best = float("inf")
        for symmetries in group.symmetries:
            ops = symmetries.generate_ops()
            for i, j in itertools.product(range(k * n), range(k * n)):
                offset = comparator.expand_coordinates((i, j))
                shift = offset / (denom * n)
                positions = (scaled_positions - shift) @ comparator.atoms.cell

                acc = 0
                for op in ops:
                    r, f1, f2, g1, g2, df1, df2, cdf = op
                    theta = 2 * np.pi * r / symmetries.nr
                    rmsd, perm = cherry_pick(comparator, positions.copy(),
                                             [theta, f1, f2, g1, g2, df1, df2, cdf], shift)
                    nrmsdsq = n * rmsd**2
                    #print(best, i, j, op, rmsd)
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
        for name in names:
            dsym = get_wallpaper_distance(name, reduced.atoms)
            results.append((reduced.factor, name, dsym + reduced.rmsd))
    for factor, name, d in results:
        print(factor, name.rjust(4), d)
