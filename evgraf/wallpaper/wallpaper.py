import itertools
import numpy as np
from evgraf import find_crystal_reductions
from evgraf.utils import standardize, plot_atoms
from evgraf.crystal_comparator import CrystalComparator
from .lattice_symmetrization import symmetrize_bravais
from .layer_standardization import standardize_layer


names = ['p1', 'p2', 'pg', 'pm', 'cm', 'pgg', 'pmg', 'pmm', 'cmm',
         'p4', 'p4g', 'p4m', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']


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
          'pg':   [[(np.diag([-1, 1, 1.]), [0, 0.5, 0])],
                   [(np.diag([1, -1, 1.]), [0.5, 0, 0])]],
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


def get_wallpaper_distance(name, atoms):
    if name == 'p1':
        return 0

    std = standardize(atoms)
    atoms, axis_permutation = standardize_layer(std.atoms)
    if name not in affine:
        raise ValueError("Not a valid wallpaper group.")

    bravais_lattice = {'p1':   'oblique',
                       'p2':   'oblique',
                       'pg':   'rectangular',
                       'pm':   'rectangular',
                       'cm':   'centred rectangular',
                       'pgg':  'rectangular',
                       'pmg':  'rectangular',
                       'pmm':  'rectangular',
                       'cmm':  'centred rectangular',
                       'p4':   'square',
                       'p4g':  'square',
                       'p4m':  'square',
                       'p3':   'hexagonal',
                       'p3m1': 'hexagonal',
                       'p31m': 'hexagonal',
                       'p6':   'hexagonal',
                       'p6m':  'hexagonal',
                       }

    dsym, sym_atoms = symmetrize_bravais(bravais_lattice[name], atoms)
    comparator = CrystalComparator(sym_atoms, subtract_barycenter=True)
    n = len(atoms)

    if 'hexagonal' in bravais_lattice[name]:
        k = 3
        denom = 3
    else:
        k = 1
        denom = 2

    scores = np.zeros((k * n, k * n))
    pos = comparator.positions.copy()
    cell = comparator.invop @ comparator.atoms.cell

    best = float("inf")
    for symmetries in affine[name]:
        for i, j in itertools.product(range(k * n), range(k * n)):
            offset = comparator.expand_coordinates((i, j))
            shift = offset / (denom * n) @ cell

            positions0 = pos - shift

            acc = 0
            for matrix in symmetries:
                if isinstance(matrix, tuple):
                    matrix, trans = matrix
                else:
                    trans = np.zeros(3)

                positions = positions0 @ matrix.T + trans @ cell + shift
                rmsd, perm = comparator.calculate_rmsd(positions)
                nrmsdsq = n * rmsd**2
                #print(best, i, j, op, rmsd)
                #plot_atoms(comparator.atoms, positions)
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
