import itertools
import numpy as np
from evgraf import find_crystal_reductions
from evgraf.utils import minkowski_reduce, standardize, plot_atoms
from evgraf.crystal_reducer import CrystalReducer, expand_coordinates
from .lattice_symmetrization import symmetrize_bravais
from .layer_standardization import standardize_layer


def flip_positions(positions, axis):
    axis = axis / np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0], 0])
    k = - 2 * np.dot(positions, normal)
    return positions + np.outer(k, normal)


def cherry_pick(reducer, c, offset, denom, invcell):

    num_atoms = len(reducer.atoms)
    offset = expand_coordinates(offset, reducer.atoms.pbc)
    shift = offset / (denom * num_atoms)
    transformed = reducer.scaled_positions - shift

    if c[3] == 1:
        transformed[:, 0] += 0.5
        transformed[:, 1] *= -1

    if c[4] == 1:
        transformed[:, 1] += 0.5
        transformed[:, 0] *= -1

    positions = transformed @ reducer.atoms.cell

    if c[1] == 1:
        positions = flip_positions(positions, reducer.atoms.cell[1])

    if c[2] == 1:
        positions = flip_positions(positions, reducer.atoms.cell[0])

    if c[5] == 1:
        cell = reducer.invop @ reducer.atoms.cell
        positions = flip_positions(positions, cell[0] + cell[1])

    if c[6] == 1:
        cell = reducer.invop @ reducer.atoms.cell
        positions = flip_positions(positions, cell[1] - cell[0])

    if c[7] == 1:
        positions += reducer.atoms.cell[0] / 2
        positions = flip_positions(positions, reducer.atoms.cell[0] + reducer.atoms.cell[1])
        positions -= reducer.atoms.cell[0] / 2

    theta = c[0]
    U = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), +np.cos(theta), 0],
                  [0, 0, 1]])
    positions = positions @ U.T

    #transformed = np.linalg.solve(reducer.atoms.cell.T, positions.T).T
    transformed = positions @ invcell
    transformed += shift
    transformed[:, 0] %= 1.0
    transformed[:, 1] %= 1.0
    positions = transformed @ reducer.atoms.cell

    #print(offset, c)
    #plot_atoms(reducer.atoms, positions)
    return reducer.calculate_rmsd(positions)


def straightforward(sym_atoms, k=1, nr=1, nf1=1, nf2=1, ng1=1, ng2=1, ndf1=1, ndf2=1, ncdf=1):

    n = len(sym_atoms)
    reducer = CrystalReducer(sym_atoms, invert=True)
    ops = [range(nr), range(nf1), range(nf2), range(ng1), range(ng2),
           range(ndf1), range(ndf2), range(ncdf)]

    scores = np.zeros((k * n, k * n))
    denom = 2
    if k >= 3:
        denom = k

    invcell = np.linalg.inv(reducer.atoms.cell)

    best = float("inf")
    for i, j in itertools.product(range(k * n), range(k * n)):
        acc = 0
        for it, op in enumerate(itertools.product(*ops)):
            if it == 0:
                continue

            r, f1, f2, g1, g2, df1, df2, cdf = op
            theta = 2 * np.pi * r / nr
            rmsd, perm = cherry_pick(reducer, [theta, f1, f2, g1, g2, df1, df2, cdf], (i, j), denom, invcell)
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
    return best


def get_wallpaper_distance(name, atoms):
    '''
    ops = {'p2':  ('oblique',             [[2, 0, 0, 0, 0, 0, 0, 0]]),
           'pm':  ('rectangular',         [[0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0, 0]]),
           'pg':  ('rectangular',         [[0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 0, 0, 0]]),
           'cm':  ('centred rectangular', [[0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 0, 0, 0]]),
           'pmm': ('rectangular',         [[0, 0, 0, 0, 0, 0, 0, 0]]),
           'pmg': ('rectangular',         [[0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 0, 0, 0]]),
    '''

    std = standardize(atoms)
    atoms, axis_permutation = standardize_layer(std.atoms)
    if name == 'p1':
        return 0

    elif name == 'p2':
        dsym, sym_atoms = symmetrize_bravais('oblique', atoms)
        return dsym + straightforward(sym_atoms, nr=2)

    elif name == 'pm':
        dsym, sym_atoms = symmetrize_bravais('rectangular', atoms)
        res1 = straightforward(sym_atoms, nf1=2)
        res2 = straightforward(sym_atoms, nf2=2)
        return min(res1, res2)

    elif name == 'pg':
        dsym, sym_atoms = symmetrize_bravais('rectangular', atoms)
        res1 = straightforward(sym_atoms, ng1=2)
        res2 = straightforward(sym_atoms, ng2=2)
        return min(res1, res2)

    elif name == 'cm':
        dsym, sym_atoms = symmetrize_bravais('centred rectangular', atoms)
        res1 = straightforward(sym_atoms, ndf1=2)
        res2 = straightforward(sym_atoms, ndf2=2)
        return min(res1, res2)

    elif name == 'pmm':
        dsym, sym_atoms = symmetrize_bravais('rectangular', atoms)
        return dsym + straightforward(sym_atoms, nf1=2, nf2=2)

    elif name == 'pmg':
        dsym, sym_atoms = symmetrize_bravais('rectangular', atoms)
        res1 = straightforward(sym_atoms, nf2=2, ng2=2)
        res2 = straightforward(sym_atoms, nf1=2, ng1=2)
        return min(res1, res2)

    elif name == 'pgg':
        dsym, sym_atoms = symmetrize_bravais('rectangular', atoms)
        return dsym + straightforward(sym_atoms, ng1=2, ng2=2)

    elif name == 'cmm':
        dsym, sym_atoms = symmetrize_bravais('centred rectangular', atoms)
        return dsym + straightforward(sym_atoms, ndf1=2, ndf2=2)

    elif name == 'p4':
        dsym, sym_atoms = symmetrize_bravais('square', atoms)
        return dsym + straightforward(sym_atoms, nr=4)

    elif name == 'p4m':
        dsym, sym_atoms = symmetrize_bravais('square', atoms)
        return dsym + straightforward(sym_atoms, nr=4, nf1=2)

    elif name == 'p4g':
        dsym, sym_atoms = symmetrize_bravais('square', atoms)
        return dsym + straightforward(sym_atoms, nr=4, ncdf=2)

    elif name == 'p3':
        dsym, sym_atoms = symmetrize_bravais('hexagonal', atoms)
        return dsym + straightforward(sym_atoms, nr=3, k=3)

    elif name == 'p3m1':
        dsym, sym_atoms = symmetrize_bravais('hexagonal', atoms)
        return dsym + straightforward(sym_atoms, nr=3, ndf2=2, k=3)

    elif name == 'p31m':
        dsym, sym_atoms = symmetrize_bravais('hexagonal', atoms)
        return dsym + straightforward(sym_atoms, nr=3, nf1=2, k=3)

    elif name == 'p6':
        dsym, sym_atoms = symmetrize_bravais('hexagonal', atoms)
        return dsym + straightforward(sym_atoms, nr=6, k=3)

    elif name == 'p6m':
        dsym, sym_atoms = symmetrize_bravais('hexagonal', atoms)
        return dsym + straightforward(sym_atoms, nr=6, nf1=2, k=3)
    else:
        raise ValueError("Not a valid wallpaper group.")


def get_distances(atoms):

    names = ['p1', 'p2', 'pg', 'pm', 'cm', 'pgg', 'pmg', 'pmm', 'cmm',
             'p4', 'p4g', 'p4m', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']

    results = []
    reductions = find_crystal_reductions(atoms)
    for reduced in reductions:
        #reduced.rmsd
        #print(reduced.factor, reduced.rmsd)
        for name in names:
            dsym = get_wallpaper_distance(name, reduced.atoms)
            results.append((reduced.factor, name, dsym + reduced.rmsd))
    for factor, name, d in results:
        print(factor, name.rjust(4), d)
