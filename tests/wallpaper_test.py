import pytest
import itertools
import numpy as np
from ase import Atoms
from ase.lattice.hexagonal import Graphene
from evgraf.wallpaper.wallpaper import get_wallpaper_distance
#from ase.build import graphene
#from ase.visualize import view


TOL = 1E-10


# TODO: collect all these disparate rotation matrix functions in one place
def rotate(p, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    U = np.array([[c, -s], [s, c]])
    return p @ U.T


def getf_br():
    return np.array([(-2, -1), (0, -1), (+2, -1), (+2, +1.)]) / 16


def getf_bl():
    return np.array([(-2, -1), (0, -1), (+2, -1), (-2, +1.)]) / 16


def getf_tr():
    return np.array([(-2, +1), (0, +1), (+2, +1), (+2, -1.)]) / 16


def getf_tl():
    return np.array([(-2, +1), (0, +1), (+2, +1), (-2, -1.)]) / 16


def build_atoms(rng, scaled_positions, cell=None):
    n = len(scaled_positions)
    numbers = np.ones(n, dtype=int)
    s = np.zeros((n, 3))
    s[:, :2] = scaled_positions

    c = np.eye(3)
    if cell is not None:
        c[:2, :2] = cell
    c[:2, :2] = rotate(c[:2, :2], rng.uniform(0, 2 * np.pi))

    scaled_positions[:, 0] += rng.uniform(0, 1, n)
    scaled_positions[:, 1] += rng.uniform(0, 1, n)
    atoms = Atoms(numbers=numbers, scaled_positions=s, cell=c,
                 pbc=[True, True, False])
    atoms.wrap(eps=0)
    return atoms


def build_p1(seed=0):
    rng = np.random.RandomState(seed=seed)
    cell = rng.uniform(0, 1, (2, 2))
    scaled_positions = getf_br()
    return build_atoms(rng, scaled_positions, cell)


def build_p2(seed=0):
    rng = np.random.RandomState(seed=seed)
    cell = np.array([[2, 0], [1/2, 3/2]])#rng.uniform(0, 1, (2, 2))
    positions = np.concatenate((getf_br() + (1/4, 1/4) @ cell,
                                getf_tl() + (3/4, 3/4) @ cell
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_pm(seed=0, orientation='horizontal'):
    rng = np.random.RandomState(seed=seed)
    cell = np.diag([2, 1])
    if orientation == 'horizontal':
        scaled_positions = np.concatenate((getf_br() + (1/2, 1/4), getf_tr() + (1/2, 3/4)))
    else:
        scaled_positions = np.concatenate((getf_bl() + (1/4, 1/2), getf_br() + (3/4, 1/2)))
    return build_atoms(rng, scaled_positions, cell)


def build_pg(seed=0, orientation='horizontal'):
    rng = np.random.RandomState(seed=seed)
    cell = np.diag([2, 1])
    if orientation == 'horizontal':
        scaled_positions = np.concatenate((getf_br() + (1/4, 1/2), getf_tr() + (3/4, 1/2)))
    else:
        scaled_positions = np.concatenate((getf_br() + (1/2, 1/4), getf_bl() + (1/2, 3/4)))
    return build_atoms(rng, scaled_positions, cell)


def build_pmm(seed=0):
    rng = np.random.RandomState(seed=seed)
    cell = np.diag([2, 1])
    scaled_positions = np.concatenate((getf_br() + (1/4, 1/4),
                                       getf_bl() + (3/4, 1/4),
                                       getf_tr() + (1/4, 3/4),
                                       getf_tl() + (3/4, 3/4),
                                      ))
    return build_atoms(rng, scaled_positions, cell)


def build_pmg(seed=0, orientation='horizontal'):
    rng = np.random.RandomState(seed=seed)
    cell = np.diag([2, 1])
    if orientation == 'horizontal':
        scaled_positions = np.concatenate((getf_br() + (1/4, 1/4),
                                           getf_tl() + (3/4, 1/4),
                                           getf_tr() + (1/4, 3/4),
                                           getf_bl() + (3/4, 3/4),
                                          ))
    else:
        raise NotImplementedError("not finished")
    return build_atoms(rng, scaled_positions, cell)


def build_pgg(seed=0, orientation='horizontal'):
    rng = np.random.RandomState(seed=seed)
    cell = np.diag([2, 1])
    scaled_positions = np.concatenate((getf_br() + (1/2, 3/8),
                                       getf_tl() + (1/2, 5/8),
                                       getf_tr() + (  0, 1/8),
                                       getf_bl() + (  0, 7/8),
                                      ))
    return build_atoms(rng, scaled_positions, cell)




def build_p4(seed=0):
    rng = np.random.RandomState(seed=seed)
    scaled_positions = np.concatenate((rotate(getf_br(), 0 * np.pi / 2) + (1/4, 1/4),
                                       rotate(getf_br(), 1 * np.pi / 2) + (3/4, 1/4),
                                       rotate(getf_br(), 2 * np.pi / 2) + (3/4, 3/4),
                                       rotate(getf_br(), 3 * np.pi / 2) + (1/4, 3/4),
                                      ))
    return build_atoms(rng, scaled_positions)


def build_p4m(seed=0):
    rng = np.random.RandomState(seed=seed)
    scaled_positions = np.concatenate((rotate(getf_br(), 0 * np.pi / 2)/2 + (3/8, 1/8),
                                       rotate(getf_bl(), 3 * np.pi / 2)/2 + (1/8, 3/8),
                                       rotate(getf_bl(), 0 * np.pi / 2)/2 + (5/8, 1/8),
                                       rotate(getf_br(), 1 * np.pi / 2)/2 + (7/8, 3/8),

                                       rotate(getf_tr(), 0 * np.pi / 2)/2 + (3/8, 7/8),
                                       rotate(getf_br(), 3 * np.pi / 2)/2 + (1/8, 5/8),

                                       rotate(getf_tl(), 0 * np.pi / 2)/2 + (5/8, 7/8),
                                       rotate(getf_tr(), 3 * np.pi / 2)/2 + (7/8, 5/8),
                                      ))
    return build_atoms(rng, scaled_positions)


def build_p4g(seed=0):
    rng = np.random.RandomState(seed=seed)
    scaled_positions = np.concatenate((rotate(getf_bl(), 1 * np.pi / 2)/2 + (1/8, 1/8),
                                       rotate(getf_br(), 0 * np.pi / 2)/2 + (3/8, 3/8),
                                       rotate(getf_br(), 1 * np.pi / 2)/2 + (5/8, 3/8),
                                       rotate(getf_bl(), 2 * np.pi / 2)/2 + (7/8, 1/8),

                                       rotate(getf_bl(), 0 * np.pi / 2)/2 + (1/8, 7/8),
                                       rotate(getf_br(), 3 * np.pi / 2)/2 + (3/8, 5/8),

                                       rotate(getf_tl(), 0 * np.pi / 2)/2 + (5/8, 5/8),
                                       rotate(getf_bl(), 3 * np.pi / 2)/2 + (7/8, 7/8),
                                      ))
    return build_atoms(rng, scaled_positions)


def build_p3(seed=0):
    rng = np.random.RandomState(seed=seed)

    theta = np.radians(60)
    cell = np.array([[1, 0], [np.cos(theta), np.sin(theta)]])

    positions = np.concatenate((rotate(getf_br(), 0 * np.pi / 3)/2 + (1/2, 1/2) @ cell,
                                rotate(getf_br(), 4 * np.pi / 3)/2 + (1/2, 0/2) @ cell,
                                rotate(getf_br(), 2 * np.pi / 3)/2 + (0/2, 1/2) @ cell
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_p3m1(seed=0):
    rng = np.random.RandomState(seed=seed)

    theta = np.radians(60)
    cell = np.array([[1, 0], [np.cos(theta), np.sin(theta)]])

    positions = np.concatenate((rotate(getf_bl(), 0 * np.pi / 3)/2 + (  1, 2/3) @ cell,
                                rotate(getf_br(), 4 * np.pi / 3)/2 + (2/3,   1) @ cell,
                                rotate(getf_bl(), 2 * np.pi / 3)/2 + (1/3,   1) @ cell,
                                rotate(getf_br(), 6 * np.pi / 3)/2 + (1/3, 2/3) @ cell,
                                rotate(getf_bl(), 4 * np.pi / 3)/2 + (2/3, 1/3) @ cell,
                                rotate(getf_br(), 8 * np.pi / 3)/2 + (  1, 1/3) @ cell,
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_p31m(seed=0):
    rng = np.random.RandomState(seed=seed)

    theta = np.radians(60)
    cell = np.array([[1, 0], [np.cos(theta), np.sin(theta)]])

    positions = np.concatenate((rotate(getf_br() - [0, 0.3], 0 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_br() - [0, 0.3], 2 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_br() - [0, 0.3], 4 * np.pi / 3)/2 + (1/3, 1/3) @ cell,

                                rotate(getf_bl() - [0, 0.3], 1 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_bl() - [0, 0.3], 3 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_bl() - [0, 0.3], 5 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_p6(seed=0):
    rng = np.random.RandomState(seed=seed)

    theta = np.radians(60)
    cell = np.array([[1, 0], [np.cos(theta), np.sin(theta)]])

    positions = np.concatenate((rotate(getf_br() - [0, 0.3], 0 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_br() - [0, 0.3], 2 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_br() - [0, 0.3], 4 * np.pi / 3)/2 + (1/3, 1/3) @ cell,

                                rotate(getf_br() - [0, 0.3], 1 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_br() - [0, 0.3], 3 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_br() - [0, 0.3], 5 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_p6m(seed=0):
    rng = np.random.RandomState(seed=seed)

    theta = np.radians(60)
    cell = np.array([[1, 0], [np.cos(theta), np.sin(theta)]])

    positions = np.concatenate((rotate(getf_br() - [0.3, 0.3], 0 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_br() - [0.3, 0.3], 2 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_br() - [0.3, 0.3], 4 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_bl() - [-0.3, 0.3], 0 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_bl() - [-0.3, 0.3], 2 * np.pi / 3)/2 + (1/3, 1/3) @ cell,
                                rotate(getf_bl() - [-0.3, 0.3], 4 * np.pi / 3)/2 + (1/3, 1/3) @ cell,

                                rotate(getf_br() - [0.3, 0.3], 1 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_br() - [0.3, 0.3], 3 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_br() - [0.3, 0.3], 5 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_bl() - [-0.3, 0.3], 1 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_bl() - [-0.3, 0.3], 3 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                                rotate(getf_bl() - [-0.3, 0.3], 5 * np.pi / 3)/2 + (2/3, 2/3) @ cell,
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_cm(seed=0):
    rng = np.random.RandomState(seed=seed)
    cell = np.array([[2, -1], [2, 1]])/2

    positions = np.concatenate((rotate(getf_br(), -0.5 * np.pi / 3)/2 + (2/3, 1/3) @ cell,
                                rotate(getf_tr(), +0.5 * np.pi / 3)/2 + (1/3, 2/3) @ cell,
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


def build_cmm(seed=0):
    rng = np.random.RandomState(seed=seed)
    cell = np.array([[2, -1], [2, 1]])/2

    conv = np.array([cell[0] + cell[1], cell[0] - cell[1]])
    positions = np.concatenate((rotate(getf_br(), -0.5 * np.pi / 3)/2 + (1/3, +1/6) @ conv,
                                rotate(getf_tr(), +0.5 * np.pi / 3)/2 + (1/3, -1/6) @ conv,

                                rotate(getf_bl(), +0.5 * np.pi / 3)/2 + (2/3, +1/6) @ conv,
                                rotate(getf_tl(), -0.5 * np.pi / 3)/2 + (2/3, -1/6) @ conv,
                              ))

    scaled_positions = np.linalg.solve(cell.T, positions.T).T
    return build_atoms(rng, scaled_positions, cell)


names = ['p1', 'p2', 'pg', 'pm', 'cm', 'pgg', 'pmg', 'pmm', 'cmm',
         'p4', 'p4g', 'p4m', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']

# TODO: add tests for vertical/horizontal for (pm, pg, pmg)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("name", names)
def test_implemented(name, seed):
    fbuild = globals()['build_{0}'.format(name)]
    atoms = fbuild(seed)
    if 0:
        from evgraf.utils import plot_atoms
        positions = atoms.get_positions()
        q = positions[:, :2]
        #q = rotate(q, 2 * np.pi / 3)
        #q[:, 0] *= -1
        #q[:, 1] *= -1
        q = np.vstack((q.T, np.zeros(len(q)) )).T
        atoms.set_positions(q)
        atoms.wrap(eps=0)
        plot_atoms(atoms)#, positions)
        asdf
    d = get_wallpaper_distance(name, atoms)
    print("d:", d)
    assert d < TOL


#def test_distances():
#    atoms = build_p4g(0)
#    from evgraf.wallpaper.wallpaper import get_distances
#    get_distances(atoms)

if __name__ == "__main__":
    test_implemented('pg', 0)
    #test_distances()
