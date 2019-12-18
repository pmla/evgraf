import pytest
import itertools
import numpy as np
from numpy.testing import assert_allclose
from ase.build import nanotube
from evgraf.utils import permute_axes
from evgraf.chains.chain_registration2d import register_clever, calculate_nrmsdsq


TOL = 1E-10


def rotation_matrix(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    U = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert_allclose(np.linalg.det(U), 1, atol=TOL)
    return U


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n", range(1, 7))
def test_regular(n, seed):
    rng = np.random.RandomState(seed=0)

    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.array([np.sin(thetas), np.cos(thetas), np.zeros(n)]).T

    phi = rng.uniform(0, 2 * np.pi)
    U = rotation_matrix(phi)
    Q = P @ U.T

    permutation = register_clever(P, Q, np.ones(n, dtype=int), 1)

    cost, U = calculate_nrmsdsq(P[permutation], Q, 1)
    assert cost < TOL
    s, c = U[1]
    dphi = np.arctan2(s, c)
    k = n * (dphi - phi) / (2 * np.pi)
    assert_allclose(k, round(k), atol=TOL)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n", range(1, 7))
def test_random(n, seed):
    rng = np.random.RandomState(seed=0)

    P = rng.normal(size=(n, 3))
    Q = rng.normal(size=(n, 3))
    P[:, 2] = 0
    Q[:, 2] = 0

    permutation = register_clever(P, Q, np.ones(n, dtype=int), 1)
    cost, u = calculate_nrmsdsq(P[permutation], Q, 1)
    U = np.eye(3)
    U[:2, :2] = u

    check = np.linalg.norm(P[permutation] @ U.T - Q)**2
    assert_allclose(cost, check, atol=TOL)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n", range(1, 9))
def test_regular_spiral(n, seed):
    rng = np.random.RandomState(seed=0)

    cell_length = 1
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.array([np.sin(thetas), np.cos(thetas), np.linspace(0, cell_length, n)]).T

    phi = 0*rng.uniform(0, 2 * np.pi)
    U = rotation_matrix(phi)
    Q = P @ U.T

    permutation = register_clever(P, Q, np.ones(n, dtype=int), cell_length)
    print(permutation)

    cost, U = calculate_nrmsdsq(P[permutation], Q, cell_length)
    s, c = U[1]
    dphi = np.arctan2(s, c)
    print(np.degrees(dphi))
    assert cost < TOL
    #k = n * (dphi - phi) / (2 * np.pi)
    #assert_allclose(k, round(k), atol=TOL)


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    test_regular_spiral(7, 0)
