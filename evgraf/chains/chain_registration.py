import numpy as np
import heapq
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def minimum_angle(a, b):

    d = (b - a) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def rotation_matrix(sint, cost):

    norm = np.linalg.norm([sint, cost])
    if norm < 1E-10:
        return np.eye(2)

    U = np.array([[cost, -sint], [sint, cost]])
    return U / norm


def optimal_rotation(P, Q):

    assert P.shape[1] == 2
    assert Q.shape[1] == 2

    A = np.dot(P.T, Q)
    sint = A[0, 1] - A[1, 0]
    cost = A[0, 0] + A[1, 1]
    return rotation_matrix(sint, cost)


def calculate_actual_cost(P, Q):

    U = optimal_rotation(P[:, :2], Q[:, :2])
    RP = np.dot(P[:, :2], U.T)

    nrmsdsq = np.sum((RP - Q[:, :2])**2)
    return nrmsdsq, U


def calculate_nrmsdsq(P, Q, cell_length):

    dz = P[:, 2] - Q[:, 2]
    dzsq = np.min([(dz + i * cell_length)**2 for i in range(-1, 2)], axis=0)

    nrmsdsq, U = calculate_actual_cost(P[:, :2], Q[:, :2])
    return nrmsdsq + np.sum(dzsq), U


def concatenate_permutations(perms):

    perm = []
    for p in perms:
        perm.extend(list(p + len(perm)))
    return np.array(perm)


def get_cost_matrix(P, Q, l):

    Cz = cdist(P[:, 2:], Q[:, 2:], metric='sqeuclidean')
    Cz = np.minimum(Cz, cdist(P[:, 2:] - l, Q[:, 2:], metric='sqeuclidean'))
    Cz = np.minimum(Cz, cdist(P[:, 2:] + l, Q[:, 2:], metric='sqeuclidean'))
    Cz += np.sum(P[:, :2]**2 + Q[:, :2]**2) / len(P)

    vf = Q[:, [1, 0]]
    vf[:, 1] *= -1
    Cx = -2 * np.dot(P[:, :2], vf.T)
    Cy = -2 * np.dot(P[:, :2], Q[:, :2].T)
    return (Cx, Cy, Cz)


def calculate_intercept(coeffs0, coeffs1):

    a, b, c = coeffs0 - coeffs1

    K = np.sqrt(a * a + b * b)
    if K < 1E-10:
        return None

    # sin(x + phi) = -c / K
    # y = x + phi
    phi = np.arctan2(b, a)
    y = np.arcsin(-c / K)
    return y - phi


def optimize_and_calc(eindices, Cs, theta):

    sint = np.sin(theta)
    cost = np.cos(theta)

    perms = []
    c = np.zeros(3)
    for indices, C in zip(eindices, Cs):

        Cx, Cy, Cz = C
        w = sint * Cx + cost * Cy + Cz

        perm = linear_sum_assignment(w)
        perms.append(perm[1])
        c += (np.sum(Cx[perm]), np.sum(Cy[perm]), np.sum(Cz[perm]))

    perm = concatenate_permutations(perms)
    return perm, c


def compute_lb_cost(eindices, Cs, dthetas, interval):

    t0, t1 = interval

    obj = 0
    perms = []
    for indices, C, dtheta in zip(eindices, Cs, dthetas):

        Cx, Cy, Cz = C

        db0 = minimum_angle(dtheta, t0)
        db1 = minimum_angle(dtheta, t1)
        indices = np.where(db1 < db0)
        boundary = np.full_like(dtheta, t0)
        boundary[indices] = t1

        indices = np.where((dtheta < t0) | (dtheta > t1))
        dtheta[indices] = boundary[indices]
        assert (dtheta >= t0).all()
        assert (dtheta <= t1).all()

        sint = np.sin(dtheta)
        cost = np.cos(dtheta)

        w = np.multiply(sint, Cx) + np.multiply(cost, Cy) + Cz
        perm = linear_sum_assignment(w)
        perms.append(perm[1])
        obj += np.sum(w[perm])

    return obj, concatenate_permutations(perms)


def _register(P, Q, eindices, cell_length):

    n = len(P)

    Cs = []
    for indices in eindices:
        Cs.append(get_cost_matrix(P[indices], Q[indices], cell_length))

    dthetas = []
    for indices in eindices:
        p = P[indices]
        q = Q[indices]
        thetasP = np.arctan2(p[:, 1], p[:, 0])
        thetasQ = np.arctan2(q[:, 1], q[:, 0])
        dtheta = np.array([thetasQ - e for e in thetasP]) % (2 * np.pi)
        dthetas.append(dtheta)

    seen = set()
    pdict = {}
    coeffs = {}

    intervals = []
    best = (float("inf"), None, None)

    for theta in [0, np.pi]:
        perm, c = optimize_and_calc(eindices, Cs, theta)
        t = tuple(perm)
        coeffs[t] = c
        seen.add(t)

        obj, U = calculate_nrmsdsq(P, Q[perm], cell_length)
        best = min(best, (obj, perm, U), key=lambda x: x[0])

        heapq.heappush(intervals, (obj, theta, theta + np.pi))

        pdict[theta] = t
        if theta == 0:
            pdict[2 * np.pi] = t

    if pdict[0] == pdict[np.pi]:
        return best

    while intervals:

        interval = sorted(intervals.pop(0)[1:])
        theta0, theta1 = interval
        perm0, perm1 = pdict[theta0], pdict[theta1]

        intercept = calculate_intercept(coeffs[perm0], coeffs[perm1])
        if intercept is None:
            continue

        theta = intercept % (2 * np.pi)

        if minimum_angle(theta, theta0) < 1E-9:
            continue
        if minimum_angle(theta, theta1) < 1E-9:
            continue

        # if theta < theta0 or theta > theta1:
        #    print(theta0, theta1, theta)
        #    raise Exception("interval failure")

        if n > 4 and abs(theta1 - theta0) < np.deg2rad(50):
            lbobj, lbperm = compute_lb_cost(eindices, Cs, dthetas, interval)
            if lbobj > best[0]:
                continue

            obj, U = calculate_nrmsdsq(P, Q[lbperm], cell_length)
            best = min(best, (obj, lbperm, U), key=lambda x: x[0])

        perm, c = optimize_and_calc(eindices, Cs, theta)
        t = tuple(perm)
        if t not in seen:
            pdict[theta] = t
            coeffs[t] = c
            seen.add(t)

            obj, U = calculate_nrmsdsq(P, Q[perm], cell_length)
            best = min(best, (obj, perm, U), key=lambda x: x[0])

            heapq.heappush(intervals, (obj, theta0, theta))
            heapq.heappush(intervals, (obj, theta, theta1))

    return best


def register_chain(P, Q, eindices, cell_length, best=None):

    n = len(P)
    if best is None or 1:
        permutation = np.arange(n)
        best = (float("inf"), permutation, np.eye(2))
    else:
        obj, permutation, U = best
        obj = obj**2 * n
        best = (obj, permutation, U)
    prev = best[0]

    best = _register(P, Q, eindices, cell_length)
    obj, permutation, U = best

    if obj != prev:
        rmsd = np.sqrt(obj / len(P))

        check, _ = calculate_nrmsdsq(P, Q[permutation], cell_length)
        assert abs(check - obj) < 1E-10
    else:
        rmsd = float("inf")

    return rmsd, permutation, U
