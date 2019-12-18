import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from evgrafcpp import linear_sum_assignment




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





def get_cost_matrix(P, Q, l):

    Cz = cdist(P[:, 2:], Q[:, 2:], metric='sqeuclidean')
    Cz = np.minimum(Cz, cdist(P[:, 2:] - l, Q[:, 2:], metric='sqeuclidean'))
    Cz = np.minimum(Cz, cdist(P[:, 2:] + l, Q[:, 2:], metric='sqeuclidean'))
    #Cz += np.sum(P[:, :2]**2 + Q[:, :2]**2) / len(P)

    vf = Q[:, [1, 0]]
    vf[:, 1] *= -1
    Cx = -2 * np.dot(P[:, :2], vf.T)
    Cy = -2 * np.dot(P[:, :2], Q[:, :2].T)
    #print(Cx)
    #print(Cy)
    #print(Cz)
    #asfd
    return Cx, Cy, (Cz - np.max(Cz))


def concatenate_permutations(perms):

    perm = []
    for p in perms:
        perm.extend(list(p + len(perm)))
    return np.array(perm)


def optimize_and_calc(eindices, Cs, plane):

    obj = 0
    c = np.zeros(2)
    perms = []
    z = 0
    for indices, C in zip(eindices, Cs):

        Cx, Cy, Cz = C
        w = plane[0] * Cx + plane[1] * Cy + Cz
        perm = linear_sum_assignment(w)		# TODO: use maximize=True
        perms.append(perm[1])
        c += (np.sum(Cx[perm]), np.sum(Cy[perm]))
        z += np.sum(Cz[perm])
        #print(plane)
        #print(Cx)
        #print(Cy)
        #print(Cz)

    #if np.linalg.norm(c
    c *= (np.sum(c**2) + z**2) / np.sum(c**2)
    perm = concatenate_permutations(perms)
    #print(perm, c, np.linalg.norm(c))
    #print()
    return c, perm


def get_facet_normals(ps, simplices):

    midpoint = np.mean(ps, axis=0)
    ts = ps[simplices]
    deltas = ts[:, 1] - ts[:, 0]

    norms = np.linalg.norm(deltas, axis=1)
    deltas = deltas / norms[:, np.newaxis]

    planes = np.array([-deltas[:, 1], deltas[:, 0]]).T

    points = ps[simplices[:, 0]]
    dots = np.einsum('ij,ij->i', points - midpoint, planes)
    indices = np.where(dots < 0)[0]
    planes[indices] *= -1
    return planes


def get_hull(n, eindices, Cs):

    perms = np.array([range(n)])
    ps = [(np.trace(Cx + Cz), np.trace(Cy + Cz)) for Cx, Cy, Cz in Cs]
    ps = np.sum(ps, axis=0).reshape((1, 2))

    p0, perm0 = optimize_and_calc(eindices, Cs, [0, +1])
    p1, perm1 = optimize_and_calc(eindices, Cs, [0, -1])
    if (perm0 == perm1).all():
        return np.array([p0, p1]), np.array([perm0, perm1])

    ps = np.array([p0, p1])
    perms = np.array([perm0, perm1])

    planes = get_facet_normals(ps, np.array([[0, 1]]))
    plane = planes[0]

    for sign in [-1, 1]:
        p, perm = optimize_and_calc(eindices, Cs, sign * plane)
        ps = np.concatenate((ps, [p]))
        perms = np.concatenate((perms, [perm]))

    rank = np.linalg.matrix_rank(ps - ps[0])
    if rank < 2:
        return ps, perms

    simplices = ConvexHull(ps).simplices
    extreme = np.unique(simplices)
    ps = ps[extreme]
    perms = perms[extreme]

    finished = set()
    while 1:
        simplices = ConvexHull(ps).simplices
        facets = get_facet_normals(ps, simplices)

        rounded = np.round(1E12 * facets).astype(np.int)
        facets = [f for f, r in zip(facets, rounded)
                  if tuple(r) not in finished]
        if len(facets) == 0:
            break

        plane = facets[0]
        sint, cost = plane
        p, perm = optimize_and_calc(eindices, Cs, [sint, cost])

        ps = np.concatenate((ps, [p]))
        perms = np.concatenate((perms, [perm]))
        simplices = ConvexHull(ps).simplices
        extreme = np.unique(simplices)
        new = len(ps) - 1 in extreme
        ps = ps[extreme]
        perms = perms[extreme]

        if not new:
            rounded = tuple(np.round(1E12 * plane).astype(np.int))
            finished.add(rounded)

    midpoint = np.mean(ps, axis=0)
    angles = [np.arctan2(e[1], e[0]) for e in ps - midpoint]
    indices = np.argsort(angles)
    return ps[indices], perms[indices]


def register_clever(P, Q, numbers, cell_length):

    eindices = [np.where(numbers == element)[0]
                for element in np.unique(numbers)]
    Cs = []
    for indices in eindices:
        Cs.append(get_cost_matrix(P[indices], Q[indices], cell_length))

    ps, perms = get_hull(len(P), eindices, Cs)
    #print(perms)
    norms = np.linalg.norm(ps, axis=1)
    #print("norms:", norms)
    index = np.argmax(norms)
    return perms[index]
