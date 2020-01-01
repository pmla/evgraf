import numpy as np
from collections import namedtuple
from .minkowski_reduction import minkowski_reduce
from evgraf.utils import pbc2pbc


StandardizedAtoms = namedtuple('StandardizedAtoms',
                               'atoms op invop barycenter zpermutation')


def standardize_cell(cell):
    Q, R = np.linalg.qr(cell.T)
    indices = np.where(np.diag(R) < 0)[0]
    Q[:, indices] *= -1
    R[indices] *= -1
    if np.sign(np.linalg.det(R)) != np.sign(np.linalg.det(cell)):
        R = -R
        Q = -Q
    return Q, R.T


def standardize(atoms, subtract_barycenter=False):
    zpermutation = np.argsort(atoms.numbers, kind='merge')
    atoms = atoms[zpermutation]
    atoms.set_pbc(pbc2pbc(atoms.pbc))
    atoms.wrap(eps=0)

    barycenter = np.mean(atoms.get_positions(), axis=0)
    if subtract_barycenter:
        atoms.positions -= barycenter

    atoms.set_cell(atoms.cell.complete(), scale_atoms=False)
    rcell, op = minkowski_reduce(atoms.cell, atoms.pbc)
    invop = np.linalg.inv(op)
    atoms.set_cell(rcell, scale_atoms=False)

    atoms.wrap(eps=0)
    return StandardizedAtoms(atoms=atoms, op = op, invop=invop,
                             barycenter=barycenter, zpermutation=zpermutation)
