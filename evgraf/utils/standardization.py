import numpy as np
from collections import namedtuple
from .minkowski_reduction import minkowski_reduce


def standardize(atoms, subtract_barycenter=False):
    zpermutation = np.argsort(atoms.numbers, kind='merge')
    atoms = atoms[zpermutation]
    atoms.wrap(eps=0)

    barycenter = np.mean(atoms.get_positions(), axis=0)
    if subtract_barycenter:
        atoms.positions -= barycenter

    rcell, op = minkowski_reduce(atoms.cell, atoms.cell.any(1) & atoms.pbc)
    invop = np.linalg.inv(op)
    atoms.set_cell(rcell, scale_atoms=False)
    atoms.wrap(eps=0)

    StandardizedAtoms = namedtuple('StandardizedAtoms',
                                   'atoms invop barycenter zpermutation')
    return StandardizedAtoms(atoms=atoms, invop=invop, barycenter=barycenter,
                             zpermutation=zpermutation)
