import numpy as np
from numpy.testing import assert_allclose
from evgraf.standardization import standardize
from ase.build import bulk


TOL = 1E-10


def test_standarization():

    # prepare input
    size = 3
    atoms = bulk("NaCl", "rocksalt", a=5.64) * size
    L = np.array([[1, 0, 0], [1, 1, 0], [2, 0, 1]])
    assert np.linalg.det(L) == 1
    atoms.set_cell(L @ atoms.cell, scale_atoms=False)
    atoms.set_positions(atoms.get_positions() + 1.23456789)
    atoms.wrap(eps=0)

    # standardize
    std = standardize(atoms, subtract_barycenter=True)

    # check atomic numbers
    assert (std.atoms.numbers == np.sort(atoms.numbers)).all()
    assert (np.sort(atoms.numbers) == std.atoms.numbers).all()
    assert (atoms.numbers[std.zpermutation] == std.atoms.numbers).all()

    # check unit cell
    assert_allclose(np.linalg.det(std.invop), np.linalg.det(L), atol=TOL)
    assert_allclose(std.invop @ std.atoms.cell, atoms.cell, atol=TOL)

    # check atomic positions
    inverse_permutation = np.argsort(std.zpermutation)
    reverted = std.atoms[inverse_permutation]
    reverted.set_cell(std.invop @ reverted.cell, scale_atoms=False)
    reverted.set_positions(reverted.get_positions() + std.barycenter)
    reverted.wrap(eps=0)
    p1 = reverted.get_positions()
    p0 = atoms.get_positions()
    assert_allclose(p0, p1, atol=TOL)
