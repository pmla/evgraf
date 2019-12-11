import numpy as np
from numpy.testing import assert_allclose
from ase.build import bulk, mx2, nanotube
from ase.geometry import permute_axes, find_mic
from evgraf import find_inversion_symmetry


TOL = 1E-10


def check_result(atoms, result):
    assert (atoms.numbers == atoms.numbers[result.permutation]).all()
    delta = result.atoms.get_positions() - atoms.get_positions()
    _, x = find_mic(delta, cell=atoms.cell)
    assert_allclose(np.sqrt(np.mean(x**2)), result.rmsd, atol=TOL)

    inverted = atoms[result.permutation]
    inverted.positions = -inverted.positions + 2 * result.axis
    inverted.wrap(eps=0)

    delta = result.atoms.get_positions() - inverted.get_positions()
    _, x = find_mic(delta, cell=atoms.cell)
    assert_allclose(np.sqrt(np.mean(x**2)), result.rmsd, atol=TOL)


# 3-dimensional: NaCl
size = 2
atoms = bulk("NaCl", "rocksalt", a=5.64) * size
result = find_inversion_symmetry(atoms)
assert result.rmsd < TOL
check_result(atoms, result)

atoms.rattle()
result = find_inversion_symmetry(atoms)
check_result(atoms, result)

# 2-dimensional: MoS2
for i in range(3):
    size = 2
    atoms = mx2(formula='MoS2', size=(size, size, 1))
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)
    result = find_inversion_symmetry(atoms)
    check_result(atoms, result)

# 1-dimensional: carbon nanotube
for i in range(3):
    size = 4
    atoms = nanotube(3, 3, length=size)
    permutation = np.roll(np.arange(3), i)
    atoms = permute_axes(atoms, permutation)

    result = find_inversion_symmetry(atoms)
    assert result.rmsd < TOL
    check_result(atoms, result)

    atoms.rattle()
    result = find_inversion_symmetry(atoms)
    check_result(atoms, result)