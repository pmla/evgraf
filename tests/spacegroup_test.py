import pytest
from ase import Atoms
from ase.build import bulk
from evgraf import find_spacegroup_symmetry


TOL = 1E-10


def build_fcc(seed=0, repeat=1):
    #rng = np.random.RandomState(seed=seed)
    return bulk('Cu', 'fcc', a=3.6, cubic=True) * repeat


def build_bcc(seed=0, repeat=1):
    #rng = np.random.RandomState(seed=seed)
    return bulk('Cu', 'bcc', a=3.6, cubic=True) * repeat


def build_perovskite(seed=0, repeat=1):
    a = 4.03646434
    k = a / 2
    atoms = Atoms(symbols='BaTiO3', pbc=True,
                  cell=[a, a, a],
                  positions = [[k, k, k],
                               [0, 0, 0],
                               [k, 0, 0],
                               [0, k, 0],
                               [0, 0, k]])
    return atoms * repeat


lut = {'fcc': 225, 'bcc': 229, 'perovskite': 221}
names = ['fcc', 'bcc', 'perovskite']


@pytest.mark.parametrize("name", names)
def test_implemented(name):
    fbuild = globals()['build_{0}'.format(name)]
    atoms = fbuild(seed=0)
    d = find_spacegroup_symmetry(atoms, lut[name])
    print("d:", d)
    assert d < TOL


if __name__ == "__main__":
    test_implemented('225', 0)
