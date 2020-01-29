import pytest
import itertools
import numpy as np
from ase import Atoms
from ase.build import bulk
from evgraf.wallpaper.spacegroup import get_spacegroup_distance


TOL = 1E-10


def build_fcc(seed=0):
    rng = np.random.RandomState(seed=seed)
    return bulk('Cu', 'fcc', a=3.6, cubic=True) * 2


def build_bcc(seed=0):
    rng = np.random.RandomState(seed=seed)
    return bulk('Cu', 'bcc', a=3.6, cubic=True) * 2


lut = {'fcc': 225, 'bcc': 229}
names = ['fcc', 'bcc']


@pytest.mark.parametrize("name", names)
def test_implemented(name):
    fbuild = globals()['build_{0}'.format(name)]
    atoms = fbuild(seed=0)
    d = get_spacegroup_distance(lut[name], atoms)
    print("d:", d)
    assert d < TOL


if __name__ == "__main__":
    test_implemented('225', 0)
