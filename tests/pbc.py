import pytest
import itertools
from evgraf.pbc import pbc2pbc


def test_true():
    assert (pbc2pbc(True) == [True, True, True]).all()


def test_false():
    assert (pbc2pbc(False) == [False, False, False]).all()


@pytest.mark.parametrize("pbc", itertools.product([False, True], repeat=3))
def test_array(pbc):
    assert (pbc2pbc(pbc) == pbc).all()
