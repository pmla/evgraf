import numpy as np


def pbc2pbc(pbc):
    newpbc = np.empty(3, bool)
    newpbc[:] = pbc
    return newpbc
