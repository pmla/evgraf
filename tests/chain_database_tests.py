import os
import pytest
import warnings
import itertools
import numpy as np
from ase.build import nanotube
from ase.geometry.dimensionality import isolate_components
from ase.db import connect
from evgraf.permute_axes import permute_axes
from evgraf.chains.chain_alignment import calculate_rmsd_chain
from chain_alignment import randomize


TOL = 1E-10

# this test is not really for public consumption
def test_chains_db():

    path = '/home/pete/Dropbox/structural_distance/chains.db'
    if not os.path.exists(path):
        return

    db = connect(path)
    rows = db.select()
    rows = sorted(list(rows), key=lambda x: len(x.toatoms()))

    for i, row in enumerate(rows):
        atoms = row.toatoms()
        res = isolate_components(atoms)
        for atoms in res['1D']:

            n = len(atoms)
            for j, reflect in enumerate(itertools.product([-1, 1], repeat=2)):
                a = randomize(atoms, rotate=1, reflect=reflect)
                b = randomize(atoms, rotate=1)

                print()
                print("it:", i, j, "natoms:", len(a), '/', len(atoms), atoms.numbers)
                res = calculate_rmsd_chain(a, b, allow_rotation=1, allow_reflection=1)
                print(res)
                assert res < TOL

warnings.simplefilter("error")
test_chains_db()
