from collections import namedtuple
from ase.spacegroup import get_bravais_class
import ase.spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError


Spacegroup = namedtuple('Spacegroup', 'lattice symmetries centrosymmetric')


def build_setting(sg_number, setting):
    sg = ase.spacegroup.Spacegroup(i, setting=setting)
    if sg.centrosymmetric:
        signs = [1, -1]
    else:
        signs = [1]

    symmetries = []
    for s in signs:
        for r, t in zip(sg.rotations, sg.translations):
            symmetries.append((s * r, s * t))

    return Spacegroup(get_bravais_class(i).longname, symmetries,
                      sg.centrosymmetric)


spacegroups = {}
for i in range(1, 231):
    result = []
    for setting in [1, 2]:
        try:
            group = build_setting(i, setting)
            result.append(group)
        except SpacegroupNotFoundError:
            break

    spacegroups[i] = result
