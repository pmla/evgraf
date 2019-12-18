import auguste


names = ['oblique', 'rectangular', 'centred rectangular', 'hexagonal', 'square']

def symmetrize_lattice(name, cell):
    """This is a temporary hack until 2D Bravais lattices are supported in
    auguste."""

    name = name.lower()
    if name.startswith('primitive '):
        name = name[len('primitive '):]

    if name not in names:
        raise ValueError("unrecognized lattice type")

    lut = {'oblique': 'primitive triclinic',
           'rectangular': 'primitive orthorhombic',
           'centred rectangular': 'base-centred orthorhombic',
           'hexagonal': 'primitive hexagonal',
           'square': 'primitive tetragonal'}

    import numpy as np
    lengths = np.linalg.norm(cell, axis=1)
    lmax = max(lengths[:2])
    cell = cell.copy()
    cell[2] *= 10 * lmax
    return auguste.symmetrize_lattice(cell, lut[name],
                                      return_correspondence=True)


def symmetrize_bravais(name, atoms):
    dsym, sym, Q, L = symmetrize_lattice(name, atoms.cell)
    print(name)
    print(dsym)
    print(atoms.cell)
    print(sym)
    print(L)

    sym_atoms = atoms.copy()
    sym_atoms.set_cell(sym, scale_atoms=True)
    sym_atoms.set_cell((sym_atoms.cell.T @ L).T, scale_atoms=False)
    sym_atoms.wrap(eps=0)
    return dsym, sym_atoms
