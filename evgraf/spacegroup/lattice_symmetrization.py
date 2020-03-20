import numpy as np
import auguste


names = ['oblique', 'rectangular', 'centred rectangular', 'hexagonal', 'square']


def symmetrize_lattice(name, cell):
    name = name.lower()
    if name in auguste.names:
        return auguste.symmetrize_lattice(cell, name,
                                          return_correspondence=True)

    """This is a temporary hack until 2D Bravais lattices are supported in
    auguste."""
    if name.startswith('primitive '):
        name = name[len('primitive '):]

    if name not in names and name not in auguste.names:
        raise ValueError("unrecognized lattice type")

    lut = {'oblique': 'primitive triclinic',
           'rectangular': 'primitive orthorhombic',
           'centred rectangular': 'base-centred orthorhombic',
           'hexagonal': 'primitive hexagonal',
           'square': 'primitive tetragonal'}

    lengths = np.linalg.norm(cell, axis=1)
    lmax = max(lengths[:2])
    cell = cell.copy()
    cell[2] *= 10 * lmax
    return auguste.symmetrize_lattice(cell, lut[name],
                                      return_correspondence=True)


def symmetrize_bravais(name, atoms):
    dsym, sym, Q, L = symmetrize_lattice(name, atoms.cell)
    #print(name)
    #print(dsym)
    #print(atoms.cell)
    #print(sym)
    #print(L)

    sym_atoms = atoms.copy()
    sym_atoms.set_cell(sym, scale_atoms=True)
    sym_atoms.set_cell((sym_atoms.cell.T @ L).T, scale_atoms=False)
    sym_atoms.set_cell(sym_atoms.cell @ Q.T, scale_atoms=True)
    sym_atoms.wrap(eps=0)

    cell = sym_atoms.cell
    #print(cell.array)
    #print(name)

    standard = ['primitive triclinic', 'primitive monoclinic',
                'primitive orthorhombic', 'primitive tetragonal',
                'primitive hexagonal', 'primitive cubic']

    if name in standard:
        conventional = np.copy(cell)

    elif 'base-centred' in name:
        conventional = np.zeros_like(cell)
        conventional[0] = cell[0] - cell[1]
        conventional[1] = cell[0] + cell[1]
        conventional[2] = cell[2]

    elif 'body-centred' in name:
        conventional = np.zeros_like(cell)
        conventional[0] = cell[1] + cell[2]
        conventional[1] = cell[0] + cell[2]
        conventional[2] = cell[0] + cell[1]

    elif 'face-centred' in name:
        conventional = np.zeros_like(cell)
        conventional[0] = cell[1] + cell[2] - cell[0]
        conventional[1] = cell[2] + cell[0] - cell[1]
        conventional[2] = cell[0] + cell[1] - cell[2]

    elif name == 'primitive rhombohedral':
        conventional = np.zeros_like(cell)
        conventional[0] = cell[0] - cell[1]
        conventional[1] = cell[1] - cell[2]
        conventional[2] = cell[0] + cell[1] + cell[2]

    '''
    print(cell.array)
    print(conventional.T)

    v = np.cross(conventional[2], conventional[0])
    v /= np.linalg.norm(v)
    v *= np.linalg.norm(conventional[0])
    a = conventional[0]
    theta = np.deg2rad(120)
    b = np.cos(theta) * a + np.sin(theta) * v

    print(np.linalg.solve(cell, conventional[0]))
    print(np.linalg.solve(cell, conventional[2]))
    print(np.linalg.solve(cell, b))
    asdf

    print(np.dot(conventional[0], conventional[2]))
    print(np.dot(conventional[1], conventional[2]))
    dot = np.dot(conventional[0], conventional[1]) / np.linalg.norm(conventional[0]) / np.linalg.norm(conventional[1])
    print(np.rad2deg(np.arccos(dot)))

    def angle(a, b):
        dot = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
        return np.rad2deg(np.arccos(dot))

    print(angle(conventional[0], conventional[2]))
    print(angle(conventional[1], conventional[2]))
    print(angle(conventional[0], conventional[1]))
    asdf
    '''

    return dsym, sym_atoms, conventional
