# evgraf
Symmetrization of crystal structures


Currently supports translational and inversion symmetries only.


### Installation:

To install the module with pip (recommended):
```
pip install --user evgraf
```

To install directly from the git repository:
```
pip install --user git+https://github.com/pmla/evgraf
```

To do a manual build and installation:
```
python3 setup.py build
python3 setup.py install --user
```

### Usage:
We can quantify the breaking of inversion symmetry in BaTiO3.
First we create the crystal structure:
```
>>> from ase import Atoms
>>> atoms = Atoms(symbols='BaTiO3', pbc=True, cell=np.diag([4.002, 4.002, 4.216]),
...               positions=np.array([[0.000, 0.000, 0.085],
...                                   [2.001, 2.001, 2.272],
...                                   [2.001, 2.001, 4.092],
...                                   [0.000, 2.001, 2.074],
...                                   [2.001, 0.000, 2.074]]))
```
We call the inversion symmetry analysis function:
```
>>> from evgraf import find_inversion_symmetry
>>> result = find_inversion_symmetry(atoms)
```
We can view the inversion-symmetrized structure:
```
>>> from ase.visualize import view
>>> view(result.atoms)
```
The degree of symmetry breaking is given by the root-mean-square distance between the input structure and the symmetrized structure:
```
>>> result.rmsd
0.10115255804971046
```
We also obtain the inversion axis:
```
>>> result.axis
array([2.001 , 2.001 , 2.1194])
```


### Information
`evgraf` is written by Peter M. Larsen.  The software is provided under the MIT license.
