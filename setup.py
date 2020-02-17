import os
import numpy
from setuptools import Extension, find_packages, setup


evgrafcpp_module = Extension(
    'evgrafcpp',
    ['src/crystalline.cpp',
     'src/crystalline.h',
     'src/evgrafcpp_module.cpp',
     'src/lup_decomposition.cpp',
     'src/lup_decomposition.h',
     'src/matrix_vector.cpp',
     'src/matrix_vector.h',
     'src/rectangular_lsap.cpp',
     'src/rectangular_lsap.h',
     'src/wrap_positions.cpp',
     'src/wrap_positions.h',
    ],
    include_dirs=[os.path.join(numpy.get_include(), 'numpy'),
                  'src',
    ],
    language='c++'
)

major_version = 0
minor_version = 1
subminor_version = 6
version = '%d.%d.%d' % (major_version, minor_version, subminor_version)

setup(name='evgraf',
      python_requires='>=3.5.0',
      version=version,
      description='Geometric analysis of crystal structures',
      author='P. M. Larsen',
      author_email='pmla@fysik.dtu.dk',
      url='https://github.com/pmla/evgraf',
      ext_modules=[evgrafcpp_module],
      install_requires=['numpy',
                        'ase>=3.19'],
      packages=find_packages()
      )
