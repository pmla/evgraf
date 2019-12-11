import os
import numpy
from setuptools import Extension, find_packages, setup


major_version = 0
minor_version = 1
subminor_version = 1
version = '%d.%d.%d' % (major_version, minor_version, subminor_version)

setup(name='evgraf',
      python_requires='>3.5.0',
      version=version,
      description='Geometric analysis of crystal structures',
      author='P. M. Larsen',
      author_email='pmla@fysik.dtu.dk',
      url='https://github.com/pmla/evgraf',
      ext_modules=[],
      install_requires=['numpy'],
      packages=['evgraf']
      )
