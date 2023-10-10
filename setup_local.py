from setuptools import setup
import os
pname='lpt4py'+os.environ['USER']
setup(name=pname,
      version='0.1',
      description='Initial conditions and Lagrangian perturbation theory in Python with mpi4py',
      url='http://github.com/marcelo-alvarez/lpt4py',
      author='Marcelo Alvarez',
      license_files = ('LICENSE',),
      packages=[pname],
      zip_safe=False)
