#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example of using polarizable embedding model in the mean-field
calculations. This example requires the cppe library

https://github.com/maxscheurer/cppe
arXiv:1804.03598

The CPPE library needs to be built from sources (according to the CPPE document):

mkdir build && cd build && cmake -DENABLE_PYTHON_INTERFACE=ON .. && make

If successfully built, find the directory where the file cppe.*.so locates
then put the directory in PYTHONPATH.

The potfile required by this example can be generated with the script
04-pe_potfile_from_pyframe.py
'''

import pyscf
from pyscf import solvent

mol = pyscf.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)
mf = mol.UKS()
mf.xc = 'b3lyp'
mf = solvent.PE(mf, '4NP_in_water/4NP_in_water.pot')
mf.run()

