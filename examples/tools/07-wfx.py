#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import pyscf
from pyscf.tools import wfx_format

'''
Write orbitals in WFX format
'''

#
# Molecule
#
mol = pyscf.M(
    atom = '''
O     0    0       0
H     0    -.757   .587
H     0    .757    .587''',
    basis = 'cc-pvdz',
    symmetry = True
)
mf = mol.HF().run()
wfx_format.from_scf(mf)

#
# Crystal
#
cell = pyscf.M(
    atom = '''
C 0.,  0.,  0.
C 0.8917,  0.8917,  0.8917''',
    a = '''0.      1.7834  1.7834
           1.7834  0.      1.7834
           1.7834  1.7834  0.    ''',
    basis = 'gth-szv',
    pseudo = 'gth-pade'
)
mf = cell.KRKS(kpts=cell.make_kpts([2,1,1])).run()
wfx_format.from_scf(mf)
