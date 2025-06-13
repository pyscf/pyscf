#!/usr/bin/env python

'''
MP2 with non-canonical orbitals
'''

import pyscf
from pyscf.mp.mp2 import MP2


mol = pyscf.M(atom='''
O    0.   0.       0.
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
basis='cc-pvdz', verbose=4)

mf = mol.RKS().run()
mf = mf.to_hf()

# Starting from non-canonical HF orbitals, MP2 is solved iteratively
pt = mf.MP2().run()
