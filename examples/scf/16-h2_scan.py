#!/usr/bin/env python

'''
Scan H2 molecule dissociation curve.
See also 30-scan_pes.py
'''

import numpy
from pyscf import scf
from pyscf import gto

ehf = []
dm = None

for b in numpy.arange(0.7, 4.01, 0.1):
    mol = gto.M(atom=[["H", 0., 0., 0.],
                      ["H", 0., 0., b ]], basis='ccpvdz', verbose=0)
    mf = scf.RHF(mol)
    ehf.append(mf.kernel(dm))
    dm = mf.make_rdm1()

print('R     E(HF)')
for i, b in enumerate(numpy.arange(0.7, 4.01, 0.1)):
    print('%.2f  %.8f' % (b, ehf[i]))
