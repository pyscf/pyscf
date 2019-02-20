#!/usr/bin/env python

'''
Showing use of UHF-based EOM-CCSD with K-point sampling.
'''

import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import eom_kccsd_uhf as eom_kuccsd


cell = gto.Cell()
cell.verbose = 7
cell.unit = 'B'
cell.mesh = [5, 5, 5]

#
# Hydrogen crystal
#
cell.a = np.eye(3) * 4.2
cell.basis = 'sto-3g'
cell.atom = '''
    H 0.000000000000   0.000000000000   0.000000000000
    H 0.000000000000   0.000000000000   1.400000000000
    '''

#
# Helium crystal
#
# cell.atom = '''
# He 0.000000000000   0.000000000000   0.000000000000
# He 1.685068664391   1.685068664391   1.685068664391
# '''
# cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
# cell.a = '''
# 0.000000000, 3.370137329, 3.370137329
# 3.370137329, 0.000000000, 3.370137329
# 3.370137329, 3.370137329, 0.000000000
# '''

cell.build()

# KUHF
kpts = cell.make_kpts([1,1,2])
kmf = scf.KUHF(cell, kpts=kpts, exxdiv=None)
euhf = kmf.kernel()

# KUCCSD
mycc = cc.KUCCSD(kmf)
euccsd = mycc.kernel()

# EOM-KUCCSD
myeom = eom_kuccsd.EOMIP(mycc)
e, v = myeom.kernel(nroots=1)