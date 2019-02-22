#!/usr/bin/env python
#
# Author: Xiao Wang <xiaowang314159@gmail.com>
#
"""
Showing use of general EOM-CCSD with K-point sampling.
"""

import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd


cell = gto.Cell()
cell.verbose = 4
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

# KRHF
kpts = cell.make_kpts([1,1,3])
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
euhf = kmf.kernel()

# KGCCSD
mycc = cc.KGCCSD(kmf)
egccsd = mycc.kernel()

# EOM-KGCCSD
myeomip = eom_kgccsd.EOMIP(mycc)
eip, vip = myeomip.kernel(nroots=2)

myeomea = eom_kgccsd.EOMEA(mycc)
eea, vea = myeomea.kernel(nroots=2)
