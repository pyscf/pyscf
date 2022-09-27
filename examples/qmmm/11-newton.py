#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Second order SCF with QM/MM charges is available
'''

import numpy
from pyscf import gto, scf, qmmm

mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
            ''',
            basis='3-21g')

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * -.1

mf = qmmm.mm_charge(scf.RHF(mol), coords, charges)

mf.run()

mf = mf.newton().run()

