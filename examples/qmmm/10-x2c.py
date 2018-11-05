#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
X2C and QM/MM charges can be used together
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

#
# The order to apply X2C and QMMM matters. X2C has to be called first then the
# QMMM charges. In this case, the non-relativistic QMMM charges is added to
# the system, i.e., picture change was not applied to the QMMM charges.
#
mf = scf.RHF(mol).x2c()
mf = qmmm.mm_charge(mf, coords, charges).run()

#
# If the other order was called, picture change is supposed to applied to QMMM
# charges as well.  However, the current implementation does not support this
# feature. This order only counts the interactions between nucleus and QMMM
# charges. The interactions between electrons and QMMM charges are excluded.
#
mf = scf.RHF(mol)
mf = qmmm.mm_charge(mf, coords, charges)
mf = mf.x2c().run()
print(mf.energy_elec())

# In the current implementation, the electronic part of the code above is
# equivalent to
mf = scf.RHF(mol)
mf = mf.x2c().run()
print(mf.energy_elec())
