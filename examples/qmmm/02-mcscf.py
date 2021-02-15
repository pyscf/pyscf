#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run MCSCF with background charges.
'''

import numpy
from pyscf import gto, scf, mcscf, qmmm

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
            basis='3-21g',
            verbose=4)

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * -.1

#
# There are two ways to add background charges to MCSCF method.
# The recommended one is to initialize it in SCF calculation. The MCSCF
# calculation takes the information from SCF objects.
#
mf = qmmm.mm_charge(scf.RHF(mol), coords, charges).run()

mc = mcscf.CASSCF(mf, 6, 6)
mc.run()

mc = mcscf.CASCI(mf, 6, 6)
mc.run()

#
# The other method is to patch the MCSCF object with the background charges.
# Note: it updates the underlying SCF object inplace.
#
mo_init = mf.mo_coeff

mf = scf.RHF(mol)
mc = mcscf.CASSCF(mf, 6, 6)
mc = qmmm.mm_charge(mc, coords, charges)
mc.run(mo_init)

mf = scf.RHF(mol)
mc = mcscf.CASCI(mf, 6, 6)
mc = qmmm.mm_charge(mc, coords, charges)
mc.run(mo_init)
