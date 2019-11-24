#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
QM/MM charges + implicit solvent model
'''

import numpy
from pyscf import gto, scf, qmmm, solvent

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * -.1

#
# The order to apply solvent model and QM/MM charges does not affect results
#
# ddCOSMO-QMMM-SCF
#
mf = scf.RHF(mol)
mf = qmmm.mm_charge(mf, coords, charges)
mf = solvent.ddCOSMO(mf)
mf.run()

#
# QMMM-ddCOSMO-SCF
#
mf = scf.RHF(mol)
mf = solvent.ddCOSMO(mf)
mf = qmmm.mm_charge(mf, coords, charges)
mf.run()
