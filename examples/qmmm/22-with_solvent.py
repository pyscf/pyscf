#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
QM/MM charges + implicit solvent model
'''

import numpy
from pyscf import gto, qmmm, solvent
from pyscf.data import radii

# load all modeuls
from pyscf import __all__

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * .1
mm_atoms = [('C', c) for c in coords]
mm_mol = qmmm.create_mm_mol(mm_atoms, charges)

# Make a giant system include both QM and MM particles
qmmm_mol = mol + mm_mol

# The solvent model is based on the giant system
sol = solvent.ddCOSMO(qmmm_mol)

# According to Lipparini's suggestion in issue #446
sol.radii_table = radii.VDW

#
# The order to apply solvent model and QM/MM charges does not affect results
#
# ddCOSMO-QMMM-SCF
#
mf = mol.RHF()
mf = mf.QMMM(coords, charges)
mf = mf.ddCOSMO(sol)
mf.run()

#
# QMMM-ddCOSMO-SCF
#
mf = mol.RHF()
mf = mf.ddCOSMO(sol)
mf = mf.QMMM(coords, charges)
mf.run()
