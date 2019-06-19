#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Using solvent model in the CASSCF calculations. There are two ways to
incorporate solvent effects. One is allow solvent to response to the
electron density. The other is to freeze the solvent effects for certain
electron density.
'''

from pyscf import gto, scf, mcscf
from pyscf import solvent

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            basis='ccpvdz',
            verbose=4)
mf = scf.RHF(mol).run()

#
# 1. Allow solvent response to CASSCF optimization
#
mc = mcscf.CASSCF(mf, 8, 8)
mc = solvent.ddCOSMO(mc)
# Adjust solvent model by modifying the attribute .with_solvent
mc.with_solvent.eps = 32.613  # methanol dielectric constant
mc.run()

#
# Freeze solvent effects in the CASSCF optimization
#
# Solvent is fully relaxed in the HF calculation.
#
mf = solvent.ddCOSMO(scf.RHF(mol)).run()
mc = mcscf.CASSCF(mf, 4, 4)
# By passing density to solvent model, the solvent potential is frozen at the
# HF level.
mc = solvent.ddCOSMO(mc, dm=mf.make_rdm1())
mc.run()
