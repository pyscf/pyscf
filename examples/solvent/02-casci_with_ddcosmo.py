#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Using solvent model in the CASCI calculations. There are two ways to
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
            basis='ccpvdz')
mf = scf.RHF(mol).run()

#
# 1. Allow solvent response to CASCI optimization
#
mc = mcscf.CASCI(mf, 4, 4)
mc = solvent.ddCOSMO(mc)
mc.run()

#
# Freeze solvent effects in the CASSCF optimization
#
# In this case, we need to decide which HF reference to use in the ddCOSMO-CASCI
# calculation. The fully relaxed solvent at HF level is preferred.
#
mf = solvent.ddCOSMO(scf.RHF(mol)).run()
mc = mcscf.CASCI(mf, 4, 4)
# By passing density to solvent model, the solvent potential is frozen at the
# HF level.
mc = solvent.ddCOSMO(mc, dm=mf.make_rdm1())
mc.run()
