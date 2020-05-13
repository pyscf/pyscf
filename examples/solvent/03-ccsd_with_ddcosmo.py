#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Using solvent model in the CCSD calculations. When applying solvent model,
this example has convergence issue.
'''

from pyscf import gto, scf, cc
from pyscf import solvent

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            basis='ccpvdz',
            verbose=3)
mf = scf.RHF(mol).run()

#
# 1. Allow solvent response to CASSCF optimization
#
mycc = solvent.ddCOSMO(cc.CCSD(mf))
# Adjust solvent model by modifying the attribute .with_solvent
mycc.with_solvent.eps = 32.613  # methanol dielectric constant
mycc.run()

#
# Freeze solvent effects in the CASSCF optimization
#
# In this case, we need to decide which HF reference to use in the ddCOSMO-CC
# calculation. The fully relaxed solvent at HF level is preferred.
#
mf = solvent.ddCOSMO(scf.RHF(mol)).run()
# By passing density to solvent model, the solvent potential is frozen at the
# HF level.
mycc = solvent.ddCOSMO(cc.CCSD(mf), dm=mf.make_rdm1())
mycc.run()
