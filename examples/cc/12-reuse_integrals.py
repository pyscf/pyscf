#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
To avoid recomputing AO to MO integral transformation, integrals for CCSD,
CCSD(T), CCSD lambda equation etc can be reused.
'''

from pyscf import gto, scf, cc

mol = gto.M(verbose = 4,
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)

#
# CCSD module allows you feed MO integrals
#
eris = mycc.ao2mo()
mycc.kernel(eris=eris)

#
# The same MO integrals can be used in CCSD lambda equation
#
mycc.solve_lambda(eris=eris)

#
# CCSD(T) module requires the same integrals used by CCSD module
#
from pyscf.cc import ccsd_t
ccsd_t.kernel(mycc, eris=eris)

#
# CCSD gradients need zeroth order MO integrals when solving the "relaxed"
# 1-particle density matrix.
#
from pyscf.grad.ccsd import Gradients
grad_e = Gradients(mycc).kernel(eris=eris)  # The electronic part only

