#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
To avoid recomputing AO to MO integral transformation, integrals for CCSD,
CCSD(T), CCSD lambda equation etc can be reused.
'''

from pyscf import gto, scf, cc
from pyscf.cc import ccsd

mol = gto.M(verbose = 4,
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)

#
# CCSD module allows you feed MO integrals
#
eris = mycc.ao2mo()
# Initialize the amplitudes from MP2 and perform two explicit CCSD updates
# with the same integrals. These are iteration energies, not converged energies.
emp2, t1, t2 = mycc.init_amps(eris)
print('Initial MP2 correlation energy:', emp2)
for iteration in range(2):
    t1, t2 = mycc.update_amps(t1, t2, eris)
    print('CCSD iteration', iteration + 1, 'correlation energy:',
          mycc.energy(t1, t2, eris))
# Continue to convergence from the updated amplitudes without rebuilding ERIs.
mycc.kernel(t1=t1, t2=t2, eris=eris)

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

# The conventional CCSD implementation can also build its four-index MO
# integrals from density-fitting tensors. Keep the converged RHF orbitals,
# and clear cached AO integrals so ao2mo uses the fitted tensors instead.
# For routine DF-CCSD, prefer the dedicated implementation in 21-dfccsd.py.
dfmf = mf.density_fit(auxbasis='weigend')
dfmf._eri = None
dfcc = ccsd.CCSD(dfmf)
df_eris = dfcc.ao2mo()
emp2, df_t1, df_t2 = dfcc.init_amps(df_eris)
df_t1, df_t2 = dfcc.update_amps(df_t1, df_t2, df_eris)
print('DF integral MP2 and first CCSD iteration energies:',
      emp2, dfcc.energy(df_t1, df_t2, df_eris))
