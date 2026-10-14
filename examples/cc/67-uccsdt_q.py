#!/usr/bin/env python

'''
Conventional UCCSDT[Q] and UCCSDT(Q) for canonical, real UHF orbitals.

The second returned correction already includes [Q].  Neither correction
changes the UCCSDT energy stored on the coupled-cluster object.
'''

import numpy as np
from pyscf import gto, scf, cc, lib


# The correction uses the PySCF OpenMP thread setting, also initialized by
# OMP_NUM_THREADS. The default correction backend is C/OpenMP.
lib.num_threads(4)

# Open-shell doublet with five correlated electrons.  This small basis is
# useful for illustrating the interface, rather than quantitative chemistry.
mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1)
mf = scf.UHF(mol).set(conv_tol=1e-13).run()
assert mf.converged

corrections = []
for compact in (True, False):
    mycc = cc.UCCSDT(mf, frozen=0, compact_tamps=compact)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    # Reuse the bare integrals for both stages; ccsdt_q otherwise repeats AO2MO.
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    assert mycc.converged

    e_ccsdt = mycc.e_tot
    e_q_bracket, e_q_paren = mycc.ccsdt_q(eris=eris)
    assert mycc.e_tot == e_ccsdt
    corrections.append((e_q_bracket, e_q_paren))

    print('compact_tamps =', compact)
    print('UCCSDT total energy       % .12f' % e_ccsdt)
    print('[Q] correction           % .12f' % e_q_bracket)
    print('Additional (Q) term      % .12f' % (e_q_paren - e_q_bracket))
    print('Complete (Q) correction  % .12f' % e_q_paren)
    print('UCCSDT(Q) total energy   % .12f' % (e_ccsdt + e_q_paren))
    del eris

np.testing.assert_allclose(corrections[0], corrections[1], atol=1e-10,
                           rtol=1e-9)
