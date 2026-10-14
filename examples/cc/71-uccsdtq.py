#!/usr/bin/env python

'''
Fully iterative UCCSDTQ for a small open-shell molecule.

Start with converged UCCSDT amplitudes and zero T4.  Compact UCCSDTQ keeps
both T3 and T4 packed; the high-memory variant expands both ranks.  Both
storage choices give the same energy.  This minimal basis illustrates the
interface; it is not a quantitative thermochemistry calculation.
'''

import copy
import numpy as np
from pyscf import gto, scf, cc, lib

lib.num_threads(4)

mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1,
            verbose=4)
mf = scf.UHF(mol).set(conv_tol=1e-13).run()
assert mf.converged

# Freeze no orbitals here: five correlated electrons make the distinction
# between the iterative quadruples and the (Q) correction visible.
ccsdt = cc.UCCSDT(mf, frozen=0, compact_tamps=True)
ccsdt.conv_tol = 1e-11
ccsdt.conv_tol_normt = 1e-9
eris = ccsdt.ao2mo()
ccsdt.kernel(eris=eris)
assert ccsdt.converged
_, e_paren = ccsdt.ccsdt_q(eris=eris)

results = []
for compact in (True, False):
    mycc = cc.UCCSDTQ(mf, frozen=0, compact_tamps=compact)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    # T4 is excluded from DIIS by default.  T1--T3 still participate in DIIS.
    mycc.do_diis_max_t = False
    mycc.max_cycle = 100
    mycc.kernel(tamps=copy.deepcopy(ccsdt.tamps), eris=eris)
    assert mycc.converged
    results.append(mycc.e_tot)
    print('compact_tamps =', compact)
    print('UCCSDT total energy       % .12f' % ccsdt.e_tot)
    print('UCCSDT(Q) total energy    % .12f' % (ccsdt.e_tot + e_paren))
    print('UCCSDTQ total energy      % .12f' % mycc.e_tot)
    print('Iterative Q - (Q)         % .12e' % (mycc.e_tot - ccsdt.e_tot - e_paren))
    print('T4 block shapes:', [x.shape for x in mycc.t4])

np.testing.assert_allclose(results[0], results[1], atol=1e-10, rtol=0)
