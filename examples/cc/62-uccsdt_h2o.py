#!/usr/bin/env python

"""Compare compact and full-storage UCCSDT for triplet H2O/6-31G.

Check correlation energies against a reference and compare singles, doubles,
and triples amplitudes, including full/triangular triples conversions.
See 62-uccsdt.py for additional restart examples.
"""

import numpy as np
from pyscf import gto, scf
from pyscf.cc.uccsdt import UCCSDT
from pyscf.cc.uccsdt_highm import UCCSDT as UCCSDThm

mol = gto.M(atom="O 0 0 0; H 0 -0.757, 0.587; H 0 0.757, 0.587", basis='631g', verbose=3, spin=2)
mf = scf.UHF(mol)
mf.level_shift = 0.0
mf.conv_tol = 1e-14
mf.max_cycle = 1000
mf.kernel()
ref_ecorr = -0.1092563391390793
mycc = UCCSDT(mf, frozen=0)
mycc.set_einsum_backend('numpy')
mycc.conv_tol = 1e-12
mycc.conv_tol_normt = 1e-10
mycc.max_cycle = 100
mycc.verbose = 5
mycc.do_diis_max_t = True
mycc.incore_complete = True
mycc.kernel()
print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_ecorr, mycc.e_corr - ref_ecorr))
print()

# comparison with the high-memory version
mycc2 = UCCSDThm(mf, frozen=0)
mycc2.set_einsum_backend('numpy')
mycc2.conv_tol = 1e-12
mycc2.conv_tol_normt = 1e-10
mycc2.max_cycle = 100
mycc2.verbose = 5
mycc2.do_diis_max_t = True
mycc2.incore_complete = True
mycc2.kernel()
print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc2.e_corr, ref_ecorr, mycc2.e_corr - ref_ecorr))
print()

t3_tri = mycc.tamps[2]
t3_full = mycc2.tamps[2]
t3_tri_from_full = mycc2.tamps_full2tri(t3_full)
t3_full_from_tri = mycc.tamps_tri2full(t3_tri)

print('energy difference                             % .10e' % (mycc.e_tot - mycc2.e_tot))
print('max(abs(t1a difference))                      % .10e' % np.max(np.abs(mycc.t1[0] - mycc2.t1[0])))
print('max(abs(t1b difference))                      % .10e' % np.max(np.abs(mycc.t1[1] - mycc2.t1[1])))
print('max(abs(t2aa difference))                     % .10e' % np.max(np.abs(mycc.t2[0] - mycc2.t2[0])))
print('max(abs(t2ab difference))                     % .10e' % np.max(np.abs(mycc.t2[1] - mycc2.t2[1])))
print('max(abs(t2bb difference))                     % .10e' % np.max(np.abs(mycc.t2[2] - mycc2.t2[2])))
print('max(abs(t3aaa_tri - t3aaa_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[0] - t3_tri_from_full[0])))
print('max(abs(t3aaa_full - t3aaa_full_from_tri))    % .10e' % np.max(np.abs(t3_full[0] - t3_full_from_tri[0])))
print('max(abs(t3aab_tri - t3aab_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[1] - t3_tri_from_full[1])))
print('max(abs(t3aab_full - t3aab_full_from_tri))    % .10e' % np.max(np.abs(t3_full[1] - t3_full_from_tri[1])))
print('max(abs(t3bba_tri - t3bba_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[2] - t3_tri_from_full[2])))
print('max(abs(t3bba_full - t3bba_full_from_tri))    % .10e' % np.max(np.abs(t3_full[2] - t3_full_from_tri[2])))
print('max(abs(t3bbb_tri - t3bbb_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[3] - t3_tri_from_full[3])))
print('max(abs(t3bbb_full - t3bbb_full_from_tri))    % .10e' % np.max(np.abs(t3_full[3] - t3_full_from_tri[3])))
