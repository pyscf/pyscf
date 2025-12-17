#!/usr/bin/env python
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
Examples of UCCSDT calculations.

This script demonstrates:
    - Consistency between `pyscf.cc.uccsdt` (triangular T3 storage) and `pyscf.cc.uccsdt_highm` (full T3 storage).
    - Restarting UCCSDT from DIIS and checkpoint files.
'''

import numpy as np
from pyscf import gto, scf, cc
from pyscf.lib import chkfile

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz', spin=2)
mf = scf.UHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

# Reference UCCSDT correlation energy
ref_e_corr = -0.167646567229198

print("\n=== Consistency: triangular vs. full T3 storage ===\n")
# '''Check the consistency between UCCSDT in `pyscf.cc.uccsdt` and `pyscf.cc.uccsdt_highm`.'''
#
# UCCSDT with symmetry-compressed (triangular) T3
# Same as cc.uccsdt.UCCSDT
#
mycc1 = cc.UCCSDT(mf, compact_tamps=True)
mycc1.conv_tol = 1e-8
mycc1.conv_tol_normt = 1e-6
mycc1.verbose = 5
# einsum_backend: numpy (default) | pyscf | pytblis (recommended)
# pytblis can be installed via `pip install pytblis==0.05` (See https://github.com/chillenb/pytblis)
mycc1.set_einsum_backend('numpy')
mycc1.incore_complete = True
mycc1.kernel()
print('Triangular UCCSDT e_corr % .12f    Ref % .12f    Diff % .12e' % (
        mycc1.e_corr, ref_e_corr, mycc1.e_corr - ref_e_corr))

#
# UCCSDT with full T3 storage
# Same as cc.uccsdt_highm.UCCSDT
#
mycc2 = cc.UCCSDT(mf, compact_tamps=False)
mycc2.conv_tol = 1e-8
mycc2.conv_tol_normt = 1e-6
mycc2.verbose = 5
mycc2.incore_complete = True
mycc2.kernel()
print('Full-T3 UCCSDT e_corr    % .12f    Ref % .12f    Diff % .12e' % (
        mycc2.e_corr, ref_e_corr, mycc2.e_corr - ref_e_corr))

#
# Compare amplitudes and conversion functions
#
t3_tri = mycc1.t3
t3_full = mycc2.t3
t3_tri_from_t3_full = mycc2.tamps_full2tri(t3_full)
t3_full_from_t3_tri = mycc1.tamps_tri2full(t3_tri)
print("\n--- Amplitude consistency checks ---")
print('total energy difference                       % .10e' % (mycc1.e_tot - mycc2.e_tot))
print('max(abs(t1a difference))                      % .10e' % np.max(np.abs(mycc1.t1[0] - mycc2.t1[0])))
print('max(abs(t1b difference))                      % .10e' % np.max(np.abs(mycc1.t1[1] - mycc2.t1[1])))
print('max(abs(t2aa difference))                     % .10e' % np.max(np.abs(mycc1.t2[0] - mycc2.t2[0])))
print('max(abs(t2ab difference))                     % .10e' % np.max(np.abs(mycc1.t2[1] - mycc2.t2[1])))
print('max(abs(t2bb difference))                     % .10e' % np.max(np.abs(mycc1.t2[2] - mycc2.t2[2])))
print('max(abs(t3aaa_tri - t3aaa_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[0] - t3_tri_from_t3_full[0])))
print('max(abs(t3aaa_full - t3aaa_full_from_tri))    % .10e' % np.max(np.abs(t3_full[0] - t3_full_from_t3_tri[0])))
print('max(abs(t3aab_tri - t3aab_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[1] - t3_tri_from_t3_full[1])))
print('max(abs(t3aab_full - t3aab_full_from_tri))    % .10e' % np.max(np.abs(t3_full[1] - t3_full_from_t3_tri[1])))
print('max(abs(t3bba_tri - t3bba_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[2] - t3_tri_from_t3_full[2])))
print('max(abs(t3bba_full - t3bba_full_from_tri))    % .10e' % np.max(np.abs(t3_full[2] - t3_full_from_t3_tri[2])))
print('max(abs(t3bbb_tri - t3bbb_tri_from_full))     % .10e' % np.max(np.abs(t3_tri[3] - t3_tri_from_t3_full[3])))
print('max(abs(t3bbb_full - t3bbb_full_from_tri))    % .10e' % np.max(np.abs(t3_full[3] - t3_full_from_t3_tri[3])))

# Rerun UCCSDT starting from full T3 amplitudes obtained by expanding the converged triangular T3 amplitudes
tamps_init = [mycc1.t1, mycc1.t2, t3_full_from_t3_tri]
mycc3 = cc.UCCSDT(mf, compact_tamps=False)
mycc3.conv_tol = 1e-8
mycc3.conv_tol_normt = 1e-6
mycc3.verbose = 5
mycc3.incore_complete = True
mycc3.kernel(tamps=tamps_init)
print('UCCSDT e_corr            % .12f    Ref % .12f    Diff % .12e' % (
        mycc3.e_corr, ref_e_corr, mycc3.e_corr - ref_e_corr))

# Rerun UCCSDT starting from triangular T3 amplitudes obtained by compressing the converged full T3 amplitudes
tamps_init = [mycc1.t1, mycc1.t2, t3_tri_from_t3_full]
mycc4 = cc.UCCSDT(mf, compact_tamps=True)
mycc4.conv_tol = 1e-8
mycc4.conv_tol_normt = 1e-6
mycc4.verbose = 5
mycc4.incore_complete = True
mycc4.kernel(tamps=tamps_init)
print('UCCSDT e_corr            % .12f    Ref % .12f    Diff % .12e' % (
        mycc4.e_corr, ref_e_corr, mycc4.e_corr - ref_e_corr))

#
# Restart from DIIS and checkpoint
#
print("\n=== Restart from DIIS file & checkpoint ===\n")

mycc5 = cc.UCCSDT(mf, compact_tamps=False)
mycc5.conv_tol = 1e-8
mycc5.conv_tol_normt = 1e-6
mycc5.verbose = 5
mycc5.diis_file = 'uccsdt_diis.h5'
mycc5.max_cycle = 5
mycc5.kernel()
print('UCCSDT e_corr            % .12f    Ref % .12f    Diff % .12e' % (
        mycc5.e_corr, ref_e_corr, mycc5.e_corr - ref_e_corr))
mycc5.chkfile = 'uccsdt.chk'
mycc5.dump_chk()

# Restore from DIIS
mycc6 = cc.UCCSDT(mf, compact_tamps=False)
mycc6.restore_from_diis_('uccsdt_diis.h5')
mycc6.conv_tol = 1e-8
mycc6.conv_tol_normt = 1e-6
mycc6.verbose = 5
mycc6.kernel(tamps=mycc6.tamps)
print('UCCSDT e_corr            % .12f    Ref % .12f    Diff % .12e' % (
        mycc6.e_corr, ref_e_corr, mycc6.e_corr - ref_e_corr))

# Restore from chk
chk = 'uccsdt.chk'
tamps_init = chkfile.load(chk, 'uccsdt_highm/tamps')
print(len(tamps_init))
mycc7 = cc.UCCSDT(mf, compact_tamps=False)
mycc7.conv_tol = 1e-8
mycc7.conv_tol_normt = 1e-6
mycc7.verbose = 5
mycc7.kernel(tamps=tamps_init)
print('UCCSDT e_corr            % .12f    Ref % .12f    Diff % .12e' % (
        mycc7.e_corr, ref_e_corr, mycc7.e_corr - ref_e_corr))
