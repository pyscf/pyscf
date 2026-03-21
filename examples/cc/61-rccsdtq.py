#!/usr/bin/env python
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
Examples of RCCSDTQ calculations.

This script demonstrates:
    - Consistency between `pyscf.cc.rccsdtq` (triangular T4 storage) and `pyscf.cc.rccsdtq_highm` (full T4 storage).
    - Restarting RCCSDTQ from DIIS and checkpoint files.
    - Invariance of RCCSDTQ energy under orbital rotations.
'''

import numpy as np
from pyscf import gto, scf, cc
from pyscf.lib import chkfile

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='631g')
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

# Reference RCCSDTQ correlation energy
ref_e_corr = -0.1427818949681492

print("\n=== Consistency: triangular vs. full T4 storage ===\n")
# '''Check the consistency between RCCSDTQ in `pyscf.cc.rccsdtq` and `pyscf.cc.rccsdtq_highm`.'''
#
# RCCSDTQ with symmetry-compressed (triangular) T4
# Same as cc.rccsdtq.RCCSDTQ
#
mycc1 = cc.RCCSDTQ(mf, compact_tamps=True)
mycc1.conv_tol = 1e-8
mycc1.conv_tol_normt = 1e-6
mycc1.verbose = 5
# einsum_backend: numpy (default) | pyscf | pytblis (recommended)
# pytblis can be installed via `pip install pytblis==0.05` (See https://github.com/chillenb/pytblis)
mycc1.set_einsum_backend('numpy')
mycc1.incore_complete = True
mycc1.kernel()
print('Triangular-T4 RCCSDTQ e_corr % .12f    Ref % .12f    Diff % .12e' % (
        mycc1.e_corr, ref_e_corr, mycc1.e_corr - ref_e_corr))

#
# RCCSDTQ with full T4 storage
# Same as cc.rccsdtq_highm.RCCSDTQ
#
mycc2 = cc.RCCSDTQ(mf, compact_tamps=False)
mycc2.conv_tol = 1e-8
mycc2.conv_tol_normt = 1e-6
mycc2.verbose = 5
mycc2.incore_complete = True
mycc2.kernel()
print('Full-T4 RCCSDQ e_corr       % .12f    Ref % .12f    Diff % .12e' % (
        mycc2.e_corr, ref_e_corr, mycc2.e_corr - ref_e_corr))

#
# Compare amplitudes and conversion functions
#
t4_tri = mycc1.t4
t4_full = mycc2.t4
t4_tri_from_t4_full = mycc2.tamps_full2tri(t4_full)
t4_full_from_t4_tri = mycc1.tamps_tri2full(t4_tri)
print("\n--- Amplitude consistency checks ---")
print('total energy difference                    % .10e' % (mycc1.e_tot - mycc2.e_tot))
print('max(abs(t1 difference))                    % .10e' % np.max(np.abs(mycc1.t1 - mycc2.t1)))
print('max(abs(t2 difference))                    % .10e' % np.max(np.abs(mycc1.t2 - mycc2.t2)))
print('max(abs(t3 difference))                    % .10e' % np.max(np.abs(mycc1.t3 - mycc2.t3)))
print('max(abs(t4_tri - t4_tri_from_t4_full))     % .10e' % np.max(np.abs(t4_tri - t4_tri_from_t4_full)))
print('max(abs(t4_full - t4_full_from_t4_tri))    % .10e' % np.max(np.abs(t4_full - t4_full_from_t4_tri)))

# Rerun RCCSDTQ starting from full T4 amplitudes obtained by expanding the converged triangular T4 amplitudes
tamps_init = [mycc1.t1, mycc1.t2, mycc1.t3, t4_full_from_t4_tri]
mycc3 = cc.RCCSDTQ(mf, compact_tamps=False)
mycc3.conv_tol = 1e-8
mycc3.conv_tol_normt = 1e-6
mycc3.verbose = 5
mycc3.incore_complete = True
mycc3.kernel(tamps=tamps_init)
print('RCCSDTQ e_corr              % .12f    Ref % .12f    Diff % .12e' % (
        mycc3.e_corr, ref_e_corr, mycc3.e_corr - ref_e_corr))

# Rerun RCCSDTQ starting from triangular T4 amplitudes obtained by compressing the converged full T4 amplitudes
tamps_init = [mycc1.t1, mycc1.t2, mycc1.t3, t4_tri_from_t4_full]
mycc4 = cc.RCCSDTQ(mf, compact_tamps=True)
mycc4.conv_tol = 1e-8
mycc4.conv_tol_normt = 1e-6
mycc4.verbose = 5
mycc4.incore_complete = True
mycc4.kernel(tamps=tamps_init)
print('RCCSDTQ e_corr              % .12f    Ref % .12f    Diff % .12e' % (
        mycc4.e_corr, ref_e_corr, mycc4.e_corr - ref_e_corr))

#
# Restart from DIIS and checkpoint
#
print("\n=== Restart from DIIS file & checkpoint ===\n")

mycc5 = cc.RCCSDTQ(mf, compact_tamps=False)
mycc5.conv_tol = 1e-8
mycc5.conv_tol_normt = 1e-6
mycc5.verbose = 5
mycc5.diis_file = 'rccsdtq_diis.h5'
mycc5.max_cycle = 5
mycc5.kernel()
print('RCCSDTQ e_corr              % .12f    Ref % .12f    Diff % .12e' % (
        mycc5.e_corr, ref_e_corr, mycc5.e_corr - ref_e_corr))
mycc5.chkfile = 'rccsdtq.chk'
mycc5.dump_chk()

# Restore from DIIS
mycc6 = cc.RCCSDTQ(mf, compact_tamps=False)
mycc6.restore_from_diis_('rccsdtq_diis.h5')
mycc6.conv_tol = 1e-8
mycc6.conv_tol_normt = 1e-6
mycc6.verbose = 5
mycc6.kernel(tamps=mycc6.tamps)
print('RCCSDTQ e_corr              % .12f    Ref % .12f    Diff % .12e' % (
        mycc6.e_corr, ref_e_corr, mycc6.e_corr - ref_e_corr))

# Restore from chk
chk = 'rccsdtq.chk'
tamps_init = chkfile.load(chk, 'rccsdtq_highm/tamps')
mycc7 = cc.RCCSDTQ(mf, compact_tamps=False)
mycc7.conv_tol = 1e-8
mycc7.conv_tol_normt = 1e-6
mycc7.verbose = 5
mycc7.kernel(tamps=tamps_init)
print('RCCSDTQ e_corr              % .12f    Ref % .12f    Diff % .12e' % (
        mycc7.e_corr, ref_e_corr, mycc7.e_corr - ref_e_corr))

#
# Orbital-rotation invariance test
#
print("\n=== Orbital-rotation invariance ===\n")

# Random orthogonal rotations in occupied and virtual spaces
np.random.seed(42)
nocc, nmo = int(np.sum(mf.mo_occ) // 2), mf.mo_coeff.shape[1]
A = np.eye(nocc) + np.random.randn(nocc, nocc) * 0.02
A += A.T
_, R = np.linalg.eigh(A)
mf.mo_coeff[:, :nocc] = mf.mo_coeff[:, :nocc] @ R
A = np.eye(nmo - nocc) + np.random.randn(nmo - nocc, nmo - nocc) * 0.02
A += A.T
_, R = np.linalg.eigh(A)
mf.mo_coeff[:, nocc:] = mf.mo_coeff[:, nocc:] @ R

mycc8 = cc.RCCSDTQ(mf, compact_tamps=False)
mycc8.conv_tol = 1e-8
mycc8.conv_tol_normt = 1e-6
mycc8.verbose = 5
mycc8.level_shift = 0.2
mycc8.max_cycle = 200
mycc8.kernel()
print('Rotated-MO RCCSDTQ e_corr   % .12f    Ref % .12f    Diff % .12e' % (
        mycc8.e_corr, ref_e_corr, mycc8.e_corr - ref_e_corr))
