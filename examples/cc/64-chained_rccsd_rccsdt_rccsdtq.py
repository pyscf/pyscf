#!/usr/bin/env python
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
Running sequential RCCSD -> RCCSDT -> RCCSDTQ calculations in PySCF.

This script demonstrates how to:
    - Initialize RCCSDT and RCCSDTQ calculations from converged lower-order amplitudes.
    - Examine the influence of DIIS acceleration on convergence.
'''

import numpy as np
from pyscf import gto, scf, cc

def run_rccsd_rccsdt_rccsdtq(do_diis=False, do_diis_max_t=False, verbose=0):

    mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()

    # Reference correlation energies for validation
    ref_ccsd_e_corr = -0.1412774234875129
    ref_ccsdt_e_corr = -0.1423431164892638
    ref_ccsdtq_e_corr = -0.1427818949681492

    # RCCSD
    myccsd = cc.CCSD(mf)
    myccsd.conv_tol = 1e-8
    myccsd.conv_tol_normt = 1e-6
    myccsd.verbose = verbose
    myccsd.diis = do_diis
    myccsd.kernel()
    print('RCCSD   e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsd.e_corr, ref_ccsd_e_corr, myccsd.e_corr - ref_ccsd_e_corr))

    # RCCSDT
    # Start with MP2 initial amplitudes
    mp2_tamps = myccsd.init_amps()[1:]
    myccsdt0 = cc.RCCSDT(mf, compact_tamps=True)
    myccsdt0.conv_tol = 1e-8
    myccsdt0.conv_tol_normt = 1e-6
    myccsdt0.verbose = verbose
    myccsdt0.blksize = 3
    myccsdt0.blksize_oooo = 3
    myccsdt0.blksize_oovv = 3
    # einsum_backend: numpy (default) | pyscf | pytblis (recommended)
    # pytblis can be installed via `pip install pytblis==0.05` (See https://github.com/chillenb/pytblis)
    myccsdt0.set_einsum_backend('numpy')
    myccsdt0.diis = do_diis
    myccsdt0.do_diis_max_t = do_diis_max_t
    myccsdt0.kernel(tamps=mp2_tamps)
    print('RCCSDT  e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsdt0.e_corr, ref_ccsdt_e_corr, myccsdt0.e_corr - ref_ccsdt_e_corr))

    # Start with converged (t1, t2) from RCCSD
    rccsd_tamps = (myccsd.t1, myccsd.t2)
    myccsdt1 = cc.RCCSDT(mf, compact_tamps=True)
    myccsdt1.conv_tol = 1e-8
    myccsdt1.conv_tol_normt = 1e-6
    myccsdt1.verbose = verbose
    myccsdt1.blksize = 3
    myccsdt1.blksize_oooo = 3
    myccsdt1.blksize_oovv = 3
    myccsdt1.set_einsum_backend('numpy')
    myccsdt1.diis = do_diis
    myccsdt1.do_diis_max_t = do_diis_max_t
    myccsdt1.kernel(tamps=rccsd_tamps)
    print('RCCSDT  e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsdt1.e_corr, ref_ccsdt_e_corr, myccsdt1.e_corr - ref_ccsdt_e_corr))

    # RCCSDTQ
    # Start with MP2 initial amplitudes
    myccsdtq0 = cc.RCCSDTQ(mf, compact_tamps=True)
    myccsdtq0.conv_tol = 1e-8
    myccsdtq0.conv_tol_normt = 1e-6
    myccsdtq0.verbose = verbose
    myccsdtq0.blksize = 3
    myccsdtq0.set_einsum_backend('numpy')
    myccsdtq0.diis = do_diis
    myccsdtq0.do_diis_max_t = do_diis_max_t
    myccsdtq0.kernel(tamps=mp2_tamps)
    print('RCCSDTQ e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsdtq0.e_corr, ref_ccsdtq_e_corr, myccsdtq0.e_corr - ref_ccsdtq_e_corr))

    # Starting with converged (t1, t2, t3) from RCCSDT
    full_t3 = myccsdt0.tamps_tri2full(myccsdt0.t3)
    rccsdt_tamps = (myccsdt0.t1, myccsdt0.t2, full_t3)
    myccsdtq1 = cc.RCCSDTQ(mf, compact_tamps=True)
    myccsdtq1.conv_tol = 1e-8
    myccsdtq1.conv_tol_normt = 1e-6
    myccsdtq1.verbose = verbose
    myccsdtq1.blksize = 3
    myccsdtq1.set_einsum_backend('numpy')
    myccsdtq1.diis = do_diis
    myccsdtq1.do_diis_max_t = do_diis_max_t
    myccsdtq1.kernel(tamps=rccsdt_tamps)
    print('RCCSDTQ e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsdtq1.e_corr, ref_ccsdtq_e_corr, myccsdtq1.e_corr - ref_ccsdtq_e_corr))

    print()
    print('RCCSD                            iterations: %2d' % myccsd.cycles)
    print('RCCSDT (MP2 starting point)      iterations: %2d' % myccsdt0.cycles)
    print('RCCSDT (RCCSD starting point)    iterations: %2d' % myccsdt1.cycles)
    print('RCCSDTQ (MP2 starting point)     iterations: %2d' % myccsdtq0.cycles)
    print('RCCSDTQ (RCCSDT starting point)  iterations: %2d' % myccsdtq1.cycles)
    print('\n' * 2)
    return

if __name__ == "__main__":
    print('=== RCCSD -> RCCSDT -> RCCSDTQ calculations without DIIS ===')
    do_diis = False
    do_diis_max_t = False
    run_rccsd_rccsdt_rccsdtq(do_diis=do_diis, do_diis_max_t=do_diis_max_t)

    print('=== RCCSD -> RCCSDT -> RCCSDTQ with DIIS (excluding T3/T4 amplitudes) ===')
    do_diis = True
    do_diis_max_t = False
    run_rccsd_rccsdt_rccsdtq(do_diis=do_diis, do_diis_max_t=do_diis_max_t)

    print('=== RCCSD -> RCCSDT -> RCCSDTQ with DIIS (including T3/T4 amplitudes) ===')
    do_diis = True
    do_diis_max_t = True
    run_rccsd_rccsdt_rccsdtq(do_diis=do_diis, do_diis_max_t=do_diis_max_t)
