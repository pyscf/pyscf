#!/usr/bin/env python
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
Running sequential UCCSD -> UCCSDT calculations in PySCF.

This script demonstrates how to:
    - Initialize a UCCSDT calculation using converged T-amplitudes from a preceding UCCSD run.
    - Understand and handle the difference in T2 amplitude conventions between UCCSD and UCCSDT implementations.
'''

import numpy as np
from pyscf import gto, scf, cc

def run_uccsd_uccsdt(do_diis=False, do_diis_max_t=False, verbose=0):

    mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz', spin=2)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()

    # Reference correlation energies for validation
    ref_ccsd_e_corr = -0.1659782005526982
    ref_ccsdt_e_corr = -0.1676465515305336

    # UCCSD
    myccsd = cc.CCSD(mf)
    myccsd.conv_tol = 1e-8
    myccsd.conv_tol_normt = 1e-6
    myccsd.verbose = verbose
    myccsd.diis = do_diis
    myccsd.kernel()
    print('RCCSD   e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsd.e_corr, ref_ccsd_e_corr, myccsd.e_corr - ref_ccsd_e_corr))

    # UCCSDT
    # Start with MP2 initial amplitudes
    mp2_tamps = myccsd.init_amps()[1:]
    # NOTE: The definition of t2ab in UCCSD differs from that in UCCSDT.
    # We must transpose indices (1 <-> 2) to ensure consistency.
    (mp2_t1a, mp2_t1b), (mp2_t2aa, mp2_t2ab, mp2_t2bb) = mp2_tamps
    mp2_t2ab = mp2_t2ab.transpose(0, 2, 1, 3)
    mp2_tamps = (mp2_t1a, mp2_t1b), (mp2_t2aa, mp2_t2ab, mp2_t2bb)

    myccsdt0 = cc.UCCSDT(mf, compact_tamps=False)
    myccsdt0.conv_tol = 1e-8
    myccsdt0.conv_tol_normt = 1e-6
    myccsdt0.verbose = verbose
    myccsdt0.blksize_o_aaa = 3
    myccsdt0.blksize_v_aaa = 8
    myccsdt0.blksize_o_aab = 3
    myccsdt0.blksize_v_aab = 8
    # einsum_backend: numpy (default) | pyscf | pytblis (recommended)
    # pytblis can be installed via `pip install pytblis==0.05` (See https://github.com/chillenb/pytblis)
    myccsdt0.set_einsum_backend('numpy')
    myccsdt0.diis = do_diis
    myccsdt0.do_diis_max_t = do_diis_max_t
    myccsdt0.kernel(tamps=mp2_tamps)
    print('UCCSDT  e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsdt0.e_corr, ref_ccsdt_e_corr, myccsdt0.e_corr - ref_ccsdt_e_corr))

    # Use converged (t1, t2) from UCCSD as the starting amplitudes
    t2aa, t2ab, t2bb = myccsd.t2
    # NOTE: Convert t2ab convention from UCCSD to UCCSDT
    t2ab = t2ab.transpose(0, 2, 1, 3)
    uccsd_tamps = (myccsd.t1, (t2aa, t2ab, t2bb))
    myccsdt1 = cc.UCCSDT(mf, compact_tamps=False)
    myccsdt1.conv_tol = 1e-8
    myccsdt1.conv_tol_normt = 1e-6
    myccsdt1.verbose = verbose
    myccsdt1.blksize_o_aaa = 3
    myccsdt1.blksize_v_aaa = 8
    myccsdt1.blksize_o_aab = 3
    myccsdt1.blksize_v_aab = 8
    myccsdt1.set_einsum_backend('numpy')
    myccsdt1.diis = do_diis
    myccsdt1.do_diis_max_t = do_diis_max_t
    myccsdt1.kernel(tamps=uccsd_tamps)
    print('UCCSDT  e_corr % .12f    Ref % .12f    Diff % .12e' % (
            myccsdt1.e_corr, ref_ccsdt_e_corr, myccsdt1.e_corr - ref_ccsdt_e_corr))

    print()
    print('UCCSD                            iterations: %2d' % myccsd.cycles)
    print('UCCSDT (MP2 starting point)      iterations: %2d' % myccsdt0.cycles)
    print('UCCSDT (UCCSD starting point)    iterations: %2d' % myccsdt1.cycles)
    print('\n' * 2)
    return

if __name__ == "__main__":
    print('=== UCCSD -> UCCSDT calculations without DIIS ===')
    do_diis = False
    do_diis_max_t = False
    run_uccsd_uccsdt(do_diis=do_diis, do_diis_max_t=do_diis_max_t)

    print('=== UCCSD -> UCCSDT with DIIS (excluding T3 amplitudes) ===')
    do_diis = True
    do_diis_max_t = False
    run_uccsd_uccsdt(do_diis=do_diis, do_diis_max_t=do_diis_max_t)

    print('=== UCCSD / UCCSDT with DIIS (including T3 amplitudes) ===')
    do_diis = True
    do_diis_max_t = True
    run_uccsd_uccsdt(do_diis=do_diis, do_diis_max_t=do_diis_max_t)
