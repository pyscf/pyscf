#!/usr/bin/env python
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
Examples of RCCSDT(Q) calculations.

This script demonstrates:
    - Consistency of the [Q] and (Q) energy corrections on top of RCCSDT between calculations
        using full and compact T3 storage.
'''

from pyscf import gto, scf, cc

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz')
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

# Reference CCSDT correlation energy, and [Q] and (Q) energy correction
ref_e_corr = -0.2188784733230733
ref_e_q_bracket = -0.0005026220700017348
ref_e_q_paren = -0.0005490746450078632

mycc1 = cc.RCCSDT(mf, compact_tamps=True)
mycc1.conv_tol = 1e-10
mycc1.conv_tol_normt = 1e-8
mycc1.verbose = 5
# einsum_backend: numpy (default) | pyscf | pytblis (recommended)
# pytblis can be installed via `pip install pytblis==0.05` (See https://github.com/chillenb/pytblis)
mycc1.set_einsum_backend('pyscf')
mycc1.incore_complete = True
mycc1.kernel()
e_q_bracket, e_q_paren = mycc1.ccsdt_q()
print('Triangular RCCSDT e_corr % .12f    Ref % .12f    Diff % .12e' % (
        mycc1.e_corr, ref_e_corr, mycc1.e_corr - ref_e_corr))
print('Triangular RCCSDT [Q]    % .12f    Ref % .12f    Diff % .12e' % (
        e_q_bracket, ref_e_q_bracket, e_q_bracket - ref_e_q_bracket))
print('Triangular RCCSDT (Q)    % .12f    Ref % .12f    Diff % .12e' % (
        e_q_paren, ref_e_q_paren, e_q_paren - ref_e_q_paren))

#
# RCCSDT with full T3 storage
# Same as cc.rccsdt_highm.RCCSDT
#
mycc2 = cc.RCCSDT(mf, compact_tamps=False)
mycc2.conv_tol = 1e-10
mycc2.conv_tol_normt = 1e-8
mycc2.verbose = 5
mycc2.incore_complete = True
mycc2.kernel()
q_bracket2, q_paren2 = mycc2.ccsdt_q()
print('Full-T3 RCCSDT e_corr    % .12f    Ref % .12f    Diff % .12e' % (
        mycc2.e_corr, ref_e_corr, mycc2.e_corr - ref_e_corr))
print('Full-T3 RCCSDT [Q]       % .12f    Ref % .12f    Diff % .12e' % (
        q_bracket2, ref_e_q_bracket, q_bracket2 - ref_e_q_bracket))
print('Full-T3 RCCSDT (Q)       % .12f    Ref % .12f    Diff % .12e' % (
        q_paren2, ref_e_q_paren, q_paren2 - ref_e_q_paren))
