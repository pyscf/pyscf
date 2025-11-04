#!/usr/bin/env python
#
# Author: Yu Jin <yjin@flatironinstitute.org>
#

'''
Verify consistency between RCCSDT and UCCSDT on a closed-shell system.

This example demonstrates:
    - Agreement between RCCSDT and UCCSDT energies and amplitudes for a closed-shell reference
    - Conversion between RHF-based and UHF-based T-amplitudes (t1, t2, t3)
    - Restarting RCCSDT/UCCSDT using converted amplitudes
'''

import numpy as np
from pyscf import gto, scf, cc

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz')
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()

mycc = cc.RCCSDT(mf, compact_tamps=False).set(conv_tol=1e-10, conv_tol_normt=1e-8, verbose=5).run()
print('RCCSDT correlation energy % .12f    Ref % .12f    Diff % .12e' % (
        mycc.e_corr, -0.2188784727114157, mycc.e_corr - -0.2188784727114157))

mf_uhf = scf.addons.convert_to_uhf(mf)
myucc = cc.UCCSDT(mf_uhf, compact_tamps=False).set(conv_tol=1e-10, conv_tol_normt=1e-8, verbose=5).run()
print('UCCSDT correlation energy % .12f    Ref % .12f    Diff % .12e' % (
        myucc.e_corr, -0.2188784727114157, myucc.e_corr - -0.2188784727114157))

t1_rhf, t2_rhf, t3_rhf = mycc.tamps
t1_uhf, t2_uhf, t3_uhf = myucc.tamps

# Transform UCCSDT amplitudes to RHF representation
tamps_uhf2rhf = myucc.tamps_uhf2rhf(myucc.tamps)
# Transform RCCSDT amplitudes to UHF representation
tamps_rhf2uhf = myucc.tamps_rhf2uhf(mycc.tamps)
t1_rhf, t2_rhf, t3_rhf = mycc.tamps
t1_uhf2rhf, t2_uhf2rhf, t3_uhf2rhf = tamps_uhf2rhf
t1_uhf, t2_uhf, t3_uhf = myucc.tamps
t1_rhf2uhf, t2_rhf2uhf, t3_rhf2uhf = tamps_rhf2uhf
# Compare correlation energy and cluster amplitudes between RCCSDT and UCCSDT
print('RCCSDT etot - UCCSDT etot             % .10e' % (mycc.e_tot - myucc.e_tot))
print('max(abs(t1_rhf - t1_uhf2rhf))         % .10e' % np.max(np.abs(t1_rhf - t1_uhf2rhf)))
print('max(abs(t2_rhf - t2_uhf2rhf))         % .10e' % np.max(np.abs(t2_rhf - t2_uhf2rhf)))
print('max(abs(t3_rhf - t3_uhf2rhf))         % .10e' % np.max(np.abs(t3_rhf - t3_uhf2rhf)))
print()
print('max(abs(t1a_uhf - t1a_rhf2uhf))       % .10e' % np.max(np.abs(t1_uhf[0] - t1_rhf2uhf[0])))
print('max(abs(t1b_uhf - t1b_rhf2uhf))       % .10e' % np.max(np.abs(t1_uhf[1] - t1_rhf2uhf[1])))
print('max(abs(t2aa_uhf - t2aa_rhf2uhf))     % .10e' % np.max(np.abs(t2_uhf[0] - t2_rhf2uhf[0])))
print('max(abs(t2ab_uhf - t2ab_rhf2uhf))     % .10e' % np.max(np.abs(t2_uhf[1] - t2_rhf2uhf[1])))
print('max(abs(t2bb_uhf - t2bb_rhf2uhf))     % .10e' % np.max(np.abs(t2_uhf[2] - t2_rhf2uhf[2])))
print('max(abs(t3aaa_uhf - t3aaa_rhf2uhf))   % .10e' % np.max(np.abs(t3_uhf[0] - t3_rhf2uhf[0])))
print('max(abs(t3aab_uhf - t3aab_rhf2uhf))   % .10e' % np.max(np.abs(t3_uhf[1] - t3_rhf2uhf[1])))

# Restart RCCSDT using amplitudes converted from UCCSDT
tamps_init_rhf = [t1_uhf2rhf, t2_uhf2rhf, t3_uhf2rhf]
mycc2 = cc.RCCSDT(mf, compact_tamps=False).set(conv_tol=1e-10, conv_tol_normt=1e-8, verbose=5)
mycc2.kernel(tamps=tamps_init_rhf)
print('RCCSDT correlation energy % .12f    Ref % .12f    Diff % .12e' % (
        mycc2.e_corr, -0.2188784727114157, mycc2.e_corr - -0.2188784727114157))

# Restart UCCSDT using amplitudes converted from RCCSDT
tamps_init_uhf = [t1_rhf2uhf, t2_rhf2uhf, t3_rhf2uhf]
myucc2 = cc.UCCSDT(mf, compact_tamps=False).set(conv_tol=1e-10, conv_tol_normt=1e-8, verbose=5)
myucc2.kernel(tamps=tamps_init_uhf)
print('UCCSDT correlation energy % .12f    Ref % .12f    Diff % .12e' % (
        myucc2.e_corr, -0.2188784727114157, myucc2.e_corr - -0.2188784727114157))
