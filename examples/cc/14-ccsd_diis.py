#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Adjust CCSD DIIS
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()

#
# Increase the DIIS space to improve convergence
#
mycc = cc.CCSD(mf)
mycc.diis_space = 10
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

#
# By default, CCSD damps the solution starting from the first iteration.
# In some systems, it'd be better to exclude the CCSD amplitudes of early
# iterations.  To start DIIS extrapolation later, you can set diis_start_cycle.
#
mycc.diis_start_cycle = 4
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

