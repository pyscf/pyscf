#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CISD calculation.
'''

from pyscf import gto, scf, ci

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
mycc = ci.CISD(mf).run()
print('RCISD correlation energy', mycc.e_corr)

mf = scf.UHF(mol).run()
mycc = ci.CISD(mf).run()
print('UCISD correlation energy', mycc.e_corr)

