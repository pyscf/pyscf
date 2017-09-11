#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

mf = scf.UHF(mol).run()
mycc = cc.UCCSD(mf).run()
et = mycc.ccsd_t()
print('UCCSD(T) correlation energy', mycc.e_corr + et)

