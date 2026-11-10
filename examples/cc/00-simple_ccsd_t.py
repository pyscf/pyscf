#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = mol.RHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

mf = mol.UHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('UCCSD(T) correlation energy', mycc.e_corr + et)

# A GHF reference gives a spin-orbital GCCSD(T) calculation.
mf = mol.GHF().run()
mycc = mf.CCSD().run()
print('GCCSD(T) correlation energy', mycc.e_corr + mycc.ccsd_t())

# QCISD(T) is also available through the CC module.
from pyscf.cc import qcisd_slow
myqcisd = qcisd_slow.QCISD(mol.RHF().run()).run()
print('QCISD(T) correlation energy', myqcisd.e_corr + myqcisd.qcisd_t())
