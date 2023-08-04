#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CISD calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = mol.HF().run()
mycc = mf.CISD().run()
print('RCISD correlation energy', mycc.e_corr)

mf = mol.UHF().run()
mycc = mf.CISD().run()
print('UCISD correlation energy', mycc.e_corr)

