#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD frozen core
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()

#
# Freeze the inner most two orbitals.
#
mycc = cc.CCSD(mf)
mycc.frozen = 2
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

#
# Auto-generate the number of core orbitals to be frozen.
# In this case, it would be 1.
#
mycc = cc.CCSD(mf)
from pyscf.data import elements
mycc.frozen = elements.chemcore(mol)
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

#
# Freeze orbitals based on the list of indices.
#
mycc.frozen = [0,1,16,17,18]
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)

