#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
AO-direct CCSD (New in PySCF 1.3)
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()

#
# AO-direct CCSD can largely reduce the IO overhead.  You need specify .direct
# attribute for CCSD object.
#
mycc = cc.CCSD(mf)
mycc.direct = True
mycc.frozen = 1 # frozen core
mycc.kernel()
print('CCSD correlation energy =', mycc.e_corr)

