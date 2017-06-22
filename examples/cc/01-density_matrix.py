#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD density matrix
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()

#
# CCSD density matrix in MO basis
#
dm1 = mycc.make_rdm1()
dm2 = mycc.make_rdm2()

#
# Relaxed CCSD density matrix in MO basis
#
from pyscf.cc import ccsd_grad
dm1 += ccsd_grad.response_dm1(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2)
