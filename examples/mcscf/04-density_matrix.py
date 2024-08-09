#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CASCI/CASSCF density matrix
'''

import numpy
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
myhf = scf.RHF(mol).run()

# 6 orbitals, 8 electrons
mycas = mcscf.CASSCF(myhf, 6, 8)
mycas.kernel()

#
# 1pdm in AO representation
#
dm1 = mycas.make_rdm1()

#
# alpha and beta 1-pdm in AO representation
#
dm1_alpha, dm1_beta = mycas.make_rdm1s()

print(numpy.allclose(dm1, dm1_alpha+dm1_beta))

dm1, dm2, dm3 = mycas.fcisolver.make_rdm123(mycas.ci, mycas.ncas, mycas.nelecas)

(dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb) = \
        mycas.fcisolver.make_rdm123s(mycas.ci, mycas.ncas, mycas.nelecas)

print(numpy.allclose(dm1a+dm1b, dm1))
print(numpy.allclose(dm2aa+dm2bb+dm2ab+dm2ab.transpose(2,3,0,1), dm2))
print(numpy.allclose(dm3aaa+dm3bbb+dm3aab+dm3aab.transpose(0,1,4,5,2,3)+\
dm3aab.transpose(4,5,0,1,2,3)+dm3abb+dm3abb.transpose(2,3,0,1,4,5)+dm3abb.transpose(2,3,4,5,0,1), dm3))

