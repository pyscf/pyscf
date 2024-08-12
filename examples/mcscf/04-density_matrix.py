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
