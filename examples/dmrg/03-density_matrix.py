#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Block code for active space N-particle density matrices.
'''

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.dmrgscf import dmrgci

b = 1.2
mol = gto.M(
    verbose = 4,
    atom = 'N 0 0 0; N 0 0 %f'%b,
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

mc = mcscf.CASSCF(m, 8, 8)
mc.fcisolver = dmrgci.DMRGCI(mol)
mc.kernel()

dm1 = mc.fcisolver.make_rdm1(0, mc.ncas, mc.nelecas)
dm2 = mc.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)[1]
dm3 = mc.fcisolver.make_rdm123(0, mc.ncas, mc.nelecas)[2]

#
# or computing DMs all together in one DMRG call
#
dm1, dm2 = mc.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)

dm1, dm2, dm3 = mc.fcisolver.make_rdm123(0, mc.ncas, mc.nelecas)

