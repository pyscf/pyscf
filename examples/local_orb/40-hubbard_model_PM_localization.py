#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf, ao2mo

#
# First, customize the Hubbard model solver
#
mol = gto.M()
n = 10
mol.nelectron = n

mf = scf.RHF(mol)
h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = 4.0

mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = ao2mo.restore(8, eri, n)
mf.kernel()

#
# Second, redefine the population tensor
#
from pyscf import lo
class HubbardPM(lo.pipek.PM):
# Construct the site-population tensor for each orbital-pair density.
# This tensor is used in cost-function and its gradients.
    def atomic_pops(self, mol, mo_coeff, method=None):
        return numpy.einsum('pi,pj->pij', mo_coeff, mo_coeff)

loc_orb_init_guess = mf.mo_coeff[:,2:8]
#mol.verbose = 5
locobj = HubbardPM(mol, loc_orb_init_guess)
print('PM cost function  ', locobj.cost_function())
loc_orb = locobj.kernel()
print('PM cost function  ', locobj.cost_function())

