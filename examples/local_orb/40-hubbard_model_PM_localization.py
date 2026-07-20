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
    r'''Construct the site-population tensor for each orbital-pair density.
        This tensor is used in cost-function and its gradients.
    '''
    pop_method = None
    def atomic_pops(self, mol, mo_coeff, method=None, mode=None):
        if mode == 'pop':
            return numpy.einsum('pi,pi->pi', mo_coeff.conj(), mo_coeff).real
        else:
            return numpy.einsum('pi,pj->pij', mo_coeff.conj(), mo_coeff)

loc_orb_init_guess = mf.mo_coeff[:,2:8]
mlo = HubbardPM(mol, loc_orb_init_guess).set(verbose=4)
print('PM cost function  ', mlo.cost_function())
loc_orb = mlo.kernel()
print('PM cost function  ', mlo.cost_function())

while True:
    mo, stable = mlo.stability_jacobi(return_status=True)
    if stable:
        break
    mlo.kernel(mo)

print('PM cost function  ', mlo.cost_function())
