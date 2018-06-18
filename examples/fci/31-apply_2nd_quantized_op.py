#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

r'''
Applying creation or annihilation operators on FCI wavefunction
        a |0>

Compute density matrices by 
        gamma_{ij} = <0| i^+ j |0>
        Gamma_{ij,kl} = <0| i^+ j^+ l k |0>
'''

import numpy
from pyscf import gto, scf, fci

mol = gto.M(atom='H 0 0 0; Li 0 0 1.1', basis='sto3g')
m = scf.RHF(mol).run()
fs = fci.FCI(mol, m.mo_coeff)
e, c = fs.kernel()

norb = m.mo_energy.size
neleca = nelecb = mol.nelectron // 2

#
# Spin-free 1-particle density matrix
# <Psi| a_{i\alpha}^\dagger a_{j\alpha} + a_{i\beta}^\dagger a_{j\beta} |Psi>
#
dm1 = numpy.zeros((norb,norb))
for i in range(norb):
    for j in range(norb):
        tmp = fci.addons.des_a(c  , norb, (neleca  ,nelecb), j)
        tmp = fci.addons.cre_a(tmp, norb, (neleca-1,nelecb), i)
        dm1[i,j] += numpy.dot(tmp.flatten(), c.flatten())

        tmp = fci.addons.des_b(c  , norb, (neleca,nelecb  ), j)
        tmp = fci.addons.cre_b(tmp, norb, (neleca,nelecb-1), i)
        dm1[i,j] += numpy.dot(tmp.flatten(), c.flatten())


#
# Note the swap of k and l indices for 2-PDM
#       tmp = i^+ j^+ l k |0>
#
dm2aaaa = numpy.zeros((norb,norb,norb,norb))
dm2abab = numpy.zeros((norb,norb,norb,norb))
dm2bbbb = numpy.zeros((norb,norb,norb,norb))
for i in range(norb):
    for j in range(norb):
        for k in range(norb):
            for l in range(norb):
                tmp = fci.addons.des_a(c  , norb, (neleca  ,nelecb), k)
                tmp = fci.addons.des_a(tmp, norb, (neleca-1,nelecb), l)
                tmp = fci.addons.cre_a(tmp, norb, (neleca-2,nelecb), j)
                tmp = fci.addons.cre_a(tmp, norb, (neleca-1,nelecb), i)
                dm2aaaa[i,j,k,l] += numpy.dot(tmp.flatten(), c.flatten())

                tmp = fci.addons.des_a(c  , norb, (neleca  ,nelecb  ), k)
                tmp = fci.addons.des_b(tmp, norb, (neleca-1,nelecb  ), l)
                tmp = fci.addons.cre_b(tmp, norb, (neleca-1,nelecb-1), j)
                tmp = fci.addons.cre_a(tmp, norb, (neleca-1,nelecb  ), i)
                dm2abab[i,j,k,l] += numpy.dot(tmp.flatten(), c.flatten())

                tmp = fci.addons.des_b(c  , norb, (neleca,nelecb  ), k)
                tmp = fci.addons.des_b(tmp, norb, (neleca,nelecb-1), l)
                tmp = fci.addons.cre_b(tmp, norb, (neleca,nelecb-2), j)
                tmp = fci.addons.cre_b(tmp, norb, (neleca,nelecb-1), i)
                dm2bbbb[i,j,k,l] += numpy.dot(tmp.flatten(), c.flatten())


ref1 = fs.make_rdm1(c, norb, (neleca,nelecb))
ref2aaaa, ref2aabb, ref2bbbb = fs.make_rdm12s(c, norb, (neleca,nelecb))[1]

print('Error in spin-free 1-PDM %g' % numpy.linalg.norm(ref1-dm1))
print('Error in 2-PDM aaaa %g' % numpy.linalg.norm(ref2aaaa.transpose(0,2,1,3)-dm2aaaa))
print('Error in 2-PDM aabb %g' % numpy.linalg.norm(ref2aabb.transpose(0,2,1,3)-dm2abab))
print('Error in 2-PDM bbbb %g' % numpy.linalg.norm(ref2bbbb.transpose(0,2,1,3)-dm2bbbb))

