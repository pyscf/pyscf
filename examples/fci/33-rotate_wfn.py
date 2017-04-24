#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Transform FCI wave functions according to the underlying one-particle
basis transformations.
'''

from functools import reduce
import numpy
from pyscf import gto, scf, fci

myhf1 = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', verbose=0).apply(scf.RHF).run()
e1, ci1 = fci.FCI(myhf1.mol, myhf1.mo_coeff).kernel()
print('FCI energy of mol1', e1)

myhf2 = gto.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', verbose=0).apply(scf.RHF).run()

s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
s12 = reduce(numpy.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
norb = myhf2.mo_energy.size
nelec = myhf2.mol.nelectron
ci2 = fci.addons.transform_ci_for_orbital_rotation(ci1, norb, nelec, s12)

print('alpha-string, beta-string,  CI coefficients')
for c,stra,strb in fci.addons.large_ci(ci2, norb, nelec):
    print(stra, strb, c)
