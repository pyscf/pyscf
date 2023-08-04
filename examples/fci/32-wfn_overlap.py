#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Overlap of two FCI wave functions
'''

from functools import reduce
import numpy
import pyscf

myhf1 = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', verbose=0).RHF().run()
e1, ci1 = pyscf.fci.FCI(myhf1).kernel()
print('FCI energy of mol1', e1)

myhf2 = pyscf.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', verbose=0).RHF().run()
e2, ci2 = pyscf.fci.FCI(myhf2).kernel()
print('FCI energy of mol2', e2)

myhf3 = pyscf.M(atom='H 0 0 0; F 0 0 1.3', basis='6-31g', verbose=0).UHF().run()
e3, ci3 = pyscf.fci.FCI(myhf3).kernel()
print('FCI energy of mol3', e3)

#
# Overlap between FCI wfn of different geometries
#
s12 = pyscf.gto.intor_cross('int1e_ovlp', myhf1.mol, myhf2.mol)
s12 = reduce(numpy.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
norb = myhf2.mo_energy.size
nelec = myhf2.mol.nelectron
print('<FCI-mol1|FCI-mol2> = ', pyscf.fci.addons.overlap(ci1, ci2, norb, nelec, s12))

#
# Overlap between RHF-FCI and UHF-FCI
#
s13 = pyscf.gto.intor_cross('int1e_ovlp', myhf1.mol, myhf2.mol)
s13a = reduce(numpy.dot, (myhf1.mo_coeff.T, s13, myhf3.mo_coeff[0]))
s13b = reduce(numpy.dot, (myhf1.mo_coeff.T, s13, myhf3.mo_coeff[1]))
s13 = (s13a, s13b)
norb = myhf3.mo_energy[0].size
nelec = myhf3.mol.nelectron
print('<FCI-mol1|FCI-mol3> = ', pyscf.fci.addons.overlap(ci1, ci3, norb, nelec, s13))
