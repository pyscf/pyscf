#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Overlap of two FCI wave functions
'''

from functools import reduce
import numpy
from pyscf import gto, scf, fci

myhf1 = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', verbose=0).apply(scf.RHF).run()
e1, ci1 = fci.FCI(myhf1.mol, myhf1.mo_coeff).kernel()
print('FCI energy of mol1', e1)

myhf2 = gto.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', verbose=0).apply(scf.RHF).run()
e2, ci2 = fci.FCI(myhf2.mol, myhf2.mo_coeff).kernel()
print('FCI energy of mol2', e2)

s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
s12 = reduce(numpy.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
norb = myhf2.mo_energy.size
nelec = myhf2.mol.nelectron
print('<FCI-mol1|FCI-mol2> = ', fci.addons.overlap(ci1, ci2, norb, nelec, s12))
