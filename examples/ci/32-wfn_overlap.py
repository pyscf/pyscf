#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Overlap of two CISD wave functions (they can be obtained from different
geometries).
'''

from functools import reduce
import numpy
from pyscf import gto, scf, ci

#
# RCISD wavefunction overlap
#
myhf1 = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', verbose=0).apply(scf.RHF).run()
ci1 = ci.CISD(myhf1).run()
print('CISD energy of mol1', ci1.e_tot)

myhf2 = gto.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', verbose=0).apply(scf.RHF).run()
ci2 = ci.CISD(myhf2).run()
print('CISD energy of mol2', ci2.e_tot)

s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
s12 = reduce(numpy.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
nmo = myhf2.mo_energy.size
nocc = myhf2.mol.nelectron // 2
print('<CISD-mol1|CISD-mol2> = ', ci.cisd.overlap(ci1.ci, ci2.ci, nmo, nocc, s12))

#
# UCISD wavefunction overlap
#
myhf1 = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', spin=2, verbose=0).apply(scf.UHF).run()
ci1 = ci.UCISD(myhf1).run()
print('CISD energy of mol1', ci1.e_tot)

myhf2 = gto.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', spin=2, verbose=0).apply(scf.UHF).run()
ci2 = ci.UCISD(myhf2).run()
print('CISD energy of mol2', ci2.e_tot)

s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
mo1a, mo1b = myhf1.mo_coeff
mo2a, mo2b = myhf2.mo_coeff
s12 = (reduce(numpy.dot, (mo1a.T, s12, mo2a)),
       reduce(numpy.dot, (mo1b.T, s12, mo2b)))
nmo = (myhf2.mo_energy[0].size, myhf2.mo_energy[1].size)
nocc = myhf2.nelec
print('<CISD-mol1|CISD-mol2> = ', ci.ucisd.overlap(ci1.ci, ci2.ci, nmo, nocc, s12))
