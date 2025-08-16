#!/usr/bin/env python

'''
An example of constructing CASSCF initial guess with the localized orbitals
'''

import numpy
from pyscf import gto, scf, mcscf, lo
from pyscf.tools import mo_mapping

mol = gto.Mole(
    verbose=4,
    atom='''
C    0.000000000000     1.398696930758     0.000000000000
C    0.000000000000    -1.398696930758     0.000000000000
C    1.211265339156     0.699329968382     0.000000000000
C    1.211265339156    -0.699329968382     0.000000000000
C   -1.211265339156     0.699329968382     0.000000000000
C   -1.211265339156    -0.699329968382     0.000000000000
H    0.000000000000     2.491406946734     0.000000000000
H    0.000000000000    -2.491406946734     0.000000000000
H    2.157597486829     1.245660462400     0.000000000000
H    2.157597486829    -1.245660462400     0.000000000000
H   -2.157597486829     1.245660462400     0.000000000000
H   -2.157597486829    -1.245660462400     0.000000000000''',
    basis='ccpvdz',
    symmetry='D2h')

mol.build()
mf = scf.RHF(mol)
mf.kernel()

#
# Search the active space orbitals based on certain AO components. In this
# example, the pi-orbitals from carbon 2pz are considered.
#
pz_pop = mo_mapping.mo_comps('C 2pz', mol, mf.mo_coeff)
cas_list = pz_pop.argsort()[-6:]
print('cas_list', cas_list)
print('pz population for active space orbitals', pz_pop[cas_list])

mycas = mcscf.CASSCF(mf, 6, 6)
# Be careful with the keyword argument "base". By default sort_mo function takes
# the 1-based indices. The return indices of .argsort() method above are 0-based
cas_orbs = mycas.sort_mo(cas_list, mf.mo_coeff, base=0)
mycas.kernel(cas_orbs)


#
# Localized orbitals are often used as the initial guess of CASSCF method.
# When localizing SCF orbitals, it's better to call the localization twice to
# obtain the localized orbitals for occupied space and virtual space
# separately. This split localization scheme can ensure that the initial core
# determinant sits inside the HF occupied space.
#
nocc = mol.nelectron // 2
loc1 = lo.PM(mol, mf.mo_coeff[:,:nocc]).kernel()
loc2 = lo.PM(mol, mf.mo_coeff[:,nocc:]).kernel()
loc_orbs = numpy.hstack((loc1, loc2))
pz_pop = mo_mapping.mo_comps('C 2pz', mol, loc_orbs)
cas_list = pz_pop.argsort()[-6:]
print('cas_list', cas_list)
print('pz population for active space orbitals', pz_pop[cas_list])

#
# Symmetry was enabled in this molecule. By default, symmetry-adapted CASSCF
# algorithm will be called for molecule with symmetry. However, the orbital
# localization above breaks the orbital spatial symmetry. If proceeding the
# symmetry-adapted CASSCF calculation, the program may complain that the
# CASSCF initial guess is not symmetrized after certain attempts of orbital
# symmetrization. The simplest operation for this problem is to mute the
# spatial symmetry of the molecule.
#
mol.symmetry = False

mycas = mcscf.CASSCF(mf, 6, 6)
# Be careful with the second argument loc_orbs. If not given, sort_mo function
# manipulates the orbitals mycas.mo_coeff. In this example, local orbitals are
# objects to be sorted.
cas_orbs = mycas.sort_mo(cas_list, loc_orbs, base=0)
mycas.kernel(cas_orbs)

