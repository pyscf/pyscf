#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Writing FCIDUMP file for given integrals or SCF orbitals
'''

from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo
from pyscf import symm
from pyscf.tools import fcidump

mol = gto.M(
    atom = [['H', 0, 0, i] for i in range(6)],
    basis = '6-31g',
    verbose = 0,
    symmetry = 1,
    symmetry_subgroup = 'D2h',
)
myhf = scf.RHF(mol)
myhf.kernel()

#
# Example 1: Convert an SCF object to FCIDUMP
#
fcidump.from_scf(myhf, 'fcidump.example1')


#
# Example 2: Given a set of orbitals to transform integrals then dump the
# integrals to FCIDUMP
#
fcidump.from_mo(mol, 'fcidump.example2', myhf.mo_coeff)


#
# Exampel 3: FCIDUMP for given 1e and 2e integrals
#
c = myhf.mo_coeff
h1e = reduce(numpy.dot, (c.T, myhf.get_hcore(), c))
eri = ao2mo.kernel(mol, c)

fcidump.from_integrals('fcidump.example3', h1e, eri, c.shape[1],
                             mol.nelectron, ms=0)

#
# Exampel 4: Ignore small matrix elements in FCIDUMP
#
fcidump.from_integrals('fcidump.example4', h1e, eri, c.shape[1],
                             mol.nelectron, ms=0, tol=1e-10)

#
# Example 5: Inculde the symmetry information in FCIDUMP
#
# to write the irreps for each orbital, first use pyscf.symm.label_orb_symm to
# get the irrep ids
MOLPRO_ID = {'D2h': { 'Ag' : 1,
                      'B1g': 4,
                      'B2g': 6,
                      'B3g': 7,
                      'Au' : 8,
                      'B1u': 5,
                      'B2u': 3,
                      'B3u': 2},
             'C2v': { 'A1' : 1,
                      'A2' : 4,
                      'B1' : 2,
                      'B2' : 3},
             'C2h': { 'Ag' : 1,
                      'Bg' : 4,
                      'Au' : 2,
                      'Bu' : 3},
             'D2' : { 'A ' : 1,
                      'B1' : 4,
                      'B2' : 3,
                      'B3' : 2},
             'Cs' : { "A'" : 1,
                      'A"' : 2},
             'C2' : { 'A'  : 1,
                      'B'  : 2},
             'Ci' : { 'Ag' : 1,
                      'Au' : 2},
             'C1' : { 'A'  : 1,}}

orbsym = [MOLPRO_ID[mol.groupname][i]
          for i in symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)]
fcidump.from_integrals('fcidump.example5', h1e, eri, c.shape[1],
                       mol.nelectron, ms=0, orbsym=orbsym)

#
# Hamiltonians of FCIDUMP file can be load
#
ctx = fcidump.read('fcidump.example1')
ctx = fcidump.read('fcidump.example5', molpro_orbsym=True)

#
# Construct an SCF object using the quantities defined in FCIDUMP
# (pyscf-1.7.4 or newer)
#
mf = fcidump.to_scf('fcidump.example5', molpro_orbsym=True)
mf.mol.verbose = 4
mf.run()
mf.MP2().run()
mf.to_uhf().newton().run()
