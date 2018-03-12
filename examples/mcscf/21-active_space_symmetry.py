#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Active space can be adjusted by specifing the number of orbitals for each irrep.
'''

from pyscf import gto, scf, mcscf

mol = gto.Mole()
mol.build(
    atom = 'N  0  0  0; N  0  0  2',
    basis = 'ccpvtz',
    symmetry = True,
)
myhf = scf.RHF(mol)
myhf.kernel()

mymc = mcscf.CASSCF(myhf, 8, 4)
# Select active orbitals which have the specified symmetry
# 2 E1gx orbitals, 2 E1gy orbitals, 2 E1ux orbitals, 2 E1uy orbitals
cas_space_symmetry = {'E1gx':2, 'E1gy':2, 'E1ux':2, 'E1uy':2}
mo = mcscf.sort_mo_by_irrep(mymc, myhf.mo_coeff, cas_space_symmetry)
mymc.kernel(mo)


mol = gto.Mole()
mol.build(
    atom = 'Ti',
    basis = 'ccpvdz',
    symmetry = True,
    spin = 2,
)
myhf = scf.RHF(mol)
myhf.kernel()
myhf.analyze()

mymc = mcscf.CASSCF(myhf, 14, 2)
# Put 3d, 4d, 4s, 4p orbtials in active space
cas_space_symmetry = {'A1g': 3,  # 4s, 3d(z^2), 4d(z^2)
                      'E2gx':2, 'E2gy':2, 'E1gx':2, 'E1gy':2,  # 3d and 4d
                      'A1u':1, 'E1ux':1, 'E1uy':1,  # 4p
                     }
mo = mcscf.sort_mo_by_irrep(mymc, myhf.mo_coeff, cas_space_symmetry)
mymc.verbose = 4
mymc.fcisolver.wfnsym = 'E3gx'
mymc.kernel(mo)
mymc.analyze()
