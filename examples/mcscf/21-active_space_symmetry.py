#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Active space can be adjusted by specifing the number of orbitals for each irrep.
'''

import pyscf

mol = pyscf.M(
    atom = 'N  0  0  0; N  0  0  2',
    basis = 'ccpvtz',
    symmetry = True,
)
myhf = mol.RHF()
myhf.kernel()

mymc = myhf.CASSCF(8, 4)
# Select active orbitals which have the specified symmetry
# 2 E1gx orbitals, 2 E1gy orbitals, 2 E1ux orbitals, 2 E1uy orbitals
cas_space_symmetry = {'E1gx':2, 'E1gy':2, 'E1ux':2, 'E1uy':2}
mo = pyscf.mcscf.sort_mo_by_irrep(mymc, myhf.mo_coeff, cas_space_symmetry)
mymc.kernel(mo)


mol = pyscf.M(
    atom = 'Ti',
    basis = 'ccpvdz',
    symmetry = True,
    spin = 2,
)
myhf = mol.RHF()
myhf.kernel()
myhf.analyze()

mymc = myhf.CASSCF(14, 2)
# Put 3d, 4d, 4s, 4p orbitals in active space
cas_space_symmetry = {'s+0': 1,  # 4s
                      'd-2':2, 'd-1':2, 'd+0':2, 'd+1':2, 'd+2':2,  # 3d and 4d
                      'p-1':1, 'p+0':1, 'p+1':1,  # 4p
                     }
mo = pyscf.mcscf.sort_mo_by_irrep(mymc, myhf.mo_coeff, cas_space_symmetry)
mymc.verbose = 4
mymc.fcisolver.wfnsym = 'f+0'
mymc.kernel(mo)
mymc.analyze()


########################################
# Spherical symmetry was not supported until PySCF-1.7.4. SO3 symmetry was
# recogonized as Dooh. Code below is token from old examples.
#
mol = pyscf.M(
    atom = 'Ti',
    basis = 'ccpvdz',
    symmetry = 'Dooh',
    spin = 2,
)
myhf = mol.RHF()
myhf.kernel()
myhf.analyze()
mymc = myhf.CASSCF(14, 2)
# Put 3d, 4d, 4s, 4p orbitals in active space
cas_space_symmetry = {'A1g': 3,  # 4s, 3d(z^2), 4d(z^2)
                      'E2gx':2, 'E2gy':2, 'E1gx':2, 'E1gy':2,  # 3d and 4d
                      'A1u':1, 'E1ux':1, 'E1uy':1,  # 4p
                     }
mo = pyscf.mcscf.sort_mo_by_irrep(mymc, myhf.mo_coeff, cas_space_symmetry)
mymc.verbose = 4
mymc.fcisolver.wfnsym = 'E3gx'
mymc.kernel(mo)
mymc.analyze()
