#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Symmetry is not immutable

CASSCF solver can have different symmetry to reference Hartree-Fock calculation.
'''


import pyscf

mol = pyscf.M(
    atom = 'Cr   0  0  0',
    basis = 'cc-pvtz',
    spin = 6,
    symmetry = True,
)
myhf = mol.RHF()
myhf.irrep_nelec = {'s+0': (4, 3), 'd-2': (1, 0), 'd-1': (1, 0),
                    'd+0': (1, 0), 'd+1': (1, 0), 'd+2': (1, 0)}
myhf.kernel()

myhf.mol.build(0, 0, symmetry='D2h')
mymc = myhf.CASSCF(9, 6)
mymc.kernel()


########################################
# Spherical symmetry was not supported until PySCF-1.7.4. SO3 symmetry was
# recogonized as Dooh. Code below is token from old examples.
#
mol = pyscf.M(
    atom = 'Cr   0  0  0',
    basis = 'cc-pvtz',
    spin = 6,
    symmetry = 'Dooh',
)
myhf = mol.RHF()
myhf.irrep_nelec = {'A1g': (5,3), 'E1gx': (1,0), 'E1gy': (1,0),
                    'E2gx': (1,0), 'E2gy': (1,0)}
myhf.kernel()

myhf.mol.build(0, 0, symmetry='D2h')
mymc = myhf.CASSCF(9, 6)
mymc.kernel()
