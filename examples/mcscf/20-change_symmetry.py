#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Symmetry is not immutable

CASSCF solver can have different symmetry to the symmetry used by the
reference Hartree-Fock calculation.
'''

mol = gto.Mole()
mol.build(
    atom = 'Cr   0  0  0',
    basis = 'cc-pvtz',
    spin = 6,
    symmetry = True,
)
myhf = scf.RHF(mol)
myhf.irrep_nelec = {'A1g': (5,3), 'E1gx': (1,0), 'E1gy': (1,0),
                    'E2gx': (1,0), 'E2gy': (1,0)}
myhf.kernel()

myhf.mol.build(0, 0, symmetry='D2h')
mymc = mcscf.CASSCF(myhf, 9, 6)
mymc.kernel()
