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
