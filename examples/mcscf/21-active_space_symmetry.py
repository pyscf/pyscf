#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Active space can be adjusted by specifing the number of orbitals for each irrep.
'''

mol = gto.Mole()
mol.build(
    atom = 'N  0  0  0; N  0  0  2',
    basis = 'ccpvtz',
    symmetry = True,
)
myhf = scf.RHF(mol)
myhf.kernel()

mymc = mcscf.CASSCF(myhf, 8, 4)
mo = mcscf.sort_mo_by_irrep(mymc, myhf.mo_coeff,
                            {'E1gx':2, 'E1gy':2, 'E1ux':2, 'E1uy':2})
mymc.kernel(mo)
