#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, ao2mo

'''
A simple example to call integral transformation for given orbitals
'''

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)

myhf = scf.RHF(mol)
myhf.kernel()

orb = myhf.mo_coeff
eri_4fold = ao2mo.kernel(mol, orb)
print('MO integrals (ij|kl) with 4-fold symmetry i>=j, k>=l have shape %s' %
      str(eri_4fold.shape))


#
# Starting from PySCF-1.7, the MO integrals can be computed with the code
# below.
#
import pyscf
mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)
orb = mol.RHF().run().mo_coeff
eri_4fold = mol.ao2mo(orb)

