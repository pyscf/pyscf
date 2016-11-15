#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run FCI
'''

from pyscf import gto, scf, fci

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '6-31g',
    symmetry = True,
)

myhf = scf.RHF(mol)
myhf.kernel()

#
# Function fci.FCI creates an FCI solver based on the given orbitals and the
# num. electrons and spin of the given mol object
#
cisolver = fci.FCI(mol, myhf.mo_coeff)
print('E(FCI) = %.12f' % cisolver.kernel()[0])

