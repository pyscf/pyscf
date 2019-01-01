#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example of using solvent model in the mean-field calculations.
'''

from pyscf import gto
from pyscf import scf
from pyscf import dftd3

mol = gto.Mole()
mol.atom = ''' O    0.00000000    0.00000000   -0.11081188
               H   -0.00000000   -0.84695236    0.59109389
               H   -0.00000000    0.89830571    0.52404783 '''
mol.basis = 'cc-pvdz'
mol.build()

mf = dftd3.dftd3(scf.RHF(mol))
print(mf.kernel()) # -75.99396273778923

