#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run HF calculation.

.kernel() function is the simple way to call HF driver.
.analyze() function calls the Mulliken population analysis etc.
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)

myhf = mol.HF()
myhf.kernel()

# Orbital energies, Mulliken population etc.
myhf.analyze()


#
# myhf object can also be created using the APIs of gto, scf module
#
from pyscf import gto, scf
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)
myhf = scf.HF(mol)
myhf.kernel()

