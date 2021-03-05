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
from systems import mol_HF_ccpvdz

myhf = mol_HF_ccpvdz.HF()
myhf.kernel()
# Orbital energies, Mulliken population etc.
myhf.analyze()

# myhf object can also be created using the APIs of gto, scf module
myhf = pyscf.scf.HF(mol_HF_ccpvdz)
myhf.kernel()
