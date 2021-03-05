#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
The contents of this example got moved to systems.py
'''

from pyscf import scf
from systems import mol_H2O_ccpvdz

mf = scf.RHF(mol_H2O_ccpvdz)
mf.kernel()
