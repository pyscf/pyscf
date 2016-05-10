#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generally, the CASSCF solver does not return the natural orbitals.
Attribute .natorb controls whether the active space orbitals are transformed
to natural orbitals.
'''

import numpy
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

myhf = scf.RHF(mol)
myhf.kernel()

# 6 orbitals, 8 electrons
mycas = mcscf.CASSCF(myhf, 6, 8)
mycas.natorb = True
mycas.kernel()
