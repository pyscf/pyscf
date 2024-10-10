#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Parity-violating contrubution to the energy
'''

import numpy
from pyscf import gto, scf
from pyscf.scf.dhf import Epv_molecule


mol = gto.M(
    atom = 'F 0 0 0; F 0 0 1.00',
    basis = 'ccpvdz',
    symmetry = True,
    verbose = 3
)

mf = scf.DHF(mol)
mf.conv_tol = 1e-5
mf.kernel()

Epv = Epv_molecule(mol, mf)
print('Epv contributions from the first nucleus (F) = %.15g, ref = -5.949099319057879e-15' % numpy.sum(Epv, axis=1)[0])