#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Parity-violating contribution to the molecular energy

Valid for the point-nuclear model within the four-component relativistic framework.
The total contribution correspond to the first term of Eq. (4) in https://doi.org/10.1002/wcms.1396.
The sum over the nuclei of the system should be done as described in Eq. (4) of https://doi.org/10.1021/acs.jpclett.3c03038
'''

import numpy
from pyscf import gto, scf
from pyscf.scf.dhf import Epv_molecule

#Alanine molecule
mol = gto.M(
    atom = 'O   -0.000008 0.000006 0.473161; O   -0.498429 1.617953 -0.942950; N   -2.916494 2.018558 0.304530; C   -2.245961 0.738717 0.446378; C   -2.933825 -0.437589 -0.265779; C   -0.836260 0.869228 -0.089564; H   -2.164332 0.502686 1.502658; H   -2.396710 -1.368611 -0.107150; H   -3.940684 -0.559206 0.124631; H   -3.002345 -0.251817 -1.334665; H   -3.903065 1.915512 0.431392; H   -2.755346 2.398021 -0.608200; H   0.850292 0.091697 0.064913',
    basis = 'sto3g',
    symmetry = False,
    verbose = 3
)

mf = scf.DHF(mol)
mf.conv_tol = 1e-9
mf.with_ssss=False
mf.kernel()

Epv = Epv_molecule(mol, mf)
print('PV contribution to energy from the first nucleus (O) of the system: %.15g, ref = -2.745821e-21' % numpy.sum(Epv, axis=1)[0])
