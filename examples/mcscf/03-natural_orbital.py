#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generally, the CASSCF solver does NOT return the natural orbitals.

1. Attribute .natorb controls whether the active space orbitals are transformed
to natural orbitals in the results.

2. When .natorb is set, the natural orbitals may NOT be sorted by the active
space occupancy.  Within each irreps
'''

import numpy
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
myhf = scf.RHF(mol).run()

# 6 orbitals, 8 electrons
mycas = mcscf.CASSCF(myhf, 6, 8)
mycas.kernel()  # Here mycas.mo_coeff are not natural orbitals

mycas.natorb = True
mycas.kernel()  # Here mycas.mo_coeff are natural orbitals


#
# The natural orbitals in active space are NOT sorted by the occupancy.
#
mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    symmetry = True,
    spin = 2)
myhf = scf.RHF(mol).run()

mycas = mcscf.CASSCF(myhf, 6, 8)
mycas.natorb = True
# Here mycas.mo_coeff are natural orbitals because .natorb is on.
# Note The active space orbitals have the same symmetry as the input HF
# canonical orbitals.  They are not fully sorted wrt the occpancies.
# The mcscf active orbitals are sorted only within each irreps.
mycas.kernel()

