#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf, mcscf, fci

'''
Use different FCI solver for CASSCF
'''

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()

mf = scf.RHF(mol)
mf.kernel()

#
# Default CASSCF
#
mc = mcscf.CASSCF(mf, 4, (4,2))
emc1 = mc.kernel()[0]
print('* Triplet, using default CI solver, E = %.15g' % emc1)

#
# fcisolver is fci.direct_spin1
#
mc = mcscf.CASSCF(mf, 4, 6)
# change the CAS space FCI solver. e.g. to DMRG, FCIQMC
mc.fcisolver = fci.direct_spin1
emc1 = mc.kernel()[0]
print('* Triplet,  using fci.direct_spin1 solver, E = %.15g' % emc1)

#
# fcisolver is fci.direct_spin0
#
mol.build(False, False, spin=0)
mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.CASSCF(mf, 6, 6)
mc.fcisolver = fci.direct_spin0
caspace = [6,7,8,9,10,12]
mo = mc.sort_mo(caspace)
emc1 = mc.kernel(mo)[0]
print('* Symmetry-broken singlet, using fci.direct_spin0 solver, E = %.15g' % emc1)

