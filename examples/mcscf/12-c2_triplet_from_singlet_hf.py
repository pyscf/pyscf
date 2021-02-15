#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
CASSCF solver can have different spin state to the HF initial guess 

The CASSCF solver only use the HF orbital as initial guess.  So it's allowed
to assign different spin state in the CASSCF solver.
'''

mol = gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

# 6 orbitals, (4 alpha electrons, 2 beta electrons)
mc = mcscf.CASSCF(mf, 6, (4, 2))
emc = mc.kernel()[0]
print('E(CAS) = %.12f, ref = -75.549397771485' % emc)

