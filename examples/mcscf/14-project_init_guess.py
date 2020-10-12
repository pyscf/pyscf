#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Use project_init_guess function to format the CASSCF initial guess orbitals

CASSCF solver can use any orbital as initial guess, no matter how and where
you get the orbital coeffcients.  It can even be the orbitals of different
molecule.
'''

mol1 = gto.M(
    atom = 'C 0 0 0; C 0 0 1.2',
    basis = '6-31g')
mf = scf.RHF(mol1)
mf.kernel()

mo_init_guess = mf.mo_coeff

#############################################################
#
# Use the inital guess from mol1 system for mol2 CASSCF
#
#############################################################
mol2 = gto.M(
    atom = 'C 0 0 0; C 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
mf = scf.RHF(mol2)
mf.kernel()
mc = mcscf.CASSCF(mf, 4, 4)
mo = mcscf.project_init_guess(mc, mo_init_guess, mol1)
mc.verbose = 4
mc.kernel(mo)
print('E(CAS) = %.12f, ref = -149.622580905034' % mc.e_tot)

