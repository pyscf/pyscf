#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf
import numpy

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

# 1. If only the geometry is different, we only need to give
#    the function the guess orbitals

mol2 = gto.M(
    atom = 'C 0 0 0; C 0 0 1.3',
    basis = '6-31g',
    verbose = 4)
mf = scf.RHF(mol2)
mf.kernel()
mc = mcscf.CASSCF(mf, 4, 4)
mo = mcscf.project_init_guess(mc, mo_init_guess)
mc.kernel(mo)
print('E(CAS) = %.12f, ref = -75.465214455907' % mc.e_tot)

# 2. When we change the basis set, we also need to give it
#    the original molecule corresponding to our guess

mol2 = gto.M(
    atom = 'C 0 0 0; C 0 0 1.2',
    basis = 'cc-pvdz',
    verbose = 4)
mf = scf.RHF(mol2)
mf.kernel()
mc = mcscf.CASSCF(mf, 4, 4)
mo = mcscf.project_init_guess(mc, mo_init_guess, prev_mol=mol1)
mc.kernel(mo)
print('E(CAS) = %.12f, ref = -75.482856483217' % mc.e_tot)


