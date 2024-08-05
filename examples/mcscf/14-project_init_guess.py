#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf
import numpy

'''
Use project_init_guess function to format the CASSCF initial guess orbitals

CASSCF solver can use any orbital as initial guess, no matter how and where
you get the orbital coefficients.  It can even be the orbitals of different
molecule.
'''

mol0 = gto.M(
    atom = 'C 0 0 0; C 0 0 1.2',
    basis = '6-31g')
mf = scf.RHF(mol0)
mf.kernel()

mo_init_guess = mf.mo_coeff

#############################################################
#
# Use the inital guess from mol1 system for mol2 CASSCF
#
#############################################################

# 1. If only the geometry is different, we only need to give
#    the function the guess orbitals

mol1 = gto.M(
atom = 'C 0 0 0; C 0 0 1.3',
basis = '6-31g',
verbose = 4)
mf = scf.RHF(mol1)
mf.kernel()
mc1 = mcscf.CASSCF(mf, 4, 4)
mo = mcscf.project_init_guess(mc1, mo_init_guess)
mc1.kernel(mo)
print('E(CAS) = %.12f, ref = -75.465214455907' % mc1.e_tot)

# 2. When we change the basis set, we also need to give it
#    the original molecule corresponding to our guess

mol2 = gto.M(
    atom = 'C 0 0 0; C 0 0 1.2',
    basis = 'cc-pvdz',
    verbose = 4)
mf = scf.RHF(mol2)
mf.kernel()
mc2 = mcscf.CASSCF(mf, 4, 4)
mo = mcscf.project_init_guess (mc2, mo_init_guess,
    prev_mol=mol0)
mc2.kernel(mo)
print('E(CAS) = %.12f, ref = -75.482856483217' % mc2.e_tot)

# 3. Changing the basis set and geometry at the same time is
#    not supported. However, you can always just call the
#    function multiple times with different arguments. 
mol3 = gto.M(
atom = 'C 0 0 0; C 0 0 1.3',
basis = 'cc-pvdz',
verbose = 4)
mf = scf.RHF(mol3)
mf.kernel()
mc3 = mcscf.CASSCF(mf, 4, 4)
mo = mcscf.project_init_guess (mc2, mo_init_guess,
    prev_mol=mol0)
mo = mcscf.project_init_guess (mc3, mo)
e3a = mc3.kernel(mo)[0] # basis, then geom
mo = mcscf.project_init_guess (mc1, mo_init_guess) 
mo = mcscf.project_init_guess (mc3, mo, prev_mol=mol1)
e3b = mc3.kernel(mo)[0] # geom, then basis
print ('E(CAS) = %.12f, %.12f, ref = -75.503011798165' % (e3a, e3b))


