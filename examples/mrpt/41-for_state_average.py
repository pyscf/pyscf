#!/usr/bin/env python

'''
NEVPT2 calculation for the wavefunction obtained by state-average CASCI/CASSCF.

NEVPT2 does not support "state-average" style calculation.  Following the
state-average CASCI/CASSCF calculation, you need to first run a multi-root
CASCI calculation then call NEVPT2 method for the specific CASCI wavefunction.
'''

from pyscf import gto, scf, mcscf
from pyscf import mrpt

r = 1.8
mol = gto.Mole()
mol.atom = [
    ['C', ( 0., 0.    , -r/2   )],
    ['C', ( 0., 0.    ,  r/2)],]
mol.basis = 'cc-pvdz'
mol.unit = 'B'
mol.symmetry = True
mol.verbose = 4
mol.build()
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1g': 4, 'E1gx': 0, 'E1gy': 0, 'A1u': 4,
                  'E1uy': 2, 'E1ux': 2, 'E2gx': 0, 'E2gy': 0, 'E2uy': 0, 'E2ux': 0}
ehf = mf.kernel()

#
# Save orbitals from state-average CASSCF calculation.
#
mc = mcscf.CASSCF(mf, 8, 8).state_average_(weights=(0.5,0.5))
mc.kernel()
orbital = mc.mo_coeff

#
# Create a multi-root CASCI calcultion to get excited state wavefunctions.
#
mc = mcscf.CASCI(mf, 8, 8)
mc.fcisolver.nroots = 4
mc.kernel(orbital)

#
# Finally compute NEVPT energy for required state
#
e_corr = mrpt.NEVPT(mc,root=1).kernel()
e_tot = mc.e_tot[1] + e_corr
print('Total energy of first excited state', e_tot)

