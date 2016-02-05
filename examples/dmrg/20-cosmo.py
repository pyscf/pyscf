#!/usr/bin/env python
#
# Contributors:
#       Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.dmrgscf.dmrgci import DMRGCI, DMRGSCF
from pyscf.mrpt.nevpt2 import sc_nevpt
from pyscf import cosmo

'''
Solvent effects can be applied with DMRG-CASSCF, DMRG-NEVPT2 calculations.
'''

mol = gto.M(
    atom = [
        ['Fe', (0.      , 0.0000  , 0.0000)],
        ['N' , (1.9764  , 0.0000  , 0.0000)],
        ['N' , (0.0000  , 1.9884  , 0.0000)],
        ['N' , (-1.9764 , 0.0000  , 0.0000)],
        ['N' , (0.0000  , -1.9884 , 0.0000)],
        ['C' , (2.8182  , -1.0903 , 0.0000)],
        ['C' , (2.8182  , 1.0903  , 0.0000)],
        ['C' , (1.0918  , 2.8249  , 0.0000)],
        ['C' , (-1.0918 , 2.8249  , 0.0000)],
        ['C' , (-2.8182 , 1.0903  , 0.0000)],
        ['C' , (-2.8182 , -1.0903 , 0.0000)],
        ['C' , (-1.0918 , -2.8249 , 0.0000)],
        ['C' , (1.0918  , -2.8249 , 0.0000)],
        ['C' , (4.1961  , -0.6773 , 0.0000)],
        ['C' , (4.1961  , 0.6773  , 0.0000)],
        ['C' , (0.6825  , 4.1912  , 0.0000)],
        ['C' , (-0.6825 , 4.1912  , 0.0000)],
        ['C' , (-4.1961 , 0.6773  , 0.0000)],
        ['C' , (-4.1961 , -0.6773 , 0.0000)],
        ['C' , (-0.6825 , -4.1912 , 0.0000)],
        ['C' , (0.6825  , -4.1912 , 0.0000)],
        ['H' , (5.0441  , -1.3538 , 0.0000)],
        ['H' , (5.0441  , 1.3538  , 0.0000)],
        ['H' , (1.3558  , 5.0416  , 0.0000)],
        ['H' , (-1.3558 , 5.0416  , 0.0000)],
        ['H' , (-5.0441 , 1.3538  , 0.0000)],
        ['H' , (-5.0441 , -1.3538 , 0.0000)],
        ['H' , (-1.3558 , -5.0416 , 0.0000)],
        ['H' , (1.3558  , -5.0416 , 0.0000)],
        ['C' , (2.4150  , 2.4083  , 0.0000)],
        ['C' , (-2.4150 , 2.4083  , 0.0000)],
        ['C' , (-2.4150 , -2.4083 , 0.0000)],
        ['C' , (2.4150  , -2.4083 , 0.0000)],
        ['H' , (3.1855  , 3.1752  , 0.0000)],
        ['H' , (-3.1855 , 3.1752  , 0.0000)],
        ['H' , (-3.1855 , -3.1752 , 0.0000)],
        ['H' , (3.1855  , -3.1752 , 0.0000)], ],
    basis = 'ccpvdz',
    verbose = 4,
    output = 'fepor-cosmo.out',
    spin = 4,
    symmetry = 'd2h'
)

mf = scf.UHF(mol)
mf.chkfile = 'fepor.chk'
mf = scf.fast_newton(mf)


##################################################
#
# Vertical excitation
#
##################################################

#
# 1. Equilibrium solvation for ground state
#
# "sol.dm = None" allows the solvation relaxing to equilibruim wrt the system
# ground state
#
sol = cosmo.COSMO(mol)
sol.dm = None

mc = cosmo.cosmo_(DMRGSCF(mf, 5, 6), sol)
mo = mc.sort_mo_by_irrep({'Ag':2, 'B1g':1, 'B2g':1, 'B3g':1})
mc.kernel(mo)
mo = mc.mo_coeff

#
# 2. Frozen solvation of ground state for excited states
#
# Assigning certain density matrix to sol.dm can force the solvent object to
# compute the solvation correction with the given density matrix.
# sol._dm_guess is the system density matrix of the last iteration from
# previous calculation.  Setting "sol.dm = sol._dm_guess" freezes the
# ground state solvent effects in the excitated calculation.
#
sol.dm = sol._dm_guess
mc = cosmo.cosmo_(mcscf.CASCI(mf, 5, 6), sol)
mc.fcisolver = DMRGCI(mol)
mc.fcisolver.nroots = 2
mc.kernel(mo)

e_state0 = mc.e_tot[0] + sc_nevpt(mc, ci=mc.ci[0])
e_state1 = mc.e_tot[1] + sc_nevpt(mc, ci=mc.ci[1])
print('Excitation E = %.9g' % (e_state1-e_state0))



##################################################
#
# Emission
#
##################################################

#
# 1. Equilibrium solvation for excited state.
#
# "sol.dm = None" relaxes the solvent to equilibruim wrt excited state
#
sol = cosmo.COSMO(mol)
sol.dm = None

mc = cosmo.cosmo_(DMRGSCF(mf, 5, 6), sol)
mc.state_specific_(1)
mo = mc.sort_mo_by_irrep({'Ag':2, 'B1g':1, 'B2g':1, 'B3g':1})
mc.kernel(mo)
mo = mc.mo_coeff
e_state1 = mc.e_tot + sc_nevpt(mc)

#
# 2. Frozen excited state solvation for ground state
#
sol.dm = sol._dm_guess
mc = cosmo.cosmo_(mcscf.CASCI(mf, 5, 6), sol)
mc.fcisolver = DMRGCI(mol)
mc.kernel(mo)
e_state0 = mc.e_tot + sc_nevpt(mc)
print('Emission E = %.9g' % (e_state1-e_state0))
