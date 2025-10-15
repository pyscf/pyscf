#!/usr/bin/env python
#

'''
Analytical nuclear gradients of MC-PDFT excited state.
TL;DR exactly the same interface as CASSCF
(CASCI-based MC-PDFT gradients are not supported)
'''

from pyscf import gto
from pyscf import scf
from pyscf import mcpdft
from pyscf import lib

mol = gto.M(
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. ,-0.757  , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
    basis = '631g'
)
mf = scf.RHF(mol).run()

mc = mcpdft.CASSCF(mf, 'ftPBE', 4, 4).state_average ([0.25, 0.25, 0.25, 0.25])
mc.run()

# PySCF-1.6.1 and newer supports the .Gradients method to create a grad
# object after grad module was imported. It is equivalent to call the
# .nuc_grad_method method.
from pyscf import grad
mc = mcpdft.CASSCF(mf, 'ftPBE', 4, 4).state_average ([0.25, 0.25, 0.25, 0.25]).run ()
mc.conv_tol = 1e-10
e3_nosymm = mc.e_states[3]
g3_nosymm = mc.Gradients().kernel(state=3)
print('Gradients of the 3rd excited state')
print(g3_nosymm)
g3_nosymm = mc.nuc_grad_method().kernel(state=3)
print('Gradients of the 3rd excited state')
print(g3_nosymm)

# The active orbitals here should be O atom 2py (b2) and 2pz (a1) and
# two OH antibonding orbitals (a1 and b1). The four states in order
# are singlet A1, triplet B2, singlet B2, and triplet A1.

#
# Use gradients scanner.
#
# Note the returned gradients are based on atomic unit.
#
g_scanner = mc.nuc_grad_method().as_scanner(state=2)
e2_nosymm, g2_nosymm = g_scanner(mol)
print('Gradients of the 2nd excited state')
print(g2_nosymm)

#
# Specify state ID for the gradients of another state.
#
# Unless explicitly specified as an input argument of set_geom_ function,
# set_geom_ function will use the same unit as the one specified in mol.unit.
#
# This has two nearby local minima consisting of different orbitals, although
# the spins and symmetries of the states are the same as above in both cases.
# The local minimum at the state-average energy of -74.7425 Eh has O atom
# 2py (b2), 2pz (a1), 3s (a1), and 3pz (a1) orbitals. That at -74.7415 has O
# atom 2py (b2), 2pz (a1), and 3py (b2) and one OH antibonding orbital (a1). 
mol.set_geom_('''O   0.   0.      0.1
                 H   0.  -0.757   0.587
                 H   0.   0.757   0.587''')
e3_nosymm_shift, g3_nosymm_shift = g_scanner(mol, state=3)
print (g_scanner.base.e_tot, g_scanner.base.e_states)
print('Energy of the 3rd excited state at a shifted geometry =', e3_nosymm_shift,'Eh')
print('Gradients of the 3rd excited state at a shifted geometry:')
print(g3_nosymm_shift)

# 
# State-average mix to average states of selected spins or symmetries
# 
mol = gto.M(
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. ,-0.757  , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
    basis = '631g',
    symmetry = True
)
mol.build ()
mf = scf.RHF (mol).run ()
from pyscf import fci
fcisolvers = [fci.solver (mol, symm=True) for i in (1,2)]
fcisolvers[0].nroots = fcisolvers[1].nroots = 2
fcisolvers[0].wfnsym = 'A1'
fcisolvers[1].wfnsym = 'B2'
mc = mcpdft.CASSCF (mf, 'ftPBE', 4, 4).state_average_mix (fcisolvers, [0.25,]*4)
mc.conv_tol = 1e-10
mc.kernel ()
# The states are now ordered first by solver, then by energy, so the 3rd
# excited state is now at index = 1.
g_scanner = mc.nuc_grad_method ().as_scanner (state=1)
e3_symm, g3_symm = g_scanner (mol)
mol.set_geom_('''O   0.   0.      0.1
                 H   0.  -0.757   0.587
                 H   0.   0.757   0.587''')
e3_symm_shift, g3_symm_shift = g_scanner (mol, state=1)
print('Gradients of the 3rd excited state using symmetry')
print(g3_symm)
print('Gradients of the 3rd excited state at a shifted geometry using symmetry')
print(g3_symm_shift)


