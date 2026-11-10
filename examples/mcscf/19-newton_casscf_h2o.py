#!/usr/bin/env python

"""Newton-CASSCF for water, with and without spatial symmetry.

Use four active electrons in six orbitals. Compare singlet and triplet
calculations, enable internal active-orbital rotations, and select a B1 state
with symmetry adaptation. Reference energies come from the original module
demonstrations; different converged orbital solutions can have different energies.
"""

from pyscf import gto, scf, mcscf, fci

mol = gto.M(
    atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
    basis='cc-pvdz', symmetry=False, verbose=4)
mf = scf.RHF(mol).run()

# Without symmetry: start from selected RHF orbitals (one-based indices).
mc = mcscf.CASSCF(mf, 6, 4).newton()
mc.fcisolver = fci.solver(mol)
mo = mc.sort_mo((3, 4, 6, 7, 8, 9), base=1)
mc.kernel(mo)
print('Singlet energy:', mc.e_tot)
# The original demonstration listed two stationary-solution references.
print('Differences from singlet references:',
      mc.e_tot - (-76.0873923174), mc.e_tot - (-76.0926176464))

# Three alpha and one beta active electrons select Ms=1.
triplet = mcscf.CASSCF(mf, 6, (3, 1)).newton()
triplet.fcisolver = fci.solver(mol, singlet=False)
mo_triplet = triplet.sort_mo((3, 4, 6, 7, 8, 9), base=1)
triplet.kernel(mo_triplet)
print('Triplet reference-energy difference:', triplet.e_tot - (-75.7155632535814))

# Active-active rotations are redundant with an exact CAS solver, but can
# be included explicitly. Repeat from the same initial orbitals.
triplet.internal_rotation = True
triplet.kernel(mo_triplet)
print('Triplet with internal rotations, reference-energy difference:',
      triplet.e_tot - (-75.7155632535814))

# Repeat with C2v spatial symmetry enabled.
mol_symm = mol.copy()
mol_symm.symmetry = True
mol_symm.build()
mf_symm = scf.RHF(mol_symm).run()

mc_symm = mcscf.CASSCF(mf_symm, 6, 4).newton()
mc_symm.fcisolver = fci.solver(mol_symm)
mo_symm = mc_symm.sort_mo((3, 4, 6, 7, 8, 9), base=1)
mc_symm.kernel(mo_symm)
print('Symmetry-adapted singlet energy:', mc_symm.e_tot)
print('Differences from singlet references:',
      mc_symm.e_tot - (-76.0873923174), mc_symm.e_tot - (-76.0926176464))

triplet_symm = mcscf.CASSCF(mf_symm, 6, (3, 1)).newton()
triplet_symm.fcisolver = fci.solver(mol_symm, singlet=False)
triplet_symm.kernel(mo_symm)
print('Symmetry-adapted triplet reference-energy difference:',
      triplet_symm.e_tot - (-75.7155632535814))

# Explicitly target a different spatial symmetry for the triplet CI state.
triplet_b1 = mcscf.CASSCF(mf_symm, 6, (3, 1)).newton()
triplet_b1.fcisolver.wfnsym = 'B1'
# The original selection has only A1/B2 active orbitals and cannot form B1.
# Include orbital 5 (B1) instead of orbital 3 (B2) for this calculation.
mo_b1 = triplet_b1.sort_mo((4, 5, 6, 7, 8, 9), base=1)
triplet_b1.kernel(mo_b1)
print('B1 triplet energy:', triplet_b1.e_tot)
# The original block printed -75.6406597705231 as a reference, but its
# B1 calculation is not executable with the original active-space selection.
