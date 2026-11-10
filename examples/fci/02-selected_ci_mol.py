
#!/usr/bin/env python
#
# Author: Chong Sun <sunchong137@gmail.com>
#
'''
Selected CI for molecules.
'''

from pyscf import gto, scf, ao2mo, fci

mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6"
mol.unit='angstrom'
mol.basis = "631g"
mol.build()

# Run RHF 
# No need to run UHF because SHCI is very accurate.
mf = scf.RHF(mol) 
mf.kernel()

# Rotate the Hamiltonians
norb = mol.nao
nelec = mol.nelectron
mo_coeff = mf.mo_coeff
h1e = mf.get_hcore()
eri = mf._eri
h1e_mo = mo_coeff.T @ h1e @ mo_coeff
eri_mo = ao2mo.kernel(eri, mo_coeff, compact=False).reshape(norb, norb, norb, norb)
e_nuc = mf.energy_nuc()

# Run SCI
scisolver = fci.SCI()
scisolver.max_cycle = 100
scisolver.conv_tol = 1e-8
e, civec = scisolver.kernel(h1e_mo, eri_mo, norb, nelec)
e_sci = e + e_nuc # add nuclear energy
print(f"Selected CI energy: {e_sci}")

# Compared to FCI
fcisolver = fci.FCI(mf)
e_fci, fcivec = fcisolver.kernel() 
print(f"Difference compared to FCI: {e_fci - e_sci}")

# Symmetry-adapted selected CI requires orbital irreps as well as integrals.
# Use a separate symmetry-enabled reference, since the orbitals above were
# optimized without symmetry. D2h is an Abelian subgroup of this linear chain.
from pyscf import symm
from pyscf.fci import selected_ci_symm, selected_ci_spin0_symm
sym_mol = mol.copy()
sym_mol.symmetry = 'D2h'
sym_mol.build(False, False)
sym_mf = scf.RHF(sym_mol).run()
mo = sym_mf.mo_coeff
orbsym = symm.label_orb_symm(sym_mol, sym_mol.irrep_id, sym_mol.symm_orb, mo)
h1 = mo.T @ sym_mf.get_hcore() @ mo
h2 = ao2mo.kernel(sym_mol, mo)
solver = selected_ci_symm.SelectedCI(sym_mol)
solver.orbsym = orbsym
solver.wfnsym = 'Ag'
e_sym, ci_sym = solver.kernel(h1, h2, mo.shape[1], sym_mol.nelectron)
# For a singlet, the spin0 variant also exploits alpha/beta exchange symmetry.
singlet = selected_ci_spin0_symm.SelectedCI(sym_mol)
singlet.orbsym, singlet.wfnsym = orbsym, 'Ag'
e_singlet, ci_singlet = singlet.kernel(h1, h2, mo.shape[1], sym_mol.nelectron)
print('Symmetry-adapted SCI total energies',
      e_sym + sym_mol.energy_nuc(), e_singlet + sym_mol.energy_nuc())
