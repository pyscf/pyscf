
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
print("Selected CI energy: {}".format(e_sci))

# Compared to FCI
fcisolver = fci.FCI(mf)
e_fci, fcivec = fcisolver.kernel() 
print("Difference compared to FCI: {}".format(e_fci - e_sci))
