from pyscf import gto

mol = gto.M(atom='He 0 0 0', basis='631g')
eri0 = mol.intor('cint2e_sph')

# erf(omega * r12) / r12
omega = .5
mol.set_range_coulomb(omega)
erf_r12 = mol.intor('cint2e_sph')
# Switch it off by setting range_coulomb parameter to 0
mol.set_range_coulomb(0)

erfc_r12 = eri0 - erf_r12

print(eri0)
print(erf_r12)
print(erfc_r12)
