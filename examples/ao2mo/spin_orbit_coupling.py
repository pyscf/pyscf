#
# Spin-orbit coupling integrals in MO space
#

import numpy
from pyscf import gto, scf, ao2mo, mcscf
mol = gto.Mole()
mol.atom = 'O 0 0 0; O 0 0 1.2'
mol.basis = 'ccpvdz'
mol.spin = 2
mol.build()
myhf = scf.RHF(mol)
myhf.kernel()
mycas = mcscf.CASSCF(myhf, 6, 8) # 6 orbital, 8 electron
mycas.kernel()

cas_orb = mycas.mo_coeff[:,mycas.ncore:mycas.ncore+mycas.ncas] # CAS space orbitals
mol.set_rinv_origin_(mol.atom_coord(1)) # set the gauge origin on second atom
h1 = mol.intor('cint1e_prinvxp_sph', comp=3) # 1-electron SOC operator; comp=3 for x,y,z components
h1 = numpy.array([reduce(numpy.dot, (cas_orb.T, v, cas_orb)) for v in h1])
h2 = ao2mo.kernel(mol, cas_orb, intor='cint2e_p1vxp1_sph', comp=3, aosym='s1') # SSO
print(h1.shape)
print(h2.shape)
