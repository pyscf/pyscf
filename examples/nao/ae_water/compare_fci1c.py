from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_conv_yzx2xyz import conv_yzx2xyz_c
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
sv = system_vars_c(gto=mol)
print(timer())
prod_log = prod_log_c(sv.ao_log)
print(timer())
me = ao_matelem_c(prod_log)
print(timer())
m1 = gto.Mole_pure()

for ia,sp in enumerate(sv.atom2sp):
  pab2v=prod_log.sp2vertex[sp]
  #coul=me.coulomb_am(sp, [0.0,0.0,0.0], sp, [0.0,0.0,0.0])
  coul=me.coulomb_ni(sp, [0.0,0.0,0.0], sp, [0.0,0.0,0.0], level=5)
  fci1c = np.einsum('abq,qcd->abcd', np.einsum('pab,pq->abq', pab2v, coul), pab2v)
  print(fci1c.shape, coul.shape)

  m1.build(atom=[mol._atom[ia]], basis=mol.basis)
  eri = m1.intor('cint2e_sph').reshape(fci1c.shape)
  eri = conv_yzx2xyz_c(m1).conv_yzx2xyz_4d(eri, 'pyscf2nao')

  print(abs(fci1c-eri).sum())

