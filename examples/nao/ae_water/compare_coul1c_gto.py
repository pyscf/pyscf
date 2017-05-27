from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_conv_yzx2xyz import conv_yzx2xyz_c
from pyscf import gto
import numpy as np
from timeit import default_timer as timer
from scipy.sparse import csr_matrix

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvtz') # coordinates in Angstrom!
sv = system_vars_c(gto=mol)
t1s = timer()
prod_log = prod_log_c(sv.ao_log, tol=1e-9)
print(timer() - t1s, 'time prod_log')
print(prod_log.lambda_check_coulomb())

me = ao_matelem_c(prod_log)

for ia,sp in enumerate(sv.atom2sp):
  print('\n atom, specie: ', ia, sp)
  pab2v=prod_log.sp2vertex[sp]
  n = pab2v.shape[1]
  pab_shape = [pab2v.shape[0], n*n]
  pab2v_csr = csr_matrix(pab2v.reshape(pab_shape))
  
  vv = np.linalg.inv( (pab2v_csr*pab2v_csr.transpose()).todense() )
  
  coul = me.coulomb_am(sp, [0.0,0.0,0.0], sp, [0.0,0.0,0.0])

  t1s  = timer()
  fci1c = ((coul*pab2v_csr).transpose()*pab2v_csr).reshape([n,n,n,n])
  print(timer() - t1s, 'time fci1c')
  
  mol1 = gto.Mole_pure(atom=[mol._atom[ia]], basis=mol.basis).build()
  eri = mol1.intor('cint2e_sph').reshape(fci1c.shape)
  eri = conv_yzx2xyz_c(mol1).conv_yzx2xyz_4d(eri, 'pyscf2nao')
  eri2 = eri.reshape([n*n, n*n])
  
  eritt = (pab2v_csr*eri2)*pab2v_csr.transpose()
  coul_gto = np.array((vv*eritt)*vv)

  t1s  = timer()
  fci1c_gto = ((coul_gto*pab2v_csr).transpose()*pab2v_csr).reshape([n,n,n,n])
  print(timer() - t1s, 'time fci1c_coul_gto')
  
  print(fci1c.shape, coul.shape, 
    abs(coul_gto-coul).sum()/coul.size, 
    abs(fci1c_gto-eri).sum()/eri.size,
    abs(fci1c_gto-eri).max(), 
    abs(fci1c-eri).sum()/eri.size,
    abs(fci1c-eri).max())
