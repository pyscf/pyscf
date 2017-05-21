from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_conv_yzx2xyz import conv_yzx2xyz_c
from pyscf.nao.m_get_atom2bas_s import get_atom2bas_s
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

tol = 1e-5
mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvtz')
sv = system_vars_c(gto=mol)
prod_log = prod_log_c(sv.ao_log, tol=tol)
print(prod_log.sp2norbs)

for ia1,n1 in zip(range(sv.natm), sv.atom2s[1:]-sv.atom2s[0:-1]):
  for ia2,n2 in zip(range(ia1+1,sv.natm+1), sv.atom2s[ia1+2:]-sv.atom2s[ia1+1:-1]):

    mol2 = gto.Mole_pure(atom=[mol._atom[ia1], mol._atom[ia2]], basis=mol.basis).build()
    bs = get_atom2bas_s(mol2._bas)
    ss = (bs[0],bs[1], bs[1],bs[2], bs[0],bs[1], bs[1],bs[2])
    eri = mol2.intor('cint2e_sph', shls_slice=ss).reshape([n1,n2,n1,n2])
    eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao', ss).reshape([n1*n2,n1*n2])
    ee,xx = np.linalg.eigh(eri)
    
    print(' ia1, ia2, n1, n2: ', ia1, ia2, n1, n2, eri.shape, n1*n2, list(ee>tol).count(True))
    
