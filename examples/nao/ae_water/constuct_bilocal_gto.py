from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_conv_yzx2xyz import conv_yzx2xyz_c
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
sv = system_vars_c(gto=mol)


for ia1,[sp1,s1,f1] in enumerate(zip(sv.atom2sp, sv.atom2s, sv.atom2s[1:])):
  n1 = f1-s1
  for ia2,sp2,s2,f2 in zip(range(ia1+1,sv.natm+1), sv.atom2sp[ia1+1:], sv.atom2s[ia1+1:], sv.atom2s[ia1+2:]):
    n2 = f2-s2    
    mol2 = gto.Mole_pure(atom=[mol._atom[ia1], mol._atom[ia2]], basis=mol.basis)
    mol2.build()
    print(mol2._bas)
    eri = mol2.intor('cint2e_sph')
    print(' ia1, ia2, n1, n2: ', ia1, ia2, sv.sp2symbol[sp1], sv.sp2symbol[sp2], n1, n2, eri.shape, n1*n2, (n1+n2)**2)

    #eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao')
