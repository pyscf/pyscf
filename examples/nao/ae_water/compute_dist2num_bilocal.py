from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_conv_yzx2xyz import conv_yzx2xyz_c
from pyscf.nao.m_get_atom2bas_s import get_atom2bas_s
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

tol = 1e-8

ia1, ia2  = 0,   1
phi,theta = 0.0, np.pi/4.0

for r in np.linspace(0.2, 7.0, 15):
  x,y,z = r*np.cos(phi)*np.sin(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(theta)
  mol = gto.M(atom=[[8, [0.0, 0.0, 0.0]], [6, [x, y, z]]], basis='ccpvtz', unit='bohr')
  sv = system_vars_c(gto=mol)
  n1,n2 = sv.atom2s[ia1+1]-sv.atom2s[ia1], sv.atom2s[ia2+1]-sv.atom2s[ia2]
  mol2 = gto.Mole_pure(atom=[mol._atom[ia1], mol._atom[ia2]], basis=mol.basis).build()
  bs = get_atom2bas_s(mol2._bas)
  ss = (bs[0],bs[1], bs[1],bs[2], bs[0],bs[1], bs[1],bs[2])
  eri = mol2.intor('cint2e_sph', shls_slice=ss).reshape([n1,n2,n1,n2])
  eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao', ss).reshape([n1*n2,n1*n2])
  ee,xx = np.linalg.eigh(eri)
  print(r, [x, y, z], n1*n2, list(ee>tol).count(True))
    
