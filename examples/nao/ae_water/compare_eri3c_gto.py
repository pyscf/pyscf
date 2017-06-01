from __future__ import print_function, division
from pyscf.nao import system_vars_c, ao_matelem_c, prod_log_c, conv_yzx2xyz_c, get_atom2bas_s, eri3c
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
sv = system_vars_c(gto=mol)
prod_log = prod_log_c(sv.ao_log)
print(prod_log.overlap_check())
print(prod_log.lambda_check_overlap())

ia1 = 0
ia2 = 1
ia3 = 0

n1,n2,n3 = [sv.atom2s[ia+1]-sv.atom2s[ia] for ia in [ia1,ia2,ia3]]
mol3 = gto.Mole_pure(atom=[sv._atom[ia1], sv._atom[ia2], sv._atom[ia3]], basis=sv.basis).build()
bs = get_atom2bas_s(mol3._bas)
ss = (bs[2],bs[3], bs[2],bs[3], bs[0],bs[1], bs[1],bs[2])
tci_ao = mol3.intor('cint2e_sph', shls_slice=ss).reshape(n3,n3,n1,n2)
tci_ao = conv_yzx2xyz_c(mol3).conv_yzx2xyz_4d(tci_ao, 'pyscf2nao', ss)


vhpf = prod_log.hartree_pot()
me = ao_matelem_c(sv.ao_log, vhpf)
sp1,sp2,sp3 = [sv.atom2sp[ia] for ia in [ia1,ia2,ia3]]
R1,R2,R3 = [sv.atom2coord[ia] for ia in [ia1,ia2,ia3]]
eri_ni = eri3c(me, sp1, sp2, R1, R2, sp3, R3, level=9)
  
  
print(ia1, ia2, ia3, tci_ao.sum(), eri_ni.sum())
