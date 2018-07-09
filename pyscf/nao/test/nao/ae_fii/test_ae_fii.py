# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from pyscf.nao import system_vars_c, prod_log_c, conv_yzx2xyz_c, get_atom2bas_s, ao_matelem_c
from pyscf.nao.m_system_vars import diag_check, overlap_check
from pyscf.nao.m_prod_log import dipole_check
from pyscf import gto
import numpy as np

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
sv  = system_vars_c().init_pyscf_gto(mol)
prod_log = prod_log_c().init_prod_log_dp(sv.ao_log)
print(prod_log.overlap_check())
print(prod_log.lambda_check_overlap())
print(dipole_check(sv, prod_log))
print('builtin simple center checks done \n')

me = ao_matelem_c(prod_log)

errmx = 0
for ia1 in range(sv.natoms):
  for ia2 in range(sv.natoms):

    n1,n2 = [sv.atom2s[ia+1]-sv.atom2s[ia] for ia in [ia1,ia2]]
    mol3 = gto.Mole_pure(atom=[sv._atom[ia1], sv._atom[ia2]], basis=sv.basis, unit='bohr').build()
    bs = get_atom2bas_s(mol3._bas)
    ss = (bs[0],bs[1], bs[0],bs[1], bs[1],bs[2], bs[1],bs[2])
    tci_ao = mol3.intor('cint2e_sph', shls_slice=ss).reshape(n1,n1,n2,n2)
    tci_ao = conv_yzx2xyz_c(mol3).conv_yzx2xyz_4d(tci_ao, 'pyscf2nao', ss)

    sp1,sp2 = [sv.atom2sp[ia] for ia in [ia1,ia2]]
    R1,R2 = [sv.atom2coord[ia] for ia in [ia1,ia2]]
    pq2v = me.coulomb_am(sp1, R1, sp2, R2)
    tci_ni = np.einsum('abq,qcd->abcd', np.einsum('pab,pq->abq', prod_log.sp2vertex[sp1], pq2v), prod_log.sp2vertex[sp2])
    print(ia1, ia2, abs(tci_ao-tci_ni).sum()/tci_ao.size, abs(tci_ao-tci_ni).max())
    errmx = max(errmx, abs(tci_ao-tci_ni).max())

assert(errmx<3e-5)


