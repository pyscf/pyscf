#!/usr/bin/env python
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

import unittest
from pyscf import lib
from pyscf.pbc import scf, gto, grad, tools
import numpy as np
from functools import reduce

def finger(mat):
    return abs(mat).sum()

disp= 1e-5
atm0 = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]]
atm1 = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2]]]
atm2 = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2]]]
gatm0 = [['GHOST-C', [0.0, 0.0, 0.0]], ['GHOST-C', [1.685068664391,1.685068664391,1.685068664391]]]
gatm1 = [['GHOST-C', [0.0, 0.0, 0.0]], ['GHOST-C', [1.685068664391,1.685068664391,1.685068664391+disp/2]]]
gatm2 = [['GHOST-C', [0.0, 0.0, 0.0]], ['GHOST-C', [1.685068664391,1.685068664391,1.685068664391-disp/2]]]
cell = gto.Cell()
cell.atom= atm0
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.basis = 'gth-szv'
cell.verbose= 4
cell.pseudo = 'gth-pade'
cell.unit = 'bohr'
cell.build()



kpts = cell.make_kpts([1,1,3])
madelung = tools.pbc.madelung(cell, kpts)
s = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
s1e = np.asarray(cell.pbc_intor("int1e_ipovlp", kpts=kpts))
aoslice = cell.aoslice_by_atom()
p0, p1 = aoslice[1][2:]
cell1 = cell.set_geom_(atm1, inplace=False)
cell2 = cell.set_geom_(atm2, inplace=False)
fcell1 = cell.set_geom_(atm0+gatm1, inplace=False)
fcell2 = cell.set_geom_(atm0+gatm2, inplace=False)

dm = np.load('dm.npy')
nao = cell.nao_nr()
nk = len(kpts)
dm0 = np.zeros([nk,2*nao,2*nao], dtype=dm.dtype)
dm0[:,:nao,:nao] = dm

mf = scf.KRHF(cell, kpts, exxdiv=None)
mfe = scf.KRHF(cell, kpts, exxdiv="ewald")
mygrad = grad.KRHF(mf)
mygrade = grad.KRHF(mfe)

mf1 = scf.KRHF(cell1, kpts, exxdiv=None)
mf2 = scf.KRHF(cell2, kpts, exxdiv=None)
fmf1 = scf.KRHF(fcell1, kpts, exxdiv=None)
fmf2 = scf.KRHF(fcell2, kpts, exxdiv=None)

mfe1 = scf.KRHF(cell1, kpts, exxdiv="ewald")
mfe2 = scf.KRHF(cell2, kpts, exxdiv="ewald")

fmfe1 = scf.KRHF(fcell1, kpts, exxdiv="ewald")
fmfe2 = scf.KRHF(fcell2, kpts, exxdiv="ewald")

k1 = mf.get_k(dm_kpts=dm)
ke1 = mfe.get_k(dm_kpts=dm)
tmp = k1 + madelung * np.einsum('kpq,kqr,krs->kps', s, dm, s)
fs1 = np.asarray(fcell1.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
fs2 = np.asarray(fcell1.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
print(np.linalg.norm(ke1-tmp))
exit()


ke = mygrad.get_k(dm)

vk1 = fmf1.get_k(dm_kpts=dm0)[:,nao:,:nao]
vk2 = fmf2.get_k(dm_kpts=dm0)[:,nao:,:nao]
fin = (vk1-vk2)/disp
tmp = np.zeros(fin.shape, dtype=fin.dtype)
p0, p1 = aoslice[1][2:]
tmp[:,p0:p1] = ke[2,:,p0:p1]

e = np.einsum('kij,kji->', fin, dm)
#print("None",np.linalg.norm(fin),np.linalg.norm(tmp),np.linalg.norm(fin-tmp), e)




kee = mygrade.get_k(dm)

vke1 = fmfe1.get_k(dm_kpts=dm0)[:,nao:,:nao]
vke2 = fmfe2.get_k(dm_kpts=dm0)[:,nao:,:nao]
fine = (vke1-vke2)/disp

tmpe = np.zeros(fine.shape, dtype=fin.dtype)
tmpe[:,p0:p1] = kee[2,:,p0:p1]
ee = np.einsum('kij,kji->', fine, dm)
ete = np.einsum('kij,kji->', tmpe, dm)
#print("EWALD",np.linalg.norm(fine),np.linalg.norm(tmpe),np.linalg.norm(fine-tmpe), ee, ete)
