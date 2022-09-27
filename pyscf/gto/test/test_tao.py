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

import numpy
from pyscf import gto

# . . . /
# . . ./
#     /. .
#    / . .
#   /  . .
#
def rotatesub(smat, i0, j0, tao):
    di = abs(tao[i0]) - i0
    dj = abs(tao[j0]) - j0
    rmat = numpy.empty((dj,di),dtype=smat.dtype)
    for i in range(di):
        for j in range(dj):
            rmat[dj-1-j,di-1-i] = smat[i,j]
    if tao[j0] < 0: # |j,-j> => -|j,j>
        rmat[1::2] = -rmat[1::2]
    else:
        rmat[::2] = -rmat[::2]
    if tao[i0] < 0:
        rmat[:,1::2] = -rmat[:,1::2]
    else:
        rmat[:,::2] = -rmat[:,::2]
    return rmat

def rotatesub1(smat, i0, j0, tao):
    di = abs(tao[i0]) - i0
    dj = abs(tao[j0]) - j0
    rmat = numpy.empty((dj,di),dtype=smat.dtype)
    for j in range(dj):
        for i in range(di):
            rmat[j,i] = smat[di-1-i,dj-1-j]
    if tao[j0] < 0: # |j,-j> => -|j,j>
        rmat[1::2] = -rmat[1::2]
    else:
        rmat[::2] = -rmat[::2]
    if tao[i0] < 0:
        rmat[:,1::2] = -rmat[:,1::2]
    else:
        rmat[:,::2] = -rmat[:,::2]
    return rmat

def test_tao():
    mol = gto.Mole()
    mol.atom = [
        ['N', (0.,0.,0.)],
        ['H', (0.,1.,1.)],
        ['H', (1.,0.,1.)],
        ['H', (1.,1.,0.)], ]
    mol.basis = {
        "N": [(0, 0, (15, 1)), ],
        "H": [(0, 0, (1, 1, 0), (3, 3, 1), (5, 1, 0)),
              (1, 0, (1, 1)), (2, 0, (.8, 1)), ]}
    mol.basis['N'].extend(gto.mole.expand_etbs(((0, 4, 1, 1.8),
                                                (1, 3, 2, 1.8),
                                                (2, 2, 1, 1.8),)))

    mol.verbose = 0
    mol.output = None
    mol.build()

    tao = mol.time_reversal_map()

    ao_loc = mol.ao_loc_2c()
    s = mol.intor('cint1e_spnucsp')
    for ish in range(mol.nbas):
        for jsh in range(mol.nbas):
            i0 = ao_loc[ish]
            i1 = abs(tao[i0])
            j0 = ao_loc[jsh]
            j1 = abs(tao[j0])
            rmat = rotatesub(s[i0:i1,j0:j1], i0, j0, tao)
            dd = abs(s[j0:j1,i0:i1]-rmat).sum()
            assert(numpy.allclose(rmat,
                                  rotatesub1(s[i0:i1,j0:j1], i0, j0, tao)))
            assert dd < 1e-12, f'{ish}, {jsh}, {dd}'

    nao = mol.nao_2c()
    j = 0
    idx0 = []
    for i in range(nao):
        if abs(tao[i]) > j:
            idx0.append(i)
            j = abs(tao[i])
    j = 0
    idx1 = [0]
    while idx1[-1] < nao:
        j += 1
        idx1.append(abs(tao[idx1[-1]]))

    assert all(numpy.array(idx0[:j-1]) == numpy.array(idx1[:j-1]))
    print('pass')
