#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Short range part of ECP under PBC
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.gto import AS_ECPBAS_OFFSET, AS_NECPBAS


def ecp_int(cell, kpts=None):
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    cell, contr_coeff = gto.cell._split_basis(cell)
    lib.logger.debug1(cell, 'nao %d -> nao %d', *(contr_coeff.shape))

    ecpcell = gto.Cell()
    ecpcell._atm = cell._atm
    # append a fictitious s function to mimic the auxiliary index in pbc.incore.
    # ptr2last_env_idx to force PBCnr3c_fill_* function to copy the entire "env"
    ptr2last_env_idx = len(cell._env) - 1
    ecpbas = numpy.vstack([[0, 0, 1, 1, 0, ptr2last_env_idx, 0, 0],
                           cell._ecpbas]).astype(numpy.int32)
    ecpcell._bas = ecpbas
    ecpcell._env = cell._env
    # In pbc.incore _ecpbas is appended to two sets of cell._bas and the
    # fictitious s function.
    cell._env[AS_ECPBAS_OFFSET] = cell.nbas * 2 + 1
    cell._env[AS_NECPBAS] = len(cell._ecpbas)
    # shls_slice of auxiliary index (0,1) corresponds to the fictitious s function
    shls_slice = (0, cell.nbas, 0, cell.nbas, 0, 1)

    kptij_lst = numpy.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)
    buf = incore.aux_e2(cell, ecpcell, 'ECPscalar', aosym='s2',
                        kptij_lst=kptij_lst, shls_slice=shls_slice)
    buf = buf.reshape(len(kpts_lst),-1)
    mat = []
    for k, kpt in enumerate(kpts_lst):
        v = lib.unpack_tril(buf[k], lib.HERMITIAN)
        if abs(kpt).sum() < 1e-9:  # gamma_point:
            v = v.real
        mat.append(reduce(numpy.dot, (contr_coeff.T, v, contr_coeff)))
    if kpts is None or numpy.shape(kpts) == (3,):
        mat = mat[0]
    return mat

