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
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.pbc.gto.cell import _split_basis
from pyscf.pbc.df import incore


def ecp_int(cell, kpts=None):
    lib.logger.debug(cell, 'PBC-ECP integrals')
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    cell, contr_coeff = _split_basis(cell)
    lib.logger.debug1(cell, 'nao %d -> nao %d', *(contr_coeff.shape))

    ecpcell = cell.copy(deep=False)
    # append a fake s function to mimic the auxiliary index in pbc.incore.
    exp_ptr = cell._ecpbas[-1,gto.PTR_EXP]
    ecpcell._bas = np.array([[0, 0, 1, 1, 0, exp_ptr, 0, 0]], dtype=np.int32)
    # _env[AS_ECPBAS_OFFSET] is to be determined in pbc.incore
    cell._env[gto.AS_NECPBAS] = len(cell._ecpbas)
    # shls_slice of auxiliary index (0,1) corresponds to the fake s function
    shls_slice = (0, cell.nbas, 0, cell.nbas, 0, 1)

    dfbuilder = incore.Int3cBuilder(cell, ecpcell, kpts_lst).build()
    int3c = dfbuilder.gen_int3c_kernel('ECPscalar', aosym='s2', comp=1,
                                       j_only=True, return_complex=True)
    buf = int3c(shls_slice)
    buf = buf.reshape(len(kpts_lst),-1)
    mat = []
    for k, kpt in enumerate(kpts_lst):
        v = lib.unpack_tril(buf[k], lib.HERMITIAN)
        if abs(kpt).max() < 1e-9:  # gamma_point:
            v = v.real
        mat.append(reduce(np.dot, (contr_coeff.T, v, contr_coeff)))
    if kpts is None or np.shape(kpts) == (3,):
        mat = mat[0]
    return mat
