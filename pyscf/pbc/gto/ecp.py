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


def ecp_int(cell, kpts=None, intor='ECPscalar'):
    assert intor in ('ECPscalar', 'ECPso')
    lib.logger.debug(cell, 'PBC-ECP integrals')
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    cell, contr_coeff = _split_basis(cell)
    lib.logger.debug1(cell, 'nao %d -> nao %d', *(contr_coeff.shape))
    nao_sorted, nao = contr_coeff.shape

    ecpcell = cell.copy(deep=False)
    # append a fake s function to mimic the auxiliary index in pbc.incore.
    exp_ptr = cell._ecpbas[-1,gto.PTR_EXP]
    ecpcell._bas = np.array([[0, 0, 1, 1, 0, exp_ptr, 0, 0]], dtype=np.int32)
    # _env[AS_ECPBAS_OFFSET] is to be determined in pbc.incore
    cell._env[gto.AS_NECPBAS] = len(cell._ecpbas)
    # shls_slice of auxiliary index (0,1) corresponds to the fake s function
    shls_slice = (0, cell.nbas, 0, cell.nbas, 0, 1)

    dfbuilder = incore.Int3cBuilder(cell, ecpcell, kpts_lst).build()
    if intor == 'ECPscalar':
        comp = 1
        int3c = dfbuilder.gen_int3c_kernel(intor, aosym='s2', comp=comp,
                                           j_only=True, return_complex=True)
        mat = int3c(shls_slice)
        mat = lib.unpack_tril(mat.reshape(nkpts*comp,-1), lib.HERMITIAN)
        mat = lib.einsum('npq,pi,qj->nij', mat, contr_coeff, contr_coeff)
        mat = mat.reshape(nkpts, comp, nao, nao)
        mat = list(mat[:,0])
        for k, kpt in enumerate(kpts_lst):
            if abs(kpt).max() < 1e-9:  # gamma_point:
                mat[k] = mat[k].real
    else:
        comp = 3
        int3c = dfbuilder.gen_int3c_kernel(intor, aosym='s1', comp=comp,
                                           j_only=True, return_complex=True)
        mat = int3c(shls_slice)
        mat = mat.reshape(nkpts, comp, nao_sorted, nao_sorted)
        mat = lib.einsum('nspq,pi,qj->nsij', mat, contr_coeff, contr_coeff)
        s = .5 * lib.PauliMatrices
        mat = np.einsum('sxy,kspq->kxpyq', -1j * s, mat)
        mat = mat.reshape(nkpts, 2*nao, 2*nao)
    if kpts is None or np.shape(kpts) == (3,):
        mat = mat[0]
    return mat
