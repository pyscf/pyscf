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
from pyscf.pbc.df import incore, ft_ao
from pyscf.pbc.tools import k2gamma


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

    dfbuilder = ECPInt3cBuilder(cell, ecpcell, kpts_lst).build()
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

class ECPInt3cBuilder(incore.Int3cBuilder):
    def build(self):
        log = lib.logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, incore.RCUT_THRESHOLD, verbose=log)

        rcut = estimate_rcut(cell)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max(), log)
        self.supmol = supmol.strip_basis(rcut)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())
        return self

def estimate_rcut(cell, precision=None):
    '''Estimate rcut for integrals of three Guassian production'''
    if precision is None:
        precision = cell.precision

    cell_exps = np.array([e.min() for e in cell.bas_exps()])
    if cell_exps.size == 0:
        return np.zeros(1)

    ls = cell._bas[:,gto.ANG_OF]
    cs = gto.gto_norm(ls, cell_exps)
    ai_idx = cell_exps.argmin()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    li = cell._bas[ai_idx,gto.ANG_OF]
    lj = ls
    ci = cs[ai_idx]
    cj = cs

    # Three GTO overlap ~ exp(-ai(Ri^2-Rc^2)-aj(Rj^2-Rc^2)-ak(Rk^2-Rc^2))
    # Rc = (ai*Ri + aj*Rj + ak*Rk) / (ai+aj+ak)
    # Let Rk = 0, given Rj=Rcut, approximately, the maximum value of this
    # overlap ~ exp(-theta Rcut^2) when Ri = aj*Rj/(aj+ak) and theta = 1/(1/aj+1/ak)

    aux_es = cell._env[cell._ecpbas[:,gto.PTR_EXP]]
    aux_cs = cell._env[cell._ecpbas[:,gto.PTR_COEFF]]
    aux_order = cell._ecpbas[:,gto.RADI_POWER]
    r0_pool = []
    for ak, ck, lk in zip(aux_es, aux_cs, aux_order):
        aij = ai + aj
        lij = li + lj
        l3 = lij + lk
        theta1 = 1./(1./aj + 1./ak)
        norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
        c1 = ci * cj * abs(ck) * norm_ang

        fac = 2**(li+1)*np.pi**2.5*c1 / (aij**(li+1.5) * aj**lj)
        fac *= 1. / precision

        r0 = cell.rcut
        r0 = (np.log(fac * r0**(l3+1) + 1.) / theta1)**.5
        r0 = (np.log(fac * r0**(l3+1) + 1.) / theta1)**.5
        r0_pool.append(r0)

    if r0_pool:
        return np.max(r0_pool, axis=0)
    else:
        return np.zeros(1)
