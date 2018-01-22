#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Short range part of ECP under PBC
'''

import copy
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.gto import AS_ECPBAS_OFFSET, AS_NECPBAS


def ecp_int(cell, kpts=None):
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    cell, contr_coeff = _uncontract_cell(cell)
    lib.logger.debug1(cell, 'nao %d -> nao %d', contr_coeff.shape)

    ecpcell = gto.Mole()
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
    buf = incore.aux_e2(cell, ecpcell, 'ECPscalar_sph', aosym='s2',
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

_THR = [1.0, 0.5, 0.25, 0.1, 0]
def _uncontract_cell(cell):
    _bas = []
    _env = cell._env.copy()
    contr_coeff = []
    for ib in range(cell.nbas):
        pexp = cell._bas[ib,gto.PTR_EXP]
        pcoeff1 = cell._bas[ib,gto.PTR_COEFF]
        np = cell.bas_nprim(ib)
        nc = cell.bas_nctr(ib)
        es = cell.bas_exp(ib)
        l = cell.bas_angular(ib)
        if cell.cart:
            degen = (l + 1) * (l + 2) // 2
        else:
            degen = l * 2 + 1

        cs = cell._env[pcoeff1:pcoeff1+np*nc].reshape(nc,np).T.copy()
        mask = numpy.ones(es.size, dtype=bool)
        count = 0
        for thr in _THR:
            idx = numpy.where(mask & (es >= thr))[0]
            np1 = len(idx)
            if np1 > 0:
                pcoeff0, pcoeff1 = pcoeff1, pcoeff1 + np1 * nc
                cs1 = cs[idx]
                _env[pcoeff0:pcoeff1] = cs1.T.ravel()
                btemp = cell._bas[ib].copy()
                btemp[gto.NPRIM_OF] = np1
                btemp[gto.PTR_COEFF] = pcoeff0
                btemp[gto.PTR_EXP] = pexp
                _bas.append(btemp)
                mask[idx] = False
                pexp += np1
                count += 1
        contr_coeff.append(numpy.vstack([numpy.eye(degen*nc)] * count))

    pcell = copy.copy(cell)
    pcell._bas = numpy.asarray(numpy.vstack(_bas), dtype=numpy.int32)
    pcell._env = _env
    return pcell, scipy.linalg.block_diag(*contr_coeff)
