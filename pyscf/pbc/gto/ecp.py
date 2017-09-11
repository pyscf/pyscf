#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Short range part of ECP under PBC
'''

import numpy
from pyscf import lib
from pyscf import gto
from pyscf.gto import PTR_ECPBAS_OFFSET, PTR_NECPBAS


def ecp_int(cell, kpts=None):
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

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
    cell._env[PTR_ECPBAS_OFFSET] = cell.nbas * 2 + 1
    cell._env[PTR_NECPBAS] = len(cell._ecpbas)
    # shls_slice of auxiliary index (0,1) corresponds to the fictitious s function
    shls_slice = (0, cell.nbas, 0, cell.nbas, 0, 1)

    kptij_lst = numpy.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)
    buf = incore.aux_e2(cell, ecpcell, 'ECPscalar_sph', aosym='s2',
                        kptij_lst=kptij_lst, shls_slice=shls_slice)
    buf = buf.reshape(len(kpts_lst),-1)
    mat = []
    for k, kpt in enumerate(kpts_lst):
        v = lib.unpack_tril(buf[k])
        if abs(kpt).sum() < 1e-9:  # gamma_point:
            v = v.real
        mat.append(v)
    if kpts is None or numpy.shape(kpts) == (3,):
        mat = mat[0]
    return mat

