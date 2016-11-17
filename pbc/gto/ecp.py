#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Short range part of ECP under PBC
'''

import copy
import ctypes
import numpy
from pyscf import gto
from pyscf.gto import PTR_ECPBAS_OFFSET, PTR_NECPBAS, PTR_COORD


def ecp_int(cell, kpts=None):
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    Ls = cell.get_lattice_Ls()
    expLk = numpy.asarray(numpy.exp(1j*numpy.dot(Ls, kpts_lst.T)), order='C')

    ecpcell = gto.Mole()
    ecpcell._atm = cell._atm
    # append a single s function for auxiliary index.
    # So the pbc fill_3c driver can handle the 2D integrals in (1,n,n) array
    ecpbas = numpy.vstack([[0, 0, 1, 1, 0, 0, 0, 0], cell._ecpbas]).astype(numpy.int32)
    ecpcell._bas = ecpbas
    ecpcell._env = cell._env

    nao = cell.nao_nr()
    buf = [numpy.zeros((nao,nao), order='F', dtype=numpy.complex128)
           for k in range(nkpts)]
    ints = incore._wrap_int3c(cell, ecpcell, 'ECPscalar_sph', 1, Ls, buf)
    atm, bas, env = ints._envs[:3]
    env[PTR_ECPBAS_OFFSET] = cell.nbas * 2 + 1
    env[PTR_NECPBAS] = len(cell._ecpbas)
    c_shls_slice = (ctypes.c_int*6)(0, cell.nbas, cell.nbas, cell.nbas*2,
                                    cell.nbas*2, cell.nbas*2+1)

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coordL = atm[:cell.natm,PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    for l, L1 in enumerate(Ls):
        env[ptr_coordL] = xyz + L1
        exp_Lk = numpy.einsum('k,ik->ik', expLk[l].conj(), expLk[:l+1])
        exp_Lk = numpy.asarray(exp_Lk, order='C')
        exp_Lk[l] = .5
        ints(exp_Lk, c_shls_slice)

    for k, kpt in enumerate(kpts_lst):
        if abs(kpt).sum() < 1e-6:
            buf[k] = buf[k].real + buf[k].real.T
        else:
            buf[k] = buf[k] + buf[k].T.conj()
    if kpts is None or numpy.shape(kpts) == (3,):
        buf = buf[0]
    return buf

