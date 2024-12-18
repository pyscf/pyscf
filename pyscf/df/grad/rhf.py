#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#

#
# Copyright 2019 Tencent America LLC. All Rights Reserved.
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

import time
import numpy
import ctypes
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import df
from pyscf.ao2mo.outcore import balance_partition
from pyscf.gto.moleintor import getints, make_cintopt
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from functools import reduce
from itertools import product
from pyscf.ao2mo import _ao2mo
from pyscf.df import df_jk
from pyscf.df.incore import LINEAR_DEP_THR

LINEAR_DEP_THRESHOLD = LINEAR_DEP_THR

def get_jk(mf_grad, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
           decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):
    '''
    Computes J and K derivatives with density fitting

    Args:
        mf_grad : instance of :class:`Gradients`

        mol : instance of :class:`gto.Mole`

        dm: numpy.ndarray
            Zeroth order density matrix
        hermi: int
            Is the density matrix hermitian or not
        with_j: boolean
            Whether to compute J matrix
        with_k: boolean
            Whether to compute K matrix
        decompose_j2c: string
            The method to decompose the metric defined by int2c. It can be set
            to CD (cholesky decomposition) or ED (eigenvalue decomposition).
        lindep : float
            The threshold to discard linearly dependent basis when decompose_j2c
            is set to ED.
    '''
    assert (with_j or with_k)
    if not with_k:
        return get_j (mf_grad, mol=mol, dm=dm, hermi=hermi), None
    t0 = (logger.process_clock (), logger.perf_counter ())
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nbas, nao, naux = mol.nbas, mol.nao, auxmol.nao
    aux_loc = auxmol.ao_loc

    # Density matrix preprocessing
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    # For j
    idx = numpy.arange(nao)
    idx = idx * (idx+1) // 2 + idx
    dm_tril = dms + dms.transpose(0,2,1)
    dm_tril = lib.pack_tril(dm_tril)
    dm_tril[:,idx] *= .5

    # For k
    if hermi == 1:
        orbol, orbor = _decompose_rdm1 (mf_grad, mol, dm)
    else:
        orbol, orbor = _decompose_rdm1_svd (mf_grad, mol, dm)
    nocc = [o.shape[-1] for o in orbor]

    # Coulomb: (P|Q) D_Q = (P|uv) D_uv for D_Q ("rhoj")
    # Exchange: (P|Q) D_Qui = (P|uv) C_vi n_i for D_Qui ("rhok")
    rhoj, get_rhok = _cho_solve_rhojk (mf_grad, mol, auxmol, orbol, orbor,
                                       decompose_j2c, lindep)

    # (d/dX i,j|P)
    t1 = (logger.process_clock (), logger.perf_counter ())
    vj = numpy.zeros((nset,3,nao,nao))
    vk = numpy.zeros((nset,3,nao,nao))
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    ao_ranges = balance_partition(aux_loc, blksize)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s1 # MO output index slower than AO output index; input AOs are asymmetric
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv # comp and aux indices are slower
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s1 # input is not tril_packed
    null = lib.c_null_ptr()
    t2 = t1
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1)).transpose (0,3,2,1)  # (P|mn'), row-major order
        t2 = logger.timer_debug1 (mf_grad, "df grad intor (P|mn')", *t2)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> multithread efficiency
            vj[i,0] += numpy.dot (rhoj[i,p0:p1], int3c[0].reshape (p1-p0, -1)).reshape (nao, nao).T
            vj[i,1] += numpy.dot (rhoj[i,p0:p1], int3c[1].reshape (p1-p0, -1)).reshape (nao, nao).T
            vj[i,2] += numpy.dot (rhoj[i,p0:p1], int3c[2].reshape (p1-p0, -1)).reshape (nao, nao).T
            t2 = logger.timer_debug1 (mf_grad, "df grad einsum rho_P (P|mn') rho_P", *t2)
            tmp = numpy.empty ((3,p1-p0,nocc[i],nao), dtype=orbol[0].dtype)
            fdrv(ftrans, fmmm, # lib.einsum ('xpmn,mi->xpin', int3c, orbol[i])
                 tmp.ctypes.data_as(ctypes.c_void_p),
                 int3c.ctypes.data_as(ctypes.c_void_p),
                 orbol[i].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                 (ctypes.c_int*4)(0, nocc[i], 0, nao),
                 null, ctypes.c_int(0))
            t2 = logger.timer_debug1 (mf_grad, "df grad einsum (P|mn') u_mi = dg_Pin", *t2)
            rhok = get_rhok (i, p0, p1)
            vk[i] += lib.einsum('xpoi,pok->xik', tmp, rhok)
            t2 = logger.timer_debug1 (mf_grad, "df grad einsum D_Pim dg_Pin = v_ij", *t2)
            rhok = tmp = None
        int3c = None
    t1 = logger.timer_debug1 (mf_grad, 'df grad vj and vk AO (P|mn) D_P eval', *t1)

    if not mf_grad.auxbasis_response:
        vj = -vj.reshape(out_shape)
        vk = -vk.reshape(out_shape)
        logger.timer (mf_grad, 'df grad vj and vk', *t0)
        if with_j: return vj, vk
        else: return None, vk

    ####### BEGIN AUXBASIS PART #######

    # ao2mo the final AO index of rhok and store in "rhok_oo":
    # dPiu C_uj -> dPij. *Not* symmetric i<->j: "i" has an occupancy
    # factor and "j" must not.
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao*max (nocc)), 20), naux))
    rhok_oo = []
    for i, j in product (range (nset), repeat=2):
        tmp = numpy.empty ((naux,nocc[i],nocc[j]))
        for p0, p1 in lib.prange(0, naux, blksize):
            rhok = get_rhok (i, p0, p1).reshape ((p1-p0)*nocc[i], nao)
            tmp[p0:p1] = lib.dot (rhok, orbol[j]).reshape (p1-p0, nocc[i], nocc[j])
        rhok_oo.append(tmp)
        rhok = tmp = None
    t1 = logger.timer_debug1 (mf_grad, 'df grad vj and vk aux d_Pim u_mj = d_Pij eval', *t1)

    vjaux = numpy.zeros((nset,nset,3,naux))
    vkaux = numpy.zeros((nset,nset,3,naux))
    # (i,j|d/dX P)
    t2 = t1
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2 # MO output index slower than AO output index; input AOs are symmetric
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv # comp and aux indices are slower
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2 # input is tril_packed
    null = lib.c_null_ptr()
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
        t2 = logger.timer_debug1 (mf_grad, "df grad intor (P'|mn)", *t2)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        drhoj = lib.dot (int3c.transpose (0,2,1).reshape (3*(p1-p0), -1),
            dm_tril.T).reshape (3, p1-p0, -1) # xpij,mij->xpm
        vjaux[:,:,:,p0:p1] = lib.einsum ('xpm,np->mnxp', drhoj, rhoj[:,p0:p1])
        t2 = logger.timer_debug1 (mf_grad, "df grad vj aux (P'|mn) eval", *t2)
        # MRH, 09/19/2022: This is a different order of operations than PySCF v2.1.0. There,
        #                  the dense matrix rhok_oo is transformed into the larger AO basis.
        #                  Here, the sparse matrix int3c is transformed into the smaller MO
        #                  basis. The latter approach is obviously more performant.
        for i in range (nset):
            buf = numpy.empty ((3, p1-p0, nocc[i], nao), dtype=orbol[i].dtype)
            fdrv(ftrans, fmmm, # lib.einsum ('pmn,ni->pim', int3c, orbol[i])
                 buf.ctypes.data_as(ctypes.c_void_p),
                 int3c.ctypes.data_as(ctypes.c_void_p),
                 orbol[i].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                 (ctypes.c_int*4)(0, nocc[i], 0, nao),
                 null, ctypes.c_int(0))
            for j in range (nset): # lib.einsum ('pim,mj->pij', buf, orbor[j])
                int3c_ij = lib.dot (buf.reshape (-1, nao), orbor[j])
                int3c_ij = int3c_ij.reshape (3, p1-p0, nocc[i], nocc[j])
                rhok_oo_ij = rhok_oo[(i*nset)+j][p0:p1]
                vkaux[i,j,:,p0:p1] += lib.einsum('xpij,pij->xp', int3c_ij,
                                                 rhok_oo_ij)
        t2 = logger.timer_debug1 (mf_grad, "df grad vk aux (P'|mn) eval", *t2)
    int3c = tmp = None
    t1 = logger.timer_debug1 (mf_grad, "df grad vj and vk aux (P'|mn) eval", *t1)

    # (d/dX P|Q)
    int2c_e1 = auxmol.intor('int2c2e_ip1')
    vjaux -= lib.einsum('xpq,mp,nq->mnxp', int2c_e1, rhoj, rhoj)
    for i, j in product (range (nset), repeat=2):
        k = (i*nset) + j
        l = (j*nset) + i
        tmp = lib.einsum('pij,qji->pq', rhok_oo[k], rhok_oo[l])
        vkaux[i,j] -= lib.einsum('xpq,pq->xp', int2c_e1, tmp)
    t1 = logger.timer_debug1 (mf_grad, "df grad vj and vk aux (P'|Q) eval", *t1)

    auxslices = auxmol.aoslice_by_atom()
    vjaux = numpy.array ([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
    vkaux = numpy.array ([-vkaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])

    vjaux = numpy.ascontiguousarray (vjaux.transpose (1,2,0,3))
    vkaux = numpy.ascontiguousarray (vkaux.transpose (1,2,0,3))

    vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
    vk = lib.tag_array(-vk.reshape(out_shape), aux=numpy.array(vkaux))
    logger.timer (mf_grad, 'df grad vj and vk', *t0)
    if with_j: return vj, vk
    else: return None, vk

def get_j(mf_grad, mol=None, dm=None, hermi=0):
    if mol is None: mol = mf_grad.mol
    if dm is None: dm = mf_grad.base.make_rdm1()
    t0 = (logger.process_clock (), logger.perf_counter ())

    with_df = mf_grad.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nbas = mol.nbas

    get_int3c_s2 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's2ij')
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')

    nao = mol.nao
    naux = auxmol.nao
    dms = numpy.asarray(dm)
    out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    idx = numpy.arange(nao)
    idx = idx * (idx+1) // 2 + idx
    dm_tril = dms + dms.transpose(0,2,1)
    dm_tril = lib.pack_tril(dm_tril)
    dm_tril[:,idx] *= .5

    aux_loc = auxmol.ao_loc
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory * .5e6/8 / (nao**2*3), 20), naux, 240))
    ao_ranges = balance_partition(aux_loc, blksize)

    # (i,j|P)
    rhoj = numpy.empty((nset,naux))
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_s2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        rhoj[:,p0:p1] = lib.einsum('wp,nw->np', int3c, dm_tril)
        int3c = None

    # (P|Q)
    int2c = auxmol.intor('int2c2e', aosym='s1')
    rhoj = scipy.linalg.solve(int2c, rhoj.T, assume_a='pos').T
    int2c = None

    # (d/dX i,j|P)
    vj = numpy.zeros((nset,3,nao,nao))
    for shl0, shl1, nL in ao_ranges:
        int3c = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        vj += lib.einsum('xijp,np->nxij', int3c, rhoj[:,p0:p1])
        int3c = None

    if mf_grad.auxbasis_response:
        # (i,j|d/dX P)
        vjaux = numpy.empty((nset,nset,3,naux))
        for shl0, shl1, nL in ao_ranges:
            int3c = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vjaux[:,:,:,p0:p1] = lib.einsum('xwp,mw,np->mnxp',
                                          int3c, dm_tril, rhoj[:,p0:p1])
            int3c = None

        # (d/dX P|Q)
        int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        vjaux -= lib.einsum('xpq,mp,nq->mnxp', int2c_e1, rhoj, rhoj)

        auxslices = auxmol.aoslice_by_atom()
        vjaux = numpy.array ([-vjaux[:,:,:,p0:p1].sum(axis=3) for p0, p1 in auxslices[:,2:]])
        vjaux = numpy.ascontiguousarray (vjaux.transpose (1,2,0,3))
        vj = lib.tag_array(-vj.reshape(out_shape), aux=numpy.array(vjaux))
    else:
        vj = -vj.reshape(out_shape)
    logger.timer (mf_grad, 'df vj', *t0)
    return vj

def _int3c_wrapper(mol, auxmol, intor, aosym):
    ''' Convenience wrapper for getints '''
    nbas = mol.nbas
    pmol = mol + auxmol
    intor = mol._add_suffix(intor)
    opt = make_cintopt(mol._atm, mol._bas, mol._env, intor)
    def get_int3c(shls_slice=None):
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, nbas, pmol.nbas)
        else:
            shls_slice = shls_slice[:4] + (nbas+shls_slice[4], nbas+shls_slice[5])
        return getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice,
                       aosym=aosym, cintopt=opt)
    return get_int3c

def _decompose_rdm1_svd (mf_grad, mol, dm):
    '''Decompose dms as U.Vh using SVD

    Args:
        mf_grad : instance of :class:`Gradients`
        mol : instance of :class:`gto.Mole`
        dm : ndarray or sequence of ndarrays of shape (nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    nao = mol.nao
    dms = numpy.asarray(dm).reshape (-1,nao,nao)
    orbor = []
    orbol = []
    for dm in dms:
        u, s, vh = numpy.linalg.svd (dm)
        idx = numpy.abs (s)>1e-8
        orbol.append (numpy.asfortranarray (u[:,idx]))
        orbor.append (numpy.asfortranarray (lib.einsum('i,ip->pi', s[idx], vh[idx])))

    return orbol, orbor

def _decompose_rdm1 (mf_grad, mol, dm):
    '''Decompose dms as U.Vh, where
    U = orbol = eigenvectors
    V = orbor = U * eigenvalues

    Args:
        mf_grad : instance of :class:`Gradients`
        mol : instance of :class:`gto.Mole`
        dm : ndarray or sequence of ndarrays of shape (nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    nao = mol.nao
    dms = numpy.asarray(dm).reshape (-1,nao,nao)
    nset = dms.shape[0]
    if hasattr (dm, 'mo_coeff') and hasattr (dm, 'mo_occ'):
        mo_coeff = dm.mo_coeff
        mo_occ = dm.mo_occ
        if getattr(mo_occ, 'ndim', None) == 1: # RHF orbitals
            mo_coeff = [mo_coeff]
            mo_occ = [mo_occ]
    else:
        s0 = mol.intor ('int1e_ovlp')
        mo_occ = []
        mo_coeff = []
        for dm in dms:
            sdms = reduce (lib.dot, (s0, dm, s0))
            n, c = scipy.linalg.eigh (sdms, b=s0)
            mo_occ.append (n)
            mo_coeff.append (c)
        mo_occ = numpy.stack (mo_occ, axis=0)

    orbor = []
    orbol = []
    for i in range(nset):
        idx = numpy.abs (mo_occ[i])>1e-8
        c = mo_coeff[i][:,idx]
        orbol.append (numpy.asfortranarray (c))
        cn = lib.einsum('pi,i->pi', c, mo_occ[i][idx])
        orbor.append (numpy.asfortranarray (cn))

    return orbol, orbor

def _gen_metric_solver(int2c, decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):
    if decompose_j2c.upper() == 'CD':
        try:
            j2c = scipy.linalg.cho_factor(int2c, lower=True)
            def j2c_solver(v):
                return scipy.linalg.cho_solve(j2c, v, overwrite_b=True)
            return j2c_solver

        except (numpy.linalg.LinAlgError, scipy.linalg.LinAlgError):
            pass

    w, v = scipy.linalg.eigh(int2c)
    mask = w > lindep
    v1 = v[:,mask]
    j2c = lib.dot(v1/w[mask], v1.conj().T)
    def j2c_solver(v): # noqa: F811
        if v.ndim == 2:
            return lib.dot(j2c, v)
        else:
            return j2c.dot(v)
    return j2c_solver

def _cho_solve_rhojk (mf_grad, mol, auxmol, orbol, orbor,
                      decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):
    ''' Solve

    (P|Q) dQ = (P|uv) D_uv
    (P|Q) dQiu = (P|uv) C_vi n_i

    for dQ ('rhoj') and dQui ('rhok'), where D_uv = C_ui n_i C_vi* is
    a density matrix.

    Args:
        mf_grad : instance of :class:`Gradients`
        mol : instance of :class:`gto.Mole`
        auxmol : instance of :class:`gto.Mole`
        orbol : list of length nset of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrices. See
            docstring for _decompose_1rdm.
        orbor : list of length nset of ndarrays of shape (nao,*)
            Contains orbol multiplied by eigenvalues. See docstring for
            _decompose_1rdm.
        decompose_j2c: string
            The method to decompose the metric defined by int2c. It can be set
            to CD (cholesky decomposition) or ED (eigenvalue decomposition).
        lindep : float
            The threshold to discard linearly dependent basis when decompose_j2c
            is set to ED.

    Returns:
        rhoj : ndarray of shape (nset, naux)
            Aux-basis densities
        get_rhok : callable taking 3 integers (i,j,k), i,j<naux, k<nset
            Returns C-contiguous ndarray of shape
            (k-j,orbol[i].shape[1],nao) which contains a mixed-basis
            density matrix
    '''

    nset = len (orbol)
    nao, naux = mol.nao, auxmol.nao
    nbas, nauxbas = mol.nbas, auxmol.nbas
    ao_loc = mol.ao_loc
    nocc = [o.shape[-1] for o in orbor]

    int2c = auxmol.intor('int2c2e', aosym='s1')
    solve_j2c = _gen_metric_solver(int2c, decompose_j2c, lindep)
    int2c = None
    get_int3c_s1 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    rhoj = numpy.zeros((nset,naux))
    f_rhok = lib.H5TmpFile()
    t1 = (logger.process_clock (), logger.perf_counter ())
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = max_memory * .5e6/8 / (naux*nao)
    mol_ao_ranges = balance_partition(ao_loc, blksize)
    nsteps = len(mol_ao_ranges)
    t2 = t1
    for istep, (shl0, shl1, nd) in enumerate(mol_ao_ranges):
        int3c = get_int3c_s1((0, nbas, shl0, shl1, 0, nauxbas))
        t2 = logger.timer_debug1 (mf_grad, 'df grad intor (P|mn)', *t2)
        p0, p1 = ao_loc[shl0], ao_loc[shl1]
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> multithread efficiency
            v = lib.dot(int3c.reshape (nao, -1, order='F').T, orbor[i]).reshape (naux, (p1-p0)*nocc[i])
            t2 = logger.timer_debug1 (mf_grad, 'df grad einsum (P|mn) u_ni N_i = v_Pmi', *t2)
            rhoj[i] += numpy.dot (v, orbol[i][p0:p1].ravel ())
            t2 = logger.timer_debug1 (mf_grad, 'df grad einsum v_Pmi u_mi = rho_P', *t2)
            v = solve_j2c(v)
            t2 = logger.timer_debug1 (mf_grad, 'df grad cho_solve (P|Q) D_Qmi = v_Pmi', *t2)
            f_rhok['%s/%s'%(i,istep)] = v.reshape(naux,p1-p0,-1)
            t2 = logger.timer_debug1 (mf_grad, 'df grad cache D_Pmi (m <-> i transpose upon retrieval)', *t2)
        int3c = v = None
    rhoj = solve_j2c(rhoj.T).T
    int2c = None
    t1 = logger.timer_debug1 (mf_grad, 'df grad vj and vk AO (P|Q) D_Q = (P|mn) D_mn solve', *t1)
    class get_rhok_class :
        def __init__(self, my_f):
            self.f_rhok = my_f
        def __call__(self, set_id, p0, p1):
            buf = numpy.empty((p1-p0,nocc[set_id],nao))
            col1 = 0
            for istep in range(nsteps):
                dat = self.f_rhok['%s/%s'%(set_id,istep)][p0:p1]
                col0, col1 = col1, col1 + dat.shape[1]
                buf[:p1-p0,:,col0:col1] = dat.transpose(0,2,1)
            return buf
    get_rhok = get_rhok_class (f_rhok)
    return rhoj, get_rhok


class Gradients(rhf_grad.Gradients):
    '''Restricted density-fitting Hartree-Fock gradients'''

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        rhf_grad.Gradients.__init__(self, mf)

    # Whether to include the response of DF auxiliary basis when computing
    # nuclear gradients of J/K matrices
    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df.df_jk._DFHF)

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if omega is None:
            return get_jk(self, mol, dm, hermi, with_j, with_k)

        with self.base.with_df.range_coulomb(omega):
            return get_jk(self, mol, dm, hermi, with_j, with_k)

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        if omega is None:
            return get_j(self, mol, dm, hermi)

        with self.base.with_df.range_coulomb(omega):
            return get_j(self, mol, dm, hermi)

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        return self.get_jk(mol, dm, with_j=False, omega=omega)[1]

    def get_veff(self, mol=None, dm=None):
        vj, vk = self.get_jk(mol, dm)
        vhf = vj - vk*.5
        if self.auxbasis_response:
            e1_aux = (vj.aux - vk.aux*.5).sum ((0,1))
            logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            vhf = lib.tag_array(vhf, aux=e1_aux)
        return vhf

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['vhf'].aux[atom_id]
        else:
            return 0

Grad = Gradients
