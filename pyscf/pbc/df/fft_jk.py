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
JK with discrete Fourier transformation
'''

import time
import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh

    ni = mydf._numint
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, mesh=mesh)
    ngrids = len(coulG)

    if hermi == 1 or gamma_point(kpts):
        vR = rhoR = np.zeros((nset,ngrids))
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for i in range(nset):
                rhoR[i,p0:p1] += make_rho(i, ao_ks, mask, 'LDA')
            ao = ao_ks = None

        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh).real

    else:  # vR may be complex if the underlying density is complex
        vR = rhoR = np.zeros((nset,ngrids), dtype=np.complex128)
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for i in range(nset):
                for k, ao in enumerate(ao_ks):
                    ao_dm = lib.dot(ao, dms[i,k])
                    rhoR[i,p0:p1] += np.einsum('xi,xi->x', ao_dm, ao.conj())
        rhoR *= 1./nkpts

        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    weight = cell.vol / ngrids
    vR *= weight
    if gamma_point(kpts_band):
        vj_kpts = np.zeros((nset,nband,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_band):
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        for i in range(nset):
            # ni.eval_mat can handle real vR only
            # vj_kpts[i] += ni.eval_mat(cell, ao_ks, 1., None, vR[i,p0:p1], mask, 'LDA')
            for k, ao in enumerate(ao_ks):
                aow = np.einsum('xi,x->xi', ao, vR[i,p0:p1])
                vj_kpts[i,k] += lib.dot(ao.conj().T, aow)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_j_e1_kpts(mydf, dm_kpts, kpts=np.zeros((1,3)), kpts_band=None):
    '''Derivatives of Coulomb (J) AO matrix at sampled k-points.
    '''

    cell = mydf.cell
    mesh = mydf.mesh

    ni = mydf._numint
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi=1)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, mesh=mesh)
    ngrids = len(coulG)

    if gamma_point(kpts):
        vR = rhoR = np.zeros((nset,ngrids))
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for i in range(nset):
                rhoR[i,p0:p1] += make_rho(i, ao_ks, mask, 'LDA')
            ao = ao_ks = None

        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh).real

    else:  # vR may be complex if the underlying density is complex
        vR = rhoR = np.zeros((nset,ngrids), dtype=np.complex128)
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for i in range(nset):
                for k, ao in enumerate(ao_ks):
                    ao_dm = lib.dot(ao, dms[i,k])
                    rhoR[i,p0:p1] += np.einsum('xi,xi->x', ao_dm, ao.conj())
        rhoR *= 1./nkpts

        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    weight = cell.vol / ngrids
    vR *= weight
    if gamma_point(kpts_band):
        vj_kpts = np.zeros((3,nset,nband,nao,nao))
    else:
        vj_kpts = np.zeros((3,nset,nband,nao,nao), dtype=np.complex128)
    rho = None
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_band, deriv=1):
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        for i in range(nset):
            # ni.eval_mat can handle real vR only
            # vj_kpts[i] += ni.eval_mat(cell, ao_ks, 1., None, vR[i,p0:p1], mask, 'LDA')
            for k, ao in enumerate(ao_ks):
                aow = np.einsum('xi,x->xi', ao[0], vR[i,p0:p1])
                vj_kpts[:,i,k] -= lib.einsum('axi,xj->aij', ao[1:].conj(), aow)

    vj_kpts = np.asarray([_format_jks(vj, dm_kpts, input_band, kpts) for vj in vj_kpts])

    return vj_kpts

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    kpts = np.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords
    ao2_kpts = [np.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [np.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]
    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                      max_memory, blksize)
    #ao1_dtype = np.result_type(*ao1_kpts)
    #ao2_dtype = np.result_type(*ao2_kpts)
    vR_dm = np.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    t1 = (time.clock(), time.time())
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            mydf.exxdiv = exxdiv
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = np.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)
        t1 = lib.logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

def get_k_e1_kpts(mydf, dm_kpts, kpts=np.zeros((1,3)), kpts_band=None,
                  exxdiv=None):
    '''Derivatives of exchange (K) AO matrix at sampled k-points.
    '''

    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    kpts = np.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((3,nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((3,nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords

    if input_band is None:
        ao2_kpts = [np.asarray(ao.transpose(0,2,1), order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts, deriv=1)]
        ao1_kpts = ao2_kpts
        ao2_kpts = [ao2_kpt[0] for ao2_kpt in ao2_kpts]
    else:
        ao2_kpts = [np.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
        ao1_kpts = [np.asarray(ao.transpose(0,2,1), order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band, deriv=1)]

    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/3/ngrids/nao)))
    lib.logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                      max_memory, blksize)
    ao1_dtype = np.result_type(*ao1_kpts)
    ao2_dtype = np.result_type(*ao2_kpts)
    vR_dm = np.empty((3,nset,nao,ngrids), dtype=vk_kpts.dtype)

    t1 = (time.clock(), time.time())
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            mydf.exxdiv = exxdiv
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = np.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = np.einsum('aig,jg->aijg', ao1T[1:,p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(3,p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('aijg,jg->aig', vR, ao_dms[i], out=vR_dm[:,i,p0:p1])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[:,i,k1] -= weight * np.einsum('aig,jg->aij', vR_dm[:,i], ao1T[0])
        t1 = lib.logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    # Ewald correction has no contribution to nuclear gradient unless range separted Coulomb is used
    # The gradient correction part is not added in the vk matrix
    if exxdiv == 'ewald' and cell.omega!=0:
        raise NotImplementedError("Range Separated Coulomb")
        # when cell.omega !=0: madelung constant will have a non-zero derivative
    vk_kpts = np.asarray([_format_jks(vk, dm_kpts, input_band, kpts) for vk in vk_kpts])
    return vk_kpts

def get_jk(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None,
           with_j=True, with_k=True, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = np.asarray(dm, order='C')
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi, kpt, kpts_band)
    if with_k:
        vk = get_k(mydf, dm, hermi, kpt, kpts_band, exxdiv)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = np.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band)
    if kpts_band is None:
        vj = vj[:,0,:,:]
    if dm.ndim == 2:
        vj = vj[0]
    return vj


def get_k(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = np.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band, exxdiv)
    if kpts_band is None:
        vk = vk[:,0,:,:]
    if dm.ndim == 2:
        vk = vk[0]
    return vk
