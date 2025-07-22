#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
JK with analytic Fourier transformation
'''


import ctypes
import numpy
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.lib import zdotNN, zdotCN, zdotNC
from pyscf.pbc import gto as pbcgto
from pyscf.gto.ft_ao import ft_aopair
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.df_jk import (_format_dms, _format_kpts_band, _format_jks,
                                _ewald_exxdiv_for_G0)
from pyscf.pbc.lib.kpts_helper import (is_zero, group_by_conj_pairs,
                                       kk_adapted_iter)
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.incore import libpbc

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    if kpts_band is not None:
        return get_j_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    vj_kpts = numpy.zeros((n_dm,nkpts,nao,nao), dtype=numpy.complex128)
    kpt_allow = numpy.zeros(3)
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(kpt_allow, False, mesh)
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
    weight = 1./len(kpts)
    for Gpq, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, max_memory=max_memory,
                                    return_complex=False):
        _update_vj_(vj_kpts, Gpq, dms, coulG[p0:p1], weight)
        Gpq = None

    if is_zero(kpts):
        vj_kpts = vj_kpts.real.copy()
    return _format_jks(vj_kpts, dm_kpts, kpts_band, kpts)

def _update_vj_(vj_kpts, Gpq, dms, coulG, weight=None):
    r'''Compute the Coulomb matrix
    J_{kl} = \sum_{ij} \sum_G 4\pi/G^2 * FT(\rho_{ij}) IFT(\rho_{kl}) dm_{ji}
    for analytical FT tensor FT(\rho_{ij})
    '''
    # FT(\rho_{ij, kpt}) = \int exp(-i G*r) conj(\psi_{i,kpt}) \psi_{j,kpt} dr
    # IFT(\rho_{ij, kpt}) = \int exp(i G*r) conj(\psi_{i,kpt}) \psi_{j,kpt} dr
    # IFT(\rho_{ij, kpt}) = conj(FT(\rho_{ji, kpt}))
    # rhoG[k] = einsum('ij,gji->g', dms[i,k], IFT(\rho))
    #         = einsum('ij,gji->g', dms[i,k], aoao.conj().transpose(0,2,1))
    #         = einsum('ij,gij->g', dms[i,k].conj(), aoao).conj()
    # vG = sum_k rhoG[k] * coulG
    # vj[k] = einsum('g,gji->g', vG, aoao[k])
    is_real = vj_kpts.dtype == np.double
    GpqR, GpqI = Gpq
    rhoR = np.einsum('nkij,kgij->ng', dms.real, GpqR)
    rhoI = -np.einsum('nkij,kgij->ng', dms.real, GpqI)
    if not is_real:
        rhoR += np.einsum('nkij,kgij->ng', dms.imag, GpqI)
        rhoI += np.einsum('nkij,kgij->ng', dms.imag, GpqR)

    if weight is not None:
        coulG = coulG * weight
    vR = coulG * rhoR
    vI = coulG * rhoI

    vj_kpts.real += np.einsum('ng,kgij->nkij', vR, GpqR)
    vj_kpts.real -= np.einsum('ng,kgij->nkij', vI, GpqI)
    if not is_real:
        vj_kpts.imag += np.einsum('ng,kgij->nkij', vR, GpqI)
        vj_kpts.imag += np.einsum('ng,kgij->nkij', vI, GpqR)
    return vj_kpts

def get_j_for_bands(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpt_allow = numpy.zeros(3)
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(kpt_allow, False, mesh)
    ngrids = len(coulG)
    rhoG = numpy.zeros((nset,ngrids), dtype=numpy.complex128)
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8

    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, max_memory=max_memory):
        for k, aoao in enumerate(aoaoks):
            rhoG[:,p0:p1] += numpy.einsum('nij,Lij->nL', dms[:,k].conj(),
                                          aoao.reshape(-1,nao,nao)).conj()
    aoao = aoaoks = p0 = p1 = None
    weight = 1./len(kpts)
    vG = rhoG * coulG * weight
    t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vj_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts_band,
                                        max_memory=max_memory):
        for k, aoao in enumerate(aoaoks):
            vj_kpts[:,k] += numpy.einsum('nL,Lij->nij', vG[:,p0:p1],
                                         aoao.reshape(-1,nao,nao))
    aoao = aoaoks = p0 = p1 = None

    if is_zero(kpts_band):
        vj_kpts = vj_kpts.real.copy()
    t1 = log.timer_debug1('get_j pass 2', *t1)
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    if kpts_band is not None:
        return get_k_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)

    cpu0 = cpu1 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mydf)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
    mo_occ = getattr(dm_kpts, 'mo_occ', None)
    dm_kpts = np.asarray(dm_kpts)

    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    vkR = np.zeros((n_dm,nkpts,nao,nao))
    vkI = np.zeros((n_dm,nkpts,nao,nao))
    vk = [vkR, vkI]
    weight = 1. / nkpts

    aosym = 's1'
    bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    rcut = ft_ao.estimate_rcut(cell)
    supmol = ft_ao.ExtendedMole.from_cell(cell, bvk_kmesh, rcut.max())
    supmol = supmol.strip_basis(rcut)

    t_rev_pairs = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    try:
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    except TypeError:
        t_rev_pairs = [[k, k] if k_conj is None else [k, k_conj]
                       for k, k_conj in t_rev_pairs]
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    log.debug1('Num time-reversal pairs %d', len(t_rev_pairs))

    time_reversal_symmetry = mydf.time_reversal_symmetry
    if time_reversal_symmetry:
        for k, k_conj in t_rev_pairs:
            if k != k_conj and abs(dms[:,k_conj] - dms[:,k].conj()).max() > 1e-6:
                time_reversal_symmetry = False
                log.debug2('Disable time_reversal_symmetry')
                break

    if time_reversal_symmetry:
        k_to_compute = np.zeros(nkpts, dtype=np.int8)
        k_to_compute[t_rev_pairs[:,0]] = 1
    else:
        k_to_compute = np.ones(nkpts, dtype=np.int8)

    contract_mo_early = False
    if mo_coeff is None:
        dmsR = np.asarray(dms.real, order='C')
        dmsI = np.asarray(dms.imag, order='C')
        dm = [dmsR, dmsI]
        dm_factor = None

        if np.count_nonzero(k_to_compute) >= 2 * lib.num_threads():
            update_vk = _update_vk1_
        else:
            update_vk = _update_vk_
        log.debug2('set update_vk to %s', update_vk)
    else:
        # dm ~= dm_factor * dm_factor.T
        n_dm, nkpts, nao = dms.shape[:3]
        # mo_coeff, mo_occ are not a list of aligned array if
        # remove_lin_dep was applied to scf object
        if dm_kpts.ndim == 4:  # KUHF
            nocc = max(max(np.count_nonzero(x > 0) for x in z) for z in mo_occ)
            dm_factor = [[x[:,:nocc] for x in mo] for mo in mo_coeff]
            occs = [[x[:nocc] for x in z] for z in mo_occ]
        else:  # KRHF
            nocc = max(np.count_nonzero(x > 0) for x in mo_occ)
            dm_factor = [[mo[:,:nocc] for mo in mo_coeff]]
            occs = [[x[:nocc] for x in mo_occ]]
        dm_factor = np.array(dm_factor, dtype=np.complex128, order='C')
        dm_factor *= np.sqrt(np.array(occs, dtype=np.double))[:,:,None]

        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        s_nao = supmol.nao
        contract_mo_early = (time_reversal_symmetry and
                             bvk_ncells*nao*4 > s_nao*nocc*n_dm)
        log.debug2('time_reversal_symmetry = %s bvk_ncells = %d '
                   's_nao = %d nocc = %d n_dm = %d',
                   time_reversal_symmetry, bvk_ncells, s_nao, nocc, n_dm)
        log.debug2('Use algorithm contract_mo_early = %s', contract_mo_early)
        if contract_mo_early:
            s_nao = supmol.nao
            moR, moI = _mo_k2gamma(supmol, dm_factor, kpts, t_rev_pairs)
            if abs(moI).max() < 1e-5:
                dm = [moR, None]
                dm_factor = moR = moI = None
                ft_kern = _gen_ft_kernel_fake_gamma(cell, supmol, aosym)
                update_vk = _update_vk_fake_gamma
            else:
                moR = moI = None
                contract_mo_early = False

        if not contract_mo_early:
            dm = [np.asarray(dm_factor.real, order='C'),
                  np.asarray(dm_factor.imag, order='C')]
            dm_factor = None
            if np.count_nonzero(k_to_compute) >= 2 * lib.num_threads():
                update_vk = _update_vk1_dmf
            else:
                update_vk = _update_vk_dmf
        log.debug2('set update_vk to %s with dm_factor', update_vk)

    if not contract_mo_early:
        ft_kern = supmol.gen_ft_kernel(aosym, return_complex=False,
                                       kpts=kpts, verbose=log)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    Gv = np.asarray(Gv, order='F')
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))
    log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)

    if contract_mo_early:
        Gblksize = max(24, int((max_memory*1e6/16-nkpts*nao**2*3)/
                               (nao*s_nao+nao*nkpts*nocc))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        log.debug1('Gblksize = %d', Gblksize)
        buf = np.empty(Gblksize*s_nao*nao*2)
    else:
        Gblksize = max(24, int(max_memory*1e6/16/nao**2/(nkpts+3))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        log.debug1('Gblksize = %d', Gblksize)
        buf = np.empty(nkpts*Gblksize*nao**2*2)

    for group_id, (kpt, ki_idx, kj_idx, self_conj) \
            in enumerate(kk_adapted_iter(cell, kpts)):
        vkcoulG = mydf.weighted_coulG(kpt, exxdiv, mesh)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            log.debug3('update_vk [%s:%s]', p0, p1)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, out=buf)
            update_vk(vk, Gpq, dm, vkcoulG[p0:p1] * weight, ki_idx, kj_idx,
                      not self_conj, k_to_compute, t_rev_pairs)
            Gpq = None
        cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)

    if is_zero(kpts) and not numpy.iscomplexobj(dm_kpts):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j

    # Add ewald_exxdiv contribution because G=0 was not included in the
    # non-uniform grids
    if exxdiv == 'ewald' and cell.low_dim_ft_type == 'inf_vacuum':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts)

    if time_reversal_symmetry:
        for k, k_conj in t_rev_pairs:
            if k != k_conj:
                vk_kpts[:,k_conj] = vk_kpts[:,k].conj()
    log.timer_debug1('get_k_kpts', *cpu0)
    return vk_kpts.reshape(dm_kpts.shape)

def get_k_for_bands(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
                    exxdiv=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    mesh = mydf.mesh
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    swap_2e = (kpts_band is None)
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    kk_todo = numpy.ones(kk_table.shape[:2], dtype=bool)
    k_to_compute = np.ones(nkpts+len(kpts_band), dtype=np.int8)
    vkR = numpy.zeros((nset,nband,nao,nao))
    vkI = numpy.zeros((nset,nband,nao,nao))
    dmsR = numpy.asarray(dms.real, order='C')
    dmsI = numpy.asarray(dms.imag, order='C')

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now)) * .8
    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    def make_kpt(kpt):  # kpt = kptj - kpti
        # search for all possible ki and kj that has ki-kj+kpt=0
        kk_match = numpy.einsum('ijx->ij', abs(kk_table + kpt)) < 1e-9
        kpti_idx, kptj_idx = numpy.where(kk_todo & kk_match)
        nkptj = len(kptj_idx)
        log.debug1('kpt = %s', kpt)
        log.debug2('kpti_idx = %s', kpti_idx)
        log.debug2('kptj_idx = %s', kptj_idx)
        kk_todo[kpti_idx,kptj_idx] = False
        if swap_2e and not is_zero(kpt):
            kk_todo[kptj_idx,kpti_idx] = False

        max_memory1 = max_memory * (nkptj+1)/(nkptj+5)
        #blksize = max(int(max_memory1*4e6/(nkptj+5)/16/nao**2), 16)

        #bufR = numpy.empty((blksize*nao**2))
        #bufI = numpy.empty((blksize*nao**2))
        # Use DF object to mimic KRHF/KUHF object in function get_coulG
        vkcoulG = mydf.weighted_coulG(kpt, exxdiv, mesh)
        weight = 1./len(kpts)
        perm_sym = swap_2e and not is_zero(kpt)
        for Gpq, p0, p1 in mydf.ft_loop(mesh, kpt, kpts, max_memory=max_memory1,
                                        return_complex=False):
            _update_vk_((vkR, vkI), Gpq, (dmsR, dmsI), vkcoulG[p0:p1]*weight,
                        kpti_idx, kptj_idx, perm_sym, k_to_compute, None)

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)
        t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%ki, *t1)

    if (is_zero(kpts) and is_zero(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j

    # Add ewald_exxdiv contribution because G=0 was not included in the
    # non-uniform grids
    if exxdiv == 'ewald' and cell.low_dim_ft_type == 'inf_vacuum':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

def _update_vk_(vk, Gpq, dms, wcoulG, kpti_idx, kptj_idx, swap_2e,
                k_to_compute, t_rev_pairs):
    '''
    contraction for exchange matrices:

    vk += np.einsum('ngij,njk,nglk,g->nil', Gpq, dm, Gpq.conj(), coulG)
    vk += np.einsum('ngij,nli,nglk,g->nkj', Gpq, dm, Gpq.conj(), coulG)
    '''
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmsR, dmsI = dms
    nG = len(wcoulG)
    n_dm, nkpts, nao = vkR.shape[:3]
    bufR = np.empty((nG*nao**2))
    bufI = np.empty((nG*nao**2))
    buf1R = np.empty((nG*nao**2))
    buf1I = np.empty((nG*nao**2))
    iLkR = np.ndarray((nao,nG,nao), buffer=buf1R)
    iLkI = np.ndarray((nao,nG,nao), buffer=buf1I)

    for k, (ki, kj) in enumerate(zip(kpti_idx, kptj_idx)):
        # case 1: k_pq = (pi|iq)
        #:v4 = np.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,jk->il', v4, dm)
        pLqR = np.ndarray((nao,nG,nao), buffer=bufR)
        pLqI = np.ndarray((nao,nG,nao), buffer=bufI)
        pLqR[:] = GpqR[kj].transpose(1,0,2)
        pLqI[:] = GpqI[kj].transpose(1,0,2)
        if k_to_compute[ki]:
            for i in range(n_dm):
                zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                       dmsR[i,kj], dmsI[i,kj], 1,
                       iLkR.reshape(-1,nao), iLkI.reshape(-1,nao))
                iLkR *= wcoulG.reshape(1,nG,1)
                iLkI *= wcoulG.reshape(1,nG,1)
                zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                       pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                       1, vkR[i,ki], vkI[i,ki], 1)

        # case 2: k_pq = (iq|pi)
        #:v4 = np.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,li->kj', v4, dm)
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        if swap_2e and k_to_compute[kj]:
            for i in range(n_dm):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1,
                       iLkR.reshape(nao,-1), iLkI.reshape(nao,-1))
                iLkR *= wcoulG.reshape(1,nG,1)
                iLkI *= wcoulG.reshape(1,nG,1)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                       1, vkR[i,kj], vkI[i,kj], 1)

def _update_vk1_(vk, Gpq, dms, wcoulG, kpti_idx, kptj_idx, swap_2e,
                 k_to_compute, t_rev_pairs):
    '''
    contraction for exchange matrices:

    vk += np.einsum('ngij,njk,nglk,g->nil', Gpq, dm, Gpq.conj(), coulG)
    vk += np.einsum('ngij,nli,nglk,g->nkj', Gpq, dm, Gpq.conj(), coulG)
    '''
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmsR, dmsI = dms
    nG = len(wcoulG)
    n_dm, nkpts, nao = vkR.shape[:3]
    nkptj = len(kptj_idx)

    assert GpqR.transpose(0,2,3,1).flags.c_contiguous
    assert vkR.flags.c_contiguous
    assert dmsR.flags.c_contiguous
    assert kpti_idx.dtype == np.int32
    assert kptj_idx.dtype == np.int32
    assert k_to_compute.dtype == np.int8
    assert k_to_compute.size == nkpts

    libpbc.PBC_kcontract(
        vkR.ctypes.data_as(ctypes.c_void_p),
        vkI.ctypes.data_as(ctypes.c_void_p),
        dmsR.ctypes.data_as(ctypes.c_void_p),
        dmsI.ctypes.data_as(ctypes.c_void_p),
        GpqR.ctypes.data_as(ctypes.c_void_p),
        GpqI.ctypes.data_as(ctypes.c_void_p),
        wcoulG.ctypes.data_as(ctypes.c_void_p),
        kpti_idx.ctypes.data_as(ctypes.c_void_p),
        kptj_idx.ctypes.data_as(ctypes.c_void_p),
        k_to_compute.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(swap_2e), ctypes.c_int(n_dm), ctypes.c_int(nao),
        ctypes.c_int(nG), ctypes.c_int(nkpts), ctypes.c_int(nkptj))

def _update_vk_dmf(vk, Gpq, dmf, wcoulG, kpti_idx, kptj_idx, swap_2e,
                   k_to_compute, t_rev_pairs):
    '''
    dmf is the factorized dm, dm = dmf * dmf.conj().T
    Computing exchange matrices with dmf:
    vk += np.einsum('ngij,njp,nkp,nglk,g->nil', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    vk += np.einsum('ngij,nlp,nip,nglk,g->nkj', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    '''
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmfR, dmfI = dmf
    nG = len(wcoulG)
    n_dm, nkpts, nao, nocc = dmfR.shape
    bufR = np.empty((nG*nao**2))
    bufI = np.empty((nG*nao**2))
    bufR1 = np.empty((nG*nao*nocc))
    bufI1 = np.empty((nG*nao*nocc))

    for k, (ki, kj) in enumerate(zip(kpti_idx, kptj_idx)):
        # case 1: k_pq = (pi|iq)
        #:v4 = np.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,jk->il', v4, dm)
        pLqR = np.ndarray((nao,nG,nao), buffer=bufR)
        pLqI = np.ndarray((nao,nG,nao), buffer=bufI)
        pLqR[:] = GpqR[kj].transpose(1,0,2)
        pLqI[:] = GpqI[kj].transpose(1,0,2)
        if k_to_compute[ki]:
            iLkR = np.ndarray((nao,nG,nocc), buffer=bufR1)
            iLkI = np.ndarray((nao,nG,nocc), buffer=bufI1)
            for i in range(n_dm):
                zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                       dmfR[i,kj], dmfI[i,kj], 1,
                       iLkR.reshape(-1,nocc), iLkI.reshape(-1,nocc))
                iLkR1 = iLkR * wcoulG[:,None]
                iLkI1 = iLkI * wcoulG[:,None]
                zdotNC(iLkR1.reshape(nao,-1), iLkI1.reshape(nao,-1),
                       iLkR.reshape(nao,-1).T, iLkI.reshape(nao,-1).T,
                       1, vkR[i,ki], vkI[i,ki], 1)

        # case 2: k_pq = (iq|pi)
        #:v4 = np.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,li->kj', v4, dm)
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        if swap_2e and k_to_compute[kj]:
            iLkR = np.ndarray((nocc,nG,nao), buffer=bufR1)
            iLkI = np.ndarray((nocc,nG,nao), buffer=bufI1)
            for i in range(n_dm):
                zdotCN(dmfR[i,ki].T, dmfI[i,ki].T, pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1,
                       iLkR.reshape(nocc,-1), iLkI.reshape(nocc,-1))
                iLkR1 = iLkR * wcoulG[:,None]
                iLkI1 = iLkI * wcoulG[:,None]
                zdotCN(iLkR.reshape(-1,nao).T, iLkI.reshape(-1,nao).T,
                       iLkR1.reshape(-1,nao), iLkI1.reshape(-1,nao),
                       1, vkR[i,kj], vkI[i,kj], 1)

def _update_vk1_dmf(vk, Gpq, dmf, wcoulG, kpti_idx, kptj_idx, swap_2e,
                    k_to_compute, t_rev_pairs):
    '''
    dmf is the factorized dm, dm = dmf * dmf.conj().T
    Computing exchange matrices with dmf:
    vk += np.einsum('ngij,njp,nkp,nglk,g->nil', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    vk += np.einsum('ngij,nlp,nip,nglk,g->nkj', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    '''
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmfR, dmfI = dmf
    nG = len(wcoulG)
    n_dm, nkpts, nao, nocc = dmfR.shape
    nkptj = len(kptj_idx)

    assert GpqR.transpose(0,2,3,1).flags.c_contiguous
    assert vkR.flags.c_contiguous
    assert dmfI.flags.c_contiguous
    assert kpti_idx.dtype == np.int32
    assert kptj_idx.dtype == np.int32
    assert k_to_compute.dtype == np.int8
    assert k_to_compute.size == nkpts

    libpbc.PBC_kcontract_dmf(
        vkR.ctypes.data_as(ctypes.c_void_p),
        vkI.ctypes.data_as(ctypes.c_void_p),
        dmfR.ctypes.data_as(ctypes.c_void_p),
        dmfI.ctypes.data_as(ctypes.c_void_p),
        GpqR.ctypes.data_as(ctypes.c_void_p),
        GpqI.ctypes.data_as(ctypes.c_void_p),
        wcoulG.ctypes.data_as(ctypes.c_void_p),
        kpti_idx.ctypes.data_as(ctypes.c_void_p),
        kptj_idx.ctypes.data_as(ctypes.c_void_p),
        k_to_compute.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(swap_2e),
        ctypes.c_int(n_dm), ctypes.c_int(nao), ctypes.c_int(nocc),
        ctypes.c_int(nG), ctypes.c_int(nkpts), ctypes.c_int(nkptj))

def _update_vk_fake_gamma(vk, Gpq, dmf, wcoulG, kpti_idx, kptj_idx, swap_2e,
                          k_to_compute, t_rev_pairs):
    '''
    dmf is the factorized dm, dm = dmf * dmf.conj().T
    Computing exchange matrices with dmf:
    vk += np.einsum('ngij,njp,nkp,nglk,g->nil', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    vk += np.einsum('ngij,nlp,nip,nglk,g->nkj', Gpq, dmf, dmf.conj(), Gpq.conj(), coulG)
    '''
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmfR, dmfI = dmf
    nG = len(wcoulG)
    n_dm, s_nao, nkpts, nocc = dmfR.shape
    nao = vkR.shape[-1]

    GpqR = GpqR.transpose(2,1,0)
    assert GpqR.flags.c_contiguous
    assert kpti_idx.dtype == np.int32
    assert kptj_idx.dtype == np.int32
    assert k_to_compute.dtype == np.int8
    assert k_to_compute.size == nkpts
    assert kptj_idx.size == nkpts

    for i in range(n_dm):
        libpbc.PBC_kcontract_fake_gamma(
            vkR[i].ctypes.data_as(ctypes.c_void_p),
            vkI[i].ctypes.data_as(ctypes.c_void_p),
            dmfR[i].ctypes.data_as(ctypes.c_void_p),
            GpqR.ctypes.data_as(ctypes.c_void_p),
            GpqI.ctypes.data_as(ctypes.c_void_p),
            wcoulG.ctypes.data_as(ctypes.c_void_p),
            kpti_idx.ctypes.data_as(ctypes.c_void_p),
            kptj_idx.ctypes.data_as(ctypes.c_void_p),
            k_to_compute.ctypes.data_as(ctypes.c_void_p),
            t_rev_pairs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(t_rev_pairs.shape[0]),
            ctypes.c_int(swap_2e), ctypes.c_int(s_nao), ctypes.c_int(nao),
            ctypes.c_int(nocc), ctypes.c_int(nG), ctypes.c_int(nkpts))

def _mo_k2gamma(supmol, dm_factor, kpts, t_rev_pairs):
    '''Transform to MO of supcell at gamma point'''
    sh_loc = supmol.sh_loc  # maps the bvk shell Id to supmol shell Id
    ao_loc = supmol.ao_loc
    rs_cell = supmol.rs_cell
    cell0_ao_loc = rs_cell.ao_loc
    nbasp = rs_cell.ref_cell.nbas
    bvk_ncells, _, nimgs = supmol.bas_mask.shape
    expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts.T))
    expLk = np.asarray(expLk, order='C')
    s_nao = supmol.nao
    n_dm, nkpts, nao, nocc = dm_factor.shape
    moR = np.empty((n_dm,s_nao,nkpts,nocc))
    moI = np.empty((n_dm,s_nao,nkpts,nocc))
    t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    for i_dm in range(n_dm):
        libpbc.PBCmo_k2gamma(
            moR[i_dm].ctypes.data_as(ctypes.c_void_p),
            moI[i_dm].ctypes.data_as(ctypes.c_void_p),
            dm_factor[i_dm].ctypes.data_as(ctypes.c_void_p),
            expLk.ctypes.data_as(ctypes.c_void_p),
            sh_loc.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            t_rev_pairs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(t_rev_pairs.shape[0]),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nbasp),
            ctypes.c_int(s_nao), ctypes.c_int(nao), ctypes.c_int(nocc),
            ctypes.c_int(nkpts), ctypes.c_int(nimgs))
    return moR, moI

def _gen_ft_kernel_fake_gamma(cell, supmol, aosym='s1'):
    ovlp_mask = supmol.get_ovlp_mask()
    ovlp_mask = np.asarray(ovlp_mask, dtype=np.int8, order='F')
    cell = supmol.rs_cell
    supmol1 = cell.to_mol() + supmol
    shls_slice = (0, cell.nbas, cell.nbas, supmol1.nbas)
    b = cell.reciprocal_vectors()

    def ft_kern(Gv, gxyz=None, Gvbase=None, q=None, kpts=None, out=None):
        return ft_aopair(supmol1, Gv, shls_slice, aosym, b,
                         gxyz=gxyz, Gvbase=Gvbase, out=out,
                         q=q, return_complex=False, ovlp_mask=ovlp_mask)
    return ft_kern

##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    vj = vk = None
    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(mydf, dm, hermi, kpt, kpts_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, dm, hermi, kpt, kpts_band)
        return vj, vk

    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, kpt.reshape(1, 3))
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = is_zero(kpt)
    k_real = is_zero(kpt) and not numpy.iscomplexobj(dms)

    mesh = mydf.mesh
    kptii = numpy.asarray((kpt,kpt))
    kpt_allow = numpy.zeros(3)

    if with_j:
        vjcoulG = mydf.weighted_coulG(kpt_allow, False, mesh)
        vjR = numpy.zeros((nset,nao,nao))
        vjI = numpy.zeros((nset,nao,nao))
    if with_k:
        vkcoulG = mydf.weighted_coulG(kpt_allow, exxdiv, mesh)
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
    dmsR = numpy.asarray(dms.real.reshape(nset,nao,nao), order='C')
    dmsI = numpy.asarray(dms.imag.reshape(nset,nao,nao), order='C')
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now)) * .8
    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    t2 = t1

    # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
    #                 == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
    #blksize = max(int(max_memory*.25e6/16/nao**2), 16)
    pLqR = pLqI = None
    for pqkR, pqkI, p0, p1 in mydf.pw_loop(mesh, kptii, max_memory=max_memory):
        t2 = log.timer_debug1('%d:%d ft_aopair'%(p0,p1), *t2)
        pqkR = pqkR.reshape(nao,nao,-1)
        pqkI = pqkI.reshape(nao,nao,-1)
        if with_j:
            #:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
            #:vj += numpy.einsum('ijkl,lk->ij', v4, dm)
            for i in range(nset):
                rhoR = numpy.einsum('pq,pqk->k', dmsR[i], pqkR)
                rhoR+= numpy.einsum('pq,pqk->k', dmsI[i], pqkI)
                rhoI = numpy.einsum('pq,pqk->k', dmsI[i], pqkR)
                rhoI-= numpy.einsum('pq,pqk->k', dmsR[i], pqkI)
                rhoR *= vjcoulG[p0:p1]
                rhoI *= vjcoulG[p0:p1]
                vjR[i] += numpy.einsum('pqk,k->pq', pqkR, rhoR)
                vjR[i] -= numpy.einsum('pqk,k->pq', pqkI, rhoI)
                if not j_real:
                    vjI[i] += numpy.einsum('pqk,k->pq', pqkR, rhoI)
                    vjI[i] += numpy.einsum('pqk,k->pq', pqkI, rhoR)
        #t2 = log.timer_debug1('        with_j', *t2)

        if with_k:
            #:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
            #:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            pLqR = lib.transpose(pqkR, axes=(0,2,1), out=pLqR).reshape(-1,nao)
            pLqI = lib.transpose(pqkI, axes=(0,2,1), out=pLqI).reshape(-1,nao)
            nG = p1 - p0
            iLkR = numpy.ndarray((nao,nG,nao), buffer=pqkR)
            iLkI = numpy.ndarray((nao,nG,nao), buffer=pqkI)
            for i in range(nset):
                if k_real:
                    lib.dot(pLqR, dmsR[i], 1, iLkR.reshape(nao*nG,nao))
                    lib.dot(pLqI, dmsR[i], 1, iLkI.reshape(nao*nG,nao))
                    iLkR *= vkcoulG[p0:p1].reshape(1,nG,1)
                    iLkI *= vkcoulG[p0:p1].reshape(1,nG,1)
                    lib.dot(iLkR.reshape(nao,-1), pLqR.reshape(nao,-1).T, 1, vkR[i], 1)
                    lib.dot(iLkI.reshape(nao,-1), pLqI.reshape(nao,-1).T, 1, vkR[i], 1)
                else:
                    zdotNN(pLqR, pLqI, dmsR[i], dmsI[i], 1,
                           iLkR.reshape(-1,nao), iLkI.reshape(-1,nao))
                    iLkR *= vkcoulG[p0:p1].reshape(1,nG,1)
                    iLkI *= vkcoulG[p0:p1].reshape(1,nG,1)
                    zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           1, vkR[i], vkI[i], 1)
            #t2 = log.timer_debug1('        with_k', *t2)
        pqkR = pqkI = pLqR = pLqI = iLkR = iLkI = None
        #t2 = log.timer_debug1('%d:%d'%(p0,p1), *t2)
    t1 = log.timer_debug1('aft_jk.get_jk', *t1)

    if with_j:
        if j_real:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj = vj.reshape(dm.shape)
    if with_k:
        if k_real:
            vk = vkR
        else:
            vk = vkR + vkI * 1j
        # Add ewald_exxdiv contribution because G=0 was not included in the
        # non-uniform grids
        if exxdiv == 'ewald' and cell.low_dim_ft_type == 'inf_vacuum':
            _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
        vk = vk.reshape(dm.shape)
    return vj, vk
