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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

'''
JK with discrete Fourier transformation
'''
import time
import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

def get_j_kpts_ibz(mydf, dm_kpts, hermi=1, kd=KPoints(), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts_ibz, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kd : :class:`KPoints` object

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts_ibz, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh
    kpts = kd.kpts_ibz

    ni = mydf._numint
    #make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi)
    mo_coeff = None
    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ = dm_kpts.mo_occ
        if isinstance(dm_kpts[0], np.ndarray) and dm_kpts[0].ndim == 2:
            mo_coeff = [mo_coeff,]
            mo_occ = [mo_occ,]
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, mesh=mesh)
    ngrids = len(coulG)

    if hermi == 1 or gamma_point(kpts):
        vR = rhoR = np.zeros((nset, ngrids))
        for k in range(nkpts):
            rhoR_k = np.zeros((nset, ngrids))
            kpt = kpts[k]
            for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpt):
                ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
                for i in range(nset):
                    if mo_coeff is None:
                        rhoR_k[i,p0:p1] += numint.eval_rho(cell, ao_ks[0], dms[i,k], mask, xctype='LDA', hermi=hermi)
                    else:
                        rhoR_k[i,p0:p1] += numint.eval_rho2(cell, ao_ks[0], mo_coeff[i][k], mo_occ[i][k], mask, xctype='LDA')
            for i in range(nset):
                rhoR[i] += kd.symmetrize_density(rhoR_k[i], k, mesh)
        rhoR *= 1./kd.nkpts
      
        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh).real
    else:
        vR = rhoR = np.zeros((nset,ngrids), dtype=np.complex128)
        for k in range(nkpts):
            rhoR_k = np.zeros((nset,ngrids), dtype=np.complex128)
            kpt = kpts[k]
            for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpt):
                ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
                for i in range(nset):
                    ao_dm = lib.dot(ao_ks, dms[i,k])
                    rhoR_k[i,p0:p1] += np.einsum('xi,xi->x', ao_dm, ao_ks.conj())
            for i in range(nset):
                rhoR[i] += kd.symmetrize_density(rhoR_k[i], k, mesh)
        rhoR *= 1./kd.nkpts

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
            for k, ao in enumerate(ao_ks):
                aow = np.einsum('xi,x->xi', ao, vR[i,p0:p1])
                vj_kpts[i,k] += lib.dot(ao.conj().T, aow)
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts_ibz(mydf, dm_kpts, hermi=1, kd=KPoints(), kpts_band=None, exxdiv=None):
    '''Get the exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts_ibz, nao, nao) ndarray
            Density matrix at each k-point
        kd : :class:`KPoints` object

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vk : (nkpts_ibz, nao, nao) ndarray
        or list of vk if the input dm_kpts is a list of DMs
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

    kpts = kd.kpts_ibz
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    dms_bz = []
    for i in range(nset):
        dms_bz.append(kd.transform_dm(dms[i]))
    dms = np.asarray(dms_bz).reshape(nset, kd.nkpts, nao, nao)

    weight = 1./kd.nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords

    ao2_kpts = [np.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kd.kpts)]
    if input_band is None:
        idx = [kd.ibz2bz[k] for k in range(kd.nkpts_ibz)]
        ao1_kpts = [ao2_kpts[k] for k in idx] #ao1 runs over k-points in IBZ
    else:
        ao1_kpts = [np.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]
    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        mo_bz = kd.transform_mo_coeff(mo_coeff)
        ao2_kpts = [np.dot(mo_bz[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    #in the worst case, all k-points will run simultaneously
    blksize = int(min(nao, max(1, max_memory*1e6/16/4/ngrids/nao/kd.nkpts - 1)))
    lib.logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                      max_memory, blksize)

    t2 = (time.clock(), time.time())
    def _get_vk(k2, nthreads):
        t1 = (time.clock(), time.time())
        lib.num_threads(nthreads)
        ao2T = ao2_kpts[k2]
        if ao2T.size == 0:
            return 0.

        kpt2 = kd.kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        vk_kpts_loc = np.zeros_like(vk_kpts)
        vR_dm_loc = np.empty((nset,nao,ngrids), dtype=vk_kpts_loc.dtype)
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
                if vR_dm_loc.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm_loc[i,p0:p1])
                vR = None
            vR_dm_loc *= expmikr.conj()

            for i in range(nset):
                vk_kpts_loc[i,k1] += weight * lib.dot(vR_dm_loc[i], ao1T.T)
        t1 = lib.logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)
        return vk_kpts_loc

    try:
        from joblib import Parallel, delayed
        with lib.with_multiproc_nproc(kd.nkpts) as mpi:
            res = Parallel(n_jobs = mpi.nproc)(delayed(_get_vk)(k, lib.num_threads()) for k in range(kd.nkpts))
    except:
        res = [_get_vk(k, lib.num_threads()) for k in range(kd.nkpts)]
    for item in res:
        vk_kpts += item
    t2 = lib.logger.timer_debug1(mydf, 'get_k_kpts: make_kpt total', *t2)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kd.kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)
