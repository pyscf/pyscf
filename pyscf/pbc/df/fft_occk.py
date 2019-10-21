#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Modified from pyscf/pbc/df/fft_jk.py
#
# ref: Lin, L. J. Chem. Theory Comput. 2016, 12, 2242\u22122249.
#

'''
JK with occ discrete Fourier transformation
'''

import time
import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

import scipy.linalg

#@profile
def get_k_kpts_occ(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
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
    Jtime=time.time()

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

######################## 
    if mo_coeff is None:
            return _format_jks(vk_kpts, dm_kpts, input_band, kpts)
    elif nset > 1:
        mem_now = lib.current_memory()[0]
        max_memory = mydf.max_memory - mem_now
        blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
        lib.logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                          max_memory, blksize)
        ao1_dtype = np.result_type(*ao1_kpts)
        ao2_dtype = np.result_type(*ao2_kpts)
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
################

    if mo_coeff is not None and nset == 1:
        # occ
        mo_coeff0 = [mo_coeff[k][:,occ>0] for k, occ in enumerate(mo_occ)]
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]
    # occ
    if gamma_point(kpts_band) and gamma_point(kpts):
        kiv = np.zeros((nset,nband,mo_coeff0[0].shape[1],nao), dtype=dms.dtype)
    else:
        kiv = [[]*nset]
        for i in range(nset):
            for k1 in range(nband):
                kiv[i].append(np.zeros((mo_coeff0[k1].shape[1],nao), dtype=np.complex128))

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                      max_memory, blksize)
    ao1_dtype = np.result_type(*ao1_kpts)
    ao2_dtype = np.result_type(*ao2_kpts)
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
            # occ
            ao3T = np.dot(mo_coeff0[k1].T, ao1T)
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

            m3T = ao3T.shape[0]
            for p0, p1 in lib.prange(0, m3T, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao3T[p0:p1].conj()*expmikr, ao2T)
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
                # occ
                kiv[i][k1] += weight * lib.dot(vR_dm[i,0:mo_coeff[k1].shape[1]], ao1T.T)

        t1 = lib.logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)
    for i in range(nset):
        for k1, ao1T in enumerate(ao1_kpts):
            kij = lib.einsum('ui,ju->ij', mo_coeff0[k1], kiv[i][k1])
            kr = scipy.linalg.solve(kij.conj(), kiv[i][k1])
            vk_kpts[i,k1] = lib.dot(kiv[i][k1].T.conj(),kr)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    print "Took this long for occ-X: ", time.time()-Jtime

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)









