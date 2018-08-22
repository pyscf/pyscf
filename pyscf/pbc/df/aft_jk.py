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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK with analytic Fourier transformation
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    if kpts_band is not None:
        return get_j_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    vj_kpts = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)
    kpt_allow = numpy.zeros(3)
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(kpt_allow, False, mesh)
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
    weight = 1./len(kpts)
    dmsC = dms.conj()
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, max_memory=max_memory):
        vG = [0] * nset
        #:rho = numpy.einsum('lkL,lk->L', pqk.conj(), dm)
        for k, aoao in enumerate(aoaoks):
            for i in range(nset):
                rho = numpy.einsum('ij,Lij->L', dmsC[i,k],
                                   aoao.reshape(-1,nao,nao)).conj()
                vG[i] += rho * coulG[p0:p1]
        for i in range(nset):
            vG[i] *= weight
        for k, aoao in enumerate(aoaoks):
            for i in range(nset):
                vj_kpts[i,k] += numpy.einsum('L,Lij->ij', vG[i],
                                             aoao.reshape(-1,nao,nao))
    aoao = aoaoks = p0 = p1 = None

    if gamma_point(kpts):
        vj_kpts = vj_kpts.real.copy()
    return _format_jks(vj_kpts, dm_kpts, kpts_band, kpts)

def get_j_for_bands(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    dmsR = dms.real.reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.reshape(nset,nkpts,nao**2)
    kpt_allow = numpy.zeros(3)
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(kpt_allow, False, mesh)
    ngrids = len(coulG)
    vG = numpy.zeros((nset,ngrids), dtype=numpy.complex128)
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8

    dmsC = dms.conj()
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, max_memory=max_memory):
        #:rho = numpy.einsum('lkL,lk->L', pqk.conj(), dm)
        for k, aoao in enumerate(aoaoks):
            for i in range(nset):
                rho = numpy.einsum('ij,Lij->L', dmsC[i,k],
                                   aoao.reshape(-1,nao,nao)).conj()
                vG[i,p0:p1] += rho * coulG[p0:p1]
    aoao = aoaoks = p0 = p1 = None
    weight = 1./len(kpts)
    vG *= weight
    t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vj_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts_band,
                                        max_memory=max_memory):
        for k, aoao in enumerate(aoaoks):
            for i in range(nset):
                vj_kpts[i,k] += numpy.einsum('L,Lij->ij', vG[i,p0:p1],
                                             aoao.reshape(-1,nao,nao))
    aoao = aoaoks = p0 = p1 = None

    if gamma_point(kpts_band):
        vj_kpts = vj_kpts.real.copy()
    t1 = log.timer_debug1('get_j pass 2', *t1)
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    mesh = mydf.mesh
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    swap_2e = (kpts_band is None)
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    kk_todo = numpy.ones(kk_table.shape[:2], dtype=bool)
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
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt, True, mesh)
        kptjs = kpts[kptj_idx]
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        #buf1R = numpy.empty((blksize*nao**2))
        #buf1I = numpy.empty((blksize*nao**2))
        for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt, kptjs, max_memory=max_memory1):
            nG = p1 - p0
            bufR = numpy.empty((nG*nao**2))
            bufI = numpy.empty((nG*nao**2))
            buf1R = numpy.empty((nG*nao**2))
            buf1I = numpy.empty((nG*nao**2))

            for k, aoao in enumerate(aoaoks):
                ki = kpti_idx[k]
                kj = kptj_idx[k]

# case 1: k_pq = (pi|iq)
#:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,jk->il', v4, dm)
                pLqR = numpy.ndarray((nao,nG,nao), buffer=bufR)
                pLqI = numpy.ndarray((nao,nG,nao), buffer=bufI)
                pLqR[:] = aoao.real.reshape(nG,nao,nao).transpose(1,0,2)
                pLqI[:] = aoao.imag.reshape(nG,nao,nao).transpose(1,0,2)
                iLkR = numpy.ndarray((nao,nG,nao), buffer=buf1R)
                iLkI = numpy.ndarray((nao,nG,nao), buffer=buf1I)
                for i in range(nset):
                    zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                           dmsR[i,kj], dmsI[i,kj], 1,
                           iLkR.reshape(-1,nao), iLkI.reshape(-1,nao))
                    iLkR *= vkcoulG[p0:p1].reshape(1,nG,1)
                    iLkI *= vkcoulG[p0:p1].reshape(1,nG,1)
                    zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           1, vkR[i,ki], vkI[i,ki], 1)

# case 2: k_pq = (iq|pi)
#:v4 = numpy.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,li->kj', v4, dm)
                if swap_2e and not is_zero(kpt):
                    for i in range(nset):
                        zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                               pLqI.reshape(nao,-1), 1,
                               iLkR.reshape(nao,-1), iLkI.reshape(nao,-1))
                        iLkR *= vkcoulG[p0:p1].reshape(1,nG,1)
                        iLkI *= vkcoulG[p0:p1].reshape(1,nG,1)
                        zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                               iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                               1, vkR[i,kj], vkI[i,kj], 1)

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)
        t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%ki, *t1)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    # Add ewald_exxdiv contribution because G=0 was not included in the
    # non-uniform grids
    if (exxdiv == 'ewald' and
        (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
         (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
        _ewald_exxdiv_for_G0(cell, kpts_band, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


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
    t1 = (time.clock(), time.time())

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not numpy.iscomplexobj(dms)

    mesh = mydf.mesh
    kptii = numpy.asarray((kpt,kpt))
    kpt_allow = numpy.zeros(3)

    if with_j:
        vjcoulG = mydf.weighted_coulG(kpt_allow, False, mesh)
        vjR = numpy.zeros((nset,nao,nao))
        vjI = numpy.zeros((nset,nao,nao))
    if with_k:
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt_allow, True, mesh)
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
    blksize = max(int(max_memory*.25e6/16/nao**2), 16)
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
                           1, vkR[i], vkI[i])
            #t2 = log.timer_debug1('        with_k', *t2)
        pqkR = pqkI = coulG = pLqR = pLqI = iLkR = iLkI = None
        #t2 = log.timer_debug1('%d:%d'%(p0,p1), *t2)
    bufR = bufI = None
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
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
        vk = vk.reshape(dm.shape)
    return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import aft

    L = 5.
    n = 10
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    df = aft.AFTDF(cell)
    df.mesh = (31,)*3
    dm = pscf.RHF(cell).get_init_guess()
    vj, vk = df.get_jk(dm)
    print(numpy.einsum('ij,ji->', df.get_nuc(), dm), 'ref=-10.577490961074622')
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=5.3766911667862516')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=8.2255177602309022')

