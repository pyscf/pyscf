#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import df
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import aft_jk
from pyscf.pbc.df.aft_jk import is_zero, gamma_point
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC, _format_dms

#
# Divide the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

def density_fit(mf, auxbasis=None, gs=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        gs : tuple
            number of grids in each (+)direction
        with_df : MDF object
    '''
    from pyscf.pbc.df import mdf
    if with_df is None:
        if hasattr(mf, 'kpts'):
            kpts = mf.kpts
        else:
            kpts = numpy.reshape(mf.kpt, (1,3))

        with_df = mdf.MDF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if gs is not None:
            with_df.gs = gs

    mf.with_df = with_df
    return mf


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None):
    if mydf._cderi is None:
        mydf.build()
    if mydf.metric is None:
        mydf.__class__, cls_bak = df.DF, mydf.__class__
        vk_kpts1 = df_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpt_band)
        mydf.__class__ = cls_bak
    else:
        vj_kpts1 = get_j_kpts_sr(mydf, dm_kpts, hermi, kpts, kpt_band)
    vj_kpts = aft_jk.get_j_kpts(mydf, dm_kpts, hermi, kpts, kpt_band)
    vj_kpts += vj_kpts1
    return vj_kpts

def get_j_kpts_sr(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer_debug1('Init get_j_kpts', *t1)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now)) * .9
    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    naux = mydf.auxcell.nao_nr()
    dmsR = dms.real.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.transpose(0,1,3,2).reshape(nset,nkpts,nao**2)
    rhoR  = numpy.zeros((nset,naux))
    rhoI  = numpy.zeros((nset,naux))
    jauxR = numpy.zeros((nset,naux))
    jauxI = numpy.zeros((nset,naux))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = LpqR + LpqI*1j
            #:j3c = j3cR + j3cI*1j
            #:rho [:,p0:p1] += numpy.einsum('Lpq,xqp->xL', Lpq, dms[:,k])
            #:jaux[:,p0:p1] += numpy.einsum('Lpq,xqp->xL', j3c, dms[:,k])
            rhoR [:,p0:p1]+= numpy.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
            rhoR [:,p0:p1]-= numpy.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
            rhoI [:,p0:p1]+= numpy.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
            rhoI [:,p0:p1]+= numpy.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            jauxR[:,p0:p1]+= numpy.einsum('Lp,xp->xL', j3cR, dmsR[:,k])
            jauxR[:,p0:p1]-= numpy.einsum('Lp,xp->xL', j3cI, dmsI[:,k])
            jauxI[:,p0:p1]+= numpy.einsum('Lp,xp->xL', j3cR, dmsI[:,k])
            jauxI[:,p0:p1]+= numpy.einsum('Lp,xp->xL', j3cI, dmsR[:,k])
            LpqR = LpqI = j3cR = j3cI = None

    weight = 1./nkpts
    jauxR *= weight
    jauxI *= weight
    rhoR *= weight
    rhoI *= weight
    vjR = numpy.zeros((nset,nband,nao,nao))
    vjI = numpy.zeros((nset,nband,nao,nao))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, True):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:v = numpy.dot(jaux, Lpq) + numpy.dot(rho, j3c)
            #:vj_kpts[:,k] += lib.unpack_tril(v)
            v  = numpy.dot(jauxR[:,p0:p1], LpqR)
            v -= numpy.dot(jauxI[:,p0:p1], LpqI)
            v += numpy.dot(rhoR [:,p0:p1], j3cR)
            v -= numpy.dot(rhoI [:,p0:p1], j3cI)
            vjR[:,k] += lib.unpack_tril(v)
            if not j_real:
                v  = numpy.dot(jauxR[:,p0:p1], LpqI)
                v += numpy.dot(jauxI[:,p0:p1], LpqR)
                v += numpy.dot(rhoR [:,p0:p1], j3cI)
                v += numpy.dot(rhoI [:,p0:p1], j3cR)
                vjI[:,k] += lib.unpack_tril(v, lib.ANTIHERMI)
            LpqR = LpqI = j3cR = j3cI = None
    t1 = log.timer_debug1('get_j pass 2', *t1)

    if j_real:
        vj_kpts = vjR.reshape(vj_kpts.shape)
    else:
        vj_kpts = (vjR+vjI*1j).reshape(vj_kpts.shape)
    return vj_kpts


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None,
               exxdiv=None):
    if mydf._cderi is None:
        mydf.build()
    if mydf.metric is None:
        mydf.__class__, cls_bak = df.DF, mydf.__class__
        vk_kpts1 = df_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpt_band, None)
        mydf.__class__ = cls_bak
    else:
        vk_kpts1 = get_k_kpts_sr(mydf, dm_kpts, hermi, kpts, kpt_band, None)
    vk_kpts = aft_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpt_band, exxdiv)
    vk_kpts += vk_kpts1
    return vk_kpts

def get_k_kpts_sr(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None,
                  exxdiv=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer_debug1('Init get_k_kpts', *t1)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    if kpt_band is None:
        kpts_band = kpts
        swap_2e = True
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
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
        kk_todo[kpti_idx,kptj_idx] = False
        if swap_2e and not is_zero(kpt):
            kk_todo[kptj_idx,kpti_idx] = False

        # Note: kj-ki for electorn 1 and ki-kj for electron 2
        # j2c ~ ({kj-ki}|{ks-kr}) ~ ({kj-ki}|-{kj-ki}) ~ ({kj-ki}|{ki-kj})
        # j3c ~ (Q|kj,ki) = j3c{ji} = (Q|ki,kj)* = conj(transpose(j3c{ij}, (0,2,1)))

        bufR = numpy.empty((mydf.blockdim*nao**2))
        bufI = numpy.empty((mydf.blockdim*nao**2))
        for ki,kj in zip(kpti_idx,kptj_idx):
            kpti = kpts_band[ki]
            kptj = kpts[kj]
            kptij = numpy.asarray((kpti,kptj))
            for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptij, max_memory, False):
                nrow = LpqR.shape[0]
                pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                pjqR = numpy.ndarray((nao,nrow,nao), buffer=LpqR)
                pjqI = numpy.ndarray((nao,nrow,nao), buffer=LpqI)
                tmpR = numpy.ndarray((nao,nrow*nao), buffer=j3cR)
                tmpI = numpy.ndarray((nao,nrow*nao), buffer=j3cI)
                pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
                pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)
                pjqR[:] = j3cR.reshape(-1,nao,nao).transpose(1,0,2)
                pjqI[:] = j3cI.reshape(-1,nao,nao).transpose(1,0,2)

                #:Lpq = LpqR + LpqI*1j
                #:j3c = j3cR + j3cI*1j
                #:for i in range(nset):
                #:    dm = dms[i,ki]
                #:    tmp = numpy.dot(dm, j3c.reshape(nao,-1))
                #:    vk1 = numpy.dot(Lpq.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
                #:    tmp = numpy.dot(dm, Lpq.reshape(nao,-1))
                #:    vk1+= numpy.dot(j3c.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
                #:    vkR[i,kj] += vk1.real
                #:    vkI[i,kj] += vk1.imag

                #:if swap_2e and not is_zero(kpt):
                #:    # K ~ 'Lij,Llk*,jk->il' + 'Llk*,Lij,jk->il'
                #:    for i in range(nset):
                #:        dm = dms[i,kj]
                #:        tmp = numpy.dot(j3c.reshape(-1,nao), dm)
                #:        vk1 = numpy.dot(tmp.reshape(nao,-1), Lpq.reshape(nao,-1).conj().T)
                #:        tmp = numpy.dot(Lpq.reshape(-1,nao), dm)
                #:        vk1+= numpy.dot(tmp.reshape(nao,-1), j3c.reshape(nao,-1).conj().T)
                #:        vkR[i,ki] += vk1.real
                #:        vkI[i,ki] += vk1.imag

                # K ~ 'iLj,lLk*,li->kj' + 'lLk*,iLj,li->kj'
                for i in range(nset):
                    tmpR, tmpI = zdotNN(dmsR[i,ki], dmsI[i,ki], pjqR.reshape(nao,-1),
                                        pjqI.reshape(nao,-1), 1, tmpR, tmpI)
                    vk1R, vk1I = zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                                        tmpR.reshape(-1,nao), tmpI.reshape(-1,nao))
                    vkR[i,kj] += vk1R
                    vkI[i,kj] += vk1I
                    if hermi:
                        vkR[i,kj] += vk1R.T
                        vkI[i,kj] -= vk1I.T
                    else:
                        tmpR, tmpI = zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                                            pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                        zdotCN(pjqR.reshape(-1,nao).T, pjqI.reshape(-1,nao).T,
                               tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                               1, vkR[i,kj], vkI[i,kj], 1)

                if swap_2e and not is_zero(kpt):
                    tmpR = tmpR.reshape(nao*nrow,nao)
                    tmpI = tmpI.reshape(nao*nrow,nao)
                    # K ~ 'iLj,lLk*,jk->il' + 'lLk*,iLj,jk->il'
                    for i in range(nset):
                        tmpR, tmpI = zdotNN(pjqR.reshape(-1,nao), pjqI.reshape(-1,nao),
                                            dmsR[i,kj], dmsI[i,kj], 1, tmpR, tmpI)
                        vk1R, vk1I = zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                                            pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T)
                        vkR[i,ki] += vk1R
                        vkI[i,ki] += vk1I
                        if hermi:
                            vkR[i,ki] += vk1R.T
                            vkI[i,ki] -= vk1I.T
                        else:
                            tmpR, tmpI = zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                                                dmsR[i,kj], dmsI[i,kj], 1, tmpR, tmpI)
                            zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                                   pjqR.reshape(nao,-1).T, pjqI.reshape(nao,-1).T,
                                   1, vkR[i,ki], vkI[i,ki], 1)
                LpqR = LpqI = j3cR = j3cI = tmpR = tmpI = None
        return None

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)
    vkR *= 1./nkpts
    vkI *= 1./nkpts

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        vk_kpts = vkR.reshape(vk_kpts.shape)
    else:
        vk_kpts = (vkR+vkI*1j).reshape(vk_kpts.shape)
    return vk_kpts


##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    vj = vk = None
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(mydf, [dm], hermi, kpt, kpt_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, [dm], hermi, kpt, kpt_band)
        return vj, vk

    if mydf._cderi is None:
        mydf.build()
    if mydf.metric is None:
        mydf.__class__, cls_bak = df.DF, mydf.__class__
        vj1, vk1 = df_jk.get_jk(mydf, dm, hermi, kpt, kpt_band, with_j, with_k, None)
        mydf.__class__ = cls_bak
    else:
        vj1, vk1 = get_jk_sr(mydf, dm, hermi, kpt, kpt_band, with_j, with_k, None)
    vj, vk = aft_jk.get_jk(mydf, dm, hermi, kpt, kpt_band, with_j, with_k, exxdiv)
    if with_j: vj += vj1
    if with_k: vk += vk1
    return vj, vk

def get_jk_sr(mydf, dm, hermi=1, kpt=numpy.zeros(3),
              kpt_band=None, with_j=True, with_k=True, exxdiv=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t2 = t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer_debug1('Init get_jk', *t1)

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not numpy.iscomplexobj(dms)
    kptii = numpy.asarray((kpt,kpt))

# .45 is estimation for the memory usage ratio  sr_loop / (sr_loop+bufR+bufI)
    dmsR = numpy.asarray(dms.real.reshape(nset,nao,nao), order='C')
    dmsI = numpy.asarray(dms.imag.reshape(nset,nao,nao), order='C')
    if with_j:
        vjR = numpy.zeros((nset,nao,nao))
        vjI = numpy.zeros((nset,nao,nao))
    if with_k:
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0])) * .45
    if with_k:
        buf1R = numpy.empty((mydf.blockdim*nao**2))
        buf2R = numpy.empty((mydf.blockdim*nao**2))
        buf3R = numpy.empty((mydf.blockdim*nao**2))
        if not k_real:
            buf1I = numpy.empty((mydf.blockdim*nao**2))
            buf2I = numpy.empty((mydf.blockdim*nao**2))
            buf3I = numpy.empty((mydf.blockdim*nao**2))
    def contract_k(pLqR, pLqI, pjqR, pjqI):
        # K ~ 'iLj,lLk*,li->kj' + 'lLk*,iLj,li->kj'
        #:Lpq = LpqR + LpqI*1j
        #:j3c = j3cR + j3cI*1j
        #:for i in range(nset):
        #:    tmp = numpy.dot(dms[i], j3c.reshape(nao,-1))
        #:    vk1 = numpy.dot(Lpq.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
        #:    tmp = numpy.dot(dms[i], Lpq.reshape(nao,-1))
        #:    vk1+= numpy.dot(j3c.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
        #:    vkR[i] += vk1.real
        #:    vkI[i] += vk1.imag
        nrow = pLqR.shape[1]
        tmpR = numpy.ndarray((nao,nrow*nao), buffer=buf3R)
        if k_real:
            for i in range(nset):
                tmpR = lib.ddot(dmsR[i], pjqR.reshape(nao,-1), 1, tmpR)
                vk1R = lib.ddot(pLqR.reshape(-1,nao).T, tmpR.reshape(-1,nao))
                vkR[i] += vk1R
                if hermi:
                    vkR[i] += vk1R.T
                else:
                    tmpR = lib.ddot(dmsR[i], pLqR.reshape(nao,-1), 1, tmpR)
                    lib.ddot(pjqR.reshape(-1,nao).T, tmpR.reshape(-1,nao),
                             1, vkR[i], 1)
        else:
            tmpI = numpy.ndarray((nao,nrow*nao), buffer=buf3I)
            for i in range(nset):
                tmpR, tmpI = zdotNN(dmsR[i], dmsI[i], pjqR.reshape(nao,-1),
                                    pjqI.reshape(nao,-1), 1, tmpR, tmpI, 0)
                vk1R, vk1I = zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                                    tmpR.reshape(-1,nao), tmpI.reshape(-1,nao))
                vkR[i] += vk1R
                vkI[i] += vk1I
                if hermi:
                    vkR[i] += vk1R.T
                    vkI[i] -= vk1I.T
                else:
                    tmpR, tmpI = zdotNN(dmsR[i], dmsI[i], pLqR.reshape(nao,-1),
                                        pLqI.reshape(nao,-1), 1, tmpR, tmpI, 0)
                    zdotCN(pjqR.reshape(-1,nao).T, pjqI.reshape(-1,nao).T,
                           tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                           1, vkR[i], vkI[i], 1)

    pLqI = pjqI = None
    thread_k = None
    for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, False):
        LpqR = LpqR.reshape(-1,nao,nao)
        LpqI = LpqI.reshape(-1,nao,nao)
        j3cR = j3cR.reshape(-1,nao,nao)
        j3cI = j3cI.reshape(-1,nao,nao)
        t2 = log.timer_debug1('        load', *t2)
        if thread_k is not None:
            thread_k.join()
        if with_j:
            #:rho_coeff = numpy.einsum('Lpq,xqp->xL', Lpq, dms)
            #:jaux = numpy.einsum('Lpq,xqp->xL', j3c, dms)
            #:vj += numpy.dot(jaux, Lpq.reshape(-1,nao**2))
            #:vj += numpy.dot(rho_coeff, j3c.reshape(-1,nao**2))
            rhoR  = numpy.einsum('Lpq,xqp->xL', LpqR, dmsR)
            jauxR = numpy.einsum('Lpq,xqp->xL', j3cR, dmsR)
            if not j_real:
                rhoR -= numpy.einsum('Lpq,xqp->xL', LpqI, dmsI)
                rhoI  = numpy.einsum('Lpq,xqp->xL', LpqR, dmsI)
                rhoI += numpy.einsum('Lpq,xqp->xL', LpqI, dmsR)
                jauxR-= numpy.einsum('Lpq,xqp->xL', j3cI, dmsI)
                jauxI = numpy.einsum('Lpq,xqp->xL', j3cR, dmsI)
                jauxI+= numpy.einsum('Lpq,xqp->xL', j3cI, dmsR)
            vjR += numpy.einsum('xL,Lpq->xpq', jauxR, LpqR)
            vjR += numpy.einsum('xL,Lpq->xpq', rhoR, j3cR)
            if not j_real:
                vjR -= numpy.einsum('xL,Lpq->xpq', jauxI, LpqI)
                vjR -= numpy.einsum('xL,Lpq->xpq', rhoI, j3cI)
                vjI += numpy.einsum('xL,Lpq->xpq', jauxR, LpqI)
                vjI += numpy.einsum('xL,Lpq->xpq', jauxI, LpqR)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoR, j3cI)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoI, j3cR)
        t2 = log.timer_debug1('        with_j', *t2)
        if with_k:
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=buf1R)
            pjqR = numpy.ndarray((nao,nrow,nao), buffer=buf2R)
            pLqR[:] = LpqR.transpose(1,0,2)
            pjqR[:] = j3cR.transpose(1,0,2)
            if not k_real:
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=buf1I)
                pjqI = numpy.ndarray((nao,nrow,nao), buffer=buf2I)
                pLqI[:] = LpqI.transpose(1,0,2)
                pjqI[:] = j3cI.transpose(1,0,2)

            thread_k = lib.background_thread(contract_k, pLqR, pLqI, pjqR, pjqI)
            t2 = log.timer_debug1('        with_k', *t2)
        LpqR = LpqI = j3cR = j3cI = None
    if thread_k is not None:
        thread_k.join()
    thread_k = None
    t1 = log.timer_debug1('mdf_jk.get_jk pass 1', *t1)

    if with_j:
        if j_real:
            vj = vjR.reshape(dm.shape)
        else:
            vj = (vjR+vjI*1j).reshape(dm.shape)
    if with_k:
        if k_real:
            vk = vkR.reshape(dm.shape)
        else:
            vk = (vkR+vkI*1j).reshape(dm.shape)
    return vj, vk


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5
    #print cell.nimgs
    #cell.nimgs = [4,4,4]

    mf = pscf.RHF(cell)
    auxbasis = 'weigend'
    mf = density_fit(mf, auxbasis)
    mf.with_df.gs = (5,) * 3
    mf.with_df.approx_sr_level = 3
    dm = mf.get_init_guess()
    vj = get_jk(mf.with_df, dm, exxdiv=mf.exxdiv, with_k=False)[0]
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.69745030912447')
    vj, vk = get_jk(mf.with_df, dm, exxdiv=mf.exxdiv)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.69745030912447')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.33704732444835')
    print(numpy.einsum('ij,ji->', mf.get_hcore(cell), dm), 'ref=-75.574414055823766')

