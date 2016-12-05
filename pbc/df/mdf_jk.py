#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools

#
# Split the Coulomb potential to two parts.  Computing short range part in
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
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer_debug1('Init get_j_kpts', *t1)

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    naux = mydf.auxcell.nao_nr()

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band)

    dmsR = dms.real.reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.reshape(nset,nkpts,nao**2)
    kpt_allow = numpy.zeros(3)
    coulG = mydf.weighted_coulG(kpt_allow, False, mydf.gs)
    ngs = len(coulG)
    vR = numpy.zeros((nset,ngs))
    vI = numpy.zeros((nset,ngs))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .9)
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts, max_memory=max_memory):
        # contract dm to rho_rs(-G+k_rs)  (Note no .T on dm)
        # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
        #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        for i in range(nset):
            rhoR = numpy.dot(dmsR[i,k], pqkR)
            rhoR+= numpy.dot(dmsI[i,k], pqkI)
            rhoI = numpy.dot(dmsI[i,k], pqkR)
            rhoI-= numpy.dot(dmsR[i,k], pqkI)
            vR[i,p0:p1] += rhoR * coulG[p0:p1]
            vI[i,p0:p1] += rhoI * coulG[p0:p1]
        pqkR = pqkI = None
    weight = 1./nkpts
    vR *= weight
    vI *= weight

    pqkR = pqkI = coulG = None
    t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

    vjR = numpy.zeros((nset,nband,nao*nao))
    vjI = numpy.zeros((nset,nband,nao*nao))
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts_band, max_memory=max_memory):
        for i in range(nset):
            vjR[i,k] += numpy.dot(pqkR, vR[i,p0:p1])
            vjR[i,k] -= numpy.dot(pqkI, vI[i,p0:p1])
            if not j_real:
                vjI[i,k] += numpy.dot(pqkI, vR[i,p0:p1])
                vjI[i,k] += numpy.dot(pqkR, vI[i,p0:p1])
        pqkR = pqkI = None

    rhoR  = numpy.zeros((nset,naux))
    rhoI  = numpy.zeros((nset,naux))
    jauxR = numpy.zeros((nset,naux))
    jauxI = numpy.zeros((nset,naux))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j).transpose(1,0,2)
            #:j3c = (j3cR + j3cI*1j).transpose(1,0,2)
            #:rho [:,p0:p1] += numpy.einsum('Lpq,xpq->xL', Lpq, dms[:,k])
            #:jaux[:,p0:p1] += numpy.einsum('Lpq,xpq->xL', j3c, dms[:,k])
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
    vjR = vjR.reshape(nset,nband,nao,nao)
    vjI = vjI.reshape(nset,nband,nao,nao)
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
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if nset == 1:  # One set of dm_kpts for KRHF
            return vj_kpts[0,0]
        else:
            return vj_kpts[:,0]
    else:
        return vj_kpts.reshape(dm_kpts.shape)


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None,
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

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    def make_kpt(kpt):  # kpt = kptj - kpti
        # search for all possible ki and kj that has ki-kj+kpt=0
        kk_match = numpy.einsum('ijx->ij', abs(kk_table + kpt)) < 1e-9
        kpti_idx, kptj_idx = numpy.where(kk_todo & kk_match)
        nkptj = len(kptj_idx)
        log.debug1('kpt = %s', kpt)
        log.debug1('kpti_idx = %s', kpti_idx)
        log.debug1('kptj_idx = %s', kptj_idx)
        kk_todo[kpti_idx,kptj_idx] = False
        if swap_2e and not is_zero(kpt):
            kk_todo[kptj_idx,kpti_idx] = False

        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0])*.9)
        max_memory = max_memory * (nkptj+1)/(nkptj+5)
        blksize = max(int(max_memory*4e6/(nkptj+5)/16/nao**2), 16)
        bufR = numpy.empty((blksize*nao**2))
        bufI = numpy.empty((blksize*nao**2))
        # Use DF object to mimic KRHF/KUHF object in function get_coulG
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt, True, mydf.gs)
        kptjs = kpts[kptj_idx]
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        for k, pqkR, pqkI, p0, p1 \
                in mydf.ft_loop(mydf.gs, kpt, kptjs, max_memory=max_memory):
            ki = kpti_idx[k]
            kj = kptj_idx[k]
            coulG = numpy.sqrt(vkcoulG[p0:p1])

# case 1: k_pq = (pi|iq)
#:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            pqkR *= coulG
            pqkI *= coulG
            pLqR = lib.transpose(pqkR.reshape(nao,nao,-1), axes=(0,2,1), out=bufR)
            pLqI = lib.transpose(pqkI.reshape(nao,nao,-1), axes=(0,2,1), out=bufI)
            iLkR = numpy.empty((nao*(p1-p0),nao))
            iLkI = numpy.empty((nao*(p1-p0),nao))
            for i in range(nset):
                iLkR, iLkI = zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                                    dmsR[i,kj], dmsI[i,kj], 1, iLkR, iLkI)
                zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                       pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                       1, vkR[i,ki], vkI[i,ki], 1)

# case 2: k_pq = (iq|pi)
#:v4 = numpy.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,li->kj', v4, dm)
            if swap_2e and not is_zero(kpt):
                iLkR = iLkR.reshape(nao,-1)
                iLkI = iLkI.reshape(nao,-1)
                for i in range(nset):
                    iLkR, iLkI = zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                                        pLqI.reshape(nao,-1), 1, iLkR, iLkI)
                    zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                           iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                           1, vkR[i,kj], vkI[i,kj], 1)
            pqkR = pqkI = pLqR = pLqI = iLkR = iLkI = None
        pqkR = pqkI = iLkR = iLkI = coulG = None

        # Note: kj-ki for electorn 1 and ki-kj for electron 2
        # j2c ~ ({kj-ki}|{ks-kr}) ~ ({kj-ki}|-{kj-ki}) ~ ({kj-ki}|{ki-kj})
        # j3c ~ (Q|kj,ki) = j3c{ji} = (Q|ki,kj)* = conj(transpose(j3c{ij}, (0,2,1)))

        bufR = numpy.empty((mydf.blockdim*nao**2))
        bufI = numpy.empty((mydf.blockdim*nao**2))
        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
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

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if nset == 1:  # One set of dm_kpts for KRHF
            return vk_kpts[0,0]
        else:
            return vk_kpts[:,0]
    else:
        return vk_kpts.reshape(dm_kpts.shape)


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

    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    if mydf._cderi is None:
        mydf.build()
        t1 = log.timer_debug1('Init get_jk', *t1)

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not numpy.iscomplexobj(dms)

    naux = mydf.auxcell.nao_nr()
    kptii = numpy.asarray((kpt,kpt))
    kpt_allow = numpy.zeros(3)

    if with_j:
        vjcoulG = mydf.weighted_coulG(kpt, False, mydf.gs)
        vjR = numpy.zeros((nset,nao**2))
        vjI = numpy.zeros((nset,nao**2))
    if with_k:
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt, True, mydf.gs)
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
    dmsR = dms.real.reshape(nset,nao,nao)
    dmsI = dms.imag.reshape(nset,nao,nao)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    max_memory *= .8  # *.8 for the temporary data in vk

    # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
    #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
    blksize = max(int(max_memory*.25e6/16/nao**2), 16)
    bufR = numpy.empty(blksize*nao**2)
    bufI = numpy.empty(blksize*nao**2)
    for pqkR, pqkI, p0, p1 \
            in mydf.pw_loop(mydf.gs, kptii, max_memory=max_memory):
        if with_j:
            for i in range(nset):
                if j_real:
                    rhoR = numpy.dot(dmsR[i].ravel(), pqkR)
                    rhoI = numpy.dot(dmsR[i].ravel(), pqkI)
                    rhoR *= vjcoulG[p0:p1]
                    rhoI *= vjcoulG[p0:p1]
                    vjR[i] += numpy.dot(pqkR, rhoR)
                    vjR[i] += numpy.dot(pqkI, rhoI)
                else:
                    rhoR = numpy.dot(dmsR[i].ravel(), pqkR)
                    rhoR+= numpy.dot(dmsI[i].ravel(), pqkI)
                    rhoI = numpy.dot(dmsI[i].ravel(), pqkR)
                    rhoI-= numpy.dot(dmsR[i].ravel(), pqkI)
                    rhoR *= vjcoulG[p0:p1]
                    rhoI *= vjcoulG[p0:p1]
                    vjR[i] += numpy.dot(pqkR, rhoR)
                    vjR[i] -= numpy.dot(pqkI, rhoI)
                    vjI[i] += numpy.dot(pqkR, rhoI)
                    vjI[i] += numpy.dot(pqkI, rhoR)

        if with_k:
            coulG = numpy.sqrt(vkcoulG[p0:p1])
            pqkR *= coulG
            pqkI *= coulG
            #:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
            #:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            pLqR = lib.transpose(pqkR.reshape(nao,nao,-1), axes=(0,2,1), out=bufR).reshape(-1,nao)
            pLqI = lib.transpose(pqkI.reshape(nao,nao,-1), axes=(0,2,1), out=bufI).reshape(-1,nao)
            iLkR = numpy.ndarray((nao*(p1-p0),nao), buffer=pqkR)
            iLkI = numpy.ndarray((nao*(p1-p0),nao), buffer=pqkI)
            for i in range(nset):
                if k_real:
                    lib.dot(pLqR, dmsR[i], 1, iLkR)
                    lib.dot(pLqI, dmsR[i], 1, iLkI)
                    lib.dot(iLkR.reshape(nao,-1), pLqR.reshape(nao,-1).T, 1, vkR[i], 1)
                    lib.dot(iLkI.reshape(nao,-1), pLqI.reshape(nao,-1).T, 1, vkR[i], 1)
                else:
                    zdotNN(pLqR, pLqI, dmsR[i], dmsI[i], 1, iLkR, iLkI)
                    zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           1, vkR[i], vkI[i])
        pqkR = pqkI = coulG = pLqR = pLqI = iLkR = iLkI = None

    bufR = numpy.empty((mydf.blockdim*nao**2))
    bufI = numpy.empty((mydf.blockdim*nao**2))
# .75 is estimation for the memory usage ratio  sr_loop / (sr_loop+bufR+bufI)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0])) * .75
    if with_j:
        vjR = vjR.reshape(nset,nao,nao)
        vjI = vjI.reshape(nset,nao,nao)
    for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, False):
        LpqR = LpqR.reshape(-1,nao,nao)
        LpqI = LpqI.reshape(-1,nao,nao)
        j3cR = j3cR.reshape(-1,nao,nao)
        j3cI = j3cI.reshape(-1,nao,nao)
        if with_j:
            #:rho_coeff = numpy.einsum('Lpq,xqp->xL', Lpq, dms)
            #:jaux = numpy.einsum('Lpq,xqp->xL', j3c, dms)
            #:vj += numpy.dot(jaux, Lpq.reshape(-1,nao**2))
            #:vj += numpy.dot(rho_coeff, j3c.reshape(-1,nao**2))
            rhoR  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsR)
            jauxR = numpy.einsum('Lpq,xpq->xL', j3cR, dmsR)
            if not j_real:
                rhoR -= numpy.einsum('Lpq,xpq->xL', LpqI, dmsI)
                rhoI  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsI)
                rhoI += numpy.einsum('Lpq,xpq->xL', LpqI, dmsR)
                jauxR-= numpy.einsum('Lpq,xpq->xL', j3cI, dmsI)
                jauxI = numpy.einsum('Lpq,xpq->xL', j3cR, dmsI)
                jauxI+= numpy.einsum('Lpq,xpq->xL', j3cI, dmsR)
            vjR += numpy.einsum('xL,Lpq->xpq', jauxR, LpqR)
            vjR += numpy.einsum('xL,Lpq->xpq', rhoR, j3cR)
            if not j_real:
                vjR -= numpy.einsum('xL,Lpq->xpq', jauxI, LpqI)
                vjR -= numpy.einsum('xL,Lpq->xpq', rhoI, j3cI)
                vjI += numpy.einsum('xL,Lpq->xpq', jauxR, LpqI)
                vjI += numpy.einsum('xL,Lpq->xpq', jauxI, LpqR)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoR, j3cI)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoI, j3cR)

        if with_k:
            #:Lpq = LpqR + LpqI*1j
            #:j3c = j3cR + j3cI*1j
            #:for i in range(nset):
            #:    tmp = numpy.dot(dms[i], j3c.reshape(nao,-1))
            #:    vk1 = numpy.dot(Lpq.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
            #:    tmp = numpy.dot(dms[i], Lpq.reshape(nao,-1))
            #:    vk1+= numpy.dot(j3c.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
            #:    vkR[i] += vk1.real
            #:    vkI[i] += vk1.imag
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
            pjqR = numpy.ndarray((nao,nrow,nao), buffer=LpqR)
            tmpR = numpy.ndarray((nao,nrow*nao), buffer=j3cR)
            pLqR[:] = LpqR.transpose(1,0,2)
            pjqR[:] = j3cR.transpose(1,0,2)
            # K ~ 'iLj,lLk*,li->kj' + 'lLk*,iLj,li->kj'
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
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                pjqI = numpy.ndarray((nao,nrow,nao), buffer=LpqI)
                tmpI = numpy.ndarray((nao,nrow*nao), buffer=j3cI)
                pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)
                pjqI[:] = j3cI.reshape(-1,nao,nao).transpose(1,0,2)
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
        LpqR = LpqI = j3cR = j3cI = vk1R = vk1I = None
        pLqR = pLqI = pjqR = pjqI = tmpR = tmpI = None

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
        vk = vk.reshape(dm.shape)
    t1 = log.timer('sr jk', *t1)
    return vj, vk


def _format_dms(dm_kpts, kpts):
    nkpts = len(kpts)
    nao = dm_kpts.shape[-1]
    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    return dms

def is_zero(kpt):
    return kpt is None or abs(kpt).sum() < 1e-9
gamma_point = is_zero

def zdotNN(aR, aI, bR, bI, alpha=1, cR=None, cI=None, beta=0):
    '''c = a*b'''
    cR = lib.ddot(aR, bR, alpha, cR, beta)
    cR = lib.ddot(aI, bI,-alpha, cR, 1   )
    cI = lib.ddot(aR, bI, alpha, cI, beta)
    cI = lib.ddot(aI, bR, alpha, cI, 1   )
    return cR, cI

def zdotCN(aR, aI, bR, bI, alpha=1, cR=None, cI=None, beta=0):
    '''c = a.conj()*b'''
    cR = lib.ddot(aR, bR, alpha, cR, beta)
    cR = lib.ddot(aI, bI, alpha, cR, 1   )
    cI = lib.ddot(aR, bI, alpha, cI, beta)
    cI = lib.ddot(aI, bR,-alpha, cI, 1   )
    return cR, cI

def zdotNC(aR, aI, bR, bI, alpha=1, cR=None, cI=None, beta=0):
    '''c = a*b.conj()'''
    cR = lib.ddot(aR, bR, alpha, cR, beta)
    cR = lib.ddot(aI, bI, alpha, cR, 1   )
    cI = lib.ddot(aR, bI,-alpha, cI, beta)
    cI = lib.ddot(aI, bR, alpha, cI, 1   )
    return cR, cI


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

