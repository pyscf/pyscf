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
        with_df : XDF object
    '''
    from pyscf.pbc.df import xdf
    if with_df is None:
        with_df = xdf.XDF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if gs is not None:
            with_df.gs = gs
        if hasattr(mf, 'kpts'):
            with_df.kpts = mf.kpts
        else:
            with_df.kpts = numpy.reshape(mf.kpt, (1,3))

    mf = copy.copy(mf)
    mf.with_df = with_df
    return mf


def get_j_kpts(xdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None, jkops=None):
    log = logger.Logger(xdf.stdout, xdf.verbose)
    t1 = (time.clock(), time.time())
    if xdf._cderi is None:
        xdf.build()
        t1 = log.timer('Init get_j_kpts', *t1)
    auxcell = xdf.auxcell
    dm_kpts = numpy.asarray(dm_kpts)
    nao = dm_kpts.shape[-1]
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    nkpts = len(kpts)

    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    nset = dms.shape[0]

    rho_tot = numpy.zeros((nset,naux))
    jaux = numpy.zeros((nset,naux))
    rho_coeffs = numpy.empty((nset,nkpts,naux))
    for k, kpt in enumerate(kpts):
        kptii = numpy.asarray((kpt,kpt))
        for Lpq in xdf.load_Lpq(kptii):
            if Lpq.shape[1] == nao_pair:
                Lpq = lib.unpack_tril(Lpq)
        for jpq in xdf.load_j3c(kptii):
            if jpq.shape[1] == nao_pair:
                jpq = lib.unpack_tril(jpq)

        for i in range(nset):
            jaux[i] += numpy.einsum('kij,ji->k', jpq, dms[i,k]).real
            rho_coeff = numpy.einsum('kij,ji->k', Lpq, dms[i,k]).real
            rho_tot[i] += rho_coeff
            rho_coeffs[i,k] = rho_coeff
        Lpq = jpq = None
    weight = 1./nkpts
    jaux *= weight
    rho_tot *= weight

    ngs = numpy.prod(numpy.asarray(xdf.gs)*2+1)
    vR = numpy.zeros((nset,ngs))
    vI = numpy.zeros((nset,ngs))
    max_memory = xdf.max_memory - lib.current_memory()[0]
    p0 = 0
    for k, kpt, pqkR, LkR, pqkI, LkI, coulG \
            in xdf.ft_loop(cell, auxcell, xdf.gs, numpy.zeros(3), kpts, max_memory):
        nG = len(coulG)
        # contract dm to rho_rs(-G+k_rs)
        # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
        #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        for i in range(nset):
            dmR = dms[i,k].real.ravel(order='C')
            dmI = dms[i,k].imag.ravel(order='C')
            rhoR = numpy.dot(dmR, pqkR)
            rhoR-= numpy.dot(dmI, pqkI)
            rhoR-= numpy.dot(rho_coeffs[i,k], LkR)
            rhoI = numpy.dot(dmR, pqkI)
            rhoI+= numpy.dot(dmI, pqkR)
            rhoI-= numpy.dot(rho_coeffs[i,k], LkI)
            vR[i,p0:p0+nG] += rhoR * coulG
            vI[i,p0:p0+nG] += rhoI * coulG
        if k+1 == nkpts:
            p0 += nG
    weight = 1./nkpts
    vR *= weight
    vI *= weight

    j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN)
    jaux -= numpy.dot(rho_tot, j2c.T)
    pqkR = LkR = pqkI = LkI = coulG = j2c = None
    t1 = log.timer('get_j pass 1 to compute J(G)', *t1)

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)

    vjR = numpy.zeros((nset,nband,nao*nao))
    vjI = numpy.zeros((nset,nband,nao*nao))
    p0 = 0
    for k, kpt, pqkR, LkR, pqkI, LkI, coulG \
            in xdf.ft_loop(cell, auxcell, xdf.gs, numpy.zeros(3),
                           kpts_band, max_memory):
        nG = len(coulG)
        for i in range(nset):
            vjR[i,k] += numpy.dot(pqkR, vR[i,p0:p0+nG])
            vjR[i,k] += numpy.dot(pqkI, vI[i,p0:p0+nG])
        if abs(kpt).sum() > 1e-9:  # if not gamma point
            for i in range(nset):
                vjI[i,k] += numpy.dot(pqkI, vR[i,p0:p0+nG])
                vjI[i,k] -= numpy.dot(pqkR, vI[i,p0:p0+nG])
        if k+1 == nband:  # Construct jaux once, it is the same for all kpts
            for i in range(nset):
                jaux[i] -= numpy.dot(LkR, vR[i,p0:p0+nG])
                jaux[i] -= numpy.dot(LkI, vI[i,p0:p0+nG])
            p0 += nG
        pqkR = LkR = pqkI = LkI = coulG = None

    if abs(kpts_band).sum() < 1e-9:  # gamma point only
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j

    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        for Lpq in xdf.load_Lpq(kptii):
            if Lpq.shape[1] == nao_pair:
                Lpq = lib.unpack_tril(Lpq).reshape(naux,-1)
        for jpq in xdf.load_j3c(kptii):
            if jpq.shape[1] == nao_pair:
                jpq = lib.unpack_tril(jpq).reshape(naux,-1)

        vj_kpts[:,k] += numpy.dot(jaux, Lpq)
        vj_kpts[:,k] += numpy.dot(rho_tot, jpq)
    vj_kpts = vj_kpts.reshape(-1,nband,nao,nao)
    t1 = log.timer('get_j pass 2', *t1)

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if dm_kpts.ndim == 3:  # One set of dm_kpts for KRHF
            return vj_kpts[0,0]
        else:
            return vj_kpts[:,0]
    else:
        return vj_kpts.reshape(dm_kpts.shape)


def get_k_kpts(xdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None):
    log = logger.Logger(xdf.stdout, xdf.verbose)
    t1 = (time.clock(), time.time())
    if xdf._cderi is None:
        xdf.build()
        t1 = log.timer('Init get_k_kpts', *t1)
    auxcell = xdf.auxcell
    dm_kpts = numpy.asarray(dm_kpts)
    nao = dm_kpts.shape[-1]
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    nkpts = len(kpts)

    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    nset = dms.shape[0]

    if kpt_band is None:
        kpts_band = kpts
        swap_2e = True
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    kk_todo = numpy.ones(kk_table.shape[:2], dtype=bool)
    vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

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
        if swap_2e and abs(kpt).sum() > 1e-9:
            kk_todo[kptj_idx,kpti_idx] = False

        LpqR = []
        LpqI = []
        for ki,kj in zip(kpti_idx,kptj_idx):
            kpti = kpts_band[ki]
            kptj = kpts[kj]
            kptij = numpy.asarray((kpti,kptj))
            for Lpq in xdf.load_Lpq(kptij):
                if Lpq.shape[1] == nao_pair:
                    Lpq = lib.unpack_tril(Lpq)
            for jpq in xdf.load_j3c(kptij):
                if jpq.shape[1] == nao_pair:
                    jpq = lib.unpack_tril(jpq)

            # Note: kj-ki for electorn 1 and ki-kj for electron 2
            # j2c ~ ({kj-ki}|{ks-kr}) ~ ({kj-ki}|-{kj-ki}) ~ ({kj-ki}|{ki-kj})
            # j3c ~ (Q|kj,ki) = j3c{ji} = (Q|ki,kj)* = conj(transpose(j3c{ij}, (0,2,1)))
            kptkl = kpti-kptj
            j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN, kptkl)

            # K ~ 'Lpq,jrs,qr->ps' + 'jpq,Lrs,qr->ps' - 'Lpq,LM,Mrs,qr->ps'
            if hermi == lib.HERMITIAN:
                jpq1 = lib.dot(j2c.T, Lpq.reshape(naux,-1), -.5).reshape(-1,nao,nao)
                jpq1 += jpq.reshape(-1,nao,nao)
                for i in range(nset):
                    dm = dms[i,kj]
                    Lpi = lib.dot(Lpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
                    v = numpy.einsum('Lpi,Lqi->pq', Lpi, jpq1.conj())
                    vk_kpts[i,ki] += v
                    vk_kpts[i,ki] += v.T.conj()
            else:
                jpq1 = lib.dot(j2c.T, Lpq.reshape(naux,-1), -1).reshape(-1,nao,nao)
                jpq1 += jpq.reshape(-1,nao,nao)
                for i in range(nset):
                    dm = dms[i,kj]
                    Lpi = lib.dot(Lpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
                    vk_kpts[i,ki] += numpy.einsum('Lpi,Lqi->pq', Lpi, jpq1.conj())
                    jpq1 = lib.dot(jpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
                    vk_kpts[i,ki] += numpy.einsum('Lpi,Lqi->pq', jpq1,
                                                  Lpq.reshape(-1,nao,nao).conj())
            Lpi = jpq1 = None

            if swap_2e and abs(kpt).sum() > 1e-9:
                # pqrs = Lpq,jrs' + 'jpq,Lrs' - 'Lpq,LM,Mrs'
                # K ~ 'pqrs,sp->rq'
                #:tmp = lib.dot(j2c.T, Lpq.reshape(naux,-1), -1).reshape(-1,nao,nao)
                #:tmp+= jpq
                #:v4 = lib.dot(Lpq.reshape(naux,-1).T, tmp.conj().reshape(naux,-1))
                #:v4+= lib.dot(jpq.reshape(naux,-1).T, Lpq.conj().reshape(naux,-1))
                #:vk_kpts[kj] += numpy.einsum('pqsr,sp->rq', v4.reshape((nao,)*4), dm)
                if hermi == lib.HERMITIAN:
                    jpq1 = lib.dot(j2c.T, Lpq.reshape(naux,-1), -.5).reshape(-1,nao,nao)
                    jpq1 += jpq.reshape(-1,nao,nao)
                    for i in range(nset):
                        dm = dms[i,ki]
                        Lip = numpy.einsum('Lpq,sp->Lsq', Lpq.reshape(-1,nao,nao), dm)
                        v = numpy.einsum('Lsq,Lsr->rq', Lip, jpq1.conj())
                        vk_kpts[i,kj] += v
                        vk_kpts[i,kj] += v.T.conj()
                else:
                    jpq1 = lib.dot(j2c.T, Lpq.reshape(naux,-1), -1).reshape(-1,nao,nao)
                    jpq1 += jpq.reshape(-1,nao,nao)
                    for i in range(nset):
                        dm = dms[i,ki]
                        Lip = numpy.einsum('Lpq,sp->Lsq', Lpq.reshape(-1,nao,nao), dm)
                        vk_kpts[i,kj] += numpy.einsum('Lsq,Lsr->rq', Lip, jpq1.conj())
                        jpq1 = numpy.einsum('jpq,sp->jsq', jpq.reshape(-1,nao,nao), dm)
                        vk_kpts[i,kj] += numpy.einsum('Lsq,Lsr->rq', jpq1,
                                                      Lpq.reshape(-1,nao,nao).conj())
            Lip = jpq1 = None

            LpqR.append(Lpq.real.reshape(naux,-1).copy())
            LpqI.append(Lpq.imag.reshape(naux,-1).copy())
            Lpq = jpq = j2c = None

        max_memory = (xdf.max_memory - lib.current_memory()[0]) * .8
        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kpt, True, xdf, xdf.gs) / cell.vol
        else:
            vkcoulG = tools.get_coulG(cell, kpt, True, vhfopt_or_mf, xdf.gs) / cell.vol
        kptjs = kpts[kptj_idx]
        p0 = 0
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        for k, kptj, pqkR, LkR, pqkI, LkI, coulG \
                in xdf.ft_loop(cell, auxcell, xdf.gs, kpt, kptjs, max_memory):
            ki = kpti_idx[k]
            kj = kptj_idx[k]
            nG = len(coulG)
            coulG = numpy.sqrt(vkcoulG[p0:p0+nG])

# case 1: k_pq = (pi|iq)
            lib.dot(LpqR[k].T, LkR, -1, pqkR, 1)
            lib.dot(LpqI[k].T, LkI,  1, pqkR, 1)
            pqkR *= coulG
            lib.dot(LpqI[k].T, LkR, -1, pqkI, 1)
            lib.dot(LpqR[k].T, LkI, -1, pqkI, 1)
            pqkI *= coulG
            rsk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            qpk = rsk.conj()
            for i in range(nset):
                qsk = lib.dot(dms[i,kj], rsk.reshape(nao,-1)).reshape(nao,nao,-1)
                vk_kpts[i,ki] += numpy.einsum('qpk,qsk->ps', qpk, qsk)
                qsk = None
            rsk = qpk = None

# case 2: k_pq = (iq|pi)
            if swap_2e and abs(kpt).sum() > 1e-9:
                srk = pqkR - pqkI*1j
                pqk = srk.reshape(nao,nao,-1).conj()
                for i in range(nset):
                    prk = lib.dot(dms[i,ki].T, srk.reshape(nao,-1)).reshape(nao,nao,-1)
                    vk_kpts[i,kj] += numpy.einsum('prk,pqk->rq', prk, pqk)
                    prk = None
                srk = pqk = None

            if k+1 == nkptj:
                p0 += nG
        LpqR = LpqI = pqkR = LkR = pqkI = LkI = coulG = None
        return None

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)

    vk_kpts *= 1./nkpts

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if dm_kpts.ndim == 3:  # One set of dm_kpts for KRHF
            return vk_kpts[0,0]
        else:
            return vk_kpts[:,0]
    else:
        return vk_kpts.reshape(dm_kpts.shape)


def get_j_kpt(xdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(xdf, cell, dm, hermi, vhfopt, kpt, kpt_band, True, False)[0]

def get_k_kpt(xdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(xdf, cell, dm, hermi, vhfopt, kpt, kpt_band, True, False)[1]

def get_jk(xdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True):
    '''JK for given k-point'''
    vj = vk = None
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(xdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)
        if with_j:
            vj = get_j_kpts(xdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)
        return vj, vk

    log = logger.Logger(xdf.stdout, xdf.verbose)
    t1 = (time.clock(), time.time())
    if xdf._cderi is None:
        xdf.build()
        t1 = log.timer('Init get_jk', *t1)
    auxcell = xdf.auxcell
    dm = numpy.asarray(dm)
    nao = dm.shape[-1]
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    kptii = numpy.asarray((kpt,kpt))
    gamma_point = abs(kpt).sum() < 1e-9

    dms = dm.reshape(-1,nao,nao)
    nset = dms.shape[0]

    for Lpq in xdf.load_Lpq(kptii):
        if Lpq.shape[1] == nao_pair:
            Lpq = lib.unpack_tril(Lpq)
    for jpq in xdf.load_j3c(kptii):
        if jpq.shape[1] == nao_pair:
            jpq = lib.unpack_tril(jpq)

    # j2c is real because for i,j,k,l, same kpt is applied
    j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN)
    if with_j:
        rho_coeff = lib.asarray([numpy.einsum('kij,ji->k', Lpq, dm) for dm in dms])
        jaux = lib.asarray([numpy.einsum('kij,ji->k', jpq, dm) for dm in dms])
        if gamma_point:
            rho_coeff = rho_coeff.real.copy()
            jaux = jaux.real.copy()
        jaux -= numpy.dot(rho_coeff, j2c.T)

    if with_k:
        vk = numpy.empty((nset,nao,nao), dtype=numpy.complex128)
        if hermi == lib.HERMITIAN:
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1), -.5).reshape(-1,nao,nao)
            tmp += jpq
            for i in range(nset):
                Lpi = lib.dot(Lpq.reshape(-1,nao), dms[i]).reshape(-1,nao,nao)
                vk[i] = numpy.einsum('Lpi,Liq->pq', Lpi, tmp)
                vk[i] += vk[i].T.conj()
        else:
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1), -1).reshape(-1,nao,nao)
            tmp += jpq
            for i in range(nset):
                Lpi = lib.dot(Lpq.reshape(-1,nao), dms[i]).reshape(-1,nao,nao)
                vk[i] = numpy.einsum('Lpi,Liq->pq', Lpi, tmp)
                tmp = lib.dot(jpq.reshape(-1,nao), dms[i]).reshape(-1,nao,nao)
                vk[i] += numpy.einsum('Lpi,Liq->pq', tmp, Lpq)
        Lpi = tmp = None
        LpqR = Lpq.real.copy()
        LpqI = Lpq.imag.copy()
    j2c = None

    if with_k:
        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, xdf, xdf.gs)
        else:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, vhfopt_or_mf, xdf.gs)
        vkcoulG *=  1./cell.vol
    vjR = numpy.zeros((nset,nao*nao))
    vjI = numpy.zeros((nset,nao*nao))
    p0 = 0
    max_memory = (xdf.max_memory - lib.current_memory()[0]) * .8
    # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
    #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
    for pqkR, LkR, pqkI, LkI, coulG \
            in xdf.pw_loop(cell, auxcell, xdf.gs, kptii, max_memory):
        if with_j:
            for i in range(nset):
                rhoR = numpy.dot(dms[i].ravel(), pqkR)
                rhoI = numpy.dot(dms[i].ravel(), pqkI)
                rhoR -= numpy.dot(rho_coeff[i], LkR)
                rhoI -= numpy.dot(rho_coeff[i], LkI)
                rhoR *= coulG
                rhoI *= coulG
                vjR[i] += numpy.dot(pqkR, rhoR)
                vjR[i] += numpy.dot(pqkI, rhoI)
                jaux[i] -= numpy.dot(LkR, rhoR)
                jaux[i] -= numpy.dot(LkI, rhoI)
                if not gamma_point:
                    vjI[i] += numpy.dot(pqkI, rhoR)
                    vjI[i] -= numpy.dot(pqkR, rhoI)
                    jaux[i] -= numpy.dot(LkI, rhoR) * 1j
                    jaux[i] += numpy.dot(LkR, rhoI) * 1j

        if with_k:
            if vhfopt_or_mf is not None:
                nG = len(coulG)
                coulG = vkcoulG[p0:p0+nG]
                p0 += nG
            coulG = numpy.sqrt(coulG)
            lib.dot(LpqR.reshape(naux,-1).T, LkR, -1, pqkR, 1)
            lib.dot(LpqI.reshape(naux,-1).T, LkI,  1, pqkR, 1)
            pqkR *= coulG
            lib.dot(LpqI.reshape(naux,-1).T, LkR, -1, pqkI, 1)
            lib.dot(LpqR.reshape(naux,-1).T, LkI, -1, pqkI, 1)
            pqkI *= coulG
            #:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
            #:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            rsk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            pqk =(pqkR+pqkI*1j).reshape(nao,nao,-1)
            for i in range(nset):
                qsk = numpy.dot(dms[i], rsk.reshape(nao,-1)).reshape(nao,nao,-1)
                vk[i] += numpy.einsum('ijG,jlG->il', pqk, qsk)
    pqkR = LkR = pqkI = LkI = coulG = None

    if with_j:
        if gamma_point:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj += numpy.dot(jaux, Lpq.reshape(naux,-1))
        vj += numpy.dot(rho_coeff, jpq.reshape(naux,-1))
        vj = vj.reshape(dm.shape)

    if with_k:
        if gamma_point:
            vk = vk.real
        vk = vk.reshape(dm.shape)
    return vj, vk


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.h = numpy.diag([L,L,L])
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
    vj = mf.get_j(cell, dm)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.69745030912447')
    vj, vk = mf.get_jk(cell, dm)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.69745030912447')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.33704732444835')
    print(numpy.einsum('ij,ji->', mf.get_hcore(cell), dm), 'ref=-75.574414055823766')
