#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Density fitting with Gaussian basis
Ref:
'''

import time
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools

def density_fit(mf, auxbasis=None, gs=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        gs : tuple
            number of grids in each (+)direction
        with_df : DF object
    '''
    from pyscf.pbc.df import df
    if with_df is None:
        if hasattr(mf, 'kpts'):
            kpts = mf.kpts
        else:
            kpts = numpy.reshape(mf.kpt, (1,3))

        with_df = df.DF(mf.cell, kpts)
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
    nao_pair = nao * (nao+1) // 2

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band)

    dmsR = dms.real.reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.reshape(nset,nkpts,nao**2)
    rhoR = numpy.zeros((nset,naux))
    rhoI = numpy.zeros((nset,naux))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j).transpose(1,0,2)
            #:rho [:,p0:p1] += numpy.einsum('Lpq,xpq->xL', Lpq, dms[:,k])
            rhoR[:,p0:p1] += numpy.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
            rhoI[:,p0:p1] += numpy.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
            if LpqI is not None:
                rhoR[:,p0:p1] -= numpy.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
                rhoI[:,p0:p1] += numpy.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            LpqR = LpqI = None
    t1 = log.timer_debug1('get_j pass 1', *t1)

    weight = 1./nkpts
    rhoR *= weight
    rhoI *= weight
    vjR = numpy.zeros((nset,nband,nao_pair))
    vjI = numpy.zeros((nset,nband,nao_pair))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI in mydf.sr_loop(kptii, max_memory, True):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:v = numpy.dot(jaux, Lpq) + numpy.dot(rho, j3c)
            #:vj_kpts[:,k] += lib.unpack_tril(v)
            vjR[:,k] += numpy.dot(rhoR[:,p0:p1], LpqR)
            if not j_real:
                vjI[:,k] += numpy.dot(rhoI[:,p0:p1], LpqR)
                if LpqI is not None:
                    vjR[:,k] -= numpy.dot(rhoI[:,p0:p1], LpqI)
                    vjI[:,k] += numpy.dot(rhoR[:,p0:p1], LpqI)
            LpqR = LpqI = None
    t1 = log.timer_debug1('get_j pass 2', *t1)

    if j_real:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = lib.unpack_tril(vj_kpts.reshape(-1,nao_pair))

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if nset == 1:  # One set of dm_kpts for KRHF
            return vj_kpts[0]
        else:
            return vj_kpts
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
    nao_pair = nao * (nao+1) // 2

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))

    nband = len(kpts_band)
    vkR = numpy.zeros((nset,nband,nao,nao))
    vkI = numpy.zeros((nset,nband,nao,nao))
    dmsR = numpy.asarray(dms.real, order='C')
    dmsI = numpy.asarray(dms.imag, order='C')

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    bufR = numpy.empty((mydf.blockdim*nao**2))
    bufI = numpy.empty((mydf.blockdim*nao**2))
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    def make_kpt(ki, kj, swap_2e):
        kpti = kpts_band[ki]
        kptj = kpts[kj]

        for LpqR, LpqI in mydf.sr_loop((kpti,kptj), max_memory, False):
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
            pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
            tmpR = numpy.ndarray((nao,nrow*nao), buffer=LpqR)
            tmpI = numpy.ndarray((nao,nrow*nao), buffer=LpqI)
            pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
            pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

            for i in range(nset):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                       1, vkR[i,kj], vkI[i,kj], 1)

            if swap_2e:
                tmpR = tmpR.reshape(nao*nrow,nao)
                tmpI = tmpI.reshape(nao*nrow,nao)
                for i in range(nset):
                    zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                           dmsR[i,kj], dmsI[i,kj], 1, tmpR, tmpI)
                    zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           1, vkR[i,ki], vkI[i,ki], 1)

    if kpt_band is None:  # normal k-points HF/DFT
        for ki in range(nkpts):
            for kj in range(ki):
                make_kpt(ki, kj, True)
            make_kpt(ki, ki, False)
    else:
        for ki in range(nband):
            for kj in range(nkpts):
                make_kpt(ki, kj, False)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j

    if exxdiv is not None:
        ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts_band)
        mydf.exxdiv = exxdiv
        fac_G0 = tools.pbc.get_coulG(cell, exx=True, mf=mydf, gs=[0]*3,
                                     Gv=numpy.zeros((1,3))) / cell.vol
        for k in range(nband):
            for i in range(nset):
                vk_kpts[i,k] += fac_G0 * reduce(numpy.dot, (ovlp[k], dms[i,k], ovlp[k]))

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
    kptii = numpy.asarray((kpt,kpt))

    if with_j:
        vjR = numpy.zeros((nset,nao**2))
        vjI = numpy.zeros((nset,nao**2))
    if with_k:
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
    dmsR = dms.real.reshape(nset,nao,nao)
    dmsI = dms.imag.reshape(nset,nao,nao)

    bufR = numpy.empty((mydf.blockdim*nao**2))
    bufI = numpy.empty((mydf.blockdim*nao**2))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    if with_j:
        vjR = vjR.reshape(nset,nao,nao)
        vjI = vjI.reshape(nset,nao,nao)
    for LpqR, LpqI in mydf.sr_loop(kptii, max_memory*.5, False):
        LpqR = LpqR.reshape(-1,nao,nao)
        if with_j:
            #:rho_coeff = numpy.einsum('Lpq,xqp->xL', Lpq, dms)
            #:vj += numpy.dot(rho_coeff, Lpq.reshape(-1,nao**2))
            rhoR  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsR)
            if not j_real:
                LpqI = LpqI.reshape(-1,nao,nao)
                rhoR -= numpy.einsum('Lpq,xpq->xL', LpqI, dmsI)
                rhoI  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsI)
                rhoI += numpy.einsum('Lpq,xpq->xL', LpqI, dmsR)
            vjR += numpy.einsum('xL,Lpq->xpq', rhoR, LpqR)
            if not j_real:
                vjR -= numpy.einsum('xL,Lpq->xpq', rhoI, LpqI)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoR, LpqI)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoI, LpqR)

        if with_k:
            #:Lpq = LpqR + LpqI*1j
            #:for i in range(nset):
            #:    tmp = numpy.dot(dms[i], Lpq.reshape(nao,-1))
            #:    vk1+= numpy.dot(Lpq.reshape(-1,nao).conj().T, tmp.reshape(-1,nao))
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
            tmpR = numpy.ndarray((nao,nrow*nao), buffer=LpqR)
            pLqR[:] = LpqR.transpose(1,0,2)
            if k_real:
                for i in range(nset):
                    lib.ddot(dmsR[i], pLqR.reshape(nao,-1), 1, tmpR)
                    lib.ddot(pLqR.reshape(-1,nao).T, tmpR.reshape(-1,nao), 1, vkR[i])
            else:
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                tmpI = numpy.ndarray((nao,nrow*nao), buffer=LpqI)
                if LpqI is None:
                    pLqI[:] = 0
                else:
                    pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)
                for i in range(nset):
                    zdotNN(dmsR[i], dmsI[i], pLqR.reshape(nao,-1),
                           pLqI.reshape(nao,-1), 1, tmpR, tmpI, 0)
                    zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                           tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                           1, vkR[i], vkI[i], 0)
        LpqR = LpqI = pLqR = pLqI = tmpR = tmpI = None

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
        if exxdiv is not None:
            ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpt)
            mydf.exxdiv = exxdiv
            fac_G0 = tools.pbc.get_coulG(cell, kpt, exx=True, mf=mydf, gs=[0]*3,
                                         Gv=numpy.zeros((1,3))) / cell.vol
            for i, dm in enumerate(dms):
                vk[i] += fac_G0 * reduce(numpy.dot, (ovlp, dm, ovlp))
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

    mf = pscf.RHF(cell)
    dm = mf.get_init_guess()
    auxbasis = 'weigend'
    #from pyscf import df
    #auxbasis = df.addons.aug_etb_for_dfbasis(cell, beta=1.5, start_at=0)
    #from pyscf.pbc.df import mdf
    #mf.with_df = mdf.MDF(cell)
    #mf.auxbasis = auxbasis
    mf = density_fit(mf, auxbasis)
    mf.with_df.gs = (5,) * 3
    vj = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv, with_k=False)[0]
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698942480902062')
    vj, vk = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698942480902062')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.348163681114187')

    kpts = cell.make_kpts([2]*3)[:4]
    from pyscf.pbc.df import DF
    with_df = DF(cell, kpts)
    with_df.auxbasis = 'weigend'
    with_df.gs = [5] * 3
    dms = numpy.array([dm]*len(kpts))
    vj, vk = with_df.get_jk(dms, exxdiv=mf.exxdiv, kpts=kpts)
    print(numpy.einsum('ij,ji->', vj[0], dms[0]) - 46.69784067248350)
    print(numpy.einsum('ij,ji->', vj[1], dms[1]) - 46.69814992718212)
    print(numpy.einsum('ij,ji->', vj[2], dms[2]) - 46.69526120279135)
    print(numpy.einsum('ij,ji->', vj[3], dms[3]) - 46.69570739526301)
    print(numpy.einsum('ij,ji->', vk[0], dms[0]) - 37.27020025046015)
    print(numpy.einsum('ij,ji->', vk[1], dms[1]) - 37.27047172558580)
    print(numpy.einsum('ij,ji->', vk[2], dms[2]) - 37.27046412080765)
    print(numpy.einsum('ij,ji->', vk[3], dms[3]) - 37.27056060718295)
