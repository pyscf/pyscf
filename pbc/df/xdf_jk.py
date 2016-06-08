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
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    max_memory = (xdf.max_memory - lib.current_memory()[0]) * .8
    rho_tot = 0
    jaux = 0
    ngs = numpy.prod(xdf.gs*2+1)
    vR = numpy.zeros(ngs)
    vI = numpy.zeros(ngs)
    for k, kpt in enumerate(kpts):
        kptii = numpy.asarray((kpt,kpt))
        for Lpq in xdf.load_Lpq(kptii):
            if Lpq.shape[1] == nao_pair:
                Lpq = lib.unpack_tril(Lpq)
        for jpq in xdf.load_j3c(kptii):
            if jpq.shape[1] == nao_pair:
                jpq = lib.unpack_tril(jpq)

        dm = dm_kpts[k]
        jaux += numpy.einsum('kij,ji->k', jpq, dm).real
        rho_coeff = numpy.einsum('kij,ji->k', Lpq, dm).real
        rho_tot += rho_coeff
        Lpq = jpq = None

        p0 = 0
        # contract dm to rho_rs(-G+k_rs)
        # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
        #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        for pqkR, LkR, pqkI, LkI, coulG \
                in xdf.pw_loop(cell, auxcell, xdf.gs, kptii, max_memory):
            nG = len(coulG)
            rhoR = numpy.dot(dm.ravel(), pqkR.reshape(nao**2,-1)) - numpy.dot(rho_coeff, LkR)
            rhoI = numpy.dot(dm.ravel(), pqkI.reshape(nao**2,-1)) - numpy.dot(rho_coeff, LkI)
            vR[p0:p0+nG] += rhoR * coulG
            vI[p0:p0+nG] += rhoI * coulG
            p0 += nG
        pqkR = LkR = pqkI = LkI = coulG = None

    j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN)
    jaux -= j2c.dot(rho_tot)
    j2c = None
    t1 = log.timer('get_j pass 1 to compute J(G)', *t1)

    if kpt_band is None:
        kpts_lst = kpts
    else:
        kpts_lst = numpy.reshape(kpt_band, (-1,3))

    vj_kpts = []
    for k, kpt in enumerate(kpts_lst):
        kptii = numpy.asarray((kpt,kpt))
        vjR = 0
        vjI = 0
        p0 = 0
        for pqkR, LkR, pqkI, LkI, coulG \
                in xdf.pw_loop(cell, auxcell, xdf.gs, kptii, max_memory):
            nG = len(coulG)
            vjR += numpy.dot(pqkR, vR[p0:p0+nG])
            vjR += numpy.dot(pqkI, vI[p0:p0+nG])
            if abs(kpt).sum() > 1e-9:  # if not gamma point
                vjI += numpy.dot(pqkI, vR[p0:p0+nG])
                vjI -= numpy.dot(pqkR, vI[p0:p0+nG])
            if k == 0:  # Construct jaux once, it is the same for all kpts
                jaux -= numpy.dot(LkR, vR[p0:p0+nG])
                jaux -= numpy.dot(LkI, vI[p0:p0+nG])
            p0 += nG
        pqkR = LkR = pqkI = LkI = coulG = None

        for Lpq in xdf.load_Lpq(kptii):
            if Lpq.shape[1] == nao_pair:
                Lpq = lib.unpack_tril(Lpq).rashape(naux,-1)
        for jpq in xdf.load_j3c(kptii):
            if jpq.shape[1] == nao_pair:
                jpq = lib.unpack_tril(jpq).rashape(naux,-1)

        if abs(kpt).sum() < 1e-9:  # gamma point
            vjR += numpy.dot(Lpq.T, jaux)
            vjR += numpy.dot(jpq.T, rho_tot)
            vj_kpts.append(vjR.reshape(nao,nao))
        else:
            vj = vjR + vjI * 1j
            vj += numpy.dot(Lpq.T, jaux)
            vj += numpy.dot(jpq.T, rho_tot)
            vj_kpts.append(vj.reshape(nao,nao))
    t1 = log.timer('get_j pass 2', *t1)

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        return vj_kpts[0]
    else:
        return vj_kpts

def get_k_kpts(xdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None):
    log = logger.Logger(xdf.stdout, xdf.verbose)
    t1 = (time.clock(), time.time())
    if xdf._cderi is None:
        xdf.build()
        t1 = log.timer('Init get_k_kpts', *t1)
    auxcell = xdf.auxcell
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    max_memory = (xdf.max_memory - lib.current_memory()[0]) * .8
    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    def make_k1k2(kpti, kptj, dm):
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
        j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN, kpti-kptj)
        Lpi = lib.dot(Lpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
        if hermi == lib.HERMITIAN:
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1), -.5).reshape(-1,nao,nao)
            tmp += jpq
            vk = numpy.einsum('Lpi,Lqi->pq', Lpi, tmp.conj())
            vk += vk.T.conj()
        else:
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1), -1).reshape(-1,nao,nao)
            tmp += jpq
            vk = numpy.einsum('Lpi,Lqi->pq', Lpi, tmp.conj())
            tmp = lib.dot(jpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
            vk += numpy.einsum('Lpi,Lqi->pq', tmp, Lpq.conj())
        jpq = tmp = j2c = None
        LpqR = Lpq.real.copy()
        LpqI = Lpq.imag.copy()

        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kptj-kpti, True, xdf,
                                      xdf.gs) / cell.vol
        else:
            vkcoulG = tools.get_coulG(cell, kptj-kpti, True, vhfopt_or_mf,
                                      xdf.gs) / cell.vol
        vjR = 0
        vjI = 0
        p0 = 0
        # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
        #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        for pqkR, LkR, pqkI, LkI, coulG \
                in xdf.pw_loop(cell, auxcell, xdf.gs, kptij, max_memory):
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
            pqk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            #:v4 = numpy.einsum('jiL,klL->ijkl', pqk.conj(), pqk)
            #:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            rqk = lib.dot(dm, pqk.reshape(nao,-1)).reshape(nao**2,-1)
            pqk = pqk.conj()
            if gamma_point:
                vk += numpy.einsum('jiL,jlL->il', pqk, rqk).real
            else:
                vk += numpy.einsum('jiL,jlL->il', pqk, rqk)
        pqkR = LkR = pqkI = LkI = coulG = None
        return vk

    if kpt_band is None:
        kpts_lst = kpts
    else:
        kpts_lst = numpy.reshape(kpt_band, (-1,3))

    vk_kpts = [0] * len(kpts_lst)
    for k1, kpti in enumerate(kpts_lst):
        for k2, kptj in enumerate(kpts):
            vk_kpts[k1] += make_k1k2(kpti, kptj, dm_kpts[k2])

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        return vk_kpts[0]
    else:
        return vk_kpts

def get_j_kpt(xdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(xdf, cell, dm, hermi, vhfopt, kpt, kpt_band, True, False)[0]

def get_k_kpt(xdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(xdf, cell, dm, hermi, vhfopt, kpt, kpt_band, True, False)[1]

def get_jk(xdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True):
    '''JK for given k-point'''
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        vj = vk = None
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(xdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)[0]
        if with_j:
            vj = get_j_kpts(xdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)[0]
        return vj, vk

    log = logger.Logger(xdf.stdout, xdf.verbose)
    t1 = (time.clock(), time.time())
    if xdf._cderi is None:
        xdf.build()
        t1 = log.timer('Init get_jk', *t1)
    auxcell = xdf.auxcell
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    kptii = numpy.asarray((kpt,kpt))
    gamma_point = abs(kpt).sum() < 1e-9
    vj = vk = None

    for Lpq in xdf.load_Lpq(kptii):
        if Lpq.shape[1] == nao_pair:
            Lpq = lib.unpack_tril(Lpq)
    for jpq in xdf.load_j3c(kptii):
        if jpq.shape[1] == nao_pair:
            jpq = lib.unpack_tril(jpq)

    # j2c is real because for i,j,k,l, same kpt is applied
    j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN)
    if with_j:
        rho_coeff = numpy.einsum('kij,ji->k', Lpq, dm)
        jaux = numpy.einsum('kij,ji->k', jpq, dm)
        if gamma_point:
            rho_coeff = rho_coeff.real.copy()
            jaux = jaux.real.copy()
        jaux -= j2c.dot(rho_coeff)

    if with_k:
        Lpi = lib.dot(Lpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
        if hermi == lib.HERMITIAN:
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1), -.5).reshape(-1,nao,nao)
            tmp += jpq
            vk = numpy.einsum('Lpi,Liq->pq', Lpi, tmp)
            vk += vk.T.conj()
        else:
            tmp = lib.dot(j2c, Lpq.reshape(naux,-1), -1).reshape(-1,nao,nao)
            tmp += jpq
            vk = numpy.einsum('Lpi,Liq->pq', Lpi, tmp)
            tmp = lib.dot(jpq.reshape(-1,nao), dm).reshape(-1,nao,nao)
            vk += numpy.einsum('Lpi,Liq->pq', tmp, Lpq)
        Lpi = tmp = None
        LpqR = Lpq.real.copy()
        LpqI = Lpq.imag.copy()
    j2c = None

    if with_k:
        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, xdf,
                                      xdf.gs) / cell.vol
        else:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, vhfopt_or_mf,
                                      xdf.gs) / cell.vol
    vjR = 0
    vjI = 0
    p0 = 0
    max_memory = (xdf.max_memory - lib.current_memory()[0]) * .8
    # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
    #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
    for pqkR, LkR, pqkI, LkI, coulG \
            in xdf.pw_loop(cell, auxcell, xdf.gs, kptii, max_memory):
        if with_j:
            rhoR = numpy.dot(dm.ravel(), pqkR.reshape(nao**2,-1)) - numpy.dot(rho_coeff, LkR)
            rhoI = numpy.dot(dm.ravel(), pqkI.reshape(nao**2,-1)) - numpy.dot(rho_coeff, LkI)
            rhoR *= coulG
            rhoI *= coulG
            vjR += numpy.dot(pqkR, rhoR)
            vjR += numpy.dot(pqkI, rhoI)
            jaux -= numpy.dot(LkR, rhoR)
            jaux -= numpy.dot(LkI, rhoI)
            if not gamma_point:
                vjI += numpy.dot(pqkI, rhoR)
                vjI -= numpy.dot(pqkR, rhoI)
                jaux -= numpy.dot(LkI, rhoR) * 1j
                jaux += numpy.dot(LkR, rhoI) * 1j

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
            pqk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            #:v4 = numpy.einsum('jiL,klL->ijkl', pqk.conj(), pqk)
            #:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            rqk = lib.dot(dm, pqk.reshape(nao,-1)).reshape(nao,nao,-1)
            pqk =(pqkR+pqkI*1j).reshape(nao,nao,-1)
            if gamma_point:
                vk += numpy.einsum('jiG,jlG->il', pqk, rqk).real
            else:
                vk += numpy.einsum('jiG,jlG->il', pqk, rqk)
    pqkR = LkR = pqkI = LkI = coulG = None

    if with_j:
        if gamma_point:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj += numpy.dot(Lpq.reshape(naux,-1).T, jaux)
        vj += numpy.dot(jpq.reshape(naux,-1).T, rho_coeff)
        vj = vj.reshape(nao,nao)
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
