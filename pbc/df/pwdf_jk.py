#!/usr/bin/env python
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
from pyscf.pbc import tools


def get_j_kpts(pwdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None, jkops=None):
    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = (time.clock(), time.time())
    dm_kpts = numpy.asarray(dm_kpts)
    nao = dm_kpts.shape[-1]
    nao_pair = nao * (nao+1) // 2
    nkpts = len(kpts)

    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    nset = dms.shape[0]

    ngs = numpy.prod(pwdf.gs*2+1)
    vR = numpy.zeros((nset,ngs))
    vI = numpy.zeros((nset,ngs))
    max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
    p0 = 0
    for k, kpt, pqkR, pqkI, coulG \
            in pwdf.ft_loop(cell, pwdf.gs, numpy.zeros(3), kpts, max_memory):
        nG = len(coulG)
        for i in range(nset):
            dmR = dms[i,k].real.ravel()
            dmI = dms[i,k].imag.ravel()
            rhoR = numpy.dot(dmR, pqkR)
            rhoR-= numpy.dot(dmI, pqkI)
            rhoI = numpy.dot(dmR, pqkI)
            rhoI+= numpy.dot(dmI, pqkR)
            vR[i,p0:p0+nG] += rhoR * coulG
            vI[i,p0:p0+nG] += rhoI * coulG
        if k+1 == nkpts:
            p0 += nG
    pqkR = pqkI = coulG = None
    weight = 1./len(kpts)
    vR *= weight
    vI *= weight

    t1 = log.timer('get_j pass 1 to compute J(G)', *t1)

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)

    vjR = numpy.zeros((nset,nband,nao*nao))
    vjI = numpy.zeros((nset,nband,nao*nao))
    p0 = 0
    for k, kpt, pqkR, pqkI, coulG \
            in pwdf.ft_loop(cell, pwdf.gs, numpy.zeros(3), kpts_band, max_memory):
        nG = len(coulG)
        for i in range(nset):
            vjR[i,k] += numpy.dot(pqkR, vR[i,p0:p0+nG])
            vjR[i,k] += numpy.dot(pqkI, vI[i,p0:p0+nG])
        if abs(kpt).sum() > 1e-9:  # if not gamma point
            for i in range(nset):
                vjI[i,k] += numpy.dot(pqkI, vR[i,p0:p0+nG])
                vjI[i,k] -= numpy.dot(pqkR, vI[i,p0:p0+nG])
        if k+1 == nband:
            p0 += nG
    pqkR = pqkI = coulG = None

    if abs(kpts_band).sum() < 1e-9:  # gamma point
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = vj_kpts.reshape(-1,nband,nao,nao)
    t1 = log.timer('get_j pass 2', *t1)

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if dm_kpts.ndim == 3:  # One set of dm_kpts for KRHF
            return vj_kpts[0,0]
        else:
            return vj_kpts[:,0]
    else:
        return vj_kpts.reshape(dm_kpts.shape)

def get_k_kpts(pwdf, cell, dm_kpts, hermi=1, vhfopt_or_mf=None,
               kpts=numpy.zeros((1,3)), kpt_band=None):
    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = (time.clock(), time.time())
    dm_kpts = numpy.asarray(dm_kpts)
    nao = dm_kpts.shape[-1]
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

    max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
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

        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kpt, True, pwdf, pwdf.gs) / cell.vol
        else:
            vkcoulG = tools.get_coulG(cell, kpt, True, vhfopt_or_mf, pwdf.gs) / cell.vol
        p0 = 0
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        for k, kptj, pqkR, pqkI, coulG \
                in pwdf.ft_loop(cell, pwdf.gs, kpt, kpts[kptj_idx], max_memory):
            ki = kpti_idx[k]
            kj = kptj_idx[k]
            nG = len(coulG)
            coulG = numpy.sqrt(vkcoulG[p0:p0+nG])

# case 1: k_pq = (pi|iq)
            pqkR *= coulG
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
        pqkR = pqkI = coulG = None
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


##################################################
#
# Single k-point
#
##################################################

def get_j_kpt(pwdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(pwdf, cell, dm, hermi, vhfopt, kpt, True, False)[0]

def get_k_kpt(pwdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
              kpt_band=None):
    return get_jk_kpt(pwdf, cell, dm, hermi, vhfopt, kpt, True, False)[1]

def get_jk(pwdf, cell, dm, hermi=1, vhfopt_or_mf=None, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True):
    '''JK for given k-point'''
    vj = vk = None
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(pwdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)
        if with_j:
            vj = get_j_kpts(pwdf, cell, [dm], hermi, vhfopt_or_mf, kpt, kpt_band)
        return vj, vk

    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = (time.clock(), time.time())
    dm = numpy.asarray(dm)
    nao = dm.shape[-1]
    nao_pair = nao * (nao+1) // 2
    kptii = numpy.asarray((kpt,kpt))
    gamma_point = abs(kpt).sum() < 1e-9

    dms = dm.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if with_k:
        vk = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        if vhfopt_or_mf is None:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, pwdf, pwdf.gs)
        else:
            vkcoulG = tools.get_coulG(cell, kpt-kpt, True, vhfopt_or_mf, pwdf.gs)
        vkcoulG *=  1./cell.vol
    vjR = numpy.zeros((nset,nao*nao))
    vjI = numpy.zeros((nset,nao*nao))
    p0 = 0
    max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
    for pqkR, pqkI, coulG in pwdf.pw_loop(cell, pwdf.gs, kptii, max_memory):
        if with_j:
            for i in range(nset):
                rhoR = numpy.dot(dms[i].ravel(), pqkR.reshape(nao**2,-1))
                rhoI = numpy.dot(dms[i].ravel(), pqkI.reshape(nao**2,-1))
                rhoR *= coulG
                rhoI *= coulG
                vjR[i] += numpy.dot(pqkR, rhoR)
                vjR[i] += numpy.dot(pqkI, rhoI)
                if not gamma_point:
                    vjI[i] += numpy.dot(pqkI, rhoR)
                    vjI[i] -= numpy.dot(pqkR, rhoI)

        if with_k:
            if vhfopt_or_mf is not None:
                nG = len(coulG)
                coulG = vkcoulG[p0:p0+nG]
                p0 += nG
            coulG = numpy.sqrt(coulG)
            pqkR *= coulG
            pqkI *= coulG
            rsk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            pqk =(pqkR+pqkI*1j).reshape(nao,nao,-1)
            for i in range(nset):
                qsk = numpy.dot(dms[i], rsk.reshape(nao,-1)).reshape(nao,nao,-1)
                vk[i] += numpy.einsum('ijG,jlG->il', pqk, qsk)
    pqkR = pqkI = coulG = None

    if with_j:
        if gamma_point:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj = vj.reshape(dm.shape)

    if with_k:
        if gamma_point:
            vk = vk.real
        vk = vk.reshape(dm.shape)
    return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import pwdf

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.h = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    df = pwdf.PWDF(cell)
    df.gs = (15,)*3
    dm = pscf.RHF(cell).get_init_guess()
    vj, vk = df.get_jk(cell, dm)
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.384051732669329')
    df.analytic_ft = True
    #print(numpy.einsum('ij,ji->', vj, dm), 'ref=5.3766911667862516')
    #print(numpy.einsum('ij,ji->', vk, dm), 'ref=8.2255177602309022')
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.447018516011319')

