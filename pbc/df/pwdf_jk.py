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


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpt_allow = numpy.zeros(3)
    coulG = mydf.weighted_coulG(kpt_allow, False, mydf.gs)

    dmsR = dms.real.reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.reshape(nset,nkpts,nao**2)
    ngs = len(coulG)
    vR = numpy.zeros((nset,ngs))
    vI = numpy.zeros((nset,ngs))
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts, max_memory=max_memory):
        for i in range(nset):
            rhoR = numpy.dot(dmsR[i,k], pqkR)
            rhoR-= numpy.dot(dmsI[i,k], pqkI)
            rhoI = numpy.dot(dmsR[i,k], pqkI)
            rhoI+= numpy.dot(dmsI[i,k], pqkR)
            vR[i,p0:p1] += rhoR * coulG[p0:p1]
            vI[i,p0:p1] += rhoI * coulG[p0:p1]
    pqkR = pqkI = coulG = None
    weight = 1./len(kpts)
    vR *= weight
    vI *= weight

    t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    gamma_point = abs(kpts_band).sum() < 1e-9
    nband = len(kpts_band)

    vjR = numpy.zeros((nset,nband,nao*nao))
    vjI = numpy.zeros((nset,nband,nao*nao))
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts_band,
                            max_memory=max_memory):
        for i in range(nset):
            vjR[i,k] += numpy.dot(pqkR, vR[i,p0:p1])
            vjR[i,k] += numpy.dot(pqkI, vI[i,p0:p1])
        if not gamma_point:
            for i in range(nset):
                vjI[i,k] += numpy.dot(pqkI, vR[i,p0:p1])
                vjI[i,k] -= numpy.dot(pqkR, vI[i,p0:p1])
    pqkR = pqkI = coulG = None

    if gamma_point:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = vj_kpts.reshape(-1,nband,nao,nao)
    t1 = log.timer_debug1('get_j pass 2', *t1)

    if kpt_band is not None and numpy.shape(kpt_band) == (3,):
        if dm_kpts.ndim == 3:  # One set of dm_kpts for KRHF
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

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    if kpt_band is None:
        kpts_band = kpts
        swap_2e = True
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    kk_todo = numpy.ones(kk_table.shape[:2], dtype=bool)
    vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
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

        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt, True, mydf.gs)
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        for k, pqkR, pqkI, p0, p1 \
                in mydf.ft_loop(mydf.gs, kpt, kpts[kptj_idx],
                                max_memory=max_memory):
            ki = kpti_idx[k]
            kj = kptj_idx[k]
            coulG = numpy.sqrt(vkcoulG[p0:p1])

# case 1: k_pq = (pi|iq)
            pqkR *= coulG
            pqkI *= coulG
            rsk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            qpk = rsk.conj()
            for i in range(nset):
                qsk = lib.dot(dms[i,kj], rsk.reshape(nao,-1)).reshape(nao,nao,-1)
                #:vk_kpts[i,ki] += numpy.einsum('qpk,qsk->ps', qpk, qsk)
                vk_kpts[i,ki] += lib.dot(qpk.transpose(1,0,2).reshape(nao,-1),
                                         qsk.transpose(1,0,2).reshape(nao,-1).T)
                qsk = None
            rsk = qpk = None

# case 2: k_pq = (iq|pi)
            if swap_2e and abs(kpt).sum() > 1e-9:
                srk = pqkR - pqkI*1j
                pqk = srk.reshape(nao,nao,-1).conj()
                for i in range(nset):
                    prk = lib.dot(dms[i,ki].T, srk.reshape(nao,-1)).reshape(nao,nao,-1)
                    #:vk_kpts[i,kj] += numpy.einsum('prk,pqk->rq', prk, pqk)
                    vk_kpts[i,kj] += lib.dot(prk.transpose(1,0,2).reshape(nao,-1),
                                             pqk.transpose(1,0,2).reshape(nao,-1).T)
                    prk = None
                srk = pqk = None

        pqkR = pqkI = coulG = None
        return None

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)

    vk_kpts *= 1./nkpts
    if abs(kpts).sum() < 1e-9 and abs(kpts_band).sum() < 1e-9:
        vk_kpts = vk_kpts.real

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

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)

    kptii = numpy.asarray((kpt,kpt))
    kpt_allow = numpy.zeros(3)
    gamma_point = abs(kpt).sum() < 1e-9

    if with_j:
        vjcoulG = mydf.weighted_coulG(kpt_allow, False, mydf.gs)
    if with_k:
        vk = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt_allow, True, mydf.gs)

    dmsR = dms.real.reshape(nset,nao**2)
    dmsI = dms.imag.reshape(nset,nao**2)
    vjR = numpy.zeros((nset,nao**2))
    vjI = numpy.zeros((nset,nao**2))
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
    for pqkR, pqkI, p0, p1 \
            in mydf.pw_loop(mydf.gs, kptii, max_memory=max_memory):
        if with_j:
            for i in range(nset):
                rhoR = numpy.dot(dmsR[i], pqkR)
                rhoR-= numpy.dot(dmsI[i], pqkI)
                rhoI = numpy.dot(dmsR[i], pqkI)
                rhoI+= numpy.dot(dmsI[i], pqkR)
                coulG = vjcoulG[p0:p1]
                rhoR *= coulG
                rhoI *= coulG
                vjR[i] += numpy.dot(pqkR, rhoR)
                vjR[i] += numpy.dot(pqkI, rhoI)
                if not gamma_point:
                    vjI[i] += numpy.dot(pqkI, rhoR)
                    vjI[i] -= numpy.dot(pqkR, rhoI)

        if with_k:
            coulG = numpy.sqrt(vkcoulG[p0:p1])
            pqkR *= coulG
            pqkI *= coulG
            rsk =(pqkR.reshape(nao,nao,-1).transpose(1,0,2) -
                  pqkI.reshape(nao,nao,-1).transpose(1,0,2)*1j)
            pqk =(pqkR+pqkI*1j).reshape(nao,nao,-1)
            for i in range(nset):
                qsk = numpy.dot(dms[i], rsk.reshape(nao,-1)).reshape(nao,nao,-1)
                #:vk[i] += numpy.einsum('ijG,jlG->il', pqk, qsk)
                vk[i] += lib.dot(pqk.reshape(nao,-1),
                                 qsk.transpose(1,0,2).reshape(nao,-1).T)
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

def _format_dms(dm_kpts, kpts):
    nkpts = len(kpts)
    nao = dm_kpts.shape[-1]
    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    return dms


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import pwdf

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
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

