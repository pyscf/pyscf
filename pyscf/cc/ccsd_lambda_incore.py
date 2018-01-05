#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
import pyscf.lib as lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.cc import _ccsd

# t2,l2 as ijab

def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2

    nocc, nvir = t1.shape
    saved = make_intermediates(mycc, t1, t2, eris)

    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1, t2)
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(mycc, t1, t2, l1, l2, eris, saved)
        normt = numpy.linalg.norm(l1new-l1) + numpy.linalg.norm(l2new-l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if mycc.diis:
            l1, l2 = mycc.diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]

    class _Saved(object):
        pass
    saved = _Saved()

# As we don't have l2 in memory, hold tau temporarily in memory
    w1 = fvv - numpy.einsum('ja,jb->ba', fov, t1)
    w2 = foo + numpy.einsum('ib,jb->ij', fov, t1)
    w3 = _cp(numpy.einsum('kc,jkbc->bj', fov, t2) * 2 + fov.T)
    w3 -= numpy.einsum('kc,kjbc->bj', fov, t2)
    w3 += reduce(numpy.dot, (t1.T, fov, t1.T))
    w4 = fov.copy()

    eris_ovvv = _cp(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nov,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)

    wovvv = numpy.empty((nocc,nvir,nvir,nvir))
    t2tmp = numpy.empty((nocc,nvir,nocc,nvir))
    for i in range(nocc):
        wovvv[i] = eris_ovvv[i].transpose(1,0,2) * 2
        t2tmp[i] = t2[i].transpose(2,0,1)
    #:wovvv += numpy.einsum('jabd,kjdc->kabc', eris_ovvv, t2) * -1.5
    tmp = lib.dot(t2tmp.reshape(nov,-1), wovvv.reshape(-1,nvir**2),
                  -1.5/2).reshape(-1,nvir,nvir,nvir)
    g2ovvv = tmp
    for i in range(nocc):
        wovvv[i] -= eris_ovvv[i].transpose(1,2,0)
        wovvv[i] += tmp[i].transpose(1,2,0)
        g2ovvv[i] = eris_ovvv[i]*2
        g2ovvv[i] -= eris_ovvv[i].transpose(1,2,0)
    tmp = t2tmp = None

    w1 += numpy.einsum('jcba,jc->ba', eris_ovvv, t1*2)
    w1 -= numpy.einsum('jabc,jc->ba', eris_ovvv, t1)
    #:w3 += numpy.einsum('kdcb,kjdc->bj', eris_ovvv, theta)
    theta = numpy.empty(t2.shape)
    for i in range(nocc):
        theta[i] = t2[i] * 2
        theta[i] -= t2[i].transpose(0,2,1)
        lib.dot(eris_ovvv[i].reshape(-1,nvir).T,
                _cp(theta[i].reshape(nocc,-1)).T, 1, w3, 1)

    theta = _cp(theta.transpose(0,2,1,3))
    #:vkbca = numpy.einsum('jdca,kbjd->kbca', g2ovvv, theta)
    vkbca = lib.dot(_cp(theta.reshape(nov,-1)),
                    g2ovvv.reshape(-1,nvir*nvir)).reshape(-1,nvir,nvir,nvir)
    for i in range(nocc):
        wovvv[i] += vkbca[i].transpose(2,0,1)
        wovvv[i] -= vkbca[i].transpose(2,1,0) * .5
    vkabc = None

    #:wOVov = numpy.einsum('jbcd,kd->jbkc', eris_ovvv, t1)
    #:wOvOv = numpy.einsum('jdcb,kd->jbkc', eris_ovvv, -t1)
    wOVov = lib.dot(eris_ovvv.reshape(-1,nvir),
                    t1.T).reshape(-1,nvir,nvir,nocc).transpose(0,1,3,2).copy()
    for i in range(nocc):
        g2ovvv[i] = eris_ovvv[i].transpose(1,2,0) * 2
    wOvOv = lib.dot(g2ovvv.reshape(-1,nvir),
                    -t1.T, .5).reshape(-1,nvir,nvir,nocc).transpose(0,1,3,2).copy()
    for i in range(nocc):
        g2ovvv[i] -= eris_ovvv[i].transpose(1,0,2)
    eris_ovov = _cp(_cp(eris.ovov).transpose(0,2,1,3))
    tau = _ccsd.make_tau(t2, t1, t1)
    #:wooov[:,j0:j1] = numpy.einsum('icbd,jkbd->ijkc', g2ovvv, tau)
    #:woooo[:,:,j0:j1] = numpy.einsum('icjd,klcd->ijkl', eris_ovov, tau)
    tmp = lib.dot(g2ovvv.reshape(-1,nvir**2), tau.reshape(-1,nvir**2).T)
    wooov = _cp(tmp.reshape(-1,nvir,nocc,nocc).transpose(0,2,3,1))
    woooo = lib.dot(eris_ovov.reshape(-1,nvir**2),
                    tau.reshape(-1,nvir**2).T).reshape(-1,nocc,nocc,nocc)
    eris_ovov = eris_ovvv = g2ovvv = tau = tmp = None

    eris_ooov = _cp(eris.ooov)
    eris_ovoo = _cp(eris.ovoo)
    #:woooo += numpy.einsum('icjl,kc->ijkl', eris_ovoo, t1)
    #:wOVov += numpy.einsum('jblk,lc->jbkc', eris_ovoo, -t1)
    for i in range(nocc):
        woooo[i] += lib.dot(t1, eris_ovoo[i].reshape(nvir,-1)).reshape((nocc,)*3).transpose(1,0,2)
    lib.dot(eris_ovoo.reshape(-1,nocc), t1, -1, wOVov.reshape(-1,nvir), 1)
    #:wooov -= numpy.einsum('ibjl,lkcb->ijkc', eris_ovoo*1.5, t2)
    t2tmp = numpy.empty((nocc,nvir,nocc,nvir))
    for i in range(nocc):
        t2tmp[i] = t2[i].transpose(2,0,1)
    tmp_ooov = _cp(-eris_ooov.transpose(2,0,1,3)).reshape(-1,nov)
    lib.dot(tmp_ooov, t2tmp.reshape(nov,-1), 1.5, wooov.reshape(-1,nov), 1)
    t2tmp = None

    g2ooov, tmp_ooov = tmp_ooov.reshape(nocc,nocc,nocc,nvir), None
    g2ooov += eris_ooov * 2
    #:vikjc = numpy.einsum('iklb,jlcb->ikjc', g2ooov, theta)
    vikjc = lib.dot(g2ooov.reshape(-1,nov), theta.reshape(-1,nov).T)
    vikjc = vikjc.reshape(nocc,nocc,nocc,nvir)
    wooov += vikjc.transpose(0,2,1,3)
    wooov -= vikjc*.5
    g2ooov = vikjc = eris_ovoo = None

    w2 += numpy.einsum('ijkb,kb->ij', eris_ooov, t1) * 2
    w2 -= numpy.einsum('kjib,kb->ij', eris_ooov, t1)
    #:w3 -= numpy.einsum('kjlc,klbc->bj', eris_ooov, theta)
    for i in range(nocc):
        lib.dot(_cp(theta[i].transpose(1,2,0)).reshape(-1,nvir).T,
                eris_ooov[i].reshape(nocc,-1).T, -1, w3, 1)
    #:woooo += numpy.einsum('ikjc,lc->ijkl', eris_ooov, t1)
    #:wOvOv += numpy.einsum('jklb,lc->jbkc', eris_ooov, t1)
    woooo += lib.dot(eris_ooov.reshape(-1,nvir),
                     t1.T).reshape((-1,nocc,nocc,nocc)).transpose(0,2,1,3)
    for i in range(nocc):
        lib.dot(_cp(eris_ooov[i].transpose(2,0,1)).reshape(-1,nocc),
                t1, 1, wOvOv[i].reshape(-1,nvir), 1)
        wooov[i] += eris_ooov[i].transpose(1,0,2)*2
        wooov[i] -= eris_ooov[i]
    eris_ooov = theta = None

    eris_ovov = _cp(eris.ovov)
    g2ovov = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        g2ovov[i] = eris_ovov[i].transpose(1,0,2)*2
        g2ovov[i] -= eris_ovov[i].transpose(1,2,0)
    tmpw4 = numpy.einsum('klcd,ld->kc', g2ovov, t1)
    #:w1 -= numpy.einsum('kcja,kjcb->ba', g2ovov, t2)
    w1 -= lib.dot(t2.reshape(-1,nvir).T, g2ovov.reshape(-1,nvir))
    w1 -= numpy.einsum('ja,jb->ba', tmpw4, t1)
    #:w2 += numpy.einsum('ibkc,jkbc->ij', g2ovov, t2)
    w2 += lib.dot(g2ovov.reshape(nocc,-1), t2.reshape(nocc,-1).T)
    w2 += numpy.einsum('ib,jb->ij', tmpw4, t1)
    w3 += reduce(numpy.dot, (t1.T, tmpw4, t1.T))
    w4 += tmpw4
    vOVov  = eris_ovov.copy()
    #:vOVov += numpy.einsum('jbld,klcd->jbkc', g2ovov, t2)
    #:vOVov -= numpy.einsum('jbld,kldc->jbkc', eris_ovov, t2)
    lib.dot(_cp(g2ovov.transpose(0,2,1,3)).reshape(-1,nov),
            _cp(t2.transpose(0,2,1,3).reshape(nov,-1).T), 1,
            vOVov.reshape(nov,-1), 1)
    lib.dot(eris_ovov.reshape(-1,nov),
            _cp(t2.transpose(0,3,1,2).reshape(nov,-1).T), -1,
            vOVov.reshape(nov,-1), 1)
    g2ovov = None

    #:tmp = numpy.einsum('jbld,kd->ljbk', eris_ovov, t1)
    #:wOVov -= numpy.einsum('ljbk,lc->jbkc', tmp, t1)
    #:tmp = numpy.einsum('jdlb,kd->ljbk', eris_ovov, t1)
    #:wOvOv += numpy.einsum('ljbk,lc->jbkc', tmp, t1)
    tmp = numpy.empty((nocc,nvir,nocc))
    for j in range(nocc):
        lib.dot(_cp(eris_ovov[j].transpose(1,0,2)).reshape(-1,nvir),
                t1.T, 1, tmp.reshape(-1,nocc))
        lib.dot(tmp.reshape(nocc,-1).T, t1, -1, wOVov[j].reshape(-1,nvir), 1)
        lib.dot(eris_ovov[j].reshape(nvir,-1).T, t1.T, 1,
                tmp.reshape(-1,nocc))
        lib.dot(tmp.reshape(nocc,-1).T, t1, 1, wOvOv[j].reshape(-1,nvir), 1)
    tmp = None

    #:vOvOv = numpy.einsum('jdlb,kldc->jbkc', eris_ovov, t2)
    ovovtmp = _cp(eris_ovov.transpose(0,3,2,1).reshape(-1,nov))
    vOvOv = numpy.empty((nocc,nvir,nocc,nvir))
    for j in range(nocc):
        lib.dot(t2[j].reshape(-1,nvir).T, ovovtmp.T, 1,
                vOvOv[j].reshape(nvir,-1))
        vOvOv[j] -= eris.oovv[j].transpose(2,0,1)
    ovovtmp = eris_ovov = None
    vOvOv = lib.transpose(vOvOv.reshape(nov,-1)).reshape(nocc,nvir,nocc,nvir)
    wOVov += vOVov
    wOvOv += vOvOv
    saved.wOVov = wOVov
    saved.wOvOv = wOvOv
    ovovtmp = wOVov = wOvOv = eris_ovov = None

    ov2 = vOVov*2 + vOvOv
    w3 += numpy.einsum('kcjb,kc->bj', ov2, t1)
    #:wooov += numpy.einsum('ibjc,kb->ijkc', ov2, t1)
    #:wovvv -= numpy.einsum('jakb,jc->kabc', ov2, t1)
    for i in range(nocc):
        wooov[i] += lib.dot(t1, ov2[i].reshape(nvir,-1)).reshape(nocc,nocc,nvir).transpose(1,0,2)
    lib.dot(_cp(ov2.transpose(0,2,1,3).reshape(nocc,-1)).T,
            t1, -1, wovvv.reshape(-1,nvir), 1)
    ov2 = None
    ov1 = vOvOv*2 + vOVov
    #:wooov -= numpy.einsum('ibkc,jb->ijkc', ov1, t1)
    #:wovvv += numpy.einsum('jakc,jb->kabc', ov1, t1)
    for i in range(nocc):
        lib.dot(t1, ov1[i].reshape(nvir,-1), -1, wooov[i].reshape(nocc,-1), 1)
    wovvv += lib.dot(_cp(ov1.reshape(nocc,-1)).T,
                     t1).reshape(nvir,-1,nvir,nvir).transpose(1,0,3,2)
    ov1 = None

    woooo += _cp(eris.oooo).transpose(0,2,1,3)
    saved.woooo = woooo
    saved.wooov = wooov
    woooo = wooov = None

    w3 += numpy.einsum('bc,jc->bj', w1, t1)
    w3 -= numpy.einsum('kj,kb->bj', w2, t1)

    eris_ooov = _cp(eris.ooov)
    g2ooov = eris_ooov * 2
    g2ooov -= eris_ooov.transpose(2,0,1,3)
    #:tmp = numpy.einsum('kjla,jb->kabl', g2ooov, t1)
    #:wovvv = numpy.einsum('kabl,lc->kabc', tmp, t1)
    #:wovvv += numpy.einsum('kjla,jlbc->kabc', g2ooov, t2)
    tmp = lib.dot(g2ooov.reshape(nocc,-1).T, t1).reshape(-1,nocc,nvir,nvir).transpose(0,2,3,1)
    lib.dot(_cp(tmp.reshape(-1,nocc)), t1, 1, wovvv.reshape(-1,nvir), 1)
    tmp = None
    lib.dot(_cp(g2ooov.transpose(0,2,1,3).reshape(nocc**2,-1)).T,
            t2.reshape(nocc**2,-1), 1, wovvv.reshape(nov,-1), 1)
    g2ooov = eris_ooov = vOVov = vOvOv = None

    saved.wovvv = wovvv
    saved.w1 = w1
    saved.w2 = w2
    saved.w3 = w3
    saved.w4 = w4
    return saved


# update L1, L2
def update_amps(mycc, t1, t2, l1, l2, eris=None, saved=None):
    if saved is None:
        saved = make_intermediates(mycc, t1, t2, eris)
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[:nocc,:nocc]

    #:mba = numpy.einsum('klca,klcb->ba', l2, t2*2-t2.transpose(0,1,3,2))
    #:mij = numpy.einsum('ikcd,jkcd->ij', l2, t2*2-t2.transpose(0,1,3,2))
    #:theta = t2*2 - t2.transpose(0,1,3,2)
    theta = _ccsd.make_0132(t2, t2, 2, -1)
    mba = lib.dot(theta.reshape(-1,nvir).T, l2.reshape(-1,nvir))
    mij = lib.dot(l2.reshape(nocc,-1), theta.reshape(nocc,-1).T)
    theta = None
    mba1 = numpy.einsum('jc,jb->bc', l1, t1) + mba
    mij1 = numpy.einsum('kb,jb->kj', l1, t1) + mij
    mia1 =(t1 + numpy.einsum('kc,jkbc->jb', l1, t2) * 2
         - numpy.einsum('kc,jkcb->jb', l1, t2)
         - reduce(numpy.dot, (t1, l1.T, t1))
         - numpy.einsum('bd,jd->jb', mba, t1)
         - numpy.einsum('lj,lb->jb', mij, t1))

    tmp = mycc.add_wvvVV(numpy.zeros_like(l1), l2, eris)
    l2new = numpy.empty((nocc,nocc,nvir,nvir))
    ij = 0
    for i in range(nocc):
        for j in range(i):
            tmp1 = tmp[ij] * .5  # *.5 because of l2+l2.transpose(1,0,3,2) later
            l2new[i,j] = tmp1
            l2new[j,i] = tmp1.T
            ij += 1
        l2new[i,i] = tmp[ij] * .5
        ij += 1
    l1new =(numpy.einsum('ijab,jb->ia', l2new, t1) * 4
          - numpy.einsum('jiab,jb->ia', l2new, t1) * 2)
    tmp = tmp1 = None

    l1new += eris.fock[:nocc,nocc:]
    l1new += numpy.einsum('ib,ba->ia', l1, saved.w1)
    l1new -= numpy.einsum('ja,ij->ia', l1, saved.w2)
    l1new -= numpy.einsum('ik,ka->ia', mij, saved.w4)
    l1new -= numpy.einsum('ca,ic->ia', mba, saved.w4)
    l1new += numpy.einsum('ijab,bj->ia', l2, saved.w3) * 2
    l1new -= numpy.einsum('ijba,bj->ia', l2, saved.w3)

    l2new += numpy.einsum('ia,jb->ijab', l1, saved.w4)
    #:l2new += numpy.einsum('jibc,ca->jiba', l2, saved.w1)
    #:l2new -= numpy.einsum('kiba,jk->jiba', l2, saved.w2)
    lib.dot(l2.reshape(-1,nvir), saved.w1, 1, l2new.reshape(-1,nvir), 1)
    lib.dot(saved.w2, l2.reshape(nocc,-1),-1, l2new.reshape(nocc,-1), 1)

    eris_ooov = _cp(eris.ooov)
    l1new -= numpy.einsum('jkia,kj->ia', eris_ooov, mij1) * 2
    l1new += numpy.einsum('ikja,kj->ia', eris_ooov, mij1)
    #:l2new -= numpy.einsum('ka,kijb->jiba', l1, eris_ooov)
    lib.dot(_cp(eris_ooov.transpose(0,2,1,3).reshape(nocc,-1)).T,
            l1, -1, l2new.reshape(-1,nvir), 1)
    eris_ooov = None

    tau = _ccsd.make_tau(t2, t1, t1)
    #:l2tau = numpy.einsum('ijcd,klcd->ijkl', l2, tau)
    l2tau = lib.dot(l2.reshape(nocc**2,-1),
                    tau.reshape(nocc**2,-1).T).reshape((nocc,)*4)
    tau = None
    l2t1 = numpy.einsum('ijcd,kc->ijkd', l2, t1)

    eris_ovvv = _cp(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nov,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)

    l1new += numpy.einsum('iabc,bc->ia', eris_ovvv, mba1) * 2
    l1new -= numpy.einsum('ibca,bc->ia', eris_ovvv, mba1)
    #:l2new += numpy.einsum('ic,jbac->jiba', l1, eris_ovvv)
    tmp = lib.dot(l1, eris_ovvv.reshape(-1,nvir).T).reshape(nocc,-1,nvir,nvir)
    for i in range(nocc):
        l2new[i] += tmp[i].transpose(0,2,1)
    #:m4 = numpy.einsum('ijkd,kadb->ijab', l2t1, eris_ovvv)
    m4 = tmp
    lib.dot(_cp(l2t1.reshape(nocc*nocc,-1)),
            _cp(eris_ovvv.transpose(0,2,1,3).reshape(-1,nvir**2)),
            1, m4.reshape(nocc*nocc,-1))
    l2new -= m4
    l1new -= numpy.einsum('ijab,jb->ia', m4, t1) * 2
    l1new -= numpy.einsum('ijab,ia->jb', m4, t1) * 2
    l1new += numpy.einsum('jiab,jb->ia', m4, t1)
    l1new += numpy.einsum('jiab,ia->jb', m4, t1)
    eris_ovvv = tmp = None

    eris_ovov = _cp(eris.ovov)
    l1new += numpy.einsum('jb,iajb->ia', l1, eris_ovov) * 2
    #:l2new -= numpy.einsum('jbic,ca->jiba', eris_ovov, mba1)
    #:l2new -= numpy.einsum('kajb,ik->ijab', eris_ovov, mij1)
    tmp = lib.dot(eris_ovov.reshape(-1,nvir), mba1).reshape(nocc,nvir,nocc,nvir)
    lib.dot(mij1, eris_ovov.reshape(nocc,-1), 1, tmp.reshape(nocc,-1), 1)
    tmp_oovv = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        tmp_oovv[i] = eris_ovov[i].transpose(1,0,2) * .5
        l2new[i] += tmp_oovv[i]
        l2new[i] -= tmp[i].transpose(1,0,2)
    tmp = None
    l1new += numpy.einsum('iajb,jb->ia', eris_ovov, mia1) * 2
    l1new -= numpy.einsum('ibja,jb->ia', eris_ovov, mia1)
    #:m4 = numpy.einsum('kalb,ijkl->ijab', eris_ovov, l2tau)
    lib.dot(l2tau.reshape(nocc*nocc,-1), tmp_oovv.reshape(-1,nvir**2),
            1, m4.reshape(nocc**2,-1))
    l2new += m4
    l1new += numpy.einsum('ijab,jb->ia', m4, t1) * 4
    l1new -= numpy.einsum('ijba,jb->ia', m4, t1) * 2
    eris_ovov = m4 = tmp_oovv = None

    eris_oovv = _cp(eris.oovv)
    l1new -= numpy.einsum('jb,ijba->ia', l1, eris_oovv)
    eris_oovv = None

    saved_wooov = _cp(saved.wooov)
    #:l1new -= numpy.einsum('jkca,ijkc->ia', l2, saved_wooov)
    l1new -= lib.dot(saved_wooov.reshape(nocc,-1), l2.reshape(-1,nvir))
    saved_wovvv = _cp(saved.wovvv)
    #:l1new += numpy.einsum('kibc,kabc->ia', l2, saved_wovvv)
    for j in range(nocc):
        l1new += lib.dot(l2[j].reshape(nocc,-1),
                         saved_wovvv[j].reshape(nvir,-1).T)
    saved_wooov = saved_wovvv = None

    saved_wOvOv = _cp(saved.wOvOv)
    tmp_ovov = _cp(saved.wOVov) * 2
    tmp_ovov += saved_wOvOv
    #:tmp = l2.transpose(0,2,1,3) - l2.transpose(0,3,1,2)*.5
    #:l2new += numpy.einsum('kcia,kcjb->jiba', tmp, tmp_ovov)
    tmp = numpy.empty((nocc,nvir,nocc,nvir))
    for i in range(nocc):
        tmp[i] = l2[i].transpose(2,0,1)*-.5
        tmp[i] += l2[i].transpose(1,0,2)
    tmp = lib.dot(tmp_ovov.reshape(-1,nov),
                  tmp.reshape(nov,-1)).reshape(-1,nvir,nocc,nvir)
    #:tmp = numpy.einsum('jkca,ibkc->ijab', l2, saved_wOvOv)
    for i in range(nocc):
        l2new[i] += tmp[i].transpose(1,0,2)
        tmp_ovov[i] = l2[i].transpose(2,0,1)
    lib.dot(saved_wOvOv.reshape(-1,nov), tmp_ovov.reshape(nov,-1),
            1, tmp.reshape(nov,-1))
    for i in range(nocc):
        l2new[i] += tmp[i].transpose(1,2,0)
        l2new[i] += tmp[i].transpose(1,0,2) * .5
    saved_wOvOv = tmp = tmp_ovov = None

    saved_woooo = _cp(saved.woooo)
    #:m3 = numpy.einsum('klab,ijkl->ijab', l2, saved_woooo)
    m3 = lib.dot(saved_woooo.reshape(-1,nocc**2),
                 l2.reshape(nocc**2,-1), .5).reshape(-1,nocc,nvir,nvir)
    l2new += m3
    l1new += numpy.einsum('ijab,jb->ia', m3, t1) * 4
    l1new -= numpy.einsum('ijba,jb->ia', m3, t1) * 2
    saved_woooo = m3 = None

    mo_e = eris.fock.diagonal()
    eia = lib.direct_sum('i-j->ij', mo_e[:nocc], mo_e[nocc:])
    l1new /= eia
    l1new += l1

#    l2new = l2new + l2new.transpose(1,0,3,2)
#    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
#    l2new += l2
    ij = 0
    for i in range(nocc):
        for j in range(i):
            dab = lib.direct_sum('a+b->ab', eia[i], eia[j])
            tmp = (l2new[i,j]+l2new[j,i].T) / dab + l2[i,j]
            l2new[i,j] = tmp
            l2new[j,i] = tmp.T
            ij += 1
        dab = lib.direct_sum('a+b->ab', eia[i], eia[i])
        l2new[i,i] = (l2new[i,i]+l2new[i,i].T)/dab + l2[i,i]
        ij += 1

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import ao2mo

    mol = gto.M()
    mf = scf.RHF(mol)

    mcc = ccsd.CCSD(mf)

    numpy.random.seed(12)
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
    eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
    fock0 = numpy.random.random((nmo,nmo))
    fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*2
    t1 = numpy.random.random((nocc,nvir))
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    l1 = numpy.random.random((nocc,nvir))
    l2 = numpy.random.random((nocc,nocc,nvir,nvir))
    l2 = l2 + l2.transpose(1,0,3,2)

    eris = lambda:None
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    idx = numpy.tril_indices(nvir)
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:][:,:,idx[0],idx[1]].copy()
    eris.vvvv = pyscf.ao2mo.restore(4,eri0[nocc:,nocc:,nocc:,nocc:],nvir)
    eris.fock = fock0

    saved = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_amps(mcc, t1, t2, l1, l2, eris, saved)
    print(abs(l1new).sum()-38172.7896467303)
    print(numpy.dot(l1new.flatten(), numpy.arange(35)) - 739312.005491083)
    print(numpy.dot(l1new.flatten(), numpy.sin(numpy.arange(35)))-7019.50937051188)
    print(numpy.dot(numpy.sin(l1new.flatten()), numpy.arange(35))-69.6652346635955)

    print(abs(l2new).sum()-72035.4931071527)
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())
    print(numpy.dot(l2new.flatten(), numpy.arange(35**2)) - 48427109.5409886)
    print(numpy.dot(l2new.flatten(), numpy.sin(numpy.arange(35**2)))-137.758016736487)
    print(numpy.dot(numpy.sin(l2new.flatten()), numpy.arange(35**2))-507.656936701192)


    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-16
    rhf.scf()

    mcc = ccsd.CCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()

    nmo = rhf.mo_energy.size
    fock0 = numpy.diag(rhf.mo_energy)
    nocc = mol.nelectron // 2
    nvir = nmo - nocc

    eris = mcc.ao2mo()
    conv, l1, l2 = kernel(mcc, eris, t1, t2, tol=1e-8)
    print(numpy.linalg.norm(l1)-0.0132626841292)
    print(numpy.linalg.norm(l2)-0.212575609057)

    import ccsd_rdm
    dm1 = ccsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = ccsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    h1 = reduce(numpy.dot, (rhf.mo_coeff.T, rhf.get_hcore(), rhf.mo_coeff))
    eri = pyscf.ao2mo.full(rhf._eri, rhf.mo_coeff)
    eri = pyscf.ao2mo.restore(1, eri, nmo).reshape((nmo,)*4)
    e1 = numpy.einsum('pq,pq', h1, dm1)
    e2 = numpy.einsum('pqrs,pqrs', eri, dm2) * .5
    print(e1+e2+mol.energy_nuc() - rhf.e_tot - ecc)
