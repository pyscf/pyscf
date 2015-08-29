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
from pyscf.cc import ccsd

# t2,l2 as ijab

# default max_memory = 2000 MB
def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, max_memory=2000, verbose=logger.INFO):
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
    saved = make_intermediates(mycc, t1, t2, eris, max_memory)

    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1, t2)
    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(mycc, t1, t2, l1, l2, eris, saved,
                                   max_memory)
        normt = numpy.linalg.norm(l1new-l1) + numpy.linalg.norm(l2new-l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if mycc.diis:
            l1, l2 = mycc.diis(l1, l2, istep, normt, 0, adiis)
        log.info('istep = %d  norm(lambda1,lambda2) = %.6g', istep, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris, max_memory=2000):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]

    class _Saved(object):
        def __init__(self):
            self._tmpfile = tempfile.NamedTemporaryFile()
            self.ftmp = h5py.File(self._tmpfile.name)
        def __del__(self):
            if hasattr(self, 'ftmp'):
                self.ftmp.close()
                self._tmpfile = None
    saved = _Saved()
    saved.woooo = saved.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    saved.wooov = saved.ftmp.create_dataset('wooov', (nocc,nocc,nocc,nvir), 'f8')
    saved.wOVov = saved.ftmp.create_dataset('wOVov', (nocc,nvir,nocc,nvir), 'f8')
    saved.wOvOv = saved.ftmp.create_dataset('wOvOv', (nocc,nvir,nocc,nvir), 'f8')
    saved.wovvv = saved.ftmp.create_dataset('wovvv', (nocc,nvir,nvir,nvir), 'f8')

# As we don't have l2 in memory, hold tau temporarily in memory
    w1 = fvv - numpy.einsum('ja,jb->ba', fov, t1)
    w2 = foo + numpy.einsum('ib,jb->ij', fov, t1)
    w3 = fov.T + numpy.einsum('kc,jkbc->bj', fov, t2) * 2
    w3 -= numpy.einsum('kc,kjbc->bj', fov, t2)
    w3 += reduce(numpy.dot, (t1.T, fov, t1.T))
    w4 = fov.copy()

    _tmpfile = tempfile.NamedTemporaryFile()
    fswap = h5py.File(_tmpfile.name)

    time1 = time.clock(), time.time()
    max_memory = max_memory - lib.current_memory()[0]
    unit = max(nocc*nvir**2*4 + nvir**3*2,
               nvir**3*3 + nocc*nvir**2,
               nocc*nvir**2*7 + nocc**2*nvir + nocc**3 + nocc**2*nvir)
    blksize = max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit))
    log.debug1('ccsd lambda make_intermediates: block size = %d, nocc = %d in %d blocks',
               blksize, nocc, int((nocc+blksize-1)//blksize))
    for istep, (p0, p1) in enumerate(ccsd.prange(0, nocc, blksize)):
        eris_ovvv = ccsd._cp(eris.ovvv[p0:p1])
        eris_ovvv = ccsd.unpack_tril(eris_ovvv.reshape((p1-p0)*nvir,-1))
        eris_ovvv = eris_ovvv.reshape(p1-p0,nvir,nvir,nvir)
        w1 += numpy.einsum('jcba,jc->ba', eris_ovvv, t1[p0:p1]*2)
        w1 -= numpy.einsum('jabc,jc->ba', eris_ovvv, t1[p0:p1])
        theta = make_theta(t2[p0:p1])
        #:w3 += numpy.einsum('kdcb,kjdc->bj', eris_ovvv, theta)
        w3 += lib.dot(eris_ovvv.reshape(-1,nvir).T,
                      ccsd._cp(theta.transpose(1,0,2,3).reshape(nocc,-1)).T)
        #:wOVov = numpy.einsum('jbcd,kd->jbkc', eris_ovvv, t1)
        #:wOvOv = numpy.einsum('jdcb,kd->jbkc', eris_ovvv, -t1)
        wOVov = lib.dot(eris_ovvv.reshape(-1,nvir),
                        t1.T).reshape(-1,nvir,nvir,nocc).transpose(0,1,3,2).copy()
        wOvOv = lib.dot(ccsd._cp(eris_ovvv.transpose(0,2,3,1).reshape(-1,nvir)),
                        -t1.T).reshape(-1,nvir,nvir,nocc).transpose(0,1,3,2).copy()
        g2ovvv = eris_ovvv.transpose(0,2,3,1)*2 - eris_ovvv.transpose(0,2,1,3)
        wooov = numpy.empty((p1-p0,nocc,nocc,nvir))
        woooo = numpy.empty((p1-p0,nocc,nocc,nocc))
        eris_ovov = ccsd._cp(ccsd._cp(eris.ovov[p0:p1]).transpose(0,2,1,3))
        for j0, j1 in ccsd.prange(0, nocc, blksize):
            tau = t2[j0:j1] + numpy.einsum('ia,jb->ijab', t1[j0:j1], t1)
            #:wooov[:,j0:j1] = numpy.einsum('icbd,jkbd->ijkc', g2ovvv, tau)
            #:woooo[:,:,j0:j1] = numpy.einsum('icjd,klcd->ijkl', eris_ovov, tau)
            tmp = lib.dot(g2ovvv.reshape(-1,nvir**2), tau.reshape(-1,nvir**2).T)
            wooov[:,j0:j1] = tmp.reshape(-1,nvir,j1-j0,nocc).transpose(0,2,3,1)
            woooo[:,:,j0:j1] = lib.dot(eris_ovov.reshape(-1,nvir**2),
                                       tau.reshape(-1,nvir**2).T).reshape(-1,nocc,j1-j0,nocc)
        eris_ovov = eris_ovvv = g2ovvv = tau = tmp = None
#==== mem usage nocc*nvir**2*2 + nocc**2*nvir + nocc**3 + nvir**3*2 + nocc*nvir**2*2

        eris_ooov = ccsd._cp(eris.ooov[p0:p1])
        w2[p0:p1] += numpy.einsum('ijkb,kb->ij', eris_ooov, t1) * 2
        w2 -= numpy.einsum('kjib,kb->ij', eris_ooov, t1[p0:p1])
        w3 -= numpy.einsum('kjlc,klbc->bj', eris_ooov, theta)
        woooo += numpy.einsum('ikjc,lc->ijkl', eris_ooov, t1)
        wOvOv += numpy.einsum('jklb,lc->jbkc', eris_ooov, t1)
        wooov += eris_ooov.transpose(0,2,1,3)*2
        wooov -= eris_ooov

        eris_ovoo = ccsd._cp(eris.ovoo[p0:p1])
        woooo += numpy.einsum('icjl,kc->ijkl', eris_ovoo, t1)
        wOVov += numpy.einsum('jblk,lc->jbkc', eris_ovoo, -t1)
        #:wooov -= numpy.einsum('iblj,klbc->ijkc', eris_ovoo*1.5, t2)
        tmp_ovoo = ccsd._cp(eris_ovoo.transpose(0,3,2,1).reshape(-1,nov))
        for j in range(nocc):
            wooov[:,:,j] -= lib.dot(tmp_ovoo, t2[j].reshape(-1,nvir),
                                    1.5).reshape(-1,nocc,nvir)
        tmp_ovoo = None
        g2ooov = eris_ooov * 2 - eris_ovoo.transpose(0,3,2,1)
        for j0, j1 in ccsd.prange(0, nocc, blksize):
            theta = make_theta(t2[j0:j1])
            vijkc = numpy.einsum('iklb,jlcb->ijkc', g2ooov, theta)
            wooov[:,j0:j1,:] += vijkc
            wooov[:,:,j0:j1] -= vijkc.transpose(0,2,1,3)*.5
        eris_ooov = eris_ovoo = g2ooov = vicjk = theta = None
#==== mem usage nocc*nvir**2*3 + nocc**2*nvir + nocc**3 + nocc*nvir**2 + nocc**2*nvir*3

        eris_ovov = ccsd._cp(eris.ovov[p0:p1])
        g2ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        tmpw4 = numpy.einsum('kcld,ld->kc', g2ovov, t1)
        #:w1 -= numpy.einsum('kcja,kjcb->ba', g2ovov, t2[p0:p1])
        w1 -= lib.dot(t2[p0:p1].reshape(-1,nvir).T,
                      ccsd._cp(g2ovov.transpose(0,2,1,3).reshape(-1,nvir)))
        w1 -= numpy.einsum('ja,jb->ba', tmpw4, t1[p0:p1])
        w2[p0:p1] += numpy.einsum('ibkc,jkbc->ij', g2ovov, t2)
        w2[p0:p1] += numpy.einsum('ib,jb->ij', tmpw4, t1)
        w3 += reduce(numpy.dot, (t1[p0:p1].T, tmpw4, t1.T))
        w4[p0:p1] += tmpw4
        vOVov  = eris_ovov.copy()
        #:vOVov += numpy.einsum('jbld,klcd->jbkc', g2ovov, t2)
        #:vOVov -= numpy.einsum('jbld,kldc->jbkc', eris_ovov, t2)
        for j in range(nocc):
            tmp =(lib.dot(g2ovov.reshape(-1,nov),
                          ccsd._cp(t2[j].transpose(0,2,1).reshape(nov,-1)))
                - lib.dot(eris_ovov.reshape(-1,nov), t2[j].reshape(nov,-1)))
            vOVov[:,:,j] += tmp.reshape(-1,nvir,nvir)
        g2ovov = tmp = None
#==== mem usage nocc*nvir**2*4 + nocc**2*nvir + nocc**3 + nocc*nvir**2

        tmp = numpy.einsum('jbld,kd->ljbk', eris_ovov, t1)
        wOVov -= numpy.einsum('ljbk,lc->jbkc', tmp, t1)
        tmp = numpy.einsum('jdlb,kd->ljbk', eris_ovov, t1)
        wOvOv += numpy.einsum('ljbk,lc->jbkc', tmp, t1)

        #:vOvOv = numpy.einsum('jdlb,kldc->jbkc', eris_ovov, t2)
        ovovtmp = ccsd._cp(eris_ovov.transpose(0,3,2,1).reshape(-1,nov))
        vOvOv = numpy.empty((p1-p0,nvir,nocc,nvir))
        for j in range(nocc):
            vOvOv[:,:,j] = lib.dot(ovovtmp, t2[j].reshape(-1,nvir)).reshape(-1,nvir,nvir)
        vOvOv -= ccsd._cp(eris.oovv[p0:p1]).transpose(0,3,1,2)
        saved.wOVov[p0:p1] = wOVov + vOVov
        saved.wOvOv[p0:p1] = wOvOv + vOvOv
        ovovtmp = wOVov = wOvOv = eris_ovov = None
#==== mem usage nocc*nvir**2*7 + nocc**2*nvir + nocc**3 + nocc**2*nvir

        ov1 = vOvOv*2 + vOVov
        ov2 = vOVov*2 + vOvOv
        w3 += numpy.einsum('kcjb,kc->bj', ov2, t1[p0:p1])
        wooov -= numpy.einsum('ibkc,jb->ijkc', ov1, t1)
        wooov += numpy.einsum('ibjc,kb->ijkc', ov2, t1)
        ov1 = lib.transpose(ov1.reshape(-1,nov))
        ov2 = lib.transpose(ov2.reshape(-1,nov))
        fswap['2vOvOv/%d'%istep] = ov1.reshape(nocc,nvir,-1,nvir)
        fswap['2vovOV/%d'%istep] = ov2.reshape(nocc,nvir,-1,nvir)
        vOVov = vOvOv = None
#==== mem usage nocc*nvir**2*5 + nocc**2*nvir + nocc**3

        woooo += ccsd._cp(eris.oooo[p0:p1]).transpose(0,2,1,3)
        saved.woooo[p0:p1] = woooo
        saved.wooov[p0:p1] = wooov
        wooo = wooov = None
        time1 = log.timer_debug1('pass1 [%d:%d]'%(p0, p1), *time1)
    tau = None

    w3 += numpy.einsum('bc,jc->bj', w1, t1)
    w3 -= numpy.einsum('kj,kb->bj', w2, t1)

    for p0, p1 in ccsd.prange(0, nocc, blksize):
        eris_ooov = ccsd._cp(eris.ooov[p0:p1])
        g2ooov = eris_ooov * 2 - eris_ooov.transpose(0,2,1,3)
        #:tmp = numpy.einsum('kjla,jb->kabl', g2ooov, t1)
        #:wovvv = numpy.einsum('kabl,lc->kabc', tmp, t1)
        #:wovvv += numpy.einsum('kjla,jlbc->kabc', g2ooov, t2)
        tmp = lib.dot(ccsd._cp(g2ooov.transpose(1,0,2,3).reshape(nocc,-1)).T,
                      t1).reshape(-1,nocc,nvir,nvir).transpose(0,2,3,1)
        wovvv = lib.dot(ccsd._cp(tmp.reshape(-1,nocc)), t1).reshape(-1,nvir,nvir,nvir)
        wovvv += lib.dot(ccsd._cp(g2ooov.transpose(0,3,1,2).reshape(-1,nocc**2)),
                         t2.reshape(nocc**2,-1)).reshape(-1,nvir,nvir,nvir)
        tmp = g2ooov = None
        ov1 = numpy.empty((p1-p0,nvir,nocc,nvir))
        ov2 = numpy.empty((p1-p0,nvir,nocc,nvir))
        for istep, (j0, j1) in enumerate(ccsd.prange(0, nocc, blksize)):
            ov1[:,:,j0:j1] = fswap['2vOvOv/%d'%istep][p0:p1]
            ov2[:,:,j0:j1] = fswap['2vovOV/%d'%istep][p0:p1]
        #:wovvv += numpy.einsum('kcja,jb->kabc', ov1, t1)
        #:wovvv -= numpy.einsum('kbja,jc->kabc', ov2, t1)
        wovvv += lib.dot(ccsd._cp(ov1.transpose(0,1,3,2).reshape(-1,nocc)),
                         t1).reshape(-1,nvir,nvir,nvir).transpose(0,2,3,1)
        wovvv -= lib.dot(ccsd._cp(ov2.transpose(0,1,3,2).reshape(-1,nocc)),
                         t1).reshape(-1,nvir,nvir,nvir).transpose(0,2,1,3)
#==== mem usage nvir**3 + nocc*nvir**2*2
        eris_ooov = ov1 = ov2 = None

        for j0, j1 in ccsd.prange(0, nocc, blksize):
            eris_ovvv = ccsd._cp(eris.ovvv[j0:j1])
            eris_ovvv = ccsd.unpack_tril(eris_ovvv.reshape((j1-j0)*nvir,-1))
            eris_ovvv = eris_ovvv.reshape(j1-j0,nvir,nvir,nvir)
            if p0 == j0:
                wovvv += eris_ovvv.transpose(0,2,1,3) * 2
                wovvv -= eris_ovvv.transpose(0,2,3,1)
            #:wovvv += numpy.einsum('jabd,kjdc->kabc', eris_ovvv, t2[p0:p1,j0:j1]) * -1.5
            tmp = lib.dot(ccsd._cp(t2[p0:p1,j0:j1].transpose(0,3,1,2).reshape((p1-p0)*nvir,-1)),
                          ccsd._cp(eris_ovvv.transpose(0,3,1,2).reshape(-1,nvir**2)), -1.5)
            wovvv += tmp.reshape(-1,nvir,nvir,nvir).transpose(0,2,3,1)
            tmp = None
            g2ovvv = eris_ovvv*2 - eris_ovvv.transpose(0,3,2,1)
#==== mem usage nvir**3*3
            eris_ovvv = None
            theta =(t2[p0:p1,j0:j1].transpose(0,2,1,3)*2
                  - t2[p0:p1,j0:j1].transpose(0,3,1,2))
            #:vkbca = numpy.einsum('jdca,kbjd->kbca', g2ovvv, theta)
            vkbca = lib.dot(theta.reshape((p1-p0)*nvir,-1),
                            g2ovvv.reshape(-1,nvir*nvir)).reshape(-1,nvir,nvir,nvir)
            wovvv += vkbca.transpose(0,3,1,2)
            wovvv -= vkbca.transpose(0,3,2,1) * .5
#==== mem usage nvir**3*3 + nocc*nvir**2
            g2ovvv = theta = vkabc = None
        saved.wovvv[p0:p1] = wovvv
        time1 = log.timer_debug1('pass2 [%d:%d]'%(p0, p1), *time1)

    fswap.close()

    saved.w1 = w1
    saved.w2 = w2
    saved.w3 = w3
    saved.w4 = w4
    saved.ftmp.flush()
    return saved


# update L1, L2
def update_amps(mycc, t1, t2, l1, l2, eris=None, saved=None, max_memory=2000):
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
    theta = make_theta(t2)
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

    tmp = mycc.add_wvvVV(numpy.zeros_like(l1), l2, eris, max_memory)
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

    eris_ooov = ccsd._cp(eris.ooov)
    l1new -= numpy.einsum('jkia,kj->ia', eris_ooov, mij1) * 2
    l1new += numpy.einsum('ikja,kj->ia', eris_ooov, mij1)
    #:l2new -= numpy.einsum('ka,kijb->jiba', l1, eris_ooov)
    l2new -= lib.dot(ccsd._cp(eris_ooov.transpose(0,2,1,3).reshape(nocc,-1)).T,
                     l1).reshape(nocc,nocc,nvir,nvir)
    eris_ooov = None

    tau = ccsd.make_tau(t2, t1, t1)
    #:l2tau = numpy.einsum('ijcd,klcd->ijkl', l2, tau)
    l2tau = lib.dot(l2.reshape(nocc**2,-1),
                    tau.reshape(nocc**2,-1).T).reshape((nocc,)*4)
    tau = None
    l2t1 = numpy.einsum('ijcd,kc->ijkd', l2, t1)

    max_memory = max_memory - lib.current_memory()[0]
    unit = max(nvir**3*2+nocc*nvir**2, nocc*nvir**2*5)
    blksize = max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit))
    log.debug1('block size = %d, nocc = %d is divided into %d blocks',
               blksize, nocc, int((nocc+blksize-1)/blksize))
    for p0, p1 in ccsd.prange(0, nocc, blksize):
        eris_ovvv = ccsd._cp(eris.ovvv[p0:p1])
        eris_ovvv = ccsd.unpack_tril(eris_ovvv.reshape((p1-p0)*nvir,-1))
        eris_ovvv = eris_ovvv.reshape(p1-p0,nvir,nvir,nvir)

        l1new[p0:p1] += numpy.einsum('iabc,bc->ia', eris_ovvv, mba1) * 2
        l1new[p0:p1] -= numpy.einsum('ibca,bc->ia', eris_ovvv, mba1)
        #:l2new[p0:p1] += numpy.einsum('ic,jbac->jiba', l1, eris_ovvv)
        tmp = lib.dot(l1, eris_ovvv.reshape(-1,nvir).T)
        l2new[p0:p1] += tmp.reshape(nocc,-1,nvir,nvir).transpose(1,0,2,3)
        tmp = None
        for j0, j1 in ccsd.prange(0, nocc, blksize):
            #:m4 = numpy.einsum('ijkd,kadb->ijab', l2t1[j0:j1,:,p0:p1], eris_ovvv)
            m4 = lib.dot(ccsd._cp(l2t1[j0:j1,:,p0:p1].reshape((j1-j0)*nocc,-1)),
                         ccsd._cp(eris_ovvv.transpose(0,2,1,3).reshape(-1,nvir**2)))
            m4 = m4.reshape(-1,nocc,nvir,nvir)
            l2new[j0:j1] -= m4
            l1new[j0:j1] -= numpy.einsum('ijab,jb->ia', m4, t1) * 2
            l1new -= numpy.einsum('ijab,ia->jb', m4, t1[j0:j1]) * 2
            l1new += numpy.einsum('jiab,jb->ia', m4, t1[j0:j1])
            l1new[j0:j1] += numpy.einsum('jiab,ia->jb', m4, t1)
            m4 = None
        eris_ovvv = None
# nvir**3*2 + nocc*nvir**2

        eris_ovov = ccsd._cp(eris.ovov[p0:p1])
        l1new[p0:p1] += numpy.einsum('jb,iajb->ia', l1, eris_ovov) * 2
        l2new[p0:p1] += eris_ovov.transpose(0,2,1,3) * .5
        #:l2new[p0:p1] -= numpy.einsum('icjb,ca->ijab', eris_ovov, mba1)
        #:l2new[p0:p1] -= numpy.einsum('jbka,ik->jiba', eris_ovov, mij1)
        for j in range(p0,p1):
            tmp = lib.dot(eris_ovov[j-p0].reshape(nvir,-1).T, mba1)
            l2new[j] -= tmp.reshape(nocc,nvir,nvir).transpose(0,2,1)
            tmp = lib.dot(mij1, ccsd._cp(eris_ovov[j-p0].transpose(1,0,2).reshape(nocc,-1)))
            l2new[j] -= tmp.reshape(nocc,nvir,nvir)
        l1new[p0:p1] += numpy.einsum('iajb,jb->ia', eris_ovov, mia1) * 2
        l1new[p0:p1] -= numpy.einsum('ibja,jb->ia', eris_ovov, mia1)
        for j0, j1 in ccsd.prange(0, nocc, blksize):
            #:m4 = numpy.einsum('kalb,ijkl->ijab', eris_ovov, l2tau[j0:j1,:,p0:p1])
            m4 = lib.dot(l2tau[j0:j1,:,p0:p1].reshape((j1-j0)*nocc,-1).copy(),
                         ccsd._cp(eris_ovov.transpose(0,2,1,3).reshape(-1,nvir**2)))
            m4 = m4.reshape(-1,nocc,nvir,nvir)
            l2new[j0:j1] += m4 * .5
            l1new[j0:j1] += numpy.einsum('ijab,jb->ia', m4, t1) * 2
            l1new[j0:j1] -= numpy.einsum('ijba,jb->ia', m4, t1)
        eris_ovov = m4 = None
# nocc*nvir**2 * 3

        eris_oovv = ccsd._cp(eris.oovv[p0:p1])
        l1new[p0:p1] -= numpy.einsum('jb,ijba->ia', l1, eris_oovv)
        eris_oovv = None

        saved_wooov = ccsd._cp(saved.wooov[p0:p1])
        #:l1new[p0:p1] -= numpy.einsum('jkca,ijkc->ia', l2, saved_wooov)
        l1new[p0:p1] -= lib.dot(saved_wooov.reshape(p1-p0,-1),
                                l2.reshape(-1,nvir))
        saved_wovvv = ccsd._cp(saved.wovvv[p0:p1])
        #:l1new += numpy.einsum('kibc,kabc->ia', l2[p0:p1], saved_wovvv)
        for j in range(p0,p1):
            l1new += lib.dot(l2[j].reshape(nocc,-1),
                             saved_wovvv[j-p0].reshape(nvir,-1).T)
        saved_wooov = saved_wovvv = None
# nvir**3 + nocc**2*nvir

        saved_wOvOv = ccsd._cp(saved.wOvOv[p0:p1])
        tmp_ovov = ccsd._cp(saved.wOVov[p0:p1]) * 2 + saved_wOvOv
        tmp_ovov = ccsd._cp(lib.transpose(tmp_ovov.reshape(-1,nov)).reshape(nocc,nvir,-1,nvir))
        for j0, j1 in ccsd.prange(0, nocc, blksize):
            #:tmp = l2[j0:j1].transpose(0,2,1,3) - l2[j0:j1].transpose(0,3,1,2)*.5
            #:l2new[p0:p1] += numpy.einsum('kcia,kcjb->jiba', tmp, tmp_ovov[j0:j1])
            tmp = l2[j0:j1].transpose(0,2,1,3) - l2[j0:j1].transpose(0,3,1,2)*.5
            tmp = lib.dot(tmp_ovov[j0:j1].reshape((j1-j0)*nvir,-1).T,
                          tmp.reshape((j1-j0)*nvir,-1))
            l2new[p0:p1] += tmp.reshape(-1,nvir,nocc,nvir).transpose(0,2,1,3)
            tmp = None
        tmp_ovov = None
# nocc*nvir**2 * 5

        #:tmp = numpy.einsum('jkca,ibkc->ijab', l2, saved_wOvOv)
        tmp = numpy.empty((nocc,(p1-p0)*nvir,nvir))
        for j in range(nocc):
            tmp[j] = lib.dot(saved_wOvOv.reshape(-1,nov), l2[j].reshape(nov,-1))
        tmp = tmp.reshape(nocc,p1-p0,nvir,nvir)
        l2new[p0:p1] += tmp.transpose(1,0,3,2)
        l2new[p0:p1] += tmp.transpose(1,0,2,3) * .5
        saved_wOvOv = tmp = None

        saved_woooo = ccsd._cp(saved.woooo[p0:p1])
        #:m3 = numpy.einsum('klab,ijkl->ijab', l2, saved_woooo)
        m3 = lib.dot(saved_woooo.reshape(-1,nocc**2),
                     l2.reshape(nocc**2,-1)).reshape(-1,nocc,nvir,nvir)
        l2new[p0:p1] += m3 * .5
        l1new[p0:p1] += numpy.einsum('ijab,jb->ia', m3, t1) * 2
        l1new[p0:p1] -= numpy.einsum('ijba,jb->ia', m3, t1)
        saved_woooo = m3 = None
        time1 = log.timer_debug1('lambda pass [%d:%d]'%(p0, p1), *time1)

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    l1new /= eia
    l1new += l1

#    l2new = l2new + l2new.transpose(1,0,3,2)
#    eiajb = eia.reshape(-1,1) + eia.reshape(1,-1)
#    l2new /= eiajb.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
#    l2new += l2
    ij = 0
    for i in range(nocc):
        for j in range(i):
            dab = eia[i].reshape(-1,1) + eia[j]
            tmp = (l2new[i,j]+l2new[j,i].T) / dab + l2[i,j]
            l2new[i,j] = tmp
            l2new[j,i] = tmp.T
            ij += 1
        dab = eia[i].reshape(-1,1) + eia[i]
        l2new[i,i] = (l2new[i,i]+l2new[i,i].T)/dab + l2[i,i]
        ij += 1

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


def make_theta(g, out=None):
    nocc = g.shape[0]
    if out is None:
        theta = numpy.empty(g.shape)
    else:
        theta = out
    for i in range(nocc):
        theta[i] = g[i] * 2
        theta[i] -= g[i].transpose(0,2,1)
    return theta


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

    from pyscf.cc import ccsd_rdm
    dm1 = ccsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = ccsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    h1 = reduce(numpy.dot, (rhf.mo_coeff.T, rhf.get_hcore(), rhf.mo_coeff))
    eri = pyscf.ao2mo.full(rhf._eri, rhf.mo_coeff)
    eri = pyscf.ao2mo.restore(1, eri, nmo).reshape((nmo,)*4)
    e1 = numpy.einsum('pq,pq', h1, dm1)
    e2 = numpy.einsum('pqrs,pqrs', eri, dm2) * .5
    print e1+e2+mol.energy_nuc() - rhf.hf_energy - ecc
