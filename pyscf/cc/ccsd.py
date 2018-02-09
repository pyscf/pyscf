#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc import _ccsd
_dgemm = lib.numpy_helper._dgemm

BLKMIN = 4

# t2 as ijab

def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=logger.INFO):
    log = logger.new_logger(mycc, verbose)

    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)[1:]
    elif t1 is None:
        nocc, nvir = t2.shape[0], t2.shape[2]
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if mycc.diis:
            t1, t2 = mycc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(mycc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc*nvir
    fock = eris.fock

    t1t2new = numpy.zeros((nov+nov**2))
    t1new = t1t2new[:nov].reshape(t1.shape)
    t2new = t1t2new[nov:].reshape(t2.shape)
    t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
    mycc.add_wvvVV_(t1, t2, eris, t2new_tril)
    idxo = numpy.tril_indices(nocc)
    lib.takebak_2d(t2new.reshape(nocc**2,nvir**2),
                   t2new_tril.reshape(nocc*(nocc+1)//2,nvir**2),
                   idxo[0]*nocc+idxo[1], numpy.arange(nvir**2))
    idxo = numpy.arange(nocc)
    t2new[idxo,idxo] *= .5
    t2new_tril = None
    time1 = log.timer_debug1('vvvv', *time0)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc].copy()
    foo[range(nocc),range(nocc)] = 0
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:].copy()
    fvv[range(nvir),range(nvir)] = 0
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    #: woooo = numpy.einsum('la,ikja->ikjl', t1, eris.ooov)
    eris_ooov = _cp(eris.ooov)
    foo += numpy.einsum('kc,jikc->ij', 2*t1, eris_ooov)
    foo += numpy.einsum('kc,jkic->ij',  -t1, eris_ooov)
    woooo = lib.ddot(eris_ooov.reshape(nocc**3,nvir), t1.T).reshape((nocc,)*4)
    woooo = lib.transpose_sum(woooo.reshape(nocc**2,nocc**2), inplace=True)
    woooo += _cp(eris.oooo).reshape(nocc**2,nocc**2)
    woooo = _cp(woooo.reshape(nocc,nocc,nocc,nocc).transpose(0,2,1,3))
    eris_ooov = None
    time1 = log.timer_debug1('woooo', *time1)

    unit = _memory_usage_inloop(nocc, nvir) + 1
    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nocc, max(BLKMIN, int(max_memory/unit)))
    blknvir = int((max_memory*.9e6/8-blksize*nov**2*6)/(blksize*nvir**2*2+1))
    blknvir = min(nvir, max(BLKMIN, blknvir))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d,%d',
               max_memory, nocc, nvir, blksize, blknvir)
    nvir_pair = nvir * (nvir+1) // 2
    def load_ovvv(p0, p1, q0, q1, buf):
        if q1 != nvir:
            q0, q1 = q1, min(nvir, q1+blknvir)
            readbuf = numpy.ndarray((p1-p0,q1-q0,nvir_pair), buffer=buf)
            readbuf[:] = eris.ovvv[p0:p1,q0:q1]
    def load_eri(eri, p0, p1, buf):
        buf[:] = eri[p0:p1]

    buflen = max(nov**2, nocc**3)
    bufs = numpy.empty((5,blksize*buflen))
    buf1, buf2, buf3, buf4, buf5 = bufs
    for p0, p1 in lib.prange(0, nocc, blksize):
    #: wOoVv += numpy.einsum('iabc,jc->ijab', eris.ovvv, t1)
    #: wOoVv -= numpy.einsum('jbik,ka->jiba', eris.ovoo, t1)
        wOoVv = numpy.ndarray((nocc,p1-p0,nvir,nvir), buffer=buf3)
        wooVV = numpy.ndarray((p1-p0,nocc,nvir,nvir), buffer=buf4)
        wooVV[:] = 0
        readbuf = numpy.empty((p1-p0,blknvir,nvir_pair))
        prefetchbuf = numpy.empty((p1-p0,blknvir,nvir_pair))
        ovvvbuf = numpy.empty((p1-p0,blknvir,nvir,nvir))
        with lib.call_in_background(load_ovvv) as prefetch_ovvv:
            prefetchbuf[:] = eris.ovvv[p0:p1,0:min(nvir,blknvir)]
            for q0, q1 in lib.prange(0, nvir, blknvir):
                readbuf, prefetchbuf = prefetchbuf, readbuf
                prefetch_ovvv(p0, p1, q0, q1, prefetchbuf)
                eris_ovvv = numpy.ndarray(((p1-p0)*(q1-q0),nvir_pair), buffer=readbuf)
                #:eris_ovvv = _cp(eris.ovvv[p0:p1,q0:q1])
                eris_ovvv = lib.unpack_tril(eris_ovvv, out=ovvvbuf)
                eris_ovvv = eris_ovvv.reshape(p1-p0,q1-q0,nvir,nvir)

                #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
                #: tmp = numpy.einsum('ijcd,kcdb->ijbk', tau, eris.ovvv)
                #: t2new += numpy.einsum('ka,ijbk->ijab', -t1, tmp)
                if not mycc.direct:
                    eris_vovv = lib.transpose(eris_ovvv.reshape(-1,nvir))
                    eris_vovv = eris_vovv.reshape(nvir*(p1-p0),-1)
                    tmp = numpy.ndarray((nocc,nocc,nvir,p1-p0), buffer=buf1)
                    for j0, j1 in lib.prange(0, nocc, blksize):
                        tau = numpy.ndarray((j1-j0,nocc,q1-q0,nvir), buffer=buf2)
                        tau = numpy.einsum('ia,jb->ijab', t1[j0:j1,q0:q1], t1, out=tau)
                        tau += t2[j0:j1,:,q0:q1]
                        lib.ddot(tau.reshape((j1-j0)*nocc,-1), eris_vovv.T, 1,
                                 tmp[j0:j1].reshape((j1-j0)*nocc,-1), 0)
                    tmp1 = numpy.ndarray((nocc,nocc,nvir,p1-p0), buffer=buf2)
                    tmp1[:] = tmp.transpose(1,0,2,3)
                    lib.ddot(tmp1.reshape(-1,p1-p0), t1[p0:p1], -1, t2new.reshape(-1,nvir), 1)
                    eris_vovv = tau = tmp1 = tmp = None

                fvv += numpy.einsum('kc,kcba->ab', 2*t1[p0:p1,q0:q1], eris_ovvv)
                fvv[:,q0:q1] += numpy.einsum('kc,kbca->ab', -t1[p0:p1], eris_ovvv)

                #: wooVV -= numpy.einsum('jc,icba->ijba', t1, eris_ovvv)
                tmp = t1[:,q0:q1].copy()
                for i in range(eris_ovvv.shape[0]):
                    lib.ddot(tmp, eris_ovvv[i].reshape(q1-q0,-1), -1,
                             wooVV[i].reshape(nocc,-1), 1)

                #: wOoVv += numpy.einsum('ibac,jc->jiba', eris_ovvv, t1)
                tmp = numpy.ndarray((nocc,p1-p0,q1-q0,nvir), buffer=buf1)
                lib.ddot(t1, eris_ovvv.reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1))
                wOoVv[:,:,q0:q1] = tmp

                #: theta = t2.transpose(1,0,2,3) * 2 - t2
                #: t1new += numpy.einsum('ijcb,jcba->ia', theta, eris.ovvv)
                theta = tmp
                theta[:] = t2[p0:p1,:,q0:q1,:].transpose(1,0,2,3)
                theta *= 2
                theta -= t2[:,p0:p1,q0:q1,:]
                lib.ddot(theta.reshape(nocc,-1), eris_ovvv.reshape(-1,nvir), 1, t1new, 1)
                theta = tmp = None
        readbuf = prefetchbuf = ovvvbuf = eris_ovvv = None
        time2 = log.timer_debug1('ovvv [%d:%d]'%(p0, p1), *time1)

        tmp = numpy.ndarray((nocc,p1-p0,nvir,nocc), buffer=buf1)
        tmp[:] = _cp(eris.ovoo[p0:p1]).transpose(2,0,1,3)
        lib.ddot(tmp.reshape(nocc*(p1-p0)*nvir,nocc), t1, -1,
                             wOoVv.reshape(nocc*(p1-p0)*nvir,nvir), 1)

        eris_ooov = _cp(eris.ooov[p0:p1])
        eris_oovv = numpy.empty((p1-p0,nocc,nvir,nvir))
        with lib.call_in_background(load_eri) as prefetch:
            prefetch(eris.oovv, p0, p1, eris_oovv)
            tmp = numpy.ndarray((p1-p0,nocc,nvir,nocc), buffer=buf1)
            tmp[:] = eris_ooov.transpose(0,1,3,2)
            #: wooVV = numpy.einsum('ka,ijkb->ijba', t1, eris.ooov[p0:p1])
            lib.ddot(tmp.reshape((p1-p0)*nov,nocc), t1, 1,
                     wooVV.reshape((p1-p0)*nov,nvir), 1)
            t2new[p0:p1] += wOoVv.transpose(1,0,2,3)

        eris_ovov = numpy.empty((p1-p0,nvir,nocc,nvir))
        with lib.call_in_background(load_eri) as prefetch:
            prefetch(eris.ovov, p0, p1, eris_ovov)
            t1new[p0:p1] += numpy.einsum('jb,ijba->ia',  -t1, eris_oovv)
            wooVV -= eris_oovv

            #tmp = numpy.einsum('ic,jkbc->jikb', t1, eris_oovv)
            #t2new[p0:p1] += numpy.einsum('ka,jikb->ijba', -t1, tmp)
            tmp1 = numpy.ndarray((nocc,nov), buffer=buf1)
            tmp2 = numpy.ndarray((nov,nocc), buffer=buf2)
            for j in range(p1-p0):
                tmp = lib.ddot(t1, eris_oovv[j].reshape(nov,nvir).T, 1, tmp1)
                lib.transpose(_cp(tmp).reshape(nocc,nocc,nvir), axes=(0,2,1), out=tmp2)
                t2new[:,p0+j] -= lib.ddot(tmp2, t1).reshape(nocc,nvir,nvir)
        eris_oovv = None

        for i in range(p1-p0):
            t2new[p0+i] += eris_ovov[i].transpose(1,0,2) * .5
        t1new[p0:p1] += numpy.einsum('jb,iajb->ia', 2*t1, eris_ovov)
        #:tmp = numpy.einsum('ic,jbkc->jibk', t1, eris_ovov)
        #:t2new[p0:p1] += numpy.einsum('ka,jibk->jiba', -t1, tmp)
        for j in range(p1-p0):
            lib.ddot(t1, eris_ovov[j].reshape(nov,nvir).T, 1, tmp1)
            lib.ddot(tmp1.reshape(nov,nocc), t1, -1,
                     t2new[p0+j].reshape(nov,nvir), 1)
        tmp1 = tmp2 = tmp = None

        fov[p0:p1] += numpy.einsum('kc,iakc->ia', t1, eris_ovov) * 2
        fov[p0:p1] -= numpy.einsum('kc,icka->ia', t1, eris_ovov)

    #: fvv -= numpy.einsum('ijca,ibjc->ab', theta, eris.ovov)
    #: foo += numpy.einsum('iakb,jkba->ij', eris.ovov, theta)
        tau = numpy.ndarray((nocc,nvir,nvir), buffer=buf1)
        theta = numpy.ndarray((nocc,nvir,nvir), buffer=buf2)
        for i in range(p1-p0):
            tau = numpy.einsum('a,jb->jab', t1[p0+i]*.5, t1, out=tau)
            tau += t2[p0+i]
            theta = lib.transpose(tau, axes=(0,2,1), out=theta)
            theta *= 2
            theta -= tau
            vov = lib.transpose(eris_ovov[i].reshape(nvir,nov), out=tau)
            lib.ddot(vov.reshape(nocc,nvir**2), theta.reshape(nocc,nvir**2).T, 1, foo, 1)
            lib.ddot(theta.reshape(nov,nvir).T, eris_ovov[i].reshape(nvir,nov).T, -1, fvv, 1)
        tau = theta = vov = None

    #: theta = t2.transpose(0,2,1,3) * 2 - t2.transpose(0,3,2,1)
    #: t1new += numpy.einsum('jb,ijba->ia', fov, theta)
    #: t1new -= numpy.einsum('kijb,kjba->ia', eris_ooov, theta)
        theta = numpy.ndarray((p1-p0,nvir,nocc,nvir), buffer=buf1)
        for i in range(p1-p0):
            tmp = t2[p0+i].transpose(0,2,1) * 2
            tmp-= t2[p0+i]
            lib.ddot(eris_ooov[i].reshape(nocc,nov),
                     tmp.reshape(nov,nvir), -1, t1new, 1)
            lib.transpose(_cp(tmp).reshape(nov,nvir), out=theta[i])  # theta[i] = tmp.transpose(2,0,1)
        t1new += numpy.einsum('jb,jbia->ia', fov[p0:p1], theta)
        eris_ooov = None

    #: wOVov += eris.ovov
    #: tau = theta - numpy.einsum('ic,kb->ikcb', t1, t1*2)
    #: wOVov += .5 * numpy.einsum('jakc,ikcb->jiba', eris.ovov, tau)
    #: wOVov -= .5 * numpy.einsum('jcka,ikcb->jiba', eris.ovov, t2)
    #: t2new += numpy.einsum('ikca,kjbc->ijba', theta, wOVov)
        for i in range(p1-p0):
            wOoVv[:,i] += wooVV[i]*.5  #: jiba + ijba*.5
        wOVov = lib.transpose(wOoVv.reshape(nocc,(p1-p0)*nvir,nvir), axes=(0,2,1), out=buf5)
        wOVov = wOVov.reshape(nocc,nvir,p1-p0,nvir)
        eris_OVov = lib.transpose(eris_ovov.reshape((p1-p0)*nvir,nov), out=buf3)
        eris_OVov = eris_OVov.reshape(nocc,nvir,p1-p0,nvir)
        wOVov += eris_OVov
        theta = theta.reshape((p1-p0)*nvir,nov)
        for i in range(nocc):  # OVov-OVov.transpose(0,3,2,1)*.5
            eris_OVov[i] -= eris_OVov[i].transpose(2,1,0)*.5
        for j0, j1 in lib.prange(0, nocc, blksize):
            tau = numpy.ndarray((j1-j0,nvir,nocc,nvir), buffer=buf2)
            for i in range(j1-j0):
                tau[i]  = t2[j0+i].transpose(1,0,2) * 2
                tau[i] -= t2[j0+i].transpose(2,0,1)
                tau[i] -= numpy.einsum('a,jb->bja', t1[j0+i]*2, t1)
            #: wOVov[j0:j1] += .5 * numpy.einsum('iakc,jbkc->jbai', eris_ovov, tau)
            lib.ddot(tau.reshape((j1-j0)*nvir,nov), eris_OVov.reshape(nov,(p1-p0)*nvir),
                     .5, wOVov[j0:j1].reshape((j1-j0)*nvir,(p1-p0)*nvir), 1)

            #theta = t2[p0:p1] * 2 - t2[p0:p1].transpose(0,1,3,2)
            #: t2new[j0:j1] += numpy.einsum('iack,jbck->jiba', theta, wOVov[j0:j1])
            tmp = lib.ddot(wOVov[j0:j1].reshape((j1-j0)*nvir,(p1-p0)*nvir), theta, 1,
                           tau.reshape((j1-j0)*nvir,nov)).reshape(j1-j0,nvir,nocc,nvir)
            for i in range(j1-j0):
                t2new[j0+i] += tmp[i].transpose(1,0,2)
        theta = wOoVv = wOVov = eris_OVov = tmp = tau = None
        time2 = log.timer_debug1('wOVov [%d:%d]'%(p0, p1), *time2)

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: woooo += numpy.einsum('ijba,klab->ijkl', eris.oOVv, tau)
    #: tau = .5*t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: woVoV += numpy.einsum('jkca,ikbc->ijba', tau, eris.oOVv)
        tmp = numpy.ndarray((p1-p0,nvir,nocc,nvir), buffer=buf1)
        tmp[:] = wooVV.transpose(0,2,1,3)
        woVoV = lib.transpose(_cp(tmp).reshape((p1-p0)*nvir,nov),
                              out=buf4).reshape(nocc,nvir,p1-p0,nvir)
        eris_oOvV = numpy.ndarray((p1-p0,nocc,nvir,nvir), buffer=buf3)
        eris_oOvV[:] = eris_ovov.transpose(0,2,1,3)
        eris_oVOv = lib.transpose(eris_oOvV.reshape(p1-p0,nov,nvir), axes=(0,2,1), out=buf5)
        eris_oVOv = eris_oVOv.reshape(p1-p0,nvir,nocc,nvir)

        for j0, j1 in lib.prange(0, nocc, blksize):
            tau = make_tau(t2[j0:j1], t1[j0:j1], t1, 1, out=buf2)
            #: woooo[p0:p1,:,j0:j1] += numpy.einsum('ijab,klab->ijkl', eris_oOvV, tau)
            _dgemm('N', 'T', (p1-p0)*nocc, (j1-j0)*nocc, nvir*nvir,
                   eris_oOvV.reshape((p1-p0)*nocc,nvir*nvir), tau.reshape((j1-j0)*nocc,nvir*nvir),
                   woooo[p0:p1].reshape((p1-p0)*nocc,nocc*nocc), 1, 1, 0, 0, j0*nocc)
            for i in range(j1-j0):
                tau[i] -= t2[j0+i] * .5
            #: woVoV[j0:j1] += numpy.einsum('jkca,ickb->jiab', tau, eris_ovov)
            lib.ddot(lib.transpose(tau.reshape(j1-j0,nov,nvir), axes=(0,2,1)).reshape((j1-j0)*nvir,nov),
                     eris_oVOv.reshape((p1-p0)*nvir,nov).T,
                    1, woVoV[j0:j1].reshape((j1-j0)*nvir,(p1-p0)*nvir), 1)
        time2 = log.timer_debug1('woVoV [%d:%d]'%(p0, p1), *time2)

        tau = make_tau(t2[p0:p1], t1[p0:p1], t1, 1, out=buf2)
        #: t2new += .5 * numpy.einsum('klij,klab->ijab', woooo[p0:p1], tau)
        lib.ddot(woooo[p0:p1].reshape((p1-p0)*nocc,nocc*nocc).T,
                 tau.reshape((p1-p0)*nocc,nvir*nvir),
                 .5, t2new.reshape(nocc*nocc,nvir**2), 1)
        eris_ovov = eris_oVOv = eris_oOvV = wooVV = tau = tmp = None

        t2ibja = lib.transpose(_cp(t2[p0:p1]).reshape(p1-p0,nov,nvir), axes=(0,2,1),
                               out=buf1).reshape(p1-p0,nvir,nocc,nvir)
        tmp = numpy.ndarray((blksize,nvir,nocc,nvir), buffer=buf2)
        for j0, j1 in lib.prange(0, nocc, blksize):
            #: t2new[j0:j1] += numpy.einsum('ibkc,kcja->ijab', woVoV[j0:j1], t2ibja)
            lib.ddot(woVoV[j0:j1].reshape((j1-j0)*nvir,(p1-p0)*nvir),
                     t2ibja.reshape((p1-p0)*nvir,nov), 1, tmp[:j1-j0].reshape((j1-j0)*nvir,nov))
            for i in range(j1-j0):
                t2new[j0+i] += tmp[i].transpose(1,2,0)
                t2new[j0+i] += tmp[i].transpose(1,0,2) * .5
        woVoV = t2ibja = tmp = None
        time1 = log.timer_debug1('contract occ [%d:%d]'%(p0, p1), *time1)
    buf1 = buf2 = buf3 = buf4 = buf5 = bufs = None
    time1 = log.timer_debug1('contract loop', *time0)

    woooo = None
    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    #: t2new += numpy.einsum('ijac,bc->ijab', t2, ft_ab)
    #: t2new -= numpy.einsum('ki,kjab->ijab', ft_ij, t2)
    lib.ddot(t2.reshape(nocc*nov,nvir), ft_ab.T, 1, t2new.reshape(nocc*nov,nvir), 1)
    lib.ddot(ft_ij.T, t2.reshape(nocc,nocc*nvir**2),-1,
             t2new.reshape(nocc,nocc*nvir**2), 1)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    t1new += numpy.einsum('ib,ab->ia', t1, fvv)
    t1new -= numpy.einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2new[i,:i] += t2new[:i,i].transpose(0,2,1)
            t2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            t2new[:i,i] = t2new[i,:i].transpose(0,2,1)
        t2new[i,i] = t2new[i,i] + t2new[i,i].T
        t2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new


def add_wvvVV_(mycc, t1, t2, eris, t2new_tril, with_ovvv=True):
    time0 = time.clock(), time.time()
    nocc, nvir = t1.shape

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
    def contract_rec_(t2new_tril, tau, eri, i0, i1, j0, j1):
        nao = tau.shape[-1]
        ic = i1 - i0
        jc = j1 - j0
        #: t2tril[:,j0:j1] += numpy.einsum('xcd,cdab->xab', tau[:,i0:i1], eri)
        _dgemm('N', 'N', nocc*(nocc+1)//2, jc*nao, ic*nao,
               tau.reshape(-1,nao*nao), eri.reshape(-1,jc*nao),
               t2new_tril.reshape(-1,nao*nao), 1, 1, i0*nao, 0, j0*nao)

        #: t2tril[:,i0:i1] += numpy.einsum('xcd,abcd->xab', tau[:,j0:j1], eri)
        _dgemm('N', 'T', nocc*(nocc+1)//2, ic*nao, jc*nao,
               tau.reshape(-1,nao*nao), eri.reshape(-1,jc*nao),
               t2new_tril.reshape(-1,nao*nao), 1, 1, j0*nao, 0, i0*nao)

    def contract_tril_(t2new_tril, tau, eri, a0, a):
        nvir = tau.shape[-1]
        #: t2new[i,:i+1, a] += numpy.einsum('xcd,cdb->xb', tau[:,a0:a+1], eri)
        _dgemm('N', 'N', nocc*(nocc+1)//2, nvir, (a+1-a0)*nvir,
               tau.reshape(-1,nvir*nvir), eri.reshape(-1,nvir),
               t2new_tril.reshape(-1,nvir*nvir), 1, 1, a0*nvir, 0, a*nvir)

        #: t2new[i,:i+1,a0:a] += numpy.einsum('xd,abd->xab', tau[:,a], eri[:a])
        if a > a0:
            _dgemm('N', 'T', nocc*(nocc+1)//2, (a-a0)*nvir, nvir,
                   tau.reshape(-1,nvir*nvir), eri.reshape(-1,nvir),
                   t2new_tril.reshape(-1,nvir*nvir), 1, 1, a*nvir, 0, a0*nvir)

    if mycc.direct:   # AO-direct CCSD
        mol = mycc.mol
        if hasattr(eris, 'mo_coeff'):
            mo = eris.mo_coeff
        else:
            mo = _mo_without_core(mycc, mycc.mo_coeff)
        nao, nmo = mo.shape
        nao_pair = nao * (nao+1) // 2
        aos = numpy.asarray(mo[:,nocc:].T, order='F')
        nocc2 = nocc*(nocc+1)//2
        outbuf = numpy.empty((nocc2,nao,nao))
        tau = numpy.ndarray((nocc2,nvir,nvir), buffer=outbuf)
        p1 = 0
        for i in range(nocc):
            p0, p1 = p1, p1 + i+1
            tau[p0:p1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p1] += t2[i,:i+1]
        tau = _ao2mo.nr_e2(tau.reshape(nocc2,nvir**2), aos, (0,nao,0,nao), 's1', 's1')
        tau = tau.reshape(nocc2,nao,nao)
        time0 = logger.timer_debug1(mycc, 'vvvv-tau', *time0)

        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        outbuf[:] = 0
        ao_loc = mol.ao_loc_nr()
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        dmax = max(4, numpy.sqrt(max_memory*.95e6/8/nao**2/2))
        sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
        dmax = max(x[2] for x in sh_ranges)
        eribuf = numpy.empty((dmax,dmax,nao,nao))
        loadbuf = numpy.empty((dmax,dmax,nao,nao))
        fint = gto.moleintor.getints4c

        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip]:
                eri = fint(intor, mol._atm, mol._bas, mol._env,
                           shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                           ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                tmp = numpy.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
                contract_rec_(outbuf, tau, tmp, i0, i1, j0, j1)
                time0 = logger.timer_debug1(mycc, 'AO-vvvv [%d:%d,%d:%d]' %
                                            (ish0,ish1,jsh0,jsh1), *time0)
            eri = fint(intor, mol._atm, mol._bas, mol._env,
                       shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                       ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            for i in range(i1-i0):
                p0, p1 = i*(i+1)//2, (i+1)*(i+2)//2
                tmp = lib.unpack_tril(eri[p0:p1], out=loadbuf)
                contract_tril_(outbuf, tau, tmp, i0, i0+i)
            time0 = logger.timer_debug1(mycc, 'AO-vvvv [%d:%d,%d:%d]' %
                                        (ish0,ish1,ish0,ish1), *time0)
        eribuf = loadbuf = eri = tmp = None

        tmp = _ao2mo.nr_e2(outbuf, mo, (nocc,nmo,nocc,nmo), 's1', 's1', out=tau)
        t2new_tril += tmp.reshape(nocc2,nvir,nvir)

        if with_ovvv:
            #: tmp = numpy.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
            #: t2new -= tmp + tmp.transpose(1,0,3,2)
            tmp = _ao2mo.nr_e2(outbuf, mo, (nocc,nmo,0,nocc), 's1', 's1', out=tau)
            t2new_tril -= lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1).reshape(nocc2,nvir,nvir)
            tmp = _ao2mo.nr_e2(outbuf, mo, (0,nocc,nocc,nmo), 's1', 's1', out=tau)
            #: t2new_tril -= numpy.einsum('xkb,ka->xab', tmp.reshape(-1,nocc,nvir), t1)
            tmp = lib.transpose(tmp.reshape(nocc2,nocc,nvir), axes=(0,2,1), out=outbuf)
            tmp = lib.ddot(tmp.reshape(nocc2*nvir,nocc), t1, 1,
                           numpy.ndarray((nocc2*nvir,nvir), buffer=tau), 0)
            tmp = lib.transpose(tmp.reshape(nocc2,nvir,nvir), axes=(0,2,1), out=outbuf)
            t2new_tril -= tmp.reshape(nocc2,nvir,nvir)

    else:
        #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
        tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
        p1 = 0
        for i in range(nocc):
            p0, p1 = p1, p1 + i+1
            tau[p0:p1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p1] += t2[i,:i+1]
        time0 = logger.timer_debug1(mycc, 'vvvv-tau', *time0)
        max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
        blksize = int(min(nvir, max(4, max_memory*.95e6/8/(nvir**3*2+1))))

        def block_contract(buf, a0, a1):
            for a in range(a0, a1):
                contract_tril_(t2new_tril, tau, buf[a-a0], 0, a)

        with lib.call_in_background(block_contract) as bcontract:
            p1 = 0
            outbuf = numpy.empty((blksize,nvir,nvir,nvir))
            outbuf1 = numpy.empty_like(outbuf)
            for a0, a1 in lib.prange(0, nvir, blksize):
                for a in range(a0, a1):
                    p0, p1 = p1, p1 + a+1
                    lib.unpack_tril(eris.vvvv[p0:p1], out=outbuf[a-a0])
                bcontract(outbuf, a0, a1)
                outbuf, outbuf1 = outbuf1, outbuf
                time0 = logger.timer_debug1(mycc, 'vvvv [%d:%d]'%(a0,a1), *time0)
    return t2new_tril


def get_nocc(mycc):
    if mycc._nocc is not None:
        return mycc._nocc
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        nocc = int(mycc.mo_occ.sum()) // 2 - mycc.frozen
        assert(nocc > 0)
        return nocc
    else:
        occ_idx = mycc.mo_occ > 0
        occ_idx[list(mycc.frozen)] = False
        return numpy.count_nonzero(occ_idx)

def get_nmo(mycc):
    if mycc._nmo is not None:
        return mycc._nmo
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        return len(mycc.mo_occ) - mycc.frozen
    else:
        return len(mycc.mo_occ) - len(mycc.frozen)

def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov + nocc*(nocc+1)//2*nvir**2
    vector = numpy.ndarray(size, t1.dtype, buffer=out)
    vector[:nov] = t1.ravel()
    vector[nov:] = t2[numpy.tril_indices(nocc)].ravel()
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].copy().reshape((nocc,nvir))
    t2 = numpy.zeros((nocc,nocc,nvir,nvir), vector.dtype)
    t2tril = vector[nov:].reshape(nocc*(nocc+1)//2,nvir,nvir)
    idx = numpy.tril_indices(nocc)
    t2[idx[0],idx[1]] = t2tril
    t2[idx[1],idx[0]] = t2tril.transpose(0,2,1)
    return t1, t2


def energy(mycc, t1, t2, eris):
    '''CCSD correlation energy'''
    nocc, nvir = t1.shape
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    tau = numpy.empty((1,nocc,nvir,nvir))
    for p0 in range(nocc):
        p1 = p0 + 1
        make_tau(t2[p0:p1], t1[p0:p1], t1, 1, out=tau)
        theta = tau*2 - tau.transpose(0,1,3,2)
        e += numpy.einsum('ijab,ijab', theta,
                          eris.ovov[p0:p1].transpose(0,2,1,3))
    return e


def as_scanner(cc):
    '''Generating a scanner/solver for CCSD PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CCSD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CCSD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, cc
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> cc_scanner = cc.CCSD(scf.RHF(mol)).as_scanner()
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    logger.info(cc, 'Set %s as a scanner', cc.__class__)
    class CCSD_Scanner(cc.__class__, lib.SinglePointScanner):
        def __init__(self, cc):
            self.__dict__.update(cc.__dict__)
            self._scf = cc._scf.as_scanner()
        def __call__(self, mol):
            mf_scanner = self._scf
            mf_scanner(mol)
            self.mol = mol
            self.mo_coeff = mf_scanner.mo_coeff
            self.mo_occ = mf_scanner.mo_occ
            self.kernel(self.t1, self.t2)[0]
            return self.e_tot
    return CCSD_Scanner(cc)


class CCSD(lib.StreamObject):
    '''restricted CCSD

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            converge threshold.  Default is 1e-7.
        conv_tol_normt : float
            converge threshold for norm(t1,t2).  Default is 1e-5.
        max_cycle : int
            max number of iterations.  Default is 50.
        diis_space : int
            DIIS space size.  Default is 6.
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        direct : bool
            AO-direct CCSD. Default is False.
        frozen : int or list
            If integer is given, the inner-most orbitals are frozen from CC
            amplitudes.  Given the orbital indices (0-based) in a list, both
            occupied and virtual orbitals can be frozen in CC calculation.

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> # freeze 2 core orbitals
            >>> mycc = cc.CCSD(mf).set(frozen = 2).run()
            >>> # freeze 2 core orbitals and 3 high lying unoccupied orbitals
            >>> mycc.set(frozen = [0,1,16,17,18]).run()

    Saved results

        converged : bool
            CCSD converged or not
        e_corr : float
            CCSD correlation correction
        e_tot : float
            Total CCSD energy (HF + correlation)
        t1, t2 : 
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
        l1, l2 : 
            Lambda amplitudes l1[i,a], l2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        from pyscf import gto
        if isinstance(mf, gto.Mole):
            raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.10.
In the new API, the first argument of CC class is HF objects.  Please see
http://sunqm.net/pyscf/code-rule.html#api-rules for the details of API conventions''')
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_cycle = 50
        self.conv_tol = 1e-7
        self.conv_tol_normt = 1e-5
        self.diis_space = 6
        self.diis_file = None
        self.diis_start_cycle = 0
# FIXME: Should we avoid DIIS starting early?
        self.diis_start_energy_diff = 1e9
        self.direct = False

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = False
        self.converged_lambda = False
        self.emp2 = None
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self._nocc = None
        self._nmo = None
        self.chkfile = None

        self._keys = set(self.__dict__.keys())

    @property
    def ecc(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    nocc = property(get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('CCSD nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def get_init_guess(self, eris=None):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return self.init_amps(eris)
    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        nvir = mo_e.size - nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
        self.emp2 = 0
        for i in range(nocc):
            gi = eris.ovov[i].transpose(1,0,2)
            t2i = t2[i] = gi/lib.direct_sum('jb,a->jba', eia, eia[i])
            self.emp2 += 4 * numpy.einsum('jab,jab', t2i[:i], gi[:i])
            self.emp2 += 2 * numpy.einsum('ab,ab'  , t2i[i] , gi[i] )
            self.emp2 -= 2 * numpy.einsum('jab,jba', t2i[:i], gi[:i])
            self.emp2 -=     numpy.einsum('ab,ba'  , t2i[i] , gi[i] )

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy


    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)
    def ccsd(self, t1=None, t2=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if self.converged:
            logger.info(self, 'CCSD converged')
        else:
            logger.note(self, 'CCSD not converged')
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    as_scanner = as_scanner


    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import ccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return ccsd_t.kernel(self, eris, t1, t2, self.verbose)

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2)

    def ao2mo(self, mo_coeff=None):
        # Pseudo code how _ERIS are implemented:
        # nocc = self.nocc
        # nmo = self.nmo
        # nvir = nmo - nocc
        # eri1 = ao2mo.incore.full(self._scf._eri, mo_coeff)
        # eri1 = ao2mo.restore(1, eri1, nmo)
        # eris = lambda:None
        # eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        # eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        # eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        # eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        # eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        # ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        # eris.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
        # for i in range(nocc):
        #     for j in range(nvir):
        #         eris.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
        # eris.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:].copy(), nvir)
        # eris.fock = numpy.diag(self._scf.mo_energy)
        # return eris
        return _ERIS(self, mo_coeff)

    add_wvvVV_ = add_wvvVV_

    def add_wvvVV(self, t1, t2, eris, with_ovvv=True):
        nocc, nvir = t1.shape
        t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
        return self.add_wvvVV_(t1, t2, eris, t2new_tril, with_ovvv)

    update_amps = update_amps

    def diis(self, t1, t2, istep, normt, de, adiis):
        return self.diis_(t1, t2, istep, normt, de, adiis)
    def diis_(self, t1, t2, istep, normt, de, adiis):
        if (istep > self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = self.amplitudes_to_vector(t1, t2)
            t1, t2 = self.vector_to_amplitudes(adiis.update(vec))
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1, t2

    def amplitudes_to_vector(self, t1, t2):
        return amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def dump_chk(self, ci=None, frozen=None, mo_coeff=None, mo_occ=None):
        if ci is None: ci = self.ci
        if frozen is None: frozen = self.frozen
        ci_chk = {'e_corr': self.e_corr,
                  'ci': ci,
                  'frozen': frozen}

        if mo_coeff is not None: ci_chk['mo_coeff'] = mo_coeff
        if mo_occ is not None: ci_chk['mo_occ'] = mo_occ
        if self._nmo is not None: ci_chk['_nmo'] = self._nmo
        if self._nocc is not None: ci_chk['_nocc'] = self._nocc

        if self.chkfile is not None:
            chkfile = self.chkfile
        else:
            chkfile = self._scf.chkfile
        lib.chkfile.save(chkfile, 'cisd', ci_chk)

CC = CCSD

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        cput0 = (time.clock(), time.time())
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = _mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = _mo_without_core(cc, mo_coeff)
# Note: Always recompute the fock matrix because cc._scf.mo_energy may not be
# the eigenvalue of Fock matrix (eg when level shift is used in the scf object
# and the scf does not converge, mo_energy has the contribution of level shift).
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):
            log.info('Build MO integrals with incore ao2mo')
            eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
            #:eri1 = ao2mo.restore(1, eri1, nmo)
            #:self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
            #:self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            #:self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
            #:self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
            #:self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
            #:ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #:self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #:for i in range(nocc):
            #:    for j in range(nvir):
            #:        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #:self.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
            nvir_pair = nvir * (nvir+1) // 2
            self.oooo = numpy.empty((nocc,nocc,nocc,nocc))
            self.ooov = numpy.empty((nocc,nocc,nocc,nvir))
            self.ovoo = numpy.empty((nocc,nvir,nocc,nocc))
            self.oovv = numpy.empty((nocc,nocc,nvir,nvir))
            self.ovov = numpy.empty((nocc,nvir,nocc,nvir))
            self.ovvv = numpy.empty((nocc,nvir,nvir_pair))
            self.vvvv = numpy.empty((nvir_pair,nvir_pair))
            ij = 0
            outbuf = numpy.empty((nmo,nmo,nmo))
            for i in range(nocc):
                buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
                for j in range(i+1):
                    self.oooo[i,j] = self.oooo[j,i] = buf[j,:nocc,:nocc]
                    self.ooov[i,j] = self.ooov[j,i] = buf[j,:nocc,nocc:]
                    self.oovv[i,j] = self.oovv[j,i] = buf[j,nocc:,nocc:]
                ij += i + 1
            ij1 = 0
            for i in range(nocc,nmo):
                buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
                self.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
                self.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
                for j in range(nocc):
                    self.ovvv[j,i-nocc] = lib.pack_tril(_cp(buf[j,nocc:,nocc:]))
                for j in range(nocc, i+1):
                    self.vvvv[ij1] = lib.pack_tril(_cp(buf[j,nocc:,nocc:]))
                    ij1 += 1
                ij += i + 1

        elif hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            log.warn('CCSD detected DF being used in the HF object. '
                     'MO integrals are computed based on the DF 3-index tensors.\n'
                     "It\'s recommended to use dfccsd.CCSD for the DF-CCSD calculations")
            nvir_pair = nvir * (nvir+1) // 2
            oooo = numpy.zeros((nocc*nocc,nocc*nocc))
            ooov = numpy.zeros((nocc*nocc,nocc*nvir))
            ovoo = numpy.zeros((nocc*nvir,nocc*nocc))
            oovv = numpy.zeros((nocc*nocc,nvir*nvir))
            ovov = numpy.zeros((nocc*nvir,nocc*nvir))
            ovvv = numpy.zeros((nocc*nvir,nvir_pair))
            vvvv = numpy.zeros((nvir_pair,nvir_pair))

            mo = numpy.asarray(mo_coeff, order='F')
            nmo = mo.shape[1]
            ijslice = (0, nmo, 0, nmo)
            Lpq = None
            for eri1 in cc._scf.with_df.loop():
                Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
                Loo = Lpq[:,:nocc,:nocc].reshape(-1,nocc**2)
                Lov = Lpq[:,:nocc,nocc:].reshape(-1,nocc*nvir)
                Lvv = Lpq[:,nocc:,nocc:].reshape(-1,nvir**2)
                lib.ddot(Loo.T, Loo, 1, oooo, 1)
                lib.ddot(Loo.T, Lov, 1, ooov, 1)
                lib.ddot(Lov.T, Loo, 1, ovoo, 1)
                lib.ddot(Loo.T, Lvv, 1, oovv, 1)
                lib.ddot(Lov.T, Lov, 1, ovov, 1)
                Lvv = lib.pack_tril(Lvv.reshape(-1,nvir,nvir))
                lib.ddot(Lov.T, Lvv, 1, ovvv, 1)
                lib.ddot(Lvv.T, Lvv, 1, vvvv, 1)

            self.feri1 = lib.H5TmpFile()
            self.feri1['oooo'] = oooo.reshape(nocc,nocc,nocc,nocc)
            self.feri1['ooov'] = ooov.reshape(nocc,nocc,nocc,nvir)
            self.feri1['ovoo'] = ovoo.reshape(nocc,nvir,nocc,nocc)
            self.feri1['oovv'] = oovv.reshape(nocc,nocc,nvir,nvir)
            self.feri1['ovov'] = ovov.reshape(nocc,nvir,nocc,nvir)
            self.feri1['ovvv'] = ovvv.reshape(nocc,nvir,nvir_pair)
            self.feri1['vvvv'] = vvvv.reshape(nvir_pair,nvir_pair)
            self.oooo = self.feri1['oooo']
            self.ooov = self.feri1['ooov']
            self.ovoo = self.feri1['ovoo']
            self.oovv = self.feri1['oovv']
            self.ovov = self.feri1['ovov']
            self.ovvv = self.feri1['ovvv']
            self.vvvv = self.feri1['vvvv']

        else:
            log.info('Build MO integrals with outcore ao2mo')
            cput1 = time.clock(), time.time()
            self.feri1 = lib.H5TmpFile()
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            nvpair = nvir * (nvir+1) // 2
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'f8')
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8')
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8')
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8')
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvpair), 'f8')
            fsort = _ccsd.libcc.CCsd_sort_inplace
            nocc_pair = nocc*(nocc+1)//2
            nvir_pair = nvir*(nvir+1)//2
            def sort_inplace(eri):
                fsort(eri.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(nocc), ctypes.c_int(nvir),
                      ctypes.c_int(eri.shape[0]))
                vv = eri[:,:nvir_pair]
                oo = eri[:,nvir_pair:nvir_pair+nocc_pair]
                ov = eri[:,nvir_pair+nocc_pair:].reshape(-1,nocc,nvir)
                return oo, ov, vv
            buf = numpy.empty((nmo,nmo,nmo))
            def save_occ_frac(i, p0, p1, eri):
                oo, ov, vv = sort_inplace(eri)
                self.oooo[i,p0:p1] = lib.unpack_tril(oo, out=buf)
                self.ooov[i,p0:p1] = ov
                self.oovv[i,p0:p1] = lib.unpack_tril(vv, out=buf)
            def save_vir_frac(i, p0, p1, eri):
                oo, ov, vv = sort_inplace(eri)
                self.ovoo[i,p0:p1] = lib.unpack_tril(oo, out=buf)
                self.ovov[i,p0:p1] = ov
                self.ovvv[i,p0:p1] = vv

            if not cc.direct:
                max_memory = max(2000,cc.max_memory-lib.current_memory()[0])
                self.feri2 = lib.H5TmpFile()
                ao2mo.full(cc.mol, orbv, self.feri2, max_memory=max_memory, verbose=log)
                self.vvvv = self.feri2['eri_mo']
                cput1 = log.timer_debug1('transforming vvvv', *cput1)

            tmpfile3 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            with h5py.File(tmpfile3.name, 'w') as feri:
                max_memory = max(2000, cc.max_memory-lib.current_memory()[0])
                mo = numpy.hstack((orbv, orbo))
                ao2mo.general(cc.mol, (orbo,mo,mo,mo),
                              feri, max_memory=max_memory, verbose=log)
                cput1 = log.timer_debug1('transforming oppp', *cput1)
                blksize = max(1, int(min(8e9,max_memory*.5e6)/8/nmo**2))

                with lib.call_in_background(save_vir_frac,
                                            save_occ_frac) as (sav_v, sav_o):
                    for i in range(nocc):
                        for p0, p1 in lib.prange(0, nvir, blksize):
                            eri = _cp(feri['eri_mo'][i*nmo+p0:i*nmo+p1])
                            sav_v(i, p0, p1, eri)
                        for p0, p1 in lib.prange(0, nocc, blksize):
                            eri = _cp(feri['eri_mo'][i*nmo+nvir+p0:i*nmo+nvir+p1])
                            sav_o(i, p0, p1, eri)
                        cput1 = log.timer_debug1('sorting %d'%i, *cput1)
                for key in feri.keys():
                    del(feri[key])
        log.timer('CCSD integral transformation', *cput0)


def get_moidx(cc):
    moidx = numpy.ones(cc.mo_occ.size, dtype=numpy.bool)
    if isinstance(cc.frozen, (int, numpy.integer)):
        moidx[:cc.frozen] = False
    elif len(cc.frozen) > 0:
        moidx[list(cc.frozen)] = False
    return moidx
def _mo_without_core(cc, mo):
    return mo[:,get_moidx(cc)]

# assume nvir > nocc, minimal requirements on memory in loop of update_amps
def _memory_usage_inloop(nocc, nvir):
    v = max(nvir**3*.3+nocc*nvir**2*6, nocc*nvir**2*7)
    return v*8/1e6
# assume nvir > nocc, minimal requirements on memory
def _mem_usage(nocc, nvir):
    basic = _memory_usage_inloop(nocc, nvir)*1e6/8 + nocc**4
    basic = max(basic, nocc*(nocc+1)//2*nvir**2) + (nocc*nvir)**2*2
    basic = basic * 8/1e6
    nmo = nocc + nvir
    incore = (max((nmo*(nmo+1)//2)**2*2*8/1e6, basic) +
              (nocc*nvir**3/2 + nvir**4/4 + nocc**2*nvir**2*2 +
               nocc**3*nvir*2)*8/1e6)
    outcore = basic
    return incore, outcore, basic

def residual_as_diis_errvec(mycc):
# Seems not quite useful
# Cannot be used with non-conanical mf object
    def fupdate(t1, t2, istep, normt, de, adiis):
        nocc, nvir = t1.shape
        nov = nocc*nvir
        moidx = get_moidx(mycc)
        mo_e = mycc._scf.mo_energy[moidx]
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        if (istep > mycc.diis_start_cycle and
            abs(de) < mycc.diis_start_energy_diff):
            if mycc.t1 is None:
                mycc.t1 = t1
                mycc.t2 = t2
            else:
                tbuf = numpy.empty(nov*(nov+1))
                tbuf[:nov] = ((t1-mycc.t1)*eia).ravel()
                pbuf = tbuf[nov:].reshape(nocc,nocc,nvir,nvir)
                for i in range(nocc):
                    pbuf[i] = (t2[i]-mycc.t2[i]) * lib.direct_sum('jb,a->jba', eia, eia[i])
                adiis.push_err_vec(tbuf)
                tbuf = numpy.empty(nov*(nov+1))
                tbuf[:nov] = t1.ravel()
                tbuf[nov:] = t2.ravel()
                t1.data = tbuf.data # release memory
                t2.data = tbuf.data

                tbuf = adiis.update(tbuf)
                mycc.t1 = t1 = tbuf[:nov].reshape(nocc,nvir)
                mycc.t2 = t2 = tbuf[nov:].reshape(nocc,nocc,nvir,nvir)
            logger.debug(mycc, 'DIIS for step %d', istep)
        return t1, t2
    return fupdate


def _fp(nocc, nvir):
    '''Total float points'''
    return (nocc**3*nvir**2*2 + nocc**2*nvir**3*2 +     # Ftilde
            nocc**4*nvir*2 * 2 + nocc**4*nvir**2*2 +    # Wijkl
            nocc*nvir**4*2 * 2 +                        # Wabcd
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 + nocc**3*nvir**3*2 +
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # Wiabj
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # t1
            nocc**3*nvir**2*2 * 2 + nocc**4*nvir**2*2 +
            nocc*(nocc+1)/2*nvir**4*2 +                 # vvvv
            nocc**2*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 2 +     # t2
            nocc**3*nvir**3*2 +
            nocc**3*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 4)      # Wiabj


# t2 + numpy.einsum('ia,jb->ijab', t1a, t1b)
def make_tau(t2, t1a, t1b, fac=1, out=None):
    return _ccsd.make_tau(t2, t1a, t1b, fac, out)

# t2.transpose(0,1,3,2)*2 - t2
def make_theta(t2, out=None):
    return _ccsd.make_0132(t2, t2, -1, 2, out)

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.M()
    nocc, nvir = 5, 12
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    numpy.random.seed(12)
    mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = numpy.random.random((nmo,nmo))
    mf.mo_energy = numpy.arange(0., nmo)
    mf.mo_occ = numpy.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = numpy.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(numpy.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = CCSD(mf)
    eris = mycc.ao2mo()
    a = numpy.random.random((nmo,nmo)) * .1
    eris.fock += a + a.T
    t1 = numpy.random.random((nocc,nvir)) * .1
    t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    r1, r2 = vector_to_amplitudes(amplitudes_to_vector(t1, t2), nocc+nvir, nocc)
    print(abs(t1-r1).max())
    print(abs(t2-r2).max())

    def finger(a):
        return numpy.dot(a.ravel(), numpy.cos(numpy.arange(a.size)))
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(finger(t1a) - -106360.5276951083)
    print(finger(t2a) - 66540.100267798145)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf() # -76.0267656731

    mcc = CCSD(rhf.density_fit(auxbasis='weigend'))
    eris = mcc.ao2mo()
    emp2, t1, t2 = mcc.init_amps(eris)
    print(abs(t2).sum() - 4.9497459404635027)
    print(emp2 - -0.20378410853394652)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(abs(t1).sum() - 0.04740146211751467)
    print(abs(t2).sum() - 5.3943058907680506 )

    mcc = CCSD(rhf)
    eris = mcc.ao2mo()
    emp2, t1, t2 = mcc.init_amps(eris)
    print(abs(t2).sum() - 4.9556571218177)
    print(emp2 - -0.2040199672883385)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(abs(t1).sum()-0.0475038989126)
    print(abs(t2).sum()-5.401823846018721)
    print(energy(mcc, t1, t2, eris) - -0.208967840546667)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(energy(mcc, t1, t2, eris) - -0.212173678670510)
    print(abs(t1).sum() - 0.05470123093500083)
    print(abs(t2).sum() - 5.5605208391876539)

    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    mcc.max_memory = 1
    mcc.direct = True
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    mcc.diis = residual_as_diis_errvec(mcc)
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)
    print(mcc.ccsd_t() - -0.003060021865720902)

