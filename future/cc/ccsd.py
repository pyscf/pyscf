#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo

libcc = lib.load_library('libcc')

# t2 as ijab

# default max_memory = 2000 MB
def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    blksize = cc.get_block_size()
    log.debug('block size = %d, nocc = %d is divided into %d blocks',
              blksize, cc.nocc, int((cc.nocc+blksize-1)/blksize))
    cput0 = log.timer('CCSD initialization', *cput0)
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, blksize)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris, blksize)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    return conv, eccsd, t1, t2


def update_amps(cc, t1, t2, eris, blksize=1):
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nov = nocc*nvir
    fock = eris.fock
    t1new = numpy.zeros_like(t1)
    t2new = numpy.zeros_like(t2)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()

    foo = fock[:nocc,:nocc].copy()
    foo[range(nocc),range(nocc)] = 0
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:].copy()
    fvv[range(nvir),range(nvir)] = 0
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    #: woooo = numpy.einsum('la,ikja->ikjl', t1, eris.ooov)
    eris_ooov = numpy.asarray(eris.ooov)
    woooo = lib.dot(eris_ooov.reshape(-1,nvir), t1.T).reshape((nocc,)*4)
    woooo = lib.transpose_sum(woooo.reshape(nocc*nocc,-1), inplace=True)
    woooo = woooo.reshape(nocc,nocc,nocc,nocc) + numpy.asarray(eris.oooo)
    woooo = numpy.asarray(woooo.transpose(0,2,1,3), order='C')
    time1 = log.timer_debug1('woooo', *time0)
    eris_ooov = None

    for p0, p1 in prange(0, nocc, blksize):
# ==== read eris.ovvv ====
        eris_ovvv = numpy.asarray(eris.ovvv[p0:p1])
        eris_ovvv = unpack_tril(eris_ovvv.reshape((p1-p0)*nvir,-1))
        eris_ovvv = eris_ovvv.reshape(p1-p0,nvir,nvir,nvir)
        eris_ooov = numpy.asarray(eris.ooov[p0:p1])

        fvv += numpy.einsum('kc,kcba->ab', 2*t1[p0:p1], eris_ovvv)
        fvv += numpy.einsum('kc,kbca->ab',  -t1[p0:p1], eris_ovvv)

        foo[:,p0:p1] += numpy.einsum('kc,jikc->ij', 2*t1, eris_ooov)
        foo[:,p0:p1] += numpy.einsum('kc,jkic->ij',  -t1, eris_ooov)

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: tmp = numpy.einsum('ijcd,kcdb->ijbk', tau, eris.ovvv)
    #: t2new += numpy.einsum('ka,ijbk->jiba', -t1, tmp)
        #: eris_vvov = eris_ovvv.transpose(1,2,0,3).copy()
        eris_vvov = eris_ovvv.transpose(1,2,0,3).reshape(nvir*nvir,-1)
        tmp = numpy.empty((nocc,nocc,p1-p0,nvir))
        for j0, j1 in prange(0, nocc, blksize):
            tau = make_tau(t2[j0:j1], t1[j0:j1], t1, 1)
            #: tmp[j0:j1] += numpy.einsum('ijcd,cdkb->ijkb', tau, eris_vvov)
            lib.dot(tau.reshape(-1,nvir*nvir), eris_vvov, 1,
                    tmp[j0:j1].reshape((j1-j0)*nocc,-1), 0)
        #: t2new += numpy.einsum('ka,ijkb->jiba', -t1[p0:p1], tmp)
        tmp = numpy.asarray(tmp.transpose(1,0,3,2).reshape(-1,p1-p0), order='C')
        lib.dot(tmp, t1[p0:p1], -1, t2new.reshape(-1,nvir), 1)
        tau = tmp = eris_vvov = None
        #==== mem usage blksize*(nvir**3*2+nvir*nocc**2*2)

    #: wovvo += numpy.einsum('iabc,jc->ijab', eris.ovvv, t1)
    #: wovvo -= numpy.einsum('jbik,ka->jiba', eris.ovoo, t1)
    #: t2new += woVoV.transpose()
        #: wovvo = -numpy.einsum('jbik,ka->ijba', eris.ovoo[p0:p1], t1)
        tmp = numpy.asarray(eris.ovoo[p0:p1].transpose(2,0,1,3), order='C')
        wovvo = lib.dot(tmp.reshape(-1,nocc), t1, -1)
        wovvo = wovvo.reshape(nocc,p1-p0,nvir,nvir)
        #: wovvo += numpy.einsum('iabc,jc->jiab', eris_ovvv, t1)
        lib.dot(t1, eris_ovvv.reshape(-1,nvir).T, 1, wovvo.reshape(nocc,-1), 1)
        t2new[p0:p1] += wovvo.transpose(1,0,2,3)

        #: woVoV = numpy.einsum('ka,ijkb->ijba', t1, eris.ooov[p0:p1])
        #: woVoV -= numpy.einsum('jc,icab->ijab', t1, eris_ovvv)
        woVoV = lib.dot(numpy.asarray(eris_ooov.transpose(0,1,3,2),
                                      order='C').reshape(-1,nocc), t1)
        woVoV = woVoV.reshape(p1-p0,nocc,nvir,nvir)
        for i in range(eris_ovvv.shape[0]):
            lib.dot(t1, eris_ovvv[i].reshape(nvir,-1), -1,
                    woVoV[i].reshape(nocc,-1), 1)

    #: theta = t2.transpose(0,1,3,2) * 2 - t2
    #: t1new += numpy.einsum('ijcb,jcba->ia', theta, eris.ovvv)
        theta = make_theta(t2[p0:p1])
        #: t1new += numpy.einsum('jibc,jcba->ia', theta, eris_ovvv)
        lib.dot(theta.transpose(1,0,3,2).reshape(nocc,-1),
                eris_ovvv.reshape(-1,nvir), 1, t1new, 1)
        eris_ovvv = None
        time2 = log.timer_debug1('ovvv [%d:%d]'%(p0, p1), *time1)
        #==== mem usage blksize*(nvir**3+nocc*nvir**2*4)

# ==== read eris.oOVv ====
        eris_oOVv = numpy.asarray(eris.ovov[p0:p1].transpose(0,2,3,1), order='C')
        #==== mem usage blksize*(nocc*nvir**2*4)

        for i in range(p1-p0):
            t2new[p0+i] += eris_oOVv[i].transpose(0,2,1) * .5

        fov[p0:p1] += numpy.einsum('kc,ikca->ia', t1, eris_oOVv) * 2
        fov[p0:p1] -= numpy.einsum('kc,ikac->ia', t1, eris_oOVv)

    #: theta = t2.transpose(1,0,2,3) * 2 - t2
    #: t1new += numpy.einsum('jb,ijab->ia', fov, theta)
    #: t1new -= numpy.einsum('ikjb,kjab->ia', eris.ooov, theta)
        t1new += numpy.einsum('jb,jiab->ia', fov[p0:p1], theta)
        #: t1new -= numpy.einsum('kijb,kjab->ia', eris.ooov[p0:p1], theta)
        lib.dot(eris_ooov.transpose(1,0,2,3).reshape(nocc,-1),
                theta.reshape(-1,nvir), -1, t1new, 1)
        eris_ooov = None

    #: wovvo += eris.ovov.transpose(0,1,3,2)
    #: theta = t2.transpose(1,0,2,3) * 2 - t2
    #: tau = theta - numpy.einsum('ic,kb->ikcb', t1, t1*2)
    #: wovvo += .5 * numpy.einsum('jakc,ikcb->jiba', eris.ovov, tau)
    #: wovvo -= .5 * numpy.einsum('jcka,ikcb->jiba', eris.ovov, t2)
    #: t2new += numpy.einsum('ikca,kjbc->ijba', theta, wovvo)
        theta = numpy.asarray(theta.transpose(1,2,3,0).reshape(nov,-1), order='C')
        wovvo = wovvo.transpose(0,3,2,1) + eris_oOVv.transpose(1,2,3,0)
        wovvo = numpy.asarray(wovvo, order='C')
        eris_OVvo = eris_oOVv.transpose(1,2,3,0).reshape(nov,-1)
        eris_OvVo = eris_oOVv.transpose(1,3,2,0).reshape(nov,-1)
        for j0, j1 in prange(0, nocc, blksize):
            t2iajb = numpy.asarray(t2[j0:j1].transpose(0,2,1,3), order='C')
            #: wovvo[j0:j1] -= .5 * numpy.einsum('icka,jkbc->jbai', eris_oOVv, t2)
            lib.dot(t2iajb.reshape(-1,nov), eris_OvVo,
                    -.5, wovvo[j0:j1].reshape((j1-j0)*nvir,-1), 1)
            tau = t2iajb
            for i in range(j1-j0):
                tau[i] *= 2
                tau[i] -= t2[j0+i].transpose(2,0,1)
                tau[i] -= numpy.einsum('a,jb->bja', t1[j0+i]*2, t1)
            #: wovvo[j0:j1] += .5 * numpy.einsum('ikca,jbkc->jbai', eris_oOVv, tau)
            lib.dot(tau.reshape(-1,nov), eris_OVvo,
                    .5, wovvo[j0:j1].reshape((j1-j0)*nvir,-1), 1)

            #theta = t2[p0:p1] * 2 - t2[p0:p1].transpose(0,1,3,2)
            #: t2new[j0:j1] += numpy.einsum('iack,jbck->jiba', theta, wovvo[j0:j1])
            tmp = lib.dot(wovvo[j0:j1].reshape((j1-j0)*nvir,-1), theta.T)
            t2new[j0:j1] += tmp.reshape(j1-j0,nvir,nocc,nvir).transpose(0,2,1,3)
            tau = tmp = None
            #==== mem usage blksize*(nocc*nvir**2*8)
        theta = wovvo = eris_OvVo = eris_OVvo = None
        time2 = log.timer_debug1('wovvo [%d:%d]'%(p0, p1), *time2)
        #==== mem usage blksize*(nocc*nvir**2*2)

    #: fvv -= numpy.einsum('ijca,ibjc->ab', theta, eris.ovov)
    #: foo += numpy.einsum('iakb,jkba->ij', eris.ovov, theta)
        tau = make_tau(t2[p0:p1], t1[p0:p1], t1, .5)
        theta = make_theta(tau)
        #: foo += numpy.einsum('kiab,kjab->ij', eris_oOVv, theta)
        #: fvv -= numpy.einsum('ijca,ijcb->ab', theta, eris_oOVv)
        for i in range(eris_oOVv.shape[0]):
            lib.dot(eris_oOVv[i].reshape(nocc,-1),
                    theta[i].reshape(nocc,-1).T, 1, foo, 1)
        lib.dot(theta.reshape(-1,nvir).T, eris_oOVv.reshape(-1,nvir),
                -1, fvv, 1)
        tau = theta = None

# ==== read eris.oovv ====
        eris_oovv = numpy.asarray(eris.oovv[p0:p1])
        #==== mem usage blksize*(nocc*nvir**2*3)

        #: tmp  = numpy.einsum('ic,kjbc->kjib', t1, eris_oovv)
        #: tmp += numpy.einsum('ic,kjbc->kijb', t1, eris_oOVv)
        tmp = lib.dot(eris_oovv.reshape(-1,nvir), t1.T).reshape(-1,nocc,nvir,nocc)
        tmp = numpy.asarray(tmp.transpose(0,3,2,1), order='C')
        lib.dot(eris_oOVv.reshape(-1,nvir), t1.T, 1, tmp.reshape(-1,nocc), 1)
        tmp = numpy.asarray(tmp.transpose(1,3,2,0), order='C')
        #: t2new += numpy.einsum('ka,jibk->ijba', -t1[p0:p1], tmp)
        lib.dot(tmp.reshape(-1,p1-p0), t1[p0:p1], -1, t2new.reshape(-1,nvir), 1)
        tmp = None

    #: g2 = 2 * eris.oOVv - eris.oovv
    #: t1new += numpy.einsum('jb,ijba->ia', t1, g2)
        t1new[p0:p1] += numpy.einsum('jb,ijba->ia', 2*t1, eris_oOVv)
        t1new[p0:p1] += numpy.einsum('jb,ijba->ia',  -t1, eris_oovv)

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: woooo += numpy.einsum('ijba,klab->ijkl', eris.oOVv, tau)
    #: woVoV -= eris.oovv
    #: tau = .5*t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: woVoV += numpy.einsum('ka,ijkb->ijab', t1, eris.ooov)
    #: woVoV += numpy.einsum('jkca,ikbc->ijab', tau, eris.oOVv)
        woVoV -= eris_oovv
        woVoV = woVoV.transpose(1,3,0,2).copy()
        eris_oVOv = eris_oOVv.transpose(0,2,1,3).reshape(-1,nov)
        eris_oOvV = eris_oOVv.transpose(0,1,3,2).reshape(-1,nvir**2)
        #==== mem usage blksize*(nocc*nvir**2*4)

        for j0, j1 in prange(0, nocc, blksize):
            tau = make_tau(t2[j0:j1], t1[j0:j1], t1, 1)
            #: woooo[p0:p1,:,j0:j1] += numpy.einsum('ijab,klab->ijkl', eris_oOvV, tau)
            lib.numpy_helper._dgemm('N', 'T', (p1-p0)*nocc, (j1-j0)*nocc, nvir*nvir,
                                    eris_oOvV.reshape(-1,nvir*nvir),
                                    tau.reshape(-1,nvir*nvir),
                                    woooo[p0:p1].reshape(-1,nocc*nocc), 1, 1,
                                    0, 0, j0*nocc)
            for i in range(j1-j0):
                tau[i] -= t2[j0+i] * .5
            #: woVoV[j0:j1] += numpy.einsum('jkca,ikbc->jiab', tau, eris_oOVv)
            tau = tau.transpose(0,3,1,2).reshape(-1,nov)
            lib.dot(tau, eris_oVOv.T,
                    1, woVoV[j0:j1].reshape((j1-j0)*nvir,-1), 1)
            #==== mem usage blksize*(nocc*nvir**2*6)
        time2 = log.timer_debug1('woVoV [%d:%d]'%(p0, p1), *time2)

        tau = make_tau(t2[p0:p1], t1[p0:p1], t1, 1)
        #: t2new += .5 * numpy.einsum('klij,klab->ijab', woooo[p0:p1], tau)
        lib.dot(woooo[p0:p1].reshape(-1,nocc*nocc).T,
                tau.reshape(-1,nvir*nvir), .5,
                t2new.reshape(nocc*nocc,-1), 1)
        eris_oovv = eris_oOVv = eris_oVOv = eris_oOvV = tau = None
        #==== mem usage blksize*(nocc*nvir**2*1)

        t2iajb = numpy.asarray(t2[p0:p1].transpose(0,2,1,3), order='C')
        t2ibja = numpy.asarray(t2[p0:p1].transpose(0,3,1,2), order='C')
        for j0, j1 in prange(0, nocc, blksize):
            #: t2new[j0:j1] += numpy.einsum('ibkc,kcja->ijab', woVoV[j0:j1], t2ibja)
            tmp = lib.dot(woVoV[j0:j1].reshape((j1-j0)*nvir,-1),
                          t2ibja.reshape(-1,nov))
            t2new[j0:j1] += tmp.reshape(j1-j0,nvir,nocc,nvir).transpose(0,2,3,1)
            tmp = None

            #: t2new[j0:j1] += numpy.einsum('iakc,kcjb->ijab', woVoV[j0:j1], t2iajb)
            tmp = lib.dot(woVoV[j0:j1].reshape((j1-j0)*nvir,-1),
                          t2iajb.reshape(-1,nov))
            t2new[j0:j1] += tmp.reshape(j1-j0,nvir,nocc,nvir).transpose(0,2,1,3)
            tmp = None
        t2ibja = t2iajb = woVoV = None
        #==== mem usage blksize*(nocc*nvir**2*3)
        time1 = log.timer_debug1('contract occ [%d:%d]'%(p0, p1), *time1)
# ==================
    time1 = log.timer_debug1('contract loop', *time0)

    woooo = None
    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    #: t2new += numpy.einsum('ijac,bc->ijab', t2, ft_ab)
    #: t2new -= numpy.einsum('ki,kjab->ijab', ft_ij, t2)
    lib.dot(t2.reshape(-1,nvir), ft_ab.T, 1, t2new.reshape(-1,nvir), 1)
    lib.dot(ft_ij.T, t2.reshape(nocc,-1),-1, t2new.reshape(nocc,-1), 1)

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    t2new_tril = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
    ij = 0
    for i in range(nocc):
        for j in range(i+1):
            t2new_tril[ij]  = t2new[i,j]
            t2new_tril[ij] += t2new[j,i].T
            ij += 1
    t2new = None
    time1 = log.timer_debug1('t2 tril', *time1)
    cc.add_wvvVV_(t1, t2, eris, t2new_tril, blksize)
    time1 = log.timer_debug1('vvvv', *time1)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    p0 = 0
    for i in range(nocc):
        dajb = (eia[i].reshape(-1,1) + eia[:i+1].reshape(1,-1))
        t2new_tril[p0:p0+i+1] /= dajb.reshape(nvir,i+1,nvir).transpose(1,0,2)
        p0 += i+1
    time1 = log.timer_debug1('g2/dijab', *time1)

    t2new = numpy.empty((nocc,nocc,nvir,nvir))
    ij = 0
    for i in range(nocc):
        for j in range(i):
            t2new[i,j] = t2new_tril[ij]
            t2new[j,i] = t2new_tril[ij].T
            ij += 1
        t2new[i,i] = t2new_tril[ij]
        ij += 1
    t2new_tril = None

#** update_amp_t1
    t1new += fock[:nocc,nocc:] \
           + numpy.einsum('ib,ab->ia', t1, fvv) \
           - numpy.einsum('ja,ji->ia', t1, foo)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    t1new /= eia
#** end update_amp_t1
    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new

def energy(cc, t1, t2, eris, blksize=1):
    nocc = cc.nocc
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    for p0 in range(nocc):
        p1 = p0 + 1
        tau = make_tau(t2[p0:p1], t1[p0:p1], t1, 1)
        theta = tau*2 - tau.transpose(0,1,3,2)
        e += numpy.einsum('ijab,ijab', theta,
                          eris.ovov[p0:p1].transpose(0,2,1,3))
    return e


class CCSD(object):
    '''CCSD

    Args

    Returns
        t1[i,a]
        t2[i,j,a,b]
    '''
    def __init__(self, mf, frozen=[]):
        from pyscf import gto
        if isinstance(mf, gto.Mole):
            raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.10.
In the new API, the first argument of CC class is HF objects.  Please see
http://sunqm.net/pyscf/code-rule.html#api-rules for the details of API conventions''')

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
        self.diis_start_cycle = 1
        self.diis_start_energy_diff = 1e-2

        self.frozen = frozen
        self.nocc = self.mol.nelectron // 2 - len(frozen)
        self.nmo = len(mf.mo_energy) - len(frozen)

        self._conv = False
        self.emp2 = None
        self.ecc = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None

        self._keys = set(self.__dict__.keys())

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        mo_e = eris.fock.diagonal()
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
        self.emp2 = 0
        for i in range(nocc):
            dajb = (eia[i].reshape(-1,1) + eia.reshape(1,-1)).reshape(-1)
            gi = eris.ovov[i].transpose(1,0,2).copy()
            t2i = t2[i] = gi/dajb.reshape(nvir,nocc,nvir).transpose(1,0,2)
            self.emp2 += 4 * numpy.einsum('jab,jab', t2i[:i], gi[:i])
            self.emp2 += 2 * numpy.einsum('ab,ab'  , t2i[i] , gi[i] )
            self.emp2 -= 2 * numpy.einsum('jab,jba', t2i[:i], gi[:i])
            self.emp2 -=     numpy.einsum('ab,ba'  , t2i[i] , gi[i] )

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2


    def kernel(self, t1=None, t2=None, mo_coeff=None):
        return self.ccsd(t1, t2, mo_coeff)
    def ccsd(self, t1=None, t2=None, mo_coeff=None):
        eris = self.ao2mo(mo_coeff)
        cput0 = (time.clock(), time.time())
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory-lib.current_memory()[0],
                       verbose=self.verbose)
        if self._conv:
            logger.info(self, 'CCSD converged')
            logger.info(self, ' E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.hf_energy, self.ecc)
        else:
            logger.info(self, 'CCSD not converge')
            logger.info(self, ' E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.hf_energy, self.ecc)
        logger.timer(self, 'CCSD', *cput0)
        return self.ecc, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, mo_coeff=None):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        eris = ccsd_lambda._ERIS(self, mo_coeff)
        conv, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   tol=self.conv_tol_normt,
                                   max_memory=self.max_memory-lib.current_memory()[0],
                                   verbose=self.verbose)
        return conv, self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None):
        '''1-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
        '''2-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2)

    def ao2mo(self, mo_coeff=None):
        #nocc = self.nocc
        #nmo = self.nmo
        #nvir = nmo - nocc
        #eri1 = pyscf.ao2mo.incore.full(self._scf._eri, self._scf.mo_coeff)
        #eri1 = pyscf.ao2mo.restore(1, eri1, nmo)
        #eris = lambda:None
        #eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        #eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        #eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        #eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        #eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        #eris.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
        #for i in range(nocc):
        #    for j in range(nvir):
        #        eris.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
        #eris.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:].copy(), nvir)
        #eris.fock = numpy.diag(self._scf.mo_energy)
        #return eris
        return _ERIS(self, mo_coeff)

    def add_wvvVV_(self, t1, t2, eris, t2new_tril, blksize=1):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nvir = self.nmo - nocc
        #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
        tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
        p0 = 0
        for i in range(nocc):
            tau[p0:p0+i+1] = t2[i,:i+1] \
                           + numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            p0 += i + 1
        time0 = logger.timer_debug1(self, 'vvvv-tau', *time0)

        p0 = 0
        for a in range(nvir):
            buf = unpack_tril(eris.vvvv[p0:p0+a+1])
            #: t2new_tril[i,:i+1, a] += numpy.einsum('xcd,cdb->xb', tau[:,:a+1], buf)
            lib.numpy_helper._dgemm('N', 'N', nocc*(nocc+1)//2, nvir, (a+1)*nvir,
                                    tau.reshape(-1,nvir*nvir), buf.reshape(-1,nvir),
                                    t2new_tril.reshape(-1,nvir*nvir), 1, 1,
                                    0, 0, a*nvir)

            #: t2new_tril[i,:i+1,:a] += numpy.einsum('xd,abd->xab', tau[:,a], buf[:a])
            if a > 0:
                lib.numpy_helper._dgemm('N', 'T', nocc*(nocc+1)//2, a*nvir, nvir,
                                        tau.reshape(-1,nvir*nvir), buf.reshape(-1,nvir),
                                        t2new_tril.reshape(-1,nvir*nvir), 1, 1,
                                        a*nvir, 0, 0)
            p0 += a+1
            time0 = logger.timer_debug1(self, 'vvvv %d'%a, *time0)
        return t2new_tril
    def add_wvvVV(self, t1, t2, eris, blksize=1):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
        return self.add_wvvVV_(t1, t2, eris, t2new_tril, blksize=1)

    def get_block_size(self):
        #return 8
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        unit = _memory_usage_inloop(nocc, nvir)*1e6/8
        rest = (self.max_memory-lib.current_memory()[0])*1e6/8*.9 \
                - nocc**4 - nocc**3*nvir - (nocc*nvir)**2*2
        return min(nocc, max(1, int(rest/unit/8)*8))

    def update_amps(self, t1, t2, eris, blksize=1):
        return update_amps(self, t1, t2, eris, blksize)

    def diis(self, t1, t2, istep, normt, de, adiis):
        return self.diis_(t1, t2, istep, normt, de, adiis)
    def diis_(self, t1, t2, istep, normt, de, adiis):
        if (istep > self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            t1t2 = numpy.hstack((t1.ravel(),t2.ravel()))
#NOTE: here overwriting .data to reduce memory usage, the contents of
# t1, t2 are CHANGED!  If the pass-in t1/t2 are used elsewhere, be very
# careful to call this function
            t1.data = t1t2.data
            t2.data = t1t2.data
            t1t2 = adiis.update(t1t2)
            t1 = t1t2[:t1.size].reshape(t1.shape)
            t2 = t1t2[t1.size:].reshape(t2.shape)
            logger.debug(self, 'DIIS for step %d', istep)
        return t1, t2

CC = CCSD

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        moidx = numpy.ones(cc.nmo+len(cc.frozen), dtype=numpy.bool)
        moidx[cc.frozen] = False
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc._scf.mo_coeff[:,moidx]
            self.fock = numpy.diag(cc._scf.mo_energy[moidx])
        else:
            mocc = mo_coeff[:,:cc.nocc+len(cc.frozen)]
            dm = numpy.dot(mocc, mocc.T) * 2
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]
        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):
            eri1 = pyscf.ao2mo.incore.full(cc._scf._eri, mo_coeff)
            #:eri1 = pyscf.ao2mo.restore(1, eri1, nmo)
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
            #:self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
            nvir_pair = nvir * (nvir+1) // 2
            self.oooo = numpy.empty((nocc,nocc,nocc,nocc))
            self.ooov = numpy.empty((nocc,nocc,nocc,nvir))
            self.ovoo = numpy.empty((nocc,nvir,nocc,nocc))
            self.oovv = numpy.empty((nocc,nocc,nvir,nvir))
            self.ovov = numpy.empty((nocc,nvir,nocc,nvir))
            self.ovvv = numpy.empty((nocc,nvir,nvir_pair))
            self.vvvv = numpy.empty((nvir_pair,nvir_pair))
            ij = 0
            for i in range(nocc):
                buf = unpack_tril(eri1[ij:ij+i+1])
                for j in range(i+1):
                    self.oooo[i,j] = self.oooo[j,i] = buf[j,:nocc,:nocc]
                    self.ooov[i,j] = self.ooov[j,i] = buf[j,:nocc,nocc:]
                    self.oovv[i,j] = self.oovv[j,i] = buf[j,nocc:,nocc:]
                    ij += 1
            ij1 = 0
            for i in range(nocc,nmo):
                buf = unpack_tril(eri1[ij:ij+i+1])
                self.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
                self.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
                for j in range(nocc):
                    self.ovvv[j,i-nocc] = lib.pack_tril(buf[j,nocc:,nocc:])
                    ij += 1
                for j in range(nocc, i+1):
                    self.vvvv[ij1] = lib.pack_tril(buf[j,nocc:,nocc:])
                    ij += 1
                    ij1 += 1
        else:
            time0 = time.clock(), time.time()
            _tmpfile1 = tempfile.NamedTemporaryFile()
            _tmpfile2 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            nvpair = nvir * (nvir+1) // 2
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'f8')
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8')
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8')
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8')
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvpair), 'f8')

            pyscf.ao2mo.full(cc.mol, orbv, _tmpfile2.name, verbose=log)
            self.feri2 = h5py.File(_tmpfile2.name, 'r')
            self.vvvv = self.feri2['eri_mo']
            time1 = log.timer_debug1('transforming vvvv', *time0)

            tmpfile3 = tempfile.NamedTemporaryFile()
            pyscf.ao2mo.general(cc.mol, (orbo,mo_coeff,mo_coeff,mo_coeff),
                                tmpfile3.name, verbose=log)
            time1 = log.timer_debug1('transforming oppp', *time1)
            with pyscf.ao2mo.load(tmpfile3.name) as eri1:
                for i in range(nocc):
                    buf = unpack_tril(numpy.asarray(eri1[i*nmo:(i+1)*nmo]))
                    self.oooo[i] = buf[:nocc,:nocc,:nocc]
                    self.ooov[i] = buf[:nocc,:nocc,nocc:]
                    self.ovoo[i] = buf[nocc:,:nocc,:nocc]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.ovov[i] = buf[nocc:,:nocc,nocc:]
                    for j in range(nvir):
                        self.ovvv[i,j] = lib.pack_tril(buf[nocc+j,nocc:,nocc:])
                    time1 = log.timer_debug1('sorting %d'%i, *time1)

    def __del__(self):
        if hasattr(self, 'feri1'):
            self.feri1.close()
            self.feri2.close()


# assume nvir > nocc, minimal requirements on memory in loop of update_amps
def _memory_usage_inloop(nocc, nvir):
    v = max(nvir**3*2+nvir*nocc**2*2,
            nvir**3+nocc*nvir**2*5+nvir*nocc**2*2,
            nocc*nvir**2*9)
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
    def fupdate(t1, t2, istep, normt, de, adiis):
        nocc = mycc.nocc
        nvir = mycc.nmo - nocc
        nov = nocc*nvir
        moidx = numpy.ones(mycc.nmo+len(mycc.frozen), dtype=numpy.bool)
        moidx[mycc.frozen] = False
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
                    djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).reshape(-1)
                    pbuf[i] = (t2[i]-mycc.t2[i]) * djba.reshape(nocc,nvir,nvir)
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


def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _fp(nocc, nvir):
    '''Total float points'''
    return (nocc**3*nvir**2*2 + nocc**2*nvir**3*2 +     # Ftilde
            nocc**4*nvir*2 * 2 + nocc**4*nvir**2*2 +    # Wijkl
            nocc*nvir**4*2 * 2 +                        # Wabcd
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 + nocc**3*nvir**3*2 +
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 +                         # Wiabj
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # t1
            nocc**3*nvir**2*2 * 2 + nocc**4*nvir**2*2 +
            nvir*(nvir+1)/2*nocc*(nocc+1)/2*nvir**2 * 2 +# vvvv
            nocc**2*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 2 +     # t2
            nocc**3*nvir**3*2 * 3 + nocc**3*nvir**2*2 * 4)      # Wiabj


def unpack_tril(tril):
    assert(tril.flags.c_contiguous)
    count = tril.shape[0]
    nd = int(numpy.sqrt(tril.shape[1]*2))
    mat = numpy.empty((count,nd,nd))
    libcc.CCunpack_tril(ctypes.c_int(count), ctypes.c_int(nd),
                        tril.ctypes.data_as(ctypes.c_void_p),
                        mat.ctypes.data_as(ctypes.c_void_p))
    return mat

def make_tau(t2, t1a, t1b, fac=1):
    nocc = t1a.shape[0]
    tau = numpy.empty(t2.shape)
    for i in range(nocc):
        tau[i] = t2[i]
        tau[i] += numpy.einsum('a,jb->jab', t1a[i]*fac, t1b)
    return tau

def make_theta(t2):
    nocc = t2.shape[0]
    theta = numpy.empty(t2.shape)
    for i in range(nocc):
        theta[i] = t2[i].transpose(0,2,1) * 2
        theta[i] -= t2[i]
    return theta


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf() # -76.0267656731

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
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    mcc.diis = residual_as_diis_errvec(mcc)
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

