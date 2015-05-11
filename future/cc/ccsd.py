#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import tempfile
import numpy
import h5py
import pyscf.lib as lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.cc import _ccsd

# t2 as ijba
#TODO: optimize diis extrapolation

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
    if cc.diis is not None:
        damp = cc.diis()
    else:
        damp = lambda t1,t2,*args: t1, t2

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, blksize)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = damp(t1, t2, istep, normt, eccsd-eold)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris, blksize)
        log.info('istep = %d, E(CCSD) = %.15g, dE = %.9g, norm(t1,t2) = %.6g',
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
    woooo = woooo.reshape(nocc,nocc,nocc,nocc).transpose(0,2,1,3) + eris.oOoO
    time1 = log.timer_debug1('woooo', *time0)

    for p0, p1 in prange(0, nocc, blksize):
# ==== read eris.ovvv ====
        eris_ovvv = _ccsd.unpack_tril(numpy.asarray( \
                eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1))
        eris_ovvv = eris_ovvv.reshape(p1-p0,nvir,nvir,nvir)
        eris_ooov = numpy.asarray(eris.ooov[p0:p1])

        #: g2 = 2 * eris_ovvv - eris_ovvv.transpose(0,2,1,3)
        #: fvv += numpy.einsum('kc,kcba->ab', t1[p0:p1], g2)
        g2 = _ccsd.make_g0213(eris_ovvv, eris_ovvv, 2, -1)
        fvv += numpy.einsum('kc,kcba->ab', t1[p0:p1], g2)
        g2 = None

        #: g2 = 2 * eris.ooov[p0:p1] - eris.ooov[p0:p1].transpose(0,2,1,3)
        g2 = _ccsd.make_g0213(eris_ooov, eris_ooov, 2, -1)
        foo[:,p0:p1] += numpy.einsum('kc,jikc->ij', t1, g2)
        g2 = None

    #: tau = t2 + numpy.einsum('ia,jb->ijba', t1, t1)
    #: tmp = numpy.einsum('jicd,kcdb->kijb', tau, eris.ovvv)
    #: t2new += numpy.einsum('ka,kijb->ijba', -t1, tmp)
        #: eris_vovv = eris_ovvv.transpose(3,0,1,2).copy()
        eris_vovv = lib.transpose(eris_ovvv.reshape(-1,nvir))
        eris_vovv = eris_vovv.reshape(nvir,p1-p0,nvir,nvir)
        tmp = numpy.empty((nocc,nocc,nvir,p1-p0))
        for j0, j1 in prange(0, nocc, blksize):
            #: tau = t2[j0:j1] + numpy.einsum('ia,jb->ijba', t1[j0:j1], t1)
            tau = _ccsd.make_tau(t2[j0:j1], t1[j0:j1], t1)
            #: tmp[j0:j1] += numpy.einsum('ijcd,bkcd->ijbk', tau, eris_vovv)
            lib.dot(tau.reshape(-1,nvir*nvir),
                    eris_vovv.reshape(-1,nvir*nvir).T, 1,
                    tmp[j0:j1].reshape((j1-j0)*nocc,-1), 0)
        #: t2new += numpy.einsum('ka,ijbk->ijba', -t1[p0:p1], tmp)
        lib.dot(tmp.transpose(1,0,2,3).reshape(-1,p1-p0),
                t1[p0:p1], -1, t2new.reshape(-1,nvir), 1)
        tmp = eris_vovv = None
        #==== mem usage blksize*(nvir**3*2+nvir*nocc**2*2)

    #: woOVv += numpy.einsum('jabc,ic->jiba', eris.ovvv, t1)
    #: woOVv -= numpy.einsum('ikja,kb->jiba', eris.ooov, t1)
    #: t2new += woOVv
        #: woOVv = -numpy.einsum('kija,kb->ijab', eris.ooov[:,:,p0:p1], t1)
        tmp = numpy.asarray(eris.ovoo[p0:p1].transpose(2,0,1,3), order='C')
        woOVv = lib.dot(tmp.reshape(-1,nocc), t1, -1).reshape(nocc,p1-p0,nvir,nvir)
        #: woOVv += numpy.einsum('jabc,ic->ijab', eris_ovvv, t1)
        lib.dot(t1, eris_ovvv.reshape(-1,nvir).T, 1, woOVv.reshape(nocc,-1), 1)
        woOVv = woOVv.transpose(1,0,3,2).copy()
        t2new[p0:p1] += woOVv
        tmp = None

        #: woovv = numpy.einsum('ka,ijkb->ijba', t1, eris.ooov[p0:p1])
        #: woovv -= numpy.einsum('jc,icab->ijab', t1, eris_ovvv)
        woovv = lib.dot(numpy.asarray(eris_ooov.transpose(0,1,3,2), order='C').reshape(-1,nocc),
                        t1).reshape(p1-p0,nocc,nvir,nvir)
        for i in range(eris_ovvv.shape[0]):
            lib.dot(t1, eris_ovvv[i].reshape(nvir,-1), -1,
                    woovv[i].reshape(nocc,-1), 1)

    #: theta = t2 * 2 - t2.transpose(0,1,3,2)
    #: t1new += numpy.einsum('ijcb,jcba->ia', theta, eris.ovvv)
        #: theta = t2[p0:p1] * 2 - t2[p0:p1].transpose(0,1,3,2)
        theta = _ccsd.make_g0132(t2[p0:p1], t2[p0:p1], 2, -1)
        #: t1new += numpy.einsum('jibc,jcba->ia', theta, eris_ovvv)
        lib.dot(theta.transpose(1,0,3,2).reshape(nocc,-1),
                eris_ovvv.reshape(-1,nvir), 1, t1new, 1)
        eris_ovvv = None
        time2 = log.timer_debug1('ovvv [%d:%d]'%(p0, p1), *time1)
        #==== mem usage blksize*(nvir**3+nocc*nvir**2*4)

# ==== read eris.oOVv ====
        eris_oOVv = numpy.asarray(eris.oOVv[p0:p1])
        #==== mem usage blksize*(nocc*nvir**2*4)

        t2new[p0:p1] += eris_oOVv * .5
    #: g2 = 2 * eris.oOVv - eris.oOVv.transpose(1,0,2,3)
    #: fov = fock[:nocc,nocc:] + numpy.einsum('kc,ikca->ia', t1, g2)
        g2 = _ccsd.make_g0132(eris_oOVv, eris_oOVv, 2, -1)
        fov[p0:p1] += numpy.einsum('kc,ikca->ia', t1, g2)
        g2 = None
        #==== mem usage blksize*(nocc*nvir**2*5)

    #: theta = t2 * 2 - t2.transpose(1,0,2,3)
    #: t1new += numpy.einsum('jb,ijba->ia', fov, theta)
    #: t1new -= numpy.einsum('ikjb,kjba->ia', eris.ooov, theta)
        #theta = t2[p0:p1] * 2 - t2[p0:p1].transpose(0,1,3,2)
        t1new += numpy.einsum('ia,ijba->jb', fov[p0:p1], theta)
        #: t1new -= numpy.einsum('kijb,kjba->ia', eris.ooov[p0:p1], theta)
        lib.dot(eris_ooov.transpose(1,0,2,3).reshape(nocc,-1),
                theta.reshape(-1,nvir), -1, t1new, 1)
        eris_ooov = None

    #: woOVv += eris.oOVv
    #: theta = t2 * 2 - t2.transpose(1,0,2,3)
    #: tau = theta - numpy.einsum('ic,kb->ikcb', t1, t1*2)
    #: woOVv += .5 * numpy.einsum('jkca,ikcb->jiba', eris.oOVv, tau)
    #: woOVv -= .5 * numpy.einsum('jkac,ikcb->jiba', eris.oOVv, t2)
    #: t2new += numpy.einsum('ikca,kjbc->ijba', theta, woOVv)
        theta = theta.transpose(1,2,3,0).copy()
        woOVv += eris_oOVv
        woOVv = woOVv.transpose(1,2,3,0).copy()
        eris_OVvo = eris_oOVv.transpose(1,2,3,0).reshape(nov,-1)
        eris_VoOv = eris_oOVv.transpose(2,0,1,3).reshape(-1,nov)
        for j0, j1 in prange(0, nocc, blksize):
            t2iajb = t2[j0:j1].transpose(0,3,1,2).copy()
            #: woOVv[j0:j1] -= .5 * numpy.einsum('ikac,jbkc->jbai', eris_oOVv, t2iajb)
            lib.dot(t2iajb.reshape(-1,nov), eris_VoOv.T,
                    -.5, woOVv[j0:j1].reshape((j1-j0)*nvir,-1), 1)
            #: tau = t2iajb*2 - t2[j0:j1].transpose(0,2,1,3) \
            #:         - numpy.einsum('ic,kb->ibkc', t1[j0:j1]*2, t1)
            tau = _ccsd.make_tau(t2[j0:j1], t1[j0:j1]*2, t1)
            tau = _ccsd.make_g0213(t2iajb, tau, 2, -1, inplace=True)
            #tau = t2iajb*2 - tau.transpose(0,2,1,3)
            #: woOVv[j0:j1] += .5 * numpy.einsum('ikca,jbkc->jbai', eris_oOVv, tau)
            lib.dot(tau.reshape(-1,nov), eris_OVvo,
                    .5, woOVv[j0:j1].reshape((j1-j0)*nvir,-1), 1)

            #theta = t2[p0:p1] * 2 - t2[p0:p1].transpose(0,1,3,2)
            #t2new[j0:j1] += numpy.einsum('kiac,jbck->jiab', theta, woOVv[j0:j1])
            _ccsd.madd_admn_bcmn(woOVv[j0:j1], theta, t2new[j0:j1])
            #==== mem usage blksize*(nocc*nvir**2*8)
        t2iajb = tau = None
        theta = woOVv = eris_VoOv = eris_OVvo = None
        time2 = log.timer_debug1('woOVv [%d:%d]'%(p0, p1), *time2)
        #==== mem usage blksize*(nocc*nvir**2*2)

    #: fvv -= numpy.einsum('ijca,ijcb->ab', theta, eris.oOVv)
    #: foo += numpy.einsum('ikba,jkba->ij', eris.oOVv, theta)
        #: tau = t2[p0:p1] + numpy.einsum('ia,jb->ijba', t1[p0:p1], t1*.5)
        tau = _ccsd.make_tau(t2[p0:p1], t1[p0:p1]*.5, t1)
        #: theta = tau * 2 - tau.transpose(0,1,3,2)
        theta = _ccsd.make_g0132(tau, tau, 2, -1)
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
        tmp = lib.dot(eris_oovv.reshape(-1,nvir),
                      t1.T).reshape(p1-p0,nocc,nvir,nocc).transpose(0,3,2,1).copy()
        lib.dot(eris_oOVv.reshape(-1,nvir), t1.T, 1, tmp.reshape(-1,nocc), 1)
        tmp = numpy.asarray(tmp.transpose(3,1,2,0), order='C')
        #: t2new += numpy.einsum('ka,ijbk->ijba', -t1[p0:p1], tmp)
        lib.dot(tmp.reshape(-1,p1-p0), t1[p0:p1], -1, t2new.reshape(-1,nvir), 1)
        tmp = None

    #: g2 = 2 * eris.oOVv - eris.oovv
    #: t1new += numpy.einsum('jb,ijba->ia', t1, g2)
        g2 = 2 * eris_oOVv - eris_oovv
        t1new[p0:p1] += numpy.einsum('jb,ijba->ia', t1, g2)
        g2 = None

    #: woVoV -= eris.oovv
    #: tau = .5*t2 + numpy.einsum('ia,jb->ijba', t1, t1)
    #: woVoV += numpy.einsum('ka,ijkb->ijab', t1, eris.ooov)
    #: woVoV += numpy.einsum('kjca,ikbc->ijab', tau, eris.oOVv)
        woovv -= eris_oovv
        woovv = woovv.transpose(1,3,0,2).copy()
        eris_oVOv = eris_oOVv.transpose(0,2,1,3).reshape(-1,nov)
        #==== mem usage blksize*(nocc*nvir**2*3)

    #: tau = t2 + numpy.einsum('ia,jb->ijba', t1, t1)
    #: woooo += numpy.einsum('ijba,klba->ijkl', eris.oOVv, tau)
        for j0, j1 in prange(0, nocc, blksize):
            #: tau = t2[j0:j1] + numpy.einsum('ia,jb->ijba', t1[j0:j1], t1)
            tau = _ccsd.make_tau(t2[j0:j1], t1[j0:j1], t1)
            #: woooo[p0:p1,:,j0:j1] += numpy.einsum('ijba,klba->ijkl', eris_oOVv, tau)
            lib.numpy_helper._dgemm('N', 'T', (p1-p0)*nocc, (j1-j0)*nocc, nvir*nvir,
                                    eris_oOVv.reshape(-1,nvir*nvir),
                                    tau.reshape(-1,nvir*nvir),
                                    woooo[p0:p1].reshape(-1,nocc*nocc), 1, 1,
                                    0, 0, j0*nocc)

            tau -= .5*t2[j0:j1]
            #: woovv[j0:j1] += numpy.einsum('jkac,ikbc->jaib', tau, eris_oOVv)
            tau = tau.transpose(0,2,1,3).reshape(-1,nov)
            lib.dot(tau, eris_oVOv.T,
                    1, woovv[j0:j1].reshape((j1-j0)*nvir,-1), 1)
            #==== mem usage blksize*(nocc*nvir**2*5)
        time2 = log.timer_debug1('woovv [%d:%d]'%(p0, p1), *time2)

        #: tau = t2[p0:p1] + numpy.einsum('ia,jb->ijba', t1[p0:p1], t1)
        tau = _ccsd.make_tau(t2[p0:p1], t1[p0:p1], t1)
        #: t2new += .5 * numpy.einsum('klij,klba->ijba', woooo[p0:p1], tau)
        lib.dot(woooo[p0:p1].reshape(-1,nocc*nocc).T,
                tau.reshape(-1,nvir*nvir), .5,
                t2new.reshape(nocc*nocc,-1), 1)
        eris_oovv = eris_oOVv = eris_oVOv = tau = None
        #==== mem usage blksize*(nocc*nvir**2*1)

        t2ibja = t2[p0:p1].transpose(0,2,1,3).copy()
        for j0, j1 in prange(0, nocc, blksize):
            #: t2new[j0:j1] += numpy.einsum('jbkc,kcia->jiba', woovv[j0:j1], t2ibja)
            _ccsd.madd_acmn_mnbd(woovv[j0:j1], t2ibja, t2new[j0:j1])

        t2iajb = t2[p0:p1].transpose(0,3,1,2).copy()
        for j0, j1 in prange(0, nocc, blksize):
            #: t2new[j0:j1] += numpy.einsum('jbkc,kcia->jiab', woovv[j0:j1], t2iajb)
            _ccsd.madd_admn_mnbc(woovv[j0:j1], t2iajb, t2new[j0:j1])
        t2ibja = t2iajb = woovv = None
        #==== mem usage blksize*(nocc*nvir**2*3)
        time1 = log.timer_debug1('contract occ [%d:%d]'%(p0, p1), *time1)
# ==================
    time1 = log.timer_debug1('contract loop', *time0)

    woooo = None
    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    #: t2new += numpy.einsum('ijbc,ac->ijba', t2, ft_ab)
    #: t2new -= numpy.einsum('ki,kjba->ijba', ft_ij, t2)
    lib.dot(t2.reshape(-1,nvir), ft_ab.T, 1, t2new.reshape(-1,nvir), 1)
    lib.dot(ft_ij.T, t2.reshape(nocc,-1),-1, t2new.reshape(nocc,-1), 1)

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    t2new_tril = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
    ij = 0
    for i in range(nocc):
        for j in range(i+1):
            t2new_tril[ij] = t2new[i,j] + t2new[j,i].T
            ij += 1
    t2new = None
    time1 = log.timer_debug1('t2 tril', *time1)
    cc.add_wvvVV_(t1, t2, eris, t2new_tril, blksize)
    time1 = log.timer_debug1('vvvv', *time1)

    mo_e = fock.diagonal()
    eia = (mo_e[:nocc,None] - mo_e[None,nocc:])
    p0 = 0
    for i in range(nocc):
        djba = (eia[:i+1].reshape(-1,1) + eia[i].reshape(1,-1))
        t2new_tril[p0:p0+i+1] /= djba.reshape(i+1,nvir,nvir)
        p0 += i+1
    time1 = log.timer_debug1('g2/dijba', *time1)

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
    for p0, p1 in prange(0, nocc, blksize):
        tau = _ccsd.make_tau(t2[p0:p1], t1[p0:p1], t1)
        #theta = tau*2 - tau.transpose(0,1,3,2)
        theta = _ccsd.make_g0132(tau, tau, 2, -1)
        e += numpy.einsum('ijab,ijab', eris.oOVv[p0:p1], theta)
    return e


class CCSD(object):
    def __init__(self, mf):
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
        self.diis_start_cycle = 1
        self.diis_start_energy_diff = 1e-2

        self.nocc = self.mol.nelectron // 2
        self.nmo = len(mf.mo_energy)

        self._conv = False
        self.emp2 = None
        self.ecc = None
        self.t1 = None
        self.t2 = None

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
            djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).reshape(-1)
            gi = eris.oOVv[i]
            t2i = (gi.reshape(-1)/djba).reshape(nocc,nvir,nvir)
            t2[i] = t2i
            self.emp2 += 4 * numpy.einsum('jab,jab', t2i[:i], gi[:i])
            self.emp2 += 2 * numpy.einsum('ab,ab'  , t2i[i] , gi[i] )
            self.emp2 -= 2 * numpy.einsum('jab,jba', t2i[:i], gi[:i])
            self.emp2 -=     numpy.einsum('ab,ba'  , t2i[i] , gi[i] )

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2


    def kernel(self, t1=None, t2=None):
        return self.ccsd(t1, t2)
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
            logger.info(self, ' E(CCSD) = %.16g, E_corr = %.16g',
                        self.ecc+self._scf.hf_energy+self.mol.energy_nuc(),
                        self.ecc)
        else:
            logger.info(self, 'CCSD not converge')
            logger.info(self, ' E(CCSD) = %.16g, E_corr = %.16g',
                        self.ecc+self._scf.hf_energy+self.mol.energy_nuc(),
                        self.ecc)
        logger.timer(self, 'CCSD', *cput0)
        return self.ecc, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        #nocc = self.nocc
        #nmo = self.nmo
        #nvir = nmo - nocc
        #eri1 = pyscf.ao2mo.incore.full(self._scf._eri, self._scf.mo_coeff)
        #eri1 = pyscf.ao2mo.restore(1, eri1, nmo)
        #eris = lambda:None
        #eris.oOoO = eri1[:nocc,:nocc,:nocc,:nocc].transpose(0,2,1,3).copy()
        #eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        #eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        #eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        #eris.oOVv = eri1[:nocc,nocc:,:nocc,nocc:].transpose(0,2,3,1).copy()
        #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        #eris.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
        #for i in range(nocc):
        #    for j in range(nvir):
        #        eris.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
        #eris.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:].copy(), nvir)
        #eris.fock = numpy.diag(self._scf.mo_energy)
        #return eris
        return _ERIS(self, mo_coeff)

    def add_wvvVV_(self, t1, t2, eris, t2new, blksize=1):
        time0 = time.clock(), time.time()
        nocc = self.nocc
        nvir = self.nmo - nocc
        #: tau = t2 + numpy.einsum('ia,jb->ijba', t1, t1)
        #: t2new += numpy.einsum('ijdc,bdca->ijba', tau, vvvv)
        tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
        p0 = 0
        for i in range(nocc):
            #: tau[p0:p0+i+1] += numpy.einsum('a,jb->jba', t1[i], t1[:i+1])
            tau[p0:p0+i+1] = _ccsd.make_tau(t2[i,:i+1], t1[i:i+1], t1[:i+1])
            p0 += i + 1
        time0 = logger.timer_debug1(self, 'vvvv-tau', *time0)

        p0 = 0
        for b in range(nvir):
            buf = _ccsd.unpack_tril(eris.vvvv[p0:p0+b+1])
            #: t2new[i,:i+1, b] += numpy.einsum('xdc,dca->xa', tau[:,:b+1], buf)
            lib.numpy_helper._dgemm('N', 'N', nocc*(nocc+1)//2, nvir, (b+1)*nvir,
                                    tau.reshape(-1,nvir*nvir), buf.reshape(-1,nvir),
                                    t2new.reshape(-1,nvir*nvir), 1, 1,
                                    0, 0, b*nvir)

            #: t2new[i,:i+1,:b] += numpy.einsum('xc,bac->xba', tau[:,b], buf[:b])
            if b > 0:
                lib.numpy_helper._dgemm('N', 'T', nocc*(nocc+1)//2, b*nvir, nvir,
                                        tau.reshape(-1,nvir*nvir), buf.reshape(-1,nvir),
                                        t2new.reshape(-1,nvir*nvir), 1, 1,
                                        b*nvir, 0, 0)
            p0 += b+1
            time0 = logger.timer_debug1(self, 'vvvv %d'%b, *time0)
        return t2new
    def add_wvvVV(self, t1, t2, eris, blksize=1):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t2new = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
        return self.add_wvvVV_(t1, t2, eris, t2new, blksize=1)

    def get_block_size(self):
        #return 8
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        unit = _memory_usage_inloop(nmo, nocc)*1e6/8
        rest = (self.max_memory-lib.current_memory()[0])*1e6/8*.9 \
                - nocc**4 - nocc**3*nvir - (nocc*nvir)**2*2
        return min(nocc, max(1, int(rest/unit/8)*8))

    def update_amps(self, *args, **kwargs):
        return update_amps(self, *args, **kwargs)

    def diis(self):
        return self.diis_()
    def diis_(self):
        damp = lib.diis.DIIS(self)
        damp.space = self.diis_space
        damp.min_space = 1
        def fupdate_(t1, t2, istep, normt, de):
            if (istep > self.diis_start_cycle and
                abs(de) < self.diis_start_energy_diff):
                t1t2 = numpy.hstack((t1.ravel(),t2.ravel()))
#NOTE: here overwriting .data to reduce memory usage, but the contents of
# t1, t2 are CHANGED!  If the pass-in t1/t2 are used elsewhere, be very
# careful to call this function
                t1.data = t1t2.data
                t2.data = t1t2.data
                t1t2 = damp.update(t1t2)
                t1 = t1t2[:t1.size].reshape(t1.shape)
                t2 = t1t2[t1.size:].reshape(t2.shape)
                logger.debug(self, 'DIIS for step %d', istep)
            return t1, t2
        return fupdate_

CC = CCSD

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        if mo_coeff is None: mo_coeff = cc._scf.mo_coeff
        nocc = cc.nocc
        nmo = mo_coeff.shape[1]
        nvir = nmo - nocc
        log = logger.Logger(cc.stdout, cc.verbose)
        self.fock = numpy.diag(cc._scf.mo_energy)
        if ((cc._scf._eri is not None) and (method == 'incore') and
            ((nocc*nvir**3/2 + nvir**4/4 + nocc**2*nvir**2*2)*8/1e6 +
             _memory_usage(nmo, nocc) + lib.current_memory()[0]) < cc.max_memory):
            eri1 = pyscf.ao2mo.incore.full(cc._scf._eri, mo_coeff)
            eri1 = pyscf.ao2mo.restore(1, eri1, nmo)
            self.oOoO = eri1[:nocc,:nocc,:nocc,:nocc].transpose(0,2,1,3).copy()
            self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
            self.oOVv = eri1[:nocc,nocc:,:nocc,nocc:].transpose(0,2,3,1).copy()
            ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            for i in range(nocc):
                for j in range(nvir):
                    self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
        else:
            time0 = time.clock(), time.time()
            _tmpfile1 = tempfile.NamedTemporaryFile()
            _tmpfile2 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            nvpair = nvir * (nvir+1) // 2
            self.oOoO = self.feri1.create_dataset('oOoO', (nocc,nocc,nocc,nocc), 'f8')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'f8')
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8')
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8')
            self.oOVv = self.feri1.create_dataset('oOVv', (nocc,nocc,nvir,nvir), 'f8')
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
                    buf = _ccsd.unpack_tril(numpy.asarray(eri1[i*nmo:(i+1)*nmo]))
                    self.oOoO[i] = buf[:nocc,:nocc,:nocc].transpose(1,0,2)
                    self.ooov[i] = buf[:nocc,:nocc,nocc:]
                    self.ovoo[i] = buf[nocc:,:nocc,:nocc]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.oOVv[i] = buf[nocc:,:nocc,nocc:].transpose(1,2,0)
                    for j in range(nvir):
                        self.ovvv[i,j] = lib.pack_tril(buf[nocc+j,nocc:,nocc:])
                    time1 = log.timer_debug1('sorting %d'%i, *time1)
            self.oOoO = numpy.asarray(self.oOoO)

    def __del__(self):
        if hasattr(self, 'feri1'):
            self.feri1.close()
            self.feri2.close()


# assume nvir > nocc, minimal requirements on memory in loop of update_amps
def _memory_usage_inloop(nmo, nocc):
    nvir = nmo - nocc
    v = max(nvir**3*2+nvir*nocc**2*2,
            nvir**3+nocc*nvir**2*5+nvir*nocc**2,
            nocc*nvir**2*9)
    return v*8/1e6
# assume nvir > nocc, minimal requirements on memory
def _memory_usage(nmo, nocc):
    nvir = nmo - nocc
    v = _memory_usage_inloop(nmo, nocc)*1e6/8 + nocc**4 + nocc**3*nvir
    v = max(v, nocc*(nocc+1)//2*nvir**2) + (nocc*nvir)**2*2
    return v*8/1e6

def residual_as_diis_errvec(mycc):
    def f_CC_diis():
        nocc = mycc.nocc
        nvir = mycc.nmo - nocc
        nov = nocc*nvir
        damp = lib.diis.DIIS(mycc)
        damp.space = mycc.diis_space
        damp.min_space = 1
        damp.last_t1 = None
        damp.last_t2 = None
        mo_e = mycc._scf.mo_energy
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        def fupdate(t1, t2, istep, normt, de):
            if (istep > mycc.diis_start_cycle and
                abs(de) < mycc.diis_start_energy_diff):
                if damp.last_t1 is None:
                    damp.last_t1 = t1
                    damp.last_t2 = t2
                else:
                    tbuf = numpy.empty(nov*(nov+1))
                    tbuf[:nov] = ((t1-damp.last_t1)*eia).ravel()
                    pbuf = tbuf[nov:].reshape(nocc,nocc,nvir,nvir)
                    for i in range(nocc):
                        djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).reshape(-1)
                        pbuf[i] = (t2[i]-damp.last_t2[i]) * djba.reshape(nocc,nvir,nvir)
                    damp.push_err_vec(tbuf)
                    tbuf = numpy.empty(nov*(nov+1))
                    tbuf[:nov] = t1.ravel()
                    tbuf[nov:] = t2.ravel()
                    t1.data = tbuf.data # release memory
                    t2.data = tbuf.data

                    tbuf = damp.update(tbuf)
                    damp.last_t1 = t1 = tbuf[:nov].reshape(nocc,nvir)
                    damp.last_t2 = t2 = tbuf[nov:].reshape(nocc,nocc,nvir,nvir)
                logger.debug(mycc, 'DIIS for step %d', istep)
            return t1, t2
        return fupdate
    return f_CC_diis


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

