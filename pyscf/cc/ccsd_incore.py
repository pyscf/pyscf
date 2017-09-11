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
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd

BLKMIN = 4

# t2 as ijab

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=logger.INFO):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(cc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    nov = nocc*nvir
    fock = eris.fock

    t1new = numpy.zeros_like(t1)
    t2new = numpy.zeros_like(t2)
    t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
    cc.add_wvvVV_(t1, t2, eris, t2new_tril)
    time1 = log.timer_debug1('vvvv', *time0)
    ij = 0
    for i in range(nocc):
        for j in range(i+1):
            t2new[i,j] = t2new_tril[ij]
            ij += 1
        t2new[i,i] *= .5
    t2new_tril = None

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
    woooo = lib.dot(eris_ooov.reshape(-1,nvir), t1.T).reshape((nocc,)*4)
    woooo = lib.transpose_sum(woooo.reshape(nocc*nocc,-1), inplace=True)
    woooo += _cp(eris.oooo).reshape(nocc**2,-1)
    woooo = _cp(woooo.reshape(nocc,nocc,nocc,nocc).transpose(0,2,1,3))
    time1 = log.timer_debug1('woooo', *time0)

    eris_ovvv = _cp(eris.ovvv)
    eris_ovvv = lib.unpack_tril(eris_ovvv.reshape(nov,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)

    fvv += numpy.einsum('kc,kcba->ab', 2*t1, eris_ovvv)
    fvv += numpy.einsum('kc,kbca->ab',  -t1, eris_ovvv)

    #: woVoV = numpy.einsum('ka,ijkb->ijba', t1, eris.ooov)
    #: woVoV -= numpy.einsum('jc,icab->ijab', t1, eris_ovvv)
    woVoV = lib.dot(_cp(eris_ooov.transpose(0,1,3,2).reshape(-1,nocc)), t1)
    woVoV = woVoV.reshape(nocc,nocc,nvir,nvir)

#: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
#: tmp = numpy.einsum('ijcd,kcdb->kijb', tau, eris.ovvv)
#: t2new += numpy.einsum('ka,kijb->jiba', -t1, tmp)
    tau = make_tau(t2, t1, t1)
    tmp = numpy.empty((nocc,nocc,nocc,nvir))
    for k in range(nocc):
        tmp[k] = lib.dot(tau.reshape(-1,nvir**2),
                         eris_ovvv[k].reshape(-1,nvir)).reshape(nocc,nocc,nvir).transpose(1,0,2)
        lib.dot(t1, eris_ovvv[k].reshape(nvir,-1), -1, woVoV[k].reshape(nocc,-1), 1)
    lib.dot(tmp.reshape(nocc,-1).T, t1, -1, t2new.reshape(-1,nvir), 1)
    tmp = None

#: wOVov += numpy.einsum('iabc,jc->ijab', eris.ovvv, t1)
#: wOVov -= numpy.einsum('jbik,ka->jiba', eris.ovoo, t1)
#: t2new += woVoV.transpose()
    #: wOVov = -numpy.einsum('jbik,ka->ijba', eris.ovoo, t1)
    wOVov, tau = tau, None
    lib.dot(_cp(_cp(eris.ooov).transpose(0,2,3,1).reshape(-1,nocc)), t1,
            -1, wOVov.reshape(-1,nvir))
    #: wOVov += numpy.einsum('iabc,jc->jiab', eris_ovvv, t1)
    lib.dot(t1, eris_ovvv.reshape(-1,nvir).T, 1, wOVov.reshape(nocc,-1), 1)
    for i in range(nocc):
        t2new[i] += wOVov[i].transpose(0,2,1)

#: theta = t2.transpose(0,1,3,2) * 2 - t2
#: t1new += numpy.einsum('ijcb,jcba->ia', theta, eris.ovvv)
    theta = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        theta[i] = t2[i].transpose(0,2,1) * 2
        theta[i] -= t2[i]
        lib.dot(_cp(theta[i].transpose(0,2,1).reshape(nocc,-1)),
                eris_ovvv[i].reshape(-1,nvir), 1, t1new, 1)
    eris_ovvv = None

    eris_ovov = _cp(eris.ovov)

    for i in range(nocc):
        t2new[i] += eris_ovov[i].transpose(1,0,2) * .5

    fov += numpy.einsum('kc,iakc->ia', t1, eris_ovov) * 2
    fov -= numpy.einsum('kc,icka->ia', t1, eris_ovov)

#: theta = t2.transpose(1,0,2,3) * 2 - t2
#: t1new += numpy.einsum('jb,ijab->ia', fov, theta)
#: t1new -= numpy.einsum('ikjb,kjab->ia', eris.ooov, theta)
    t1new += numpy.einsum('jb,jiab->ia', fov, theta)
    #: t1new -= numpy.einsum('kijb,kjab->ia', eris.ooov, theta)
    lib.dot(_cp(eris_ooov.transpose(1,0,2,3).reshape(nocc,-1)),
            theta.reshape(-1,nvir), -1, t1new, 1)
    eris_ooov = None

#: wOVov += eris.ovov.transpose(0,1,3,2)
#: theta = t2.transpose(1,0,2,3) * 2 - t2
#: tau = theta - numpy.einsum('ic,kb->ikcb', t1, t1*2)
#: wOVov += .5 * numpy.einsum('jakc,ikcb->jiba', eris.ovov, tau)
#: wOVov -= .5 * numpy.einsum('jcka,ikcb->jiba', eris.ovov, t2)
#: t2new += numpy.einsum('ikca,kjbc->ijba', theta, wOVov)
    wOVov = _cp(wOVov.transpose(0,3,1,2))
    eris_OVov = lib.transpose(eris_ovov.reshape(-1,nov)).reshape(nocc,nvir,-1,nvir)
    eris_OvoV = _cp(eris_OVov.transpose(0,3,2,1))
    wOVov += eris_OVov
    t2iajb = t2.transpose(0,2,1,3).copy()
    #: wOVov[j0:j1] -= .5 * numpy.einsum('iakc,jkbc->jbai', eris_ovov, t2)
    lib.dot(t2iajb.reshape(-1,nov), eris_OvoV.reshape(nov,-1),
            -.5, wOVov.reshape(nov,-1), 1)
    tau, t2iajb = t2iajb, None
    for i in range(nocc):
        tau[i] = tau[i]*2 - t2[i].transpose(2,0,1)
        tau[i] -= numpy.einsum('a,jb->bja', t1[i]*2, t1)
    #: wOVov += .5 * numpy.einsum('iakc,jbkc->jbai', eris_ovov, tau)
    lib.dot(tau.reshape(-1,nov), eris_OVov.reshape(nov,-1),
            .5, wOVov.reshape(nov,-1), 1)

    #theta = t2 * 2 - t2.transpose(0,1,3,2)
    #: t2new[j0:j1] += numpy.einsum('iack,jbck->jiba', theta, wOVov[j0:j1])
    tmp, tau = tau, None
    theta = _cp(theta.transpose(0,3,1,2).reshape(nov,-1))
    lib.dot(wOVov.reshape(nov,-1), theta.T, 1, tmp.reshape(nov,-1))
    for i in range(nocc):
        t2new[i] += tmp[i].transpose(1,0,2)
    tmp = wOVov = eris_OvoV = eris_OVov = None

#: fvv -= numpy.einsum('ijca,ibjc->ab', theta, eris.ovov)
#: foo += numpy.einsum('iakb,jkba->ij', eris.ovov, theta)
    for i in range(nocc):
        tau = numpy.einsum('a,jb->jab', t1[i]*.5, t1) + t2[i]
        theta = tau.transpose(0,2,1)*2 - tau
        lib.dot(_cp(eris_ovov[i].transpose(1,2,0)).reshape(nocc,-1),
                theta.reshape(nocc,-1).T, 1, foo, 1)
        lib.dot(theta.reshape(-1,nvir).T,
                eris_ovov[i].reshape(nvir,-1).T, -1, fvv, 1)
    tau = theta = None

    eris_oovv = _cp(eris.oovv)
    #:tmp = numpy.einsum('ic,jkbc->jibk', t1, eris_oovv)
    #:t2new += numpy.einsum('ka,jibk->jiab', -t1, tmp)
    #:tmp = numpy.einsum('ic,jbkc->jibk', t1, eris_ovov)
    #:t2new += numpy.einsum('ka,jibk->jiba', -t1, tmp)
    for j in range(nocc):
        tmp = lib.dot(t1, eris_oovv[j].reshape(-1,nvir).T)
        tmp = _cp(tmp.reshape(nocc,nocc,nvir).transpose(0,2,1))
        t2new[j] += lib.dot(tmp.reshape(-1,nocc), t1,
                            -1).reshape(nocc,nvir,nvir).transpose(0,2,1)
        lib.dot(t1, eris_ovov[j].reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1))
        lib.dot(tmp.reshape(-1,nocc), t1, -1, t2new[j].reshape(-1,nvir), 1)
    tmp = None

#: g2 = 2 * eris.oOVv - eris.oovv
#: t1new += numpy.einsum('jb,ijba->ia', t1, g2)
    t1new += numpy.einsum('jb,iajb->ia', 2*t1, eris_ovov)
    t1new += numpy.einsum('jb,ijba->ia',  -t1, eris_oovv)

#: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
#: woooo += numpy.einsum('ijba,klab->ijkl', eris.oOVv, tau)
#: woVoV -= eris.oovv
#: tau = .5*t2 + numpy.einsum('ia,jb->ijab', t1, t1)
#: woVoV += numpy.einsum('ka,ijkb->ijab', t1, eris.ooov)
#: woVoV += numpy.einsum('jkca,ikbc->ijab', tau, eris.oOVv)
    woVoV -= eris_oovv
    woVoV = woVoV.transpose(1,3,0,2).copy()
    eris_oVOv = _cp(eris_ovov.transpose(0,3,2,1))
    eris_oOvV = _cp(eris_ovov.transpose(0,2,1,3))

    tau = make_tau(t2, t1, t1)
    #: woooo += numpy.einsum('ijab,klab->ijkl', eris_oOvV, tau)
    lib.dot(eris_oOvV.reshape(-1,nvir**2), tau.reshape(-1,nvir**2).T,
            1, woooo.reshape(nocc**2,-1), 1)
    #: t2new += .5 * numpy.einsum('klij,klab->ijab', woooo, tau)
    lib.dot(woooo.reshape(-1,nocc*nocc).T, tau.reshape(-1,nvir*nvir),
            .5, t2new.reshape(nocc*nocc,-1), 1)
    for i in range(nocc):
        tau[i] -= t2[i] * .5
    #: woVoV[j0:j1] += numpy.einsum('jkca,ickb->jiab', tau, eris_ovov)
    tau = _cp(tau.transpose(0,3,1,2))
    lib.dot(tau.reshape(-1,nov), eris_oVOv.reshape(-1,nov).T,
            1, woVoV.reshape(nov,-1), 1)
    eris_oovv = eris_ovov = eris_oOvV = taubuf = None

    tmp, tau = tau, None
    t2ibja, eris_oVOv = eris_oVOv, None
    for i in range(nocc):
        t2ibja[i] = t2[i].transpose(2,0,1)
    #: t2new += numpy.einsum('ibkc,kcja->ijab', woVoV, t2ibja)
    lib.dot(woVoV.reshape(nov,-1), t2ibja.reshape(-1,nov), 1, tmp.reshape(nov,-1))
    for i in range(nocc):
        t2new[i] += tmp[i].transpose(1,2,0)

    #: t2new[j0:j1] += numpy.einsum('iakc,kcjb->ijab', woVoV[j0:j1], t2iajb)
    t2iajb = t2ibja
    for i in range(nocc):
        t2iajb[i] = t2[i].transpose(1,0,2)
    lib.dot(woVoV.reshape(nov,-1), t2iajb.reshape(-1,nov), 1, tmp.reshape(nov,-1))
    for i in range(nocc):
        t2new[i] += tmp[i].transpose(1,0,2)
    t2ibja = t2iajb = woVoV = tmp = None
    time1 = log.timer_debug1('contract loop', *time0)

    woooo = None
    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    #: t2new += numpy.einsum('ijac,bc->ijab', t2, ft_ab)
    #: t2new -= numpy.einsum('ki,kjab->ijab', ft_ij, t2)
    lib.dot(t2.reshape(-1,nvir), ft_ab.T, 1, t2new.reshape(-1,nvir), 1)
    lib.dot(ft_ij.T, t2.reshape(nocc,-1),-1, t2new.reshape(nocc,-1), 1)

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
    #if hasattr(pyscf, 'MKL_NUM_THREADS'):
    #    pyscf._libmkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
    return t1new, t2new

def energy(cc, t1, t2, eris):
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


class CCSD(ccsd.CCSD):
    '''CCSD

    Args

    Returns
        t1[i,a]
        t2[i,j,a,b]
    '''

    def ccsd(self, t1=None, t2=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if self.converged:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.ecc, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, mo_coeff=None,
                     eris=None):
        from pyscf.cc import ccsd_lambda_incore
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda_incore.kernel(self, eris, t1, t2, l1, l2,
                                          max_cycle=self.max_cycle,
                                          tol=self.conv_tol_normt,
                                          verbose=self.verbose)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None):
        '''1-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm_incore
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm_incore.make_rdm1(self, t1, t2, l1, l2)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
        '''2-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm_incore
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return ccsd_rdm_incore.make_rdm2(self, t1, t2, l1, l2)

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)

CC = CCSD

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

    mcc.diis = ccsd.residual_as_diis_errvec(mcc)
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

