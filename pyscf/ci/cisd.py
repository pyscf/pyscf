#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Solve CISD equation  H C = C e  where e = E_HF + E_CORR
'''

import time
import ctypes
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.fci import cistring
from functools import reduce
_dgemm = lib.numpy_helper._dgemm

def kernel(myci, eris, ci0=None, max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(myci.stdout, verbose)

    mol = myci.mol
    nmo = myci.nmo
    nocc = myci.nocc
    nvir = nmo - nocc
    diag = myci.make_diagonal(eris)
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
        ci0 = myci.get_init_guess(eris)[1]

    def op(xs):
        return [myci.contract(x, eris) for x in xs]

    def precond(x, e, *args):
        diagd = diag - (e-myci.level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return x / diagd

    def cisd_dot(x1, x2):
        return dot(x1, x2, nocc, nvir)

    conv, ecisd, ci = lib.davidson1(op, ci0, precond, tol=tol,
                                    max_cycle=max_cycle, max_space=myci.max_space,
                                    lindep=myci.lindep, dot=cisd_dot,
                                    nroots=myci.nroots, verbose=log)
    if myci.nroots == 1:
        conv = conv[0]
        ecisd = ecisd[0]
        ci = ci[0]
    return conv, ecisd, ci

def make_diagonal(myci, eris):
    mo_energy = eris.fock.diagonal()
    nmo = mo_energy.size
    jdiag = numpy.zeros((nmo,nmo))
    kdiag = numpy.zeros((nmo,nmo))
    eris_oooo = _cp(eris.oooo)
    nocc = eris.nocc
    nvir = nmo - nocc
    jdiag[:nocc,:nocc] = numpy.einsum('iijj->ij', eris.oooo)
    kdiag[:nocc,:nocc] = numpy.einsum('jiij->ij', eris.oooo)
    jdiag[:nocc,nocc:] = numpy.einsum('iijj->ji', eris.vvoo)
    kdiag[:nocc,nocc:] = numpy.einsum('jiij->ij', eris.voov)
    if eris.vvvv is not None:
        #:eris_vvvv = ao2mo.restore(1, eris.vvvv, nvir)
        #:jdiag1 = numpy.einsum('iijj->ij', eris_vvvv)
        diag_idx = numpy.arange(nvir)
        diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
        for i, ii in enumerate(diag_idx):
            jdiag[nocc+i,nocc:] = eris.vvvv[ii][diag_idx]

    jksum = (jdiag[:nocc,:nocc] * 2 - kdiag[:nocc,:nocc]).sum()
    ehf = mo_energy[:nocc].sum() * 2 - jksum
    e_ia = lib.direct_sum('a-i->ia', mo_energy[nocc:], mo_energy[:nocc])
    e_ia -= jdiag[:nocc,nocc:] - kdiag[:nocc,nocc:]
    e1diag = ehf + e_ia
    e2diag = lib.direct_sum('ia+jb->ijab', e_ia, e_ia)
    e2diag += ehf
    e2diag += jdiag[:nocc,:nocc].reshape(nocc,nocc,1,1)
    e2diag -= jdiag[:nocc,nocc:].reshape(nocc,1,1,nvir)
    e2diag -= jdiag[:nocc,nocc:].reshape(1,nocc,nvir,1)
    e2diag += jdiag[nocc:,nocc:].reshape(1,1,nvir,nvir)
    return numpy.hstack((ehf, e1diag.reshape(-1), e2diag.reshape(-1)))

def contract(myci, civec, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(myci.stdout, myci.verbose)
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    nov = nocc * nvir
    noo = nocc**2
    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc)

    cinew = numpy.zeros_like(civec)
    t1 = cinew[1:nov+1].reshape(nocc,nvir)
    t2 = cinew[nov+1:].reshape(nocc,nocc,nvir,nvir)
    t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
    myci.add_wvvVV_(c2, eris, t2new_tril)
    for i in range(nocc):
        for j in range(i+1):
            t2[i,j] = t2new_tril[i*(i+1)//2+j]
        t2[i,i] *= .5
    t2new_tril = None
    time1 = log.timer_debug1('vvvv', *time0)
    #:t2 += numpy.einsum('iklj,klab->ijab', _cp(eris.oooo)*.5, c2)
    oooo = lib.transpose(_cp(eris.oooo).reshape(nocc,noo,nocc), axes=(0,2,1))
    lib.ddot(oooo.reshape(noo,noo), c2.reshape(noo,-1), .5, t2.reshape(noo,-1), 1)

    foo = eris.fock[:nocc,:nocc].copy()
    fov = eris.fock[:nocc,nocc:].copy()
    fvv = eris.fock[nocc:,nocc:].copy()

    t1+= fov * c0
    t1+= numpy.einsum('ib,ab->ia', c1, fvv)
    t1-= numpy.einsum('ja,ji->ia', c1, foo)

    #:t2 += numpy.einsum('bc,ijac->ijab', fvv, c2)
    #:t2 -= numpy.einsum('kj,kiba->ijab', foo, c2)
    #:t2 += numpy.einsum('ia,jb->ijab', c1, fov)
    lib.ddot(c2.reshape(-1,nvir), fvv, 1, t2.reshape(-1,nvir), 1)
    lib.ddot(foo, c2.reshape(nocc,-1),-1, t2.reshape(nocc,-1), 1)
    for j in range(nocc):
        t2[:,j] += numpy.einsum('ia,b->iab', c1, fov[j])

    unit = _memory_usage_inloop(nocc, nvir)
    max_memory = max(2000, myci.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)
    nvir_pair = nvir * (nvir+1) // 2
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_vvoo = _cp(eris.vvoo[p0:p1])
        oovv = lib.transpose(eris_vvoo.reshape(-1,nocc**2))
        #:eris_oVoV = eris_vvoo.transpose(2,0,3,1)
        eris_oVoV = numpy.ndarray((nocc,p1-p0,nocc,nvir))
        eris_oVoV[:] = oovv.reshape(nocc,nocc,p1-p0,nvir).transpose(0,2,1,3)
        eris_vvoo = oovv = None
        #:tmp = numpy.einsum('ikca,jbkc->jiba', c2, eris_oVoV)
        #:t2[:,:,p0:p1] -= tmp*.5
        #:t2[:,:,p0:p1] -= tmp.transpose(1,0,2,3)
        for i in range(nocc):
            tmp = lib.ddot(eris_oVoV.reshape(-1,nov), c2[i].reshape(nov,nvir))
            tmp = tmp.reshape(nocc,p1-p0,nvir)
            t2[:,i,p0:p1] -= tmp*.5
            t2[i,:,p0:p1] -= tmp

        eris_voov = _cp(eris.voov[p0:p1])
        for i in range(p0, p1):
            t2[:,:,i] += eris_voov[i-p0] * (c0 * .5)
        t1[:,p0:p1] += numpy.einsum('jb,aijb->ia', c1, eris_voov) * 2
        t1[:,p0:p1] -= numpy.einsum('jb,iajb->ia', c1, eris_oVoV)

        #:ovov = eris_voov.transpose(2,0,1,3) - eris_vvoo.transpose(2,0,3,1)
        ovov = eris_oVoV
        ovov *= -.5
        for i in range(nocc):
            ovov[i] += eris_voov[:,:,i]
        eris_oVoV = eris_vvoo = None
        #:theta = c2[:,:,p0:p1]
        #:theta = theta * 2 - theta.transpose(1,0,2,3)
        #:theta = theta.transpose(2,0,1,3)
        theta = numpy.ndarray((p1-p0,nocc,nocc,nvir))
        for i in range(p0, p1):
            theta[i-p0] = c2[:,:,i] * 2
            theta[i-p0]-= c2[:,:,i].transpose(1,0,2)
        #:t2 += numpy.einsum('ckia,jckb->ijab', theta, ovov)
        for j in range(nocc):
            tmp = lib.ddot(theta.reshape(-1,nov).T, ovov[j].reshape(-1,nvir))
            t2[:,j] += tmp.reshape(nocc,nvir,nvir)
        tmp = ovov = None

        t1[:,p0:p1] += numpy.einsum('aijb,jb->ia', theta, fov)

        eris_vooo = _cp(eris.vooo[p0:p1])
        #:t1 -= numpy.einsum('bjka,bjki->ia', theta, eris_vooo)
        #:t2[:,:,p0:p1] -= numpy.einsum('ka,bjik->jiba', c1, eris_vooo)
        lib.ddot(eris_vooo.reshape(-1,nocc).T, theta.reshape(-1,nvir), -1, t1, 1)
        for i in range(p0, p1):
            t2[:,:,i] -= lib.ddot(eris_vooo[i-p0].reshape(noo,-1), c1).reshape(nocc,nocc,-1)
        eris_vooo = None

        eris_vovv = _cp(eris.vovv[p0:p1]).reshape(-1,nvir_pair)
        eris_vovv = lib.unpack_tril(eris_vovv).reshape(p1-p0,nocc,nvir,nvir)
        #:t1 += numpy.einsum('cjib,cjba->ia', theta, eris_vovv)
        #:t2[:,:,p0:p1] += numpy.einsum('jc,aibc->ijab', c1, eris_vovv)
        theta = lib.transpose(theta.reshape(-1,nocc,nvir), axes=(0,2,1))
        lib.ddot(theta.reshape(-1,nocc).T, eris_vovv.reshape(-1,nvir), 1, t1, 1)
        for i in range(p0, p1):
            tmp = lib.ddot(c1, eris_vovv[i-p0].reshape(-1,nvir).T)
            t2[:,:,i] += tmp.reshape(nocc,nocc,nvir).transpose(1,0,2)
        tmp = eris_vovv = None

    #:t2 + t2.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2[i,:i]+= t2[:i,i].transpose(0,2,1)
            t2[:i,i] = t2[i,:i].transpose(0,2,1)
        t2[i,i] = t2[i,i] + t2[i,i].T

    cinew[0] += numpy.einsum('ia,ia->', fov, c1) * 2
    cinew[0] += numpy.einsum('aijb,ijab->', eris.voov, c2) * 2
    cinew[0] -= numpy.einsum('aijb,jiab->', eris.voov, c2)
    return cinew

def amplitudes_to_cisdvec(c0, c1, c2):
    return numpy.hstack((c0, c1.ravel(), c2.ravel()))

def cisdvec_to_amplitudes(civec, nmo, nocc):
    nvir = nmo - nocc
    c0 = civec[0]
    c1 = civec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
    return c0, c1, c2

def dot(v1, v2, nocc, nvir):
    hijab = v2[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    cijab = v1[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    val = numpy.dot(v1, v2) * 2 - v1[0]*v2[0]
    val-= numpy.einsum('jiab,ijab->', cijab, hijab)
    return val

def t1strs(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    addrs = []
    signs = []
    for a in range(nocc, norb):
        for i in reversed(range(nocc)):
            str1 = hf_str ^ (1 << i) | (1 << a)
            addrs.append(cistring.str2addr(norb, nelec, str1))
            signs.append(cistring.cre_des_sign(a, i, hf_str))
    return numpy.asarray(addrs), numpy.asarray(signs)

def to_fci(cisdvec, norb, nelec):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    nocc = neleca
    nvir = norb - nocc
    c0, c1, c2 = cisdvec_to_amplitudes(cisdvec, norb, nocc)
    t1addr, t1sign = t1strs(norb, nocc)

    na = cistring.num_strings(norb, nocc)
    fcivec = numpy.zeros((na,na))
    fcivec[0,0] = c0
    c1 = c1[::-1].T.ravel()
    fcivec[0,t1addr] = fcivec[t1addr,0] = c1 * t1sign
    c2ab = c2[::-1,::-1].transpose(2,0,3,1).reshape(nocc*nvir,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1sign, t1sign, c2ab)
    lib.takebak_2d(fcivec, c2ab, t1addr, t1addr)

    if nocc > 1 and nvir > 1:
        hf_str = int('1'*nocc, 2)
        for a in range(nocc, norb):
            for b in range(nocc, a):
                for i in reversed(range(1, nocc)):
                    for j in reversed(range(i)):
                        c2aa = c2[i,j,a-nocc,b-nocc] - c2[j,i,a-nocc,b-nocc]
                        str1 = hf_str ^ (1 << j) | (1 << b)
                        c2aa*= cistring.cre_des_sign(b, j, hf_str)
                        c2aa*= cistring.cre_des_sign(a, i, str1)
                        str1^= (1 << i) | (1 << a)
                        addr = cistring.str2addr(norb, nocc, str1)
                        fcivec[0,addr] = fcivec[addr,0] = c2aa
    return fcivec

def from_fci(ci0, norb, nelec):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    nocc = neleca
    nvir = norb - nocc
    t1addr, t1sign = t1strs(norb, nocc)

    c0 = ci0[0,0]
    c1 = ci0[0,t1addr] * t1sign
    c2 = numpy.einsum('i,j,ij->ij', t1sign, t1sign, ci0[t1addr][:,t1addr])
    c1 = c1.reshape(nvir,nocc).T
    c2 = c2.reshape(nvir,nocc,nvir,nocc).transpose(1,3,0,2)
    return amplitudes_to_cisdvec(c0, c1[::-1], c2[::-1,::-1])


def make_rdm1(ci, nmo, nocc):
    nvir = nmo - nocc
    c0, c1, c2 = cisdvec_to_amplitudes(ci, nmo, nocc)
    dov = c0*c1 * 2
    dov += numpy.einsum('jb,ijab->ia', c1, c2) * 4
    dov -= numpy.einsum('jb,ijba->ia', c1, c2) * 2
    doo  = numpy.einsum('ia,ka->ik', c1, c1) * -2
    #:doo -= numpy.einsum('ijab,ikab->jk', c2, c2) * 4
    #:doo += numpy.einsum('ijab,kiab->jk', c2, c2) * 2
    theta = c2*2 - c2.transpose(0,1,3,2)
    lib.ddot(c2.reshape(nocc,-1), theta.reshape(nocc,-1).T, -2, doo, 1)
    dvv  = numpy.einsum('ia,ic->ca', c1, c1) * 2
    #:dvv += numpy.einsum('ijab,ijac->cb', c2, c2) * 4
    #:dvv -= numpy.einsum('ijab,jiac->cb', c2, c2) * 2
    lib.ddot(c2.reshape(-1,nvir).T, theta.reshape(-1,nvir), 2, dvv, 1)

    rdm1 = numpy.empty((nmo,nmo))
    rdm1[:nocc,nocc:] = dov
    rdm1[nocc:,:nocc] = dov.T
    rdm1[:nocc,:nocc] = doo
    rdm1[nocc:,nocc:] = dvv

    for i in range(nocc):
        rdm1[i,i] += 2
    return rdm1

def make_rdm2(ci, nmo, nocc):
    '''spin-traced 2pdm in chemist's notation
    '''
    nvir = nmo - nocc
    noo = nocc**2
    nov = nocc * nvir
    c0, c1, c2 = cisdvec_to_amplitudes(ci, nmo, nocc)
    doovv = c0*c2
    dvvvo = numpy.einsum('ia,ikcd->cdak', c1, c2)
    dovoo =-numpy.einsum('ia,klac->ickl', c1, c2)
    doooo = lib.ddot(c2.reshape(noo,-1), c2.reshape(noo,-1).T).reshape((nocc,)*4)
    dvvvv = lib.ddot(c2.reshape(noo,-1).T, c2.reshape(noo,-1)).reshape((nvir,)*4)

    rdm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    rdm2[:nocc,:nocc,:nocc,:nocc] = doooo*4-doooo.transpose(1,0,2,3)*2
    rdm2[:nocc,nocc:,:nocc,:nocc] = dovoo*4-dovoo.transpose(0,1,3,2)*2
    rdm2[nocc:,:nocc,:nocc,:nocc] = rdm2[:nocc,nocc:,:nocc,:nocc].transpose(1,0,3,2)
    rdm2[:nocc,:nocc,:nocc,nocc:] = rdm2[:nocc,nocc:,:nocc,:nocc].transpose(2,3,0,1)
    rdm2[:nocc,:nocc,nocc:,:nocc] = rdm2[:nocc,nocc:,:nocc,:nocc].transpose(3,2,1,0)

    rdm2[:nocc,:nocc,nocc:,nocc:] = doovv*4-doovv.transpose(1,0,2,3)*2
    rdm2[nocc:,nocc:,:nocc,:nocc] = rdm2[:nocc,:nocc,nocc:,nocc:].transpose(2,3,0,1)
    rdm2[nocc:,nocc:,nocc:,:nocc] = dvvvo*4-dvvvo.transpose(1,0,2,3)*2
    rdm2[nocc:,nocc:,:nocc,nocc:] = rdm2[nocc:,nocc:,nocc:,:nocc].transpose(1,0,3,2)
    rdm2[nocc:,:nocc,nocc:,nocc:] = rdm2[nocc:,nocc:,nocc:,:nocc].transpose(2,3,0,1)
    rdm2[:nocc,nocc:,nocc:,nocc:] = rdm2[nocc:,nocc:,nocc:,:nocc].transpose(3,2,1,0)
    rdm2[nocc:,nocc:,nocc:,nocc:] = dvvvv*4-dvvvv.transpose(1,0,2,3)*2

    theta = c2*2 - c2.transpose(1,0,2,3)
    dovov  = numpy.einsum('ia,kc->icka', c1, c1) * -2
    #:dovov -= numpy.einsum('kjcb,kica->jaib', c2, theta) * 2
    #:dovov -= numpy.einsum('ikcb,jkca->iajb', c2, theta) * 2
    dovov -= lib.ddot(c2.transpose(0,2,1,3).reshape(nov,-1).T,
                      theta.transpose(0,2,1,3).reshape(nov,-1),
                      2).reshape(nocc,nvir,nocc,nvir).transpose(0,3,2,1)
    dovov -= lib.ddot(c2.transpose(0,3,1,2).reshape(nov,-1),
                      theta.transpose(0,3,1,2).reshape(nov,-1).T,
                      2).reshape(nocc,nvir,nocc,nvir).transpose(0,3,2,1)
    dvoov  = numpy.einsum('ia,kc->cika', c1, c1) * 4
    #:dvoov += numpy.einsum('kica,kjcb->ajib', theta, theta) * 2
    dvoov += lib.ddot(theta.transpose(0,2,1,3).reshape(nov,-1).T,
                      theta.transpose(0,2,1,3).reshape(nov,-1),
                      2).reshape(nocc,nvir,nocc,nvir).transpose(1,2,0,3)

    rdm2[:nocc,nocc:,:nocc,nocc:] = dovov
    rdm2[nocc:,:nocc,nocc:,:nocc] = dovov.transpose(1,0,3,2)
    rdm2[nocc:,:nocc,:nocc,nocc:] = dvoov
    rdm2[:nocc,nocc:,nocc:,:nocc] = dvoov.transpose(1,0,3,2)

    rdm1 = make_rdm1(ci, nmo, nocc)
    for i in range(nocc):
        rdm1[i,i] -= 2
    for i in range(nocc):
        for j in range(nocc):
            rdm2[i,j,i,j] += 4
            rdm2[i,j,j,i] -= 2
        rdm2[i,:,i,:] += rdm1 * 2
        rdm2[:,i,:,i] += rdm1 * 2
        rdm2[:,i,i,:] -= rdm1
        rdm2[i,:,:,i] -= rdm1

    return rdm2.transpose(0,2,1,3)  # to chemist's notation


class CISD(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ   is None: mo_occ   = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.conv_tol = 1e-9
        self.max_cycle = 50
        self.max_space = 12
        self.lindep = 1e-14
        self.nroots = 1
        self.level_shift = 0  # in precond

        self.frozen = frozen
        self.direct = False
        self.chkfile = None

##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.ci = None
        self._nocc = None
        self._nmo = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('CISD nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('max_space = %d', self.max_space)
        log.info('lindep = %d', self.lindep)
        log.info('nroots = %d', self.nroots)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def e_tot(self):
        return numpy.asarray(self.e_corr) + self._scf.e_tot

    nocc = property(ccsd.get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(ccsd.get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = ccsd.get_nocc
    get_nmo = ccsd.get_nmo

    def kernel(self, ci0=None, eris=None):
        return self.cisd(ci0, eris)
    def cisd(self, ci0=None, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.converged, self.e_corr, self.ci = \
                kernel(self, eris, ci0, max_cycle=self.max_cycle,
                       tol=self.conv_tol, verbose=self.verbose)
        if numpy.all(self.converged):
            logger.info(self, 'CISD converged')
        else:
            logger.info(self, 'CISD not converged')
        if self.nroots > 1:
            for i,e in enumerate(self.e_tot):
                logger.note(self, 'CISD root %d  E = %.16g', i, e)
        else:
            logger.note(self, 'E(CISD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.ci

    def get_init_guess(self, eris=None):
        # MP2 initial guess
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        mo_e = eris.fock.diagonal()
        e_ia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
        ci0 = 1
        ci1 = numpy.zeros_like(e_ia)
        eris_voov = _cp(eris.voov)
        ci2 = 2 * eris_voov.transpose(1,2,0,3)
        ci2-= eris_voov.transpose(1,2,3,0)
        ci2 /= lib.direct_sum('ia,jb->ijab', e_ia, e_ia)
        self.emp2 = numpy.einsum('ijab,aijb', ci2, eris_voov)
        eris_voov = None
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, amplitudes_to_cisdvec(ci0, ci1, ci2)

    contract = contract
    make_diagonal = make_diagonal

    def ao2mo(self, mo_coeff=None):
        nocc = self.nocc
        nvir = self.nmo - nocc
        mem_incore, mem_outcore, mem_basic = ccsd._mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            raise NotImplementedError

        else:
            return _make_eris_outcore(self, mo_coeff)

    def add_wvvVV_(self, t2, eris, t2new_tril):
        nocc, nvir = t2.shape[1:3]
        t1 = numpy.zeros((nocc,nvir))
        return ccsd.add_wvvVV_(self, t1, t2, eris, t2new_tril, with_ovvv=False)

    def to_fci(self, cisdvec, norb, nelec):
        return to_fci(cisdvec, norb, nelec)

    def from_fci(self, fcivec, norb, nelec):
        return from_fci(fcivec, norb, nelec)

    def make_rdm1(self, ci=None):
        if ci is None: ci = self.ci
        return make_rdm1(ci, self.nmo, self.nocc)

    def make_rdm2(self, ci=None):
        if ci is None: ci = self.ci
        return make_rdm2(ci, self.nmo, self.nocc)

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


class _RCISD_ERIs:
    def __init__(self, myci, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = ccsd._mo_without_core(myci, myci.mo_coeff)
        else:
            mo_coeff = ccsd._mo_without_core(myci, mo_coeff)
# Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = myci._scf.make_rdm1(myci.mo_coeff, myci.mo_occ)
        fockao = myci._scf.get_hcore() + myci._scf.get_veff(myci.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))
        self.mo_coeff = mo_coeff
        self.nocc = myci.nocc

        self.oooo = None
        self.vooo = None
        self.vvoo = None
        self.voov = None
        self.vovv = None
        self.vvvv = None

def _make_eris_incore(myci, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    eris = _RCISD_ERIs(myci, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eri1 = ao2mo.incore.full(myci._scf._eri, eris.mo_coeff)
    #:eri1 = ao2mo.restore(1, eri1, nmo)
    #:eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    #:eris.vooo = eri1[nocc:,:nocc,:nocc,:nocc].copy()
    #:eris.voov = eri1[nocc:,:nocc,:nocc,nocc:].copy()
    #:eris.vvoo = eri1[nocc:,nocc:,:nocc,:nocc].copy()
    #:vovv = eri1[nocc:,:nocc,nocc:,nocc:].copy()
    #:eris.vovv = lib.pack_tril(vovv.reshape(-1,nvir,nvir))
    #:eris.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = numpy.empty((nocc,nocc,nocc,nocc))
    eris.vooo = numpy.empty((nvir,nocc,nocc,nocc))
    eris.voov = numpy.empty((nvir,nocc,nocc,nvir))
    eris.vovv = numpy.empty((nvir,nocc,nvir_pair))
    eris.vvvv = numpy.empty((nvir_pair,nvir_pair))

    ij = 0
    outbuf = numpy.empty((nmo,nmo,nmo))
    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        for j in range(i+1):
            eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
            oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
        ij += i + 1
    eris.vvoo = lib.transpose(oovv.reshape(nocc**2,-1)).reshape(nvir,nvir,nocc,nocc)
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.vooo[i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.voov[i-nocc] = buf[:nocc,:nocc,nocc:]
        lib.pack_tril(_cp(buf[:nocc,nocc:,nocc:]), out=eris.vovv[i-nocc])
        dij = i - nocc + 1
        lib.pack_tril(_cp(buf[nocc:i+1,nocc:,nocc:]),
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(myci, 'CISD integral transformation', *cput0)
    return eris

def _make_eris_outcore(myci, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(myci.stdout, myci.verbose)
    eris = _RCISD_ERIs(myci, mo_coeff)

    mol = myci.mol
    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.vvoo = eris.feri1.create_dataset('vvoo', (nvir,nvir,nocc,nocc), 'f8')
    eris.vooo = eris.feri1.create_dataset('vooo', (nvir,nocc,nocc,nocc), 'f8')
    eris.voov = eris.feri1.create_dataset('voov', (nvir,nocc,nocc,nvir), 'f8')
    eris.vovv = eris.feri1.create_dataset('vovv', (nvir,nocc,nvpair), 'f8')

    nvir_pair = nvir*(nvir+1)//2
    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    def save_occ_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.oooo[p0:p1] = eri[:,:,:nocc,:nocc]
        oovv[p0:p1] = eri[:,:,nocc:,nocc:]

    def save_vir_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.vooo[p0:p1] = eri[:,:,:nocc,:nocc]
        eris.voov[p0:p1] = eri[:,:,:nocc,nocc:]
        vv = _cp(eri[:,:,nocc:,nocc:].reshape((p1-p0)*nocc,nvir,nvir))
        eris.vovv[p0:p1] = lib.pack_tril(vv).reshape(p1-p0,nocc,nvir_pair)

    cput1 = time.clock(), time.time()
    if not myci.direct:
        max_memory = max(2000, myci.max_memory-lib.current_memory()[0])
        eris.feri2 = lib.H5TmpFile()
        ao2mo.full(mol, orbv, eris.feri2, max_memory=max_memory, verbose=log)
        eris.vvvv = eris.feri2['eri_mo']
        cput1 = log.timer_debug1('transforming vvvv', *cput1)

    tmpfile3 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    with h5py.File(tmpfile3.name, 'w') as fswap:
        mo_coeff = numpy.asarray(mo_coeff, order='F')
        max_memory = max(2000, myci.max_memory-lib.current_memory()[0])
        int2e = mol._add_suffix('int2e')
        ao2mo.outcore.half_e1(mol, (mo_coeff,mo_coeff[:,:nocc]), fswap, int2e,
                              's4', 1, max_memory, verbose=log)

        ao_loc = mol.ao_loc_nr()
        nao_pair = nao * (nao+1) // 2
        blksize = int(min(8e9,max_memory*.5e6)/8/(nao_pair+nmo**2)/nocc)
        blksize = max(1, min(nmo*nocc, blksize))
        fload = ao2mo.outcore._load_from_h5g
        def prefetch(p0, p1, rowmax, buf):
            p0, p1 = p1, min(rowmax, p1+blksize)
            if p0 < p1:
                fload(fswap['0'], p0*nocc, p1*nocc, buf)

        buf = numpy.empty((blksize*nocc,nao_pair))
        buf_prefetch = numpy.empty_like(buf)
        outbuf = numpy.empty((blksize*nocc,nmo**2))
        with lib.call_in_background(prefetch) as bprefetch:
            fload(fswap['0'], 0, min(nocc,blksize)*nocc, buf_prefetch)
            for p0, p1 in lib.prange(0, nocc, blksize):
                nrow = (p1 - p0) * nocc
                buf, buf_prefetch = buf_prefetch, buf
                bprefetch(p0, p1, nocc, buf_prefetch)
                dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                         's4', 's1', out=outbuf, ao_loc=ao_loc)
                save_occ_frac(p0, p1, dat)

            fload(fswap['0'], nocc**2, min(nmo,nocc+blksize)*nocc, buf_prefetch)
            for p0, p1 in lib.prange(0, nvir, blksize):
                nrow = (p1 - p0) * nocc
                buf, buf_prefetch = buf_prefetch, buf
                bprefetch(nocc+p0, nocc+p1, nmo, buf_prefetch)
                dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                         's4', 's1', out=outbuf, ao_loc=ao_loc)
                save_vir_frac(p0, p1, dat)

        cput1 = log.timer_debug1('transforming oppp', *cput1)
    eris.vvoo[:] = lib.transpose(oovv.reshape(nocc**2,-1)).reshape(nvir,nvir,nocc,nocc)
    log.timer('CISD integral transformation', *cput0)
    return eris


def _cp(a):
    return numpy.array(a, copy=False, order='C')

def _memory_usage_inloop(nocc, nvir):
    v = nocc*nvir**2 + nocc**2*nvir*3
    return v*8/1e6


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import fci
    def finger(a):
        return numpy.dot(a.ravel(), numpy.cos(numpy.arange(a.size)))
    def fcicontract(h1, h2, norb, nelec, ci0):
        g2e = fci.direct_spin1.absorb_h1e(h1, h2, norb, nelec, .5)
        ci1 = fci.direct_spin1.contract_2e(g2e, ci0, norb, nelec)
        return ci1

    mol = gto.M()
    mol.nelectron = 6
    nocc, nvir = mol.nelectron//2, 4
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    numpy.random.seed(12)
    mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = numpy.random.random((nmo,nmo))
    mf.mo_occ = numpy.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    h1 = numpy.random.random((nmo,nmo)) * .1
    h1 = h1 + h1.T
    mf.get_hcore = lambda *args: h1

    myci = CISD(mf)
    eris = myci.ao2mo(mf.mo_coeff)
    eris.ehf = (h1*dm).sum() + (vhf*dm).sum()*.5

    c2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
    c2 = c2 + c2.transpose(1,0,3,2)
    civec = numpy.hstack((numpy.random.random(nocc*nvir+1) * .1,
                          c2.ravel()))
    hcivec = contract(myci, civec, eris)
    print(finger(hcivec) - 2059.5730673341673)

    ci0 = to_fci(civec, nmo, mol.nelectron)
    print(abs(civec-from_fci(ci0, nmo, nocc*2)).sum())
    h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, h1, mf.mo_coeff))
    ci1 = fcicontract(h1e, h2e, nmo, mol.nelectron, ci0)
    ci2 = to_fci(hcivec, nmo, mol.nelectron)
    e1 = numpy.dot(ci1.ravel(), ci0.ravel())
    e2 = dot(civec, hcivec+eris.ehf*civec, nocc, nvir)
    print(e1-e2)

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0.,-1.    , 1.   )],
    ]
    mol.charge = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.RHF(mol).run()
    ecisd = CISD(mf).kernel()[0]
    print(ecisd - -0.024780739973407784)
    h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eci = fci.direct_spin0.kernel(h1e, h2e, mf.mo_coeff.shape[1], mol.nelectron)[0]
    eci = eci + mol.energy_nuc() - mf.e_tot
    print(ecisd - eci)

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g',}
    mol.build()
    mf = scf.RHF(mol).run()
    myci = CISD(mf)
    ecisd, civec = myci.kernel()
    print(ecisd - -0.048878084082066106)

    nmo = myci.nmo
    nocc = myci.nocc
    rdm1 = myci.make_rdm1(civec)
    rdm2 = myci.make_rdm2(civec)
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, nmo)
    e2 = (numpy.einsum('ij,ji', h1e, rdm1) +
          numpy.einsum('ijkl,jilk', h2e, rdm2) * .5)
    print(ecisd + mf.e_tot - mol.energy_nuc() - e2)   # = 0

    print(abs(rdm1 - numpy.einsum('ijkk->ij', rdm2)/(mol.nelectron-1)).sum())

