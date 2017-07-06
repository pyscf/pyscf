#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Solve CISD equation  H C = C e  where e = E_HF + E_CORR
'''

import time
import numpy
from functools import reduce
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
    mo_energy = eris.fock.diagonal()
    diag = make_diagonal(mol, mo_energy, eris, nocc)
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
# MP2 initial guess
        e_i = mo_energy[:nocc]
        e_a = mo_energy[nocc:]
        ci0 = numpy.zeros(1+nocc*nvir+(nocc*nvir)**2)
        ci0[0] = 1
        t2 = 2*eris.voov.transpose(1,2,0,3) - eris.voov.transpose(1,2,3,0)
        t2 /= lib.direct_sum('i+j-a-b', e_i, e_i, e_a, e_a)
        ci0[1+nocc*nvir:] = t2.ravel()

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
        ecisd = ecisd[0]
        ci = ci[0]
    return conv, ecisd, ci

def make_diagonal(mol, mo_energy, eris, nocc):
    nmo = mo_energy.size
    nvir = nmo - nocc
    jdiag = numpy.zeros((nmo,nmo))
    kdiag = numpy.zeros((nmo,nmo))
    eris_vvvv = ao2mo.restore(1, eris.vvvv, nvir)
    jdiag[:nocc,:nocc] = numpy.einsum('iijj->ij', eris.oooo)
    kdiag[:nocc,:nocc] = numpy.einsum('jiij->ij', eris.oooo)
    jdiag[nocc:,nocc:] = numpy.einsum('iijj->ij', eris_vvvv)
    kdiag[nocc:,nocc:] = numpy.einsum('jiij->ij', eris_vvvv)
    jdiag[:nocc,nocc:] = numpy.einsum('iijj->ji', eris.vvoo)
    kdiag[:nocc,nocc:] = numpy.einsum('jiij->ij', eris.voov)
    jksum = (jdiag[:nocc,:nocc] * 2 - kdiag[:nocc,:nocc]).sum()
    ehf = mo_energy[:nocc].sum() * 2 - jksum
    e1diag = numpy.empty((nocc,nvir))
    e2diag = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        for a in range(nocc, nmo):
            e1diag[i,a-nocc] = ehf - mo_energy[i] + mo_energy[a] \
                    - jdiag[i,a] + kdiag[i,a]
            for j in range(nocc):
                for b in range(nocc, nmo):
                    e2diag[i,j,a-nocc,b-nocc] = ehf \
                            - mo_energy[i] - mo_energy[j] \
                            + mo_energy[a] + mo_energy[b] \
                            + jdiag[i,j] - jdiag[i,a] + kdiag[i,a] \
                            - jdiag[j,a] - jdiag[i,b] - jdiag[j,b] \
                            + kdiag[j,b] + jdiag[a,b]
    return numpy.hstack((ehf, e1diag.reshape(-1), e2diag.reshape(-1)))

def contract(myci, civec, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(myci.stdout, myci.verbose)
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    nov = nocc * nvir
    noo = nocc**2
    c0 = civec[0]
    c1 = civec[1:nov+1].reshape(nocc,nvir)
    c2 = civec[nov+1:].reshape(nocc,nocc,nvir,nvir)

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

    eris_vovv = lib.unpack_tril(eris.vovv).reshape(nvir,nocc,nvir,-1)
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

    for i in range(nocc):
        for j in range(i+1):
            t2[i,j]+= t2[j,i].T
            t2[j,i] = t2[i,j].T

    cinew[0] += numpy.einsum('ia,ia->', fov, c1) * 2
    cinew[0] += numpy.einsum('aijb,ijab->', eris.voov, c2) * 2
    cinew[0] -= numpy.einsum('aijb,jiab->', eris.voov, c2)
    return cinew

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
    c0 = cisdvec[0]
    c1 = cisdvec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = cisdvec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
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
    return numpy.hstack((c0, c1[::-1].ravel(), c2[::-1,::-1].ravel()))



def make_rdm1(ci, nmo, nocc):
    nvir = nmo - nocc
    c0 = ci[0]
    c1 = ci[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = ci[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
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
    nvir = nmo - nocc
    noo = nocc**2
    nov = nocc * nvir
    c0 = ci[0]
    c1 = ci[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = ci[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
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

    # Fixme: This seems giving right answer, but not based on solid formula
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

#    rdm2[:nocc,nocc:,:nocc,nocc:] = dovov*4-dvoov.transpose(1,0,2,3)*2
#    rdm2[nocc:,:nocc,nocc:,:nocc] = rdm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2)
#    rdm2[nocc:,:nocc,:nocc,nocc:] = dvoov*4-dovov.transpose(1,0,2,3)*2
#    rdm2[:nocc,nocc:,nocc:,:nocc] = rdm2[nocc:,:nocc,:nocc,nocc:].transpose(1,0,3,2)
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

    return rdm2.transpose(0,2,1,3)


class CISD(lib.StreamObject):
    def __init__(self, mf, frozen=[], mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ   is None: mo_occ   = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.conv_tol = 1e-7
        self.max_cycle = 50
        self.max_space = 12
        self.lindep = 1e-14
        self.nroots = 1
        self.level_shift = 0  # in precond

        self.frozen = frozen
        self.direct = False

##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.ci = None
        self._nocc = None
        self._nmo = None

    @property
    def e_tot(self):
        return numpy.asarray(self.e_corr) + self._scf.e_tot

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        elif isinstance(self.frozen, (int, numpy.integer)):
            return numpy.count_nonzero(self.mo_occ) - self.frozen
        elif self.frozen:
            occ_idx = self.mo_occ > 0
            occ_idx[numpy.asarray(self.frozen)] = False
            return numpy.count_nonzero(occ_idx)
        else:
            return numpy.count_nonzero(self.mo_occ)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        if isinstance(self.frozen, (int, numpy.integer)):
            return len(self.mo_occ) - self.frozen
        else:
            return len(self.mo_occ) - len(self.frozen)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def kernel(self, ci0=None, mo_coeff=None, eris=None):
        return self.cisd(ci0, mo_coeff, eris)
    def cisd(self, ci0=None, mo_coeff=None, eris=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        self.converged, self.e_corr, self.ci = \
                kernel(self, eris, ci0, max_cycle=self.max_cycle,
                       tol=self.conv_tol, verbose=self.verbose)
        if self.converged:
            logger.info(self, 'CISD converged')
        else:
            logger.info(self, 'CISD not converged')
        if self.nroots > 1:
            for i,e in enumerate(self.e_tot):
                logger.note(self, 'CISD root %d  E = %.16g', i, e)
        else:
            if self._scf.e_tot == 0:
                logger.note(self, 'E_corr = %.16g', self.e_corr)
            else:
                logger.note(self, 'E(CISD) = %.16g  E_corr = %.16g',
                            self.e_tot, self.e_corr)
        return self.e_corr, self.ci

    contract = contract

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def add_wvvVV(self, t2, eris):
        nocc, nvir = t2.shape[1:3]
        t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
        return self.add_wvvVV_(t2, eris, t2new_tril)
    def add_wvvVV_(self, t2, eris, t2new_tril):
        time0 = time.clock(), time.time()
        nocc, nvir = t2.shape[1:3]

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

        if self.direct:   # AO-direct CCSD
            mol = self.mol
            nao, nmo = self.mo_coeff.shape
            nao_pair = nao * (nao+1) // 2
            aos = numpy.asarray(self.mo_coeff[:,nocc:].T, order='F')
            outbuf = numpy.empty((nocc*(nocc+1)//2,nao,nao))
            tau = numpy.ndarray((nocc*(nocc+1)//2,nvir,nvir), buffer=outbuf)
            p0 = 0
            for i in range(nocc):
                tau[p0:p0+i+1] = t2[i,:i+1]
                p0 += i + 1
            tau = _ao2mo.nr_e2(tau.reshape(-1,nvir**2), aos, (0,nao,0,nao), 's1', 's1')
            tau = tau.reshape(-1,nao,nao)
            time0 = logger.timer_debug1(self, 'vvvv-tau', *time0)

            intor = mol._add_suffix('int2e')
            ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                     'CVHFsetnr_direct_scf')
            outbuf[:] = 0
            ao_loc = mol.ao_loc_nr()
            max_memory = max(0, self.max_memory - lib.current_memory()[0])
            dmax = max(4, int(numpy.sqrt(max_memory*.95e6/8/nao**2/2)))
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
                    time0 = logger.timer_debug1(self, 'AO-vvvv [%d:%d,%d:%d]' %
                                                (ish0,ish1,jsh0,jsh1), *time0)
                eri = fint(intor, mol._atm, mol._bas, mol._env,
                           shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                           ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                for i in range(i1-i0):
                    p0, p1 = i*(i+1)//2, (i+1)*(i+2)//2
                    tmp = lib.unpack_tril(eri[p0:p1], out=loadbuf)
                    contract_tril_(outbuf, tau, tmp, i0, i0+i)
                time0 = logger.timer_debug1(self, 'AO-vvvv [%d:%d,%d:%d]' %
                                            (ish0,ish1,ish0,ish1), *time0)
            eribuf = loadbuf = eri = tmp = None

            mo = numpy.asarray(self.mo_coeff, order='F')
            tmp = _ao2mo.nr_e2(outbuf, mo, (nocc,nmo,nocc,nmo), 's1', 's1', out=tau)
            t2new_tril += tmp.reshape(-1,nvir,nvir)

        else:
            tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
            p0 = 0
            for i in range(nocc):
                tau[p0:p0+i+1] = t2[i,:i+1]
                p0 += i + 1
            p0 = 0
            outbuf = numpy.empty((nvir,nvir,nvir))
            outbuf1 = numpy.empty((nvir,nvir,nvir))
            handler = None
            for a in range(nvir):
                buf = lib.unpack_tril(eris.vvvv[p0:p0+a+1], out=outbuf)
                outbuf, outbuf1 = outbuf1, outbuf
                handler = async_do(handler, contract_tril_, t2new_tril, tau, buf, 0, a)
                p0 += a+1
                time0 = logger.timer_debug1(self, 'vvvv %d'%a, *time0)
            handler.join()
        return t2new_tril

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


class _ERIS:
    def __init__(self, myci, mo_coeff=None, method='incore'):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(myci.mo_occ.size, dtype=numpy.bool)
        if isinstance(myci.frozen, (int, numpy.integer)):
            moidx[:myci.frozen] = False
        elif len(myci.frozen) > 0:
            moidx[numpy.asarray(myci.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = myci.mo_coeff[:,moidx]
        else:
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
        dm = myci._scf.make_rdm1(myci.mo_coeff, myci.mo_occ)
        fockao = myci._scf.get_hcore() + myci._scf.get_veff(myci.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = myci.nocc
        nmo = myci.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = ccsd._mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]

        log = logger.Logger(myci.stdout, myci.verbose)
        if (method == 'incore' and myci._scf._eri is not None and
            (mem_incore+mem_now < myci.max_memory) or myci.mol.incore_anyway):
            eri1 = ao2mo.incore.full(myci._scf._eri, mo_coeff)
            #:eri1 = ao2mo.restore(1, eri1, nmo)
            #:self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
            #:self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            #:self.vooo = eri1[nocc:,:nocc,:nocc,:nocc].copy()
            #:self.voov = eri1[nocc:,:nocc,:nocc,nocc:].copy()
            #:self.vvoo = eri1[nocc:,nocc:,:nocc,:nocc].copy()
            #:vovv = eri1[nocc:,:nocc,nocc:,nocc:].copy()
            #:self.vovv = lib.pack_tril(vovv.reshape(-1,nvir,nvir))
            #:self.vvvv = ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
            nvir_pair = nvir * (nvir+1) // 2
            self.oooo = numpy.empty((nocc,nocc,nocc,nocc))
            self.ooov = numpy.empty((nocc,nocc,nocc,nvir))
            self.vooo = numpy.empty((nvir,nocc,nocc,nocc))
            self.voov = numpy.empty((nvir,nocc,nocc,nvir))
            self.vovv = numpy.empty((nvir,nocc,nvir_pair))
            self.vvvv = numpy.empty((nvir_pair,nvir_pair))
            ij = 0
            outbuf = numpy.empty((nmo,nmo,nmo))
            oovv = numpy.empty((nocc,nocc,nvir,nvir))
            for i in range(nocc):
                buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
                for j in range(i+1):
                    self.oooo[i,j] = self.oooo[j,i] = buf[j,:nocc,:nocc]
                    self.ooov[i,j] = self.ooov[j,i] = buf[j,:nocc,nocc:]
                    oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
                ij += i + 1
            self.vvoo = lib.transpose(oovv.reshape(nocc**2,-1)).reshape(nvir,nvir,nocc,nocc)
            oovv = None
            ij1 = 0
            for i in range(nocc,nmo):
                buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
                self.vooo[i-nocc] = buf[:nocc,:nocc,:nocc]
                self.voov[i-nocc] = buf[:nocc,:nocc,nocc:]
                lib.pack_tril(_cp(buf[:nocc,nocc:,nocc:]), out=self.vovv[i-nocc])
                dij = i - nocc + 1
                lib.pack_tril(_cp(buf[nocc:i+1,nocc:,nocc:]),
                              out=self.vvvv[ij1:ij1+dij])
                ij += i + 1
                ij1 += dij
        else:
            cput1 = time.clock(), time.time()
            self.feri1 = lib.H5TmpFile()
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            nvpair = nvir * (nvir+1) // 2
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'f8')
            self.vvoo = self.feri1.create_dataset('vvoo', (nvir,nvir,nocc,nocc), 'f8')
            self.vooo = self.feri1.create_dataset('vooo', (nvir,nocc,nocc,nocc), 'f8')
            self.voov = self.feri1.create_dataset('voov', (nvir,nocc,nocc,nvir), 'f8')
            self.vovv = self.feri1.create_dataset('vovv', (nvir,nocc,nvpair), 'f8')
            fsort = _ccsd.libcc.CCsd_sort_inplace
            nocc_pair = nocc*(nocc+1)//2
            nvir_pair = nvir*(nvir+1)//2
            def sort_inplace(p0, p1, eri):
                fsort(eri.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(nocc), ctypes.c_int(nvir),
                      ctypes.c_int((p1-p0)*nocc))
                vv = eri[:,:nvir_pair]
                oo = eri[:,nvir_pair:nvir_pair+nocc_pair]
                ov = eri[:,nvir_pair+nocc_pair:].reshape(-1,nocc,nvir)
                return oo, ov, vv
            buf = numpy.empty((nmo,nmo,nmo))
            oovv = numpy.empty((nocc,nocc,nvir,nvir))
            def save_occ_frac(p0, p1, eri):
                oo, ov, vv = sort_inplace(p0, p1, eri)
                self.oooo[p0:p1] = lib.unpack_tril(oo, out=buf).reshape(p1-p0,nocc,nocc,nocc)
                self.ooov[p0:p1] = ov.reshape(p1-p0,nocc,nocc,nvir)
                oovv[p0:p1] = lib.unpack_tril(vv, out=buf).reshape(p1-p0,nocc,nvir,nvir)
            def save_vir_frac(p0, p1, eri):
                oo, ov, vv = sort_inplace(p0, p1, eri)
                self.vooo[p0:p1] = lib.unpack_tril(oo, out=buf).reshape(p1-p0,nocc,nocc,nocc)
                self.voov[p0:p1] = ov.reshape(p1-p0,nocc,nocc,nvir)
                self.vovv[p0:p1] = vv.reshape(p1-p0,nocc,-1)

            if not myci.direct:
                max_memory = max(2000,myci.max_memory-lib.current_memory()[0])
                self.feri2 = lib.H5TmpFile()
                ao2mo.full(myci.mol, orbv, self.feri2, max_memory=max_memory, verbose=log)
                self.vvvv = self.feri2['eri_mo']
                cput1 = log.timer_debug1('transforming vvvv', *cput1)

            tmpfile3 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            with h5py.File(tmpfile3.name, 'w') as feri:
                max_memory = max(2000, myci.max_memory-lib.current_memory()[0])
                mo = numpy.hstack((orbv, orbo))
                ao2mo.general(myci.mol, (mo,orbo,mo,mo),
                              feri, max_memory=max_memory, verbose=log)
                cput1 = log.timer_debug1('transforming oppp', *cput1)
                blksize = max(1, int(min(8e9,max_memory*.5e6)/8/nmo**2/nocc))
                handler = None
                for p0, p1 in lib.prange(0, nvir, blksize):
                    eri = _cp(feri['eri_mo'][p0*nocc:p1*nocc])
                    handler = async_do(handler, save_vir_frac, p0, p1, eri)
                for p0, p1 in lib.prange(0, nocc, blksize):
                    eri = _cp(feri['eri_mo'][(p0+nvir)*nocc:(p1+nvir)*nocc])
                    handler = async_do(handler, save_occ_frac, p0, p1, eri)
                if handler is not None:
                    handler.join()
            self.vvoo[:] = lib.transpose(oovv.reshape(nocc**2,-1)).reshape(nvir,nvir,nocc,nocc)
        log.timer('CISD integral transformation', *cput0)

def _cp(a):
    return numpy.array(a, copy=False, order='C')

def _memory_usage_inloop(nocc, nvir):
    v = nocc*nvir**2 + nocc**2*nvir*3
    return v*8/1e6

def async_do(handler, fn, *args):
    if handler is not None:
        handler.join()
    handler = lib.background_thread(fn, *args)
    return handler



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
    eris = myci.ao2mo()
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
    ecisd , civec = myci.kernel()
    print(ecisd - -0.048878084082066106)

    nmo = mf.mo_coeff.shape[1]
    nocc = mol.nelectron//2
    rdm1 = make_rdm1(civec, nmo, nocc)
    rdm2 = make_rdm2(civec, nmo, nocc)
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, nmo)
    e2 = (numpy.dot(h1e.flatten(),rdm1.flatten()) +
          numpy.dot(h2e.transpose(0,2,1,3).flatten(),rdm2.flatten()) * .5)
    print(ecisd + mf.e_tot - mol.energy_nuc() - e2)   # = 0

    print(abs(rdm1 - numpy.einsum('ikjk->ij', rdm2)/(mol.nelectron-1)).sum())
