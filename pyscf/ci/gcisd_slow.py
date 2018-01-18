#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
General spin-orbital CISD
'''

import time
from functools import reduce
import numpy
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import uccsd
from pyscf.ci.cisd_slow import t1strs, t2strs
from pyscf.fci import cistring

einsum = lib.einsum

def kernel(myci, eris, ci0=None, max_cycle=50, tol=1e-8,
           verbose=logger.INFO):
    mol = myci.mol
    nmo = myci.nmo
    nocc = myci.nocc
    mo_energy = eris.fock.diagonal()
    diag = make_diagonal(myci, eris)
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
        ci0 = myci.get_init_guess(eris)[1]

    def op(x):
        return contract(myci, x, eris)

    def precond(x, e, *args):
        return x / (diag - e)

    conv, ecisd, ci = lib.davidson1(lambda xs: [op(x) for x in xs],
                                    ci0, precond,
                                    max_cycle=max_cycle, tol=tol,
                                    verbose=verbose)
    if myci.nroots == 1:
        conv = conv[0]
        ecisd = ecisd[0]
        ci = ci[0]
    return conv, ecisd, ci


def make_diagonal(myci, eris):
    nocc, nvir = eris.ovoo.shape[:2]
    nmo = nocc + nvir
    jkdiag = numpy.zeros((nmo,nmo))
    jkdiag[:nocc,:nocc] = numpy.einsum('ijij->ij', eris.oooo)
    jkdiag[nocc:,nocc:] = numpy.einsum('ijij->ij', eris.vvvv)
    jkdiag[:nocc,nocc:] = numpy.einsum('ijij->ij', eris.ovov)
    jksum = jkdiag[:nocc,:nocc].sum()
    mo_energy = eris.fock.diagonal()
    ehf = mo_energy[:nocc].sum() - jksum * .5
    e1diag = numpy.empty((nocc,nvir))
    e2diag = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        for a in range(nocc, nmo):
            e1diag[i,a-nocc] = ehf - mo_energy[i] + mo_energy[a] - jkdiag[i,a]
            for j in range(nocc):
                for b in range(nocc, nmo):
                    e2diag[i,j,a-nocc,b-nocc] = ehf \
                            - mo_energy[i] - mo_energy[j] \
                            + mo_energy[a] + mo_energy[b] \
                            + jkdiag[i,j] + jkdiag[a,b] \
                            - jkdiag[i,a] - jkdiag[j,a] \
                            - jkdiag[i,b] - jkdiag[j,b]
    return amplitudes_to_cisdvec(ehf, e1diag, e2diag)

def contract(myci, civec, eris):
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc

    c0, c1, c2 = cisdvec_to_amplitudes(civec, nocc, nvir)
    c2 = c2

    fock = eris.fock
    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    t1  = einsum('ie,ae->ia', c1, fvv)
    t1 -= einsum('ma,mi->ia', c1, foo)
    t1 += einsum('imae,me->ia', c2, fov)
    #:t1 -= einsum('nf,naif->ia', c1, eris.ovov)
    t1 += einsum('nf,nafi->ia', c1, eris.ovvo)
    t1 -= 0.5*einsum('imef,maef->ia', c2, eris.ovvv)
    t1 -= 0.5*einsum('mnae,mnie->ia', c2, eris.ooov)

    tmp = einsum('ijae,be->ijab', c2, fvv)
    t2  = tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('imab,mj->ijab', c2, foo)
    t2 -= tmp - tmp.transpose(1,0,2,3)
    t2 += 0.5*einsum('mnab,mnij->ijab', c2, eris.oooo)
    t2 += 0.5*einsum('ijef,abef->ijab', c2, eris.vvvv)
    tmp = einsum('imae,mbej->ijab', c2, eris.ovvo)
    tmp+= numpy.einsum('ia,jb->ijab', c1, fov)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2 += tmp - tmp.transpose(1,0,2,3)
    tmp = einsum('ie,jeba->ijab', c1, numpy.asarray(eris.ovvv).conj())
    t2 += tmp - tmp.transpose(1,0,2,3)
    tmp = einsum('ma,mbij->ijab', c1, eris.ovoo)
    t2 -= tmp - tmp.transpose(0,1,3,2)

    t1 += fov * c0
    t2 += eris.oovv * c0
    t0  = numpy.einsum('ia,ia', fov, c1)
    t0 += numpy.einsum('ijab,ijab', eris.oovv, c2) * .25

    return amplitudes_to_cisdvec(t0, t1, t2)

def amplitudes_to_cisdvec(c0, c1, c2):
    nocc, nvir = c1.shape
    ooidx = numpy.tril_indices(nocc, -1)
    vvidx = numpy.tril_indices(nvir, -1)
    return numpy.hstack((c0, c1.ravel(), c2[ooidx][:,vvidx[0],vvidx[1]].ravel()))

def cisdvec_to_amplitudes(civec, nocc, nvir):
    c0 = civec[0]
    c1 = civec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = _unpack_4fold(civec[nocc*nvir+1:], nocc, nvir)
    return c0, c1, c2

def _unpack_4fold(c2vec, nocc, nvir):
    ooidx = numpy.tril_indices(nocc, -1)
    vvidx = numpy.tril_indices(nvir, -1)
    c2tmp = numpy.zeros((nocc,nocc,nvir*(nvir-1)//2))
    c2tmp[ooidx] = c2vec.reshape(nocc*(nocc-1)//2,-1)
    c2tmp = c2tmp - c2tmp.transpose(1,0,2)
    c2 = numpy.zeros((nocc,nocc,nvir,nvir))
    c2[:,:,vvidx[0],vvidx[1]] = c2tmp
    c2 = c2 - c2.transpose(0,1,3,2)
    return c2

def to_fci(cisdvec, nelec, orbspin):
    from pyscf import fci
    nocc = nelec
    norb = len(orbspin)
    c0, c1, c2 = cisdvec_to_amplitudes(cisdvec, nocc, norb-nocc)
    oidxa = orbspin[:nocc] == 0
    oidxb = orbspin[:nocc] == 1
    vidxa = orbspin[nocc:] == 0
    vidxb = orbspin[nocc:] == 1
    c1a = c1[oidxa][:,vidxa]
    c1b = c1[oidxb][:,vidxb]
    c2aa = c2[oidxa][:,oidxa][:,:,vidxa][:,:,:,vidxa]
    c2bb = c2[oidxb][:,oidxb][:,:,vidxb][:,:,:,vidxb]
    c2ab = c2[oidxa][:,oidxb][:,:,vidxa][:,:,:,vidxb]
    nocca = numpy.count_nonzero(oidxa)
    noccb = numpy.count_nonzero(oidxb)
    nvira = numpy.count_nonzero(vidxa)
    nvirb = numpy.count_nonzero(vidxb)
    norba = nocca + nvira
    norbb = noccb + nvirb
    t1addra, t1signa = t1strs(norba, nocca)
    t1addrb, t1signb = t1strs(norbb, noccb)

    na = fci.cistring.num_strings(norba, nocca)
    nb = fci.cistring.num_strings(norbb, noccb)
    fcivec = numpy.zeros((na,nb))
    fcivec[0,0] = c0
    fcivec[t1addra,0] = c1a[::-1].T.ravel() * t1signa
    fcivec[0,t1addrb] = c1b[::-1].T.ravel() * t1signb
    c2ab = c2ab[::-1,::-1].transpose(2,0,3,1).reshape(nocca*nvira,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1signa, t1signb, c2ab)
    lib.takebak_2d(fcivec, c2ab, t1addra, t1addrb)

    if nocca > 1 and nvira > 1:
        ooidx = numpy.tril_indices(nocca, -1)
        vvidx = numpy.tril_indices(nvira, -1)
        c2aa = c2aa[ooidx][:,vvidx[0],vvidx[1]]
        t2addra, t2signa = t2strs(norba, nocca)
        fcivec[t2addra,0] = c2aa[::-1].T.ravel() * t2signa
    if noccb > 1 and nvirb > 1:
        ooidx = numpy.tril_indices(noccb, -1)
        vvidx = numpy.tril_indices(nvirb, -1)
        c2bb = c2bb[ooidx][:,vvidx[0],vvidx[1]]
        t2addrb, t2signb = t2strs(norbb, noccb)
        fcivec[0,t2addrb] = c2bb[::-1].T.ravel() * t2signb
    return fcivec

def from_fci(ci0, nelec, orbspin):
    from pyscf.cc.addons import spatial2spin
    nocc = nelec
    oidxa = orbspin[:nocc] == 0
    oidxb = orbspin[:nocc] == 1
    vidxa = orbspin[nocc:] == 0
    vidxb = orbspin[nocc:] == 1
    nocca = numpy.count_nonzero(oidxa)
    noccb = numpy.count_nonzero(oidxb)
    nvira = numpy.count_nonzero(vidxa)
    nvirb = numpy.count_nonzero(vidxb)
    norba = nocca+nvira
    norbb = noccb+nvirb
    t1addra, t1signa = t1strs(norba, nocca)
    t1addrb, t1signb = t1strs(norbb, noccb)

    na = fci.cistring.num_strings(norba, nocca)
    nb = fci.cistring.num_strings(norbb, noccb)
    ci0 = ci0.reshape(na,nb)
    c0 = ci0[0,0]
    c1a = ((ci0[t1addra,0] * t1signa).reshape(nvira,nocca).T)[::-1]
    c1b = ((ci0[0,t1addrb] * t1signb).reshape(nvirb,noccb).T)[::-1]
    c1 = spatial2spin((c1a, c1b), orbspin)

    c2ab = numpy.einsum('i,j,ij->ij', t1signa, t1signb, ci0[t1addra][:,t1addrb])
    c2ab = c2ab.reshape(nvira,nocca,nvirb,noccb).transpose(1,3,0,2)
    c2ab = c2ab[::-1,::-1]
    t2addra, t2signa = t2strs(norba, nocca)
    c2aa = (ci0[t2addra,0] * t2signa).reshape(nvira*(nvira-1)//2,-1).T
    c2aa = _unpack_4fold(c2aa[::-1], nocca, nvira)
    t2addrb, t2signb = t2strs(norbb, noccb)
    c2bb = (ci0[0,t2addrb] * t2signb).reshape(nvirb*(nvirb-1)//2,-1).T
    c2bb = _unpack_4fold(c2bb[::-1], noccb, nvirb)
    c2 = spatial2spin((c2aa, c2ab, c2bb), orbspin)

    cisdvec = amplitudes_to_cisdvec(c0, c1, c2)
    return cisdvec


def make_rdm1(ci, nmo, nocc):
    nvir = nmo - nocc
    c0, c1, c2 = cisdvec_to_amplitudes(ci, nocc, nvir)
    dov = c0 * c1
    dov += numpy.einsum('jb,ijab->ia', c1, c2)
    doo  =-numpy.einsum('ia,ka->ik', c1, c1)
    doo -= numpy.einsum('ijab,ikab->jk', c2, c2) * .25
    doo -= numpy.einsum('jiab,kiab->jk', c2, c2) * .25
    dvv  = numpy.einsum('ia,ic->ca', c1, c1)
    dvv += numpy.einsum('ijab,ijac->cb', c2, c2) * .25
    dvv += numpy.einsum('ijba,ijca->cb', c2, c2) * .25

    rdm1 = numpy.empty((nmo,nmo))
    rdm1[:nocc,nocc:] = dov
    rdm1[nocc:,:nocc] = dov.T.conj()
    rdm1[:nocc,:nocc] = doo
    rdm1[nocc:,nocc:] = dvv

    for i in range(nocc):
        rdm1[i,i] += 1
    return rdm1

def make_rdm2(ci, nmo, nocc):
    '''spin-orbital 2pdm in physicist's notation
    '''
    nvir = nmo - nocc
    c0, c1, c2 = cisdvec_to_amplitudes(ci, nocc, nvir)
    doovv = c0 * c2 * .5
    dvvvo = numpy.einsum('ia,ikcd->cdak', c1, c2) * .5
    dovoo = numpy.einsum('ia,klac->ickl', c1, c2) *-.5
    doooo = numpy.einsum('klab,ijab->klij', c2, c2) * .25
    dvvvv = numpy.einsum('ijcd,ijab->cdab', c2, c2) * .25
    dovov =-numpy.einsum('ijab,ikac->jckb', c2, c2)
    dovov-= numpy.einsum('ia,jb->jaib', c1, c1)
    dvoov = numpy.einsum('ijab,ikac->cjkb', c2, c2)
    dvoov+= numpy.einsum('ia,jb->ajib', c1, c1)

    rdm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    rdm2[:nocc,:nocc,:nocc,:nocc] = doooo - doooo.transpose(1,0,2,3)
    rdm2[:nocc,nocc:,:nocc,:nocc] = dovoo - dovoo.transpose(0,1,3,2)
    rdm2[nocc:,:nocc,:nocc,:nocc] = rdm2[:nocc,nocc:,:nocc,:nocc].transpose(1,0,3,2)
    rdm2[:nocc,:nocc,:nocc,nocc:] = rdm2[:nocc,nocc:,:nocc,:nocc].transpose(2,3,0,1)
    rdm2[:nocc,:nocc,nocc:,:nocc] = rdm2[:nocc,nocc:,:nocc,:nocc].transpose(3,2,1,0)

    rdm2[:nocc,:nocc,nocc:,nocc:] = doovv - doovv.transpose(1,0,2,3)
    rdm2[nocc:,nocc:,:nocc,:nocc] = rdm2[:nocc,:nocc,nocc:,nocc:].transpose(2,3,0,1)
    rdm2[nocc:,nocc:,nocc:,:nocc] = dvvvo - dvvvo.transpose(1,0,2,3)
    rdm2[nocc:,nocc:,:nocc,nocc:] = rdm2[nocc:,nocc:,nocc:,:nocc].transpose(1,0,3,2)
    rdm2[nocc:,:nocc,nocc:,nocc:] = rdm2[nocc:,nocc:,nocc:,:nocc].transpose(2,3,0,1)
    rdm2[:nocc,nocc:,nocc:,nocc:] = rdm2[nocc:,nocc:,nocc:,:nocc].transpose(3,2,1,0)
    rdm2[nocc:,nocc:,nocc:,nocc:] = dvvvv - dvvvv.transpose(1,0,2,3)

    rdm2[:nocc,nocc:,:nocc,nocc:] = dovov
    rdm2[nocc:,:nocc,nocc:,:nocc] = rdm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2)
    rdm2[nocc:,:nocc,:nocc,nocc:] = dvoov
    rdm2[:nocc,nocc:,nocc:,:nocc] = rdm2[nocc:,:nocc,:nocc,nocc:].transpose(1,0,3,2)

    rdm1 = make_rdm1(ci, nmo, nocc)
    for i in range(nocc):
        rdm1[i,i] -= 1
    for i in range(nocc):
        for j in range(nocc):
            rdm2[i,j,i,j] += 1
            rdm2[i,j,j,i] -= 1
        rdm2[i,:,i,:] += rdm1
        rdm2[:,i,:,i] += rdm1
        rdm2[:,i,i,:] -= rdm1
        rdm2[i,:,:,i] -= rdm1

    return rdm2


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
        self.frozen = frozen
        self.nroots = 1

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
        nocca, noccb = uccsd.get_nocc(self)
        return nocca + noccb

    @property
    def nmo(self):
        nmoa, nmob = uccsd.get_nmo(self)
        return nmoa + nmob

    def kernel(self, ci0=None, mo_coeff=None, eris=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        self.converged, self.e_corr, self.ci = \
                kernel(self, eris, ci0, max_cycle=self.max_cycle,
                       tol=self.conv_tol, verbose=self.verbose)
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(UCISD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.ci

    def get_init_guess(self, eris=None):
        # MP2 initial guess
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        t1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = numpy.array(eris.oovv)
        t2 = eris_oovv / eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2.conj(), eris_oovv).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, amplitudes_to_cisdvec(1, t1, t2)

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def to_fci(self, cisdvec, nelec, orbspin):
        return to_fci(cisdvec, nelec, orbspin)

    def from_fci(self, fcivec, nelec, orbspin):
        return from_fci(fcivec, nelec, orbspin)

    def make_rdm1(self, ci=None):
        if ci is None: ci = self.ci
        return make_rdm1(ci, self.nmo, self.nocc)

    def make_rdm2(self, ci=None):
        if ci is None: ci = self.ci
        return make_rdm2(ci, self.nmo, self.nocc)

class _ERIS(object):
    def __init__(self, myci, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = myci._scf.mo_coeff
        mol = myci.mol
        nocc = myci.nocc
        nmo = myci.nmo
        nvir = nmo - nocc
        moidx = uccsd.get_umoidx(myci)
        self.fock, mo_coeff, orbspin = uccsd.uspatial2spin(myci, moidx, mo_coeff)
        self.mo_coeff = mo_coeff
        myci.orbspin = self.orbspin = orbspin

        eri = ao2mo.kernel(myci._scf._eri, mo_coeff, compact=False)
        eri = eri.reshape([nmo]*4)
        for i in range(nmo):
            for j in range(i):
                if orbspin[i] != orbspin[j]:
                    eri[i,j,:,:] = eri[j,i,:,:] = 0.
                    eri[:,:,i,j] = eri[:,:,j,i] = 0.
        eri = eri - eri.transpose(0,3,2,1)
        eri = eri.transpose(0,2,1,3)

        self.oooo = eri[:nocc,:nocc,:nocc,:nocc]
        self.ovvo = eri[:nocc,nocc:,nocc:,:nocc]
        self.ovov = eri[:nocc,nocc:,:nocc,nocc:]
        self.ooov = eri[:nocc,:nocc,:nocc,nocc:]
        self.ovoo = eri[:nocc,nocc:,:nocc,:nocc]
        self.oovv = eri[:nocc,:nocc,nocc:,nocc:]
        self.ovvv = eri[:nocc,nocc:,nocc:,nocc:]
        self.vvvv = eri[nocc:,nocc:,nocc:,nocc:]


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import fci
    from pyscf.cc.addons import spatial2spin
    numpy.random.seed(12)
    nocc = 3
    nvir = 5
    nmo = nocc + nvir

    orbspin = numpy.zeros(nmo*2, dtype=int)
    orbspin[1::2] = 1
    c1a = numpy.random.random((nocc,nvir))
    c1b = numpy.random.random((nocc,nvir))
    c2aa = numpy.random.random((nocc,nocc,nvir,nvir))
    c2bb = numpy.random.random((nocc,nocc,nvir,nvir))
    c2ab = numpy.random.random((nocc,nocc,nvir,nvir))
    c1 = spatial2spin((c1a, c1b), orbspin)
    c2 = spatial2spin((c2aa, c2ab, c2bb), orbspin)
    cisdvec = amplitudes_to_cisdvec(1., c1, c2)
    fcivec = to_fci(cisdvec, nocc*2, orbspin)
    cisdvec1 = from_fci(fcivec, nocc*2, orbspin)
    print(abs(cisdvec-cisdvec1).sum())
    ci1 = to_fci(cisdvec1, nocc*2, orbspin)
    print(abs(fcivec-ci1).sum())

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0.,-1.    , 1.   )],
    ]
    mol.charge = -2
    mol.spin = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    ehf0 = mf.e_tot - mol.energy_nuc()
    myci = CISD(mf)
    eris = myci.ao2mo()

    numpy.random.seed(12)
    nocca, noccb = mol.nelec
    nmo = mf.mo_occ[0].size
    nvira = nmo - nocca
    nvirb = nmo - noccb
    #cisdvec = myci.get_init_guess(eris)[1]
    c1a  = .1 * numpy.random.random((nocca,nvira))
    c1b  = .1 * numpy.random.random((noccb,nvirb))
    c2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
    c2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
    c2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
    c1 = spatial2spin((c1a, c1b), eris.orbspin)
    c2 = spatial2spin((c2aa, c2ab, c2bb), eris.orbspin)
    cisdvec = amplitudes_to_cisdvec(1., c1, c2)

    hcisd0 = contract(myci, cisdvec, eris)
    eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
    eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
    eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                    mf.mo_coeff[1], mf.mo_coeff[1]])
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    h2e = fci.direct_uhf.absorb_h1e((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                    h1a.shape[0], mol.nelec, .5)
    fcivec = to_fci(cisdvec, mol.nelectron, eris.orbspin)
    hci1 = fci.direct_uhf.contract_2e(h2e, fcivec, h1a.shape[0], mol.nelec)
    hci1 -= ehf0 * fcivec
    hcisd1 = from_fci(hci1, mol.nelectron, eris.orbspin)
    print(numpy.linalg.norm(hcisd1-hcisd0) / numpy.linalg.norm(hcisd0))

    hdiag0 = make_diagonal(myci, eris)
    hdiag0 = to_fci(hdiag0, mol.nelectron, eris.orbspin).ravel()
    hdiag0 = from_fci(hdiag0, mol.nelectron, eris.orbspin).ravel()
    hdiag1 = fci.direct_uhf.make_hdiag((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                       h1a.shape[0], mol.nelec)
    hdiag1 = from_fci(hdiag1, mol.nelectron, eris.orbspin).ravel()
    print(numpy.linalg.norm(abs(hdiag0)-abs(hdiag1)))

    ecisd = myci.kernel()[0]
    print(ecisd, mf.e_tot)
    efci = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                 h1a.shape[0], mol.nelec)[0]
    print(ecisd, ecisd - -0.037067274690894436, '> E(fci)', efci-ehf0)

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0.,-1.    , 1.   )],
    ]
    mol.charge = 2
    mol.spin = 2
    mol.basis = '6-31g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    ehf0 = mf.e_tot - mol.energy_nuc()
    myci = CISD(mf)
    eris = myci.ao2mo()
    ecisd = myci.kernel(eris=eris)[0]
    eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
    eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
    eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                    mf.mo_coeff[1], mf.mo_coeff[1]])
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    efci, fcivec = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                         h1a.shape[0], mol.nelec)
    print(ecisd, '== E(fci)', efci-ehf0)
    dm1ref, dm2ref = fci.direct_uhf.make_rdm12s(fcivec, h1a.shape[0], mol.nelec)
    dm2ref = [x.transpose(0,2,1,3) for x in dm2ref]
    nmo = myci.nmo
    rdm1 = make_rdm1(myci.ci, nmo, mol.nelectron)
    rdm2 = make_rdm2(myci.ci, nmo, mol.nelectron)
    idxa = eris.orbspin == 0
    idxb = eris.orbspin == 1
    print('dm1a', abs(dm1ref[0] - rdm1[idxa][:,idxa]).max())
    print('dm1b', abs(dm1ref[1] - rdm1[idxb][:,idxb]).max())
    print('dm2aa', abs(dm2ref[0] - rdm2[idxa][:,idxa][:,:,idxa][:,:,:,idxa]).max())
    print('dm2ab', abs(dm2ref[1] - rdm2[idxa][:,idxb][:,:,idxa][:,:,:,idxb]).max())
    print('dm2bb', abs(dm2ref[2] - rdm2[idxb][:,idxb][:,:,idxb][:,:,:,idxb]).max())
#    nocca, noccb = mol.nelec
#    rdm2ab = rdm2[idxa][:,idxb][:,:,idxa][:,:,:,idxb]
#    print(abs(rdm2ab[:nocca,:noccb,:nocca,:noccb]-dm2ref[1][:nocca,:noccb,:nocca,:noccb]).sum(), 'oooo')
#    print(abs(rdm2ab[nocca:,noccb:,nocca:,noccb:]-dm2ref[1][nocca:,noccb:,nocca:,noccb:]).sum(), 'vvvv')
#    print(abs(rdm2ab[nocca:,:noccb,:nocca,:noccb]-dm2ref[1][nocca:,:noccb,:nocca,:noccb]).sum(), 'vooo')
#    print(abs(rdm2ab[:nocca,noccb:,nocca:,noccb:]-dm2ref[1][:nocca,noccb:,nocca:,noccb:]).sum(), 'ovvv')
#    print(abs(rdm2ab[nocca:,:noccb,:nocca,noccb:]-dm2ref[1][nocca:,:noccb,:nocca,noccb:]).sum(), 'voov')
#    print(abs(rdm2ab[:nocca,:noccb,nocca:,noccb:]-dm2ref[1][:nocca,:noccb,nocca:,noccb:]).sum(), 'ovov')

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g',}
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    myci = CISD(mf)
    eris = myci.ao2mo()
    ecisd, civec = myci.kernel(eris=eris)
    print(ecisd - -0.048878084082066106)

    nmo = eris.mo_coeff.shape[1]
    rdm1 = make_rdm1(civec, nmo, mol.nelectron)
    rdm2 = make_rdm2(civec, nmo, mol.nelectron)

    eri = ao2mo.kernel(mf._eri, eris.mo_coeff, compact=False)
    eri = eri.reshape([nmo]*4)
    for i in range(nmo):
        for j in range(i):
            if eris.orbspin[i] != eris.orbspin[j]:
                eri[i,j,:,:] = eri[j,i,:,:] = 0.
                eri[:,:,i,j] = eri[:,:,j,i] = 0.
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    h1e = numpy.zeros((nmo,nmo))
    idxa = eris.orbspin == 0
    idxb = eris.orbspin == 1
    h1e[idxa[:,None]&idxa] = h1a.ravel()
    h1e[idxb[:,None]&idxb] = h1b.ravel()
    e2 = (numpy.einsum('ij,ji', h1e, rdm1) +
          numpy.einsum('ikjl,ijkl', eri, rdm2) * .5)
    print(ecisd + mf.e_tot - mol.energy_nuc() - e2)   # = 0

    print(abs(rdm1 - numpy.einsum('ikjk->ij', rdm2)/(mol.nelectron-1)).sum())

