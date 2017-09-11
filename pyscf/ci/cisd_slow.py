#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
non-relativistic RCISD

The RCISD equation is  H C = C e  where e = E_HF + E_CORR
'''

import time
import numpy
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.fci import cistring
from functools import reduce

def kernel(myci, eris, ci0=None, max_cycle=50, tol=1e-8,
           verbose=logger.INFO):
    mol = myci.mol
    nmo = myci.nmo
    nocc = myci.nocc
    mo_energy = eris.fock.diagonal()
    diag = make_diagonal(eris)
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
# MP2 initial guess
        nvir = nmo - nocc
        e_i = mo_energy[:nocc]
        e_a = mo_energy[nocc:]
        ci0 = numpy.zeros(1+nocc*nvir+(nocc*nvir)**2)
        ci0[0] = 1
        t2 = 2*eris.voov.transpose(1,2,0,3) - eris.voov.transpose(1,2,3,0)
        t2 /= lib.direct_sum('i+j-a-b', e_i, e_i, e_a, e_a)
        ci0[1+nocc*nvir:] = t2.ravel()

    def op(x):
        return contract(myci, x, eris)

    def precond(x, e, *args):
        return x / (diag - e)

    def cisd_dot(x1, x2):
        return dot(x1, x2, nocc, nvir)

    ecisd, ci = lib.davidson(op, ci0, precond, max_cycle=max_cycle, tol=tol,
                             dot=cisd_dot, verbose=verbose)
    conv = True  # It should be checked in lib.davidson function
    return conv, ecisd, ci

def make_diagonal(eris):
    mo_energy = eris.fock.diagonal()
    nocc = eris.nocc
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
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    c0 = civec[0]
    c1 = civec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)

    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    eris_vovv = lib.unpack_tril(eris.vovv).reshape(nvir,nocc,nvir,-1)

    t1 = numpy.einsum('ib,ab->ia', c1, fvv)
    t1-= numpy.einsum('ja,ji->ia', c1, foo)
    t1+= numpy.einsum('jb,aijb->ia', c1, eris.voov) * 2
    t1-= numpy.einsum('jb,abij->ia', c1, eris.vvoo)
    theta = c2 * 2 - c2.transpose(1,0,2,3)
    t1 += numpy.einsum('ijab,jb->ia', theta, fov)
    t1 += numpy.einsum('ijbc,cjba->ia', theta, eris_vovv)
    t1 -= numpy.einsum('jkba,bjki->ia', theta, eris.vooo)

    tw = numpy.einsum('bc,ijac->ijab', fvv, c2)
    tw-= numpy.einsum('kj,kiba->ijab', foo, c2)

    theta = c2 * 2 - c2.transpose(1,0,2,3)
    tw -= numpy.einsum('ikac,bckj->ijab', c2, eris.vvoo)
    tw -= numpy.einsum('ikca,bckj->ijba', c2, eris.vvoo)
    tw += numpy.einsum('ikac,ckjb->ijab', theta, eris.voov)

    tw += numpy.einsum('ia,jb->ijab', c1, fov)
    tw += numpy.einsum('jc,aicb->jiba', c1, eris_vovv)
    tw -= numpy.einsum('ka,bjik->jiba', c1, eris.vooo)

    t2  = tw + tw.transpose(1,0,3,2)
    t2 += numpy.einsum('kilj,klab->ijab', eris.oooo, c2)
    t2 += myci.add_wvvVV(c2, eris)

    t1 += fov * c0
    t2 += eris.voov.transpose(1,2,0,3) * c0
    tau = c2*2 - c2.transpose(1,0,2,3)
    t0  = numpy.einsum('ia,ia', fov, c1) * 2
    t0 += numpy.einsum('aijb,ijab', eris.voov, tau)
    cinew = numpy.hstack((t0, t1.ravel(), t2.ravel()))
    return cinew

def dot(v1, v2, nocc, nvir):
    p0 = nocc*nvir+1
    p1 = p0+(nocc*nvir)**2
    hijab = v2[p0:p1].reshape(nocc,nocc,nvir,nvir)
    cijab = v1[p0:p1].reshape(nocc,nocc,nvir,nvir)
    hIJAB = hijab - hijab.transpose(1,0,2,3)
    cIJAB = cijab - cijab.transpose(1,0,2,3)
    return v1[0] * v2[0] + 2*numpy.dot(v1[1:p0], v2[1:p0]) \
            + numpy.dot(v1[p0:p1], v2[p0:p1]) \
            + .5*numpy.dot(cIJAB.reshape(-1), hIJAB.reshape(-1))

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

def t2strs(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    addrs = []
    signs = []
    for a in range(nocc+1, norb):
        for b in range(nocc, a):
            for i in reversed(range(1,nocc)):
                for j in reversed(range(i)):
                    str1 = hf_str ^ (1 << j) | (1 << b)
                    sign = cistring.cre_des_sign(b, j, hf_str)
                    sign*= cistring.cre_des_sign(a, i, str1)
                    str1^= (1 << i) | (1 << a)
                    addrs.append(cistring.str2addr(norb, nelec, str1))
                    signs.append(sign)
    return numpy.asarray(addrs), numpy.asarray(signs)

def to_fci(cisdvec, norb, nelec):
    nocc = nelec // 2
    nvir = norb - nocc
    c0 = cisdvec[0]
    c1 = cisdvec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = cisdvec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
    t1addr, t1sign = t1strs(norb, nocc)

    na = fci.cistring.num_strings(norb, nocc)
    fcivec = numpy.zeros((na,na))
    fcivec[0,0] = c0
    c1 = c1[::-1].T.ravel()
    fcivec[0,t1addr] = fcivec[t1addr,0] = c1 * t1sign
    c2ab = c2[::-1,::-1].transpose(2,0,3,1).reshape(nocc*nvir,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1sign, t1sign, c2ab)
    lib.takebak_2d(fcivec, c2ab, t1addr, t1addr)

    if nocc > 1 and nvir > 1:
        t2addr, t2sign = t2strs(norb, nocc)
        c2aa = []
        for a in range(1,nvir):
            for b in range(a):
                for i in reversed(range(1,nocc)):
                    for j in reversed(range(i)):
                        c2aa.append((c2[i,j,a,b] - c2[j,i,a,b]))
        c2aa = numpy.asarray(c2aa)
        fcivec[0,t2addr] = fcivec[t2addr,0] = c2aa * t2sign
    return fcivec

def from_fci(ci0, norb, nelec):
    nocc = nelec // 2
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
    C2 = c2 - c2.transpose(1,0,2,3)
    dov = c0*c1 * 2
    dov += numpy.einsum('jb,ijab->ia', c1, c2+C2) * 2
    doo  =-numpy.einsum('ia,ka->ik', c1, c1) * 2
    doo -= numpy.einsum('ijab,ikab->jk', c2, c2) * 2
    doo -= numpy.einsum('ijab,ikab->jk', C2, C2)
    dvv  = numpy.einsum('ia,ic->ca', c1, c1) * 2
    dvv += numpy.einsum('ijab,ijac->cb', c2, c2) * 2
    dvv += numpy.einsum('ijab,ijac->cb', C2, C2)

    rdm1 = numpy.empty((nmo,nmo))
    rdm1[:nocc,nocc:] = dov
    rdm1[nocc:,:nocc] = dov.T
    rdm1[:nocc,:nocc] = doo
    rdm1[nocc:,nocc:] = dvv

    for i in range(nocc):
        rdm1[i,i] += 2
    return rdm1

def make_rdm2(ci, nmo, nocc):
    '''spin-traced 2pdm in physicist's notation
    '''
    nvir = nmo - nocc
    c0 = ci[0]
    c1 = ci[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = ci[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
    C2 = c2 - c2.transpose(1,0,2,3)
    theta = C2 + c2
    doovv = c0*c2
    dvvvo = numpy.einsum('ia,ikcd->cdak', c1, c2)
    dovoo =-numpy.einsum('ia,klac->ickl', c1, c2)
    doooo = numpy.einsum('klab,ijab->klij', c2, c2)
    dvvvv = numpy.einsum('ijcd,ijab->cdab', c2, c2)

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
    dovov  = numpy.einsum('ijab,ikac->jckb', c2, theta) * -2
    dovov -= numpy.einsum('ijab,jkca->ickb', c2, theta) * 2
    dovov -= numpy.einsum('ia,kc->icka', c1, c1) * 2
    dvoov  = numpy.einsum('ijab,ikac->cjkb', theta, theta) * 2
    dvoov += numpy.einsum('ia,kc->cika', c1, c1) * 4

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

##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.ci = None

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    @property
    def nocc(self):
        self._nocc = numpy.count_nonzero(self.mo_occ)
        return self._nocc

    @property
    def nmo(self):
        self._nmo = len(self.mo_occ)
        return self._nmo

    def kernel(self, ci0=None, mo_coeff=None, eris=None):
        return self.cisd(ci0, mo_coeff, eris)
    def cisd(self, ci0=None, mo_coeff=None, eris=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        self.converged, self.e_corr, self.ci = \
                kernel(self, eris, ci0, max_cycle=self.max_cycle,
                       tol=self.conv_tol, verbose=self.verbose)
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(CISD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.ci

    def ao2mo(self, mo_coeff=None):
        return _RCISD_ERIs(self, mo_coeff)

    def add_wvvVV(self, t2, eris):
        nvir = self.nmo - self.nocc
        eris_vvvv = ao2mo.restore(1, eris.vvvv, nvir)
        return numpy.einsum('ijcd,acbd->ijab', t2, eris_vvvv)

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

class _RCISD_ERIs(object):
    def __init__(self, myci, mo_coeff, method='incore'):
        mol = myci.mol
        mf = myci._scf
        nocc = self.nocc = myci.nocc
        nmo = myci.nmo
        nvir = nmo - nocc
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = myci.mo_coeff
        if (method == 'incore' and mf._eri is not None):
            eri = ao2mo.kernel(mf._eri, mo_coeff, verbose=myci.verbose)
        else:
            eri = ao2mo.kernel(mol, mo_coeff, verbose=myci.verbose)
        eri = ao2mo.restore(1, eri, nmo)
        eri = eri.reshape(nmo,nmo,nmo,nmo)

        self.oooo = eri[:nocc,:nocc,:nocc,:nocc]
        self.vvoo = eri[nocc:,nocc:,:nocc,:nocc]
        self.vooo = eri[nocc:,:nocc,:nocc,:nocc]
        self.voov = eri[nocc:,:nocc,:nocc,nocc:]
        self.vovv = lib.pack_tril(eri[nocc:,:nocc,nocc:,nocc:].reshape(-1,nvir,nvir))
        self.vvvv = ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:].copy(), nvir)

        dm = mf.make_rdm1()
        vhf = mf.get_veff(mol, dm)
        h1 = mf.get_hcore(mol)
        self.fock = reduce(numpy.dot, (mo_coeff.T, h1 + vhf, mo_coeff))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import fci
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
    mf = scf.RHF(mol).run(max_cycle=1)
    etot = CISD(mf).kernel()[0] + mf.e_tot
    mf = scf.RHF(mol).run()
    ecisd, civec = CISD(mf).kernel()
    print(ecisd + mf.e_tot - etot)
    print(ecisd - -0.024780739973407784)
    nmo = mf.mo_occ.size
    nocc = mol.nelectron // 2
    h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eci,fcivec = fci.direct_spin0.kernel(h1e, h2e, nmo, mol.nelectron)
    eci = eci + mol.energy_nuc() - mf.e_tot
    print(ecisd - eci)
    rdm1 = make_rdm1(civec, nmo, nocc)
    rdm2 = make_rdm2(civec, nmo, nocc)
    dm1ref, dm2ref = fci.direct_spin0.make_rdm12(fcivec, nmo, mol.nelectron)
    dm2ref = dm2ref.transpose(0,2,1,3)
    print(abs(rdm1-dm1ref).max(), 'r')
    print(abs(rdm2-dm2ref).max(), 'r')

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
