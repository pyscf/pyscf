#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Solve CISD equation  H C = C e  where e = E_HF + E_CORR
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import ccsd_rdm
from pyscf.fci import cistring
from functools import reduce
from pyscf import __config__

BLKMIN = getattr(__config__, 'ci_cisd_blkmin', 4)


def kernel(myci, eris, ci0=None, max_cycle=50, tol=1e-8, verbose=logger.INFO):
    log = logger.new_logger(myci, verbose)
    mol = myci.mol
    diag = myci.make_diagonal(eris)
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
        ci0 = myci.get_init_guess(eris=eris, nroots=myci.nroots, diag=diag)[1]

    def op(xs):
        return [myci.contract(x, eris) for x in xs]

    def precond(x, e, *args):
        diagd = diag - (e-myci.level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return x / diagd

    if myci._dot is not None:
        nmo = myci.nmo
        nocc = myci.nocc
        def cisd_dot(x1, x2):
            return myci._dot(x1, x2, nmo, nocc)
    else:
        cisd_dot = numpy.dot

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
    jdiag[:nocc,nocc:] = numpy.einsum('iijj->ij', eris.oovv)
    kdiag[:nocc,nocc:] = numpy.einsum('ijji->ij', eris.ovvo)
    if eris.vvvv is not None and len(eris.vvvv.shape) == 2:
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
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc)

    t2 = myci._add_vvvv(c2, eris, t2sym='jiba')
    t2 *= .5  # due to t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

    foo = eris.fock[:nocc,:nocc].copy()
    fov = eris.fock[:nocc,nocc:].copy()
    fvv = eris.fock[nocc:,nocc:].copy()

    t1  = fov * c0
    t1 += numpy.einsum('ib,ab->ia', c1, fvv)
    t1 -= numpy.einsum('ja,ji->ia', c1, foo)

    t2 += lib.einsum('kilj,klab->ijab', _cp(eris.oooo)*.5, c2)
    t2 += lib.einsum('ijac,bc->ijab', c2, fvv)
    t2 -= lib.einsum('kj,kiba->jiba', foo, c2)
    t2 += numpy.einsum('ia,jb->ijab', c1, fov)

    unit = nocc*nvir**2 + nocc**2*nvir*3 + 1
    max_memory = max(0, myci.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(BLKMIN, int(max_memory*.9e6/8/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)
    nvir_pair = nvir * (nvir+1) // 2
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_oVoV = _cp(_cp(eris.oovv[:,:,p0:p1]).transpose(0,2,1,3))
        tmp = lib.einsum('kbjc,ikca->jiba', eris_oVoV, c2)
        t2[:,:,p0:p1] -= tmp*.5
        t2[:,:,p0:p1] -= tmp.transpose(1,0,2,3)
        tmp = None

        eris_ovvo = _cp(eris.ovvo[:,p0:p1])
        t2[:,:,p0:p1] += eris_ovvo.transpose(0,3,1,2) * (c0*.5)
        t1 += numpy.einsum('ia,iabj->jb', c1[:,p0:p1], eris_ovvo) * 2
        t1[:,p0:p1] -= numpy.einsum('ib,iajb->ja', c1, eris_oVoV)

        ovov = -.5 * eris_oVoV
        ovov += eris_ovvo.transpose(3,1,0,2)
        eris_oVoV = eris_oovv = None
        theta = c2[:,:,p0:p1].transpose(2,0,1,3) * 2
        theta-= c2[:,:,p0:p1].transpose(2,1,0,3)
        for j in range(nocc):
            t2[:,j] += lib.einsum('ckb,ckia->iab', ovov[j], theta)
        tmp = ovov = None

        t1 += numpy.einsum('aijb,ia->jb', theta, fov[:,p0:p1])

        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        t1 -= lib.einsum('bjka,jbki->ia', theta, eris_ovoo)
        t2[:,:,p0:p1] -= lib.einsum('jbik,ka->jiba', eris_ovoo.conj(), c1)
        eris_vooo = None

        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1)).conj()
        t1 += lib.einsum('cjib,jcba->ia', theta, eris_ovvv)
        t2[:,:,p0:p1] += lib.einsum('iacb,jc->ijab', eris_ovvv, c1)
        tmp = eris_ovvv = None

    #:t2 + t2.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2[i,:i]+= t2[:i,i].transpose(0,2,1)
            t2[:i,i] = t2[i,:i].transpose(0,2,1)
        t2[i,i] = t2[i,i] + t2[i,i].T

    t0  = numpy.einsum('ia,ia->', fov, c1) * 2
    t0 += numpy.einsum('iabj,ijab->', eris.ovvo, c2) * 2
    t0 -= numpy.einsum('iabj,jiab->', eris.ovvo, c2)
    cinew = numpy.hstack((t0, t1.ravel(), t2.ravel()))
    return cinew

def amplitudes_to_cisdvec(c0, c1, c2):
    return numpy.hstack((c0, c1.ravel(), c2.ravel()))

def cisdvec_to_amplitudes(civec, nmo, nocc):
    nvir = nmo - nocc
    c0 = civec[0]
    c1 = civec[1:nocc*nvir+1].reshape(nocc,nvir)
    c2 = civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir)
    return c0, c1, c2

def dot(v1, v2, nmo, nocc):
    nvir = nmo - nocc
    hijab = v2[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    cijab = v1[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    val = numpy.dot(v1, v2) * 2 - v1[0]*v2[0]
    val-= numpy.einsum('jiab,ijab->', cijab, hijab)
    return val

def t1strs(norb, nelec):
    '''FCI strings (address) for CIS single-excitation amplitues'''
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

def to_fcivec(cisdvec, norb, nelec, frozen=0):
    '''Convert CISD coefficients to FCI coefficients'''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)

    frozen_mask = numpy.zeros(norb, dtype=bool)
    if isinstance(frozen, (int, numpy.integer)):
        nfroz = frozen
        frozen_mask[:frozen] = True
    else:
        nfroz = len(frozen)
        frozen_mask[frozen] = True

    nocc = numpy.count_nonzero(~frozen_mask[:neleca])
    nmo = norb - nfroz
    nvir = nmo - nocc
    c0, c1, c2 = cisdvec_to_amplitudes(cisdvec, nmo, nocc)
    t1addr, t1sign = t1strs(nmo, nocc)

    na = cistring.num_strings(nmo, nocc)
    fcivec = numpy.zeros((na,na))
    fcivec[0,0] = c0
    c1 = c1[::-1].T.ravel()
    fcivec[0,t1addr] = fcivec[t1addr,0] = c1 * t1sign
    c2ab = c2[::-1,::-1].transpose(2,0,3,1).reshape(nocc*nvir,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1sign, t1sign, c2ab)
    lib.takebak_2d(fcivec, c2ab, t1addr, t1addr)

    if nocc > 1 and nvir > 1:
        hf_str = int('1'*nocc, 2)
        for a in range(nocc, nmo):
            for b in range(nocc, a):
                for i in reversed(range(1, nocc)):
                    for j in reversed(range(i)):
                        c2aa = c2[i,j,a-nocc,b-nocc] - c2[j,i,a-nocc,b-nocc]
                        str1 = hf_str ^ (1 << j) | (1 << b)
                        c2aa*= cistring.cre_des_sign(b, j, hf_str)
                        c2aa*= cistring.cre_des_sign(a, i, str1)
                        str1^= (1 << i) | (1 << a)
                        addr = cistring.str2addr(nmo, nocc, str1)
                        fcivec[0,addr] = fcivec[addr,0] = c2aa

    if nfroz == 0:
        return fcivec

    assert(norb < 63)

    strs = cistring.gen_strings4orblist(range(norb), neleca)
    na = len(strs)
    count = numpy.zeros(na, dtype=int)
    parity = numpy.zeros(na, dtype=bool)
    core_mask = numpy.ones(na, dtype=bool)
    # During the loop, count saves the number of occupied orbitals that
    # lower (with small orbital ID) than the present orbital i.
    # Moving all the frozen orbitals to the beginning of the orbital list
    # (before the occupied orbitals) leads to parity odd (= True, with
    # negative sign) or even (= False, with positive sign).
    for i in range(norb):
        if frozen_mask[i]:
            if i < neleca:
                # frozen occupied orbital should be occupied
                core_mask &= (strs & (1<<i)) != 0
                parity ^= (count & 1) == 1
            else:
                # frozen virtual orbital should not be occupied.
                # parity is not needed since it's unoccupied
                core_mask &= (strs & (1<<i)) == 0
        else:
            count += (strs & (1<<i)) != 0
    sub_strs = strs[core_mask & (count == nocc)]
    addrs = cistring.strs2addr(norb, neleca, sub_strs)
    fcivec1 = numpy.zeros((na,na))
    fcivec1[addrs[:,None],addrs] = fcivec
    fcivec1[parity,:] *= -1
    fcivec1[:,parity] *= -1
    return fcivec1

def from_fcivec(ci0, norb, nelec, frozen=0):
    '''Extract CISD coefficients from FCI coefficients'''
    if frozen is not 0:
        raise NotImplementedError
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

def make_rdm1(myci, civec=None, nmo=None, nocc=None):
    '''
    Spin-traced one-particle density matrix in MO basis (the occupied-virtual
    blocks from the orbital response contribution are not included).

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    return ccsd_rdm._make_rdm1(myci, d1, with_frozen=True)

def make_rdm2(myci, civec=None, nmo=None, nocc=None):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    f = lib.H5TmpFile()
    d2 = _gamma2_outcore(myci, civec, nmo, nocc, f, False)
    return ccsd_rdm._make_rdm2(myci, d1, d2, with_dm1=True, with_frozen=True)

def _gamma1_intermediates(myci, civec, nmo, nocc):
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc)
    dvo = c0.conj() * c1.T
    dvo += numpy.einsum('jb,ijab->ai', c1.conj(), c2) * 2
    dvo -= numpy.einsum('jb,ijba->ai', c1.conj(), c2)
    dov = dvo.T.conj()

    theta = c2*2 - c2.transpose(0,1,3,2)
    doo  =-numpy.einsum('ia,ka->ik', c1.conj(), c1)
    doo -= lib.einsum('ijab,ikab->jk', c2.conj(), theta)
    dvv  = numpy.einsum('ia,ic->ac', c1, c1.conj())
    dvv += lib.einsum('ijab,ijac->bc', theta, c2.conj())
    return doo, dov, dvo, dvv

def _gamma2_intermediates(myci, civec, nmo, nocc, compress_vvvv=False):
    f = lib.H5TmpFile()
    _gamma2_outcore(myci, civec, nmo, nocc, f, compress_vvvv)
    d2 = (f['dovov'].value, f['dvvvv'].value, f['doooo'].value, f['doovv'].value,
          f['dovvo'].value, None,             f['dovvv'].value, f['dooov'].value)
    return d2

def _gamma2_outcore(myci, civec, nmo, nocc, h5fobj, compress_vvvv=False):
    log = logger.Logger(myci.stdout, myci.verbose)
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    nvir_pair = nvir * (nvir+1) // 2
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc)

    h5fobj['dovov'] = (2*c0*c2.conj().transpose(0,2,1,3) -
                       c0*c2.conj().transpose(1,2,0,3))

    doooo = lib.einsum('ijab,klab->ijkl', c2.conj(), c2)
    h5fobj['doooo'] = doooo.transpose(0,2,1,3) - doooo.transpose(1,2,0,3)*.5
    doooo = None

    dooov =-lib.einsum('ia,klac->klic', c1*2, c2.conj())
    h5fobj['dooov'] = dooov.transpose(0,2,1,3)*2 - dooov.transpose(1,2,0,3)
    dooov = None

    #:dvovv = numpy.einsum('ia,ikcd->akcd', c1, c2) * 2
    #:dvvvv = lib.einsum('ijab,ijcd->abcd', c2, c2)
    max_memory = max(0, myci.max_memory - lib.current_memory()[0])
    unit = max(nocc**2*nvir*2+nocc*nvir**2*3 + 1, nvir**3*2+nocc*nvir**2 + 1)
    blksize = min(nvir, max(BLKMIN, int(max_memory*.9e6/8/unit)))
    iobuflen = int(256e6/8/blksize)
    log.debug1('rdm intermediates: block size = %d, nvir = %d in %d blocks',
               blksize, nocc, int((nvir+blksize-1)/blksize))
    dtype = numpy.result_type(civec).char
    dovvv = h5fobj.create_dataset('dovvv', (nocc,nvir,nvir,nvir), dtype,
                                  chunks=(nocc,min(nocc,nvir),1,nvir))
    if compress_vvvv:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir_pair,nvir_pair), dtype)
    else:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir,nvir,nvir,nvir), dtype)

    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        theta = c2[:,:,p0:p1] - c2[:,:,p0:p1].transpose(1,0,2,3) * .5
        gvvvv = lib.einsum('ijab,ijcd->abcd', theta.conj(), c2)
        if compress_vvvv:
# symmetrize dvvvv because it does not affect the results of cisd_grad
# dvvvv = (dvvvv+dvvvv.transpose(0,1,3,2)) * .5
# dvvvv = (dvvvv+dvvvv.transpose(1,0,2,3)) * .5
# now dvvvv == dvvvv.transpose(0,1,3,2) == dvvvv.transpose(1,0,3,2)
            tmp = numpy.empty((nvir,nvir,nvir))
            tmpvvvv = numpy.empty((p1-p0,nvir,nvir_pair))
            for i in range(p1-p0):
                tmp[:] = gvvvv[i].conj().transpose(1,0,2)
                lib.pack_tril(tmp+tmp.transpose(0,2,1), out=tmpvvvv[i])
            # tril of (dvvvv[p0:p1,p0:p1]+dvvvv[p0:p1,p0:p1].T)
            for i in range(p0, p1):
                for j in range(p0, i):
                    tmpvvvv[i-p0,j] += tmpvvvv[j-p0,i]
                tmpvvvv[i-p0,i] *= 2
            for i in range(p1, nvir):
                off = i * (i+1) // 2
                dvvvv[off+p0:off+p1] = tmpvvvv[:,i]
            for i in range(p0, p1):
                off = i * (i+1) // 2
                if p0 > 0:
                    tmpvvvv[i-p0,:p0] += dvvvv[off:off+p0]
                dvvvv[off:off+i+1] = tmpvvvv[i-p0,:i+1] * .25
            tmp = tmpvvvv = None
        else:
            for i in range(p0, p1):
                dvvvv[i] = gvvvv[i-p0].conj().transpose(1,0,2)

        gvovv = numpy.einsum('ia,ikcd->akcd', c1[:,p0:p1].conj()*2, c2)
        gvovv = gvovv.conj()
        dovvv[:,:,p0:p1] = gvovv.transpose(1,3,0,2)*2 - gvovv.transpose(1,2,0,3)

    theta = c2*2 - c2.transpose(1,0,2,3)
    doovv  = numpy.einsum('ia,kc->ikca', c1.conj(), -c1)
    doovv -= lib.einsum('kjcb,kica->jiab', c2.conj(), theta)
    doovv -= lib.einsum('ikcb,jkca->ijab', c2.conj(), theta)
    h5fobj['doovv'] = doovv
    doovv = None

    dovvo  = lib.einsum('ikac,jkbc->iabj', theta.conj(), theta)
    dovvo += numpy.einsum('ia,kc->iack', c1.conj(), c1) * 2
    h5fobj['dovvo'] = dovvo
    theta = dovvo = None

    dvvov = None
    return (h5fobj['dovov'], h5fobj['dvvvv'], h5fobj['doooo'], h5fobj['doovv'],
            h5fobj['dovvo'], dvvov          , h5fobj['dovvv'], h5fobj['dooov'])

def trans_rdm1(myci, cibra, ciket, nmo=None, nocc=None):
    '''
    Spin-traced one-particle transition density matrix in MO basis.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    c0bra, c1bra, c2bra = myci.cisdvec_to_amplitudes(cibra, nmo, nocc)
    c0ket, c1ket, c2ket = myci.cisdvec_to_amplitudes(ciket, nmo, nocc)

    dvo = c0bra.conj() * c1ket.T
    dvo += numpy.einsum('jb,ijab->ai', c1bra.conj(), c2ket) * 2
    dvo -= numpy.einsum('jb,ijba->ai', c1bra.conj(), c2ket)

    dov = c0ket * c1bra.conj()
    dov += numpy.einsum('jb,ijab->ia', c1ket, c2bra.conj()) * 2
    dov -= numpy.einsum('jb,ijba->ia', c1ket, c2bra.conj())

    theta = c2ket*2 - c2ket.transpose(0,1,3,2)
    doo  =-numpy.einsum('ia,ka->ik', c1bra.conj(), c1ket)
    doo -= lib.einsum('ijab,ikab->jk', c2bra.conj(), theta)
    dvv  = numpy.einsum('ia,ic->ac', c1ket, c1bra.conj())
    dvv += lib.einsum('ijab,ijac->bc', theta, c2bra.conj())

    dm1 = numpy.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo * 2
    dm1[:nocc,nocc:] = dov * 2
    dm1[nocc:,:nocc] = dvo * 2
    dm1[nocc:,nocc:] = dvv * 2
    norm = dot(cibra, ciket, nmo, nocc)
    dm1[numpy.diag_indices(nocc)] += 2 * norm

    if not (myci.frozen is 0 or myci.frozen is None):
        nmo = myci.mo_occ.size
        nocc = numpy.count_nonzero(myci.mo_occ > 0)
        rdm1 = numpy.zeros((nmo,nmo), dtype=dm1.dtype)
        rdm1[numpy.diag_indices(nocc)] = 2 * norm
        moidx = numpy.where(myci.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1
    return dm1


def as_scanner(ci):
    '''Generating a scanner/solver for CISD PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CISD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CISD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, ci
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> ci_scanner = ci.CISD(scf.RHF(mol)).as_scanner()
        >>> e_tot = ci_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot = ci_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(ci, lib.SinglePointScanner):
        return ci

    logger.info(ci, 'Set %s as a scanner', ci.__class__)

    class CISD_Scanner(ci.__class__, lib.SinglePointScanner):
        def __init__(self, ci):
            self.__dict__.update(ci.__dict__)
            self._scf = ci._scf.as_scanner()
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mf_scanner = self._scf
            mf_scanner(mol)
            self.mol = mol
            self.mo_coeff = mf_scanner.mo_coeff
            self.mo_occ = mf_scanner.mo_occ
# FIXME: Whether to use the initial guess from last step? If root flips, large
# errors may be found in the solutions
            self.kernel(self.ci, **kwargs)[0]
            return self.e_tot
    return CISD_Scanner(ci)


class CISD(lib.StreamObject):
    '''restricted CISD

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            converge threshold.  Default is 1e-9.
        max_cycle : int
            max number of iterations.  Default is 50.
        max_space : int
            Davidson diagonalization space size.  Default is 12.
        direct : bool
            AO-direct CISD. Default is False.
        async_io : bool
            Allow for asynchronous function execution. Default is True.
        frozen : int or list
            If integer is given, the inner-most orbitals are frozen from CI
            amplitudes.  Given the orbital indices (0-based) in a list, both
            occupied and virtual orbitals can be frozen in CI calculation.

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> # freeze 2 core orbitals
            >>> myci = ci.CISD(mf).set(frozen = 2).run()
            >>> # freeze 2 core orbitals and 3 high lying unoccupied orbitals
            >>> myci.set(frozen = [0,1,16,17,18]).run()

    Saved results

        converged : bool
            CISD converged or not
        e_corr : float
            CISD correlation correction
        e_tot : float
            Total CCSD energy (HF + correlation)
        ci :
            CI wavefunction coefficients
    '''

    conv_tol = getattr(__config__, 'ci_cisd_CISD_conv_tol', 1e-9)
    max_cycle = getattr(__config__, 'ci_cisd_CISD_max_cycle', 50)
    max_space = getattr(__config__, 'ci_cisd_CISD_max_space', 12)
    lindep = getattr(__config__, 'ci_cisd_CISD_lindep', 1e-14)
    level_shift = getattr(__config__, 'ci_cisd_CISD_level_shift', 0)  # in preconditioner
    direct = getattr(__config__, 'ci_cisd_CISD_direct', False)
    async_io = getattr(__config__, 'ci_cisd_CISD_async_io', True)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if 'dft' in str(mf.__module__):
            raise RuntimeError('CISD Warning: The first argument mf is a DFT object. '
                               'CISD calculation should be initialized with HF object.\n'
                               'DFT object can be converted to HF object with '
                               'the code below:\n'
                               '    mf_hf = scf.RHF(mol)\n'
                               '    mf_hf.__dict__.update(mf_dft.__dict__)\n')

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ   is None: mo_occ   = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.nroots = 1
        self.frozen = frozen
        self.chkfile = mf.chkfile

##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.emp2 = None
        self.ci = None
        self._nocc = None
        self._nmo = None

        keys = set(('conv_tol', 'max_cycle', 'max_space', 'lindep',
                    'level_shift', 'direct'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        log.info('CISD nocc = %s, nmo = %s', self.nocc, self.nmo)
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

    @property
    def nstates(self):
        return self.nroots
    @nstates.setter
    def nstates(self, x):
        self.nroots = x

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def vector_size(self):
        '''The size of the vector which was returned from
        :func:`amplitudes_to_cisdvec`
        '''
        nocc = self.nocc
        nvir = self.nmo - nocc
        return 1 + nocc*nvir + (nocc*nvir)**2

    get_nocc = ccsd.get_nocc
    get_nmo = ccsd.get_nmo
    get_frozen_mask = ccsd.get_frozen_mask

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
        self._finalize()
        return self.e_corr, self.ci

    def _finalize(self):
        citype = self.__class__.__name__
        if numpy.all(self.converged):
            logger.info(self, '%s converged', citype)
        else:
            logger.info(self, '%s not converged', citype)
        if self.nroots > 1:
            for i,e in enumerate(self.e_tot):
                logger.note(self, '%s root %d  E = %.16g', citype, i, e)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        citype, self.e_tot, self.e_corr)
        return self

    def get_init_guess(self, eris=None, nroots=1, diag=None):
        # MP2 initial guess
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        mo_e = eris.fock.diagonal()
        e_ia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
        ci0 = 1
        ci1 = eris.fock[:nocc,nocc:] / e_ia
        eris_ovvo = _cp(eris.ovvo)
        ci2  = 2 * eris_ovvo.transpose(0,3,1,2)
        ci2 -= eris_ovvo.transpose(0,3,2,1)
        ci2 /= lib.direct_sum('ia,jb->ijab', e_ia, e_ia)
        self.emp2 = numpy.einsum('ijab,iabj', ci2, eris_ovvo)
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)

        if abs(self.emp2) < 1e-3 and abs(ci1).sum() < 1e-3:
            # To avoid ci1 being stuck at local minimum
            ci1 = 1e-1 / e_ia

        ci_guess = amplitudes_to_cisdvec(ci0, ci1, ci2)

        if nroots > 1:
            civec_size = ci_guess.size
            dtype = ci_guess.dtype
            nroots = min(ci1.size+1, nroots)  # Consider Koopmans' theorem only

            if diag is None:
                idx = range(1, nroots)
            else:
                idx = diag[:ci1.size+1].argsort()[1:nroots]  # exclude HF determinant

            ci_guess = [ci_guess]
            for i in idx:
                g = numpy.zeros(civec_size, dtype)
                g[i] = 1.0
                ci_guess.append(g)
        return self.emp2, ci_guess

    contract = contract
    make_diagonal = make_diagonal

    def _dot(self, x1, x2, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return dot(x1, x2, nmo, nocc)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return ccsd._make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            logger.warn(self, 'CISD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CISD calculations')
            return ccsd._make_df_eris_outcore(self, mo_coeff)

        else:
            return ccsd._make_eris_outcore(self, mo_coeff)

    def _add_vvvv(self, c2, eris, out=None, t2sym=None):
        return ccsd._add_vvvv(self, None, c2, eris, out, False, t2sym)

    def to_fcivec(self, cisdvec, norb=None, nelec=None, frozen=0):
        if norb is None: norb = self.nmo
        if nelec is None: nelec = self.nocc*2
        return to_fcivec(cisdvec, norb, nelec, frozen)

    def from_fcivec(self, fcivec, norb=None, nelec=None):
        if norb is None: norb = self.nmo
        if nelec is None: nelec = self.nocc*2
        return from_fcivec(fcivec, norb, nelec)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    trans_rdm1 = trans_rdm1

    as_scanner = as_scanner

    def dump_chk(self, ci=None, frozen=None, mo_coeff=None, mo_occ=None):
        if not self.chkfile:
            return self

        if ci is None: ci = self.ci
        if frozen is None: frozen = self.frozen
        ci_chk = {'e_corr': self.e_corr,
                  'ci': ci,
                  'frozen': frozen}

        if mo_coeff is not None: ci_chk['mo_coeff'] = mo_coeff
        if mo_occ is not None: ci_chk['mo_occ'] = mo_occ
        if self._nmo is not None: ci_chk['_nmo'] = self._nmo
        if self._nocc is not None: ci_chk['_nocc'] = self._nocc

        lib.chkfile.save(self.chkfile, 'cisd', ci_chk)

    def amplitudes_to_cisdvec(self, c0, c1, c2):
        return amplitudes_to_cisdvec(c0, c1, c2)

    def cisdvec_to_amplitudes(self, civec, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return cisdvec_to_amplitudes(civec, nmo, nocc)

    def density_fit(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        from pyscf.grad import cisd
        return cisd.Gradients(self)

class RCISD(CISD):
    pass

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import fci
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol).run()
    myci = CISD(mf)
    eris = ccsd._make_eris_outcore(myci, mf.mo_coeff)
    ecisd, civec = myci.kernel(eris=eris)
    print(ecisd - -0.048878084082066106)

    nmo = myci.nmo
    nocc = myci.nocc
    rdm1 = myci.make_rdm1(civec)
    rdm2 = myci.make_rdm2(civec)
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, nmo)
    e2 = (numpy.einsum('ij,ji', h1e, rdm1) +
          numpy.einsum('ijkl,ijkl', h2e, rdm2) * .5)
    print(ecisd + mf.e_tot - mol.energy_nuc() - e2)   # = 0

    print(abs(rdm1 - numpy.einsum('ijkk->ji', rdm2)/(mol.nelectron-1)).sum())

