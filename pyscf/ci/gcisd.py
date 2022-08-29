#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
General spin-orbital CISD
'''

import warnings

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import gccsd
from pyscf.cc import gccsd_rdm
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf.ci import cisd
from pyscf.ci import ucisd


def make_diagonal(myci, eris):
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    mo_energy = eris.fock.diagonal()
    jkdiag = numpy.zeros((nmo,nmo), dtype=mo_energy.dtype)
    jkdiag[:nocc,:nocc] = numpy.einsum('ijij->ij', eris.oooo)
    jkdiag[nocc:,nocc:] = numpy.einsum('ijij->ij', eris.vvvv)
    jkdiag[:nocc,nocc:] = numpy.einsum('ijij->ij', eris.ovov)
    jksum = jkdiag[:nocc,:nocc].sum()
    ehf = mo_energy[:nocc].sum() - jksum * .5
    e1diag = numpy.empty((nocc,nvir), dtype=mo_energy.dtype)
    e2diag = numpy.empty((nocc,nocc,nvir,nvir), dtype=mo_energy.dtype)
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

    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)

    fock = eris.fock
    foo = fock[:nocc,:nocc]
    fov = fock[:nocc,nocc:]
    fvo = fock[nocc:,:nocc]
    fvv = fock[nocc:,nocc:]

    t1  = lib.einsum('ie,ae->ia', c1, fvv)
    t1 -= lib.einsum('ma,mi->ia', c1, foo)
    t1 += lib.einsum('imae,me->ia', c2, fov)
    t1 += lib.einsum('nf,nafi->ia', c1, eris.ovvo)
    t1 -= 0.5*lib.einsum('imef,maef->ia', c2, eris.ovvv)
    t1 -= 0.5*lib.einsum('mnae,mnie->ia', c2, eris.ooov)

    tmp = lib.einsum('ijae,be->ijab', c2, fvv)
    t2  = tmp - tmp.transpose(0,1,3,2)
    tmp = lib.einsum('imab,mj->ijab', c2, foo)
    t2 -= tmp - tmp.transpose(1,0,2,3)
    t2 += 0.5*lib.einsum('mnab,mnij->ijab', c2, eris.oooo)
    t2 += 0.5*lib.einsum('ijef,abef->ijab', c2, eris.vvvv)
    tmp = lib.einsum('imae,mbej->ijab', c2, eris.ovvo)
    tmp+= numpy.einsum('ia,bj->ijab', c1, fvo)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2 += tmp - tmp.transpose(1,0,2,3)
    tmp = lib.einsum('ie,jeba->ijab', c1, numpy.asarray(eris.ovvv).conj())
    t2 += tmp - tmp.transpose(1,0,2,3)
    tmp = lib.einsum('ma,ijmb->ijab', c1, numpy.asarray(eris.ooov).conj())
    t2 -= tmp - tmp.transpose(0,1,3,2)

    eris_oovv = numpy.asarray(eris.oovv)
    t1 += fov.conj() * c0
    t2 += eris_oovv.conj() * c0
    t0  = numpy.einsum('ia,ia', fov, c1)
    t0 += numpy.einsum('ijab,ijab', eris_oovv, c2) * .25

    return amplitudes_to_cisdvec(t0, t1, t2)

def amplitudes_to_cisdvec(c0, c1, c2):
    nocc, nvir = c1.shape
    ooidx = numpy.tril_indices(nocc, -1)
    vvidx = numpy.tril_indices(nvir, -1)
    c2tril = lib.take_2d(c2.reshape(nocc**2,nvir**2),
                         ooidx[0]*nocc+ooidx[1], vvidx[0]*nvir+vvidx[1])
    return numpy.hstack((c0, c1.ravel(), c2tril.ravel()))

def cisdvec_to_amplitudes(civec, nmo, nocc, copy=True):
    nvir = nmo - nocc
    c0 = civec[0]
    cp = lambda x: (x.copy() if copy else x)
    c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir))
    c2 = ccsd._unpack_4fold(civec[nocc*nvir+1:], nocc, nvir)
    return c0, c1, c2

def from_ucisdvec(civec, nocc, orbspin):
    '''Convert the (spin-separated) CISD coefficient vector to GCISD
    coefficient vector'''
    nmoa = numpy.count_nonzero(orbspin == 0)
    nmob = numpy.count_nonzero(orbspin == 1)
    if isinstance(nocc, int):
        nocca = numpy.count_nonzero(orbspin[:nocc] == 0)
        noccb = numpy.count_nonzero(orbspin[:nocc] == 1)
    else:
        nocca, noccb = nocc
    nvira = nmoa - nocca

    if civec.size == nocca*nvira + (nocca*nvira)**2 + 1:  # RCISD
        c0, c1, c2 = cisd.cisdvec_to_amplitudes(civec, nmoa, nocca, copy=False)
    else:  # UCISD
        c0, c1, c2 = ucisd.cisdvec_to_amplitudes(civec, (nmoa,nmob), (nocca,noccb), copy=False)
    c1 = spatial2spin(c1, orbspin)
    c2 = spatial2spin(c2, orbspin)
    return amplitudes_to_cisdvec(c0, c1, c2)
from_rcisdvec = from_ucisdvec

def to_ucisdvec(civec, nmo, nocc, orbspin):
    '''Convert the GCISD coefficient vector to UCISD coefficient vector'''
    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    c1 = spin2spatial(c1, orbspin)
    c2 = spin2spatial(c2, orbspin)
    ucisdvec = ucisd.amplitudes_to_cisdvec(c0, c1, c2)
    unorm = numpy.linalg.norm(ucisdvec)
    if unorm < 1e-2:
        raise RuntimeError('GCISD vector corresponds to spin-flip excitation. '
                           'It cannot be converted to UCISD wfn.'
                           'norm(UCISD) = %s' % unorm)
    elif unorm < 0.99:
        warnings.warn('GCISD vector has spin-flip excitation. '
                      'They are ignored when converting to UCISD wfn. '
                      'norm(UCISD) = %s' % unorm)
    return ucisdvec

def to_fcivec(cisdvec, nelec, orbspin, frozen=None):
    assert (numpy.count_nonzero(orbspin == 0) ==
           numpy.count_nonzero(orbspin == 1))
    norb = len(orbspin)
    frozen_mask = numpy.zeros(norb, dtype=bool)
    if frozen is None:
        pass
    elif isinstance(frozen, (int, numpy.integer)):
        frozen_mask[:frozen] = True
    else:
        frozen_mask[frozen] = True
    frozen = (numpy.where(frozen_mask[orbspin == 0])[0],
              numpy.where(frozen_mask[orbspin == 1])[0])
    nelec = (numpy.count_nonzero(orbspin[:nelec] == 0),
             numpy.count_nonzero(orbspin[:nelec] == 1))
    orbspin = orbspin[~frozen_mask]
    nmo = len(orbspin)
    nocc = numpy.count_nonzero(~frozen_mask[:sum(nelec)])
    ucisdvec = to_ucisdvec(cisdvec, nmo, nocc, orbspin)
    return ucisd.to_fcivec(ucisdvec, norb//2, nelec, frozen)

def from_fcivec(ci0, nelec, orbspin, frozen=None):
    if not (frozen is None or frozen == 0):
        raise NotImplementedError

    assert (numpy.count_nonzero(orbspin == 0) ==
           numpy.count_nonzero(orbspin == 1))
    norb = len(orbspin)
    frozen_mask = numpy.zeros(norb, dtype=bool)
    if frozen is None:
        pass
    elif isinstance(frozen, (int, numpy.integer)):
        frozen_mask[:frozen] = True
    else:
        frozen_mask[frozen] = True
    #frozen = (numpy.where(frozen_mask[orbspin == 0])[0],
    #          numpy.where(frozen_mask[orbspin == 1])[0])
    nelec = (numpy.count_nonzero(orbspin[:nelec] == 0),
             numpy.count_nonzero(orbspin[:nelec] == 1))
    ucisdvec = ucisd.from_fcivec(ci0, norb//2, nelec, frozen)
    nocc = numpy.count_nonzero(~frozen_mask[:sum(nelec)])
    return from_ucisdvec(ucisdvec, nocc, orbspin[~frozen_mask])


def make_rdm1(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    return gccsd_rdm._make_rdm1(myci, d1, with_frozen=True, ao_repr=ao_repr)

def make_rdm2(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    Two-particle density matrix in the molecular spin-orbital representation

    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
    correspond to another particle.  The contraction between ERIs (in
    Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    d2 = _gamma2_intermediates(myci, civec, nmo, nocc)
    return gccsd_rdm._make_rdm2(myci, d1, d2, with_dm1=True, with_frozen=True,
                                ao_repr=ao_repr)

def _gamma1_intermediates(myci, civec, nmo, nocc):
    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    dvo = c0.conj() * c1.T
    dvo += numpy.einsum('jb,ijab->ai', c1.conj(), c2)
    dov = dvo.T.conj()
    doo  =-numpy.einsum('ia,ka->ik', c1.conj(), c1)
    doo -= numpy.einsum('jiab,kiab->jk', c2.conj(), c2) * .5
    dvv  = numpy.einsum('ia,ic->ac', c1, c1.conj())
    dvv += numpy.einsum('ijab,ijac->bc', c2, c2.conj()) * .5
    return doo, dov, dvo, dvv

def _gamma2_intermediates(myci, civec, nmo, nocc):
    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    goovv = c0 * c2.conj() * .5
    govvv = numpy.einsum('ia,ikcd->kadc', c1, c2.conj()) * .5
    gooov = numpy.einsum('ia,klac->klic', c1, c2.conj()) *-.5
    goooo = numpy.einsum('ijab,klab->ijkl', c2.conj(), c2) * .25
    gvvvv = numpy.einsum('ijab,ijcd->abcd', c2, c2.conj()) * .25
    govvo = numpy.einsum('ijab,ikac->jcbk', c2.conj(), c2)
    govvo+= numpy.einsum('ia,jb->ibaj', c1.conj(), c1)

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    dovvo = govvo.transpose(0,2,1,3)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    doovv = None
    dvvov = None
    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov

def trans_rdm1(myci, cibra, ciket, nmo=None, nocc=None):
    r'''
    One-particle transition density matrix in the molecular spin-orbital
    representation.

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    c0bra, c1bra, c2bra = myci.cisdvec_to_amplitudes(cibra, nmo, nocc, copy=False)
    c0ket, c1ket, c2ket = myci.cisdvec_to_amplitudes(ciket, nmo, nocc, copy=False)

    dvo = c0bra.conj() * c1ket.T
    dvo += numpy.einsum('jb,ijab->ai', c1bra.conj(), c2ket)

    dov = c0ket * c1bra.conj()
    dov += numpy.einsum('jb,ijab->ia', c1ket, c2bra.conj())

    doo  =-numpy.einsum('ia,ka->ik', c1bra.conj(), c1ket)
    doo -= numpy.einsum('jiab,kiab->jk', c2bra.conj(), c2ket) * .5
    dvv  = numpy.einsum('ia,ic->ac', c1ket, c1bra.conj())
    dvv += numpy.einsum('ijab,ijac->bc', c2ket, c2bra.conj()) * .5

    dm1 = numpy.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo
    dm1[:nocc,nocc:] = dov
    dm1[nocc:,:nocc] = dvo
    dm1[nocc:,nocc:] = dvv
    norm = numpy.dot(cibra, ciket)
    dm1[numpy.diag_indices(nocc)] += norm

    if myci.frozen is not None:
        nmo = myci.mo_occ.size
        nocc = numpy.count_nonzero(myci.mo_occ > 0)
        rdm1 = numpy.zeros((nmo,nmo), dtype=dm1.dtype)
        rdm1[numpy.diag_indices(nocc)] = norm
        moidx = numpy.where(myci.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1
    return dm1


class GCISD(cisd.CISD):

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        noo = nocc * (nocc-1) // 2
        nvv = nvir * (nvir-1) // 2
        return 1 + nocc*nvir + noo*nvv

    def get_init_guess(self, eris=None, nroots=1, diag=None):
        # MP2 initial guess
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        time0 = logger.process_clock(), logger.perf_counter()
        mo_e = eris.mo_energy
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        ci0 = 1
        ci1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = numpy.array(eris.oovv)
        ci2 = eris_oovv / eijab
        self.emp2 = 0.25*numpy.einsum('ijab,ijab', ci2.conj(), eris_oovv).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)

        if abs(self.emp2) < 1e-3 and abs(ci1).sum() < 1e-3:
            ci1 = 1. / eia

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

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        mem_incore = nmo**4*2 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return gccsd._make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return gccsd._make_eris_outcore(self, mo_coeff)

    contract = contract
    make_diagonal = make_diagonal
    _dot = None

    def to_fcivec(self, cisdvec, nelec, orbspin, frozen=None):
        return to_fcivec(cisdvec, nelec, orbspin, frozen)

    def from_fcivec(self, fcivec, nelec, orbspin, frozen=None):
        return from_fcivec(fcivec, nelec, orbspin, frozen)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    trans_rdm1 = trans_rdm1

    def amplitudes_to_cisdvec(self, c0, c1, c2):
        return amplitudes_to_cisdvec(c0, c1, c2)

    def cisdvec_to_amplitudes(self, civec, nmo=None, nocc=None, copy=True):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return cisdvec_to_amplitudes(civec, nmo, nocc, copy=copy)

    @lib.with_doc(from_ucisdvec.__doc__)
    def from_ucisdvec(self, civec, nocc=None, orbspin=None):
        if nocc is None: nocc = self.nocc
        if orbspin is None:
            orbspin = getattr(self.mo_coeff, 'orbspin', None)
            if orbspin is not None:
                orbspin = orbspin[self.get_frozen_mask()]
        assert (orbspin is not None)
        return from_ucisdvec(civec, nocc, orbspin=orbspin)
    from_rcisdvec = from_ucisdvec

    @lib.with_doc(to_ucisdvec.__doc__)
    def to_ucisdvec(self, civec, orbspin=None):
        if orbspin is None:
            orbspin = getattr(self.mo_coeff, 'orbspin', None)
            if orbspin is not None:
                orbspin = orbspin[self.get_frozen_mask()]
        return to_ucisdvec(civec, self.nmo, self.nocc, orbspin)

    def spatial2spin(self, tx, orbspin=None):
        if orbspin is None:
            orbspin = getattr(self.mo_coeff, 'orbspin', None)
            if orbspin is not None:
                orbspin = orbspin[self.get_frozen_mask()]
        return spatial2spin(tx, orbspin)

    def spin2spatial(self, tx, orbspin=None):
        if orbspin is None:
            orbspin = getattr(self.mo_coeff, 'orbspin', None)
            if orbspin is not None:
                orbspin = orbspin[self.get_frozen_mask()]
        return spin2spatial(tx, orbspin)

CISD = GCISD

from pyscf import scf
scf.ghf.GHF.CISD = lib.class_as_method(CISD)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

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
    gmf = scf.addons.convert_to_ghf(mf)
    myci = GCISD(gmf)
    eris = myci.ao2mo()
    ecisd, civec = myci.kernel(eris=eris)
    print(ecisd - -0.048878084082066106)

    nmo = eris.mo_coeff.shape[1]
    rdm1 = myci.make_rdm1(civec, nmo, mol.nelectron)
    rdm2 = myci.make_rdm2(civec, nmo, mol.nelectron)

    mo = eris.mo_coeff[:7] + eris.mo_coeff[7:]
    eri = ao2mo.kernel(mf._eri, mo, compact=False).reshape([nmo]*4)
    eri[eris.orbspin[:,None]!=eris.orbspin,:,:] = 0
    eri[:,:,eris.orbspin[:,None]!=eris.orbspin] = 0
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    h1e = numpy.zeros((nmo,nmo))
    idxa = eris.orbspin == 0
    idxb = eris.orbspin == 1
    h1e[idxa[:,None] & idxa] = h1a.ravel()
    h1e[idxb[:,None] & idxb] = h1b.ravel()
    e2 = (numpy.einsum('ij,ji', h1e, rdm1) +
          numpy.einsum('ijkl,ijkl', eri, rdm2) * .5)
    e2 += mol.energy_nuc()
    print(myci.e_tot - e2)   # = 0

    print(abs(rdm1 - numpy.einsum('ijkk->ji', rdm2)/(mol.nelectron-1)).sum())

