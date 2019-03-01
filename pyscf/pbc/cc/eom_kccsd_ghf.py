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
# Authors: James D. McClain
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#

import itertools
import time
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd
from pyscf.cc import eom_rccsd
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from pyscf.pbc.cc import kintermediates as imd

einsum = lib.einsum

def kernel(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''Calculate excitation energy via eigenvalue solver

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested per k-point
        koopmans : bool
            Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
        left : bool
            If True, calculates left eigenvectors rather than right eigenvectors.
        eris : `object(uccsd._ChemistsERIs)`
            Holds uccsd electron repulsion integrals in chemist notation.
        imds : `object(_IMDS)`
            Holds eom intermediates in chemist notation.
        partition : bool or str
            Use a matrix-partitioning for the doubles-doubles block.
            Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
            or 'full' (full diagonal elements).
        kptlist : list
            List of k-point indices for which eigenvalues are requested.
        dtype : type
            Type for eigenvectors.
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    size = eom.vector_size()
    nroots = min(nroots,size)
    nkpts = eom.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    if dtype is None:
        dtype = np.result_type(*imds.t1)

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    convs = np.zeros((len(kptlist),nroots), dtype)

    for k, kshift in enumerate(kptlist):
        matvec, diag = eom.gen_matvec(kshift, imds, left=left, **kwargs)

        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            user_guess = False
            guess = eom.get_init_guess(nroots, koopmans, diag)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = lib.davidson_nosym1
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            conv_k, evals_k, evecs_k = eig(matvec, guess, precond, pick=pickeig,
                                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                                           max_space=eom.max_space, nroots=nroots, verbose=log)
        else:
            conv_k, evals_k, evecs_k = eig(matvec, guess, precond,
                                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                                           max_space=eom.max_space, nroots=nroots, verbose=log)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        convs[k] = conv_k

        if nroots == 1:
            evals_k, evecs_k = [evals_k], [evecs_k]
        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn)
            qp_weight = np.linalg.norm(r1)**2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    return convs, evals, evecs

def enforce_2p_spin_doublet(r2, kconserv, kshift, orbspin, excitation):
    '''Enforces condition that net spin can only change by +/- 1/2'''
    assert(excitation in ['ip', 'ea'])
    if excitation == 'ip':
        nkpts, nocc, nvir = np.array(r2.shape)[[1, 3, 4]]
    elif excitation == 'ea':
        nkpts, nocc, nvir = np.array(r2.shape)[[1, 2, 3]]
    else:
        raise NotImplementedError

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    if excitation == 'ip':
        for ki, kj in itertools.product(range(nkpts), repeat=2):
            if ki > kj:  # Avoid double-counting of anti-symmetrization
                continue
            ka = kconserv[ki, kshift, kj]
            idxoaa = idxoa[ki][:,None] * nocc + idxoa[kj]
            idxoab = idxoa[ki][:,None] * nocc + idxob[kj]
            idxoba = idxob[ki][:,None] * nocc + idxoa[kj]
            idxobb = idxob[ki][:,None] * nocc + idxob[kj]

            r2_tmp = 0.5 * (r2[ki, kj] - r2[kj, ki].transpose(1, 0, 2))
            r2_tmp = r2_tmp.reshape(nocc**2, nvir)

            # Zero out states with +/- 3 unpaired spins
            r2_tmp[idxobb.ravel()[:, None], idxva[ka]] = 0.0
            r2_tmp[idxoaa.ravel()[:, None], idxvb[ka]] = 0.0

            r2[ki, kj] = r2_tmp.reshape(nocc, nocc, nvir)
            r2[kj, ki] = -r2[ki, kj].transpose(1, 0, 2)  # Enforce antisymmetry
    else:
        for kj, ka in itertools.product(range(nkpts), repeat=2):
            kb = kconserv[kshift, ka, kj]
            if ka > kb:  # Avoid double-counting of anti-symmetrization
                continue

            idxvaa = idxva[ka][:,None] * nvir + idxva[kb]
            idxvab = idxva[ka][:,None] * nvir + idxvb[kb]
            idxvba = idxvb[ka][:,None] * nvir + idxva[kb]
            idxvbb = idxvb[ka][:,None] * nvir + idxvb[kb]

            r2_tmp = 0.5 * (r2[kj, ka] - r2[kj, kb].transpose(0, 2, 1))
            r2_tmp = r2_tmp.reshape(nocc, nvir**2)

            # Zero out states with +/- 3 unpaired spins
            r2_tmp[idxoa[kshift], idxvbb.ravel()[:, None]] = 0.0
            r2_tmp[idxob[kshift], idxvaa.ravel()[:, None]] = 0.0

            r2[kj, ka] = r2_tmp.reshape(nocc, nvir, nvir)
            r2[kj, kb] = -r2[kj, ka].transpose(0, 2, 1)  # Enforce antisymmetry
    return r2

########################################
# EOM-IP-CCSD
########################################

def enforce_2p_spin_ip_doublet(r2, kconserv, kshift, orbspin):
    return enforce_2p_spin_doublet(r2, kconserv, kshift, orbspin, 'ip')

def spin2spatial_ip_doublet(r1, r2, kconserv, kshift, orbspin):
    '''Convert R1/R2 of spin orbital representation to R1/R2 of
    spatial orbital representation '''
    nkpts, nocc, nvir = np.array(r2.shape)[[1, 3, 4]]

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]
    nocc_a = len(idxoa[0])  # Assume nocc/nvir same for each k-point
    nocc_b = len(idxob[0])
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

    r1a = r1[idxoa[kshift]]
    r1b = r1[idxob[kshift]]

    r2aaa = np.zeros((nkpts,nkpts,nocc_a,nocc_a,nvir_a), dtype=r2.dtype)
    r2baa = np.zeros((nkpts,nkpts,nocc_b,nocc_a,nvir_a), dtype=r2.dtype)
    r2abb = np.zeros((nkpts,nkpts,nocc_a,nocc_b,nvir_b), dtype=r2.dtype)
    r2bbb = np.zeros((nkpts,nkpts,nocc_b,nocc_b,nvir_b), dtype=r2.dtype)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        idxoaa = idxoa[ki][:,None] * nocc + idxoa[kj]
        idxoab = idxoa[ki][:,None] * nocc + idxob[kj]
        idxoba = idxob[ki][:,None] * nocc + idxoa[kj]
        idxobb = idxob[ki][:,None] * nocc + idxob[kj]

        r2_tmp = r2[ki, kj].reshape(nocc**2, nvir)
        r2aaa_tmp = lib.take_2d(r2_tmp, idxoaa.ravel(), idxva[ka])
        r2baa_tmp = lib.take_2d(r2_tmp, idxoba.ravel(), idxva[ka])
        r2abb_tmp = lib.take_2d(r2_tmp, idxoab.ravel(), idxvb[ka])
        r2bbb_tmp = lib.take_2d(r2_tmp, idxobb.ravel(), idxvb[ka])

        r2aaa[ki, kj] = r2aaa_tmp.reshape(nocc_a, nocc_a, nvir_a)
        r2baa[ki, kj] = r2baa_tmp.reshape(nocc_b, nocc_a, nvir_a)
        r2abb[ki, kj] = r2abb_tmp.reshape(nocc_a, nocc_b, nvir_b)
        r2bbb[ki, kj] = r2bbb_tmp.reshape(nocc_b, nocc_b, nvir_b)
    return [r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb]

def spatial2spin_ip_doublet(r1, r2, kconserv, kshift, orbspin=None):
    '''Convert R1/R2 of spatial orbital representation to R1/R2 of
    spin orbital representation '''
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    nkpts, nocc_a, nvir_a = np.array(r2aaa.shape)[[1, 3, 4]]
    nkpts, nocc_b, nvir_b = np.array(r2bbb.shape)[[1, 3, 4]]

    if orbspin is None:
        orbspin = np.zeros((nkpts, nocc_a+nocc_b+nvir_a+nvir_b), dtype=int)
        orbspin[:,1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    r1 = np.zeros(nocc, dtype = r1a.dtype)
    r1[idxoa[kshift]] = r1a
    r1[idxob[kshift]] = r1b

    r2 = np.zeros((nkpts, nkpts, nocc**2, nvir), dtype = r2aaa.dtype)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        idxoaa = idxoa[ki][:,None] * nocc + idxoa[kj]
        idxoab = idxoa[ki][:,None] * nocc + idxob[kj]
        idxoba = idxob[ki][:,None] * nocc + idxoa[kj]
        idxobb = idxob[ki][:,None] * nocc + idxob[kj]

        r2aaa_tmp = r2aaa[ki,kj].reshape(nocc_a * nocc_a, nvir_a)
        r2baa_tmp = r2baa[ki,kj].reshape(nocc_b * nocc_a, nvir_a)
        r2abb_tmp = r2abb[ki,kj].reshape(nocc_a * nocc_b, nvir_b)
        r2bbb_tmp = r2bbb[ki,kj].reshape(nocc_b * nocc_b, nvir_b)

        lib.takebak_2d(r2[ki, kj], r2aaa_tmp, idxoaa.ravel(), idxva[ka])
        lib.takebak_2d(r2[ki, kj], r2baa_tmp, idxoba.ravel(), idxva[ka])
        lib.takebak_2d(r2[ki, kj], r2abb_tmp, idxoab.ravel(), idxvb[ka])
        lib.takebak_2d(r2[ki, kj], r2bbb_tmp, idxobb.ravel(), idxvb[ka])

        r2aba_tmp = - r2baa[kj,ki].reshape(nocc_a * nocc_b, nvir_a)
        r2bab_tmp = - r2abb[kj,ki].reshape(nocc_a * nocc_b, nvir_b)

        lib.takebak_2d(r2[ki, kj], r2aba_tmp, idxoab.T.ravel(), idxva[ka])
        lib.takebak_2d(r2[ki, kj], r2bab_tmp, idxoba.T.ravel(), idxvb[ka])

    r2 = r2.reshape(nkpts, nkpts, nocc, nocc, nvir)
    return r1, r2

def vector_to_amplitudes_ip(vector, nkpts, nmo, nocc):
    nvir = nmo - nocc

    # TODO: some redundancy; can use symmetry of operator to reduce size
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    return [r1,r2]

def amplitudes_to_vector_ip(r1,r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{a }, i.e. 'ia' indices are coupled.
    This differs from the restricted case that uses s_{ij}^{ b}.'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ip(vector, nkpts, nmo, nocc)

    Hr1 = -np.einsum('mi,m->i', imds.Foo[kshift], r1)
    for km in range(nkpts):
        Hr1 += np.einsum('me,mie->i', imds.Fov[km], r2[km, kshift])
        for kn in range(nkpts):
            Hr1 += - 0.5 * np.einsum('nmie,mne->i', imds.Wooov[kn, km, kshift],
                                     r2[km, kn])

    Hr2 = np.zeros_like(r2)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        Hr2[ki, kj] += lib.einsum('ae,ije->ija', imds.Fvv[ka], r2[ki, kj])

        Hr2[ki, kj] -= lib.einsum('mi,mja->ija', imds.Foo[ki], r2[ki, kj])
        Hr2[ki, kj] += lib.einsum('mj,mia->ija', imds.Foo[kj], r2[kj, ki])

        Hr2[ki, kj] -= np.einsum('maji,m->ija', imds.Wovoo[kshift, ka, kj], r1)
        for km in range(nkpts):
            kn = kconserv[ki, km, kj]
            Hr2[ki, kj] += 0.5 * lib.einsum('mnij,mna->ija',
                                            imds.Woooo[km, kn, ki], r2[km, kn])

    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        for km in range(nkpts):
            ke = kconserv[km, kshift, kj]
            Hr2[ki, kj] += lib.einsum('maei,mje->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, kj])

            ke = kconserv[km, kshift, ki]
            Hr2[ki, kj] -= lib.einsum('maej,mie->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, ki])

    tmp = lib.einsum('xymnef,xymnf->e', imds.Woovv[:, :, kshift], r2[:, :])  # contract_{km, kn}
    Hr2[:, :] += 0.5 * lib.einsum('e,yxjiea->xyija', tmp, imds.t2[:, :, kshift])  # sum_{ki, kj}

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = -np.diag(imds.Foo[kshift])

    Hr2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir), dtype=t1.dtype)
    if eom.partition == 'mp':
        foo = eom.eris.fock[:,:nocc,:nocc]
        fvv = eom.eris.fock[:,nocc:,nocc:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kconserv[ki,kshift,kj]
                Hr2[ki,kj] -= foo[ki].diagonal()[:,None,None]
                Hr2[ki,kj] -= foo[kj].diagonal()[None,:,None]
                Hr2[ki,kj] += fvv[ka].diagonal()[None,None,:]
    else:
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kconserv[ki,kshift,kj]
                Hr2[ki,kj] -= imds.Foo[ki].diagonal()[:,None,None]
                Hr2[ki,kj] -= imds.Foo[kj].diagonal()[None,:,None]
                Hr2[ki,kj] += imds.Fvv[ka].diagonal()[None,None,:]

                if ki == kconserv[ki,kj,kj]:
                    Hr2[ki,kj] += np.einsum('ijij->ij', imds.Woooo[ki, kj, ki])[:,:,None]

                Hr2[ki, kj] += lib.einsum('iaai->ia', imds.Wovvo[ki, ka, ka])[:,None,:]
                Hr2[ki, kj] += lib.einsum('jaaj->ja', imds.Wovvo[kj, ka, ka])[None,:,:]

                Hr2[ki, kj] += lib.einsum('ijea,jiea->ija',imds.Woovv[ki,kj,kshift], imds.t2[kj,ki,kshift])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''See `kernel()` for a description of arguments.'''
    if partition:
        eom.partition = partition.lower()
        assert eom.partition in ['mp','full']
        if eom.partition in ['mp', 'full']:
            raise NotImplementedError
    eom.converged, eom.e, eom.v \
            = kernel(eom, nroots, koopmans, guess, left, eris=eris, imds=imds,
                     partition=partition, kptlist=kptlist, dtype=dtype)
    return eom.e, eom.v

class EOMIP(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        eom_rccsd.EOM.__init__(self, cc)

    kernel = ipccsd
    ipccsd = ipccsd
    get_diag = ipccsd_diag
    matvec = ipccsd_matvec

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc-n-1] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    @property
    def nkpts(self):
        return len(self.kpts)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nkpts=None, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes_ip(vector, nkpts, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    def make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ip()
        return imds

########################################
# EOM-EA-CCSD
########################################

def enforce_2p_spin_ea_doublet(r2, kconserv, kshift, orbspin):
    return enforce_2p_spin_doublet(r2, kconserv, kshift, orbspin, 'ea')

def spin2spatial_ea_doublet(r1, r2, kconserv, kshift, orbspin):
    '''Convert R1/R2 of spin orbital representation to R1/R2 of
    spatial orbital representation'''
    nkpts, nocc, nvir = np.array(r2.shape)[[1, 2, 3]]

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]
    nocc_a = len(idxoa[0])
    nocc_b = len(idxob[0])
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

    r1a = r1[idxva[kshift]]
    r1b = r1[idxvb[kshift]]

    r2aaa = np.zeros((nkpts,nkpts,nocc_a,nvir_a,nvir_a), dtype=r2.dtype)
    r2aba = np.zeros((nkpts,nkpts,nocc_a,nvir_b,nvir_a), dtype=r2.dtype)
    r2bab = np.zeros((nkpts,nkpts,nocc_b,nvir_a,nvir_b), dtype=r2.dtype)
    r2bbb = np.zeros((nkpts,nkpts,nocc_b,nvir_b,nvir_b), dtype=r2.dtype)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift, ka, kj]
        idxvaa = idxva[ka][:,None] * nvir + idxva[kb]
        idxvab = idxva[ka][:,None] * nvir + idxvb[kb]
        idxvba = idxvb[ka][:,None] * nvir + idxva[kb]
        idxvbb = idxvb[ka][:,None] * nvir + idxvb[kb]

        r2_tmp = r2[kj, ka].reshape(nocc, nvir**2)
        r2aaa_tmp = lib.take_2d(r2_tmp, idxoa[kj], idxvaa.ravel())
        r2aba_tmp = lib.take_2d(r2_tmp, idxoa[kj], idxvba.ravel())
        r2bab_tmp = lib.take_2d(r2_tmp, idxob[kj], idxvab.ravel())
        r2bbb_tmp = lib.take_2d(r2_tmp, idxob[kj], idxvbb.ravel())

        r2aaa[kj, ka] = r2aaa_tmp.reshape(nocc_a, nvir_a, nvir_a)
        r2aba[kj, ka] = r2aba_tmp.reshape(nocc_a, nvir_b, nvir_a)
        r2bab[kj, ka] = r2bab_tmp.reshape(nocc_b, nvir_a, nvir_b)
        r2bbb[kj, ka] = r2bbb_tmp.reshape(nocc_b, nvir_b, nvir_b)
    return [r1a, r1b], [r2aaa, r2aba, r2bab, r2bbb]

def spatial2spin_ea_doublet(r1, r2, kconserv, kshift, orbspin=None):
    '''Convert R1/R2 of spatial orbital representation to R1/R2 of
    spin orbital representation'''
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2

    nkpts, nocc_a, nvir_a = np.array(r2aaa.shape)[[0, 2, 3]]
    nkpts, nocc_b, nvir_b = np.array(r2bbb.shape)[[0, 2, 3]]

    if orbspin is None:
        orbspin = np.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    r1 = np.zeros((nvir), dtype=r1a.dtype)
    r1[idxva[kshift]] = r1a
    r1[idxvb[kshift]] = r1b

    r2 = np.zeros((nkpts,nkpts,nocc,nvir**2), dtype=r2aaa.dtype)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift, ka, kj]
        idxvaa = idxva[ka][:,None] * nvir + idxva[kb]
        idxvab = idxva[ka][:,None] * nvir + idxvb[kb]
        idxvba = idxvb[ka][:,None] * nvir + idxva[kb]
        idxvbb = idxvb[ka][:,None] * nvir + idxvb[kb]

        r2aaa_tmp = r2aaa[kj,ka].reshape(nocc_a, nvir_a*nvir_a)
        r2aba_tmp = r2aba[kj,ka].reshape(nocc_a, nvir_b*nvir_a)
        r2bab_tmp = r2bab[kj,ka].reshape(nocc_b, nvir_a*nvir_b)
        r2bbb_tmp = r2bbb[kj,ka].reshape(nocc_b, nvir_b*nvir_b)

        lib.takebak_2d(r2[kj,ka], r2aaa_tmp, idxoa[kj], idxvaa.ravel())
        lib.takebak_2d(r2[kj,ka], r2aba_tmp, idxoa[kj], idxvba.ravel())
        lib.takebak_2d(r2[kj,ka], r2bab_tmp, idxob[kj], idxvab.ravel())
        lib.takebak_2d(r2[kj,ka], r2bbb_tmp, idxob[kj], idxvbb.ravel())

        r2aab_tmp = -r2aba[kj,kb].reshape(nocc_a, nvir_b*nvir_a)
        r2bba_tmp = -r2bab[kj,kb].reshape(nocc_b, nvir_a*nvir_b)
        lib.takebak_2d(r2[kj,ka], r2bba_tmp, idxob[kj], idxvba.T.ravel())
        lib.takebak_2d(r2[kj,ka], r2aab_tmp, idxoa[kj], idxvab.T.ravel())

    r2 = r2.reshape(nkpts, nkpts, nocc, nvir, nvir)
    return r1, r2

def vector_to_amplitudes_ea(vector, nkpts, nmo, nocc):
    nvir = nmo - nocc

    # TODO: some redundancy; can use symmetry of operator to reduce size
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
    return [r1,r2]

def amplitudes_to_vector_ea(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def eaccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None):
    '''See `ipccsd()` for a description of arguments.'''
    return ipccsd(eom, nroots, koopmans, guess, left, eris, imds,
                  partition, kptlist, dtype)

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{a }, i.e. 'ia' indices are coupled.
    This differs from the restricted case that uses s_{ij}^{ b}.'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ea(vector, nkpts, nmo, nocc)

    Hr1 = np.einsum('ac,c->a', imds.Fvv[kshift], r1)
    for kl in range(nkpts):
        Hr1 += np.einsum('ld,lad->a', imds.Fov[kl], r2[kl, kshift])
        for kc in range(nkpts):
            Hr1 += 0.5*np.einsum('alcd,lcd->a', imds.Wvovv[kshift,kl,kc], r2[kl,kc])

    Hr2 = np.zeros_like(r2)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        Hr2[kj,ka] += np.einsum('abcj,c->jab', imds.Wvvvo[ka,kb,kshift], r1)
        Hr2[kj,ka] += lib.einsum('ac,jcb->jab', imds.Fvv[ka], r2[kj,ka])
        Hr2[kj,ka] -= lib.einsum('bc,jca->jab', imds.Fvv[kb], r2[kj,kb])
        Hr2[kj,ka] -= lib.einsum('lj,lab->jab', imds.Foo[kj], r2[kj,ka])

        for kd in range(nkpts):
            kl = kconserv[kj, kb, kd]
            Hr2[kj, ka] += lib.einsum('lbdj,lad->jab', imds.Wovvo[kl, kb, kd], r2[kl, ka])

            # P(ab)
            kl = kconserv[kj, ka, kd]
            Hr2[kj, ka] -= lib.einsum('ladj,lbd->jab', imds.Wovvo[kl, ka, kd], r2[kl, kb])

            kc = kconserv[ka, kd, kb]
            Hr2[kj, ka] += 0.5 * lib.einsum('abcd,jcd->jab', imds.Wvvvv[ka, kb, kc], r2[kj, kc])

    tmp = lib.einsum('xyklcd,xylcd->k', imds.Woovv[kshift, :, :], r2[:, :])  # contract_{kl, kc}
    Hr2[:, :] -= 0.5*lib.einsum('k,xykjab->xyjab', tmp, imds.t2[kshift, :, :])  # sum_{kj, ka]

    vector = amplitudes_to_vector_ea(Hr1, Hr2)
    return vector

def eaccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = np.diag(imds.Fvv[kshift])

    Hr2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), dtype=t1.dtype)
    if eom.partition == 'mp': # This case is untested
        foo = eom.eris.fock[:,:nocc,:nocc]
        fvv = eom.eris.fock[:,nocc:,nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                Hr2[kj,ka] -= foo[kj].diagonal()[:,None,None]
                Hr2[kj,ka] -= fvv[ka].diagonal()[None,:,None]
                Hr2[kj,ka] += fvv[kb].diagonal()[None,None,:]
    else:
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]
                Hr2[kj,ka] -= imds.Foo[kj].diagonal()[:,None,None]
                Hr2[kj,ka] += imds.Fvv[ka].diagonal()[None,:,None]
                Hr2[kj,ka] += imds.Fvv[kb].diagonal()[None,None,:]

                Hr2[kj,ka] += np.einsum('jbbj->jb', imds.Wovvo[kj,kb,kb])[:, None, :]
                Hr2[kj,ka] += np.einsum('jaaj->ja', imds.Wovvo[kj,ka,ka])[:, :, None]

                if ka == kconserv[ka,kb,kb]:
                    Hr2[kj,ka] += np.einsum('abab->ab', imds.Wvvvv[ka,kb,ka])[None,:,:]

                Hr2[kj,ka] -= np.einsum('kjab,kjab->jab',imds.Woovv[kshift,kj,ka],imds.t2[kshift,kj,ka])

    vector = amplitudes_to_vector_ea(Hr1, Hr2)
    return vector

class EOMEA(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        eom_rccsd.EOM.__init__(self, cc)
        self.kshift = 0

    kernel = eaccsd
    eaccsd = eaccsd
    get_diag = eaccsd_diag
    matvec = eaccsd_matvec

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(size, dtype=dtype)
                g[n] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(size, dtype=dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    @property
    def nkpts(self):
        return len(self.kpts)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nkpts=None, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes_ea(vector, nkpts, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    def make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ea()
        return imds

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc, eris=None, t1=None, t2=None):
        self._cc = cc
        self.verbose = cc.verbose
        self.kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        self.stdout = cc.stdout
        if t1 is None:
            t1 = cc.t1
        self.t1 = t1
        if t2 is None:
            t2 = cc.t2
        self.t2 = t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        self.Foo = imd.Foo(self._cc, t1, t2, eris, kconserv)
        self.Fvv = imd.Fvv(self._cc, t1, t2, eris, kconserv)
        self.Fov = imd.Fov(self._cc, t1, t2, eris, kconserv)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(self._cc, t1, t2, eris, kconserv)
        self.Woovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-CCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(self._cc, t1, t2, eris, kconserv)
        self.Wooov = imd.Wooov(self._cc, t1, t2, eris, kconserv)
        self.Wovoo = imd.Wovoo(self._cc, t1, t2, eris, kconserv)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        # FIXME DELETE WOOOO
        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(self._cc, t1, t2, eris, kconserv)
        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(self._cc, t1, t2, eris, kconserv)
        self.Wvvvv = imd.Wvvvv(self._cc, t1, t2, eris, kconserv)
        self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris, kconserv)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.Woooo = imd.Woooo(self._cc, t1, t2, eris, kconserv)
            self.Wooov = imd.Wooov(self._cc, t1, t2, eris, kconserv)
            self.Wovoo = imd.Wovoo(self._cc, t1, t2, eris, kconserv)
        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(self._cc, t1, t2, eris, kconserv)
            self.Wvvvv = imd.Wvvvv(self._cc, t1, t2, eris, kconserv)
            self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris, self.Wvvvv, kconserv)

        self.made_ee_imds = True
        logger.timer(self, 'EOM-CCSD EE intermediates', *cput0)
        return self

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = { 'C': [[0, (0.8, 1.0)],
                         [1, (1.0, 1.0)]]}
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
    kmf.conv_tol_grad = 1e-8
    ehf = kmf.kernel()

    mycc = cc.KGCCSD(kmf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    eris = mycc.ao2mo(mycc.mo_coeff)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.155298393321855)

    eom = EOMIP(mycc)
    e, v = eom.ipccsd(nroots=2, kptlist=[0])

    eom = EOMEA(mycc)
    eom.max_cycle = 100
    e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[0])
