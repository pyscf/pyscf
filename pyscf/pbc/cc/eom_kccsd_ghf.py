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
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from pyscf.pbc.cc import kintermediates as imd
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)

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
        imds = eom.make_imds(eris=eris)

    size = eom.vector_size()
    nroots = min(nroots,size)
    nkpts = eom.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    # Make the max number of roots the maximum number of occupied orbitals at any given
    # kpoint in the list
    for k, kshift in enumerate(kptlist):
        frozen_orbs = eom.mask_frozen(np.zeros(size, dtype=int), kshift, const=1)
        if isinstance(frozen_orbs, tuple):
            nfrozen  = (np.sum(frozen_orbs[0]), np.sum(frozen_orbs[1]))
            nroots = min(nroots, size - nfrozen[0])
            nroots = min(nroots, size - nfrozen[1])
        else:
            nfrozen = np.sum(frozen_orbs)
            nroots = min(nroots, size - nfrozen)

    if dtype is None:
        dtype = np.result_type(*imds.t1)

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    convs = np.zeros((len(kptlist),nroots), dtype)

    for k, kshift in enumerate(kptlist):
        print("kshift =", kshift)
        matvec, diag = eom.gen_matvec(kshift, imds, left=left, **kwargs)
        diag = eom.mask_frozen(diag, kshift, const=LARGE_DENOM)

        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            user_guess = False
            guess = eom.get_init_guess(kshift, nroots, koopmans, diag)
        for ig, g in enumerate(guess):
            guess_norm = np.linalg.norm(g)
            guess_norm_tol = LOOSE_ZERO_TOL
            if guess_norm < guess_norm_tol:
                raise ValueError('Guess vector (id=%d) with norm %.4g is below threshold %.4g.\n'
                                 'This could possibly be due to masking/freezing orbitals.\n'
                                 'Check your guess vector to make sure it has sufficiently large norm.'
                                 % (ig, guess_norm, guess_norm_tol))

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
                                           max_space=eom.max_space, nroots=nroots, verbose=eom.verbose)
        else:
            conv_k, evals_k, evecs_k = eig(matvec, guess, precond,
                                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                                           max_space=eom.max_space, nroots=nroots, verbose=eom.verbose)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        convs[k] = conv_k

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn, kshift=kshift)
            if isinstance(r1, np.ndarray):
                qp_weight = np.linalg.norm(r1)**2
            else: # for EOM-UCCSD
                r1 = np.hstack([x.ravel() for x in r1])
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

def get_padding_k_idx(eom, cc):
    return padding_k_idx(cc, kind="split")

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

def vector_to_amplitudes_ip(vector, kshift, nkpts, nmo, nocc, kconserv):
    nvir = nmo - nocc

    r1 = vector[:nocc].copy()
    r2_tril = vector[nocc:].copy().reshape(nkpts*nocc*(nkpts*nocc-1)//2,nvir)
    idx, idy = np.tril_indices(nkpts*nocc, -1)
    r2 = np.zeros((nkpts*nocc,nkpts*nocc,nvir), dtype=vector.dtype)
    r2[idx, idy] = r2_tril
    r2[idy, idx] = -r2_tril
    r2 = r2.reshape(nkpts,nocc,nkpts,nocc,nvir).transpose(0,2,1,3,4)
    return [r1,r2]

def amplitudes_to_vector_ip(r1, r2, kshift, kconserv):
    nkpts, nocc, nvir = np.asarray(r2.shape)[[0,2,4]]
    # From symmetry for aaa and bbb terms, only store lower
    # triangular part (ki,i) < (kj,j)
    idx, idy = np.tril_indices(nkpts*nocc, -1)
    r2 = r2.transpose(0,2,1,3,4).reshape(nkpts*nocc,nkpts*nocc,nvir)
    return np.hstack((r1, r2[idx,idy].ravel()))

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{a }, i.e. 'ia' indices are coupled.
    This differs from the restricted case that uses s_{ij}^{ b}.'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ip(vector, kshift, nkpts, nmo, nocc, kconserv)

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

    vector = amplitudes_to_vector_ip(Hr1, Hr2, kshift, kconserv)
    return vector

def lipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled.

    See also `ipccsd_matvec`'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ip(vector, kshift, nkpts, nmo, nocc, kconserv)
    dtype = np.result_type(r1, r2)

    Hr1 = -lib.einsum('mi,i->m', imds.Foo[kshift], r1)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        Hr1 += -0.5 * lib.einsum('maji,ija->m', imds.Wovoo[kshift,ka,kj], r2[ki,kj])

    Hr2 = np.zeros_like(r2)
    for km, kn in itertools.product(range(nkpts), repeat=2):
        ke = kconserv[km, kshift, kn]
        Hr2[km,kn] += -lib.einsum('nmie,i->mne', imds.Wooov[kn,km,kshift], r1)
        Hr2[km,kshift] += (km==ke)*lib.einsum('me,n->mne', imds.Fov[km], r1)
        Hr2[kshift,kn] -= (kn==ke)*lib.einsum('ne,m->mne', imds.Fov[kn], r1)

    for km, kn in itertools.product(range(nkpts), repeat=2):
        ke = kconserv[km, kshift, kn]
        Hr2[km,kn] += lib.einsum('ae,mna->mne', imds.Fvv[ke], r2[km,kn])
        tmp1 = lib.einsum('mi,ine->mne', imds.Foo[km], r2[km,kn])
        tmp1T = lib.einsum('ni,ime->mne', imds.Foo[kn], r2[kn,km])
        Hr2[km,kn] += (-tmp1 + tmp1T)

        for ki in range(nkpts):
            kj = kconserv[km,ki,kn]
            Hr2[km,kn] += 0.5 * lib.einsum('mnij,ije->mne', imds.Woooo[km,kn,ki], r2[ki,kj])

            ka = kconserv[ke,km,ki]
            tmp2 = lib.einsum('maei,ina->mne', imds.Wovvo[km,ka,ke], r2[ki,kn])
            ka = kconserv[ke,kn,ki]
            tmp2T = lib.einsum('naei,ima->mne', imds.Wovvo[kn,ka,ke], r2[ki,km])
            Hr2[km,kn] += (tmp2 - tmp2T)

    tmp = np.zeros(nvir, dtype=dtype)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki,kshift,kj]
        kf = kshift
        tmp += lib.einsum('ija,ijaf->f',r2[ki,kj],imds.t2[ki,kj,ka])

    for km, kn in itertools.product(range(nkpts), repeat=2):
        ke = kconserv[km, kshift, kn]
        Hr2[km,kn] += 0.5 * lib.einsum('mnfe,f->mne', imds.Woovv[km,kn,kf], tmp)

    vector = amplitudes_to_vector_ip(Hr1, Hr2, kshift, kconserv)
    return vector

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = -np.diag(imds.Foo[kshift])
    print("ipccsd: Hr1 shape =", Hr1.shape)
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

    vector = amplitudes_to_vector_ip(Hr1, Hr2, kshift, kconserv)
    return vector


def ipccsd_star_contract(eom, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift, imds=None):
    """
    Returns:
        e_star (list of float):
            The IP-CCSD* energy.

    Notes:
        The user should check to make sure the right and left eigenvalues
        before running the perturbative correction.

        The 2hp right amplitudes are assumed to be of the form s^{a }_{ij}, i.e.
        the (ia) indices are coupled.

    Reference:
        Saeh, Stanton "...energy surfaces of radicals" JCP 111, 8275 (1999)
    """
    assert (eom.partition == None)
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    nmo = nocc + nvir
    dtype = np.result_type(t1, t2)
    kconserv = eom.kconserv

    fov = fock[:, :nocc, nocc:]
    foo = [fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)]
    fvv = [fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)]
    mo_energy_occ = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    ipccsd_evecs = np.array(ipccsd_evecs)
    lipccsd_evecs = np.array(lipccsd_evecs)
    e_star = []
    ipccsd_evecs, lipccsd_evecs = [np.atleast_2d(x) for x in [ipccsd_evecs, lipccsd_evecs]]
    ipccsd_evals = np.atleast_1d(ipccsd_evals)
    for ip_eval, ip_evec, ip_levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        # Enforcing <L|R> = 1
        l1, l2 = vector_to_amplitudes_ip(ip_levec, kshift, nkpts, nmo, nocc, kconserv)
        r1, r2 = vector_to_amplitudes_ip(ip_evec, kshift, nkpts, nmo, nocc, kconserv)
        ldotr = np.dot(l1, r1) + 0.5 * np.dot(l2.ravel(), r2.ravel())

        logger.info(eom, 'Left-right amplitude overlap : %14.8e + 1j %14.8e',
                    ldotr.real, ldotr.imag)
        if abs(ldotr) < 1e-7:
            logger.warn(eom, 'Small %s left-right amplitude overlap. Results '
                             'may be inaccurate.', ldotr)

        l1 /= ldotr
        l2 /= ldotr

        deltaE = 0.0 + 1j*0.0
        for ka, kb in itertools.product(range(nkpts), repeat=2):
            lijkab = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir,nvir),dtype=dtype)
            rijkab = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir,nvir),dtype=dtype)
            kklist = kpts_helper.get_kconserv3(eom._cc._scf.cell, eom._cc.kpts,
                          [ka,kb,kshift,range(nkpts),range(nkpts)])

            for ki, kj in itertools.product(range(nkpts), repeat=2):
                kk = kklist[ki,kj]
                #TODO: can reduce size of ijkab arrays since `kk` fixed from other k-points

                # lijkab update
                if kk == kshift and kb == kconserv[ki,ka,kj]:
                    lijkab[ki,kj,kk] += lib.einsum('ijab,k->ijkab', eris.oovv[ki,kj,ka], l1)

                km = kconserv[kj,ka,ki]
                tmp = lib.einsum('jima,mkb->ijkab', eris.ooov[kj,ki,km], l2[km,kk])
                km = kconserv[kj,kb,ki]
                tmpT = lib.einsum('jimb,mka->ijkab', eris.ooov[kj,ki,km], l2[km,kk])
                lijkab[ki,kj,kk] += (-tmp + tmpT)

                ke = kconserv[ka,ki,kb]
                lijkab[ki,kj,kk] += lib.einsum('ieab,jke->ijkab', eris.ovvv[ki,ke,ka], l2[kj,kk])

                # rijkab update
                tmp = lib.einsum('mbke,m->bke', eris.ovov[kshift,kb,kk], r1)
                tmp = lib.einsum('bke,ijae->ijkab', tmp, t2[ki,kj,ka])
                tmpT = lib.einsum('make,m->ake', eris.ovov[kshift,ka,kk], r1)
                tmpT = lib.einsum('ake,ijbe->ijkab', tmpT, t2[ki,kj,kb])
                rijkab[ki,kj,kk] -= (tmp - tmpT)

                km = kconserv[kj,kshift,kk]
                tmp = lib.einsum('mnjk,n->mjk', eris.oooo[km,kshift,kj], r1)
                tmp = lib.einsum('mjk,imab->ijkab', tmp, t2[ki,km,ka])
                rijkab[ki,kj,kk] += tmp

                km = kconserv[kj,ka,ki]
                tmp = lib.einsum('jima,mkb->ijkab', eris.ooov[kj,ki,km].conj(), r2[km,kk])
                km = kconserv[kj,kb,ki]
                tmpT = lib.einsum('jimb,mka->ijkab', eris.ooov[kj,ki,km].conj(), r2[km,kk])
                rijkab[ki,kj,kk] -= (tmp - tmpT)

                ke = kconserv[ka,ki,kb]
                rijkab[ki,kj,kk] += lib.einsum('ieab,jke->ijkab', eris.ovvv[ki,ke,ka].conj(), r2[kj,kk])

            # P(ijk)
            lijkab = lijkab + lijkab.transpose(1,2,0,4,5,3,6,7) + lijkab.transpose(2,0,1,5,3,4,6,7)
            rijkab = rijkab + rijkab.transpose(1,2,0,4,5,3,6,7) + rijkab.transpose(2,0,1,5,3,4,6,7)

            # Creating denominator
            eijk = (mo_e_o[:, None, None, :, None, None] + mo_e_o[None, :, None, None, :, None] +
                    mo_e_o[None, None, :, None, None, :])
            eab = mo_e_v[ka][:, None] + mo_e_v[kb][None, :]
            eijkab = (eijk[:, :, :, :, :, :, None, None] -
                      eab[None, None, None, None, None, None, :, :])
            denom = eijkab + ip_eval
            denom = 1. / denom

            deltaE += lib.einsum('xyzijkab,xyzijkab,xyzijkab', lijkab, rijkab, denom)

        deltaE *= 1./12
        deltaE = deltaE.real
        logger.info(eom, "Exc. energy, delta energy = %16.12f, %16.12f",
        ip_eval + deltaE, deltaE)
        e_star.append(ip_eval + deltaE)
    return e_star


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


def perturbed_ccsd_kernel(eom, nroots=1, koopmans=False, right_guess=None,
                          left_guess=None, eris=None, imds=None, partition=None,
                          kptlist=None, dtype=None):
    '''Wrapper for running perturbative excited-states that require both left
    and right amplitudes.'''
    from pyscf.cc.eom_rccsd import _sort_left_right_eigensystem
    if imds is None:
        imds = eom.make_imds(eris=eris)

    e_star = []
    for k, kshift in enumerate(kptlist):
        # Right eigenvectors
        r_converged, r_e, r_v = \
                   kernel(eom, nroots, koopmans=koopmans, guess=right_guess, left=False,
                          eris=eris, imds=imds, partition=partition, kptlist=[kshift,], dtype=dtype)
        # Left eigenvectors
        l_converged, l_e, l_v = \
                   kernel(eom, nroots, koopmans=koopmans, guess=right_guess, left=True,
                          eris=eris, imds=imds, partition=partition, kptlist=[kshift,], dtype=dtype)

        ek, r_vk, l_vk = _sort_left_right_eigensystem(eom, r_converged[0], r_e[0], r_v[0],
                                                      l_converged[0], l_e[0], l_v[0])
        e_star.append(eom.ccsd_star_contract(ek, r_vk, l_vk, kshift, imds=imds))
    return e_star


def ipccsd_star(eom, nroots=1, koopmans=False, right_guess=None, left_guess=None,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''See `kernel()` for a description of arguments.'''
    if partition:
        raise NotImplementedError
    return perturbed_ccsd_kernel(eom, nroots=nroots, koopmans=koopmans,
                                 right_guess=right_guess, left_guess=left_guess, eris=eris,
                                 imds=imds, partition=partition, kptlist=kptlist, dtype=dtype)


def mask_frozen_ip(eom, vector, kshift, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    r1, r2 = eom.vector_to_amplitudes(vector, kshift=kshift)
    nkpts = eom.nkpts
    nocc, nmo = eom.nocc, eom.nmo
    nvir = nmo - nocc
    kconserv = eom.kconserv

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = eom.nonzero_opadding, eom.nonzero_vpadding

    new_r1 = const * np.ones_like(r1)
    new_r2 = const * np.ones_like(r2)

    new_r1[nonzero_opadding[kshift]] = r1[nonzero_opadding[kshift]]
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, kshift, kj]
            idx = np.ix_([ki], [kj], nonzero_opadding[ki], nonzero_opadding[kj], nonzero_vpadding[kb])
            new_r2[idx] = r2[idx]

    return eom.amplitudes_to_vector(new_r1, new_r2, kshift, kconserv)

class EOMIP(eom_rccsd.EOMIP):
    def __init__(self, cc):
        self.kpts = cc.kpts
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(cc)
        self.kconserv = cc.khelper.kconserv
        eom_rccsd.EOM.__init__(self, cc)

    kernel = ipccsd
    ipccsd = ipccsd
    ipccsd_star = ipccsd_star
    ccsd_star_contract = ipccsd_star_contract

    get_diag = ipccsd_diag
    matvec = ipccsd_matvec
    l_matvec = lipccsd_matvec
    mask_frozen = mask_frozen_ip
    get_padding_k_idx = get_padding_k_idx

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift, imds=None):
        return self.ccsd_star_contract(ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift, imds=imds)

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in self.nonzero_opadding[kshift][::-1][:nroots]:
                g = np.zeros(int(size), dtype=dtype)
                g[n] = 1.0
                g = self.mask_frozen(g, kshift, const=0.0)
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype=dtype)
                g[i] = 1.0
                g = self.mask_frozen(g, kshift, const=0.0)
                guess.append(g)
        return guess

    @property
    def nkpts(self):
        return len(self.kpts)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, kshift=None, nkpts=None, nmo=None, nocc=None, kconserv=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        if kconserv is None: kconserv = self.kconserv
        return vector_to_amplitudes_ip(vector, kshift, nkpts, nmo, nocc, kconserv)

    def amplitudes_to_vector(self, r1, r2, kshift, kconserv=None):
        if kconserv is None: kconserv = self.kconserv
        return amplitudes_to_vector_ip(r1, r2, kshift, kconserv)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts*nocc*(nkpts*nocc-1)*nvir//2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ip()
        return imds

class EOMIP_Ta(EOMIP):
    '''Class for EOM IPCCSD(T)*(a) method by Matthews and Stanton.'''
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_t3p2_ip(self._cc)
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

def amplitudes_to_vector_ea(r1, r2, kshift, kconserv):
    nkpts, nocc, nvir = np.asarray(r2.shape)[[0,2,3]]
    r2_tril = np.zeros((nocc*nkpts*nvir*(nkpts*nvir-1)//2), dtype=r2.dtype)
    index = 0
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        if ka < kb:
            idx, idy = np.tril_indices(nvir, 0)
        else:
            idx, idy = np.tril_indices(nvir, -1)
        r2_tril[index:index + nocc*len(idy)] = r2[kj,ka,:,idx,idy].reshape(-1)
        index = index + nocc*len(idy)
    vector = np.hstack((r1, r2_tril))
    return vector

def vector_to_amplitudes_ea(vector, kshift, nkpts, nmo, nocc, kconserv):
    nvir = nmo - nocc

    r1 = vector[:nvir].copy()
    r2_tril = vector[nvir:].copy().reshape(nocc*nkpts*nvir*(nkpts*nvir-1)//2)
    r2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), dtype=vector.dtype)

    index = 0
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        if ka < kb:
            idx, idy = np.tril_indices(nvir, 0)
        else:
            idx, idy = np.tril_indices(nvir, -1)
        tmp = r2_tril[index:index + nocc*len(idy)].reshape(-1,nocc)
        r2[kj,ka,:,idx,idy] = tmp
        r2[kj,kb,:,idy,idx] = -tmp
        index = index + nocc*len(idy)

    return [r1,r2]

def eaccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None):
    '''See `ipccsd()` for a description of arguments.'''
    return ipccsd(eom, nroots, koopmans, guess, left, eris, imds,
                  partition, kptlist, dtype)

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2hp operators are of the form s_{ j}^{ab}, i.e. 'jb' indices are coupled.'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ea(vector, kshift, nkpts, nmo, nocc, kconserv)

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

    vector = eom.amplitudes_to_vector(Hr1, Hr2, kshift)
    return vector

def leaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2hp operators are of the form s_{ j}^{ab}, i.e. 'jb' indices are coupled.

    See also `eaccsd_matvec`'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ea(vector, kshift, nkpts, nmo, nocc, kconserv)
    dtype = np.result_type(r1, r2)

    Hr1 = np.einsum('ca,c->a', imds.Fvv[kshift], r1)
    for kj, kb in itertools.product(range(nkpts), repeat=2):
        kc = kconserv[kshift,kb,kj]
        Hr1 += 0.5*lib.einsum('cbaj,jcb->a',imds.Wvvvo[kc,kb,kshift],r2[kj,kc])

    Hr2 = np.zeros_like(r2)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        Hr2[kj,ka] += lib.einsum('cjab,c->jab',imds.Wvovv[kshift,kj,ka],r1)
        Hr2[kj,kshift] += (kj==kb)*lib.einsum('jb,a->jab',imds.Fov[kj],r1)
        Hr2[kj,ka] -= (kj==ka)*lib.einsum('ja,b->jab',imds.Fov[kj],r1)

    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        tmp1 = lib.einsum('ca,jcb->jab',imds.Fvv[ka],r2[kj,ka])
        tmp1T = lib.einsum('cb,jca->jab',imds.Fvv[kb],r2[kj,kb])
        Hr2[kj,ka] += (tmp1 - tmp1T)
        Hr2[kj,ka] += -lib.einsum('jl,lab->jab',imds.Foo[kj],r2[kj,ka])

        for kd in range(nkpts):
            km = kconserv[kj,kb,kd]
            tmp2 = lib.einsum('jdbm,mad->jab',imds.Wovvo[kj,kd,kb],r2[km,ka])
            km = kconserv[kj,ka,kd]
            tmp2T = lib.einsum('jdam,mbd->jab',imds.Wovvo[kj,kd,ka],r2[km,kb])
            Hr2[kj,ka] += (tmp2 - tmp2T)

            kc = kconserv[ka,kd,kb]
            Hr2[kj,ka] += 0.5*lib.einsum('cdab,jcd->jab',imds.Wvvvv[kc,kd,ka],r2[kj,kc])

    tmp = np.zeros(nocc, dtype=dtype)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        tmp += lib.einsum('jab,kjab->k',r2[kj,ka],imds.t2[kshift,kj,ka])

    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        Hr2[kj,ka] += -0.5*lib.einsum('kjab,k->jab',imds.Woovv[kshift,kj,ka],tmp)

    vector = eom.amplitudes_to_vector(Hr1, Hr2, kshift)
    return vector


def eaccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = np.diag(imds.Fvv[kshift])
    print("eaccsd: Hr1 shape =", Hr1.shape)
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

    vector = amplitudes_to_vector_ea(Hr1, Hr2, kshift, kconserv)
    return vector

def eaccsd_star_contract(eom, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift, imds=None):
    """
    Returns:
        e_star (list of float):
            The EA-CCSD* energy.

    Notes:
        The user should check to make sure the right and left eigenvalues
        before running the perturbative correction.

    Reference:
        Saeh, Stanton "...energy surfaces of radicals" JCP 111, 8275 (1999)
    """
    assert (eom.partition == None)
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    nmo = nocc + nvir
    dtype = np.result_type(t1, t2)
    kconserv = eom.kconserv

    fov = fock[:, :nocc, nocc:]
    foo = [fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)]
    fvv = [fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)]
    mo_energy_occ = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    eaccsd_evecs = np.array(eaccsd_evecs)
    leaccsd_evecs = np.array(leaccsd_evecs)
    e_star = []
    eaccsd_evecs, leaccsd_evecs = [np.atleast_2d(x) for x in [eaccsd_evecs, leaccsd_evecs]]
    eaccsd_evals = np.atleast_1d(eaccsd_evals)
    for ea_eval, ea_evec, ea_levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        # Enforcing <L|R> = 1
        l1, l2 = vector_to_amplitudes_ea(ea_levec, kshift, nkpts, nmo, nocc, kconserv)
        r1, r2 = vector_to_amplitudes_ea(ea_evec, kshift, nkpts, nmo, nocc, kconserv)
        ldotr = np.dot(l1, r1) + 0.5 * np.dot(l2.ravel(), r2.ravel())

        logger.info(eom, 'Left-right amplitude overlap : %14.8e + 1j %14.8e',
                    ldotr.real, ldotr.imag)
        if abs(ldotr) < 1e-7:
            logger.warn(eom, 'Small %s left-right amplitude overlap. Results '
                             'may be inaccurate.', ldotr)

        l1 /= ldotr
        l2 /= ldotr

        deltaE = 0.0 + 1j*0.0
        for ki, kj in itertools.product(range(nkpts), repeat=2):
            lijabc = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir,nvir),dtype=dtype)
            rijabc = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir,nvir),dtype=dtype)
            kklist = kpts_helper.get_kconserv3(eom._cc._scf.cell, eom._cc.kpts,
                          [ki,kj,kshift,range(nkpts),range(nkpts)])

            for ka, kb in itertools.product(range(nkpts), repeat=2):
                #TODO: can reduce size of ijabc arrays since `kc` fixed from other k-points
                kc = kklist[ka,kb]

                # lijabc update
                if kc == kshift and kb == kconserv[ki,ka,kj]:
                    lijabc[ka,kb,kc] -= lib.einsum('ijab,c->ijabc', eris.oovv[ki,kj,ka], l1)

                km = kconserv[kj,ka,ki]
                lijabc[ka,kb,kc] -= lib.einsum('jima,mbc->ijabc', eris.ooov[kj,ki,km], l2[km,kb])

                ke = kconserv[ka,ki,kb]
                tmp = lib.einsum('ieab,jce->ijabc', eris.ovvv[ki,ke,ka], l2[kj,kc])
                ke = kconserv[ka,kj,kb]
                tmpT = lib.einsum('jeab,ice->ijabc', eris.ovvv[kj,ke,ka], l2[ki,kc])
                lijabc[ka,kb,kc] -= (tmp - tmpT)

                # rijabc update
                ke = kconserv[kb,kshift,kc]
                tmp = lib.einsum('bcef,f->bce', eris.vvvv[kb,kc,ke], r1)
                tmp = lib.einsum('bce,ijae->ijabc', tmp, t2[ki,kj,ka])
                rijabc[ka,kb,kc] -= tmp

                km = kconserv[kj,kc,kshift]
                tmp = lib.einsum('mcje,e->mcj', eris.ovov[km,kc,kj], r1)
                tmp = lib.einsum('mcj,imab->ijabc', tmp, t2[ki,km,ka])
                km = kconserv[ki,kc,kshift]
                tmpT = lib.einsum('mcie,e->mci', eris.ovov[km,kc,ki], r1)
                tmpT = lib.einsum('mci,jmab->ijabc', tmpT, t2[kj,km,ka])
                rijabc[ka,kb,kc] += (tmp - tmpT)

                km = kconserv[kj,ka,ki]
                rijabc[ka,kb,kc] += lib.einsum('jima,mcb->ijabc', eris.ooov[kj,ki,km].conj(), r2[km,kc])

                ke = kconserv[ka,ki,kb]
                tmp = lib.einsum('ieab,jce->ijabc', eris.ovvv[ki,ke,ka].conj(), r2[kj,kc])
                ke = kconserv[ka,kj,kb]
                tmpT = lib.einsum('jeab,ice->ijabc', eris.ovvv[kj,ke,ka].conj(), r2[ki,kc])
                rijabc[ka,kb,kc] -= (tmp - tmpT)

            # P(ijk)
            lijabc = lijabc + lijabc.transpose(1,2,0,3,4,6,7,5) + lijabc.transpose(2,0,1,3,4,7,5,6)
            rijabc = rijabc + rijabc.transpose(1,2,0,3,4,6,7,5) + rijabc.transpose(2,0,1,3,4,7,5,6)

            # Creating denominator
            eabc = (mo_e_v[:, None, None, :, None, None] + mo_e_v[None, :, None, None, :, None] +
                    mo_e_v[None, None, :, None, None, :])
            eij = mo_e_o[ki][:, None] + mo_e_o[kj][None, :]
            eijabc = (eij[None, None, None, :, :, None, None, None] -
                      eabc[:, :, :, None, None, :, :, :])
            denom = eijabc + ea_eval
            denom = 1. / denom

            deltaE += lib.einsum('xyzijabc,xyzijabc,xyzijabc', lijabc, rijabc, denom)

        deltaE *= 1./12
        deltaE = deltaE.real
        logger.info(eom, "Exc. energy, delta energy = %16.12f, %16.12f",
        ea_eval + deltaE, deltaE)
        e_star.append(ea_eval + deltaE)
    return e_star

def eaccsd_star(eom, nroots=1, koopmans=False, right_guess=None, left_guess=None,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''See `kernel()` for a description of arguments.'''
    if partition:
        raise NotImplementedError
    return perturbed_ccsd_kernel(eom, nroots=nroots, koopmans=koopmans,
                                 right_guess=right_guess, left_guess=left_guess, eris=eris,
                                 imds=imds, partition=partition, kptlist=kptlist, dtype=dtype)


def mask_frozen_ea(eom, vector, kshift, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    r1, r2 = eom.vector_to_amplitudes(vector, kshift=kshift)
    kconserv = eom.kconserv
    nkpts = eom.nkpts
    nocc, nmo = eom.nocc, eom.nmo
    nvir = nmo - nocc

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = eom.nonzero_opadding, eom.nonzero_vpadding

    new_r1 = const * np.ones_like(r1)
    new_r2 = const * np.ones_like(r2)

    new_r1[nonzero_vpadding[kshift]] = r1[nonzero_vpadding[kshift]]
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding[kj], nonzero_vpadding[ka], nonzero_vpadding[kb])
            new_r2[idx] = r2[idx]

    return eom.amplitudes_to_vector(new_r1, new_r2, kshift, kconserv)

class EOMEA(eom_rccsd.EOMEA):
    def __init__(self, cc):
        self.kpts = cc.kpts
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(cc)
        self.kconserv = cc.khelper.kconserv
        eom_rccsd.EOM.__init__(self, cc)

    kernel = eaccsd
    eaccsd = eaccsd
    eaccsd_star = eaccsd_star
    ccsd_star_contract = eaccsd_star_contract

    get_diag = eaccsd_diag
    matvec = eaccsd_matvec
    l_matvec = leaccsd_matvec
    mask_frozen = mask_frozen_ea
    get_padding_k_idx = get_padding_k_idx

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift, imds=None):
        return self.ccsd_star_contract(eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift, imds=imds)

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in self.nonzero_vpadding[kshift][:nroots]:
                g = np.zeros(int(size), dtype=dtype)
                g[n] = 1.0
                g = self.mask_frozen(g, kshift, const=0.0)
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype=dtype)
                g[i] = 1.0
                g = self.mask_frozen(g, kshift, const=0.0)
                guess.append(g)
        return guess

    @property
    def nkpts(self):
        return len(self.kpts)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, kshift=None, nkpts=None, nmo=None, nocc=None, kconserv=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        if kconserv is None: kconserv = self.kconserv
        return vector_to_amplitudes_ea(vector, kshift, nkpts, nmo, nocc, kconserv)

    def amplitudes_to_vector(self, r1, r2, kshift, kconserv=None):
        if kconserv is None: kconserv = self.kconserv
        return amplitudes_to_vector_ea(r1, r2, kshift, kconserv)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nvir + nocc*nkpts*nvir*(nkpts*nvir-1)//2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ea()
        return imds

class EOMEA_Ta(EOMEA):
    '''Class for EOM EACCSD(T)*(a) method by Matthews and Stanton.'''
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_t3p2_ea(self._cc)
        return imds

########################################
# EOM-EE-CCSD
########################################

def kernel_ee(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''See `kernel()` for a description of arguments.

    This method is merely a simplified version of kernel() with a few parts
    removed, such as those involving `eom.mask_frozen()`. Slowly they will be
    added back for the completion of program.
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris=eris)

    size = eom.vector_size()
    nroots = min(nroots, size)
    nkpts = eom.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    # TODO mask frozen-orbital indices

    if dtype is None:
        dtype = np.result_type(*imds.t1)

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    convs = np.zeros((len(kptlist),nroots), dtype)

    for k, kshift in enumerate(kptlist):
        eom.get_diag(kshift, imds)

    for k, kshift in enumerate(kptlist):
        matvec, diag = eom.gen_matvec(kshift, imds, left=left, **kwargs)
        raise NotImplementedError # Remove after finishing gen_matvec()
        # TODO update `diag` in case of frozen orbitals

        # TODO allow user provided guess vector
        if guess:
            raise NotImplementedError
        else:
            user_guess = False
            guess = eom.get_init_guess(kshift, nroots, koopmans, diag)
        for ig, g in enumerate(guess):
            guess_norm = np.linalg.norm(g)
            guess_norm_tol = LOOSE_ZERO_TOL
            if guess_norm < guess_norm_tol:
                raise ValueError('Guess vector (id=%d) with norm %.4g is below threshold %.4g.\n'
                                 'This could possibly be due to masking/freezing orbitals.\n'
                                 'Check your guess vector to make sure it has sufficiently large norm.'
                                 % (ig, guess_norm, guess_norm_tol))

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = lib.davidson_nosym1
        # TODO allow user provided guess vector or Koopmans
        if user_guess or koopmans:
            raise NotImplementedError
        else:
            conv_k, evals_k, evecs_k = eig(matvec, guess, precond,
                                       tol=eom.conv_tol, max_cycle=eom.max_cycle,
                                       max_space=eom.max_space, nroots=nroots, verbose=eom.verbose)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        convs[k] = conv_k

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn, kshift=kshift)
            if isinstance(r1, np.ndarray):
                qp_weight = np.linalg.norm(r1) ** 2
            else:  # for EOM-UCCSD
                r1 = np.hstack([x.ravel() for x in r1])
                qp_weight = np.linalg.norm(r1) ** 2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    return convs, evals, evecs


def eeccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None):
    '''See `ipccsd()` for a description of arguments.'''
    eom.converged, eom.e, eom.v \
            = kernel_ee(eom, nroots, koopmans, guess, left, eris=eris, imds=imds,
                  partition=partition, kptlist=kptlist, dtype=dtype)
    return eom.e, eom.v


# TODO complete this method
def eeccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    return None


# TODO complete this method
def eeccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = eom.kconserv
    kconserv_r1 = eom.get_kconserv_r1(kshift)


    Hr1 = np.zeros((nkpts, nocc, nvir), dtype=t1.dtype)


    return None


# TODO complete this method
def mask_frozen_ee(eom, vector, kshift, const=LARGE_DENOM):
    '''Replace all frozen orbital indices of `vector` with the value `const`.'''
    return None


# TODO complete this method
def vector_to_amplitudes_ee(vector, kshift, nkpts, nmo, nocc, kconserv):
    nvir = nmo - nocc

    r1 = vector[:nkpts*nocc*nvir].copy()
    r2_tril = vector[nkpts*nocc*nvir:].copy()

    return None


# TODO complete this method
def amplitudes_to_vector_ee(r1, r2, kshift, kconserv):
    return None


class EOMEE(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        self.kconserv = cc.khelper.kconserv
        eom_rccsd.EOM.__init__(self, cc)

    kernel = eeccsd
    eeccsd = eeccsd
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag
    mask_frozen = mask_frozen_ee

    @property
    def nkpts(self):
        return len(self.kpts)

    def vector_size(self):
        '''Size of the linear excitation operator R vector based on spin-orbital basis'''
        nocc = self.nocc  # alpha+beta
        nvir = self.nmo-nocc  # alpha+beta
        nkpts = self.nkpts

        size_r1 = nkpts*nocc*nvir
        if nkpts % 2 == 1:
            size_r2 = nkpts*nocc*(nkpts*nocc-1)//2*nvir*(nkpts*nvir-1)//2
        else:
            size_oo = nocc*(nocc-1)//2  # When ki==kj, there are size_oo ways to create 2 holes
            size_vv = nvir*(nvir-1)//2  # When ka==kb, there are size_vv ways to create 2 particles
            size_r2 = 0
            kconserv = self.kconserv
            # TODO Optimize this 3-layer for loop, or find an elegant solution
            for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
                kb = kconserv[ki, ka, kj]
                if ki == kj:
                    if ka == kb:
                        size_r2 += size_oo*size_vv
                    elif ka > kb:
                        size_r2 += size_oo*nvir**2
                elif ki > kj:
                    if ka == kb:
                        size_r2 += nocc**2*size_vv
                    elif ka > kb:
                        size_r2 += nocc**2*nvir**2

        return size_r1 + size_r2

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        """Initial guess vectors of R coefficients"""
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        # TODO do Koopmans later
        if koopmans:
            raise NotImplementedError
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype=dtype)
                g[i] = 1.0
                # TODO do mask_frozen later
                guess.append(g)
        return guess

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            # TODO allow left vectors to be computed
            raise NotImplementedError
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    # TODO complete this method
    def vector_to_amplitudes(self, vector, kshift=None, nkpts=None, nmo=None, nocc=None, kconserv=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        if kconserv is None: kconserv = self.kconserv
        return vector_to_amplitudes_ee(vector, kshift, nkpts, nmo, nocc, kconserv)

    # TODO complete this method
    def amplitudes_to_vector(self, r1, r2, kshift, kconserv=None):
        return amplitudes_to_vector_ee(r1, r2, kshift, kconserv)

    def get_kconserv_r1(self, kshift=0):
        '''Get the momentum conservation array for a set of k-points.

        Given k-point index m the array kconserv_r1[m] returns the index n that
        satisfies momentum conservation,

            (k(m) - k(n) - kshift) \dot a = 2n\pi

        This is used for symmetry of 1p-1h excitation operator vector
        R_{m k_m}^{n k_n} is zero unless n satisfies the above.

        Note that this method is adapted from `kpts_helper.get_kconserv()`.
        '''
        kconserv_r1 = self.kconserv[:,kshift,0].copy()
        return kconserv_r1

    # TODO Complete this method and merge it with `kpts_helper.get_kconserv()`
    def get_kconserv_r2(self, kshift=0):
        r'''Get the momentum conservation array for a set of k-points.

        Given k-point indices (k, l, m) the array kconserv_r2[k,l,m] returns
        the index n that satisfies momentum conservation,

            (k(k) - k(l) + k(m) - k(n) - kshift) \dot a = 2n\pi

        This is used for symmetry of 2p-2h excitation operator vector
        R_{k k_k, m k_m}^{l k_l n k_n} is zero unless n satisfies the above.

        Note that this method is adapted from `kpts_helper.get_kconserv()`.
        '''
        nkpts = kpts.shape[0]
        a = cell.lattice_vectors() / (2 * np.pi)

        kconserv_r2 = np.zeros((nkpts, nkpts, nkpts), dtype=int)
        # TODO add kshift in this line!!!
        kvKLM = kpts[:, None, None, :] - kpts[:, None, :] + kpts
        for N, kvN in enumerate(kpts):
            kvKLMN = np.einsum('wx,klmx->wklm', a, kvKLM - kvN)
            # check whether (1/(2pi) k_{KLMN} dot a) is an integer
            kvKLMN_int = np.rint(kvKLMN)
            mask = np.einsum('wklm->klm', abs(kvKLMN - kvKLMN_int)) < 1e-9
            print("mask:\n", mask)
            kconserv_r2[mask] = N
            print("kconserv_r2:\n", kconserv_r2)
        return kconserv_r2

    # TODO complete this method
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ee()
        return imds

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc, eris=None):
        self._cc = cc
        self.verbose = cc.verbose
        self.kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        self.stdout = cc.stdout
        self.t1, self.t2 = cc.t1, cc.t2
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

    def make_t3p2_ip(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a IP intermediates', *cput0)
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

    def make_t3p2_ea(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared = False  # Force update
        self.make_ea()  # Make after t1/t2 updated
        self.Wvvvo = self.Wvvvo + Wvvvo

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a EA intermediates', *cput0)
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
            self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris, kconserv, self.Wvvvv)

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
