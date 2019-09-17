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
# Authors: Artem Pulkin, pyscf authors

from pyscf.lib import logger, linalg_helper, einsum
from pyscf.lib.parameters import LARGE_DENOM

from pyscf.pbc.lib.kpts_helper import VectorSplitter, VectorComposer
from pyscf.pbc.mp.kmp2 import padding_k_idx

import numpy as np

import time


def iter_12(cc, k):
    """Iterates over EA index slices."""
    o, v = padding_k_idx(cc, kind="split")
    kconserv = cc.khelper.kconserv

    yield (v[k],)

    for ki in range(cc.nkpts):
        for ka in range(cc.nkpts):
            kb = kconserv[k, ka, ki]
            yield (ki,), (ka,), o[ki], v[ka], v[kb]


def amplitudes_to_vector(cc, t1, t2, k):
    """EA amplitudes to vector."""
    itr = iter_12(cc, k)
    t1, t2 = np.asarray(t1), np.asarray(t2)

    vc = VectorComposer(t1.dtype)
    vc.put(t1[np.ix_(*next(itr))])
    for slc in itr:
        vc.put(t2[np.ix_(*slc)])
    return vc.flush()


def vector_to_amplitudes(cc, vec, k):
    """EA vector to apmplitudes."""
    expected_vs = vector_size(cc, k)
    if expected_vs != len(vec):
        raise ValueError("The size of the vector passed {:d} should be exactly {:d}".format(len(vec), expected_vs))

    itr = iter_12(cc, k)
    nvirt = cc.nmo - cc.nocc

    vs = VectorSplitter(vec)
    r1 = vs.get(nvirt, slc=next(itr))
    r2 = np.zeros((cc.nkpts, cc.nkpts, cc.nocc, nvirt, nvirt), vec.dtype)
    for slc in itr:
        vs.get(r2, slc=slc)
    return r1, r2


def vector_size(cc, k):
    """The total number of elements in EA vector."""
    size = 0
    for slc in iter_12(cc, k):
        size += np.prod(tuple(len(i) for i in slc))
    return size


def kernel(cc, nroots=1, koopmans=False, guess=None, partition=None,
           kptlist=None):
    '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

    Kwargs:
        See ipccd()
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    nkpts = cc.nkpts
    if kptlist is None:
        kptlist = range(nkpts)
    for k, kshift in enumerate(kptlist):
        size = vector_size(cc, kshift)
        nfrozen = np.sum(mask_frozen(cc, np.zeros(size, dtype=int), kshift, const=1))
        nroots = min(nroots, size - nfrozen)
    if partition:
        partition = partition.lower()
        assert partition in ['mp', 'full']
    cc.ea_partition = partition
    evals = np.zeros((len(kptlist), nroots), np.float)
    evecs = []

    for k, kshift in enumerate(kptlist):
        adiag = diag(cc, kshift)
        adiag = mask_frozen(cc, adiag, kshift, const=LARGE_DENOM)
        size = vector_size(cc, kshift)
        if partition == 'full':
            cc._eaccsd_diag_matrix2 = vector_to_amplitudes(cc, adiag, kshift)[1]

        if guess is not None:
            guess_k = guess[k]
            # assert len(guess_k) == nroots
            for g in guess_k:
                assert g.size == size
        else:
            guess_k = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[n] = 1.0
                    g = mask_frozen(cc, g, kshift, const=0.0)
                    guess_k.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    g = mask_frozen(cc, g, kshift, const=0.0)
                    guess_k.append(g)

        def precond(r, e0, x0):
            return r / (e0 - adiag + 1e-12)

        eig = linalg_helper.eig
        if guess is not None or koopmans:
            def pickeig(w, v, nroots, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                s = np.dot(np.asarray(guess_k).conj(), np.asarray(x0).T)
                snorm = np.einsum('pi,pi->i', s.conj(), s)
                idx = np.argsort(-snorm)[:nroots]
                return linalg_helper._eigs_cmplx2real(w, v, idx, real_eigenvectors=False)

            evals_k, evecs_k = eig(lambda _arg: matvec(cc, _arg, kshift), guess_k, precond, pick=pickeig,
                                   tol=cc.conv_tol, max_cycle=cc.max_cycle,
                                   max_space=cc.max_space, nroots=len(guess_k), verbose=cc.verbose)
        else:
            evals_k, evecs_k = eig(lambda _arg: matvec(cc, _arg, kshift), guess_k, precond,
                                   tol=cc.conv_tol, max_cycle=cc.max_cycle,
                                   max_space=cc.max_space, nroots=len(guess_k), verbose=cc.verbose)

        if nroots == 1:
            evals_k, evecs_k = np.array([evals_k]), np.array([evecs_k])

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs.append(evecs_k)

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = vector_to_amplitudes(cc, vn, kshift)
            qp_weight = np.linalg.norm(r1) ** 2
            logger.info(cc, 'EOM root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    cc.eea = evals
    return cc.eea, evecs


def matvec(cc, vector, k):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if not cc.imds.made_ea_imds:
        cc.imds.make_ea(cc.ea_partition)
    imds = cc.imds

    vector = mask_frozen(cc, vector, k, const=0.0)
    r1, r2 = vector_to_amplitudes(cc, vector, k)

    t1, t2 = cc.t1, cc.t2
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv

    # Eq. (30)
    # 1p-1p block
    Hr1 = einsum('ac,c->a', imds.Lvv[k], r1)
    # 1p-2p1h block
    for kl in range(nkpts):
        Hr1 += 2. * einsum('ld,lad->a', imds.Fov[kl], r2[kl, k])
        Hr1 += -einsum('ld,lda->a', imds.Fov[kl], r2[kl, kl])
        for kc in range(nkpts):
            kd = kconserv[k, kc, kl]
            Hr1 += 2. * einsum('alcd,lcd->a', imds.Wvovv[k, kl, kc], r2[kl, kc])
            Hr1 += -einsum('aldc,lcd->a', imds.Wvovv[k, kl, kd], r2[kl, kc])

    # Eq. (31)
    # 2p1h-1p block
    Hr2 = np.zeros(r2.shape, dtype=np.common_type(imds.Wvvvo[0, 0, 0], r1))
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[k, ka, kj]
            Hr2[kj, ka] += einsum('abcj,c->jab', imds.Wvvvo[ka, kb, k], r1)

    # 2p1h-2p1h block
    if cc.ea_partition == 'mp':
        nkpts, nocc, nvir = cc.t1.shape
        fock = cc.eris.fock
        foo = fock[:, :nocc, :nocc]
        fvv = fock[:, nocc:, nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[k, ka, kj]
                Hr2[kj, ka] -= einsum('lj,lab->jab', foo[kj], r2[kj, ka])
                Hr2[kj, ka] += einsum('ac,jcb->jab', fvv[ka], r2[kj, ka])
                Hr2[kj, ka] += einsum('bd,jad->jab', fvv[kb], r2[kj, ka])
    elif cc.ea_partition == 'full':
        Hr2 += cc._eaccsd_diag_matrix2 * r2
    else:
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[k, ka, kj]
                Hr2[kj, ka] -= einsum('lj,lab->jab', imds.Loo[kj], r2[kj, ka])
                Hr2[kj, ka] += einsum('ac,jcb->jab', imds.Lvv[ka], r2[kj, ka])
                Hr2[kj, ka] += einsum('bd,jad->jab', imds.Lvv[kb], r2[kj, ka])
                for kd in range(nkpts):
                    kc = kconserv[ka, kd, kb]
                    Hr2[kj, ka] += einsum('abcd,jcd->jab', imds.Wvvvv[ka, kb, kc], r2[kj, kc])
                    kl = kconserv[kd, kb, kj]
                    Hr2[kj, ka] += 2. * einsum('lbdj,lad->jab', imds.Wovvo[kl, kb, kd], r2[kl, ka])
                    # imds.Wvovo[kb,kl,kd,kj] <= imds.Wovov[kl,kb,kj,kd].transpose(1,0,3,2)
                    Hr2[kj, ka] += -einsum('bldj,lad->jab', imds.Wovov[kl, kb, kj].transpose(1, 0, 3, 2),
                                           r2[kl, ka])
                    # imds.Wvoov[kb,kl,kj,kd] <= imds.Wovvo[kl,kb,kd,kj].transpose(1,0,3,2)
                    Hr2[kj, ka] += -einsum('bljd,lda->jab', imds.Wovvo[kl, kb, kd].transpose(1, 0, 3, 2),
                                           r2[kl, kd])
                    kl = kconserv[kd, ka, kj]
                    # imds.Wvovo[ka,kl,kd,kj] <= imds.Wovov[kl,ka,kj,kd].transpose(1,0,3,2)
                    Hr2[kj, ka] += -einsum('aldj,ldb->jab', imds.Wovov[kl, ka, kj].transpose(1, 0, 3, 2),
                                           r2[kl, kd])
        tmp = (2. * einsum('xyklcd,xylcd->k', imds.Woovv[k, :, :], r2[:, :])
               - einsum('xylkcd,xylcd->k', imds.Woovv[:, k, :], r2[:, :]))
        Hr2[:, :] += -einsum('k,xykjab->xyjab', tmp, t2[k, :, :])

    return mask_frozen(cc, amplitudes_to_vector(cc, Hr1, Hr2, k), k, const=0.0)


def diag(cc, k):
    """Diagonal for the EA vector update."""
    if not cc.imds.made_ea_imds:
        cc.imds.make_ea(cc.ea_partition)
    imds = cc.imds

    t1, t2 = cc.t1, cc.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv

    Hr1 = np.diag(imds.Lvv[k])

    Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=t2.dtype)
    if cc.ea_partition == 'mp':
        foo = cc.eris.fock[:, :nocc, :nocc]
        fvv = cc.eris.fock[:, nocc:, nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[k, ka, kj]
                Hr2[kj, ka] -= foo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += fvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += fvv[kb].diagonal()
    else:
        idx = np.eye(nvir, dtype=bool)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[k, ka, kj]
                Hr2[kj, ka] -= imds.Loo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += imds.Lvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += imds.Lvv[kb].diagonal()

                Hr2[kj, ka] += np.einsum('abab->ab', imds.Wvvvv[ka, kb, ka])

                Hr2[kj, ka] -= np.einsum('jbjb->jb', imds.Wovov[kj, kb, kj])[:, None, :]
                Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])
                Hr2[kj, ka] += 2. * Wovvo[:, None, :]
                if ka == kb:
                    for a in range(nvir):
                        Hr2[kj, ka, :, a, a] -= Wovvo[:, a]

                Hr2[kj, ka] -= np.einsum('jaja->ja', imds.Wovov[kj, ka, kj])[:, :, None]

                Hr2[kj, ka] -= 2 * np.einsum('ijab,ijab->jab', t2[k, kj, ka], imds.Woovv[k, kj, ka])
                Hr2[kj, ka] += np.einsum('ijab,ijba->jab', t2[k, kj, ka], imds.Woovv[k, kj, kb])

    return amplitudes_to_vector(cc, Hr1, Hr2, k)


def mask_frozen(cc, vector, k, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    r1, r2 = vector_to_amplitudes(cc, vector, k)
    nkpts, nocc, nvir = cc.t1.shape
    kconserv = cc.khelper.kconserv

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    new_r1 = const * np.ones_like(r1)
    new_r2 = const * np.ones_like(r2)

    new_r1[nonzero_vpadding[k]] = r1[nonzero_vpadding[k]]
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[k, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding[kj], nonzero_vpadding[ka], nonzero_vpadding[kb])
            new_r2[idx] = r2[idx]

    return amplitudes_to_vector(cc, new_r1, new_r2, k)
