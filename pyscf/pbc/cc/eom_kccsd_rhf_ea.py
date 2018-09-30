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
# Author: Artem Pulkin
from pyscf.lib import logger, linalg_helper, einsum
from pyscf.lib.parameters import LARGE_DENOM

from pyscf.pbc.lib.kpts_helper import nested_to_vector, vector_to_nested
from pyscf.pbc.mp.kmp2 import padding_k_idx

import numpy as np

import time


def ea_vector_desc(cc):
    """Description of the EA vector."""
    nvir = cc.nmo - cc.nocc
    return [(nvir,), (cc.nkpts, cc.nkpts, cc.nocc, nvir, nvir)]


def ea_amplitudes_to_vector(cc, t1, t2):
    """Ground state amplitudes to a vector."""
    return nested_to_vector((t1, t2))[0]


def ea_vector_to_amplitudes(cc, vec):
    """Ground state vector to apmplitudes."""
    return vector_to_nested(vec, ea_vector_desc(cc))


def vector_size_ea(self):
    nocc = self.nocc
    nvir = self.nmo - nocc
    nkpts = self.nkpts

    size = nvir + nkpts ** 2 * nvir ** 2 * nocc
    return size


def eaccsd(self, nroots=1, koopmans=False, guess=None, partition=None,
           kptlist=None):
    '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

    Kwargs:
        See ipccd()
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(self.stdout, self.verbose)
    nocc = self.nocc
    nvir = self.nmo - nocc
    nkpts = self.nkpts
    if kptlist is None:
        kptlist = range(nkpts)
    size = vector_size_ea(self)
    for k, kshift in enumerate(kptlist):
        nfrozen = np.sum(mask_frozen_ea(self, np.zeros(size, dtype=int), kshift, const=1))
        nroots = min(nroots, size - nfrozen)
    if partition:
        partition = partition.lower()
        assert partition in ['mp', 'full']
    self.ea_partition = partition
    evals = np.zeros((len(kptlist), nroots), np.float)
    evecs = np.zeros((len(kptlist), nroots, size), np.complex)

    for k, kshift in enumerate(kptlist):
        adiag = eaccsd_diag(self, kshift)
        adiag = mask_frozen_ea(self, adiag, kshift, const=LARGE_DENOM)
        if partition == 'full':
            self._eaccsd_diag_matrix2 = ea_vector_to_amplitudes(self, adiag)[1]

        if guess is not None:
            guess_k = guess[k]
            assert len(guess_k) == nroots
            for g in guess_k:
                assert g.size == size
        else:
            guess_k = []
            if koopmans:
                # Get location of padded elements in occupied and virtual space
                nonzero_vpadding = padding_k_idx(self, kind="split")[1][kshift]

                for n in nonzero_vpadding[:nroots]:
                    g = np.zeros(size)
                    g[n] = 1.0
                    g = mask_frozen_ea(self, g, kshift, const=0.0)
                    guess_k.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    g = mask_frozen_ea(self, g, kshift, const=0.0)
                    guess_k.append(g)

        def precond(r, e0, x0):
            return r / (e0 - adiag + 1e-12)

        eig = linalg_helper.eig
        if guess is not None or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax(np.abs(np.dot(np.array(guess_k).conj(), np.array(x0).T)), axis=1)
                return linalg_helper._eigs_cmplx2real(w, v, idx)

            evals_k, evecs_k = eig(lambda _arg: eaccsd_matvec(self, _arg, kshift), guess_k, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            evals_k, evecs_k = eig(lambda _arg: eaccsd_matvec(self, _arg, kshift), guess_k, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k

        if nroots == 1:
            evals_k, evecs_k = [evals_k], [evecs_k]

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = ea_vector_to_amplitudes(self, vn)
            qp_weight = np.linalg.norm(r1) ** 2
            logger.info(self, 'EOM root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    self.eea = evals
    return self.eea, evecs


def eaccsd_matvec(self, vector, kshift):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if not self.imds.made_ea_imds:
        self.imds.make_ea(self.ea_partition)
    imds = self.imds

    vector = mask_frozen_ea(self, vector, kshift, const=0.0)
    r1, r2 = ea_vector_to_amplitudes(self, vector)

    t1, t2 = self.t1, self.t2
    nkpts = self.nkpts
    kconserv = self.khelper.kconserv

    # Eq. (30)
    # 1p-1p block
    Hr1 = einsum('ac,c->a', imds.Lvv[kshift], r1)
    # 1p-2p1h block
    for kl in range(nkpts):
        Hr1 += 2. * einsum('ld,lad->a', imds.Fov[kl], r2[kl, kshift])
        Hr1 += -einsum('ld,lda->a', imds.Fov[kl], r2[kl, kl])
        for kc in range(nkpts):
            kd = kconserv[kshift, kc, kl]
            Hr1 += 2. * einsum('alcd,lcd->a', imds.Wvovv[kshift, kl, kc], r2[kl, kc])
            Hr1 += -einsum('aldc,lcd->a', imds.Wvovv[kshift, kl, kd], r2[kl, kc])

    # Eq. (31)
    # 2p1h-1p block
    Hr2 = np.zeros(r2.shape, dtype=np.common_type(imds.Wvvvo[0, 0, 0], r1))
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            Hr2[kj, ka] += einsum('abcj,c->jab', imds.Wvvvo[ka, kb, kshift], r1)

    # 2p1h-2p1h block
    if self.ea_partition == 'mp':
        nkpts, nocc, nvir = self.t1.shape
        fock = self.eris.fock
        foo = fock[:, :nocc, :nocc]
        fvv = fock[:, nocc:, nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= einsum('lj,lab->jab', foo[kj], r2[kj, ka])
                Hr2[kj, ka] += einsum('ac,jcb->jab', fvv[ka], r2[kj, ka])
                Hr2[kj, ka] += einsum('bd,jad->jab', fvv[kb], r2[kj, ka])
    elif self.ea_partition == 'full':
        Hr2 += self._eaccsd_diag_matrix2 * r2
    else:
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
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
        tmp = (2. * einsum('xyklcd,xylcd->k', imds.Woovv[kshift, :, :], r2[:, :])
               - einsum('xylkcd,xylcd->k', imds.Woovv[:, kshift, :], r2[:, :]))
        Hr2[:, :] += -einsum('k,xykjab->xyjab', tmp, t2[kshift, :, :])

    return mask_frozen_ea(self, ea_amplitudes_to_vector(self, Hr1, Hr2), kshift, const=0.0)


def eaccsd_diag(self, kshift):
    if not self.imds.made_ea_imds:
        self.imds.make_ea(self.ea_partition)
    imds = self.imds

    t1, t2 = self.t1, self.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = self.khelper.kconserv

    Hr1 = np.diag(imds.Lvv[kshift])

    Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=t2.dtype)
    if self.ea_partition == 'mp':
        foo = self.eris.fock[:, :nocc, :nocc]
        fvv = self.eris.fock[:, nocc:, nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= foo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += fvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += fvv[kb].diagonal()
    else:
        idx = np.eye(nvir, dtype=bool)
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
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

                Hr2[kj, ka] -= 2 * np.einsum('ijab,ijab->jab', t2[kshift, kj, ka], imds.Woovv[kshift, kj, ka])
                Hr2[kj, ka] += np.einsum('ijab,ijba->jab', t2[kshift, kj, ka], imds.Woovv[kshift, kj, kb])

    return ea_amplitudes_to_vector(self, Hr1, Hr2)


def mask_frozen_ea(self, vector, kshift, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    r1, r2 = ea_vector_to_amplitudes(self, vector)
    nkpts, nocc, nvir = self.t1.shape
    kconserv = self.khelper.kconserv

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

    new_r1 = const * np.ones_like(r1)
    new_r2 = const * np.ones_like(r2)

    new_r1[nonzero_vpadding[kshift]] = r1[nonzero_vpadding[kshift]]
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding[kj], nonzero_vpadding[ka], nonzero_vpadding[kb])
            new_r2[idx] = r2[idx]

    return ea_amplitudes_to_vector(self, new_r1, new_r2)
