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
    log = logger.Logger(self.stdout, self.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots,size)
    if partition:
        partition = partition.lower()
        assert partition in ['mp','full']
        if partition in ['mp', 'full']:
            raise NotImplementedError
    eom.partition = partition

    if kptlist is None:
        kptlist = range(nkpts)

    if dtype is None:
        dtype = eom.t1.dtype

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)

    for k, kshift in enumerate(kptlist):
        self.kshift = kshift
        diag = self.ipccsd_diag(kshift)
        if partition == 'full':
            eom.diag = diag

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

        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            evals_k, evecs_k = eig(self.ipccsd_matvec, guess, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            evals_k, evecs_k = eig(self.ipccsd_matvec, guess, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k

        if nroots == 1:
            evals_k, evecs_k = [evals_k], [evecs_k]
        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn)
            qp_weight = np.linalg.norm(r1)**2
            logger.info(self, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    return evals, evecs

########################################
# EOM-IP-CCSD
########################################
def vector_to_amplitudes_ip(self,vector):
    nocc = self.nocc
    nvir = self.nmo - nocc
    nkpts = self.nkpts

    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    return [r1,r2]

def amplitudes_to_vector_ip(self,r1,r2):
    nocc = self.nocc
    nvir = self.nmo - nocc
    nkpts = self.nkpts
    size = nocc + nkpts*nkpts*nocc*nocc*nvir

    vector = np.zeros((size), r1.dtype)
    vector[:nocc] = r1.copy()
    vector[nocc:] = r2.copy().reshape(nkpts*nkpts*nocc*nocc*nvir)
    return vector

class EOMIP(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        eom_rccsd.EOM.__init__(self, cc)

    @property
    def nkpts(self):
        return len(self.kpts)

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
        return nocc + nocc*nocc*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds

########################################
# EOM-EA-CCSD
########################################

class EOMEA(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        eom_rccsd.EOM.__init__(self, cc)

    @property
    def nkpts(self):
        return len(self.kpts)

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
        return nocc + nocc*nocc*nvir

    def make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ip()
        return imds

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc, eris=None, t1=None, t2=None):
        self._cc = cc
        self.verbose = cc.verbose
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

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo = imd.Foo(self._cc, t1, t2, eris)
        self.Fvv = imd.Fvv(self._cc, t1, t2, eris)
        self.Fov = imd.Fov(self._cc, t1, t2, eris)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(self._cc, t1, t2, eris)
        self.Woovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-CCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(self._cc, t1, t2, eris)
        self.Wooov = imd.Wooov(self._cc, t1, t2, eris)
        self.Wovoo = imd.Wovoo(self._cc, t1, t2, eris)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(self._cc, t1, t2, eris)
        self.Wvvvv = imd.Wvvvv(self._cc, t1, t2, eris)
        self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris,self.Wvvvv)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.Woooo = imd.Woooo(self._cc, t1, t2, eris)
            self.Wooov = imd.Wooov(self._cc, t1, t2, eris)
            self.Wovoo = imd.Wovoo(self._cc, t1, t2, eris)
        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(self._cc, t1, t2, eris)
            self.Wvvvv = imd.Wvvvv(self._cc, t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris,self.Wvvvv)

        self.made_ee_imds = True
        logger.timer(self, 'EOM-CCSD EE intermediates', *cput0)
        return self
