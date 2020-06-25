#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

'''base class for CTF-based CCSD'''

import time
import ctf

from pyscf.cc import ccsd
from pyscf.lib import logger
from pyscf.ctfcc.linalg_helper.diis import DIIS
from pyscf.ctfcc.mpi_helper import comm
from symtensor.sym_ctf import tensor

def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    if eris is None: mycc.eris = eris = mycc.ao2mo(self.mo_coeff)
    if t1 is None: t1, t2 = mycc.init_amps(eris)[1:]

    if isinstance(mycc.diis, DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = DIIS(mycc)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    eccsd = 0
    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        delta_vec = mycc.amplitudes_to_vector(t1new, t2new) - mycc.amplitudes_to_vector(t1, t2)
        normt = ctf.norm(delta_vec)
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        logger.info(mycc, 'cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = logger.timer(mycc, 'CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    logger.timer(mycc, 'CCSD', *cput0)

    return conv, eccsd, t1, t2

class CCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        comm.barrier()
        mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)
        mf.mo_occ = comm.bcast(mf.mo_occ, root=0)
        self.symlib = None
        self.SYMVERBOSE = SYMVERBOSE
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])

    @property
    def _sym1(self):
        """symmetry identifier for T1, None for mole system"""
        return None

    @property
    def _sym2(self):
        """symmetry identifier for T2, None for mole system"""
        return None

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nov = nocc * nvir
        t1 = vec[:nov].reshape(nocc,nvir)
        t2 = vec[nov:].reshape(nocc,nocc,nvir,nvir)
        t1 = tensor(t1, self._sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2 = tensor(t2, self._sym2, symlib=self.symlib, verbose=self.SYMVERBOSE)
        return t1, t2

    def amplitudes_to_vector(self, t1, t2):
        vector = ctf.hstack((t1.array.ravel(), t2.array.ravel()))
        return vector

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)

    def ccsd(self, t1=None, t2=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            self.eris = eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, 'e_hf', None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def init_amps(self, eris):
        raise NotImplementedError

    def update_amps(self, t1, t2, eris):
        raise NotImplementedError

    def energy(cc, t1, t2, eris):
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None):
        raise NotImplementedError
