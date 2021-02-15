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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Coupled cluster doubles
'''

import numpy
from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm

class CCD(ccsd.CCSD):
    def update_amps(self, t1, t2, eris):
        t1, t2 = ccsd.update_amps(self, t1, t2, eris)
        return numpy.zeros_like(t1), t2

    def kernel(self, t2=None, eris=None):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1 = numpy.zeros((nocc, nvir))
        ccsd.CCSD.kernel(self, t1, t2, eris)
        return self.e_corr, self.t2

    def solve_lambda(self, t2=None, l2=None, eris=None):
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
            l1, l2 = ccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
            return numpy.zeros_like(l1), l2

        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose, fupdate=update_lambda)
        return self.l2

    def make_rdm1(self, t2=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t2)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t2)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)
