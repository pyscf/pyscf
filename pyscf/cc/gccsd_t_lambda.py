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
Lambda equation of GHF-CCSD(T) with spin-orbital integrals

Ref:
JCP 98, 8718 (1993); DOI:10.1063/1.464480
JCP 147, 044104 (2017); DOI:10.1063/1.4994918
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_lambda
from pyscf.cc import gccsd_lambda


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    imds = gccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    bcei = numpy.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = numpy.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = numpy.asarray(eris.oovv).conj().transpose(2,3,0,1)

    mo_e = eris.mo_energy
    eia = mo_e[:nocc,None] - mo_e[nocc:]
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    t3c =(numpy.einsum('jkae,bcei->ijkabc', t2, bcei) -
          numpy.einsum('imbc,majk->ijkabc', t2, majk))
    t3c = t3c - t3c.transpose(0,1,2,4,3,5) - t3c.transpose(0,1,2,5,4,3)
    t3c = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
    t3c /= d3

    t3d = numpy.einsum('ia,bcjk->ijkabc', t1, bcjk)
    t3d += numpy.einsum('ai,jkbc->ijkabc', eris.fock[nocc:,:nocc], t2)
    t3d = t3d - t3d.transpose(0,1,2,4,3,5) - t3d.transpose(0,1,2,5,4,3)
    t3d = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)
    t3d /= d3

    l1_t = numpy.einsum('ijkabc,jkbc->ia', t3c.conj(), eris.oovv) / eia
    imds.l1_t = l1_t * .25

    m3 = t3c * 2 + t3d
    tmp = numpy.einsum('ijkaef,kbfe->ijab', m3.conj(), eris.ovvv) * .5
    l2_t = tmp - tmp.transpose(0,1,3,2)
    tmp = numpy.einsum('imnabc,mnjc->ijab', m3.conj(), eris.ooov) * .5
    l2_t -= tmp - tmp.transpose(1,0,2,3)
    l2_t += numpy.einsum('kc,ijkabc->ijab', eris.fock[:nocc,nocc:], t3c.conj())
    imds.l2_t = l2_t / lib.direct_sum('ia+jb->ijab', eia, eia)

    return imds


def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None: eris = mycc.ao2mo()
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = gccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 += imds.l1_t
    l2 += imds.l2_t
    return l1, l2
