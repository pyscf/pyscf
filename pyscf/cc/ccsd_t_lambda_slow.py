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
Spin-free lambda equation of RHF-CCSD(T)

Ref:
JCP, 147, 044104
'''

import time
import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_lambda

# Note: not support fov != 0

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    imds = ccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)

    mo_e = eris.mo_energy
    eia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
    d3 = lib.direct_sum('ia,jb,kc->ijkabc', eia, eia, eia)

    def p6(t):
        t1 = t + t.transpose(0,2,1,3,5,4)
        return t1 + t1.transpose(1,0,2,4,3,5) + t1.transpose(1,2,0,4,5,3)

    def r6(w):
        return (4 * w + w.transpose(0,1,2,4,5,3) + w.transpose(0,1,2,5,3,4)
                - 2 * w.transpose(0,1,2,5,4,3) - 2 * w.transpose(0,1,2,3,5,4)
                - 2 * w.transpose(0,1,2,4,3,5))

    w =(numpy.einsum('iafb,kjcf->ijkabc', eris_ovvv.conj(), t2)
      - numpy.einsum('iajm,mkbc->ijkabc', eris_ovoo.conj(), t2)) / d3
    v =(numpy.einsum('iajb,kc->ijkabc', eris_ovov.conj(), t1)
      + numpy.einsum('ck,ijab->ijkabc', eris.fock[nocc:,:nocc], t2)) / d3
    w = p6(w)
    v = p6(v)

    imds.l1_t = numpy.einsum('jbkc,ijkabc->ia', eris_ovov, r6(w)).conj() / eia * .5

    def as_r6(m):
        # When making derivative over t2, r6 should be called on the 6-index
        # tensor. It gives the equation for lambda2, but not corresponding to
        # the lambda equation used by RCCSD-lambda code.  A transformation was
        # applied in RCCSD-lambda equation  F(lambda)_{ijab} = 0:
        #       2/3 * # F(lambda)_{ijab} + 1/3 * F(lambda)_{jiab} = 0
        # Combining this transformation with r6 operation, leads to the
        # transformation code below
        return m * 2 - m.transpose(0,1,2,5,4,3) - m.transpose(0,1,2,3,5,4)

    m = as_r6(w * 2 + v * .5)
    joovv = numpy.einsum('kfbe,ijkaef->ijab', eris_ovvv, m.conj())
    joovv-= numpy.einsum('ncmj,imnabc->ijab', eris_ovoo, m.conj())
    joovv = joovv + joovv.transpose(1,0,3,2)
    rw = as_r6(w)
    joovv+= numpy.einsum('kc,ijkabc->ijab', eris.fock[:nocc,nocc:], rw.conj())
    imds.l2_t = joovv / lib.direct_sum('ia+jb->ijab', eia, eia)

    return imds

def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None: eris = mycc.ao2mo()
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = ccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 += imds.l1_t
    l2 += imds.l2_t
    return l1, l2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-16
    rhf.scf()

    mcc = ccsd.CCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()
    #l1, l2 = mcc.solve_lambda()
    #print(numpy.linalg.norm(l1)-0.0132626841292)
    #print(numpy.linalg.norm(l2)-0.212575609057)

    conv, l1, l2 = kernel(mcc, mcc.ao2mo(), t1, t2, tol=1e-8)
    print(numpy.linalg.norm(l1)-0.013575484203926739)
    print(numpy.linalg.norm(l2)-0.22029981372536928)

