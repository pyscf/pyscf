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
Spin-free lambda equation of UHF-CCSD(T)
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_lambda
from pyscf.cc import uccsd_lambda


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

def make_intermediates(mycc, t1, t2, eris):
    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))

    imds = uccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:,:nocca]
    fVO = eris.fockb[noccb:,:noccb]

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,kceb->ijkabc', t2aa, numpy.asarray(eris.get_ovvv()).conj())
    w-= numpy.einsum('mkbc,iajm->ijkabc', t2aa, numpy.asarray(eris.ovoo.conj()))
    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.ovov).conj(), t1a)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5

    rw = r6(p6(w)) / d3
    imds.l1a_t = numpy.einsum('ijkabc,jbkc->ia', rw.conj(),
                              numpy.asarray(eris.ovov)) / eia * .25
    wvd = r6(p6(w * 2 + v)) / d3
    l2_t  = numpy.einsum('ijkabc,kceb->ijae', wvd, numpy.asarray(eris.get_ovvv()).conj())
    l2_t -= numpy.einsum('ijkabc,iajm->mkbc', wvd, numpy.asarray(eris.ovoo.conj()))
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fvo)
    imds.l2aa_t = l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eia) * .5

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,kceb->ijkabc', t2bb, numpy.asarray(eris.get_OVVV()).conj())
    w-= numpy.einsum('imab,kcjm->ijkabc', t2bb, numpy.asarray(eris.OVOO.conj()))
    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.OVOV).conj(), t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5

    rw = r6(p6(w)) / d3
    imds.l1b_t = numpy.einsum('ijkabc,jbkc->ia', rw.conj(),
                              numpy.asarray(eris.OVOV)) / eIA * .25
    wvd = r6(p6(w * 2 + v)) / d3
    l2_t  = numpy.einsum('ijkabc,kceb->ijae', wvd, numpy.asarray(eris.get_OVVV()).conj())
    l2_t -= numpy.einsum('ijkabc,iajm->mkbc', wvd, numpy.asarray(eris.OVOO.conj()))
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fVO)
    imds.l2bb_t = l2_t.conj() / lib.direct_sum('ia+jb->ijab', eIA, eIA) * .5

    # baa
    def r4(w):
        w = w - w.transpose(0,2,1,3,4,5)
        w = w + w.transpose(0,2,1,3,5,4)
        return w
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    w  = numpy.einsum('jIeA,kceb->IjkAbc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
    w += numpy.einsum('jIbE,kcEA->IjkAbc', t2ab, numpy.asarray(eris.get_ovVV()).conj()) * 2
    w += numpy.einsum('jkbe,IAec->IjkAbc', t2aa, numpy.asarray(eris.get_OVvv()).conj())
    w -= numpy.einsum('mIbA,kcjm->IjkAbc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
    w -= numpy.einsum('jMbA,kcIM->IjkAbc', t2ab, numpy.asarray(eris.ovOO).conj()) * 2
    w -= numpy.einsum('jmbc,IAkm->IjkAbc', t2aa, numpy.asarray(eris.OVoo).conj())
    v  = numpy.einsum('jbkc,IA->IjkAbc', numpy.asarray(eris.ovov).conj(), t1b)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += numpy.einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2

    rw = r4(w) / d3
    imds.l1a_t += numpy.einsum('ijkabc,kcia->jb', rw.conj(),
                               numpy.asarray(eris.ovOV)) / eia * .5
    imds.l1b_t += numpy.einsum('ijkabc,jbkc->ia', rw.conj(),
                               numpy.asarray(eris.ovov)) / eIA * .25
    wvd = r4(w * 2 + v) / d3
    l2_t  = numpy.einsum('ijkabc,iaec->jkbe', wvd, numpy.asarray(eris.get_OVvv()).conj())
    l2_t -= numpy.einsum('ijkabc,iakm->jmbc', wvd, numpy.asarray(eris.OVoo).conj())
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fVO)
    imds.l2aa_t += l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eia) * .5
    l2_t  = numpy.einsum('ijkabc,kceb->jiea', wvd, numpy.asarray(eris.get_ovvv()).conj())
    l2_t += numpy.einsum('ijkabc,kcea->jibe', wvd, numpy.asarray(eris.get_ovVV()).conj())
    l2_t -= numpy.einsum('ijkabc,kcjm->miba', wvd, numpy.asarray(eris.ovoo).conj())
    l2_t -= numpy.einsum('ijkabc,kcim->jmba', wvd, numpy.asarray(eris.ovOO).conj())
    l2_t += numpy.einsum('ijkabc,bj->kica', rw, fvo)
    imds.l2ab_t = l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eIA) * .5

    # bba
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    w  = numpy.einsum('ijae,kceb->ijkabc', t2ab, numpy.asarray(eris.get_OVVV()).conj()) * 2
    w += numpy.einsum('ijeb,kcea->ijkabc', t2ab, numpy.asarray(eris.get_OVvv()).conj()) * 2
    w += numpy.einsum('jkbe,iaec->ijkabc', t2bb, numpy.asarray(eris.get_ovVV()).conj())
    w -= numpy.einsum('imab,kcjm->ijkabc', t2ab, numpy.asarray(eris.OVOO).conj()) * 2
    w -= numpy.einsum('mjab,kcim->ijkabc', t2ab, numpy.asarray(eris.OVoo).conj()) * 2
    w -= numpy.einsum('jmbc,iakm->ijkabc', t2bb, numpy.asarray(eris.ovOO).conj())
    v  = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.OVOV).conj(), t1a)
    v += numpy.einsum('iakc,jb->ijkabc', numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum('iakc,jb->ijkabc', numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum('JKBC,ai->iJKaBC', t2bb, fvo) * .5
    v += numpy.einsum('iKaC,BJ->iJKaBC', t2ab, fVO) * 2

    rw = r4(w) / d3
    imds.l1a_t += numpy.einsum('ijkabc,jbkc->ia', rw.conj(),
                               numpy.asarray(eris.OVOV)) / eia * .25
    imds.l1b_t += numpy.einsum('ijkabc,iakc->jb', rw.conj(),
                               numpy.asarray(eris.ovOV)) / eIA * .5
    wvd = r4(w * 2 + v) / d3
    l2_t  = numpy.einsum('ijkabc,iaec->jkbe', wvd, numpy.asarray(eris.get_ovVV()).conj())
    l2_t -= numpy.einsum('ijkabc,iakm->jmbc', wvd, numpy.asarray(eris.ovOO).conj())
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fvo)
    imds.l2bb_t += l2_t.conj() / lib.direct_sum('ia+jb->ijab', eIA, eIA) * .5
    l2_t  = numpy.einsum('ijkabc,kceb->ijae', wvd, numpy.asarray(eris.get_OVVV()).conj())
    l2_t += numpy.einsum('ijkabc,kcea->ijeb', wvd, numpy.asarray(eris.get_OVvv()).conj())
    l2_t -= numpy.einsum('ijkabc,kcjm->imab', wvd, numpy.asarray(eris.OVOO).conj())
    l2_t -= numpy.einsum('ijkabc,kcim->mjab', wvd, numpy.asarray(eris.OVoo).conj())
    l2_t += numpy.einsum('ijkabc,bj->ikac', rw, fVO)
    imds.l2ab_t += l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eIA) * .5

    return imds


def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None: eris = mycc.ao2mo()
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = uccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    l1a += imds.l1a_t
    l1b += imds.l1b_t
    l2aa += imds.l2aa_t
    l2ab += imds.l2ab_t
    l2bb += imds.l2bb_t
    return (l1a, l1b), (l2aa, l2ab, l2bb)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf0 = mf = scf.RHF(mol).run(conv_tol=1)
    mf = scf.addons.convert_to_uhf(mf)
    mycc = cc.UCCSD(mf)
    eris = mycc.ao2mo()

    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    mycc0 = cc.CCSD(mf0)
    eris0 = mycc0.ao2mo()
    mycc0.kernel(eris=eris0)
    t1 = mycc0.t1
    t2 = mycc0.t2
    imds = ccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
    l1, l2 = ccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
    l1ref, l2ref = ccsd_t_lambda.update_lambda(mycc0, t1, t2, l1, l2, eris0, imds)

    t1 = (t1, t1)
    t2aa = t2 - t2.transpose(1,0,2,3)
    t2 = (t2aa, t2, t2aa)
    l1 = (l1, l1)
    l2aa = l2 - l2.transpose(1,0,2,3)
    l2 = (l2aa, l2, l2aa)
    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    print(abs(l2[1]-l2[1].transpose(1,0,2,3)-l2[0]).max())
    print(abs(l2[1]-l2[1].transpose(0,1,3,2)-l2[2]).max())
    print(abs(l1[0]-l1ref).max())
    print(abs(l2[1]-l2ref).max())
