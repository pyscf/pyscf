#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Spin-free lambda equation of UHF-CCSD(T)
'''

import time
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
    from pyscf.cc import uccsd_t_slow
    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))

    imds = uccsd_lambda.make_intermediates(mycc, t1, t2, eris)
    eris = uccsd_t_slow._ChemistsERIs(mycc)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    mo_ea = eris.focka.diagonal().real
    mo_eb = eris.fockb.diagonal().real
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:,:nocca]
    fVO = eris.fockb[noccb:,:noccb]

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2aa, eris.vovp[:,:,:,nocca:])
    w-= numpy.einsum('mkbc,aimj->ijkabc', t2aa, eris.vooo)
    v = numpy.einsum('bjck,ia->ijkabc', eris.vovp[:,:,:,:nocca], t1a)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5

    rw = r6(p6(w)) / d3
    imds.l1a_t = numpy.einsum('ijkabc,bjck->ia', rw,
                              eris.vovp[:,:,:,:nocca]).conj() / eia * .25
    wvd = r6(p6(w * 2 + v)) / d3
    l2_t  = numpy.einsum('ijkabc,ckbe->ijae', wvd, eris.vovp[:,:,:,nocca:])
    l2_t -= numpy.einsum('ijkabc,aimj->mkbc', wvd, eris.vooo)
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fvo)
    imds.l2aa_t = l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eia) * .5

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2bb, eris.VOVP[:,:,:,noccb:])
    w-= numpy.einsum('imab,ckmj->ijkabc', t2bb, eris.VOOO)
    v = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5

    rw = r6(p6(w)) / d3
    imds.l1b_t = numpy.einsum('ijkabc,bjck->ia', rw,
                              eris.VOVP[:,:,:,:noccb]).conj() / eIA * .25
    wvd = r6(p6(w * 2 + v)) / d3
    l2_t  = numpy.einsum('ijkabc,ckbe->ijae', wvd, eris.VOVP[:,:,:,noccb:])
    l2_t -= numpy.einsum('ijkabc,aimj->mkbc', wvd, eris.VOOO)
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fVO)
    imds.l2bb_t = l2_t.conj() / lib.direct_sum('ia+jb->ijab', eIA, eIA) * .5

    # baa
    def r4(w):
        w = w - w.transpose(0,2,1,3,4,5)
        w = w + w.transpose(0,2,1,3,5,4)
        return w
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    w  = numpy.einsum('jIeA,ckbe->IjkAbc', t2ab, eris.vovp[:,:,:,nocca:]) * 2
    w += numpy.einsum('jIbE,ckAE->IjkAbc', t2ab, eris.voVP[:,:,:,noccb:]) * 2
    w += numpy.einsum('jkbe,AIce->IjkAbc', t2aa, eris.VOvp[:,:,:,nocca:])
    w -= numpy.einsum('mIbA,ckmj->IjkAbc', t2ab, eris.vooo) * 2
    w -= numpy.einsum('jMbA,ckMI->IjkAbc', t2ab, eris.voOO) * 2
    w -= numpy.einsum('jmbc,AImk->IjkAbc', t2aa, eris.VOoo)
    v  = numpy.einsum('bjck,IA->IjkAbc', eris.vovp[:,:,:,:nocca], t1b)
    v += numpy.einsum('AIck,jb->IjkAbc', eris.VOvp[:,:,:,:nocca], t1a)
    v += numpy.einsum('ckAI,jb->IjkAbc', eris.voVP[:,:,:,:noccb], t1a)
    v += numpy.einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += numpy.einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2

    rw = r4(w) / d3
    imds.l1a_t += numpy.einsum('ijkabc,aick->jb', rw,
                               eris.VOvp[:,:,:,:nocca]).conj() / eia * .5
    imds.l1b_t += numpy.einsum('ijkabc,bjck->ia', rw,
                               eris.vovp[:,:,:,:nocca]).conj() / eIA * .25
    wvd = r4(w * 2 + v) / d3
    l2_t  = numpy.einsum('ijkabc,aice->jkbe', wvd, eris.VOvp[:,:,:,nocca:])
    l2_t -= numpy.einsum('ijkabc,aimk->jmbc', wvd, eris.VOoo)
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fVO)
    imds.l2aa_t += l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eia) * .5
    l2_t  = numpy.einsum('ijkabc,ckbe->jiea', wvd, eris.vovp[:,:,:,nocca:])
    l2_t += numpy.einsum('ijkabc,ckae->jibe', wvd, eris.voVP[:,:,:,noccb:])
    l2_t -= numpy.einsum('ijkabc,ckmj->miba', wvd, eris.vooo)
    l2_t -= numpy.einsum('ijkabc,ckmi->jmba', wvd, eris.voOO)
    l2_t += numpy.einsum('ijkabc,bj->kica', rw, fvo)
    imds.l2ab_t = l2_t.conj() / lib.direct_sum('ia+jb->ijab', eia, eIA) * .5

    # bba
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    w  = numpy.einsum('ijae,ckbe->ijkabc', t2ab, eris.VOVP[:,:,:,noccb:]) * 2
    w += numpy.einsum('ijeb,ckae->ijkabc', t2ab, eris.VOvp[:,:,:,nocca:]) * 2
    w += numpy.einsum('jkbe,aice->ijkabc', t2bb, eris.voVP[:,:,:,noccb:])
    w -= numpy.einsum('imab,ckmj->ijkabc', t2ab, eris.VOOO) * 2
    w -= numpy.einsum('mjab,ckmi->ijkabc', t2ab, eris.VOoo) * 2
    w -= numpy.einsum('jmbc,aimk->ijkabc', t2bb, eris.voOO)
    v  = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1a)
    v += numpy.einsum('aick,jb->ijkabc', eris.voVP[:,:,:,:noccb], t1b)
    v += numpy.einsum('ckai,jb->ijkabc', eris.VOvp[:,:,:,:nocca], t1b)
    v += numpy.einsum('JKBC,ai->iJKaBC', t2bb, fvo) * .5
    v += numpy.einsum('iKaC,BJ->iJKaBC', t2ab, fVO) * 2

    rw = r4(w) / d3
    imds.l1a_t += numpy.einsum('ijkabc,bjck->ia', rw,
                               eris.VOVP[:,:,:,:noccb]).conj() / eia * .25
    imds.l1b_t += numpy.einsum('ijkabc,aick->jb', rw,
                               eris.voVP[:,:,:,:noccb]).conj() / eIA * .5
    wvd = r4(w * 2 + v) / d3
    l2_t  = numpy.einsum('ijkabc,aice->jkbe', wvd, eris.voVP[:,:,:,noccb:])
    l2_t -= numpy.einsum('ijkabc,aimk->jmbc', wvd, eris.voOO)
    l2_t = l2_t + l2_t.transpose(1,0,3,2)
    l2_t += numpy.einsum('ijkabc,ai->jkbc', rw, fvo)
    imds.l2bb_t += l2_t.conj() / lib.direct_sum('ia+jb->ijab', eIA, eIA) * .5
    l2_t  = numpy.einsum('ijkabc,ckbe->ijae', wvd, eris.VOVP[:,:,:,noccb:])
    l2_t += numpy.einsum('ijkabc,ckae->ijeb', wvd, eris.VOvp[:,:,:,nocca:])
    l2_t -= numpy.einsum('ijkabc,ckmj->imab', wvd, eris.VOOO)
    l2_t -= numpy.einsum('ijkabc,ckmi->mjab', wvd, eris.VOoo)
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
