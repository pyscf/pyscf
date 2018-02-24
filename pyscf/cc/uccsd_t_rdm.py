#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd_rdm

def _gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None):
    d1 = uccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None: eris = mycc.ao2mo()

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
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

    rw = p6(r6(w)) / d3
    wvd = p6(w + v) / d3
    goo = numpy.einsum('iklabc,jklabc->ij', wvd.conj(), rw) * .125
    gvv = numpy.einsum('ijkacd,ijkbcd->ab', wvd, rw.conj()) * .125
    gvo = numpy.einsum('jkbc,ijkabc->ai', t2aa.conj(), rw) * .125

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2bb, eris.VOVP[:,:,:,noccb:])
    w-= numpy.einsum('imab,ckmj->ijkabc', t2bb, eris.VOOO)
    v = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5

    rw = p6(r6(w)) / d3
    wvd = p6(w + v) / d3
    gOO = numpy.einsum('iklabc,jklabc->ij', wvd.conj(), rw) * .125
    gVV = numpy.einsum('ijkacd,ijkbcd->ab', wvd, rw.conj()) * .125
    gVO = numpy.einsum('jkbc,ijkabc->ai', t2bb.conj(), rw) * .125

    # baa
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
    wvd = (w + v) / d3
    goo += numpy.einsum('kilabc,kjlabc->ij', wvd.conj(), rw) * .25
    goo += numpy.einsum('kliabc,kljabc->ij', wvd.conj(), rw) * .25
    gOO += numpy.einsum('iklabc,jklabc->ij', wvd.conj(), rw) * .25
    gvv += numpy.einsum('ijkcad,ijkcbd->ab', wvd, rw.conj()) * .25
    gvv += numpy.einsum('ijkcda,ijkcdb->ab', wvd, rw.conj()) * .25
    gVV += numpy.einsum('ijkacd,ijkbcd->ab', wvd, rw.conj()) * .25
    gvo += numpy.einsum('kica,ijkabc->bj', t2ab.conj(), rw) * .5
    gVO += numpy.einsum('jkbc,ijkabc->ai', t2aa.conj(), rw) * .125

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
    wvd = (w + v) / d3
    goo += numpy.einsum('iklabc,jklabc->ij', wvd.conj(), rw) * .25
    gOO += numpy.einsum('kilabc,kjlabc->ij', wvd.conj(), rw) * .25
    gOO += numpy.einsum('kliabc,kljabc->ij', wvd.conj(), rw) * .25
    gvv += numpy.einsum('ijkacd,ijkbcd->ab', wvd, rw.conj()) * .25
    gVV += numpy.einsum('ijkcad,ijkcbd->ab', wvd, rw.conj()) * .25
    gVV += numpy.einsum('ijkcda,ijkcdb->ab', wvd, rw.conj()) * .25
    gVO += numpy.einsum('ikac,ijkabc->bj', t2ab.conj(), rw) * .5
    gvo += numpy.einsum('jkbc,ijkabc->ai', t2bb.conj(), rw) * .125

    doo, dOO = d1[0]
    dov, dOV = d1[1]
    dvo, dVO = d1[2]
    dvv, dVV = d1[3]

    doo[numpy.diag_indices(nocca)] -= goo.diagonal()
    dOO[numpy.diag_indices(noccb)] -= gOO.diagonal()
    dvv[numpy.diag_indices(nvira)] += gvv.diagonal()
    dVV[numpy.diag_indices(nvirb)] += gVV.diagonal()
    dvo += gvo
    dVO += gVO

    return d1

# gamma2 intermediates in Chemist's notation
def _gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None):
    d2 = uccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)

    if eris is None: eris = mycc.ao2mo()

    dovov, dovOV, dOVov, dOVOV = d2[0]
    dvvvv, dvvVV, dVVvv, dVVVV = d2[1]
    doooo, dooOO, dOOoo, dOOOO = d2[2]
    doovv, dooVV, dOOvv, dOOVV = d2[3]
    dovvo, dovVO, dOVvo, dOVVO = d2[4]
    dvvov, dvvOV, dVVov, dVVOV = d2[5]
    dovvv, dovVV, dOVvv, dOVVV = d2[6]
    dooov, dooOV, dOOov, dOOOV = d2[7]

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
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
    wvd = r6(p6(w * 2 + v)) / d3
    dovov += numpy.einsum('ia,ijkabc->jbkc', t1a, rw.conj()) * 0.25
    dooov -= numpy.einsum('mkbc,ijkabc->jmia', t2aa, wvd.conj()) * .125
    dovvv += numpy.einsum('kjcf,ijkabc->iafb', t2aa, wvd.conj()) * .125

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2bb, eris.VOVP[:,:,:,noccb:])
    w-= numpy.einsum('imab,ckmj->ijkabc', t2bb, eris.VOOO)
    v = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5

    rw = r6(p6(w)) / d3
    wvd = r6(p6(w * 2 + v)) / d3
    dOVOV += numpy.einsum('ia,ijkabc->jbkc', t1b, rw.conj()) * .25
    dOOOV -= numpy.einsum('mkbc,ijkabc->jmia', t2bb, wvd.conj()) * .125
    dOVVV += numpy.einsum('kjcf,ijkabc->iafb', t2bb, wvd.conj()) * .125

    # baa
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
    wvd = r4(w * 2 + v) / d3
    dovvv += numpy.einsum('jiea,ijkabc->kceb', t2ab, wvd.conj()) * .25
    dovVV += numpy.einsum('jibe,ijkabc->kcea', t2ab, wvd.conj()) * .25
    dOVvv += numpy.einsum('jkbe,ijkabc->iaec', t2aa, wvd.conj()) * .125
    dooov -= numpy.einsum('miba,ijkabc->jmkc', t2ab, wvd.conj()) * .25
    dOOov -= numpy.einsum('jmba,ijkabc->imkc', t2ab, wvd.conj()) * .25
    dooOV -= numpy.einsum('jmbc,ijkabc->kmia', t2aa, wvd.conj()) * .125
    dovov += numpy.einsum('ia,ijkabc->jbkc', t1b, rw.conj()) * .25
    #dOVov += numpy.einsum('jb,ijkabc->iakc', t1a, rw.conj()) * .25
    dovOV += numpy.einsum('jb,ijkabc->kcia', t1a, rw.conj()) * .25

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
    wvd = r4(w * 2 + v) / d3
    dOVVV += numpy.einsum('ijae,ijkabc->kceb', t2ab, wvd.conj()) * .25
    dOVvv += numpy.einsum('ijeb,ijkabc->kcea', t2ab, wvd.conj()) * .25
    dovVV += numpy.einsum('jkbe,ijkabc->iaec', t2bb, wvd.conj()) * .125
    dOOOV -= numpy.einsum('imab,ijkabc->jmkc', t2ab, wvd.conj()) * .25
    dooOV -= numpy.einsum('mjab,ijkabc->imkc', t2ab, wvd.conj()) * .25
    dOOov -= numpy.einsum('jmbc,ijkabc->kmia', t2bb, wvd.conj()) * .125
    dOVOV += numpy.einsum('ia,ijkabc->jbkc', t1a, rw.conj()) * .25
    dovOV += numpy.einsum('jb,ijkabc->iakc', t1b, rw.conj()) * .25
    #dOVov += numpy.einsum('jb,ijkabc->kcia', t1b, rw.conj()) * .25

    return d2

def make_rdm1(mycc, t1, t2, l1, l2, eris=None):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    return uccsd_rdm._make_rdm1(mycc, d1, True)

# rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, eris=None):
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    return uccsd_rdm._make_rdm2(mycc, d1, d2, True, True)

def p6(t):
    return (t + t.transpose(1,2,0,4,5,3) +
            t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
            t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))

def r6(w):
    return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
            - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
            - w.transpose(1,0,2,3,4,5))

def r4(w):
    w = w - w.transpose(0,2,1,3,4,5)
    w = w + w.transpose(0,2,1,3,5,4)
    return w

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import cc
    from pyscf.cc import uccsd_t_slow
    from pyscf.cc import uccsd_t_lambda

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.957 , .587)],
        [1 , (0.2,  .757 , .487)]]
    #mol.basis = '631g'
    mol.build()
    mf0 = mf = scf.RHF(mol).run(conv_tol=1.)
    mf = scf.addons.convert_to_uhf(mf)
    mycc = cc.UCCSD(mf)
    eris = uccsd_t_slow._ChemistsERIs(mycc)

    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
    mycc0 = cc.CCSD(mf0)
    eris0 = mycc0.ao2mo()
    mycc0.kernel(eris=eris0)
    t1 = mycc0.t1
    t2 = mycc0.t2
    imds = ccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
    l1, l2 = ccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
    dm1ref = ccsd_t_rdm.make_rdm1(mycc0, t1, t2, l1, l2, eris0)
    dm2ref = ccsd_t_rdm.make_rdm2(mycc0, t1, t2, l1, l2, eris0)

    t1 = (t1, t1)
    t2aa = t2 - t2.transpose(1,0,2,3)
    t2 = (t2aa, t2, t2aa)
    l1 = (l1, l1)
    l2aa = l2 - l2.transpose(1,0,2,3)
    l2 = (l2aa, l2, l2aa)
    dm1 = make_rdm1(mycc, t1, t2, l1, l2, eris)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2, eris)
    trdm1 = dm1[0] + dm1[1]
    trdm2 = dm2[0] + dm2[1] + dm2[1].transpose(2,3,0,1) + dm2[2]
    print(abs(trdm1 - dm1ref).max())
    print(abs(trdm2 - dm2ref).max())
