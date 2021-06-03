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


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd
from pyscf.cc import ccsd_lambda

einsum = lib.einsum

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    fooa = eris.focka[:nocca,:nocca]
    fova = eris.focka[:nocca,nocca:]
    fvoa = eris.focka[nocca:,:nocca]
    fvva = eris.focka[nocca:,nocca:]
    foob = eris.fockb[:noccb,:noccb]
    fovb = eris.fockb[:noccb,noccb:]
    fvob = eris.fockb[noccb:,:noccb]
    fvvb = eris.fockb[noccb:,noccb:]

    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)

    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    OVOV = numpy.asarray(eris.OVOV)
    OVOV = OVOV - OVOV.transpose(0,3,2,1)
    ovOV = numpy.asarray(eris.ovOV)

    v1a  = fvva - einsum('ja,jb->ba', fova, t1a)
    v1b  = fvvb - einsum('ja,jb->ba', fovb, t1b)
    v1a += einsum('jcka,jkbc->ba', ovov, tauaa) * .5
    v1a -= einsum('jaKC,jKbC->ba', ovOV, tauab) * .5
    v1a -= einsum('kaJC,kJbC->ba', ovOV, tauab) * .5
    v1b += einsum('jcka,jkbc->ba', OVOV, taubb) * .5
    v1b -= einsum('kcJA,kJcB->BA', ovOV, tauab) * .5
    v1b -= einsum('jcKA,jKcB->BA', ovOV, tauab) * .5

    v2a  = fooa + einsum('ib,jb->ij', fova, t1a)
    v2b  = foob + einsum('ib,jb->ij', fovb, t1b)
    v2a += einsum('ibkc,jkbc->ij', ovov, tauaa) * .5
    v2a += einsum('ibKC,jKbC->ij', ovOV, tauab)
    v2b += einsum('ibkc,jkbc->ij', OVOV, taubb) * .5
    v2b += einsum('kcIB,kJcB->IJ', ovOV, tauab)

    ovoo = numpy.asarray(eris.ovoo)
    ovoo = ovoo - ovoo.transpose(2,1,0,3)
    OVOO = numpy.asarray(eris.OVOO)
    OVOO = OVOO - OVOO.transpose(2,1,0,3)
    OVoo = numpy.asarray(eris.OVoo)
    ovOO = numpy.asarray(eris.ovOO)
    v2a -= numpy.einsum('ibkj,kb->ij', ovoo, t1a)
    v2a += numpy.einsum('KBij,KB->ij', OVoo, t1b)
    v2b -= numpy.einsum('ibkj,kb->ij', OVOO, t1b)
    v2b += numpy.einsum('kbIJ,kb->IJ', ovOO, t1a)

    v5a  = fvoa + numpy.einsum('kc,jkbc->bj', fova, t2aa)
    v5a += numpy.einsum('KC,jKbC->bj', fovb, t2ab)
    v5b  = fvob + numpy.einsum('kc,jkbc->bj', fovb, t2bb)
    v5b += numpy.einsum('kc,kJcB->BJ', fova, t2ab)
    tmp  = fova - numpy.einsum('kdlc,ld->kc', ovov, t1a)
    tmp += numpy.einsum('kcLD,LD->kc', ovOV, t1b)
    v5a += einsum('kc,kb,jc->bj', tmp, t1a, t1a)
    tmp  = fovb - numpy.einsum('kdlc,ld->kc', OVOV, t1b)
    tmp += numpy.einsum('ldKC,ld->KC', ovOV, t1a)
    v5b += einsum('kc,kb,jc->bj', tmp, t1b, t1b)
    v5a -= einsum('lckj,klbc->bj', ovoo, t2aa) * .5
    v5a -= einsum('LCkj,kLbC->bj', OVoo, t2ab)
    v5b -= einsum('LCKJ,KLBC->BJ', OVOO, t2bb) * .5
    v5b -= einsum('lcKJ,lKcB->BJ', ovOO, t2ab)

    oooo = numpy.asarray(eris.oooo)
    OOOO = numpy.asarray(eris.OOOO)
    ooOO = numpy.asarray(eris.ooOO)
    woooo  = einsum('icjl,kc->ikjl', ovoo, t1a)
    wOOOO  = einsum('icjl,kc->ikjl', OVOO, t1b)
    wooOO  = einsum('icJL,kc->ikJL', ovOO, t1a)
    wooOO += einsum('JCil,KC->ilJK', OVoo, t1b)
    woooo += (oooo - oooo.transpose(0,3,2,1)) * .5
    wOOOO += (OOOO - OOOO.transpose(0,3,2,1)) * .5
    wooOO += ooOO.copy()
    woooo += einsum('icjd,klcd->ikjl', ovov, tauaa) * .25
    wOOOO += einsum('icjd,klcd->ikjl', OVOV, taubb) * .25
    wooOO += einsum('icJD,kLcD->ikJL', ovOV, tauab)

    v4ovvo  = einsum('jbld,klcd->jbck', ovov, t2aa)
    v4ovvo += einsum('jbLD,kLcD->jbck', ovOV, t2ab)
    v4ovvo += numpy.asarray(eris.ovvo)
    v4ovvo -= numpy.asarray(eris.oovv).transpose(0,3,2,1)
    v4OVVO  = einsum('jbld,klcd->jbck', OVOV, t2bb)
    v4OVVO += einsum('ldJB,lKdC->JBCK', ovOV, t2ab)
    v4OVVO += numpy.asarray(eris.OVVO)
    v4OVVO -= numpy.asarray(eris.OOVV).transpose(0,3,2,1)
    v4OVvo  = einsum('ldJB,klcd->JBck', ovOV, t2aa)
    v4OVvo += einsum('JBLD,kLcD->JBck', OVOV, t2ab)
    v4OVvo += numpy.asarray(eris.OVvo)
    v4ovVO  = einsum('jbLD,KLCD->jbCK', ovOV, t2bb)
    v4ovVO += einsum('jbld,lKdC->jbCK', ovov, t2ab)
    v4ovVO += numpy.asarray(eris.ovVO)
    v4oVVo  = einsum('jdLB,kLdC->jBCk', ovOV, t2ab)
    v4oVVo -= numpy.asarray(eris.ooVV).transpose(0,3,2,1)
    v4OvvO  = einsum('lbJD,lKcD->JbcK', ovOV, t2ab)
    v4OvvO -= numpy.asarray(eris.OOvv).transpose(0,3,2,1)

    woovo  = einsum('ibck,jb->ijck', v4ovvo, t1a)
    wOOVO  = einsum('ibck,jb->ijck', v4OVVO, t1b)
    wOOvo  = einsum('IBck,JB->IJck', v4OVvo, t1b)
    wOOvo -= einsum('IbcK,jb->IKcj', v4OvvO, t1a)
    wooVO  = einsum('ibCK,jb->ijCK', v4ovVO, t1a)
    wooVO -= einsum('iBCk,JB->ikCJ', v4oVVo, t1b)
    woovo += ovoo.conj().transpose(3,2,1,0) * .5
    wOOVO += OVOO.conj().transpose(3,2,1,0) * .5
    wooVO += OVoo.conj().transpose(3,2,1,0)
    wOOvo += ovOO.conj().transpose(3,2,1,0)
    woovo -= einsum('iclk,jlbc->ikbj', ovoo, t2aa)
    woovo += einsum('LCik,jLbC->ikbj', OVoo, t2ab)
    wOOVO -= einsum('iclk,jlbc->ikbj', OVOO, t2bb)
    wOOVO += einsum('lcIK,lJcB->IKBJ', ovOO, t2ab)
    wooVO -= einsum('iclk,lJcB->ikBJ', ovoo, t2ab)
    wooVO += einsum('LCik,JLBC->ikBJ', OVoo, t2bb)
    wooVO -= einsum('icLK,jLcB->ijBK', ovOO, t2ab)
    wOOvo -= einsum('ICLK,jLbC->IKbj', OVOO, t2ab)
    wOOvo += einsum('lcIK,jlbc->IKbj', ovOO, t2aa)
    wOOvo -= einsum('IClk,lJbC->IJbk', OVoo, t2ab)

    wvvvo  = einsum('jack,jb->back', v4ovvo, t1a)
    wVVVO  = einsum('jack,jb->back', v4OVVO, t1b)
    wVVvo  = einsum('JAck,JB->BAck', v4OVvo, t1b)
    wVVvo -= einsum('jACk,jb->CAbk', v4oVVo, t1a)
    wvvVO  = einsum('jaCK,jb->baCK', v4ovVO, t1a)
    wvvVO -= einsum('JacK,JB->caBK', v4OvvO, t1b)
    wvvvo += einsum('lajk,jlbc->back', .25*ovoo, tauaa)
    wVVVO += einsum('lajk,jlbc->back', .25*OVOO, taubb)
    wVVvo -= einsum('LAjk,jLcB->BAck', OVoo, tauab)
    wvvVO -= einsum('laJK,lJbC->baCK', ovOO, tauab)

    w3a  = numpy.einsum('jbck,jb->ck', v4ovvo, t1a)
    w3a += numpy.einsum('JBck,JB->ck', v4OVvo, t1b)
    w3b  = numpy.einsum('jbck,jb->ck', v4OVVO, t1b)
    w3b += numpy.einsum('jbCK,jb->CK', v4ovVO, t1a)

    wovvo  = v4ovvo
    wOVVO  = v4OVVO
    wovVO  = v4ovVO
    wOVvo  = v4OVvo
    woVVo  = v4oVVo
    wOvvO  = v4OvvO
    wovvo += lib.einsum('jbld,kd,lc->jbck', ovov, t1a, -t1a)
    wOVVO += lib.einsum('jbld,kd,lc->jbck', OVOV, t1b, -t1b)
    wovVO += lib.einsum('jbLD,KD,LC->jbCK', ovOV, t1b, -t1b)
    wOVvo += lib.einsum('ldJB,kd,lc->JBck', ovOV, t1a, -t1a)
    woVVo += lib.einsum('jdLB,kd,LC->jBCk', ovOV, t1a,  t1b)
    wOvvO += lib.einsum('lbJD,KD,lc->JbcK', ovOV, t1b,  t1a)
    wovvo -= einsum('jblk,lc->jbck', ovoo, t1a)
    wOVVO -= einsum('jblk,lc->jbck', OVOO, t1b)
    wovVO -= einsum('jbLK,LC->jbCK', ovOO, t1b)
    wOVvo -= einsum('JBlk,lc->JBck', OVoo, t1a)
    woVVo += einsum('LBjk,LC->jBCk', OVoo, t1b)
    wOvvO += einsum('lbJK,lc->JbcK', ovOO, t1a)

    if nvira > 0 and nocca > 0:
        ovvv = numpy.asarray(eris.get_ovvv())
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        v1a -= numpy.einsum('jabc,jc->ba', ovvv, t1a)
        v5a += einsum('kdbc,jkcd->bj', ovvv, t2aa) * .5
        woovo += einsum('idcb,kjbd->ijck', ovvv, tauaa) * .25
        wovvo += einsum('jbcd,kd->jbck', ovvv, t1a)
        wvvvo -= ovvv.conj().transpose(3,2,1,0) * .5
        wvvvo += einsum('jacd,kjbd->cabk', ovvv, t2aa)
        wvvVO += einsum('jacd,jKdB->caBK', ovvv, t2ab)
        ovvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OVVV = numpy.asarray(eris.get_OVVV())
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        v1b -= numpy.einsum('jabc,jc->ba', OVVV, t1b)
        v5b += einsum('KDBC,JKCD->BJ', OVVV, t2bb) * .5
        wOOVO += einsum('idcb,kjbd->ijck', OVVV, taubb) * .25
        wOVVO += einsum('jbcd,kd->jbck', OVVV, t1b)
        wVVVO -= OVVV.conj().transpose(3,2,1,0) * .5
        wVVVO += einsum('jacd,kjbd->cabk', OVVV, t2bb)
        wVVvo += einsum('JACD,kJbD->CAbk', OVVV, t2ab)
        OVVV = tmp = None

    if nvirb > 0 and nocca > 0:
        OVvv = numpy.asarray(eris.get_OVvv())
        v1a += numpy.einsum('JCba,JC->ba', OVvv, t1b)
        v5a += einsum('KDbc,jKcD->bj', OVvv, t2ab)
        wOOvo += einsum('IDcb,kJbD->IJck', OVvv, tauab)
        wOVvo += einsum('JBcd,kd->JBck', OVvv, t1a)
        wOvvO -= einsum('JDcb,KD->JbcK', OVvv, t1b)
        wvvVO -= OVvv.conj().transpose(3,2,1,0)
        wvvvo -= einsum('KDca,jKbD->cabj', OVvv, t2ab)
        wvvVO -= einsum('KDca,JKBD->caBJ', OVvv, t2bb)
        wVVvo += einsum('KAcd,jKdB->BAcj', OVvv, t2ab)
        OVvv = tmp = None

    if nvira > 0 and noccb > 0:
        ovVV = numpy.asarray(eris.get_ovVV())
        v1b += numpy.einsum('jcBA,jc->BA', ovVV, t1a)
        v5b += einsum('kdBC,kJdC->BJ', ovVV, t2ab)
        wooVO += einsum('idCB,jKdB->ijCK', ovVV, tauab)
        wovVO += einsum('jbCD,KD->jbCK', ovVV, t1b)
        woVVo -= einsum('jdCB,kd->jBCk', ovVV, t1a)
        wVVvo -= ovVV.conj().transpose(3,2,1,0)
        wVVVO -= einsum('kdCA,kJdB->CABJ', ovVV, t2ab)
        wVVvo -= einsum('kdCA,jkbd->CAbj', ovVV, t2aa)
        wvvVO += einsum('kaCD,kJbD->baCJ', ovVV, t2ab)
        ovVV = tmp = None

    w3a += v5a
    w3b += v5b
    w3a += lib.einsum('cb,jb->cj', v1a, t1a)
    w3b += lib.einsum('cb,jb->cj', v1b, t1b)
    w3a -= lib.einsum('jk,jb->bk', v2a, t1a)
    w3b -= lib.einsum('jk,jb->bk', v2b, t1b)

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    dtype = numpy.result_type(t2ab, eris.vvvv).char
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocca,nocca,nocca,nocca), dtype)
    imds.wooOO = imds.ftmp.create_dataset('wooOO', (nocca,nocca,noccb,noccb), dtype)
    imds.wOOOO = imds.ftmp.create_dataset('wOOOO', (noccb,noccb,noccb,noccb), dtype)
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocca,nvira,nvira,nocca), dtype)
    imds.wOVVO = imds.ftmp.create_dataset('wOVVO', (noccb,nvirb,nvirb,noccb), dtype)
    imds.wovVO = imds.ftmp.create_dataset('wovVO', (nocca,nvira,nvirb,noccb), dtype)
    imds.wOVvo = imds.ftmp.create_dataset('wOVvo', (noccb,nvirb,nvira,nocca), dtype)
    imds.woVVo = imds.ftmp.create_dataset('woVVo', (nocca,nvirb,nvirb,nocca), dtype)
    imds.wOvvO = imds.ftmp.create_dataset('wOvvO', (noccb,nvira,nvira,noccb), dtype)
    imds.woovo = imds.ftmp.create_dataset('woovo', (nocca,nocca,nvira,nocca), dtype)
    imds.wOOVO = imds.ftmp.create_dataset('wOOVO', (noccb,noccb,nvirb,noccb), dtype)
    imds.wOOvo = imds.ftmp.create_dataset('wOOvo', (noccb,noccb,nvira,nocca), dtype)
    imds.wooVO = imds.ftmp.create_dataset('wooVO', (nocca,nocca,nvirb,noccb), dtype)
    imds.wvvvo = imds.ftmp.create_dataset('wvvvo', (nvira,nvira,nvira,nocca), dtype)
    imds.wVVVO = imds.ftmp.create_dataset('wVVVO', (nvirb,nvirb,nvirb,noccb), dtype)
    imds.wVVvo = imds.ftmp.create_dataset('wVVvo', (nvirb,nvirb,nvira,nocca), dtype)
    imds.wvvVO = imds.ftmp.create_dataset('wvvVO', (nvira,nvira,nvirb,noccb), dtype)

    imds.woooo[:] = woooo
    imds.wOOOO[:] = wOOOO
    imds.wooOO[:] = wooOO
    imds.wovvo[:] = wovvo
    imds.wOVVO[:] = wOVVO
    imds.wovVO[:] = wovVO
    imds.wOVvo[:] = wOVvo
    imds.woVVo[:] = woVVo
    imds.wOvvO[:] = wOvvO
    imds.woovo[:] = woovo
    imds.wOOVO[:] = wOOVO
    imds.wOOvo[:] = wOOvo
    imds.wooVO[:] = wooVO
    imds.wvvvo[:] = wvvvo
    imds.wVVVO[:] = wVVVO
    imds.wVVvo[:] = wVVvo
    imds.wvvVO[:] = wvvVO
    imds.v1a = v1a
    imds.v1b = v1b
    imds.v2a = v2a
    imds.v2b = v2b
    imds.w3a = w3a
    imds.w3b = w3b
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    u1a = numpy.zeros_like(l1a)
    u1b = numpy.zeros_like(l1b)
    u2aa = numpy.zeros_like(l2aa)
    u2ab = numpy.zeros_like(l2ab)
    u2bb = numpy.zeros_like(l2bb)
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + mycc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + mycc.level_shift

    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    v1a = imds.v1a - numpy.diag(mo_ea_v)
    v1b = imds.v1b - numpy.diag(mo_eb_v)
    v2a = imds.v2a - numpy.diag(mo_ea_o)
    v2b = imds.v2b - numpy.diag(mo_eb_o)

    mvv = einsum('klca,klcb->ba', l2aa, t2aa) * .5
    mvv+= einsum('lKaC,lKbC->ba', l2ab, t2ab)
    mVV = einsum('klca,klcb->ba', l2bb, t2bb) * .5
    mVV+= einsum('kLcA,kLcB->BA', l2ab, t2ab)
    moo = einsum('kicd,kjcd->ij', l2aa, t2aa) * .5
    moo+= einsum('iKdC,jKdC->ij', l2ab, t2ab)
    mOO = einsum('kicd,kjcd->ij', l2bb, t2bb) * .5
    mOO+= einsum('kIcD,kJcD->IJ', l2ab, t2ab)

    #m3 = lib.einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5
    m3aa, m3ab, m3bb = mycc._add_vvvv(None, (l2aa.conj(),l2ab.conj(),l2bb.conj()), eris)
    m3aa = m3aa.conj()
    m3ab = m3ab.conj()
    m3bb = m3bb.conj()
    m3aa += lib.einsum('klab,ikjl->ijab', l2aa, numpy.asarray(imds.woooo))
    m3bb += lib.einsum('klab,ikjl->ijab', l2bb, numpy.asarray(imds.wOOOO))
    m3ab += lib.einsum('kLaB,ikJL->iJaB', l2ab, numpy.asarray(imds.wooOO))

    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    OVOV = numpy.asarray(eris.OVOV)
    OVOV = OVOV - OVOV.transpose(0,3,2,1)
    ovOV = numpy.asarray(eris.ovOV)
    mvv1 = einsum('jc,jb->bc', l1a, t1a) + mvv
    mVV1 = einsum('jc,jb->bc', l1b, t1b) + mVV
    moo1 = einsum('ic,kc->ik', l1a, t1a) + moo
    mOO1 = einsum('ic,kc->ik', l1b, t1b) + mOO
    if nvira > 0 and nocca > 0:
        ovvv = numpy.asarray(eris.get_ovvv())
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        tmp = lib.einsum('ijcd,kd->ijck', l2aa, t1a)
        m3aa -= lib.einsum('kbca,ijck->ijab', ovvv, tmp)

        tmp = einsum('ic,jbca->jiba', l1a, ovvv)
        tmp+= einsum('kiab,jk->ijab', l2aa, v2a)
        tmp-= einsum('ik,kajb->ijab', moo1, ovov)
        u2aa += tmp - tmp.transpose(1,0,2,3)
        u1a += numpy.einsum('iacb,bc->ia', ovvv, mvv1)
        ovvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OVVV = numpy.asarray(eris.get_OVVV())
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        tmp = lib.einsum('ijcd,kd->ijck', l2bb, t1b)
        m3bb -= lib.einsum('kbca,ijck->ijab', OVVV, tmp)

        tmp = einsum('ic,jbca->jiba', l1b, OVVV)
        tmp+= einsum('kiab,jk->ijab', l2bb, v2b)
        tmp-= einsum('ik,kajb->ijab', mOO1, OVOV)
        u2bb += tmp - tmp.transpose(1,0,2,3)
        u1b += numpy.einsum('iaCB,BC->ia', OVVV, mVV1)
        OVVV = tmp = None

    if nvirb > 0 and nocca > 0:
        OVvv = numpy.asarray(eris.get_OVvv())
        tmp = lib.einsum('iJcD,KD->iJcK', l2ab, t1b)
        m3ab -= lib.einsum('KBca,iJcK->iJaB', OVvv, tmp)

        tmp = einsum('ic,JAcb->JibA', l1a, OVvv)
        tmp-= einsum('kIaB,jk->IjaB', l2ab, v2a)
        tmp-= einsum('IK,jaKB->IjaB', mOO1, ovOV)
        u2ab += tmp.transpose(1,0,2,3)
        u1b += numpy.einsum('iacb,bc->ia', OVvv, mvv1)
        OVvv = tmp = None

    if nvira > 0 and noccb > 0:
        ovVV = numpy.asarray(eris.get_ovVV())
        tmp = lib.einsum('iJdC,kd->iJCk', l2ab, t1a)
        m3ab -= lib.einsum('kaCB,iJCk->iJaB', ovVV, tmp)

        tmp = einsum('IC,jbCA->jIbA', l1b, ovVV)
        tmp-= einsum('iKaB,JK->iJaB', l2ab, v2b)
        tmp-= einsum('ik,kaJB->iJaB', moo1, ovOV)
        u2ab += tmp
        u1a += numpy.einsum('iaCB,BC->ia', ovVV, mVV1)
        ovVV = tmp = None

    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    tmp = lib.einsum('ijcd,klcd->ijkl', l2aa, tauaa)
    ovov = numpy.asarray(eris.ovov)
    ovov = ovov - ovov.transpose(0,3,2,1)
    m3aa += lib.einsum('kalb,ijkl->ijab', ovov, tmp) * .25

    tmp = lib.einsum('ijcd,klcd->ijkl', l2bb, taubb)
    OVOV = numpy.asarray(eris.OVOV)
    OVOV = OVOV - OVOV.transpose(0,3,2,1)
    m3bb += lib.einsum('kalb,ijkl->ijab', OVOV, tmp) * .25

    tmp = lib.einsum('iJcD,kLcD->iJkL', l2ab, tauab)
    ovOV = numpy.asarray(eris.ovOV)
    m3ab += lib.einsum('kaLB,iJkL->iJaB', ovOV, tmp) * .5
    tmp = lib.einsum('iJdC,lKdC->iJKl', l2ab, tauab)
    m3ab += lib.einsum('laKB,iJKl->iJaB', ovOV, tmp) * .5

    u1a += numpy.einsum('ijab,jb->ia', m3aa, t1a)
    u1a += numpy.einsum('iJaB,JB->ia', m3ab, t1b)
    u1b += numpy.einsum('IJAB,JB->IA', m3bb, t1b)
    u1b += numpy.einsum('jIbA,jb->IA', m3ab, t1a)

    u2aa += m3aa
    u2bb += m3bb
    u2ab += m3ab
    u2aa += ovov.transpose(0,2,1,3)
    u2bb += OVOV.transpose(0,2,1,3)
    u2ab += ovOV.transpose(0,2,1,3)

    fov1 = fova + numpy.einsum('kcjb,kc->jb', ovov, t1a)
    fov1+= numpy.einsum('jbKC,KC->jb', ovOV, t1b)
    tmp = numpy.einsum('ia,jb->ijab', l1a, fov1)
    tmp+= einsum('kica,jbck->ijab', l2aa, imds.wovvo)
    tmp+= einsum('iKaC,jbCK->ijab', l2ab, imds.wovVO)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2aa += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + numpy.einsum('kcjb,kc->jb', OVOV, t1b)
    fov1+= numpy.einsum('kcJB,kc->JB', ovOV, t1a)
    tmp = numpy.einsum('ia,jb->ijab', l1b, fov1)
    tmp+= einsum('kica,jbck->ijab', l2bb, imds.wOVVO)
    tmp+= einsum('kIcA,JBck->IJAB', l2ab, imds.wOVvo)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2bb += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + numpy.einsum('kcjb,kc->jb', OVOV, t1b)
    fov1+= numpy.einsum('kcJB,kc->JB', ovOV, t1a)
    u2ab += numpy.einsum('ia,JB->iJaB', l1a, fov1)
    u2ab += einsum('iKaC,JBCK->iJaB', l2ab, imds.wOVVO)
    u2ab += einsum('kica,JBck->iJaB', l2aa, imds.wOVvo)
    u2ab += einsum('kIaC,jBCk->jIaB', l2ab, imds.woVVo)
    u2ab += einsum('iKcA,JbcK->iJbA', l2ab, imds.wOvvO)
    fov1 = fova + numpy.einsum('kcjb,kc->jb', ovov, t1a)
    fov1+= numpy.einsum('jbKC,KC->jb', ovOV, t1b)
    u2ab += numpy.einsum('ia,jb->jiba', l1b, fov1)
    u2ab += einsum('kIcA,jbck->jIbA', l2ab, imds.wovvo)
    u2ab += einsum('KICA,jbCK->jIbA', l2bb, imds.wovVO)

    ovoo = numpy.asarray(eris.ovoo)
    ovoo = ovoo - ovoo.transpose(2,1,0,3)
    OVOO = numpy.asarray(eris.OVOO)
    OVOO = OVOO - OVOO.transpose(2,1,0,3)
    OVoo = numpy.asarray(eris.OVoo)
    ovOO = numpy.asarray(eris.ovOO)
    tmp = einsum('ka,jbik->ijab', l1a, ovoo)
    tmp+= einsum('ijca,cb->ijab', l2aa, v1a)
    tmp+= einsum('ca,icjb->ijab', mvv1, ovov)
    u2aa -= tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ka,jbik->ijab', l1b, OVOO)
    tmp+= einsum('ijca,cb->ijab', l2bb, v1b)
    tmp+= einsum('ca,icjb->ijab', mVV1, OVOV)
    u2bb -= tmp - tmp.transpose(0,1,3,2)
    u2ab -= einsum('ka,JBik->iJaB', l1a, OVoo)
    u2ab += einsum('iJaC,CB->iJaB', l2ab, v1b)
    u2ab -= einsum('ca,icJB->iJaB', mvv1, ovOV)
    u2ab -= einsum('KA,ibJK->iJbA', l1b, ovOO)
    u2ab += einsum('iJcA,cb->iJbA', l2ab, v1a)
    u2ab -= einsum('CA,ibJC->iJbA', mVV1, ovOV)

    u1a += fova
    u1b += fovb
    u1a += einsum('ib,ba->ia', l1a, v1a)
    u1a -= einsum('ja,ij->ia', l1a, v2a)
    u1b += einsum('ib,ba->ia', l1b, v1b)
    u1b -= einsum('ja,ij->ia', l1b, v2b)

    u1a += numpy.einsum('jb,iabj->ia', l1a, eris.ovvo)
    u1a -= numpy.einsum('jb,ijba->ia', l1a, eris.oovv)
    u1a += numpy.einsum('JB,iaBJ->ia', l1b, eris.ovVO)
    u1b += numpy.einsum('jb,iabj->ia', l1b, eris.OVVO)
    u1b -= numpy.einsum('jb,ijba->ia', l1b, eris.OOVV)
    u1b += numpy.einsum('jb,iabj->ia', l1a, eris.OVvo)

    u1a -= einsum('kjca,ijck->ia', l2aa, imds.woovo)
    u1a -= einsum('jKaC,ijCK->ia', l2ab, imds.wooVO)
    u1b -= einsum('kjca,ijck->ia', l2bb, imds.wOOVO)
    u1b -= einsum('kJcA,IJck->IA', l2ab, imds.wOOvo)

    u1a -= einsum('ikbc,back->ia', l2aa, imds.wvvvo)
    u1a -= einsum('iKbC,baCK->ia', l2ab, imds.wvvVO)
    u1b -= einsum('IKBC,BACK->IA', l2bb, imds.wVVVO)
    u1b -= einsum('kIcB,BAck->IA', l2ab, imds.wVVvo)

    u1a += numpy.einsum('jiba,bj->ia', l2aa, imds.w3a)
    u1a += numpy.einsum('iJaB,BJ->ia', l2ab, imds.w3b)
    u1b += numpy.einsum('JIBA,BJ->IA', l2bb, imds.w3b)
    u1b += numpy.einsum('jIbA,bj->IA', l2ab, imds.w3a)

    tmpa  = t1a + numpy.einsum('kc,kjcb->jb', l1a, t2aa)
    tmpa += numpy.einsum('KC,jKbC->jb', l1b, t2ab)
    tmpa -= einsum('bd,jd->jb', mvv1, t1a)
    tmpa -= einsum('lj,lb->jb', moo, t1a)
    tmpb  = t1b + numpy.einsum('kc,kjcb->jb', l1b, t2bb)
    tmpb += numpy.einsum('kc,kJcB->JB', l1a, t2ab)
    tmpb -= einsum('bd,jd->jb', mVV1, t1b)
    tmpb -= einsum('lj,lb->jb', mOO, t1b)
    u1a += numpy.einsum('jbia,jb->ia', ovov, tmpa)
    u1a += numpy.einsum('iaJB,JB->ia', ovOV, tmpb)
    u1b += numpy.einsum('jbia,jb->ia', OVOV, tmpb)
    u1b += numpy.einsum('jbIA,jb->IA', ovOV, tmpa)

    u1a -= numpy.einsum('iajk,kj->ia', ovoo, moo1)
    u1a -= numpy.einsum('iaJK,KJ->ia', ovOO, mOO1)
    u1b -= numpy.einsum('iajk,kj->ia', OVOO, mOO1)
    u1b -= numpy.einsum('IAjk,kj->IA', OVoo, moo1)

    tmp  = fova - numpy.einsum('kbja,jb->ka', ovov, t1a)
    tmp += numpy.einsum('kaJB,JB->ka', ovOV, t1b)
    u1a -= lib.einsum('ik,ka->ia', moo, tmp)
    u1a -= lib.einsum('ca,ic->ia', mvv, tmp)
    tmp  = fovb - numpy.einsum('kbja,jb->ka', OVOV, t1b)
    tmp += numpy.einsum('jbKA,jb->KA', ovOV, t1a)
    u1b -= lib.einsum('ik,ka->ia', mOO, tmp)
    u1b -= lib.einsum('ca,ic->ia', mVV, tmp)

    eia = lib.direct_sum('i-j->ij', mo_ea_o, mo_ea_v)
    eIA = lib.direct_sum('i-j->ij', mo_eb_o, mo_eb_v)
    u1a /= eia
    u1b /= eIA

    u2aa /= lib.direct_sum('ia+jb->ijab', eia, eia)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia, eIA)
    u2bb /= lib.direct_sum('ia+jb->ijab', eIA, eIA)

    time0 = log.timer_debug1('update l1 l2', *time0)
    return (u1a,u1b), (u2aa,u2ab,u2bb)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mycc = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
    eris = mycc.ao2mo()
    mycc.kernel()
    l1, l2 = mycc.solve_lambda(mycc.t1, mycc.t2, eris=eris)
    l1ref = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2ref = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)

    mycc = uccsd.UCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel()
    conv, l1, l2 = kernel(mycc, eris, mycc.t1, mycc.t2, tol=1e-8)
    print(abs(l1[0]-l1ref[0]).max())
    print(abs(l1[1]-l1ref[1]).max())
    print(abs(l2[0]-l2ref[0]).max())
    print(abs(l2[1]-l2ref[1]).max())
    print(abs(l2[2]-l2ref[2]).max())
