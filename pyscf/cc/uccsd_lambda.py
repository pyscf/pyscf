#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import uccsd

einsum = lib.einsum

def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2

    imds = make_intermediates(mycc, t1, t2, eris)

    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput0 = log.timer('UCCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(mycc, t1, t2, l1, l2, eris, imds)
        normt = numpy.linalg.norm(mycc.amplitudes_to_vector(l1new, l2new) -
                                  mycc.amplitudes_to_vector(l1, l2))
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if mycc.diis:
            l1, l2 = mycc.diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('UCCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
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

    ovov1 = numpy.asarray(eris.ovov)
    ovov1 = ovov1 - ovov1.transpose(0,3,2,1)
    OVOV1 = numpy.asarray(eris.OVOV)
    OVOV1 = OVOV1 - OVOV1.transpose(0,3,2,1)
    ovOV1 = numpy.asarray(eris.ovOV)

    v1a  = fvva - einsum('ja,jb->ba', fova, t1a)
    v1b  = fvvb - einsum('ja,jb->ba', fovb, t1b)
    v1a += einsum('jcka,jkbc->ba', ovov1, tauaa) * .5
    v1a -= einsum('jaKC,jKbC->ba', ovOV1, tauab) * .5
    v1a -= einsum('kaJC,kJbC->ba', ovOV1, tauab) * .5
    v1b += einsum('jcka,jkbc->ba', OVOV1, taubb) * .5
    v1b -= einsum('kcJA,kJcB->BA', ovOV1, tauab) * .5
    v1b -= einsum('jcKA,jKcB->BA', ovOV1, tauab) * .5

    v2a  = fooa + einsum('ib,jb->ij', fova, t1a)
    v2b  = foob + einsum('ib,jb->ij', fovb, t1b)
    v2a += einsum('ibkc,jkbc->ij', ovov1, tauaa) * .5
    v2a += einsum('ibKC,jKbC->ij', ovOV1, tauab)
    v2b += einsum('ibkc,jkbc->ij', OVOV1, taubb) * .5
    v2b += einsum('kcIB,kJcB->IJ', ovOV1, tauab)

    ovoo = numpy.asarray(eris.ovoo)
    ovoo = ovoo - ovoo.transpose(2,1,0,3)
    OVOO = numpy.asarray(eris.OVOO)
    OVOO = OVOO - OVOO.transpose(2,1,0,3)
    OVoo = numpy.asarray(eris.OVoo)
    ovOO = numpy.asarray(eris.ovOO)
    v2a -= einsum('ibkj,kb->ij', ovoo, t1a)
    v2a += einsum('KBij,KB->ij', OVoo, t1b)
    v2b -= einsum('ibkj,kb->ij', OVOO, t1b)
    v2b += einsum('kbIJ,kb->IJ', ovOO, t1a)

    v5a  = fvoa + numpy.einsum('kc,jkbc->bj', fova, t2aa)
    v5a += numpy.einsum('KC,jKbC->bj', fovb, t2ab)
    v5b  = fvob + numpy.einsum('kc,jkbc->bj', fovb, t2bb)
    v5b += numpy.einsum('kc,kJcB->BJ', fova, t2ab)
    tmp  = fova - numpy.einsum('kdlc,ld->kc', ovov1, t1a)
    tmp += numpy.einsum('kcLD,LD->kc', ovOV1, t1b)
    v5a += numpy.einsum('kc,kb,jc->bj', tmp, t1a, t1a)
    tmp  = fovb - numpy.einsum('kdlc,ld->kc', OVOV1, t1b)
    tmp += numpy.einsum('ldKC,ld->KC', ovOV1, t1a)
    v5b += numpy.einsum('kc,kb,jc->bj', tmp, t1b, t1b)
    v5a -= einsum('lckj,klbc->bj', ovoo, t2aa) * .5
    v5a -= einsum('LCkj,kLbC->bj', OVoo, t2ab)
    v5b -= einsum('LCKJ,KLBC->BJ', OVOO, t2bb) * .5
    v5b -= einsum('lcKJ,lKcB->BJ', ovOO, t2ab)

    oooo = numpy.asarray(eris.oooo)
    OOOO = numpy.asarray(eris.OOOO)
    ooOO = numpy.asarray(eris.ooOO)
    woooo1  = einsum('icjl,kc->ikjl', ovoo, t1a)
    wOOOO1  = einsum('icjl,kc->ikjl', OVOO, t1b)
    wooOO1  = einsum('icJL,kc->ikJL', ovOO, t1a)
    wooOO1 += einsum('JCil,KC->ilJK', OVoo, t1b)
    woooo1 += (oooo - oooo.transpose(0,3,2,1)) * .5
    wOOOO1 += (OOOO - OOOO.transpose(0,3,2,1)) * .5
    wooOO1 += ooOO.copy()
    woooo1 += einsum('icjd,klcd->ikjl', ovov1, tauaa) * .25
    wOOOO1 += einsum('icjd,klcd->ikjl', OVOV1, taubb) * .25
    wooOO1 += einsum('icJD,kLcD->ikJL', ovOV1, tauab)

    v4ovvo  = einsum('jbld,klcd->jbck', ovov1, t2aa)
    v4ovvo += einsum('jbLD,kLcD->jbck', ovOV1, t2ab)
    v4ovvo += numpy.asarray(eris.ovvo)
    v4ovvo -= numpy.asarray(eris.oovv).transpose(0,3,2,1)
    v4OVVO  = einsum('jbld,klcd->jbck', OVOV1, t2bb)
    v4OVVO += einsum('ldJB,lKdC->JBCK', ovOV1, t2ab)
    v4OVVO += numpy.asarray(eris.OVVO)
    v4OVVO -= numpy.asarray(eris.OOVV).transpose(0,3,2,1)
    v4OVvo  = einsum('ldJB,klcd->JBck', ovOV1, t2aa)
    v4OVvo += einsum('JBLD,kLcD->JBck', OVOV1, t2ab)
    v4OVvo += numpy.asarray(eris.OVvo)
    v4ovVO  = einsum('jbLD,KLCD->jbCK', ovOV1, t2bb)
    v4ovVO += einsum('jbld,lKdC->jbCK', ovov1, t2ab)
    v4ovVO += numpy.asarray(eris.ovVO)
    v4oVVo  = einsum('jdLB,kLdC->jBCk', ovOV1, t2ab)
    v4oVVo -= numpy.asarray(eris.ooVV).transpose(0,3,2,1)
    v4OvvO  = einsum('lbJD,lKcD->JbcK', ovOV1, t2ab)
    v4OvvO -= numpy.asarray(eris.OOvv).transpose(0,3,2,1)

    wovvo1  = numpy.einsum('jbld,kd,lc->jbck', ovov1, t1a, -t1a)
    wOVVO1  = numpy.einsum('jbld,kd,lc->jbck', OVOV1, t1b, -t1b)
    wovVO1  = numpy.einsum('jbLD,KD,LC->jbCK', ovOV1, t1b, -t1b)
    wOVvo1  = numpy.einsum('ldJB,kd,lc->JBck', ovOV1, t1a, -t1a)
    woVVo1  = numpy.einsum('jdLB,kd,LC->jBCk', ovOV1, t1a,  t1b)
    wOvvO1  = numpy.einsum('lbJD,KD,lc->JbcK', ovOV1, t1b,  t1a)
    wovvo1 += v4ovvo
    wOVVO1 += v4OVVO
    wovVO1 += v4ovVO
    wOVvo1 += v4OVvo
    woVVo1 += v4oVVo
    wOvvO1 += v4OvvO
    wovvo1 -= einsum('jblk,lc->jbck', ovoo, t1a)
    wOVVO1 -= einsum('jblk,lc->jbck', OVOO, t1b)
    wovVO1 -= einsum('jbLK,LC->jbCK', ovOO, t1b)
    wOVvo1 -= einsum('JBlk,lc->JBck', OVoo, t1a)
    woVVo1 += einsum('LBjk,LC->jBCk', OVoo, t1b)
    wOvvO1 += einsum('lbJK,lc->JbcK', ovOO, t1a)

    wovoo1  = einsum('ibck,jb->kcji', v4ovvo, t1a)
    wOVOO1  = einsum('ibck,jb->kcji', v4OVVO, t1b)
    wovOO1  = einsum('IBck,JB->kcJI', v4OVvo, t1b)
    wovOO1 -= einsum('IbcK,jb->jcKI', v4OvvO, t1a)
    wOVoo1  = einsum('ibCK,jb->KCji', v4ovVO, t1a)
    wOVoo1 -= einsum('iBCk,JB->JCki', v4oVVo, t1b)
    wovoo1 += ovoo.conj() * .5
    wOVOO1 += OVOO.conj() * .5
    wOVoo1 += OVoo.conj()
    wovOO1 += ovOO.conj()

    wovvv1  = einsum('jack,jb->kcab', v4ovvo, t1a)
    wOVVV1  = einsum('jack,jb->kcab', v4OVVO, t1b)
    wovVV1  = einsum('JAck,JB->kcAB', v4OVvo, t1b)
    wovVV1 -= einsum('jACk,jb->kbAC', v4oVVo, t1a)
    wOVvv1  = einsum('jaCK,jb->KCab', v4ovVO, t1a)
    wOVvv1 -= einsum('JacK,JB->KBac', v4OvvO, t1b)
    wovvv1 += einsum('lajk,jlbc->kcab', ovoo, tauaa) * .25
    wOVVV1 += einsum('lajk,jlbc->kcab', OVOO, taubb) * .25
    wovVV1 -= einsum('LAjk,jLcB->kcAB', OVoo, tauab)
    wOVvv1 -= einsum('laJK,lJbC->KCab', ovOO, tauab)

    if nvira > 0 and nocca > 0:
        ovvv = numpy.asarray(eris.ovvv).reshape(nocca*nvira,-1)
        ovvv = lib.unpack_tril(ovvv).reshape(nocca,nvira,nvira,nvira)
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        v1a -= einsum('jabc,jc->ba', ovvv, t1a)
        v5a += einsum('kdbc,jkcd->bj', ovvv, t2aa) * .5
        wovoo1 += einsum('idcb,kjbd->kcji', ovvv, tauaa) * .25
        wovvo1 += einsum('jbcd,kd->jbck', ovvv, t1a)
        wovvv1 -= ovvv.conj() * .5
        ovvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OVVV = numpy.asarray(eris.OVVV).reshape(noccb*nvirb,-1)
        OVVV = lib.unpack_tril(OVVV).reshape(noccb,nvirb,nvirb,nvirb)
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        v1b -= einsum('jabc,jc->ba', OVVV, t1b)
        v5b += einsum('KDBC,JKCD->BJ', OVVV, t2bb) * .5
        wOVOO1 += einsum('idcb,kjbd->kcji', OVVV, taubb) * .25
        wOVVO1 += einsum('jbcd,kd->jbck', OVVV, t1b)
        wOVVV1 -= OVVV.conj() * .5
        OVVV = tmp = None

    if nvirb > 0 and noccb > 0:
        OVvv = numpy.asarray(eris.OVvv).reshape(noccb*nvirb,-1)
        OVvv = lib.unpack_tril(OVvv).reshape(noccb,nvirb,nvira,nvira)
        v1a += einsum('JCba,JC->ba', OVvv, t1b)
        v5a += einsum('KDbc,jKcD->bj', OVvv, t2ab)
        wovOO1 += einsum('IDcb,kJbD->kcJI', OVvv, tauab)
        wOVvo1 += einsum('JBcd,kd->JBck', OVvv, t1a)
        wOvvO1 -= einsum('JDcb,KD->JbcK', OVvv, t1b)
        wOVvv1 -= OVvv.conj()
        OVvv = tmp = None

    if nvira > 0 and nocca > 0:
        ovVV = numpy.asarray(eris.ovVV).reshape(nocca*nvira,-1)
        ovVV = lib.unpack_tril(ovVV).reshape(nocca,nvira,nvirb,nvirb)
        v1b += einsum('jcBA,jc->BA', ovVV, t1a)
        v5b += einsum('kdBC,kJdC->BJ', ovVV, t2ab)
        wOVoo1 += einsum('idCB,jKdB->KCji', ovVV, tauab)
        wovVO1 += einsum('jbCD,KD->jbCK', ovVV, t1b)
        woVVo1 -= einsum('jdCB,kd->jBCk', ovVV, t1a)
        wovVV1 -= ovVV.conj()
        ovVV = tmp = None

    w3a  = v5a + numpy.einsum('jbck,jb->ck', v4ovvo, t1a)
    w3a += numpy.einsum('JBck,JB->ck', v4OVvo, t1b)
    w3b  = v5b + numpy.einsum('jbck,jb->ck', v4OVVO, t1b)
    w3b += numpy.einsum('jbCK,jb->CK', v4ovVO, t1a)
    w3a += numpy.einsum('cb,jb->cj', v1a, t1a)
    w3b += numpy.einsum('cb,jb->cj', v1b, t1b)
    w3a -= numpy.einsum('jk,jb->bk', v2a, t1a)
    w3b -= numpy.einsum('jk,jb->bk', v2b, t1b)

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    imds.woooo1 = imds.ftmp.create_dataset('woooo1', (nocca,nocca,nocca,nocca), 'f8')
    imds.wooOO1 = imds.ftmp.create_dataset('wooOO1', (nocca,nocca,noccb,noccb), 'f8')
    imds.wOOOO1 = imds.ftmp.create_dataset('wOOOO1', (noccb,noccb,noccb,noccb), 'f8')
    imds.wovvo1 = imds.ftmp.create_dataset('wovvo1', (nocca,nvira,nvira,nocca), 'f8')
    imds.wOVVO1 = imds.ftmp.create_dataset('wOVVO1', (noccb,nvirb,nvirb,noccb), 'f8')
    imds.wovVO1 = imds.ftmp.create_dataset('wovVO1', (nocca,nvira,nvirb,noccb), 'f8')
    imds.wOVvo1 = imds.ftmp.create_dataset('wOVvo1', (noccb,nvirb,nvira,nocca), 'f8')
    imds.woVVo1 = imds.ftmp.create_dataset('woVVo1', (nocca,nvirb,nvirb,nocca), 'f8')
    imds.wOvvO1 = imds.ftmp.create_dataset('wOvvO1', (noccb,nvira,nvira,noccb), 'f8')
    imds.wovoo1 = imds.ftmp.create_dataset('wovoo1', (nocca,nvira,nocca,nocca), 'f8')
    imds.wOVOO1 = imds.ftmp.create_dataset('wOVOO1', (noccb,nvirb,noccb,noccb), 'f8')
    imds.wovOO1 = imds.ftmp.create_dataset('wovOO1', (nocca,nvira,noccb,noccb), 'f8')
    imds.wOVoo1 = imds.ftmp.create_dataset('wOVoo1', (noccb,nvirb,nocca,nocca), 'f8')
    imds.wovvv1 = imds.ftmp.create_dataset('wovvv1', (nocca,nvira,nvira,nvira), 'f8')
    imds.wOVVV1 = imds.ftmp.create_dataset('wOVVV1', (noccb,nvirb,nvirb,nvirb), 'f8')
    imds.wovVV1 = imds.ftmp.create_dataset('wovVV1', (nocca,nvira,nvirb,nvirb), 'f8')
    imds.wOVvv1 = imds.ftmp.create_dataset('wOVvv1', (noccb,nvirb,nvira,nvira), 'f8')

    imds.woooo1[:] = woooo1
    imds.wOOOO1[:] = wOOOO1
    imds.wooOO1[:] = wooOO1
    imds.wovvo1[:] = wovvo1
    imds.wOVVO1[:] = wOVVO1
    imds.wovVO1[:] = wovVO1
    imds.wOVvo1[:] = wOVvo1
    imds.woVVo1[:] = woVVo1
    imds.wOvvO1[:] = wOvvO1
    imds.wovoo1[:] = wovoo1
    imds.wOVOO1[:] = wOVOO1
    imds.wovOO1[:] = wovOO1
    imds.wOVoo1[:] = wOVoo1
    imds.wovvv1[:] = wovvv1
    imds.wOVVV1[:] = wOVVV1
    imds.wovVV1[:] = wovVV1
    imds.wOVvv1[:] = wOVvv1
    imds.v1a = v1a
    imds.v1b = v1b
    imds.v2a = v2a
    imds.v2b = v2b
    imds.w3a = w3a
    imds.w3b = w3b
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_amps(mycc, t1, t2, l1, l2, eris, imds):
    time1 = time0 = time.clock(), time.time()
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

    fooa = eris.focka[:nocca,:nocca]
    fova = eris.focka[:nocca,nocca:]
    fvva = eris.focka[nocca:,nocca:]
    foob = eris.fockb[:noccb,:noccb]
    fovb = eris.fockb[:noccb,noccb:]
    fvvb = eris.fockb[noccb:,noccb:]

    mvv = einsum('klca,klcb->ba', l2aa, t2aa) * .5
    mvv+= einsum('lKaC,lKbC->ba', l2ab, t2ab)
    mVV = einsum('klca,klcb->ba', l2bb, t2bb) * .5
    mVV+= einsum('kLcA,kLcB->BA', l2ab, t2ab)
    moo = einsum('kicd,kjcd->ij', l2aa, t2aa) * .5
    moo+= einsum('iKdC,jKdC->ij', l2ab, t2ab)
    mOO = einsum('kicd,kjcd->ij', l2bb, t2bb) * .5
    mOO+= einsum('kIcD,kJcD->IJ', l2ab, t2ab)

    #m3 = numpy.einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5
    m3aa, m3ab, m3bb = mycc._add_vvvv(None, (l2aa,l2ab,l2bb), eris)
    m3aa += numpy.einsum('klab,ikjl->ijab', l2aa, numpy.asarray(imds.woooo1))
    m3bb += numpy.einsum('klab,ikjl->ijab', l2bb, numpy.asarray(imds.wOOOO1))
    m3ab += numpy.einsum('kLaB,ikJL->iJaB', l2ab, numpy.asarray(imds.wooOO1))

    ovov1 = numpy.asarray(eris.ovov)
    ovov1 = ovov1 - ovov1.transpose(0,3,2,1)
    OVOV1 = numpy.asarray(eris.OVOV)
    OVOV1 = OVOV1 - OVOV1.transpose(0,3,2,1)
    ovOV1 = numpy.asarray(eris.ovOV)
    mvv1 = numpy.einsum('jc,jb->bc', l1a, t1a) + mvv
    mVV1 = numpy.einsum('jc,jb->bc', l1b, t1b) + mVV
    moo1 = moo + einsum('ic,kc->ik', l1a, t1a)
    mOO1 = mOO + einsum('ic,kc->ik', l1b, t1b)
    if nvira > 0 and nocca > 0:
        ovvv = numpy.asarray(eris.ovvv).reshape(nocca*nvira,-1)
        ovvv = lib.unpack_tril(ovvv).reshape(nocca,nvira,nvira,nvira)
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        tmp = numpy.einsum('ijcd,kd->ijck', l2aa, t1a)
        m3aa -= numpy.einsum('kbca,ijck->ijab', ovvv, tmp)

        tmp = einsum('ic,jbca->jiba', l1a, ovvv)
        tmp+= einsum('kiab,jk->ijab', l2aa, imds.v2a)
        tmp-= einsum('ik,kajb->ijab', moo1, ovov1)
        u2aa += tmp - tmp.transpose(1,0,2,3)

        u1a += numpy.einsum('iacb,bc->ia', ovvv, mvv1)
        tmp  = einsum('kacd,jkbd->jacb', ovvv, t2aa)
        u1a -= einsum('ijcb,jacb->ia', l2aa, tmp)
        tmp  = einsum('kacd,kJdB->JacB', ovvv, t2ab)
        u1a -= einsum('iJcB,JacB->ia', l2ab, tmp)
        ovvv = tmp = None

    if nvirb > 0 and noccb > 0:
        OVVV = numpy.asarray(eris.OVVV).reshape(noccb*nvirb,-1)
        OVVV = lib.unpack_tril(OVVV).reshape(noccb,nvirb,nvirb,nvirb)
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        tmp = numpy.einsum('ijcd,kd->ijck', l2bb, t1b)
        m3bb -= numpy.einsum('kbca,ijck->ijab', OVVV, tmp)

        tmp = einsum('ic,jbca->jiba', l1b, OVVV)
        tmp+= einsum('kiab,jk->ijab', l2bb, imds.v2b)
        tmp-= einsum('ik,kajb->ijab', mOO1, OVOV1)
        u2bb += tmp - tmp.transpose(1,0,2,3)

        u1b += numpy.einsum('iaCB,BC->ia', OVVV, mVV1)
        tmp  = einsum('KACD,JKBD->JACB', OVVV, t2bb)
        u1b -= einsum('IJCB,JACB->IA', l2bb, tmp)
        tmp  = einsum('KACD,jKbD->jACb', OVVV, t2ab)
        u1b -= einsum('jIbC,jACb->IA', l2ab, tmp)
        OVVV = tmp = None

    if nvirb > 0 and noccb > 0:
        OVvv = numpy.asarray(eris.OVvv).reshape(noccb*nvirb,-1)
        OVvv = lib.unpack_tril(OVvv).reshape(noccb,nvirb,nvira,nvira)
        tmp = numpy.einsum('iJcD,KD->iJcK', l2ab, t1b)
        m3ab -= numpy.einsum('KBca,iJcK->iJaB', OVvv, tmp)

        tmp = einsum('ic,JAcb->JibA', l1a, OVvv)
        tmp-= einsum('kIaB,jk->IjaB', l2ab, imds.v2a)
        tmp-= einsum('IK,jaKB->IjaB', mOO1, ovOV1)
        u2ab += tmp.transpose(1,0,2,3)

        tmp  = einsum('KDca,jKbD->jacb', OVvv, t2ab)
        u1a += einsum('ijcb,jacb->ia', l2aa, tmp)
        tmp  = einsum('KDca,JKBD->JacB', OVvv, t2bb)
        u1a += einsum('iJcB,JacB->ia', l2ab, tmp)

        u1b += numpy.einsum('iacb,bc->ia', OVvv, mvv1)
        tmp  = einsum('KAcd,jKdB->jAcB', OVvv, t2ab)
        u1b -= einsum('jIcB,jAcB->IA', l2ab, tmp)
        OVvv = tmp = None

    if nvira > 0 and nocca > 0:
        ovVV = numpy.asarray(eris.ovVV).reshape(nocca*nvira,-1)
        ovVV = lib.unpack_tril(ovVV).reshape(nocca,nvira,nvirb,nvirb)
        tmp = numpy.einsum('iJdC,kd->iJCk', l2ab, t1a)
        m3ab -= numpy.einsum('kaCB,iJCk->iJaB', ovVV, tmp)

        tmp = einsum('IC,jbCA->jIbA', l1b, ovVV)
        tmp-= einsum('iKaB,JK->iJaB', l2ab, imds.v2b)
        tmp-= einsum('ik,kaJB->iJaB', moo1, ovOV1)
        u2ab += tmp

        u1a += numpy.einsum('iaCB,BC->ia', ovVV, mVV1)
        tmp  = einsum('kaCD,kJbD->JaCb', ovVV, t2ab)
        u1a -= einsum('iJbC,JaCb->ia', l2ab, tmp)

        tmp  = einsum('kdCA,kJdB->JACB', ovVV, t2ab)
        u1b += einsum('IJCB,JACB->IA', l2bb, tmp)
        tmp  = einsum('kdCA,jkbd->jACb', ovVV, t2aa)
        u1b += einsum('jIbC,jACb->IA', l2ab, tmp)
        ovVV = tmp = None

    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    tmp = numpy.einsum('ijcd,klcd->ijkl', l2aa, tauaa)
    ovov1 = numpy.asarray(eris.ovov)
    ovov1 = ovov1 - ovov1.transpose(0,3,2,1)
    m3aa += numpy.einsum('kalb,ijkl->ijab', ovov1, tmp) * .25

    tmp = numpy.einsum('ijcd,klcd->ijkl', l2bb, taubb)
    OVOV1 = numpy.asarray(eris.OVOV)
    OVOV1 = OVOV1 - OVOV1.transpose(0,3,2,1)
    m3bb += numpy.einsum('kalb,ijkl->ijab', OVOV1, tmp) * .25

    tmp = numpy.einsum('iJcD,kLcD->iJkL', l2ab, tauab)
    ovOV1 = numpy.asarray(eris.ovOV)
    m3ab += numpy.einsum('kaLB,iJkL->iJaB', ovOV1, tmp) * .5
    tmp = numpy.einsum('iJdC,lKdC->iJKl', l2ab, tauab)
    m3ab += numpy.einsum('laKB,iJKl->iJaB', ovOV1, tmp) * .5

    u1a += numpy.einsum('ijab,jb->ia', m3aa, t1a)
    u1a += numpy.einsum('iJaB,JB->ia', m3ab, t1b)
    u1b += numpy.einsum('IJAB,JB->IA', m3bb, t1b)
    u1b += numpy.einsum('jIbA,jb->IA', m3ab, t1a)

    u2aa += m3aa
    u2bb += m3bb
    u2ab += m3ab
    u2aa += ovov1.transpose(0,2,1,3)
    u2bb += OVOV1.transpose(0,2,1,3)
    u2ab += ovOV1.transpose(0,2,1,3)

    fov1 = fova + einsum('kcjb,kc->jb', ovov1, t1a)
    fov1+= einsum('jbKC,KC->jb', ovOV1, t1b)
    tmp = einsum('ia,jb->ijab', l1a, fov1)
    tmp+= einsum('kica,jbck->ijab', l2aa, imds.wovvo1)
    tmp+= einsum('iKaC,jbCK->ijab', l2ab, imds.wovVO1)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2aa += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + einsum('kcjb,kc->jb', OVOV1, t1b)
    fov1+= einsum('kcJB,kc->JB', ovOV1, t1a)
    tmp = einsum('ia,jb->ijab', l1b, fov1)
    tmp+= einsum('kica,jbck->ijab', l2bb, imds.wOVVO1)
    tmp+= einsum('kIcA,JBck->IJAB', l2ab, imds.wOVvo1)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2bb += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + einsum('kcjb,kc->jb', OVOV1, t1b)
    fov1+= einsum('kcJB,kc->JB', ovOV1, t1a)
    u2ab += einsum('ia,JB->iJaB', l1a, fov1)
    u2ab += einsum('iKaC,JBCK->iJaB', l2ab, imds.wOVVO1)
    u2ab += einsum('kica,JBck->iJaB', l2aa, imds.wOVvo1)
    u2ab += einsum('kIaC,jBCk->jIaB', l2ab, imds.woVVo1)
    u2ab += einsum('iKcA,JbcK->iJbA', l2ab, imds.wOvvO1)
    fov1 = fova + einsum('kcjb,kc->jb', ovov1, t1a)
    fov1+= einsum('jbKC,KC->jb', ovOV1, t1b)
    u2ab += einsum('ia,jb->jiba', l1b, fov1)
    u2ab += einsum('kIcA,jbck->jIbA', l2ab, imds.wovvo1)
    u2ab += einsum('KICA,jbCK->jIbA', l2bb, imds.wovVO1)

    ovoo = numpy.asarray(eris.ovoo)
    ovoo = ovoo - ovoo.transpose(2,1,0,3)
    OVOO = numpy.asarray(eris.OVOO)
    OVOO = OVOO - OVOO.transpose(2,1,0,3)
    OVoo = numpy.asarray(eris.OVoo)
    ovOO = numpy.asarray(eris.ovOO)
    tmp = einsum('ka,jbik->ijab', l1a, ovoo)
    tmp+= einsum('ijca,cb->ijab', l2aa, imds.v1a)
    tmp+= einsum('ca,icjb->ijab', mvv1, ovov1)
    u2aa -= tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ka,jbik->ijab', l1b, OVOO)
    tmp+= einsum('ijca,cb->ijab', l2bb, imds.v1b)
    tmp+= einsum('ca,icjb->ijab', mVV1, OVOV1)
    u2bb -= tmp - tmp.transpose(0,1,3,2)
    u2ab -= einsum('ka,JBik->iJaB', l1a, eris.OVoo)
    u2ab += einsum('iJaC,CB->iJaB', l2ab, imds.v1b)
    u2ab -= einsum('ca,icJB->iJaB', mvv1, ovOV1)
    u2ab -= einsum('KA,ibJK->iJbA', l1b, eris.ovOO)
    u2ab += einsum('iJcA,cb->iJbA', l2ab, imds.v1a)
    u2ab -= einsum('CA,ibJC->iJbA', mVV1, ovOV1)

    u1a += fova
    u1b += fovb
    u1a += einsum('ib,ba->ia', l1a, imds.v1a)
    u1a -= einsum('ja,ij->ia', l1a, imds.v2a)
    u1b += einsum('ib,ba->ia', l1b, imds.v1b)
    u1b -= einsum('ja,ij->ia', l1b, imds.v2b)

    u1a += numpy.einsum('jb,iabj->ia', l1a, eris.ovvo)
    u1a -= numpy.einsum('jb,ijba->ia', l1a, eris.oovv)
    u1a += numpy.einsum('JB,iaBJ->ia', l1b, eris.ovVO)
    u1b += numpy.einsum('jb,iabj->ia', l1b, eris.OVVO)
    u1b -= numpy.einsum('jb,ijba->ia', l1b, eris.OOVV)
    u1b += numpy.einsum('jb,iabj->ia', l1a, eris.OVvo)

    u1a -= einsum('kjca,kcji->ia', l2aa, imds.wovoo1)
    u1a -= einsum('jKaC,KCji->ia', l2ab, imds.wOVoo1)
    u1b -= einsum('kjca,kcji->ia', l2bb, imds.wOVOO1)
    u1b -= einsum('kJcA,kcJI->IA', l2ab, imds.wovOO1)

    u1a -= einsum('ikbc,kcab->ia', l2aa, imds.wovvv1)
    u1a -= einsum('iKbC,KCab->ia', l2ab, imds.wOVvv1)
    u1b -= einsum('IKBC,KCAB->IA', l2bb, imds.wOVVV1)
    u1b -= einsum('kIcB,kcAB->IA', l2ab, imds.wovVV1)

    u1a += numpy.einsum('jiba,bj->ia', l2aa, imds.w3a)
    u1a += numpy.einsum('iJaB,BJ->ia', l2ab, imds.w3b)
    u1b += numpy.einsum('JIBA,BJ->IA', l2bb, imds.w3b)
    u1b += numpy.einsum('jIbA,bj->IA', l2ab, imds.w3a)

    tmpa  = t1a + einsum('kc,kjcb->jb', l1a, t2aa)
    tmpa += einsum('KC,jKbC->jb', l1b, t2ab)
    tmpa -= einsum('bd,jd->jb', mvv1, t1a)
    tmpa -= einsum('lj,lb->jb', moo, t1a)
    tmpb  = t1b + einsum('kc,kjcb->jb', l1b, t2bb)
    tmpb += einsum('kc,kJcB->JB', l1a, t2ab)
    tmpb -= einsum('bd,jd->jb', mVV1, t1b)
    tmpb -= einsum('lj,lb->jb', mOO, t1b)
    u1a += numpy.einsum('jbia,jb->ia', ovov1, tmpa)
    u1a += numpy.einsum('iaJB,JB->ia', ovOV1, tmpb)
    u1b += numpy.einsum('jbia,jb->ia', OVOV1, tmpb)
    u1b += numpy.einsum('jbIA,jb->IA', ovOV1, tmpa)

    u1a -= numpy.einsum('iajk,kj->ia', ovoo, moo1)
    u1a -= numpy.einsum('iaJK,KJ->ia', ovOO, mOO1)
    u1b -= numpy.einsum('iajk,kj->ia', OVOO, mOO1)
    u1b -= numpy.einsum('IAjk,kj->IA', OVoo, moo1)

    tmp  = fova - einsum('kbja,jb->ka', ovov1, t1a)
    tmp += einsum('kaJB,JB->ka', ovOV1, t1b)
    u1a -= numpy.einsum('ik,ka->ia', moo, tmp)
    u1a -= numpy.einsum('ca,ic->ia', mvv, tmp)
    tmp  = fovb - einsum('kbja,jb->ka', OVOV1, t1b)
    tmp += einsum('jbKA,jb->KA', ovOV1, t1a)
    u1b -= numpy.einsum('ik,ka->ia', mOO, tmp)
    u1b -= numpy.einsum('ca,ic->ia', mVV, tmp)

    tmp  = einsum('iclk,jlbc->jkib', ovoo, t2aa)
    tmp -= einsum('LCik,jLbC->jkib', OVoo, t2ab)
    u1a += einsum('kjab,jkib->ia', l2aa, tmp)
    tmp  = einsum('iclk,lJcB->JkiB', ovoo, t2ab)
    tmp -= einsum('LCik,JLBC->JkiB', OVoo, t2bb)
    u1a += einsum('kJaB,JkiB->ia', l2ab, tmp)
    tmp  = einsum('icLK,jLcB->jKiB', ovOO, t2ab)
    u1a += einsum('jKaB,jKiB->ia', l2ab, tmp)
    tmp  = einsum('iclk,jlbc->jkib', OVOO, t2bb)
    tmp -= einsum('lcIK,lJcB->JKIB', ovOO, t2ab)
    u1b += einsum('kjab,jkib->ia', l2bb, tmp)
    tmp  = einsum('ICLK,jLbC->jKIb', OVOO, t2ab)
    tmp -= einsum('lcIK,jlbc->jKIb', ovOO, t2aa)
    u1b += einsum('jKbA,jKIb->IA', l2ab, tmp)
    tmp  = einsum('IClk,lJbC->JkIb', OVoo, t2ab)
    u1b += einsum('kJbA,JkIb->IA', l2ab, tmp)

    eia = lib.direct_sum('i-j->ij', fooa.diagonal(), fvva.diagonal())
    eIA = lib.direct_sum('i-j->ij', foob.diagonal(), fvvb.diagonal())
    u1a /= eia
    u1a += l1a
    u1b /= eIA
    u1b += l1b

    u2aa /= lib.direct_sum('ia+jb->ijab', eia, eia)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia, eIA)
    u2bb /= lib.direct_sum('ia+jb->ijab', eIA, eIA)
    u2aa += l2aa
    u2ab += l2ab
    u2bb += l2bb

    time0 = log.timer_debug1('update l1 l2', *time0)
    return (u1a,u1b), (u2aa,u2ab,u2bb)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import gccsd
    from pyscf.cc import gccsd_lambda

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    #mf.__dict__.update(scf.chkfile.load('h2o1.chk', 'scf'))
    mycc = uccsd.UCCSD(mf)
    eris = mycc.ao2mo()
    nocca, noccb = 6,4
    nmo = mol.nao_nr()
    nvira,nvirb = nmo-nocca, nmo-noccb
    numpy.random.seed(9)
    t1 = (numpy.random.random((nocca,nvira))-.9,
          numpy.random.random((noccb,nvirb))-.9)
    l1 = (numpy.random.random((nocca,nvira))-.9,
          numpy.random.random((noccb,nvirb))-.9)
    t2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
          numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
          numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
    t2[0] = t2[0] - t2[0].transpose(1,0,2,3)
    t2[0] = t2[0] - t2[0].transpose(0,1,3,2)
    t2[2] = t2[2] - t2[2].transpose(1,0,2,3)
    t2[2] = t2[2] - t2[2].transpose(0,1,3,2)
    l2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
          numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
          numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
    l2[0] = l2[0] - l2[0].transpose(1,0,2,3)
    l2[0] = l2[0] - l2[0].transpose(0,1,3,2)
    l2[2] = l2[2] - l2[2].transpose(1,0,2,3)
    l2[2] = l2[2] - l2[2].transpose(0,1,3,2)

    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = update_amps(mycc, t1, t2, l1, l2, eris, imds)
    print(lib.finger(l1[0]) --104.55975252585894)
    print(lib.finger(l1[1]) --241.12677819375281)
    print(lib.finger(l2[0]) --0.4957533529669417)
    print(lib.finger(l2[1]) - 15.46423057451851 )
    print(lib.finger(l2[2]) - 5.8430776663704407)

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
    conv, l1, l2 = gccsd_lambda.kernel(mycc, eris, mycc.t1, mycc.t2, tol=1e-8)
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
