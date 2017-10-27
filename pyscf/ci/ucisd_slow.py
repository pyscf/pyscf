#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Unrestricted CISD
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ci import cisd
from pyscf.cc import uccsd

einsum = lib.einsum

def kernel(myci, eris, ci0=None, max_cycle=50, tol=1e-8,
           verbose=logger.INFO):
    mol = myci.mol
    diag = myci.make_diagonal(eris)
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
        ci0 = myci.get_init_guess(eris)[1]

    def op(xs):
        return [myci.contract(x, eris) for x in xs]

    def precond(x, e, *args):
        diagd = diag - (e-myci.level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return x / diagd

    conv, ecisd, ci = lib.davidson1(op, ci0, precond, tol=tol,
                                    max_cycle=max_cycle, max_space=myci.max_space,
                                    lindep=myci.lindep, nroots=myci.nroots,
                                    verbose=verbose)
    if myci.nroots == 1:
        conv = conv[0]
        ecisd = ecisd[0]
        ci = ci[0]
    return conv, ecisd, ci


def make_diagonal(myci, eris):
    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.focka.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    jdiag_aa = numpy.zeros((nmoa,nmoa))
    jdiag_ab = numpy.zeros((nmoa,nmob))
    jdiag_bb = numpy.zeros((nmob,nmob))
    def diag_idx(n):
        idx = numpy.arange(n)
        return idx * (idx + 1) // 2 + idx
    jdiag_aa[:nocca,:nocca] = numpy.einsum('iijj->ij', eris.oooo)
    jdiag_aa[nocca:,nocca:] = eris.vvvv[diag_idx(nvira)[:,None],diag_idx(nvira)]
    jdiag_aa[:nocca,nocca:] = numpy.einsum('iijj->ij', eris.oovv)
    jdiag_aa[nocca:,:nocca] = jdiag_aa[:nocca,nocca:].T
    jdiag_ab[:nocca,:noccb] = numpy.einsum('iijj->ij', eris.ooOO)
    jdiag_ab[nocca:,noccb:] = eris.vvVV[diag_idx(nvira)[:,None],diag_idx(nvirb)]
    jdiag_ab[:nocca,noccb:] = numpy.einsum('iijj->ij', eris.ooVV)
    jdiag_ab[nocca:,:noccb] = numpy.einsum('iijj->ij', eris.vvOO)
    jdiag_bb[:noccb,:noccb] = numpy.einsum('iijj->ij', eris.OOOO)
    jdiag_bb[noccb:,noccb:] = eris.VVVV[diag_idx(nvirb)[:,None],diag_idx(nvirb)]
    jdiag_bb[:noccb,noccb:] = numpy.einsum('iijj->ij', eris.OOVV)
    jdiag_bb[noccb:,:noccb] = jdiag_bb[:noccb,noccb:].T

    kdiag_aa = numpy.zeros((nmoa,nmoa))
    kdiag_bb = numpy.zeros((nmob,nmob))
    kdiag_aa[:nocca,:nocca] = numpy.einsum('ijji->ij', eris.oooo)
    kdiag_aa[nocca:,nocca:] = lib.unpack_tril(eris.vvvv.diagonal())
    kdiag_aa[:nocca,nocca:] = numpy.einsum('ijji->ji', eris.voov)
    kdiag_aa[nocca:,:nocca] = kdiag_aa[:nocca,nocca:].T
    kdiag_bb[:noccb,:noccb] = numpy.einsum('ijji->ij', eris.OOOO)
    kdiag_bb[noccb:,noccb:] = lib.unpack_tril(eris.VVVV.diagonal())
    kdiag_bb[:noccb,noccb:] = numpy.einsum('ijji->ji', eris.VOOV)
    kdiag_bb[noccb:,:noccb] = kdiag_bb[:noccb,noccb:].T
    jkdiag_aa = jdiag_aa - kdiag_aa
    jkdiag_ab = jdiag_ab
    jkdiag_bb = jdiag_bb - kdiag_bb

    mo_ea = eris.focka.diagonal()
    mo_eb = eris.fockb.diagonal()
    ehf = (mo_ea[:nocca].sum() + mo_eb[:noccb].sum()
           - (jdiag_aa[:nocca,:nocca].sum() - kdiag_aa[:nocca,:nocca].sum()) * .5
           - jdiag_ab[:nocca,:noccb].sum()
           - (jdiag_bb[:noccb,:noccb].sum() - kdiag_bb[:noccb,:noccb].sum()) * .5)

    e1diag_a = numpy.empty((nocca,nvira))
    e1diag_b = numpy.empty((noccb,nvirb))
    e2diag_aa = numpy.empty((nocca,nocca,nvira,nvira))
    e2diag_ab = numpy.empty((nocca,noccb,nvira,nvirb))
    e2diag_bb = numpy.empty((noccb,noccb,nvirb,nvirb))
    for i in range(nocca):
        for a in range(nocca, nmoa):
            e1diag_a[i,a-nocca] = ehf - mo_ea[i] + mo_ea[a] - jkdiag_aa[i,a]
            for j in range(nocca):
                for b in range(nocca, nmoa):
                    e2diag_aa[i,j,a-nocca,b-nocca] = ehf \
                            - mo_ea[i] - mo_ea[j] \
                            + mo_ea[a] + mo_ea[b] \
                            + jkdiag_aa[i,j] + jkdiag_aa[a,b] \
                            - jkdiag_aa[i,a] - jkdiag_aa[j,a] \
                            - jkdiag_aa[i,b] - jkdiag_aa[j,b]

    for i in range(nocca):
        for a in range(nocca, nmoa):
            for j in range(noccb):
                for b in range(noccb, nmob):
                    e2diag_ab[i,j,a-nocca,b-noccb] = ehf \
                            - mo_ea[i] - mo_eb[j] \
                            + mo_ea[a] + mo_eb[b] \
                            + jdiag_ab[i,j] + jdiag_ab[a,b] \
                            - jkdiag_aa[i,a] - jkdiag_bb[j,b] \
                            - jdiag_ab[i,b] - jdiag_ab[a,j]

    for i in range(noccb):
        for a in range(noccb, nmob):
            e1diag_b[i,a-noccb] = ehf - mo_eb[i] + mo_eb[a] - jkdiag_bb[i,a]
            for j in range(noccb):
                for b in range(noccb, nmob):
                    e2diag_bb[i,j,a-noccb,b-noccb] = ehf \
                            - mo_eb[i] - mo_eb[j] \
                            + mo_eb[a] + mo_eb[b] \
                            + jkdiag_bb[i,j] + jkdiag_bb[a,b] \
                            - jkdiag_bb[i,a] - jkdiag_bb[j,a] \
                            - jkdiag_bb[i,b] - jkdiag_bb[j,b]

    return amplitudes_to_cisdvec(ehf, (e1diag_a, e1diag_b),
                                 (e2diag_aa, e2diag_ab, e2diag_bb))

def contract(myci, civec, eris):
    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    c0, (c1a,c1b), (c2aa,c2ab,c2bb) = \
            cisdvec_to_amplitudes(civec, (nmoa,nmob), (nocca,noccb))

    fooa = eris.focka[:nocca,:nocca]
    foob = eris.fockb[:noccb,:noccb]
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    fvva = eris.focka[nocca:,nocca:]
    fvvb = eris.fockb[noccb:,noccb:]

    #:t1  = einsum('ie,ae->ia', c1, fvv)
    t1a = einsum('ie,ae->ia', c1a, fvva)
    t1b = einsum('ie,ae->ia', c1b, fvvb)
    #:t1 -= einsum('ma,mi->ia', c1, foo)
    t1a -= einsum('ma,mi->ia', c1a, fooa)
    t1b -= einsum('ma,mi->ia', c1b, foob)
    #:t1 += einsum('imae,me->ia', c2, fov)
    t1a += numpy.einsum('imae,me->ia', c2aa, fova)
    t1a += numpy.einsum('imae,me->ia', c2ab, fovb)
    t1b += numpy.einsum('imae,me->ia', c2bb, fovb)
    t1b += numpy.einsum('miea,me->ia', c2ab, fova)
    #:t1 += einsum('nf,nafi->ia', c1, eris.ovvo)
    t1a += numpy.einsum('nf,ainf->ia', c1a, eris.voov)
    t1a -= numpy.einsum('nf,nifa->ia', c1a, eris.oovv)
    t1b += numpy.einsum('nf,ainf->ia', c1b, eris.VOOV)
    t1b -= numpy.einsum('nf,nifa->ia', c1b, eris.OOVV)
    t1b += numpy.einsum('nf,ainf->ia', c1a, eris.VOov)
    t1a += numpy.einsum('nf,ainf->ia', c1b, eris.voOV)
    #:t1 -= 0.5*einsum('imef,maef->ia', c2, eris.ovvv)
    eris_Hovvv = _cp(eris.vovv).transpose(1,0,2,3)
    eris_HOVVV = _cp(eris.VOVV).transpose(1,0,2,3)
    eris_HovVV = _cp(eris.voVV).transpose(1,0,2,3)
    eris_HOVvv = _cp(eris.VOvv).transpose(1,0,2,3)
    ovvv = eris_Hovvv - eris_Hovvv.transpose(0,3,2,1)
    OVVV = eris_HOVVV - eris_HOVVV.transpose(0,3,2,1)
    t1a += 0.5*lib.einsum('mief,meaf->ia', c2aa, ovvv)
    t1b += 0.5*lib.einsum('MIEF,MEAF->IA', c2bb, OVVV)
    t1a += lib.einsum('iMfE,MEaf->ia', c2ab, eris_HOVvv)
    t1b += lib.einsum('mIeF,meAF->IA', c2ab, eris_HovVV)
    #:t1 -= 0.5*einsum('mnae,mnie->ia', c2, eris.ooov)
    eris_Hooov = _cp(eris.ooov)
    eris_HOOOV = _cp(eris.OOOV)
    eris_HooOV = _cp(eris.ooOV)
    eris_HOOov = _cp(eris.OOov)
    ooov = eris_Hooov - eris_Hooov.transpose(2,1,0,3)
    OOOV = eris_HOOOV - eris_HOOOV.transpose(2,1,0,3)
    t1a += 0.5*lib.einsum('mnae,nime->ia', c2aa, ooov)
    t1b += 0.5*lib.einsum('mnae,nime->ia', c2bb, OOOV)
    t1a -= lib.einsum('nMaE,niME->ia', c2ab, eris_HooOV)
    t1b -= lib.einsum('mNeA,NIme->IA', c2ab, eris_HOOov)

    #:tmp = einsum('ijae,be->ijab', c2, fvv)
    #:t2  = tmp - tmp.transpose(0,1,3,2)
    t2aa = lib.einsum('ijae,be->ijab', c2aa, fvva)
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2bb = lib.einsum('ijae,be->ijab', c2bb, fvvb)
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2ab = lib.einsum('iJaE,BE->iJaB', c2ab, fvvb)
    t2ab+= lib.einsum('iJeA,be->iJbA', c2ab, fvva)
    #:tmp = einsum('imab,mj->ijab', c2, foo)
    #:t2 -= tmp - tmp.transpose(1,0,2,3)
    tmp = lib.einsum('imab,mj->ijab', c2aa, fooa)
    t2aa-= tmp - tmp.transpose(1,0,2,3)
    tmp = lib.einsum('imab,mj->ijab', c2bb, foob)
    t2bb-= tmp - tmp.transpose(1,0,2,3)
    t2ab-= lib.einsum('iMaB,MJ->iJaB', c2ab, foob)
    t2ab-= lib.einsum('mIaB,mj->jIaB', c2ab, fooa)
    #:t2 += 0.5*einsum('mnab,mnij->ijab', c2, eris.oooo)
    Woooo = _cp(eris.oooo).transpose(0,2,1,3)
    WOOOO = _cp(eris.OOOO).transpose(0,2,1,3)
    WoOoO = _cp(eris.ooOO).transpose(0,2,1,3)
    Woooo = Woooo - Woooo.transpose(1,0,2,3)
    WOOOO = WOOOO - WOOOO.transpose(1,0,2,3)
    t2aa += 0.5*lib.einsum('mnab,mnij->ijab', c2aa, Woooo)
    t2bb += 0.5*lib.einsum('mnab,mnij->ijab', c2bb, WOOOO)
    t2ab += lib.einsum('mNaB,mNiJ->iJaB', c2ab, WoOoO)
    #:t2 += 0.5*einsum('ijef,abef->ijab', c2, eris.vvvv)
    eris_Hvvvv = ao2mo.restore(1, eris.vvvv, nvira)
    eris_HvvVV = _restore(eris.vvVV, nvira, nvirb)
    eris_HVVVV = ao2mo.restore(1, eris.VVVV, nvirb)
    t2aa += lib.einsum('ijef,aebf->ijab', c2aa, eris_Hvvvv)
    t2bb += lib.einsum('ijef,aebf->ijab', c2bb, eris_HVVVV)
    t2ab += lib.einsum('iJeF,aeBF->iJaB', c2ab, eris_HvvVV)
    #:tmp = einsum('imae,mbej->ijab', c2, eris.ovvo)
    #:tmp = tmp - tmp.transpose(0,1,3,2)
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    eris_Hoovv = _cp(eris.oovv)
    eris_HooVV = _cp(eris.ooVV)
    eris_HOOvv = _cp(eris.vvOO).transpose(2,3,0,1)
    eris_HOOVV = _cp(eris.OOVV)
    wovvo = -eris_Hoovv.transpose(0,2,3,1)
    woVVo = -eris_HooVV.transpose(0,2,3,1)
    wOvvO = -eris_HOOvv.transpose(0,2,3,1)
    wOVVO = -eris_HOOVV.transpose(0,2,3,1)
    eris_Hovvo = _cp(eris.voov).transpose(2,3,0,1)
    eris_HovVO = _cp(eris.VOov).transpose(2,3,0,1)
    eris_HOVvo = _cp(eris.voOV).transpose(2,3,0,1)
    eris_HOVVO = _cp(eris.VOOV).transpose(2,3,0,1)
    oovv = eris_Hoovv - eris_Hovvo.transpose(0,3,2,1)
    OOVV = eris_HOOVV - eris_HOVVO.transpose(0,3,2,1)
    wovvo = wovvo + eris_Hovvo.transpose(0,2,1,3)
    wOVVO = wOVVO + eris_HOVVO.transpose(0,2,1,3)
    woVvO = eris_HovVO.transpose(0,2,1,3)
    wOvVo = eris_HOVvo.transpose(0,2,1,3)
    tmpaa = lib.einsum('imae,mbej->ijab', c2aa, wovvo)
    tmpaa+= lib.einsum('iMaE,MbEj->ijab', c2ab, wOvVo)
    tmpbb = lib.einsum('imae,mbej->ijab', c2bb, wOVVO)
    tmpbb+= lib.einsum('mIeA,mBeJ->IJAB', c2ab, woVvO)
    tmpab = lib.einsum('imae,mBeJ->iJaB', c2aa, woVvO)
    tmpab+= lib.einsum('iMaE,MBEJ->iJaB', c2ab, wOVVO)
    tmpba = lib.einsum('IMAE,MbEj->IjAb', c2bb, wOvVo)
    tmpba+= lib.einsum('mIeA,mbej->IjAb', c2ab, wovvo)
    tmpabba =-lib.einsum('iMeA,MbeJ->iJAb', c2ab, wOvvO)
    tmpbaab =-lib.einsum('mIaE,mBEj->IjaB', c2ab, woVVo)
    tmpaa = tmpaa - tmpaa.transpose(0,1,3,2)
    t2aa += tmpaa - tmpaa.transpose(1,0,2,3)
    tmpbb = tmpbb - tmpbb.transpose(0,1,3,2)
    t2bb += tmpbb - tmpbb.transpose(1,0,2,3)
    tmpab -= tmpabba.transpose(0,1,3,2)
    tmpab -= tmpbaab.transpose(1,0,2,3)
    tmpab += tmpba.transpose(1,0,3,2)
    t2ab += tmpab
    #:tmp = numpy.einsum('ia,jb->ijab', c1, fov)
    #:tmp = tmp - tmp.transpose(0,1,3,2)
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    tmpaa = numpy.einsum('ia,jb->ijab', c1a, fova)
    tmpab = numpy.einsum('ia,jb->ijab', c1a, fovb)
    tmpba = numpy.einsum('ia,jb->ijab', c1b, fova)
    tmpbb = numpy.einsum('ia,jb->ijab', c1b, fovb)
    tmpaa = tmpaa - tmpaa.transpose(0,1,3,2)
    t2aa += tmpaa - tmpaa.transpose(1,0,2,3)
    tmpbb = tmpbb - tmpbb.transpose(0,1,3,2)
    t2bb += tmpbb - tmpbb.transpose(1,0,2,3)
    t2ab += tmpab + tmpba.transpose(1,0,3,2)
    #:tmp = einsum('ie,jeba->ijab', c1, numpy.asarray(eris.ovvv).conj())
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    tmpaa = lib.einsum('ie,mbea->imab', c1a, ovvv.conj())
    t2aa += tmpaa - tmpaa.transpose(1,0,2,3)
    tmpbb = lib.einsum('ie,mbea->imab', c1b, OVVV.conj())
    t2bb += tmpbb - tmpbb.transpose(1,0,2,3)
    t2ab += lib.einsum('ie,MBea->iMaB', c1a, eris_HOVvv.conj())
    t2ab += lib.einsum('IE,maEB->mIaB', c1b, eris_HovVV.conj())
    #:tmp = einsum('ma,mbij->ijab', c1, eris.ovoo)
    #:t2 -= tmp - tmp.transpose(0,1,3,2)
    eris_Hoovo = _cp(eris.oovo)
    eris_HOOVO = _cp(eris.OOVO)
    eris_HooVO = _cp(eris.ooVO)
    eris_HOOvo = _cp(eris.OOvo)
    oovo = eris_Hoovo - eris_Hoovo.transpose(0,3,2,1)
    OOVO = eris_HOOVO - eris_HOOVO.transpose(0,3,2,1)
    tmpaa = lib.einsum('ma,mibj->ijab', c1a, oovo)
    t2aa -= tmpaa - tmpaa.transpose(0,1,3,2)
    tmpbb = lib.einsum('ma,mibj->ijab', c1b, OOVO)
    t2bb -= tmpbb - tmpbb.transpose(0,1,3,2)
    t2ab -= lib.einsum('ma,miBJ->iJaB', c1a, eris_HooVO)
    t2ab -= lib.einsum('MA,MJbi->iJbA', c1b, eris_HOOvo)

    #:t1 += fov * c0
    t1a += fova * c0
    t1b += fovb * c0
    #:t2 += eris.oovv * c0
    t2aa += eris.voov.transpose(1,2,0,3) * c0
    t2aa -= eris.voov.transpose(1,2,3,0) * c0
    t2bb += eris.VOOV.transpose(1,2,0,3) * c0
    t2bb -= eris.VOOV.transpose(1,2,3,0) * c0
    t2ab += eris.voOV.transpose(1,2,0,3) * c0
    #:t0  = numpy.einsum('ia,ia', fov, c1)
    #:t0 += numpy.einsum('ijab,ijab', eris.oovv, c2) * .25
    t0  = numpy.einsum('ia,ia', fova, c1a)
    t0 += numpy.einsum('ia,ia', fovb, c1b)
    t0 += numpy.einsum('aijb,ijab', eris.voov, c2aa) * .25
    t0 -= numpy.einsum('ajib,ijab', eris.voov, c2aa) * .25
    t0 += numpy.einsum('aijb,ijab', eris.VOOV, c2bb) * .25
    t0 -= numpy.einsum('ajib,ijab', eris.VOOV, c2bb) * .25
    t0 += numpy.einsum('aijb,ijab', eris.voOV, c2ab)

    return amplitudes_to_cisdvec(t0, (t1a,t1b), (t2aa,t2ab,t2bb))

def amplitudes_to_cisdvec(c0, c1, c2):
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2
    nocca, nvira = c1a.shape
    noccb, nvirb = c1b.shape
    def trilidx(n):
        idx = numpy.tril_indices(n, -1)
        return idx[0] * n + idx[1]
    ooidxa = trilidx(nocca)
    vvidxa = trilidx(nvira)
    ooidxb = trilidx(noccb)
    vvidxb = trilidx(nvirb)
    size = (1, nocca*nvira, noccb*nvirb, nocca*noccb*nvira*nvirb,
            len(ooidxa)*len(vvidxa), len(ooidxb)*len(vvidxb))
    loc = numpy.cumsum(size)
    civec = numpy.empty(loc[-1])
    civec[0] = c0
    civec[loc[0]:loc[1]] = c1a.ravel()
    civec[loc[1]:loc[2]] = c1b.ravel()
    civec[loc[2]:loc[3]] = c2ab.ravel()
    lib.take_2d(c2aa.reshape(nocca**2,nvira**2), ooidxa, vvidxa, out=civec[loc[3]:loc[4]])
    lib.take_2d(c2bb.reshape(noccb**2,nvirb**2), ooidxb, vvidxb, out=civec[loc[4]:loc[5]])
    return civec

def cisdvec_to_amplitudes(civec, nmoa_nmob, nocca_noccb):
    norba, norbb = nmoa_nmob
    nocca, noccb = nocca_noccb
    nvira = norba - nocca
    nvirb = norbb - noccb
    nooa = nocca * (nocca-1) // 2
    nvva = nvira * (nvira-1) // 2
    noob = noccb * (noccb-1) // 2
    nvvb = nvirb * (nvirb-1) // 2
    size = (1, nocca*nvira, noccb*nvirb, nocca*noccb*nvira*nvirb,
            nooa*nvva, noob*nvvb)
    loc = numpy.cumsum(size)
    c0 = civec[0]
    c1a = civec[loc[0]:loc[1]].reshape(nocca,nvira)
    c1b = civec[loc[1]:loc[2]].reshape(noccb,nvirb)
    c2ab = civec[loc[2]:loc[3]].reshape(nocca,noccb,nvira,nvirb)
    c2aa = _unpack_4fold(civec[loc[3]:loc[4]], nocca, nvira)
    c2bb = _unpack_4fold(civec[loc[4]:loc[5]], noccb, nvirb)
    return c0, (c1a,c1b), (c2aa,c2ab,c2bb)

def _unpack_4fold(c2vec, nocc, nvir):
    ooidx = numpy.tril_indices(nocc, -1)
    vvidx = numpy.tril_indices(nvir, -1)
    c2tmp = numpy.zeros((nocc,nocc,nvir*(nvir-1)//2))
    c2tmp[ooidx] = c2vec.reshape(nocc*(nocc-1)//2, nvir*(nvir-1)//2)
    c2tmp = c2tmp - c2tmp.transpose(1,0,2)
    c2 = numpy.zeros((nocc,nocc,nvir,nvir))
    c2[:,:,vvidx[0],vvidx[1]] = c2tmp
    c2 = c2 - c2.transpose(0,1,3,2)
    return c2

def to_fci(cisdvec, nmoa_nmob, nocca_noccb):
    from pyscf import fci
    from pyscf.ci.cisd_slow import t2strs
    norba, norbb = nmoa_nmob
    nocca, noccb = nocca_noccb
    nvira = norba - nocca
    nvirb = norbb - noccb
    c0, c1, c2 = cisdvec_to_amplitudes(cisdvec, nmoa_nmob, nocca_noccb)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2
    t1addra, t1signa = cisd.t1strs(norba, nocca)
    t1addrb, t1signb = cisd.t1strs(norbb, noccb)

    na = fci.cistring.num_strings(norba, nocca)
    nb = fci.cistring.num_strings(norbb, noccb)
    fcivec = numpy.zeros((na,nb))
    fcivec[0,0] = c0
    fcivec[t1addra,0] = c1a[::-1].T.ravel() * t1signa
    fcivec[0,t1addrb] = c1b[::-1].T.ravel() * t1signb
    c2ab = c2ab[::-1,::-1].transpose(2,0,3,1).reshape(nocca*nvira,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1signa, t1signb, c2ab)
    lib.takebak_2d(fcivec, c2ab, t1addra, t1addrb)

    if nocca > 1 and nvira > 1:
        ooidx = numpy.tril_indices(nocca, -1)
        vvidx = numpy.tril_indices(nvira, -1)
        c2aa = c2aa[ooidx][:,vvidx[0],vvidx[1]]
        t2addra, t2signa = t2strs(norba, nocca)
        fcivec[t2addra,0] = c2aa[::-1].T.ravel() * t2signa
    if noccb > 1 and nvirb > 1:
        ooidx = numpy.tril_indices(noccb, -1)
        vvidx = numpy.tril_indices(nvirb, -1)
        c2bb = c2bb[ooidx][:,vvidx[0],vvidx[1]]
        t2addrb, t2signb = t2strs(norbb, noccb)
        fcivec[0,t2addrb] = c2bb[::-1].T.ravel() * t2signb
    return fcivec

def from_fci(ci0, nmoa_nmob, nocca_noccb):
    from pyscf import fci
    from pyscf.ci.cisd_slow import t2strs
    norba, norbb = nmoa_nmob
    nocca, noccb = nocca_noccb
    nvira = norba - nocca
    nvirb = norbb - noccb
    t1addra, t1signa = cisd.t1strs(norba, nocca)
    t1addrb, t1signb = cisd.t1strs(norbb, noccb)

    na = fci.cistring.num_strings(norba, nocca)
    nb = fci.cistring.num_strings(norbb, noccb)
    ci0 = ci0.reshape(na,nb)
    c0 = ci0[0,0]
    c1a = ((ci0[t1addra,0] * t1signa).reshape(nvira,nocca).T)[::-1]
    c1b = ((ci0[0,t1addrb] * t1signb).reshape(nvirb,noccb).T)[::-1]

    c2ab = numpy.einsum('i,j,ij->ij', t1signa, t1signb, ci0[t1addra][:,t1addrb])
    c2ab = c2ab.reshape(nvira,nocca,nvirb,noccb).transpose(1,3,0,2)
    c2ab = c2ab[::-1,::-1]
    t2addra, t2signa = t2strs(norba, nocca)
    c2aa = (ci0[t2addra,0] * t2signa).reshape(nvira*(nvira-1)//2,-1).T
    c2aa = _unpack_4fold(c2aa[::-1], nocca, nvira)
    t2addrb, t2signb = t2strs(norbb, noccb)
    c2bb = (ci0[0,t2addrb] * t2signb).reshape(nvirb*(nvirb-1)//2,-1).T
    c2bb = _unpack_4fold(c2bb[::-1], noccb, nvirb)

    return amplitudes_to_cisdvec(c0, (c1a,c1b), (c2aa,c2ab,c2bb))


def make_rdm1(ci, nmoa_nmob, nocca_noccb):
    nmoa, nmob = nmoa_nmob
    nocca, noccb = nocca_noccb
    c0, c1, c2 = cisdvec_to_amplitudes(ci, nmoa_nmob, nocca_noccb)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2

    dova = c0 * c1a
    dovb = c0 * c1b
    dova += numpy.einsum('jb,ijab->ia', c1a, c2aa)
    dova += numpy.einsum('jb,ijab->ia', c1b, c2ab)
    dovb += numpy.einsum('jb,ijab->ia', c1b, c2bb)
    dovb += numpy.einsum('jb,jiba->ia', c1a, c2ab)

    dooa  =-numpy.einsum('ia,ka->ik', c1a, c1a)
    doob  =-numpy.einsum('ia,ka->ik', c1b, c1b)
    dooa -= numpy.einsum('ijab,ikab->jk', c2aa, c2aa) * .5
    dooa -= numpy.einsum('jiab,kiab->jk', c2ab, c2ab)
    doob -= numpy.einsum('ijab,ikab->jk', c2bb, c2bb) * .5
    doob -= numpy.einsum('ijab,ikab->jk', c2ab, c2ab)

    dvva  = numpy.einsum('ia,ic->ca', c1a, c1a)
    dvvb  = numpy.einsum('ia,ic->ca', c1b, c1b)
    dvva += numpy.einsum('ijab,ijac->cb', c2aa, c2aa) * .5
    dvva += numpy.einsum('ijba,ijca->cb', c2ab, c2ab)
    dvvb += numpy.einsum('ijba,ijca->cb', c2bb, c2bb) * .5
    dvvb += numpy.einsum('ijab,ijac->cb', c2ab, c2ab)

    dm1a = numpy.empty((nmoa,nmoa))
    dm1b = numpy.empty((nmob,nmob))
    dm1a[:nocca,nocca:] = dova
    dm1b[:noccb,noccb:] = dovb
    dm1a[nocca:,:nocca] = dova.T.conj()
    dm1b[noccb:,:noccb] = dovb.T.conj()
    dm1a[:nocca,:nocca] = dooa
    dm1b[:noccb,:noccb] = doob
    dm1a[nocca:,nocca:] = dvva
    dm1b[noccb:,noccb:] = dvvb
    dm1a[numpy.diag_indices(nocca)] += 1
    dm1b[numpy.diag_indices(noccb)] += 1
    return dm1a, dm1b

def make_rdm2(ci, nmoa_nmob, nocca_noccb):
    '''spin-orbital 2pdm in chemist's notation
    '''
    nmoa, nmob = nmoa_nmob
    nocca, noccb = nocca_noccb
    c0, c1, c2 = cisdvec_to_amplitudes(ci, nmoa_nmob, nocca_noccb)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2

    d2oovv = c0 * c2aa * .5
    d2oOvV = c0 * c2ab
    d2OOVV = c0 * c2bb * .5
    #:dvvvo = numpy.einsum('ia,ikcd->cdak', c1, c2) * .5
    d2vvvo = numpy.einsum('ia,ikcd->cdak', c1a, c2aa) * .5
    d2vVvO = numpy.einsum('ia,ikcd->cdak', c1a, c2ab)
    d2VvVo = numpy.einsum('ia,kidc->cdak', c1b, c2ab)
    d2VVVO = numpy.einsum('ia,ikcd->cdak', c1b, c2bb) * .5
    #:dovoo = numpy.einsum('ia,klac->ickl', c1, c2) *-.5
    d2ovoo = numpy.einsum('ia,klac->ickl', c1a, c2aa) *-.5
    d2oVoO =-numpy.einsum('ia,klac->ickl', c1a, c2ab)
    d2OvOo =-numpy.einsum('ia,lkca->ickl', c1b, c2ab)
    d2OVOO = numpy.einsum('ia,klac->ickl', c1b, c2bb) *-.5
    #:doooo = numpy.einsum('klab,ijab->klij', c2, c2) * .25
    d2oooo = numpy.einsum('klab,ijab->klij', c2aa, c2aa) * .25
    d2oOoO = numpy.einsum('klab,ijab->klij', c2ab, c2ab)
    d2OOOO = numpy.einsum('klab,ijab->klij', c2bb, c2bb) * .25
    #:dvvvv = numpy.einsum('ijcd,ijab->cdab', c2, c2) * .25
    d2vvvv = numpy.einsum('ijcd,ijab->cdab', c2aa, c2aa) * .25
    d2vVvV = numpy.einsum('ijcd,ijab->cdab', c2ab, c2ab)
    d2VVVV = numpy.einsum('ijcd,ijab->cdab', c2bb, c2bb) * .25
    #:dovov =-numpy.einsum('ijab,ikac->jckb', c2, c2)
    #:dovov-= numpy.einsum('ia,jb->jaib', c1, c1)
    d2ovov =-numpy.einsum('ijab,ikac->jckb', c2aa, c2aa)
    d2ovov-= numpy.einsum('jiba,kica->jckb', c2ab, c2ab)
    d2oVoV =-numpy.einsum('jIaB,kIaC->jCkB', c2ab, c2ab)
    d2OvOv =-numpy.einsum('iJbA,iKcA->JcKb', c2ab, c2ab)
    d2OVOV =-numpy.einsum('ijab,ikac->jckb', c2bb, c2bb)
    d2OVOV-= numpy.einsum('ijab,ikac->jckb', c2ab, c2ab)
    d2ovov-= numpy.einsum('ia,jb->jaib', c1a, c1a)
    d2OVOV-= numpy.einsum('ia,jb->jaib', c1b, c1b)
    #:dvoov = numpy.einsum('ijab,ikac->cjkb', c2, c2)
    #:dvoov+= numpy.einsum('ia,jb->ajib', c1, c1)
    d2voov = numpy.einsum('ijab,ikac->cjkb', c2aa, c2aa)
    d2voov+= numpy.einsum('jIbA,kIcA->cjkb', c2ab, c2ab)
    d2vOoV = numpy.einsum('iJaB,ikac->cJkB', c2ab, c2aa)
    d2vOoV+= numpy.einsum('IJAB,kIcA->cJkB', c2bb, c2ab)
    d2VOOV = numpy.einsum('ijab,ikac->cjkb', c2bb, c2bb)
    d2VOOV+= numpy.einsum('iJaB,iKaC->CJKB', c2ab, c2ab)
    d2voov+= numpy.einsum('ia,jb->ajib', c1a, c1a)
    d2vOoV+= numpy.einsum('ia,jb->ajib', c1a, c1b)
    d2VOOV+= numpy.einsum('ia,jb->ajib', c1b, c1b)

    dm2aa = numpy.zeros((nmoa,nmoa,nmoa,nmoa))
    dm2ab = numpy.zeros((nmoa,nmoa,nmob,nmob))
    dm2bb = numpy.zeros((nmob,nmob,nmob,nmob))
    dm2aa[:nocca,:nocca,:nocca,:nocca] = d2oooo - d2oooo.transpose(1,0,2,3)
    dm2ab[:nocca,:noccb,:nocca,:noccb] = d2oOoO
    dm2bb[:noccb,:noccb,:noccb,:noccb] = d2OOOO - d2OOOO.transpose(1,0,2,3)

    dm2aa[:nocca,nocca:,:nocca,:nocca] = d2ovoo - d2ovoo.transpose(0,1,3,2)
    dm2aa[nocca:,:nocca,:nocca,:nocca] = dm2aa[:nocca,nocca:,:nocca,:nocca].transpose(1,0,3,2)
    dm2aa[:nocca,:nocca,:nocca,nocca:] = dm2aa[:nocca,nocca:,:nocca,:nocca].transpose(2,3,0,1)
    dm2aa[:nocca,:nocca,nocca:,:nocca] = dm2aa[:nocca,nocca:,:nocca,:nocca].transpose(3,2,1,0)
    dm2ab[:nocca,noccb:,:nocca,:noccb] = d2oVoO
    dm2ab[nocca:,:noccb,:nocca,:noccb] = d2OvOo.transpose(1,0,3,2)
    dm2ab[:nocca,:noccb,:nocca,noccb:] = dm2ab[:nocca,noccb:,:nocca,:noccb].transpose(2,3,0,1)
    dm2ab[:nocca,:noccb,nocca:,:noccb] = dm2ab[nocca:,:noccb,:nocca,:noccb].transpose(2,3,0,1)
    dm2bb[:noccb,noccb:,:noccb,:noccb] = d2OVOO - d2OVOO.transpose(0,1,3,2)
    dm2bb[noccb:,:noccb,:noccb,:noccb] = dm2bb[:noccb,noccb:,:noccb,:noccb].transpose(1,0,3,2)
    dm2bb[:noccb,:noccb,:noccb,noccb:] = dm2bb[:noccb,noccb:,:noccb,:noccb].transpose(2,3,0,1)
    dm2bb[:noccb,:noccb,noccb:,:noccb] = dm2bb[:noccb,noccb:,:noccb,:noccb].transpose(3,2,1,0)

    dm2aa[:nocca,:nocca,nocca:,nocca:] = d2oovv - d2oovv.transpose(1,0,2,3)
    dm2ab[:nocca,:noccb,nocca:,noccb:] = d2oOvV
    dm2bb[:noccb,:noccb,noccb:,noccb:] = d2OOVV - d2OOVV.transpose(1,0,2,3)
    dm2aa[nocca:,nocca:,:nocca,:nocca] = dm2aa[:nocca,:nocca,nocca:,nocca:].transpose(2,3,0,1)
    dm2ab[nocca:,noccb:,:nocca,:noccb] = dm2ab[:nocca,:noccb,nocca:,noccb:].transpose(2,3,0,1)
    dm2bb[noccb:,noccb:,:noccb,:noccb] = dm2bb[:noccb,:noccb,noccb:,noccb:].transpose(2,3,0,1)

    dm2aa[nocca:,nocca:,nocca:,:nocca] = d2vvvo - d2vvvo.transpose(1,0,2,3)
    dm2aa[nocca:,nocca:,:nocca,nocca:] = dm2aa[nocca:,nocca:,nocca:,:nocca].transpose(1,0,3,2)
    dm2aa[nocca:,:nocca,nocca:,nocca:] = dm2aa[nocca:,nocca:,nocca:,:nocca].transpose(2,3,0,1)
    dm2aa[:nocca,nocca:,nocca:,nocca:] = dm2aa[nocca:,nocca:,nocca:,:nocca].transpose(3,2,1,0)
    dm2ab[nocca:,noccb:,nocca:,:noccb] = d2vVvO
    dm2ab[nocca:,noccb:,:nocca,noccb:] = d2VvVo.transpose(1,0,3,2)
    dm2ab[nocca:,:noccb,nocca:,noccb:] = dm2ab[nocca:,noccb:,nocca:,:noccb].transpose(2,3,0,1)
    dm2ab[:nocca,noccb:,nocca:,noccb:] = dm2ab[nocca:,noccb:,:nocca,noccb:].transpose(2,3,0,1)
    dm2bb[noccb:,noccb:,noccb:,:noccb] = d2VVVO - d2VVVO.transpose(1,0,2,3)
    dm2bb[noccb:,noccb:,:noccb,noccb:] = dm2bb[noccb:,noccb:,noccb:,:noccb].transpose(1,0,3,2)
    dm2bb[noccb:,:noccb,noccb:,noccb:] = dm2bb[noccb:,noccb:,noccb:,:noccb].transpose(2,3,0,1)
    dm2bb[:noccb,noccb:,noccb:,noccb:] = dm2bb[noccb:,noccb:,noccb:,:noccb].transpose(3,2,1,0)

    dm2aa[nocca:,nocca:,nocca:,nocca:] = d2vvvv - d2vvvv.transpose(1,0,2,3)
    dm2ab[nocca:,noccb:,nocca:,noccb:] = d2vVvV
    dm2bb[noccb:,noccb:,noccb:,noccb:] = d2VVVV - d2VVVV.transpose(1,0,2,3)

    dm2aa[:nocca,nocca:,:nocca,nocca:] = d2ovov
    dm2aa[nocca:,:nocca,nocca:,:nocca] = dm2aa[:nocca,nocca:,:nocca,nocca:].transpose(1,0,3,2)
    dm2ab[:nocca,noccb:,:nocca,noccb:] = d2oVoV
    dm2ab[nocca:,:noccb,nocca:,:noccb] = d2OvOv.transpose(1,0,3,2)
    dm2bb[:noccb,noccb:,:noccb,noccb:] = d2OVOV
    dm2bb[noccb:,:noccb,noccb:,:noccb] = dm2bb[:noccb,noccb:,:noccb,noccb:].transpose(1,0,3,2)

    dm2aa[nocca:,:nocca,:nocca,nocca:] = d2voov
    dm2aa[:nocca,nocca:,nocca:,:nocca] = dm2aa[nocca:,:nocca,:nocca,nocca:].transpose(1,0,3,2)
    dm2ab[nocca:,:noccb,:nocca,noccb:] = d2vOoV
    dm2ab[:nocca,noccb:,nocca:,:noccb] = d2vOoV.transpose(2,3,0,1)
    dm2bb[noccb:,:noccb,:noccb,noccb:] = d2VOOV
    dm2bb[:noccb,noccb:,noccb:,:noccb] = dm2bb[noccb:,:noccb,:noccb,noccb:].transpose(1,0,3,2)

    dm1a, dm1b = make_rdm1(ci, nmoa_nmob, nocca_noccb)
    dm1a[numpy.diag_indices(nocca)] -= 1
    dm1b[numpy.diag_indices(noccb)] -= 1
    for i in range(nocca):
        for j in range(nocca):
            dm2aa[i,j,i,j] += 1
            dm2aa[i,j,j,i] -= 1
        dm2aa[i,:,i,:] += dm1a
        dm2aa[:,i,:,i] += dm1a
        dm2aa[:,i,i,:] -= dm1a
        dm2aa[i,:,:,i] -= dm1a
    for i in range(noccb):
        for j in range(noccb):
            dm2bb[i,j,i,j] += 1
            dm2bb[i,j,j,i] -= 1
        dm2bb[i,:,i,:] += dm1b
        dm2bb[:,i,:,i] += dm1b
        dm2bb[:,i,i,:] -= dm1b
        dm2bb[i,:,:,i] -= dm1b
    for i in range(nocca):
        for j in range(noccb):
            dm2ab[i,j,i,j] += 1
        dm2ab[i,:,i,:] += dm1b
    for i in range(noccb):
        dm2ab[:,i,:,i] += dm1a

    # To chemist's convention
    dm2aa = dm2aa.transpose(0,2,1,3)
    dm2ab = dm2ab.transpose(0,2,1,3)
    dm2bb = dm2bb.transpose(0,2,1,3)
    return dm2aa, dm2ab, dm2bb


class CISD(cisd.CISD):

    @property
    def nocc(self):
        nocca, noccb = self.get_nocc()
        return nocca + noccb

    @property
    def nmo(self):
        nmoa, nmob = self.get_nmo()
        return nmoa + nmob

    get_nocc = uccsd.get_nocc
    get_nmo = uccsd.get_nmo

    def kernel(self, ci0=None, mo_coeff=None, eris=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        self.converged, self.e_corr, self.ci = \
                kernel(self, eris, ci0, max_cycle=self.max_cycle,
                       tol=self.conv_tol, verbose=self.verbose)
        if numpy.all(self.converged):
            logger.info(self, 'CISD converged')
        else:
            logger.info(self, 'CISD not converged')
        if self.nroots > 1:
            for i,e in enumerate(self.e_tot):
                logger.note(self, 'CISD root %d  E = %.16g', i, e)
        else:
            logger.note(self, 'E(CISD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.ci

    def get_init_guess(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocca = eris.nocca
        noccb = eris.noccb
        mo_ea = eris.focka.diagonal()
        mo_eb = eris.fockb.diagonal()
        eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
        eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]
        t1a = eris.focka[:nocca,nocca:] / eia_a
        t1b = eris.fockb[:noccb,noccb:] / eia_b

        eris_voov = _cp(eris.voov)
        eris_voOV = _cp(eris.voOV)
        eris_VOOV = _cp(eris.VOOV)
        t2aa = eris_voov.transpose(1,2,0,3) - eris_voov.transpose(2,1,0,3)
        t2bb = eris_VOOV.transpose(1,2,0,3) - eris_VOOV.transpose(2,1,0,3)
        t2ab = eris_voOV.transpose(1,2,0,3).copy()
        t2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
        t2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
        t2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

        emp2  = numpy.einsum('ia,ia', eris.focka[:nocca,nocca:], t1a)
        emp2 += numpy.einsum('ia,ia', eris.fockb[:noccb,noccb:], t1b)
        emp2 += numpy.einsum('aijb,ijab', eris_voov, t2aa) * .25
        emp2 -= numpy.einsum('ajib,ijab', eris_voov, t2aa) * .25
        emp2 += numpy.einsum('aijb,ijab', eris_VOOV, t2bb) * .25
        emp2 -= numpy.einsum('ajib,ijab', eris_VOOV, t2bb) * .25
        emp2 += numpy.einsum('aijb,ijab', eris_voOV, t2ab)
        self.emp2 = emp2
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, amplitudes_to_cisdvec(1, (t1a,t1b), (t2aa,t2ab,t2bb))

    contract = contract
    make_diagonal = make_diagonal

    def ao2mo(self, mo_coeff=None):
        return _make_eris_incore(self, mo_coeff)

    def to_fci(self, cisdvec, nmoa_nmob=None, nocca_noccb=None):
        return to_fci(cisdvec, nmoa_nmob, nocca_noccb)

    def from_fci(self, fcivec, nmoa_nmob=None, nocca_noccb=None):
        return from_fci(fcivec, nmoa_nmob, nocca_noccb)

    def make_rdm1(self, ci=None, nmoa_nmob=None, nocca_noccb=None):
        if ci is None: ci = self.ci
        if nmoa_nmob is None: nmoa_nmob = self.get_nmo()
        if nocca_noccb is None: nocca_noccb = self.get_nocc()
        return make_rdm1(ci, nmoa_nmob, nocca_noccb)

    def make_rdm2(self, ci=None, nmoa_nmob=None, nocca_noccb=None):
        if ci is None: ci = self.ci
        if nmoa_nmob is None: nmoa_nmob = self.get_nmo()
        if nocca_noccb is None: nocca_noccb = self.get_nocc()
        return make_rdm2(ci, nmoa_nmob, nocca_noccb)


class _UCISD_ERIs:
    def __init__(self, myci, mo_coeff=None):
        moidx = uccsd.get_umoidx(myci)
        if mo_coeff is None:
            mo_coeff = (myci.mo_coeff[0][:,moidx[0]], myci.mo_coeff[1][:,moidx[1]])
        else:
            mo_coeff = (mo_coeff[0][:,moidx[0]], mo_coeff[1][:,moidx[1]])
# Note: Always recompute the fock matrix in UCISD because the mf object may be
# converted from ROHF object in which orbital energies are eigenvalues of
# Roothaan Fock rather than the true alpha, beta orbital energies. 
        dm = myci._scf.make_rdm1(myci.mo_coeff, myci.mo_occ)
        fockao = myci._scf.get_hcore() + myci._scf.get_veff(myci.mol, dm)
        self.focka = reduce(numpy.dot, (mo_coeff[0].T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(numpy.dot, (mo_coeff[1].T, fockao[1], mo_coeff[1]))
        self.mo_coeff = mo_coeff
        self.nocca, self.noccb = myci.get_nocc()

        self.oooo = None
        self.vooo = None
        self.voov = None
        self.vvoo = None
        self.vovv = None
        self.vvvv = None

        self.OOOO = None
        self.VOOO = None
        self.VOOV = None
        self.VVOO = None
        self.VOVV = None
        self.VVVV = None

        self.ooOO = None
        self.voOO = None
        self.voOV = None
        self.vvOO = None
        self.voVV = None
        self.vvVV = None

        self.VOoo = None
        self.VVoo = None
        self.VOvv = None
        self.OVvv = None

def _make_eris_incore(myci, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    eris = _UCISD_ERIs(myci, mo_coeff)
    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    moa, mob = eris.mo_coeff

    eri_aa = ao2mo.restore(1, ao2mo.full(myci._scf._eri, moa), nmoa)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ooov = eri_aa[:nocca,:nocca,:nocca,nocca:].copy()
    eris.vooo = eri_aa[nocca:,:nocca,:nocca,:nocca].copy()
    eris.oovo = eri_aa[:nocca,:nocca,nocca:,:nocca].copy()
    eris.voov = eri_aa[nocca:,:nocca,:nocca,nocca:].copy()
    eris.vvoo = eri_aa[nocca:,nocca:,:nocca,:nocca].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.vovo = eri_aa[nocca:,:nocca,nocca:,:nocca].copy()
    eris.vovv = eri_aa[nocca:,:nocca,nocca:,nocca:].copy()
    #vovv = eri_aa[nocca:,:nocca,nocca:,nocca:].reshape(-1,nvira,nvira)
    #eris.vovv = lib.pack_tril(vovv).reshape(nvira,nocca,nvira*(nvira+1)//2)
    eris.vvvv = ao2mo.restore(4, eri_aa[nocca:,nocca:,nocca:,nocca:].copy(), nvira)
    vovv = eri_aa = None

    eri_bb = ao2mo.restore(1, ao2mo.full(myci._scf._eri, mob), nmob)
    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OOOV = eri_bb[:noccb,:noccb,:noccb,noccb:].copy()
    eris.VOOO = eri_bb[noccb:,:noccb,:noccb,:noccb].copy()
    eris.OOVO = eri_bb[:noccb,:noccb,noccb:,:noccb].copy()
    eris.VOOV = eri_bb[noccb:,:noccb,:noccb,noccb:].copy()
    eris.VVOO = eri_bb[noccb:,noccb:,:noccb,:noccb].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.VOVO = eri_bb[noccb:,:noccb,noccb:,:noccb].copy()
    eris.VOVV = eri_bb[noccb:,:noccb,noccb:,noccb:].copy()
    #VOVV = eri_bb[noccb:,:noccb,noccb:,noccb:].reshape(-1,nvirb,nvirb)
    #eris.VOVV = lib.pack_tril(VOVV).reshape(nvirb,noccb,nvirb*(nvirb+1)//2)
    eris.VVVV = ao2mo.restore(4, eri_bb[noccb:,noccb:,noccb:,noccb:].copy(), nvirb)
    VOVV = eri_bb = None

    eri_ab = ao2mo.general(myci._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ooOV = eri_ab[:nocca,:nocca,:noccb,noccb:].copy()
    eris.voOO = eri_ab[nocca:,:nocca,:noccb,:noccb].copy()
    eris.ooVO = eri_ab[:nocca,:nocca,noccb:,:noccb].copy()
    eris.voOV = eri_ab[nocca:,:nocca,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.vvOO = eri_ab[nocca:,nocca:,:noccb,:noccb].copy()
    eris.voVO = eri_ab[nocca:,:nocca,noccb:,:noccb].copy()
    eris.voVV = eri_ab[nocca:,:nocca,noccb:,noccb:].copy()
    #voVV = eri_ab[nocca:,:nocca,noccb:,noccb:].reshape(nocca*nvira,nvirb,nvirb)
    #eris.voVV = lib.pack_tril(voVV).reshape(nvira,nocca,nvirb*(nvirb+1)//2)
    voVV = None
    vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].reshape(nvira**2,nvirb**2)
    idxa = numpy.tril_indices(nvira)
    idxb = numpy.tril_indices(nvirb)
    eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

    eri_ba = lib.transpose(eri_ab.reshape(nmoa**2,nmob**2))
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OOov = eri_ba[:noccb,:noccb,:nocca,nocca:].copy()
    eris.VOoo = eri_ba[noccb:,:noccb,:nocca,:nocca].copy()
    eris.OOvo = eri_ba[:noccb,:noccb,nocca:,:nocca].copy()
    eris.VOov = eri_ba[noccb:,:noccb,:nocca,nocca:].copy()
    eris.VVoo = eri_ba[noccb:,noccb:,:nocca,:nocca].copy()
    eris.VOvo = eri_ba[noccb:,:noccb,nocca:,:nocca].copy()
    eris.VOvv = eri_ba[noccb:,:noccb,nocca:,nocca:].copy()
    #VOvv = eri_ba[noccb:,:noccb,nocca:,nocca:].reshape(noccb*nvirb,nvira,nvira)
    #eris.VOvv = lib.pack_tril(VOvv).reshape(nvirb,noccb,nvira*(nvira+1)//2)
    VOvv = None
    eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()
    return eris


def _cp(a):
    return numpy.array(a, copy=False, order='C')

def _restore(mat, nvira, nvirb):
    mat1 = numpy.zeros((nvira*nvira,nvirb*nvirb))
    idxa = numpy.tril_indices(nvira)
    idxb = numpy.tril_indices(nvirb)
    lib.takebak_2d(mat1, mat, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])
    lib.takebak_2d(mat1, mat, idxa[0]*nvira+idxa[1], idxb[1]*nvirb+idxb[0])
    lib.takebak_2d(mat1, mat, idxa[1]*nvira+idxa[0], idxb[0]*nvirb+idxb[1])
    lib.takebak_2d(mat1, mat, idxa[1]*nvira+idxa[0], idxb[1]*nvirb+idxb[0])
    mat1 = mat1.reshape(nvira,nvira,nvirb,nvirb)
    idxa = numpy.arange(nvira)
    idxb = numpy.arange(nvirb)
    mat1[idxa,idxa] *= .5
    mat1[:,:,idxb,idxb] *= .5
    return mat1

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import fci
    numpy.random.seed(12)
    nocc = 3
    nvir = 5
    nmo = nocc + nvir

    c1a = numpy.random.random((nocc,nvir))
    c1b = numpy.random.random((nocc,nvir))
    c2aa = numpy.random.random((nocc,nocc,nvir,nvir))
    c2bb = numpy.random.random((nocc,nocc,nvir,nvir))
    c2ab = numpy.random.random((nocc,nocc,nvir,nvir))
    c1 = (c1a, c1b)
    c2 = (c2aa, c2ab, c2bb)
    cisdvec = amplitudes_to_cisdvec(1., c1, c2)
    fcivec = to_fci(cisdvec, (nmo,nmo), (nocc,nocc))
    cisdvec1 = from_fci(fcivec, (nmo,nmo), (nocc,nocc))
    print(abs(cisdvec-cisdvec1).sum())
    ci1 = to_fci(cisdvec1, (nmo,nmo), (nocc,nocc))
    print(abs(fcivec-ci1).sum())

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0.,-1.    , 1.   )],
    ]
    mol.charge = -2
    mol.spin = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    ehf0 = mf.e_tot - mol.energy_nuc()
    myci = CISD(mf)
    numpy.random.seed(10)
    mo = numpy.random.random(myci.mo_coeff.shape)
    eris = myci.ao2mo(mo)

    numpy.random.seed(12)
    nocca, noccb = mol.nelec
    nmo = mf.mo_occ[0].size
    nvira = nmo - nocca
    nvirb = nmo - noccb
    c1a  = .1 * numpy.random.random((nocca,nvira))
    c1b  = .1 * numpy.random.random((noccb,nvirb))
    c2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
    c2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
    c2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
    cisdvec = amplitudes_to_cisdvec(1., (c1a, c1b), (c2aa, c2ab, c2bb))

    hcisd0 = contract(myci, amplitudes_to_cisdvec(1., (c1a,c1b), (c2aa,c2ab,c2bb)), eris)
#    from pyscf.ci import gcisd_slow
#    res = cisdvec_to_amplitudes(hcisd0, nmoa_nmob, nocca_noccb)
#    res = (res[0],
#           uccsd.spatial2spin(res[1], eris.orbspin),
#           uccsd.spatial2spin(res[2], eris.orbspin))
#    print(lib.finger(gcisd_slow.amplitudes_to_cisdvec(*res)) - 187.10206473716548)
    print(lib.finger(hcisd0) - 466.56620234351681)
    eris = myci.ao2mo(mf.mo_coeff)
    hcisd0 = contract(myci, cisdvec, eris)
    eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
    eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
    eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                    mf.mo_coeff[1], mf.mo_coeff[1]])
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    h2e = fci.direct_uhf.absorb_h1e((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                    h1a.shape[0], mol.nelec, .5)
    nmo = (mf.mo_coeff[0].shape[1],mf.mo_coeff[1].shape[1])
    fcivec = to_fci(cisdvec, nmo, mol.nelec)
    hci1 = fci.direct_uhf.contract_2e(h2e, fcivec, h1a.shape[0], mol.nelec)
    hci1 -= ehf0 * fcivec
    hcisd1 = from_fci(hci1, nmo, mol.nelec)
    print(numpy.linalg.norm(hcisd1-hcisd0) / numpy.linalg.norm(hcisd0))

    hdiag0 = make_diagonal(myci, eris)
    hdiag0 = to_fci(hdiag0, nmo, mol.nelec).ravel()
    hdiag0 = from_fci(hdiag0, nmo, mol.nelec).ravel()
    hdiag1 = fci.direct_uhf.make_hdiag((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                       h1a.shape[0], mol.nelec)
    hdiag1 = from_fci(hdiag1, nmo, mol.nelec).ravel()
    print(numpy.linalg.norm(abs(hdiag0)-abs(hdiag1)))

    ecisd = myci.kernel(eris=eris)[0]
    efci = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                 h1a.shape[0], mol.nelec)[0]
    print(ecisd, ecisd - -0.037067274690894436, '> E(fci)', efci-ehf0)

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0.,-1.    , 1.   )],
    ]
    mol.charge = 2
    mol.spin = 2
    mol.basis = '6-31g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    ehf0 = mf.e_tot - mol.energy_nuc()
    myci = CISD(mf)
    eris = myci.ao2mo()
    ecisd = myci.kernel(eris=eris)[0]
    eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
    eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
    eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                    mf.mo_coeff[1], mf.mo_coeff[1]])
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    efci, fcivec = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                         h1a.shape[0], mol.nelec)
    print(ecisd, '== E(fci)', efci-ehf0)
    dm1ref, dm2ref = fci.direct_uhf.make_rdm12s(fcivec, h1a.shape[0], mol.nelec)
    rdm1 = make_rdm1(myci.ci, myci.get_nmo(), myci.get_nocc())
    rdm2 = make_rdm2(myci.ci, myci.get_nmo(), myci.get_nocc())
    print('dm1a', abs(dm1ref[0] - rdm1[0]).max())
    print('dm1b', abs(dm1ref[1] - rdm1[1]).max())
    print('dm2aa', abs(dm2ref[0] - rdm2[0]).max())
    print('dm2ab', abs(dm2ref[1] - rdm2[1]).max())
    print('dm2bb', abs(dm2ref[2] - rdm2[2]).max())

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g',}
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    myci = CISD(mf)
    eris = myci.ao2mo()
    ecisd, civec = myci.kernel(eris=eris)
    print(ecisd - -0.048878084082066106)

    nmoa = mf.mo_energy[0].size
    nmob = mf.mo_energy[1].size
    rdm1 = myci.make_rdm1(civec)
    rdm2 = myci.make_rdm2(civec)
    eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0], compact=False).reshape([nmoa]*4)
    eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1], compact=False).reshape([nmob]*4)
    eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                    mf.mo_coeff[1], mf.mo_coeff[1]], compact=False)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    e2 = (numpy.einsum('ij,ji', h1a, rdm1[0]) +
          numpy.einsum('ij,ji', h1b, rdm1[1]) +
          numpy.einsum('ijkl,ijkl', eri_aa, rdm2[0]) * .5 +
          numpy.einsum('ijkl,ijkl', eri_ab, rdm2[1])      +
          numpy.einsum('ijkl,ijkl', eri_bb, rdm2[2]) * .5)
    print(ecisd + mf.e_tot - mol.energy_nuc() - e2)   # = 0

    print(abs(rdm1[0] - (numpy.einsum('ijkk->ij', rdm2[0]) +
                         numpy.einsum('ijkk->ij', rdm2[1]))/(mol.nelectron-1)).sum())
    print(abs(rdm1[1] - (numpy.einsum('ijkk->ij', rdm2[2]) +
                         numpy.einsum('kkij->ij', rdm2[1]))/(mol.nelectron-1)).sum())

