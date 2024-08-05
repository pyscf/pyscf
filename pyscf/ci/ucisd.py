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
Unrestricted CISD
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd
from pyscf.cc import uccsd_rdm
from pyscf.ci import cisd
from pyscf.fci import cistring
from pyscf.cc.ccsd import _unpack_4fold

def make_diagonal(myci, eris):
    nocca, noccb = eris.nocc
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    jdiag_aa = numpy.zeros((nmoa,nmoa))
    jdiag_ab = numpy.zeros((nmoa,nmob))
    jdiag_bb = numpy.zeros((nmob,nmob))
    jdiag_aa[:nocca,:nocca] = numpy.einsum('iijj->ij', eris.oooo)
    jdiag_aa[:nocca,nocca:] = numpy.einsum('iijj->ij', eris.oovv)
    jdiag_aa[nocca:,:nocca] = jdiag_aa[:nocca,nocca:].T
    jdiag_ab[:nocca,:noccb] = numpy.einsum('iijj->ij', eris.ooOO)
    jdiag_ab[:nocca,noccb:] = numpy.einsum('iijj->ij', eris.ooVV)
    jdiag_ab[nocca:,:noccb] = numpy.einsum('iijj->ji', eris.OOvv)
    jdiag_bb[:noccb,:noccb] = numpy.einsum('iijj->ij', eris.OOOO)
    jdiag_bb[:noccb,noccb:] = numpy.einsum('iijj->ij', eris.OOVV)
    jdiag_bb[noccb:,:noccb] = jdiag_bb[:noccb,noccb:].T

    kdiag_aa = numpy.zeros((nmoa,nmoa))
    kdiag_bb = numpy.zeros((nmob,nmob))
    kdiag_aa[:nocca,:nocca] = numpy.einsum('ijji->ij', eris.oooo)
    kdiag_aa[:nocca,nocca:] = numpy.einsum('ijji->ij', eris.ovvo)
    kdiag_aa[nocca:,:nocca] = kdiag_aa[:nocca,nocca:].T
    kdiag_bb[:noccb,:noccb] = numpy.einsum('ijji->ij', eris.OOOO)
    kdiag_bb[:noccb,noccb:] = numpy.einsum('ijji->ij', eris.OVVO)
    kdiag_bb[noccb:,:noccb] = kdiag_bb[:noccb,noccb:].T

#    if eris.vvvv is not None and eris.vvVV is not None and eris.VVVV is not None:
#        def diag_idx(n):
#            idx = numpy.arange(n)
#            return idx * (idx + 1) // 2 + idx
#        jdiag_aa[nocca:,nocca:] = eris.vvvv[diag_idx(nvira)[:,None],diag_idx(nvira)]
#        jdiag_ab[nocca:,noccb:] = eris.vvVV[diag_idx(nvira)[:,None],diag_idx(nvirb)]
#        jdiag_bb[noccb:,noccb:] = eris.VVVV[diag_idx(nvirb)[:,None],diag_idx(nvirb)]
#        kdiag_aa[nocca:,nocca:] = lib.unpack_tril(eris.vvvv.diagonal())
#        kdiag_bb[noccb:,noccb:] = lib.unpack_tril(eris.VVVV.diagonal())

    jkdiag_aa = jdiag_aa - kdiag_aa
    jkdiag_bb = jdiag_bb - kdiag_bb

    mo_ea = eris.focka.diagonal()
    mo_eb = eris.fockb.diagonal()
    ehf = (mo_ea[:nocca].sum() + mo_eb[:noccb].sum()
           - jkdiag_aa[:nocca,:nocca].sum() * .5
           - jdiag_ab[:nocca,:noccb].sum()
           - jkdiag_bb[:noccb,:noccb].sum() * .5)

    dia_a = lib.direct_sum('a-i->ia', mo_ea[nocca:], mo_ea[:nocca])
    dia_a -= jkdiag_aa[:nocca,nocca:]
    dia_b = lib.direct_sum('a-i->ia', mo_eb[noccb:], mo_eb[:noccb])
    dia_b -= jkdiag_bb[:noccb,noccb:]
    e1diag_a = dia_a + ehf
    e1diag_b = dia_b + ehf

    e2diag_aa = lib.direct_sum('ia+jb->ijab', dia_a, dia_a)
    e2diag_aa += ehf
    e2diag_aa += jkdiag_aa[:nocca,:nocca].reshape(nocca,nocca,1,1)
    e2diag_aa -= jkdiag_aa[:nocca,nocca:].reshape(nocca,1,1,nvira)
    e2diag_aa -= jkdiag_aa[:nocca,nocca:].reshape(1,nocca,nvira,1)
    e2diag_aa += jkdiag_aa[nocca:,nocca:].reshape(1,1,nvira,nvira)

    e2diag_ab = lib.direct_sum('ia+jb->ijab', dia_a, dia_b)
    e2diag_ab += ehf
    e2diag_ab += jdiag_ab[:nocca,:noccb].reshape(nocca,noccb,1,1)
    e2diag_ab += jdiag_ab[nocca:,noccb:].reshape(1,1,nvira,nvirb)
    e2diag_ab -= jdiag_ab[:nocca,noccb:].reshape(nocca,1,1,nvirb)
    e2diag_ab -= jdiag_ab[nocca:,:noccb].T.reshape(1,noccb,nvira,1)

    e2diag_bb = lib.direct_sum('ia+jb->ijab', dia_b, dia_b)
    e2diag_bb += ehf
    e2diag_bb += jkdiag_bb[:noccb,:noccb].reshape(noccb,noccb,1,1)
    e2diag_bb -= jkdiag_bb[:noccb,noccb:].reshape(noccb,1,1,nvirb)
    e2diag_bb -= jkdiag_bb[:noccb,noccb:].reshape(1,noccb,nvirb,1)
    e2diag_bb += jkdiag_bb[noccb:,noccb:].reshape(1,1,nvirb,nvirb)

    return amplitudes_to_cisdvec(ehf, (e1diag_a, e1diag_b),
                                 (e2diag_aa, e2diag_ab, e2diag_bb))

def contract(myci, civec, eris):
    nocca, noccb = eris.nocc
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    c0, (c1a,c1b), (c2aa,c2ab,c2bb) = \
            cisdvec_to_amplitudes(civec, (nmoa,nmob), (nocca,noccb), copy=False)

    #:t2 += 0.5*einsum('ijef,abef->ijab', c2, eris.vvvv)
    #:eris_vvvv = ao2mo.restore(1, eris.vvvv, nvira)
    #:eris_vvVV = ucisd_slow._restore(eris.vvVV, nvira, nvirb)
    #:eris_VVVV = ao2mo.restore(1, eris.VVVV, nvirb)
    #:t2aa += lib.einsum('ijef,aebf->ijab', c2aa, eris_vvvv)
    #:t2bb += lib.einsum('ijef,aebf->ijab', c2bb, eris_VVVV)
    #:t2ab += lib.einsum('iJeF,aeBF->iJaB', c2ab, eris_vvVV)
    t2aa, t2ab, t2bb = myci._add_vvvv(None, (c2aa,c2ab,c2bb), eris)
    t2aa *= .25
    t2bb *= .25

    fooa = eris.focka[:nocca,:nocca]
    foob = eris.fockb[:noccb,:noccb]
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    fvoa = eris.focka[nocca:,:nocca]
    fvob = eris.fockb[noccb:,:noccb]
    fvva = eris.focka[nocca:,nocca:]
    fvvb = eris.fockb[noccb:,noccb:]

    t0 = 0
    t1a = 0
    t1b = 0
    eris_oovv = _cp(eris.oovv)
    eris_ooVV = _cp(eris.ooVV)
    eris_OOvv = _cp(eris.OOvv)
    eris_OOVV = _cp(eris.OOVV)
    eris_ovov = _cp(eris.ovov)
    eris_ovOV = _cp(eris.ovOV)
    eris_OVOV = _cp(eris.OVOV)
    #:t2 += eris.oovv * c0
    t2aa += .25 * c0 * eris_ovov.conj().transpose(0,2,1,3)
    t2aa -= .25 * c0 * eris_ovov.conj().transpose(0,2,3,1)
    t2bb += .25 * c0 * eris_OVOV.conj().transpose(0,2,1,3)
    t2bb -= .25 * c0 * eris_OVOV.conj().transpose(0,2,3,1)
    t2ab += c0 * eris_ovOV.conj().transpose(0,2,1,3)
    #:t0 += numpy.einsum('ijab,ijab', eris.oovv, c2) * .25
    t0 += numpy.einsum('iajb,ijab', eris_ovov, c2aa) * .25
    t0 -= numpy.einsum('jaib,ijab', eris_ovov, c2aa) * .25
    t0 += numpy.einsum('iajb,ijab', eris_OVOV, c2bb) * .25
    t0 -= numpy.einsum('jaib,ijab', eris_OVOV, c2bb) * .25
    t0 += numpy.einsum('iajb,ijab', eris_ovOV, c2ab)
    eris_ovov = eris_ovOV = eris_OVOV = None

    #:tmp = einsum('imae,mbej->ijab', c2, eris.ovvo)
    #:tmp = tmp - tmp.transpose(0,1,3,2)
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    eris_ovvo = _cp(eris.ovvo)
    eris_ovVO = _cp(eris.ovVO)
    eris_OVVO = _cp(eris.OVVO)
    ovvo = eris_ovvo - eris_oovv.transpose(0,3,2,1)
    OVVO = eris_OVVO - eris_OOVV.transpose(0,3,2,1)
    t2aa += lib.einsum('imae,jbem->ijab', c2aa, ovvo)
    t2aa += lib.einsum('iMaE,jbEM->ijab', c2ab, eris_ovVO)
    t2bb += lib.einsum('imae,jbem->ijab', c2bb, OVVO)
    t2bb += lib.einsum('mIeA,meBJ->IJAB', c2ab, eris_ovVO)
    t2ab += lib.einsum('imae,meBJ->iJaB', c2aa, eris_ovVO)
    t2ab += lib.einsum('iMaE,MEBJ->iJaB', c2ab, OVVO)
    t2ab += lib.einsum('IMAE,jbEM->jIbA', c2bb, eris_ovVO)
    t2ab += lib.einsum('mIeA,jbem->jIbA', c2ab, ovvo)
    t2ab -= lib.einsum('iMeA,JMeb->iJbA', c2ab, eris_OOvv)
    t2ab -= lib.einsum('mIaE,jmEB->jIaB', c2ab, eris_ooVV)

    #:t1 += einsum('nf,nafi->ia', c1, eris.ovvo)
    t1a += numpy.einsum('nf,nfai->ia', c1a, eris_ovvo)
    t1a -= numpy.einsum('nf,nifa->ia', c1a, eris_oovv)
    t1b += numpy.einsum('nf,nfai->ia', c1b, eris_OVVO)
    t1b -= numpy.einsum('nf,nifa->ia', c1b, eris_OOVV)
    t1b += numpy.einsum('nf,nfai->ia', c1a, eris_ovVO)
    t1a += numpy.einsum('nf,iafn->ia', c1b, eris_ovVO)

    #:t1 -= 0.5*einsum('mnae,mnie->ia', c2, eris.ooov)
    eris_ovoo = _cp(eris.ovoo)
    eris_OVOO = _cp(eris.OVOO)
    eris_OVoo = _cp(eris.OVoo)
    eris_ovOO = _cp(eris.ovOO)
    t1a += lib.einsum('mnae,meni->ia', c2aa, eris_ovoo)
    t1b += lib.einsum('mnae,meni->ia', c2bb, eris_OVOO)
    t1a -= lib.einsum('nMaE,MEni->ia', c2ab, eris_OVoo)
    t1b -= lib.einsum('mNeA,meNI->IA', c2ab, eris_ovOO)
    #:tmp = einsum('ma,mbij->ijab', c1, eris.ovoo)
    #:t2 -= tmp - tmp.transpose(0,1,3,2)
    t2aa -= lib.einsum('ma,jbmi->jiba', c1a, eris_ovoo)
    t2bb -= lib.einsum('ma,jbmi->jiba', c1b, eris_OVOO)
    t2ab -= lib.einsum('ma,JBmi->iJaB', c1a, eris_OVoo)
    t2ab -= lib.einsum('MA,ibMJ->iJbA', c1b, eris_ovOO)

    #:#:t1 -= 0.5*einsum('imef,maef->ia', c2, eris.ovvv)
    #:eris_ovvv = _cp(eris.ovvv)
    #:eris_OVVV = _cp(eris.OVVV)
    #:eris_ovVV = _cp(eris.ovVV)
    #:eris_OVvv = _cp(eris.OVvv)
    #:t1a += lib.einsum('mief,mefa->ia', c2aa, eris_ovvv)
    #:t1b += lib.einsum('MIEF,MEFA->IA', c2bb, eris_OVVV)
    #:t1a += lib.einsum('iMfE,MEaf->ia', c2ab, eris_OVvv)
    #:t1b += lib.einsum('mIeF,meAF->IA', c2ab, eris_ovVV)
    #:#:tmp = einsum('ie,jeba->ijab', c1, numpy.asarray(eris.ovvv).conj())
    #:#:t2 += tmp - tmp.transpose(1,0,2,3)
    #:t2aa += lib.einsum('ie,mbae->imab', c1a, eris_ovvv)
    #:t2bb += lib.einsum('ie,mbae->imab', c1b, eris_OVVV)
    #:t2ab += lib.einsum('ie,MBae->iMaB', c1a, eris_OVvv)
    #:t2ab += lib.einsum('IE,maBE->mIaB', c1b, eris_ovVV)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    if nvira > 0 and nocca > 0:
        blksize = max(int(max_memory*1e6/8/(nvira**2*nocca*2)), 2)
        for p0,p1 in lib.prange(0, nvira, blksize):
            ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
            t1a += lib.einsum('mief,mefa->ia', c2aa[:,:,p0:p1], ovvv)
            t2aa[:,:,p0:p1] += lib.einsum('mbae,ie->miba', ovvv, c1a)
            ovvv = None

    if nvirb > 0 and noccb > 0:
        blksize = max(int(max_memory*1e6/8/(nvirb**2*noccb*2)), 2)
        for p0,p1 in lib.prange(0, nvirb, blksize):
            OVVV = eris.get_OVVV(slice(None), slice(p0,p1))
            t1b += lib.einsum('MIEF,MEFA->IA', c2bb[:,:,p0:p1], OVVV)
            t2bb[:,:,p0:p1] += lib.einsum('mbae,ie->miba', OVVV, c1b)
            OVVV = None

    if nvirb > 0 and nocca > 0:
        blksize = max(int(max_memory*1e6/8/(nvirb**2*nocca*2)), 2)
        for p0,p1 in lib.prange(0, nvira, blksize):
            ovVV = eris.get_ovVV(slice(None), slice(p0,p1))
            t1b += lib.einsum('mIeF,meAF->IA', c2ab[:,:,p0:p1], ovVV)
            t2ab[:,:,p0:p1] += lib.einsum('maBE,IE->mIaB', ovVV, c1b)
            ovVV = None

    if nvira > 0 and noccb > 0:
        blksize = max(int(max_memory*1e6/8/(nvira**2*noccb*2)), 2)
        for p0,p1 in lib.prange(0, nvirb, blksize):
            OVvv = eris.get_OVvv(slice(None), slice(p0,p1))
            t1a += lib.einsum('iMfE,MEaf->ia', c2ab[:,:,:,p0:p1], OVvv)
            t2ab[:,:,:,p0:p1] += lib.einsum('MBae,ie->iMaB', OVvv, c1a)
            OVvv = None

    #:t1  = einsum('ie,ae->ia', c1, fvv)
    t1a += lib.einsum('ie,ae->ia', c1a, fvva)
    t1b += lib.einsum('ie,ae->ia', c1b, fvvb)
    #:t1 -= einsum('ma,mi->ia', c1, foo)
    t1a -= lib.einsum('ma,mi->ia', c1a, fooa)
    t1b -= lib.einsum('ma,mi->ia', c1b, foob)
    #:t1 += einsum('imae,me->ia', c2, fov)
    t1a += numpy.einsum('imae,me->ia', c2aa, fova)
    t1a += numpy.einsum('imae,me->ia', c2ab, fovb)
    t1b += numpy.einsum('imae,me->ia', c2bb, fovb)
    t1b += numpy.einsum('miea,me->ia', c2ab, fova)

    #:tmp = einsum('ijae,be->ijab', c2, fvv)
    #:t2  = tmp - tmp.transpose(0,1,3,2)
    t2aa += lib.einsum('ijae,be->ijab', c2aa, fvva*.5)
    t2bb += lib.einsum('ijae,be->ijab', c2bb, fvvb*.5)
    t2ab += lib.einsum('iJaE,BE->iJaB', c2ab, fvvb)
    t2ab += lib.einsum('iJeA,be->iJbA', c2ab, fvva)
    #:tmp = einsum('imab,mj->ijab', c2, foo)
    #:t2 -= tmp - tmp.transpose(1,0,2,3)
    t2aa -= lib.einsum('imab,mj->ijab', c2aa, fooa*.5)
    t2bb -= lib.einsum('imab,mj->ijab', c2bb, foob*.5)
    t2ab -= lib.einsum('iMaB,MJ->iJaB', c2ab, foob)
    t2ab -= lib.einsum('mIaB,mj->jIaB', c2ab, fooa)

    #:tmp = numpy.einsum('ia,bj->ijab', c1, fvo)
    #:tmp = tmp - tmp.transpose(0,1,3,2)
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    t2aa += numpy.einsum('ia,bj->ijab', c1a, fvoa)
    t2bb += numpy.einsum('ia,bj->ijab', c1b, fvob)
    t2ab += numpy.einsum('ia,bj->ijab', c1a, fvob)
    t2ab += numpy.einsum('ia,bj->jiba', c1b, fvoa)

    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)

    #:t2 += 0.5*einsum('mnab,mnij->ijab', c2, eris.oooo)
    eris_oooo = _cp(eris.oooo)
    eris_OOOO = _cp(eris.OOOO)
    eris_ooOO = _cp(eris.ooOO)
    t2aa += lib.einsum('mnab,minj->ijab', c2aa, eris_oooo)
    t2bb += lib.einsum('mnab,minj->ijab', c2bb, eris_OOOO)
    t2ab += lib.einsum('mNaB,miNJ->iJaB', c2ab, eris_ooOO)

    #:t1 += fov.conj() * c0
    t1a += fova.conj() * c0
    t1b += fovb.conj() * c0
    #:t0  = numpy.einsum('ia,ia', fov, c1)
    t0 += numpy.einsum('ia,ia', fova, c1a)
    t0 += numpy.einsum('ia,ia', fovb, c1b)
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
    civec = numpy.empty(loc[-1], dtype=c2ab.dtype)
    civec[0] = c0
    civec[loc[0]:loc[1]] = c1a.ravel()
    civec[loc[1]:loc[2]] = c1b.ravel()
    civec[loc[2]:loc[3]] = c2ab.ravel()
    lib.take_2d(c2aa.reshape(nocca**2,nvira**2), ooidxa, vvidxa, out=civec[loc[3]:loc[4]])
    lib.take_2d(c2bb.reshape(noccb**2,nvirb**2), ooidxb, vvidxb, out=civec[loc[4]:loc[5]])
    return civec

def cisdvec_to_amplitudes(civec, nmo, nocc, copy=True):
    norba, norbb = nmo
    nocca, noccb = nocc
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
    cp = lambda x: (x.copy() if copy else x)
    c1a = cp(civec[loc[0]:loc[1]].reshape(nocca,nvira))
    c1b = cp(civec[loc[1]:loc[2]].reshape(noccb,nvirb))
    c2ab = cp(civec[loc[2]:loc[3]].reshape(nocca,noccb,nvira,nvirb))
    c2aa = _unpack_4fold(civec[loc[3]:loc[4]], nocca, nvira)
    c2bb = _unpack_4fold(civec[loc[4]:loc[5]], noccb, nvirb)
    return c0, (c1a,c1b), (c2aa,c2ab,c2bb)

def to_fcivec(cisdvec, norb, nelec, frozen=None):
    '''Convert CISD coefficients to FCI coefficients'''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    frozena_mask = numpy.zeros(norb, dtype=bool)
    frozenb_mask = numpy.zeros(norb, dtype=bool)
    if frozen is None:
        nfroza = nfrozb = 0
    elif isinstance(frozen, (int, numpy.integer)):
        nfroza = nfrozb = frozen
        frozena_mask[:frozen] = True
        frozenb_mask[:frozen] = True
    else:
        nfroza = len(frozen[0])
        nfrozb = len(frozen[1])
        frozena_mask[frozen[0]] = True
        frozenb_mask[frozen[1]] = True

#    if nfroza != nfrozb:
#        raise NotImplementedError
    nocca = numpy.count_nonzero(~frozena_mask[:neleca])
    noccb = numpy.count_nonzero(~frozenb_mask[:nelecb])
    nmo = nmoa, nmob = norb - nfroza, norb - nfrozb
    nocc = nocca, noccb
    nvira, nvirb = nmoa - nocca, nmob - noccb

    c0, c1, c2 = cisdvec_to_amplitudes(cisdvec, nmo, nocc, copy=False)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2
    t1addra, t1signa = cisd.tn_addrs_signs(nmoa, nocca, 1)
    t1addrb, t1signb = cisd.tn_addrs_signs(nmob, noccb, 1)

    na = cistring.num_strings(nmoa, nocca)
    nb = cistring.num_strings(nmob, noccb)
    fcivec = numpy.zeros((na,nb))
    fcivec[0,0] = c0
    fcivec[t1addra,0] = c1a.ravel() * t1signa
    fcivec[0,t1addrb] = c1b.ravel() * t1signb
    c2ab = c2ab.transpose(0,2,1,3).reshape(nocca*nvira,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1signa, t1signb, c2ab)
    fcivec[t1addra[:,None],t1addrb] = c2ab

    if nocca > 1 and nvira > 1:
        ooidx = numpy.tril_indices(nocca, -1)
        vvidx = numpy.tril_indices(nvira, -1)
        c2aa = c2aa[ooidx][:,vvidx[0],vvidx[1]]
        t2addra, t2signa = cisd.tn_addrs_signs(nmoa, nocca, 2)
        fcivec[t2addra,0] = c2aa.ravel() * t2signa
    if noccb > 1 and nvirb > 1:
        ooidx = numpy.tril_indices(noccb, -1)
        vvidx = numpy.tril_indices(nvirb, -1)
        c2bb = c2bb[ooidx][:,vvidx[0],vvidx[1]]
        t2addrb, t2signb = cisd.tn_addrs_signs(nmob, noccb, 2)
        fcivec[0,t2addrb] = c2bb.ravel() * t2signb

    if nfroza == nfrozb == 0:
        return fcivec

    assert (norb < 63)

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    na = len(strsa)
    nb = len(strsb)
    count_a = numpy.zeros(na, dtype=int)
    count_b = numpy.zeros(nb, dtype=int)
    parity_a = numpy.zeros(na, dtype=bool)
    parity_b = numpy.zeros(nb, dtype=bool)
    core_a_mask = numpy.ones(na, dtype=bool)
    core_b_mask = numpy.ones(nb, dtype=bool)

    for i in range(norb):
        if frozena_mask[i]:
            if i < neleca:
                core_a_mask &= (strsa & (1 <<i )) != 0
                parity_a ^= (count_a & 1) == 1
            else:
                core_a_mask &= (strsa & (1 << i)) == 0
        else:
            count_a += (strsa & (1 << i)) != 0

        if frozenb_mask[i]:
            if i < nelecb:
                core_b_mask &= (strsb & (1 <<i )) != 0
                parity_b ^= (count_b & 1) == 1
            else:
                core_b_mask &= (strsb & (1 << i)) == 0
        else:
            count_b += (strsb & (1 << i)) != 0

    sub_strsa = strsa[core_a_mask & (count_a == nocca)]
    sub_strsb = strsb[core_b_mask & (count_b == noccb)]
    addrsa = cistring.strs2addr(norb, neleca, sub_strsa)
    addrsb = cistring.strs2addr(norb, nelecb, sub_strsb)
    fcivec1 = numpy.zeros((na,nb))
    fcivec1[addrsa[:,None],addrsb] = fcivec
    fcivec1[parity_a,:] *= -1
    fcivec1[:,parity_b] *= -1
    return fcivec1

def from_fcivec(ci0, norb, nelec, frozen=None):
    '''Extract CISD coefficients from FCI coefficients'''
    if not (frozen is None or frozen == 0):
        raise NotImplementedError

    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    norba = norbb = norb
    nocca, noccb = neleca, nelecb
    nvira = norba - nocca
    nvirb = norbb - noccb
    t1addra, t1signa = cisd.tn_addrs_signs(norba, nocca, 1)
    t1addrb, t1signb = cisd.tn_addrs_signs(norbb, noccb, 1)

    na = cistring.num_strings(norba, nocca)
    nb = cistring.num_strings(norbb, noccb)
    ci0 = ci0.reshape(na,nb)
    c0 = ci0[0,0]
    c1a = (ci0[t1addra,0] * t1signa).reshape(nocca,nvira)
    c1b = (ci0[0,t1addrb] * t1signb).reshape(noccb,nvirb)

    c2ab = numpy.einsum('i,j,ij->ij', t1signa, t1signb, ci0[t1addra[:,None],t1addrb])
    c2ab = c2ab.reshape(nocca,nvira,noccb,nvirb).transpose(0,2,1,3)
    t2addra, t2signa = cisd.tn_addrs_signs(norba, nocca, 2)
    t2addrb, t2signb = cisd.tn_addrs_signs(norbb, noccb, 2)
    c2aa = (ci0[t2addra,0] * t2signa).reshape(nocca*(nocca-1)//2, nvira*(nvira-1)//2)
    c2aa = _unpack_4fold(c2aa, nocca, nvira)
    c2bb = (ci0[0,t2addrb] * t2signb).reshape(noccb*(noccb-1)//2, nvirb*(nvirb-1)//2)
    c2bb = _unpack_4fold(c2bb, noccb, nvirb)

    return amplitudes_to_cisdvec(c0, (c1a,c1b), (c2aa,c2ab,c2bb))

def overlap(cibra, ciket, nmo, nocc, s=None):
    '''Overlap between two CISD wavefunctions.

    Args:
        s : a list of 2D arrays
            The overlap matrix of non-orthogonal one-particle basis
    '''
    if s is None:
        return numpy.dot(cibra, ciket, nmo, nocc)

    if isinstance(nmo, (int, numpy.integer)):
        nmoa = nmob = nmo
    else:
        nmoa, nmob = nmo
    nocca, noccb = nocc
    nvira, nvirb = nmoa - nocca, nmob - noccb

    bra0, bra1, bra2 = cisdvec_to_amplitudes(cibra, (nmoa,nmob), nocc, copy=False)
    ket0, ket1, ket2 = cisdvec_to_amplitudes(ciket, (nmoa,nmob), nocc, copy=False)

    ooidx = numpy.tril_indices(nocca, -1)
    vvidx = numpy.tril_indices(nvira, -1)
    bra2aa = lib.take_2d(bra2[0].reshape(nocca**2,nvira**2),
                         ooidx[0]*nocca+ooidx[1], vvidx[0]*nvira+vvidx[1])
    ket2aa = lib.take_2d(ket2[0].reshape(nocca**2,nvira**2),
                         ooidx[0]*nocca+ooidx[1], vvidx[0]*nvira+vvidx[1])

    ooidx = numpy.tril_indices(noccb, -1)
    vvidx = numpy.tril_indices(nvirb, -1)
    bra2bb = lib.take_2d(bra2[2].reshape(noccb**2,nvirb**2),
                         ooidx[0]*noccb+ooidx[1], vvidx[0]*nvirb+vvidx[1])
    ket2bb = lib.take_2d(ket2[2].reshape(noccb**2,nvirb**2),
                         ooidx[0]*noccb+ooidx[1], vvidx[0]*nvirb+vvidx[1])

    nova = nocca * nvira
    novb = noccb * nvirb
    occlist0a = numpy.arange(nocca).reshape(1,nocca)
    occlist0b = numpy.arange(noccb).reshape(1,noccb)
    occlistsa = numpy.repeat(occlist0a, 1+nova+bra2aa.size, axis=0)
    occlistsb = numpy.repeat(occlist0b, 1+novb+bra2bb.size, axis=0)
    occlist0a = occlistsa[:1]
    occlist1a = occlistsa[1:1+nova]
    occlist2a = occlistsa[1+nova:]
    occlist0b = occlistsb[:1]
    occlist1b = occlistsb[1:1+novb]
    occlist2b = occlistsb[1+novb:]

    ia = 0
    for i in range(nocca):
        for a in range(nocca, nmoa):
            occlist1a[ia,i] = a
            ia += 1
    ia = 0
    for i in range(noccb):
        for a in range(noccb, nmob):
            occlist1b[ia,i] = a
            ia += 1

    ia = 0
    for i in range(nocca):
        for j in range(i):
            for a in range(nocca, nmoa):
                for b in range(nocca, a):
                    occlist2a[ia,i] = a
                    occlist2a[ia,j] = b
                    ia += 1
    ia = 0
    for i in range(noccb):
        for j in range(i):
            for a in range(noccb, nmob):
                for b in range(noccb, a):
                    occlist2b[ia,i] = a
                    occlist2b[ia,j] = b
                    ia += 1

    na = len(occlistsa)
    trans_a = numpy.empty((na,na))
    for i, idx in enumerate(occlistsa):
        s_sub = s[0][idx].T.copy()
        minors = s_sub[occlistsa]
        trans_a[i,:] = numpy.linalg.det(minors)
    nb = len(occlistsb)
    trans_b = numpy.empty((nb,nb))
    for i, idx in enumerate(occlistsb):
        s_sub = s[1][idx].T.copy()
        minors = s_sub[occlistsb]
        trans_b[i,:] = numpy.linalg.det(minors)

    # Mimic the transformation einsum('ab,ap->pb', FCI, trans).
    # The wavefunction FCI has the [excitation_alpha,excitation_beta]
    # representation.  The zero blocks like FCI[S_alpha,D_beta],
    # FCI[D_alpha,D_beta], are explicitly excluded.
    bra_mat = numpy.zeros((na,nb))
    bra_mat[0,0] = bra0
    bra_mat[1:1+nova,0] = bra1[0].ravel()
    bra_mat[0,1:1+novb] = bra1[1].ravel()
    bra_mat[1+nova:,0] = bra2aa.ravel()
    bra_mat[0,1+novb:] = bra2bb.ravel()
    bra_mat[1:1+nova,1:1+novb] = bra2[1].transpose(0,2,1,3).reshape(nova,novb)
    c_s = lib.einsum('ab,ap,bq->pq', bra_mat, trans_a, trans_b)
    ovlp  =  c_s[0,0] * ket0
    ovlp += numpy.dot(c_s[1:1+nova,0], ket1[0].ravel())
    ovlp += numpy.dot(c_s[0,1:1+novb], ket1[1].ravel())
    ovlp += numpy.dot(c_s[1+nova:,0] , ket2aa.ravel())
    ovlp += numpy.dot(c_s[0,1+novb:] , ket2bb.ravel())
    ovlp += numpy.einsum('ijab,iajb->', ket2[1],
                         c_s[1:1+nova,1:1+novb].reshape(nocca,nvira,noccb,nvirb))
    return ovlp


def make_rdm1(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    One-particle spin density matrices dm1a, dm1b in MO basis (the
    occupied-virtual blocks due to the orbital response contribution are not
    included).

    dm1a[p,q] = <q_alpha^\dagger p_alpha>
    dm1b[p,q] = <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    return uccsd_rdm._make_rdm1(myci, d1, with_frozen=True, ao_repr=ao_repr)

def make_rdm2(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    Two-particle spin density matrices dm2aa, dm2ab, dm2bb in MO basis

    dm2aa[p,q,r,s] = <q_alpha^\dagger s_alpha^\dagger r_alpha p_alpha>
    dm2ab[p,q,r,s] = <q_alpha^\dagger s_beta^\dagger r_beta p_alpha>
    dm2bb[p,q,r,s] = <q_beta^\dagger s_beta^\dagger r_beta p_beta>

    (p,q correspond to one particle and r,s correspond to another particle)
    Two-particle density matrix should be contracted to integrals with the
    pattern below to compute energy

    E = numpy.einsum('pqrs,pqrs', eri_aa, dm2_aa)
    E+= numpy.einsum('pqrs,pqrs', eri_ab, dm2_ab)
    E+= numpy.einsum('pqrs,rspq', eri_ba, dm2_ab)
    E+= numpy.einsum('pqrs,pqrs', eri_bb, dm2_bb)

    where eri_aa[p,q,r,s] = (p_alpha q_alpha | r_alpha s_alpha )
    eri_ab[p,q,r,s] = ( p_alpha q_alpha | r_beta s_beta )
    eri_ba[p,q,r,s] = ( p_beta q_beta | r_alpha s_alpha )
    eri_bb[p,q,r,s] = ( p_beta q_beta | r_beta s_beta )
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    d2 = _gamma2_intermediates(myci, civec, nmo, nocc)
    return uccsd_rdm._make_rdm2(myci, d1, d2, with_dm1=True, with_frozen=True,
                                ao_repr=ao_repr)

def _gamma1_intermediates(myci, civec, nmo, nocc):
    nmoa, nmob = nmo
    nocca, noccb = nocc
    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2

    dvoa = c0.conj() * c1a.T
    dvob = c0.conj() * c1b.T
    dvoa += numpy.einsum('jb,ijab->ai', c1a.conj(), c2aa)
    dvoa += numpy.einsum('jb,ijab->ai', c1b.conj(), c2ab)
    dvob += numpy.einsum('jb,ijab->ai', c1b.conj(), c2bb)
    dvob += numpy.einsum('jb,jiba->ai', c1a.conj(), c2ab)
    dova = dvoa.T.conj()
    dovb = dvob.T.conj()

    dooa  =-numpy.einsum('ia,ka->ik', c1a.conj(), c1a)
    doob  =-numpy.einsum('ia,ka->ik', c1b.conj(), c1b)
    dooa -= numpy.einsum('ijab,ikab->jk', c2aa.conj(), c2aa) * .5
    dooa -= numpy.einsum('jiab,kiab->jk', c2ab.conj(), c2ab)
    doob -= numpy.einsum('ijab,ikab->jk', c2bb.conj(), c2bb) * .5
    doob -= numpy.einsum('ijab,ikab->jk', c2ab.conj(), c2ab)

    dvva  = numpy.einsum('ia,ic->ac', c1a, c1a.conj())
    dvvb  = numpy.einsum('ia,ic->ac', c1b, c1b.conj())
    dvva += numpy.einsum('ijab,ijac->bc', c2aa, c2aa.conj()) * .5
    dvva += numpy.einsum('ijba,ijca->bc', c2ab, c2ab.conj())
    dvvb += numpy.einsum('ijba,ijca->bc', c2bb, c2bb.conj()) * .5
    dvvb += numpy.einsum('ijab,ijac->bc', c2ab, c2ab.conj())
    return (dooa, doob), (dova, dovb), (dvoa, dvob), (dvva, dvvb)

def _gamma2_intermediates(myci, civec, nmo, nocc):
    nmoa, nmob = nmo
    nocca, noccb = nocc
    c0, c1, c2 = cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2

    goovv = c0 * c2aa.conj() * .5
    goOvV = c0 * c2ab.conj()
    gOOVV = c0 * c2bb.conj() * .5

    govvv = numpy.einsum('ia,ikcd->kadc', c1a, c2aa.conj()) * .5
    gOvVv = numpy.einsum('ia,ikcd->kadc', c1a, c2ab.conj())
    goVvV = numpy.einsum('ia,kidc->kadc', c1b, c2ab.conj())
    gOVVV = numpy.einsum('ia,ikcd->kadc', c1b, c2bb.conj()) * .5

    gooov = numpy.einsum('ia,klac->klic', c1a, c2aa.conj()) *-.5
    goOoV =-numpy.einsum('ia,klac->klic', c1a, c2ab.conj())
    gOoOv =-numpy.einsum('ia,lkca->klic', c1b, c2ab.conj())
    gOOOV = numpy.einsum('ia,klac->klic', c1b, c2bb.conj()) *-.5

    goooo = numpy.einsum('ijab,klab->ijkl', c2aa.conj(), c2aa) * .25
    goOoO = numpy.einsum('ijab,klab->ijkl', c2ab.conj(), c2ab)
    gOOOO = numpy.einsum('ijab,klab->ijkl', c2bb.conj(), c2bb) * .25
    gvvvv = numpy.einsum('ijab,ijcd->abcd', c2aa, c2aa.conj()) * .25
    gvVvV = numpy.einsum('ijab,ijcd->abcd', c2ab, c2ab.conj())
    gVVVV = numpy.einsum('ijab,ijcd->abcd', c2bb, c2bb.conj()) * .25

    goVoV = numpy.einsum('jIaB,kIaC->jCkB', c2ab.conj(), c2ab)
    gOvOv = numpy.einsum('iJbA,iKcA->JcKb', c2ab.conj(), c2ab)

    govvo = numpy.einsum('ijab,ikac->jcbk', c2aa.conj(), c2aa)
    govvo+= numpy.einsum('jIbA,kIcA->jcbk', c2ab.conj(), c2ab)
    goVvO = numpy.einsum('jIbA,IKAC->jCbK', c2ab.conj(), c2bb)
    goVvO+= numpy.einsum('ijab,iKaC->jCbK', c2aa.conj(), c2ab)
    gOVVO = numpy.einsum('ijab,ikac->jcbk', c2bb.conj(), c2bb)
    gOVVO+= numpy.einsum('iJaB,iKaC->JCBK', c2ab.conj(), c2ab)
    govvo+= numpy.einsum('ia,jb->ibaj', c1a.conj(), c1a)
    goVvO+= numpy.einsum('ia,jb->ibaj', c1a.conj(), c1b)
    gOVVO+= numpy.einsum('ia,jb->ibaj', c1b.conj(), c1b)

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    dovvo = govvo.transpose(0,2,1,3)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    doovv =-dovvo.transpose(0,3,2,1)
    dvvov = None

    dOVOV = gOOVV.transpose(0,2,1,3) - gOOVV.transpose(0,3,1,2)
    dOOOO = gOOOO.transpose(0,2,1,3) - gOOOO.transpose(0,3,1,2)
    dVVVV = gVVVV.transpose(0,2,1,3) - gVVVV.transpose(0,3,1,2)
    dOVVO = gOVVO.transpose(0,2,1,3)
    dOOOV = gOOOV.transpose(0,2,1,3) - gOOOV.transpose(1,2,0,3)
    dOVVV = gOVVV.transpose(0,2,1,3) - gOVVV.transpose(0,3,1,2)
    dOOVV =-dOVVO.transpose(0,3,2,1)
    dVVOV = None

    dovOV = goOvV.transpose(0,2,1,3)
    dooOO = goOoO.transpose(0,2,1,3)
    dvvVV = gvVvV.transpose(0,2,1,3)
    dovVO = goVvO.transpose(0,2,1,3)
    dooOV = goOoV.transpose(0,2,1,3)
    dovVV = goVvV.transpose(0,2,1,3)
    dooVV = goVoV.transpose(0,2,1,3)
    dooVV = -(dooVV + dooVV.transpose(1,0,3,2).conj()) * .5
    dvvOV = None

    dOVov = None
    dOOoo = None
    dVVvv = None
    dOVvo = dovVO.transpose(3,2,1,0).conj()
    dOOov = gOoOv.transpose(0,2,1,3)
    dOVvv = gOvVv.transpose(0,2,1,3)
    dOOvv = gOvOv.transpose(0,2,1,3)
    dOOvv =-(dOOvv + dOOvv.transpose(1,0,3,2).conj()) * .5
    dVVov = None

    return ((dovov, dovOV, dOVov, dOVOV),
            (dvvvv, dvvVV, dVVvv, dVVVV),
            (doooo, dooOO, dOOoo, dOOOO),
            (doovv, dooVV, dOOvv, dOOVV),
            (dovvo, dovVO, dOVvo, dOVVO),
            (dvvov, dvvOV, dVVov, dVVOV),
            (dovvv, dovVV, dOVvv, dOVVV),
            (dooov, dooOV, dOOov, dOOOV))

def trans_rdm1(myci, cibra, ciket, nmo=None, nocc=None):
    r'''
    One-particle spin density matrices dm1a, dm1b in MO basis (the
    occupied-virtual blocks due to the orbital response contribution are not
    included).

    dm1a[p,q] = <q_alpha^\dagger p_alpha>
    dm1b[p,q] = <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    '''
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    c0bra, c1bra, c2bra = myci.cisdvec_to_amplitudes(cibra, nmo, nocc, copy=False)
    c0ket, c1ket, c2ket = myci.cisdvec_to_amplitudes(ciket, nmo, nocc, copy=False)

    nmoa, nmob = nmo
    nocca, noccb = nocc
    bra1a, bra1b = c1bra
    bra2aa, bra2ab, bra2bb = c2bra
    ket1a, ket1b = c1ket
    ket2aa, ket2ab, ket2bb = c2ket

    dvoa = c0bra.conj() * ket1a.T
    dvob = c0bra.conj() * ket1b.T
    dvoa += numpy.einsum('jb,ijab->ai', bra1a.conj(), ket2aa)
    dvoa += numpy.einsum('jb,ijab->ai', bra1b.conj(), ket2ab)
    dvob += numpy.einsum('jb,ijab->ai', bra1b.conj(), ket2bb)
    dvob += numpy.einsum('jb,jiba->ai', bra1a.conj(), ket2ab)

    dova = c0ket * bra1a.conj()
    dovb = c0ket * bra1b.conj()
    dova += numpy.einsum('jb,ijab->ia', ket1a.conj(), bra2aa)
    dova += numpy.einsum('jb,ijab->ia', ket1b.conj(), bra2ab)
    dovb += numpy.einsum('jb,ijab->ia', ket1b.conj(), bra2bb)
    dovb += numpy.einsum('jb,jiba->ia', ket1a.conj(), bra2ab)

    dooa  =-numpy.einsum('ia,ka->ik', bra1a.conj(), ket1a)
    doob  =-numpy.einsum('ia,ka->ik', bra1b.conj(), ket1b)
    dooa -= numpy.einsum('ijab,ikab->jk', bra2aa.conj(), ket2aa) * .5
    dooa -= numpy.einsum('jiab,kiab->jk', bra2ab.conj(), ket2ab)
    doob -= numpy.einsum('ijab,ikab->jk', bra2bb.conj(), ket2bb) * .5
    doob -= numpy.einsum('ijab,ikab->jk', bra2ab.conj(), ket2ab)

    dvva  = numpy.einsum('ia,ic->ac', ket1a, bra1a.conj())
    dvvb  = numpy.einsum('ia,ic->ac', ket1b, bra1b.conj())
    dvva += numpy.einsum('ijab,ijac->bc', ket2aa, bra2aa.conj()) * .5
    dvva += numpy.einsum('ijba,ijca->bc', ket2ab, bra2ab.conj())
    dvvb += numpy.einsum('ijba,ijca->bc', ket2bb, bra2bb.conj()) * .5
    dvvb += numpy.einsum('ijab,ijac->bc', ket2ab, bra2ab.conj())

    dm1a = numpy.empty((nmoa,nmoa), dtype=dooa.dtype)
    dm1a[:nocca,:nocca] = dooa
    dm1a[:nocca,nocca:] = dova
    dm1a[nocca:,:nocca] = dvoa
    dm1a[nocca:,nocca:] = dvva
    norm = numpy.dot(cibra, ciket)
    dm1a[numpy.diag_indices(nocca)] += norm

    dm1b = numpy.empty((nmob,nmob), dtype=dooa.dtype)
    dm1b[:noccb,:noccb] = doob
    dm1b[:noccb,noccb:] = dovb
    dm1b[noccb:,:noccb] = dvob
    dm1b[noccb:,noccb:] = dvvb
    dm1b[numpy.diag_indices(noccb)] += norm

    if myci.frozen is not None:
        nmoa = myci.mo_occ[0].size
        nmob = myci.mo_occ[1].size
        nocca = numpy.count_nonzero(myci.mo_occ[0] > 0)
        noccb = numpy.count_nonzero(myci.mo_occ[1] > 0)
        rdm1a = numpy.zeros((nmoa,nmoa), dtype=dm1a.dtype)
        rdm1b = numpy.zeros((nmob,nmob), dtype=dm1b.dtype)
        rdm1a[numpy.diag_indices(nocca)] = norm
        rdm1b[numpy.diag_indices(noccb)] = norm
        moidx = myci.get_frozen_mask()
        moidxa = numpy.where(moidx[0])[0]
        moidxb = numpy.where(moidx[1])[0]
        rdm1a[moidxa[:,None],moidxa] = dm1a
        rdm1b[moidxb[:,None],moidxb] = dm1b
        dm1a = rdm1a
        dm1b = rdm1b
    return dm1a, dm1b


class UCISD(cisd.CISD):

    def vector_size(self):
        norba, norbb = self.nmo
        nocca, noccb = self.nocc
        nvira = norba - nocca
        nvirb = norbb - noccb
        nooa = nocca * (nocca-1) // 2
        nvva = nvira * (nvira-1) // 2
        noob = noccb * (noccb-1) // 2
        nvvb = nvirb * (nvirb-1) // 2
        size = (1 + nocca*nvira + noccb*nvirb +
                nocca*noccb*nvira*nvirb + nooa*nvva + noob*nvvb)
        return size

    get_nocc = uccsd.get_nocc
    get_nmo = uccsd.get_nmo
    get_frozen_mask = uccsd.get_frozen_mask

    def get_init_guess(self, eris=None, nroots=1, diag=None):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        nocca, noccb = self.nocc
        mo_ea, mo_eb = eris.mo_energy
        eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
        eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]
        t1a = eris.focka[:nocca,nocca:].conj() / eia_a
        t1b = eris.fockb[:noccb,noccb:].conj() / eia_b

        eris_ovov = _cp(eris.ovov)
        eris_ovOV = _cp(eris.ovOV)
        eris_OVOV = _cp(eris.OVOV)
        t2aa = eris_ovov.transpose(0,2,1,3) - eris_ovov.transpose(0,2,3,1)
        t2bb = eris_OVOV.transpose(0,2,1,3) - eris_OVOV.transpose(0,2,3,1)
        t2ab = eris_ovOV.transpose(0,2,1,3).copy()
        t2aa = t2aa.conj()
        t2ab = t2ab.conj()
        t2bb = t2bb.conj()
        t2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
        t2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
        t2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

        emp2  = numpy.einsum('iajb,ijab', eris_ovov, t2aa) * .25
        emp2 -= numpy.einsum('jaib,ijab', eris_ovov, t2aa) * .25
        emp2 += numpy.einsum('iajb,ijab', eris_OVOV, t2bb) * .25
        emp2 -= numpy.einsum('jaib,ijab', eris_OVOV, t2bb) * .25
        emp2 += numpy.einsum('iajb,ijab', eris_ovOV, t2ab)
        self.emp2 = emp2.real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)

        if abs(emp2) < 1e-3 and (abs(t1a).sum()+abs(t1b).sum()) < 1e-3:
            t1a = 1e-1 / eia_a
            t1b = 1e-1 / eia_b

        ci_guess = amplitudes_to_cisdvec(1, (t1a,t1b), (t2aa,t2ab,t2bb))

        if nroots > 1:
            civec_size = ci_guess.size
            ci1_size = t1a.size + t1b.size
            dtype = ci_guess.dtype
            nroots = min(ci1_size+1, nroots)

            if diag is None:
                idx = range(1, nroots)
            else:
                idx = diag[:ci1_size+1].argsort()[1:nroots]  # exclude HF determinant

            ci_guess = [ci_guess]
            for i in idx:
                g = numpy.zeros(civec_size, dtype)
                g[i] = 1.0
                ci_guess.append(g)

        return self.emp2, ci_guess

    contract = contract
    make_diagonal = make_diagonal
    _dot = None
    _add_vvvv = uccsd._add_vvvv

    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.get_nmo()
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmoa * (nmoa+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return uccsd._make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            return uccsd._make_df_eris_outcore(self, mo_coeff)
        else:
            return uccsd._make_eris_outcore(self, mo_coeff)

    def to_fcivec(self, cisdvec, nmo=None, nocc=None):
        return to_fcivec(cisdvec, nmo, nocc)

    def from_fcivec(self, fcivec, nmo=None, nocc=None):
        return from_fcivec(fcivec, nmo, nocc)

    def amplitudes_to_cisdvec(self, c0, c1, c2):
        return amplitudes_to_cisdvec(c0, c1, c2)

    def cisdvec_to_amplitudes(self, civec, nmo=None, nocc=None, copy=True):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return cisdvec_to_amplitudes(civec, nmo, nocc, copy=copy)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2
    trans_rdm1 = trans_rdm1

    def nuc_grad_method(self):
        from pyscf.grad import ucisd
        return ucisd.Gradients(self)

CISD = UCISD

from pyscf import scf
scf.uhf.UHF.CISD = lib.class_as_method(CISD)

def _cp(a):
    return numpy.asarray(a, order='C')
