#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Unrestricted CISD
'''

import time
from functools import reduce
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import uccsd
from pyscf.ci import cisd
from pyscf.cc.rccsd import _unpack_4fold, _mem_usage
from pyscf.ci.ucisd_slow import from_fci, to_fci
from pyscf.ci.ucisd_slow import make_rdm1, make_rdm2

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
    jdiag_aa[:nocca,:nocca] = numpy.einsum('iijj->ij', eris.oooo)
    jdiag_aa[:nocca,nocca:] = numpy.einsum('iijj->ji', eris.vvoo)
    jdiag_aa[nocca:,:nocca] = jdiag_aa[:nocca,nocca:].T
    jdiag_ab[:nocca,:noccb] = numpy.einsum('iijj->ij', eris.ooOO)
    jdiag_ab[:nocca,noccb:] = numpy.einsum('iijj->ji', eris.VVoo)
    jdiag_ab[nocca:,:noccb] = numpy.einsum('iijj->ij', eris.vvOO)
    jdiag_bb[:noccb,:noccb] = numpy.einsum('iijj->ij', eris.OOOO)
    jdiag_bb[:noccb,noccb:] = numpy.einsum('iijj->ji', eris.VVOO)
    jdiag_bb[noccb:,:noccb] = jdiag_bb[:noccb,noccb:].T

    kdiag_aa = numpy.zeros((nmoa,nmoa))
    kdiag_bb = numpy.zeros((nmob,nmob))
    kdiag_aa[:nocca,:nocca] = numpy.einsum('ijji->ij', eris.oooo)
    kdiag_aa[:nocca,nocca:] = numpy.einsum('ijji->ji', eris.voov)
    kdiag_aa[nocca:,:nocca] = kdiag_aa[:nocca,nocca:].T
    kdiag_bb[:noccb,:noccb] = numpy.einsum('ijji->ij', eris.OOOO)
    kdiag_bb[:noccb,noccb:] = numpy.einsum('ijji->ji', eris.VOOV)
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

    t0 = 0
    t1a = 0
    t1b = 0
    t2aa = 0
    t2ab = 0
    t2bb = 0
    eris_vvoo = _cp(eris.vvoo)
    eris_VVoo = _cp(eris.VVoo)
    eris_vvOO = _cp(eris.vvOO)
    eris_VVOO = _cp(eris.VVOO)
    eris_voov = _cp(eris.voov)
    eris_voOV = _cp(eris.voOV)
    eris_VOOV = _cp(eris.VOOV)
    #:t2 += eris.oovv * c0
    t2aa += .25 * c0 * eris_voov.transpose(1,2,0,3)
    t2aa -= .25 * c0 * eris_voov.transpose(1,2,3,0)
    t2bb += .25 * c0 * eris_VOOV.transpose(1,2,0,3)
    t2bb -= .25 * c0 * eris_VOOV.transpose(1,2,3,0)
    t2ab += c0 * eris_voOV.transpose(1,2,0,3)
    #:t0 += numpy.einsum('ijab,ijab', eris.oovv, c2) * .25
    t0 += numpy.einsum('aijb,ijab', eris.voov, c2aa) * .25
    t0 -= numpy.einsum('ajib,ijab', eris.voov, c2aa) * .25
    t0 += numpy.einsum('aijb,ijab', eris.VOOV, c2bb) * .25
    t0 -= numpy.einsum('ajib,ijab', eris.VOOV, c2bb) * .25
    t0 += numpy.einsum('aijb,ijab', eris.voOV, c2ab)

    #:tmp = einsum('imae,mbej->ijab', c2, eris.ovvo)
    #:tmp = tmp - tmp.transpose(0,1,3,2)
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    voov = eris_voov - eris_vvoo.transpose(0,3,2,1)
    VOOV = eris_VOOV - eris_VVOO.transpose(0,3,2,1)
    t2aa += lib.einsum('imae,bjme->ijab', c2aa, voov)
    t2aa += lib.einsum('iMaE,bjME->ijab', c2ab, eris_voOV)
    t2bb += lib.einsum('imae,bjme->ijab', c2bb, VOOV)
    t2bb += lib.einsum('mIeA,emJB->IJAB', c2ab, eris_voOV)
    t2ab += lib.einsum('imae,emJB->iJaB', c2aa, eris_voOV)
    t2ab += lib.einsum('iMaE,EMJB->iJaB', c2ab, VOOV)
    t2ab += lib.einsum('IMAE,bjME->jIbA', c2bb, eris_voOV)
    t2ab += lib.einsum('mIeA,bjme->jIbA', c2ab, voov)
    t2ab -= lib.einsum('iMeA,ebJM->iJbA', c2ab, eris_vvOO)
    t2ab -= lib.einsum('mIaE,EBjm->jIaB', c2ab, eris_VVoo)

    #:t1 += einsum('nf,nafi->ia', c1, eris.ovvo)
    t1a += numpy.einsum('nf,ainf->ia', c1a, eris_voov)
    t1a -= numpy.einsum('nf,fani->ia', c1a, eris_vvoo)
    t1b += numpy.einsum('nf,ainf->ia', c1b, eris_VOOV)
    t1b -= numpy.einsum('nf,fani->ia', c1b, eris_VVOO)
    t1b += numpy.einsum('nf,fnia->ia', c1a, eris_voOV)
    t1a += numpy.einsum('nf,ainf->ia', c1b, eris_voOV)

    #:t1 -= 0.5*einsum('mnae,mnie->ia', c2, eris.ooov)
    eris_vooo = _cp(eris.vooo)
    eris_VOOO = _cp(eris.VOOO)
    eris_VOoo = _cp(eris.VOoo)
    eris_voOO = _cp(eris.voOO)
    t1a += lib.einsum('mnae,emni->ia', c2aa, eris_vooo)
    t1b += lib.einsum('mnae,emni->ia', c2bb, eris_VOOO)
    t1a -= lib.einsum('nMaE,EMni->ia', c2ab, eris_VOoo)
    t1b -= lib.einsum('mNeA,emNI->IA', c2ab, eris_voOO)
    #:tmp = einsum('ma,mbij->ijab', c1, eris.ovoo)
    #:t2 -= tmp - tmp.transpose(0,1,3,2)
    t2aa -= lib.einsum('ma,bjmi->jiba', c1a, eris_vooo)
    t2bb -= lib.einsum('ma,bjmi->jiba', c1b, eris_VOOO)
    t2ab -= lib.einsum('ma,BJmi->iJaB', c1a, eris_VOoo)
    t2ab -= lib.einsum('MA,biMJ->iJbA', c1b, eris_voOO)

    #:#:t1 -= 0.5*einsum('imef,maef->ia', c2, eris.ovvv)
    #:eris_vovv = _cp(eris.vovv)
    #:eris_VOVV = _cp(eris.VOVV)
    #:eris_voVV = _cp(eris.voVV)
    #:eris_VOvv = _cp(eris.VOvv)
    #:t1a += lib.einsum('mief,emfa->ia', c2aa, eris_vovv)
    #:t1b += lib.einsum('MIEF,EMFA->IA', c2bb, eris_VOVV)
    #:t1a += lib.einsum('iMfE,EMaf->ia', c2ab, eris_VOvv)
    #:t1b += lib.einsum('mIeF,emAF->IA', c2ab, eris_voVV)
    #:#:tmp = einsum('ie,jeba->ijab', c1, numpy.asarray(eris.ovvv).conj())
    #:#:t2 += tmp - tmp.transpose(1,0,2,3)
    #:t2aa += lib.einsum('ie,bmae->imab', c1a, eris_vovv)
    #:t2bb += lib.einsum('ie,bmae->imab', c1b, eris_VOVV)
    #:t2ab += lib.einsum('ie,BMae->iMaB', c1a, eris_VOvv)
    #:t2ab += lib.einsum('IE,amBE->mIaB', c1b, eris_voVV)
    if nvira > 0 and nocca > 0:
        mem_now = lib.current_memory()[0]
        max_memory = myci.max_memory - mem_now
        blksize = max(int(max_memory*1e6/8/(nvira**2*nocca*2)), 2)
        for p0,p1 in lib.prange(0, nvira, blksize):
            vovv = _cp(eris.vovv[p0:p1]).reshape((p1-p0)*nocca,-1)
            vovv = lib.unpack_tril(vovv).reshape(p1-p0,nocca,nvira,nvira)
            t1a += lib.einsum('mief,emfa->ia', c2aa[:,:,p0:p1], vovv)
            t2aa[:,:,p0:p1] += lib.einsum('ie,bmae->miba', c1a, vovv)
            vovv = None

    if nvirb > 0 and noccb > 0:
        mem_now = lib.current_memory()[0]
        max_memory = myci.max_memory - mem_now
        blksize = max(int(max_memory*1e6/8/(nvirb**2*noccb*2)), 2)
        for p0,p1 in lib.prange(0, nvirb, blksize):
            VOVV = _cp(eris.VOVV[p0:p1]).reshape((p1-p0)*noccb,-1)
            VOVV = lib.unpack_tril(VOVV).reshape(p1-p0,noccb,nvirb,nvirb)
            t1b += lib.einsum('MIEF,EMFA->IA', c2bb[:,:,p0:p1], VOVV)
            t2bb[:,:,p0:p1] += lib.einsum('ie,bmae->miba', c1b, VOVV)
            VOVV = None

    if nvirb > 0 and nocca > 0:
        mem_now = lib.current_memory()[0]
        max_memory = myci.max_memory - mem_now
        blksize = max(int(max_memory*1e6/8/(nvirb**2*nocca*2)), 2)
        for p0,p1 in lib.prange(0, nvira, blksize):
            voVV = _cp(eris.voVV[p0:p1]).reshape((p1-p0)*nocca,-1)
            voVV = lib.unpack_tril(voVV).reshape(p1-p0,nocca,nvirb,nvirb)
            t1b += lib.einsum('mIeF,emAF->IA', c2ab[:,:,p0:p1], voVV)
            t2ab[:,:,p0:p1] += lib.einsum('IE,amBE->mIaB', c1b, voVV)
            voVV = None

    if nvira > 0 and noccb > 0:
        mem_now = lib.current_memory()[0]
        max_memory = myci.max_memory - mem_now
        blksize = max(int(max_memory*1e6/8/(nvira**2*noccb*2)), 2)
        for p0,p1 in lib.prange(0, nvirb, blksize):
            VOvv = _cp(eris.VOvv[p0:p1]).reshape((p1-p0)*noccb,-1)
            VOvv = lib.unpack_tril(VOvv).reshape(p1-p0,noccb,nvira,nvira)
            t1a += lib.einsum('iMfE,EMaf->ia', c2ab[:,:,:,p0:p1], VOvv)
            t2ab[:,:,:,p0:p1] += lib.einsum('ie,BMae->iMaB', c1a, VOvv)
            VOvv = None

    #:t1  = einsum('ie,ae->ia', c1, fvv)
    t1a += einsum('ie,ae->ia', c1a, fvva)
    t1b += einsum('ie,ae->ia', c1b, fvvb)
    #:t1 -= einsum('ma,mi->ia', c1, foo)
    t1a -=einsum('ma,mi->ia', c1a, fooa)
    t1b -=einsum('ma,mi->ia', c1b, foob)
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

    #:tmp = numpy.einsum('ia,jb->ijab', c1, fov)
    #:tmp = tmp - tmp.transpose(0,1,3,2)
    #:t2 += tmp - tmp.transpose(1,0,2,3)
    t2aa += numpy.einsum('ia,jb->ijab', c1a, fova)
    t2bb += numpy.einsum('ia,jb->ijab', c1b, fovb)
    t2ab += numpy.einsum('ia,jb->ijab', c1a, fovb)
    t2ab += numpy.einsum('ia,jb->jiba', c1b, fova)

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

    #:t2 += 0.5*einsum('ijef,abef->ijab', c2, eris.vvvv)
    #:eris_vvvv = ao2mo.restore(1, eris.vvvv, nvira)
    #:eris_vvVV = ucisd_slow._restore(eris.vvVV, nvira, nvirb)
    #:eris_VVVV = ao2mo.restore(1, eris.VVVV, nvirb)
    #:t2aa += lib.einsum('ijef,aebf->ijab', c2aa, eris_vvvv)
    #:t2bb += lib.einsum('ijef,aebf->ijab', c2bb, eris_VVVV)
    #:t2ab += lib.einsum('iJeF,aeBF->iJaB', c2ab, eris_vvVV)
    uccsd._add_vvvv_(myci, (c2aa,c2ab,c2bb), eris, (t2aa,t2ab,t2bb))

    #:t1 += fov * c0
    t1a += fova * c0
    t1b += fovb * c0
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


class UCISD(cisd.CISD):

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
            logger.info(self, 'UCISD converged')
        else:
            logger.info(self, 'UCISD not converged')
        if self.nroots > 1:
            for i,e in enumerate(self.e_tot):
                logger.note(self, 'UCISD root %d  E = %.16g', i, e)
        else:
            logger.note(self, 'E(UCISD) = %.16g  E_corr = %.16g',
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
        nocc = self.nocc
        nvir = self.nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            raise NotImplementedError

        else:
            return _make_eris_outcore(self, mo_coeff)

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

CISD = UCISD


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

def _make_eris_incore(myci, mo_coeff=None):
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
    eris.vooo = eri_aa[nocca:,:nocca,:nocca,:nocca].copy()
    eris.voov = eri_aa[nocca:,:nocca,:nocca,nocca:].copy()
    eris.vvoo = eri_aa[nocca:,nocca:,:nocca,:nocca].copy()
    vovv = eri_aa[nocca:,:nocca,nocca:,nocca:].reshape(-1,nvira,nvira)
    eris.vovv = lib.pack_tril(vovv).reshape(nvira,nocca,nvira*(nvira+1)//2)
    eris.vvvv = ao2mo.restore(4, eri_aa[nocca:,nocca:,nocca:,nocca:].copy(), nvira)
    vovv = eri_aa = None

    eri_bb = ao2mo.restore(1, ao2mo.full(myci._scf._eri, mob), nmob)
    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.VOOO = eri_bb[noccb:,:noccb,:noccb,:noccb].copy()
    eris.VOOV = eri_bb[noccb:,:noccb,:noccb,noccb:].copy()
    eris.VVOO = eri_bb[noccb:,noccb:,:noccb,:noccb].copy()
    VOVV = eri_bb[noccb:,:noccb,noccb:,noccb:].reshape(-1,nvirb,nvirb)
    eris.VOVV = lib.pack_tril(VOVV).reshape(nvirb,noccb,nvirb*(nvirb+1)//2)
    eris.VVVV = ao2mo.restore(4, eri_bb[noccb:,noccb:,noccb:,noccb:].copy(), nvirb)
    VOVV = eri_bb = None

    eri_ab = ao2mo.general(myci._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.voOO = eri_ab[nocca:,:nocca,:noccb,:noccb].copy()
    eris.voOV = eri_ab[nocca:,:nocca,:noccb,noccb:].copy()
    eris.vvOO = eri_ab[nocca:,nocca:,:noccb,:noccb].copy()
    voVV = eri_ab[nocca:,:nocca,noccb:,noccb:].reshape(nocca*nvira,nvirb,nvirb)
    eris.voVV = lib.pack_tril(voVV).reshape(nvira,nocca,nvirb*(nvirb+1)//2)
    voVV = None
    vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].reshape(nvira**2,nvirb**2)
    idxa = numpy.tril_indices(nvira)
    idxb = numpy.tril_indices(nvirb)
    eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

    eri_ba = lib.transpose(eri_ab.reshape(nmoa**2,nmob**2))
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eris.VOoo = eri_ba[noccb:,:noccb,:nocca,:nocca].copy()
    eris.VVoo = eri_ba[noccb:,noccb:,:nocca,:nocca].copy()
    VOvv = eri_ba[noccb:,:noccb,nocca:,nocca:].reshape(noccb*nvirb,nvira,nvira)
    eris.VOvv = lib.pack_tril(VOvv).reshape(nvirb,noccb,nvira*(nvira+1)//2)
    VOvv = None
    eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy() #X
    return eris

def _make_eris_outcore(myci, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(myci.stdout, myci.verbose)
    eris = _UCISD_ERIs(myci, mo_coeff)

    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    moa, mob = eris.mo_coeff
    mol = myci.mol

    eris.feri = lib.H5TmpFile()
    dtype = 'f8'
    eris.oooo = eris.feri.create_dataset('oooo', (nocca,nocca,nocca,nocca), dtype)
    eris.vooo = eris.feri.create_dataset('vooo', (nvira,nocca,nocca,nocca), dtype)
    eris.voov = eris.feri.create_dataset('voov', (nvira,nocca,nocca,nvira), dtype)
    eris.vvoo = eris.feri.create_dataset('vvoo', (nvira,nvira,nocca,nocca), dtype)
    eris.vovv = eris.feri.create_dataset('vovv', (nvira,nocca,nvira*(nvira+1)//2), dtype)
    #eris.vvvv = eris.feri.create_dataset('vvvv', (nvira,nvira,nvira,nvira), dtype)
    eris.OOOO = eris.feri.create_dataset('OOOO', (noccb,noccb,noccb,noccb), dtype)
    eris.VOOO = eris.feri.create_dataset('VOOO', (nvirb,noccb,noccb,noccb), dtype)
    eris.VOOV = eris.feri.create_dataset('VOOV', (nvirb,noccb,noccb,nvirb), dtype)
    eris.VVOO = eris.feri.create_dataset('VVOO', (nvirb,nvirb,noccb,noccb), dtype)
    eris.VOVV = eris.feri.create_dataset('VOVV', (nvirb,noccb,nvirb*(nvirb+1)//2), dtype)
    #eris.VVVV = eris.feri.create_dataset('VVVV', (nvirb,nvirb,nvirb,nvirb), dtype)
    eris.ooOO = eris.feri.create_dataset('ooOO', (nocca,nocca,noccb,noccb), dtype)
    eris.voOO = eris.feri.create_dataset('voOO', (nvira,nocca,noccb,noccb), dtype)
    eris.voOV = eris.feri.create_dataset('voOV', (nvira,nocca,noccb,nvirb), dtype)
    eris.vvOO = eris.feri.create_dataset('vvOO', (nvira,nvira,noccb,noccb), dtype)
    eris.voVV = eris.feri.create_dataset('voVV', (nvira,nocca,nvirb*(nvirb+1)//2), dtype)
    #eris.vvVV = eris.feri.create_dataset('vvVV', (nvira,nvira,nvirb,nvirb), dtype)
    eris.VOoo = eris.feri.create_dataset('VOoo', (nvirb,noccb,nocca,nocca), dtype)
    eris.VVoo = eris.feri.create_dataset('VVoo', (nvirb,nvirb,nocca,nocca), dtype)
    eris.VOvv = eris.feri.create_dataset('VOvv', (nvirb,noccb,nvira*(nvira+1)//2), dtype)

    cput1 = time.clock(), time.time()
    # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
    tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    ao2mo.general(mol, (moa,moa[:,:nocca],moa,moa), tmpfile2.name, 'aa')
    with h5py.File(tmpfile2.name) as f:
        buf = lib.unpack_tril(f['aa'][:nocca*nocca])
        buf = buf.reshape(nocca,nocca,nmoa,nmoa)
        eris.oooo[:] = buf[:,:,:nocca,:nocca]
        oovv = buf[:,:,nocca:,nocca:].reshape(nocca**2,nvira**2)
        eris.vvoo[:] = lib.transpose(oovv).reshape(nvira,nvira,nocca,nocca)
        buf = oovv = None
        for i0, i1 in lib.prange(0, nvira, nocca):
            buf = lib.unpack_tril(f['aa'][(nocca+i0)*nocca:(nocca+i1)*nocca])
            eris.vovv[i0:i1] = lib.pack_tril(buf[:,nocca:,nocca:]).reshape(i1-i0,nocca,-1)
            buf = buf.reshape(i1-i0,nocca,nmoa,nmoa)
            eris.vooo[i0:i1] = buf[:,:nocca,:nocca,:nocca]
            eris.voov[i0:i1] = buf[:,:nocca,:nocca,nocca:]
            buf = None
        del(f['aa'])

    if noccb > 0:
        ao2mo.general(mol, (mob,mob[:,:noccb],mob,mob), tmpfile2.name, 'bb')
        with h5py.File(tmpfile2.name) as f:
            buf = lib.unpack_tril(f['bb'][:noccb*noccb])
            buf = buf.reshape(noccb,noccb,nmob,nmob)
            eris.OOOO[:] = buf[:,:,:noccb,:noccb]
            oovv = buf[:,:,noccb:,noccb:].reshape(noccb**2,nvirb**2)
            eris.VVOO[:] = lib.transpose(oovv).reshape(nvirb,nvirb,noccb,noccb)
            buf = oovv = None
            for i0, i1 in lib.prange(0, nvirb, noccb):
                buf = lib.unpack_tril(f['bb'][(noccb+i0)*noccb:(noccb+i1)*noccb])
                eris.VOVV[i0:i1] = lib.pack_tril(buf[:,noccb:,noccb:]).reshape(i1-i0,noccb,-1)
                buf = buf.reshape(i1-i0,noccb,nmob,nmob)
                eris.VOOO[i0:i1] = buf[:,:noccb,:noccb,:noccb]
                eris.VOOV[i0:i1] = buf[:,:noccb,:noccb,noccb:]
                buf = None
            del(f['bb'])

    ao2mo.general(mol, (moa,moa[:,:nocca],mob,mob), tmpfile2.name, 'ab')
    with h5py.File(tmpfile2.name) as f:
        buf = lib.unpack_tril(f['ab'][:nocca*nocca])
        buf = buf.reshape(nocca,nocca,nmob,nmob)
        eris.ooOO[:] = buf[:,:,:noccb,:noccb]
        oovv = buf[:,:,noccb:,noccb:].reshape(nocca**2,nvirb**2)
        eris.VVoo[:] = lib.transpose(oovv).reshape(nvirb,nvirb,nocca,nocca)
        buf = oovv = None
        for i0, i1 in lib.prange(0, nvira, nocca):
            buf = lib.unpack_tril(f['ab'][(nocca+i0)*nocca:(nocca+i1)*nocca])
            eris.voVV[i0:i1] = lib.pack_tril(buf[:,noccb:,noccb:]).reshape(i1-i0,nocca,-1)
            buf = buf.reshape(i1-i0,nocca,nmob,nmob)
            eris.voOO[i0:i1] = buf[:,:nocca,:noccb,:noccb]
            eris.voOV[i0:i1] = buf[:,:nocca,:noccb,noccb:]
            buf = None
        del(f['ab'])

    if noccb > 0:
        ao2mo.general(mol, (mob,mob[:,:noccb],moa,moa), tmpfile2.name, 'ba')
        with h5py.File(tmpfile2.name) as f:
            buf = lib.unpack_tril(f['ba'][:noccb*noccb])
            buf = buf.reshape(noccb,noccb,nmoa,nmoa)
            oovv = buf[:,:,nocca:,nocca:].reshape(noccb**2,nvira**2)
            eris.vvOO[:] = lib.transpose(oovv).reshape(nvira,nvira,noccb,noccb)
            buf = oovv = None
            for i0, i1 in lib.prange(0, nvirb, noccb):
                buf = lib.unpack_tril(f['ba'][(noccb+i0)*noccb:(noccb+i1)*noccb])
                eris.VOvv[i0:i1] = lib.pack_tril(buf[:,nocca:,nocca:]).reshape(i1-i0,noccb,-1)
                buf = buf.reshape(i1-i0,noccb,nmoa,nmoa)
                eris.VOoo[i0:i1] = buf[:,:noccb,:nocca,:nocca]
                buf = None
            del(f['ba'])

    cput1 = log.timer_debug1('transforming vopq', *cput1)

    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]
    ao2mo.full(mol, orbva, eris.feri, dataname='vvvv')
    ao2mo.full(mol, orbvb, eris.feri, dataname='VVVV')
    ao2mo.general(mol, (orbva,orbva,orbvb,orbvb), eris.feri, dataname='vvVV')
    eris.vvvv = eris.feri['vvvv']
    eris.VVVV = eris.feri['VVVV']
    eris.vvVV = eris.feri['vvVV']

    cput1 = log.timer_debug1('transforming vvvv', *cput1)
    log.timer('CISD integral transformation', *cput0)
    return eris


def _cp(a):
    return numpy.array(a, copy=False, order='C')


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

    eris0 = _make_eris_incore(myci, mo)
    eris1 = _make_eris_outcore(myci, mo)
    print('oooo', abs(eris0.oooo - eris1.oooo).max())
    print('vooo', abs(eris0.vooo - eris1.vooo).max())
    print('voov', abs(eris0.voov - eris1.voov).max())
    print('vvoo', abs(eris0.vvoo - eris1.vvoo).max())
    print('vovv', abs(eris0.vovv - eris1.vovv).max())
    print('vvvv', abs(eris0.vvvv - eris1.vvvv).max())

    print('OOOO', abs(eris0.OOOO - eris1.OOOO).max())
    print('VOOO', abs(eris0.VOOO - eris1.VOOO).max())
    print('VOOV', abs(eris0.VOOV - eris1.VOOV).max())
    print('VVOO', abs(eris0.VVOO - eris1.VVOO).max())
    print('VOVV', abs(eris0.VOVV - eris1.VOVV).max())
    print('VVVV', abs(eris0.VVVV - eris1.VVVV).max())

    print('ooOO', abs(eris0.ooOO - eris1.ooOO).max())
    print('voOO', abs(eris0.voOO - eris1.voOO).max())
    print('voOV', abs(eris0.voOV - eris1.voOV).max())
    print('vvOO', abs(eris0.vvOO - eris1.vvOO).max())
    print('voVV', abs(eris0.voVV - eris1.voVV).max())
    print('vvVV', abs(eris0.vvVV - eris1.vvVV).max())

    print('VOoo', abs(eris0.VOoo - eris1.VOoo).max())
    print('VVoo', abs(eris0.VVoo - eris1.VVoo).max())
    print('VOvv', abs(eris0.VOvv - eris1.VOvv).max())

    eris = myci.ao2mo(mo)
    print(lib.finger(myci.make_diagonal(eris)) - -838.45507742639279)

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

