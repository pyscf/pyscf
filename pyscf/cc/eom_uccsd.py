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

import time
import numpy as np

from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import uccsd
from pyscf.cc import eom_rccsd
from pyscf.cc import eom_gccsd
from pyscf.cc import addons


########################################
# EOM-IP-CCSD
########################################

class EOMIP(eom_gccsd.EOMIP):
    def __init__(self, cc):
        gcc = addons.convert_to_gccsd(cc)
        eom_gccsd.EOMIP.__init__(self, gcc)

########################################
# EOM-EA-CCSD
########################################

class EOMEA(eom_gccsd.EOMEA):
    def __init__(self, cc):
        gcc = addons.convert_to_gccsd(cc)
        eom_gccsd.EOMEA.__init__(self, gcc)

########################################
# EOM-EE-CCSD
########################################

def eeccsd(eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None):
    '''Calculate N-electron neutral excitations via EOM-EE-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        koopmans : bool
            Calculate Koopmans'-like (1p1h) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    if eris is None: eris = eom._cc.ao2mo()
    if imds is None: imds = eom.make_imds(eris)

    spinvec_size = eom.vector_size()
    nroots = min(nroots, spinvec_size)

    diag_ee, diag_sf = eom.get_diag(imds)
    guess_ee = []
    guess_sf = []
    if guess and guess[0].size == spinvec_size:
        raise NotImplementedError
        #TODO: initial guess from GCCSD EOM amplitudes
        #orbspin = scf.addons.get_ghf_orbspin(eris.mo_coeff)
        #nmo = np.sum(eom.nmo)
        #nocc = np.sum(eom.nocc)
        #for g in guess:
        #    r1, r2 = eom_gccsd.vector_to_amplitudes_ee(g, nmo, nocc)
        #    r1aa = r1[orbspin==0][:,orbspin==0]
        #    r1ab = r1[orbspin==0][:,orbspin==1]
        #    if abs(r1aa).max() > 1e-7:
        #        r1 = addons.spin2spatial(r1, orbspin)
        #        r2 = addons.spin2spatial(r2, orbspin)
        #        guess_ee.append(eom.amplitudes_to_vector(r1, r2))
        #    else:
        #        r1 = spin2spatial_eomsf(r1, orbspin)
        #        r2 = spin2spatial_eomsf(r2, orbspin)
        #        guess_sf.append(amplitudes_to_vector_eomsf(r1, r2))
        #    r1 = r2 = r1aa = r1ab = g = None
        #nroots_ee = len(guess_ee)
        #nroots_sf = len(guess_sf)
    elif guess:
        for g in guess:
            if g.size == diag_ee.size:
                guess_ee.append(g)
            else:
                guess_sf.append(g)
        nroots_ee = len(guess_ee)
        nroots_sf = len(guess_sf)
    else:
        dee = np.sort(diag_ee)[:nroots]
        dsf = np.sort(diag_sf)[:nroots]
        dmax = np.sort(np.hstack([dee,dsf]))[nroots-1]
        nroots_ee = np.count_nonzero(dee <= dmax)
        nroots_sf = np.count_nonzero(dsf <= dmax)
        guess_ee = guess_sf = None

    def eomee_sub(cls, nroots, guess, diag):
        ee_sub = cls(eom._cc)
        ee_sub.__dict__.update(eom.__dict__)
        e, v = ee_sub.kernel(nroots, koopmans, guess, eris, imds, diag=diag)
        if nroots == 1:
            e, v = [e], [v]
            ee_sub.converged = [ee_sub.converged]
        return list(ee_sub.converged), list(e), list(v)

    e0 = e1 = []
    v0 = v1 = []
    conv0 = conv1 = []
    if nroots_ee > 0:
        conv0, e0, v0 = eomee_sub(EOMEESpinKeep, nroots_ee, guess_ee, diag_ee)
    if nroots_sf > 0:
        conv1, e1, v1 = eomee_sub(EOMEESpinFlip, nroots_sf, guess_sf, diag_sf)

    e = np.hstack([e0,e1])
    idx = e.argsort()
    e = e[idx]
    conv = conv0 + conv1
    conv = [conv[x] for x in idx]
    v = v0 + v1
    v = [v[x] for x in idx]

    if nroots == 1:
        conv = conv[0]
        e = e[0]
        v = v[0]
    eom.converged = conv
    eom.e = e
    eom.v = v
    return eom.e, eom.v

def eomee_ccsd(eom, nroots=1, koopmans=False, guess=None,
               eris=None, imds=None, diag=None):
    if eris is None: eris = eom._cc.ao2mo()
    if imds is None: imds = eom.make_imds(eris)
    eom.converged, eom.e, eom.v \
            = eom_rccsd.kernel(eom, nroots, koopmans, guess, imds=imds, diag=diag)
    return eom.e, eom.v

def eomsf_ccsd(eom, nroots=1, koopmans=False, guess=None,
               eris=None, imds=None, diag=None):
    '''Spin flip EOM-EE-CCSD
    '''
    return eomee_ccsd(eom, nroots, koopmans, guess, eris, imds, diag)

amplitudes_to_vector_ee = uccsd.amplitudes_to_vector
vector_to_amplitudes_ee = uccsd.vector_to_amplitudes

def amplitudes_to_vector_eomsf(t1, t2, out=None):
    t1ab, t1ba = t1
    t2baaa, t2aaba, t2abbb, t2bbab = t2
    nocca, nvirb = t1ab.shape
    noccb, nvira = t1ba.shape

    otrila = np.tril_indices(nocca, k=-1)
    otrilb = np.tril_indices(noccb, k=-1)
    vtrila = np.tril_indices(nvira, k=-1)
    vtrilb = np.tril_indices(nvirb, k=-1)
    baaa = np.take(t2baaa.reshape(noccb*nocca,nvira*nvira),
                   vtrila[0]*nvira+vtrila[1], axis=1)
    abbb = np.take(t2abbb.reshape(nocca*noccb,nvirb*nvirb),
                   vtrilb[0]*nvirb+vtrilb[1], axis=1)
    vector = np.hstack((t1ab.ravel(), t1ba.ravel(),
                        baaa.ravel(), t2aaba[otrila].ravel(),
                        abbb.ravel(), t2bbab[otrilb].ravel()))
    return vector

def vector_to_amplitudes_eomsf(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    t1ab = vector[:nocca*nvirb].reshape(nocca,nvirb).copy()
    t1ba = vector[nocca*nvirb:nocca*nvirb+noccb*nvira].reshape(noccb,nvira).copy()
    pvec = vector[t1ab.size+t1ba.size:]

    nbaaa = noccb*nocca*nvira*(nvira-1)//2
    naaba = nocca*(nocca-1)//2*nvirb*nvira
    nabbb = nocca*noccb*nvirb*(nvirb-1)//2
    nbbab = noccb*(noccb-1)//2*nvira*nvirb
    t2baaa = np.zeros((noccb*nocca,nvira*nvira), dtype=vector.dtype)
    t2aaba = np.zeros((nocca*nocca,nvirb*nvira), dtype=vector.dtype)
    t2abbb = np.zeros((nocca*noccb,nvirb*nvirb), dtype=vector.dtype)
    t2bbab = np.zeros((noccb*noccb,nvira*nvirb), dtype=vector.dtype)
    otrila = np.tril_indices(nocca, k=-1)
    otrilb = np.tril_indices(noccb, k=-1)
    vtrila = np.tril_indices(nvira, k=-1)
    vtrilb = np.tril_indices(nvirb, k=-1)
    oidxab = np.arange(nocca*noccb, dtype=np.int32)
    vidxab = np.arange(nvira*nvirb, dtype=np.int32)

    v = pvec[:nbaaa].reshape(noccb*nocca,-1)
    lib.takebak_2d(t2baaa, v, oidxab, vtrila[0]*nvira+vtrila[1])
    lib.takebak_2d(t2baaa,-v, oidxab, vtrila[1]*nvira+vtrila[0])
    v = pvec[nbaaa:nbaaa+naaba].reshape(-1,nvirb*nvira)
    lib.takebak_2d(t2aaba, v, otrila[0]*nocca+otrila[1], vidxab)
    lib.takebak_2d(t2aaba,-v, otrila[1]*nocca+otrila[0], vidxab)
    v = pvec[nbaaa+naaba:nbaaa+naaba+nabbb].reshape(nocca*noccb,-1)
    lib.takebak_2d(t2abbb, v, oidxab, vtrilb[0]*nvirb+vtrilb[1])
    lib.takebak_2d(t2abbb,-v, oidxab, vtrilb[1]*nvirb+vtrilb[0])
    v = pvec[nbaaa+naaba+nabbb:].reshape(-1,nvira*nvirb)
    lib.takebak_2d(t2bbab, v, otrilb[0]*noccb+otrilb[1], vidxab)
    lib.takebak_2d(t2bbab,-v, otrilb[1]*noccb+otrilb[0], vidxab)
    t2baaa = t2baaa.reshape(noccb,nocca,nvira,nvira)
    t2aaba = t2aaba.reshape(nocca,nocca,nvirb,nvira)
    t2abbb = t2abbb.reshape(nocca,noccb,nvirb,nvirb)
    t2bbab = t2bbab.reshape(noccb,noccb,nvira,nvirb)
    return (t1ab,t1ba), (t2baaa, t2aaba, t2abbb, t2bbab)

def spatial2spin_eomsf(rx, orbspin):
    '''Convert EOM spatial R1,R2 to spin-orbital R1,R2'''
    if len(rx) == 2:  # r1
        r1ab, r1ba = rx
        nocca, nvirb = r1ab.shape
        noccb, nvira = r1ba.shape
    else:
        r2baaa,r2aaba,r2abbb,r2bbab = rx
        noccb, nocca, nvira = r2baaa.shape[:3]
        nvirb = r2aaba.shape[2]

    nocc = nocca + noccb
    nvir = nvira + nvirb
    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]

    if len(rx) == 2:  # r1
        r1 = np.zeros((nocc,nvir), dtype=r1ab.dtype)
        lib.takebak_2d(r1, r1ab, idxoa, idxvb)
        lib.takebak_2d(r1, r1ba, idxob, idxva)
        return r1

    else:
        r2 = np.zeros((nocc**2,nvir**2), dtype=r2aaba.dtype)
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxoba = idxob[:,None] * nocc + idxoa
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvba = idxvb[:,None] * nvir + idxva
        idxvbb = idxvb[:,None] * nvir + idxvb
        r2baaa = r2baaa.reshape(noccb*nocca,nvira*nvira)
        r2aaba = r2aaba.reshape(nocca*nocca,nvirb*nvira)
        r2abbb = r2abbb.reshape(nocca*noccb,nvirb*nvirb)
        r2bbab = r2bbab.reshape(noccb*noccb,nvira*nvirb)
        lib.takebak_2d(r2, r2baaa, idxoba.ravel(), idxvaa.ravel())
        lib.takebak_2d(r2, r2aaba, idxoaa.ravel(), idxvba.ravel())
        lib.takebak_2d(r2, r2abbb, idxoab.ravel(), idxvbb.ravel())
        lib.takebak_2d(r2, r2bbab, idxobb.ravel(), idxvab.ravel())
        lib.takebak_2d(r2, r2baaa, idxoab.T.ravel(), idxvaa.T.ravel())
        lib.takebak_2d(r2, r2aaba, idxoaa.T.ravel(), idxvab.T.ravel())
        lib.takebak_2d(r2, r2abbb, idxoba.T.ravel(), idxvbb.T.ravel())
        lib.takebak_2d(r2, r2bbab, idxobb.T.ravel(), idxvba.T.ravel())
        return r2.reshape(nocc,nocc,nvir,nvir)

def spin2spatial_eomsf(rx, orbspin):
    '''Convert EOM spin-orbital R1,R2 to spatial R1,R2'''
    if rx.ndim == 2:  # r1
        nocc, nvir = rx.shape
    else:
        nocc, nvir = rx.shape[1:3]

    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]
    nocca = len(idxoa)
    noccb = len(idxob)
    nvira = len(idxva)
    nvirb = len(idxvb)

    if rx.ndim == 2:
        r1ab = lib.take_2d(rx, idxoa, idxvb)
        r1ba = lib.take_2d(rx, idxob, idxva)
        return r1ab, r1ba
    else:
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxoba = idxob[:,None] * nocc + idxoa
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvba = idxvb[:,None] * nvir + idxva
        idxvbb = idxvb[:,None] * nvir + idxvb
        r2 = rx.reshape(nocc**2,nvir**2)
        r2baaa = lib.take_2d(r2, idxoba.ravel(), idxvaa.ravel())
        r2aaba = lib.take_2d(r2, idxoaa.ravel(), idxvba.ravel())
        r2abbb = lib.take_2d(r2, idxoab.ravel(), idxvbb.ravel())
        r2bbab = lib.take_2d(r2, idxobb.ravel(), idxvab.ravel())
        r2baaa = r2baaa.reshape(noccb,nocca,nvira,nvira)
        r2aaba = r2aaba.reshape(nocca,nocca,nvirb,nvira)
        r2abbb = r2abbb.reshape(nocca,noccb,nvirb,nvirb)
        r2bbab = r2bbab.reshape(noccb,noccb,nvira,nvirb)
        return r2baaa,r2aaba,r2abbb,r2bbab

# Ref: Wang, Tu, and Wang, J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10)
# Note: Last line in Eq. (10) is superfluous.
# See, e.g. Gwaltney, Nooijen, and Barlett, Chem. Phys. Lett. 248, 189 (1996)
def eomee_ccsd_matvec(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()

    t1, t2, eris = imds.t1, imds.t2, imds.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca+nvira, noccb+nvirb
    r1, r2 = vector_to_amplitudes_ee(vector, (nmoa,nmob), (nocca,noccb))
    r1a, r1b = r1
    r2aa, r2ab, r2bb = r2

    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:Hr2aa += lib.einsum('ijef,aebf->ijab', tau2aa, eris_vvvv) * .5
    #:Hr2bb += lib.einsum('ijef,aebf->ijab', tau2bb, eris_VVVV) * .5
    #:Hr2ab += lib.einsum('iJeF,aeBF->iJaB', tau2ab, eris_vvVV)
    tau2aa, tau2ab, tau2bb = uccsd.make_tau(r2, r1, t1, 2)
    Hr2aa, Hr2ab, Hr2bb = eom._cc._add_vvvv(None, (tau2aa,tau2ab,tau2bb), eris)
    Hr2aa *= .5
    Hr2bb *= .5
    tau2aa = tau2ab = tau2bb = None

    Hr1a  = lib.einsum('ae,ie->ia', imds.Fvva, r1a)
    Hr1a -= lib.einsum('mi,ma->ia', imds.Fooa, r1a)
    Hr1a += np.einsum('me,imae->ia',imds.Fova, r2aa)
    Hr1a += np.einsum('ME,iMaE->ia',imds.Fovb, r2ab)
    Hr1b  = lib.einsum('ae,ie->ia', imds.Fvvb, r1b)
    Hr1b -= lib.einsum('mi,ma->ia', imds.Foob, r1b)
    Hr1b += np.einsum('me,imae->ia',imds.Fovb, r2bb)
    Hr1b += np.einsum('me,mIeA->IA',imds.Fova, r2ab)

    Hr2aa += lib.einsum('mnij,mnab->ijab', imds.woooo, r2aa) * .25
    Hr2bb += lib.einsum('mnij,mnab->ijab', imds.wOOOO, r2bb) * .25
    Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', imds.woOoO, r2ab)
    Hr2aa += lib.einsum('be,ijae->ijab', imds.Fvva, r2aa)
    Hr2bb += lib.einsum('be,ijae->ijab', imds.Fvvb, r2bb)
    Hr2ab += lib.einsum('BE,iJaE->iJaB', imds.Fvvb, r2ab)
    Hr2ab += lib.einsum('be,iJeA->iJbA', imds.Fvva, r2ab)
    Hr2aa -= lib.einsum('mj,imab->ijab', imds.Fooa, r2aa)
    Hr2bb -= lib.einsum('mj,imab->ijab', imds.Foob, r2bb)
    Hr2ab -= lib.einsum('MJ,iMaB->iJaB', imds.Foob, r2ab)
    Hr2ab -= lib.einsum('mj,mIaB->jIaB', imds.Fooa, r2ab)

    #:tau2aa, tau2ab, tau2bb = uccsd.make_tau(r2, r1, t1, 2)
    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    #:Hr1a += lib.einsum('mfae,imef->ia', eris_ovvv, r2aa)
    #:tmpaa = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2aa)
    #:Hr2aa+= lib.einsum('mb,maij->ijab', t1a, tmpaa)
    #:tmpa = lib.einsum('mfae,me->af', eris_ovvv, r1a)
    #:tmpa-= lib.einsum('meaf,me->af', eris_ovvv, r1a)

    #:Hr1b += lib.einsum('mfae,imef->ia', eris_OVVV, r2bb)
    #:tmpbb = lib.einsum('meaf,ijef->maij', eris_OVVV, tau2bb)
    #:Hr2bb+= lib.einsum('mb,maij->ijab', t1b, tmpbb)
    #:tmpb = lib.einsum('mfae,me->af', eris_OVVV, r1b)
    #:tmpb-= lib.einsum('meaf,me->af', eris_OVVV, r1b)

    #:Hr1b += lib.einsum('mfAE,mIfE->IA', eris_ovVV, r2ab)
    #:tmpab = lib.einsum('meAF,iJeF->mAiJ', eris_ovVV, tau2ab)
    #:Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1a, tmpab)
    #:tmpb-= lib.einsum('meAF,me->AF', eris_ovVV, r1a)

    #:Hr1a += lib.einsum('MFae,iMeF->ia', eris_OVvv, r2ab)
    #:tmpba =-lib.einsum('MEaf,iJfE->MaiJ', eris_OVvv, tau2ab)
    #:Hr2ab+= lib.einsum('MB,MaiJ->iJaB', t1b, tmpba)
    #:tmpa-= lib.einsum('MEaf,ME->af', eris_OVvv, r1b)
    tau2aa = uccsd.make_tau_aa(r2aa, r1a, t1a, 2)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    tmpa = np.zeros((nvira,nvira))
    tmpb = np.zeros((nvirb,nvirb))
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3))))
    for p0, p1 in lib.prange(0, nocca, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Hr1a += lib.einsum('mfae,imef->ia', ovvv, r2aa[:,p0:p1])
        tmpaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aa)
        Hr2aa+= lib.einsum('mb,maij->ijab', t1a[p0:p1], tmpaa)
        tmpa+= lib.einsum('mfae,me->af', ovvv, r1a[p0:p1])
        tmpa-= lib.einsum('meaf,me->af', ovvv, r1a[p0:p1])
        ovvv = tmpaa = None
    tau2aa = None

    tau2bb = uccsd.make_tau_aa(r2bb, r1b, t1b, 2)
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
        Hr1b += lib.einsum('mfae,imef->ia', OVVV, r2bb[:,p0:p1])
        tmpbb = lib.einsum('meaf,ijef->maij', OVVV, tau2bb)
        Hr2bb+= lib.einsum('mb,maij->ijab', t1b[p0:p1], tmpbb)
        tmpb+= lib.einsum('mfae,me->af', OVVV, r1b[p0:p1])
        tmpb-= lib.einsum('meaf,me->af', OVVV, r1b[p0:p1])
        OVVV = tmpbb = None
    tau2bb = None

    tau2ab = uccsd.make_tau_ab(r2ab, r1 , t1 , 2)
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
    for p0, p1 in lib.prange(0, nocca, blksize):
        ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
        Hr1b += lib.einsum('mfAE,mIfE->IA', ovVV, r2ab[p0:p1])
        tmpab = lib.einsum('meAF,iJeF->mAiJ', ovVV, tau2ab)
        Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1a[p0:p1], tmpab)
        tmpb-= lib.einsum('meAF,me->AF', ovVV, r1a[p0:p1])
        ovVV = tmpab = None

    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
        Hr1a += lib.einsum('MFae,iMeF->ia', OVvv, r2ab[:,p0:p1])
        tmpba = lib.einsum('MEaf,iJfE->MaiJ', OVvv, tau2ab)
        Hr2ab-= lib.einsum('MB,MaiJ->iJaB', t1b[p0:p1], tmpba)
        tmpa-= lib.einsum('MEaf,ME->af', OVvv, r1b[p0:p1])
        OVvv = tmpba = None
    tau2ab = None

    Hr2aa-= lib.einsum('af,ijfb->ijab', tmpa, t2aa)
    Hr2bb-= lib.einsum('af,ijfb->ijab', tmpb, t2bb)
    Hr2ab-= lib.einsum('af,iJfB->iJaB', tmpa, t2ab)
    Hr2ab-= lib.einsum('AF,iJbF->iJbA', tmpb, t2ab)

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    tau2aa = uccsd.make_tau_aa(r2aa, r1a, t1a, 2)
    tauaa = uccsd.make_tau_aa(t2aa, t1a, t1a)
    tmpaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aa)
    Hr2aa += lib.einsum('mnij,mnab->ijab', tmpaa, tauaa) * 0.25
    tau2aa = tauaa = None

    tau2bb = uccsd.make_tau_aa(r2bb, r1b, t1b, 2)
    taubb = uccsd.make_tau_aa(t2bb, t1b, t1b)
    tmpbb = lib.einsum('menf,ijef->mnij', eris_OVOV, tau2bb)
    Hr2bb += lib.einsum('mnij,mnab->ijab', tmpbb, taubb) * 0.25
    tau2bb = taubb = None

    tau2ab = uccsd.make_tau_ab(r2ab, r1 , t1 , 2)
    tauab = uccsd.make_tau_ab(t2ab, t1 , t1)
    tmpab = lib.einsum('meNF,iJeF->mNiJ', eris_ovOV, tau2ab)
    Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', tmpab, tauab)
    tau2ab = tauab = None

    tmpa = lib.einsum('menf,imef->ni', eris_ovov, r2aa)
    tmpa-= lib.einsum('neMF,iMeF->ni', eris_ovOV, r2ab)
    tmpb = lib.einsum('menf,imef->ni', eris_OVOV, r2bb)
    tmpb-= lib.einsum('mfNE,mIfE->NI', eris_ovOV, r2ab)
    Hr1a += lib.einsum('na,ni->ia', t1a, tmpa)
    Hr1b += lib.einsum('na,ni->ia', t1b, tmpb)
    Hr2aa+= lib.einsum('mj,imab->ijab', tmpa, t2aa)
    Hr2bb+= lib.einsum('mj,imab->ijab', tmpb, t2bb)
    Hr2ab+= lib.einsum('MJ,iMaB->iJaB', tmpb, t2ab)
    Hr2ab+= lib.einsum('mj,mIaB->jIaB', tmpa, t2ab)

    tmp1a = np.einsum('menf,mf->en', eris_ovov, r1a)
    tmp1a-= np.einsum('mfne,mf->en', eris_ovov, r1a)
    tmp1a-= np.einsum('neMF,MF->en', eris_ovOV, r1b)
    tmp1b = np.einsum('menf,mf->en', eris_OVOV, r1b)
    tmp1b-= np.einsum('mfne,mf->en', eris_OVOV, r1b)
    tmp1b-= np.einsum('mfNE,mf->EN', eris_ovOV, r1a)
    tmpa = np.einsum('en,nb->eb', tmp1a, t1a)
    tmpa+= lib.einsum('menf,mnfb->eb', eris_ovov, r2aa)
    tmpa-= lib.einsum('meNF,mNbF->eb', eris_ovOV, r2ab)
    tmpb = np.einsum('en,nb->eb', tmp1b, t1b)
    tmpb+= lib.einsum('menf,mnfb->eb', eris_OVOV, r2bb)
    tmpb-= lib.einsum('nfME,nMfB->EB', eris_ovOV, r2ab)
    Hr2aa+= lib.einsum('eb,ijae->ijab', tmpa, t2aa)
    Hr2bb+= lib.einsum('eb,ijae->ijab', tmpb, t2bb)
    Hr2ab+= lib.einsum('EB,iJaE->iJaB', tmpb, t2ab)
    Hr2ab+= lib.einsum('eb,iJeA->iJbA', tmpa, t2ab)
    eirs_ovov = eris_ovOV = eris_OVOV = None

    Hr2aa-= lib.einsum('mbij,ma->ijab', imds.wovoo, r1a)
    Hr2bb-= lib.einsum('mbij,ma->ijab', imds.wOVOO, r1b)
    Hr2ab-= lib.einsum('mBiJ,ma->iJaB', imds.woVoO, r1a)
    Hr2ab-= lib.einsum('MbJi,MA->iJbA', imds.wOvOo, r1b)

    Hr1a-= 0.5*lib.einsum('mnie,mnae->ia', imds.wooov, r2aa)
    Hr1a-=     lib.einsum('mNiE,mNaE->ia', imds.woOoV, r2ab)
    Hr1b-= 0.5*lib.einsum('mnie,mnae->ia', imds.wOOOV, r2bb)
    Hr1b-=     lib.einsum('MnIe,nMeA->IA', imds.wOoOv, r2ab)
    tmpa = lib.einsum('mnie,me->ni', imds.wooov, r1a)
    tmpa-= lib.einsum('nMiE,ME->ni', imds.woOoV, r1b)
    tmpb = lib.einsum('mnie,me->ni', imds.wOOOV, r1b)
    tmpb-= lib.einsum('NmIe,me->NI', imds.wOoOv, r1a)
    Hr2aa+= lib.einsum('ni,njab->ijab', tmpa, t2aa)
    Hr2bb+= lib.einsum('ni,njab->ijab', tmpb, t2bb)
    Hr2ab+= lib.einsum('ni,nJaB->iJaB', tmpa, t2ab)
    Hr2ab+= lib.einsum('NI,jNaB->jIaB', tmpb, t2ab)
    for p0, p1 in lib.prange(0, nvira, nocca):
        Hr2aa+= lib.einsum('ejab,ie->ijab', imds.wvovv[p0:p1], r1a[:,p0:p1])
        Hr2ab+= lib.einsum('eJaB,ie->iJaB', imds.wvOvV[p0:p1], r1a[:,p0:p1])
    for p0, p1 in lib.prange(0, nvirb, noccb):
        Hr2bb+= lib.einsum('ejab,ie->ijab', imds.wVOVV[p0:p1], r1b[:,p0:p1])
        Hr2ab+= lib.einsum('EjBa,IE->jIaB', imds.wVoVv[p0:p1], r1b[:,p0:p1])

    Hr1a += np.einsum('maei,me->ia',imds.wovvo,r1a)
    Hr1a += np.einsum('MaEi,ME->ia',imds.wOvVo,r1b)
    Hr1b += np.einsum('maei,me->ia',imds.wOVVO,r1b)
    Hr1b += np.einsum('mAeI,me->IA',imds.woVvO,r1a)
    Hr2aa+= lib.einsum('mbej,imae->ijab', imds.wovvo, r2aa) * 2
    Hr2aa+= lib.einsum('MbEj,iMaE->ijab', imds.wOvVo, r2ab) * 2
    Hr2bb+= lib.einsum('mbej,imae->ijab', imds.wOVVO, r2bb) * 2
    Hr2bb+= lib.einsum('mBeJ,mIeA->IJAB', imds.woVvO, r2ab) * 2
    Hr2ab+= lib.einsum('mBeJ,imae->iJaB', imds.woVvO, r2aa)
    Hr2ab+= lib.einsum('MBEJ,iMaE->iJaB', imds.wOVVO, r2ab)
    Hr2ab+= lib.einsum('mBEj,mIaE->jIaB', imds.woVVo, r2ab)
    Hr2ab+= lib.einsum('mbej,mIeA->jIbA', imds.wovvo, r2ab)
    Hr2ab+= lib.einsum('MbEj,IMAE->jIbA', imds.wOvVo, r2bb)
    Hr2ab+= lib.einsum('MbeJ,iMeA->iJbA', imds.wOvvO, r2ab)

    Hr2aa *= .5
    Hr2bb *= .5
    Hr2aa = Hr2aa - Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa - Hr2aa.transpose(1,0,2,3)
    Hr2bb = Hr2bb - Hr2bb.transpose(0,1,3,2)
    Hr2bb = Hr2bb - Hr2bb.transpose(1,0,2,3)

    vector = amplitudes_to_vector_ee((Hr1a,Hr1b), (Hr2aa,Hr2ab,Hr2bb))
    return vector

def eomsf_ccsd_matvec(eom, vector, imds=None):
    '''Spin flip EOM-CCSD'''
    if imds is None: imds = eom.make_imds()

    t1, t2, eris = imds.t1, imds.t2, imds.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca+nvira, noccb+nvirb
    r1, r2 = vector_to_amplitudes_eomsf(vector, (nmoa,nmob), (nocca,noccb))
    r1ab, r1ba = r1
    r2baaa, r2aaba, r2abbb, r2bbab = r2

    Hr1ab  = np.einsum('ae,ie->ia', imds.Fvvb, r1ab)
    Hr1ab -= np.einsum('mi,ma->ia', imds.Fooa, r1ab)
    Hr1ab += np.einsum('me,imae->ia', imds.Fovb, r2abbb)
    Hr1ab += np.einsum('me,imae->ia', imds.Fova, r2aaba)
    Hr1ba  = np.einsum('ae,ie->ia', imds.Fvva, r1ba)
    Hr1ba -= np.einsum('mi,ma->ia', imds.Foob, r1ba)
    Hr1ba += np.einsum('me,imae->ia', imds.Fova, r2baaa)
    Hr1ba += np.einsum('me,imae->ia', imds.Fovb, r2bbab)
    Hr2baaa = .5 *lib.einsum('nMjI,Mnab->Ijab', imds.woOoO, r2baaa)
    Hr2aaba = .25*lib.einsum('mnij,mnAb->ijAb', imds.woooo, r2aaba)
    Hr2abbb = .5 *lib.einsum('mNiJ,mNAB->iJAB', imds.woOoO, r2abbb)
    Hr2bbab = .25*lib.einsum('MNIJ,MNaB->IJaB', imds.wOOOO, r2bbab)
    Hr2baaa += lib.einsum('be,Ijae->Ijab', imds.Fvva   , r2baaa)
    Hr2baaa -= lib.einsum('mj,imab->ijab', imds.Fooa*.5, r2baaa)
    Hr2baaa -= lib.einsum('MJ,Miab->Jiab', imds.Foob*.5, r2baaa)
    Hr2bbab -= lib.einsum('mj,imab->ijab', imds.Foob   , r2bbab)
    Hr2bbab += lib.einsum('BE,IJaE->IJaB', imds.Fvvb*.5, r2bbab)
    Hr2bbab += lib.einsum('be,IJeA->IJbA', imds.Fvva*.5, r2bbab)
    Hr2aaba -= lib.einsum('mj,imab->ijab', imds.Fooa   , r2aaba)
    Hr2aaba += lib.einsum('be,ijAe->ijAb', imds.Fvva*.5, r2aaba)
    Hr2aaba += lib.einsum('BE,ijEa->ijBa', imds.Fvvb*.5, r2aaba)
    Hr2abbb += lib.einsum('BE,iJAE->iJAB', imds.Fvvb   , r2abbb)
    Hr2abbb -= lib.einsum('mj,imab->ijab', imds.Foob*.5, r2abbb)
    Hr2abbb -= lib.einsum('mj,mIAB->jIAB', imds.Fooa*.5, r2abbb)

    tau2baaa = np.einsum('ia,jb->ijab', r1ba, t1a)
    tau2baaa = tau2baaa - tau2baaa.transpose(0,1,3,2)
    tau2abbb = np.einsum('ia,jb->ijab', r1ab, t1b)
    tau2abbb = tau2abbb - tau2abbb.transpose(0,1,3,2)
    tau2aaba = np.einsum('ia,jb->ijab', r1ab, t1a)
    tau2aaba = tau2aaba - tau2aaba.transpose(1,0,2,3)
    tau2bbab = np.einsum('ia,jb->ijab', r1ba, t1b)
    tau2bbab = tau2bbab - tau2bbab.transpose(1,0,2,3)
    tau2baaa += r2baaa
    tau2bbab += r2bbab
    tau2abbb += r2abbb
    tau2aaba += r2aaba
    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    #:Hr1ba += lib.einsum('mfae,Imef->Ia', eris_ovvv, r2baaa)
    #:tmp1aaba = lib.einsum('meaf,Ijef->maIj', eris_ovvv, tau2baaa)
    #:Hr2baaa += lib.einsum('mb,maIj->Ijab', t1a   , tmp1aaba)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Hr1ba += lib.einsum('mfae,Imef->Ia', ovvv, r2baaa[:,p0:p1])
        tmp1aaba = lib.einsum('meaf,Ijef->maIj', ovvv, tau2baaa)
        Hr2baaa += lib.einsum('mb,maIj->Ijab', t1a[p0:p1], tmp1aaba)
        ovvv = tmp1aaba = None

    #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    #:Hr1ab += lib.einsum('MFAE,iMEF->iA', eris_OVVV, r2abbb)
    #:tmp1bbab = lib.einsum('MEAF,iJEF->MAiJ', eris_OVVV, tau2abbb)
    #:Hr2abbb += lib.einsum('MB,MAiJ->iJAB', t1b   , tmp1bbab)
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
        Hr1ab += lib.einsum('MFAE,iMEF->iA', OVVV, r2abbb[:,p0:p1])
        tmp1bbab = lib.einsum('MEAF,iJEF->MAiJ', OVVV, tau2abbb)
        Hr2abbb += lib.einsum('MB,MAiJ->iJAB', t1b[p0:p1], tmp1bbab)
        OVVV = tmp1bbab = None

    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:Hr1ab += lib.einsum('mfAE,imEf->iA', eris_ovVV, r2aaba)
    #:tmp1abaa = lib.einsum('meAF,ijFe->mAij', eris_ovVV, tau2aaba)
    #:tmp1abbb = lib.einsum('meAF,IJeF->mAIJ', eris_ovVV, tau2bbab)
    #:tmp1ba = lib.einsum('mfAE,mE->Af', eris_ovVV, r1ab)
    #:Hr2bbab -= lib.einsum('mb,mAIJ->IJbA', t1a*.5, tmp1abbb)
    #:Hr2aaba -= lib.einsum('mb,mAij->ijAb', t1a*.5, tmp1abaa)
    tmp1ba = np.zeros((nvirb,nvira))
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
        Hr1ab += lib.einsum('mfAE,imEf->iA', ovVV, r2aaba[:,p0:p1])
        tmp1abaa = lib.einsum('meAF,ijFe->mAij', ovVV, tau2aaba)
        tmp1abbb = lib.einsum('meAF,IJeF->mAIJ', ovVV, tau2bbab)
        tmp1ba += lib.einsum('mfAE,mE->Af', ovVV, r1ab[p0:p1])
        Hr2bbab -= lib.einsum('mb,mAIJ->IJbA', t1a[p0:p1]*.5, tmp1abbb)
        Hr2aaba -= lib.einsum('mb,mAij->ijAb', t1a[p0:p1]*.5, tmp1abaa)

    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:Hr1ba += lib.einsum('MFae,IMeF->Ia', eris_OVvv, r2bbab)
    #:tmp1baaa = lib.einsum('MEaf,ijEf->Maij', eris_OVvv, tau2aaba)
    #:tmp1babb = lib.einsum('MEaf,IJfE->MaIJ', eris_OVvv, tau2bbab)
    #:tmp1ab = lib.einsum('MFae,Me->aF', eris_OVvv, r1ba)
    #:Hr2aaba -= lib.einsum('MB,Maij->ijBa', t1b*.5, tmp1baaa)
    #:Hr2bbab -= lib.einsum('MB,MaIJ->IJaB', t1b*.5, tmp1babb)
    tmp1ab = np.zeros((nvira,nvirb))
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
        Hr1ba += lib.einsum('MFae,IMeF->Ia', OVvv, r2bbab[:,p0:p1])
        tmp1baaa = lib.einsum('MEaf,ijEf->Maij', OVvv, tau2aaba)
        tmp1babb = lib.einsum('MEaf,IJfE->MaIJ', OVvv, tau2bbab)
        tmp1ab+= lib.einsum('MFae,Me->aF', OVvv, r1ba[p0:p1])
        Hr2aaba -= lib.einsum('MB,Maij->ijBa', t1b[p0:p1]*.5, tmp1baaa)
        Hr2bbab -= lib.einsum('MB,MaIJ->IJaB', t1b[p0:p1]*.5, tmp1babb)

    Hr2baaa += lib.einsum('aF,jIbF->Ijba', tmp1ab   , t2ab)
    Hr2bbab -= lib.einsum('aF,IJFB->IJaB', tmp1ab*.5, t2bb)
    Hr2abbb += lib.einsum('Af,iJfB->iJBA', tmp1ba   , t2ab)
    Hr2aaba -= lib.einsum('Af,ijfb->ijAb', tmp1ba*.5, t2aa)
    Hr2baaa -= lib.einsum('MbIj,Ma->Ijab', imds.wOvOo, r1ba   )
    Hr2bbab -= lib.einsum('MBIJ,Ma->IJaB', imds.wOVOO, r1ba*.5)
    Hr2abbb -= lib.einsum('mBiJ,mA->iJAB', imds.woVoO, r1ab   )
    Hr2aaba -= lib.einsum('mbij,mA->ijAb', imds.wovoo, r1ab*.5)

    Hr1ab -= 0.5*lib.einsum('mnie,mnAe->iA', imds.wooov, r2aaba)
    Hr1ab -=     lib.einsum('mNiE,mNAE->iA', imds.woOoV, r2abbb)
    Hr1ba -= 0.5*lib.einsum('MNIE,MNaE->Ia', imds.wOOOV, r2bbab)
    Hr1ba -=     lib.einsum('MnIe,Mnae->Ia', imds.wOoOv, r2baaa)
    tmp1ab = lib.einsum('MnIe,Me->nI', imds.wOoOv, r1ba)
    tmp1ba = lib.einsum('mNiE,mE->Ni', imds.woOoV, r1ab)
    Hr2baaa += lib.einsum('nI,njab->Ijab', tmp1ab*.5, t2aa)
    Hr2bbab += lib.einsum('nI,nJaB->IJaB', tmp1ab   , t2ab)
    Hr2abbb += lib.einsum('Ni,NJAB->iJAB', tmp1ba*.5, t2bb)
    Hr2aaba += lib.einsum('Ni,jNbA->ijAb', tmp1ba   , t2ab)
    for p0, p1 in lib.prange(0, nvira, nocca):
        Hr2baaa += lib.einsum('ejab,Ie->Ijab', imds.wvovv[p0:p1], r1ba[:,p0:p1]*.5)
        Hr2bbab += lib.einsum('eJaB,Ie->IJaB', imds.wvOvV[p0:p1], r1ba[:,p0:p1]   )
    for p0, p1 in lib.prange(0, nvirb, noccb):
        Hr2abbb += lib.einsum('EJAB,iE->iJAB', imds.wVOVV[p0:p1], r1ab[:,p0:p1]*.5)
        Hr2aaba += lib.einsum('EjAb,iE->ijAb', imds.wVoVv[p0:p1], r1ab[:,p0:p1]   )

    Hr1ab += np.einsum('mAEi,mE->iA', imds.woVVo, r1ab)
    Hr1ba += np.einsum('MaeI,Me->Ia', imds.wOvvO, r1ba)
    Hr2baaa += lib.einsum('mbej,Imae->Ijab', imds.wovvo, r2baaa)
    Hr2baaa += lib.einsum('MbeJ,Miae->Jiab', imds.wOvvO, r2baaa)
    Hr2baaa += lib.einsum('MbEj,IMaE->Ijab', imds.wOvVo, r2bbab)
    Hr2bbab += lib.einsum('MBEJ,IMaE->IJaB', imds.wOVVO, r2bbab)
    Hr2bbab += lib.einsum('MbeJ,IMeA->IJbA', imds.wOvvO, r2bbab)
    Hr2bbab += lib.einsum('mBeJ,Imae->IJaB', imds.woVvO, r2baaa)
    Hr2aaba += lib.einsum('mbej,imAe->ijAb', imds.wovvo, r2aaba)
    Hr2aaba += lib.einsum('mBEj,imEa->ijBa', imds.woVVo, r2aaba)
    Hr2aaba += lib.einsum('MbEj,iMAE->ijAb', imds.wOvVo, r2abbb)
    Hr2abbb += lib.einsum('MBEJ,iMAE->iJAB', imds.wOVVO, r2abbb)
    Hr2abbb += lib.einsum('mBEj,mIAE->jIAB', imds.woVVo, r2abbb)
    Hr2abbb += lib.einsum('mBeJ,imAe->iJAB', imds.woVvO, r2aaba)

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    tmp1baaa = lib.einsum('nfME,ijEf->Mnij', eris_ovOV, tau2aaba)
    tmp1aaba = lib.einsum('menf,Ijef->mnIj', eris_ovov, tau2baaa)
    tmp1abbb = lib.einsum('meNF,IJeF->mNIJ', eris_ovOV, tau2bbab)
    tmp1bbab = lib.einsum('MENF,iJEF->MNiJ', eris_OVOV, tau2abbb)
    Hr2baaa += 0.5*.5*lib.einsum('mnIj,mnab->Ijab', tmp1aaba, tauaa)
    Hr2bbab +=     .5*lib.einsum('nMIJ,nMaB->IJaB', tmp1abbb, tauab)
    Hr2aaba +=     .5*lib.einsum('Nmij,mNbA->ijAb', tmp1baaa, tauab)
    Hr2abbb += 0.5*.5*lib.einsum('MNiJ,MNAB->iJAB', tmp1bbab, taubb)
    tauaa = tauab = taubb = None

    tmpab  = lib.einsum('menf,Imef->nI', eris_ovov, r2baaa)
    tmpab -= lib.einsum('nfME,IMfE->nI', eris_ovOV, r2bbab)
    tmpba  = lib.einsum('MENF,iMEF->Ni', eris_OVOV, r2abbb)
    tmpba -= lib.einsum('meNF,imFe->Ni', eris_ovOV, r2aaba)
    Hr1ab += np.einsum('NA,Ni->iA', t1b, tmpba)
    Hr1ba += np.einsum('na,nI->Ia', t1a, tmpab)
    Hr2baaa -= lib.einsum('mJ,imab->Jiab', tmpab*.5, t2aa)
    Hr2bbab -= lib.einsum('mJ,mIaB->IJaB', tmpab*.5, t2ab) * 2
    Hr2aaba -= lib.einsum('Mj,iMbA->ijAb', tmpba*.5, t2ab) * 2
    Hr2abbb -= lib.einsum('Mj,IMAB->jIAB', tmpba*.5, t2bb)

    tmp1ab = np.einsum('meNF,mF->eN', eris_ovOV, r1ab)
    tmp1ba = np.einsum('nfME,Mf->En', eris_ovOV, r1ba)
    tmpab = np.einsum('eN,NB->eB', tmp1ab, t1b)
    tmpba = np.einsum('En,nb->Eb', tmp1ba, t1a)
    tmpab -= lib.einsum('menf,mnBf->eB', eris_ovov, r2aaba)
    tmpab += lib.einsum('meNF,mNFB->eB', eris_ovOV, r2abbb)
    tmpba -= lib.einsum('MENF,MNbF->Eb', eris_OVOV, r2bbab)
    tmpba += lib.einsum('nfME,Mnfb->Eb', eris_ovOV, r2baaa)
    Hr2baaa -= lib.einsum('Eb,jIaE->Ijab', tmpba*.5, t2ab) * 2
    Hr2bbab -= lib.einsum('Eb,IJAE->IJbA', tmpba*.5, t2bb)
    Hr2aaba -= lib.einsum('eB,ijae->ijBa', tmpab*.5, t2aa)
    Hr2abbb -= lib.einsum('eB,iJeA->iJAB', tmpab*.5, t2ab) * 2
    eris_ovov = eris_OVOV = eris_ovOV = None

    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:Hr2baaa += .5*lib.einsum('Ijef,aebf->Ijab', tau2baaa, eris_vvvv)
    #:Hr2abbb += .5*lib.einsum('iJEF,AEBF->iJAB', tau2abbb, eris_VVVV)
    #:Hr2bbab += .5*lib.einsum('IJeF,aeBF->IJaB', tau2bbab, eris_vvVV)
    #:Hr2aaba += .5*lib.einsum('ijEf,bfAE->ijAb', tau2aaba, eris_vvVV)
    fakeri = uccsd._ChemistsERIs()
    fakeri.mol = eris.mol

    if eom._cc.direct:
        orbva = eris.mo_coeff[0][:,nocca:]
        orbvb = eris.mo_coeff[1][:,noccb:]
        tau2baaa = lib.einsum('ijab,pa,qb->ijpq', tau2baaa, .5*orbva, orbva)
        tmp = eris._contract_vvvv_t2(eom._cc, tau2baaa, True)
        Hr2baaa += lib.einsum('ijpq,pa,qb->ijab', tmp, orbva.conj(), orbva.conj())
        tmp = None

        tau2abbb = lib.einsum('ijab,pa,qb->ijpq', tau2abbb, .5*orbvb, orbvb)
        tmp = eris._contract_VVVV_t2(eom._cc, tau2abbb, True)
        Hr2abbb += lib.einsum('ijpq,pa,qb->ijab', tmp, orbvb.conj(), orbvb.conj())
        tmp = None
    else:
        tau2baaa *= .5
        Hr2baaa += eris._contract_vvvv_t2(eom._cc, tau2baaa, False)
        tau2abbb *= .5
        Hr2abbb += eris._contract_VVVV_t2(eom._cc, tau2abbb, False)

    tau2bbab *= .5
    Hr2bbab += eom._cc._add_vvVV(None, tau2bbab, eris)
    tau2aaba = tau2aaba.transpose(0,1,3,2)*.5
    Hr2aaba += eom._cc._add_vvVV(None, tau2aaba, eris).transpose(0,1,3,2)

    Hr2baaa = Hr2baaa - Hr2baaa.transpose(0,1,3,2)
    Hr2bbab = Hr2bbab - Hr2bbab.transpose(1,0,2,3)
    Hr2abbb = Hr2abbb - Hr2abbb.transpose(0,1,3,2)
    Hr2aaba = Hr2aaba - Hr2aaba.transpose(1,0,2,3)
    vector = amplitudes_to_vector_eomsf((Hr1ab, Hr1ba), (Hr2baaa,Hr2aaba,Hr2abbb,Hr2bbab))
    return vector

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    eris = imds.eris
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    nocca, noccb, nvira, nvirb = t2ab.shape

    Foa = imds.Fooa.diagonal()
    Fob = imds.Foob.diagonal()
    Fva = imds.Fvva.diagonal()
    Fvb = imds.Fvvb.diagonal()
    Wovaa = np.einsum('iaai->ia', imds.wovvo)
    Wovbb = np.einsum('iaai->ia', imds.wOVVO)
    Wovab = np.einsum('iaai->ia', imds.woVVo)
    Wovba = np.einsum('iaai->ia', imds.wOvvO)

    Hr1aa = lib.direct_sum('-i+a->ia', Foa, Fva)
    Hr1bb = lib.direct_sum('-i+a->ia', Fob, Fvb)
    Hr1ab = lib.direct_sum('-i+a->ia', Foa, Fvb)
    Hr1ba = lib.direct_sum('-i+a->ia', Fob, Fva)
    Hr1aa += Wovaa
    Hr1bb += Wovbb
    Hr1ab += Wovab
    Hr1ba += Wovba

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Wvvaa = .5*np.einsum('mnab,manb->ab', tauaa, eris_ovov)
    Wvvbb = .5*np.einsum('mnab,manb->ab', taubb, eris_OVOV)
    Wvvab =    np.einsum('mNaB,maNB->aB', tauab, eris_ovOV)
    ijb = np.einsum('iejb,ijbe->ijb',      ovov, t2aa)
    IJB = np.einsum('iejb,ijbe->ijb',      OVOV, t2bb)
    iJB =-np.einsum('ieJB,iJeB->iJB', eris_ovOV, t2ab)
    Ijb =-np.einsum('jbIE,jIbE->Ijb', eris_ovOV, t2ab)
    iJb =-np.einsum('ibJE,iJbE->iJb', eris_ovOV, t2ab)
    IjB =-np.einsum('jeIB,jIeB->IjB', eris_ovOV, t2ab)
    jab = np.einsum('kajb,jkab->jab',      ovov, t2aa)
    JAB = np.einsum('kajb,jkab->jab',      OVOV, t2bb)
    jAb =-np.einsum('jbKA,jKbA->jAb', eris_ovOV, t2ab)
    JaB =-np.einsum('kaJB,kJaB->JaB', eris_ovOV, t2ab)
    jaB =-np.einsum('jaKB,jKaB->jaB', eris_ovOV, t2ab)
    JAb =-np.einsum('kbJA,kJbA->JAb', eris_ovOV, t2ab)
    eris_ovov = eris_ovOV = eris_OVOV = ovov = OVOV = None
    Hr2aa = lib.direct_sum('ijb+a->ijba', ijb, Fva)
    Hr2bb = lib.direct_sum('ijb+a->ijba', IJB, Fvb)
    Hr2ab = lib.direct_sum('iJb+A->iJbA', iJb, Fvb)
    Hr2ab+= lib.direct_sum('iJB+a->iJaB', iJB, Fva)
    Hr2aa+= lib.direct_sum('-i+jab->ijab', Foa, jab)
    Hr2bb+= lib.direct_sum('-i+jab->ijab', Fob, JAB)
    Hr2ab+= lib.direct_sum('-i+JaB->iJaB', Foa, JaB)
    Hr2ab+= lib.direct_sum('-I+jaB->jIaB', Fob, jaB)
    Hr2aa = Hr2aa + Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa + Hr2aa.transpose(1,0,2,3)
    Hr2bb = Hr2bb + Hr2bb.transpose(0,1,3,2)
    Hr2bb = Hr2bb + Hr2bb.transpose(1,0,2,3)
    Hr2aa *= .5
    Hr2bb *= .5
    Hr2baaa = lib.direct_sum('Ijb+a->Ijba', Ijb, Fva)
    Hr2aaba = lib.direct_sum('ijb+A->ijAb', ijb, Fvb)
    Hr2aaba+= Fva.reshape(1,1,1,-1)
    Hr2abbb = lib.direct_sum('iJB+A->iJBA', iJB, Fvb)
    Hr2bbab = lib.direct_sum('IJB+a->IJaB', IJB, Fva)
    Hr2bbab+= Fvb.reshape(1,1,1,-1)
    Hr2baaa = Hr2baaa + Hr2baaa.transpose(0,1,3,2)
    Hr2abbb = Hr2abbb + Hr2abbb.transpose(0,1,3,2)
    Hr2baaa+= lib.direct_sum('-I+jab->Ijab', Fob, jab)
    Hr2baaa-= Foa.reshape(1,-1,1,1)
    tmpaaba = lib.direct_sum('-i+jAb->ijAb', Foa, jAb)
    Hr2abbb+= lib.direct_sum('-i+JAB->iJAB', Foa, JAB)
    Hr2abbb-= Fob.reshape(1,-1,1,1)
    tmpbbab = lib.direct_sum('-I+JaB->IJaB', Fob, JaB)
    Hr2aaba+= tmpaaba + tmpaaba.transpose(1,0,2,3)
    Hr2bbab+= tmpbbab + tmpbbab.transpose(1,0,2,3)
    tmpaaba = tmpbbab = None
    Hr2aa += Wovaa.reshape(1,nocca,1,nvira)
    Hr2aa += Wovaa.reshape(nocca,1,1,nvira)
    Hr2aa += Wovaa.reshape(nocca,1,nvira,1)
    Hr2aa += Wovaa.reshape(1,nocca,nvira,1)
    Hr2ab += Wovbb.reshape(1,noccb,1,nvirb)
    Hr2ab += Wovab.reshape(nocca,1,1,nvirb)
    Hr2ab += Wovaa.reshape(nocca,1,nvira,1)
    Hr2ab += Wovba.reshape(1,noccb,nvira,1)
    Hr2bb += Wovbb.reshape(1,noccb,1,nvirb)
    Hr2bb += Wovbb.reshape(noccb,1,1,nvirb)
    Hr2bb += Wovbb.reshape(noccb,1,nvirb,1)
    Hr2bb += Wovbb.reshape(1,noccb,nvirb,1)
    Hr2baaa += Wovaa.reshape(1,nocca,1,nvira)
    Hr2baaa += Wovba.reshape(noccb,1,1,nvira)
    Hr2baaa += Wovba.reshape(noccb,1,nvira,1)
    Hr2baaa += Wovaa.reshape(1,nocca,nvira,1)
    Hr2aaba += Wovaa.reshape(1,nocca,1,nvira)
    Hr2aaba += Wovaa.reshape(nocca,1,1,nvira)
    Hr2aaba += Wovab.reshape(nocca,1,nvirb,1)
    Hr2aaba += Wovab.reshape(1,nocca,nvirb,1)
    Hr2abbb += Wovbb.reshape(1,noccb,1,nvirb)
    Hr2abbb += Wovab.reshape(nocca,1,1,nvirb)
    Hr2abbb += Wovab.reshape(nocca,1,nvirb,1)
    Hr2abbb += Wovbb.reshape(1,noccb,nvirb,1)
    Hr2bbab += Wovbb.reshape(1,noccb,1,nvirb)
    Hr2bbab += Wovbb.reshape(noccb,1,1,nvirb)
    Hr2bbab += Wovba.reshape(noccb,1,nvira,1)
    Hr2bbab += Wovba.reshape(1,noccb,nvira,1)

    Wooaa  = np.einsum('ijij->ij', imds.woooo).copy()
    Wooaa -= np.einsum('ijji->ij', imds.woooo)
    Woobb  = np.einsum('ijij->ij', imds.wOOOO).copy()
    Woobb -= np.einsum('ijji->ij', imds.wOOOO)
    Wooab = np.einsum('ijij->ij', imds.woOoO)
    Wooba = Wooab.T
    Wooaa *= .5
    Woobb *= .5
    Hr2aa += Wooaa.reshape(nocca,nocca,1,1)
    Hr2ab += Wooab.reshape(nocca,noccb,1,1)
    Hr2bb += Woobb.reshape(noccb,noccb,1,1)
    Hr2baaa += Wooba.reshape(noccb,nocca,1,1)
    Hr2aaba += Wooaa.reshape(nocca,nocca,1,1)
    Hr2abbb += Wooab.reshape(nocca,noccb,1,1)
    Hr2bbab += Woobb.reshape(noccb,noccb,1,1)

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
    #:Wvvaa += np.einsum('mb,maab->ab', t1a, eris_ovvv)
    #:Wvvaa -= np.einsum('mb,mbaa->ab', t1a, eris_ovvv)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Wvvaa += np.einsum('mb,maab->ab', t1a[p0:p1], ovvv)
        Wvvaa -= np.einsum('mb,mbaa->ab', t1a[p0:p1], ovvv)
        ovvv = None
    #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
    #:Wvvbb += np.einsum('mb,maab->ab', t1b, eris_OVVV)
    #:Wvvbb -= np.einsum('mb,mbaa->ab', t1b, eris_OVVV)
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
        Wvvbb += np.einsum('mb,maab->ab', t1b[p0:p1], OVVV)
        Wvvbb -= np.einsum('mb,mbaa->ab', t1b[p0:p1], OVVV)
        OVVV = None
    #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
    #:Wvvab -= np.einsum('mb,mbaa->ba', t1a, eris_ovVV)
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
        Wvvab -= np.einsum('mb,mbaa->ba', t1a[p0:p1], ovVV)
        ovVV = None
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
    #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
    #:Wvvab -= np.einsum('mb,mbaa->ab', t1b, eris_OVvv)
    #idxa = np.arange(nvira)
    #idxa = idxa*(idxa+1)//2+idxa
    #for p0, p1 in lib.prange(0, noccb, blksize):
    #    OVvv = np.asarray(eris.OVvv[p0:p1])
    #    Wvvab -= np.einsum('mb,mba->ab', t1b[p0:p1], OVvv[:,:,idxa])
    #    OVvv = None
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
        Wvvab -= np.einsum('mb,mbaa->ab', t1b[p0:p1], OVvv)
        OVvv = None
    Wvvaa = Wvvaa + Wvvaa.T
    Wvvbb = Wvvbb + Wvvbb.T
    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:Wvvaa += np.einsum('aabb->ab', eris_vvvv) - np.einsum('abba->ab', eris_vvvv)
    #:Wvvbb += np.einsum('aabb->ab', eris_VVVV) - np.einsum('abba->ab', eris_VVVV)
    #:Wvvab += np.einsum('aabb->ab', eris_vvVV)
    if eris.vvvv is not None:
        for i in range(nvira):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvaa[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvaa[i,:i+1] -= tmp
            Wvvaa[:i  ,i] -= tmp[:i]
            vvv = lib.unpack_tril(np.asarray(eris.vvVV[i0:i0+i+1]))
            Wvvab[i] += np.einsum('bb->b', vvv[i])
            vvv = None
        for i in range(nvirb):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.VVVV[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvbb[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvbb[i,:i+1] -= tmp
            Wvvbb[:i  ,i] -= tmp[:i]
            vvv = None
    Wvvba = Wvvab.T
    Hr2aa += Wvvaa.reshape(1,1,nvira,nvira)
    Hr2ab += Wvvab.reshape(1,1,nvira,nvirb)
    Hr2bb += Wvvbb.reshape(1,1,nvirb,nvirb)
    Hr2baaa += Wvvaa.reshape(1,1,nvira,nvira)
    Hr2aaba += Wvvba.reshape(1,1,nvirb,nvira)
    Hr2abbb += Wvvbb.reshape(1,1,nvirb,nvirb)
    Hr2bbab += Wvvab.reshape(1,1,nvira,nvirb)

    vec_ee = amplitudes_to_vector_ee((Hr1aa,Hr1bb), (Hr2aa,Hr2ab,Hr2bb))
    vec_sf = amplitudes_to_vector_eomsf((Hr1ab,Hr1ba), (Hr2baaa,Hr2aaba,Hr2abbb,Hr2bbab))
    return vec_ee, vec_sf

class EOMEE(eom_rccsd.EOMEE):
    def __init__(self, cc):
        eom_rccsd.EOMEE.__init__(self, cc)
        self.nocc = cc.get_nocc()
        self.nmo = cc.get_nmo()

    kernel = eeccsd
    eeccsd = eeccsd
    get_diag = eeccsd_diag

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocc = np.sum(self.nocc)
        nvir = np.sum(self.nmo) - nocc
        return nocc*nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds

class EOMEESpinKeep(EOMEE):
    kernel = eomee_ccsd
    eomee_ccsd = eomee_ccsd
    matvec = eomee_ccsd_matvec
    get_diag = eeccsd_diag

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if koopmans:
            nocca, noccb = self.nocc
            nmoa, nmob = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
# amplitudes are compressed by the function amplitudes_to_vector_ee. sizea is
# the offset in the compressed vector that points to the amplitudes R1_beta
# The addresses of R1_alpha and R1_beta are not contiguous in the compressed
# vector.
            sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
            diag = np.append(diag[:nocca*nvira], diag[sizea:sizea+noccb*nvirb])
            addr = np.append(np.arange(nocca*nvira),
                             np.arange(sizea,sizea+noccb*nvirb))
            idx = addr[diag.argsort()]
        else:
            idx = diag.argsort()

        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[0]
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ee(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ee(r1, r2)

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
        sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
        sizeab = nocca * noccb * nvira * nvirb
        return sizea+sizeb+sizeab

class EOMEESpinFlip(EOMEE):
    kernel = eomsf_ccsd
    eomsf_ccsd = eomsf_ccsd
    matvec = eomsf_ccsd_matvec

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if koopmans:
            nocca, noccb = self.nocc
            nmoa, nmob = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
            idx = diag[:nocca*nvirb+noccb*nvira].argsort()
        else:
            idx = diag.argsort()

        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[1]
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_eomsf(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_eomsf(r1, r2)

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb

        nbaaa = noccb*nocca*nvira*(nvira-1)//2
        naaba = nocca*(nocca-1)//2*nvirb*nvira
        nabbb = nocca*noccb*nvirb*(nvirb-1)//2
        nbbab = noccb*(noccb-1)//2*nvira*nvirb
        return nocca*nvirb + noccb*nvira + nbaaa + naaba + nabbb + nbbab


class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> uintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def make_ip(self):
        raise NotImplementedError

    def make_ea(self):
        raise NotImplementedError

    def make_ee(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape

        fooa = eris.focka[:nocca,:nocca]
        foob = eris.fockb[:noccb,:noccb]
        fova = eris.focka[:nocca,nocca:]
        fovb = eris.fockb[:noccb,noccb:]
        fvva = eris.focka[nocca:,nocca:]
        fvvb = eris.fockb[noccb:,noccb:]

        self.Fooa = np.zeros((nocca,nocca))
        self.Foob = np.zeros((noccb,noccb))
        self.Fvva = np.zeros((nvira,nvira))
        self.Fvvb = np.zeros((nvirb,nvirb))

        wovvo = np.zeros((nocca,nvira,nvira,nocca))
        wOVVO = np.zeros((noccb,nvirb,nvirb,noccb))
        woVvO = np.zeros((nocca,nvirb,nvira,noccb))
        woVVo = np.zeros((nocca,nvirb,nvirb,nocca))
        wOvVo = np.zeros((noccb,nvira,nvirb,nocca))
        wOvvO = np.zeros((noccb,nvira,nvira,noccb))

        wovoo = np.zeros((nocca,nvira,nocca,nocca))
        wOVOO = np.zeros((noccb,nvirb,noccb,noccb))
        woVoO = np.zeros((nocca,nvirb,nocca,noccb))
        wOvOo = np.zeros((noccb,nvira,noccb,nocca))

        tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #:self.Fvva  = np.einsum('mf,mfae->ae', t1a, ovvv)
        #:self.wovvo = lib.einsum('jf,mebf->mbej', t1a, ovvv)
        #:self.wovoo  = 0.5 * lib.einsum('mebf,ijef->mbij', eris_ovvv, tauaa)
        #:self.wovoo -= 0.5 * lib.einsum('mfbe,ijef->mbij', eris_ovvv, tauaa)
        mem_now = lib.current_memory()[0]
        max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
        blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3))))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            self.Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] = lib.einsum('jf,mebf->mbej', t1a, ovvv)
            wovoo[p0:p1] = 0.5 * lib.einsum('mebf,ijef->mbij', ovvv, tauaa)
            ovvv = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #:self.Fvvb  = np.einsum('mf,mfae->ae', t1b, OVVV)
        #:self.wOVVO = lib.einsum('jf,mebf->mbej', t1b, OVVV)
        #:self.wOVOO  = 0.5 * lib.einsum('mebf,ijef->mbij', OVVV, taubb)
        blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3))))
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            self.Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            wOVOO[p0:p1] = 0.5 * lib.einsum('mebf,ijef->mbij', OVVV, taubb)
            OVVV = None

        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:self.Fvvb += np.einsum('mf,mfAE->AE', t1a, eris_ovVV)
        #:self.woVvO = lib.einsum('JF,meBF->mBeJ', t1b, eris_ovVV)
        #:self.woVVo = lib.einsum('jf,mfBE->mBEj',-t1a, eris_ovVV)
        #:self.woVoO  = 0.5 * lib.einsum('meBF,iJeF->mBiJ', eris_ovVV, tauab)
        #:self.woVoO += 0.5 * lib.einsum('mfBE,iJfE->mBiJ', eris_ovVV, tauab)
        blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            self.Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            woVoO[p0:p1] = 0.5 * lib.einsum('meBF,iJeF->mBiJ', ovVV, tauab)
            woVoO[p0:p1]+= 0.5 * lib.einsum('mfBE,iJfE->mBiJ', ovVV, tauab)
            ovVV = None

        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:self.Fvva += np.einsum('MF,MFae->ae', t1b, eris_OVvv)
        #:self.wOvVo = lib.einsum('jf,MEbf->MbEj', t1a, eris_OVvv)
        #:self.wOvvO = lib.einsum('JF,MFbe->MbeJ',-t1b, eris_OVvv)
        #:self.wOvOo  = 0.5 * lib.einsum('MEbf,jIfE->MbIj', eris_OVvv, tauab)
        #:self.wOvOo += 0.5 * lib.einsum('MFbe,jIeF->MbIj', eris_OVvv, tauab)
        blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            self.Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            wOvOo[p0:p1] = 0.5 * lib.einsum('MEbf,jIfE->MbIj', OVvv, tauab)
            wOvOo[p0:p1]+= 0.5 * lib.einsum('MFbe,jIeF->MbIj', OVvv, tauab)
            OVvv = None

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        self.Fova = np.einsum('nf,menf->me', t1a,      ovov)
        self.Fova+= np.einsum('NF,meNF->me', t1b, eris_ovOV)
        self.Fova += fova
        self.Fovb = np.einsum('nf,menf->me', t1b,      OVOV)
        self.Fovb+= np.einsum('nf,nfME->ME', t1a, eris_ovOV)
        self.Fovb += fovb
        tilaa, tilab, tilbb = uccsd.make_tau(t2,t1,t1,fac=0.5)
        self.Fooa  = lib.einsum('inef,menf->mi', tilaa, eris_ovov)
        self.Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
        self.Foob  = lib.einsum('inef,menf->mi', tilbb, eris_OVOV)
        self.Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)
        self.Fvva -= lib.einsum('mnaf,menf->ae', tilaa, eris_ovov)
        self.Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
        self.Fvvb -= lib.einsum('mnaf,menf->ae', tilbb, eris_OVOV)
        self.Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
        wovvo -= lib.einsum('jnfb,menf->mbej', t2aa,      ovov)
        wovvo += lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
        wOVVO -= lib.einsum('jnfb,menf->mbej', t2bb,      OVOV)
        wOVVO += lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
        woVvO += lib.einsum('nJfB,menf->mBeJ', t2ab,      ovov)
        woVvO -= lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
        wOvVo -= lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
        wOvVo += lib.einsum('jNbF,MENF->MbEj', t2ab,      OVOV)
        woVVo += lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
        wOvvO += lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)

        eris_ovoo = np.asarray(eris.ovoo)
        eris_OVOO = np.asarray(eris.OVOO)
        eris_OVoo = np.asarray(eris.OVoo)
        eris_ovOO = np.asarray(eris.ovOO)
        self.Fooa += np.einsum('ne,nemi->mi', t1a, eris_ovoo)
        self.Fooa -= np.einsum('ne,meni->mi', t1a, eris_ovoo)
        self.Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
        self.Foob += np.einsum('ne,nemi->mi', t1b, eris_OVOO)
        self.Foob -= np.einsum('ne,meni->mi', t1b, eris_OVOO)
        self.Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)
        eris_ovoo = eris_ovoo + np.einsum('nfme,jf->menj', eris_ovov, t1a)
        eris_OVOO = eris_OVOO + np.einsum('nfme,jf->menj', eris_OVOV, t1b)
        eris_OVoo = eris_OVoo + np.einsum('nfme,jf->menj', eris_ovOV, t1a)
        eris_ovOO = eris_ovOO + np.einsum('menf,jf->menj', eris_ovOV, t1b)
        ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
        OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
        wovvo += lib.einsum('nb,nemj->mbej', t1a,      ovoo)
        wOVVO += lib.einsum('nb,nemj->mbej', t1b,      OVOO)
        woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
        wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
        woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
        wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)
        eris_ooov = eris_OOOV = eris_OOov = eris_ooOV = None

        self.Fooa += fooa + 0.5*lib.einsum('me,ie->mi', self.Fova+fova, t1a)
        self.Foob += foob + 0.5*lib.einsum('me,ie->mi', self.Fovb+fovb, t1b)
        self.Fvva += fvva - 0.5*lib.einsum('me,ma->ae', self.Fova+fova, t1a)
        self.Fvvb += fvvb - 0.5*lib.einsum('me,ma->ae', self.Fovb+fovb, t1b)

        # 0 or 1 virtuals
        eris_ovoo = np.asarray(eris.ovoo)
        eris_OVOO = np.asarray(eris.OVOO)
        eris_OVoo = np.asarray(eris.OVoo)
        eris_ovOO = np.asarray(eris.ovOO)
        ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
        OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
        woooo = lib.einsum('je,nemi->mnij', t1a,      ovoo)
        wOOOO = lib.einsum('je,nemi->mnij', t1b,      OVOO)
        woOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
        woOOo = lib.einsum('je,meNI->mNIj',-t1a, eris_ovOO)
        tmpaa = lib.einsum('nemi,jnbe->mbij',      ovoo, t2aa)
        tmpaa+= lib.einsum('NEmi,jNbE->mbij', eris_OVoo, t2ab)
        tmpbb = lib.einsum('nemi,jnbe->mbij',      OVOO, t2bb)
        tmpbb+= lib.einsum('neMI,nJeB->MBIJ', eris_ovOO, t2ab)
        woVoO += lib.einsum('nemi,nJeB->mBiJ',      ovoo, t2ab)
        woVoO += lib.einsum('NEmi,JNBE->mBiJ', eris_OVoo, t2bb)
        woVoO -= lib.einsum('meNI,jNeB->mBjI', eris_ovOO, t2ab)
        wOvOo += lib.einsum('NEMI,jNbE->MbIj',      OVOO, t2ab)
        wOvOo += lib.einsum('neMI,jnbe->MbIj', eris_ovOO, t2aa)
        wOvOo -= lib.einsum('MEni,nJbE->MbJi', eris_OVoo, t2ab)
        wovoo += tmpaa - tmpaa.transpose(0,1,3,2)
        wOVOO += tmpbb - tmpbb.transpose(0,1,3,2)
        self.wooov =      ovoo.transpose(2,0,3,1).copy()
        self.wOOOV =      OVOO.transpose(2,0,3,1).copy()
        self.woOoV = eris_OVoo.transpose(2,0,3,1).copy()
        self.wOoOv = eris_ovOO.transpose(2,0,3,1).copy()
        self.wOooV =-eris_OVoo.transpose(0,2,3,1).copy()
        self.woOOv =-eris_ovOO.transpose(0,2,3,1).copy()
        eris_ovoo = eris_OVOO = eris_ovOO = eris_OVoo = None

        woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
        wOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
        woOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
        self.woooo = woooo - woooo.transpose(0,1,3,2)
        self.wOOOO = wOOOO - wOOOO.transpose(0,1,3,2)
        self.woOoO = woOoO - woOOo.transpose(0,1,3,2)

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        tauaa, tauab, taubb = uccsd.make_tau(t2,t1,t1)
        self.woooo += 0.5*lib.einsum('ijef,menf->mnij', tauaa,      ovov)
        self.wOOOO += 0.5*lib.einsum('ijef,menf->mnij', taubb,      OVOV)
        self.woOoO +=     lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)

        self.wooov += lib.einsum('if,mfne->mnie', t1a,      ovov)
        self.wOOOV += lib.einsum('if,mfne->mnie', t1b,      OVOV)
        self.woOoV += lib.einsum('if,mfNE->mNiE', t1a, eris_ovOV)
        self.wOoOv += lib.einsum('IF,neMF->MnIe', t1b, eris_ovOV)
        self.wOooV -= lib.einsum('if,nfME->MniE', t1a, eris_ovOV)
        self.woOOv -= lib.einsum('IF,meNF->mNIe', t1b, eris_ovOV)

        tmp1aa = lib.einsum('njbf,menf->mbej', t2aa,      ovov)
        tmp1aa-= lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
        tmp1bb = lib.einsum('njbf,menf->mbej', t2bb,      OVOV)
        tmp1bb-= lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
        tmp1ab = lib.einsum('NJBF,meNF->mBeJ', t2bb, eris_ovOV)
        tmp1ab-= lib.einsum('nJfB,menf->mBeJ', t2ab,      ovov)
        tmp1ba = lib.einsum('njbf,nfME->MbEj', t2aa, eris_ovOV)
        tmp1ba-= lib.einsum('jNbF,MENF->MbEj', t2ab,      OVOV)
        tmp1abba =-lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
        tmp1baab =-lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
        tmpaa = lib.einsum('ie,mbej->mbij', t1a, tmp1aa)
        tmpbb = lib.einsum('ie,mbej->mbij', t1b, tmp1bb)
        tmpab = lib.einsum('ie,mBeJ->mBiJ', t1a, tmp1ab)
        tmpab-= lib.einsum('IE,mBEj->mBjI', t1b, tmp1abba)
        tmpba = lib.einsum('IE,MbEj->MbIj', t1b, tmp1ba)
        tmpba-= lib.einsum('ie,MbeJ->MbJi', t1a, tmp1baab)
        wovoo -= tmpaa - tmpaa.transpose(0,1,3,2)
        wOVOO -= tmpbb - tmpbb.transpose(0,1,3,2)
        woVoO -= tmpab
        wOvOo -= tmpba
        eris_ovov = eris_OVOV = eris_ovOV = None
        eris_ovoo = np.asarray(eris.ovoo)
        eris_OVOO = np.asarray(eris.OVOO)
        eris_ovOO = np.asarray(eris.ovOO)
        eris_OVoo = np.asarray(eris.OVoo)
        wovoo += eris_ovoo.transpose(3,1,2,0) - eris_ovoo.transpose(2,1,0,3)
        wOVOO += eris_OVOO.transpose(3,1,2,0) - eris_OVOO.transpose(2,1,0,3)
        woVoO += eris_OVoo.transpose(3,1,2,0)
        wOvOo += eris_ovOO.transpose(3,1,2,0)
        eris_ovoo = eris_OVOO = eris_ovOO = eris_OVoo = None

        eris_ovvo = np.asarray(eris.ovvo)
        eris_OVVO = np.asarray(eris.OVVO)
        eris_OVvo = np.asarray(eris.OVvo)
        eris_ovVO = np.asarray(eris.ovVO)
        eris_oovv = np.asarray(eris.oovv)
        eris_OOVV = np.asarray(eris.OOVV)
        eris_OOvv = np.asarray(eris.OOvv)
        eris_ooVV = np.asarray(eris.ooVV)
        wovvo += eris_ovvo.transpose(0,2,1,3)
        wOVVO += eris_OVVO.transpose(0,2,1,3)
        woVvO += eris_ovVO.transpose(0,2,1,3)
        wOvVo += eris_OVvo.transpose(0,2,1,3)
        wovvo -= eris_oovv.transpose(0,2,3,1)
        wOVVO -= eris_OOVV.transpose(0,2,3,1)
        woVVo -= eris_ooVV.transpose(0,2,3,1)
        wOvvO -= eris_OOvv.transpose(0,2,3,1)

        tmpaa = lib.einsum('ie,mebj->mbij', t1a, eris_ovvo)
        tmpbb = lib.einsum('ie,mebj->mbij', t1b, eris_OVVO)
        tmpaa-= lib.einsum('ie,mjbe->mbij', t1a, eris_oovv)
        tmpbb-= lib.einsum('ie,mjbe->mbij', t1b, eris_OOVV)
        woVoO += lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
        woVoO -= lib.einsum('IE,mjBE->mBjI',-t1b, eris_ooVV)
        wOvOo += lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
        wOvOo -= lib.einsum('ie,MJbe->MbJi',-t1a, eris_OOvv)
        wovoo += tmpaa - tmpaa.transpose(0,1,3,2)
        wOVOO += tmpbb - tmpbb.transpose(0,1,3,2)
        wovoo -= lib.einsum('me,ijbe->mbij', self.Fova, t2aa)
        wOVOO -= lib.einsum('me,ijbe->mbij', self.Fovb, t2bb)
        woVoO += lib.einsum('me,iJeB->mBiJ', self.Fova, t2ab)
        wOvOo += lib.einsum('ME,jIbE->MbIj', self.Fovb, t2ab)
        wovoo -= lib.einsum('nb,mnij->mbij', t1a, self.woooo)
        wOVOO -= lib.einsum('nb,mnij->mbij', t1b, self.wOOOO)
        woVoO -= lib.einsum('NB,mNiJ->mBiJ', t1b, self.woOoO)
        wOvOo -= lib.einsum('nb,nMjI->MbIj', t1a, self.woOoO)
        eris_ovvo = eris_OVVO = eris_OVvo = eris_ovVO = None
        eris_oovv = eris_OOVV = eris_OOvv = eris_ooVV = None

        self.saved = lib.H5TmpFile()
        self.saved['ovvo'] = wovvo
        self.saved['OVVO'] = wOVVO
        self.saved['oVvO'] = woVvO
        self.saved['OvVo'] = wOvVo
        self.saved['oVVo'] = woVVo
        self.saved['OvvO'] = wOvvO
        self.wovvo = self.saved['ovvo']
        self.wOVVO = self.saved['OVVO']
        self.woVvO = self.saved['oVvO']
        self.wOvVo = self.saved['OvVo']
        self.woVVo = self.saved['oVVo']
        self.wOvvO = self.saved['OvvO']
        self.saved['ovoo'] = wovoo
        self.saved['OVOO'] = wOVOO
        self.saved['oVoO'] = woVoO
        self.saved['OvOo'] = wOvOo
        self.wovoo = self.saved['ovoo']
        self.wOVOO = self.saved['OVOO']
        self.woVoO = self.saved['oVoO']
        self.wOvOo = self.saved['OvOo']

        self.wvovv = self.saved.create_dataset('vovv', (nvira,nocca,nvira,nvira), t1a.dtype.char)
        self.wVOVV = self.saved.create_dataset('VOVV', (nvirb,noccb,nvirb,nvirb), t1a.dtype.char)
        self.wvOvV = self.saved.create_dataset('vOvV', (nvira,noccb,nvira,nvirb), t1a.dtype.char)
        self.wVoVv = self.saved.create_dataset('VoVv', (nvirb,nocca,nvirb,nvira), t1a.dtype.char)

        # 3 or 4 virtuals
        eris_ovoo = np.asarray(eris.ovoo)
        eris_ovov = np.asarray(eris.ovov)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        eris_oovv = np.asarray(eris.oovv)
        eris_ovvo = np.asarray(eris.ovvo)
        oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
        eris_oovv = eris_ovvo = None
        #:wvovv  = .5 * lib.einsum('meni,mnab->eiab', eris_ovoo, tauaa)
        #:wvovv -= .5 * lib.einsum('me,miab->eiab', self.Fova, t2aa)
        #:tmp1aa = lib.einsum('nibf,menf->mbei', t2aa,      ovov)
        #:tmp1aa-= lib.einsum('iNbF,meNF->mbei', t2ab, eris_ovOV)
        #:wvovv+= lib.einsum('ma,mbei->eiab', t1a, tmp1aa)
        #:wvovv+= lib.einsum('ma,mibe->eiab', t1a,      oovv)
        for p0, p1 in lib.prange(0, nvira, nocca):
            wvovv  = .5*lib.einsum('meni,mnab->eiab', eris_ovoo[:,p0:p1], tauaa)
            wvovv -= .5*lib.einsum('me,miab->eiab', self.Fova[:,p0:p1], t2aa)

            tmp1aa = lib.einsum('nibf,menf->mbei', t2aa, ovov[:,p0:p1])
            tmp1aa-= lib.einsum('iNbF,meNF->mbei', t2ab, eris_ovOV[:,p0:p1])
            wvovv += lib.einsum('ma,mbei->eiab', t1a, tmp1aa)
            wvovv += lib.einsum('ma,mibe->eiab', t1a, oovv[:,:,:,p0:p1])
            self.wvovv[p0:p1] = wvovv
            tmp1aa = None
        eris_ovov = eris_ovoo = eris_ovOV = None

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #:wvovv += lib.einsum('mebf,miaf->eiab',      ovvv, t2aa)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:wvovv += lib.einsum('MFbe,iMaF->eiab', eris_OVvv, t2ab)
        #:wvovv += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvovv -= wvovv - wvovv.transpose(0,1,3,2)
        mem_now = lib.current_memory()[0]
        max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
        blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*6))))
        for i0,i1 in lib.prange(0, nocca, blksize):
            wvovv = self.wvovv[:,i0:i1]
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
                wvovv -= lib.einsum('MFbe,iMaF->eiab', OVvv, t2ab[i0:i1,p0:p1])
                OVvv = None
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
                if p0 == i0:
                    wvovv += ovvv.transpose(2,0,3,1).conj()
                ovvv = ovvv - ovvv.transpose(0,3,2,1)
                wvovv -= lib.einsum('mebf,miaf->eiab', ovvv, t2aa[p0:p1,i0:i1])
                ovvv = None
            wvovv = wvovv - wvovv.transpose(0,1,3,2)
            self.wvovv[:,i0:i1] = wvovv

        eris_OVOO = np.asarray(eris.OVOO)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        eris_OOVV = np.asarray(eris.OOVV)
        eris_OVVO = np.asarray(eris.OVVO)
        OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
        eris_OOVV = eris_OVVO = None
        #:wVOVV  = .5*lib.einsum('meni,mnab->eiab', eris_OVOO, taubb)
        #:wVOVV -= .5*lib.einsum('me,miab->eiab', self.Fovb, t2bb)
        #:tmp1bb = lib.einsum('nibf,menf->mbei', t2bb,      OVOV)
        #:tmp1bb-= lib.einsum('nIfB,nfME->MBEI', t2ab, eris_ovOV)
        #:wVOVV += lib.einsum('ma,mbei->eiab', t1b, tmp1bb)
        #:wVOVV += lib.einsum('ma,mibe->eiab', t1b,      OOVV)
        for p0, p1 in lib.prange(0, nvirb, noccb):
            wVOVV  = .5*lib.einsum('meni,mnab->eiab', eris_OVOO[:,p0:p1], taubb)
            wVOVV -= .5*lib.einsum('me,miab->eiab', self.Fovb[:,p0:p1], t2bb)

            tmp1bb = lib.einsum('nibf,menf->mbei', t2bb, OVOV[:,p0:p1])
            tmp1bb-= lib.einsum('nIfB,nfME->MBEI', t2ab, eris_ovOV[:,:,:,p0:p1])
            wVOVV += lib.einsum('ma,mbei->eiab', t1b, tmp1bb)
            wVOVV += lib.einsum('ma,mibe->eiab', t1b, OOVV[:,:,:,p0:p1])
            self.wVOVV[p0:p1] = wVOVV
            tmp1bb = None
        eris_OVOV = eris_OVOO = eris_ovOV = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #:wVOVV -= lib.einsum('MEBF,MIAF->EIAB',      OVVV, t2bb)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:wVOVV -= lib.einsum('mfBE,mIfA->EIAB', eris_ovVV, t2ab)
        #:wVOVV += eris_OVVV.transpose(2,0,3,1).conj()
        #:self.wVOVV += wVOVV - wVOVV.transpose(0,1,3,2)
        blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*6))))
        for i0,i1 in lib.prange(0, noccb, blksize):
            wVOVV = self.wVOVV[:,i0:i1]
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
                wVOVV -= lib.einsum('mfBE,mIfA->EIAB', ovVV, t2ab[p0:p1,i0:i1])
                ovVV = None
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
                if p0 == i0:
                    wVOVV += OVVV.transpose(2,0,3,1).conj()
                OVVV = OVVV - OVVV.transpose(0,3,2,1)
                wVOVV -= lib.einsum('mebf,miaf->eiab', OVVV, t2bb[p0:p1,i0:i1])
                OVVV = None
            wVOVV = wVOVV - wVOVV.transpose(0,1,3,2)
            self.wVOVV[:,i0:i1] = wVOVV

        eris_ovOV = np.asarray(eris.ovOV)
        eris_ovOO = np.asarray(eris.ovOO)
        eris_OOvv = np.asarray(eris.OOvv)
        eris_ovVO = np.asarray(eris.ovVO)
        #:self.wvOvV = lib.einsum('meNI,mNaB->eIaB', eris_ovOO, tauab)
        #:self.wvOvV -= lib.einsum('me,mIaB->eIaB', self.Fova, t2ab)
        #:tmp1ab = lib.einsum('NIBF,meNF->mBeI', t2bb, eris_ovOV)
        #:tmp1ab-= lib.einsum('nIfB,menf->mBeI', t2ab,      ovov)
        #:tmp1baab = lib.einsum('nIbF,neMF->MbeI', t2ab, eris_ovOV)
        #:tmpab = lib.einsum('ma,mBeI->eIaB', t1a, tmp1ab)
        #:tmpab+= lib.einsum('MA,MbeI->eIbA', t1b, tmp1baab)
        #:tmpab-= lib.einsum('MA,MIbe->eIbA', t1b, eris_OOvv)
        #:tmpab-= lib.einsum('ma,meBI->eIaB', t1a, eris_ovVO)
        #:self.wvOvV += tmpab
        for p0, p1 in lib.prange(0, nvira, nocca):
            wvOvV  = lib.einsum('meNI,mNaB->eIaB', eris_ovOO[:,p0:p1], tauab)
            wvOvV -= lib.einsum('me,mIaB->eIaB', self.Fova[:,p0:p1], t2ab)
            tmp1ab = lib.einsum('NIBF,meNF->mBeI', t2bb, eris_ovOV[:,p0:p1])
            tmp1ab-= lib.einsum('nIfB,menf->mBeI', t2ab, ovov[:,p0:p1])
            wvOvV+= lib.einsum('ma,mBeI->eIaB', t1a, tmp1ab)
            tmp1ab = None
            tmp1baab = lib.einsum('nIbF,neMF->MbeI', t2ab, eris_ovOV[:,p0:p1])
            wvOvV+= lib.einsum('MA,MbeI->eIbA', t1b, tmp1baab)
            tmp1baab = None
            wvOvV-= lib.einsum('MA,MIbe->eIbA', t1b, eris_OOvv[:,:,:,p0:p1])
            wvOvV-= lib.einsum('ma,meBI->eIaB', t1a, eris_ovVO[:,p0:p1])
            self.wvOvV[p0:p1] = wvOvV
        eris_ovOV = eris_ovOO = eris_OOvv = eris_ovVO = None

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #:self.wvOvV -= lib.einsum('mebf,mIfA->eIbA',      ovvv, t2ab)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:self.wvOvV -= lib.einsum('meBF,mIaF->eIaB', eris_ovVV, t2ab)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:self.wvOvV -= lib.einsum('MFbe,MIAF->eIbA', eris_OVvv, t2bb)
        #:self.wvOvV += eris_OVvv.transpose(2,0,3,1).conj()
        blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*6))))
        for i0,i1 in lib.prange(0, noccb, blksize):
            wvOvV = self.wvOvV[:,i0:i1]
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
                wvOvV -= lib.einsum('meBF,mIaF->eIaB', ovVV, t2ab[p0:p1,i0:i1])
                ovVV = None
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
                ovvv = ovvv - ovvv.transpose(0,3,2,1)
                wvOvV -= lib.einsum('mebf,mIfA->eIbA',ovvv, t2ab[p0:p1,i0:i1])
                ovvv = None
            self.wvOvV[:,i0:i1] = wvOvV

        blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
        for i0,i1 in lib.prange(0, noccb, blksize):
            wvOvV = self.wvOvV[:,i0:i1]
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
                if p0 == i0:
                    wvOvV += OVvv.transpose(2,0,3,1).conj()
                wvOvV -= lib.einsum('MFbe,MIAF->eIbA', OVvv, t2bb[p0:p1,i0:i1])
                OVvv = None
            self.wvOvV[:,i0:i1] = wvOvV

        eris_ovOV = np.asarray(eris.ovOV)
        eris_OVoo = np.asarray(eris.OVoo)
        eris_ooVV = np.asarray(eris.ooVV)
        eris_OVvo = np.asarray(eris.OVvo)
        #:self.wVoVv = lib.einsum('MEni,nMbA->EiAb', eris_OVoo, tauab)
        #:self.wVoVv -= lib.einsum('ME,iMbA->EiAb', self.Fovb, t2ab)
        #:tmp1ba = lib.einsum('nibf,nfME->MbEi', t2aa, eris_ovOV)
        #:tmp1ba-= lib.einsum('iNbF,MENF->MbEi', t2ab,      OVOV)
        #:tmp1abba = lib.einsum('iNfB,mfNE->mBEi', t2ab, eris_ovOV)
        #:tmpba = lib.einsum('MA,MbEi->EiAb', t1b, tmp1ba)
        #:tmpba+= lib.einsum('ma,mBEi->EiBa', t1a, tmp1abba)
        #:tmpba-= lib.einsum('ma,miBE->EiBa', t1a, eris_ooVV)
        #:tmpba-= lib.einsum('MA,MEbi->EiAb', t1b, eris_OVvo)
        #:self.wVoVv += tmpba
        for p0, p1 in lib.prange(0, nvirb, noccb):
            wVoVv  = lib.einsum('MEni,nMbA->EiAb', eris_OVoo[:,p0:p1], tauab)
            wVoVv -= lib.einsum('ME,iMbA->EiAb', self.Fovb[:,p0:p1], t2ab)
            tmp1ba = lib.einsum('nibf,nfME->MbEi', t2aa, eris_ovOV[:,:,:,p0:p1])
            tmp1ba-= lib.einsum('iNbF,MENF->MbEi', t2ab, OVOV[:,p0:p1])
            wVoVv += lib.einsum('MA,MbEi->EiAb', t1b, tmp1ba)
            tmp1ba = None
            tmp1abba = lib.einsum('iNfB,mfNE->mBEi', t2ab, eris_ovOV[:,:,:,p0:p1])
            wVoVv += lib.einsum('ma,mBEi->EiBa', t1a, tmp1abba)
            tmp1abba = None
            wVoVv -= lib.einsum('ma,miBE->EiBa', t1a, eris_ooVV[:,:,:,p0:p1])
            wVoVv -= lib.einsum('MA,MEbi->EiAb', t1b, eris_OVvo[:,p0:p1])
            self.wVoVv[p0:p1] = wVoVv
        eris_ovOV = eris_OVoo = eris_ooVV = eris_OVvo = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #:self.wVoVv -= lib.einsum('MEBF,iMaF->EiBa',      OVVV, t2ab)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:self.wVoVv -= lib.einsum('MEbf,iMfA->EiAb', eris_OVvv, t2ab)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:self.wVoVv -= lib.einsum('mfBE,miaf->EiBa', eris_ovVV, t2aa)
        #:self.wVoVv += eris_ovVV.transpose(2,0,3,1).conj()
        blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*6))))
        for i0,i1 in lib.prange(0, nocca, blksize):
            wVoVv = self.wVoVv[:,i0:i1]
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
                wVoVv -= lib.einsum('MEbf,iMfA->EiAb', OVvv, t2ab[i0:i1,p0:p1])
                OVvv = None
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
                OVVV = OVVV - OVVV.transpose(0,3,2,1)
                wVoVv -= lib.einsum('MEBF,iMaF->EiBa', OVVV, t2ab[i0:i1,p0:p1])
                OVVV = None
            self.wVoVv[:,i0:i1] = wVoVv

        blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
        for i0,i1 in lib.prange(0, nocca, blksize):
            wVoVv = self.wVoVv[:,i0:i1]
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
                if p0 == i0:
                    wVoVv += ovVV.transpose(2,0,3,1).conj()
                wVoVv -= lib.einsum('mfBE,miaf->EiBa', ovVV, t2aa[p0:p1,i0:i1])
                ovVV = None
            self.wVoVv[:,i0:i1] = wVoVv

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)

