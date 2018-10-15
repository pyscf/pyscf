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
# Authors: James D. McClain
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#

import time
from functools import reduce
import itertools
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from pyscf.pbc.cc import kintermediates_uhf

einsum = lib.einsum

########################################
# EOM-IP-CCSD
########################################

def amplitudes_to_vector_ip(r1, r2):
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    return np.hstack((r1a, r1b,
                      r2aaa.ravel(), r2baa.ravel(),
                      r2abb.ravel(), r2bbb.ravel()))

def vector_to_amplitudes_ip(vector, nkpts, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nocca, noccb, nkpts**2*nocca*nocca*nvira, nkpts**2*noccb*nocca*nvira,
             nkpts**2*nocca*noccb*nvirb, nkpts**2*noccb*noccb*nvirb)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2aaa, r2baa, r2abb, r2bbb = np.split(vector, sections)

    r2aaa = r2aaa.reshape(nkpts,nkpts,nocca,nocca,nvira).copy()
    r2baa = r2baa.reshape(nkpts,nkpts,noccb,nocca,nvira).copy()
    r2abb = r2abb.reshape(nkpts,nkpts,nocca,noccb,nvirb).copy()
    r2bbb = r2bbb.reshape(nkpts,nkpts,noccb,noccb,nvirb).copy()

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2baa, r2abb, r2bbb)
    return r1, r2

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    nocca, noccb = nocc
    nvira, nvirb = nvir
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = eom.vector_to_amplitudes(vector, nkpts, nmo[0]+nmo[1], nocc[0]+nocc[1])  # spin
    orbspin = imds.eris.orbspin
    spatial_r1, spatial_r2 = eom_kgccsd.spin2spatial_ip_doublet(r1, r2, kconserv, kshift, orbspin)
    uccsd_imds = imds._uccsd_imds
    t2aa, t2ab, t2bb = t2

    # k-point spin orbital version of ipccsd

    Hr1 = -0.0*np.einsum('mi,m->i', imds.Foo[kshift], r1)

    Hr2 = np.zeros_like(r2)

    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        for km in range(nkpts):
            ke = kconserv[km, kshift, kj]
            Hr2[ki, kj] += lib.einsum('maei,mje->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, kj])

            ke = kconserv[km, kshift, ki]
            Hr2[ki, kj] -= lib.einsum('maej,mie->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, ki])

    r1a, r1b = spatial_r1
    Hr1a = np.zeros((nocca), dtype=r1.dtype)
    Hr1b = np.zeros((noccb), dtype=r1.dtype)

    r2aaa, r2baa, r2abb, r2bbb = spatial_r2
    #Foo term
    for km in range(nkpts):
        Hr1a += np.einsum('me,mie->i', uccsd_imds.Fov[km], r2aaa[km,kshift])
        Hr1a -= np.einsum('ME,iME->i', uccsd_imds.FOV[km], r2abb[kshift,km])
        Hr1b += np.einsum('ME,MIE->I', uccsd_imds.FOV[km], r2bbb[km,kshift])
        Hr1b -= np.einsum('me,Ime->I', uccsd_imds.Fov[km], r2baa[kshift,km])
    #Fov term
    spatial_Foo = uccsd_imds.Foo
    spatial_FOO = uccsd_imds.FOO
    Hr1a += -np.einsum('mi,m->i', spatial_Foo[kshift], r1a)
    Hr1b += -np.einsum('MI,M->I', spatial_FOO[kshift], r1b)
    #Wooov
    for km in range(nkpts):
        for kn in range(nkpts):
            Hr1a += -0.5 * np.einsum('nime,mne->i', uccsd_imds.Wooov[kn,kshift,km], r2aaa[km,kn])
            Hr1b +=        np.einsum('NIme,Nme->I', uccsd_imds.WOOov[kn,kshift,km], r2baa[kn,km])
            Hr1b += -0.5 * np.einsum('NIME,MNE->I', uccsd_imds.WOOOV[kn,kshift,km], r2bbb[km,kn])
            Hr1a +=        np.einsum('niME,nME->i', uccsd_imds.WooOV[kn,kshift,km], r2abb[kn,km])

    Hr2aaa = np.zeros((nkpts, nkpts, nocca, nocca, nvira), dtype=r2.dtype)
    Hr2baa = np.zeros((nkpts, nkpts, noccb, nocca, nvira), dtype=r2.dtype)
    Hr2abb = np.zeros((nkpts, nkpts, nocca, noccb, nvirb), dtype=r2.dtype)
    Hr2bbb = np.zeros((nkpts, nkpts, noccb, noccb, nvirb), dtype=r2.dtype)

    # Fvv term
    for kb, ki in itertools.product(range(nkpts),repeat=2):
        kj = kconserv[kshift,ki,kb]
        Hr2aaa[ki,kj] += lib.einsum('be,ije->ijb', uccsd_imds.Fvv[kb], r2aaa[ki,kj])
        Hr2abb[ki,kj] += lib.einsum('BE,iJE->iJB', uccsd_imds.FVV[kb], r2abb[ki,kj])
        Hr2bbb[ki,kj] += lib.einsum('BE,IJE->IJB', uccsd_imds.FVV[kb], r2bbb[ki,kj])
        Hr2baa[ki,kj] += lib.einsum('be,Ije->Ijb', uccsd_imds.Fvv[kb], r2baa[ki,kj])

    # Foo term
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        tmpa = lib.einsum('mi,mjb->ijb', uccsd_imds.Foo[ki], r2aaa[ki,kj])
        tmpb = lib.einsum('mj,mib->ijb', uccsd_imds.Foo[kj], r2aaa[kj,ki])
        Hr2aaa[ki,kj] -= tmpa - tmpb
        Hr2abb[ki,kj] -= lib.einsum('mi,mJB->iJB', uccsd_imds.Foo[ki], r2abb[ki,kj])
        Hr2abb[ki,kj] -= lib.einsum('MJ,iMB->iJB', uccsd_imds.FOO[kj], r2abb[ki,kj])
        Hr2baa[ki,kj] -= lib.einsum('MI,Mjb->Ijb', uccsd_imds.FOO[ki], r2baa[ki,kj])
        Hr2baa[ki,kj] -= lib.einsum('mj,Imb->Ijb', uccsd_imds.Foo[kj], r2baa[ki,kj])
        tmpb = lib.einsum('MI,MJB->IJB', uccsd_imds.FOO[ki], r2bbb[ki,kj])
        tmpa = lib.einsum('MJ,MIB->IJB', uccsd_imds.FOO[kj], r2bbb[kj,ki])
        Hr2bbb[ki,kj] -= tmpb - tmpa

    # Wovoo term
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]
        Hr2aaa[ki,kj] -= np.einsum('mjbi,m->ijb', uccsd_imds.Woovo[kshift,kj,kb], r1a)
        Hr2abb[ki,kj] += np.einsum('miBJ,m->iJB', uccsd_imds.WooVO[kshift,ki,kb], r1a)
        Hr2baa[ki,kj] += np.einsum('MIbj,M->Ijb', uccsd_imds.WOOvo[kshift,ki,kb], r1b)
        Hr2bbb[ki,kj] -= np.einsum('MJBI,M->IJB', uccsd_imds.WOOVO[kshift,kj,kb], r1b)

    # Woooo term
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]
        for kn in range(nkpts):
            km = kconserv[kj, kn, ki]
            Hr2aaa[ki, kj] += .5 * lib.einsum('minj,mnb->ijb', uccsd_imds.Woooo[km, ki, kn], r2aaa[km, kn])
            Hr2abb[ki, kj] +=      lib.einsum('miNJ,mNB->iJB', uccsd_imds.WooOO[km, ki, kn], r2abb[km, kn])
            Hr2bbb[ki, kj] += .5 * lib.einsum('MINJ,MNB->IJB', uccsd_imds.WOOOO[km, ki, kn], r2bbb[km, kn])
            Hr2baa[ki, kj] +=      lib.einsum('njMI,Mnb->Ijb', uccsd_imds.WooOO[kn, kj, km], r2baa[km, kn])

    ## T2 term
    tmp_aaa = lib.einsum('xymenf,xymnf->e', uccsd_imds.Wovov[:,kshift,:], r2aaa)
    tmp_bbb = lib.einsum('xyMENF,xyMNF->E', uccsd_imds.WOVOV[:,kshift,:], r2bbb)
    tmp_abb = lib.einsum('xymeNF,xymNF->e', uccsd_imds.WovOV[:,kshift,:], r2abb)
    tmp_baa = np.zeros(tmp_bbb.shape, dtype=tmp_bbb.dtype)
    for km, kn in itertools.product(range(nkpts), repeat=2):
        kf = kconserv[kn, kshift, km]
        tmp_baa += lib.einsum('nfME, Mnf->E', uccsd_imds.WovOV[kn, kf, km], r2baa[km, kn])


    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]

        Hr2aaa[ki,kj] -= 0.5 * lib.einsum('e,jibe->ijb', tmp_aaa, t2aa[kj,ki,kb])
        Hr2aaa[ki,kj] -= lib.einsum('e,jibe->ijb', tmp_abb, t2aa[kj,ki,kb])

        Hr2abb[ki,kj] -= 0.5 * lib.einsum('e,iJeB->iJB', tmp_aaa, t2ab[ki,kj,kshift])
        Hr2abb[ki,kj] -= lib.einsum('e,iJeB->iJB', tmp_abb, t2ab[ki,kj,kshift])

        Hr2baa[ki,kj] -= 0.5 * lib.einsum('E,jIbE->Ijb', tmp_bbb, t2ab[kj,ki,kb])
        Hr2baa[ki,kj] -= lib.einsum('E,jIbE->Ijb', tmp_baa, t2ab[kj,ki,kb])

        Hr2bbb[ki,kj] -= 0.5 * lib.einsum('E,JIBE->IJB', tmp_bbb, t2bb[kj,ki,kb])
        Hr2bbb[ki,kj] -= lib.einsum('E,JIBE->IJB', tmp_baa, t2bb[kj,ki,kb])

    spatial_Hr1 = [Hr1a, Hr1b]
    spatial_Hr2 = [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb]
    spin_Hr1, spin_Hr2 = eom_kgccsd.spatial2spin_ip_doublet(spatial_Hr1, spatial_Hr2,
                                                            kconserv, kshift, orbspin)
    Hr1 += spin_Hr1
    Hr2 += spin_Hr2
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMIP(eom_kgccsd.EOMIP):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMIP.__init__(self, cc)

    #get_diag = ipccsd_diag
    matvec = ipccsd_matvec
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        #TODO check this
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            nocca, noccb = self.nocc
            idx = diag[:nocca + noccb].argsort()
        else:
            idx = diag.argsort()
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def _vector_to_amplitudes(self, vector, nkpts=None, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes_ip(vector, nkpts, nmo, nocc)

    def _amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        nkpts = self.nkpts
        return nocca + noccb + nkpts**2 * (nocca*(nocca - 1)//2*nvira + noccb * nocca * nvira+ nocca * noccb * nvirb + noccb*(noccb - 1)//2*nvirb)

    def _make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ip()
        return imds


########################################
# EOM-EA-CCSD
########################################

def vector_to_amplitudes_ea(vector, nkpts, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nvira, nvirb, nkpts**2*nocca*nvira*nvira, nkpts**2*nocca*nvirb*nvira,
             nkpts**2*noccb*nvira*nvirb, nkpts**2*noccb*nvirb*nvirb)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2a, r2aba, r2bab, r2b = np.split(vector, sections)

    r2aaa = r2aaa.reshape(nocca,nvira,nvira).copy()
    r2aba = r2aba.reshape(nocca,nvirb,nvira).copy()
    r2bab = r2bab.reshape(noccb,nvira,nvirb).copy()
    r2bbb = r2bab.reshape(noccb,nvirb,nvirb).copy()

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2aba, r2bab, r2bbb)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    return np.hstack((r1a, r1b,
                      r2aaa.ravel(),
                      r2aba.ravel(), r2bab.ravel(),
                      r2bbb.ravel()))

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ j}^{ab}, i.e. 'jb' indices are coupled'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    nocca, noccb = nocc
    nvira, nvirb = nvir
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = eom.vector_to_amplitudes(vector, nkpts, nmo[0]+nmo[1], nocc[0]+nocc[1])  # spin
    orbspin = imds.eris.orbspin
    spatial_r1, spatial_r2 = eom_kgccsd.spin2spatial_ea_doublet(r1, r2, kconserv, kshift, orbspin)

    uccsd_imds = imds._uccsd_imds
    t2aa, t2ab, t2bb = t2

    # k-point spin orbital version of eaccsd

    Hr1 = 0.0*np.einsum('ac,c->a', imds.Fvv[kshift], r1)

    Hr2 = np.zeros_like(r2)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]

        for kd in range(nkpts):
            kl = kconserv[kj, kb, kd]
            Hr2[kj, ka] += lib.einsum('lbdj,lad->jab', imds.Wovvo[kl, kb, kd], r2[kl, ka])

            # P(ab)
            kl = kconserv[kj, ka, kd]
            Hr2[kj, ka] -= lib.einsum('ladj,lbd->jab', imds.Wovvo[kl, ka, kd], r2[kl, kb])

    r1a, r1b = spatial_r1
    r2aaa, r2aba, r2bab, r2bbb = spatial_r2
    Hr1a = np.zeros((nvira), dtype=r1.dtype)
    Hr1b = np.zeros((nvirb), dtype=r1.dtype)

    # Fov terms
    for kl in range(nkpts):
        Hr1a += np.einsum('ld,lad->a', uccsd_imds.Fov[kl], r2aaa[kl,kshift])
        Hr1a += np.einsum('LD,LaD->a', uccsd_imds.FOV[kl], r2bab[kl,kshift])
        Hr1b += np.einsum('ld,lAd->A', uccsd_imds.Fov[kl], r2aba[kl,kshift])
        Hr1b += np.einsum('LD,LAD->A', uccsd_imds.FOV[kl], r2bbb[kl,kshift])

    # Fvv terms
    Hr1a += np.einsum('ac,c->a', uccsd_imds.Fvv[kshift], r1a)
    Hr1b += np.einsum('AC,C->A', uccsd_imds.FVV[kshift], r1b)

    # Wvovv
    for kc, kl in itertools.product(range(nkpts), repeat=2):
        Hr1a += 0.5*lib.einsum('acld,lcd->a', uccsd_imds.Wvvov[kshift,kc,kl], r2aaa[kl,kc])
        Hr1a +=     lib.einsum('acLD,LcD->a', uccsd_imds.WvvOV[kshift,kc,kl], r2bab[kl,kc])
        Hr1b += 0.5*lib.einsum('ACLD,LCD->A', uccsd_imds.WVVOV[kshift,kc,kl], r2bbb[kl,kc])
        Hr1b +=     lib.einsum('ACld,lCd->A', uccsd_imds.WVVov[kshift,kc,kl], r2aba[kl,kc])

    Hr2aaa = np.zeros((nkpts, nkpts, nocca, nvira, nvira), dtype=r2.dtype)
    Hr2aba = np.zeros((nkpts, nkpts, nocca, nvirb, nvira), dtype=r2.dtype)
    Hr2bab = np.zeros((nkpts, nkpts, noccb, nvira, nvirb), dtype=r2.dtype)
    Hr2bbb = np.zeros((nkpts, nkpts, noccb, nvirb, nvirb), dtype=r2.dtype)

    uimds = imds._uccsd_imds
    #** Wvvvv term
    if uimds.Wvvvv is not None:
        for kj, ka in itertools.product(range(nkpts), repeat=2):
            kb = kconserv[kshift,ka,kj]
            for kd in range(nkpts):
                kc = kconserv[ka, kd, kb]
                Hr2aaa[kj,ka] += .5 * lib.einsum('acbd,jcd->jab', uimds.Wvvvv[ka,kc,kb], r2aaa[kj,kc])
                Hr2aba[kj,ka] +=      lib.einsum('bdac,jcd->jab', uimds.WvvVV[kb,kd,ka], r2aba[kj,kc])
                Hr2bab[kj,ka] +=      lib.einsum('acbd,jcd->jab', uimds.WvvVV[ka,kc,kb], r2bab[kj,kc])
                Hr2bbb[kj,ka] += .5 * lib.einsum('acbd,jcd->jab', uimds.WVVVV[ka,kc,kb], r2bbb[kj,kc])

    #Wvvvo term
    for ka, kj, in itertools.product(range(nkpts),repeat=2):
        kb = kconserv[kshift,ka,kj]
        kc = kshift
        Hr2aaa[kj,ka] += np.einsum('acbj,c->jab', uimds.Wvvvo[ka,kc,kb], r1a)
        Hr2bbb[kj,ka] += np.einsum('ACBJ,C->JAB', uimds.WVVVO[ka,kc,kb], r1b)

        Hr2bab[kj,ka] += np.einsum('acBJ,c->JaB', uimds.WvvVO[ka,kc,kb], r1a)
        Hr2aba[kj,ka] += np.einsum('ACbj,C->jAb', uimds.WVVvo[ka,kc,kb], r1b)

    #Fvv Term
    for ka, kj in itertools.product(range(nkpts), repeat=2):
        # kb = kshift - ka + kj
        kb = kconserv[kshift, ka, kj]
        tmpa = lib.einsum('ac,jcb->jab', uccsd_imds.Fvv[ka], r2aaa[kj,ka])
        tmpb = lib.einsum('bc,jca->jab', uccsd_imds.Fvv[kb], r2aaa[kj,kb])
        Hr2aaa[kj,ka] += tmpa - tmpb
        Hr2aba[kj,ka] += lib.einsum('AC,jCb->jAb', uccsd_imds.FVV[ka], r2aba[kj,ka])
        Hr2bab[kj,ka] += lib.einsum('ac,JcB->JaB', uccsd_imds.Fvv[ka], r2bab[kj,ka])
        Hr2aba[kj,ka] += lib.einsum('bc, jAc -> jAb', uccsd_imds.Fvv[kb], r2aba[kj,ka])
        Hr2bab[kj,ka] += lib.einsum('BC, JaC -> JaB', uccsd_imds.FVV[kb], r2bab[kj,ka])
        tmpb = lib.einsum('AC,JCB->JAB', uccsd_imds.FVV[ka], r2bbb[kj,ka])
        tmpa = lib.einsum('BC,JCA->JAB', uccsd_imds.FVV[kb], r2bbb[kj,kb])
        Hr2bbb[kj,ka] += tmpb - tmpa

    #Foo Term
    for kl, ka in itertools.product(range(nkpts), repeat=2):
        Hr2aaa[kl,ka] -= lib.einsum('lj,lab->jab', uccsd_imds.Foo[kl], r2aaa[kl,ka])
        Hr2bbb[kl,ka] -= lib.einsum('LJ,LAB->JAB', uccsd_imds.FOO[kl], r2bbb[kl,ka])
        Hr2bab[kl,ka] -= lib.einsum('LJ,LaB->JaB', uccsd_imds.FOO[kl], r2bab[kl,ka])
        Hr2aba[kl,ka] -= lib.einsum('lj,lAb->jAb', uccsd_imds.Foo[kl], r2aba[kl,ka])

    ## Woovv term
    tmp_aaa = lib.einsum('xykcld, yxlcd->k', uccsd_imds.Wovov[kshift,:,:], r2aaa)
    tmp_bbb = lib.einsum('xyKCLD, yxLCD->K', uccsd_imds.WOVOV[kshift,:,:], r2bbb)
    tmp_bab = lib.einsum('xykcLD, yxLcD->k', uccsd_imds.WovOV[kshift], r2bab)
    tmp_aba = np.zeros(tmp_bbb.shape, dtype = tmp_bbb.dtype)

    for kl, kc in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kl,kc,kshift]
        tmp_aba += lib.einsum('ldKC, lCd->K', uccsd_imds.WovOV[kl,kd,kshift], r2aba[kl,kc])

    Hr2aaa -= 0.5 * lib.einsum('k, xykjab->xyjab', tmp_aaa, t2aa[kshift])
    Hr2bab -= 0.5 * lib.einsum('k, xykJaB->xyJaB', tmp_aaa, t2ab[kshift])

    Hr2aaa -= lib.einsum('k, xykjab->xyjab', tmp_bab, t2aa[kshift])
    Hr2bbb -= 0.5 * lib.einsum('K, xyKJAB->xyJAB', tmp_bbb, t2bb[kshift])

    Hr2bbb -= lib.einsum('K, xyKJAB->xyJAB', tmp_aba, t2bb[kshift])
    Hr2bab -= lib.einsum('k, xykJaB->xyJaB', tmp_bab, t2ab[kshift])


    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift, ka, kj]
        Hr2aba[kj, ka] -= lib.einsum('K, jKbA->jAb', tmp_aba, t2ab[kj, kshift, kb])
        Hr2aba[kj, ka] -= 0.5 * einsum('K, jKbA->jAb', tmp_bbb, t2ab[kj, kshift, kb])

    spatial_Hr1 = [Hr1a, Hr1b]
    spatial_Hr2 = [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb]
    spin_Hr1, spin_Hr2 = eom_kgccsd.spatial2spin_ea_doublet(spatial_Hr1, spatial_Hr2,
                                                            kconserv, kshift, orbspin)
    Hr1 += spin_Hr1
    Hr2 += spin_Hr2
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMEA(eom_kgccsd.EOMEA):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMEA.__init__(self, cc)

    #get_diag = eaccsd_diag
    matvec = eaccsd_matvec

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            nocca, noccb = self.nocc
            nmoa, nmob = self.nmo
            nvira, nvirb = nmoa - nocca, nmob - noccb
            idx = diag[:nvira+nvirb].argsort()
        else:
            idx = diag.argsort()
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    def _vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc is self.nocc
        return vector_to_amplitudes_ea(vector, nmo, nocc)

    def _amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        nkpts = self.nkpts
        return nvira + nvirb + nkpts**2 * (nocca*nvira*(nvira-1)//2+nocca*nvirb*nvira+noccb*nvira*nvirb+noccb*nvirb*(nvirb-1)//2)

    def _make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ea()
        return imds

class _IMDS:
    def __init__(self, cc, eris=None, t1=None, t2=None):
        self._cc = cc
        self.verbose = cc.verbose
        self.kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        self.stdout = cc.stdout
        if t1 is None:
            t1 = cc.t1
        self.t1 = t1
        if t2 is None:
            t2 = cc.t2
        self.t2 = t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo, self.FOO = kintermediates_uhf.Foo(self._cc, t1, t2, eris)
        self.Fvv, self.FVV = kintermediates_uhf.Fvv(self._cc, t1, t2, eris)
        self.Fov, self.FOV = kintermediates_uhf.Fov(self._cc, t1, t2, eris)

        # 2 virtuals
        self.Wovvo, self.WovVO, self.WOVvo, self.WOVVO = kintermediates_uhf.Wovvo(self._cc, t1, t2, eris)
        self.Wovov = eris.ovov - eris.ovov.transpose(2,1,0,5,4,3,6)
        self.WOVOV = eris.OVOV - eris.OVOV.transpose(2,1,0,5,4,3,6)
        self.WovOV = eris.ovOV
        self.WOVov = None

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-KCCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        kconserv = self.kconserv
        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo, self.WooOO, _         , self.WOOOO = kintermediates_uhf.Woooo(self._cc, t1, t2, eris)
        self.Wooov, self.WooOV, self.WOOov, self.WOOOV = kintermediates_uhf.Wooov(self._cc, t1, t2, eris, kconserv)  # TODO
        self.Woovo, self.WooVO, self.WOOvo, self.WOOVO = kintermediates_uhf.Woovo(self._cc, t1, t2, eris)  # TODO

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-KUCCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        #self.Wvovv, self.WvoVV, self.WVOvv, self.WVOVV = kintermediates_uhf.Wvovv(self._cc, t1, t2, eris)
        self.Wvvov, self.WvvOV, self.WVVov, self.WVVOV = kintermediates_uhf.Wvvov(self._cc, t1, t2, eris)
        self.Wvvvv, self.WvvVV, self.WVVVV = Wvvvv = kintermediates_uhf.Wvvvv(self._cc, t1, t2, eris)
        self.Wvvvo, self.WvvVO, self.WVVvo, self.WVVVO = kintermediates_uhf.Wvvvo(self._cc, t1, t2, eris)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-KUCCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from pyscf import lo

    cell = gto.Cell()
    cell.atom='''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    #cell.basis = [[0, (1., 1.)], [1, (.5, 1.)]]
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [5, 5, 5]
    cell.build()

    np.random.seed(1)
    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((2,3,nmo))
    kmf.mo_occ[0,:,:3] = 1
    kmf.mo_occ[1,:,:1] = 1
    kmf.mo_energy = np.arange(nmo) + np.random.random((2,3,nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2

    mo = (np.random.random((2,3,nmo,nmo)) +
          np.random.random((2,3,nmo,nmo))*1j - .5-.5j)
    s = kmf.get_ovlp()
    kmf.mo_coeff = np.empty_like(mo)
    nkpts = len(kmf.kpts)
    for k in range(nkpts):
        kmf.mo_coeff[0,k] = lo.orth.vec_lowdin(mo[0,k], s[k])
        kmf.mo_coeff[1,k] = lo.orth.vec_lowdin(mo[1,k], s[k])

    def rand_t1_t2(mycc):
        nkpts = mycc.nkpts
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        np.random.seed(1)
        t1a = (np.random.random((nkpts,nocca,nvira)) +
               np.random.random((nkpts,nocca,nvira))*1j - .5-.5j)
        t1b = (np.random.random((nkpts,noccb,nvirb)) +
               np.random.random((nkpts,noccb,nvirb))*1j - .5-.5j)
        t2aa = (np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira))*1j - .5-.5j)
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        t2aa = t2aa - t2aa.transpose(1,0,2,4,3,5,6)
        tmp = t2aa.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
        t2ab = (np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
        t2bb = (np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
        t2bb = t2bb - t2bb.transpose(1,0,2,4,3,5,6)
        tmp = t2bb.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2bb[ki,kj,kk] = t2bb[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)

        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        return t1, t2

    from pyscf.pbc.cc import kccsd_uhf
    mycc = kccsd_uhf.KUCCSD(kmf)
    eris = mycc.ao2mo()

    t1, t2 = rand_t1_t2(mycc)
    mycc.t1 = t1
    mycc.t2 = t2
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)

    import kccsd
    kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(kmf))
    kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
    spin_t1 = kccsd.spatial2spin(t1, kccsd_eris.orbspin, kconserv)
    spin_t2 = kccsd.spatial2spin(t2, kccsd_eris.orbspin, kconserv)
    orbspin = kccsd_eris.orbspin

    nkpts = mycc.nkpts
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    kshift = 0  # excitation out of 0th k-point
    nmo = nmoa + nmob
    nocc = nocca + noccb
    nvir = nmo - nocc

    np.random.seed(0)
    # IP version
    myeom = EOMIP(mycc)
    imds = myeom.make_imds(eris=kccsd_eris, t1=spin_t1, t2=spin_t2)
    imds._uccsd_eris = eris
    imds._uccsd_imds = _IMDS(mycc, eris=eris)
    imds._uccsd_imds.make_ip()

    spin_r1_ip = (np.random.rand(nocc)*1j +
                  np.random.rand(nocc) - 0.5 - 0.5*1j)
    spin_r2_ip = (np.random.rand(nkpts**2 * nocc**2 * nvir) +
                  np.random.rand(nkpts**2 * nocc**2 * nvir)*1j - 0.5 - 0.5*1j)
    spin_r2_ip = spin_r2_ip.reshape(nkpts, nkpts, nocc, nocc, nvir)
    spin_r2_ip = eom_kgccsd.enforce_2p_spin_ip_doublet(spin_r2_ip, kconserv, kshift, orbspin)
    vector = eom_kgccsd.amplitudes_to_vector_ip(spin_r1_ip, spin_r2_ip)

    kshift = 0
    vector = myeom.matvec(vector, kshift=kshift, imds=imds)
    Hr1, Hr2 = myeom.vector_to_amplitudes(vector, nkpts, nmo, nocc)
    [Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb] = \
            eom_kgccsd.spin2spatial_ip_doublet(Hr1, Hr2, kconserv, kshift, orbspin)

    print('ip Hr1a',   abs(lib.finger(Hr1a)   - (-0.34462696543560045-1.6104596956729178j)))
    print('ip Hr1b',   abs(lib.finger(Hr1b)   - (-0.055793611517250929+0.22169994342782473j)))
    print('ip Hr2aaa', abs(lib.finger(Hr2aaa) - (0.692705827672665420-1.958639508839846943j)))
    print('ip Hr2baa', abs(lib.finger(Hr2baa) - (2.892194153603884654+2.039530776282815872j)))
    print('ip Hr2abb', abs(lib.finger(Hr2abb) - (1.618257685489421727-5.489218743953674817j)))
    print('ip Hr2bbb', abs(lib.finger(Hr2bbb) - (0.479835513829048044+0.108406393138471210j)))

    # EA version
    myeom = EOMEA(mycc)
    imds = myeom.make_imds(eris=kccsd_eris, t1=spin_t1, t2=spin_t2)
    imds._uccsd_eris = eris
    imds._uccsd_imds = _IMDS(mycc, eris=eris)
    imds._uccsd_imds.make_ea()

    spin_r1_ea = (np.random.rand(nvir)*1j +
                  np.random.rand(nvir) - 0.5 - 0.5*1j)
    spin_r2_ea = (np.random.rand(nkpts**2 * nocc * nvir**2) +
                  np.random.rand(nkpts**2 * nocc * nvir**2)*1j - 0.5 - 0.5*1j)
    spin_r2_ea = spin_r2_ea.reshape(nkpts, nkpts, nocc, nvir, nvir)
    spin_r2_ea = eom_kgccsd.enforce_2p_spin_ea_doublet(spin_r2_ea, kconserv, kshift, orbspin)
    vector = eom_kgccsd.amplitudes_to_vector_ea(spin_r1_ea, spin_r2_ea)

    kshift = 0
    vector = myeom.matvec(vector, kshift=kshift, imds=imds)
    Hr1, Hr2 = myeom.vector_to_amplitudes(vector, nkpts, nmo, nocc)
    [Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb] = \
        eom_kgccsd.spin2spatial_ea_doublet(Hr1, Hr2, kconserv, kshift, orbspin)

    print('ea Hr1a',  abs(lib.finger(Hr1a)   - (-0.081373075311041126-0.51422895644026023j)))
    print('ea Hr1b',  abs(lib.finger(Hr1b)   - (-0.39518588661294807-1.3063424820239824j))  )
    print('ea Hr2aaa',abs(lib.finger(Hr2aaa) - (-2.6502079691200251-0.61302655915003545j))  )
    print('ea Hr2aba',abs(lib.finger(Hr2aba) - (5.5723208649566036-5.4202659143496286j))    )
    print('ea Hr2bab',abs(lib.finger(Hr2bab) - (-1.2822293707887937+0.3026476580141586j))   )
    print('ea Hr2bbb',abs(lib.finger(Hr2bbb) - (-4.0202809577487253-0.46985725132191702j))  )
