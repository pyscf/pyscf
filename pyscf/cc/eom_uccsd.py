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
#         James D. McClain
#         Jason Yu
#         Shining Sun
#         Mario Motta
#         Chong Sun
#


import numpy as np

from pyscf import lib
from pyscf.lib import logger, module_method
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import uccsd
from pyscf.cc import addons
from pyscf.cc import eom_rccsd
from pyscf.cc import uintermediates

########################################
# EOM-IP-CCSD
########################################

def vector_to_amplitudes_ip(vector, nmo, nocc):
    '''For spin orbitals'''
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nocca, noccb, nocca*(nocca-1)//2*nvira, noccb*nocca*nvira,
             nocca*noccb*nvirb, noccb*(noccb-1)//2*nvirb)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2a, r2baa, r2abb, r2b = np.split(vector, sections)
    r2a = r2a.reshape(nocca*(nocca-1)//2,nvira)
    r2b = r2b.reshape(noccb*(noccb-1)//2,nvirb)
    r2baa = r2baa.reshape(noccb,nocca,nvira).copy()
    r2abb = r2abb.reshape(nocca,noccb,nvirb).copy()

    idxa = np.tril_indices(nocca, -1)
    idxb = np.tril_indices(noccb, -1)
    r2aaa = np.zeros((nocca,nocca,nvira), vector.dtype)
    r2bbb = np.zeros((noccb,noccb,nvirb), vector.dtype)
    r2aaa[idxa[0],idxa[1]] = r2a
    r2aaa[idxa[1],idxa[0]] =-r2a
    r2bbb[idxb[0],idxb[1]] = r2b
    r2bbb[idxb[1],idxb[0]] =-r2b

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2baa, r2abb, r2bbb)
    return r1, r2

def amplitudes_to_vector_ip(r1, r2):
    '''For spin orbitals'''
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    nocca, noccb, nvirb = r2abb.shape
    idxa = np.tril_indices(nocca, -1)
    idxb = np.tril_indices(noccb, -1)
    return np.hstack((r1a, r1b,
                      r2aaa[idxa].ravel(), r2baa.ravel(),
                      r2abb.ravel(), r2bbb[idxb].ravel()))

def spatial2spin_ip(rx, orbspin=None):
    '''Convert EOMIP spatial-orbital R1/R2 to spin-orbital R1/R2'''
    if len(rx) == 2:  # r1
        r1a, r1b = rx
        nocc_a = r1a.size
        nocc_b = r1b.size
        nocc = nocc_a + nocc_b

        r1 = np.zeros(nocc, dtype=r1a.dtype)
        if orbspin is None:
            assert nocc_a == nocc_b
            r1[0::2] = r1a
            r1[1::2] = r1b
        else:
            r1[orbspin[:nocc] == 0] = r1a
            r1[orbspin[:nocc] == 1] = r1b
        return r1

    r2aaa, r2baa, r2abb, r2bbb = rx
    nocc_a, nvir_a = r2aaa.shape[1:]
    nocc_b, nvir_b = r2bbb.shape[1:]

    if orbspin is None:
        assert nocc_a == nocc_b
        orbspin = np.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]

    r2 = np.zeros((nocc**2, nvir), dtype=r2aaa.dtype)
    idxoaa = idxoa[:,None] * nocc + idxoa
    idxoab = idxoa[:,None] * nocc + idxob
    idxoba = idxob[:,None] * nocc + idxoa
    idxobb = idxob[:,None] * nocc + idxob
    # idxvaa = idxva[:,None] * nvir + idxva
    # idxvab = idxva[:,None] * nvir + idxvb
    # idxvba = idxvb[:,None] * nvir + idxva
    # idxvbb = idxvb[:,None] * nvir + idxvb
    r2aaa = r2aaa.reshape(nocc_a*nocc_a, nvir_a)
    r2baa = r2baa.reshape(nocc_b*nocc_a, nvir_a)
    r2abb = r2abb.reshape(nocc_a*nocc_b, nvir_b)
    r2bbb = r2bbb.reshape(nocc_b*nocc_b, nvir_b)
    lib.takebak_2d(r2, r2aaa, idxoaa.ravel(), idxva.ravel())
    lib.takebak_2d(r2, r2baa, idxoba.ravel(), idxva.ravel())
    lib.takebak_2d(r2, r2abb, idxoab.ravel(), idxvb.ravel())
    lib.takebak_2d(r2, r2bbb, idxobb.ravel(), idxvb.ravel())
    r2aba = -r2baa
    r2bab = -r2abb
    lib.takebak_2d(r2, r2aba, idxoab.T.ravel(), idxva.ravel())
    lib.takebak_2d(r2, r2bab, idxoba.T.ravel(), idxvb.ravel())
    return r2.reshape(nocc, nocc, nvir)

def spin2spatial_ip(rx, orbspin):
    '''Convert EOMIP spin-orbital R1/R2 to spatial-orbital R1/R2'''
    if rx.ndim == 1:
        nocc = rx.size
        r1a = rx[orbspin[:nocc] == 0]
        r1b = rx[orbspin[:nocc] == 1]
        return r1a, r1b

    nocc, nvir = rx.shape[1:]

    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]
    nocc_a = len(idxoa)
    nocc_b = len(idxob)
    nvir_a = len(idxva)
    nvir_b = len(idxvb)

    idxoaa = idxoa[:,None] * nocc + idxoa
    idxoab = idxoa[:,None] * nocc + idxob
    idxoba = idxob[:,None] * nocc + idxoa
    idxobb = idxob[:,None] * nocc + idxob
    #idxvaa = idxva[:,None] * nvir + idxva
    #idxvab = idxva[:,None] * nvir + idxvb
    #idxvba = idxvb[:,None] * nvir + idxva
    #idxvbb = idxvb[:,None] * nvir + idxvb

    r2 = rx.reshape(nocc**2, nvir)
    r2aaa = lib.take_2d(r2, idxoaa.ravel(), idxva.ravel())
    r2baa = lib.take_2d(r2, idxoba.ravel(), idxva.ravel())
    r2abb = lib.take_2d(r2, idxoab.ravel(), idxvb.ravel())
    r2bbb = lib.take_2d(r2, idxobb.ravel(), idxvb.ravel())

    r2aaa = r2aaa.reshape(nocc_a, nocc_a, nvir_a)
    r2baa = r2baa.reshape(nocc_b, nocc_a, nvir_a)
    r2abb = r2abb.reshape(nocc_a, nocc_b, nvir_b)
    r2bbb = r2bbb.reshape(nocc_b, nocc_b, nvir_b)
    return r2aaa, r2baa, r2abb, r2bbb

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    '''For spin orbitals
    R2 operators of the form s_{ij}^{ b}, i.e. indices jb are coupled.'''
    # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca+nvira, noccb+nvirb
    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa,nmob), (nocca,noccb))
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2

    #Foo, Fov, and Wooov
    Hr1a  = np.einsum('me,mie->i', imds.Fov, r2aaa)
    Hr1a -= np.einsum('ME,iME->i', imds.FOV, r2abb)
    Hr1b  = np.einsum('ME,MIE->I', imds.FOV, r2bbb)
    Hr1b -= np.einsum('me,Ime->I', imds.Fov, r2baa)

    Hr1a += -np.einsum('mi,m->i', imds.Foo, r1a)
    Hr1b += -np.einsum('MI,M->I', imds.FOO, r1b)

    Hr1a += -0.5*np.einsum('nime,mne->i', imds.Wooov, r2aaa)
    Hr1b +=      np.einsum('NIme,Nme->I', imds.WOOov, r2baa)
    Hr1b += -0.5*np.einsum('NIME,MNE->I', imds.WOOOV, r2bbb)
    Hr1a +=      np.einsum('niME,nME->i', imds.WooOV, r2abb)

    # Fvv term
    Hr2aaa = lib.einsum('be,ije->ijb', imds.Fvv, r2aaa)
    Hr2abb = lib.einsum('BE,iJE->iJB', imds.FVV, r2abb)
    Hr2bbb = lib.einsum('BE,IJE->IJB', imds.FVV, r2bbb)
    Hr2baa = lib.einsum('be,Ije->Ijb', imds.Fvv, r2baa)

    # Foo term
    tmpa = lib.einsum('mi,mjb->ijb', imds.Foo, r2aaa)
    Hr2aaa -= tmpa - tmpa.transpose((1,0,2))
    Hr2abb -= lib.einsum('mi,mJB->iJB', imds.Foo, r2abb)
    Hr2abb -= lib.einsum('MJ,iMB->iJB', imds.FOO, r2abb)
    Hr2baa -= lib.einsum('MI,Mjb->Ijb', imds.FOO, r2baa)
    Hr2baa -= lib.einsum('mj,Imb->Ijb', imds.Foo, r2baa)
    tmpb = lib.einsum('MI,MJB->IJB', imds.FOO, r2bbb)
    Hr2bbb -= tmpb - tmpb.transpose((1,0,2))

    # Wovoo term
    Hr2aaa -= np.einsum('mjbi,m->ijb', imds.Woovo, r1a)
    Hr2abb += np.einsum('miBJ,m->iJB', imds.WooVO, r1a)
    Hr2baa += np.einsum('MIbj,M->Ijb', imds.WOOvo, r1b)
    Hr2bbb -= np.einsum('MJBI,M->IJB', imds.WOOVO, r1b)

    # Woooo term
    Hr2aaa += .5 * lib.einsum('minj,mnb->ijb', imds.Woooo, r2aaa)
    Hr2abb +=      lib.einsum('miNJ,mNB->iJB', imds.WooOO, r2abb)
    Hr2bbb += .5 * lib.einsum('MINJ,MNB->IJB', imds.WOOOO, r2bbb)
    Hr2baa +=      lib.einsum('njMI,Mnb->Ijb', imds.WooOO, r2baa)

    # Wovvo terms
    tmp = lib.einsum('mebj,ime->ijb', imds.Wovvo, r2aaa)
    tmp += lib.einsum('MEbj,iME->ijb', imds.WOVvo, r2abb)
    Hr2aaa += tmp - tmp.transpose(1, 0, 2)

    WooVV = -imds.WoVVo.transpose(0,3,2,1)
    WOOvv = -imds.WOvvO.transpose(0,3,2,1)
    Hr2abb += lib.einsum('MEBJ,iME->iJB', imds.WOVVO, r2abb)
    Hr2abb += lib.einsum('meBJ,ime->iJB', imds.WovVO, r2aaa)
    Hr2abb += -lib.einsum('miBE,mJE->iJB', WooVV, r2abb)

    Hr2baa += lib.einsum('meaj,Ime->Ija', imds.Wovvo, r2baa)
    Hr2baa += lib.einsum('MEaj,IME->Ija', imds.WOVvo, r2bbb)
    Hr2baa += -lib.einsum('MIab,Mjb->Ija', WOOvv, r2baa)

    tmp = lib.einsum('MEBJ,IME->IJB', imds.WOVVO, r2bbb)
    tmp += lib.einsum('meBJ,Ime->IJB', imds.WovVO, r2baa)
    Hr2bbb += tmp - tmp.transpose(1, 0, 2)

    # T2 term
    Hr2aaa -= 0.5 * lib.einsum('menf,mnf,jibe->ijb', imds.Wovov, r2aaa, t2aa)
    Hr2aaa -= lib.einsum('meNF,mNF,jibe->ijb', imds.WovOV, r2abb, t2aa)

    Hr2abb -= 0.5 * lib.einsum('menf,mnf,iJeB->iJB', imds.Wovov, r2aaa, t2ab)
    Hr2abb -= lib.einsum('meNF,mNF,iJeB->iJB', imds.WovOV, r2abb, t2ab)

    Hr2baa -= 0.5 * lib.einsum('MENF,MNF,jIbE->Ijb', imds.WOVOV, r2bbb, t2ab)
    Hr2baa -= lib.einsum('nfME,Mnf,jIbE->Ijb', imds.WovOV, r2baa, t2ab)

    Hr2bbb -= 0.5 * lib.einsum('MENF,MNF,JIBE->IJB', imds.WOVOV, r2bbb, t2bb)
    Hr2bbb -= lib.einsum('nfME,Mnf,JIBE->IJB', imds.WovOV, r2baa, t2bb)

    vector = amplitudes_to_vector_ip([Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb])
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape

    Hr1a = -np.diag(imds.Foo)
    Hr1b = -np.diag(imds.FOO)

    Fvv_diag = np.diag(imds.Fvv)
    Foo_diag = np.diag(imds.Foo)
    FOO_diag = np.diag(imds.FOO)
    FVV_diag = np.diag(imds.FVV)

    Woooo_slice = np.einsum('iijj->ij',imds.Woooo)
    Wovvo_slice = np.einsum('iaai->ia',imds.Wovvo)
    WooOO_slice = np.einsum('jjii->ij',imds.WooOO)
    WOvvO_slice = np.einsum('iaai->ia',imds.WOvvO)
    WooOO_slice_T = np.einsum('iijj->ij',imds.WooOO)
    WoVVo_slice = np.einsum('iaai->ia',imds.WoVVo)
    WOVVO_slice = np.einsum('jaaj->ja',imds.WOVVO)
    WOOOO_slice = np.einsum('iijj->ij',imds.WOOOO)

    Wovov_t2_dot = np.einsum('jaib,jiab->ija',imds.Wovov,t2aa)
    WovOV_t2_dot = np.einsum('ibja,ijba->ija',imds.WovOV,t2ab)
    WovOV_t2_dot_T = np.einsum('jaib,jiab->ija',imds.WovOV,t2ab)
    WOVOV_t2_dot = np.einsum('jaib,jiab->ija',imds.WOVOV,t2bb)

    Hr2aaa = Fvv_diag[None,None,:] - Foo_diag[:,None,None] - Foo_diag[None,:,None] \
             + Woooo_slice[:,:,None] + Wovvo_slice[:,None,:] + Wovvo_slice[None,:,:] \
             - Wovov_t2_dot

    Hr2baa = Fvv_diag[None,None,:] - FOO_diag[:,None,None] - Foo_diag[None,:,None] \
             + WooOO_slice[:,:,None] + WOvvO_slice[:,None,:] + Wovvo_slice[None,:,:] \
             - WovOV_t2_dot_T

    Hr2abb = FVV_diag[None,None,:] - Foo_diag[:,None,None] - FOO_diag[None,:,None] \
             + WooOO_slice_T[:,:,None] + WoVVo_slice[:,None,:] + WOVVO_slice[None,:,:] \
             - WovOV_t2_dot

    Hr2bbb = FVV_diag[None,None,:] - FOO_diag[:,None,None] - FOO_diag[None,:,None] \
             + WOOOO_slice[:,:,None] + WOVVO_slice[:,None,:] + WOVVO_slice[None,:,:] \
             - WOVOV_t2_dot

    vector = amplitudes_to_vector_ip([Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb])
    return vector


class EOMIP(eom_rccsd.EOMIP):
    matvec = ipccsd_matvec
    l_matvec = None
    get_diag = ipccsd_diag
    ipccsd_star = None
    ccsd_star_contract = None

    def __init__(self, cc):
        eom_rccsd.EOMIP.__init__(self, cc)
        self.nocc = cc.get_nocc()
        self.nmo = cc.get_nmo()

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if diag is None:
            diag = self.get_diag()
        if koopmans:
            nocca, noccb = self.nocc
            idx = diag[:nocca+noccb].argsort()
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

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ip)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ip,
                                         absences=['nmo', 'nocc'])
    spatial2spin = staticmethod(spatial2spin_ip)
    spin2spatial = staticmethod(spin2spatial_ip)

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        return (nocca + noccb
                + nocca*(nocca-1)//2*nvira + noccb*nocca*nvira
                + nocca*noccb*nvirb + noccb*(noccb-1)//2*nvirb)

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds

########################################
# EOM-EA-CCSD
########################################

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nvira, nvirb, nocca*nvira*(nvira-1)//2, nocca*nvirb*nvira,
             noccb*nvira*nvirb, noccb*nvirb*(nvirb-1)//2)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2a, r2aba, r2bab, r2b = np.split(vector, sections)
    r2a = r2a.reshape(nocca,nvira*(nvira-1)//2)
    r2b = r2b.reshape(noccb,nvirb*(nvirb-1)//2)
    r2aba = r2aba.reshape(nocca,nvirb,nvira).copy()
    r2bab = r2bab.reshape(noccb,nvira,nvirb).copy()

    idxa = np.tril_indices(nvira, -1)
    idxb = np.tril_indices(nvirb, -1)
    r2aaa = np.zeros((nocca,nvira,nvira), vector.dtype)
    r2bbb = np.zeros((noccb,nvirb,nvirb), vector.dtype)
    r2aaa[:,idxa[0],idxa[1]] = r2a
    r2aaa[:,idxa[1],idxa[0]] =-r2a
    r2bbb[:,idxb[0],idxb[1]] = r2b
    r2bbb[:,idxb[1],idxb[0]] =-r2b

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2aba, r2bab, r2bbb)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    nocca, nvirb, nvira = r2aba.shape
    idxa = np.tril_indices(nvira, -1)
    idxb = np.tril_indices(nvirb, -1)
    return np.hstack((r1a, r1b,
                      r2aaa[:,idxa[0],idxa[1]].ravel(),
                      r2aba.ravel(), r2bab.ravel(),
                      r2bbb[:,idxb[0],idxb[1]].ravel()))

def spatial2spin_ea(rx, orbspin=None):
    '''Convert EOMEA spatial-orbital R1/R2 to spin-orbital R1/R2'''
    if len(rx) == 2:  # r1
        r1a, r1b = rx
        nvir_a = r1a.size
        nvir_b = r1b.size
        nvir = nvir_a + nvir_b

        r1 = np.zeros(nvir, dtype=r1a.dtype)
        if orbspin is None:
            assert nvir_a == nvir_b
            r1[0::2] = r1a
            r1[1::2] = r1b
        else:
            r1[orbspin[-nvir:] == 0] = r1a
            r1[orbspin[-nvir:] == 1] = r1b
        return r1

    r2aaa, r2aba, r2bab, r2bbb = rx
    nocc_a, nvir_a = r2aaa.shape[:2]
    nocc_b, nvir_b = r2bbb.shape[:2]

    if orbspin is None:
        assert nvir_a == nvir_b
        orbspin = np.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]

    r2 = np.zeros((nocc, nvir**2), dtype=r2aaa.dtype)
    #idxoaa = idxoa[:,None] * nocc + idxoa
    #idxoab = idxoa[:,None] * nocc + idxob
    #idxoba = idxob[:,None] * nocc + idxoa
    #idxobb = idxob[:,None] * nocc + idxob
    idxvaa = idxva[:,None] * nvir + idxva
    idxvab = idxva[:,None] * nvir + idxvb
    idxvba = idxvb[:,None] * nvir + idxva
    idxvbb = idxvb[:,None] * nvir + idxvb

    r2aaa = r2aaa.reshape(nocc_a, nvir_a*nvir_a)
    r2aba = r2aba.reshape(nocc_a, nvir_b*nvir_a)
    r2bab = r2bab.reshape(nocc_b, nvir_a*nvir_b)
    r2bbb = r2bbb.reshape(nocc_b, nvir_b*nvir_b)

    lib.takebak_2d(r2, r2aaa, idxoa.ravel(), idxvaa.ravel())
    lib.takebak_2d(r2, r2aba, idxoa.ravel(), idxvba.ravel())
    lib.takebak_2d(r2, r2bab, idxob.ravel(), idxvab.ravel())
    lib.takebak_2d(r2, r2bbb, idxob.ravel(), idxvbb.ravel())
    r2aab = -r2aba
    r2bba = -r2bab
    lib.takebak_2d(r2, r2bba, idxob.ravel(), idxvba.T.ravel())
    lib.takebak_2d(r2, r2aab, idxoa.ravel(), idxvab.T.ravel())
    r2 = r2.reshape(nocc, nvir, nvir)
    return r2

def spin2spatial_ea(rx, orbspin):
    '''Convert EOMEA spin-orbital R1/R2 to spatial-orbital R1/R2'''
    if rx.ndim == 1:
        nvir = rx.size
        r1a = rx[orbspin[-nvir:] == 0]
        r1b = rx[orbspin[-nvir:] == 1]
        return r1a, r1b

    nocc, nvir = rx.shape[:2]

    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]
    nocc_a = len(idxoa)
    nocc_b = len(idxob)
    nvir_a = len(idxva)
    nvir_b = len(idxvb)

    idxvaa = idxva[:,None] * nvir + idxva
    idxvab = idxva[:,None] * nvir + idxvb
    idxvba = idxvb[:,None] * nvir + idxva
    idxvbb = idxvb[:,None] * nvir + idxvb

    r2 = rx.reshape(nocc, nvir**2)
    r2aaa = lib.take_2d(r2, idxoa.ravel(), idxvaa.ravel())
    r2aba = lib.take_2d(r2, idxoa.ravel(), idxvba.ravel())
    r2bab = lib.take_2d(r2, idxob.ravel(), idxvab.ravel())
    r2bbb = lib.take_2d(r2, idxob.ravel(), idxvbb.ravel())

    r2aaa = r2aaa.reshape(nocc_a, nvir_a, nvir_a)
    r2aba = r2aba.reshape(nocc_a, nvir_b, nvir_a)
    r2bab = r2bab.reshape(nocc_b, nvir_a, nvir_b)
    r2bbb = r2bbb.reshape(nocc_b, nvir_b, nvir_b)
    return r2aaa, r2aba, r2bab, r2bbb

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    '''For spin orbitals.

    R2 operators of the form s_{ j}^{ab}, i.e. indices jb are coupled.'''
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca+nvira, noccb+nvirb
    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa,nmob), (nocca,noccb))
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2

    # Fov terms
    Hr1a  = np.einsum('ld,lad->a', imds.Fov, r2aaa)
    Hr1a += np.einsum('LD,LaD->a', imds.FOV, r2bab)
    Hr1b  = np.einsum('ld,lAd->A', imds.Fov, r2aba)
    Hr1b += np.einsum('LD,LAD->A', imds.FOV, r2bbb)

    # Fvv terms
    Hr1a += np.einsum('ac,c->a', imds.Fvv, r1a)
    Hr1b += np.einsum('AC,C->A', imds.FVV, r1b)

    # Wvovv
    Hr1a += 0.5*lib.einsum('acld,lcd->a', imds.Wvvov, r2aaa)
    Hr1a +=     lib.einsum('acLD,LcD->a', imds.WvvOV, r2bab)
    Hr1b += 0.5*lib.einsum('ACLD,LCD->A', imds.WVVOV, r2bbb)
    Hr1b +=     lib.einsum('ACld,lCd->A', imds.WVVov, r2aba)

    #** Wvvvv term
    #:Hr2aaa = lib.einsum('acbd,jcd->jab', eris_vvvv, r2aaa)
    #:Hr2aba = lib.einsum('bdac,jcd->jab', eris_vvVV, r2aba)
    #:Hr2bab = lib.einsum('acbd,jcd->jab', eris_vvVV, r2bab)
    #:Hr2bbb = lib.einsum('acbd,jcd->jab', eris_VVVV, r2bbb)
    u2 = (r2aaa + np.einsum('c,jd->jcd', r1a, t1a) - np.einsum('d,jc->jcd', r1a, t1a),
          r2aba + np.einsum('c,jd->jcd', r1b, t1a),
          r2bab + np.einsum('c,jd->jcd', r1a, t1b),
          r2bbb + np.einsum('c,jd->jcd', r1b, t1b) - np.einsum('d,jc->jcd', r1b, t1b))
    Hr2aaa, Hr2aba, Hr2bab, Hr2bbb = _add_vvvv_ea(eom._cc, u2, eris)
    u2 = None

    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    tmpaaa = lib.einsum('menf,jef->mnj', eris_ovov, r2aaa) * .5
    Hr2aaa += lib.einsum('mnj,mnab->jab', tmpaaa, tauaa)
    tmpaaa = tauaa = None

    tmpbbb = lib.einsum('menf,jef->mnj', eris_OVOV, r2bbb) * .5
    Hr2bbb += lib.einsum('mnj,mnab->jab', tmpbbb, taubb)
    tmpbbb = taubb = None

    tmpabb = lib.einsum('menf,jef->mnj', eris_ovOV, r2bab)
    Hr2bab += lib.einsum('mnj,mnab->jab', tmpabb, tauab)
    tmpaba = lib.einsum('nfme,jef->nmj', eris_ovOV, r2aba)
    Hr2aba += lib.einsum('nmj,nmba->jab', tmpaba, tauab)
    tmpaba = tauab = None
    eris_ovov = eris_OVOV = eris_ovOV = None

    eris_ovvv = imds.eris.get_ovvv(slice(None))
    tmpaaa = lib.einsum('mebf,jef->mjb', eris_ovvv, r2aaa)
    tmpaaa = lib.einsum('mjb,ma->jab', tmpaaa, t1a)
    Hr2aaa-= tmpaaa - tmpaaa.transpose(0,2,1)
    tmpaaa = eris_ovvv = None

    eris_OVVV = imds.eris.get_OVVV(slice(None))
    tmpbbb = lib.einsum('mebf,jef->mjb', eris_OVVV, r2bbb)
    tmpbbb = lib.einsum('mjb,ma->jab', tmpbbb, t1b)
    Hr2bbb-= tmpbbb - tmpbbb.transpose(0,2,1)
    tmpbbb = eris_OVVV = None

    eris_ovVV = imds.eris.get_ovVV(slice(None))
    eris_OVvv = imds.eris.get_OVvv(slice(None))
    tmpaab = lib.einsum('meBF,jFe->mjB', eris_ovVV, r2aba)
    Hr2aba-= lib.einsum('mjB,ma->jBa', tmpaab, t1a)
    tmpabb = lib.einsum('meBF,JeF->mJB', eris_ovVV, r2bab)
    Hr2bab-= lib.einsum('mJB,ma->JaB', tmpabb, t1a)
    tmpaab = tmpabb = eris_ovVV = None

    tmpbaa = lib.einsum('MEbf,jEf->Mjb', eris_OVvv, r2aba)
    Hr2aba-= lib.einsum('Mjb,MA->jAb', tmpbaa, t1b)
    tmpbba = lib.einsum('MEbf,JfE->MJb', eris_OVvv, r2bab)
    Hr2bab-= lib.einsum('MJb,MA->JbA', tmpbba, t1b)
    tmpbaa = tmpbba = eris_OVvv = None
    #** Wvvvv term end

    # Wvvvo
    Hr2aaa += np.einsum('acbj,c->jab', imds.Wvvvo, r1a)
    Hr2bbb += np.einsum('ACBJ,C->JAB', imds.WVVVO, r1b)
    Hr2bab += np.einsum('acBJ,c->JaB', imds.WvvVO, r1a)
    Hr2aba += np.einsum('ACbj,C->jAb', imds.WVVvo, r1b)

    # Wovvo
    tmp2aa = lib.einsum('ldbj,lad->jab', imds.Wovvo, r2aaa)
    tmp2aa += lib.einsum('ldbj,lad->jab', imds.WOVvo, r2bab)
    Hr2aaa += tmp2aa - tmp2aa.transpose(0,2,1)

    Hr2bab += lib.einsum('ldbj,lad->jab', imds.WovVO, r2aaa)
    Hr2bab += lib.einsum('ldbj,lad->jab', imds.WOVVO, r2bab)
    Hr2bab += lib.einsum('ldaj,ldb->jab', imds.WOvvO, r2bab)

    Hr2aba += lib.einsum('ldbj,lad->jab', imds.WOVvo, r2bbb)
    Hr2aba += lib.einsum('ldbj,lad->jab', imds.Wovvo, r2aba)
    Hr2aba += lib.einsum('ldaj,ldb->jab', imds.WoVVo, r2aba)

    tmp2bb = lib.einsum('ldbj,lad->jab', imds.WOVVO, r2bbb)
    tmp2bb += lib.einsum('ldbj,lad->jab', imds.WovVO, r2aba)
    Hr2bbb += tmp2bb - tmp2bb.transpose(0,2,1)

    #Fvv Term
    tmpa = lib.einsum('ac,jcb->jab', imds.Fvv, r2aaa)
    Hr2aaa += tmpa - tmpa.transpose((0,2,1))
    Hr2aba += lib.einsum('AC,jCb->jAb', imds.FVV, r2aba)
    Hr2bab += lib.einsum('ac,JcB->JaB', imds.Fvv, r2bab)
    Hr2aba += lib.einsum('bc, jAc -> jAb', imds.Fvv, r2aba)
    Hr2bab += lib.einsum('BC, JaC -> JaB', imds.FVV, r2bab)
    tmpb = lib.einsum('AC,JCB->JAB', imds.FVV, r2bbb)
    Hr2bbb += tmpb - tmpb.transpose((0,2,1))

    #Foo Term
    Hr2aaa -= lib.einsum('lj,lab->jab', imds.Foo, r2aaa)
    Hr2bbb -= lib.einsum('LJ,LAB->JAB', imds.FOO, r2bbb)
    Hr2bab -= lib.einsum('LJ,LaB->JaB', imds.FOO, r2bab)
    Hr2aba -= lib.einsum('lj,lAb->jAb', imds.Foo, r2aba)

    # Woovv term
    Hr2aaa -= 0.5 * lib.einsum('kcld,lcd,kjab->jab', imds.Wovov, r2aaa, t2aa)
    Hr2bab -= 0.5 * lib.einsum('kcld,lcd,kJaB->JaB', imds.Wovov, r2aaa, t2ab)

    Hr2aba -= lib.einsum('ldKC,lCd,jKbA->jAb', imds.WovOV, r2aba, t2ab)
    Hr2aaa -= lib.einsum('kcLD,LcD,kjab->jab', imds.WovOV, r2bab, t2aa)

    Hr2aba -= 0.5 * lib.einsum('KCLD,LCD,jKbA->jAb', imds.WOVOV, r2bbb, t2ab)
    Hr2bbb -= 0.5 * lib.einsum('KCLD,LCD,KJAB->JAB', imds.WOVOV, r2bbb, t2bb)

    Hr2bbb -= lib.einsum('ldKC,lCd,KJAB->JAB', imds.WovOV, r2aba, t2bb)
    Hr2bab -= lib.einsum('kcLD,LcD,kJaB->JaB', imds.WovOV, r2bab, t2ab)

    vector = amplitudes_to_vector_ea([Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb])
    return vector

def _add_vvvv_ea(mycc, r2, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    r2aaa, r2aba, r2bab, r2bbb = r2
    nocca, noccb = mycc.nocc

    if mycc.direct:
        if getattr(eris, 'mo_coeff', None) is not None:
            mo_a, mo_b = eris.mo_coeff
        else:
            moidxa, moidxb = mycc.get_frozen_mask()
            mo_a = mycc.mo_coeff[0][:,moidxa]
            mo_b = mycc.mo_coeff[1][:,moidxb]

        r2aaa = lib.einsum('xab,pa->xpb', r2aaa, mo_a[:,nocca:])
        r2aaa = lib.einsum('xab,pb->xap', r2aaa, mo_a[:,nocca:])
        r2aba = lib.einsum('xab,pa->xpb', r2aba, mo_b[:,noccb:])
        r2aba = lib.einsum('xab,pb->xap', r2aba, mo_a[:,nocca:])
        r2bab = lib.einsum('xab,pa->xpb', r2bab, mo_a[:,nocca:])
        r2bab = lib.einsum('xab,pb->xap', r2bab, mo_b[:,noccb:])
        r2bbb = lib.einsum('xab,pa->xpb', r2bbb, mo_b[:,noccb:])
        r2bbb = lib.einsum('xab,pb->xap', r2bbb, mo_b[:,noccb:])

        r2 = np.vstack((r2aaa, r2aba, r2bab, r2bbb))
        r2aaa = r2aba = r2bab = r2bbb = None
        time0 = log.timer_debug1('vvvv-tau', *time0)

        buf = ccsd._contract_vvvv_t2(mycc, mycc.mol, None, r2, verbose=log)
        sections = np.cumsum([nocca,nocca,noccb])
        Hr2aaa, Hr2aba, Hr2bab, Hr2bbb = np.split(buf, sections)
        buf = None

        Hr2aaa = lib.einsum('xpb,pa->xab', Hr2aaa, mo_a[:,nocca:])
        Hr2aaa = lib.einsum('xap,pb->xab', Hr2aaa, mo_a[:,nocca:])
        Hr2aba = lib.einsum('xpb,pa->xab', Hr2aba, mo_b[:,noccb:])
        Hr2aba = lib.einsum('xap,pb->xab', Hr2aba, mo_a[:,nocca:])
        Hr2bab = lib.einsum('xpb,pa->xab', Hr2bab, mo_a[:,nocca:])
        Hr2bab = lib.einsum('xap,pb->xab', Hr2bab, mo_b[:,noccb:])
        Hr2bbb = lib.einsum('xpb,pa->xab', Hr2bbb, mo_b[:,noccb:])
        Hr2bbb = lib.einsum('xap,pb->xab', Hr2bbb, mo_b[:,noccb:])

    elif r2aaa.dtype == np.double:
        r2aab = np.asarray(r2aba.transpose(0,2,1), order='C')
        Hr2aab = eris._contract_vvVV_t2(mycc, r2aab, mycc.direct, None)
        Hr2aba = np.asarray(Hr2aab.transpose(0,2,1), order='C')
        r2aab = Hr2aab = None
        Hr2bab = eris._contract_vvVV_t2(mycc, r2bab, mycc.direct, None)
        Hr2aaa = eris._contract_vvvv_t2(mycc, r2aaa, mycc.direct, None)
        Hr2bbb = eris._contract_VVVV_t2(mycc, r2bbb, mycc.direct, None)

    else:
        noccb, nvira, nvirb = r2bab.shape
        eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvira)
        Hr2aaa = lib.einsum('acbd,jcd->jab', eris_vvvv, r2aaa)
        eris_vvvv = None

        eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
        Hr2bbb = lib.einsum('acbd,jcd->jab', eris_VVVV, r2bbb)
        eris_VVVV = None

        sqa = lib.square_mat_in_trilu_indices(nvira)
        sqb = lib.square_mat_in_trilu_indices(nvirb)
        eris_vvVV = np.asarray(eris.vvVV)[:,sqb][sqa]
        Hr2aba = lib.einsum('bdac,jcd->jab', eris_vvVV, r2aba)
        Hr2bab = lib.einsum('acbd,jcd->jab', eris_vvVV, r2bab)
        eris_vvVV = None

    return Hr2aaa, Hr2aba, Hr2bab, Hr2bbb

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    eris = imds.eris
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t2ba = t2ab.transpose(1,0,3,2)

    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    Hr1a = np.diag(imds.Fvv)
    Hr1b = np.diag(imds.FVV)

    #-------------- intermediates

    Fvv_diag = np.diag(imds.Fvv)
    Foo_diag = np.diag(imds.Foo)
    FOO_diag = np.diag(imds.FOO)
    FVV_diag = np.diag(imds.FVV)

    Wovvo_slice = np.einsum('jbbj->jb',imds.Wovvo)
    Wovov_t2_dot = np.einsum('iajb,ijab->jab',imds.Wovov,t2aa)
    WoVVo_slice  = np.einsum('jaaj->ja',imds.WoVVo)
    WovOV_t2_dot = np.einsum('jbia,ijab->jab',imds.WovOV,t2ba)
    WOVVO_slice = np.einsum('jaaj->ja',imds.WOVVO)
    WOvvO_slice = np.einsum('jbbj->jb',imds.WOvvO)
    WovOV_t2_dot_T = np.einsum('ibja,ijba->jab',imds.WovOV,t2ab)
    WOVOV_t2_dot = np.einsum('iajb,ijab->jab',imds.WOVOV,t2bb)

    #-------------- contraction

    Hr2aaa = Fvv_diag[None,:,None]+Fvv_diag[None,None,:]-Foo_diag[:,None,None]+ \
             Wovvo_slice[:,None,:]+Wovvo_slice[:,:,None]-Wovov_t2_dot

    Hr2aba = FVV_diag[None,:,None]+Fvv_diag[None,None,:]-Foo_diag[:,None,None]+ \
             Wovvo_slice[:,None,:]+WoVVo_slice[:,:,None]-WovOV_t2_dot

    Hr2bab = -FOO_diag[:,None,None]+FVV_diag[None,:,None]+Fvv_diag[None,None,:]+ \
             WOVVO_slice[:,:,None]+WOvvO_slice[:,None,:]-WovOV_t2_dot_T
    Hr2bab = Hr2bab.transpose(0,2,1)

    Hr2bbb = -FOO_diag[:,None,None]+FVV_diag[None,:,None]+FVV_diag[None,None,:]+ \
             WOVVO_slice[:,:,None]+WOVVO_slice[:,None,:]-WOVOV_t2_dot

#    if imds.Wvvvv is not None:
#        Wvvvv_slice = np.einsum('aabb->ab',imds.Wvvvv)
#        Hr2aaa += 0.5 * Wvvvv_slice[None,:,:]
#        WVVvv_slice = np.einsum('aabb->ba',imds.WvvVV)
#        Hr2aba += WVVvv_slice[None,:,:]
#        WvvVV_slice = np.einsum('aabb->ab',imds.WvvVV)
#        Hr2bab += WvvVV_slice[None,:,:]
#        WVVVV_slice = np.einsum('aabb->ab',imds.WVVVV)
#        Hr2bbb += 0.5 * WVVVV_slice[None,:,:]

# TODO: test Wvvvv contribution
    # See also the code for Wvvvv contribution in function eeccsd_diag
    tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    Wvvaa = .5*np.einsum('mnab,manb->ab', tauaa, eris_ovov)
    Wvvbb = .5*np.einsum('mnab,manb->ab', taubb, eris_OVOV)
    Wvvab =    np.einsum('mNaB,maNB->aB', tauab, eris_ovOV)
    eris_ovov = eris_OVOV = eris_ovOV = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Wvvaa += np.einsum('mb,maab->ab', t1a[p0:p1], ovvv)
        Wvvaa -= np.einsum('mb,mbaa->ab', t1a[p0:p1], ovvv)
        ovvv = None
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
        Wvvbb += np.einsum('mb,maab->ab', t1b[p0:p1], OVVV)
        Wvvbb -= np.einsum('mb,mbaa->ab', t1b[p0:p1], OVVV)
        OVVV = None
    blksize = min(nocca, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3))))
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
        Wvvab -= np.einsum('mb,mbaa->ba', t1a[p0:p1], ovVV)
        ovVV = None
    blksize = min(noccb, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3))))
    for p0, p1 in lib.prange(0, noccb, blksize):
        OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
        Wvvab -= np.einsum('mb,mbaa->ab', t1b[p0:p1], OVvv)
        OVvv = None
    Wvvaa = Wvvaa + Wvvaa.T
    Wvvbb = Wvvbb + Wvvbb.T
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

    Hr2aaa += Wvvaa[None,:,:]
    Hr2aba += Wvvba[None,:,:]
    Hr2bab += Wvvab[None,:,:]
    Hr2bbb += Wvvbb[None,:,:]
    # Wvvvv contribution end

    vector = amplitudes_to_vector_ea((Hr1a,Hr1b), (Hr2aaa,Hr2aba,Hr2bab,Hr2bbb))
    return vector


class EOMEA(eom_rccsd.EOMEA):
    matvec = eaccsd_matvec
    l_matvec = None
    get_diag = eaccsd_diag
    eaccsd_star = None
    ccsd_star_contract = None

    def __init__(self, cc):
        eom_rccsd.EOMEA.__init__(self, cc)
        self.nocc = cc.get_nocc()
        self.nmo = cc.get_nmo()

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if diag is None:
            diag = self.get_diag()
        if koopmans:
            nocca, noccb = self.nocc
            nmoa, nmob = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
            idx = diag[:nvira+nvirb].argsort()
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

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ea)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ea,
                                         absences=['nmo', 'nocc'])
    spatial2spin = staticmethod(spatial2spin_ea)
    spin2spatial = staticmethod(spin2spatial_ea)

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        return (nvira + nvirb
                + nocca*nvira*(nvira-1)//2 + nocca*nvirb*nvira
                + noccb*nvira*nvirb + noccb*nvirb*(nvirb-1)//2)

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ea()
        return imds

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
        #from pyscf.cc import addons
        #from pyscf.cc import eom_gccsd
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

amplitudes_to_vector_ee = amplitudes_to_vector_eomee = uccsd.amplitudes_to_vector
vector_to_amplitudes_ee = vector_to_amplitudes_eomee = uccsd.vector_to_amplitudes

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

    nbaaa = noccb*nocca*nvira*(nvira-1)//2
    naaba = nocca*(nocca-1)//2*nvirb*nvira
    nabbb = nocca*noccb*nvirb*(nvirb-1)//2
    nbbab = noccb*(noccb-1)//2*nvira*nvirb
    sizes = (nocca*nvirb, noccb*nvira, nbaaa, naaba, nabbb, nbbab)
    sections = np.cumsum(sizes[:-1])
    t1ab, t1ba, vbaaa, vaaba, vabbb, vbbab = np.split(vector, sections)

    t1ab = t1ab.reshape(nocca,nvirb).copy()
    t1ba = t1ba.reshape(noccb,nvira).copy()

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

    vbaaa = vbaaa.reshape(noccb*nocca,-1)
    lib.takebak_2d(t2baaa, vbaaa, oidxab, vtrila[0]*nvira+vtrila[1])
    lib.takebak_2d(t2baaa,-vbaaa, oidxab, vtrila[1]*nvira+vtrila[0])
    vaaba = vaaba.reshape(-1,nvirb*nvira)
    lib.takebak_2d(t2aaba, vaaba, otrila[0]*nocca+otrila[1], vidxab)
    lib.takebak_2d(t2aaba,-vaaba, otrila[1]*nocca+otrila[0], vidxab)
    vabbb = vabbb.reshape(nocca*noccb,-1)
    lib.takebak_2d(t2abbb, vabbb, oidxab, vtrilb[0]*nvirb+vtrilb[1])
    lib.takebak_2d(t2abbb,-vabbb, oidxab, vtrilb[1]*nvirb+vtrilb[0])
    vbbab = vbbab.reshape(-1,nvira*nvirb)
    lib.takebak_2d(t2bbab, vbbab, otrilb[0]*noccb+otrilb[1], vidxab)
    lib.takebak_2d(t2bbab,-vbbab, otrilb[1]*noccb+otrilb[0], vidxab)
    t2baaa = t2baaa.reshape(noccb,nocca,nvira,nvira)
    t2aaba = t2aaba.reshape(nocca,nocca,nvirb,nvira)
    t2abbb = t2abbb.reshape(nocca,noccb,nvirb,nvirb)
    t2bbab = t2bbab.reshape(noccb,noccb,nvira,nvirb)
    return (t1ab,t1ba), (t2baaa, t2aaba, t2abbb, t2bbab)

spatial2spin_eomee = addons.spatial2spin
spin2spatial_eomee = addons.spin2spatial

def spatial2spin_eomsf(rx, orbspin=None):
    '''Convert spin-flip EOMEE spatial-orbital R1/R2 to spin-orbital R1/R2'''
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
    if orbspin is None:
        assert nocca == noccb
        orbspin = np.zeros(nocc+nvir, dtype=int)
        orbspin[1::2] = 1

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
    '''Convert EOMEE spin-orbital R1/R2 to spin-flip EOMEE spatial-orbital R1/R2'''
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
    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa,nmob), (nocca,noccb))
    r1a, r1b = r1
    r2aa, r2ab, r2bb = r2

    #:Hr2aa += lib.einsum('ijef,aebf->ijab', tau2aa, eris.vvvv) * .5
    #:Hr2bb += lib.einsum('ijef,aebf->ijab', tau2bb, eris.VVVV) * .5
    #:Hr2ab += lib.einsum('iJeF,aeBF->iJaB', tau2ab, eris.vvVV)
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

    Hr2aa += lib.einsum('minj,mnab->ijab', imds.woooo, r2aa) * .25
    Hr2bb += lib.einsum('minj,mnab->ijab', imds.wOOOO, r2bb) * .25
    Hr2ab += lib.einsum('miNJ,mNaB->iJaB', imds.wooOO, r2ab)
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
    tmpaa = tau2aa = tauaa = None

    tau2bb = uccsd.make_tau_aa(r2bb, r1b, t1b, 2)
    taubb = uccsd.make_tau_aa(t2bb, t1b, t1b)
    tmpbb = lib.einsum('menf,ijef->mnij', eris_OVOV, tau2bb)
    Hr2bb += lib.einsum('mnij,mnab->ijab', tmpbb, taubb) * 0.25
    tmpbb = tau2bb = taubb = None

    tau2ab = uccsd.make_tau_ab(r2ab, r1 , t1 , 2)
    tauab = uccsd.make_tau_ab(t2ab, t1 , t1)
    tmpab = lib.einsum('meNF,iJeF->mNiJ', eris_ovOV, tau2ab)
    Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', tmpab, tauab)
    tmpab = tau2ab = tauab = None

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
    eris_ovOV = eris_OVOV = None

    Hr2aa-= lib.einsum('mbij,ma->ijab', imds.wovoo, r1a)
    Hr2bb-= lib.einsum('mbij,ma->ijab', imds.wOVOO, r1b)
    Hr2ab-= lib.einsum('mBiJ,ma->iJaB', imds.woVoO, r1a)
    Hr2ab-= lib.einsum('MbJi,MA->iJbA', imds.wOvOo, r1b)

    Hr1a-= 0.5*lib.einsum('mine,mnae->ia', imds.wooov, r2aa)
    Hr1a-=     lib.einsum('miNE,mNaE->ia', imds.wooOV, r2ab)
    Hr1b-= 0.5*lib.einsum('mine,mnae->ia', imds.wOOOV, r2bb)
    Hr1b-=     lib.einsum('MIne,nMeA->IA', imds.wOOov, r2ab)
    tmpa = lib.einsum('mine,me->ni', imds.wooov, r1a)
    tmpa-= lib.einsum('niME,ME->ni', imds.wooOV, r1b)
    tmpb = lib.einsum('mine,me->ni', imds.wOOOV, r1b)
    tmpb-= lib.einsum('NIme,me->NI', imds.wOOov, r1a)
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
    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa,nmob), (nocca,noccb))
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
    Hr2baaa = .5 *lib.einsum('njMI,Mnab->Ijab', imds.wooOO, r2baaa)
    Hr2aaba = .25*lib.einsum('minj,mnAb->ijAb', imds.woooo, r2aaba)
    Hr2abbb = .5 *lib.einsum('miNJ,mNAB->iJAB', imds.wooOO, r2abbb)
    Hr2bbab = .25*lib.einsum('MINJ,MNaB->IJaB', imds.wOOOO, r2bbab)
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

    Hr1ab -= 0.5*lib.einsum('mine,mnAe->iA', imds.wooov, r2aaba)
    Hr1ab -=     lib.einsum('miNE,mNAE->iA', imds.wooOV, r2abbb)
    Hr1ba -= 0.5*lib.einsum('MINE,MNaE->Ia', imds.wOOOV, r2bbab)
    Hr1ba -=     lib.einsum('MIne,Mnae->Ia', imds.wOOov, r2baaa)
    tmp1ab = lib.einsum('MIne,Me->nI', imds.wOOov, r1ba)
    tmp1ba = lib.einsum('miNE,mE->Ni', imds.wooOV, r1ab)
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

    #:Hr2baaa += .5*lib.einsum('Ijef,aebf->Ijab', tau2baaa, eris.vvvv)
    #:Hr2abbb += .5*lib.einsum('iJEF,AEBF->iJAB', tau2abbb, eris.VVVV)
    #:Hr2bbab += .5*lib.einsum('IJeF,aeBF->IJaB', tau2bbab, eris.vvVV)
    #:Hr2aaba += .5*lib.einsum('ijEf,bfAE->ijAb', tau2aaba, eris.vvVV)
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
    jab = np.einsum('kajb,jkab->jab',      ovov, t2aa)
    JAB = np.einsum('kajb,jkab->jab',      OVOV, t2bb)
    jAb =-np.einsum('jbKA,jKbA->jAb', eris_ovOV, t2ab)
    JaB =-np.einsum('kaJB,kJaB->JaB', eris_ovOV, t2ab)
    jaB =-np.einsum('jaKB,jKaB->jaB', eris_ovOV, t2ab)
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

    Wooaa  = np.einsum('iijj->ij', imds.woooo).copy()
    Wooaa -= np.einsum('ijji->ij', imds.woooo)
    Woobb  = np.einsum('iijj->ij', imds.wOOOO).copy()
    Woobb -= np.einsum('ijji->ij', imds.wOOOO)
    Wooab = np.einsum('iijj->ij', imds.wooOO)
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
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ee()
        return imds

class EOMEESpinKeep(EOMEE):
    kernel = eomee_ccsd
    eomee_ccsd = eomee_ccsd
    matvec = eomee_ccsd_matvec
    get_diag = eeccsd_diag

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if diag is None:
            diag = self.get_diag()
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

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ee)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ee,
                                         absences=['nmo', 'nocc'])
    spatial2spin = staticmethod(spatial2spin_eomee)
    spin2spatial = staticmethod(spin2spatial_eomee)

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
        if diag is None:
            diag = self.get_diag()
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

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_eomsf)
    vector_to_amplitudes = module_method(vector_to_amplitudes_eomsf,
                                         absences=['nmo', 'nocc'])
    spatial2spin = staticmethod(spatial2spin_eomsf)
    spin2spatial = staticmethod(spin2spatial_eomsf)

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

uccsd.UCCSD.EOMIP         = lib.class_as_method(EOMIP)
uccsd.UCCSD.EOMEA         = lib.class_as_method(EOMEA)
uccsd.UCCSD.EOMEE         = lib.class_as_method(EOMEE)
uccsd.UCCSD.EOMEESpinKeep = lib.class_as_method(EOMEESpinKeep)
uccsd.UCCSD.EOMEESpinFlip = lib.class_as_method(EOMEESpinFlip)


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

    def _make_shared(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo, self.FOO = uintermediates.Foo(t1, t2, eris)
        self.Fvv, self.FVV = uintermediates.Fvv(t1, t2, eris)
        self.Fov, self.FOV = uintermediates.Fov(t1, t2, eris)

        # 2 virtuals
        self.Wovvo, self.WovVO, self.WOVvo, self.WOVVO, self.WoVVo, self.WOvvO = \
                uintermediates.Wovvo(t1, t2, eris)
        Wovov = np.asarray(eris.ovov)
        WOVOV = np.asarray(eris.OVOV)
        Wovov = Wovov - Wovov.transpose(0,3,2,1)
        WOVOV = WOVOV - WOVOV.transpose(0,3,2,1)
        self.Wovov = Wovov
        self.WovOV = eris.ovOV
        self.WOVov = None
        self.WOVOV = WOVOV

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-CCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo, self.WooOO, _         , self.WOOOO = uintermediates.Woooo(t1, t2, eris)
        self.Wooov, self.WooOV, self.WOOov, self.WOOOV = uintermediates.Wooov(t1, t2, eris)
        self.Woovo, self.WooVO, self.WOOvo, self.WOOVO = uintermediates.Woovo(t1, t2, eris)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-UCCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvvov, self.WvvOV, self.WVVov, self.WVVOV = uintermediates.Wvvov(t1, t2, eris)
        self.Wvvvv = None  # too expensive to hold Wvvvv
        self.Wvvvo, self.WvvVO, self.WVVvo, self.WVVVO = uintermediates.Wvvvo(t1, t2, eris)

        # The contribution of Wvvvv
        t1a, t1b = t1
        # The contraction to eris.vvvv is included in eaccsd_matvec
        #:vvvv = eris.vvvv - eris.vvvv.transpose(0,3,2,1)
        #:VVVV = eris.VVVV - eris.VVVV.transpose(0,3,2,1)
        #:self.Wvvvo += lib.einsum('abef,if->abei',      vvvv, t1a)
        #:self.WvvVO += lib.einsum('abef,if->abei', eris_vvVV, t1b)
        #:self.WVVvo += lib.einsum('efab,if->abei', eris_vvVV, t1a)
        #:self.WVVVO += lib.einsum('abef,if->abei',      VVVV, t1b)

        tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        tmp = lib.einsum('menf,if->meni',      ovov, t1a) * .5
        self.Wvvvo += lib.einsum('meni,mnab->aebi', tmp, tauaa)
        tmp = tauaa = None

        tmp = lib.einsum('menf,if->meni',      OVOV, t1b) * .5
        self.WVVVO += lib.einsum('meni,mnab->aebi', tmp, taubb)
        tmp = taubb = None

        tmp = lib.einsum('menf,if->meni', eris_ovOV, t1b)
        self.WvvVO += lib.einsum('meni,mnab->aebi', tmp, tauab)
        tmp = lib.einsum('nfme,if->meni', eris_ovOV, t1a)
        self.WVVvo += lib.einsum('meni,nmba->aebi', tmp, tauab)
        tauab = None
        ovov = OVOV = eris_ovov = eris_OVOV = eris_ovOV = None

        eris_ovvv = eris.get_ovvv(slice(None))
        ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        tmp = lib.einsum('mebf,if->mebi', ovvv, t1a)
        tmp = lib.einsum('mebi,ma->aebi', tmp, t1a)
        self.Wvvvo -= tmp - tmp.transpose(2,1,0,3)
        tmp = eris_ovvv = ovvv = None

        eris_OVVV = eris.get_OVVV(slice(None))
        OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        tmp = lib.einsum('mebf,if->mebi', OVVV, t1b)
        tmp = lib.einsum('mebi,ma->aebi', tmp, t1b)
        self.WVVVO -= tmp - tmp.transpose(2,1,0,3)
        tmp = eris_OVVV = OVVV = None

        eris_ovVV = eris.get_ovVV(slice(None))
        eris_OVvv = eris.get_OVvv(slice(None))
        tmpaabb = lib.einsum('mebf,if->mebi', eris_ovVV, t1b)
        tmpbaab = lib.einsum('mebf,ie->mfbi', eris_OVvv, t1b)
        tmp  = lib.einsum('mebi,ma->aebi', tmpaabb, t1a)
        tmp += lib.einsum('mfbi,ma->bfai', tmpbaab, t1b)
        self.WvvVO -= tmp
        tmp = tmpaabb = tmpbaab = None

        tmpbbaa = lib.einsum('mebf,if->mebi', eris_OVvv, t1a)
        tmpabba = lib.einsum('mebf,ie->mfbi', eris_ovVV, t1a)
        tmp  = lib.einsum('mebi,ma->aebi', tmpbbaa, t1b)
        tmp += lib.einsum('mfbi,ma->bfai', tmpabba, t1a)
        self.WVVvo -= tmp
        tmp = tmpbbaa = tmpabba = None
        eris_ovVV = eris_OVvv = None
        # The contribution of Wvvvv end

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-UCCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape
        dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

        fooa = eris.focka[:nocca,:nocca]
        foob = eris.fockb[:noccb,:noccb]
        fova = eris.focka[:nocca,nocca:]
        fovb = eris.fockb[:noccb,noccb:]
        fvva = eris.focka[nocca:,nocca:]
        fvvb = eris.fockb[noccb:,noccb:]

        self.Fooa = np.zeros((nocca,nocca), dtype=dtype)
        self.Foob = np.zeros((noccb,noccb), dtype=dtype)
        self.Fvva = np.zeros((nvira,nvira), dtype=dtype)
        self.Fvvb = np.zeros((nvirb,nvirb), dtype=dtype)

        wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
        wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
        woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
        woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
        wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
        wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

        wovoo = np.zeros((nocca,nvira,nocca,nocca), dtype=dtype)
        wOVOO = np.zeros((noccb,nvirb,noccb,noccb), dtype=dtype)
        woVoO = np.zeros((nocca,nvirb,nocca,noccb), dtype=dtype)
        wOvOo = np.zeros((noccb,nvira,noccb,nocca), dtype=dtype)

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
        woooo = lib.einsum('je,nemi->minj', t1a,      ovoo)
        wOOOO = lib.einsum('je,nemi->minj', t1b,      OVOO)
        wooOO = lib.einsum('JE,NEmi->miNJ', t1b, eris_OVoo)
        woOOo = lib.einsum('je,meNI->mINj',-t1a, eris_ovOO)
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
        self.wooov = np.array(     ovoo.transpose(2,3,0,1), dtype=dtype)
        self.wOOOV = np.array(     OVOO.transpose(2,3,0,1), dtype=dtype)
        self.wooOV = np.array(eris_OVoo.transpose(2,3,0,1), dtype=dtype)
        self.wOOov = np.array(eris_ovOO.transpose(2,3,0,1), dtype=dtype)
#X        self.wOooV =-np.array(eris_OVoo.transpose(0,3,2,1), dtype=dtype)
#X        self.woOOv =-np.array(eris_ovOO.transpose(0,3,2,1), dtype=dtype)
        eris_ovoo = eris_OVOO = eris_ovOO = eris_OVoo = None

        woooo += np.asarray(eris.oooo)
        wOOOO += np.asarray(eris.OOOO)
        wooOO += np.asarray(eris.ooOO)
        self.woooo = woooo - woooo.transpose(0,3,2,1)
        self.wOOOO = wOOOO - wOOOO.transpose(0,3,2,1)
        self.wooOO = wooOO - woOOo.transpose(0,3,2,1)

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        tauaa, tauab, taubb = uccsd.make_tau(t2,t1,t1)
        self.woooo += 0.5*lib.einsum('ijef,menf->minj', tauaa,      ovov)
        self.wOOOO += 0.5*lib.einsum('ijef,menf->minj', taubb,      OVOV)
        self.wooOO +=     lib.einsum('iJeF,meNF->miNJ', tauab, eris_ovOV)

        self.wooov += lib.einsum('if,mfne->mine', t1a,      ovov)
        self.wOOOV += lib.einsum('if,mfne->mine', t1b,      OVOV)
        self.wooOV += lib.einsum('if,mfNE->miNE', t1a, eris_ovOV)
        self.wOOov += lib.einsum('IF,neMF->MIne', t1b, eris_ovOV)
#X        self.wOooV -= lib.einsum('if,nfME->MinE', t1a, eris_ovOV)
#X        self.woOOv -= lib.einsum('IF,meNF->mINe', t1b, eris_ovOV)

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
        wovoo -= lib.einsum('nb,minj->mbij', t1a, self.woooo)
        wOVOO -= lib.einsum('nb,minj->mbij', t1b, self.wOOOO)
        woVoO -= lib.einsum('NB,miNJ->mBiJ', t1b, self.wooOO)
        wOvOo -= lib.einsum('nb,njMI->MbIj', t1a, self.wooOO)
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
        log.timer('EOM-UCCSD EE intermediates', *cput0)
