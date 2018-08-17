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

    # k-point spin orbital version of ipccsd

    Hr1 = -0.0*np.einsum('mi,m->i', imds.Foo[kshift], r1)
    for km in range(nkpts):
        Hr1 += np.einsum('me,mie->i', imds.Fov[km], r2[km, kshift])
        for kn in range(nkpts):
            Hr1 += - 0.5 * np.einsum('nmie,mne->i', imds.Wooov[kn, km, kshift],
                                     r2[km, kn])

    Hr2 = np.zeros_like(r2)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        Hr2[ki, kj] += lib.einsum('ae,ije->ija', imds.Fvv[ka], r2[ki, kj])

        Hr2[ki, kj] -= lib.einsum('mi,mja->ija', imds.Foo[ki], r2[ki, kj])
        Hr2[ki, kj] += lib.einsum('mj,mia->ija', imds.Foo[kj], r2[kj, ki])

        Hr2[ki, kj] -= np.einsum('maji,m->ija', imds.Wovoo[kshift, ka, kj], r1)
        #for km in range(nkpts):
        #    kn = kconserv[ki, km, kj]
        #    Hr2[ki, kj] += 0.5 * lib.einsum('mnij,mna->ija',
        #                                    imds.Woooo[km, kn, ki], r2[km, kn])

    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        for km in range(nkpts):
            ke = kconserv[km, kshift, kj]
            Hr2[ki, kj] += lib.einsum('maei,mje->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, kj])

            ke = kconserv[km, kshift, ki]
            Hr2[ki, kj] -= lib.einsum('maej,mie->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, ki])

    tmp = lib.einsum('xymnef,xymnf->e', imds.Woovv[:, :, kshift], r2[:, :])  # contract_{km, kn}
    Hr2[:, :] += 0.5 * lib.einsum('e,xyjiea->xyija', tmp, imds.t2[:, :, kshift])  # sum_{ki, kj}

    # molecular version of ipccsd

    r1a, r1b = spatial_r1
    Hr1a = np.zeros((nocca), dtype=r1.dtype)
    Hr1b = np.zeros((noccb), dtype=r1.dtype)

    ##Foo, Fov, and Wooov
    #Hr1a  = np.einsum('me,mie->i', imds.Fov, r2aaa)
    #Hr1a -= np.einsum('ME,iME->i', imds.FOV, r2abb)
    #Hr1b  = np.einsum('ME,MIE->I', imds.FOV, r2bbb)
    #Hr1b -= np.einsum('me,Ime->I', imds.Fov, r2baa)

    spatial_Foo = uccsd_imds.Foo
    spatial_FOO = uccsd_imds.FOO
    Hr1a += -np.einsum('mi,m->i', spatial_Foo[kshift], r1a)
    Hr1b += -np.einsum('MI,M->I', spatial_FOO[kshift], r1b)

    #Hr1a += -0.5*np.einsum('nime,mne->i', imds.Wooov, r2aaa)
    #Hr1b +=      np.einsum('NIme,Nme->I', imds.WOOov, r2baa)
    #Hr1b += -0.5*np.einsum('NIME,MNE->I', imds.WOOOV, r2bbb)
    #Hr1a +=      np.einsum('niME,nME->i', imds.WooOV, r2abb)

    r2aaa, r2baa, r2abb, r2bbb = spatial_r2
    Hr2aaa = np.zeros((nkpts, nkpts, nocca, nocca, nvira), dtype=r2.dtype)
    Hr2baa = np.zeros((nkpts, nkpts, noccb, nocca, nvira), dtype=r2.dtype)
    Hr2abb = np.zeros((nkpts, nkpts, nocca, noccb, nvirb), dtype=r2.dtype)
    Hr2bbb = np.zeros((nkpts, nkpts, noccb, noccb, nvirb), dtype=r2.dtype)

    ## Fvv term
    #Hr2aaa = lib.einsum('be,ije->ijb', imds.Fvv, r2aaa)
    #Hr2abb = lib.einsum('BE,iJE->iJB', imds.FVV, r2abb)
    #Hr2bbb = lib.einsum('BE,IJE->IJB', imds.FVV, r2bbb)
    #Hr2baa = lib.einsum('be,Ije->Ijb', imds.Fvv, r2baa)

    ## Foo term
    #tmpa = lib.einsum('mi,mjb->ijb', imds.Foo, r2aaa)
    #Hr2aaa -= tmpa - tmpa.transpose((1,0,2))
    #Hr2abb -= lib.einsum('mi,mJB->iJB', imds.Foo, r2abb)
    #Hr2abb -= lib.einsum('MJ,iMB->iJB', imds.FOO, r2abb)
    #Hr2baa -= lib.einsum('MI,Mjb->Ijb', imds.FOO, r2baa)
    #Hr2baa -= lib.einsum('mj,Imb->Ijb', imds.Foo, r2baa)
    #tmpb = lib.einsum('MI,MJB->IJB', imds.FOO, r2bbb)
    #Hr2bbb -= tmpb - tmpb.transpose((1,0,2))

    ## Wovoo term
    #Hr2aaa -= np.einsum('mjbi,m->ijb', imds.Woovo, r1a)
    #Hr2abb += np.einsum('miBJ,m->iJB', imds.WooVO, r1a)
    #Hr2baa += np.einsum('MIbj,M->Ijb', imds.WOOvo, r1b)
    #Hr2bbb -= np.einsum('MJBI,M->IJB', imds.WOOVO, r1b)

    # Woooo term
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]
        for kn in range(nkpts):
            km = kconserv[kj, kn, ki]
            Hr2aaa[ki, kj] += .5 * lib.einsum('minj,mnb->ijb', imds._uccsd_imds.Woooo[km, ki, kn], r2aaa[km, kn])
            Hr2abb[ki, kj] +=      lib.einsum('miNJ,mNB->iJB', imds._uccsd_imds.WooOO[km, ki, kn], r2abb[km, kn])
            Hr2bbb[ki, kj] += .5 * lib.einsum('MINJ,MNB->IJB', imds._uccsd_imds.WOOOO[km, ki, kn], r2bbb[km, kn])
            Hr2baa[ki, kj] +=      lib.einsum('njMI,Mnb->Ijb', imds._uccsd_imds.WooOO[kn, kj, km], r2baa[km, kn])

    ## Wovvo terms
    #tmp = lib.einsum('mebj,ime->ijb', imds.Wovvo, r2aaa)
    #tmp += lib.einsum('MEbj,iME->ijb', imds.WOVvo, r2abb)
    #Hr2aaa += tmp - tmp.transpose(1, 0, 2)

    #WooVV = -imds.WoVVo.transpose(0,3,2,1)
    #WOOvv = -imds.WOvvO.transpose(0,3,2,1)
    #Hr2abb += lib.einsum('MEBJ,iME->iJB', imds.WOVVO, r2abb)
    #Hr2abb += lib.einsum('meBJ,ime->iJB', imds.WovVO, r2aaa)
    #Hr2abb += -lib.einsum('miBE,mJE->iJB', WooVV, r2abb)

    #Hr2baa += lib.einsum('meaj,Ime->Ija', imds.Wovvo, r2baa)
    #Hr2baa += lib.einsum('MEaj,IME->Ija', imds.WOVvo, r2bbb)
    #Hr2baa += -lib.einsum('MIab,Mjb->Ija', WOOvv, r2baa)

    #tmp = lib.einsum('MEBJ,IME->IJB', imds.WOVVO, r2bbb)
    #tmp += lib.einsum('meBJ,Ime->IJB', imds.WovVO, r2baa)
    #Hr2bbb += tmp - tmp.transpose(1, 0, 2)

    ## T2 term
    #Hr2aaa -= 0.5 * lib.einsum('menf,mnf,jibe->ijb', imds.Wovov, r2aaa, t2aa)
    #Hr2aaa -= lib.einsum('meNF,mNF,jibe->ijb', imds.WovOV, r2abb, t2aa)

    #Hr2abb -= 0.5 * lib.einsum('menf,mnf,iJeB->iJB', imds.Wovov, r2aaa, t2ab)
    #Hr2abb -= lib.einsum('meNF,mNF,iJeB->iJB', imds.WovOV, r2abb, t2ab)

    #Hr2baa -= 0.5 * lib.einsum('MENF,MNF,jIbE->Ijb', imds.WOVOV, r2bbb, t2ab)
    #Hr2baa -= lib.einsum('nfME,Mnf,jIbE->Ijb', imds.WovOV, r2baa, t2ab)

    #Hr2bbb -= 0.5 * lib.einsum('MENF,MNF,JIBE->IJB', imds.WOVOV, r2bbb, t2bb)
    #Hr2bbb -= lib.einsum('nfME,Mnf,JIBE->IJB', imds.WovOV, r2baa, t2bb)

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

    # k-point spin orbital version of eaccsd

    Hr1 = 0.0*np.einsum('ac,c->a', imds.Fvv[kshift], r1)
    for kl in range(nkpts):
        Hr1 += np.einsum('ld,lad->a', imds.Fov[kl], r2[kl, kshift])
        for kc in range(nkpts):
            Hr1 += 0.5*np.einsum('alcd,lcd->a', imds.Wvovv[kshift,kl,kc], r2[kl,kc])

    Hr2 = np.zeros_like(r2)
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        Hr2[kj,ka] += np.einsum('abcj,c->jab', imds.Wvvvo[ka,kb,kshift], r1)
        Hr2[kj,ka] += lib.einsum('ac,jcb->jab', imds.Fvv[ka], r2[kj,ka])
        Hr2[kj,ka] -= lib.einsum('bc,jca->jab', imds.Fvv[kb], r2[kj,kb])
        Hr2[kj,ka] -= lib.einsum('lj,lab->jab', imds.Foo[kj], r2[kj,ka])

        for kd in range(nkpts):
            kl = kconserv[kj, kb, kd]
            Hr2[kj, ka] += lib.einsum('lbdj,lad->jab', imds.Wovvo[kl, kb, kd], r2[kl, ka])

            # P(ab)
            kl = kconserv[kj, ka, kd]
            Hr2[kj, ka] -= lib.einsum('ladj,lbd->jab', imds.Wovvo[kl, ka, kd], r2[kl, kb])

            kc = kconserv[ka, kd, kb]
            Hr2[kj, ka] += 0.5 * lib.einsum('abcd,jcd->jab', imds.Wvvvv[ka, kb, kc], r2[kj, kc])

    tmp = lib.einsum('xyklcd,xylcd->k', imds.Woovv[kshift, :, :], r2[:, :])  # contract_{kl, kc}
    Hr2[:, :] -= 0.5*lib.einsum('k,xykjab->xyjab', tmp, imds.t2[kshift, :, :])  # sum_{kj, ka]

    # molecular version of eaccsd

    r1a, r1b = spatial_r1
    Hr1a = np.zeros((nvira), dtype=r1.dtype)
    Hr1b = np.zeros((nvirb), dtype=r1.dtype)

    ## Fov terms
    #Hr1a  = np.einsum('ld,lad->a', imds.Fov, r2aaa)
    #Hr1a += np.einsum('LD,LaD->a', imds.FOV, r2bab)
    #Hr1b  = np.einsum('ld,lAd->A', imds.Fov, r2aba)
    #Hr1b += np.einsum('LD,LAD->A', imds.FOV, r2bbb)

    # Fvv terms
    Hr1a += np.einsum('ac,c->a', uccsd_imds.Fvv[kshift], r1a)
    Hr1b += np.einsum('AC,C->A', uccsd_imds.FVV[kshift], r1b)

    ## Wvovv
    #Hr1a += 0.5*lib.einsum('acld,lcd->a', imds.Wvvov, r2aaa)
    #Hr1a +=     lib.einsum('acLD,LcD->a', imds.WvvOV, r2bab)
    #Hr1b += 0.5*lib.einsum('ACLD,LCD->A', imds.WVVOV, r2bbb)
    #Hr1b +=     lib.einsum('ACld,lCd->A', imds.WVVov, r2aba)

    r2aaa, r2aba, r2bab, r2bbb = spatial_r2
    Hr2aaa = np.zeros((nkpts, nkpts, nocca, nvira, nvira), dtype=r2.dtype)
    Hr2aba = np.zeros((nkpts, nkpts, nocca, nvirb, nvira), dtype=r2.dtype)
    Hr2bab = np.zeros((nkpts, nkpts, noccb, nvira, nvirb), dtype=r2.dtype)
    Hr2bbb = np.zeros((nkpts, nkpts, noccb, nvirb, nvirb), dtype=r2.dtype)

    ##** Wvvvv term
    ##:Hr2aaa = lib.einsum('acbd,jcd->jab', eris_vvvv, r2aaa)
    ##:Hr2aba = lib.einsum('bdac,jcd->jab', eris_vvVV, r2aba)
    ##:Hr2bab = lib.einsum('acbd,jcd->jab', eris_vvVV, r2bab)
    ##:Hr2bbb = lib.einsum('acbd,jcd->jab', eris_VVVV, r2bbb)
    #u2 = (r2aaa + np.einsum('c,jd->jcd', r1a, t1a) - np.einsum('d,jc->jcd', r1a, t1a),
    #      r2aba + np.einsum('c,jd->jcd', r1b, t1a),
    #      r2bab + np.einsum('c,jd->jcd', r1a, t1b),
    #      r2bbb + np.einsum('c,jd->jcd', r1b, t1b) - np.einsum('d,jc->jcd', r1b, t1b))
    #Hr2aaa, Hr2aba, Hr2bab, Hr2bbb = _add_vvvv_ea(eom._cc, u2, eris)
    #u2 = None

    #tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
    #eris_ovov = np.asarray(eris.ovov)
    #eris_OVOV = np.asarray(eris.OVOV)
    #eris_ovOV = np.asarray(eris.ovOV)
    #tmpaaa = lib.einsum('menf,jef->mnj', eris_ovov, r2aaa) * .5
    #Hr2aaa += lib.einsum('mnj,mnab->jab', tmpaaa, tauaa)
    #tmpaaa = tauaa = None

    #tmpbbb = lib.einsum('menf,jef->mnj', eris_OVOV, r2bbb) * .5
    #Hr2bbb += lib.einsum('mnj,mnab->jab', tmpbbb, taubb)
    #tmpbbb = taubb = None

    #tmpabb = lib.einsum('menf,jef->mnj', eris_ovOV, r2bab)
    #Hr2bab += lib.einsum('mnj,mnab->jab', tmpabb, tauab)
    #tmpaba = lib.einsum('nfme,jef->nmj', eris_ovOV, r2aba)
    #Hr2aba += lib.einsum('nmj,nmba->jab', tmpaba, tauab)
    #tmpbab = tmpaba = tauab = None
    #eris_ovov = eris_OVOV = eris_ovOV = None

    #eris_ovvv = imds.eris.get_ovvv(slice(None))
    #tmpaaa = lib.einsum('mebf,jef->mjb', eris_ovvv, r2aaa)
    #tmpaaa = lib.einsum('mjb,ma->jab', tmpaaa, t1a)
    #Hr2aaa-= tmpaaa - tmpaaa.transpose(0,2,1)
    #tmpaaa = eris_ovvv = None

    #eris_OVVV = imds.eris.get_OVVV(slice(None))
    #tmpbbb = lib.einsum('mebf,jef->mjb', eris_OVVV, r2bbb)
    #tmpbbb = lib.einsum('mjb,ma->jab', tmpbbb, t1b)
    #Hr2bbb-= tmpbbb - tmpbbb.transpose(0,2,1)
    #tmpbbb = eris_OVVV = None

    #eris_ovVV = imds.eris.get_ovVV(slice(None))
    #eris_OVvv = imds.eris.get_OVvv(slice(None))
    #tmpaab = lib.einsum('meBF,jFe->mjB', eris_ovVV, r2aba)
    #Hr2aba-= lib.einsum('mjB,ma->jBa', tmpaab, t1a)
    #tmpabb = lib.einsum('meBF,JeF->mJB', eris_ovVV, r2bab)
    #Hr2bab-= lib.einsum('mJB,ma->JaB', tmpabb, t1a)
    #tmpaab = tmpabb = eris_ovVV = None

    #tmpbaa = lib.einsum('MEbf,jEf->Mjb', eris_OVvv, r2aba)
    #Hr2aba-= lib.einsum('Mjb,MA->jAb', tmpbaa, t1b)
    #tmpbba = lib.einsum('MEbf,JfE->MJb', eris_OVvv, r2bab)
    #Hr2bab-= lib.einsum('MJb,MA->JbA', tmpbba, t1b)
    #tmpbaa = tmpbba = eris_OVvv = None
    ##** Wvvvv term end

    ## Wvvvo
    #Hr2aaa += np.einsum('acbj,c->jab', imds.Wvvvo, r1a)
    #Hr2bbb += np.einsum('ACBJ,C->JAB', imds.WVVVO, r1b)
    #Hr2bab += np.einsum('acBJ,c->JaB', imds.WvvVO, r1a)
    #Hr2aba += np.einsum('ACbj,C->jAb', imds.WVVvo, r1b)

    ## Wovvo
    #tmp2aa = lib.einsum('ldbj,lad->jab', imds.Wovvo, r2aaa)
    #tmp2aa += lib.einsum('ldbj,lad->jab', imds.WOVvo, r2bab)
    #Hr2aaa += tmp2aa - tmp2aa.transpose(0,2,1)

    #Hr2bab += lib.einsum('ldbj,lad->jab', imds.WovVO, r2aaa)
    #Hr2bab += lib.einsum('ldbj,lad->jab', imds.WOVVO, r2bab)
    #Hr2bab += lib.einsum('ldaj,ldb->jab', imds.WOvvO, r2bab)

    #Hr2aba += lib.einsum('ldbj,lad->jab', imds.WOVvo, r2bbb)
    #Hr2aba += lib.einsum('ldbj,lad->jab', imds.Wovvo, r2aba)
    #Hr2aba += lib.einsum('ldaj,ldb->jab', imds.WoVVo, r2aba)

    #tmp2bb = lib.einsum('ldbj,lad->jab', imds.WOVVO, r2bbb)
    #tmp2bb += lib.einsum('ldbj,lad->jab', imds.WovVO, r2aba)
    #Hr2bbb += tmp2bb - tmp2bb.transpose(0,2,1)

    ##Fvv Term
    #tmpa = lib.einsum('ac,jcb->jab', imds.Fvv, r2aaa)
    #Hr2aaa += tmpa - tmpa.transpose((0,2,1))
    #Hr2aba += lib.einsum('AC,jCb->jAb', imds.FVV, r2aba)
    #Hr2bab += lib.einsum('ac,JcB->JaB', imds.Fvv, r2bab)
    #Hr2aba += lib.einsum('bc, jAc -> jAb', imds.Fvv, r2aba)
    #Hr2bab += lib.einsum('BC, JaC -> JaB', imds.FVV, r2bab)
    #tmpb = lib.einsum('AC,JCB->JAB', imds.FVV, r2bbb)
    #Hr2bbb += tmpb - tmpb.transpose((0,2,1))

    ##Foo Term
    #Hr2aaa -= lib.einsum('lj,lab->jab', imds.Foo, r2aaa)
    #Hr2bbb -= lib.einsum('LJ,LAB->JAB', imds.FOO, r2bbb)
    #Hr2bab -= lib.einsum('LJ,LaB->JaB', imds.FOO, r2bab)
    #Hr2aba -= lib.einsum('lj,lAb->jAb', imds.Foo, r2aba)

    ## Woovv term
    #Hr2aaa -= 0.5 * lib.einsum('kcld,lcd,kjab->jab', imds.Wovov, r2aaa, t2aa)
    #Hr2bab -= 0.5 * lib.einsum('kcld,lcd,kJaB->JaB', imds.Wovov, r2aaa, t2ab)

    #Hr2aba -= lib.einsum('ldKC,lCd,jKbA->jAb', imds.WovOV, r2aba, t2ab)
    #Hr2aaa -= lib.einsum('kcLD,LcD,kjab->jab', imds.WovOV, r2bab, t2aa)

    #Hr2aba -= 0.5 * lib.einsum('KCLD,LCD,jKbA->jAb', imds.WOVOV, r2bbb, t2ab)
    #Hr2bbb -= 0.5 * lib.einsum('KCLD,LCD,KJAB->JAB', imds.WOVOV, r2bbb, t2bb)

    #Hr2bbb -= lib.einsum('ldKC,lCd,KJAB->JAB', imds.WovOV, r2aba, t2bb)
    #Hr2bab -= lib.einsum('kcLD,LcD,kJaB->JaB', imds.WovOV, r2bab, t2ab)

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
        #self.Fov, self.FOV = kintermediates_uhf.Fov(t1, t2, eris)  # TODO

        ## 2 virtuals
        #self.Wovvo, self.WovVO, self.WOVvo, self.WOVVO, self.WoVVo, self.WOvvO = \
        #        kintermediates_uhf.Wovvo(t1, t2, eris)
        #Wovov = np.asarray(eris.ovov)
        #WOVOV = np.asarray(eris.OVOV)
        #Wovov = Wovov - Wovov.transpose(0,3,2,1)
        #WOVOV = WOVOV - WOVOV.transpose(0,3,2,1)
        #self.Wovov = Wovov
        #self.WovOV = eris.ovOV
        #self.WOVov = None
        #self.WOVOV = WOVOV

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-KCCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo, self.WooOO, _         , self.WOOOO = kintermediates_uhf.Woooo(self._cc, t1, t2, eris)
        #self.Wooov, self.WooOV, self.WOOov, self.WOOOV = kintermediates_uhf.Wooov(t1, t2, eris)  # TODO
        #self.Woovo, self.WooVO, self.WOOvo, self.WOOVO = kintermediates_uhf.Woovo(t1, t2, eris)  # TODO

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-KUCCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        #self.Wvvov, self.WvvOV, self.WVVov, self.WVVOV = kintermediates_uhf.Wvvov(t1, t2, eris)  # TODO
        #self.Wvvvv = None  # too expensive to hold Wvvvv
        #self.Wvvvo, self.WvvVO, self.WVVvo, self.WVVVO = kintermediates_uhf.Wvvvo(t1, t2, eris)  # TODO

        ## The contribution of Wvvvv
        #t1a, t1b = t1
        ## The contraction to eris.vvvv is included in eaccsd_matvec
        ## TODO: Not included in current kccsd implementation
        ##:vvvv = eris.vvvv - eris.vvvv.transpose(0,3,2,1)
        ##:VVVV = eris.VVVV - eris.VVVV.transpose(0,3,2,1)
        ##:self.Wvvvo += lib.einsum('abef,if->abei',      vvvv, t1a)
        ##:self.WvvVO += lib.einsum('abef,if->abei', eris_vvVV, t1b)
        ##:self.WVVvo += lib.einsum('efab,if->abei', eris_vvVV, t1a)
        ##:self.WVVVO += lib.einsum('abef,if->abei',      VVVV, t1b)

        #tauaa, tauab, taubb = uccsd.make_tau(t2, t1, t1)
        #eris_ovov = np.asarray(eris.ovov)
        #eris_OVOV = np.asarray(eris.OVOV)
        #eris_ovOV = np.asarray(eris.ovOV)
        #ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        #OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        #tmp = lib.einsum('menf,if->meni',      ovov, t1a) * .5
        #self.Wvvvo += lib.einsum('meni,mnab->aebi', tmp, tauaa)
        #tmp = tauaa = None

        #tmp = lib.einsum('menf,if->meni',      OVOV, t1b) * .5
        #self.WVVVO += lib.einsum('meni,mnab->aebi', tmp, taubb)
        #tmp = taubb = None

        #tmp = lib.einsum('menf,if->meni', eris_ovOV, t1b)
        #self.WvvVO += lib.einsum('meni,mnab->aebi', tmp, tauab)
        #tmp = lib.einsum('nfme,if->meni', eris_ovOV, t1a)
        #self.WVVvo += lib.einsum('meni,nmba->aebi', tmp, tauab)
        #tmpbab = tmpaba = tauab = None
        #ovov = OVOV = eris_ovov = eris_OVOV = eris_ovOV = None

        #eris_ovvv = eris.get_ovvv(slice(None))
        #ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #tmp = lib.einsum('mebf,if->mebi', ovvv, t1a)
        #tmp = lib.einsum('mebi,ma->aebi', tmp, t1a)
        #self.Wvvvo -= tmp - tmp.transpose(2,1,0,3)
        #tmp = eris_ovvv = ovvv = None

        #eris_OVVV = eris.get_OVVV(slice(None))
        #OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #tmp = lib.einsum('mebf,if->mebi', OVVV, t1b)
        #tmp = lib.einsum('mebi,ma->aebi', tmp, t1b)
        #self.WVVVO -= tmp - tmp.transpose(2,1,0,3)
        #tmp = eris_OVVV = OVVV = None

        #eris_ovVV = eris.get_ovVV(slice(None))
        #eris_OVvv = eris.get_OVvv(slice(None))
        #tmpaabb = lib.einsum('mebf,if->mebi', eris_ovVV, t1b)
        #tmpbaab = lib.einsum('mebf,ie->mfbi', eris_OVvv, t1b)
        #tmp  = lib.einsum('mebi,ma->aebi', tmpaabb, t1a)
        #tmp += lib.einsum('mfbi,ma->bfai', tmpbaab, t1b)
        #self.WvvVO -= tmp
        #tmp = tmpaabb = tmpbaab = None

        #tmpbbaa = lib.einsum('mebf,if->mebi', eris_OVvv, t1a)
        #tmpabba = lib.einsum('mebf,ie->mfbi', eris_ovVV, t1a)
        #tmp  = lib.einsum('mebi,ma->aebi', tmpbbaa, t1b)
        #tmp += lib.einsum('mfbi,ma->bfai', tmpabba, t1a)
        #self.WVVvo -= tmp
        #tmp = tmpbbaa = tmpabba = None
        #eris_ovVV = eris_OVvv = None
        ## The contribution of Wvvvv end

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

    import kccsd_uhf
    mycc = kccsd_uhf.KUCCSD(kmf)
    eris = mycc.ao2mo()

    t1, t2 = rand_t1_t2(mycc)
    mycc.t1 = t1
    mycc.t2 = t2
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)

    from pyscf.pbc.cc import kccsd
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

    print('ea Hr1a',   abs(lib.finger(Hr1a)   - (-0.34462696543560045-1.6104596956729178j)))
    print('ea Hr1b',   abs(lib.finger(Hr1b)   - (-0.055793611517250929+0.22169994342782473j)))
    print('ea Hr2aaa', abs(lib.finger(Hr2aaa) - (0.64317341651618687-1.9454081575555195j)))
    print('ea Hr2baa', abs(lib.finger(Hr2baa) - (2.8978462590588068+2.1220361054795198j)))
    print('ea Hr2abb', abs(lib.finger(Hr2abb) - (1.5482378135185952-5.4536785087220636j)))
    print('ea Hr2bbb', abs(lib.finger(Hr2bbb) - (0.43434684782030114+0.24259097270424451j)))

    # EA version
    myeom = EOMEA(mycc)
    imds = myeom.make_imds(eris=kccsd_eris, t1=spin_t1, t2=spin_t2)
    imds._uccsd_eris = eris
    imds._uccsd_imds = _IMDS(mycc, eris=eris)
    imds._uccsd_imds.make_ip()

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
