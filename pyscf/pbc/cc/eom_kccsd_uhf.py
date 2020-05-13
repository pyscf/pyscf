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
# Authors: James D. McClain
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#

import time
import itertools
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa
from pyscf.pbc.cc import kintermediates_uhf
from pyscf.pbc.mp.kump2 import (get_frozen_mask, get_nocc, get_nmo,
                                padded_mo_coeff, padding_k_idx)  # noqa

einsum = lib.einsum

########################################
# EOM-IP-CCSD
########################################

def amplitudes_to_vector_ip(r1, r2, kshift, kconserv):
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    nkpts = r2aaa.shape[0]
    nocca, noccb = r1a.shape[0], r1b.shape[0]
    nvira, nvirb = r2aaa.shape[-1], r2bbb.shape[-1]
    # From symmetry for aaa and bbb terms, only store lower
    # triangular part (ki,i) < (kj,j)
    idxa, idya = np.tril_indices(nkpts*nocca, -1)
    idxb, idyb = np.tril_indices(nkpts*noccb, -1)
    r2aaa = r2aaa.transpose(0,2,1,3,4).reshape(nkpts*nocca,nkpts*nocca,nvira)
    r2bbb = r2bbb.transpose(0,2,1,3,4).reshape(nkpts*noccb,nkpts*noccb,nvirb)
    return np.hstack((r1a, r1b, r2aaa[idxa,idya].ravel(),
                      r2baa.ravel(), r2abb.ravel(),
                      r2bbb[idxb,idyb].ravel()))

def vector_to_amplitudes_ip(vector, kshift, nkpts, nmo, nocc, kconserv):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nocca, noccb, (nkpts*nocca)*(nkpts*nocca-1)*nvira//2,
             nkpts**2*noccb*nocca*nvira, nkpts**2*nocca*noccb*nvirb,
             nkpts*noccb*(nkpts*noccb-1)*nvirb//2)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2a, r2baa, r2abb, r2b = np.split(vector, sections)

    r2a = r2a.reshape(nkpts*nocca*(nkpts*nocca-1)//2,nvira)
    r2b = r2b.reshape(nkpts*noccb*(nkpts*noccb-1)//2,nvirb)

    idxa, idya = np.tril_indices(nkpts*nocca, -1)
    idxb, idyb = np.tril_indices(nkpts*noccb, -1)

    r2aaa = np.zeros((nkpts*nocca,nkpts*nocca,nvira), dtype=r2a.dtype)
    r2aaa[idxa,idya] = r2a.copy()
    r2aaa[idya,idxa] = -r2a.copy()  # Fill in value :  kj, j < ki, i
    r2aaa = r2aaa.reshape(nkpts,nocca,nkpts,nocca,nvira)
    r2aaa = r2aaa.transpose(0,2,1,3,4)
    r2baa = r2baa.reshape(nkpts,nkpts,noccb,nocca,nvira).copy()
    r2abb = r2abb.reshape(nkpts,nkpts,nocca,noccb,nvirb).copy()
    r2bbb = np.zeros((nkpts*noccb,nkpts*noccb,nvirb), dtype=r2b.dtype)
    r2bbb[idxb,idyb] = r2b.copy()
    r2bbb[idyb,idxb] = -r2b.copy()  # Fill in value :  kj, j < ki, i
    r2bbb = r2bbb.reshape(nkpts,noccb,nkpts,noccb,nvirb)
    r2bbb = r2bbb.transpose(0,2,1,3,4)

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2baa, r2abb, r2bbb)
    return r1, r2

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled'''
    if imds is None: imds = eom.make_imds()
    t1, t2= imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape[3:]
    nmoa, nmob = nocca + nvira, noccb + nvirb
    kconserv = imds.kconserv
    nkpts = eom.nkpts

    r1, r2 = eom.vector_to_amplitudes(vector, kshift, nkpts, (nmoa, nmob), (nocca, noccb), kconserv)




    #nocc = eom.nocc
    #nmo = eom.nmo
    #nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
    #nocca, noccb = nocc
    #nvira, nvirb = nvir
    #nkpts = eom.nkpts
    #r1, r2 = eom.vector_to_amplitudes(vector, nkpts, nmo[0]+nmo[1], nocc[0]+nocc[1])  # spin
    #spatial_r1, spatial_r2 = eom_kgccsd.spin2spatial_ip_doublet(r1, r2, kconserv, kshift, orbspin)
    #imds = imds._imds
    #t2aa, t2ab, t2bb = t2

    # k-point spin orbital version of ipccsd

    #Hr1 = -0.0*np.einsum('mi,m->i', imds.Foo[kshift], r1)

    #Hr2 = np.zeros_like(r2)

    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2

    #Foo term
    # -\sum_{kk,k} U_{kk,k,ki,i} s_{kk,k}
    Hr1a = -np.einsum('mi,m->i', imds.Foo[kshift], r1a)
    Hr1b = -np.einsum('MI,M->I', imds.FOO[kshift], r1b)

    #Fov term
    # \sum_{kL,kD,L,D} U_{kL,kD,L,D} S_{ki,i,kL,L}^{kD,D} + \sum_{kl,kd,l,d} U_{kl,kd,l,d} S_{ki,i,kl,l}^{kd,d}
    for km in range(nkpts):
        Hr1a += einsum('me,mie->i', imds.Fov[km], r2aaa[km,kshift])
        Hr1a -= einsum('ME,iME->i', imds.FOV[km], r2abb[kshift,km])
        Hr1b += einsum('ME,MIE->I', imds.FOV[km], r2bbb[km,kshift])
        Hr1b -= einsum('me,Ime->I', imds.Fov[km], r2baa[kshift,km])

    #Wooov
    # \sum_{kk,kl,kd,k,l,d} W_{kk,ki,kl,kd,k,i,l,d} s_{kl,kk,l,k}^{kd,d}
    # \sum_{kk,kL,kD,k,L,D} W_{kk,ki,kL,kD,k,i,L,D} s_{kL,kk,L,k}^{kD,D}
    for km in range(nkpts):
        for kn in range(nkpts):
            Hr1a += -0.5 * einsum('nime,mne->i', imds.Wooov[kn,kshift,km], r2aaa[km,kn])
            Hr1b +=        einsum('NIme,Nme->I', imds.WOOov[kn,kshift,km], r2baa[kn,km])
            Hr1b += -0.5 * einsum('NIME,MNE->I', imds.WOOOV[kn,kshift,km], r2bbb[km,kn])
            Hr1a +=        einsum('niME,nME->i', imds.WooOV[kn,kshift,km], r2abb[kn,km])

    dtype = np.result_type(Hr1a, *r2)
    Hr2aaa = np.zeros((nkpts, nkpts, nocca, nocca, nvira), dtype=dtype)
    Hr2baa = np.zeros((nkpts, nkpts, noccb, nocca, nvira), dtype=dtype)
    Hr2abb = np.zeros((nkpts, nkpts, nocca, noccb, nvirb), dtype=dtype)
    Hr2bbb = np.zeros((nkpts, nkpts, noccb, noccb, nvirb), dtype=dtype)

    # Fvv term
    # \sum_{kd,d} U_{kb,kd,b,d} S_{ki,kj,i,j}^{kd,d} = (\bar{H}S)_{ki,kj,i,j}^{kb,b}
    # \sum_{kD,D} S_{ki,kJ,i,J}^{kD,D} U_{kB,kD,B,D} = (\bar{H}S)_{ki,kJ,i,J}^{kB,B}
    for kb, ki in itertools.product(range(nkpts),repeat=2):
        kj = kconserv[kshift,ki,kb]
        Hr2aaa[ki,kj] += lib.einsum('be,ije->ijb', imds.Fvv[kb], r2aaa[ki,kj])
        Hr2abb[ki,kj] += lib.einsum('BE,iJE->iJB', imds.FVV[kb], r2abb[ki,kj])
        Hr2bbb[ki,kj] += lib.einsum('BE,IJE->IJB', imds.FVV[kb], r2bbb[ki,kj])
        Hr2baa[ki,kj] += lib.einsum('be,Ije->Ijb', imds.Fvv[kb], r2baa[ki,kj])

    # Foo term
    # \sum_{kl,l} U_{kl,ki,l,i} s_{kl,kj,l,j}^{kb,b} = (\bar{H}S)_{ki,kj,i,j}^{kb,b}
    # \sum_{kl,l} U_{kl,kj,l,j} S_{ki,kl,i,l}^{kb,b} = (\bar{H}S)_{ki,kj,i,j}^{kb,b}

    # \sum_{kl,l} S_{kl,kJ,l,J}^{kB,B} U_{kl,ki,l,i} = (\bar{H}S)_{ki,kJ,i,J}^{kB,B}
    # \sum_{KL,L} S_{ki,kL,i,L}^{kB,B} U_{kL,kJ,L,J} = (\bar{H}S)_{ki,kJ,i,J}^{kB,B}
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        tmpa = lib.einsum('mi,mjb->ijb', imds.Foo[ki], r2aaa[ki,kj])
        tmpb = lib.einsum('mj,mib->ijb', imds.Foo[kj], r2aaa[kj,ki])
        Hr2aaa[ki,kj] -= tmpa - tmpb
        Hr2abb[ki,kj] -= lib.einsum('mi,mJB->iJB', imds.Foo[ki], r2abb[ki,kj])
        Hr2abb[ki,kj] -= lib.einsum('MJ,iMB->iJB', imds.FOO[kj], r2abb[ki,kj])
        Hr2baa[ki,kj] -= lib.einsum('MI,Mjb->Ijb', imds.FOO[ki], r2baa[ki,kj])
        Hr2baa[ki,kj] -= lib.einsum('mj,Imb->Ijb', imds.Foo[kj], r2baa[ki,kj])
        tmpb = lib.einsum('MI,MJB->IJB', imds.FOO[ki], r2bbb[ki,kj])
        tmpa = lib.einsum('MJ,MIB->IJB', imds.FOO[kj], r2bbb[kj,ki])
        Hr2bbb[ki,kj] -= tmpb - tmpa

    # Wovoo term
    # \sum_{kk,k} W_{kk,kb,kj,ki,k,b,j,i} s_{kk,k} = (\bar{H}S)_{ki,kj,i,j}^{kb,b}
    # \sum_{kk,k} W_{kk,kB,ki,kJ,k,B,i,J} S_{kk,k} = (\bar{H}S)_{ki,kJ,i,J}^{kB,B}
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]
        Hr2aaa[ki,kj] -= einsum('mjbi,m->ijb', imds.Woovo[kshift,kj,kb], r1a)
        Hr2abb[ki,kj] += einsum('miBJ,m->iJB', imds.WooVO[kshift,ki,kb], r1a)
        Hr2baa[ki,kj] += einsum('MIbj,M->Ijb', imds.WOOvo[kshift,ki,kb], r1b)
        Hr2bbb[ki,kj] -= einsum('MJBI,M->IJB', imds.WOOVO[kshift,kj,kb], r1b)

    # Woooo term
    # \sum_{kk,kl,k,l} W_{kk,ki,kl,kj,k,i,l,j} S_{kk,kl,k,l}^{kb,b} = (\bar{H}S)_{ki,kj,i,j}^{kb,b}
    # \sum_{kk,kL,k,L} W_{kk,kL,ki,kJ,k,L,i,J} S_{kk,kl,k,L}^{kB,B} = (\bar{H}S)_{ki,kJ,i,J}^{kB,B}
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]
        for kn in range(nkpts):
            km = kconserv[kj, kn, ki]
            Hr2aaa[ki, kj] += .5 * lib.einsum('minj,mnb->ijb', imds.Woooo[km, ki, kn], r2aaa[km, kn])
            Hr2abb[ki, kj] +=      lib.einsum('miNJ,mNB->iJB', imds.WooOO[km, ki, kn], r2abb[km, kn])
            Hr2bbb[ki, kj] += .5 * lib.einsum('MINJ,MNB->IJB', imds.WOOOO[km, ki, kn], r2bbb[km, kn])
            Hr2baa[ki, kj] +=      lib.einsum('njMI,Mnb->Ijb', imds.WooOO[kn, kj, km], r2baa[km, kn])

    # T2 term
    # - \sum_{kc,c} t_{kj,ki,j,i}^{kb,kc,b,c} [ \sum_{kk,kL,kD,k,L,D} W_{kL,kk,kD,kc,L,k,D,c} S_{kk,kL,k,L}^{kD,D}
    # + \sum{kk,kl,kd,k,l,d} W_{kl,kk,kd,kc,l,k,d,c} S_{kk,kl,k,l}^{kd,d} ] = (\bar{H}S)_{ki,kj,i,j}^{kb,b}
    #
    # - \sum_{kc,c} t_{ki,kJ,i,J}^{kc,kB,c,B} [ \sum_{kk,kL,kD,k,L,D} W_{kL,kk,kD,kc,L,k,D,c} S_{Kk,kL,k,L}^{kD,D}
    # + \sum{kk,kl,kd,k,l,d} W_{kl,kk,kd,kc,l,k,d,c} S_{kk,kl,k,l}^{kd,d} ] = (\bar{H}S)_{ki,kJ,i,J}^{kB,B}

    tmp_aaa = lib.einsum('xymenf,xymnf->e', imds.Wovov[:,kshift,:], r2aaa)
    tmp_bbb = lib.einsum('xyMENF,xyMNF->E', imds.WOVOV[:,kshift,:], r2bbb)
    tmp_abb = lib.einsum('xymeNF,xymNF->e', imds.WovOV[:,kshift,:], r2abb)
    tmp_baa = np.zeros(tmp_bbb.shape, dtype=tmp_bbb.dtype)
    for km, kn in itertools.product(range(nkpts), repeat=2):
        kf = kconserv[kn, kshift, km]
        tmp_baa += lib.einsum('nfME, Mnf->E', imds.WovOV[kn, kf, km], r2baa[km, kn])


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

    #idxoa = [np.where(orbspin[k][:nocca+noccb] == 0)[0] for k in range(nkpts)]
    #idxva = [np.where(orbspin[k][nocca+noccb:] == 0)[0] for k in range(nkpts)]
    #idxob = [np.where(orbspin[k][:nocca+noccb] == 1)[0] for k in range(nkpts)]
    #idxvb = [np.where(orbspin[k][nocca+noccb:] == 1)[0] for k in range(nkpts)]

    # j \/ b   |  i
    #    ---   |
    #      /\  |
    #    m \/ e|
    #     -------
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kshift, kj]
        for km in range(nkpts):
            ke = kconserv[km, kshift, ki]

            # \sum_{kL,kD,L,D} W_{kL,kD,kb,kj,L,D,b,j} S_{ki,kL,i,L}^{kb,b}
            # \sum_{kl,kd,l,d} W_{kl,kd,kb,kj,l,d,b,j} S_{ki,kl,i,l}^{kb,b}
            Hr2aaa[ki, kj] += lib.einsum('mebj,ime->ijb', imds.Wovvo[km, ke, kb],
                                         r2aaa[ki, km])
            Hr2aaa[ki, kj] += lib.einsum('MEbj,iME->ijb', imds.WOVvo[km, ke, kb],
                                         r2abb[ki, km])
            # P(ij)
            ke = kconserv[km, kshift, kj]
            Hr2aaa[ki, kj] -= lib.einsum('mebi,jme->ijb', imds.Wovvo[km, ke, kb],
                                         r2aaa[kj, km])
            Hr2aaa[ki, kj] -= lib.einsum('MEbi,jME->ijb', imds.WOVvo[km, ke, kb],
                                         r2abb[kj, km])

            # \sum_{kL,kD,L,D} W_{kL,kD,kb,kJ,L,D,b,J} S_{ki,kL,i,L}^{kD,D}
            # \sum_{kl,kd,l,d} W_{kl,kd,kB,kJ,l,d,B,J} S_{ki,kl,i,l}^{kd,d}
            ke = kconserv[km, kshift, ki]
            Hr2abb[ki, kj] += lib.einsum('meBJ,ime->iJB', imds.WovVO[km, ke, kb],
                                         r2aaa[ki, km])
            Hr2abb[ki, kj] += lib.einsum('MEBJ,iME->iJB', imds.WOVVO[km, ke, kb],
                                         r2abb[ki, km])
            ke = kconserv[km, kshift, kj]
            Hr2abb[ki, kj] -= lib.einsum('miBE,mJE->iJB', imds.WooVV[km, ki, kb],
                                         r2abb[km, kj])

            ke = kconserv[km, kshift, ki]
            Hr2baa[ki, kj] += lib.einsum('MEbj,IME->Ijb', imds.WOVvo[km, ke, kb],
                                         r2bbb[ki, km])
            Hr2baa[ki, kj] += lib.einsum('mebj,Ime->Ijb', imds.Wovvo[km, ke, kb],
                                         r2baa[ki, km])
            ke = kconserv[km, kshift, kj]
            Hr2baa[ki, kj] -= lib.einsum('MIbe,Mje->Ijb', imds.WOOvv[km, ki, kb],
                                         r2baa[km, kj])

            ke = kconserv[km, kshift, ki]
            Hr2bbb[ki, kj] += lib.einsum('MEBJ,IME->IJB', imds.WOVVO[km, ke, kb],
                                         r2bbb[ki, km])
            Hr2bbb[ki, kj] += lib.einsum('meBJ,Ime->IJB', imds.WovVO[km, ke, kb],
                                         r2baa[ki, km])
            # P(ij)
            ke = kconserv[km, kshift, kj]
            Hr2bbb[ki, kj] -= lib.einsum('MEBI,JME->IJB', imds.WOVVO[km, ke, kb],
                                         r2bbb[kj, km])
            Hr2bbb[ki, kj] -= lib.einsum('meBI,Jme->IJB', imds.WovVO[km, ke, kb],
                                         r2baa[kj, km])

    #spatial_Hr1 = [Hr1a, Hr1b]
    #spatial_Hr2 = [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb]
    #spin_Hr1, spin_Hr2 = eom_kgccsd.spatial2spin_ip_doublet(spatial_Hr1, spatial_Hr2,
#                                                            kconserv, kshift, orbspin)
    #Hr1 += spin_Hr1
    #Hr2 += spin_Hr2
    #vector = eom.amplitudes_to_vector(Hr1, Hr2)
    vector = amplitudes_to_vector_ip([Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb], kshift, kconserv)
    return vector

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc_a, nvir_a = t1a.shape
    nkpts, nocc_b, nvir_b = t1b.shape
    kconserv = imds.kconserv

    Hr1a = -np.diag(imds.Foo[kshift])
    Hr1b = -np.diag(imds.FOO[kshift])

    Hr2aaa = np.zeros((nkpts,nkpts,nocc_a,nocc_a,nvir_a), dtype=t1[0].dtype)
    Hr2bbb = np.zeros((nkpts,nkpts,nocc_b,nocc_b,nvir_b), dtype=t1[0].dtype)
    Hr2abb = np.zeros((nkpts,nkpts,nocc_a,nocc_b,nvir_b), dtype=t1[0].dtype)
    Hr2baa = np.zeros((nkpts,nkpts,nocc_b,nocc_a,nvir_a), dtype=t1[0].dtype)
    if eom.partition == 'mp':
        raise Exception("MP diag is not tested") # remove this to use untested code
        #foo = eris.fock[0][:,:nocc_a,:nocc_a]
        #fOO = eris.fock[1][:,:nocc_b,:nocc_b]
        #fvv = eris.fock[0][:,:nvir_a,:nvir_a]
        #fVV = eris.fock[1][:,:nvir_b,:nvir_b]
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kconserv[ki,kshift,kj]
                Hr2aaa[ki,kj]  = imds.Fvv[ka].diagonal()
                Hr2aaa[ki,kj] -= imds.Foo[ki].diagonal()[:,None,None]
                Hr2aaa[ki,kj] -= imds.Foo[kj].diagonal()[None,:,None]
                Hr2bbb[ki,kj]  = imds.FVV[ka].diagonal()
                Hr2bbb[ki,kj] -= imds.FOO[ki].diagonal()[:,None,None]
                Hr2bbb[ki,kj] -= imds.FOO[kj].diagonal()[None,:,None]
                Hr2aba[ki,kj]  = imds.Fvv[ka].diagonal()
                Hr2aba[ki,kj] -= imds.Foo[ki].diagonal()[:,None,None]
                Hr2aba[ki,kj] -= imds.FOO[kj].diagonal()[None,:,None]
                Hr2bab[ki,kj]  = imds.FVV[ka].diagonal()
                Hr2bab[ki,kj] -= imds.FOO[ki].diagonal()[:,None,None]
                Hr2bab[ki,kj] -= imds.Foo[kj].diagonal()[None,:,None]
    else:
        for ka in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kshift,ki,ka]
                Hr2aaa[ki,kj] += imds.Fvv[ka].diagonal()
                Hr2abb[ki,kj] += imds.FVV[ka].diagonal()
                Hr2bbb[ki,kj] += imds.FVV[ka].diagonal()
                Hr2baa[ki,kj] += imds.Fvv[ka].diagonal()

                Hr2aaa[ki,kj] -= imds.Foo[ki].diagonal()[:,None,None]
                Hr2aaa[ki,kj] -= imds.Foo[kj].diagonal()[None,:,None]
                Hr2abb[ki,kj] -= imds.Foo[ki].diagonal()[:,None,None]
                Hr2abb[ki,kj] -= imds.FOO[kj].diagonal()[None,:,None]
                Hr2baa[ki,kj] -= imds.FOO[ki].diagonal()[:,None,None]
                Hr2baa[ki,kj] -= imds.Foo[kj].diagonal()[None,:,None]
                Hr2bbb[ki,kj] -= imds.FOO[ki].diagonal()[:,None,None]
                Hr2bbb[ki,kj] -= imds.FOO[kj].diagonal()[None,:,None]

        for ki, kj in itertools.product(range(nkpts), repeat=2):
        #for ki in range(nkpts):
        #    for kj in range(nkpts):
            Hr2aaa[ki, kj] += lib.einsum('iijj->ij', imds.Woooo[ki, ki, kj])[:,:,None]
            Hr2abb[ki, kj] += lib.einsum('iiJJ->iJ', imds.WooOO[ki, ki, kj])[:,:,None]
            Hr2bbb[ki, kj] += lib.einsum('IIJJ->IJ', imds.WOOOO[ki, ki, kj])[:,:,None]
            Hr2baa[ki, kj] += lib.einsum('jjII->Ij', imds.WooOO[kj, kj, ki])[:,:,None]

            kb = kconserv[ki, kshift, kj]
            Hr2aaa[ki,kj] -= lib.einsum('iejb,jibe->ijb', imds.Wovov[ki,kshift,kj], t2aa[kj,ki,kb])
            Hr2abb[ki,kj] -= lib.einsum('ieJB,iJeB->iJB', imds.WovOV[ki,kshift,kj], t2ab[ki,kj,kshift])
            Hr2baa[ki,kj] -= lib.einsum('jbIE,jIbE->Ijb', imds.WovOV[kj,kb,ki], t2ab[kj,ki,kb])
            Hr2bbb[ki,kj] -= lib.einsum('IEJB,JIBE->IJB', imds.WOVOV[ki,kshift,kj], t2bb[kj,ki,kb])

            Hr2aaa[ki, kj] += lib.einsum('ibbi->ib', imds.Wovvo[ki, kb, kb])[:,None,:]
            Hr2aaa[ki, kj] += lib.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])[None,:,:]

            Hr2baa[ki, kj] += lib.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])[None,:,:]
            Hr2baa[ki, kj] -= lib.einsum('IIbb->Ib', imds.WOOvv[ki, ki, kb])[:,None,:]

            Hr2abb[ki, kj] += lib.einsum('JBBJ->JB', imds.WOVVO[kj, kb, kb])[None,:,:]
            Hr2abb[ki, kj] -= lib.einsum('iiBB->iB', imds.WooVV[ki, ki, kb])[:,None,:]

            Hr2bbb[ki, kj] += lib.einsum('IBBI->IB', imds.WOVVO[ki, kb, kb])[:,None,:]
            Hr2bbb[ki, kj] += lib.einsum('JBBJ->JB', imds.WOVVO[kj, kb, kb])[None,:,:]

    vector = amplitudes_to_vector_ip((Hr1a,Hr1b), (Hr2aaa,Hr2baa,Hr2abb,Hr2bbb), kshift, kconserv)
    return vector

def mask_frozen_ip(eom, vector, kshift, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    nkpts = eom.nkpts
    nocca, noccb = eom.nocc
    nmoa, nmob = eom.nmo
    kconserv = eom.kconserv

    r1, r2 = eom.vector_to_amplitudes(vector, kshift, nkpts, (nmoa, nmob), (nocca, noccb), kconserv)
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = eom.nonzero_opadding, eom.nonzero_vpadding
    nonzero_opadding_a, nonzero_opadding_b = nonzero_opadding
    nonzero_vpadding_a, nonzero_vpadding_b = nonzero_vpadding

    new_r1a = const * np.ones_like(r1a)
    new_r1b = const * np.ones_like(r1b)
    new_r2aaa = const * np.ones_like(r2aaa)
    new_r2baa = const * np.ones_like(r2baa)
    new_r2abb = const * np.ones_like(r2abb)
    new_r2bbb = const * np.ones_like(r2bbb)

    # r1a/b case
    new_r1a[nonzero_opadding_a[kshift]] = r1a[nonzero_opadding_a[kshift]]
    new_r1b[nonzero_opadding_b[kshift]] = r1b[nonzero_opadding_b[kshift]]

    # r2aaa case
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, kshift, kj]
            idx = np.ix_([ki], [kj], nonzero_opadding_a[ki], nonzero_opadding_a[kj], nonzero_vpadding_a[kb])
            new_r2aaa[idx] = r2aaa[idx]

    # r2baa case
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, kshift, kj]
            idx = np.ix_([ki], [kj], nonzero_opadding_b[ki], nonzero_opadding_a[kj], nonzero_vpadding_a[kb])
            new_r2baa[idx] = r2baa[idx]

    # r2abb case
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, kshift, kj]
            idx = np.ix_([ki], [kj], nonzero_opadding_a[ki], nonzero_opadding_b[kj], nonzero_vpadding_b[kb])
            new_r2abb[idx] = r2abb[idx]

    # r2bbb case
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, kshift, kj]
            idx = np.ix_([ki], [kj], nonzero_opadding_b[ki], nonzero_opadding_b[kj], nonzero_vpadding_b[kb])
            new_r2bbb[idx] = r2bbb[idx]

    return eom.amplitudes_to_vector((new_r1a,new_r1b), (new_r2aaa,new_r2baa,new_r2abb,new_r2bbb), kshift, kconserv)


def get_padding_k_idx(eom, cc):
    # Get location of padded elements in occupied and virtual space
    nonzero_padding_alpha, nonzero_padding_beta = padding_k_idx(cc, kind="split")
    nonzero_opadding_alpha, nonzero_vpadding_alpha = nonzero_padding_alpha
    nonzero_opadding_beta, nonzero_vpadding_beta = nonzero_padding_beta
    return ((nonzero_opadding_alpha, nonzero_opadding_beta),
            (nonzero_vpadding_alpha, nonzero_vpadding_beta))


class EOMIP(eom_kgccsd.EOMIP):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMIP.__init__(self, cc)

    get_diag = ipccsd_diag
    matvec = ipccsd_matvec
    get_padding_k_idx = get_padding_k_idx
    mask_frozen = mask_frozen_ip

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        nocca, noccb = self.nocc
        guess = []
        if koopmans:
            idx = np.zeros(nroots, dtype=np.int)
            tmp_oalpha, tmp_obeta = self.nonzero_opadding[kshift]
            tmp_oalpha = list(tmp_oalpha)
            tmp_obeta = list(tmp_obeta)
            if len(tmp_obeta) + len(tmp_oalpha) < nroots:
                raise ValueError("Max number of roots for k-point (idx=%3d) for koopmans "
                                 "is %3d.\nRequested %3d." %
                                 (kshift, len(tmp_obeta)+len(tmp_oalpha), nroots))

            total_count = 0
            while(total_count < nroots):
                if total_count % 2 == 0 and len(tmp_oalpha) > 0:
                    idx[total_count] = tmp_oalpha.pop()
                else:
                    # Careful! index depends on how we create vector
                    # (here the first elements are r1a, then r1b)
                    idx[total_count] = nocca + tmp_obeta.pop()
                total_count += 1
        else:
            idx = diag.argsort()

        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            g = self.mask_frozen(g, kshift, const=0.0)
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

    def vector_to_amplitudes(self, vector, kshift, nkpts=None, nmo=None, nocc=None, kconserv=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        if kconserv is None: kconserv = self.kconserv
        return vector_to_amplitudes_ip(vector, kshift, nkpts, nmo, nocc, kconserv)

    def amplitudes_to_vector(self, r1, r2, kshift, kconserv=None):
        if kconserv is None: kconserv = self.kconserv
        return amplitudes_to_vector_ip(r1, r2, kshift, kconserv)

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        nkpts = self.nkpts
        return nocca + noccb + nkpts*nocca*(nkpts*nocca-1)*nvira//2 + nkpts**2*noccb*nocca*nvira + nkpts**2*nocca*noccb*nvirb + nkpts*noccb*(nkpts*noccb-1)*nvirb//2

    def make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ip()
        return imds


########################################
# EOM-EA-CCSD
########################################

def amplitudes_to_vector_ea(r1, r2, kshift, kconserv):
    r1a, r1b = r1
    r2a, r2aba, r2bab, r2b = r2
    nkpts = r2a.shape[0]
    nocca, noccb = r2a.shape[2], r2b.shape[2]
    nvira, nvirb = r2a.shape[3], r2b.shape[3]
    # From symmetry for aaa and bbb terms, only store lower
    # triangular part (ka,a) < (kb,b)
    r2aaa = np.zeros((nocca*nkpts*nvira*(nkpts*nvira-1))//2, dtype=r2a.dtype)
    r2bbb = np.zeros((noccb*nkpts*nvirb*(nkpts*nvirb-1))//2, dtype=r2b.dtype)

    index = 0
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        if ka < kb:  # Take diagonal part
            idxa, idya = np.tril_indices(nvira, 0)
        else:  # Don't take diagonal (equal to zero)
            idxa, idya = np.tril_indices(nvira, -1)
        r2aaa[index:index + nocca*len(idya)] = r2a[kj,ka,:,idxa,idya].reshape(-1)
        index = index + nocca*len(idya)

    index = 0
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        if ka < kb:  # Take diagonal part
            idxb, idyb = np.tril_indices(nvirb, 0)
        else:
            idxb, idyb = np.tril_indices(nvirb, -1)
        r2bbb[index:index + noccb*len(idyb)] = r2b[kj,ka,:,idxb,idyb].reshape(-1)
        index = index + noccb*len(idyb)

    return np.hstack((r1a, r1b, r2aaa.ravel(),
                      r2aba.ravel(), r2bab.ravel(),
                      r2bbb.ravel()))

def vector_to_amplitudes_ea(vector, kshift, nkpts, nmo, nocc, kconserv):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nvira, nvirb, nkpts*nocca*(nkpts*nvira-1)*nvira//2,
             nkpts**2*nocca*nvirb*nvira, nkpts**2*noccb*nvira*nvirb,
             nkpts*noccb*(nkpts*nvirb-1)*nvirb//2)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2a, r2aba, r2bab, r2b = np.split(vector, sections)

    r2aaa = np.zeros((nkpts,nkpts,nocca,nvira,nvira), dtype=r2a.dtype)
    r2aba = r2aba.reshape(nkpts,nkpts,nocca,nvirb,nvira).copy()
    r2bab = r2bab.reshape(nkpts,nkpts,noccb,nvira,nvirb).copy()
    r2bbb = np.zeros((nkpts,nkpts,noccb,nvirb,nvirb), dtype=r2b.dtype)

    index = 0
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        if ka < kb:  # Take diagonal part
            idxa, idya = np.tril_indices(nvira, 0)
        else:
            idxa, idya = np.tril_indices(nvira, -1)
        tmp = r2a[index:index + nocca*len(idya)].reshape(-1,nocca)
        r2aaa[kj,ka,:,idxa,idya] = tmp
        r2aaa[kj,kb,:,idya,idxa] = -tmp
        index = index + nocca*len(idya)

    index = 0
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        if ka < kb:  # Take diagonal part
            idxb, idyb = np.tril_indices(nvirb, 0)
        else:
            idxb, idyb = np.tril_indices(nvirb, -1)
        tmp = r2b[index:index + noccb*len(idyb)].reshape(-1,noccb)
        r2bbb[kj,ka,:,idxb,idyb] = tmp
        r2bbb[kj,kb,:,idyb,idxb] = -tmp
        index = index + noccb*len(idyb)

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2aba, r2bab, r2bbb)
    return r1, r2

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ j}^{ab}, i.e. 'jb' indices are coupled'''
    if imds is None: imds = eom.make_imds()
    t1, t2= imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape[3:]
    nmoa, nmob = nocca + nvira, noccb + nvirb
    kconserv = imds.kconserv
    nkpts = eom.nkpts

    r1, r2 = eom.vector_to_amplitudes(vector, kshift, nkpts, (nmoa, nmob), (nocca, noccb), kconserv)

    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2


    # BEGINNING OF MATVEC CONTRACTIONS: ref - Nooijen 1995 EOM-CC for EA

    # Fvv terms
    # (\bar{H}S)^a = \sum_{kc,c} U_{ac} s^c
    Hr1a = einsum('ac,c->a', imds.Fvv[kshift], r1a)
    Hr1b = einsum('AC,C->A', imds.FVV[kshift], r1b)

    # Fov terms
    # (\bar{H}S)^a = \sum_{kL,kD, L, D} U_{kL,kD,L,D} s^{a,kD,D}_{kL,L} + \sum_{kl,kd,l,d} U_{kl, d}^{a,kd,d}_{kl,l}
    for kl in range(nkpts):
        Hr1a += einsum('ld,lad->a', imds.Fov[kl], r2aaa[kl,kshift])
        Hr1a += einsum('LD,LaD->a', imds.FOV[kl], r2bab[kl,kshift])
        Hr1b += einsum('ld,lAd->A', imds.Fov[kl], r2aba[kl,kshift])
        Hr1b += einsum('LD,LAD->A', imds.FOV[kl], r2bbb[kl,kshift])

    # Wvovv
    # (\bar{H}S)^a = \sum_{kc,kL,kD,c,L,D} W_{kL,kc,kD,a,l,c,D} s_{kL,L}^{kc,kD,c,D}
    # + \sum_{kc,kd,kl,c,d,l} W_{ka,kl,kc,kd,a,l,c,d} s_{kl,l}^{kc,kd,c,d}
    for kc, kl in itertools.product(range(nkpts), repeat=2):
        Hr1a += 0.5*lib.einsum('acld,lcd->a', imds.Wvvov[kshift,kc,kl], r2aaa[kl,kc])
        Hr1a +=     lib.einsum('acLD,LcD->a', imds.WvvOV[kshift,kc,kl], r2bab[kl,kc])
        Hr1b += 0.5*lib.einsum('ACLD,LCD->A', imds.WVVOV[kshift,kc,kl], r2bbb[kl,kc])
        Hr1b +=     lib.einsum('ACld,lCd->A', imds.WVVov[kshift,kc,kl], r2aba[kl,kc])

    dtype = np.result_type(Hr1a, *r2)
    Hr2aaa = np.zeros((nkpts, nkpts, nocca, nvira, nvira), dtype=dtype)
    Hr2aba = np.zeros((nkpts, nkpts, nocca, nvirb, nvira), dtype=dtype)
    Hr2bab = np.zeros((nkpts, nkpts, noccb, nvira, nvirb), dtype=dtype)
    Hr2bbb = np.zeros((nkpts, nkpts, noccb, nvirb, nvirb), dtype=dtype)

    # Wvvvv
    # \sum_{kc,kd,c,d} W_{ka,kb,kc,kd,a,b,c,d} s_{kj,j}^{kc,kd,c,d} = (\bar{H}S)^{kb, a, b}_{kj,j}
    # \sum_{kc,kD,c,D} W{ka,kB,kc,kD,a,B,c,D} s_{kJ,kc,kD,J,c,D} = (\bar{H}S)^{kB, a, B}_{kJ,J}
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift,ka,kj]
        for kc in range(nkpts):
            kd = kconserv[ka, kc, kb]
            Wvvvv, WvvVV, WVVVV = imds.get_Wvvvv(ka, kb, kc)
            Hr2aaa[kj,ka] += .5 * lib.einsum('acbd,jcd->jab', Wvvvv, r2aaa[kj,kc])
            Hr2aba[kj,kb] +=      lib.einsum('bcad,jdc->jab', WvvVV, r2aba[kj,kd])
            Hr2bab[kj,ka] +=      lib.einsum('acbd,jcd->jab', WvvVV, r2bab[kj,kc])
            Hr2bbb[kj,ka] += .5 * lib.einsum('acbd,jcd->jab', WVVVV, r2bbb[kj,kc])

    #Wvvvo
    # \sum_{kc,ka,kj,c,a,j} W_{kb,kc,kj,a,b,c,j} s^{kc,c} = (\bar{H}S)^{kb, a, b}_{kj,j}
    # \sum_{kc,ka,kJ,c,a,J} W_{kB,kc,kJ,a,B,c,J} s^{kc,c} = (\bar{H}S)^{kB, a, B}_{kJ,J}
    for ka, kj, in itertools.product(range(nkpts),repeat=2):
        kb = kconserv[kshift,ka,kj]
        kc = kshift
        Hr2aaa[kj,ka] += einsum('acbj,c->jab', imds.Wvvvo[ka,kc,kb], r1a)
        Hr2bbb[kj,ka] += einsum('ACBJ,C->JAB', imds.WVVVO[ka,kc,kb], r1b)

        Hr2bab[kj,ka] += einsum('acBJ,c->JaB', imds.WvvVO[ka,kc,kb], r1a)
        Hr2aba[kj,ka] += einsum('ACbj,C->jAb', imds.WVVvo[ka,kc,kb], r1b)

    #Fvv Terms
    # sum_{kc,ka,kj,c,a,j} s_{kj,j}^{kc,kb,c,b} U_{ka,kc,a,c} = (\bar{H}S)^{kb, a, b}_{kj,j}
    # sum_{kd,ka,kj,d,b,j} s_{kj,j}^{ka,kd,a,d} U_{kb,kd,b,d} = (\bar{H}S)^{kb, a, b}_{kj,j}

    # sum_{kc,ka,kJ,c,a,J} U_{ka,kc,a,c} s_{kJ,J}^{kc,kB,c,B} = (\bar{H}S)^{kB, a, B}_{kJ,J}
    # sum_{kD,ka,kj,D,a,j} U_{kb,kd,b,d} s_{kj,j}^{ka,kd,a,d} = (\bar{H}S)^{kB, a, B}_{kJ,J}
    for ka, kj in itertools.product(range(nkpts), repeat=2):
        # kb = kshift - ka + kj
        kb = kconserv[kshift, ka, kj]
        tmpa = lib.einsum('ac,jcb->jab', imds.Fvv[ka], r2aaa[kj,ka])
        tmpb = lib.einsum('bc,jca->jab', imds.Fvv[kb], r2aaa[kj,kb])
        Hr2aaa[kj,ka] += tmpa - tmpb
        Hr2aba[kj,ka] += lib.einsum('AC,jCb->jAb', imds.FVV[ka], r2aba[kj,ka])
        Hr2bab[kj,ka] += lib.einsum('ac,JcB->JaB', imds.Fvv[ka], r2bab[kj,ka])
        Hr2aba[kj,ka] += lib.einsum('bc, jAc -> jAb', imds.Fvv[kb], r2aba[kj,ka])
        Hr2bab[kj,ka] += lib.einsum('BC, JaC -> JaB', imds.FVV[kb], r2bab[kj,ka])
        tmpb = lib.einsum('AC,JCB->JAB', imds.FVV[ka], r2bbb[kj,ka])
        tmpa = lib.einsum('BC,JCA->JAB', imds.FVV[kb], r2bbb[kj,kb])
        Hr2bbb[kj,ka] += tmpb - tmpa

    #Foo Term
    # \sum_{ka,kl,l} U_{kl,l,kj,j} s^{ka,a,kb,b}^{kl,l} = (\bar{H}S)^{kb, a, b}_{kj,j}
    # \sum_{ka,kL,L} s^{ka,a,kB,B}_{kL,L} U_{kL,L,kJ,J} = (\bar{H}S)^{kB, a, B}_{kJ,J}
    for kl, ka in itertools.product(range(nkpts), repeat=2):
        Hr2aaa[kl,ka] -= lib.einsum('lj,lab->jab', imds.Foo[kl], r2aaa[kl,ka])
        Hr2bbb[kl,ka] -= lib.einsum('LJ,LAB->JAB', imds.FOO[kl], r2bbb[kl,ka])
        Hr2bab[kl,ka] -= lib.einsum('LJ,LaB->JaB', imds.FOO[kl], r2bab[kl,ka])
        Hr2aba[kl,ka] -= lib.einsum('lj,lAb->jAb', imds.Foo[kl], r2aba[kl,ka])

    # Woovv term
    # - \sum{kk,k} t_{kk,kj,k,j}^{ka,kb,a,b} [\sum_{kc,kD,kL,c,D,L} W_{kL,kk,kD,kc,L,k,D,c} s_{kL,L}^{kc,kD,c,D}
    # + \sum{kc,kd,kl,c,d,l} W_{kk,kl,kc,kd,k,l,c,d} s_{kl,l}^{kc,kd,c,d} ] = (\bar{H}S)^{kb, a, b}_{kj,j}
    #
    # - \sum_{kk,k} t_{kk,kJ,k,J}^{ka,kB,a,B} [ \sum{kc,kD,kL,c,D,L} W_{kk,kL,kc,kD,k,L,c,D} s_{kL,L}^{kc,kD,c,D}
    # + \sum_{kc,kd,kl,c,d,l} W_{kk,kl,kc,kd,k,l,c,d} s_{kl,l}^{kc,kd,c,d} ] = (\bar{H}S)^{kB, a, B}_{kJ,J}
    tmp_aaa = lib.einsum('xykcld, yxlcd->k', imds.Wovov[kshift,:,:], r2aaa)
    tmp_bbb = lib.einsum('xyKCLD, yxLCD->K', imds.WOVOV[kshift,:,:], r2bbb)
    tmp_bab = lib.einsum('xykcLD, yxLcD->k', imds.WovOV[kshift], r2bab)
    tmp_aba = np.zeros(tmp_bbb.shape, dtype = tmp_bbb.dtype)

    for kl, kc in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kl,kc,kshift]
        tmp_aba += lib.einsum('ldKC, lCd->K', imds.WovOV[kl,kd,kshift], r2aba[kl,kc])

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

    # j \/ b   |  a
    #    ---   |
    #      /\  |
    #    l \/ d|
    #     -------
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kshift, ka, kj]
        for kd in range(nkpts):
            kl = kconserv[ka, kshift, kd]

            # \sum_{kL,kD,L,D} W_{kL,kb,kD,kj,L,b,D,j} s_{kL,L}^{ka,kD,a,D} = (\bar{H}S)^{kb, a, b}_{kj,j}
            # \sum_{kl,kd,l,d} W_{kl,kb,kd,kj,l,b,d,j} s_{kl,l}^{ka,kd,a,d} = (\bar{H}S)^{kb, a, b}_{kj,j}
            Hr2aaa[kj, ka] += lib.einsum('ldbj,lad->jab', imds.Wovvo[kl,kd,kb],
                                         r2aaa[kl,ka])
            Hr2aaa[kj, ka] += lib.einsum('LDbj,LaD->jab', imds.WOVvo[kl,kd,kb],
                                         r2bab[kl,ka])
            # P(ab)
            kl = kconserv[kb, kshift, kd]
            Hr2aaa[kj, ka] -= lib.einsum('ldaj,lbd->jab', imds.Wovvo[kl,kd,ka],
                                         r2aaa[kl,kb])
            Hr2aaa[kj, ka] -= lib.einsum('LDaj,LbD->jab', imds.WOVvo[kl,kd,ka],
                                         r2bab[kl,kb])

            kl = kconserv[ka, kshift, kd]

            # \sum_{kL,kD,L,D} W_{kL,kB,kD,kJ,L,B,D,J} s_{kL,L}^{ka,kD,a,D} = (\bar{H}S)^{kB, a, B}_{kJ,J}
            # \sum_{kl,kd,l,d} W_{kl,kB,kd,kJ,l,B,d,J} s_{kl,l}^{ka,kd,a,d} = (\bar{H}S)^{kB, a, B}_{kJ,J}
            # - \sum_{kc,kL,c,L} W_{ka,kL,kc,kJ,a,L,c,J} s_{kL,L}^{kc,kB,c,B} = (\bar{H}S)^{kB, a, B}_{kJ,J}
            Hr2bab[kj, ka] += lib.einsum('ldBJ,lad->JaB', imds.WovVO[kl,kd,kb],
                                         r2aaa[kl,ka])
            Hr2bab[kj, ka] += lib.einsum('LDBJ,LaD->JaB', imds.WOVVO[kl,kd,kb],
                                         r2bab[kl,ka])
            kl = kconserv[kb, kshift, kd]
            Hr2bab[kj, ka] -= lib.einsum('LJad,LdB->JaB', imds.WOOvv[kl,kj,ka],
                                         r2bab[kl,kd])

            kl = kconserv[ka, kshift, kd]
            Hr2aba[kj, ka] += lib.einsum('LDbj,LAD->jAb', imds.WOVvo[kl,kd,kb],
                                         r2bbb[kl,ka])
            Hr2aba[kj, ka] += lib.einsum('ldbj,lAd->jAb', imds.Wovvo[kl,kd,kb],
                                         r2aba[kl,ka])
            kl = kconserv[kb, kshift, kd]
            Hr2aba[kj, ka] -= lib.einsum('ljAD,lDb->jAb', imds.WooVV[kl,kj,ka],
                                         r2aba[kl,kd])

            kl = kconserv[ka, kshift, kd]
            Hr2bbb[kj, ka] += lib.einsum('LDBJ,LAD->JAB', imds.WOVVO[kl,kd,kb],
                                         r2bbb[kl,ka])
            Hr2bbb[kj, ka] += lib.einsum('ldBJ,lAd->JAB', imds.WovVO[kl,kd,kb],
                                         r2aba[kl,ka])
            # P(ab)
            kl = kconserv[kb, kshift, kd]
            Hr2bbb[kj, ka] -= lib.einsum('LDAJ,LBD->JAB', imds.WOVVO[kl,kd,ka],
                                         r2bbb[kl,kb])
            Hr2bbb[kj, ka] -= lib.einsum('ldAJ,lBd->JAB', imds.WovVO[kl,kd,ka],
                                         r2aba[kl,kb])

    vector = amplitudes_to_vector_ea([Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb], kshift, kconserv)
    return vector

def eaccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    nkpts, noccb, nvirb = t1b.shape
    kconserv = imds.kconserv

    #Hr1a = np.zeros((nvira), dtype=t1a.dtype)
    #Hr1b = np.zeros((nvirb), dtype=t1b.dtype)
    Hr2aaa = np.zeros((nkpts,nkpts,nocca,nvira,nvira), dtype=t1a.dtype)
    Hr2aba = np.zeros((nkpts,nkpts,nocca,nvirb,nvira), dtype=t1a.dtype)
    Hr2bab = np.zeros((nkpts,nkpts,noccb,nvira,nvirb), dtype=t1a.dtype)
    Hr2bbb = np.zeros((nkpts,nkpts,noccb,nvirb,nvirb), dtype=t1b.dtype)

    Hr1a = np.diag(imds.Fvv[kshift])
    Hr1b = np.diag(imds.FVV[kshift])
    if eom.partition == 'mp':
        raise Exception("MP diag is not tested") # remove this to use untested code
        for kj, ka in itertools.product(range(nkpts), repeat=2):
            kb = kconserv[kshift, ka, kj]
            Hr2aaa[kj,ka] -= imds.Foo[kj,:,None,None]
            Hr2aaa[kj,ka] += imds.Fvv[ka,None,:,None]
            Hr2aaa[kj,ka] += imds.Fvv[kb,None,None,:]

            Hr2aba[kj,ka] -= imds.Foo[kj,:,None,None]
            Hr2aba[kj,ka] += imds.FVV[ka,None,:,None]
            Hr2aba[kj,ka] += imds.Fvv[kb,None,None,:]

            Hr2bab[kj,ka] -= imds.FOO[kj,:,None,None]
            Hr2bab[kj,ka] += imds.Fvv[ka,None,:,None]
            Hr2bab[kj,ka] += imds.FVV[kb,None,None,:]

            Hr2bbb[kj,ka] -= imds.FOO[kj,:,None,None]
            Hr2bbb[kj,ka] += imds.FVV[ka,None,:,None]
            Hr2bbb[kj,ka] += imds.FVV[kb,None,None,:]
    else:
        for kj, ka in itertools.product(range(nkpts), repeat=2):
            kb = kconserv[kshift, ka, kj]
            # Fvv
            Hr2aaa[kj,ka] += imds.Fvv[ka].diagonal()[None,:,None]
            Hr2aaa[kj,ka] += imds.Fvv[kb].diagonal()[None,None,:]
            Hr2aba[kj,ka] += imds.FVV[ka].diagonal()[None,:,None]
            Hr2aba[kj,ka] += imds.Fvv[kb].diagonal()[None,None,:]
            Hr2bab[kj,ka] += imds.Fvv[ka].diagonal()[None,:,None]
            Hr2bab[kj,ka] += imds.FVV[kb].diagonal()[None,None,:]
            Hr2bbb[kj,ka] += imds.FVV[ka].diagonal()[None,:,None]
            Hr2bbb[kj,ka] += imds.FVV[kb].diagonal()[None,None,:]

            # Foo
            Hr2aaa[kj,ka] -= imds.Foo[kj].diagonal()[:,None,None]
            Hr2bbb[kj,ka] -= imds.FOO[kj].diagonal()[:,None,None]
            Hr2bab[kj,ka] -= imds.FOO[kj].diagonal()[:,None,None]
            Hr2aba[kj,ka] -= imds.Foo[kj].diagonal()[:,None,None]

            # Wvvvv
            Wvvvv, WvvVV, WVVVV = imds.get_Wvvvv(ka, kb, ka)
            # FIXME: Do Wvvvv and WVVVV have a factor 0.5?
            Hr2aaa[kj,ka] += lib.einsum('aabb->ab', Wvvvv)[None,:,:]
            Hr2aba[kj,kb] += lib.einsum('bbAA->Ab', WvvVV)[None,:,:]
            Hr2bab[kj,ka] += lib.einsum('aaBB->aB', WvvVV)[None,:,:]
            Hr2bbb[kj,ka] += lib.einsum('AABB->AB', WVVVV)[None,:,:]

            # Wovov term (physicist's Woovv)
            Hr2aaa[kj,ka] -= lib.einsum('kajb, kjab->jab', imds.Wovov[kshift,ka,kj], t2aa[kshift,kj,ka])
            Hr2aba[kj,ka] -= lib.einsum('jbKA, jKbA->jAb', imds.WovOV[kj,kb,kshift], t2ab[kj,kshift,kb])
            Hr2bab[kj,ka] -= lib.einsum('kaJB, kJaB->JaB', imds.WovOV[kshift,ka,kj], t2ab[kshift,kj,ka])
            Hr2bbb[kj,ka] -= lib.einsum('kajb, kjab->jab', imds.WOVOV[kshift,ka,kj], t2bb[kshift,kj,ka])

            # Wovvo term
            Hr2aaa[kj, ka] += lib.einsum('jbbj->jb', imds.Wovvo[kj,kb,kb])[:,None,:]
            Hr2aaa[kj, ka] += lib.einsum('jaaj->ja', imds.Wovvo[kj,ka,ka])[:,:,None]

            Hr2aba[kj, ka] += lib.einsum('jbbj->jb', imds.Wovvo[kj,kb,kb])[:,None,:]
            Hr2aba[kj, ka] -= lib.einsum('jjAA->jA', imds.WooVV[kj,kj,ka])[:,:,None]

            Hr2bab[kj, ka] += lib.einsum('JBBJ->JB', imds.WOVVO[kj,kb,kb])[:,None,:]
            Hr2bab[kj, ka] -= lib.einsum('JJaa->Ja', imds.WOOvv[kj,kj,ka])[:,:,None]

            Hr2bbb[kj, ka] += lib.einsum('JBBJ->JB', imds.WOVVO[kj,kb,kb])[:,None,:]
            Hr2bbb[kj, ka] += lib.einsum('JAAJ->JA', imds.WOVVO[kj,ka,ka])[:,:,None]

    vector = amplitudes_to_vector_ea([Hr1a,Hr1b], [Hr2aaa,Hr2aba,Hr2bab,Hr2bbb], kshift, kconserv)
    return vector

def mask_frozen_ea(eom, vector, kshift, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    nkpts = eom.nkpts
    nocca, noccb = eom.nocc
    nmoa, nmob = eom.nmo
    kconserv = eom.kconserv

    r1, r2 = eom.vector_to_amplitudes(vector, kshift, nkpts, (nmoa, nmob), (nocca, noccb), kconserv)
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = eom.nonzero_opadding, eom.nonzero_vpadding
    nonzero_opadding_a, nonzero_opadding_b = nonzero_opadding
    nonzero_vpadding_a, nonzero_vpadding_b = nonzero_vpadding

    new_r1a = const * np.ones_like(r1a)
    new_r1b = const * np.ones_like(r1b)
    new_r2aaa = const * np.ones_like(r2aaa)
    new_r2aba = const * np.ones_like(r2aba)
    new_r2bab = const * np.ones_like(r2bab)
    new_r2bbb = const * np.ones_like(r2bbb)

    # r1a/b case
    new_r1a[nonzero_vpadding_a[kshift]] = r1a[nonzero_vpadding_a[kshift]]
    new_r1b[nonzero_vpadding_b[kshift]] = r1b[nonzero_vpadding_b[kshift]]

    # r2aaa case
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding_a[kj], nonzero_vpadding_a[ka], nonzero_vpadding_a[kb])
            new_r2aaa[idx] = r2aaa[idx]

    # r2aba case
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding_a[kj], nonzero_vpadding_b[ka], nonzero_vpadding_a[kb])
            new_r2aba[idx] = r2aba[idx]

    # r2bab case
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding_b[kj], nonzero_vpadding_a[ka], nonzero_vpadding_b[kb])
            new_r2bab[idx] = r2bab[idx]

    # r2bbb case
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            idx = np.ix_([kj], [ka], nonzero_opadding_b[kj], nonzero_vpadding_b[ka], nonzero_vpadding_b[kb])
            new_r2bbb[idx] = r2bbb[idx]

    return eom.amplitudes_to_vector((new_r1a,new_r1b), (new_r2aaa,new_r2aba,new_r2bab,new_r2bbb), kshift)

class EOMEA(eom_kgccsd.EOMEA):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMEA.__init__(self, cc)

    get_diag = eaccsd_diag
    matvec = eaccsd_matvec
    get_padding_k_idx = get_padding_k_idx
    mask_frozen = mask_frozen_ea

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.complex)
        nroots = min(nroots, size)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        guess = []
        if koopmans:
            idx = np.zeros(nroots, dtype=np.int)
            tmp_valpha, tmp_vbeta = self.nonzero_vpadding[kshift]
            tmp_valpha = list(tmp_valpha)
            tmp_vbeta = list(tmp_vbeta)
            if len(tmp_vbeta) + len(tmp_valpha) < nroots:
                raise ValueError("Max number of roots for k-point (idx=%3d) for koopmans "
                                 "is %3d.\nRequested %3d." %
                                 (kshift, len(tmp_vbeta)+len(tmp_valpha), nroots))

            total_count = 0
            while(total_count < nroots):
                if total_count % 2 == 0 and len(tmp_valpha) > 0:
                    idx[total_count] = tmp_valpha.pop(0)
                else:
                    # Careful! index depends on how we create vector
                    # (here the first elements are r1a, then r1b)
                    idx[total_count] = nvira + tmp_vbeta.pop(0)
                total_count += 1
        else:
            idx = diag.argsort()

        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            g = self.mask_frozen(g, kshift, const=0.0)
            guess.append(g)
        return guess

    def vector_to_amplitudes(self, vector, kshift, nkpts=None, nmo=None, nocc=None, kconserv=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        if kconserv is None: kconserv = self.kconserv
        return vector_to_amplitudes_ea(vector, kshift, nkpts, nmo, nocc, kconserv)

    def amplitudes_to_vector(self, r1, r2, kshift, kconserv=None):
        if kconserv is None: kconserv = self.kconserv
        return amplitudes_to_vector_ea(r1, r2, kshift, kconserv)

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        nkpts = self.nkpts
        #return nvira + nvirb + nocca*nkpts*nvira*nkpts*nvira + nkpts**2*nocca*nvirb*nvira + nkpts**2*noccb*nvira*nvirb + noccb*nkpts*nvirb*nkpts*nvirb
        return nvira + nvirb + nocca*nkpts*nvira*(nkpts*nvira-1)//2 + nkpts**2*nocca*nvirb*nvira + nkpts**2*noccb*nvira*nvirb + noccb*nkpts*nvirb*(nkpts*nvirb-1)//2

    def make_imds(self, eris=None, t1=None, t2=None):
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

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo, self.FOO = kintermediates_uhf.Foo(self._cc, t1, t2, eris)
        self.Fvv, self.FVV = kintermediates_uhf.Fvv(self._cc, t1, t2, eris)
        self.Fov, self.FOV = kintermediates_uhf.Fov(self._cc, t1, t2, eris)

        # 2 virtuals
        self.Wovvo, self.WovVO, self.WOVvo, self.WOVVO = kintermediates_uhf.Wovvo(self._cc, t1, t2, eris)
        self.Woovv, self.WooVV, self.WOOvv, self.WOOVV = kintermediates_uhf.Woovv(self._cc, t1, t2, eris)
        self.Wovov = eris.ovov - np.asarray(eris.ovov).transpose(2,1,0,5,4,3,6)
        self.WOVOV = eris.OVOV - np.asarray(eris.OVOV).transpose(2,1,0,5,4,3,6)
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
        if eris.vvvv is not None:
            self.Wvvvv, self.WvvVV, self.WVVVV = Wvvvv = kintermediates_uhf.Wvvvv(self._cc, t1, t2, eris)
        else:
            self.Wvvvv = self.WvvVV = self.WVVVV = None
        self.Wvvvo, self.WvvVO, self.WVVvo, self.WVVVO = kintermediates_uhf.Wvvvo(self._cc, t1, t2, eris)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-KUCCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        raise NotImplementedError

    def get_Wvvvv(self, ka, kb, kc):
        if not self.made_ea_imds:
            self.make_ea()

        if self.Wvvvv is None:
            return kintermediates_uhf.get_Wvvvv(self._cc, self.t1, self.t2, self.eris,
                                                ka, kb, kc)
        else:
            return self.Wvvvv[ka,kc,kb], self.WvvVV[ka,kc,kb], self.WVVVV[ka,kc,kb]

if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf
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

    kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(kmf))
    kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
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
    imds = myeom.make_imds()
    imds.make_ip()

    spin_r1_ip = (np.random.rand(nocc)*1j +
                  np.random.rand(nocc) - 0.5 - 0.5*1j)
    spin_r2_ip = (np.random.rand(nkpts**2 * nocc**2 * nvir) +
                  np.random.rand(nkpts**2 * nocc**2 * nvir)*1j - 0.5 - 0.5*1j)
    spin_r2_ip = spin_r2_ip.reshape(nkpts, nkpts, nocc, nocc, nvir)
    spin_r2_ip = eom_kgccsd.enforce_2p_spin_ip_doublet(spin_r2_ip, kconserv, kshift, orbspin)

    r1, r2 = eom_kgccsd.spin2spatial_ip_doublet(spin_r1_ip, spin_r2_ip, kconserv, kshift, orbspin)
    vector = myeom.amplitudes_to_vector(r1, r2, kshift)
    vector = myeom.matvec(vector, kshift=kshift, imds=imds)
    Hr1, Hr2 = myeom.vector_to_amplitudes(vector, nkpts, (nmoa, nmob), (nocca, noccb))
    Hr1a, Hr1b = Hr1
    Hr2aaa, Hr2baa, Hr2abb, Hr2bbb = Hr2
    print('ip Hr1a',   abs(lib.finger(Hr1a)   - (-0.34462696543560045-1.6104596956729178j)))
    print('ip Hr1b',   abs(lib.finger(Hr1b)   - (-0.055793611517250929+0.22169994342782473j)))
    print('ip Hr2aaa', abs(lib.finger(Hr2aaa) - (0.692705827672665420-1.958639508839846943j)))
    print('ip Hr2baa', abs(lib.finger(Hr2baa) - (2.892194153603884654+2.039530776282815872j)))
    print('ip Hr2abb', abs(lib.finger(Hr2abb) - (1.618257685489421727-5.489218743953674817j)))
    print('ip Hr2bbb', abs(lib.finger(Hr2bbb) - (0.479835513829048044+0.108406393138471210j)))
    # EA version

    myeom = EOMEA(mycc)
    imds = myeom.make_imds()
    imds.make_ea()

    spin_r1_ea = (np.random.rand(nvir)*1j +
                  np.random.rand(nvir) - 0.5 - 0.5*1j)
    spin_r2_ea = (np.random.rand(nkpts**2 * nocc * nvir**2) +
                  np.random.rand(nkpts**2 * nocc * nvir**2)*1j - 0.5 - 0.5*1j)
    spin_r2_ea = spin_r2_ea.reshape(nkpts, nkpts, nocc, nvir, nvir)
    spin_r2_ea = eom_kgccsd.enforce_2p_spin_ea_doublet(spin_r2_ea, kconserv, kshift, orbspin)
    r1, r2 = eom_kgccsd.spin2spatial_ea_doublet(spin_r1_ea, spin_r2_ea, kconserv, kshift, orbspin)

    vector = myeom.amplitudes_to_vector(r1, r2, kshift)
    vector = myeom.matvec(vector, kshift=kshift, imds=imds)
    Hr1, Hr2 = myeom.vector_to_amplitudes(vector, nkpts, (nmoa, nmob), (nocca, noccb))
    Hr1a, Hr1b = Hr1
    Hr2aaa, Hr2aba, Hr2bab, Hr2bbb = Hr2

    print('ea Hr1a',  abs(lib.finger(Hr1a)   - (-0.081373075311041126-0.51422895644026023j)))
    print('ea Hr1b',  abs(lib.finger(Hr1b)   - (-0.39518588661294807-1.3063424820239824j))  )
    print('ea Hr2aaa',abs(lib.finger(Hr2aaa) - (-2.6502079691200251-0.61302655915003545j))  )
    print('ea Hr2aba',abs(lib.finger(Hr2aba) - (5.5723208649566036-5.4202659143496286j))    )
    print('ea Hr2bab',abs(lib.finger(Hr2bab) - (-1.2822293707887937+0.3026476580141586j))   )
    print('ea Hr2bbb',abs(lib.finger(Hr2bbb) - (-4.0202809577487253-0.46985725132191702j))  )
