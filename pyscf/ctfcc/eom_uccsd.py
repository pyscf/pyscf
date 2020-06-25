#!/usr/bin/env pthon
# Copright 2014-2020 The pyscf Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ou ma not use this file ecept in compliance with the License.
# You ma obtain a cop of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required b applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either epress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yang Gao <ounggao1994@gmail.com>
#     Qiming Sun <osirpt.sun@gmail.com>


import numpy
import ctf
import time
from pyscf.lib import logger
from pyscf.ctfcc import eom_rccsd
import pyscf.ctfcc.uintermediates as imd
from pyscf.ctfcc import mpi_helper
from symtensor.sym_ctf import tensor, einsum, zeros

comm = mpi_helper.comm
rank = mpi_helper.rank

def pack_tril_ip(array, nocc=None, nvir=None):
    if nocc is None or nvir is None:
        nocc, nvir = array.shape[1:]
    ind, val = array.read_local()
    idxi, idxj, idxa = mpi_helper.unpack_idx(ind, nocc,nocc,nvir)
    mask = idxi<idxj
    idxij_tri = mpi_helper.sqr_to_tri(idxi, idxj, nocc)
    idx = idxij_tri[mask] * nvir + idxa[mask]
    vec = ctf.zeros([nocc*(nocc-1)//2*nvir], dtype=array.dtype)
    vec.write(idx.ravel(), val[mask].ravel())
    return vec

def pack_tril_ea(array, nocc=None, nvir=None):
    if nocc is None or nvir is None:
        nocc, nvir = array.shape[:2]
    ind, val = array.read_local()
    idxi, idxa, idxb = mpi_helper.unpack_idx(ind, nocc,nvir,nvir)
    mask = idxa<idxb
    idxab_tri = mpi_helper.sqr_to_tri(idxa, idxb, nvir)
    idx = idxi[mask] * nvir*(nvir-1)//2 + idxab_tri[mask]
    vec = ctf.zeros([nvir*(nvir-1)//2*nocc], dtype=array.dtype)
    vec.write(idx.ravel(), val[mask].ravel())
    return vec

def unpack_tril_ip(vector, nocc, nvir):
    array = ctf.zeros([nocc,nocc,nvir], dtype=vector.dtype)
    ind, val = vector.read_local()
    idxij, idxa = mpi_helper.unpack_idx(ind, nocc*(nocc-1)//2, nvir)
    idxi, idxj = mpi_helper.tri_to_sqr(idxij, nocc)
    idxija = idxi*nocc*nvir+idxj*nvir+idxa
    idxjia = idxj*nocc*nvir+idxi*nvir+idxa
    array.write(idxija.ravel(), val.ravel())
    array.write(idxjia.ravel(),-val.ravel())
    return array

def unpack_tril_ea(vector, nocc, nvir):
    array = ctf.zeros([nocc,nvir,nvir], dtype=vector.dtype)
    ind, val = vector.read_local()
    idxi, idxab = mpi_helper.unpack_idx(ind, nocc, nvir*(nvir-1)//2)
    
    idxa, idxb = mpi_helper.tri_to_sqr(idxab, nvir)
    idxiab = idxi*nvir**2+idxa*nvir+idxb
    idxiba = idxi*nvir**2+idxb*nvir+idxa
    array.write(idxiab.ravel(), val.ravel())
    array.write(idxiba.ravel(),-val.ravel())
    return array

def amplitudes_to_vector_ip(eom, r1, r2):
    '''only the triangular block for r2aaa/bbb are saved
       otherwise davidson might miss some low-lying roots'''
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    r2avec = pack_tril_ip(r2aaa.array)
    r2bvec = pack_tril_ip(r2bbb.array)
    return ctf.hstack((r1[0].array.ravel(), r1[1].array.ravel(),\
                       r2avec, r2baa.array.ravel(),\
                       r2abb.array.ravel(), r2bvec))

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    size1a = nocca
    size1b = size1a + noccb
    size2aaa = size1b + nocca*(nocca-1)//2*nvira
    size2baa = size2aaa + nocca*noccb*nvira
    size2abb = size2baa + nocca*noccb*nvirb
    r1a = tensor(vector[:size1a].reshape(nocca))
    r1b = tensor(vector[size1a:size1b].reshape(noccb))
    r2baa = tensor(vector[size2aaa:size2baa].reshape(noccb,nocca,nvira))
    r2abb = tensor(vector[size2baa:size2abb].reshape(nocca,noccb,nvirb))
    r2aaa = tensor(unpack_tril_ip(vector[size1b:size2aaa], nocca, nvira))
    r2bbb = tensor(unpack_tril_ip(vector[size2abb:], noccb, nvirb))
    return (r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)

def amplitudes_to_vector_ea(eom, r1, r2):
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    r2avec = pack_tril_ea(r2aaa.array)
    r2bvec = pack_tril_ea(r2bbb.array)
    return ctf.hstack((r1[0].array.ravel(), r1[1].array.ravel(),\
                       r2avec, r2aba.array.ravel(),\
                       r2bab.array.ravel(), r2bvec))

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    size1a = nvira
    size1b = size1a + nvirb
    size2aaa = size1b + nvira*(nvira-1)//2*nocca
    size2aba = size2aaa + nocca*nvirb*nvira
    size2bab = size2aba + noccb*nvira*nvirb
    r1a = tensor(vector[:size1a].reshape(nvira))
    r1b = tensor(vector[size1a:size1b].reshape(nvirb))
    r2aba = tensor(vector[size2aaa:size2aba].reshape(nocca,nvirb,nvira))
    r2bab = tensor(vector[size2aba:size2bab].reshape(noccb,nvira,nvirb))
    r2aaa = tensor(unpack_tril_ea(vector[size1b:size2aaa], nocca, nvira))
    r2bbb = tensor(unpack_tril_ea(vector[size2bab:], noccb, nvirb))
    return (r1a, r1b), (r2aaa, r2aba, r2bab, r2bbb)

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled'''
    if imds is None: imds = eom.make_imds()
    t1, t2= imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca + nvira, noccb + nvirb

    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa, nmob), (nocca, noccb))

    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2

    Hr1a = -einsum('mi,m->i', imds.Foo, r1a)
    Hr1b = -einsum('mi,m->i', imds.FOO, r1b)

    Hr1a += einsum('me,mie->i', imds.Fov, r2aaa)
    Hr1a -= einsum('me,ime->i', imds.FOV, r2abb)
    Hr1b += einsum('me,mie->i', imds.FOV, r2bbb)
    Hr1b -= einsum('me,ime->i', imds.Fov, r2baa)

    Hr1a += -0.5 * einsum('nime,mne->i', imds.Wooov, r2aaa)
    Hr1b +=    einsum('nime,nme->i', imds.WOOov, r2baa)
    Hr1b += -0.5 * einsum('nime,mne->i', imds.WOOOV, r2bbb)
    Hr1a +=    einsum('nime,nme->i', imds.WooOV, r2abb)

    Hr2aaa = einsum('be,ije->ijb', imds.Fvv, r2aaa)
    Hr2abb = einsum('be,ije->ijb', imds.FVV, r2abb)
    Hr2bbb = einsum('be,ije->ijb', imds.FVV, r2bbb)
    Hr2baa = einsum('be,ije->ijb', imds.Fvv, r2baa)

    tmpa = einsum('mi,mjb->ijb', imds.Foo, r2aaa)
    tmpb = einsum('mj,mib->ijb', imds.Foo, r2aaa)
    Hr2aaa -= tmpa - tmpb
    Hr2abb -= einsum('mi,mjb->ijb', imds.Foo, r2abb)
    Hr2abb -= einsum('mj,imb->ijb', imds.FOO, r2abb)
    Hr2baa -= einsum('mi,mjb->ijb', imds.FOO, r2baa)
    Hr2baa -= einsum('mj,imb->ijb', imds.Foo, r2baa)
    tmpb = einsum('mi,mjb->ijb', imds.FOO, r2bbb)
    tmpa = einsum('mj,mib->ijb', imds.FOO, r2bbb)
    Hr2bbb -= tmpb - tmpa

    Hr2aaa -= einsum('mjbi,m->ijb', imds.Woovo, r1a)
    Hr2abb += einsum('mibj,m->ijb', imds.WooVO, r1a)
    Hr2baa += einsum('mibj,m->ijb', imds.WOOvo, r1b)
    Hr2bbb -= einsum('mjbi,m->ijb', imds.WOOVO, r1b)


    Hr2aaa += .5 * einsum('minj,mnb->ijb', imds.Woooo, r2aaa)
    Hr2abb +=      einsum('minj,mnb->ijb', imds.WooOO, r2abb)
    Hr2bbb += .5 * einsum('minj,mnb->ijb', imds.WOOOO, r2bbb)
    Hr2baa +=      einsum('njmi,mnb->ijb', imds.WooOO, r2baa)


    tmp_aaa = einsum('menf,mnf->e', imds.Wovov, r2aaa)
    tmp_bbb = einsum('menf,mnf->e', imds.WOVOV, r2bbb)
    tmp_abb = einsum('menf,mnf->e', imds.WovOV, r2abb)
    tmp_baa = einsum('nfme,mnf->e', imds.WovOV, r2baa)

    Hr2aaa -= 0.5 * einsum('e,jibe->ijb', tmp_aaa, t2aa)
    Hr2aaa -= einsum('e,jibe->ijb', tmp_abb, t2aa)

    Hr2abb -= 0.5 * einsum('e,ijeb->ijb', tmp_aaa, t2ab)
    Hr2abb -= einsum('e,ijeb->ijb', tmp_abb, t2ab)

    Hr2baa -= 0.5 * einsum('e,jibe->ijb', tmp_bbb, t2ab)
    Hr2baa -= einsum('e,jibe->ijb', tmp_baa, t2ab)

    Hr2bbb -= 0.5 * einsum('e,jibe->ijb', tmp_bbb, t2bb)
    Hr2bbb -= einsum('e,jibe->ijb', tmp_baa, t2bb)

    Hr2aaa += einsum('mebj,ime->ijb', imds.Wovvo, r2aaa)
    Hr2aaa += einsum('mebj,ime->ijb', imds.WOVvo, r2abb)
    # P(ij)
    Hr2aaa -= einsum('mebi,jme->ijb', imds.Wovvo, r2aaa)
    Hr2aaa -= einsum('mebi,jme->ijb', imds.WOVvo, r2abb)


    Hr2abb += einsum('mebj,ime->ijb', imds.WovVO, r2aaa)
    Hr2abb += einsum('mebj,ime->ijb', imds.WOVVO, r2abb)
    Hr2abb -= einsum('mibe,mje->ijb', imds.WooVV, r2abb)

    Hr2baa += einsum('mebj,ime->ijb', imds.WOVvo, r2bbb)
    Hr2baa += einsum('mebj,ime->ijb', imds.Wovvo, r2baa)
    Hr2baa -= einsum('mibe,mje->ijb', imds.WOOvv, r2baa)


    Hr2bbb += einsum('mebj,ime->ijb', imds.WOVVO, r2bbb)
    Hr2bbb += einsum('mebj,ime->ijb', imds.WovVO, r2baa)
    # P(ij)
    Hr2bbb -= einsum('mebi,jme->ijb', imds.WOVVO, r2bbb)
    Hr2bbb -= einsum('mebi,jme->ijb', imds.WovVO, r2baa)
    vector = eom.amplitudes_to_vector([Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb])
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape


    Hr1a = tensor(-imds.Foo.diagonal())
    Hr1b = tensor(-imds.FOO.diagonal())

    Hr2aaa = zeros((nocca,nocca,nvira), dtype=t1[0].dtype)
    Hr2bbb = zeros((noccb,noccb,nvirb), dtype=t1[0].dtype)
    Hr2abb = zeros((nocca,noccb,nvirb), dtype=t1[0].dtype)
    Hr2baa = zeros((noccb,nocca,nvira), dtype=t1[0].dtype)

    Hr2aaa += imds.Fvv.diagonal().reshape(1,1,nvira)
    Hr2aaa -= imds.Foo.diagonal().reshape(nocca,1,1)
    Hr2aaa -= imds.Foo.diagonal().reshape(1,nocca,1)

    Hr2bbb += imds.FVV.diagonal().reshape(1,1,nvirb)
    Hr2bbb -= imds.FOO.diagonal().reshape(noccb,1,1)
    Hr2bbb -= imds.FOO.diagonal().reshape(1,noccb,1)

    Hr2abb += imds.FVV.diagonal().reshape(1,1,nvirb)
    Hr2abb -= imds.Foo.diagonal().reshape(nocca,1,1)
    Hr2abb -= imds.FOO.diagonal().reshape(1,noccb,1)

    Hr2baa += imds.Fvv.diagonal().reshape(1,1,nvira)
    Hr2baa -= imds.FOO.diagonal().reshape(noccb,1,1)
    Hr2baa -= imds.Foo.diagonal().reshape(1,nocca,1)

    if eom.partition != 'mp':
        Hr2aaa += ctf.einsum('iijj->ij', imds.Woooo.array).reshape(nocca,nocca,1)
        Hr2abb += ctf.einsum('iijj->ij', imds.WooOO.array).reshape(nocca,noccb,1)
        Hr2bbb += ctf.einsum('iijj->ij', imds.WOOOO.array).reshape(noccb,noccb,1)
        Hr2baa += ctf.einsum('jjii->ij', imds.WooOO.array).reshape(noccb,nocca,1)
        Hr2aaa -= einsum('iejb,jibe->ijb', imds.Wovov, t2aa)
        Hr2abb -= einsum('iejb,ijeb->ijb', imds.WovOV, t2ab)
        Hr2baa -= einsum('jbie,jibe->ijb', imds.WovOV, t2ab)
        Hr2bbb -= einsum('iejb,jibe->ijb', imds.WOVOV, t2bb)
        Hr2aaa += ctf.einsum('ibbi->ib', imds.Wovvo.array).reshape(nocca,1,nvira)
        Hr2aaa += ctf.einsum('jbbj->jb', imds.Wovvo.array).reshape(1,nocca,nvira)

        Hr2baa += ctf.einsum('jbbj->jb', imds.Wovvo.array).reshape(1,nocca,nvira)
        Hr2baa -= ctf.einsum('iibb->ib', imds.WOOvv.array).reshape(noccb,1,nvira)

        Hr2abb += ctf.einsum('jbbj->jb', imds.WOVVO.array).reshape(1,noccb,nvirb)
        Hr2abb -= ctf.einsum('iibb->ib', imds.WooVV.array).reshape(nocca,1,nvirb)

        Hr2bbb += ctf.einsum('ibbi->ib', imds.WOVVO.array).reshape(noccb,1,nvirb)
        Hr2bbb += ctf.einsum('jbbj->jb', imds.WOVVO.array).reshape(1,noccb,nvirb)

    vector = eom.amplitudes_to_vector((Hr1a,Hr1b), (Hr2aaa,Hr2baa,Hr2abb,Hr2bbb))
    return vector

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    '''2ph operators are of the form s_{ j}^{ab}, i.e. 'jb' indices are coupled'''
    if imds is None: imds = eom.make_imds()
    t1, t2= imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca + nvira, noccb + nvirb

    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa, nmob), (nocca, noccb))

    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2

    Hr1a = einsum('ac,c->a', imds.Fvv, r1a)
    Hr1b = einsum('ac,c->a', imds.FVV, r1b)

    Hr1a += einsum('ld,lad->a', imds.Fov, r2aaa)
    Hr1a += einsum('ld,lad->a', imds.FOV, r2bab)
    Hr1b += einsum('ld,lad->a', imds.Fov, r2aba)
    Hr1b += einsum('ld,lad->a', imds.FOV, r2bbb)

    Hr1a += 0.5*einsum('acld,lcd->a', imds.Wvvov, r2aaa)
    Hr1a +=     einsum('acld,lcd->a', imds.WvvOV, r2bab)
    Hr1b += 0.5*einsum('acld,lcd->a', imds.WVVOV, r2bbb)
    Hr1b +=     einsum('acld,lcd->a', imds.WVVov, r2aba)

    Hr2aaa = .5 * einsum('acbd,jcd->jab', imds.Wvvvv, r2aaa)
    Hr2aba =      einsum('bcad,jdc->jab', imds.WvvVV, r2aba)
    Hr2bab =      einsum('acbd,jcd->jab', imds.WvvVV, r2bab)
    Hr2bbb = .5 * einsum('acbd,jcd->jab', imds.WVVVV, r2bbb)

    Hr2aaa += einsum('acbj,c->jab', imds.Wvvvo, r1a)
    Hr2bbb += einsum('acbj,c->jab', imds.WVVVO, r1b)

    Hr2bab += einsum('acbj,c->jab', imds.WvvVO, r1a)
    Hr2aba += einsum('acbj,c->jab', imds.WVVvo, r1b)

    tmpa = einsum('ac,jcb->jab', imds.Fvv, r2aaa)
    tmpb = einsum('bc,jca->jab', imds.Fvv, r2aaa)
    Hr2aaa += tmpa - tmpb
    Hr2aba += einsum('ac,jcb->jab', imds.FVV, r2aba)
    Hr2bab += einsum('ac,jcb->jab', imds.Fvv, r2bab)
    Hr2aba += einsum('bc,jac->jab', imds.Fvv, r2aba)
    Hr2bab += einsum('bc,jac->jab', imds.FVV, r2bab)
    tmpb = einsum('ac,jcb->jab', imds.FVV, r2bbb)
    tmpa = einsum('bc,jca->jab', imds.FVV, r2bbb)
    Hr2bbb += tmpb - tmpa


    Hr2aaa -= einsum('lj,lab->jab', imds.Foo, r2aaa)
    Hr2bbb -= einsum('lj,lab->jab', imds.FOO, r2bbb)
    Hr2bab -= einsum('lj,lab->jab', imds.FOO, r2bab)
    Hr2aba -= einsum('lj,lab->jab', imds.Foo, r2aba)


    tmp_aaa = einsum('kcld,lcd->k', imds.Wovov, r2aaa)
    tmp_bbb = einsum('kcld,lcd->k', imds.WOVOV, r2bbb)
    tmp_bab = einsum('kcld,lcd->k', imds.WovOV, r2bab)
    tmp_aba = einsum('ldkc,lcd->k', imds.WovOV, r2aba)

    Hr2aaa -= 0.5 * einsum('k,kjab->jab', tmp_aaa, t2aa)
    Hr2bab -= 0.5 * einsum('k,kjab->jab', tmp_aaa, t2ab)

    Hr2aaa -= einsum('k,kjab->jab', tmp_bab, t2aa)
    Hr2bbb -= 0.5 * einsum('k,kjab->jab', tmp_bbb, t2bb)

    Hr2bbb -= einsum('k,kjab->jab', tmp_aba, t2bb)
    Hr2bab -= einsum('k,kjab->jab', tmp_bab, t2ab)


    Hr2aba -= einsum('k,jkba->jab', tmp_aba, t2ab)
    Hr2aba -= 0.5 * einsum('k,jkba->jab', tmp_bbb, t2ab)

    Hr2aaa += einsum('ldbj,lad->jab', imds.Wovvo, r2aaa)
    Hr2aaa += einsum('ldbj,lad->jab', imds.WOVvo, r2bab)
    # P(ab)
    Hr2aaa -= einsum('ldaj,lbd->jab', imds.Wovvo, r2aaa)
    Hr2aaa -= einsum('ldaj,lbd->jab', imds.WOVvo, r2bab)


    Hr2bab += einsum('ldbj,lad->jab', imds.WovVO, r2aaa)
    Hr2bab += einsum('ldbj,lad->jab', imds.WOVVO, r2bab)
    Hr2bab -= einsum('ljad,ldb->jab', imds.WOOvv, r2bab)

    Hr2aba += einsum('ldbj,lad->jab', imds.WOVvo, r2bbb)
    Hr2aba += einsum('ldbj,lad->jab', imds.Wovvo, r2aba)
    Hr2aba -= einsum('ljad,ldb->jab', imds.WooVV, r2aba)

    Hr2bbb += einsum('ldbj,lad->jab', imds.WOVVO, r2bbb)
    Hr2bbb += einsum('ldbj,lad->jab', imds.WovVO, r2aba)
    # P(ab)
    Hr2bbb -= einsum('ldaj,lbd->jab', imds.WOVVO, r2bbb)
    Hr2bbb -= einsum('ldaj,lbd->jab', imds.WovVO, r2aba)

    vector = eom.amplitudes_to_vector([Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb])
    return vector

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    Hr1a = tensor(imds.Fvv.diagonal())
    Hr1b = tensor(imds.FVV.diagonal())

    Hr2aaa = zeros((nocca,nvira,nvira), dtype=t1a.dtype)
    Hr2aba = zeros((nocca,nvirb,nvira), dtype=t1a.dtype)
    Hr2bab = zeros((noccb,nvira,nvirb), dtype=t1a.dtype)
    Hr2bbb = zeros((noccb,nvirb,nvirb), dtype=t1b.dtype)

    Hr2aaa -= imds.Foo.diagonal().reshape(nocca,1,1)
    Hr2aaa += imds.Fvv.diagonal().reshape(1,nvira,1)
    Hr2aaa += imds.Fvv.diagonal().reshape(1,1,nvira)

    Hr2aba -= imds.Foo.diagonal().reshape(nocca,1,1)
    Hr2aba += imds.FVV.diagonal().reshape(1,nvirb,1)
    Hr2aba += imds.Fvv.diagonal().reshape(1,1,nvira)

    Hr2bab -= imds.FOO.diagonal().reshape(noccb,1,1)
    Hr2bab += imds.Fvv.diagonal().reshape(1,nvira,1)
    Hr2bab += imds.FVV.diagonal().reshape(1,1,nvirb)

    Hr2bbb -= imds.FOO.diagonal().reshape(noccb,1,1)
    Hr2bbb += imds.FVV.diagonal().reshape(1,nvirb,1)
    Hr2bbb += imds.FVV.diagonal().reshape(1,1,nvirb)


    if eom.partition != 'mp':
        Hr2aaa += ctf.einsum('aabb->ab', imds.Wvvvv.array).reshape(1,nvira,nvira)
        Hr2aba += ctf.einsum('bbaa->ab', imds.WvvVV.array).reshape(1,nvirb,nvira)
        Hr2bab += ctf.einsum('aabb->ab', imds.WvvVV.array).reshape(1,nvira,nvirb)
        Hr2bbb += ctf.einsum('aabb->ab', imds.WVVVV.array).reshape(1,nvirb,nvirb)

        # Wovov term (physicist's Woovv)
        Hr2aaa -= einsum('kajb,kjab->jab', imds.Wovov, t2aa)
        Hr2aba -= einsum('jbka,jkba->jab', imds.WovOV, t2ab)
        Hr2bab -= einsum('kajb,kjab->jab', imds.WovOV, t2ab)
        Hr2bbb -= einsum('kajb,kjab->jab', imds.WOVOV, t2bb)

    # Wovvo term
        Hr2aaa += ctf.einsum('jbbj->jb', imds.Wovvo.array).reshape(nocca,1,nvira)
        Hr2aaa += ctf.einsum('jaaj->ja', imds.Wovvo.array).reshape(nocca,nvira,1)

        Hr2aba += ctf.einsum('jbbj->jb', imds.Wovvo.array).reshape(nocca,1,nvira)
        Hr2aba -= ctf.einsum('jjaa->ja', imds.WooVV.array).reshape(nocca,nvirb,1)

        Hr2bab += ctf.einsum('jbbj->jb', imds.WOVVO.array).reshape(noccb,1,nvirb)
        Hr2bab -= ctf.einsum('jjaa->ja', imds.WOOvv.array).reshape(noccb,nvira,1)

        Hr2bbb += ctf.einsum('jbbj->jb', imds.WOVVO.array).reshape(noccb,1,nvirb)
        Hr2bbb += ctf.einsum('jaaj->ja', imds.WOVVO.array).reshape(noccb,nvirb,1)

    vector = eom.amplitudes_to_vector([Hr1a,Hr1b], [Hr2aaa,Hr2aba,Hr2bab,Hr2bbb])
    return vector

class EOMIP(eom_rccsd.EOMIP):

    amplitudes_to_vector = amplitudes_to_vector_ip
    matvec = ipccsd_matvec
    get_diag = ipccsd_diag

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ip(vec, nmo, nocc)

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa,  nmob  = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        return nocca + noccb + nocca*(nocca-1)//2*nvira + noccb*nocca*nvira + \
               nocca*noccb*nvirb + noccb*(noccb-1)//2*nvirb

    def make_imds(self, eris=None):
        self.imds = imds = _IMDS(self._cc, eris=eris)
        imds.make_ip()
        return imds

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.double)
        nroots = min(nroots, size)
        nocc = sum(self.nocc)
        if koopmans:
            idx = range(nocc-nroots, nocc)[::-1]
        else:
            if diag is None: diag = self.get_diag()
            idx = mpi_helper.argsort(diag, nroots)
        guess = ctf.zeros([nroots,size], dtype)
        if rank==0:
            idx = numpy.arange(nroots)*size + numpy.asarray(idx)
            guess.write(idx, numpy.ones(nroots))
        else:
            guess.write([],[])
        return guess

class EOMEA(eom_rccsd.EOMEA):
    amplitudes_to_vector = amplitudes_to_vector_ea
    matvec = eaccsd_matvec
    get_diag = eaccsd_diag

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ea(vec, nmo, nocc)

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa,  nmob  = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        return nvira + nvirb + nocca*nvira*(nvira-1)//2 + nocca*nvirb*nvira + \
               noccb*nvira*nvirb + noccb*nvirb*(nvirb-1)//2

    def make_imds(self, eris=None):
        self.imds = imds = _IMDS(self._cc, eris=eris)
        imds.make_ea()
        return imds

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            idx = range(nroots)
        else:
            if diag is None: diag = self.get_diag()
            idx = mpi_helper.argsort(diag, nroots)
        guess = ctf.zeros([nroots,size], dtype)
        idx = numpy.arange(nroots)*size + numpy.asarray(idx)
        if rank==0:
            guess.write(idx, numpy.ones(nroots))
        else:
            guess.write([], [])
        return guess

class _IMDS:
    def __init__(self, mycc, eris=None, t1=None, t2=None):
        self._cc = mycc
        self.verbose = mycc.verbose
        self.stdout = mycc.stdout
        if t1 is None:
            t1 = mycc.t1
        self.t1 = t1
        if t2 is None:
            t2 = mycc.t2
        self.t2 = t2
        if eris is None:
            if getattr(mycc, 'eris') is None:
                eris = mycc.ao2mo()
            else:
                eris = mycc.eris
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo, self.FOO = imd.Foo(t1, t2, eris)
        self.Fvv, self.FVV = imd.Fvv(t1, t2, eris)
        self.Fov, self.FOV = imd.Fov(t1, t2, eris)
        # 2 virtuals
        self.Wovvo, self.WovVO, self.WOVvo, self.WOVVO = imd.Wovvo(t1, t2, eris)
        self.Woovv, self.WooVV, self.WOOvv, self.WOOVV = imd.Woovv(t1, t2, eris)
        self.Wovov = eris.ovov - eris.ovov.transpose(2,1,0,3)
        self.WOVOV = eris.OVOV - eris.OVOV.transpose(2,1,0,3)
        self.WovOV = eris.ovOV
        self.WOVov = None
        self._made_shared = True
        logger.timer_debug1(self, 'EOM-KCCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        # 0 or 1 virtuals
        self.Woooo, self.WooOO, _         , self.WOOOO = imd.Woooo(t1, t2, eris)
        self.Wooov, self.WooOV, self.WOOov, self.WOOOV = imd.Wooov(t1, t2, eris)
        self.Woovo, self.WooVO, self.WOOvo, self.WOOVO = imd.Woovo(t1, t2, eris)
        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-UCCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        # 3 or 4 virtuals
        self.Wvvov, self.WvvOV, self.WVVov, self.WVVOV = imd.Wvvov(t1, t2, eris)
        self.Wvvvv, self.WvvVV, self.WVVVV = imd.Wvvvv(t1, t2, eris)
        self.Wvvvo, self.WvvVO, self.WVVvo, self.WVVVO = imd.Wvvvo(t1, t2, eris)
        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-KUCCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto, scf, cc
    from pyscf.ctfcc.uccsd import UCCSD
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.spin = 0
    mol.build()

    mf = scf.UHF(mol)
    if rank==0:
        mf.kernel()

    mycc = UCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    eip = myeom.ipccsd(nroots=3)[1]
    print(numpy.amax(eip[0]-0.4335604229241659))
    print(numpy.amax(eip[1]-0.4335604229241659))
    print(numpy.amax(eip[2]-0.5187659782655635))

    myeom = EOMEA(mycc)
    eea = myeom.eaccsd(nroots=3)[1]
    print(numpy.amax(eea[0]-0.1673788639606518))
    print(numpy.amax(eea[1]-0.1673788639606518))
    print(numpy.amax(eea[2]-0.2402762272383755))
