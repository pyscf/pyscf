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
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

import numpy
import ctf
from pyscf.pbc.mp.kump2 import padding_k_idx
from pyscf.ctfcc import eom_rccsd, mpi_helper, eom_kccsd_rhf
from symtensor.sym_ctf import tensor, zeros, einsum
from pyscf.ctfcc import eom_uccsd

comm = mpi_helper.comm
rank = mpi_helper.rank

def vector_to_amplitudes_ip(eom, vector, kshift):
    # From symmetry for aaa and bbb terms, only store lower
    # triangular part (ki,i) < (kj,j)
    nkpts, kpts= eom.nkpts, eom.kpts
    nocca, noccb = eom.nocc
    nmoa, nmob = eom.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    size1a = nocca
    size1b = size1a + noccb
    sizeaaa = size1b + nkpts*nocca*(nkpts*nocca-1)//2*nvira
    sizebaa = sizeaaa+ nkpts**2*noccb*nocca*nvira
    sizeabb = sizebaa+ nkpts**2*nocca*noccb*nvirb

    r1a = vector[:size1a]
    r1b = vector[size1a:size1b]

    r2aaa = eom_uccsd.unpack_tril_ip(vector[size1b:sizeaaa], nkpts*noccb, nvirb)
    r2aaa = r2aaa.reshape(nkpts,nocca,nkpts,nocca,nvira).transpose(0,2,1,3,4)
    r2baa = vector[sizeaaa:sizebaa].reshape(nkpts,nkpts,noccb,nocca,nvira)
    r2abb = vector[sizebaa:sizeabb].reshape(nkpts,nkpts,nocca,noccb,nvirb)
    r2bbb = eom_uccsd.unpack_tril_ip(vector[sizeabb:], nkpts*noccb, nvirb)
    r2bbb = r2bbb.reshape(nkpts,noccb,nkpts,noccb,nvirb).transpose(0,2,1,3,4)

    r1a = tensor(r1a, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r1b = tensor(r1b, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2aaa = tensor(r2aaa, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2baa = tensor(r2baa, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2abb = tensor(r2abb, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2bbb = tensor(r2bbb, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)

    return (r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)

def vector_to_amplitudes_ea(eom, vector, kshift):
    # From symmetry for aaa and bbb terms, only store lower
    # triangular part (ka,a) < (kb,b)
    nkpts, kpts= eom.nkpts, eom.kpts
    nocca, noccb = eom.nocc
    nmoa, nmob = eom.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]

    size1a = nvira
    size1b = size1a + nvirb
    sizeaaa = size1b + nkpts*nvira*(nkpts*nvira-1)//2*nocca
    sizeaba = sizeaaa+ nkpts**2*nocca*nvirb*nvira
    sizebab = sizeaba+ nkpts**2*noccb*nvira*nvirb

    r1a = vector[:size1a]
    r1b = vector[size1a:size1b]

    abi = eom_uccsd.unpack_tril_ea(vector[size1b:sizeaaa], nocca, nkpts*nvira)
    abi = abi.reshape(nocca, nkpts, nvira, nkpts, nvira).transpose(1,3,2,4,0)
    abi = tensor(abi, ['++-', [kpts,]*3, kpts[kshift], gvec], \
                 symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2aaa = abi.transpose(2,0,1)

    r2aba = vector[sizeaaa:sizeaba].reshape(nkpts,nkpts,nocca,nvirb,nvira)
    r2bab = vector[sizeaba:sizebab].reshape(nkpts,nkpts,noccb,nvira,nvirb)

    ABI = eom_uccsd.unpack_tril_ea(vector[sizebab:], noccb, nkpts*nvirb)
    ABI = ABI.reshape(noccb, nkpts, nvirb, nkpts, nvirb).transpose(1,3,2,4,0)
    ABI = tensor(ABI, ['++-', [kpts,]*3, kpts[kshift], gvec], \
                 symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2bbb = ABI.transpose(2,0,1)
    r1a = tensor(r1a, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r1b = tensor(r1b, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2aba = tensor(r2aba, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2bab = tensor(r2bab, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)

    return (r1a, r1b), (r2aaa, r2aba, r2bab, r2bbb)


def amplitudes_to_vector_ip(eom, r1, r2):
    nkpts = eom.nkpts
    nocca, noccb = eom.nocc
    nmoa, nmob = eom.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    r2avec = eom_uccsd.pack_tril_ip(r2aaa.array.transpose(0,2,1,3,4), nkpts*nocca, nvira)
    r2bvec = eom_uccsd.pack_tril_ip(r2bbb.array.transpose(0,2,1,3,4), nkpts*noccb, nvirb)

    vector = ctf.hstack((r1a.array.ravel(),   r1b.array.ravel(), \
                         r2avec,              r2baa.array.ravel(), \
                         r2abb.array.ravel(), r2bvec))
    return vector

def amplitudes_to_vector_ea(eom, r1, r2):
    nkpts = eom.nkpts
    nocca, noccb = eom.nocc
    nmoa, nmob = eom.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    abi = r2aaa.transpose(1,2,0).array.transpose(0,2,1,3,4).reshape(nkpts*nvira,nkpts*nvira,nocca)
    r2avec = eom_uccsd.pack_tril_ea(abi.transpose(2,0,1),nocca,nkpts*nvira)

    ABI = r2bbb.transpose(1,2,0).array.transpose(0,2,1,3,4).reshape(nkpts*nvirb,nkpts*nvirb,noccb)
    r2bvec = eom_uccsd.pack_tril_ea(ABI.transpose(2,0,1),noccb,nkpts*nvirb)

    vector = ctf.hstack((r1a.array.ravel(),   r1b.array.ravel(), \
                         r2avec,              r2aba.array.ravel(), \
                         r2bab.array.ravel(), r2bvec))
    return vector

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    t1, t2= imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca + nvira, noccb + nvirb

    r1, r2 = eom.vector_to_amplitudes(vector, kshift)

    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2

    #Foo term
    # -\sum_{k} U_{k,i} s_{k}
    Hr1a = -einsum('mi,m->i', imds.Foo, r1a)
    Hr1b = -einsum('mi,m->i', imds.FOO, r1b)
    #Fov term
    # \sum_{L,D} U_{L,D} S_{i,L}^{D} + \sum_{l,d} U_{l,d} S_{i,l}^{d}
    Hr1a += einsum('me,mie->i', imds.Fov, r2aaa)
    Hr1a -= einsum('me,ime->i', imds.FOV, r2abb)
    Hr1b += einsum('me,mie->i', imds.FOV, r2bbb)
    Hr1b -= einsum('me,ime->i', imds.Fov, r2baa)

    #Wooov
    # \sum_{k,l,d} W_{k,i,l,d} s_{l,k}^{d}
    # \sum_{k,L,D} W_{k,i,L,D} s_{L,k}^{D}
    Hr1a += -0.5 * einsum('nime,mne->i', imds.Wooov, r2aaa)
    Hr1b +=    einsum('nime,nme->i', imds.WOOov, r2baa)
    Hr1b += -0.5 * einsum('nime,mne->i', imds.WOOOV, r2bbb)
    Hr1a +=    einsum('nime,nme->i', imds.WooOV, r2abb)

    # Fvv term
    # \sum_{d} U_{b,d} S_{i,j}^{d} = (\bar{H}S)_{i,j}^{b}
    # \sum_{D} S_{i,J}^{D} U_{B,D} = (\bar{H}S)_{i,J}^{B}
    Hr2aaa = einsum('be,ije->ijb', imds.Fvv, r2aaa)
    Hr2abb = einsum('be,ije->ijb', imds.FVV, r2abb)
    Hr2bbb = einsum('be,ije->ijb', imds.FVV, r2bbb)
    Hr2baa = einsum('be,ije->ijb', imds.Fvv, r2baa)

    # Foo term
    # \sum_{l} U_{l,i} s_{l,j}^{b} = (\bar{H}S)_{i,j}^{b}
    # \sum_{l} U_{l,j} S_{i,l}^{b} = (\bar{H}S)_{i,j}^{b}

    # \sum_{l} S_{l,J}^{B} U_{l,i} = (\bar{H}S)_{i,J}^{B}
    # \sum_{L} S_{i,L}^{B} U_{L,J} = (\bar{H}S)_{i,J}^{B}

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

    # Wovoo term
    # \sum_{k} W_{k,b,j,i} s_{k} = (\bar{H}S)_{i,j}^{b}
    # \sum_{k} W_{k,B,i,J} S_{k} = (\bar{H}S)_{i,J}^{B}

    Hr2aaa -= einsum('mjbi,m->ijb', imds.Woovo, r1a)
    Hr2abb += einsum('mibj,m->ijb', imds.WooVO, r1a)
    Hr2baa += einsum('mibj,m->ijb', imds.WOOvo, r1b)
    Hr2bbb -= einsum('mjbi,m->ijb', imds.WOOVO, r1b)

    # Woooo term
    # \sum_{k,l} W_{k,i,l,j} S_{k,l}^{b} = (\bar{H}S)_{i,j}^{b}
    # \sum_{k,L} W_{k,L,i,J} S_{k,L}^{B} = (\bar{H}S)_{i,J}^{B}
    Hr2aaa += .5 * einsum('minj,mnb->ijb', imds.Woooo, r2aaa)
    Hr2abb +=      einsum('minj,mnb->ijb', imds.WooOO, r2abb)
    Hr2bbb += .5 * einsum('minj,mnb->ijb', imds.WOOOO, r2bbb)
    Hr2baa +=      einsum('njmi,mnb->ijb', imds.WooOO, r2baa)

    # T2 term
    # - \sum_{c} t_{j,i}^{b,c} [ \sum_{k,L,D} W_{L,k,D,c} S_{k,L}^{D}
    # + \sum{k,l,d} W_{l,k,d,c} S_{k,l}^{d} ] = (\bar{H}S)_{i,j}^{b}
    #
    # - \sum_{c} t_{i,J}^{c,B} [ \sum_{k,L,D} W_{L,k,D,c} S_{k,L}^{D}
    # + \sum{k,l,d} W_{l,k,d,c} S_{k,l}^{d} ] = (\bar{H}S)_{i,J}^{B}

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

    # j \/ b   |  i
    #    ---   |
    #      /\  |
    #    m \/ e|
    #     -------

    # \sum_{L,D} W_{L,D,b,j} S_{i,L}^{b}
    # \sum_{l,d} W_{l,d,b,j} S_{i,l}^{b}
    Hr2aaa += einsum('mebj,ime->ijb', imds.Wovvo, r2aaa)
    Hr2aaa += einsum('mebj,ime->ijb', imds.WOVvo, r2abb)
    # P(ij)
    Hr2aaa -= einsum('mebi,jme->ijb', imds.Wovvo, r2aaa)
    Hr2aaa -= einsum('mebi,jme->ijb', imds.WOVvo, r2abb)

    # \sum_{L,D} W_{L,D,b,J} S_{i,L}^{D}
    # \sum_{l,d} W_{l,d,B,J} S_{i,l}^{d}
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

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    kconserv = eom.kconserv
    nkpts = len(kpts)
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]

    Hr1a = tensor(-imds.Foo.diagonal()[kshift], sym=sym1, symlib=eom.symlib)
    Hr1b = tensor(-imds.FOO.diagonal()[kshift], sym=sym1, symlib=eom.symlib)

    Hr2aaa = zeros((nocca,nocca,nvira), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)
    Hr2bbb = zeros((noccb,noccb,nvirb), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)
    Hr2abb = zeros((nocca,noccb,nvirb), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)
    Hr2baa = zeros((noccb,nocca,nvira), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)

    tasks = mpi_helper.static_partition(range(nkpts**2))
    ntasks = max(comm.allgather(len(tasks)))

    Foo = imds.Foo.diagonal().to_nparray()
    Fvv = imds.Fvv.diagonal().to_nparray()
    FOO = imds.FOO.diagonal().to_nparray()
    FVV = imds.FVV.diagonal().to_nparray()

    if eom.partition == 'mp':
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2aaa.write([], [])
                Hr2bbb.write([], [])
                Hr2abb.write([], [])
                Hr2baa.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
            kb = kconserv[ki,kshift,kj]
            off = ki * nkpts + kj
            ijb = -Foo[ki][:,None,None]-Foo[kj][None,:,None]+Fvv[kb][None,None,:]
            IJB = -FOO[ki][:,None,None]-FOO[kj][None,:,None]+FVV[kb][None,None,:]
            iJB = -Foo[ki][:,None,None]-FOO[kj][None,:,None]+FVV[kb][None,None,:]
            Ijb = -FOO[ki][:,None,None]-Foo[kj][None,:,None]+Fvv[kb][None,None,:]
            Hr2aaa.write(off*ijb.size+numpy.arange(ijb.size), ijb.ravel())
            Hr2bbb.write(off*IJB.size+numpy.arange(IJB.size), IJB.ravel())
            Hr2abb.write(off*iJB.size+numpy.arange(iJB.size), iJB.ravel())
            Hr2baa.write(off*Ijb.size+numpy.arange(Ijb.size), Ijb.ravel())

    else:
        Woo = ctf.einsum('IIJiijj->IJij', imds.Woooo.array).to_nparray()
        WOO = ctf.einsum('IIJiijj->IJij', imds.WOOOO.array).to_nparray()
        WoO = ctf.einsum('IIJiijj->IJij', imds.WooOO.array).to_nparray()
        wib = ctf.einsum('IBBibbi->IBib', imds.Wovvo.array).to_nparray()
        wIB = ctf.einsum('IBBibbi->IBib', imds.WOVVO.array).to_nparray()
        wIb = ctf.einsum('IIBiibb->IBib', imds.WOOvv.array).to_nparray()
        wiB = ctf.einsum('IIBiibb->IBib', imds.WooVV.array).to_nparray()

        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2aaa.write([], [])
                Hr2bbb.write([], [])
                Hr2abb.write([], [])
                Hr2baa.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
            kb = kconserv[ki,kshift,kj]
            off = ki * nkpts + kj
            ijb = -Foo[ki][:,None,None]-Foo[kj][None,:,None]+Fvv[kb][None,None,:]
            IJB = -FOO[ki][:,None,None]-FOO[kj][None,:,None]+FVV[kb][None,None,:]
            iJB = -Foo[ki][:,None,None]-FOO[kj][None,:,None]+FVV[kb][None,None,:]
            Ijb = -FOO[ki][:,None,None]-Foo[kj][None,:,None]+Fvv[kb][None,None,:]

            ijb+= Woo[ki,kj][:,:,None]
            IJB+= WOO[ki,kj][:,:,None]
            iJB+= WoO[ki,kj][:,:,None]
            Ijb+= WoO[kj,ki][:,:,None].transpose(1,0,2)

            ijb += wib[ki,kb][:,None,:] + wib[kj,kb][None,:,:]
            IJB += wIB[ki,kb][:,None,:] + wIB[kj,kb][None,:,:]
            Ijb += wIb[ki,kb][:,None,:] + wib[kj,kb][None,:,:]
            iJB += wiB[ki,kb][:,None,:] + wIB[kj,kb][None,:,:]

            Hr2aaa.write(off*ijb.size+numpy.arange(ijb.size), ijb.ravel())
            Hr2bbb.write(off*IJB.size+numpy.arange(IJB.size), IJB.ravel())
            Hr2abb.write(off*iJB.size+numpy.arange(iJB.size), iJB.ravel())
            Hr2baa.write(off*Ijb.size+numpy.arange(Ijb.size), Ijb.ravel())

        tmpW = imds.WovOV.transpose(0,2,3,1).array[:,:,kshift]
        tmpt2 = t2ab.transpose(1,0,3,2).array[:,:,kshift]
        Hr2aaa -= ctf.einsum('IJiejb,IJijeb->IJijb', imds.Wovov.array[:,kshift], t2aa.array[:,:,kshift])
        Hr2abb -= ctf.einsum('IJiejb,IJijeb->IJijb', imds.WovOV.array[:,kshift], t2ab.array[:,:,kshift])
        Hr2bbb -= ctf.einsum('IJiejb,IJijeb->IJijb', imds.WOVOV.array[:,kshift], t2bb.array[:,:,kshift])
        Hr2baa -= ctf.einsum('JIjieb,IJijeb->IJijb', tmpW, tmpt2)
    vector = eom.amplitudes_to_vector((Hr1a,Hr1b), (Hr2aaa,Hr2baa,Hr2abb,Hr2bbb))
    return vector

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    t1, t2= imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nmoa, nmob = nocca + nvira, noccb + nvirb

    r1, r2 = eom.vector_to_amplitudes(vector, kshift)

    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    # BEGINNING OF MATVEC CONTRACTIONS: ref - Nooijen 1995 EOM-CC for EA

    # Fvv terms
    # (\bar{H}S)^a = \sum_{c} U_{ac} s^c
    Hr1a = einsum('ac,c->a', imds.Fvv, r1a)
    Hr1b = einsum('ac,c->a', imds.FVV, r1b)

    # Fov terms
    # (\bar{H}S)^a = \sum_{L,D} U_{L,D} s^{a,D}_{L} + \sum_{l,d} U_{d}^{a,d}_{l}
    Hr1a += einsum('ld,lad->a', imds.Fov, r2aaa)
    Hr1a += einsum('ld,lad->a', imds.FOV, r2bab)
    Hr1b += einsum('ld,lad->a', imds.Fov, r2aba)
    Hr1b += einsum('ld,lad->a', imds.FOV, r2bbb)

    # Wvovv
    # (\bar{H}S)^a = \sum_{c,L,D} W_{a,l,c,D} s_{L}^{c,D}
    # + \sum_{c,d,l} W_{a,l,c,d} s_{l}^{c,d}
    Hr1a += 0.5*einsum('acld,lcd->a', imds.Wvvov, r2aaa)
    Hr1a +=     einsum('acld,lcd->a', imds.WvvOV, r2bab)
    Hr1b += 0.5*einsum('acld,lcd->a', imds.WVVOV, r2bbb)
    Hr1b +=     einsum('acld,lcd->a', imds.WVVov, r2aba)

    # Wvvvv
    # \sum_{c,d} W_{a,b,c,d} s_{j}^{c,d} = (\bar{H}S)^{a,b}_{j}
    # \sum_{c,D} W_{a,B,c,D} s_{J}^{c,D} = (\bar{H}S)^{a,B}_{J}

    Hr2aaa = .5 * einsum('acbd,jcd->jab', imds.Wvvvv, r2aaa)
    Hr2aba =      einsum('bcad,jdc->jab', imds.WvvVV, r2aba)
    Hr2bab =      einsum('acbd,jcd->jab', imds.WvvVV, r2bab)
    Hr2bbb = .5 * einsum('acbd,jcd->jab', imds.WVVVV, r2bbb)

    #Wvvvo
    # \sum_{c,a,j} W_{a,b,c,j} s^{c} = (\bar{H}S)^{a,b}_{j}
    # \sum_{c,a,J} W_{a,B,c,J} s^{c} = (\bar{H}S)^{a,B}_{J}
    Hr2aaa += einsum('acbj,c->jab', imds.Wvvvo, r1a)
    Hr2bbb += einsum('acbj,c->jab', imds.WVVVO, r1b)

    Hr2bab += einsum('acbj,c->jab', imds.WvvVO, r1a)
    Hr2aba += einsum('acbj,c->jab', imds.WVVvo, r1b)

    #Fvv Terms
    # sum_{c,a,j} s_{j}^{c,b} U_{a,c} = (\bar{H}S)^{a,b}_{j}
    # sum_{d,b,j} s_{j}^{a,d} U_{b,d} = (\bar{H}S)^{a,b}_{j}

    # sum_{c,a,J} U_{a,c} s_{J}^{c,B} = (\bar{H}S)^{a,B}_{J}
    # sum_{D,a,j} U_{b,d} s_{j}^{a,d} = (\bar{H}S)^{a,B}_{J}
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

    #Foo Term
    # \sum_{l} U_{l,j} s^{a,b}^{l} = (\bar{H}S)^{a, b}_{j}
    # \sum_{L} s^{a,B}_{L} U_{L,J} = (\bar{H}S)^{a, B}_{J}
    Hr2aaa -= einsum('lj,lab->jab', imds.Foo, r2aaa)
    Hr2bbb -= einsum('lj,lab->jab', imds.FOO, r2bbb)
    Hr2bab -= einsum('lj,lab->jab', imds.FOO, r2bab)
    Hr2aba -= einsum('lj,lab->jab', imds.Foo, r2aba)

    # Woovv term
    # - \sum_{k} t_{k,j}^{a,b} [\sum_{c,D,L} W_{L,k,D,c} s_{L}^{c,D}
    # + \sum_{c,d,l} W_{k,l,c,d} s_{l}^{c,d} ] = (\bar{H}S)^{a, b}_{j}
    #
    # - \sum_{k} t_{k,J}^{a,B} [ \sum{c,D,L} W_{k,L,c,D} s_{L}^{c,D}
    # + \sum_{c,d,l} W_{k,l,c,d} s_{l}^{c,d} ] = (\bar{H}S)^{a, B}_{J}
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
    # j \/ b   |  a
    #    ---   |
    #      /\  |
    #    l \/ d|
    #     -------

    # \sum_{L,D} W_{L,b,D,j} s_{L}^{a,D} = (\bar{H}S)^{a,b}_{j}
    # \sum_{l,d} W_{l,b,d,j} s_{l}^{a,d} = (\bar{H}S)^{a,b}_{j}
    Hr2aaa += einsum('ldbj,lad->jab', imds.Wovvo, r2aaa)
    Hr2aaa += einsum('ldbj,lad->jab', imds.WOVvo, r2bab)
    # P(ab)
    Hr2aaa -= einsum('ldaj,lbd->jab', imds.Wovvo, r2aaa)
    Hr2aaa -= einsum('ldaj,lbd->jab', imds.WOVvo, r2bab)
    # \sum_{L,D} W_{L,B,D,J} s_{L}^{a,D} = (\bar{H}S)^{a,B}_{J}
    # \sum_{l,d} W_{l,B,d,J} s_{l}^{a,d} = (\bar{H}S)^{a,B}_{J}
    # - \sum_{c,L} W_{a,L,c,J} s_{L}^{c,B} = (\bar{H}S)^{a, B}_{J}
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

def eaccsd_diag(eom, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    kconserv = eom.kconserv
    nkpts = len(kpts)
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]

    Hr1a = tensor(imds.Fvv.diagonal()[kshift], sym=sym1, symlib=eom.symlib)
    Hr1b = tensor(imds.FVV.diagonal()[kshift], sym=sym1, symlib=eom.symlib)

    Hr2aaa = zeros((nocca,nvira,nvira), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)
    Hr2bbb = zeros((noccb,nvirb,nvirb), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)
    Hr2aba = zeros((nocca,nvirb,nvira), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)
    Hr2bab = zeros((noccb,nvira,nvirb), sym=sym2, symlib=eom.symlib, dtype=t2aa.dtype)

    tasks = mpi_helper.static_partition(range(nkpts**2))
    ntasks = max(comm.allgather(len(tasks)))

    Foo = imds.Foo.diagonal().to_nparray()
    Fvv = imds.Fvv.diagonal().to_nparray()
    FOO = imds.FOO.diagonal().to_nparray()
    FVV = imds.FVV.diagonal().to_nparray()
    if eom.partition == 'mp':
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2aaa.write([], [])
                Hr2bbb.write([], [])
                Hr2aba.write([], [])
                Hr2bab.write([], [])
                continue
            kiab = tasks[itask]
            ki, ka = (kiab).__divmod__(nkpts)
            kb = kconserv[ki,ka,kshift]
            off = ki*nkpts+ka
            iab = -Foo[ki][:,None,None]+Fvv[ka][None,:,None]+Fvv[kb][None,None,:]
            IAB = -FOO[ki][:,None,None]+FVV[ka][None,:,None]+FVV[kb][None,None,:]
            iAb = -Foo[ki][:,None,None]+FVV[ka][None,:,None]+Fvv[kb][None,None,:]
            IaB = -FOO[ki][:,None,None]+Fvv[ka][None,:,None]+FVV[kb][None,None,:]
            Hr2aaa.write(off*iab.size+numpy.arange(iab.size), iab.ravel())
            Hr2bbb.write(off*IAB.size+numpy.arange(IAB.size), IAB.ravel())
            Hr2aba.write(off*iAb.size+numpy.arange(iAb.size), iAb.ravel())
            Hr2bab.write(off*IaB.size+numpy.arange(IaB.size), IaB.ravel())
    else:
        wab = ctf.einsum('AABaabb->ABab', imds.Wvvvv.array).to_nparray()
        waB = ctf.einsum('AABaabb->ABab', imds.WvvVV.array).to_nparray()
        wAB = ctf.einsum('AABaabb->ABab', imds.WVVVV.array).to_nparray()

        wov = ctf.einsum('IBBibbi->IBib', imds.Wovvo.array).to_nparray()
        wOV = ctf.einsum('IBBibbi->IBib', imds.WOVVO.array).to_nparray()
        wOv = ctf.einsum('IIBiibb->IBib', imds.WOOvv.array).to_nparray()
        woV = ctf.einsum('IIBiibb->IBib', imds.WooVV.array).to_nparray()
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2aaa.write([], [])
                Hr2bbb.write([], [])
                Hr2aba.write([], [])
                Hr2bab.write([], [])
                continue
            kiab = tasks[itask]
            ki, ka = (kiab).__divmod__(nkpts)
            kb = kconserv[ki,ka,kshift]
            off = ki*nkpts+ka
            iab = -Foo[ki][:,None,None]+Fvv[ka][None,:,None]+Fvv[kb][None,None,:]
            IAB = -FOO[ki][:,None,None]+FVV[ka][None,:,None]+FVV[kb][None,None,:]
            iAb = -Foo[ki][:,None,None]+FVV[ka][None,:,None]+Fvv[kb][None,None,:]
            IaB = -FOO[ki][:,None,None]+Fvv[ka][None,:,None]+FVV[kb][None,None,:]

            iab += wab[ka,kb][None,:,:]
            IAB += wAB[ka,kb][None,:,:]
            iAb += waB[kb,ka][None,:,:].transpose(0,2,1)
            IaB += waB[ka,kb][None,:,:]

            iab += wov[ki,kb][:,None,:] + wov[ki,ka][:,:,None]
            IAB += wOV[ki,kb][:,None,:] + wOV[ki,ka][:,:,None]
            iAb += wov[ki,kb][:,None,:] + woV[ki,ka][:,:,None]
            IaB += wOV[ki,kb][:,None,:] + wOv[ki,ka][:,:,None]

            Hr2aaa.write(off*iab.size+numpy.arange(iab.size), iab.ravel())
            Hr2bbb.write(off*IAB.size+numpy.arange(IAB.size), IAB.ravel())
            Hr2aba.write(off*iAb.size+numpy.arange(iAb.size), iAb.ravel())
            Hr2bab.write(off*IaB.size+numpy.arange(IaB.size), IaB.ravel())

        Hr2aaa -= ctf.einsum('AJkajb,JAkjab->JAjab', imds.Wovov.array[kshift], t2aa.array[kshift])
        tmpw = imds.WovOV.transpose(2,3,0,1).array[kshift]
        tmpt = t2ab.transpose(0,1,3,2).array[:,kshift]
        Hr2aba -= ctf.einsum('AJkajb,JAjkab->JAjab', tmpw, tmpt)
        Hr2bab -= ctf.einsum('AJkajb,JAkjab->JAjab', imds.WovOV.array[kshift], t2ab.array[kshift])
        Hr2bbb -= ctf.einsum('AJkajb,JAkjab->JAjab', imds.WOVOV.array[kshift], t2bb.array[kshift])

    return eom.amplitudes_to_vector([Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb])

class EOMIP(eom_kccsd_rhf.EOMIP):

    vector_to_amplitudes = vector_to_amplitudes_ip
    amplitudes_to_vector = amplitudes_to_vector_ip
    matvec = ipccsd_matvec
    get_diag = ipccsd_diag

    def get_padding_k_idx(eom, mycc):
        nonzero_padding_alpha, nonzero_padding_beta = padding_k_idx(mycc, kind="split")
        nonzero_opadding_alpha, nonzero_vpadding_alpha = nonzero_padding_alpha
        nonzero_opadding_beta, nonzero_vpadding_beta = nonzero_padding_beta
        return ((nonzero_opadding_alpha, nonzero_opadding_beta),
                (nonzero_vpadding_alpha, nonzero_vpadding_beta))

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        nkpts = self.nkpts
        return nocca + noccb + nkpts*nocca*(nkpts*nocca-1)//2*nvira +\
               nkpts**2*noccb*nocca*(nvira+nvirb) + nkpts*noccb*(nkpts*noccb-1)//2*nvirb

    @property
    def nkpts(self):
        return len(self.kpts)

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        nocca, noccb = self.nocc
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.complex)
        nroots = min(nroots, size)
        guess = ctf.zeros([nroots, int(size)])
        if koopmans:
            idx = numpy.zeros(nroots, dtype=numpy.int)
            nonzero_opadding_alpha, nonzero_opadding_beta = self.nonzero_opadding
            tmp_oalpha, tmp_obeta = nonzero_opadding_alpha[kshift], nonzero_opadding_beta[kshift]
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
            ind = [kn*size+n for kn, n in enumerate(idx)]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None).real
            idx = mpi_helper.argsort(diag, nroots)
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = numpy.ones(nroots)
        if rank==0:
            guess.write(ind, fill)
        else:
            guess.write([], [])
        return guess

    def make_imds(self, eris=None):
        imds = eom_uccsd._IMDS(self._cc, eris=eris)
        imds.make_ip()
        return imds

class EOMEA(EOMIP):

    vector_to_amplitudes = vector_to_amplitudes_ea
    amplitudes_to_vector = amplitudes_to_vector_ea
    matvec = eaccsd_matvec
    get_diag = eaccsd_diag
    eaccsd = eom_kccsd_rhf.kernel

    def vector_size(self):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        nkpts = self.nkpts
        return nvira + nvirb + nkpts*nvira*(nkpts*nvira-1)//2*nocca +\
               nkpts**2*nvirb*nvira*(nocca+noccb) + nkpts*nvirb*(nkpts*nvirb-1)//2*noccb

    def update_symlib(self,kshift):
        kpts, gvec = self.kpts, self._cc._scf.cell.reciprocal_vectors()
        sym1 = ['+', [kpts,], kpts[kshift], gvec]
        sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
        self.symlib.update(sym1,sym2)

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb=  nmoa-nocca, nmob-noccb
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.complex)
        nroots = min(nroots, size)
        guess = ctf.zeros([nroots, int(size)])
        if koopmans:
            idx = numpy.zeros(nroots, dtype=numpy.int)
            nonzero_vpadding_alpha, nonzero_vpadding_beta = self.nonzero_vpadding
            tmp_valpha, tmp_vbeta = nonzero_vpadding_alpha[kshift], nonzero_vpadding_beta[kshift]
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
            ind = [kn*size+n for kn, n in enumerate(idx)]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None).real
            idx = mpi_helper.argsort(diag, nroots)
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = numpy.ones(nroots)
        if rank==0:
            guess.write(ind, fill)
        else:
            guess.write([],[])
        return guess

    def make_imds(self, eris=None):
        imds = eom_uccsd._IMDS(self._cc, eris=eris)
        imds.make_ea()
        return imds

    @property
    def eea(self):
        return self.e

if __name__ == '__main__':

    from pyscf.pbc import gto, scf
    from pyscf.ctfcc.kccsd_uhf import KUCCSD
    cell = gto.Cell()
    cell.verbose = 5
    cell.unit = 'B'
    cell.atom = '''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.mesh = [13,13,13]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000
    '''
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KUHF(cell,kpts, exxdiv=None)
    if rank ==0:
        mf.kernel()
    mycc = KUCCSD(mf)
    mycc.kernel()

    myip = EOMIP(mycc)
    myea = EOMEA(mycc)

    eip = myip.ipccsd(nroots=3, kptlist=[1])[1].ravel()
    eea = myea.eaccsd(nroots=3, kptlist=[2])[1].ravel()
    print(abs(eip[0]-0.13448793))
    print(abs(eip[1]-0.13448793))
    print(abs(eip[2]-0.48273328))
    print(abs(eea[0]-1.6094025))
    print(abs(eea[1]-1.6094025))
    print(abs(eea[2]-2.22843578))
