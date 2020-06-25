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
#         Qiming Sun <osirpt.sun@gmail.com>


import numpy
import ctf
import time
from pyscf.lib import logger
from pyscf.ctfcc import eom_gccsd, eom_uccsd, eom_kccsd_rhf
import pyscf.ctfcc.gintermediates as imd
from pyscf.ctfcc import mpi_helper
from symtensor.sym_ctf import tensor, einsum, zeros

comm = mpi_helper.comm
rank = mpi_helper.rank

def amplitudes_to_vector_ip(eom, r1, r2):
    nkpts = eom.nkpts
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r2array = r2.array.transpose(0,2,1,3,4)
    r2vec = eom_uccsd.pack_tril_ip(r2array, nocc*nkpts, nvir)
    return ctf.hstack((r1.array.ravel(), r2vec))

def vector_to_amplitudes_ip(eom, vector, kshift):
    kpts = eom.kpts
    nkpts = eom.nkpts
    nocc = eom.nocc
    nvir = eom.nmo-nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    r1 = tensor(vector[:nocc], sym1, symlib=eom.symlib)
    r2 = eom_uccsd.unpack_tril_ip(vector[nocc:], nocc*nkpts, nvir)
    r2 = r2.reshape(nkpts,nocc,nkpts,nocc,nvir).transpose(0,2,1,3,4)
    r2 = tensor(r2, sym2, symlib=eom.symlib)
    return r1, r2

def amplitudes_to_vector_ea(eom, r1, r2):
    nkpts = eom.nkpts
    nocc = eom.nocc
    nmo = eom.nmo
    abi = r2.transpose(1,2,0).array.transpose(4,0,2,1,3)
    r2vec = eom_uccsd.pack_tril_ea(abi, nocc, (nmo-nocc)*nkpts)
    return ctf.hstack((r1.array.ravel(), r2vec))

def vector_to_amplitudes_ea(eom, vector, kshift):
    kpts = eom.kpts
    nkpts = eom.nkpts
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]

    r1 = tensor(vector[:nvir], sym1, symlib=eom.symlib)
    iab = eom_uccsd.unpack_tril_ea(vector[nvir:], nocc, nvir*nkpts)
    iab = iab.reshape(nocc,nkpts,nvir,nkpts,nvir).transpose(1,3,2,4,0)
    r2 = tensor(iab, sym2, symlib=eom.symlib).transpose(2,0,1)
    return r1, r2

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, kshift)
    # Eq. (8)
    Hr1 = -einsum('mi,m->i', imds.Foo, r1)
    Hr1 += einsum('me,mie->i', imds.Fov, r2)
    Hr1 += -0.5*einsum('nmie,mne->i', imds.Wooov, r2)
    # Eq. (9)
    Hr2 =  einsum('ae,ije->ija', imds.Fvv, r2)
    tmp1 = einsum('mi,mja->ija', imds.Foo, r2)
    Hr2 -= tmp1 - tmp1.transpose(1,0,2)
    Hr2 -= einsum('maji,m->ija', imds.Wovoo, r1)
    Hr2 += 0.5*einsum('mnij,mna->ija', imds.Woooo, r2)
    tmp2 = einsum('maei,mje->ija', imds.Wovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(1,0,2)
    Hr2 += 0.5*einsum('mnef,mnf,ijae->ija', imds.Woovv, r2, imds.t2)
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    kpts = eom.kpts
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    nkpts = eom.nkpts
    kconserv = eom.kconserv
    Hr1 = tensor(-imds.Foo.diagonal()[kshift], sym=sym1, symlib=eom.symlib)
    Hr2 = zeros([nocc,nocc,nvir], sym=sym2, symlib=eom.symlib, dtype=t2.dtype)
    foo = imds.Foo.diagonal().to_nparray()
    fvv = imds.Fvv.diagonal().to_nparray()
    Woo = ctf.einsum('IJIijij->IJij', imds.Woooo.array).to_nparray()
    Wia = ctf.einsum('IAAiaai->IAia', imds.Wovvo.array).to_nparray()
    jobs = numpy.arange(nkpts**2)
    tasks = mpi_helper.static_partition(jobs)
    ntasks = max(comm.allgather(len(tasks)))
    for itask in range(ntasks):
        if itask>=len(tasks):
            Hr2.write([],[])
            continue
        ki, kj = numpy.divmod(tasks[itask], nkpts)
        ka = kconserv[ki,kshift,kj]
        ija = -foo[ki][:,None,None] - foo[kj][None,:,None] + fvv[ka][None,None,:]
        ija+= Woo[ki,kj][:,:,None]
        ija+= Wia[ki,ka][:,None,:] + Wia[kj,ka][None,:,:]
        off = (ki*nkpts+kj)*ija.size
        Hr2.write(off+numpy.arange(ija.size), ija.ravel())
    Hr2 += ctf.einsum('IJijea,JIjiea->IJija',imds.Woovv.array[:,:,kshift], t2.array[:,:,kshift])

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, kshift)

    # Eq. (30)
    Hr1  = einsum('ac,c->a', imds.Fvv, r1)
    Hr1 += einsum('ld,lad->a', imds.Fov, r2)
    Hr1 += 0.5*einsum('alcd,lcd->a', imds.Wvovv, r2)
    # Eq. (31)
    Hr2 = einsum('abcj,c->jab', imds.Wvvvo, r1)
    tmp1 = einsum('ac,jcb->jab', imds.Fvv, r2)
    Hr2 += tmp1 - tmp1.transpose(0,2,1)
    Hr2 -= einsum('lj,lab->jab', imds.Foo, r2)
    tmp2 = einsum('lbdj,lad->jab', imds.Wovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(0,2,1)
    Hr2 += 0.5*einsum('abcd,jcd->jab', imds.Wvvvv, r2)
    Hr2 -= 0.5*einsum('klcd,lcd,kjab->jab', imds.Woovv, r2, imds.t2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eaccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    kpts = eom.kpts
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    nkpts = eom.nkpts
    kconserv = eom.kconserv
    Hr1 = tensor(imds.Fvv.diagonal()[kshift], sym=sym1, symlib=eom.symlib)
    Hr2 = zeros([nocc, nvir,nvir], sym=sym2, symlib=eom.symlib, dtype=t2.dtype)
    fvv = imds.Fvv.diagonal().to_nparray()
    foo = imds.Foo.diagonal().to_nparray()
    wov = ctf.einsum('JAAjaaj->JAja', imds.Wovvo.array).to_nparray()
    wvv = ctf.einsum('ABAabab->ABab', imds.Wvvvv.array).to_nparray()

    jobs = numpy.arange(nkpts**2)
    tasks = mpi_helper.static_partition(jobs)
    ntasks = max(comm.allgather(len(tasks)))
    for itask in range(ntasks):
        if itask>=len(tasks):
            Hr2.write([],[])
            continue
        ki, ka = numpy.divmod(tasks[itask], nkpts)
        kb = kconserv[ki,ka,kshift]
        iab = -foo[ki][:,None,None]+fvv[ka][None,:,None] + fvv[kb][None,None,:]

        iab += wvv[ka,kb][None]
        iab += wov[ki,ka][:,:,None] + wov[ki,kb][:,None,:]
        off = (ki*nkpts+ka)*iab.size
        Hr2.write(off+numpy.arange(iab.size), iab.ravel())
    Hr2 -= ctf.einsum('JAkjab,JAkjab->JAjab', imds.Woovv.array[kshift], t2.array[kshift])
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMIP(eom_kccsd_rhf.EOMIP):
    matvec = ipccsd_matvec
    amplitudes_to_vector = amplitudes_to_vector_ip
    vector_to_amplitudes = vector_to_amplitudes_ip
    get_diag = ipccsd_diag

    def vector_size(self):
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*nkpts*(nocc*nkpts-1)//2*nvir

    make_imds = eom_gccsd.EOMIP.make_imds

class EOMEA(eom_kccsd_rhf.EOMEA):

    matvec = eaccsd_matvec
    amplitudes_to_vector = amplitudes_to_vector_ea
    vector_to_amplitudes = vector_to_amplitudes_ea
    get_diag = eaccsd_diag

    def vector_size(self):
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nvir*nkpts*(nvir*nkpts-1)//2*nocc

    make_imds = eom_gccsd.EOMEA.make_imds

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from pyscf.ctfcc import kccsd

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

    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    kmf.conv_tol_grad = 1e-8
    if rank==0:
        kmf.kernel()
    kmf = kmf.to_ghf(kmf)

    mycc = kccsd.KGCCSD(kmf)
    mycc.kernel()

    myip = EOMIP(mycc)
    myea = EOMEA(mycc)

    eip = myip.ipccsd(nroots=3, kptlist=[1])[1].ravel()
    eea = myea.eaccsd(nroots=3, kptlist=[2])[1].ravel()

    print(eip[0] - 0.13448793)
    print(eip[1] - 0.13448793)
    print(eip[2] - 0.48273328)
    print(eea[0] - 1.6094025)
    print(eea[1] - 1.6094025)
    print(eea[2] - 2.22843578)
