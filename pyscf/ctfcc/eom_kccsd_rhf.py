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
import time
import ctf
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.ctfcc import eom_rccsd, mpi_helper
from pyscf.ctfcc.linalg_helper.davidson import eigs
from symtensor.sym_ctf import tensor, zeros, einsum


comm = mpi_helper.comm
rank = mpi_helper.rank

def kernel(eom, nroots=1, koopmans=True, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):

    cput0 = (time.clock(), time.time())
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    size = eom.vector_size()
    nroots = min(nroots,size)
    nkpts = eom.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    if dtype is None:
        if isinstance(imds.t1, tuple):
            dtype = imds.t1[0].dtype
        else:
            dtype = imds.t1.dtype

    evals = numpy.zeros((len(kptlist),nroots), numpy.float)
    evecs = []
    convs = numpy.zeros((len(kptlist),nroots), dtype)

    for k, kshift in enumerate(kptlist):
        matvec, diag = eom.gen_matvec(kshift, imds, left=left, **kwargs)
        eom.update_symlib(kshift)
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            user_guess = False
            guess = eom.get_init_guess(kshift, nroots, koopmans, diag)

        conv_k, evals_k, evecs_k = eigs(matvec, size, nroots, x0=guess, Adiag=diag, verbose=eom.verbose)
        evals_k = evals_k.real
        evals[k] = evals_k.real
        evecs.append(evecs_k)
        convs[k] = conv_k

        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn, kshift)
            if isinstance(r1, tuple) or isinstance(r1, list):
                qp_weight = sum([r.norm()**2 for r in r1])
            else:
                qp_weight = r1.norm()**2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    logger.timer(eom, 'EOM-CCSD', *cput0)
    evecs = ctf.vstack(tuple(evecs))
    return convs, evals, evecs

def vector_to_amplitudes_ip(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    r1 = tensor(r1, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    return [r1,r2]

def vector_to_amplitudes_ea(eom, vector, kshift):
    nkpts, kpts, nocc = eom.nkpts, eom.kpts, eom.nocc
    nvir = eom.nmo - nocc
    gvec = eom._cc._scf.cell.reciprocal_vectors()
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
    r1 = tensor(r1,sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2,sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    return [r1,r2]

def amplitudes_to_vector(eom, r1, r2):
    vector = ctf.hstack((r1.array.ravel(), r2.array.ravel()))
    return vector

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector, kshift)

    # 1h-1h block
    Hr1 = -einsum('ki,k->i',imds.Loo,r1)
    #1h-2h1p block
    Hr1 += 2*einsum('ld,ild->i',imds.Fov,r2)
    Hr1 +=  -einsum('kd,kid->i',imds.Fov,r2)
    Hr1 += -2*einsum('klid,kld->i',imds.Wooov,r2)
    Hr1 +=    einsum('lkid,kld->i',imds.Wooov,r2)

    # 2h1p-1h block
    Hr2 = -einsum('kbij,k->ijb',imds.Wovoo,r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        foo = self.eris.foo
        fvv = self.eris.fvv
        Hr2 += einsum('bd,ijd->ijb',fvv,r2)
        Hr2 += -einsum('ki,kjb->ijb',foo,r2)
        Hr2 += -einsum('lj,ilb->ijb',foo,r2)
    elif eom.partition == 'full':
        Hr2 += self._ipccsd_diag_matrix2*r2
    else:
        Hr2 += einsum('bd,ijd->ijb',imds.Lvv,r2)
        Hr2 += -einsum('ki,kjb->ijb',imds.Loo,r2)
        Hr2 += -einsum('lj,ilb->ijb',imds.Loo,r2)
        Hr2 +=  einsum('klij,klb->ijb',imds.Woooo,r2)
        Hr2 += 2*einsum('lbdj,ild->ijb',imds.Wovvo,r2)
        Hr2 +=  -einsum('kbdj,kid->ijb',imds.Wovvo,r2)
        Hr2 +=  -einsum('lbjd,ild->ijb',imds.Wovov,r2) #typo in Ref
        Hr2 +=  -einsum('kbid,kjd->ijb',imds.Wovov,r2)
        tmp = 2*einsum('lkdc,kld->c',imds.Woovv,r2)
        tmp += -einsum('kldc,kld->c',imds.Woovv,r2)
        Hr2 += -einsum('c,ijcb->ijb',tmp,imds.t2)

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def ipccsd_diag(eom, kshift, imds=None):
    if imds is None: imds = eom.make_imds()
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    dtype = t2.dtype
    nocc, nvir = t1.shape
    nkpts = len(kpts)
    kconserv = eom.kconserv
    Hr1array = -imds.Loo.diagonal()[kshift]
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
    Hr1 = tensor(Hr1array, sym1, symlib=eom.symlib, verbose=eom.SYMVERBOSE)
    Hr2 = zeros([nocc,nocc,nvir], dtype, sym2, symlib=eom.symlib, verbose=eom.SYMVERBOSE)

    tasks = mpi_helper.static_partition(range(nkpts**2))
    ntasks = max(comm.allgather(len(tasks)))

    idx_ijb = numpy.arange(nocc*nocc*nvir)
    if eom.partition == 'mp':
        foo = eom.eris.foo.diagonal().to_nparray()
        fvv = eom.eris.fvv.diagonal().to_nparray()
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
            kb = kconserv[ki,kshift,kj]
            off = ki * nkpts + kj
            ijb = numpy.zeros([nocc,nocc,nvir], dtype=dtype)
            ijb += fvv[kb].reshape(1,1,-1)
            ijb -= foo[ki][:,None,None]
            ijb -= foo[kj][None,:,None]
            Hr2.write(off*idx_ijb.size+idx_ijb, ijb.ravel())
    else:
        lvv = imds.Lvv.diagonal().to_nparray()
        loo = imds.Loo.diagonal().to_nparray()
        wij = ctf.einsum('IJIijij->IJij', imds.Woooo.array).to_nparray()
        wjb = ctf.einsum('JBJjbjb->JBjb', imds.Wovov.array).to_nparray()
        wjb2 = ctf.einsum('JBBjbbj->JBjb', imds.Wovvo.array).to_nparray()
        wib = ctf.einsum('IBIibib->IBib', imds.Wovov.array).to_nparray()
        idx = numpy.arange(nocc)
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kijb = tasks[itask]
            ki, kj = (kijb).__divmod__(nkpts)
            kb = kconserv[ki,kshift,kj]
            ijb = numpy.zeros([nocc,nocc,nvir], dtype=dtype)
            ijb += lvv[kb][None,None,:]
            ijb -= loo[ki][:,None,None]
            ijb -= loo[kj][None,:,None]
            ijb += wij[ki,kj][:,:,None]
            ijb -= wjb[kj,kb][None,:,:]
            ijb += 2*wjb2[kj,kb][None,:,:]
            if ki == kj:
                ijb[idx,idx] -= wjb2[kj,kb]
            ijb -= wib[ki,kb][:,None,:]
            off = ki * nkpts + kj
            Hr2.write(off*idx_ijb.size+idx_ijb, ijb.ravel())

        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[:,:,kshift]
        Hr2 -= 2.*ctf.einsum('IJijcb,JIjicb->IJijb', t2[:,:,kshift], Woovvtmp)
        Hr2 += ctf.einsum('IJijcb,IJijcb->IJijb', t2[:,:,kshift], Woovvtmp)
    return eom.amplitudes_to_vector(Hr1, Hr2)

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector, kshift)

    # Eq. (30)
    # 1p-1p block
    Hr1 =  einsum('ac,c->a',imds.Lvv,r1)
    # 1p-2p1h block
    Hr1 += einsum('ld,lad->a',2.*imds.Fov,r2)
    Hr1 += einsum('ld,lda->a',  -imds.Fov,r2)
    Hr1 += 2*einsum('alcd,lcd->a',imds.Wvovv,r2)
    Hr1 +=  -einsum('aldc,lcd->a',imds.Wvovv,r2)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        foo = imds.eris.foo
        fvv = imds.eris.fvv
        Hr2 +=  einsum('ac,jcb->jab',fvv,r2)
        Hr2 +=  einsum('bd,jad->jab',fvv,r2)
        Hr2 += -einsum('lj,lab->jab',foo,r2)
    elif eom.partition == 'full':
        Hr2 += eom._eaccsd_diag_matrix2*r2
    else:
        Hr2 +=  einsum('ac,jcb->jab',imds.Lvv,r2)
        Hr2 +=  einsum('bd,jad->jab',imds.Lvv,r2)
        Hr2 += -einsum('lj,lab->jab',imds.Loo,r2)
        Hr2 += 2*einsum('lbdj,lad->jab',imds.Wovvo,r2)
        Hr2 +=  -einsum('lbjd,lad->jab',imds.Wovov,r2)
        Hr2 +=  -einsum('lajc,lcb->jab',imds.Wovov,r2)
        Hr2 +=  -einsum('lbcj,lca->jab',imds.Wovvo,r2)

        Hr2 +=   einsum('abcd,jcd->jab',imds.Wvvvv,r2)
        tmp = (2*einsum('klcd,lcd->k',imds.Woovv,r2)
                -einsum('kldc,lcd->k',imds.Woovv,r2))
        Hr2 += -einsum('k,kjab->jab',tmp,imds.t2)

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def eaccsd_diag(eom, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    kpts, gvec = eom.kpts, eom._cc._scf.cell.reciprocal_vectors()
    t1, t2 = imds.t1, imds.t2
    dtype = t2.dtype
    nocc, nvir = t1.shape
    nkpts = len(kpts)
    kconserv = eom.kconserv
    sym1 = ['+', [kpts,], kpts[kshift], gvec]
    sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]

    Hr1array = imds.Lvv.diagonal()[kshift]
    Hr1 = tensor(Hr1array, sym1, symlib=eom.symlib)
    Hr2 = zeros([nocc,nvir,nvir], dtype, sym2, symlib=eom.symlib)

    tasks = mpi_helper.static_partition(range(nkpts**2))
    ntasks = max(comm.allgather(len(tasks)))
    idx_jab = numpy.arange(nocc*nvir*nvir)
    if eom.partition == 'mp':
        foo = imds.eris.foo.diagonal().to_nparray()
        fvv = imds.eris.fvv.diagonal().to_nparray()
        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kjab = tasks[itask]
            kj, ka = (kjab).__divmod__(nkpts)
            kb = kconserv[ki,ka,kshift]
            off = kj * nkpts + ka
            jab = numpy.zeros([nocc,nvir,nvir], dtype=dtype)
            jab += -foo[kj][:,None,None]
            jab += fvv[ka][None,:,None]
            jab += fvv[kb][None,None,:]
            Hr2.write(off*idx_jab.size+idx_jab, jab.ravel())
    else:
        idx = numpy.arange(nvir)
        loo = imds.Loo.diagonal().to_nparray()
        lvv = imds.Lvv.diagonal().to_nparray()
        wab = ctf.einsum("ABAabab->ABab", imds.Wvvvv.array).to_nparray()
        wjb = ctf.einsum('JBJjbjb->JBjb', imds.Wovov.array).to_nparray()
        wjb2 = ctf.einsum('JBBjbbj->JBjb', imds.Wovvo.array).to_nparray()
        wja = ctf.einsum('JAJjaja->JAja', imds.Wovov.array).to_nparray()

        for itask in range(ntasks):
            if itask >= len(tasks):
                Hr2.write([], [])
                continue
            kjab = tasks[itask]
            kj, ka = (kjab).__divmod__(nkpts)
            kb = kconserv[kj,ka,kshift]
            jab = numpy.zeros([nocc,nvir,nvir], dtype=dtype)
            jab -= loo[kj][:,None,None]
            jab += lvv[ka][None,:,None]
            jab += lvv[kb][None,None,:]
            jab += wab[ka,kb][None,:,:]
            jab -= wjb[kj,kb][:,None,:]
            jab += 2*wjb2[kj,kb][:,None,:]
            if ka == kb:
                jab[:,idx,idx] -= wjb2[kj,ka]
            jab -= wja[kj,ka][:,:,None]
            off = kj * nkpts + ka
            Hr2.write(off*idx_jab.size+idx_jab, jab.ravel())
        Hr2 -= 2*ctf.einsum('JAijab,JAijab->JAjab', t2[kshift], imds.Woovv[kshift])
        Woovvtmp = imds.Woovv.transpose(0,1,3,2)[kshift]
        Hr2 += ctf.einsum('JAijab,JAijab->JAjab', t2[kshift], Woovvtmp)

    return eom.amplitudes_to_vector(Hr1, Hr2)

class EOMIP(eom_rccsd.EOMIP):

    def __init__(self, mycc):
        self.kpts = mycc.kpts
        self.symlib = mycc.symlib
        self.t1, self.t2 = mycc.t1, mycc.t2
        self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(mycc)
        self.kconserv = mycc.khelper.kconserv
        eom_rccsd.EOMIP.__init__(self,mycc)
        #self.kpts = mycc.kpts
        #self.symlib = mycc.symlib
        #self.t1, self.t2 = mycc.t1, mycc.t2
        #self.nonzero_opadding, self.nonzero_vpadding = self.get_padding_k_idx(mycc)
        #self.kconserv = mycc.khelper.kconserv

    vector_to_amplitudes = vector_to_amplitudes_ip
    matvec = ipccsd_matvec
    get_diag = ipccsd_diag
    kernel = kernel
    ipccsd = kernel

    def get_padding_k_idx(self, mycc):
        return padding_k_idx(mycc, kind='split')

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    @property
    def nkpts(self):
        return len(self.kpts)

    def update_symlib(self,kshift):
        kpts, gvec = self.kpts, self._cc._scf.cell.reciprocal_vectors()
        sym1 = ['+', [kpts,], kpts[kshift], gvec]
        sym2 = ['++-', [kpts,]*3, kpts[kshift], gvec]
        self.symlib.update(sym1,sym2)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda x: self.matvec(x, kshift, imds, diag)
        return matvec, diag

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.complex)
        nroots = min(nroots, size)
        guess = ctf.zeros([nroots, int(size)], dtype=dtype)
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(self.nonzero_opadding[kshift][::-1][:nroots])]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None)
            idx = mpi_helper.argsort(diag, nroots)
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = numpy.ones(nroots)
        if rank==0:
            guess.write(ind, fill)
        else:
            guess.write([], [])
        return guess

class EOMEA(EOMIP):

    vector_to_amplitudes = vector_to_amplitudes_ea
    amplitudes_to_vector = amplitudes_to_vector
    matvec = eaccsd_matvec
    get_diag = eaccsd_diag
    eaccsd = kernel
    kernel = kernel

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nvir + nkpts**2*nocc*nvir*nvir

    def update_symlib(self,kshift):
        kpts, gvec = self.kpts, self._cc._scf.cell.reciprocal_vectors()
        sym1 = ['+', [kpts,], kpts[kshift], gvec]
        sym2 = ['-++', [kpts,]*3, kpts[kshift], gvec]
        self.symlib.update(sym1,sym2)

    def get_init_guess(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.complex)
        nroots = min(nroots, size)
        guess = ctf.zeros([nroots, int(size)], dtype=dtype)
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(self.nonzero_vpadding[kshift][:nroots])]
        else:
            if diag is None:
                diag = self.get_diag(kshift, imds=None)
            idx = mpi_helper.argsort(diag, nroots)
            ind = [kn*size+n for kn, n in enumerate(idx)]
        fill = numpy.ones(nroots)
        if rank==0:
            guess.write(ind, fill)
        else:
            guess.write([],[])
        return guess

    def make_imds(self, eris=None):
        imds = eom_rccsd._IMDS(self._cc, eris=eris)
        imds.make_ea(self.partition)
        return imds

    @property
    def eea(self):
        return self.e


if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.ctfcc.kccsd_rhf import KRCCSD
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [13,13,13]
    cell.verbose = 5
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell,kpts, exxdiv=None)
    if rank ==0:
        mf.kernel()
    mycc = KRCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    eip = myeom.ipccsd(nroots=3, kptlist=[1], koopmans=True)[1].ravel()
    myeom = EOMEA(mycc)
    eea = myeom.eaccsd(nroots=3, kptlist=[1], koopmans=True)[1].ravel()
    print(abs(eip[0] - -0.53939626))
    print(abs(eip[1] - -0.53939629))
    print(abs(eip[2] - -0.31601216))
     
    print(abs(eea[0] - 1.14702038))
    print(abs(eea[1] - 1.16075426))
    print(abs(eea[2] - 1.16075426))
