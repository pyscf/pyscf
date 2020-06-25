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
import time
from pyscf.lib import logger
from pyscf.cc.eom_rccsd import EOM
import pyscf.ctfcc.rintermediates as imd
from pyscf.ctfcc import mpi_helper
from pyscf.ctfcc.linalg_helper import davidson
from symtensor.sym_ctf import tensor, einsum

comm = mpi_helper.comm
rank = mpi_helper.rank

def kernel(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, **kwargs):

    cput0 = (time.clock(), time.time())
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    size = eom.vector_size()
    nroots = min(nroots,size)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)
    user_guess = False
    if guess:
        user_guess = True
        assert len(guess) == nroots
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)
    conv, evals, evecs = davidson.davidson(matvec, size, nroots, x0=guess, Adiag=diag, verbose=eom.verbose)
    evals = evals.real
    for n, en, vn in zip(range(nroots), evals, evecs):
        r1, r2 = eom.vector_to_amplitudes(vn)
        if isinstance(r1, tuple):
            qp_weight = sum([ri.norm()**2 for ri in r1])
        else:
            qp_weight = r1.norm()**2
        logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)
    logger.timer(eom, 'EOM-CCSD', *cput0)
    return conv, evals, evecs


def amplitudes_to_vector(eom, r1, r2):
    vector = ctf.hstack((r1.array.ravel(), r2.array.ravel()))
    return vector

def vector_to_amplitudes_ip(eom, vector):
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1 = vector[:nocc]
    r2 = vector[nocc:].reshape(nocc,nocc,nvir)
    r1 = tensor(r1, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2, verbose=eom.SYMVERBOSE)
    return (r1,r2)

def vector_to_amplitudes_ea(eom, vector):
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1 = vector[:nvir]
    r2 = vector[nvir:].reshape(nocc,nvir,nvir)
    r1 = tensor(r1, verbose=eom.SYMVERBOSE)
    r2 = tensor(r2, verbose=eom.SYMVERBOSE)
    return (r1, r2)

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector)

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

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    foo = imds.eris.foo
    fvv = imds.eris.fvv

    Hr1 = - imds.Loo.diagonal()
    Hr1 = tensor(Hr1)
    if eom.partition == 'mp':
        ijb = - foo.diagonal().reshape(nocc,1,1)
        ijb = ijb - foo.diagonal().reshape(1,nocc,1)
        ijb = ijb + fvv.diagonal().reshape(1,1,nvir)
        Hr2 = tensor(ijb)
    else:
        wij = ctf.einsum('ijij->ij', imds.Woooo.array)
        wjb = ctf.einsum('jbjb->jb', imds.Wovov.array)
        wjb2 = ctf.einsum('jbbj->jb', imds.Wovvo.array)
        wib = ctf.einsum('ibib->ib', imds.Wovov.array)
        ijb  = imds.Lvv.diagonal().reshape(1,1,nvir)
        ijb = ijb - imds.Loo.diagonal().reshape(nocc,1,1)
        ijb = ijb -imds.Loo.diagonal().reshape(1,nocc,1)
        ijb = ijb + wij.reshape(nocc,nocc,1)
        ijb = ijb - wjb.reshape(1,nocc,nvir)
        ijb = ijb + 2*wjb2.reshape(1,nocc,nvir)
        ijb = ijb - ctf.einsum('jb->jjb', wjb2)
        ijb = ijb - wib.reshape(nocc,1,nvir)
        Hr2 = tensor(ijb)
        Hr2 -= 2.*einsum('ijcb,jibc->ijb', t2, imds.Woovv)
        Hr2 += einsum('ijcb,ijbc->ijb', t2, imds.Woovv)
    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector



def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    r1,r2 = eom.vector_to_amplitudes(vector)

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

def eaccsd_diag(eom, imds=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    foo = imds.eris.foo
    fvv = imds.eris.fvv
    Hr1 = imds.Lvv.diagonal()
    Hr1 = tensor(Hr1)

    if eom.partition == 'mp':
        jab = fvv.diagonal().reshape(1,-1,1)
        jab+= fvv.diagonal().reshape(1,1,-1)
        jab-= jab - foo.diagonal().reshape(-1,1,1)
        Hr2 = tensor(jab)
    else:
        jab = imds.Lvv.diagonal().reshape(1,nvir,1)
        jab = jab + imds.Lvv.diagonal().reshape(1,1,nvir)
        jab = jab - imds.Loo.diagonal().reshape(nocc,1,1)
        wab = ctf.einsum("abab->ab", imds.Wvvvv.array)
        wjb = ctf.einsum('jbjb->jb', imds.Wovov.array)
        wjb2 = ctf.einsum('jbbj->jb', imds.Wovvo.array)
        wja = ctf.einsum('jaja->ja', imds.Wovov.array)
        jab = jab + wab.reshape(1,nvir,nvir)
        jab = jab - wjb.reshape(nocc,1,nvir)
        jab = jab + 2*wjb2.reshape(nocc,1,nvir)
        jab -= ctf.einsum('jb,ab->jab', wjb2, ctf.eye(nvir))
        jab = jab - wja.reshape(nocc,nvir,1)
        Hr2 = tensor(jab)
        Hr2 -= 2*einsum('ijab,ijab->jab', t2, imds.Woovv)
        Hr2 += einsum('ijab,ijba->jab', t2, imds.Woovv)
    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

class EOMIP(EOM):
    def __init__(self, mycc):
        EOM.__init__(self, mycc)
        self.symlib = mycc.symlib
        self.t1, self.t2 = mycc.t1, mycc.t2
        self.SYMVERBOSE = mycc.SYMVERBOSE

    amplitudes_to_vector = amplitudes_to_vector
    vector_to_amplitudes = vector_to_amplitudes_ip
    matvec = ipccsd_matvec
    get_diag = ipccsd_diag

    kernel = kernel
    ipccsd = kernel

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc**2*nvir

    def make_imds(self, eris=None):
        self.imds = imds = _IMDS(self._cc, eris=eris)
        imds.make_ip(self.partition)
        return imds

    @property
    def eip(self):
        return self.e

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        if left:
            matvec = lambda x: self.l_matvec(x, imds, diag)
        else:
            matvec = lambda x: self.matvec(x, imds, diag)
        return matvec, diag

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.double)
        nroots = min(nroots, size)
        if koopmans:
            idx = range(self.nocc-nroots, self.nocc)[::-1]
        else:
            if diag is None: diag = self.get_diag()
            idx = mpi_helper.argsort(diag, nroots)
        guess = ctf.zeros([nroots,size], dtype)
        idx = numpy.arange(nroots)*size + numpy.asarray(idx)
        if rank==0:
            guess.write(idx, numpy.ones(nroots))
        else:
            guess.write([],[])
        return guess

class EOMEA(EOMIP):

    vector_to_amplitudes = vector_to_amplitudes_ea
    matvec = eaccsd_matvec
    get_diag = eaccsd_diag
    eaccsd = kernel

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nocc*nvir**2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ea(self.partition)
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

    @property
    def eea(self):
        return self.e

class _IMDS:
    def __init__(self, mycc, eris=None):
        self.t1 = mycc.t1
        self.t2 = mycc.t2
        self.stdout, self.verbose= mycc.stdout, mycc.verbose
        if eris is None:
            if getattr(mycc, 'eris') is None:
                self.eris = eris = mycc.ao2mo()
            else:
                self.eris = mycc.eris
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        logger.timer(self, 'EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.ovov.transpose(0,2,1,3)

        logger.timer(self, 'EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        logger.timer(self, 'EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        logger.timer(self, 'EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.ctfcc.rccsd import RCCSD
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    if rank==0:
        mf.kernel()

    mycc = RCCSD(mf)
    mycc.kernel()

    myeom = EOMIP(mycc)
    eip = myeom.ipccsd(nroots=2)[1]

    myeom = EOMEA(mycc)
    eea = myeom.eaccsd(nroots=2)[1]

    print(numpy.amax(eip[0]-0.4335604229241659))
    print(numpy.amax(eip[1]-0.5187659782655635))

    print(numpy.amax(eea[0]-0.1673788639606518))
    print(numpy.amax(eea[1]-0.2402762272383755))
