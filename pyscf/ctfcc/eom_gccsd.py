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
from pyscf.ctfcc import eom_rccsd, eom_uccsd
import pyscf.ctfcc.gintermediates as imd
from pyscf.ctfcc import mpi_helper
from symtensor.sym_ctf import tensor, einsum, zeros

comm = mpi_helper.comm
rank = mpi_helper.rank

def amplitudes_to_vector_ip(eom, r1, r2):
    r2vec = eom_uccsd.pack_tril_ip(r2.array)
    return ctf.hstack((r1.array.ravel(), r2vec))

def vector_to_amplitudes_ip(vector, nmo, nocc):
    r1 = tensor(vector[:nocc])
    r2 = tensor(eom_uccsd.unpack_tril_ip(vector[nocc:], nocc, nmo-nocc))
    return r1, r2

def amplitudes_to_vector_ea(eom, r1, r2):
    r2vec = eom_uccsd.pack_tril_ea(r2.array)
    return ctf.hstack((r1.array.ravel(), r2vec))

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nvir = nmo-nocc
    r1 = tensor(vector[:nvir])
    r2 = tensor(eom_uccsd.unpack_tril_ea(vector[nvir:], nocc, nvir))
    return r1, r2

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
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

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    Hr1 = tensor(-imds.Foo.diagonal())
    ija = imds.Fvv.diagonal().reshape(1,1,-1) \
        - imds.Foo.diagonal().reshape(-1,1,1) \
        - imds.Foo.diagonal().reshape(1,-1,1)

    wij = 0.5*(ctf.einsum('ijij->ij', imds.Woooo.array) \
        - ctf.einsum('jiij->ij', imds.Woooo.array))
    wia = ctf.einsum('iaai->ia', imds.Wovvo.array)
    ija+= wij.reshape(nocc,nocc,1) + wia.reshape(nocc,1,nvir) + wia.reshape(1,nocc,nvir)
    ija+= 0.5*(ctf.einsum('ijea,ijae->ija', imds.Woovv.array, t2.array) - \
               ctf.einsum('jiea,ijae->ija', imds.Woovv.array, t2.array))
    Hr2 = tensor(ija)
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

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

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    Hr1 = tensor(imds.Fvv.diagonal())
    jab = imds.Fvv.diagonal().reshape(1,-1,1)+imds.Fvv.diagonal().reshape(1,1,-1)-\
          imds.Foo.diagonal().reshape(-1,1,1)
    wja = ctf.einsum('jaaj->ja', imds.Wovvo.array)
    wab = 0.5*(ctf.einsum('abab->ab',imds.Wvvvv.array) - ctf.einsum('abba->ab',imds.Wvvvv.array))
    jab += wja.reshape(nocc,nvir,1) + wja.reshape(nocc,1,nvir) + wab.reshape(1,nvir,nvir)
    jab -= 0.5*(ctf.einsum('ejab,ejab->jab', imds.Woovv.array, t2.array)-\
                ctf.einsum('ejba,ejab->jab', imds.Woovv.array, t2.array))
    Hr2 = tensor(jab)
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMIP(eom_rccsd.EOMIP):
    matvec = ipccsd_matvec
    amplitudes_to_vector = amplitudes_to_vector_ip
    get_diag = ipccsd_diag
    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return  vector_to_amplitudes_ip(vector, nmo, nocc)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*(nocc-1)//2*nvir

    def make_imds(self, eris=None):
        self.imds = imds = _IMDS(self._cc, eris=eris)
        imds.make_ip()
        return imds

class EOMEA(eom_rccsd.EOMEA):
    matvec = eaccsd_matvec
    amplitudes_to_vector = amplitudes_to_vector_ea
    get_diag = eaccsd_diag
    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return  vector_to_amplitudes_ea(vector, nmo, nocc)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nvir*(nvir-1)//2*nocc

    def make_imds(self, eris=None):
        self.imds = imds = _IMDS(self._cc, eris=eris)
        imds.make_ea()
        return imds

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, mycc, eris=None):
        self.verbose = mycc.verbose
        self.stdout = mycc.stdout
        self.t1 = mycc.t1
        self.t2 = mycc.t2
        if eris is None:
            eris = mycc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo = imd.Foo(t1, t2, eris)
        self.Fvv = imd.Fvv(t1, t2, eris)
        self.Fov = imd.Fov(t1, t2, eris)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-CCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        self.Wvvvv = imd.Wvvvv(t1, t2, eris)
        self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.Woooo = imd.Woooo(t1, t2, eris)
            self.Wooov = imd.Wooov(t1, t2, eris)
            self.Wovoo = imd.Wovoo(t1, t2, eris)
        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(t1, t2, eris)
            self.Wvvvv = imd.Wvvvv(t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris,self.Wvvvv)

        self.made_ee_imds = True
        logger.timer(self, 'EOM-CCSD EE intermediates', *cput0)
        return self

if __name__ == '__main__':
    from pyscf import gto, scf, cc
    from pyscf.ctfcc import gccsd
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.verbose=4
    mol.build()
    mf = scf.UHF(mol)
    if rank==0:
        mf.run()
    mf = scf.addons.convert_to_ghf(mf)

    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)

    myip = EOMIP(mycc)
    eip = myip.ipccsd(nroots=3)[1]
    print(eip[0]-0.4335604447145262)
    print(eip[1]-0.4335604491514221)
    print(eip[2]-0.5187659567026139)

    myea = EOMEA(mycc)
    eea = myea.eaccsd(nroots=3)[1]
    print(eea[0] - 0.1673789735691068)
    print(eea[1] - 0.1673789812544622)
    print(eea[2] - 0.2402763379294816)
