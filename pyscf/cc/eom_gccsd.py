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

import time
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import eom_rccsd
from pyscf.cc import gintermediates as imd


########################################
# EOM-IP-CCSD
########################################

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = np.zeros((nocc,nocc,nvir), dtype=vector.dtype)
    idx, idy = np.tril_indices(nocc, -1)
    r2[idx,idy] = vector[nocc:].reshape(nocc*(nocc-1)//2,nvir)
    r2[idy,idx] =-vector[nocc:].reshape(nocc*(nocc-1)//2,nvir)
    return r1, r2

def amplitudes_to_vector_ip(r1, r2):
    nocc = r1.size
    return np.hstack((r1, r2[np.tril_indices(nocc, -1)].ravel()))

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    # Eq. (8)
    Hr1 = -np.einsum('mi,m->i', imds.Foo, r1)
    Hr1 += np.einsum('me,mie->i', imds.Fov, r2)
    Hr1 += -0.5*np.einsum('nmie,mne->i', imds.Wooov, r2)
    # Eq. (9)
    Hr2 =  lib.einsum('ae,ije->ija', imds.Fvv, r2)
    tmp1 = lib.einsum('mi,mja->ija', imds.Foo, r2)
    Hr2 -= tmp1 - tmp1.transpose(1,0,2)
    Hr2 -= np.einsum('maji,m->ija', imds.Wovoo, r1)
    Hr2 += 0.5*lib.einsum('mnij,mna->ija', imds.Woooo, r2)
    tmp2 = lib.einsum('maei,mje->ija', imds.Wovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(1,0,2)
    Hr2 += 0.5*lib.einsum('mnef,mnf,ijae->ija', imds.Woovv, r2, imds.t2)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = -np.diag(imds.Foo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype=t1.dtype)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                Hr2[i,j,a] += imds.Fvv[a,a]
                Hr2[i,j,a] += -imds.Foo[i,i]
                Hr2[i,j,a] += -imds.Foo[j,j]
                Hr2[i,j,a] += 0.5*(imds.Woooo[i,j,i,j]-imds.Woooo[j,i,i,j])
                Hr2[i,j,a] += imds.Wovvo[i,a,a,i]
                Hr2[i,j,a] += imds.Wovvo[j,a,a,j]
                Hr2[i,j,a] += 0.5*(np.dot(imds.Woovv[i,j,:,a], t2[i,j,a,:])
                                  -np.dot(imds.Woovv[j,i,:,a], t2[i,j,a,:]))

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector


class EOMIP(eom_rccsd.EOMIP):
    matvec = ipccsd_matvec
    l_matvec = None
    get_diag = ipccsd_diag
    ipccsd_star = None

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ip(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*(nocc-1)/2*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds

########################################
# EOM-EA-CCSD
########################################

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nvir].copy()
    r2 = np.zeros((nocc,nvir,nvir), vector.dtype)
    idx, idy = np.tril_indices(nvir, -1)
    r2[:,idx,idy] = vector[nvir:].reshape(nocc,-1)
    r2[:,idy,idx] =-vector[nvir:].reshape(nocc,-1)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    nvir = r1.size
    idx, idy = np.tril_indices(nvir, -1)
    return np.hstack((r1, r2[:,idx,idy].ravel()))

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    # Eq. (30)
    Hr1  = np.einsum('ac,c->a', imds.Fvv, r1)
    Hr1 += np.einsum('ld,lad->a', imds.Fov, r2)
    Hr1 += 0.5*np.einsum('alcd,lcd->a', imds.Wvovv, r2)
    # Eq. (31)
    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)
    tmp1 = lib.einsum('ac,jcb->jab', imds.Fvv, r2)
    Hr2 += tmp1 - tmp1.transpose(0,2,1)
    Hr2 -= lib.einsum('lj,lab->jab', imds.Foo, r2)
    tmp2 = lib.einsum('lbdj,lad->jab', imds.Wovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(0,2,1)
    Hr2 += 0.5*lib.einsum('abcd,jcd->jab', imds.Wvvvv, r2)
    Hr2 -= 0.5*lib.einsum('klcd,lcd,kjab->jab', imds.Woovv, r2, imds.t2)

    vector = amplitudes_to_vector_ea(Hr1, Hr2)
    return vector

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = np.diag(imds.Fvv)
    Hr2 = np.zeros((nocc,nvir,nvir),dtype=t1.dtype)
    for a in range(nvir):
        _Wvvvva = np.array(imds.Wvvvv[a])
        for b in range(a):
            for j in range(nocc):
               Hr2[j,a,b] += imds.Fvv[a,a]
               Hr2[j,a,b] += imds.Fvv[b,b]
               Hr2[j,a,b] += -imds.Foo[j,j]
               Hr2[j,a,b] += imds.Wovvo[j,b,b,j]
               Hr2[j,a,b] += imds.Wovvo[j,a,a,j]
               Hr2[j,a,b] += 0.5*(_Wvvvva[b,a,b]-_Wvvvva[b,b,a])
               Hr2[j,a,b] += -0.5*(np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b])
                                  -np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b]))

    vector = amplitudes_to_vector_ea(Hr1, Hr2)
    return vector


class EOMEA(eom_rccsd.EOMEA):
    matvec = eaccsd_matvec
    l_matvec = None
    get_diag = eaccsd_diag
    eaccsd_star = None

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ea(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nocc*nvir*(nvir-1)//2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ea()
        return imds


########################################
# EOM-EE-CCSD
########################################

vector_to_amplitudes_ee = ccsd.vector_to_amplitudes_s4
amplitudes_to_vector_ee = ccsd.amplitudes_to_vector_s4

def eeccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Wang, Tu, and Wang, J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10)
    # Note: Last line in Eq. (10) is superfluous.
    # See, e.g. Gwaltney, Nooijen, and Barlett, Chem. Phys. Lett. 248, 189 (1996)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ee(vector, nmo, nocc)

    # Eq. (9)
    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += lib.einsum('me,imae->ia', imds.Fov, r2)
    Hr1 += lib.einsum('maei,me->ia', imds.Wovvo, r1)
    Hr1 -= 0.5*lib.einsum('mnie,mnae->ia', imds.Wooov, r2)
    Hr1 += 0.5*lib.einsum('amef,imef->ia', imds.Wvovv, r2)
    # Eq. (10)
    tmpab = lib.einsum('be,ijae->ijab', imds.Fvv, r2)
    tmpab -= 0.5*lib.einsum('mnef,ijae,mnbf->ijab', imds.Woovv, imds.t2, r2)
    tmpab -= lib.einsum('mbij,ma->ijab', imds.Wovoo, r1)
    tmpab -= lib.einsum('amef,ijfb,me->ijab', imds.Wvovv, imds.t2, r1)
    tmpij  = lib.einsum('mj,imab->ijab', -imds.Foo, r2)
    tmpij -= 0.5*lib.einsum('mnef,imab,jnef->ijab', imds.Woovv, imds.t2, r2)
    tmpij += lib.einsum('abej,ie->ijab', imds.Wvvvo, r1)
    tmpij += lib.einsum('mnie,njab,me->ijab', imds.Wooov, imds.t2, r1)

    tmpabij = lib.einsum('mbej,imae->ijab', imds.Wovvo, r2)
    tmpabij = tmpabij - tmpabij.transpose(1,0,2,3)
    tmpabij = tmpabij - tmpabij.transpose(0,1,3,2)
    Hr2 = tmpabij

    Hr2 += tmpab - tmpab.transpose(0,1,3,2)
    Hr2 += tmpij - tmpij.transpose(1,0,2,3)
    Hr2 += 0.5*lib.einsum('mnij,mnab->ijab', imds.Woooo, r2)
    Hr2 += 0.5*lib.einsum('abef,ijef->ijab', imds.Wvvvv, r2)

    vector = amplitudes_to_vector_ee(Hr1, Hr2)
    return vector

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = np.zeros((nocc,nvir), dtype=t1.dtype)
    Hr2 = np.zeros((nocc,nocc,nvir,nvir), dtype=t1.dtype)
    for i in range(nocc):
        for a in range(nvir):
            Hr1[i,a] = imds.Fvv[a,a] - imds.Foo[i,i] + imds.Wovvo[i,a,a,i]
    for a in range(nvir):
        tmp = 0.5*(np.einsum('ijeb,ijbe->ijb', imds.Woovv, t2)
                  -np.einsum('jieb,ijbe->ijb', imds.Woovv, t2))
        Hr2[:,:,:,a] += imds.Fvv[a,a] + tmp
        Hr2[:,:,a,:] += imds.Fvv[a,a] + tmp
        _Wvvvva = np.array(imds.Wvvvv[a])
        for b in range(a):
            Hr2[:,:,a,b] += 0.5*(_Wvvvva[b,a,b]-_Wvvvva[b,b,a])
        for i in range(nocc):
            tmp = imds.Wovvo[i,a,a,i]
            Hr2[:,i,:,a] += tmp
            Hr2[i,:,:,a] += tmp
            Hr2[:,i,a,:] += tmp
            Hr2[i,:,a,:] += tmp
    for i in range(nocc):
        tmp = 0.5*(np.einsum('kjab,jkab->jab', imds.Woovv, t2)
                  -np.einsum('kjba,jkab->jab', imds.Woovv, t2))
        Hr2[:,i,:,:] += -imds.Foo[i,i] + tmp
        Hr2[i,:,:,:] += -imds.Foo[i,i] + tmp
        for j in range(i):
            Hr2[i,j,:,:] += 0.5*(imds.Woooo[i,j,i,j]-imds.Woooo[j,i,i,j])

    vector = amplitudes_to_vector_ee(Hr1, Hr2)
    return vector


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
    return eom_rccsd.eomee_ccsd_singlet(eom, nroots, koopmans, guess, eris, imds)


class EOMEE(eom_rccsd.EOMEE):

    kernel = eeccsd
    eeccsd = eeccsd
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag

    def gen_matvec(self, imds=None, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ee(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ee(r1, r2)

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc*nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
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
        self.Wvvvo = imd.Wvvvo(t1, t2, eris,self.Wvvvv)

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
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc import gccsd
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    e,v = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    #mycc.verbose = 5
    e,v = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)

    e,v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
