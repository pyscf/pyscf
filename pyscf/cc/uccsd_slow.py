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

from functools import reduce

import numpy
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import uccsd
from pyscf.mp import ump2
from pyscf.cc import gintermediates as imd

#einsum = np.einsum
einsum = lib.einsum

# This is unrestricted (U)CCSD, i.e. spin-orbital form.


def update_amps(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv -= np.diag(np.diag(fvv))
    Foo -= np.diag(np.diag(foo))

    # T1 equation
    t1new = np.array(fov).conj()
    t1new +=  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)

    # T2 equation
    t2new = np.array(eris.oovv).conj()
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new += (tmp - tmp.transpose(0,1,3,2))
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= (tmp - tmp.transpose(1,0,2,3))
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    t2new += (tmp - tmp.transpose(0,1,3,2)
              - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2) )
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,mbij->ijab', t1, eris.ovoo)
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:] - cc.level_shift
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = einsum('ia,ia', fock[:nocc,nocc:], t1)
    e += 0.25*np.einsum('ijab,ijab', t2, eris.oovv)
    e += 0.5 *np.einsum('ia,jb,ijab', t1, t1, eris.oovv)
    return e.real


get_frozen_mask = ump2.get_frozen_mask


class UCCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        # Spin-orbital CCSD needs a stricter tolerance than spatial-orbital
        self.conv_tol_normt = 1e-6

    @property
    def nocc(self):
        nocca, noccb = self.get_nocc()
        return nocca + noccb

    @property
    def nmo(self):
        nmoa, nmob = self.get_nmo()
        return nmoa + nmob

    get_nocc = uccsd.get_nocc
    get_nmo = uccsd.get_nmo
    get_frozen_mask = get_frozen_mask

    def init_amps(self, eris):
        time0 = logger.process_clock(), logger.perf_counter()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        t1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = np.array(eris.oovv)
        t2 = eris_oovv/eijab
        self.emp2 = 0.25*einsum('ijab,ijab',t2,eris_oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unrestricted (U)CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if mbpt2:
            #cctyp = 'MBPT2'
            #self.e_corr, self.t1, self.t2 = self.init_amps(eris)
            raise NotImplementedError

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        return _PhysicistsERIs(self, mo_coeff)

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)

    def nip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nip = nocc + nocc*(nocc-1)//2*nvir
        return self._nip

    def nea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nea = nvir + nocc*nvir*(nvir-1)//2
        return self._nea

    def nee(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nee = nocc*nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        return self._nee

    def ipccsd_matvec(self, vector):
        # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        # Eq. (8)
        Hr1 = -einsum('mi,m->i',imds.Foo,r1)
        Hr1 += einsum('me,mie->i',imds.Fov,r2)
        Hr1 += -0.5*einsum('nmie,mne->i',imds.Wooov,r2)
        # Eq. (9)
        Hr2 =  einsum('ae,ije->ija',imds.Fvv,r2)
        tmp1 = einsum('mi,mja->ija',imds.Foo,r2)
        Hr2 += (-tmp1 + tmp1.transpose(1,0,2))
        Hr2 += -einsum('maji,m->ija',imds.Wovoo,r1)
        Hr2 += 0.5*einsum('mnij,mna->ija',imds.Woooo,r2)
        tmp2 = einsum('maei,mje->ija',imds.Wovvo,r2)
        Hr2 += (tmp2 - tmp2.transpose(1,0,2))
        Hr2 += 0.5*einsum('mnef,ijae,mnf->ija',imds.Woovv,self.t2,r2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape

        Hr1 = -np.diag(imds.Foo)
        Hr2 = np.zeros((nocc,nocc,nvir),dtype=t1.dtype)
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    Hr2[i,j,a] += imds.Fvv[a,a]
                    Hr2[i,j,a] += -imds.Foo[i,i]
                    Hr2[i,j,a] += -imds.Foo[j,j]
                    Hr2[i,j,a] += 0.5*(imds.Woooo[i,j,i,j]-imds.Woooo[j,i,i,j])
                    Hr2[i,j,a] += imds.Wovvo[i,a,a,i]
                    Hr2[i,j,a] += imds.Wovvo[j,a,a,j]
                    Hr2[i,j,a] += 0.5*(np.dot(imds.Woovv[i,j,:,a],t2[i,j,a,:])
                                       -np.dot(imds.Woovv[j,i,:,a],t2[i,j,a,:]))

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc].copy()
        r2 = np.zeros((nocc,nocc,nvir), vector.dtype)
        index = nocc
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    r2[i,j,a] =  vector[index]
                    r2[j,i,a] = -vector[index]
                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = nocc + nocc*(nocc-1)//2*nvir
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        index = nocc
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    vector[index] = r2[i,j,a]
                    index += 1
        return vector

    def eaccsd_matvec(self,vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        # Eq. (30)
        Hr1 = einsum('ac,c->a',imds.Fvv,r1)
        Hr1 += einsum('ld,lad->a',imds.Fov,r2)
        Hr1 += 0.5*einsum('alcd,lcd->a',imds.Wvovv,r2)
        # Eq. (31)
        Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
        tmp1 = einsum('ac,jcb->jab',imds.Fvv,r2)
        Hr2 += (tmp1 - tmp1.transpose(0,2,1))
        Hr2 += -einsum('lj,lab->jab',imds.Foo,r2)
        tmp2 = einsum('lbdj,lad->jab',imds.Wovvo,r2)
        Hr2 += (tmp2 - tmp2.transpose(0,2,1))
        nvir = self.nmo-self.nocc
        for a in range(nvir):
            Hr2[:,a,:] += 0.5*einsum('bcd,jcd->jb',imds.Wvvvv[a],r2)
        Hr2 += -0.5*einsum('klcd,lcd,kjab->jab',imds.Woovv,r2,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds

        t1, t2 = self.t1, self.t2
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
                    Hr2[j,a,b] -= 0.5*(np.dot(imds.Woovv[:,j,a,b],t2[:,j,a,b]) -
                                       np.dot(imds.Woovv[:,j,b,a],t2[:,j,a,b]))

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nvir].copy()
        r2 = np.zeros((nocc,nvir,nvir), vector.dtype)
        index = nvir
        for i in range(nocc):
            for a in range(nvir):
                for b in range(a):
                    r2[i,a,b] =  vector[index]
                    r2[i,b,a] = -vector[index]
                    index += 1
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = nvir + nvir*(nvir-1)//2*nocc
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        index = nvir
        for i in range(nocc):
            for a in range(nvir):
                for b in range(a):
                    vector[index] = r2[i,a,b]
                    index += 1
        return vector

    def eeccsd_matvec(self,vector):
        # Ref: Wang, Tu, and Wang, J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10)
        # Note: Last line in Eq. (10) is superfluous.
        # See, e.g. Gwaltney, Nooijen, and Barlett, Chem. Phys. Lett. 248, 189 (1996)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ee(vector)

        # Eq. (9)
        Hr1 = einsum('ae,ie->ia',imds.Fvv,r1)
        Hr1 += -einsum('mi,ma->ia',imds.Foo,r1)
        Hr1 += einsum('me,imae->ia',imds.Fov,r2)
        Hr1 += einsum('maei,me->ia',imds.Wovvo,r1)
        Hr1 += -0.5*einsum('mnie,mnae->ia',imds.Wooov,r2)
        Hr1 += 0.5*einsum('amef,imef->ia',imds.Wvovv,r2)
        # Eq. (10)
        tmpab = einsum('be,ijae->ijab',imds.Fvv,r2)
        tmpab += -0.5*einsum('mnef,ijae,mnbf->ijab',imds.Woovv,self.t2,r2)
        tmpab += -einsum('mbij,ma->ijab',imds.Wovoo,r1)
        tmpab += -einsum('amef,ijfb,me->ijab',imds.Wvovv,self.t2,r1)
        tmpij = -einsum('mj,imab->ijab',imds.Foo,r2)
        tmpij += -0.5*einsum('mnef,imab,jnef->ijab',imds.Woovv,self.t2,r2)
        tmpij += einsum('abej,ie->ijab',imds.Wvvvo,r1)
        tmpij += einsum('mnie,njab,me->ijab',imds.Wooov,self.t2,r1)

        tmpabij = einsum('mbej,imae->ijab',imds.Wovvo,r2)

        Hr2 = (tmpab - tmpab.transpose(0,1,3,2)
               + tmpij - tmpij.transpose(1,0,2,3)
               + 0.5*einsum('mnij,mnab->ijab',imds.Woooo,r2)
               + 0.5*einsum('abef,ijef->ijab',imds.Wvvvv,r2)
               + tmpabij - tmpabij.transpose(0,1,3,2)
               - tmpabij.transpose(1,0,2,3) + tmpabij.transpose(1,0,3,2) )

        vector = self.amplitudes_to_vector_ee(Hr1,Hr2)
        return vector

    def eeccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape

        Hr1 = np.zeros((nocc,nvir), dtype=t1.dtype)
        Hr2 = np.zeros((nocc,nocc,nvir,nvir), dtype=t1.dtype)
        for i in range(nocc):
            for a in range(nvir):
                Hr1[i,a] = imds.Fvv[a,a] - imds.Foo[i,i] + imds.Wovvo[i,a,a,i]
        for a in range(nvir):
            tmp = 0.5*(np.einsum('ijeb,ijbe->ijb', imds.Woovv, t2) -
                       np.einsum('jieb,ijbe->ijb', imds.Woovv, t2))
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
            tmp = 0.5*(np.einsum('kjab,jkab->jab', imds.Woovv, t2) -
                       np.einsum('kjba,jkab->jab', imds.Woovv, t2))
            Hr2[:,i,:,:] += -imds.Foo[i,i] + tmp
            Hr2[i,:,:,:] += -imds.Foo[i,i] + tmp
            for j in range(i):
                Hr2[i,j,:,:] += 0.5*(imds.Woooo[i,j,i,j]-imds.Woooo[j,i,i,j])

        vector = self.amplitudes_to_vector_ee(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ee(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc*nvir].copy().reshape((nocc,nvir))
        r2 = np.zeros((nocc,nocc,nvir,nvir), vector.dtype)
        index = nocc*nvir
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    for b in range(a):
                        r2[i,j,a,b] =  vector[index]
                        r2[j,i,a,b] = -vector[index]
                        r2[i,j,b,a] = -vector[index]
                        r2[j,i,b,a] =  vector[index]
                        index += 1
        return [r1,r2]

    def amplitudes_to_vector_ee(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = nocc*nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        vector = np.zeros(size, r1.dtype)
        vector[:nocc*nvir] = r1.copy().reshape(nocc*nvir)
        index = nocc*nvir
        for i in range(nocc):
            for j in range(i):
                for a in range(nvir):
                    for b in range(a):
                        vector[index] = r2[i,j,a,b]
                        index += 1
        return vector

    def amplitudes_from_rccsd(self, t1, t2):
        '''Convert spatial orbital T1,T2 to spin-orbital T1,T2'''
        nocc, nvir = t1.shape
        nocc2 = nocc * 2
        nvir2 = nvir * 2
        t1s = np.zeros((nocc2,nvir2))
        t1s[:nocc,:nvir] = t1
        t1s[nocc:,nvir:] = t1

        t2s = np.zeros((nocc2,nocc2,nvir2,nvir2))
        t2s[:nocc,nocc:,:nvir,nvir:] = t2
        t2s[nocc:,:nocc,nvir:,:nvir] = t2
        t2s[:nocc,nocc:,nvir:,:nvir] =-t2.transpose(0,1,3,2)
        t2s[nocc:,:nocc,:nvir,nvir:] =-t2.transpose(0,1,3,2)
        t2s[:nocc,:nocc,:nvir,:nvir] = t2 - t2.transpose(0,1,3,2)
        t2s[nocc:,nocc:,nvir:,nvir:] = t2 - t2.transpose(0,1,3,2)
        return t1s, t2s


class _PhysicistsERIs:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.outcore.general_iofree):
        cput0 = (logger.process_clock(), logger.perf_counter())
        moidx = cc.get_frozen_mask()
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = [cc.mo_coeff[0][:,moidx[0]],
                                        cc.mo_coeff[1][:,moidx[1]]]
        else:
            self.mo_coeff = mo_coeff = [mo_coeff[0][:,moidx[0]],
                                        mo_coeff[1][:,moidx[1]]]

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore = nmo**4*2 * 8/1e6
        mem_now = lib.current_memory()[0]

        self.fock, so_coeff, spin = uspatial2spin(cc, moidx, mo_coeff)

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            eri = ao2mofn(cc._scf.mol, (so_coeff,so_coeff,so_coeff,so_coeff), compact=0)
            if mo_coeff[0].dtype == np.double: eri = eri.real
            eri = eri.reshape((nmo,)*4)
            for i in range(nmo):
                for j in range(i):
                    if spin[i] != spin[j]:
                        eri[i,j,:,:] = eri[j,i,:,:] = 0.
                        eri[:,:,i,j] = eri[:,:,j,i] = 0.
            eri1 = eri - eri.transpose(0,3,2,1)
            eri1 = eri1.transpose(0,2,1,3)

            self.dtype = eri1.dtype
            self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
            self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        else:
            self.feri1 = lib.H5TmpFile()
            orbo = so_coeff[:,:nocc]
            orbv = so_coeff[:,nocc:]
            if mo_coeff[0].dtype == np.complex128: ds_type = 'c16'
            else: ds_type = 'f8'
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), ds_type)
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), ds_type)
            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            cput1 = logger.process_clock(), logger.perf_counter()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            buf = ao2mofn(cc._scf.mol, (orbo,so_coeff,orbo,so_coeff), compact=0)
            if mo_coeff[0].dtype == np.double: buf = buf.real
            buf = buf.reshape((nocc,nmo,nocc,nmo))
            for i in range(nocc):
                for p in range(nmo):
                    if spin[i] != spin[p]:
                        buf[i,p,:,:] = 0.
                        buf[:,:,i,p] = 0.
            buf1 = buf - buf.transpose(0,3,2,1)
            buf1 = buf1.transpose(0,2,1,3)
            cput1 = log.timer_debug1('transforming oopq', *cput1)
            self.dtype = buf1.dtype
            self.oooo[:,:,:,:] = buf1[:,:,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf1[:,:,:nocc,nocc:]
            self.oovv[:,:,:,:] = buf1[:,:,nocc:,nocc:]

            cput1 = logger.process_clock(), logger.perf_counter()
            # <ia||pq> = <ia|pq> - <ia|qp> = (ip|aq) - (iq|ap)
            buf = ao2mofn(cc._scf.mol, (orbo,so_coeff,orbv,so_coeff), compact=0)
            if mo_coeff[0].dtype == np.double: buf = buf.real
            buf = buf.reshape((nocc,nmo,nvir,nmo))
            for p in range(nmo):
                for i in range(nocc):
                    if spin[i] != spin[p]:
                        buf[i,p,:,:] = 0.
                for a in range(nvir):
                    if spin[nocc+a] != spin[p]:
                        buf[:,:,a,p] = 0.
            buf1 = buf - buf.transpose(0,3,2,1)
            buf1 = buf1.transpose(0,2,1,3)
            cput1 = log.timer_debug1('transforming ovpq', *cput1)
            self.ovoo[:,:,:,:] = buf1[:,:,:nocc,:nocc]
            self.ovov[:,:,:,:] = buf1[:,:,:nocc,nocc:]
            self.ovvv[:,:,:,:] = buf1[:,:,nocc:,nocc:]

            for a in range(nvir):
                orbva = orbv[:,a].reshape(-1,1)
                buf = ao2mofn(cc._scf.mol, (orbva,orbv,orbv,orbv), compact=0)
                if mo_coeff[0].dtype == np.double: buf = buf.real
                buf = buf.reshape((1,nvir,nvir,nvir))
                for b in range(nvir):
                    if spin[nocc+a] != spin[nocc+b]:
                        buf[0,b,:,:] = 0.
                    for c in range(nvir):
                        if spin[nocc+b] != spin[nocc+c]:
                            buf[:,:,b,c] = buf[:,:,c,b] = 0.
                buf1 = buf - buf.transpose(0,3,2,1)
                buf1 = buf1.transpose(0,2,1,3)
                self.vvvv[a] = buf1[:]

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)


def uspatial2spin(cc, moidx, mo_coeff):
    '''Convert the results of an unrestricted mean-field calculation to spin-orbital form.

    Spin-orbital ordering is determined by orbital energy without regard for spin.

    Returns:
        fock : (nso,nso) ndarray
            The Fock matrix in the basis of spin-orbitals
        so_coeff : (nao, nso) ndarray
            The matrix of spin-orbital coefficients in the AO basis
        spin : (nso,) ndarary
            The spin (0 or 1) of each spin-orbital
    '''

    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
    fockab = list()
    for a in range(2):
        fockab.append( reduce(numpy.dot, (mo_coeff[a].T, fockao[a], mo_coeff[a])) )

    nocc = cc.nocc
    nao = cc.mo_coeff[0].shape[0]
    nmo = cc.nmo
    so_coeff = np.zeros((nao, nmo), dtype=mo_coeff[0].dtype)
    nocc_a = int(sum(cc.mo_occ[0]*moidx[0]))
    nocc_b = int(sum(cc.mo_occ[1]*moidx[1]))
    nmo_a = fockab[0].shape[0]
    nmo_b = fockab[1].shape[0]
    nvir_a = nmo_a - nocc_a
    #nvir_b = nmo_b - nocc_b
    oa = range(0,nocc_a)
    ob = range(nocc_a,nocc)
    va = range(nocc,nocc+nvir_a)
    vb = range(nocc+nvir_a,nmo)
    spin = np.zeros(nmo, dtype=int)
    spin[oa] = 0
    spin[ob] = 1
    spin[va] = 0
    spin[vb] = 1
    so_coeff[:,oa] = mo_coeff[0][:,:nocc_a]
    so_coeff[:,ob] = mo_coeff[1][:,:nocc_b]
    so_coeff[:,va] = mo_coeff[0][:,nocc_a:nmo_a]
    so_coeff[:,vb] = mo_coeff[1][:,nocc_b:nmo_b]

    fock = np.zeros((nmo, nmo), dtype=fockab[0].dtype)
    fock[np.ix_(oa,oa)] = fockab[0][:nocc_a,:nocc_a]
    fock[np.ix_(oa,va)] = fockab[0][:nocc_a,nocc_a:]
    fock[np.ix_(va,oa)] = fockab[0][nocc_a:,:nocc_a]
    fock[np.ix_(va,va)] = fockab[0][nocc_a:,nocc_a:]
    fock[np.ix_(ob,ob)] = fockab[1][:nocc_b,:nocc_b]
    fock[np.ix_(ob,vb)] = fockab[1][:nocc_b,nocc_b:]
    fock[np.ix_(vb,ob)] = fockab[1][nocc_b:,:nocc_b]
    fock[np.ix_(vb,vb)] = fockab[1][nocc_b:,nocc_b:]

# Do not sort because it's different to the orbital ordering generated by
# get_frozen_mask function in AO-direct vvvv contraction
#    idxo = np.diagonal(fock[:nocc,:nocc]).argsort()
#    idxv = nocc + np.diagonal(fock[nocc:,nocc:]).argsort()
#    idx = np.concatenate((idxo,idxv))
#    spin = spin[idx]
#    so_coeff = so_coeff[:,idx]
#    fock = fock[:, idx][idx]

    return fock, so_coeff, spin


class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> uintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc):
        self.cc = cc
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris
        self.Foo = imd.Foo(t1,t2,eris)
        self.Fvv = imd.Fvv(t1,t2,eris)
        self.Fov = imd.Fov(t1,t2,eris)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.oovv

        log.timer('EOM-CCSD shared intermediates', *cput0)

    def make_ip(self):
        if self._made_shared is False:
            self._make_shared()
            self._made_shared = True

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)

        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self):
        if self._made_shared is False:
            self._make_shared()
            self._made_shared = True

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        self.Wvvvv = imd.Wvvvv(t1,t2,eris)
        self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)

        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        if self._made_shared is False:
            self._make_shared()
            self._made_shared = True

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        if self.made_ip_imds is False:
            # 0 or 1 virtuals
            self.Woooo = imd.Woooo(t1,t2,eris)
            self.Wooov = imd.Wooov(t1,t2,eris)
            self.Wovoo = imd.Wovoo(t1,t2,eris)
        if self.made_ea_imds is False:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(t1,t2,eris)
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol)
    print(mf.scf())

    # Freeze 1s electrons
    frozen = [[0,1], [0,1]]
    # also acceptable
    #frozen = 4
    ucc = UCCSD(mf, frozen=frozen)
    ecc, t1, t2 = ucc.kernel()
    print(ecc - -0.3486987472235819)
    exit()

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol)
    print(mf.scf())

    mycc = UCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    e,v = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    mycc.verbose = 5
    e,v = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)
    print("e=", e)

    e,v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
