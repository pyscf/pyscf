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

'''
Restricted CCSD

Ref: Stanton et al., J. Chem. Phys. 94, 4334 (1990)
Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

from functools import reduce

import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.lib import linalg_helper

#einsum = np.einsum
einsum = lib.einsum

# note MO integrals are treated in chemist's notation

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= np.diag(np.diag(foo))
    Fvv -= np.diag(np.diag(fvv))

    # T1 equation
    t1new = np.asarray(fov).conj().copy()
    t1new +=-2*einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -einsum('ki,ka->ia', Foo, t1)
    t1new += 2*einsum('kc,kica->ia', Fov, t2)
    t1new +=  -einsum('kc,ikca->ia', Fov, t2)
    t1new +=   einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += 2*einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    t1new += 2*einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=-2*einsum('kilc,klac->ia', eris.ooov, t2)
    t1new +=   einsum('likc,klac->ia', eris.ooov, t2)
    t1new +=-2*einsum('kilc,lc,ka->ia', eris.ooov, t1, t1)
    t1new +=   einsum('likc,lc,ka->ia', eris.ooov, t1, t1)

    # T2 equation
    t2new = np.asarray(eris.ovov).conj().transpose(0,2,1,3).copy()
    if cc.cc2:
        Woooo2 = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
        Woooo2 += einsum('kilc,jc->klij', eris.ooov, t1)
        Woooo2 += einsum('ljkc,ic->klij', eris.ooov, t1)
        Woooo2 += einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
        t2new += einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv = einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv += np.asarray(eris.vvvv).transpose(0,2,1,3)
        t2new += einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = fvv - einsum('kc,ka->ac', fov, t1)
        Lvv2 -= np.diag(np.diag(fvv))
        tmp = einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + einsum('kc,ic->ki', fov, t1)
        Loo2 -= np.diag(np.diag(foo))
        tmp = einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo -= np.diag(np.diag(foo))
        Lvv -= np.diag(np.diag(fvv))
        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        tau = t2 + einsum('ia,jb->ijab', t1, t1)
        t2new += einsum('klij,klab->ijab', Woooo, tau)
        t2new += einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += np.asarray(eris.ovvv).conj().transpose(1,3,0,2)
    tmp = einsum('abic,jc->ijab', tmp2, t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += np.asarray(eris.ooov).transpose(3,1,2,0).conj()
    tmp = einsum('akij,kb->ijab', tmp2, t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*einsum('ia,ia', fock[:nocc,nocc:], t1)
    tau = einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*einsum('ijab,iajb', tau, eris_ovov)
    e +=  -einsum('ijab,ibja', tau, eris_ovov)
    return e.real


class RCCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])

    def dump_flags(self, verbose=None):
        ccsd.CCSD.dump_flags(self, verbose)

    def init_amps(self, eris):
        nocc = self.nocc
        mo_e = eris.fock.diagonal().real
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        t1 = eris.fock[:nocc,nocc:].conj() / eia
        eris_ovov = np.asarray(eris.ovov)
        t2 = eris_ovov.transpose(0,2,1,3).conj() / eijab
        self.emp2  = 2*einsum('ijab,iajb', t2, eris_ovov)
        self.emp2 -=   einsum('ijab,ibja', t2, eris_ovov)
        lib.logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2


    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        return self.ccsd(t1, t2, eris, mbpt2, cc2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
            cc2 : bool
                Use CC2 approximation to CCSD.
        '''
        if mbpt2 and cc2:
            raise RuntimeError('MBPT2 and CC2 are mutually exclusive approximations to the CCSD ground state.')
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.e_hf = self.get_e_hf()
        self.eris = eris
        self.dump_flags()
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            if cc2:
                cctyp = 'CC2'
                self.cc2 = True
            else:
                cctyp = 'CCSD'
                self.cc2 = False
            self.converged, self.e_corr, self.t1, self.t2 = \
                    ccsd.kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                                tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                                verbose=self.verbose)
            if self.converged:
                logger.info(self, '%s converged', cctyp)
            else:
                logger.info(self, '%s not converged', cctyp)
        if self.e_hf == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    energy = energy
    update_amps = update_amps

    def nip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nip = nocc + nocc*nocc*nvir
        return self._nip

    def nea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nea = nvir + nocc*nvir*nvir
        return self._nea

    def nee(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nee = nocc*nvir + nocc*nocc*nvir*nvir
        return self._nee

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            koopmans : bool
                Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nip()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ip_partition = partition
        if partition == 'full':
            self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(self.ipccsd_diag())[1]

        adiag = self.ipccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[self.nocc-n-1] = 1.0
                    guess.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        if left:
            matvec = self.lipccsd_matvec
        else:
            matvec = self.ipccsd_matvec
        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            eip, evecs = eig(matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            eip, evecs = eig(matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eip = eip.real

        if nroots == 1:
            eip, evecs = [self.eip], [evecs]
        for n, en, vn in zip(range(nroots), eip, evecs):
            logger.info(self, 'IP root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:self.nocc])**2)
        log.timer('IP-CCSD', *cput0)
        if nroots == 1:
            return eip[0], evecs[0]
        else:
            return eip, evecs

    def ipccsd_matvec(self, vector):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

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
        if self.ip_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 += einsum('bd,ijd->ijb',fvv,r2)
            Hr2 += -einsum('ki,kjb->ijb',foo,r2)
            Hr2 += -einsum('lj,ilb->ijb',foo,r2)
        elif self.ip_partition == 'full':
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
            Hr2 += -einsum('c,ijcb->ijb',tmp,self.t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def lipccsd_matvec(self, vector):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        # 1h-1h block
        Hr1 = -einsum('ki,i->k',imds.Loo,r1)
        #1h-2h1p block
        Hr1 += -einsum('kbij,ijb->k',imds.Wovoo,r2)

        # 2h1p-1h block
        Hr2 = -einsum('kd,l->kld',imds.Fov,r1)
        Hr2 += 2.*einsum('ld,k->kld',imds.Fov,r1)
        Hr2 += -2.*einsum('klid,i->kld',imds.Wooov,r1)
        Hr2 += einsum('lkid,i->kld',imds.Wooov,r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 += einsum('bd,klb->kld',fvv,r2)
            Hr2 += -einsum('ki,ild->kld',foo,r2)
            Hr2 += -einsum('lj,kjd->kld',foo,r2)
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('bd,klb->kld',imds.Lvv,r2)
            Hr2 += -einsum('ki,ild->kld',imds.Loo,r2)
            Hr2 += -einsum('lj,kjd->kld',imds.Loo,r2)
            Hr2 += 2.*einsum('lbdj,kjb->kld',imds.Wovvo,r2)
            Hr2 += -einsum('kbdj,ljb->kld',imds.Wovvo,r2)
            Hr2 += -einsum('lbjd,kjb->kld',imds.Wovov,r2)
            Hr2 += einsum('klij,ijd->kld',imds.Woooo,r2)
            Hr2 += -einsum('kbid,ilb->kld',imds.Wovov,r2)
            tmp = einsum('ijcb,ijb->c',t2,r2)
            Hr2 += einsum('kldc,c->kld',imds.Woovv,tmp)
            Hr2 += -2.*einsum('lkdc,c->kld',imds.Woovv,tmp)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape
        fock = self.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        Hr1 = -np.diag(imds.Loo)
        Hr2 = np.zeros((nocc,nocc,nvir), t1.dtype)
        for i in range(nocc):
            for j in range(nocc):
                for b in range(nvir):
                    if self.ip_partition == 'mp':
                        Hr2[i,j,b] += fvv[b,b]
                        Hr2[i,j,b] += -foo[i,i]
                        Hr2[i,j,b] += -foo[j,j]
                    else:
                        Hr2[i,j,b] += imds.Lvv[b,b]
                        Hr2[i,j,b] += -imds.Loo[i,i]
                        Hr2[i,j,b] += -imds.Loo[j,j]
                        Hr2[i,j,b] += imds.Woooo[i,j,i,j]
                        Hr2[i,j,b] += 2*imds.Wovvo[j,b,b,j]
                        Hr2[i,j,b] += -imds.Wovvo[i,b,b,i]*(i==j)
                        Hr2[i,j,b] += -imds.Wovov[j,b,j,b]
                        Hr2[i,j,b] += -imds.Wovov[i,b,i,b]
                        Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:],t2[i,j,:,b])
                        Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:],t2[i,j,:,b])

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = self.nip()
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nocc*nocc*nvir)
        return vector

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        assert self.ip_partition is None
        t2, eris = self.t2, self.eris
        fock = eris.fock
        nocc = self.nocc
        nvir = self.nmo - nocc

        #fov = fock[:nocc,nocc:]
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        ovov = _cp(eris.ovov)
        ovvv = _cp(eris.ovvv)
        vovv = ovvv.conj().transpose(1,0,3,2)
        oovv = _cp(eris.oovv)
        ovvo = _cp(eris.ovvo)
        ooov = _cp(eris.ooov)
        vooo = ooov.conj().transpose(3,2,1,0)
        oooo = _cp(eris.oooo)

        eijkab = np.zeros((nocc,nocc,nocc,nvir,nvir))
        for i,j,k in lib.cartesian_prod([range(nocc),range(nocc),range(nocc)]):
            for a,b in lib.cartesian_prod([range(nvir),range(nvir)]):
                eijkab[i,j,k,a,b] = foo[i,i] + foo[j,j] + foo[k,k] - fvv[a,a] - fvv[b,b]

        ipccsd_evecs  = np.array(ipccsd_evecs)
        lipccsd_evecs = np.array(lipccsd_evecs)
        e = []
        for _eval, _evec, _levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ip(_levec)
            r1,r2 = self.vector_to_amplitudes_ip(_evec)
            ldotr = np.dot(l1.conj(),r1) + np.dot(l2.ravel(),r2.ravel())
            l1 /= ldotr
            l2 /= ldotr
            l2 = 1./3*(l2 + 2.*l2.transpose(1,0,2))

            _eijkab = eijkab + _eval
            _eijkab = 1./_eijkab

            lijkab = 0.5*einsum('iajb,k->ijkab', ovov, l1)
            lijkab += einsum('iaeb,jke->ijkab', ovvv, l2)
            lijkab += -einsum('kmjb,ima->ijkab', ooov, l2)
            lijkab += -einsum('imjb,mka->ijkab', ooov, l2)
            lijkab = lijkab + lijkab.transpose(1,0,2,4,3)

            rijkab = -einsum('mkbe,m,ijae->ijkab', oovv, r1, t2)
            rijkab -= einsum('mebj,m,ikae->ijkab', ovvo, r1, t2)
            rijkab += einsum('mjnk,n,imab->ijkab', oooo, r1, t2)
            rijkab +=  einsum('aibe,kje->ijkab', vovv, r2)
            rijkab += -einsum('bjmk,mia->ijkab', vooo, r2)
            rijkab += -einsum('bjmi,kma->ijkab', vooo, r2)
            rijkab = rijkab + rijkab.transpose(1,0,2,4,3)

            lijkab = 4.*lijkab \
                   - 2.*lijkab.transpose(1,0,2,3,4) \
                   - 2.*lijkab.transpose(2,1,0,3,4) \
                   - 2.*lijkab.transpose(0,2,1,3,4) \
                   + 1.*lijkab.transpose(1,2,0,3,4) \
                   + 1.*lijkab.transpose(2,0,1,3,4)

            deltaE = 0.5*einsum('ijkab,ijkab,ijkab',lijkab,rijkab,_eijkab)
            deltaE = deltaE.real
            logger.note(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        _eval+deltaE, deltaE)
            e.append(_eval+deltaE)
        return e

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

        Kwargs:
            See ipccd()
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nea()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        if partition == 'full':
            self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(self.eaccsd_diag())[1]

        adiag = self.eaccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[n] = 1.0
                    guess.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        if left:
            matvec = self.leaccsd_matvec
        else:
            matvec = self.eaccsd_matvec
        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            eea, evecs = eig(matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            eea, evecs = eig(matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eea = eea.real

        if nroots == 1:
            eea, evecs = [self.eea], [evecs]
        nvir = self.nmo - self.nocc
        for n, en, vn in zip(range(nroots), eea, evecs):
            logger.info(self, 'EA root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:nvir])**2)
        log.timer('EA-CCSD', *cput0)
        if nroots == 1:
            return eea[0], evecs[0]
        else:
            return eea, evecs

    def eaccsd_matvec(self,vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

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
        if self.ea_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('ac,jcb->jab',fvv,r2)
            Hr2 +=  einsum('bd,jad->jab',fvv,r2)
            Hr2 += -einsum('lj,lab->jab',foo,r2)
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            Hr2 +=  einsum('ac,jcb->jab',imds.Lvv,r2)
            Hr2 +=  einsum('bd,jad->jab',imds.Lvv,r2)
            Hr2 += -einsum('lj,lab->jab',imds.Loo,r2)
            Hr2 += 2*einsum('lbdj,lad->jab',imds.Wovvo,r2)
            Hr2 +=  -einsum('lbjd,lad->jab',imds.Wovov,r2)
            Hr2 +=  -einsum('lajc,lcb->jab',imds.Wovov,r2)
            Hr2 +=  -einsum('lbcj,lca->jab',imds.Wovvo,r2)
            nvir = self.nmo-self.nocc
            for a in range(nvir):
                Hr2[:,a,:] += einsum('bcd,jcd->jb',imds.Wvvvv[a],r2)
            tmp = (2*einsum('klcd,lcd->k',imds.Woovv,r2)
                    -einsum('kldc,lcd->k',imds.Woovv,r2))
            Hr2 += -einsum('k,kjab->jab',tmp,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def leaccsd_matvec(self,vector):
        # Note this is not the same left EA equations used by Nooijen and Bartlett.
        # Small changes were made so that the same type L2 basis was used for both the
        # left EA and left IP equations.  You will note more similarity for these
        # equations to the left IP equations than for the left EA equations by Nooijen.
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,a->c',imds.Lvv,r1)
        # 1p-2p1h block
        Hr1 += einsum('abcj,jab->c',imds.Wvvvo,r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = 2.*einsum('c,ld->lcd',r1,imds.Fov)
        Hr2 +=   -einsum('d,lc->lcd',r1,imds.Fov)
        Hr2 += 2.*einsum('a,alcd->lcd',r1,imds.Wvovv)
        Hr2 +=   -einsum('a,aldc->lcd',r1,imds.Wvovv)
        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 += einsum('lad,ac->lcd',r2,fvv)
            Hr2 += einsum('lcb,bd->lcd',r2,fvv)
            Hr2 += -einsum('jcd,lj->lcd',r2,foo)
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('lad,ac->lcd',r2,imds.Lvv)
            Hr2 += einsum('lcb,bd->lcd',r2,imds.Lvv)
            Hr2 += -einsum('jcd,lj->lcd',r2,imds.Loo)
            Hr2 += 2.*einsum('jcb,lbdj->lcd',r2,imds.Wovvo)
            Hr2 +=   -einsum('jcb,lbjd->lcd',r2,imds.Wovov)
            Hr2 +=   -einsum('lajc,jad->lcd',imds.Wovov,r2)
            Hr2 +=   -einsum('lbcj,jdb->lcd',imds.Wovvo,r2)
            nvir = self.nmo-self.nocc
            for a in range(nvir):
                Hr2 += einsum('lb,bcd->lcd',r2[:,a,:],imds.Wvvvv[a])
            tmp = einsum('ijcb,ibc->j',t2,r2)
            Hr2 +=     einsum('kjef,j->kef',imds.Woovv,tmp)
            Hr2 += -2.*einsum('kjfe,j->kef',imds.Woovv,tmp)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape

        fock = self.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        Hr1 = np.diag(imds.Lvv)
        Hr2 = np.zeros((nocc,nvir,nvir), t1.dtype)
        for a in range(nvir):
            if self.ea_partition != 'mp':
                _Wvvvva = np.array(imds.Wvvvv[a])
            for b in range(nvir):
                for j in range(nocc):
                    if self.ea_partition == 'mp':
                        Hr2[j,a,b] += fvv[a,a]
                        Hr2[j,a,b] += fvv[b,b]
                        Hr2[j,a,b] += -foo[j,j]
                    else:
                        Hr2[j,a,b] += imds.Lvv[a,a]
                        Hr2[j,a,b] += imds.Lvv[b,b]
                        Hr2[j,a,b] += -imds.Loo[j,j]
                        Hr2[j,a,b] += 2*imds.Wovvo[j,b,b,j]
                        Hr2[j,a,b] += -imds.Wovov[j,b,j,b]
                        Hr2[j,a,b] += -imds.Wovov[j,a,j,a]
                        Hr2[j,a,b] += -imds.Wovvo[j,b,b,j]*(a==b)
                        Hr2[j,a,b] += _Wvvvva[b,a,b]
                        Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b],t2[:,j,a,b])
                        Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a],t2[:,j,a,b])

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = self.nea()
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nocc*nvir*nvir)
        return vector

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        assert self.ea_partition is None
        t2, eris = self.t2, self.eris
        fock = eris.fock
        nocc = self.nocc
        nvir = self.nmo - nocc

        #fov = fock[:nocc,nocc:]
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        ovov = _cp(eris.ovov)
        ovvv = _cp(eris.ovvv)
        vovv = ovvv.conj().transpose(1,0,3,2)
        ooov = _cp(eris.ooov)
        vooo = ooov.conj().transpose(3,2,1,0)
        oovv = _cp(eris.oovv)
        #oooo = _cp(eris.oooo)
        vvvv = _cp(eris.vvvv)
        ovvo = _cp(eris.ovvo)

        eijabc = np.zeros((nocc,nocc,nvir,nvir,nvir))
        for i,j in lib.cartesian_prod([range(nocc),range(nocc)]):
            for a,b,c in lib.cartesian_prod([range(nvir),range(nvir),range(nvir)]):
                eijabc[i,j,a,b,c] = foo[i,i] + foo[j,j] - fvv[a,a] - fvv[b,b] - fvv[c,c]

        eaccsd_evecs  = np.array(eaccsd_evecs)
        leaccsd_evecs = np.array(leaccsd_evecs)
        e = []
        for _eval, _evec, _levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ea(_levec)
            r1,r2 = self.vector_to_amplitudes_ea(_evec)
            ldotr = np.dot(l1.conj(),r1) + np.dot(l2.ravel(),r2.ravel())
            l1 /= ldotr
            l2 /= ldotr
            l2 = 1./3*(1.*l2 + 2.*l2.transpose(0,2,1))
            r2 = r2.transpose(0,2,1)

            _eijabc = eijabc + _eval
            _eijabc = 1./_eijabc

            lijabc = -0.5*einsum('c,iajb->ijabc', l1, ovov)
            lijabc += einsum('jmia,mbc->ijabc', ooov, l2)
            lijabc -= einsum('iaeb,jec->ijabc', ovvv, l2)
            lijabc -= einsum('jbec,iae->ijabc', ovvv, l2)
            lijabc = lijabc + lijabc.transpose(1,0,3,2,4)

            rijabc = -einsum('becf,f,ijae->ijabc', vvvv, r1, t2)
            rijabc += einsum('mjce,e,imab->ijabc', oovv, r1, t2)
            rijabc += einsum('mebj,e,imac->ijabc', ovvo, r1, t2)
            rijabc += einsum('aimj,mbc->ijabc', vooo, r2)
            rijabc += -einsum('bjce,iae->ijabc', vovv, r2)
            rijabc += -einsum('aibe,jec->ijabc', vovv, r2)
            rijabc = rijabc + rijabc.transpose(1,0,3,2,4)

            lijabc =  4.*lijabc \
                    - 2.*lijabc.transpose(0,1,3,2,4) \
                    - 2.*lijabc.transpose(0,1,4,3,2) \
                    - 2.*lijabc.transpose(0,1,2,4,3) \
                    + 1.*lijabc.transpose(0,1,3,4,2) \
                    + 1.*lijabc.transpose(0,1,4,2,3)
            deltaE = 0.5*einsum('ijabc,ijabc,ijabc',lijabc,rijabc,_eijabc)
            deltaE = deltaE.real
            logger.note(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        _eval+deltaE, deltaE)
            e.append(_eval+deltaE)
        return e


    def eeccsd(self, nroots=1, koopmans=False, guess=None, partition=None):
        '''Calculate N-electron neutral excitations via EE-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            koopmans : bool
                Calculate Koopmans'-like (1p1h) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nee()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ee_partition = partition
        if partition == 'full':
            self._eeccsd_diag_matrix2 = self.vector_to_amplitudes_ee(self.eeccsd_diag())[1]

        nvir = self.nmo - self.nocc
        adiag = self.eeccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            idx = adiag.argsort()
            n = 0
            for i in idx:
                g = np.zeros(size)
                g[i] = 1.0
                if koopmans:
                    if np.linalg.norm(g[:self.nocc*nvir])**2 > 0.8:
                        guess.append(g)
                        n += 1
                else:
                    guess.append(g)
                    n += 1
                if n == nroots:
                    break

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            eee, evecs = eig(self.eeccsd_matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            eee, evecs = eig(self.eeccsd_matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eee = eee.real

        if nroots == 1:
            eee, evecs = [self.eee], [evecs]
        for n, en, vn in zip(range(nroots), eee, evecs):
            logger.info(self, 'EE root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:self.nocc*nvir])**2)
        log.timer('EE-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eeccsd_matvec(self,vector):
        raise NotImplementedError

    def eeccsd_diag(self):
        raise NotImplementedError

    def vector_to_amplitudes_ee(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc*nvir].copy().reshape((nocc,nvir))
        r2 = vector[nocc*nvir:].copy().reshape((nocc,nocc,nvir,nvir))
        return [r1,r2]

    def amplitudes_to_vector_ee(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = self.nee()
        vector = np.zeros(size, r1.dtype)
        vector[:nocc*nvir] = r1.copy().reshape(nocc*nvir)
        vector[nocc*nvir:] = r2.copy().reshape(nocc*nocc*nvir*nvir)
        return vector


class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.outcore.general_iofree):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nmo = cc.nmo
        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        self.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if slices:
            return self.ovvv[slices]
        else:
            return self.ovvv

class _IMDS:
    def __init__(self, cc):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = cc.eris
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = np.asarray(eris.ovov).transpose(0,2,1,3)

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError

def _cp(a):
    return np.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto

    mol = gto.M()
    nocc, nvir = 5, 12
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    np.random.seed(12)
    mf._eri = np.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = np.random.random((nmo,nmo))
    mf.mo_energy = np.arange(0., nmo)
    mf.mo_occ = np.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = np.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(np.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    a = np.random.random((nmo,nmo)) * .1
    eris.fock += a + a.T.conj()
    t1 = np.random.random((nocc,nvir)) * .1
    t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)

    mycc.cc2 = False
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - 66540.100267798145)
    mycc.cc2 = True
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - -1517.9391800662809)

    eri1 = np.random.random((nmo,nmo,nmo,nmo)) + np.random.random((nmo,nmo,nmo,nmo))*1j
    eri1 = eri1.transpose(0,2,1,3)
    eri1 = eri1 + eri1.transpose(1,0,3,2).conj()
    eri1 = eri1 + eri1.transpose(2,3,0,1)
    eri1 *= .1
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    a = np.random.random((nmo,nmo)) * .1j
    eris.fock = eris.fock + a + a.T.conj()

    t1 = t1 + np.random.random((nocc,nvir)) * .1j
    t2 = t2 + np.random.random((nocc,nocc,nvir,nvir)) * .1j
    t2 = t2 + t2.transpose(1,0,3,2)
    mycc.cc2 = False
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - (-13.32050019680894-1.8825765910430254j))
    print(lib.finger(t2a) - (9.2521062044785189+29.999480274811873j))
    mycc.cc2 = True
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - (-13.32050019680894-1.8825765910430254j))
    print(lib.finger(t2a) - (-0.056223856104895858+0.025472249329733986j))

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    #mol.basis = '3-21G'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    emp2, t1, t2 = mycc.init_amps(eris)
    print(lib.finger(t2) - 0.044540097905897198)
    np.random.seed(1)
    t1 = np.random.random(t1.shape)*.1
    t2 = np.random.random(t2.shape)*.1
    t2 = t2 + t2.transpose(1,0,3,2)
    t1, t2 = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1) - 0.25118555558133576)
    print(lib.finger(t2) - 0.02352137419932243)

    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    part = None
    print("IP energies... (right eigenvector)")
    e,v = mycc.ipccsd(nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    print("IP energies... (left eigenvector)")
    le,lv = mycc.ipccsd(nroots=3,left=True)
    print(le[0] - 0.43356040428879794)
    print(le[1] - 0.51876597800180335)
    print(le[2] - 0.67828755013874864)

    e = mycc.ipccsd_star_contract(e,v,lv)
    print(e[0] - 0.43793202073189047)
    print(e[1] - 0.52287073446559729)
    print(e[2] - 0.67994597948852287)

    print("EA energies... (right eigenvector)")
    e,v = mycc.eaccsd(nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

    print("EA energies... (left eigenvector)")
    le,lv = mycc.eaccsd(nroots=3,left=True)
    print(le[0] - 0.16737896537079733)
    print(le[1] - 0.24027634198123343)
    print(le[2] - 0.51006809015066612)

    e = mycc.eaccsd_star_contract(e,v,lv)
    print(e[0] - 0.16656250953550664)
    print(e[1] - 0.23944144521387614)
    print(e[2] - 0.41399436888830721)

    # Note: Not implemented
    #e,v = mycc.eeccsd(nroots=4)
