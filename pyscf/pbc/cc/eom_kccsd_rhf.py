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

import itertools
import time
import sys
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.cc.kccsd_rhf import vector_to_nested, nested_to_vector
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa

einsum = lib.einsum

########################################
# EOM-IP-CCSD
########################################

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{ b}, i.e. 'jb' indices are coupled.'''
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nmo = eom.nmo
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv

    vector = eom.mask_frozen(vector, kshift, const=0.0)
    r1, r2 = eom.vector_to_amplitudes(vector)

    # 1h-1h block
    Hr1 = -einsum('ki,k->i', imds.Loo[kshift], r1)
    # 1h-2h1p block
    for kl in range(nkpts):
        Hr1 += 2. * einsum('ld,ild->i', imds.Fov[kl], r2[kshift, kl])
        Hr1 += -einsum('ld,lid->i', imds.Fov[kl], r2[kl, kshift])
        for kk in range(nkpts):
            kd = kconserv[kk, kshift, kl]
            Hr1 += -2. * einsum('klid,kld->i', imds.Wooov[kk, kl, kshift], r2[kk, kl])
            Hr1 += einsum('lkid,kld->i', imds.Wooov[kl, kk, kshift], r2[kk, kl])

    Hr2 = np.zeros(r2.shape, dtype=np.result_type(imds.Wovoo.dtype, r1.dtype))
    # 2h1p-1h block
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kconserv[ki, kshift, kj]
            Hr2[ki, kj] -= einsum('kbij,k->ijb', imds.Wovoo[kshift, kb, ki], r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:, :nocc, :nocc]
        fvv = fock[:, nocc:, nocc:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] += einsum('bd,ijd->ijb', fvv[kb], r2[ki, kj])
                Hr2[ki, kj] -= einsum('li,ljb->ijb', foo[ki], r2[ki, kj])
                Hr2[ki, kj] -= einsum('lj,ilb->ijb', foo[kj], r2[ki, kj])
    elif eom.partition == 'full':
        if diag is not None:
            diag = eom.get_diag(imds=imds)
        diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] += einsum('bd,ijd->ijb', imds.Lvv[kb], r2[ki, kj])
                Hr2[ki, kj] -= einsum('li,ljb->ijb', imds.Loo[ki], r2[ki, kj])
                Hr2[ki, kj] -= einsum('lj,ilb->ijb', imds.Loo[kj], r2[ki, kj])
                for kl in range(nkpts):
                    kk = kconserv[ki, kl, kj]
                    Hr2[ki, kj] += einsum('klij,klb->ijb', imds.Woooo[kk, kl, ki], r2[kk, kl])
                    kd = kconserv[kl, kj, kb]
                    Hr2[ki, kj] += 2. * einsum('lbdj,ild->ijb', imds.Wovvo[kl, kb, kd], r2[ki, kl])
                    Hr2[ki, kj] += -einsum('lbdj,lid->ijb', imds.Wovvo[kl, kb, kd], r2[kl, ki])
                    Hr2[ki, kj] += -einsum('lbjd,ild->ijb', imds.Wovov[kl, kb, kj], r2[ki, kl])  # typo in Ref
                    kd = kconserv[kl, ki, kb]
                    Hr2[ki, kj] += -einsum('lbid,ljd->ijb', imds.Wovov[kl, kb, ki], r2[kl, kj])
        tmp = (2. * einsum('xyklcd,xykld->c', imds.Woovv[:, :, kshift], r2[:, :])
                  - einsum('yxlkcd,xykld->c', imds.Woovv[:, :, kshift], r2[:, :]))
        Hr2[:, :] += -einsum('c,xyijcb->xyijb', tmp, t2[:, :, kshift])

    return eom.mask_frozen(eom.amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

def lipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2hp operators are of the form s_{kl}^{ d}, i.e. 'ld' indices are coupled.'''
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    assert(eom.partition == None)
    if imds is None: imds = eom.make_imds()

    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv

    vector = eom.mask_frozen(vector, kshift, const=0.0)
    r1, r2 = eom.vector_to_amplitudes(vector)

    Hr1 = -einsum('ki,i->k',imds.Loo[kshift],r1)
    for ki, kb in itertools.product(range(nkpts), repeat=2):
        kj = kconserv[kshift,ki,kb]
        Hr1 -= einsum('kbij,ijb->k',imds.Wovoo[kshift,kb,ki],r2[ki,kj])

    Hr2 = np.zeros(r2.shape, dtype=np.result_type(imds.Wovoo.dtype, r1.dtype))
    for kl, kk in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kk,kshift,kl]
        SWooov = (2. * imds.Wooov[kk,kl,kshift] -
                       imds.Wooov[kl,kk,kshift].transpose(1, 0, 2, 3))
        Hr2[kk,kl] -= einsum('klid,i->kld',SWooov,r1)

        Hr2[kk,kshift] -= (kk==kd)*einsum('kd,l->kld',imds.Fov[kk],r1)
        Hr2[kshift,kl] += (kl==kd)*2.*einsum('ld,k->kld',imds.Fov[kl],r1)

    for kl, kk in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kk,kshift,kl]
        Hr2[kk,kl] -= einsum('ki,ild->kld',imds.Loo[kk],r2[kk,kl])
        Hr2[kk,kl] -= einsum('lj,kjd->kld',imds.Loo[kl],r2[kk,kl])
        Hr2[kk,kl] += einsum('bd,klb->kld',imds.Lvv[kd],r2[kk,kl])

        for kj in range(nkpts):
            kb = kconserv[kd, kl, kj]
            SWovvo = (2. * imds.Wovvo[kl,kb,kd] -
                           imds.Wovov[kl,kb,kj].transpose(0, 1, 3, 2))
            Hr2[kk,kl] += einsum('lbdj,kjb->kld',SWovvo,r2[kk,kj])

            kb = kconserv[kd, kk, kj]
            Hr2[kk,kl] -= einsum('kbdj,ljb->kld',imds.Wovvo[kk,kb,kd],r2[kl,kj])
            Hr2[kk,kl] -= einsum('kbjd,jlb->kld',imds.Wovov[kk,kb,kj],r2[kj,kl])

            ki = kconserv[kk,kj,kl]
            Hr2[kk,kl] += einsum('klji,jid->kld',imds.Woooo[kk,kl,kj],r2[kj,ki])

    tmp = np.zeros(nvir, dtype=np.result_type(imds.Wovoo.dtype, r1.dtype))
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        kc = kshift
        tmp += einsum('ijcb,ijb->c',t2[ki, kj, kc],r2[ki, kj])

    for kl, kk in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kk,kshift,kl]
        SWoovv = (2. * imds.Woovv[kl, kk, kd] -
                       imds.Woovv[kk, kl, kd].transpose(1, 0, 2, 3))
        Hr2[kk, kl] -= einsum('lkdc,c->kld',SWoovv, tmp)

    return eom.mask_frozen(eom.amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)

def ipccsd_diag(eom, kshift, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()

    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = -np.diag(imds.Loo[kshift])

    Hr2 = np.zeros((nkpts, nkpts, nocc, nocc, nvir), dtype=t1.dtype)
    if eom.partition == 'mp':
        foo = eom.eris.fock[:, :nocc, :nocc]
        fvv = eom.eris.fock[:, nocc:, nocc:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] = fvv[kb].diagonal()
                Hr2[ki, kj] -= foo[ki].diagonal()[:, None, None]
                Hr2[ki, kj] -= foo[kj].diagonal()[:, None]
    else:
        idx = np.arange(nocc)
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki, kshift, kj]
                Hr2[ki, kj] = imds.Lvv[kb].diagonal()
                Hr2[ki, kj] -= imds.Loo[ki].diagonal()[:, None, None]
                Hr2[ki, kj] -= imds.Loo[kj].diagonal()[:, None]

                if ki == kconserv[ki, kj, kj]:
                    Hr2[ki, kj] += np.einsum('ijij->ij', imds.Woooo[ki, kj, ki])[:, :, None]

                Hr2[ki, kj] -= np.einsum('jbjb->jb', imds.Wovov[kj, kb, kj])

                Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])
                Hr2[ki, kj] += 2. * Wovvo
                if ki == kj:  # and i == j
                    Hr2[ki, ki, idx, idx] -= Wovvo

                Hr2[ki, kj] -= np.einsum('ibib->ib', imds.Wovov[ki, kb, ki])[:, None, :]

                kd = kconserv[kj, kshift, ki]
                Hr2[ki, kj] -= 2. * np.einsum('ijcb,jibc->ijb', t2[ki, kj, kshift], imds.Woovv[kj, ki, kd])
                Hr2[ki, kj] += np.einsum('ijcb,ijbc->ijb', t2[ki, kj, kshift], imds.Woovv[ki, kj, kd])

    return eom.amplitudes_to_vector(Hr1, Hr2)


def ipccsd_star_contract(eom, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift, imds=None):
    '''For description of arguments, see `ipccsd_star_contract` in `kccsd_ghf.py`.'''
    assert (eom.partition == None)
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    dtype = np.result_type(t1, t2)
    kconserv = eom.kconserv

    mo_energy_occ = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    def contract_l3p(l1,l2,kptvec):
        '''Create perturbed left 3p2h amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,kk,ka,kb]
        '''
        ki, kj, kk, ka, kb = kptvec
        out = np.zeros((nocc,)*3 + (nvir,)*2, dtype=dtype)
        if kk == kshift and kj == kconserv[ka,ki,kb]:
            out += 0.5*np.einsum('ijab,k->ijkab', eris.oovv[ki,kj,ka], l1)
        ke = kconserv[kb,ki,ka]
        out += lib.einsum('eiba,jke->ijkab', eris.vovv[ke,ki,kb], l2[kj,kk])
        km = kconserv[kshift,ki,ka]
        out += -lib.einsum('kjmb,ima->ijkab', eris.ooov[kk,kj,km], l2[ki,km])
        km = kconserv[ki,kb,kj]
        out += -lib.einsum('ijmb,mka->ijkab', eris.ooov[ki,kj,km], l2[km,kk])
        return out

    def contract_pl3p(l1,l2,kptvec):
        '''Create P(ia|jb) of perturbed left 3p2h amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,kk,ka,kb]
        '''
        kptvec = np.asarray(kptvec)
        out = contract_l3p(l1,l2,kptvec)
        out += contract_l3p(l1,l2,kptvec[[1,0,2,4,3]]).transpose(1,0,2,4,3)  # P(ia|jb)
        return out

    def contract_r3p(r1,r2,kptvec):
        '''Create perturbed right 3p2h amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,kk,ka,kb]
        '''
        ki, kj, kk, ka, kb = kptvec
        out = np.zeros((nocc,)*3 + (nvir,)*2, dtype=dtype)
        tmp = np.einsum('mbke,m->bke', eris.ovov[kshift,kb,kk], r1)
        out += -lib.einsum('bke,ijae->ijkab', tmp, t2[ki,kj,ka])
        ke = kconserv[kb,kshift,kj]
        tmp = np.einsum('bmje,m->bej', eris.voov[kb,kshift,kj], r1)
        out += -lib.einsum('bej,ikae->ijkab', tmp, t2[ki,kk,ka])
        km = kconserv[ka,ki,kb]
        tmp = np.einsum('mnjk,n->mjk', eris.oooo[km,kshift,kj], r1)
        out += lib.einsum('mjk,imab->ijkab', tmp, t2[ki,km,ka])
        ke = kconserv[kk,kshift,kj]
        out += lib.einsum('eiba,kje->ijkab', eris.vovv[ke,ki,kb].conj(), r2[kk,kj])
        km = kconserv[kk,kb,kj]
        out += -lib.einsum('kjmb,mia->ijkab', eris.ooov[kk,kj,km].conj(), r2[km,ki])
        km = kconserv[ki,kb,kj]
        out += -lib.einsum('ijmb,kma->ijkab', eris.ooov[ki,kj,km].conj(), r2[kk,km])
        return out

    def contract_pr3p(r1,r2,kptvec):
        '''Create P(ia|jb) of perturbed right 3p2h amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,kk,ka,kb]
        '''
        kptvec = np.asarray(kptvec)
        out = contract_r3p(r1,r2,kptvec)
        out += contract_r3p(r1,r2,kptvec[[1,0,2,4,3]]).transpose(1,0,2,4,3)  # P(ia|jb)
        return out

    ipccsd_evecs = np.array(ipccsd_evecs)
    lipccsd_evecs = np.array(lipccsd_evecs)
    e_star = []
    ipccsd_evecs, lipccsd_evecs = [np.atleast_2d(x) for x in [ipccsd_evecs, lipccsd_evecs]]
    ipccsd_evals = np.atleast_1d(ipccsd_evals)
    for ip_eval, ip_evec, ip_levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        # Enforcing <L|R> = 1
        l1, l2 = eom.vector_to_amplitudes(ip_levec, kshift)
        r1, r2 = eom.vector_to_amplitudes(ip_evec, kshift)
        ldotr = np.dot(l1, r1) + np.dot(l2.ravel(), r2.ravel())

        # Transposing the l2 operator
        l2T = np.zeros_like(l2)
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kconserv[ki,kshift,kj]
                l2T[ki,kj] = l2[kj,ki].transpose(1,0,2)
        l2 = (l2 + 2.*l2T)/3.

        logger.info(eom, 'Left-right amplitude overlap : %14.8e + 1j %14.8e',
                    ldotr.real, ldotr.imag)
        if abs(ldotr) < 1e-7:
            logger.warn(eom, 'Small %s left-right amplitude overlap. Results '
                             'may be inaccurate.', ldotr)
        l1 /= ldotr
        l2 /= ldotr

        deltaE = 0.0 + 1j*0.0
        #eij = (mo_e_o[:, None, :, None, None] + mo_e_o[None, :, None, :, None])
        #        #mo_e_o[None, None, :, None, None, :])
        for ka, kb in itertools.product(range(nkpts), repeat=2):
            lijkab = np.zeros((nkpts,nkpts,nocc,nocc,nocc,nvir,nvir),dtype=dtype)
            Plijkab = np.zeros((nkpts,nkpts,nocc,nocc,nocc,nvir,nvir),dtype=dtype)
            rijkab = np.zeros((nkpts,nkpts,nocc,nocc,nocc,nvir,nvir),dtype=dtype)
            eijk = np.zeros((nkpts,nkpts,nocc,nocc,nocc),dtype=mo_e_o.dtype)
            kklist = kpts_helper.get_kconserv3(eom._cc._scf.cell, eom._cc.kpts,
                         [ka,kb,kshift,range(nkpts),range(nkpts)])

            for ki, kj in itertools.product(range(nkpts), repeat=2):
                kk = kklist[ki,kj]

                kptvec = [ki,kj,kk,ka,kb]
                lijkab[ki,kj] = contract_pl3p(l1,l2,kptvec)

                rijkab[ki,kj] = contract_pr3p(r1,r2,kptvec)

            for ki, kj in itertools.product(range(nkpts), repeat=2):
                kk = kklist[ki,kj]
                Plijkab[ki,kj] = (4.*lijkab[ki,kj] +
                                  1.*lijkab[kj,kk].transpose(2,0,1,3,4) +
                                  1.*lijkab[kk,ki].transpose(1,2,0,3,4) -
                                  2.*lijkab[ki,kk].transpose(0,2,1,3,4) -
                                  2.*lijkab[kk,kj].transpose(2,1,0,3,4) -
                                  2.*lijkab[kj,ki].transpose(1,0,2,3,4))
                eijk[ki,kj] = _get_epqr([0,nocc,ki,mo_e_o,eom.nonzero_opadding],
                                        [0,nocc,kj,mo_e_o,eom.nonzero_opadding],
                                        [0,nocc,kk,mo_e_o,eom.nonzero_opadding])

            # Creating denominator
            eab = _get_epq([0,nvir,ka,mo_e_v,eom.nonzero_vpadding],
                           [0,nvir,kb,mo_e_v,eom.nonzero_vpadding],
                           fac=[-1.,-1.])

            # Creating denominator
            eijkab = (eijk[:, :, :, :, :, None, None] +
                      eab[None, None, None, None, None, :, :])
            denom = eijkab + ip_eval
            denom = 1. / denom

            deltaE += lib.einsum('xyijkab,xyijkab,xyijkab', Plijkab, rijkab, denom)

        deltaE *= 0.5
        deltaE = deltaE.real
        logger.info(eom, "ipccsd energy, star energy, delta energy = %16.12f, %16.12f, %16.12f",
                    ip_eval, ip_eval + deltaE, deltaE)
        e_star.append(ip_eval + deltaE)
    return e_star


class EOMIP(eom_kgccsd.EOMIP):
    matvec = ipccsd_matvec
    l_matvec = lipccsd_matvec
    get_diag = ipccsd_diag
    ccsd_star_contract = ipccsd_star_contract

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def ip_vector_desc(self):
        """Description of the IP vector."""
        return [(self.nocc,), (self.nkpts, self.nkpts, self.nocc, self.nocc, self.nmo - self.nocc)]

    def ip_amplitudes_to_vector(self, t1, t2):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]

    def ip_vector_to_amplitudes(self, vec):
        """Ground state vector to amplitudes."""
        return vector_to_nested(vec, self.ip_vector_desc)

    def vector_to_amplitudes(self, vector, kshift=None):
        return self.ip_vector_to_amplitudes(vector)

    def amplitudes_to_vector(self, r1, r2, kshift=None, kconserv=None):
        return self.ip_amplitudes_to_vector(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds

class EOMIP_Ta(EOMIP):
    '''Class for EOM IPCCSD(T)*(a) method by Matthews and Stanton.'''
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_t3p2_ip(self._cc)
        return imds

########################################
# EOM-EA-CCSD
########################################

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nmo = eom.nmo
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv

    vector = eom.mask_frozen(vector, kshift, const=0.0)
    r1, r2 = eom.vector_to_amplitudes(vector)

    # Eq. (30)
    # 1p-1p block
    Hr1 = einsum('ac,c->a', imds.Lvv[kshift], r1)
    # 1p-2p1h block
    for kl in range(nkpts):
        Hr1 += 2. * einsum('ld,lad->a', imds.Fov[kl], r2[kl, kshift])
        Hr1 += -einsum('ld,lda->a', imds.Fov[kl], r2[kl, kl])
        for kc in range(nkpts):
            kd = kconserv[kshift, kc, kl]
            Hr1 += 2. * einsum('alcd,lcd->a', imds.Wvovv[kshift, kl, kc], r2[kl, kc])
            Hr1 += -einsum('aldc,lcd->a', imds.Wvovv[kshift, kl, kd], r2[kl, kc])

    # Eq. (31)
    # 2p1h-1p block
    Hr2 = np.zeros(r2.shape, dtype=np.result_type(imds.Wvvvo.dtype, r1.dtype))
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift,ka,kj]
            Hr2[kj,ka] += einsum('abcj,c->jab',imds.Wvvvo[ka,kb,kshift],r1)

    # 2p1h-2p1h block
    if eom.partition == 'mp':
        fock = eom.eris.fock
        foo = fock[:, :nocc, :nocc]
        fvv = fock[:, nocc:, nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= einsum('lj,lab->jab', foo[kj], r2[kj, ka])
                Hr2[kj, ka] += einsum('ac,jcb->jab', fvv[ka], r2[kj, ka])
                Hr2[kj, ka] += einsum('bd,jad->jab', fvv[kb], r2[kj, ka])
    elif eom.partition == 'full':
        if diag is not None:
            diag = eom.get_diag(imds=imds)
        diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= einsum('lj,lab->jab', imds.Loo[kj], r2[kj, ka])
                Hr2[kj, ka] += einsum('ac,jcb->jab', imds.Lvv[ka], r2[kj, ka])
                Hr2[kj, ka] += einsum('bd,jad->jab', imds.Lvv[kb], r2[kj, ka])
                for kd in range(nkpts):
                    kc = kconserv[ka, kd, kb]
                    Wvvvv = imds.get_Wvvvv(ka, kb, kc)
                    Hr2[kj, ka] += einsum('abcd,jcd->jab', Wvvvv, r2[kj, kc])
                    kl = kconserv[kd, kb, kj]
                    Hr2[kj, ka] += 2. * einsum('lbdj,lad->jab', imds.Wovvo[kl, kb, kd], r2[kl, ka])
                    # imds.Wvovo[kb,kl,kd,kj] <= imds.Wovov[kl,kb,kj,kd].transpose(1,0,3,2)
                    Hr2[kj, ka] += -einsum('bldj,lad->jab', imds.Wovov[kl, kb, kj].transpose(1, 0, 3, 2),
                                           r2[kl, ka])
                    # imds.Wvoov[kb,kl,kj,kd] <= imds.Wovvo[kl,kb,kd,kj].transpose(1,0,3,2)
                    Hr2[kj, ka] += -einsum('bljd,lda->jab', imds.Wovvo[kl, kb, kd].transpose(1, 0, 3, 2),
                                           r2[kl, kd])
                    kl = kconserv[kd, ka, kj]
                    # imds.Wvovo[ka,kl,kd,kj] <= imds.Wovov[kl,ka,kj,kd].transpose(1,0,3,2)
                    Hr2[kj, ka] += -einsum('aldj,ldb->jab', imds.Wovov[kl, ka, kj].transpose(1, 0, 3, 2),
                                           r2[kl, kd])
        tmp = (2. * einsum('xyklcd,xylcd->k', imds.Woovv[kshift, :, :], r2[:, :])
                  - einsum('xylkcd,xylcd->k', imds.Woovv[:, kshift, :], r2[:, :]))
        Hr2[:, :] += -einsum('k,xykjab->xyjab', tmp, t2[kshift, :, :])

    return eom.mask_frozen(eom.amplitudes_to_vector(Hr1, Hr2, kshift), kshift, const=0.0)


def leaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2hp operators are of the form s_{ l}^{cd}, i.e. 'ld' indices are coupled.'''
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    assert(eom.partition == None)
    if imds is None: imds = eom.make_imds()

    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = imds.t1.shape
    kconserv = imds.kconserv

    vector = eom.mask_frozen(vector, kshift, const=0.0)
    r1, r2 = eom.vector_to_amplitudes(vector)

    # 1p-1p block
    Hr1 = np.einsum('ac,a->c', imds.Lvv[kshift], r1)
    # 1p-2p1h block
    for kj, ka in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[kj, ka, kshift]
        Hr1 += np.einsum('abcj,jab->c', imds.Wvvvo[ka, kb, kshift], r2[kj, ka])

    # 2p1h-1p block
    Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)
    for kl, kc in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kl, kc, kshift]
        Hr2[kl, kc] += 2. * (kl==kd) * np.einsum('c,ld->lcd', r1, imds.Fov[kd])
        Hr2[kl, kc] += - (kl==kc) * np.einsum('d,lc->lcd', r1, imds.Fov[kl])

        SWvovv = (2. * imds.Wvovv[kshift, kl, kc] -
                       imds.Wvovv[kshift, kl, kd].transpose(0, 1, 3, 2))
        Hr2[kl, kc] += np.einsum('a,alcd->lcd', r1, SWvovv)

    # 2p1h-2p1h block
    for kl, kc in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kl, kc, kshift]
        Hr2[kl, kc] += lib.einsum('lad,ac->lcd', r2[kl, kc], imds.Lvv[kc])
        Hr2[kl, kc] += lib.einsum('lcb,bd->lcd', r2[kl, kc], imds.Lvv[kd])
        Hr2[kl, kc] += -lib.einsum('jcd,lj->lcd', r2[kl, kc], imds.Loo[kl])

        for kb in range(nkpts):
            kj = kconserv[kl, kd, kb]
            SWovvo = (2. * imds.Wovvo[kl, kb, kd] -
                           imds.Wovov[kl, kb, kj].transpose(0, 1, 3, 2))
            Hr2[kl, kc] += lib.einsum('jcb,lbdj->lcd', r2[kj, kc], SWovvo)
            kj = kconserv[kl, kc, kb]
            Hr2[kl, kc] += -lib.einsum('lbjc,jbd->lcd', imds.Wovov[kl, kb, kj], r2[kj, kb])
            Hr2[kl, kc] += -lib.einsum('lbcj,jdb->lcd', imds.Wovvo[kl, kb, kc], r2[kj, kd])

            ka = kconserv[kc, kb, kd]
            Wvvvv = imds.get_Wvvvv(ka, kb, kc)
            Hr2[kl, kc] += lib.einsum('lab,abcd->lcd', r2[kl, ka], Wvvvv)

    tmp = np.zeros((nocc),dtype=t1.dtype)
    for ki, kc in itertools.product(range(nkpts), repeat=2):
        kb = kconserv[ki, kc, kshift]
        tmp += np.einsum('ijcb,ibc->j', imds.t2[ki, kshift, kc], r2[ki, kb])

    for kl, kc in itertools.product(range(nkpts), repeat=2):
        kd = kconserv[kl, kc, kshift]
        SWoovv = (2. * imds.Woovv[kl, kshift, kd] -
                       imds.Woovv[kl, kshift, kc].transpose(0, 1, 3, 2))
        Hr2[kl,kc] += -np.einsum('ljdc,j->lcd', SWoovv, tmp)

    return eom.mask_frozen(eom.amplitudes_to_vector(Hr1, Hr2), kshift, const=0.0)


def eaccsd_diag(eom, kshift, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()

    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = np.diag(imds.Lvv[kshift])

    Hr2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=t2.dtype)
    if eom.partition == 'mp':
        foo = imds.eris.fock[:, :nocc, :nocc]
        fvv = imds.eris.fock[:, nocc:, nocc:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= foo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += fvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += fvv[kb].diagonal()
    else:
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                Hr2[kj, ka] -= imds.Loo[kj].diagonal()[:, None, None]
                Hr2[kj, ka] += imds.Lvv[ka].diagonal()[None, :, None]
                Hr2[kj, ka] += imds.Lvv[kb].diagonal()

                Wvvvv = imds.get_Wvvvv(ka, kb, ka)
                Hr2[kj, ka] += np.einsum('abab->ab', Wvvvv)

                Hr2[kj, ka] -= np.einsum('jbjb->jb', imds.Wovov[kj, kb, kj])[:, None, :]
                Wovvo = np.einsum('jbbj->jb', imds.Wovvo[kj, kb, kb])
                Hr2[kj, ka] += 2. * Wovvo[:, None, :]
                if ka == kb:
                    for a in range(nvir):
                        Hr2[kj, ka, :, a, a] -= Wovvo[:, a]

                Hr2[kj, ka] -= np.einsum('jaja->ja', imds.Wovov[kj, ka, kj])[:, :, None]

                Hr2[kj, ka] -= 2 * np.einsum('ijab,ijab->jab', t2[kshift, kj, ka], imds.Woovv[kshift, kj, ka])
                Hr2[kj, ka] += np.einsum('ijab,ijba->jab', t2[kshift, kj, ka], imds.Woovv[kshift, kj, kb])

    return eom.amplitudes_to_vector(Hr1, Hr2, kshift)

def eaccsd_star_contract(eom, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift, imds=None):
    '''For descreation of arguments, see `eaccsd_star_contract` in `kccsd_ghf.py`.'''
    assert (eom.partition == None)
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    dtype = np.result_type(t1, t2)
    kconserv = eom.kconserv

    mo_energy_occ = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    def contract_l3p(l1,l2,kptvec):
        '''Create perturbed left 3h2p amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,ka,kb,kc]
        '''
        ki, kj, ka, kb, kc = kptvec
        out = np.zeros((nocc,)*2 + (nvir,)*3, dtype=dtype)
        if kc == kshift and kb == kconserv[ki,ka,kj]:
            out -= 0.5*lib.einsum('ijab,c->ijabc', eris.oovv[ki,kj,ka], l1)
        km = kconserv[ki,ka,kj]
        out += lib.einsum('jima,mbc->ijabc', eris.ooov[kj,ki,km], l2[km,kb])
        ke = kconserv[kshift,ka,ki]
        out -= lib.einsum('ejcb,iae->ijabc', eris.vovv[ke,kj,kc], l2[ki,ka])
        ke = kconserv[kshift,kc,ki]
        out -= lib.einsum('ejab,iec->ijabc', eris.vovv[ke,kj,ka], l2[ki,ke])
        return out

    def contract_pl3p(l1,l2,kptvec):
        '''Create P(ia|jb) of perturbed left 3h2p amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,ka,kb,kc]
        '''
        kptvec = np.asarray(kptvec)
        out = contract_l3p(l1,l2,kptvec)
        out += contract_l3p(l1,l2,kptvec[[1,0,3,2,4]]).transpose(1,0,3,2,4)  # P(ia|jb)
        return out

    def contract_r3p(r1,r2,kptvec):
        '''Create perturbed right 3j2p amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,ka,kb,kc]
        '''
        ki, kj, ka, kb, kc = kptvec
        out = np.zeros((nocc,)*2 + (nvir,)*3, dtype=dtype)
        ke = kconserv[ki,ka,kj]
        tmp = lib.einsum('bcef,f->bce', eris.vvvv[kb,kc,ke], r1)
        out -= lib.einsum('bce,ijae->ijabc', tmp, t2[ki,kj,ka])
        km = kconserv[kshift,kc,kj]
        tmp = einsum('mcje,e->mcj',eris.ovov[km,kc,kj],r1)
        out += einsum('mcj,imab->ijabc',tmp,t2[ki,km,ka])
        km = kconserv[kc,ki,ka]
        tmp = einsum('bmje,e->mbj',eris.voov[kb,km,kj],r1)
        out += einsum('mbj,imac->ijabc',tmp,t2[ki,km,ka])
        km = kconserv[ki,ka,kj]
        out += einsum('jima,mcb->ijabc',eris.ooov[kj,ki,km].conj(),r2[km,kc])
        ke = kconserv[kshift,ka,ki]
        out += -einsum('ejcb,iea->ijabc',eris.vovv[ke,kj,kc].conj(),r2[ki,ke])
        ke = kconserv[kshift,kc,kj]
        out += -einsum('eiba,jce->ijabc',eris.vovv[ke,ki,kb].conj(),r2[kj,kc])
        return out

    def contract_pr3p(r1,r2,kptvec):
        '''Create P(ia|jb) of perturbed right 3h2p amplitude.

        Args:
            kptvec (`ndarray`):
                Array of k-vectors [ki,kj,ka,kb,kc]
        '''
        kptvec = np.asarray(kptvec)
        out = contract_r3p(r1,r2,kptvec)
        out += contract_r3p(r1,r2,kptvec[[1,0,3,2,4]]).transpose(1,0,3,2,4)  # P(ia|jb)
        return out

    eaccsd_evecs = np.array(eaccsd_evecs)
    leaccsd_evecs = np.array(leaccsd_evecs)
    e_star = []
    eaccsd_evecs, leaccsd_evecs = [np.atleast_2d(x) for x in [eaccsd_evecs, leaccsd_evecs]]
    eaccsd_evals = np.atleast_1d(eaccsd_evals)
    for ea_eval, ea_evec, ea_levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        # Enforcing <L|R> = 1
        l1, l2 = eom.vector_to_amplitudes(ea_levec, kshift)
        r1, r2 = eom.vector_to_amplitudes(ea_evec, kshift)
        ldotr = np.dot(l1, r1) + np.dot(l2.ravel(), r2.ravel())

        # Transposing the l2 operator
        l2T = np.zeros_like(l2)
        for kj, ka in itertools.product(range(nkpts), repeat=2):
            kb = kconserv[ka,kj,kshift]
            l2T[kj,kb] = l2[kj,ka].transpose(0,2,1)
        l2 = (l2 + 2.*l2T)/3.

        logger.info(eom, 'Left-right amplitude overlap : %14.8e + 1j %14.8e',
                    ldotr.real, ldotr.imag)
        if abs(ldotr) < 1e-7:
            logger.warn(eom, 'Small %s left-right amplitude overlap. Results '
                             'may be inaccurate.', ldotr)
        l1 /= ldotr
        l2 /= ldotr

        deltaE = 0.0 + 1j*0.0
        for ki, kj in itertools.product(range(nkpts), repeat=2):
            lijabc = np.zeros((nkpts,nkpts,nocc,nocc,nvir,nvir,nvir),dtype=dtype)
            Plijabc = np.zeros((nkpts,nkpts,nocc,nocc,nvir,nvir,nvir),dtype=dtype)
            rijabc = np.zeros((nkpts,nkpts,nocc,nocc,nvir,nvir,nvir),dtype=dtype)
            eabc = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=dtype)
            kclist = kpts_helper.get_kconserv3(eom._cc._scf.cell, eom._cc.kpts,
                         [ki,kj,kshift,range(nkpts),range(nkpts)])

            for ka, kb in itertools.product(range(nkpts), repeat=2):
                kc = kclist[ka,kb]

                kptvec = [ki,kj,ka,kb,kc]
                lijabc[ka,kb] = contract_pl3p(l1,l2,kptvec)

                rijabc[ka,kb] = contract_pr3p(r1,r2,kptvec)

            for ka, kb in itertools.product(range(nkpts), repeat=2):
                kc = kclist[ka,kb]
                Plijabc[ka,kb] = (4.*lijabc[ka,kb] +
                                  1.*lijabc[kb,kc].transpose(0,1,4,2,3) +
                                  1.*lijabc[kc,ka].transpose(0,1,3,4,2) -
                                  2.*lijabc[ka,kc].transpose(0,1,2,4,3) -
                                  2.*lijabc[kc,kb].transpose(0,1,4,3,2) -
                                  2.*lijabc[kb,ka].transpose(0,1,3,2,4))
                eabc[ka,kb] = _get_epqr([0,nvir,ka,mo_e_v,eom.nonzero_vpadding],
                                        [0,nvir,kb,mo_e_v,eom.nonzero_vpadding],
                                        [0,nvir,kc,mo_e_v,eom.nonzero_vpadding],
                                        fac=[-1.,-1.,-1.])

            # Creating denominator
            eij = _get_epq([0,nocc,ki,mo_e_o,eom.nonzero_opadding],
                           [0,nocc,kj,mo_e_o,eom.nonzero_opadding])
            eijabc = (eij[None, None, :, :, None, None, None] +
                      eabc[:, :, None, None, :, :, :])
            denom = eijabc + ea_eval
            denom = 1. / denom

            deltaE += lib.einsum('xyijabc,xyijabc,xyijabc', Plijabc, rijabc, denom)

        deltaE *= 0.5
        deltaE = deltaE.real
        logger.info(eom, "eaccsd energy, star energy, delta energy = %16.12f, %16.12f, %16.12f",
                    ea_eval, ea_eval + deltaE, deltaE)
        e_star.append(ea_eval + deltaE)
    return e_star


class EOMEA(eom_kgccsd.EOMEA):
    matvec = eaccsd_matvec
    l_matvec = leaccsd_matvec
    get_diag = eaccsd_diag
    ccsd_star_contract = eaccsd_star_contract

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def ea_vector_desc(self):
        """Description of the EA vector."""
        nvir = self.nmo - self.nocc
        return [(nvir,), (self.nkpts, self.nkpts, self.nocc, nvir, nvir)]

    def ea_amplitudes_to_vector(self, t1, t2, kshift=None, kconserv=None):
        """Ground state amplitudes to a vector."""
        return nested_to_vector((t1, t2))[0]

    def ea_vector_to_amplitudes(self, vec):
        """Ground state vector to apmplitudes."""
        return vector_to_nested(vec, self.ea_vector_desc)

    def vector_to_amplitudes(self, vector, kshift=None):
        return self.ea_vector_to_amplitudes(vector)

    def amplitudes_to_vector(self, r1, r2, kshift=None, kconserv=None):
        return self.ea_amplitudes_to_vector(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nvir + nkpts**2*nocc*nvir*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ea()
        return imds

class EOMEA_Ta(EOMEA):
    '''Class for EOM EACCSD(T)*(a) method by Matthews and Stanton.'''
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_t3p2_ea(self._cc)
        return imds


########################################
# EOM-EE-CCSD
########################################

def eeccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None):
    '''See `kernel_ee()` for a description of arguments.'''
    raise NotImplementedError


def eomee_ccsd_singlet(eom, nroots=1, koopmans=False, guess=None, left=False,
                       eris=None, imds=None, diag=None, partition=None,
                       kptlist=None, dtype=None):
    '''See `eom_kgccsd.kernel()` for a description of arguments.'''
    eom.converged, eom.e, eom.v  \
            = eom_kgccsd.kernel_ee(eom, nroots, koopmans, guess, left, eris=eris,
                                   imds=imds, diag=diag, partition=partition,
                                   kptlist=kptlist, dtype=dtype)
    return eom.e, eom.v


def vector_to_amplitudes_singlet(vector, nkpts, nmo, nocc, kconserv):
    '''Transform 1-dimensional array to 3- and 7-dimensional arrays, r1 and r2.

    For example:
        vector: a 1-d array with all r1 elements, and r2 elements whose indices
    satisfy (i k_i a k_a) >= (j k_j b k_b)

        return: [r1, r2], where
        r1 = r_{i k_i}^{a k_a} is a 3-d array whose elements can be accessed via
            r1[k_i, i, a].

        r2 = r_{i k_i, j k_j}^{a k_a, b k_b} is a 7-d array whose elements can
            be accessed via r2[k_i, k_j, k_a, i, j, a, b]
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(sys.stdout, logger.DEBUG)
    nvir = nmo - nocc
    nov = nocc*nvir

    r1 = vector[:nkpts*nov].copy().reshape(nkpts, nocc, nvir)

    r2 = np.zeros((nkpts**2, nkpts, nov, nov), dtype=vector.dtype)

    idx, idy = np.tril_indices(nov)
    nov2_tril = nov * (nov + 1) // 2
    nov2 = nov * nov

    r2_tril = vector[nkpts*nov:].copy()
    offset = 0
    for ki, ka, kj in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        kika = ki * nkpts + ka
        kjkb = kj * nkpts + kb
        if kika == kjkb:
            tmp = r2_tril[offset:offset+nov2_tril]
            r2[kika, kj, idx, idy] = tmp
            r2[kjkb, ki, idy, idx] = tmp
            offset += nov2_tril
        elif kika > kjkb:
            tmp = r2_tril[offset:offset+nov2].reshape(nov, nov)
            r2[kika, kj] = tmp
            r2[kjkb, ki] = tmp.transpose()
            offset += nov2

    # r2 indices (old): (k_i, k_a), (k_J), (i, a), (J, B)
    # r2 indices (new): k_i, k_J, k_a, i, J, a, B
    r2 = r2.reshape(nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir).transpose(0,2,1,3,5,4,6)

    log.timer("vector_to_amplitudes_singlet", *cput0)
    return [r1, r2]


def amplitudes_to_vector_singlet(r1, r2, kconserv):
    '''Transform 3- and 7-dimensional arrays, r1 and r2, to a 1-dimensional
    array with unique indices.

    For example:
        r1: t_{i k_i}^{a k_a}
        r2: t_{i k_i, j k_j}^{a k_a, b k_b}
        return: a vector with all r1 elements, and r2 elements whose indices
    satisfy (i k_i a k_a) >= (j k_j b k_b)
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(sys.stdout, logger.DEBUG)
    # r1 indices: k_i, i, a
    nkpts, nocc, nvir = np.asarray(r1.shape)[[0, 1, 2]]
    nov = nocc * nvir

    # r2 indices (old): k_i, k_J, k_a, i, J, a, B
    # r2 indices (new): (k_i, k_a), (k_J), (i, a), (J, B)
    r2 = r2.transpose(0,2,1,3,5,4,6).reshape(nkpts**2, nkpts, nov, nov)

    idx, idy = np.tril_indices(nov)
    nov2_tril = nov * (nov + 1) // 2
    nov2 = nov * nov

    vector = np.empty(r2.size, dtype=r2.dtype)
    offset = 0
    for ki, ka, kj in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        kika = ki * nkpts + ka
        kjkb = kj * nkpts + kb
        r2ovov = r2[kika, kj]
        if kika == kjkb:
            vector[offset:offset+nov2_tril] = r2ovov[idx, idy]
            offset += nov2_tril
        elif kika > kjkb:
            vector[offset:offset+nov2] = r2ovov.ravel()
            offset += nov2

    vector = np.hstack((r1.ravel(), vector[:offset]))
    log.timer("amplitudes_to_vector_singlet", *cput0)
    return vector


def join_indices(indices, struct):
    '''Returns a joined index for an array of indices.

    Args:
        indices (np.array): an array of indices
        struct (np.array): an array of index ranges

    Example:
        indices = np.array((3, 4, 5))
        struct = np.array((10, 10, 10))
        join_indices(indices, struct): 345
    '''
    if not isinstance(indices, np.ndarray) or not isinstance(struct, np.ndarray):
        raise TypeError("Arguments %s and %s should both be numpy.ndarray" %
                        (repr(indices), repr(struct)))
    if indices.size != struct.size:
        raise ValueError("Structure shape mismatch: expected dimension = %d, found %d" %
                         (struct.size, indices.size))
    if (indices >= struct).all():
        raise ValueError("Indices are out of range")

    result = 0
    for dim in range(struct.size):
        result += indices[dim] * np.prod(struct[dim+1:])

    return result


def eeccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    raise NotImplementedError


def eeccsd_matvec_singlet(eom, vector, kshift, imds=None, diag=None):
    """Spin-restricted, k-point EOM-EE-CCSD equations for singlet excitation only.
    
    This implementation can be checked against the spin-orbital version in 
    `eom_kccsd_ghf.eeccsd_matvec()`.
    """
    cput0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)

    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    kconserv_r1 = eom.get_kconserv_ee_r1(kshift)
    kconserv_r2 = eom.get_kconserv_ee_r2(kshift)
    r1, r2 = vector_to_amplitudes_singlet(vector, nkpts, nmo, nocc, kconserv_r2)

    # Build antisymmetrized tensors that will be used later
    #   antisymmetrized r2   : rbar_ijab = 2 r_ijab - r_ijba
    #   antisymmetrized woOoV: wbar_nmie = 2 W_nmie - W_nmei
    #   antisymmetrized wvOvV: wbar_amfe = 2 W_amfe - W_amef
    #   antisymmetrized woVvO: wbar_mbej = 2 W_mbej - W_mbje
    r2bar = np.zeros_like(r2)
    woOoV_bar = np.zeros_like(imds.woOoV)
    wvOvV_bar = np.zeros_like(imds.wvOvV)
    woVvO_bar = np.zeros_like(imds.woVvO)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # rbar_ijab = 2 r_ijab - r_ijba
        #  ki - ka + kj - kb = kshift
        kb = kconserv_r2[ki, ka, kj]
        r2bar[ki, kj, ka] = 2. * r2[ki, kj, ka] - r2[ki, kj, kb].transpose(0,1,3,2)
        # wbar_nmie = 2 W_nmie - W_nmei = 2 W_nmie - W_mnie
        #  ki->kn, kj->km, ka->ki
        wkn = ki
        wkm = kj
        wki = ka
        #  kn + km - ki - ke = G
        wke = kconserv[wkn, wki, wkm]
        woOoV_bar[wkn, wkm, wki] = 2. * imds.woOoV[wkn, wkm, wki] - imds.woOoV[wkm, wkn, wki].transpose(1,0,2,3)
        # wbar_amfe = 2 W_amfe - W_amef
        #  ki->ka, kj->km, ka->kf, kb->ke
        wka = ki
        wkm = kj
        wkf = ka
        #  ka + km - kf - ke = G
        wke = kconserv[wka, wkf, wkm]
        wvOvV_bar[wka, wkm, wkf] = 2. * imds.wvOvV[wka, wkm, wkf] - imds.wvOvV[wka, wkm, wke].transpose(0,1,3,2)
        # wbar_mbej = 2 W_mbej - W_mbje
        #  ki->km, kj->kb, ka->ke
        wkm = ki
        wkb = kj
        wke = ka
        #  km + kb - ke - kj = G
        wkj = kconserv[wkm, wke, wkb]
        woVvO_bar[wkm, wkb, wke] = 2. * imds.woVvO[wkm, wkb, wke] - imds.woVoV[wkm, wkb, wkj].transpose(0,1,3,2)

    Hr1 = np.zeros_like(r1)
    for ki in range(nkpts):
        #  ki - ka = kshift
        ka = kconserv_r1[ki]
        # r_ia <- - F_mi r_ma
        #  km = ki
        Hr1[ki] -= einsum('mi,ma->ia', imds.Foo[ki], r1[ki])
        # r_ia <- F_ac r_ic
        Hr1[ki] += einsum('ac,ic->ia', imds.Fvv[ka], r1[ki])
        for km in range(nkpts):
            # r_ia <- (2 W_amie - W_maie) r_me
            #  km - ke = kshift
            ke = kconserv_r1[km]
            Hr1[ki] += 2. * einsum('maei,me->ia', imds.woVvO[km, ka, ke], r1[km])
            Hr1[ki] -= einsum('maie,me->ia', imds.woVoV[km, ka, ki], r1[km])

            # r_ia <- F_me (2 r_imae - r_miae)
            Hr1[ki] += 2. * einsum('me,imae->ia', imds.Fov[km], r2[ki, km, ka])
            Hr1[ki] -= einsum('me,miae->ia', imds.Fov[km], r2[km, ki, ka])

            for ke in range(nkpts):
                # r_ia <- (2 W_amef - W_amfe) r_imef
                Hr1[ki] += 2. * einsum('amef,imef->ia', imds.wvOvV[ka, km, ke], r2[ki, km, ke])
                #  ka + km - ke - kf = G
                kf = kconserv[ka, ke, km]
                Hr1[ki] -= einsum('amfe,imef->ia', imds.wvOvV[ka, km, kf], r2[ki, km, ke])

                # r_ia <- -W_mnie (2 r_mnae - r_nmae)
                # Rename dummy index ke -> kn
                kn = ke
                Hr1[ki] -= 2. * np.einsum('mnie,mnae->ia', imds.woOoV[km, kn, ki], r2[km, kn, ka])
                Hr1[ki] += np.einsum('mnie,nmae->ia', imds.woOoV[km, kn, ki], r2[kn, km, ka])

    Hr2 = np.zeros_like(r2)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # ki + kj - ka - kb = kshift
        kb = kconserv_r2[ki, ka, kj]

        # r_ijab <= - F_mj r_imab
        #  km = kj
        Hr2[ki, kj, ka] -= einsum('mj,imab->ijab', imds.Foo[kj], r2[ki, kj, ka])
        # r_ijab <= - F_mi r_jmba
        #  km = ki
        Hr2[ki, kj, ka] -= einsum('mi,jmba->ijab', imds.Foo[ki], r2[kj, ki, kb])
        # r_ijab <= F_be r_ijae
        Hr2[ki, kj, ka] += einsum('be,ijae->ijab', imds.Fvv[kb], r2[ki, kj, ka])
        # r_ijab <= F_ae r_jibe
        Hr2[ki, kj, ka] += einsum('ae,jibe->ijab', imds.Fvv[ka], r2[kj, ki, kb])

        # r_ijab <= W_abej r_ie
        #  ki - ke = kshift
        ke = kconserv_r1[ki]
        Hr2[ki, kj, ka] += einsum('abej,ie->ijab', imds.wvVvO[ka, kb, ke], r1[ki])
        # r_ijab <= W_baei r_je
        #  kj - ke = kshift
        ke = kconserv_r1[kj]
        Hr2[ki, kj, ka] += einsum('baei,je->ijab', imds.wvVvO[kb, ka, ke], r1[kj])
        # r_ijab <= - W_mbij r_ma
        #  km + kb - ki - kj = G
        # => ki - kb + kj - km = G
        km = kconserv[ki, kb, kj]
        Hr2[ki, kj, ka] -= einsum('mbij,ma->ijab', imds.woVoO[km, kb, ki], r1[km])
        # r_ijab <= - W_maji r_mb
        #  km + ka - kj - ki = G
        # => ki -ka + kj - km = G
        km = kconserv[ki, ka, kj]
        Hr2[ki, kj, ka] -= einsum('maji,mb->ijab', imds.woVoO[km, ka, kj], r1[km])

        tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=r2.dtype)
        for km in range(nkpts):
            # r_ijab <= (2 W_mbej - W_mbje) r_imae - W_mbej r_imea
            #  km + kb - ke - kj = G
            ke = kconserv[km, kj, kb]
            tmp += einsum('mbej,imae->ijab', woVvO_bar[km, kb, ke], r2[ki, km, ka])
            tmp -= einsum('mbej,imea->ijab', imds.woVvO[km, kb, ke], r2[ki, km, ke])
            # r_ijab <= - W_maje r_imeb
            #  km + ka - kj - ke = G
            ke = kconserv[km, kj, ka]
            tmp -= einsum('maje,imeb->ijab', imds.woVoV[km, ka, kj], r2[ki, km, ke])
        Hr2[ki, kj, ka] += tmp
        # The following two lines can be obtained by simply transposing tmp:
        #   r_ijab <= (2 W_maei - W_maie) r_jmbe - W_maei r_jmeb
        #   r_ijab <= - W_mbie r_jmea
        Hr2[kj, ki, kb] += tmp.transpose(1,0,3,2)
        tmp = None

        for km in range(nkpts):
            # r_ijab <= W_abef r_ijef
            # Rename dummy index km -> ke
            ke = km
            Hr2[ki, kj, ka] += einsum('abef,ijef->ijab', imds.wvVvV[ka, kb, ke], r2[ki, kj, ke])
            # r_ijab <= W_mnij r_mnab
            #  km + kn - ki - kj = G
            # => ki - km + kj - kn = G
            kn = kconserv[ki, km, kj]
            Hr2[ki, kj, ka] += einsum('mnij,mnab->ijab', imds.woOoO[km, kn, ki], r2[km, kn, ka])

    #
    # r_ijab <= - W_mnef t_imab (2 r_jnef - r_jnfe)
    # r_ijab <= - W_mnef t_jmba (2 r_inef - r_infe)
    # r_ijab <= - W_mnef t_ijae (2 r_mnbf - r_mnfb)
    # r_ijab <= - W_mnef t_jibe (2 r_mnaf - r_mnfa)
    #
    # r_ijab <= - (2 W_nmie - W_nmei) t_jnba r_me
    # r_ijab <= - (2 W_nmje - W_nmej) t_inab r_me
    # r_ijab <= + (2 W_amfe - W_amef) t_jibf r_me
    # r_ijab <= + (2 W_bmfe - W_bmef) t_ijaf r_me
    #
    # First, build intermediates M = W.r
    #
    wr2_oo = np.zeros((nkpts, nocc, nocc), dtype=r2.dtype)
    wr2_vv = np.zeros((nkpts, nvir, nvir), dtype=r2.dtype)
    wr1_oo = np.zeros_like(wr2_oo)
    wr1_vv = np.zeros_like(wr2_vv)
    for kj in range(nkpts):
        # Wr2_jm = W_mnef (2 r_jnef - r_jnfe) = W_mnef rbar_jnef
        #  km + kn - ke - kf = G
        #  kj + kn - ke - kf = kshift
        # => kj - km = kshift
        km = kconserv_r1[kj]
        # x: kn, y: ke
        wr2_oo[kj] += einsum('xymnef,xyjnef->jm', imds.woOvV[km], r2bar[kj])

        # Wr2_eb = W_mnef (2 r_mnbf - r_mnfb) = W_mnef rbar_mnbf
        ke = kj
        #  km + kn - ke - kf = G
        #  km + kn - kb - kf = kshift
        # => ke - kb = kshift
        kb = kconserv_r1[ke]
        # x: km, y: kn
        wr2_vv[ke] += einsum('xymnef,xymnbf->eb', imds.woOvV[:, :, ke], r2bar[:, :, kb])

        # Wr1_in = (2 W_nmie - W_nmei) r_me = wbar_nmie r_me
        ki = kj
        #  kn + km - ki - ke = G
        #  km - ke = kshift
        # => ki - kn = kshift
        kn = kconserv_r1[ki]
        # x: km
        wr1_oo[ki] += einsum('xnmie,xme->in', woOoV_bar[kn, :, ki], r1)

        # Wr1_fa = (2 W_amfe - W_amef) r_me = wbar_amfe r_me
        kf = kj
        #  ka + km - kf - ke = G
        #  km - ke = kshift
        # => kf - ka = kshift
        ka = kconserv_r1[kf]
        # x: km
        wr1_vv[kf] += einsum('xamfe,xme->fa', wvOvV_bar[ka, :, kf], r1)
    #
    # Second, compute the whole contraction
    #
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # ki + kj - ka - kb = kshift
        kb = kconserv_r2[ki, ka, kj]
        # r_ijab <= - Wr2_jm t_imab
        #  kj - km = kshift
        km = kconserv_r1[kj]
        Hr2[ki, kj, ka] -= einsum('jm,imab->ijab', wr2_oo[kj], imds.t2[ki, km, ka])
        # r_ijab <= - Wr2_im t_jmba
        #  ki - km = kshift
        km = kconserv_r1[ki]
        Hr2[ki, kj, ka] -= einsum('im,jmba->ijab', wr2_oo[ki], imds.t2[kj, km, kb])
        # r_ijab <= - Wr2_eb t_ijae
        #  ki + kj - ka - ke = G
        ke = kconserv[ki, ka, kj]
        Hr2[ki, kj, ka] -= einsum('eb,ijae->ijab', wr2_vv[ke], imds.t2[ki, kj, ka])
        # r_ijab <= - Wr2_ea t_jibe
        #  kj + ki - kb - ke = G
        ke = kconserv[kj, kb, ki]
        Hr2[ki, kj, ka] -= einsum('ea,jibe->ijab', wr2_vv[ke], imds.t2[kj, ki, kb])

        # r_ijab <= - Wr1_in t_jnba
        #  ki - kn = kshift
        kn = kconserv_r1[ki]
        Hr2[ki, kj, ka] -= einsum('in,jnba->ijab', wr1_oo[ki], imds.t2[kj, kn, kb])
        # r_ijab <= - Wr1_jn t_inab
        #  kj - kn = kshift
        kn = kconserv_r1[kj]
        Hr2[ki, kj, ka] -= einsum('jn,inab->ijab', wr1_oo[kj], imds.t2[ki, kn, ka])
        # r_ijab <= Wr1_fa t_jibf
        #  kj + ki - kb - kf = G
        kf = kconserv[kj, kb, ki]
        Hr2[ki, kj, ka] += einsum('fa,jibf->ijab', wr1_vv[kf], imds.t2[kj, ki, kb])
        # r_ijab <= Wr1_fb t_ijaf
        #  ki + kj - ka - kf = G
        kf = kconserv[ki, ka, kj]
        Hr2[ki, kj, ka] += einsum('fb,ijaf->ijab', wr1_vv[kf], imds.t2[ki, kj, ka])

    vector = amplitudes_to_vector_singlet(Hr1, Hr2, kconserv_r2)
    log.timer("matvec EOMEE Singlet", *cput0)
    return vector


def eeccsd_diag(eom, kshift=0, imds=None):
    '''Diagonal elements of similarity-transformed Hamiltonian'''
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = eom.kconserv
    kconserv_r1 = eom.get_kconserv_ee_r1(kshift)
    kconserv_r2 = eom.get_kconserv_ee_r2(kshift)

    Hr1 = np.zeros((nkpts, nocc, nvir), dtype=t1.dtype)
    for ki in range(nkpts):
        ka = kconserv_r1[ki]
        Hr1[ki] -= imds.Foo[ki].diagonal()[:,None]
        Hr1[ki] += imds.Fvv[ka].diagonal()[None,:]
        Hr1[ki] += np.einsum('iaai->ia', imds.woVvO[ki, ka, ka])
        Hr1[ki] -= np.einsum('iaia->ia', imds.woVoV[ki, ka, ki])

    Hr2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=t1.dtype)
    # TODO Allow partition='mp'
    if eom.partition == "mp":
        raise NotImplementedError
    else:
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv_r2[ki, ka, kj]
            Hr2[ki, kj, ka] -= imds.Foo[ki].diagonal()[:, None, None, None]
            Hr2[ki, kj, ka] -= imds.Foo[kj].diagonal()[None, :, None, None]
            Hr2[ki, kj, ka] += imds.Fvv[ka].diagonal()[None, None, :, None]
            Hr2[ki, kj, ka] += imds.Fvv[kb].diagonal()[None, None, None, :]

            Hr2[ki, kj, ka] += np.einsum('jbbj->jb', imds.woVvO[kj, kb, kb])[None, :, None, :]
            Hr2[ki, kj, ka] -= np.einsum('jbjb->jb', imds.woVoV[kj, kb, kj])[None, :, None, :]
            Hr2[ki, kj, ka] -= np.einsum('jaja->ja', imds.woVoV[kj, ka, kj])[None, :, :, None]
            Hr2[ki, kj, ka] -= np.einsum('ibib->ib', imds.woVoV[ki, kb, ki])[:, None, None, :]
            Hr2[ki, kj, ka] += np.einsum('iaai->ia', imds.woVvO[ki, ka, ka])[:, None, :, None]
            Hr2[ki, kj, ka] -= np.einsum('iaia->ia', imds.woVoV[ki, ka, ki])[:, None, :, None]

            Hr2[ki, kj, ka] += np.einsum('abab->ab', imds.wvVvV[ka, kb, ka])[None, None, :, :]
            Hr2[ki, kj, ka] += np.einsum('ijij->ij', imds.woOoO[ki, kj, ki])[:, :, None, None]

            # ki - ka + km - kb = G
            # => ka - ki + kb - km = G
            km = kconserv[ka, ki, kb]
            Hr2[ki, kj, ka] -= np.einsum('imab,imab->iab', imds.woOvV[ki, km, ka], imds.t2[ki, km, ka])[:, None, :, :]
            # km - ka + kj - kb = G
            # => ka - kj + kb - km = G
            km = kconserv[ka, kj, kb]
            Hr2[ki, kj, ka] -= np.einsum('mjab,mjab->jab', imds.woOvV[km, kj, ka], imds.t2[km, kj, ka])[None, :, :, :]
            # ki - ka + kj - ke = G
            Hr2[ki, kj, ka] -= np.einsum('ijae,ijae->ija', imds.woOvV[ki, kj, ka], imds.t2[ki, kj, ka])[:, :, :, None]
            # ki - ke + kj - kb = G
            ke = kconserv[ki, kb, kj]
            Hr2[ki, kj, ka] -= np.einsum('ijeb,ijeb->ijb', imds.woOvV[ki, kj, ke], imds.t2[ki, kj, ke])[:, :, None, :]

    vector = amplitudes_to_vector_singlet(Hr1, Hr2, kconserv_r2)
    return vector


def eeccsd_matvec_singlet_Hr1(eom, vector, kshift, imds=None):
    '''A mini version of eeccsd_matvec_singlet(), in the sense that
    only Hbar.r1 is performed.'''

    if imds is None: imds = eom.make_imds()
    nkpts = eom.nkpts
    nocc = eom.nocc
    nvir = eom.nmo - nocc
    r1_size = nkpts * nocc * nvir
    kconserv_r1 = eom.get_kconserv_ee_r1(kshift)

    if len(vector) != r1_size:
        raise ValueError("vector length mismatch: expected {0}, "
                         "found {1}".format(r1_size, len(vector)))
    r1 = vector.reshape(nkpts, nocc, nvir)

    Hr1 = np.zeros_like(r1)
    for ki in range(nkpts):
        #  ki - ka = kshift
        ka = kconserv_r1[ki]
        # r_ia <- - F_mi r_ma
        #  km = ki
        Hr1[ki] -= einsum('mi,ma->ia', imds.Foo[ki], r1[ki])
        # r_ia <- F_ac r_ic
        Hr1[ki] += einsum('ac,ic->ia', imds.Fvv[ka], r1[ki])
        for km in range(nkpts):
            # r_ia <- (2 W_amie - W_maie) r_me
            #  km - ke = kshift
            ke = kconserv_r1[km]
            Hr1[ki] += 2. * einsum('maei,me->ia', imds.woVvO[km, ka, ke], r1[km])
            Hr1[ki] -= einsum('maie,me->ia', imds.woVoV[km, ka, ki], r1[km])

    return Hr1.ravel()


def eeccsd_cis_approx_slow(eom, kshift, nroots=1, imds=None, **kwargs):
    '''Build initial R vector through diagonalization of <r1|Hbar|r1>

    This method evaluates the matrix elements of Hbar in r1 space in the following way:
    - 1st col of Hbar = matvec(r1_col1) where r1_col1 = [1, 0, 0, 0, ...]
    - 2nd col of Hbar = matvec(r1_col2) where r1_col2 = [0, 1, 0, 0, ...]
    - and so on

    Note that such evaluation has N^3 cost, but error free (because matvec() has been proven correct).
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)

    if imds is None: imds = eom.make_imds()
    nkpts, nocc, nvir = imds.t1.shape
    dtype = imds.t1.dtype
    r1_size = nkpts * nocc * nvir

    H1 = np.zeros([r1_size, r1_size], dtype=dtype)
    for col in range(r1_size):
        vec = np.zeros(r1_size, dtype=dtype)
        vec[col] = 1.0
        H1[:, col] = eeccsd_matvec_singlet_Hr1(eom, vec, kshift, imds=imds)

    eigval, eigvec = np.linalg.eig(H1)
    idx = eigval.argsort()[:nroots]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    log.timer("EOMEE CIS approx", *cput0)

    return eigval, eigvec


def get_init_guess_cis(eom, kshift, nroots=1, imds=None, **kwargs):
    '''Build initial R vector through diagonalization of <r1|Hbar|r1>

    Check eeccsd_cis_approx_slow() for details.
    '''
    if imds is None: imds = eom.make_imds()
    nkpts, nocc, nvir = imds.t1.shape
    dtype = imds.t1.dtype
    r1_size = nkpts * nocc * nvir
    vector_size = eom.vector_size(kshift)

    eigval, eigvec = eeccsd_cis_approx_slow(eom, kshift, nroots, imds)
    guess = []
    for i in range(nroots):
        g = np.zeros(int(vector_size), dtype=dtype)
        g[:r1_size] = eigvec[:, i]
        guess.append(g)

    return guess


def cis_easy(eom, nroots=1, kptlist=None, imds=None, **kwargs):
    '''An easy implementation of k-point CIS based on EOMCC infrastructure.'''

    print("\n******** <function 'pyscf.pbc.cc.eom_kccsd_rhf.cis_easy'> ********")

    if imds is None:
        cc = eom._cc
        t1_old, t2_old = cc.t1.copy(), cc.t2.copy()

        # Zero t1, t2
        cc.t1 = np.zeros_like(t1_old)
        cc.t2 = np.zeros_like(t2_old)

        # Remake intermediates using zero t1, t2 => get bare Hamiltonian back
        imds = eom.make_imds()

        # Recover t1, t2 so that the following calculations based on `eom` are
        # not affected.
        cc.t1, cc.t2 = None, None
        cc.t1, cc.t2 = t1_old, t2_old

    evals = [None]*len(kptlist)
    evecs = [None]*len(kptlist)
    for k, kshift in enumerate(kptlist):
        print("\nkshift =", kshift)
        eigval, eigvec = eeccsd_cis_approx_slow(eom, kshift, nroots, imds)
        evals[k] = eigval
        evecs[k] = eigvec
        for i in range(nroots):
            print('CIS root {:d} E = {:.16g}'.format(i, eigval[i].real))

    return evals, evecs


class EOMEE(eom_kgccsd.EOMEE):
    kernel = eeccsd
    eeccsd = eeccsd
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag

    @property
    def nkpts(self):
        return len(self.kpts)

    def vector_size(self, kshift=0):
        raise NotImplementedError

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds


class EOMEESinglet(EOMEE):
    kernel = eomee_ccsd_singlet
    eomee_ccsd_singlet = eomee_ccsd_singlet
    matvec = eeccsd_matvec_singlet
    get_init_guess = get_init_guess_cis
    cis = cis_easy

    def vector_size(self, kshift=0):
        '''Size of the linear excitation operator R vector based on spatial
        orbital basis.

        r1 : r_{i k_i}${a k_a}
        r2 : r_{i k_i, J k_J}^{a k_a, B k_B}

        Only r1aa, r2abab spin blocks are considered.
        '''
        nocc = self.nocc
        nvir = self.nmo - nocc
        nov = nocc * nvir
        nkpts = self.nkpts

        size_r1 = nkpts*nov

        kconserv = self.get_kconserv_ee_r2(kshift)
        size_r2 = 0
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki, ka, kj]
            kika = ki * nkpts + ka
            kjkb = kj * nkpts + kb
            if kika == kjkb:
                size_r2 += nov * (nov + 1) // 2
            elif kika > kjkb:
                size_r2 += nov**2

        return size_r1 + size_r2

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            # TODO allow left vectors to be computed
            raise NotImplementedError
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, kshift=None, nkpts=None, nmo=None, nocc=None, kconserv=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        if kconserv is None: kconserv = self.get_kconserv_ee_r2(kshift)
        return vector_to_amplitudes_singlet(vector, nkpts, nmo, nocc, kconserv)

    def amplitudes_to_vector(self, r1, r2, kshift=None, kconserv=None):
        if kconserv is None: kconserv = self.get_kconserv_ee_r2(kshift)
        return amplitudes_to_vector_singlet(r1, r2, kconserv)


class EOMEETriplet(EOMEE):

    def vector_size(self, kshift=0):
        return None


class EOMEESpinFlip(EOMEE):

    def vector_size(self, kshift=0):
        return None


imd = imdk

class _IMDS:
    # Identical to molecular rccsd_slow
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self.kconserv = cc.khelper.kconserv
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False
        # TODO: check whether to hold all stuff in memory
        if getattr(self.eris, "feri1", None):
            self._fimd = lib.H5TmpFile()
        else:
            self._fimd = None

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv
        self.Loo = imd.Loo(t1, t2, eris, kconserv)
        self.Lvv = imd.Lvv(t1, t2, eris, kconserv)
        self.Fov = imd.cc_Fov(t1, t2, eris, kconserv)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t1.shape
            ovov_dest = self._fimd.create_dataset('ovov', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), t1.dtype.char)
            ovvo_dest = self._fimd.create_dataset('ovvo', (nkpts, nkpts, nkpts, nocc, nvir, nvir, nocc), t1.dtype.char)
        else:
            ovov_dest = ovvo_dest = None

        # 2 virtuals
        self.Wovov = imd.Wovov(t1, t2, eris, kconserv, ovov_dest)
        self.Wovvo = imd.Wovvo(t1, t2, eris, kconserv, ovvo_dest)
        self.Woovv = eris.oovv

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t1.shape
            oooo_dest = self._fimd.create_dataset('oooo', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), t1.dtype.char)
            ooov_dest = self._fimd.create_dataset('ooov', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), t1.dtype.char)
            ovoo_dest = self._fimd.create_dataset('ovoo', (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), t1.dtype.char)
        else:
            oooo_dest = ooov_dest = ovoo_dest = None

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1, t2, eris, kconserv, oooo_dest)
        self.Wooov = imd.Wooov(t1, t2, eris, kconserv, ooov_dest)
        self.Wovoo = imd.Wovoo(t1, t2, eris, kconserv, ovoo_dest)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_t3p2_ip(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_tot, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a IP intermediates', *cput0)
        return self

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        if self._fimd is not None:
            nkpts, nocc, nvir = t1.shape
            vovv_dest = self._fimd.create_dataset('vovv', (nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir), t1.dtype.char)
            vvvo_dest = self._fimd.create_dataset('vvvo', (nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc), t1.dtype.char)
            if eris.vvvv is not None:
                vvvv_dest = self._fimd.create_dataset('vvvv', (nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), t1.dtype.char)
        else:
            vovv_dest = vvvo_dest = vvvv_dest = None

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris, kconserv, vovv_dest)
        if ea_partition == 'mp' and np.all(t1 == 0):
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, kconserv, vvvo_dest)
        else:
            if eris.vvvv is None:
                self.Wvvvv = None
            else:
                self.Wvvvv = imd.Wvvvv(t1, t2, eris, kconserv, vvvv_dest)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, kconserv, self.Wvvvv, vvvo_dest)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_t3p2_ea(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_tot, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ea()  # Make after t1/t2 updated
        self.Wvvvo = self.Wvvvo + Wvvvo

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a EA intermediates', *cput0)
        return self

    def make_t3p2_ip_ea(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_tot, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.make_ea()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo
        self.Wvvvo = self.Wvvvo + Wvvvo

        self.made_ip_imds = True
        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a IP/EA intermediates', *cput0)
        return self

    def make_ee(self, ee_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False:
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        kconserv = self.kconserv

        # Rename imds to match the notations in pyscf.cc.eom_rccsd
        self.Foo = self.Loo
        self.Fvv = self.Lvv
        self.woOvV = self.Woovv
        self.woVvO = self.Wovvo
        self.woVoV = self.Wovov

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.woOoO = imd.Woooo(t1, t2, eris, kconserv)
            self.woOoV = imd.Wooov(t1, t2, eris, kconserv)
            self.woVoO = imd.Wovoo(t1, t2, eris, kconserv)
        else:
            self.woOoO = self.Woooo
            self.woOoV = self.Wooov
            self.woVoO = self.Wovoo

        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.wvOvV = imd.Wvovv(t1, t2, eris, kconserv)
            self.wvVvV = imd.Wvvvv(t1, t2, eris, kconserv)
            self.wvVvO = imd.Wvvvo(t1, t2, eris, kconserv, self.wvVvV)
        else:
            self.wvOvV = self.Wvovv
            self.wvVvV = self.Wvvvv
            self.wvVvO = self.Wvvvo

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)

    def get_Wvvvv(self, ka, kb, kc):
        if not self.made_ea_imds:
            self.make_ea()

        if self.Wvvvv is None:
            return imd.get_Wvvvv(self.t1, self.t2, self.eris, self.kconserv,
                                 ka, kb, kc)
        else:
            return self.Wvvvv[ka,kb,kc]
