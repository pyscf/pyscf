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

import numpy as np
from pyscf import lib
from pyscf.lib import logger

#einsum = np.einsum
einsum = lib.einsum

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# Section (a)

def make_tau(t2, t1a, t1b, fac=1, out=None):
    t1t1 = einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    tau1 += t2
    return tau1

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    eris_vovv = np.asarray(eris.ovvv).transpose(1,0,3,2)
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)
    Fae = fvv - 0.5*einsum('me,ma->ae',fov, t1)
    Fae += einsum('mf,amef->ae', t1, eris_vovv)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    foo = eris.fock[:nocc,:nocc]
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)
    Fmi = ( foo + 0.5*einsum('me,ie->mi',fov, t1)
            + einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
    return Fmi

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fme = fov + einsum('nf,mnef->me', t1, eris.oovv)
    return Fme

def cc_Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('mb,mafe->bafe', t1, eris_ovvv)
    Wabef = np.asarray(eris.vvvv) - tmp + tmp.transpose(1,0,2,3)
    Wabef += einsum('mnab,mnef->abef', tau, 0.25*np.asarray(eris.oovv))
    return Wabef

def cc_Wovvo(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej

### Section (b)

def Fvv(t1, t2, eris):
    ccFov = cc_Fov(t1, t2, eris)
    Fae = cc_Fvv(t1, t2, eris) - 0.5*einsum('ma,me->ae', t1,ccFov)
    return Fae

def Foo(t1, t2, eris):
    ccFov = cc_Fov(t1, t2, eris)
    Fmi = cc_Foo(t1, t2, eris) + 0.5*einsum('ie,me->mi', t1,ccFov)
    return Fmi

def Fov(t1, t2, eris):
    Fme = cc_Fov(t1, t2, eris)
    return Fme

def Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    Wmnij = 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    Wmnij += cc_Woooo(t1, t2, eris)
    return Wmnij

def Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    Wabef = cc_Wvvvv(t1, t2, eris)
    Wabef += einsum('mnab,mnef->abef', tau, .25*np.asarray(eris.oovv))
    return Wabef

def Wovvo(t1, t2, eris):
    Wmbej = -0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej += cc_Wovvo(t1, t2, eris)
    return Wmbej

def Wooov(t1, t2, eris):
    Wmnie = einsum('if,mnfe->mnie', t1, eris.oovv)
    Wmnie += eris.ooov
    return Wmnie

def Wvovv(t1, t2, eris):
    Wamef = einsum('na,nmef->amef', -t1, eris.oovv)
    Wamef -= np.asarray(eris.ovvv).transpose(1,0,2,3)
    return Wamef

def Wovoo(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    tmp1 = einsum('mnie,jnbe->mbij', eris.ooov, t2)
    tmp2 = einsum('ie,mbej->mbij', t1, eris_ovvo)
    tmp2 -= einsum('ie,njbf,mnef->mbij', t1, t2, eris.oovv)
    FFov = Fov(t1, t2, eris)
    WWoooo = Woooo(t1, t2, eris)
    tau = make_tau(t2, t1, t1)
    Wmbij = einsum('me,ijbe->mbij', -FFov, t2)
    Wmbij -= einsum('nb,mnij->mbij', t1, WWoooo)
    Wmbij += 0.5 * einsum('mbef,ijef->mbij', eris.ovvv, tau)
    Wmbij += tmp1 - tmp1.transpose(0,1,3,2)
    Wmbij += tmp2 - tmp2.transpose(0,1,3,2)
    Wmbij += np.asarray(eris.ooov).conj().transpose(2,3,0,1)
    return Wmbij

def Wvvvo(t1, t2, eris, _Wvvvv=None):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_vvvo = -np.asarray(eris.ovvv).transpose(2,3,1,0).conj()
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    tmp1 = einsum('mbef,miaf->abei', eris.ovvv, t2)
    tmp2 = einsum('ma,mbei->abei', t1, eris_ovvo)
    tmp2 -= einsum('ma,nibf,mnef->abei', t1, t2, eris.oovv)
    FFov = Fov(t1, t2, eris)
    tau = make_tau(t2, t1, t1)
    Wabei  = 0.5 * einsum('mnei,mnab->abei', eris_oovo, tau)
    Wabei -= einsum('me,miab->abei', FFov, t2)
    Wabei += eris_vvvo
    Wabei -= tmp1 - tmp1.transpose(1,0,2,3)
    Wabei -= tmp2 - tmp2.transpose(1,0,2,3)
    nocc,nvir = t1.shape
    if _Wvvvv is None:
        _Wvvvv = Wvvvv(t1, t2, eris)
    Wabei += einsum('abef,if->abei', _Wvvvv, t1)
    return Wabei

########################################
# T3[2] related contributions to T1/T2
########################################

def _cp(a):
    return np.array(a, copy=False, order='C')

def get_t3p2_imds_slow(cc, t1, t2, eris=None, t3p2_ip_out=None, t3p2_ea_out=None):
    """Calculates T1, T2 amplitudes corrected by second-order T3 contribution
    and intermediates used in IP/EA-CCSD(T)a

    Args:
        cc (:obj:`GCCSD`):
            Object containing coupled-cluster results.
        t1 (:obj:`ndarray`):
            T1 amplitudes.
        t2 (:obj:`ndarray`):
            T2 amplitudes from which the T3[2] amplitudes are formed.
        eris (:obj:`_PhysicistsERIs`):
            Antisymmetrized electron-repulsion integrals in physicist's notation.
        t3p2_ip_out (:obj:`ndarray`):
            Store results of the intermediate used in IP-EOM-CCSD(T)a.
        t3p2_ea_out (:obj:`ndarray`):
            Store results of the intermediate used in EA-EOM-CCSD(T)a.

    Returns:
        delta_ccsd (float):
            Difference of perturbed and unperturbed CCSD ground-state energy,
                energy(T1 + T1[2], T2 + T2[2]) - energy(T1, T2)
        pt1 (:obj:`ndarray`):
            Perturbatively corrected T1 amplitudes.
        pt2 (:obj:`ndarray`):
            Perturbatively corrected T2 amplitudes.

    Reference:
        D. A. Matthews, J. F. Stanton "A new approach to approximate..."
            JCP 145, 124102 (2016); DOI:10.1063/1.4962910, Equation 14
        Shavitt and Bartlett "Many-body Methods in Physics and Chemistry"
            2009, Equation 10.33
    """
    if eris is None:
        eris = cc.ao2mo()
    fock = eris.fock
    nocc, nvir = t1.shape

    fov = fock[:nocc, nocc:]
    #foo = fock[:nocc, :nocc].diagonal()
    #fvv = fock[nocc:, nocc:].diagonal()

    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:]

    oovv = _cp(eris.oovv)
    ovvv = _cp(eris.ovvv)
    ooov = _cp(eris.ooov)
    oovv = _cp(eris.oovv)
    vooo = _cp(ooov).conj().transpose(3, 2, 1, 0)
    vvvo = _cp(ovvv).conj().transpose(3, 2, 1, 0)

    ccsd_energy = cc.energy(t1, t2, eris)
    dtype = np.result_type(t1, t2)
    if np.issubdtype(dtype, np.dtype(complex).type):
        logger.error(cc, 't3p2 imds has not been strictly checked for use with complex integrals')

    if t3p2_ip_out is None:
        t3p2_ip_out = np.zeros((nocc,nvir,nocc,nocc), dtype=dtype)
    Wmcik = t3p2_ip_out

    if t3p2_ea_out is None:
        t3p2_ea_out = np.zeros((nvir,nvir,nvir,nocc), dtype=dtype)
    Wacek = t3p2_ea_out

    tmp_t3 = lib.einsum('bcdk,ijad->ijkabc', vvvo, t2)
    tmp_t3 -= lib.einsum('cmkj,imab->ijkabc', vooo, t2)

    # P(ijk)
    tmp_t3 = (tmp_t3 + tmp_t3.transpose(1, 2, 0, 3, 4, 5) +
                       tmp_t3.transpose(2, 0, 1, 3, 4, 5))
    # P(abc)
    tmp_t3 = (tmp_t3 + tmp_t3.transpose(0, 1, 2, 4, 5, 3) +
                       tmp_t3.transpose(0, 1, 2, 5, 3, 4))

    eia = mo_e_o[:, None] - mo_e_v[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    eijkabc = eijab[:, :, None, :, :, None] + eia[None, None, :, None, None, :]
    tmp_t3 /= eijkabc

    pt1 = 0.25 * lib.einsum('mnef,imnaef->ia', oovv, tmp_t3)

    pt2 = lib.einsum('ijmabe,me->ijab', tmp_t3, fov)
    tmp2 = ovvv - 0.0*lib.einsum('nmef,nb->mbfe', oovv, t1)
    tmp = lib.einsum('ijmaef,mbfe->ijab', tmp_t3, tmp2)
    tmp = tmp - tmp.transpose(0, 1, 3, 2)  # P(ab)
    tmp *= 0.5
    pt2 = pt2 + tmp
    tmp2 = ooov + 0.0*lib.einsum('mnef,jf->nmje', oovv, t1)
    tmp = lib.einsum('inmabe,nmje->ijab', tmp_t3, tmp2)
    tmp *= -0.5
    tmp = tmp - tmp.transpose(1, 0, 2, 3)  # P(ij)
    pt2 = pt2 + tmp

    eia = mo_e_o[:, None] - mo_e_v[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]

    pt1 /= eia
    pt2 /= eijab

    pt1 = pt1 + t1
    pt2 = pt2 + t2

    Wmcik += 0.5*lib.einsum('ijkabc,mjab->mcik', tmp_t3, oovv)
    Wacek += -0.5*lib.einsum('ijkabc,ijeb->acek', tmp_t3, oovv)

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)
    return delta_ccsd_energy, pt1, pt2, Wmcik, Wacek
