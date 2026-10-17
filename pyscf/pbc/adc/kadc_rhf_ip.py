# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import time
import numpy as np
from pyscf import ao2mo
import pyscf.adc
import pyscf.adc.radc
from pyscf.adc import radc_ao2mo
import itertools

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import mp
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_rhf
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx, _padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM

from pyscf.pbc import tools
import h5py
import tempfile


def vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc
    size = n_singles + n_doubles

    return size


def get_imds(adc, eris=None):

    cput0 = (time.process_time(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_occ = np.identity(nocc)
    M_ij = np.empty((nkpts, nocc, nocc), dtype=mo_coeff[0].dtype)

    if eris is None:
        eris = adc.transform_integrals()

    # i-j block
    # Zeroth-order terms

    t2_1 = adc.t2[0]
    eris_ovov = eris.ovov
    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ, e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj, kd, kl]
                t2_1 = adc.t2[0]
                t2_1_ild = adc.t2[0][ki, kl, kd]

                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ilde,jdle->ij', t2_1_ild, eris_ovov[kj, kd, kl], optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ilde,jeld->ij', t2_1_ild, eris_ovov[kj, ke, kl], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ilde,jdle->ij', t2_1_ild,
                                             eris_ovov[kj, kd, kl], optimize=True)
                del t2_1_ild

                t2_1_lid = adc.t2[0][kl, ki, kd]
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('lide,jdle->ij', t2_1_lid, eris_ovov[kj, kd, kl], optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('lide,jeld->ij', t2_1_lid, eris_ovov[kj, ke, kl], optimize=True)
                del t2_1_lid

                t2_1_jld = adc.t2[0][kj, kl, kd]
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('jlde,idle->ij', t2_1_jld.conj(),
                               eris_ovov[ki, kd, kl].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('jlde,ield->ij', t2_1_jld.conj(),
                               eris_ovov[ki, ke, kl].conj(), optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('jlde,idle->ij', t2_1_jld.conj(),
                                             eris_ovov[ki, kd, kl].conj(), optimize=True)
                del t2_1_jld

                t2_1_ljd = adc.t2[0][kl, kj, kd]
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ljde,idle->ij', t2_1_ljd.conj(),
                               eris_ovov[ki, kd, kl].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ljde,ield->ij', t2_1_ljd.conj(),
                               eris_ovov[ki, ke, kl].conj(), optimize=True)
                del t2_1_ljd
                del t2_1

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    if (method == "adc(3)"):
        t1_2 = adc.t1[0]
        eris_ovoo = eris.ovoo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv
        eris_oooo = eris.oooo

        for ki in range(nkpts):
            kj = ki
            for kl in range(nkpts):
                kd = kconserv[kj, ki, kl]
                M_ij[ki] += 2. * lib.einsum('ld,ldji->ij', t1_2[kl],
                                            eris_ovoo[kl, kd, kj], optimize=True)
                M_ij[ki] -= lib.einsum('ld,jdli->ij', t1_2[kl],
                                       eris_ovoo[kj, kd, kl], optimize=True)

                kd = kconserv[ki, kj, kl]
                M_ij[ki] += 2. * lib.einsum('ld,ldij->ij', t1_2[kl].conj(),
                                            eris_ovoo[kl, kd, ki].conj(), optimize=True)
                M_ij[ki] -= lib.einsum('ld,idlj->ij', t1_2[kl].conj(),
                                       eris_ovoo[ki, kd, kl].conj(), optimize=True)

                for kd in range(nkpts):
                    t2_1 = adc.t2[0]

                    ke = kconserv[kj, kd, kl]
                    t2_2_ild = adc.t2[1][ki, kl, kd]
                    M_ij[ki] += 0.5 * 0.5 * \
                        lib.einsum('ilde,jdle->ij', t2_2_ild, eris_ovov[kj, kd, kl], optimize=True)
                    M_ij[ki] -= 0.5 * 0.5 * \
                        lib.einsum('ilde,jeld->ij', t2_2_ild, eris_ovov[kj, ke, kl], optimize=True)
                    M_ij[ki] += 0.5 * \
                        lib.einsum('ilde,jdle->ij', t2_2_ild, eris_ovov[kj, kd, kl], optimize=True)
                    del t2_2_ild

                    t2_2_lid = adc.t2[1][kl, ki, kd]
                    M_ij[ki] -= 0.5 * 0.5 * \
                        lib.einsum('lide,jdle->ij', t2_2_lid, eris_ovov[kj, kd, kl], optimize=True)
                    M_ij[ki] += 0.5 * 0.5 * \
                        lib.einsum('lide,jeld->ij', t2_2_lid, eris_ovov[kj, ke, kl], optimize=True)

                    t2_2_jld = adc.t2[1][kj, kl, kd]
                    M_ij[ki] += 0.5 * 0.5 * \
                        lib.einsum('jlde,leid->ij', t2_2_jld.conj(),
                                   eris_ovov[kl, ke, ki].conj(), optimize=True)
                    M_ij[ki] -= 0.5 * 0.5 * \
                        lib.einsum('jlde,ield->ij', t2_2_jld.conj(),
                                   eris_ovov[ki, ke, kl].conj(), optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('jlde,leid->ij', t2_2_jld.conj(),
                                                 eris_ovov[kl, ke, ki].conj(), optimize=True)

                    t2_2_ljd = adc.t2[1][kl, kj, kd]
                    M_ij[ki] -= 0.5 * 0.5 * \
                        lib.einsum('ljde,leid->ij', t2_2_ljd.conj(),
                                   eris_ovov[kl, ke, ki].conj(), optimize=True)
                    M_ij[ki] += 0.5 * 0.5 * \
                        lib.einsum('ljde,ield->ij', t2_2_ljd.conj(),
                                   eris_ovov[ki, ke, kl].conj(), optimize=True)

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                t2_1 = adc.t2[0]

                kl = kconserv[kd, km, ke]
                kf = kconserv[kj, kd, kl]
                temp_t2_v_1 = lib.einsum(
                    'lmde,jldf->mejf', t2_1[kl, km, kd].conj(), t2_1[kj, kl, kd], optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('mejf,ifem->ij',
                                                 temp_t2_v_1, eris_ovvo[ki, kf, ke], optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('meif,jfem->ij', temp_t2_v_1.conj(),
                                                 eris_ovvo[kj, kf, ke].conj(), optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('mejf,imef->ij',
                                             temp_t2_v_1, eris_oovv[ki, km, ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('meif,jmef->ij', temp_t2_v_1.conj(),
                                             eris_oovv[kj, km, ke].conj(), optimize=True)

                temp_t2_v_new = lib.einsum(
                    'mlde,ljdf->mejf', t2_1[km, kl, kd].conj(), t2_1[kl, kj, kd], optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('mejf,ifem->ij',
                                                 temp_t2_v_new, eris_ovvo[ki, kf, ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('mejf,imef->ij',
                                             temp_t2_v_new, eris_oovv[ki, km, ke], optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('meif,jfem->ij', temp_t2_v_new.conj(),
                                                 eris_ovvo[kj, kf, ke].conj(), optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('meif,jmef->ij', temp_t2_v_new.conj(),
                                             eris_oovv[kj, km, ke].conj(), optimize=True)
                del temp_t2_v_new

                temp_t2_v_2 = lib.einsum(
                    'lmde,ljdf->mejf', t2_1[kl, km, kd].conj(), t2_1[kl, kj, kd], optimize=True)
                M_ij[ki] += 0.5 * 4 * lib.einsum('mejf,ifem->ij',
                                                 temp_t2_v_2, eris_ovvo[ki, kf, ke], optimize=True)
                M_ij[ki] += 0.5 * 4 * lib.einsum('meif,jfem->ij', temp_t2_v_2.conj(),
                                                 eris_ovvo[kj, kf, ke].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('meif,jmef->ij', temp_t2_v_2.conj(),
                                                 eris_oovv[kj, km, ke].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('mejf,imef->ij',
                                                 temp_t2_v_2, eris_oovv[ki, km, ke], optimize=True)
                del temp_t2_v_2

                temp_t2_v_3 = lib.einsum(
                    'mlde,jldf->mejf', t2_1[km, kl, kd].conj(), t2_1[kj, kl, kd], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('mejf,ifem->ij',
                                             temp_t2_v_3, eris_ovvo[ki, kf, ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('meif,jfem->ij', temp_t2_v_3.conj(),
                                             eris_ovvo[kj, kf, ke].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('meif,jmef->ij', temp_t2_v_3.conj(),
                                                 eris_oovv[kj, km, ke].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 2 * lib.einsum('mejf,imef->ij',
                                                 temp_t2_v_3, eris_oovv[ki, km, ke], optimize=True)
                del temp_t2_v_3

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):

                kl = kconserv[kd, km, ke]
                kf = kconserv[kl, kd, km]
                temp_t2_v_8 = lib.einsum(
                    'lmdf,lmde->fe', t2_1[kl, km, kd], t2_1[kl, km, kd].conj(), optimize=True)
                M_ij[ki] += 3.0 * lib.einsum('fe,jief->ij', temp_t2_v_8,
                                             eris_oovv[kj, ki, ke], optimize=True)
                M_ij[ki] -= 1.5 * lib.einsum('fe,jfei->ij', temp_t2_v_8,
                                             eris_ovvo[kj, kf, ke], optimize=True)
                M_ij[ki] += lib.einsum('ef,jief->ij', temp_t2_v_8.T,
                                       eris_oovv[kj, ki, ke], optimize=True)
                M_ij[ki] -= 0.5 * lib.einsum('ef,jfei->ij', temp_t2_v_8.T,
                                             eris_ovvo[kj, kf, ke], optimize=True)
                del temp_t2_v_8

                temp_t2_v_9 = lib.einsum(
                    'lmdf,mlde->fe', t2_1[kl, km, kd], t2_1[km, kl, kd].conj(), optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('fe,jief->ij', temp_t2_v_9,
                                             eris_oovv[kj, ki, ke], optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('ef,jief->ij', temp_t2_v_9.T,
                                             eris_oovv[kj, ki, ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('fe,jfei->ij', temp_t2_v_9,
                                             eris_ovvo[kj, kf, ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ef,jfei->ij', temp_t2_v_9.T,
                                             eris_ovvo[kj, kf, ke], optimize=True)
                del temp_t2_v_9

                kl = kconserv[kd, km, ke]
                kn = kconserv[kd, kl, ke]
                temp_t2_v_10 = lib.einsum(
                    'lnde,lmde->nm', t2_1[kl, kn, kd], t2_1[kl, km, kd].conj(), optimize=True)
                M_ij[ki] -= 3.0 * lib.einsum('nm,jinm->ij', temp_t2_v_10,
                                             eris_oooo[kj, ki, kn], optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('mn,jinm->ij', temp_t2_v_10.T,
                                             eris_oooo[kj, ki, kn], optimize=True)
                M_ij[ki] += 1.5 * lib.einsum('nm,jmni->ij', temp_t2_v_10,
                                             eris_oooo[kj, km, kn], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('mn,jmni->ij', temp_t2_v_10.T,
                                             eris_oooo[kj, km, kn], optimize=True)
                del temp_t2_v_10

                temp_t2_v_11 = lib.einsum(
                    'lnde,mlde->nm', t2_1[kl, kn, kd], t2_1[km, kl, kd].conj(), optimize=True)
                M_ij[ki] += 1.0 * lib.einsum('nm,jinm->ij', temp_t2_v_11,
                                             eris_oooo[kj, ki, kn], optimize=True)
                M_ij[ki] -= 0.5 * lib.einsum('nm,jmni->ij', temp_t2_v_11,
                                             eris_oooo[kj, km, kn], optimize=True)
                M_ij[ki] -= 0.5 * lib.einsum('mn,jmni->ij', temp_t2_v_11.T,
                                             eris_oooo[kj, km, kn], optimize=True)
                M_ij[ki] += 1.0 * lib.einsum('mn,jinm->ij', temp_t2_v_11.T,
                                             eris_oooo[kj, ki, kn], optimize=True)
                del temp_t2_v_11

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                t2_1 = adc.t2[0]
                kl = kconserv[kd, km, ke]
                kn = kconserv[kd, ki, ke]
                temp_t2_v_12 = lib.einsum(
                    'inde,lmde->inlm', t2_1[ki, kn, kd].conj(), t2_1[kl, km, kd], optimize=True)
                M_ij[ki] += 0.5 * 1.25 * \
                    lib.einsum('inlm,jlnm->ij', temp_t2_v_12,
                               eris_oooo[kj, kl, kn].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('inlm,jmnl->ij', temp_t2_v_12,
                               eris_oooo[kj, km, kn].conj(), optimize=True)

                M_ij[ki] += 0.5 * 1.25 * \
                    lib.einsum('jnlm,ilnm->ij', temp_t2_v_12.conj(),
                               eris_oooo[ki, kl, kn], optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('jnlm,imnl->ij', temp_t2_v_12.conj(),
                               eris_oooo[ki, km, kn], optimize=True)
                del temp_t2_v_12

                temp_t2_v_12_1 = lib.einsum(
                    'nide,mlde->inlm', t2_1[kn, ki, kd].conj(), t2_1[km, kl, kd], optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('inlm,jlnm->ij', temp_t2_v_12_1,
                               eris_oooo[kj, kl, kn].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('inlm,jmnl->ij', temp_t2_v_12_1,
                               eris_oooo[kj, km, kn].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('jnlm,ilnm->ij', temp_t2_v_12_1.conj(),
                               eris_oooo[ki, kl, kn], optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('jnlm,imnl->ij', temp_t2_v_12_1.conj(),
                               eris_oooo[ki, km, kn], optimize=True)

                temp_t2_v_13 = lib.einsum(
                    'inde,mlde->inml', t2_1[ki, kn, kd].conj(), t2_1[km, kl, kd], optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('inml,jlnm->ij', temp_t2_v_13,
                               eris_oooo[kj, kl, kn].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('inml,jmnl->ij', temp_t2_v_13,
                               eris_oooo[kj, km, kn].conj(), optimize=True)

                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('jnml,ilnm->ij', temp_t2_v_13.conj(),
                               eris_oooo[ki, kl, kn], optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('jnml,imnl->ij', temp_t2_v_13.conj(),
                               eris_oooo[ki, km, kn], optimize=True)
                del temp_t2_v_13

                temp_t2_v_13_1 = lib.einsum(
                    'nide,lmde->inml', t2_1[kn, ki, kd].conj(), t2_1[kl, km, kd], optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('inml,jmnl->ij', temp_t2_v_13_1,
                               eris_oooo[kj, km, kn].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('jnml,imnl->ij', temp_t2_v_13_1.conj(),
                               eris_oooo[ki, km, kn], optimize=True)
                del temp_t2_v_13_1

    return M_ij


def get_diag(adc, kshift, M_ij=None, eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv
    nocc = adc.nocc
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    nocc = adc.nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    diag = np.zeros((dim), dtype=np.complex128)
    doubles = np.zeros((nkpts, nkpts, nvir * nocc * nocc), dtype=np.complex128)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij[kshift])

    # Compute precond in 2p1h-2p1h block
    for ka in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[kshift, ki, ka]
            d_ij = e_occ[ki][:, None] + e_occ[kj]
            d_a = e_vir[ka][:nvir, None]
            D_n = -d_a + d_ij.reshape(-1)
            doubles[ka, ki] += D_n.reshape(-1)

    if (adc.frozen is not None) or (not np.all([x.shape[1] == adc.nmo for x in mo_coeff])):
        doubles = doubles.reshape(nkpts, nkpts, nvir, nocc, nocc)
        (M_ij_diag, doubles) = mask_frozen_ip(adc, M_ij_diag, doubles, kshift)

    diag[s1:f1] = M_ij_diag.copy()
    diag[s2:f2] = doubles.reshape(-1)
    diag = -diag

    log.timer_debug1("Completed ip_diag calculation")

    return diag


def matvec(adc, kshift, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    s_singles = 0
    f_singles = n_singles
    s_doubles = f_singles
    f_doubles = s_doubles + n_doubles

    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:nocc + nvir] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    if M_ij is None:
        M_ij = adc.get_imds()

    # Calculate sigma vector
    def sigma_(r):
        cput0 = (time.process_time(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        r1 = r[s_singles:f_singles]
        r2 = r[s_doubles:f_doubles]

        r2 = r2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        s2 = np.zeros((nkpts, nkpts, nvir, nocc, nocc), dtype=np.complex128)
        cell = adc.cell
        kpts = adc.kpts
        madelung = tools.madelung(cell, kpts)

        eris_ovoo = eris.ovoo

############ ADC(2) ij block ############################

        s1 = lib.einsum('ij,j->i', M_ij[kshift], r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]
                ki = kconserv[kj, kk, ka]

                s1 += 2. * lib.einsum('jaki,ajk->i',
                                      eris_ovoo[kj, ka, kk].conj(), r2[ka, kj], optimize=True)
                s1 -= lib.einsum('kaji,ajk->i',
                                 eris_ovoo[kk, ka, kj].conj(), r2[ka, kj], optimize=True)
#################### ADC(2) ajk - i block ############################

                s2[ka, kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj, ka, kk], r1, optimize=True)

################# ADC(2) ajk - bil block ############################

                s2[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2[ka, kj])
                s2[ka, kj] += lib.einsum('j,ajk->ajk', e_occ[kj], r2[ka, kj])
                s2[ka, kj] += lib.einsum('k,ajk->ajk', e_occ[kk], r2[ka, kj])

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    for kl in range(nkpts):
                        ki = kconserv[kj, kl, kk]

                        s2[ka, kj] -= 0.5 * lib.einsum('kijl,ali->ajk',
                                                       eris_oooo[kk, ki, kj], r2[ka, kl], optimize=True)
                        s2[ka, kj] -= 0.5 * lib.einsum('klji,ail->ajk',
                                                       eris_oooo[kk, kl, kj], r2[ka, ki], optimize=True)

                    for kl in range(nkpts):
                        kb = kconserv[ka, kk, kl]
                        s2[ka, kj] += 0.5 * lib.einsum('klba,bjl->ajk',
                                                       eris_oovv[kk, kl, kb], r2[kb, kj], optimize=True)

                        kb = kconserv[kl, kj, ka]
                        s2[ka, kj] += 0.5 * lib.einsum('jabl,bkl->ajk',
                                                       eris_ovvo[kj, ka, kb], r2[kb, kk], optimize=True)
                        s2[ka, kj] -= lib.einsum('jabl,blk->ajk',
                                                 eris_ovvo[kj, ka, kb], r2[kb, kl], optimize=True)
                        kb = kconserv[ka, kj, kl]
                        s2[ka, kj] += 0.5 * lib.einsum('jlba,blk->ajk',
                                                       eris_oovv[kj, kl, kb], r2[kb, kl], optimize=True)

                    for ki in range(nkpts):
                        kb = kconserv[ka, kk, ki]
                        s2[ka, kj] += 0.5 * lib.einsum('kiba,bji->ajk',
                                                       eris_oovv[kk, ki, kb], r2[kb, kj], optimize=True)

                        kb = kconserv[ka, kj, ki]
                        s2[ka, kj] += 0.5 * lib.einsum('jiba,bik->ajk',
                                                       eris_oovv[kj, ki, kb], r2[kb, ki], optimize=True)
                        s2[ka, kj] -= lib.einsum('jabi,bik->ajk', eris_ovvo[kj,
                                                                            ka, kb], r2[kb, ki], optimize=True)
                        kb = kconserv[ki, kj, ka]
                        s2[ka, kj] += 0.5 * lib.einsum('jabi,bki->ajk',
                                                       eris_ovvo[kj, ka, kb], r2[kb, kk], optimize=True)

            if adc.exxdiv is not None:
                s2 += madelung * r2

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

################# ADC(3) i - kja block and ajk - i ############################

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kj, kshift, kk]

                    for kb in range(nkpts):
                        kc = kconserv[kj, kb, kk]
                        t2_1 = adc.t2[0]
                        temp_1 = lib.einsum(
                            'jkbc,ajk->abc', t2_1[kj, kk, kb], r2[ka, kj], optimize=True)
                        temp = 0.25 * lib.einsum('jkbc,ajk->abc',
                                                 t2_1[kj, kk, kb], r2[ka, kj], optimize=True)
                        temp -= 0.25 * lib.einsum('jkbc,akj->abc',
                                                  t2_1[kj, kk, kb], r2[ka, kk], optimize=True)
                        temp -= 0.25 * lib.einsum('kjbc,ajk->abc',
                                                  t2_1[kk, kj, kb], r2[ka, kj], optimize=True)
                        temp += 0.25 * lib.einsum('kjbc,akj->abc',
                                                  t2_1[kk, kj, kb], r2[ka, kk], optimize=True)
                        ki = kconserv[kc, ka, kb]
                        if eris.ovvv is None:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nocc)
                            a = 0
                            for p in range(0, nocc, chnk_size):
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[ki, kc], eris.Lvv[ka, kb], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                k = eris_ovvv.shape[0]
                                s1[a:a + k] += lib.einsum('abc,icab->i', temp_1,
                                                          eris_ovvv, optimize=True)
                                s1[a:a + k] += lib.einsum('abc,icab->i', temp,
                                                          eris_ovvv, optimize=True)
                                del eris_ovvv
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[ki, kb], eris.Lvv[ka, kc], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                s1[a:a + k] -= lib.einsum('abc,ibac->i', temp,
                                                          eris_ovvv, optimize=True)
                                del eris_ovvv
                                a += k
                        else:
                            eris_ovvv = eris.ovvv[:]
                            s1 += lib.einsum('abc,icab->i', temp_1,
                                             eris_ovvv[ki, kc, ka], optimize=True)
                            s1 += lib.einsum('abc,icab->i', temp,
                                             eris_ovvv[ki, kc, ka], optimize=True)
                            s1 -= lib.einsum('abc,ibac->i', temp,
                                             eris_ovvv[ki, kb, ka], optimize=True)
                            del eris_ovvv
            del temp
            del temp_1

            t2_1 = adc.t2[0]

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kj, kshift, kk]
                    for kc in range(nkpts):
                        kb = kconserv[kj, kc, kk]
                        ki = kconserv[kb, ka, kc]
                        if eris.ovvv is None:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nocc)
                            a = 0
                            for p in range(0, nocc, chnk_size):

                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[ki, kc], eris.Lvv[ka, kb], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                k = eris_ovvv.shape[0]
                                temp = lib.einsum(
                                    'i,icab->cba', r1[a:a + k], eris_ovvv.conj(), optimize=True)
                                del eris_ovvv
                                a += k
                        else:
                            eris_ovvv = eris.ovvv[:]
                            temp = lib.einsum(
                                'i,icab->cba', r1, eris_ovvv[ki, kc, ka].conj(), optimize=True)
                            del eris_ovvv
                        s2[ka, kj] += lib.einsum('cba,jkbc->ajk', temp,
                                                 t2_1[kj, kk, kb].conj(), optimize=True)
            del temp

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kj, kshift, kk]
                    for kb in range(nkpts):
                        kl = kconserv[ka, kj, kb]

                        t2_1 = adc.t2[0]
                        temp = lib.einsum('ljba,ajk->blk', t2_1[kl, kj, kb], r2[ka, kj], optimize=True)
                        temp_2 = lib.einsum('jlba,akj->blk', t2_1[kj, kl, kb], r2[ka, kk], optimize=True)
                        del t2_1

                        t2_1_jla = adc.t2[0][kj, kl, ka]
                        temp += lib.einsum('jlab,ajk->blk', t2_1_jla, r2[ka, kj], optimize=True)
                        temp -= lib.einsum('jlab,akj->blk', t2_1_jla, r2[ka, kk], optimize=True)

                        temp_1 = lib.einsum('jlab,ajk->blk', t2_1_jla, r2[ka, kj], optimize=True)
                        temp_1 -= lib.einsum('jlab,akj->blk', t2_1_jla, r2[ka, kk], optimize=True)
                        temp_1 += lib.einsum('jlab,ajk->blk', t2_1_jla, r2[ka, kj], optimize=True)
                        del t2_1_jla

                        t2_1_lja = adc.t2[0][kl, kj, ka]
                        temp -= lib.einsum('ljab,ajk->blk', t2_1_lja, r2[ka, kj], optimize=True)
                        temp += lib.einsum('ljab,akj->blk', t2_1_lja, r2[ka, kk], optimize=True)

                        temp_1 -= lib.einsum('ljab,ajk->blk', t2_1_lja, r2[ka, kj], optimize=True)
                        del t2_1_lja

                        ki = kconserv[kk, kl, kb]
                        s1 += 0.5 * lib.einsum('blk,lbik->i', temp, eris_ovoo[kl, kb, ki], optimize=True)
                        s1 -= 0.5 * lib.einsum('blk,iblk->i', temp, eris_ovoo[ki, kb, kl], optimize=True)
                        s1 += 0.5 * lib.einsum('blk,lbik->i', temp_1, eris_ovoo[kl, kb, ki], optimize=True)
                        s1 -= 0.5 * lib.einsum('blk,iblk->i', temp_2, eris_ovoo[ki, kb, kl], optimize=True)
                        del temp
                        del temp_1
                        del temp_2

                    for kb in range(nkpts):
                        kl = kconserv[ka, kk, kb]

                        t2_1 = adc.t2[0]

                        temp = -lib.einsum('lkba,akj->blj', t2_1[kl, kk, kb], r2[ka, kk], optimize=True)
                        temp_2 = -lib.einsum('klba,ajk->blj', t2_1[kk, kl, kb], r2[ka, kj], optimize=True)
                        del t2_1

                        t2_1_kla = adc.t2[0][kk, kl, ka]
                        temp -= lib.einsum('klab,akj->blj', t2_1_kla, r2[ka, kk], optimize=True)
                        temp += lib.einsum('klab,ajk->blj', t2_1_kla, r2[ka, kj], optimize=True)
                        temp_1 = -2.0 * lib.einsum('klab,akj->blj',
                                                   t2_1_kla, r2[ka, kk], optimize=True)
                        temp_1 += lib.einsum('klab,ajk->blj', t2_1_kla, r2[ka, kj], optimize=True)
                        del t2_1_kla

                        t2_1_lka = adc.t2[0][kl, kk, ka]
                        temp += lib.einsum('lkab,akj->blj', t2_1_lka, r2[ka, kk], optimize=True)
                        temp -= lib.einsum('lkab,ajk->blj', t2_1_lka, r2[ka, kj], optimize=True)
                        temp_1 += lib.einsum('lkab,akj->blj', t2_1_lka, r2[ka, kk], optimize=True)
                        del t2_1_lka

                        ki = kconserv[kj, kl, kb]
                        s1 -= 0.5 * lib.einsum('blj,lbij->i', temp, eris_ovoo[kl, kb, ki], optimize=True)
                        s1 += 0.5 * lib.einsum('blj,iblj->i', temp, eris_ovoo[ki, kb, kl], optimize=True)
                        s1 -= 0.5 * lib.einsum('blj,lbij->i', temp_1, eris_ovoo[kl, kb, ki], optimize=True)
                        s1 += 0.5 * lib.einsum('blj,iblj->i', temp_2, eris_ovoo[ki, kb, kl], optimize=True)

                        del temp
                        del temp_1
                        del temp_2

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    for kl in range(nkpts):
                        kb = kconserv[kj, ka, kl]
                        ki = kconserv[kk, kl, kb]
                        temp_1 = lib.einsum(
                            'i,lbik->kbl', r1, eris_ovoo[kl, kb, ki].conj(), optimize=True)
                        temp = lib.einsum(
                            'i,lbik->kbl', r1, eris_ovoo[kl, kb, ki].conj(), optimize=True)
                        temp -= lib.einsum('i,iblk->kbl', r1,
                                           eris_ovoo[ki, kb, kl].conj(), optimize=True)

                        t2_1 = adc.t2[0]
                        s2[ka, kj] += lib.einsum('kbl,ljba->ajk', temp,
                                                 t2_1[kl, kj, kb].conj(), optimize=True)
                        s2[ka, kj] += lib.einsum('kbl,jlab->ajk', temp_1,
                                                 t2_1[kj, kl, ka].conj(), optimize=True)
                        s2[ka, kj] -= lib.einsum('kbl,ljab->ajk', temp_1,
                                                 t2_1[kl, kj, ka].conj(), optimize=True)

                        kb = kconserv[kk, ka, kl]
                        ki = kconserv[kj, kl, kb]
                        temp_2 = -lib.einsum('i,iblj->jbl', r1,
                                             eris_ovoo[ki, kb, kl].conj(), optimize=True)
                        s2[ka, kj] += lib.einsum('jbl,klba->ajk', temp_2,
                                                 t2_1[kk, kl, kb].conj(), optimize=True)
                        del t2_1
        s2 = s2.reshape(-1)
        s = np.hstack((s1, s2))
        del s1
        del s2
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s
    return sigma_


def get_trans_moments(adc, kshift):

    nmo = adc.nmo
    T = []
    for orb in range(nmo):
        T_a = get_trans_moments_orbital(adc, orb, kshift)
        T.append(T_a)

    T = np.array(T)
    return T


def get_trans_moments_orbital(adc, orb, kshift):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    t2_1 = adc.t2[0]

    idn_occ = np.identity(nocc)

    T1 = np.zeros((nocc), dtype=np.complex128)
    T2 = np.zeros((nkpts, nkpts, nvir, nocc, nocc), dtype=np.complex128)

######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T1 += idn_occ[orb, :]
        for kk in range(nkpts):
            for kc in range(nkpts):
                kd = adc.khelper.kconserv[kk, kc, kshift]
                ki = adc.khelper.kconserv[kc, kk, kd]
                T1 += 0.25 * lib.einsum('kdc,ikdc->i', t2_1[kk, kshift, kd]
                                        [:, orb, :, :].conj(), t2_1[ki, kk, kd], optimize=True)
                T1 -= 0.25 * lib.einsum('kcd,ikdc->i', t2_1[kk, kshift, kc]
                                        [:, orb, :, :].conj(), t2_1[ki, kk, kd], optimize=True)
                T1 -= 0.25 * lib.einsum('kdc,ikcd->i', t2_1[kk, kshift, kd]
                                        [:, orb, :, :].conj(), t2_1[ki, kk, kc], optimize=True)
                T1 += 0.25 * lib.einsum('kcd,ikcd->i', t2_1[kk, kshift, kc]
                                        [:, orb, :, :].conj(), t2_1[ki, kk, kc], optimize=True)
                T1 -= 0.25 * lib.einsum('kdc,ikdc->i', t2_1[kshift, kk, kd]
                                        [orb, :, :, :].conj(), t2_1[ki, kk, kd], optimize=True)
                T1 -= 0.25 * lib.einsum('kcd,ikcd->i', t2_1[kshift, kk, kc]
                                        [orb, :, :, :].conj(), t2_1[ki, kk, kc], optimize=True)
    else:
        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            t1_2 = adc.t1[0]
            T1 += t1_2[kshift][:, (orb - nocc)]

######## ADC(2) 2h-1p  part  ############################################

        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = adc.khelper.kconserv[kj, kshift, ki]

                t2_1_t = -t2_1[ki, kj, ka].transpose(2, 3, 1, 0)
                T2[ka, kj] += t2_1_t[:, (orb - nocc), :, :].conj()

        del t2_1_t
####### ADC(3) 2h-1p  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1]

        if orb >= nocc:
            for ki in range(nkpts):
                for kj in range(nkpts):
                    ka = adc.khelper.kconserv[kj, kshift, ki]

                    t2_2_t = -t2_2[ki, kj, ka].transpose(2, 3, 1, 0)
                    T2[ka, kj] += t2_2_t[:, (orb - nocc), :, :].conj()


######### ADC(3) 1h part  ############################################

    if (method == 'adc(3)'):
        if orb < nocc:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = adc.khelper.kconserv[kk, kc, kshift]
                    ki = adc.khelper.kconserv[kd, kk, kc]
                    T1 += 0.25 * \
                        lib.einsum('kdc,ikdc->i', t2_1[kk, ki, kd][:, orb,
                                   :, :].conj(), t2_2[ki, kk, kd], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('kcd,ikdc->i', t2_1[kk, ki, kc][:, orb,
                                   :, :].conj(), t2_2[ki, kk, kd], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('kdc,ikcd->i', t2_1[kk, ki, kd][:, orb,
                                   :, :].conj(), t2_2[ki, kk, kc], optimize=True)
                    T1 += 0.25 * \
                        lib.einsum('kcd,ikcd->i', t2_1[kk, ki, kc][:, orb,
                                   :, :].conj(), t2_2[ki, kk, kc], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('kdc,ikdc->i', t2_1[ki, kk, kd][orb, :,
                                   :, :].conj(), t2_2[ki, kk, kd], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('kcd,ikcd->i', t2_1[ki, kk, kc][orb, :,
                                   :, :].conj(), t2_2[ki, kk, kc], optimize=True)

                    T1 += 0.25 * \
                        lib.einsum('ikdc,kdc->i', t2_1[ki, kk, kd],
                                   t2_2[kk, ki, kd][:, orb, :, :].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('ikcd,kdc->i', t2_1[ki, kk, kc],
                                   t2_2[kk, ki, kd][:, orb, :, :].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('ikdc,kcd->i', t2_1[ki, kk, kd],
                                   t2_2[kk, ki, kc][:, orb, :, :].conj(), optimize=True)
                    T1 += 0.25 * \
                        lib.einsum('ikcd,kcd->i', t2_1[ki, kk, kc],
                                   t2_2[kk, ki, kc][:, orb, :, :].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('ikcd,kcd->i', t2_1[ki, kk, kc],
                                   t2_2[ki, kk, kc][orb, :, :, :].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('ikdc,kdc->i', t2_1[ki, kk, kd],
                                   t2_2[ki, kk, kd][orb, :, :, :].conj(), optimize=True)
        else:

            for kk in range(nkpts):
                for kc in range(nkpts):
                    ki = adc.khelper.kconserv[kshift, kk, kc]

                    T1 += 0.5 * lib.einsum('kic,kc->i',
                                           t2_1[kk, ki, kc][:, :, :, (orb - nocc)], t1_2[kk], optimize=True)
                    T1 -= 0.5 * lib.einsum('ikc,kc->i', t2_1[ki, kk, kc]
                                           [:, :, :, (orb - nocc)], t1_2[kk], optimize=True)
                    T1 += 0.5 * lib.einsum('kic,kc->i', t2_1[kk, ki, kc]
                                           [:, :, :, (orb - nocc)], t1_2[kk], optimize=True)

        del t2_2
    del t2_1

    for ki in range(nkpts):
        for kj in range(nkpts):
            ka = adc.khelper.kconserv[kj, kshift, ki]
            T2[ka, kj] += T2[ka, kj] - T2[ka, ki].transpose(0, 2, 1)

    T2 = T2.reshape(-1)
    T = np.hstack((T1, T2))

    return T


def renormalize_eigenvectors(adc, kshift, U, nroots=1):

    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    n_singles = nocc
    nvir = adc.nmo - adc.nocc

    for I in range(U.shape[1]):
        U1 = U[:n_singles, I]
        U2 = U[n_singles:, I].reshape(nkpts, nkpts, nvir, nocc, nocc)
        UdotU = np.dot(U1.conj().ravel(), U1.ravel())
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = adc.khelper.kconserv[kj, kshift, kk]
                UdotU += 2. * np.dot(U2[ka, kj].conj().ravel(), U2[ka, kj].ravel()) - \
                    np.dot(U2[ka, kj].conj().ravel(),
                           U2[ka, kk].transpose(0, 2, 1).ravel())
        U[:, I] /= np.sqrt(UdotU)

    U = U.reshape(-1, nroots)

    return U


def make_rdm1(adc, root=None, kptlist=None, K_idx=None, if_ss=False):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    rdm1 = []

    for kpt, kshift in enumerate(K_idx):
        k = kptlist.index(kshift)
        U = np.array(adc.U[k]).T.copy()
        U = adc.renormalize_eigenvectors(kshift, U, adc.U.shape[1])
        rdm1_k = []
        for i in root[kpt]:
            rdm1_k.append(make_rdm1_eigenvectors(adc, U[:, i], U[:, i], kshift, if_ss))
        rdm1_k = np.stack(rdm1_k, axis=0)
        rdm1.append(rdm1_k)
    cput0 = log.timer_debug1("completed OPDM calculation", *cput0)
    return rdm1


def make_rdm1_eigenvectors_slow(adc, L, R, kshift):

    L = np.array(L).ravel().conj()
    R = np.array(R).ravel()

    t2_ce = adc.t1[0]
    t1_ccee = adc.t2[0]

    einsum = lib.einsum
    kconserv = adc.khelper.kconserv

    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    nkpts = adc.nkpts

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    kd_oc = np.identity(nocc)

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L2 = L2.reshape(nkpts, nkpts, nvir, nocc, nocc)
    R2 = R2.reshape(nkpts, nkpts, nvir, nocc, nocc)

# block- ij
    # 000
    for ki in range(nkpts):
        rdm1[ki][:nocc, :nocc] += 2 * einsum('ij,m,m->ij', kd_oc, L1, R1, optimize=True)
    rdm1[kshift][:nocc, :nocc] -= einsum('i,j->ij', L1, R1, optimize=True)
    # 101
    for ki in range(nkpts):
        for ka in range(nkpts):
            kj = kconserv[ka, ki, kshift]
            for kl in range(nkpts):
                rdm1[kl][:nocc, :nocc] += 4 * einsum('aij,aij,IJ->IJ', L2[ka][ki], R2[ka][ki], kd_oc, optimize=True)
                rdm1[kl][:nocc, :nocc] -= 2 * einsum('aij,aji,IJ->IJ', L2[ka][ki], R2[ka][kj], kd_oc, optimize=True)

            rdm1[kj][:nocc, :nocc] -= 2 * einsum('aJi,aIi->IJ', L2[ka][kj], R2[ka][kj], optimize=True)
            rdm1[kj][:nocc, :nocc] += einsum('aJi,aiI->IJ', L2[ka][kj], R2[ka][ki], optimize=True)
            rdm1[kj][:nocc, :nocc] += einsum('aiJ,aIi->IJ', L2[ka][ki], R2[ka][kj], optimize=True)
            rdm1[kj][:nocc, :nocc] -= 2 * einsum('aiJ,aiI->IJ', L2[ka][ki], R2[ka][ki], optimize=True)
    # 020
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            ki = kshift
            rdm1[ki][:nocc, :nocc] += einsum('J,i,Ijab,ijab->IJ', L1, R1, t1_ccee[ki]
                                             [kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
            rdm1[ki][:nocc, :nocc] -= 1 / 2 * einsum('J,i,Ijab,ijba->IJ', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)
            rdm1[ki][:nocc, :nocc] += einsum('i,I,ijab,Jjab->IJ', L1, R1, t1_ccee[ki]
                                             [kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
            rdm1[ki][:nocc, :nocc] -= 1 / 2 * einsum('i,I,ijab,Jjba->IJ', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)

            rdm1[kj][:nocc, :nocc] += 2 * einsum('i,j,Iiab,Jjab->IJ', L1, R1,
                                                 t1_ccee[kj][ki][ka], t1_ccee[kj][ki][ka].conj(), optimize=True)
            rdm1[kj][:nocc, :nocc] -= einsum('i,j,Iiab,Jjba->IJ', L1, R1, t1_ccee[kj]
                                             [ki][ka], t1_ccee[kj][ki][kb].conj(), optimize=True)
            for ki in range(nkpts):
                kb = kconserv[ki, ka, kj]
                rdm1[ki][:nocc, :nocc] -= 4 * einsum('i,i,Ijab,Jjab->IJ', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
                rdm1[ki][:nocc, :nocc] += 2 * einsum('i,i,Ijab,Jjba->IJ', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)
# block- ab
    # 101
    for ki in range(nkpts):
        for ka in range(nkpts):
            kj = kconserv[ka, ki, kshift]
            rdm1[ka][nocc:, nocc:] += 2 * einsum('Aij,Bij->AB', L2[ka][ki], R2[ka][ki], optimize=True)
            rdm1[ka][nocc:, nocc:] -= einsum('Aij,Bji->AB', L2[ka][ki], R2[ka][kj], optimize=True)

    # 020
    for kk in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kk]
            ki = kshift
            rdm1[ka][nocc:, nocc:] -= 2 * einsum('i,j,ikBa,jkAa->AB', L1, R1,
                                                 t1_ccee[ki][kk][ka], t1_ccee[ki][kk][ka].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] += einsum('i,j,ikBa,jkaA->AB', L1, R1, t1_ccee[ki]
                                             [kk][ka], t1_ccee[ki][kk][kb].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] += einsum('i,j,ikaB,jkAa->AB', L1, R1, t1_ccee[ki]
                                             [kk][kb], t1_ccee[ki][kk][ka].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] -= 2 * einsum('i,j,ikaB,jkaA->AB', L1, R1,
                                                 t1_ccee[ki][kk][kb], t1_ccee[ki][kk][kb].conj(), optimize=True)

            for kj in range(nkpts):
                kb = kconserv[kj, ka, kk]
                rdm1[ka][nocc:, nocc:] += 4 * einsum('i,i,jkBa,jkAa->AB', L1, R1,
                                                     t1_ccee[kj][kk][ka], t1_ccee[kj][kk][ka].conj(), optimize=True)
                rdm1[ka][nocc:, nocc:] -= 2 * einsum('i,i,jkBa,jkaA->AB', L1, R1,
                                                     t1_ccee[kj][kk][ka], t1_ccee[kj][kk][kb].conj(), optimize=True)

# block- ia
    for ki in range(nkpts):
        for kj in range(nkpts):
            kb = kshift
            ka = kconserv[ki, kb, kj]
            rdm1[kb][:nocc, nocc:] += einsum('aij,I,ijAa->IA', L2[ka][ki], R1, t1_ccee[ki][kj][kb], optimize=True)
            rdm1[kb][:nocc, nocc:] -= 2 * einsum('aij,I,ijaA->IA', L2[ka][ki], R1, t1_ccee[ki][kj][ka], optimize=True)
    for ka in range(nkpts):
        ki = kshift
        kj = kconserv[ka, ki, kshift]
        for kl in range(nkpts):
            km = kl
            rdm1[kl][:nocc, nocc:] -= 2 * einsum('aij,i,IjAa->IA', L2[ka][ki], R1, t1_ccee[kl][kj][km], optimize=True)
            rdm1[kl][:nocc, nocc:] += 1 * einsum('aij,i,IjaA->IA', L2[ka][ki], R1, t1_ccee[kl][kj][ka], optimize=True)
            rdm1[kl][:nocc, nocc:] += 4 * einsum('aij,j,IiAa->IA', L2[ka][kj], R1, t1_ccee[kl][kj][km], optimize=True)
            rdm1[kl][:nocc, nocc:] -= 2 * einsum('aij,j,IiaA->IA', L2[ka][kj], R1, t1_ccee[kl][kj][ka], optimize=True)
    for ka in range(nkpts):
        ki = kshift
        rdm1[ka][:nocc, nocc:] += 2 * einsum('i,AIi->IA', L1, R2[ka][ka], optimize=True)
        rdm1[ka][:nocc, nocc:] -= einsum('i,AiI->IA', L1, R2[ka][ki], optimize=True)

# block- ai
    for ki in range(nkpts):
        rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

    if adc.approx_trans_moments is False or adc.method == "adc(3)":
        ki = kshift
        ### block- ia ###
        rdm1[ki][:nocc, nocc:] -= einsum('i,I,iA->IA', L1, R1, t2_ce[ki], optimize=True)
        for ki in range(nkpts):
            rdm1[ki][:nocc, nocc:] += 2 * einsum('i,i,IA->IA', L1, R1, t2_ce[ki], optimize=True)
        ### block- ai ###
            rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

    return rdm1


def make_rdm1_eigenvectors(adc, L, R, kshift, if_ss):

    L = np.array(L).ravel().conj()
    R = np.array(R).ravel()

    t2_ce = adc.t1[0]
    t1_ccee = adc.t2[0]

    einsum = lib.einsum
    kconserv = adc.khelper.kconserv
    kconserv_s = kconserv[kshift, :, :]

    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    nkpts = adc.nkpts

    t1_ccee_sjb = np.zeros_like(t1_ccee[0])
    t1_ccee_isb = np.zeros_like(t1_ccee[0])
    t1_ccee_sja = t1_ccee[kshift]
    t1_ccee_isa = t1_ccee[:, kshift, :, :, :, :]
    t1_ccee_ijs = t1_ccee[:, :, kshift, :, :, :]

    kj, ka = np.indices((nkpts, nkpts))
    kb = adc.khelper.kconserv[kshift, ka, kj]
    t1_ccee_sjb[kj, ka] = t1_ccee_sja[kj, kb]
    t1_ccee_isb[kj, ka] = t1_ccee_isa[kj, kb]

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc
    occ_list = range(nocc)
    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L2 = L2.reshape(nkpts, nkpts, nvir, nocc, nocc)
    R2 = R2.reshape(nkpts, nkpts, nvir, nocc, nocc)
    R2_aji = np.zeros_like(R2)
    L2_aji = np.zeros_like(L2)
    L2_jia = np.zeros_like(L2)

    ka, ki = np.indices((nkpts, nkpts))
    kj = kconserv_s[ki, ka]
    R2_aji[ka, ki] = R2[ka, kj]
    L2_aji[ka, ki] = L2[ka, kj]
    L2_asj = L2[:, kshift, :, :, :]

    kconserv_s = kconserv[:, kshift, :]
    kj, ki = np.indices((nkpts, nkpts))
    ka = kconserv_s[ki, kj]
    L2_jia[kj, ki] = L2[ka, ki]

# block- ij
    # 000
    rdm1[:, occ_list, occ_list] += 2 * einsum('m,m->', L1, R1, optimize=True)
    rdm1[kshift, :nocc, :nocc] -= einsum('i,j->ij', L1, R1, optimize=True)
    # 101
    rdm1[:, occ_list, occ_list] += 4 * einsum('kKaij,kKaij->', L2, R2, optimize=True)
    rdm1[:, occ_list, occ_list] -= 2 * einsum('kKaij,kKaji->', L2, R2_aji, optimize=True)
    rdm1[:, :nocc, :nocc] -= 2 * einsum('kKaJi,kKaIi->KIJ', L2, R2, optimize=True)
    rdm1[:, :nocc, :nocc] += einsum('kKaJi,kKaiI->KIJ', L2, R2_aji, optimize=True)
    rdm1[:, :nocc, :nocc] += einsum('kKaiJ,kKaIi->KIJ', L2_aji, R2, optimize=True)
    rdm1[:, :nocc, :nocc] -= 2 * einsum('kKaiJ,kKaiI->KIJ', L2_aji, R2_aji, optimize=True)
    # 020
    rdm1[kshift, :nocc, :nocc] += einsum('J,i,kKIjab,kKijab->IJ', L1, R1,
                                         t1_ccee_sja, t1_ccee_sja.conj(), optimize=True)
    rdm1[kshift, :nocc, :nocc] -= 1 / 2 * einsum('J,i,kKIjab,kKijba->IJ', L1, R1,
                                                 t1_ccee_sja, t1_ccee_sjb.conj(), optimize=True)
    rdm1[kshift, :nocc, :nocc] += einsum('i,I,kKijab,kKJjab->IJ', L1, R1,
                                         t1_ccee_sja, t1_ccee_sja.conj(), optimize=True)
    rdm1[kshift, :nocc, :nocc] -= 1 / 2 * einsum('i,I,kKijab,kKJjba->IJ', L1, R1,
                                                 t1_ccee_sja, t1_ccee_sjb.conj(), optimize=True)
    rdm1[:, :nocc, :nocc] += 2 * einsum('i,j,KkIiab,KkJjab->KIJ', L1, R1,
                                        t1_ccee_isa, t1_ccee_isa.conj(), optimize=True)
    rdm1[:, :nocc, :nocc] -= einsum('i,j,KkIiab,KkJjba->KIJ', L1, R1,
                                    t1_ccee_isa, t1_ccee_isb.conj(), optimize=True)

# block- ab
    # 101
    rdm1[:, nocc:, nocc:] += 2 * einsum('KkAij,KkBij->KAB', L2, R2, optimize=True)
    rdm1[:, nocc:, nocc:] -= einsum('KkAij,KkBji->KAB', L2, R2_aji, optimize=True)
    # 020
    rdm1[:, nocc:, nocc:] -= 2 * einsum('i,m,kKijBa,kKmjAa->KAB', L1, R1,
                                        t1_ccee_sja, t1_ccee_sja.conj(), optimize=True)
    rdm1[:, nocc:, nocc:] += einsum('i,m,kKijBa,kKmjaA->KAB', L1, R1,
                                    t1_ccee_sja, t1_ccee_sjb.conj(), optimize=True)
    rdm1[:, nocc:, nocc:] += einsum('i,m,kKijaB,kKmjAa->KAB', L1, R1,
                                    t1_ccee_sjb, t1_ccee_sja.conj(), optimize=True)
    rdm1[:, nocc:, nocc:] -= 2 * einsum('i,m,kKijaB,kKmjaA->KAB', L1, R1,
                                        t1_ccee_sjb, t1_ccee_sjb.conj(), optimize=True)

# block- ia
    rdm1[kshift, :nocc, nocc:] += einsum('kKaij,I,KkijAa->IA', L2_jia, R1, t1_ccee_ijs, optimize=True)
    rdm1[:, :nocc, nocc:] += 2 * einsum('i,KKAIi->KIA', L1, R2, optimize=True)
    rdm1[:, :nocc, nocc:] -= einsum('i,KAiI->KIA', L1, R2[:, kshift, :, :, :], optimize=True)

    del (t1_ccee_sjb)
    del (t1_ccee_sja)
    del (t1_ccee_isa)
    del (t1_ccee_ijs)
    del (t1_ccee_isb)
    del (R2_aji)
    del (L2_aji)

    kj, ka = np.indices((nkpts, nkpts))
    idx0 = np.arange(nkpts)[:, None]
    LR = np.dot(L1, R1)
    if if_ss:
        rdm1[:, :nocc, :nocc] += np.identity(nocc)
        LR += 1.
    t1_ccee_ijs_b = np.zeros_like(t1_ccee[0])
    oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
    vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
    ov = np.zeros((nkpts, nocc, nvir), dtype=np.complex128)
    for ki in range(nkpts):
        kb = adc.khelper.kconserv[ki, ka, kj]
        t1_ccee_np = np.array(t1_ccee[ki])
        t1_ccee_ijb = t1_ccee_np[idx0, kb]
        t1_ccee_ijs_b[ki] = t1_ccee_ijb[:, kshift].copy()
# block- ij
        oo[ki] -= 4 * einsum('KkIjab,KkJjab->IJ', t1_ccee_np, t1_ccee_np.conj(), optimize=True)
        oo[ki] += 2 * einsum('KkIjab,KkJjba->IJ', t1_ccee_np, t1_ccee_ijb.conj(), optimize=True)
# block- ab
        vv += 4 * einsum('kKijBa,kKijAa->KAB', t1_ccee_np, t1_ccee_np.conj(), optimize=True)
        vv -= 2 * einsum('kKijBa,kKijaA->KAB', t1_ccee_np, t1_ccee_ijb.conj(), optimize=True)
# block- ia
        ov[ki] -= 2 * einsum('kaij,i,kIjAa->IA', L2_asj, R1, t1_ccee_np[:, ki], optimize=True)
        ov[ki] += 1 * einsum('kaij,i,kkIjaA->IA', L2_asj, R1, t1_ccee_np, optimize=True)
        ov[ki] += 4 * einsum('kkaij,j,kIiAa->IA', L2, R1, t1_ccee_np[:, ki], optimize=True)
        ov[ki] -= 2 * einsum('kkaij,j,kkIiaA->IA', L2, R1, t1_ccee_np, optimize=True)
        del (t1_ccee_ijb)
        del (t1_ccee_np)

    oo *= LR
    vv *= LR
    rdm1[:, :nocc, :nocc] += oo
    rdm1[:, nocc:, nocc:] += vv
    rdm1[:, :nocc, nocc:] += ov
    del (oo)
    del (vv)
    del (ov)
    del (L2_asj)
# block- ia
    rdm1[kshift, :nocc, nocc:] -= 2 * einsum('kKaij,I,KkijaA->IA', L2_jia, R1, t1_ccee_ijs_b, optimize=True)
    del (L2_jia)
    del (t1_ccee_ijs_b)

# block- ai
    rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

    if adc.approx_trans_moments is False or adc.method == "adc(3)":
        rdm1[kshift, :nocc, nocc:] -= einsum('i,I,iA->IA', L1, R1, t2_ce[kshift], optimize=True)
        rdm1[:, :nocc, nocc:] += 2 * LR * t2_ce
        rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

    return rdm1


def get_properties(adc, kshift, U, nroots=1):

    # Transition moments
    T = adc.get_trans_moments(kshift)

    # Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(kshift, U, nroots)
    X = np.dot(T, U).reshape(-1, nroots)

    # Spectroscopic factors
    P = 2.0 * lib.einsum("pi,pi->i", X.conj(), X)
    P = P.real

    return P, X


def mask_frozen_ip(adc, v1, v2, kshift, const=-LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(adc)

    new_v1 = const * np.ones_like(v1)
    new_v2 = const * np.ones_like(v2)

    new_v1[nonzero_opadding[kshift]] = v1[nonzero_opadding[kshift]]
    for ka in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[kshift, ki, ka]
            idx = np.ix_([ka], [ki], nonzero_vpadding[ka], nonzero_opadding[ki], nonzero_opadding[kj])
            new_v2[idx] = v2[idx]

    return (new_v1, new_v2)


class RADCIP(kadc_rhf.RADC):
    '''restricted ADC for IP energies and spectroscopic amplitudes

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).
        conv_tol : float
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcip = adc.RADC(myadc).run()

    Saved results

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list
            of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''

    _keys = {
        'tol_residual', 'conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b',
        't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'kpts', 'exxdiv', 'khelper', 'cell', 'nkop_chk', 'kop_npick', 'chnk_size',
    }

    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol = adc.conv_tol
        self.tol_residual = adc.tol_residual

        self.t1 = adc.t1
        self.t2 = adc.t2
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments

        self.kpts = adc._scf.kpts
        self.exxdiv = adc.exxdiv
        self.verbose = adc.verbose
        self.max_memory = adc.max_memory
        self.method = adc.method

        self.khelper = adc.khelper
        self.cell = adc.cell
        self.mo_coeff = adc.mo_coeff
        self.mo_occ = adc.mo_occ
        self.frozen = adc.frozen

        self._nocc = adc._nocc
        self._nmo = adc._nmo
        self._nvir = adc._nvir
        self.nkop_chk = adc.nkop_chk
        self.kop_npick = adc.kop_npick

        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.mo_energy = adc.mo_energy
        self.imds = adc.imds
        self.chnk_size = adc.chnk_size
        self.U = adc.U
        self.if_naf = adc.if_naf
        self.naux = adc.naux

    kernel = kadc_rhf.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    vector_size = vector_size
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
    get_properties = get_properties
    make_rdm1 = make_rdm1

    def get_init_guess(self, nroots=1, diag=None, ascending=True, type=None, ini=None, kshift=None):
        ncore = self.nocc
        nextern = self.nmo - ncore
        nkpts = self.nkpts
        n_singles = ncore
        n_doubles = nkpts * nkpts * nextern * ncore * ncore
        dim = n_singles + n_doubles

        if (type == "read"):
            logger.info(self, "obtain initial guess from input variable")
            g = ini.T
            if g.shape[0] != dim or g.shape[1] != nroots:
                raise ValueError(f"Shape of guess each k point should be ({dim},{nroots})")
        else:
            if diag is None:
                diag = self.get_diag()
            idx = None
            dtype = getattr(diag, 'dtype', np.complex128)
            g = np.zeros((diag.shape[0], nroots), dtype=dtype)
            if ascending:
                idx = np.argsort(diag)
            else:
                idx = np.argsort(diag)[::-1]
            guess = np.zeros((diag.shape[0], nroots), dtype=dtype)
            min_shape = min(diag.shape[0], nroots)
            guess[:min_shape, :min_shape] = np.identity(min_shape)
            g[idx] = guess.copy()
        if (self.frozen is not None or not np.all([x.shape[1] == self.nmo for x in self.mo_coeff])) \
                and (type == "read"):
            for p in range(g.shape[1]):
                singles = g[:n_singles, p]
                doubles = g[n_singles:, p].reshape(nkpts, nkpts, nextern, ncore, ncore)
                (singles, doubles) = mask_frozen_ip(self, singles, doubles, kshift, const=0.0)
                g[:, p] = np.hstack((singles, doubles.ravel()))

        guess = []
        for p in range(g.shape[1]):
            if (self.frozen is not None) or (not np.all([x.shape[1] == self.nmo for x in self.mo_coeff])) \
                    or (type == "read"):
                guess_norm = np.linalg.norm(g[:, p])
                guess_norm_tol = LOOSE_ZERO_TOL
                if guess_norm < guess_norm_tol:
                    raise ValueError('Guess vector for root %d with norm %.4g is below threshold %.4g.\n'
                                     'This could possibly be due to freezing orbitals.\n'
                                     'Check your guess vector to make sure it has sufficiently large norm.'
                                     % (p, guess_norm, guess_norm_tol))

            guess.append(g[:, p])
        return guess

    def gen_matvec(self, kshift, imds=None, eris=None):
        if imds is None:
            imds = self.get_imds(eris)
        diag = self.get_diag(kshift, imds, eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag
        # return diag
