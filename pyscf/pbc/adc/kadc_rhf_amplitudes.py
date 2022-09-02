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
import pyscf.ao2mo as ao2mo
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
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx,_padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa

from pyscf.pbc import tools
import h5py
import tempfile


# Note : All interals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba

def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1,t2,myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    cput0 = (time.process_time(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nmo = myadc.nmo
    nocc = myadc.nocc
    nvir = nmo - nocc
    nkpts = myadc.nkpts
    cell = myadc.cell
    kpts = myadc.kpts
    madelung = tools.madelung(cell, kpts)

    # Compute first-order doubles t2 (tijab)
    tf = tempfile.TemporaryFile()
    f = h5py.File(tf, 'a')
    t2_1 = f.create_dataset('t2_1', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)

    mo_energy =  myadc.mo_energy
    mo_coeff =  myadc.mo_coeff
    mo_coeff, mo_energy = _add_padding(myadc, mo_coeff, mo_energy)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(myadc, kind="split")

    kconserv = myadc.khelper.kconserv
    touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        if touched[ki, kj, ka]:
            continue

        kb = kconserv[ki, ka, kj]
        # For discussion of LARGE_DENOM, see t1new update above
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])

        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        t2_1[ki,kj,ka] = eris.ovov[ki,ka,kj].conj().transpose((0,2,1,3)) / eijab

        if ka != kb:
            eijba = eijab.transpose(0, 1, 3, 2)
            t2_1[ki, kj, kb] = eris.ovov[ki,kb,kj].conj().transpose((0,2,1,3)) / eijba

        touched[ki, kj, ka] = touched[ki, kj, kb] = True

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    t1_2 = None
    if myadc.approx_trans_moments is False or myadc.method == "adc(3)":
        # Compute second-order singles t1 (tij)
        t1_2 = np.zeros((nkpts,nocc,nvir), dtype=t2_1.dtype)
        eris_ovoo = eris.ovoo
        for ki in range(nkpts):
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = kconserv[ki, kc, kk]
                    ka = kconserv[kc, kk, kd]

                    if isinstance(eris.ovvv, type(None)):
                        chnk_size = myadc.chnk_size
                        if chnk_size > nocc:
                            chnk_size = nocc
                        a = 0
                        for p in range(0,nocc,chnk_size):
                            eris_ovvv = dfadc.get_ovvv_df(
                                myadc, eris.Lov[kk,kd], eris.Lvv[ka,kc], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                            k = eris_ovvv.shape[0]
                            t1_2[ki] += 1.5*lib.einsum('kdac,ikcd->ia',
                                                       eris_ovvv,t2_1[ki,kk,kc,:,a:a+k],optimize=True)
                            t1_2[ki] -= 0.5*lib.einsum('kdac,kicd->ia',
                                                       eris_ovvv,t2_1[kk,ki,kc,a:a+k,:],optimize=True)
                            del eris_ovvv
                            eris_ovvv = dfadc.get_ovvv_df(
                                myadc, eris.Lov[kk,kc], eris.Lvv[ka,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                            t1_2[ki] -= 0.5*lib.einsum('kcad,ikcd->ia',
                                                       eris_ovvv,t2_1[ki,kk,kc,:,a:a+k],optimize=True)
                            t1_2[ki] += 0.5*lib.einsum('kcad,kicd->ia',
                                                       eris_ovvv,t2_1[kk,ki,kc,a:a+k,:],optimize=True)
                            del eris_ovvv
                            a += k
                    else:
                        eris_ovvv = eris.ovvv[:]
                        t1_2[ki] += 1.5*lib.einsum('kdac,ikcd->ia',
                                                   eris_ovvv[kk,kd,ka],t2_1[ki,kk,kc],optimize=True)
                        t1_2[ki] -= 0.5*lib.einsum('kdac,kicd->ia',
                                                   eris_ovvv[kk,kd,ka],t2_1[kk,ki,kc],optimize=True)
                        t1_2[ki] -= 0.5*lib.einsum('kcad,ikcd->ia',
                                                   eris_ovvv[kk,kc,ka],t2_1[ki,kk,kc],optimize=True)
                        t1_2[ki] += 0.5*lib.einsum('kcad,kicd->ia',
                                                   eris_ovvv[kk,kc,ka],t2_1[kk,ki,kc],optimize=True)
                        del eris_ovvv

                for kl in range(nkpts):
                    kc = kconserv[kk, ki, kl]
                    ka = kconserv[kl, kc, kk]

                    t1_2[ki] -= 1.5*lib.einsum('lcki,klac->ia',
                                               eris_ovoo[kl,kc,kk],t2_1[kk,kl,ka],optimize=True)
                    t1_2[ki] += 0.5*lib.einsum('lcki,lkac->ia',
                                               eris_ovoo[kl,kc,kk],t2_1[kl,kk,ka],optimize=True)
                    t1_2[ki] -= 0.5*lib.einsum('kcli,lkac->ia',
                                               eris_ovoo[kk,kc,kl],t2_1[kl,kk,ka],optimize=True)
                    t1_2[ki] += 0.5*lib.einsum('kcli,klac->ia',
                                               eris_ovoo[kk,kc,kl],t2_1[kk,kl,ka],optimize=True)

        for ki in range(nkpts):
            ka = ki
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            t1_2[ki] = t1_2[ki] / eia

        cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    t2_2 = None
    t1_3 = None
    t2_1_vvvv = None

    if (myadc.method == "adc(2)-x" and myadc.approx_trans_moments is False) or (myadc.method == "adc(3)"):
        # Compute second-order doubles t2 (tijab)
        t2_1_vvvv = f.create_dataset(
            't2_1_vvvv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
        eris_oooo = eris.oooo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv

        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka, kc, kb]
            for ki in range(nkpts):
                kj = kconserv[ka, ki, kb]
                if isinstance(eris.vvvv, np.ndarray):
                    eris_vvvv = eris.vvvv.reshape(nkpts,nkpts,nkpts,nvir*nvir,nvir*nvir)
                    t2_1_a = t2_1[:].reshape(nkpts,nkpts,nkpts,nocc*nocc,nvir*nvir)
                    t2_1_vvvv[ki, kj, ka] += np.dot(t2_1_a[ki,kj,kc],
                                                    eris_vvvv[kc,kd,ka].conj()).reshape(nocc,nocc,nvir,nvir)
                elif isinstance(eris.vvvv, type(None)):
                    t2_1_vvvv[ki,kj,ka] += contract_ladder(myadc,t2_1[ki,kj,kc],eris.Lvv,ka,kb,kc)
                else :
                    t2_1_vvvv[ki,kj,ka] += contract_ladder(myadc,t2_1[ki,kj,kc],eris.vvvv,kc,kd,ka)

        t2_2 = f.create_dataset('t2_2', (nkpts,nkpts,nkpts,nocc,nocc,
                                nvir,nvir), dtype=eris.ovov.dtype)
        t2_2 = t2_1_vvvv[:]

        if myadc.exxdiv is not None:
            t2_2 -= 2.0 * madelung * t2_1

        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            for kk in range(nkpts):

                kc = kconserv[ki,ka,kk]
                kb = kconserv[kj,kk,kc]
                t2_2[ki,kj,ka] -= lib.einsum('kjbc,kica->ijab',
                                             eris_oovv[kk,kj,kb],t2_1[kk,ki,kc],optimize=True)

                kc = kconserv[kk,ka,kj]
                kb = kconserv[ki,kk,kc]
                t2_2[ki,kj,ka] -= lib.einsum('kibc,jkca->ijab',
                                             eris_oovv[kk,ki,kb],t2_1[kj,kk,kc],optimize=True)

                kc = kconserv[ka,kj,kk]
                kb = kconserv[ki,kc,kk]
                t2_2[ki,kj,ka] -= lib.einsum('kjac,ikcb->ijab',
                                             eris_oovv[kk,kj,ka],t2_1[ki,kk,kc],optimize=True)

                kc = kconserv[ka,ki,kk]
                kb = kconserv[kk,kc,kj]
                t2_2[ki,kj,ka] -= lib.einsum('kiac,kjcb->ijab',
                                             eris_oovv[kk,ki,ka],t2_1[kk,kj,kc],optimize=True)

            for kl in range(nkpts):
                kk = kconserv[kj,kl,ki]
                t2_2[ki,kj,ka] += lib.einsum('kilj,klab->ijab',
                                             eris_oooo[kk,ki,kl],t2_1[kk,kl,ka],optimize=True)

            for kk in range(nkpts):

                kc = kconserv[ki,ka,kk]
                kb = kconserv[kc,kk,kj]

                t2_2[ki,kj,ka] += 2 * lib.einsum('kcbj,kica->ijab',
                                                 eris_ovvo[kk,kc,kb],t2_1[kk,ki,kc],optimize=True)
                kc = kconserv[kk,ka,ki]
                t2_2[ki,kj,ka] -= lib.einsum('kcbj,ikca->ijab',
                                             eris_ovvo[kk,kc,kb],t2_1[ki,kk,kc],optimize=True)

                kc = kconserv[kk,ki,ka]
                kb = kconserv[kk,kj,kc]
                t2_2[ki,kj,ka] += 2 * lib.einsum('kcai,kjcb->ijab',
                                                 eris_ovvo[kk,kc,ka],t2_1[kk,kj,kc],optimize=True)

                kc = kconserv[kk,ki,ka]
                kb = kconserv[kj,kk,kc]
                t2_2[ki,kj,ka] -= lib.einsum('kcai,jkcb->ijab',
                                             eris_ovvo[kk,kc,ka],t2_1[kj,kk,kc],optimize=True)

            kb = kconserv[ki, ka, kj]
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])

            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                           [0,nvir,kb,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            t2_2[ki,kj,ka] /= eijab

        cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

        if (myadc.method == "adc(3)" and myadc.approx_trans_moments is False):
            raise NotImplementedError('3rd order singles amplitues not implemented')

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2, t2_1_vvvv


def compute_energy(myadc, t2, eris):

    nkpts = myadc.nkpts

    emp2 = 0.0
    eris_ovov = eris.ovov
    t2_amp = t2[0][:]

    if (myadc.method == "adc(3)"):
        t2_amp += t2[1][:]

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):

        emp2 += 2 * lib.einsum('ijab,iajb', t2_amp[ki,kj,ka], eris_ovov[ki,ka,kj],optimize=True)
        emp2 -= 1 * lib.einsum('ijab,jaib', t2_amp[ki,kj,ka], eris_ovov[kj,ka,ki],optimize=True)

    del t2_amp
    emp2 = emp2.real / nkpts
    return emp2


def contract_ladder(myadc,t_amp,vvvv,ka,kb,kc):

    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc
    nkpts = myadc.nkpts
    kconserv = myadc.khelper.kconserv

    kd = kconserv[ka, kc, kb]
    t_amp = np.ascontiguousarray(t_amp.reshape(nocc*nocc,nvir*nvir))
    t = np.zeros((nocc,nocc, nvir, nvir),dtype=t_amp.dtype)
    chnk_size = myadc.chnk_size
    if chnk_size > nvir:
        chnk_size = nvir
    a = 0
    if isinstance(vvvv, np.ndarray):
        vv1 = vvvv[kc,ka]
        vv2 = vvvv[kd,kb]
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vv1, vv2, p, chnk_size)/nkpts
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            k = vvvv_p.shape[0]
            t += np.dot(t_amp[:,a:a+k],vvvv_p.conj()).reshape(nocc,nocc,nvir,nvir)
            del vvvv_p
            a += k
    else :
        for p in range(0,nvir,chnk_size):
            vvvv_p = vvvv[ka,kb,kc,p:p+chnk_size,:,:,:].reshape(-1,nvir*nvir)
            k = vvvv_p.shape[0]
            t += np.dot(t_amp[:,a:a+k],vvvv_p.conj()).reshape(nocc,nocc,nvir,nvir)
            del vvvv_p
            a += k

    return t
