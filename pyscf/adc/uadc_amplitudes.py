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
# Author: Abdelrahman Ahmed <>
#         Samragni Banerjee <samragnibanerjee4@gmail.com>
#         James Serna <jamcar456@gmail.com>
#         Terrence Stahl <>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Unrestricted algebraic diagrammatic construction
'''

import numpy as np
import pyscf.lib as lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf.adc.radc_amplitudes import _create_t2_h5cache
from pyscf import scf

def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1, t2, myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_oovv = eris.oovv
    eris_OOVV = eris.OOVV
    eris_ooVV = eris.ooVV
    eris_OOvv = eris.OOvv
    eris_ovoo = eris.ovoo
    eris_OVoo = eris.OVoo
    eris_ovOO = eris.ovOO
    eris_OVOO = eris.OVOO

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO

    e_a = myadc.mo_energy_a
    e_b = myadc.mo_energy_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris_ovvo[:].transpose(0,3,1,2).copy()
    v2e_oovv -= eris_ovvo[:].transpose(0,3,2,1).copy()

    d_ij_a = e_a[:nocc_a][:,None] + e_a[:nocc_a]
    d_ab_a = e_a[nocc_a:][:,None] + e_a[nocc_a:]

    D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
    D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))

    t2_1_a = v2e_oovv/D2_a
    h5cache_t2 = _create_t2_h5cache()
    if not isinstance(eris.oooo, np.ndarray):
        t2_1_a = h5cache_t2.create_dataset('t2_1_a', data=t2_1_a)

    del v2e_oovv
    del D2_a

    v2e_OOVV = eris_OVVO[:].transpose(0,3,1,2).copy()
    v2e_OOVV -= eris_OVVO[:].transpose(0,3,2,1).copy()

    d_ij_b = e_b[:nocc_b][:,None] + e_b[:nocc_b]
    d_ab_b = e_b[nocc_b:][:,None] + e_b[nocc_b:]

    D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
    D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))

    t2_1_b = v2e_OOVV/D2_b
    if not isinstance(eris.oooo, np.ndarray):
        t2_1_b = h5cache_t2.create_dataset('t2_1_b', data=t2_1_b)
    del v2e_OOVV
    del D2_b

    v2e_oOvV = eris_ovVO[:].transpose(0,3,1,2).copy()

    d_ij_ab = e_a[:nocc_a][:,None] + e_b[:nocc_b]
    d_ab_ab = e_a[nocc_a:][:,None] + e_b[nocc_b:]

    D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
    D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))

    t2_1_ab = v2e_oOvV/D2_ab
    if not isinstance(eris.oooo, np.ndarray):
        t2_1_ab = h5cache_t2.create_dataset('t2_1_ab', data=t2_1_ab)
    del v2e_oOvV
    del D2_ab

    D1_a = e_a[:nocc_a][:None].reshape(-1,1) - e_a[nocc_a:].reshape(-1)
    D1_b = e_b[:nocc_b][:None].reshape(-1,1) - e_b[nocc_b:].reshape(-1)
    D1_a = D1_a.reshape((nocc_a,nvir_a))
    D1_b = D1_b.reshape((nocc_b,nvir_b))

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    t1_1 = (None,)

    if isinstance(myadc._scf, scf.rohf.ROHF):

        f_ov_a, f_ov_b = myadc.f_ov

        t1_1_a = f_ov_a/D1_a
        t1_1_b = f_ov_b/D1_b
        t1_1 = (t1_1_a, t1_1_b)
    else:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_1_a = np.zeros((nocc_a, nvir_a))
        t1_1_b = np.zeros((nocc_b, nvir_b))


    t1_2 = (None,)
    if myadc.approx_trans_moments is False or myadc.method == "adc(3)":
        # Compute second-order singles t1 (tij)

        t1_2_a = np.zeros((nocc_a,nvir_a))
        t1_2_b = np.zeros((nocc_b,nvir_b))

        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(
                    myadc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                t1_2_a += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1_a[:,a:b],optimize=True)
                t1_2_a -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1_a[:,a:b],optimize=True)
                del eris_ovvv
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            t1_2_a += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1_a[:],optimize=True)
            t1_2_a -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1_a[:],optimize=True)
            del eris_ovvv

        t1_2_a -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1_a[:],optimize=True)
        t1_2_a += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1_a[:],optimize=True)

        if isinstance(myadc._scf, scf.rohf.ROHF):
            t1_2_a += lib.einsum('d,ld,ilad->ia',e_a[nocc_a:],t1_1_a,t2_1_a[:], optimize = True)
            t1_2_a += lib.einsum('d,ld,ilad->ia',e_b[nocc_b:],t1_1_b,t2_1_ab[:], optimize = True)

            t1_2_a -= lib.einsum('l,ld,ilad->ia',e_a[:nocc_a],t1_1_a,t2_1_a[:], optimize = True)
            t1_2_a -= lib.einsum('l,ld,ilad->ia',e_b[:nocc_b],t1_1_b,t2_1_ab[:], optimize = True)


            t1_2_a += 0.5*lib.einsum('a,ld,ilad->ia',e_a[nocc_a:],t1_1_a,t2_1_a[:], optimize = True)
            t1_2_a += 0.5*lib.einsum('a,ld,ilad->ia',
                                     e_a[nocc_a:],t1_1_b,t2_1_ab[:], optimize = True)

            t1_2_a -= 0.5*lib.einsum('i,ld,ilad->ia',e_a[:nocc_a],t1_1_a,t2_1_a[:], optimize = True)
            t1_2_a -= 0.5*lib.einsum('i,ld,ilad->ia',
                                     e_a[:nocc_a],t1_1_b,t2_1_ab[:], optimize = True)

            t1_2_a += lib.einsum('ld,ilad->ia',f_ov_a,t2_1_a[:], optimize = True)
            t1_2_a += lib.einsum('ld,ilad->ia',f_ov_b,t2_1_ab[:], optimize = True)

            t1_2_a += lib.einsum('ld,iadl->ia',t1_1_a, eris.ovvo, optimize = True)
            t1_2_a -= lib.einsum('ld,idal->ia',t1_1_a, eris.ovvo, optimize = True)
            t1_2_a += lib.einsum('ld,iadl->ia',t1_1_b, eris.ovVO, optimize = True)

            t1_2_a += lib.einsum('ld,iadl->ia',t1_1_a,eris.ovvo, optimize = True)
            t1_2_a -= lib.einsum('ld,liad->ia',t1_1_a,eris.oovv, optimize = True)
            t1_2_a += lib.einsum('ld,iadl->ia',t1_1_b,eris.ovVO, optimize = True)

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(
                    myadc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                t1_2_a += lib.einsum('kdac,ikcd->ia',eris_OVvv,t2_1_ab[:,a:b],optimize=True)
                del eris_OVvv
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            t1_2_a += lib.einsum('kdac,ikcd->ia',eris_OVvv,t2_1_ab[:],optimize=True)
            del eris_OVvv

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(
                    myadc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                t1_2_b += lib.einsum('kdac,kidc->ia',eris_ovVV,t2_1_ab[a:b],optimize=True)
                del eris_ovVV

        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            t1_2_b += lib.einsum('kdac,kidc->ia',eris_ovVV,t2_1_ab[:],optimize=True)
            del eris_ovVV

        t1_2_a -= lib.einsum('lcki,klac->ia',eris_OVoo,t2_1_ab[:],optimize=True)
        t1_2_b -= lib.einsum('lcki,lkca->ia',eris_ovOO,t2_1_ab[:],optimize=True)

        if isinstance(myadc._scf, scf.rohf.ROHF):
            t1_2_b += lib.einsum('d,ld,ilad->ia',e_b[nocc_b:],t1_1_b,t2_1_b[:], optimize = True)
            t1_2_b += lib.einsum('d,ld,lida->ia',e_a[nocc_a:],t1_1_a,t2_1_ab[:], optimize = True)

            t1_2_b -= lib.einsum('l,ld,ilad->ia',e_b[:nocc_b],t1_1_b,t2_1_b[:], optimize = True)
            t1_2_b -= lib.einsum('l,ld,lida->ia',e_a[:nocc_a],t1_1_a,t2_1_ab[:], optimize = True)

            t1_2_b += 0.5*lib.einsum('a,ld,ilad->ia',e_b[nocc_b:],t1_1_b,t2_1_b[:], optimize = True)
            t1_2_b += 0.5*lib.einsum('a,ld,lida->ia',
                                     e_b[nocc_b:],t1_1_a,t2_1_ab[:], optimize = True)

            t1_2_b -= 0.5*lib.einsum('i,ld,ilad->ia',e_b[:nocc_b],t1_1_b,t2_1_b[:], optimize = True)
            t1_2_b -= 0.5*lib.einsum('i,ld,lida->ia',
                                     e_b[:nocc_b],t1_1_a,t2_1_ab[:], optimize = True)

            t1_2_b += lib.einsum('ld,ilad->ia',f_ov_b,t2_1_b[:], optimize = True)
            t1_2_b += lib.einsum('ld,lida->ia',f_ov_a,t2_1_ab[:], optimize = True)

            t1_2_b += lib.einsum('ld,iadl->ia',t1_1_b, eris.OVVO, optimize = True)
            t1_2_b -= lib.einsum('ld,idal->ia',t1_1_b, eris.OVVO, optimize = True)
            t1_2_b += lib.einsum('ld,ldai->ia',t1_1_a, eris.ovVO, optimize = True)

            t1_2_b += lib.einsum('ld,iadl->ia',t1_1_b,eris.OVVO, optimize = True)
            t1_2_b -= lib.einsum('ld,liad->ia',t1_1_b,eris.OOVV, optimize = True)
            t1_2_b += lib.einsum('ld,ldai->ia',t1_1_a,eris.ovVO, optimize = True)

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(
                    myadc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                t1_2_b += 0.5*lib.einsum('kdac,ikcd->ia',eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_2_b -= 0.5*lib.einsum('kcad,ikcd->ia',eris_OVVV,t2_1_b[:,a:b],optimize=True)
                del eris_OVVV

        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            t1_2_b += 0.5*lib.einsum('kdac,ikcd->ia',eris_OVVV,t2_1_b[:],optimize=True)
            t1_2_b -= 0.5*lib.einsum('kcad,ikcd->ia',eris_OVVV,t2_1_b[:],optimize=True)
            del eris_OVVV

        t1_2_b -= 0.5*lib.einsum('lcki,klac->ia',eris_OVOO,t2_1_b[:],optimize=True)
        t1_2_b += 0.5*lib.einsum('kcli,klac->ia',eris_OVOO,t2_1_b[:],optimize=True)

        t1_2_a = t1_2_a/D1_a
        t1_2_b = t1_2_b/D1_b

        cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

        t1_2 = (t1_2_a , t1_2_b)

    t2_2 = (None,)
    t1_3 = (None,)
    t2_1_vvvv = (None,)

    if ((myadc.method == "adc(2)" and myadc.method_type == "ee" and myadc.approx_trans_moments is False)
        or (myadc.method == "adc(2)-x" and myadc.approx_trans_moments is False)
        or (myadc.method == "adc(3)")):

        # Compute second-order doubles t2 (tijab)
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO

        if isinstance(eris.vvvv_p, np.ndarray):
            eris_vvvv = eris.vvvv_p
            temp = np.ascontiguousarray(
                t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]).reshape(nocc_a*nocc_a,-1)
            temp = np.dot(temp,eris_vvvv.T).reshape(nocc_a, nocc_a, -1)
            t2_1_vvvv_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
            t2_1_vvvv_a[:,:,ab_ind_a[0],ab_ind_a[1]] = temp
            t2_1_vvvv_a[:,:,ab_ind_a[1],ab_ind_a[0]] = -temp
            del eris_vvvv
        elif isinstance(eris.vvvv_p, list):
            t2_1_vvvv_a = contract_ladder_antisym(myadc,t2_1_a[:], eris.vvvv_p, pack = False)
        else:
            t2_1_vvvv_a = contract_ladder(myadc, t2_1_a[:], (eris.Lvv, eris.Lvv))

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv_a = h5cache_t2.create_dataset('t2_1_vvvv_a', data=t2_1_vvvv_a)

        t2_2_a = t2_1_vvvv_a[:].copy()

        t2_2_a += 0.5*lib.einsum('kilj,klab->ijab', eris_oooo, t2_1_a[:],optimize=True)
        t2_2_a -= 0.5*lib.einsum('kjli,klab->ijab', eris_oooo, t2_1_a[:],optimize=True)

        temp = lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1_a[:],optimize=True)
        temp -= lib.einsum('kjbc,kica->ijab',eris_oovv,t2_1_a[:],optimize=True)
        temp_1 = lib.einsum('kcbj,ikac->ijab',eris_OVvo,t2_1_ab[:],optimize=True)

        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - \
                                            temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        del temp
        del temp_1

        if isinstance(myadc._scf, scf.rohf.ROHF):
            t2_2_a += lib.einsum('la,ibjl->ijab',t1_1_a,eris.ovoo, optimize = True)
            t2_2_a -= lib.einsum('la,jbil->ijab',t1_1_a,eris.ovoo, optimize = True)


            t2_2_a -= lib.einsum('lb,iajl->ijab',t1_1_a,eris.ovoo, optimize = True)
            t2_2_a += lib.einsum('lb,jail->ijab',t1_1_a,eris.ovoo, optimize = True)

            if isinstance(eris.ovvv, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
                for a,b in lib.prange(0,nocc_a,chnk_size):
                    eris_ovvv = dfadc.get_ovvv_spin_df(
                        myadc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                    t2_2_a[:,a:b] += lib.einsum('id,jbad->ijab',t1_1_a,eris_ovvv, optimize = True)
                    t2_2_a[:,a:b] -= lib.einsum('id,jabd->ijab',t1_1_a,eris_ovvv, optimize = True)
                    t2_2_a[a:b] -= lib.einsum('jd,ibad->ijab',t1_1_a,eris_ovvv, optimize = True)
                    t2_2_a[a:b] += lib.einsum('jd,iabd->ijab',t1_1_a,eris_ovvv, optimize = True)
                    del eris_ovvv
            else:
                eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                t2_2_a += lib.einsum('id,jbad->ijab',t1_1_a,eris_ovvv, optimize = True)
                t2_2_a -= lib.einsum('id,jabd->ijab',t1_1_a,eris_ovvv, optimize = True)
                t2_2_a -= lib.einsum('jd,ibad->ijab',t1_1_a,eris_ovvv, optimize = True)
                t2_2_a += lib.einsum('jd,iabd->ijab',t1_1_a,eris_ovvv, optimize = True)
                del eris_ovvv

        if isinstance(eris.VVVV_p, np.ndarray):
            eris_VVVV = eris.VVVV_p
            temp = np.ascontiguousarray(
                t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]).reshape(nocc_b*nocc_b,-1)
            temp = np.dot(temp,eris_VVVV.T).reshape(nocc_b, nocc_b, -1)
            t2_1_vvvv_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
            t2_1_vvvv_b[:,:,ab_ind_b[0],ab_ind_b[1]] = temp
            t2_1_vvvv_b[:,:,ab_ind_b[1],ab_ind_b[0]] = -temp
            del eris_VVVV
        elif isinstance(eris.VVVV_p, list) :
            t2_1_vvvv_b = contract_ladder_antisym(myadc,t2_1_b[:],eris.VVVV_p, pack = False)
        else:
            t2_1_vvvv_b = contract_ladder(myadc, t2_1_b[:], (eris.LVV, eris.LVV))

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv_b = h5cache_t2.create_dataset('t2_1_vvvv_b', data=t2_1_vvvv_b)

        t2_2_b = t2_1_vvvv_b[:].copy()

        t2_2_b += 0.5*lib.einsum('kilj,klab->ijab', eris_OOOO, t2_1_b[:],optimize=True)
        t2_2_b -= 0.5*lib.einsum('kjli,klab->ijab', eris_OOOO, t2_1_b[:],optimize=True)

        temp = lib.einsum('kcbj,kica->ijab',eris_OVVO,t2_1_b[:],optimize=True)
        temp -= lib.einsum('kjbc,kica->ijab',eris_OOVV,t2_1_b[:],optimize=True)
        temp_1 = lib.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_ab[:],optimize=True)

        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - \
                                            temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        del temp
        del temp_1

        if isinstance(myadc._scf, scf.rohf.ROHF):
            t2_2_b += lib.einsum('la,ibjl->ijab',t1_1_b,eris.OVOO, optimize = True)
            t2_2_b -= lib.einsum('la,jbil->ijab',t1_1_b,eris.OVOO, optimize = True)

            t2_2_b -= lib.einsum('lb,iajl->ijab',t1_1_b,eris.OVOO, optimize = True)
            t2_2_b += lib.einsum('lb,jail->ijab',t1_1_b,eris.OVOO, optimize = True)

            if isinstance(eris.OVVV, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
                for a,b in lib.prange(0,nocc_b,chnk_size):
                    eris_OVVV = dfadc.get_ovvv_spin_df(
                        myadc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                    t2_2_b[:,a:b] += lib.einsum('id,jbad->ijab',t1_1_b,eris_OVVV, optimize = True)
                    t2_2_b[:,a:b] -= lib.einsum('id,jabd->ijab',t1_1_b,eris_OVVV, optimize = True)
                    t2_2_b[a:b] -= lib.einsum('jd,ibad->ijab',t1_1_b,eris_OVVV, optimize = True)
                    t2_2_b[a:b] += lib.einsum('jd,iabd->ijab',t1_1_b,eris_OVVV, optimize = True)
                    del eris_OVVV
            else:
                eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                t2_2_b += lib.einsum('id,jbad->ijab',t1_1_b,eris_OVVV, optimize = True)
                t2_2_b -= lib.einsum('id,jabd->ijab',t1_1_b,eris_OVVV, optimize = True)
                t2_2_b -= lib.einsum('jd,ibad->ijab',t1_1_b,eris_OVVV, optimize = True)
                t2_2_b += lib.einsum('jd,iabd->ijab',t1_1_b,eris_OVVV, optimize = True)
                del eris_OVVV

        if isinstance(eris.vVvV_p, np.ndarray):
            temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
            eris_vVvV = eris.vVvV_p
            t2_1_vvvv_ab = np.dot(temp,eris_vVvV.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        elif isinstance(eris.vVvV_p, list):
            t2_1_vvvv_ab = contract_ladder(myadc,t2_1_ab[:],eris.vVvV_p)
        else:
            t2_1_vvvv_ab = contract_ladder(myadc,t2_1_ab[:],(eris.Lvv,eris.LVV))

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv_ab = h5cache_t2.create_dataset('t2_1_vvvv_ab', data=t2_1_vvvv_ab)

        t2_2_ab = t2_1_vvvv_ab[:].copy()

        t2_2_ab += lib.einsum('kilj,klab->ijab',eris_ooOO,t2_1_ab[:],optimize=True)
        t2_2_ab += lib.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_a[:],optimize=True)
        t2_2_ab += lib.einsum('kcbj,ikac->ijab',eris_OVVO,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kjbc,ikac->ijab',eris_OOVV,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kibc,kjac->ijab',eris_ooVV,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kjac,ikcb->ijab',eris_OOvv,t2_1_ab[:],optimize=True)
        t2_2_ab += lib.einsum('kcai,kjcb->ijab',eris_OVvo,t2_1_b[:],optimize=True)
        t2_2_ab += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1_ab[:],optimize=True)

        if isinstance(myadc._scf, scf.rohf.ROHF):
            t2_2_ab -= lib.einsum('la,jbil->ijab',t1_1_a,eris.OVoo, optimize = True)
            t2_2_ab -= lib.einsum('lb,iajl->ijab',t1_1_b,eris.ovOO, optimize = True)

            if isinstance(eris.OVvv, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
                for a,b in lib.prange(0,nocc_b,chnk_size):
                    eris_OVvv = dfadc.get_ovvv_spin_df(
                        myadc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                    t2_2_ab[:, a:b] += lib.einsum('id,jbad->ijab',
                                                    t1_1_a,eris_OVvv, optimize = True)
                    del eris_OVvv
            else:
                eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                t2_2_ab += lib.einsum('id,jbad->ijab',t1_1_a,eris_OVvv, optimize = True)
                del eris_OVvv

            if isinstance(eris.ovVV, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
                for a,b in lib.prange(0,nocc_a,chnk_size):
                    eris_ovVV = dfadc.get_ovvv_spin_df(
                        myadc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                    t2_2_ab[a:b] += lib.einsum('jd,iabd->ijab',t1_1_b,eris_ovVV, optimize = True)
                    del eris_ovVV
            else:
                eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                t2_2_ab += lib.einsum('jd,iabd->ijab',t1_1_b,eris_ovVV, optimize = True)
                del eris_ovVV

        D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
        D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
        t2_2_a = t2_2_a/D2_a
        if not isinstance(eris.oooo, np.ndarray):
            t2_2_a = h5cache_t2.create_dataset('t2_2_a', data=t2_2_a)
        del D2_a

        D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
        D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))
        t2_2_b = t2_2_b/D2_b
        if not isinstance(eris.oooo, np.ndarray):
            t2_2_b = h5cache_t2.create_dataset('t2_2_b', data=t2_2_b)
        del D2_b

        D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
        D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))
        t2_2_ab = t2_2_ab/D2_ab
        if not isinstance(eris.oooo, np.ndarray):
            t2_2_ab = h5cache_t2.create_dataset('t2_2_ab', data=t2_2_ab)
        del D2_ab

        cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

    if (myadc.method == "adc(3)" and myadc.approx_trans_moments is False):
        # Compute third-order singles (tij)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

        t1_3 = (None,)

        t1_3_a = lib.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_a[:],t1_2_a,optimize=True)
        t1_3_a += lib.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_ab[:],t1_2_b,optimize=True)

        t1_3_b  = lib.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b += lib.einsum('d,lida,ld->ia',e_a[nocc_a:],t2_1_ab[:],t1_2_a,optimize=True)

        t1_3_a -= lib.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a[:], t1_2_a,optimize=True)
        t1_3_a -= lib.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab[:],t1_2_b,optimize=True)

        t1_3_b -= lib.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b -= lib.einsum('l,lida,ld->ia',e_a[:nocc_a],t2_1_ab[:],t1_2_a,optimize=True)

        t1_3_a += 0.5*lib.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a[:], t1_2_a,optimize=True)
        t1_3_a += 0.5*lib.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab[:],t1_2_b,optimize=True)

        t1_3_b += 0.5*lib.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b += 0.5*lib.einsum('a,lida,ld->ia',e_b[nocc_b:],t2_1_ab[:],t1_2_a,optimize=True)

        t1_3_a -= 0.5*lib.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a[:], t1_2_a,optimize=True)
        t1_3_a -= 0.5*lib.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab[:],t1_2_b,optimize=True)

        t1_3_b -= 0.5*lib.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b -= 0.5*lib.einsum('i,lida,ld->ia',e_b[:nocc_b],t2_1_ab[:],t1_2_a,optimize=True)

        t1_3_a += lib.einsum('ld,iadl->ia',t1_2_a,eris_ovvo,optimize=True)
        t1_3_a -= lib.einsum('ld,ladi->ia',t1_2_a,eris_ovvo,optimize=True)
        t1_3_a += lib.einsum('ld,iadl->ia',t1_2_b,eris_ovVO,optimize=True)

        t1_3_b += lib.einsum('ld,iadl->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b -= lib.einsum('ld,ladi->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b += lib.einsum('ld,ldai->ia',t1_2_a,eris_ovVO,optimize=True)

        t1_3_a += lib.einsum('ld,ldai->ia',t1_2_a,eris_ovvo ,optimize=True)
        t1_3_a -= lib.einsum('ld,liad->ia',t1_2_a,eris_oovv ,optimize=True)
        t1_3_a += lib.einsum('ld,ldai->ia',t1_2_b,eris_OVvo,optimize=True)

        t1_3_b += lib.einsum('ld,ldai->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b -= lib.einsum('ld,liad->ia',t1_2_b,eris_OOVV ,optimize=True)
        t1_3_b += lib.einsum('ld,ldai->ia',t1_2_a,eris_ovVO,optimize=True)

        t1_3_a -= 0.5*lib.einsum('lmad,mdli->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3_a += 0.5*lib.einsum('lmad,ldmi->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3_a -=     lib.einsum('lmad,mdli->ia',t2_2_ab,eris_OVoo,optimize=True)

        t1_3_b -= 0.5*lib.einsum('lmad,mdli->ia',t2_2_b,eris_OVOO,optimize=True)
        t1_3_b += 0.5*lib.einsum('lmad,ldmi->ia',t2_2_b,eris_OVOO,optimize=True)
        t1_3_b -=     lib.einsum('mlda,mdli->ia',t2_2_ab,eris_ovOO,optimize=True)

        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(
                    myadc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                t1_3_a += 0.5*lib.einsum('ilde,lead->ia',t2_2_a[:,a:b],eris_ovvv,optimize=True)
                t1_3_a -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_a[:,a:b],eris_ovvv,optimize=True)
                t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',
                                     t2_1_a[:], eris_ovvv,  t2_1_a[:,a:b] ,optimize=True)
                t1_3_a += lib.einsum('ildf,mafe,lmde->ia',
                                     t2_1_a[:], eris_ovvv,  t2_1_a[:,a:b] ,optimize=True)
                t1_3_a += lib.einsum('ilfd,mefa,mled->ia',
                                     t2_1_ab[:],eris_ovvv, t2_1_ab[a:b],optimize=True)
                t1_3_a -= lib.einsum('ilfd,mafe,mled->ia',
                                     t2_1_ab[:],eris_ovvv, t2_1_ab[a:b],optimize=True)
                t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                         t2_1_a[:],eris_ovvv,t2_1_a[:,a:b],optimize=True)
                t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                         t2_1_a[:],eris_ovvv,t2_1_a[:,a:b],optimize=True)
                t1_3_b += 0.5*lib.einsum('lifa,mefd,lmde->ia',
                                         t2_1_ab[:],eris_ovvv,t2_1_a[:,a:b],optimize=True)
                t1_3_b -= 0.5*lib.einsum('lifa,mdfe,lmde->ia',
                                         t2_1_ab[:],eris_ovvv,t2_1_a[:,a:b],optimize=True)
                t1_3_a[a:b] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                                t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
                t1_3_a[a:b] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',
                                                t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
                t1_3_a[a:b] += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab[:],
                                            eris_ovvv,t2_1_ab[:],optimize=True)
                t1_3_a[a:b] -= lib.einsum('mlfd,ifea,mled->ia',t2_1_ab[:],
                                            eris_ovvv,t2_1_ab[:],optimize=True)
                t1_3_a[a:b] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',
                                                 t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
                t1_3_a[a:b] += 0.25*lib.einsum('lmef,ifde,lmad->ia',
                                                 t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
                del eris_ovvv
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            t1_3_a += 0.5*lib.einsum('ilde,lead->ia',t2_2_a[:],eris_ovvv,optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_a[:],eris_ovvv,optimize=True)
            t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',
                                 t2_1_a[:], eris_ovvv,  t2_1_a[:] ,optimize=True)
            t1_3_a += lib.einsum('ildf,mafe,lmde->ia',
                                 t2_1_a[:], eris_ovvv,  t2_1_a[:] ,optimize=True)
            t1_3_a += lib.einsum('ilfd,mefa,mled->ia',
                                 t2_1_ab[:],eris_ovvv, t2_1_ab[:],optimize=True)
            t1_3_a -= lib.einsum('ilfd,mafe,mled->ia',
                                 t2_1_ab[:],eris_ovvv, t2_1_ab[:],optimize=True)
            t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                     t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                     t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_b += 0.5*lib.einsum('lifa,mefd,lmde->ia',
                                     t2_1_ab[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_b -= 0.5*lib.einsum('lifa,mdfe,lmde->ia',
                                     t2_1_ab[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                     t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',
                                     t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab[:],eris_ovvv,t2_1_ab[:],optimize=True)
            t1_3_a -= lib.einsum('mlfd,ifea,mled->ia',t2_1_ab[:],eris_ovvv,t2_1_ab[:],optimize=True)
            t1_3_a -= 0.25*lib.einsum('lmef,iedf,lmad->ia',
                                      t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a += 0.25*lib.einsum('lmef,ifde,lmad->ia',
                                      t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            del eris_ovvv

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(
                    myadc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                t1_3_b += 0.5*lib.einsum('ilde,lead->ia',t2_2_b[:,a:b],eris_OVVV,optimize=True)
                t1_3_b -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_b[:,a:b],eris_OVVV,optimize=True)
                t1_3_b -= lib.einsum('ildf,mefa,lmde->ia',
                                     t2_1_b[:],eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_3_b += lib.einsum('ildf,mafe,lmde->ia',
                                     t2_1_b[:],eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_3_b += lib.einsum('lidf,mefa,lmde->ia',
                                     t2_1_ab[:],eris_OVVV,t2_1_ab[:,a:b],optimize=True)
                t1_3_b -= lib.einsum('lidf,mafe,lmde->ia',
                                     t2_1_ab[:],eris_OVVV,t2_1_ab[:,a:b],optimize=True)
                t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                         t2_1_ab[:],eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                         t2_1_ab[:],eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_3_b += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                         t2_1_b[:],eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_3_b -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                         t2_1_b[:],eris_OVVV,t2_1_b[:,a:b],optimize=True)
                t1_3_b[a:b] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                                t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
                t1_3_b[a:b] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',
                                                t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
                t1_3_b[a:b] += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab[:],
                                            eris_OVVV,t2_1_ab[:],optimize=True)
                t1_3_b[a:b] -= lib.einsum('lmdf,ifea,lmde->ia',t2_1_ab[:],
                                            eris_OVVV,t2_1_ab[:],optimize=True)
                t1_3_b[a:b] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',
                                                 t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
                t1_3_b[a:b] += 0.25*lib.einsum('lmef,ifde,lmad->ia',
                                                 t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
                del eris_OVVV

        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            t1_3_b += 0.5*lib.einsum('ilde,lead->ia',t2_2_b[:],eris_OVVV,optimize=True)
            t1_3_b -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_b[:],eris_OVVV,optimize=True)
            t1_3_b -= lib.einsum('ildf,mefa,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b += lib.einsum('ildf,mafe,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b += lib.einsum('lidf,mefa,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:],optimize=True)
            t1_3_b -= lib.einsum('lidf,mafe,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:],optimize=True)
            t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                     t2_1_ab[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                     t2_1_ab[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                     t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                     t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                     t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',
                                     t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:],optimize=True)
            t1_3_b -= lib.einsum('lmdf,ifea,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:],optimize=True)
            t1_3_b -= 0.25*lib.einsum('lmef,iedf,lmad->ia',
                                      t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b += 0.25*lib.einsum('lmef,ifde,lmad->ia',
                                      t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            del eris_OVVV

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(
                    myadc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                t1_3_b += lib.einsum('lied,lead->ia',t2_2_ab[a:b],eris_ovVV,optimize=True)
                t1_3_a -= lib.einsum('ildf,mafe,mlde->ia',
                                     t2_1_ab[:],eris_ovVV,t2_1_ab[a:b],optimize=True)
                t1_3_b -= lib.einsum('ildf,mefa,mled->ia',
                                     t2_1_b[:],eris_ovVV,t2_1_ab[a:b],optimize=True)
                t1_3_b += lib.einsum('lidf,mefa,lmde->ia',
                                     t2_1_ab[:],eris_ovVV,t2_1_a[:,a:b],optimize=True)
                t1_3_a += lib.einsum('ilaf,mefd,mled->ia',
                                     t2_1_ab[:],eris_ovVV,t2_1_ab[a:b],optimize=True)
                t1_3_b += lib.einsum('ilaf,mefd,mled->ia',
                                     t2_1_b[:],eris_ovVV,t2_1_ab[a:b],optimize=True)
                t1_3_a[a:b] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                                t2_1_b[:],eris_ovVV,t2_1_b[:],optimize=True)
                t1_3_a[a:b] += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab[:],
                                            eris_ovVV,t2_1_ab[:],optimize=True)
                t1_3_a[a:b] -= lib.einsum('lmef,iedf,lmad->ia',t2_1_ab[:],
                                            eris_ovVV,t2_1_ab[:],optimize=True)
                del eris_ovVV
        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            t1_3_b += lib.einsum('lied,lead->ia',t2_2_ab[:],eris_ovVV,optimize=True)
            t1_3_a -= lib.einsum('ildf,mafe,mlde->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[:],optimize=True)
            t1_3_b -= lib.einsum('ildf,mefa,mled->ia',t2_1_b[:],eris_ovVV,t2_1_ab[:],optimize=True)
            t1_3_b += lib.einsum('lidf,mefa,lmde->ia',t2_1_ab[:],eris_ovVV,t2_1_a[:],optimize=True)
            t1_3_a += lib.einsum('ilaf,mefd,mled->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[:],optimize=True)
            t1_3_b += lib.einsum('ilaf,mefd,mled->ia',t2_1_b[:],eris_ovVV,t2_1_ab[:],optimize=True)
            t1_3_a += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                     t2_1_b[:],eris_ovVV,t2_1_b[:],optimize=True)
            t1_3_a += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[:],optimize=True)
            t1_3_a -= lib.einsum('lmef,iedf,lmad->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[:],optimize=True)
            del eris_ovVV

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
            for a,b in lib.prange(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(
                    myadc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                t1_3_a += lib.einsum('ilde,lead->ia',t2_2_ab[:,a:b],eris_OVvv,optimize=True)
                t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',
                                     t2_1_a[:],eris_OVvv, t2_1_ab[:,a:b],optimize=True)
                t1_3_a += lib.einsum('ilfd,mefa,lmde->ia',
                                     t2_1_ab[:],eris_OVvv,t2_1_b[:,a:b] ,optimize=True)
                t1_3_b -= lib.einsum('lifd,mafe,lmed->ia',
                                     t2_1_ab[:],eris_OVvv,t2_1_ab[:,a:b],optimize=True)
                t1_3_a += lib.einsum('ilaf,mefd,lmde->ia',
                                     t2_1_a[:],eris_OVvv,t2_1_ab[:,a:b],optimize=True)
                t1_3_b += lib.einsum('lifa,mefd,lmde->ia',
                                     t2_1_ab[:],eris_OVvv,t2_1_ab[:,a:b],optimize=True)
                t1_3_b[a:b] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                                t2_1_a[:],eris_OVvv,t2_1_a[:],optimize=True)
                t1_3_b[a:b] += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab[:],
                                            eris_OVvv,t2_1_ab[:],optimize=True)
                t1_3_b[a:b] -= lib.einsum('mlfe,iedf,mlda->ia',t2_1_ab[:],
                                            eris_OVvv,t2_1_ab[:],optimize=True)

                del eris_OVvv
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            t1_3_a += lib.einsum('ilde,lead->ia',t2_2_ab[:],eris_OVvv,optimize=True)
            t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a[:], eris_OVvv,t2_1_ab[:],optimize=True)
            t1_3_a += lib.einsum('ilfd,mefa,lmde->ia',t2_1_ab[:],eris_OVvv,t2_1_b[:] ,optimize=True)
            t1_3_b -= lib.einsum('lifd,mafe,lmed->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:],optimize=True)
            t1_3_a += lib.einsum('ilaf,mefd,lmde->ia',t2_1_a[:],eris_OVvv,t2_1_ab[:],optimize=True)
            t1_3_b += lib.einsum('lifa,mefd,lmde->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:],optimize=True)
            t1_3_b += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                     t2_1_a[:],eris_OVvv,t2_1_a[:],optimize=True)
            t1_3_b += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:],optimize=True)
            t1_3_b -= lib.einsum('mlfe,iedf,mlda->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:],optimize=True)

            del eris_OVvv

        t1_3_a += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('inde,lamn,lmde->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)

        t1_3_b += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('nied,lamn,mled->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)

        t1_3_a += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= 0.5 * lib.einsum('inad,lemn,mlde->ia',
                                   t2_1_a[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5 * lib.einsum('inad,meln,lmde->ia',
                                   t2_1_a[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5 *lib.einsum('inad,lemn,lmed->ia',
                                  t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,mled->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_a += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_b[:],optimize=True)

        t1_3_b += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= 0.5 * lib.einsum('inad,meln,mled->ia',
                                   t2_1_b[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5 * lib.einsum('inad,lemn,lmed->ia',
                                   t2_1_b[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5 *lib.einsum('nida,meln,lmde->ia',
                                  t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('nida,lemn,mlde->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_b += 0.5*lib.einsum('nida,lemn,lmde->ia',t2_1_ab[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('nida,meln,lmde->ia',t2_1_ab[:],eris_ovoo,t2_1_a[:],optimize=True)

        t1_3_a -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= lib.einsum('nled,ianm,mled->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a += lib.einsum('nled,naim,mled->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_b[:],eris_ovOO,t2_1_b[:],optimize=True)
        t1_3_a -= lib.einsum('lnde,ianm,lmde->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)

        t1_3_b -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= lib.einsum('lnde,ianm,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b += lib.einsum('lnde,naim,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a[:],eris_OVoo,t2_1_a[:],optimize=True)
        t1_3_b -= lib.einsum('nled,ianm,mled->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)

        t1_3_a -= lib.einsum('lnde,ienm,lmad->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('lnde,neim,lmad->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('lnde,neim,lmad->ia',t2_1_ab[:],eris_OVoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('nled,ienm,mlad->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a -= lib.einsum('nled,neim,mlad->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a += lib.einsum('lned,ienm,lmad->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_a -= lib.einsum('lnde,neim,mlad->ia',t2_1_b[:],eris_OVoo,t2_1_ab[:],optimize=True)

        t1_3_b -= lib.einsum('lnde,ienm,lmad->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('lnde,neim,lmad->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('nled,neim,lmad->ia',t2_1_ab[:],eris_ovOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('lnde,ienm,lmda->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b -= lib.einsum('lnde,neim,lmda->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b += lib.einsum('nlde,ienm,mlda->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_b -= lib.einsum('lnde,neim,lmda->ia',t2_1_a[:],eris_ovOO,t2_1_ab[:],optimize=True)

        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b

        t1_3 = (t1_3_a, t1_3_b)

    del D1_a, D1_b

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)

    if ((myadc.method == "adc(2)" and myadc.method_type == "ee" and myadc.approx_trans_moments is False)
        or (myadc.method == "adc(2)-x" and myadc.approx_trans_moments is False)
        or (myadc.method == "adc(3)")):
        t2_2 = (t2_2_a , t2_2_ab, t2_2_b)
        t2_1_vvvv = (t2_1_vvvv_a, t2_1_vvvv_ab, t2_1_vvvv_b)

    t1 = (t1_2, t1_3, t1_1)
    t2 = (t2_1, t2_2)

    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)

    return t1, t2, t2_1_vvvv


def compute_energy(myadc, t1, t2, eris):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO

    #Compute MPn correlation energy
    t2_a  = t2[0][0][:].copy()
    if (myadc.method == "adc(3)"):
        t2_a += t2[1][0][:]

    e_mp = 0.25 * lib.einsum('ijab,iabj', t2_a, eris_ovvo)
    e_mp -= 0.25 * lib.einsum('ijab,ibaj', t2_a, eris_ovvo)
    del t2_a

    t2_ab  = t2[0][1][:].copy()
    if (myadc.method == "adc(3)"):
        t2_ab += t2[1][1][:]

    e_mp += lib.einsum('ijab,iabj', t2_ab, eris_ovVO)
    del t2_ab

    t2_b  = t2[0][2][:].copy()
    if (myadc.method == "adc(3)"):
        t2_b += t2[1][2][:]

    e_mp += 0.25 * lib.einsum('ijab,iabj', t2_b, eris_OVVO)
    e_mp -= 0.25 * lib.einsum('ijab,ibaj', t2_b, eris_OVVO)
    del t2_b

    logger.info(myadc, "Reference correlation energy (doubles): %.8f", e_mp)

    if isinstance(myadc._scf, scf.rohf.ROHF):
        f_ov_a = myadc.f_ov[0]
        f_ov_b = myadc.f_ov[1]
        t1_1_a = t1[2][0].copy()
        t1_1_b = t1[2][1].copy()

        if (myadc.method == "adc(3)"):
            t1_1_a += t1[0][0]
            t1_1_b += t1[0][1]

        singles = lib.einsum('ia,ia', f_ov_a, t1_1_a)
        singles += lib.einsum('ia,ia', f_ov_b, t1_1_b)

        e_mp += singles

        logger.info(myadc, "Reference correlation energy (singles): %.8f", singles)

    cput0 = log.timer_debug1("Completed energy calculation", *cput0)

    return e_mp


def contract_ladder(myadc,t_amp,vvvv_p, prefactor = 1.0, pack = False):

    nocc_a = t_amp.shape[0]
    nocc_b = t_amp.shape[1]
    nvir_a = t_amp.shape[2]
    nvir_b = t_amp.shape[3]

    tril_idx = np.tril_indices(nvir_a, k=-1)
    t_amp_t = np.ascontiguousarray(t_amp.reshape(nocc_a*nocc_b,-1).T)
    t = np.zeros((nvir_a,nvir_b, nocc_a*nocc_b))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)

    if isinstance(vvvv_p, list):
        a = 0
        for dataset in vvvv_p:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir_a * nvir_b)
            t[a:a+k] = np.dot(dataset,t_amp_t).reshape(-1,nvir_b,nocc_a*nocc_b)
            a += k
    elif getattr(myadc, 'with_df', None):
        for a,b in lib.prange(0,nvir_a,chnk_size):
            Lvv = vvvv_p[0]
            LVV = vvvv_p[1]
            vvvv = dfadc.get_vVvV_df(myadc, Lvv, LVV, a, chnk_size)
            vvvv = vvvv.reshape(-1,nvir_a*nvir_b)
            t[a:b] = np.dot(vvvv,t_amp_t).reshape(-1,nvir_b,nocc_a*nocc_b)
            del vvvv
    else:
        raise Exception("Unknown vvvv type")

    t = prefactor * np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc_a, nocc_b, nvir_a, nvir_b)
    if pack:
        t = t[:, :, tril_idx[0], tril_idx[1]]

    return t


def contract_ladder_antisym(myadc,t_amp,vvvv_d, pack = True):

    nocc = t_amp.shape[0]
    nvir = t_amp.shape[2]

    nv_pair = nvir  *  (nvir - 1) // 2
    tril_idx = np.tril_indices(nvir, k=-1)

    t_amp = t_amp[:,:,tril_idx[0],tril_idx[1]]
    t_amp_t = np.ascontiguousarray(t_amp.reshape(nocc*nocc,-1).T)

    t = np.zeros((nvir,nvir, nocc*nocc))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)

    if isinstance(vvvv_d, list):
        a = 0
        for dataset in vvvv_d:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nv_pair)
            t[a:a+k] = np.dot(dataset,t_amp_t).reshape(-1,nvir,nocc*nocc)
            a += k
    elif getattr(myadc, 'with_df', None):
        for a,b in lib.prange(0,nvir,chnk_size):
            vvvv = dfadc.get_vvvv_antisym_df(myadc, vvvv_d, a, chnk_size)
            vvvv = vvvv.reshape(-1,nv_pair)
            t[a:b] = np.dot(vvvv,t_amp_t).reshape(-1,nvir,nocc*nocc)
            del vvvv
    else:
        raise Exception("Unknown vvvv type")

    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc, nocc, nvir, nvir)

    if pack:
        t = t[:, :, tril_idx[0], tril_idx[1]]

    return t
