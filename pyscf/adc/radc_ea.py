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

'''
Restricted algebraic diagrammatic construction
'''
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import symm


def get_imds(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2 = t1[0]

    eris_ovvo = eris.ovvo
    nocc = adc._nocc
    nvir = adc._nvir

    e_vir = adc.mo_energy[nocc:].copy()

    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    # a-b block
    # Zeroth-order terms

    M_ab = lib.einsum('ab,a->ab', idn_vir, e_vir)

    # Second-order terms
    t2_1 = t2[0][:]

    M_ab -= 1.5 * 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 * 0.5 *  lib.einsum('mlad,lbdm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 * 0.5 *  lib.einsum('lmad,ldbm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab -= 0.5 * 0.5 *  lib.einsum('mlad,ldbm->ab',t2_1, eris_ovvo,optimize=True)
    #M_ab -= 0.5 *        lib.einsum('lmad,lbdm->ab',t2_1, eris_ovvo,optimize=True)

    M_ab -= 1.5 * 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 * 0.5 *  lib.einsum('mlbd,ladm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 * 0.5 *  lib.einsum('lmbd,ldam->ab',t2_1, eris_ovvo,optimize=True)
    M_ab -= 0.5 * 0.5 *  lib.einsum('mlbd,ldam->ab',t2_1, eris_ovvo,optimize=True)
    #M_ab -= 0.5 *        lib.einsum('lmbd,ladm->ab',t2_1, eris_ovvo,optimize=True)

    del t2_1
    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    #Third-order terms

    if(method =='adc(3)'):

        eris_oovv = eris.oovv

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc,chnk_size):
                eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv,
                                              p, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = eris_ovvv.shape[0]
                M_ab += 4. * lib.einsum('ld,ldab->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
                M_ab -=  lib.einsum('ld,lbad->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
                M_ab -= lib.einsum('ld,ladb->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
                del eris_ovvv
                a += k
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            M_ab += 4. * lib.einsum('ld,ldab->ab',t1_2, eris_ovvv,optimize=True)
            M_ab -=  lib.einsum('ld,lbad->ab',t1_2, eris_ovvv,optimize=True)
            M_ab -= lib.einsum('ld,ladb->ab',t1_2, eris_ovvv,optimize=True)
            del eris_ovvv

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)
        t2_2 = t2[1][:]

        M_ab -= 0.5 * 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab += 0.5 * 0.5 *  lib.einsum('mlad,lbdm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab += 0.5 * 0.5 *  lib.einsum('lmad,ldbm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -= 0.5 * 0.5 *  lib.einsum('mlad,ldbm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -= 0.5 *        lib.einsum('lmad,lbdm->ab',t2_2, eris_ovvo,optimize=True)

        M_ab -= 0.5 * 0.5 * lib.einsum('lmbd,ladm->ab',t2_2,eris_ovvo,optimize=True)
        M_ab += 0.5 * 0.5 * lib.einsum('mlbd,ladm->ab',t2_2,eris_ovvo,optimize=True)
        M_ab += 0.5 * 0.5 * lib.einsum('lmbd,ldam->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -= 0.5 * 0.5 * lib.einsum('mlbd,ldam->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -= 0.5 * 1.0 * lib.einsum('lmbd,ladm->ab',t2_2,eris_ovvo,optimize=True)
        t2_1 = t2[0][:]

        log.timer_debug1("Starting the small integrals  calculation")
        temp_t2_v_1 = lib.einsum('lned,mlbd->nemb',t2_1, t2_1,optimize=True)
        M_ab -= 0.5 * lib.einsum('nemb,nmae->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab -= 0.5 * lib.einsum('mbne,nmae->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab += 0.5 * lib.einsum('nemb,maen->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += 0.5 * lib.einsum('mbne,maen->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += 0.5 * lib.einsum('nemb,neam->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab -= 0.5 * lib.einsum('name,nmeb->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab -= 0.5 * lib.einsum('mena,nmeb->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab += 0.5 * 2. * lib.einsum('name,nbem->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += 0.5 * 2. * lib.einsum('mena,nbem->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += 0.5 * lib.einsum('nbme,mean->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        del temp_t2_v_1

        temp_t2_v_2 = lib.einsum('nled,mlbd->nemb',t2_1, t2_1,optimize=True)
        M_ab += 0.5 * 2. * lib.einsum('nemb,nmae->ab',temp_t2_v_2, eris_oovv, optimize=True)
        M_ab -= 0.5 * 4. * lib.einsum('nemb,maen->ab',temp_t2_v_2, eris_ovvo, optimize=True)
        M_ab += 0.5 * 2. * lib.einsum('mena,nmeb->ab',temp_t2_v_2, eris_oovv, optimize=True)
        M_ab -= 0.5 * 4. * lib.einsum('mena,nbem->ab',temp_t2_v_2, eris_ovvo, optimize=True)
        del temp_t2_v_2

        temp_t2_v_3 = lib.einsum('lned,lmbd->nemb',t2_1, t2_1,optimize=True)
        M_ab -= 0.5 * lib.einsum('nemb,maen->ab',temp_t2_v_3, eris_ovvo, optimize=True)
        M_ab += 0.5 * 2. * lib.einsum('nemb,nmae->ab',temp_t2_v_3, eris_oovv, optimize=True)
        M_ab += 0.5 * 2. * lib.einsum('mena,nmeb->ab',temp_t2_v_3, eris_oovv, optimize=True)
        M_ab -= 0.5 * lib.einsum('mena,nbem->ab',temp_t2_v_3, eris_ovvo, optimize=True)
        del temp_t2_v_3

        temp_t2_v_8 = lib.einsum('lned,mled->mn',t2_1, t2_1,optimize=True)
        M_ab += 2.* lib.einsum('mn,nmab->ab',temp_t2_v_8, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mn,nbam->ab', temp_t2_v_8, eris_ovvo, optimize=True)
        del temp_t2_v_8

        temp_t2_v_9 = lib.einsum('nled,mled->mn',t2_1, t2_1,optimize=True)
        M_ab -= 4.* lib.einsum('mn,nmab->ab',temp_t2_v_9, eris_oovv, optimize=True)
        M_ab += 2. * lib.einsum('mn,nbam->ab',temp_t2_v_9, eris_ovvo, optimize=True)
        del temp_t2_v_9

        log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

        log.timer_debug1("Starting M_ab vvvv ADC(3) calculation")

        if isinstance(eris.vvvv, np.ndarray):
            temp_t2 = adc.imds.t2_1_vvvv
            M_ab -= 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlaf,lmbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('lmaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmaf,lmbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlaf,mlfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('mlaf,lmfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmaf,mlfb->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('lmaf,lmfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.5 * lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)

            M_ab -= 0.5 * 0.25*lib.einsum('mlad,mlbd->ab', temp_t2, t2_1, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlad,lmbd->ab', temp_t2, t2_1, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('lmad,mlbd->ab', temp_t2, t2_1, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmad,lmbd->ab', temp_t2, t2_1, optimize=True)
            M_ab -= 0.5 * lib.einsum('mlad,mlbd->ab', temp_t2, t2_1, optimize=True)

            M_ab += 0.5 * 0.25*lib.einsum('lmad,mlbd->ab',temp_t2, t2_1, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmad,lmbd->ab',temp_t2, t2_1, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('mlad,mlbd->ab',temp_t2, t2_1, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlad,lmbd->ab',temp_t2, t2_1, optimize=True)
            del temp_t2

            eris_vvvv =  eris.vvvv
            eris_vvvv = eris_vvvv.reshape(nvir,nvir,nvir,nvir)
            M_ab -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            eris_vvvv = eris_vvvv.reshape(nvir*nvir,nvir*nvir)

        else:
            temp_t2_vvvv = adc.imds.t2_1_vvvv[:]
            M_ab -= 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlaf,lmbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('lmaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmaf,lmbf->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab += 0.5 * 0.25*lib.einsum('mlaf,mlfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('mlaf,lmfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmaf,mlfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('lmaf,lmfb->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab -= 0.5 * lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab += 0.5 * 0.25*lib.einsum('lmad,mlbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmad,lmbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('mlad,mlbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlad,lmbd->ab',temp_t2_vvvv, t2_1, optimize=True)

            M_ab -= 0.5 * 0.25*lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('mlad,lmbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.5 * 0.25*lib.einsum('lmad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.5 * 0.25*lib.einsum('lmad,lmbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.5 * lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            del temp_t2_vvvv

            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            temp = np.zeros((nvir,nvir))

            if isinstance(eris.vvvv, list):
                for dataset in eris.vvvv:
                    k = dataset.shape[0]
                    eris_vvvv = dataset[:].reshape(-1,nvir,nvir,nvir)
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1,
                                              t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1,
                                              t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1,
                                              t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1,
                                              t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',
                                                  t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',
                                                  t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',
                                                  t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',
                                                  t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',
                                                 t2_1, t2_1, eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1,
                                              t2_1, eris_vvvv, optimize=True)
                    del eris_vvvv
                    a += k
            else :
                for p in range(0,nvir,chnk_size):

                    vvvv = dfadc.get_vvvv_df(adc, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                    k = vvvv.shape[0]
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',
                                                  t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',
                                                  t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',
                                                  t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',
                                                  t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',
                                                 t2_1, t2_1, vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, vvvv, optimize=True)
                    del vvvv
                    a += k

            M_ab += temp
            del temp
            del t2_1

    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
    return M_ab


def get_diag(adc,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ab = adc.get_imds()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ab = e_vir[:,None] + e_vir
    d_i = e_occ[:,None]
    D_n = -d_i + d_ab.reshape(-1)
    D_iab = D_n.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in p1-p1 block

    M_ab_diag = np.diagonal(M_ab)
    diag[s1:f1] = M_ab_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] = D_iab.copy()
    del D_iab

#    ###### Additional terms for the preconditioner ####
#
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        #TODO Implement this for out-of-core and density-fitted algorithms
#        if isinstance(eris.vvvv, np.ndarray):
#
#            eris_oovv = eris.oovv
#            eris_ovvo = eris.ovvo
#            eris_vvvv = eris.vvvv
#
#            temp = np.zeros((nocc, eris_vvvv.shape[0]))
#            temp[:] += np.diag(eris_vvvv)
#            diag[s2:f2] += temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nvir, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(1,0,2))
#            diag[s2:f2] += -temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nvir, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(1,2,0))
#            diag[s2:f2] += -temp.reshape(-1)
#        else:
#           raise Exception("Precond not available for out-of-core and density-fitted algo")

    log.timer_debug1("Completed ea_diag calculation")
    return diag


def matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method


    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ab = e_vir[:,None] + e_vir
    d_i = e_occ[:,None]
    D_n = -d_i + d_ab.reshape(-1)
    D_iab = D_n.reshape(-1)

    if M_ab is None:
        M_ab = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nocc,nvir,nvir)

############ ADC(2) ab block ############################

        s[s1:f1] = lib.einsum('ab,b->a',M_ab,r1)

############## ADC(2) a - ibc and ibc - a coupling blocks #########################

        temp_doubles = np.zeros((nocc,nvir,nvir))
        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc,chnk_size):
                eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv,
                                              p, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = eris_ovvv.shape[0]
                s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv, r2[a:a+k], optimize=True)
                s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv, r2[a:a+k], optimize=True)

                temp_doubles[a:a+k] += lib.einsum('icab,a->ibc', eris_ovvv, r1, optimize=True)
                del eris_ovvv
                a += k

        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)

            s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv, r2, optimize=True)
            s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv, r2, optimize=True)

            temp_doubles += lib.einsum('icab,a->ibc', eris_ovvv, r1, optimize=True)
            del eris_ovvv

        s[s2:f2] +=  temp_doubles.reshape(-1)
################ ADC(2) iab - jcd block ############################

        s[s2:f2] +=  D_iab * r2.reshape(-1)

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            r2 = r2.reshape(nocc, nvir, nvir)

            if isinstance(eris.vvvv, np.ndarray):
                r_bab_t = r2.reshape(nocc,-1)
                eris_vvvv = eris.vvvv
                s[s2:f2] += np.dot(r_bab_t,eris_vvvv.T).reshape(-1)
            elif isinstance(eris.vvvv, list):
                s[s2:f2] += contract_r_vvvv(adc,r2,eris.vvvv)
            else :
                s[s2:f2] += contract_r_vvvv(adc,r2,eris.Lvv)

            s[s2:f2] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo,r2,optimize=True).reshape(-1)
            s[s2:f2] += lib.einsum('jzyi,jxz->ixy',eris_ovvo,r2,optimize=True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv,r2,optimize=True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r2,optimize=True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jixw,jwy->ixy',eris_oovv,r2,optimize=True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r2,optimize=True).reshape(-1)
            s[s2:f2] += lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2,optimize=True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jwyi,jwx->ixy',eris_ovvo,r2,optimize=True).reshape(-1)

            #print("Calculating additional terms for adc(3)")

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################

            t2_1 = adc.t2[0][:]

            temp =   0.25 * lib.einsum('lmab,jab->lmj',t2_1,r2)
            temp -=  0.25 * lib.einsum('lmab,jba->lmj',t2_1,r2)
            temp -=  0.25 * lib.einsum('mlab,jab->lmj',t2_1,r2)
            temp +=  0.25 * lib.einsum('mlab,jba->lmj',t2_1,r2)

            s[s1:f1] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
            s[s1:f1] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)
            del temp

            temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1,r2)
            s[s1:f1] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovoo, optimize=True)

            temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_s_a -= lib.einsum('jlwd,jwz->lzd',t2_1,r2,optimize=True)
            temp_s_a -= lib.einsum('ljwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_s_a += lib.einsum('ljwd,jwz->lzd',t2_1,r2,optimize=True)
            temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1,r2,optimize=True)

            temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 += lib.einsum('jlzd,jzw->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 += lib.einsum('ljzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 -= lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 += -lib.einsum('ljdz,jwz->lwd',t2_1,r2,optimize=True)

            temp_t2_r2_1 = lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_t2_r2_1 -= lib.einsum('jlwd,jwz->lzd',t2_1,r2,optimize=True)
            temp_t2_r2_1 += lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_t2_r2_1 -= lib.einsum('ljwd,jzw->lzd',t2_1,r2,optimize=True)

            temp_t2_r2_2 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_t2_r2_2 += lib.einsum('jlzd,jzw->lwd',t2_1,r2,optimize=True)
            temp_t2_r2_2 -= lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_t2_r2_2 += lib.einsum('ljzd,jwz->lwd',t2_1,r2,optimize=True)

            temp_t2_r2_3 = -lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)

            temp_a = t2_1.transpose(0,3,1,2).copy()
            temp_b = temp_a.reshape(nocc*nvir,nocc*nvir)
            r2_t = r2.reshape(nocc*nvir,-1)
            temp_c = np.dot(temp_b,r2_t).reshape(nocc,nvir,nvir)
            temp_t2_r2_4 = temp_c.transpose(0,2,1).copy()

            del t2_1

            temp = np.zeros((nocc,nvir,nvir))
            temp_1_1 = np.zeros((nocc,nvir,nvir))
            temp_2_1 = np.zeros((nocc,nvir,nvir))
            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
                a = 0
                for p in range(0,nocc,chnk_size):
                    eris_ovvv = dfadc.get_ovvv_df(
                        adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                    k = eris_ovvv.shape[0]
                    temp_1_1[a:a+k] = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
                    temp_1_1[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r1,optimize=True)
                    temp_2_1[a:a+k] = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)

                    s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a[a:a+k],
                                               eris_ovvv,optimize=True)
                    s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a[a:a+k],
                                               eris_ovvv,optimize=True)
                    s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',
                                               temp_s_a_1[a:a+k],eris_ovvv,optimize=True)
                    s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',
                                               temp_s_a_1[a:a+k],eris_ovvv,optimize=True)

                    s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',
                                               temp_t2_r2_1[a:a+k],eris_ovvv,optimize=True)

                    s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',
                                               temp_t2_r2_2[a:a+k],eris_ovvv,optimize=True)

                    s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',
                                               temp_t2_r2_3[a:a+k],eris_ovvv,optimize=True)

                    s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',
                                               temp_t2_r2_4[a:a+k],eris_ovvv,optimize=True)

                    temp[a:a+k]  -= lib.einsum('lbyd,b->lyd',eris_ovvv,r1,optimize=True)

                    del eris_ovvv
                    a += k

            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)

                temp_1_1 = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
                temp_1_1 -= lib.einsum('lbxd,b->lxd', eris_ovvv,r1,optimize=True)
                temp_2_1 = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)

                s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a,eris_ovvv,optimize=True)
                s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a,eris_ovvv,optimize=True)
                s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1,eris_ovvv,optimize=True)
                s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1,eris_ovvv,optimize=True)

                s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_t2_r2_1,eris_ovvv,optimize=True)

                s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_t2_r2_2,eris_ovvv,optimize=True)

                s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_t2_r2_3,eris_ovvv,optimize=True)

                s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_t2_r2_4,eris_ovvv,optimize=True)

                temp  -= lib.einsum('lbyd,b->lyd',eris_ovvv,r1,optimize=True)

                del eris_ovvv

            t2_1 = adc.t2[0][:]
            temp_1 = -lib.einsum('lyd,lixd->ixy',temp,t2_1,optimize=True)
            s[s2:f2] -= temp_1.reshape(-1)

            del temp_s_a
            del temp_s_a_1
            del temp_t2_r2_1
            del temp_t2_r2_2
            del temp_t2_r2_3
            del temp_t2_r2_4

            temp_1 = lib.einsum('b,lbmi->lmi',r1,eris_ovoo)
            s[s2:f2] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1, optimize=True).reshape(-1)

            temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1,optimize=True)
            temp  += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1,optimize=True)
            temp  -= lib.einsum('lxd,ildy->ixy',temp_2_1,t2_1,optimize=True)
            s[s2:f2] += temp.reshape(-1)

            del t2_1
            del temp
            del temp_1
            del temp_1_1
            del temp_2_1

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        return s

    return sigma_


def get_trans_moments(adc):

    nmo  = adc.nmo
    T = []
    for orb in range(nmo):

        T_a = get_trans_moments_orbital(adc,orb)
        T.append(T_a)

    T = np.array(T)
    return T


def get_trans_moments_orbital(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0][:]
    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        t1_2 = adc.t1[0][:]

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    idn_vir = np.identity(nvir)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T = np.zeros((dim))

######## ADC(2) part  ############################################

    if orb < nocc:

        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            T[s1:f1] = -t1_2[orb,:]

        t2_1_t = -t2_1.transpose(1,0,2,3)

        T[s2:f2] += t2_1_t[:,orb,:,:].reshape(-1)

    else:

        T[s1:f1] += idn_vir[(orb-nocc), :]
        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize=True)

        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize=True)
        T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize=True)
        T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize=True)

######### ADC(3) 2p-1h  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1][:]

        if orb < nocc:

            t2_2_t = -t2_2.transpose(1,0,2,3)

            T[s2:f2] += t2_2_t[:,orb,:,:].reshape(-1)

########### ADC(3) 1p part  ############################################

    if(method=='adc(3)'):

        t2_2 = adc.t2[1][:]
        if (adc.approx_trans_moments is False):
            t1_3 = adc.t1[1]

        if orb < nocc:
            T[s1:f1] += 0.5*lib.einsum('kac,ck->a',t2_1[:,orb,:,:], t1_2.T,optimize=True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize=True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize=True)
            if (adc.approx_trans_moments is False):
                T[s1:f1] -= t1_3[orb,:]

        else:

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize=True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize=True)

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize=True)
            T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize=True)
            T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize=True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize=True)
            T[s1:f1] += 0.25*lib.einsum('klac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize=True)
            T[s1:f1] += 0.25*lib.einsum('lkac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize=True)

        del t2_2
    del t2_1

    T_aaa = T[n_singles:].reshape(nocc,nvir,nvir).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def analyze_eigenvector(adc):

    nocc = adc._nocc
    nvir = adc._nvir
    evec_print_tol = adc.evec_print_tol

    logger.info(adc, "Number of occupied orbitals = %d", nocc)
    logger.info(adc, "Number of virtual orbitals =  %d", nvir)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

    n_singles = nvir
    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nvir,nvir)
        U1dotU1 = np.dot(U1, U1)
        U2dotU2 =  2.*np.dot(U2.ravel(), U2.ravel()) - \
                             np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx]
        U_sorted = U[ind_idx,I].copy()

        U_sorted = U_sorted[U_sq > evec_print_tol**2]
        ind_idx = ind_idx[U_sq > evec_print_tol**2]

        singles_idx = []
        doubles_idx = []
        singles_val = []
        doubles_val = []
        iter_num = 0

        for orb_idx in ind_idx:

            if orb_idx < n_singles:
                a_idx = orb_idx + 1 + nocc
                singles_idx.append(a_idx)
                singles_val.append(U_sorted[iter_num])

            if orb_idx >= n_singles:
                iab_idx = orb_idx - n_singles
                ab_rem = iab_idx % (nvir*nvir)
                i_idx = iab_idx //(nvir*nvir)
                a_idx = ab_rem//nvir
                b_idx = ab_rem % nvir
                doubles_idx.append((i_idx + 1, a_idx + 1 + nocc, b_idx + 1 + nocc))
                doubles_val.append(U_sorted[iter_num])

            iter_num += 1

        logger.info(adc, '%s | root %d | norm(1p)  = %6.4f | norm(1h2p) = %6.4f ',
                    adc.method ,I, U1dotU1, U2dotU2)

        if singles_val:
            logger.info(adc, "\n1p block: ")
            logger.info(adc, "     a     U(a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_val[idx])

        if doubles_val:
            logger.info(adc, "\n1h2p block: ")
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], doubles_val[idx])

        logger.info(adc, "\n*************************************************************\n")


def analyze_spec_factor(adc):

    X = adc.X
    X_2 = (X.copy()**2)*2
    thresh = adc.spec_factor_print_tol

    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)

    for i in range(X_2.shape[1]):

        sort = np.argsort(-X_2[:,i])
        X_2_row = X_2[:,i]
        X_2_row = X_2_row[sort]

        if not adc.mol.symmetry:
            sym = np.repeat(['A'], X_2_row.shape[0])
        else:
            sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff.orbsym]
            sym = np.array(sym)

            sym = sym[sort]

        spec_Contribution = X_2_row[X_2_row > thresh]
        index_mo = sort[X_2_row > thresh]+1

        if np.sum(spec_Contribution) == 0.0:
            continue

        logger.info(adc,'%s | root %d \n',adc.method ,i)
        logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
        logger.info(adc, "-----------------------------------------------------------")

        for c in range(index_mo.shape[0]):
            logger.info(adc, '     %3.d          %10.8f                %s',
                        index_mo[c], spec_Contribution[c], sym[c])

        logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
        logger.info(adc, "\n*************************************************************\n")


def renormalize_eigenvectors(adc, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - \
                       np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[:,I] /= np.sqrt(UdotU)

    return U


def get_properties(adc, nroots=1):

    #Transition moments
    T = adc.get_trans_moments()

    #Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(nroots)
    X = np.dot(T, U).reshape(-1, nroots)

    #Spectroscopic factors
    P = 2.0*lib.einsum("pi,pi->i", X, X)

    return P,X


def analyze(myadc):

    header = ("\n*************************************************************"
              "\n           Eigenvector analysis summary"
              "\n*************************************************************")
    logger.info(myadc, header)

    myadc.analyze_eigenvector()

    if myadc.compute_properties:

        header = ("\n*************************************************************"
                  "\n            Spectroscopic factors analysis summary"
                  "\n*************************************************************")
        logger.info(myadc, header)

        myadc.analyze_spec_factor()

def make_rdm1(adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    nroots = adc.U.shape[1]
    U = adc.renormalize_eigenvectors(nroots)

    list_rdm1 = []

    for i in range(U.shape[1]):
        rdm1 = make_rdm1_eigenvectors(adc, U[:,i], U[:,i])
        list_rdm1.append(rdm1)

    cput0 = log.timer_debug1("completed OPDM calculation", *cput0)
    return list_rdm1

def make_rdm1_eigenvectors(adc, L, R):
    # Using SQA EA
    L = np.array(L).ravel()
    R = np.array(R).ravel()

    t1_ccee = adc.t2[0][:]
    t2_ce = adc.t1[0][:]

    nocc = adc._nocc
    nvir = adc._nvir
    nmo = nocc + nvir
    n_singles = nvir
    n_doubles = nvir * nvir * nocc

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1  = np.zeros((nmo,nmo))
    kd_oc = np.identity(nocc)

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L2 = L2.reshape(nocc,nvir,nvir)
    R2 = R2.reshape(nocc,nvir,nvir)

    einsum = lib.einsum
    einsum_type = True

############# block- ij
    ### 000 ###
    rdm1[:nocc, :nocc] += 2 * einsum('a,a,IJ->IJ', L1, R1, kd_oc, optimize = einsum_type)

    ### 101 ###
    rdm1[:nocc, :nocc] -= 2 * einsum('Jab,Iab->IJ', L2, R2, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 1 * einsum('Jab,Iba->IJ', L2, R2, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 4 * einsum('iab,iab,IJ->IJ', L2, R2, kd_oc, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 2 * einsum('iab,iba,IJ->IJ', L2, R2, kd_oc, optimize = einsum_type)

    ### 020 ###
    rdm1[:nocc, :nocc] -= 2 * einsum('a,a,Iibc,Jibc->IJ', L1, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('a,a,Iibc,Jicb->IJ', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('a,b,Iiac,Jibc->IJ', L1, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= einsum('a,b,Iiac,Jicb->IJ', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= einsum('a,b,Iica,Jibc->IJ', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('a,b,Iica,Jicb->IJ', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 2 * einsum('a,a,Iibc,Jibc->IJ', L1, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('a,a,Iibc,Jicb->IJ', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('a,b,Iica,Jicb->IJ', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)

############# block- ab
    ### 000 ###
    rdm1[nocc:, nocc:] += einsum('B,A->AB', L1, R1, optimize = einsum_type)

    ### 020 ###
    rdm1[nocc:, nocc:] -= einsum('A,a,ijab,ijBb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 1/2 * einsum('A,a,ijab,jiBb->AB', L1,
                                       R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('a,B,ijab,ijAb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 1/2 * einsum('a,B,ijab,jiAb->AB', L1,
                                       R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 2 * einsum('a,a,ijAb,ijBb->AB', L1, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('a,a,ijAb,jiBb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('a,b,ijBa,ijAb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += einsum('a,b,ijBa,jiAb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 2 * einsum('a,a,ijAb,ijBb->AB', L1, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('a,a,ijAb,jiBb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('a,b,ijBa,ijAb->AB', L1, R1, t1_ccee,
                                 t1_ccee, optimize = einsum_type)

    ### 101 ###
    rdm1[nocc:, nocc:] += 2 * einsum('iAa,iBa->AB', R2, R2, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('iAa,iaB->AB', R2, R2, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('iaA,iBa->AB', L2, R2, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 2 * einsum('iaA,iaB->AB', L2, R2, optimize = einsum_type)

############# block- ai
    # 020 #
    rdm1[nocc:, :nocc] -= einsum('a,A,Ia->AI', L1, R1, t2_ce, optimize = einsum_type)
    rdm1[nocc:, :nocc] += 2 * einsum('a,a,IA->AI', L1, R1, t2_ce, optimize = einsum_type)

    # 011 #
    rdm1[nocc:, :nocc] -= 2 * einsum('A,iab,Iiab->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] += einsum('A,iab,Iiba->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] += 2 * einsum('a,iab,IiAb->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] -= einsum('a,iab,iIAb->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] -= 2 * einsum('a,iba,IiAb->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] += einsum('a,iba,iIAb->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] += 2 * einsum('a,iab,IiAb->AI', L1, R2, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, :nocc] -= einsum('a,iab,iIAb->AI', L1, R2, t1_ccee, optimize = einsum_type)

    # 110 #
    rdm1[nocc:, :nocc] -= einsum('IAa,a->AI',L2, R1, optimize = einsum_type)
    rdm1[nocc:, :nocc] += 2 * einsum('IaA,a->AI',L2, R1, optimize = einsum_type)

############# block- ia
    rdm1[:nocc, nocc:] = rdm1[nocc:, :nocc].T

    ####### ADC(3) SPIN ADAPTED EXCITED STATE OPDM WITH SQA ################
    if adc.method == "adc(3)":
        ### Redudant Variables used for names from SQA
        einsum_type = True
        t3_ce = adc.t1[1][:]
        t2_ccee = adc.t2[1][:]

############# block- ij
        # 120 #
        rdm1[:nocc, :nocc] -= 2 * einsum('Jab,a,Ib->IJ', L2, R1, t2_ce, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Jab,b,Ia->IJ', L2, R1, t2_ce, optimize = einsum_type)

        # 021 #
        rdm1[:nocc, :nocc] -= 2 * einsum('a,Iab,Jb->IJ', L1, R2, t2_ce, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('a,Iba,Jb->IJ', L1, R2, t2_ce, optimize = einsum_type)

        # 030 #
        rdm1[:nocc, :nocc] -= 4 * einsum('a,a,Iibc,Jibc->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('a,a,Iibc,Jicb->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 4 * einsum('a,a,Jibc,Iibc->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('a,a,Jibc,Iicb->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('a,b,Iibc,Jiac->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('a,b,Iibc,Jica->IJ', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('a,b,Iicb,Jiac->IJ', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('a,b,Iicb,Jica->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('a,b,Jiac,Iibc->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('a,b,Jiac,Iicb->IJ', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('a,b,Jica,Iibc->IJ', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('a,b,Jica,Iicb->IJ', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)

############# block- ab
        # 120 #
        rdm1[nocc:, nocc:] -= einsum('iAa,a,iB->AB', L2, R1, t2_ce, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('iaA,a,iB->AB', L2, R1, t2_ce, optimize = einsum_type)

        # 021 #
        rdm1[nocc:, nocc:] += 2 * einsum('a,iaB,iA->AB', L1, R2, t2_ce, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('a,iBa,iA->AB', L1, R2, t2_ce, optimize = einsum_type)

        # 030 #
        rdm1[nocc:, nocc:] -= einsum('B,a,ijAb,ijab->AB', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('B,a,ijAb,jiab->AB',
                                           L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('B,a,ijab,ijAb->AB', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('B,a,ijab,jiAb->AB',
                                           L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('a,A,ijBb,ijab->AB', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('a,A,ijBb,jiab->AB',
                                           L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('a,A,ijab,ijBb->AB', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('a,A,ijab,jiBb->AB',
                                           L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 4 * einsum('a,a,ijAb,ijBb->AB', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('a,a,ijAb,jiBb->AB', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 4 * einsum('a,a,ijBb,ijAb->AB', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('a,a,ijBb,jiAb->AB', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('a,b,ijAa,ijBb->AB', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('a,b,ijAa,jiBb->AB', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('a,b,ijBb,ijAa->AB', L1,
                                         R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('a,b,ijBb,jiAa->AB', L1, R1,
                                     t1_ccee, t2_ccee, optimize = einsum_type)

############# block- ia
        # 120 #
        rdm1[:nocc, nocc:] -= 2 * einsum('iab,A,Iiab->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('iab,A,Iiba->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 4 * einsum('iab,a,IiAb->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('iab,a,iIAb->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('iab,b,IiAa->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('iab,b,iIAa->IA', L2, R1, t2_ccee, optimize = einsum_type)

        # 021 #
        rdm1[:nocc, nocc:] -= einsum('a,Iab,ijbc,ijAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,Iab,ijbc,jiAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,Ibc,ijAa,jibc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,iaA,ijbc,Ijbc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iaA,ijbc,Ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('a,iab,ijbc,IjAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijbc,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijcb,IjAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iab,ijcb,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,ibA,Ijca,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibc,IjAa,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)

        rdm1[:nocc, nocc:] -= einsum('a,Iab,ijbc,ijAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,Iab,ijbc,jiAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('a,Iba,ijbc,ijAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,Iba,ijbc,jiAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,Ibc,ijAa,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,Ibc,ijAa,jibc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('a,iAa,ijbc,Ijbc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iAa,ijbc,Ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,iAb,Ijac,ijbc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iAb,Ijac,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iAb,Ijca,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iAb,Ijca,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,iaA,ijbc,Ijbc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iaA,ijbc,Ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('a,iab,ijbc,IjAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijbc,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijcb,IjAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iab,ijcb,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('a,ibA,Ijac,ijbc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibA,Ijac,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibA,Ijca,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,ibA,Ijca,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,iba,ijbc,IjAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iba,ijbc,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iba,ijcb,IjAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iba,ijcb,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,ibc,IjAa,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibc,IjAa,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibc,jIAa,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,ibc,jIAa,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)

        rdm1[:nocc, nocc:] += einsum('a,iab,ijbc,IjAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijbc,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijcb,IjAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,iba,ijbc,IjAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iba,ijbc,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iba,ijcb,IjAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,ibc,IjAa,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibc,IjAa,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)

        rdm1[:nocc, nocc:] -= einsum('a,iAb,Ijac,ijbc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iAb,Ijac,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,iAb,Ijca,ijbc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('a,iab,ijbc,IjAc->IA', L1, R2,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijbc,jIAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,iab,ijcb,IjAc->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('a,ibc,IjAa,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,ibc,jIAa,ijcb->IA',
                                           L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)

        # 030 #
        rdm1[:nocc, nocc:] -= einsum('A,a,Ia->IA', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 2 * einsum('a,a,IA->IA', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('A,a,Iiab,ib->IA', L1, R1, t1_ccee,
                                     t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('A,a,Iiba,ib->IA', L1,
                                           R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 2 * einsum('a,a,IiAb,ib->IA', L1, R1,
                                         t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,a,iIAb,ib->IA', L1, R1, t1_ccee,
                                     t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('a,b,IiAb,ia->IA', L1, R1, t1_ccee,
                                     t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('a,b,iIAb,ia->IA', L1,
                                           R1, t1_ccee, t2_ce, optimize = einsum_type)

############# block- ai
        # 120 #
        rdm1[nocc:, :nocc] -= einsum('Iab,a,ijbc,ijAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('Iab,a,ijbc,jiAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('Iab,c,ijab,jiAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('iaA,a,ijbc,Ijbc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iaA,a,ijbc,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iaA,b,ijca,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('iab,a,ijbc,IjAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijbc,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijcb,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,a,ijcb,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,c,ijba,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)

        rdm1[nocc:, :nocc] -= einsum('Iab,a,ijbc,ijAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('Iab,a,ijbc,jiAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('Iab,b,ijac,ijAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('Iab,b,ijac,jiAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('Iab,c,ijab,ijAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('Iab,c,ijab,jiAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('iAa,a,ijbc,Ijbc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iAa,a,ijbc,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('iAa,b,ijac,Ijbc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iAa,b,ijac,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iAa,b,ijca,Ijbc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iAa,b,ijca,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('iaA,a,ijbc,Ijbc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iaA,a,ijbc,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('iaA,b,ijac,Ijbc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iaA,b,ijac,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iaA,b,ijca,Ijbc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iaA,b,ijca,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('iab,a,ijbc,IjAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijbc,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijcb,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,a,ijcb,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('iab,b,ijac,IjAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,b,ijac,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,b,ijca,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,b,ijca,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,c,ijab,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,c,ijab,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,c,ijba,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,c,ijba,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)

        rdm1[nocc:, :nocc] += einsum('iab,a,ijbc,IjAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijbc,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijcb,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('iab,b,ijac,IjAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,b,ijac,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,b,ijca,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,c,ijab,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,c,ijba,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)

        rdm1[nocc:, :nocc] -= einsum('iAa,b,ijac,Ijbc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iAa,b,ijac,Ijcb->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iAa,b,ijca,Ijbc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('iab,a,ijbc,IjAc->AI', L2, R1,
                                     t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijbc,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,a,ijcb,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 1/2 * einsum('iab,c,ijba,IjAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('iab,c,ijba,jIAc->AI',
                                           L2, R1, t1_ccee, t1_ccee, optimize = einsum_type)

        # 021 #
        rdm1[nocc:, :nocc] -= 2 * einsum('A,iab,Iiab->AI', L1, R2, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('A,iab,Iiba->AI', L1, R2, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 4 * einsum('a,iab,IiAb->AI', L1, R2, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 2 * einsum('a,iab,iIAb->AI', L1, R2, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= 2 * einsum('a,iba,IiAb->AI', L1, R2, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, :nocc] += einsum('a,iba,iIAb->AI', L1, R2, t2_ccee, optimize = einsum_type)

        # 030 #
        rdm1[nocc:, :nocc] -= einsum('a,A,Ia->AI', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 2 * einsum('a,a,IA->AI', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('a,A,Iiab,ib->AI', L1, R1, t1_ccee,
                                     t2_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('a,A,Iiba,ib->AI', L1,
                                           R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 2 * einsum('a,a,IiAb,ib->AI', L1, R1,
                                         t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('a,a,iIAb,ib->AI', L1, R1, t1_ccee,
                                     t2_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] -= einsum('a,b,IiAa,ib->AI', L1, R1, t1_ccee,
                                     t2_ce, optimize = einsum_type)
        rdm1[nocc:, :nocc] += 1/2 * einsum('a,b,iIAa,ib->AI', L1,
                                           R1, t1_ccee, t2_ce, optimize = einsum_type)
    return rdm1


def compute_dyson_mo(myadc):

    X = myadc.X

    if X is None:
        nroots = myadc.U.shape[1]
        P,X = myadc.get_properties(nroots)

    nroots = X.shape[1]
    dyson_mo = np.dot(myadc.mo_coeff,X)

    return dyson_mo


class RADCEA(radc.RADC):
    '''restricted ADC for EA energies and spectroscopic amplitudes

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
            Space size to hold trial vectors for Davidson iterative
            diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcea = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
        'mo_energy', 't1', 'max_space', 't2', 'max_cycle',
        'nmo', 'transform_integrals', 'with_df', 'compute_properties',
        'approx_trans_moments', 'E', 'U', 'P', 'X',
        'evec_print_tol', 'spec_factor_print_tol',
    }

    def __init__(self, adc):
        self.mol = adc.mol
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.imds = adc.imds
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments

        self.evec_print_tol = adc.evec_print_tol
        self.spec_factor_print_tol = adc.spec_factor_print_tol

        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

    kernel = radc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag
    get_trans_moments = get_trans_moments
    get_properties = get_properties

    renormalize_eigenvectors = renormalize_eigenvectors
    analyze = analyze
    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector
    compute_dyson_mo = compute_dyson_mo
    make_rdm1 = make_rdm1

    def get_init_guess(self, nroots=1, diag=None, ascending=True):
        if diag is None :
            diag = self.get_diag()
        idx = None
        if ascending:
            idx = np.argsort(diag)
        else:
            idx = np.argsort(diag)[::-1]
        guess = np.zeros((diag.shape[0], nroots))
        min_shape = min(diag.shape[0], nroots)
        guess[:min_shape,:min_shape] = np.identity(min_shape)
        g = np.zeros((diag.shape[0], nroots))
        g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess


    def gen_matvec(self, imds=None, eris=None):
        if imds is None:
            imds = self.get_imds(eris)
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag


def contract_r_vvvv(myadc,r2,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    r2_vvvv = np.zeros((nocc,nvir,nvir))
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
            del dataset
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc,-1,nvir)
            del vvvv_p
            a += k
    else:
        raise Exception("Unknown vvvv type")

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv
