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
# Author: Terrence Stahl <terrencestahl@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Restricted algebraic diagrammatic construction
'''
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc
from pyscf.adc import radc_ao2mo, radc_amplitudes
from pyscf.adc import dfadc

def get_imds(adc, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    t1 = adc.t1
    t2 = adc.t2

    t1_ccee = t2[0][:]

    t2_ce = t1[0]

    einsum = lib.einsum
    einsum_type = True

    ncore = adc._nocc
    nextern = adc._nvir

    nocc = adc._nocc
    nvir = adc._nvir
    n_singles = ncore * nextern

    e_core = adc.mo_energy[:ncore].copy()
    e_extern = adc.mo_energy[ncore:].copy()

    if eris is None:
        eris = adc.transform_integrals()

    v_ccee = eris.oovv
    v_cece = eris.ovvo
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    occ_list = np.array(range(ncore))
    vir_list = np.array(range(nextern))
    M_ab = np.zeros((ncore*nextern, ncore*nextern))

    ####000#####################
    d_ai_a = adc.mo_energy[ncore:][:,None] - adc.mo_energy[:ncore]
    np.fill_diagonal(M_ab, d_ai_a.transpose().reshape(-1))
    M_ab = M_ab.reshape(ncore,nextern,ncore,nextern).copy()

    ####010#####################
    M_ab -= einsum('ILAD->IDLA', v_ccee, optimize = einsum_type).copy()
    M_ab += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()

    M_ab += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()

    ####020#####################
    M_ab += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += einsum('iIDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += einsum('iLAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab -= einsum('LAai,IiDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab += 1/2 * einsum('LAai,iIDa->IDLA',v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab += 1/2 * einsum('iAaL,IiDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab -= 1/2 * einsum('iAaL,iIDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab -= einsum('LiAa,IDai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab += 1/2 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('iLAa,IDai->IDLA',t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab -= 1/2 * einsum('iLAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab -= einsum('IiDa,LAai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab += 1/2 * einsum('IiDa,iAaL->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab += 1/2 * einsum('iIDa,LAai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##

    M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Iiab,Labi->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Lbai->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Liab,Iabi->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] += einsum('Liab,Ibai->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijAa,iDaj->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] += einsum('ijAa,jDai->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijDa,iAaj->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] += einsum('ijDa,jAai->DA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab[occ_list,:,occ_list,:] += einsum('iAaj,ijDa->DA', v_ceec, t1_ccee, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] -= 1/2 * einsum('iAaj,jiDa->DA', v_ceec, t1_ccee, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] += einsum('ijAa,iDaj->DA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] -= 1/2 * einsum('ijAa,jDai->DA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab[:,vir_list,:,vir_list] += einsum('Iabi,Liab->IL', v_ceec, t1_ccee, optimize = einsum_type)##
    M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('Iabi,Liba->IL', v_ceec, t1_ccee, optimize = einsum_type)##
    M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Labi->IL', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('Iiab,Lbai->IL', t1_ccee, v_ceec, optimize = einsum_type) ##

    M_ab += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab -= einsum('LiAa,IDai->IDLA',t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('iLAa,IDai->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#

    if (adc.method == "adc(3)"):
        t2_ccee = t2[1][:]

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nocc,chnk_size):
                v_ceee = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir,nvir,nvir)
                M_ab[:,:,a:b,:] += einsum('Ia,LADa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
                M_ab[:,:,a:b,:] -= einsum('Ia,LaDA->IDLA', t2_ce, v_ceee, optimize = einsum_type)
                M_ab[a:b,:,:,:] += einsum('La,IDAa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
                M_ab[a:b,:,:,:] -= einsum('La,IaAD->IDLA', t2_ce, v_ceee, optimize = einsum_type)
                M_ab[occ_list,:,occ_list,:] -= einsum('ia,iADa->DA', t2_ce[a:b,:], v_ceee, optimize = einsum_type)
                M_ab[occ_list,:,occ_list,:] -= einsum('ia,iDAa->DA', t2_ce[a:b,:], v_ceee, optimize = einsum_type)
                M_ab[occ_list,:,occ_list,:] += 2 * einsum('ia,iaAD->DA', t2_ce[a:b,:], v_ceee, optimize = einsum_type)
                M_ab[occ_list,:,occ_list,:] += 2 * einsum('ia,iaDA->DA', t2_ce[a:b,:], v_ceee, optimize = einsum_type)
                M_ab[:,:,a:b,:] += einsum('Ia,LADa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
                M_ab[a:b,:,:,:] += einsum('La,IDAa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
                del v_ceee
        else:
            v_ceee = radc_ao2mo.unpack_eri_1(eris.ovvv, nextern)
            M_ab += einsum('Ia,LADa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab -= einsum('Ia,LaDA->IDLA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab += einsum('La,IDAa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab -= einsum('La,IaAD->IDLA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab[occ_list,:,occ_list,:] -= einsum('ia,iADa->DA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab[occ_list,:,occ_list,:] -= einsum('ia,iDAa->DA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab[occ_list,:,occ_list,:] += 2 * einsum('ia,iaAD->DA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab[occ_list,:,occ_list,:] += 2 * einsum('ia,iaDA->DA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab += einsum('Ia,LADa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
            M_ab += einsum('La,IDAa->IDLA', t2_ce, v_ceee, optimize = einsum_type)

        if isinstance(eris.vvvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nvir,chnk_size):
                v_eeee = dfadc.get_vvvv_df(adc, eris.Lvv, a, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = v_eeee.shape[0]
                M_ab[:,:,:,a:b] -= 2 * einsum('AaDb,Iiac,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] += einsum('AaDb,Iiac,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] += einsum('AaDb,Iica,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] -= 2 * einsum('AaDb,Iica,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] += 2 * einsum('AbaD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] -= einsum('AbaD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] -= einsum('AbaD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] += einsum('AbaD,Iicb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] += 2 * einsum('Abac,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] -= einsum('Abac,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] -= einsum('Abac,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] += einsum('Abac,iIDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:b,:,:] += 2 * einsum('Dbac,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,a:b,:,:] -= einsum('Dbac,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:b,:,:] -= einsum('Dbac,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:b,:,:] += einsum('Dbac,iLAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,vir_list,:,vir_list] -= 2 * einsum('acbd,Iiac,Libd->IL', v_eeee, t1_ccee[:,:,a:b,:], t1_ccee,
                optimize = einsum_type)
                M_ab[:,vir_list,:,vir_list] += einsum('acbd,Iiac,Lidb->IL', v_eeee, t1_ccee[:,:,a:b,:], t1_ccee,
                optimize = einsum_type)
                M_ab[occ_list,:,occ_list,a:b] += 4 * einsum('AaDb,ijac,ijbc->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:b] -= 2 * einsum('AaDb,ijac,jibc->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:b] -= 2 * einsum('AbaD,ijbc,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:b] += einsum('AbaD,ijbc,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:b] -= 2 * einsum('Abac,ijDb,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:b] += einsum('Abac,ijDb,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,a:b,occ_list,:] -= 2 * einsum('Dbac,ijAb,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,a:b,occ_list,:] += einsum('Dbac,ijAb,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] += 2 * einsum('AbaD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] -= einsum('AbaD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] -= einsum('AbaD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] += 2 * einsum('Abac,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:b] -= einsum('Abac,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:b] -= einsum('Abac,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:b,:,:] += 2 * einsum('Dbac,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,a:b,:,:] -= einsum('Dbac,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:b,:,:] -= einsum('Dbac,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                del v_eeee
        elif isinstance(eris.vvvv, list):
            a = 0
            for dataset in eris.vvvv:
                k = dataset.shape[0]
                v_eeee = dataset[:].reshape(-1,nvir,nvir,nvir)
                M_ab[:,:,:,a:a+k] -= 2 * einsum('AaDb,Iiac,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] += einsum('AaDb,Iiac,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] += einsum('AaDb,Iica,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] -= 2 * einsum('AaDb,Iica,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] += 2 * einsum('AbaD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('AbaD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('AbaD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] += einsum('AbaD,Iicb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] += 2 * einsum('Abac,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('Abac,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('Abac,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] += einsum('Abac,iIDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:a+k,:,:] += 2 * einsum('Dbac,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,a:a+k,:,:] -= einsum('Dbac,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:a+k,:,:] -= einsum('Dbac,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:a+k,:,:] += einsum('Dbac,iLAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,vir_list,:,vir_list] -= 2 * einsum('acbd,Iiac,Libd->IL', v_eeee, t1_ccee[:,:,a:a+k,:], t1_ccee,
                optimize = einsum_type)
                M_ab[:,vir_list,:,vir_list] += einsum('acbd,Iiac,Lidb->IL', v_eeee, t1_ccee[:,:,a:a+k,:], t1_ccee,
                optimize = einsum_type)
                M_ab[occ_list,:,occ_list,a:a+k] += 4 * einsum('AaDb,ijac,ijbc->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:a+k] -= 2 * einsum('AaDb,ijac,jibc->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:a+k] -= 2 * einsum('AbaD,ijbc,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:a+k] += einsum('AbaD,ijbc,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:a+k] -= 2 * einsum('Abac,ijDb,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,:,occ_list,a:a+k] += einsum('Abac,ijDb,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,a:a+k,occ_list,:] -= 2 * einsum('Dbac,ijAb,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[occ_list,a:a+k,occ_list,:] += einsum('Dbac,ijAb,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] += 2 * einsum('AbaD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('AbaD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('AbaD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] += 2 * einsum('Abac,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('Abac,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,:,:,a:a+k] -= einsum('Abac,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:a+k,:,:] += 2 * einsum('Dbac,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize =
                einsum_type)
                M_ab[:,a:a+k,:,:] -= einsum('Dbac,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                M_ab[:,a:a+k,:,:] -= einsum('Dbac,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
                del v_eeee
                a += k
            del a
            del k
        else:
            v_eeee = eris.vvvv.reshape(nextern, nextern, nextern, nextern)
            M_ab -= 2 * einsum('AaDb,Iiac,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += einsum('AaDb,Iiac,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += einsum('AaDb,Iica,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= 2 * einsum('AaDb,Iica,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += 2 * einsum('AbaD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('AbaD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('AbaD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += einsum('AbaD,Iicb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += 2 * einsum('Abac,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Abac,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Abac,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += einsum('Abac,iIDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += 2 * einsum('Dbac,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Dbac,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Dbac,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += einsum('Dbac,iLAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab[:,vir_list,:,vir_list] -= 2 * einsum('acbd,Iiac,Libd->IL', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[:,vir_list,:,vir_list] += einsum('acbd,Iiac,Lidb->IL', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] += 4 * einsum('AaDb,ijac,ijbc->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] -= 2 * einsum('AaDb,ijac,jibc->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] -= 2 * einsum('AbaD,ijbc,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] += einsum('AbaD,ijbc,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] -= 2 * einsum('Abac,ijDb,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] += einsum('Abac,ijDb,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] -= 2 * einsum('Dbac,ijAb,ijac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab[occ_list,:,occ_list,:] += einsum('Dbac,ijAb,jiac->DA', v_eeee, t1_ccee, t1_ccee, optimize =
            einsum_type)
            M_ab += 2 * einsum('AbaD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('AbaD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('AbaD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += 2 * einsum('Abac,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Abac,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Abac,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab += 2 * einsum('Dbac,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Dbac,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
            M_ab -= einsum('Dbac,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)

        M_ab += 2 * einsum('IiDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)

        M_ab -= einsum('IiDa,iAaL->IDLA', t2_ccee, v_cece, optimize = einsum_type)

        M_ab += 2 * einsum('LiAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab -= einsum('LiAa,iDaI->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab += einsum('iA,iDIL->IDLA', t2_ce, v_cecc, optimize = einsum_type)
        M_ab -= einsum('iA,IDiL->IDLA', t2_ce, v_cecc, optimize = einsum_type)
        M_ab += einsum('iD,iALI->IDLA', t2_ce, v_cecc, optimize = einsum_type)
        M_ab -= einsum('iD,LAiI->IDLA', t2_ce, v_cecc, optimize = einsum_type)
        M_ab -= einsum('iIDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab += einsum('iIDa,iAaL->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab -= einsum('iLAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab += einsum('iLAa,iDaI->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab += einsum('A,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('A,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('A,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('A,iIDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += einsum('D,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('D,iLAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('I,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('I,iLAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('L,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('L,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('L,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('L,iIDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('IDai,LiAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('a,iIDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('a,iLAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('IDai,LiAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('IDai,iLAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2* einsum('LAai,IiDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2* einsum('LAai,IiDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('LAai,iIDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('iDaI,LiAa->IDLA',v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('i,iIDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('iDaI,iLAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 1/2*einsum('iAaL,IiDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('i,iLAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('iAaL,iIDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)

        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Iiab,Labi->IL', t2_ccee, v_cece, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Lbai->IL', t2_ccee, v_cece, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Liab,Iabi->IL', t2_ccee, v_cece, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('Liab,Ibai->IL',  t2_ccee, v_cece, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('ia,iaIL->IL', t2_ce, v_cecc, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('ia,iaLI->IL', t2_ce, v_cecc, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('ia,LaiI->IL', t2_ce, v_cecc, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('ia,IaiL->IL', t2_ce, v_cecc, optimize = einsum_type)

        M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijAa,iDaj->DA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += einsum('ijAa,jDai->DA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijDa,iAaj->DA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += einsum('ijDa,jAai->DA',t2_ccee, v_cece, optimize = einsum_type)

        M_ab -= einsum('IDaL,ijab,ijAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('IDaL,ijab,jiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('IDai,ijab,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IDai,ijab,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IDai,ijba,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('IDai,ijba,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ILAa,ijab,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('ILAa,ijab,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ILij,ikAa,jkDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ILij,ikAa,kjDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ILij,kiAa,jkDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ILij,kiAa,kjDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iabi,ijDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Iabi,ijDb,jLAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Iabi,jiDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('Iabi,jiDb,jLAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('IiAD,ijab,Ljab->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('IiAD,ijab,Ljba->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('IiAa,Ljab,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('IiAa,Ljab,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('IiAa,Ljba,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IiAa,Ljba,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Iiab,ijDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iiab,ijDa,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iiab,jiDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Iiab,jiDa,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('IijL,jkAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IijL,jkAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IijL,kjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('IijL,kjAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('Iijk,LjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Iijk,LjAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Iijk,jLAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iijk,jLAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LADi,ijab,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('LADi,ijab,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LAaI,ijab,ijDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('LAaI,ijab,jiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('LAai,ijab,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LAai,ijab,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LAai,ijba,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('LAai,ijba,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('LIDa,ijab,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('LIDa,ijab,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Labi,ijAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Labi,ijAb,jIDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Labi,jiAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('Labi,jiAb,jIDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('LiDa,Ijab,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('LiDa,Ijab,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('LiDa,Ijba,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LiDa,Ijba,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Liab,ijAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Liab,ijAa,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Liab,jiAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Liab,jiAa,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('Lijk,IjDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Lijk,IjDa,kiAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Lijk,jIDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Lijk,jIDa,kiAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iADI,ijab,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iADI,ijab,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iADj,Ljab,Iiab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iADj,Ljab,Iiba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('iAaI,ijDb,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iAaI,ijDb,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iAaI,jiDb,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iAaI,jiDb,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iAaj,Ljab,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iAaj,Ljab,iIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iAaj,Ljba,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('iAaj,Ljba,iIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('iDaL,ijAb,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iDaL,ijAb,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iDaL,jiAb,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iDaL,jiAb,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iDaj,Ijab,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iDaj,Ijab,iLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iDaj,Ijba,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('iDaj,Ijba,iLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iIDa,ijab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iIDa,ijab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iIDa,ijba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('iIDa,ijba,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iLAD,ijab,Ijab->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('iLAD,ijab,Ijba->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iLAa,ijab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iLAa,ijab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iLAa,ijba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('iLAa,ijba,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 4 * einsum('iabj,LiAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iabj,LiAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iabj,iLAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iabj,iLAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('ijAD,Liab,Ijab->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('ijAD,Liab,Ijba->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ijAa,Liab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijAa,Liab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijAa,Liba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('ijAa,Liba,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ijDa,Iiab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijDa,Iiab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijDa,Iiba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('ijDa,Iiba,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ijab,LjAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijab,LjAa,iIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijab,jLAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('ijab,jLAa,iIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)

        M_ab[occ_list,:,occ_list,:] -= einsum('A,ijAa,ijDa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 1/2 * einsum('A,ijAa,jiDa->DA', e_extern, t1_ccee, t2_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('A,ijDa,ijAa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 1/2 * einsum('A,ijDa,jiAa->DA', e_extern, t1_ccee, t2_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('D,ijAa,ijDa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 1/2 * einsum('D,ijAa,jiDa->DA', e_extern, t1_ccee, t2_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('D,ijDa,ijAa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 1/2 * einsum('D,ijDa,jiAa->DA', e_extern, t1_ccee, t2_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('I,Iiab,Liab->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('I,Iiab,Liba->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('I,Liab,Iiab->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('I,Liab,Iiba->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('L,Iiab,Liab->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('L,Iiab,Liba->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('L,Liab,Iiab->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('L,Liab,Iiba->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('a,Iiab,Liab->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('a,Iiab,Liba->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('a,Iiba,Liab->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('a,Iiba,Liba->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('a,Liab,Iiab->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('a,Liab,Iiba->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('a,Liba,Iiab->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('a,Liba,Iiba->IL', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 2 * einsum('a,ijAa,ijDa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += einsum('a,ijAa,jiDa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 2 * einsum('a,ijDa,ijAa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += einsum('a,ijDa,jiAa->DA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('i,Iiab,Liab->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('i,Iiab,Liba->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('i,Liab,Iiab->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('i,Liab,Iiba->IL', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('i,ijAa,ijDa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('i,ijAa,jiDa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('i,ijDa,ijAa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('i,ijDa,jiAa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('i,jiAa,ijDa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('i,jiAa,jiDa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('i,jiDa,ijAa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('i,jiDa,jiAa->DA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)

        M_ab[:,vir_list,:,vir_list] -= 4 * einsum('ILab,ijac,ijbc->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('ILab,ijac,jibc->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 4 * einsum('ILij,ikab,jkab->IL', v_cccc, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('ILij,ikab,jkba->IL', v_cccc, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('IabL,ijbc,ijac->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('IabL,ijbc,jiac->IL', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 4 * einsum('Iabi,ijbc,Ljac->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Iabi,ijbc,Ljca->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Iabi,jibc,Ljac->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('Iabi,jibc,Ljca->IL', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Iiab,ijac,Ljbc->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('Iiab,ijac,Ljcb->IL', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('Iiab,jiac,Ljbc->IL', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Iiab,jiac,Ljcb->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('IijL,jkab,ikab->IL', v_cccc, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('IijL,jkab,ikba->IL', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Iijk,Ljab,ikab->IL', v_cccc, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('Iijk,Ljab,ikba->IL', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 4 * einsum('Labi,ijbc,Ijac->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Labi,ijbc,Ijca->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Labi,jibc,Ijac->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('Labi,jibc,Ijca->IL', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Liab,ijac,Ijbc->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('Liab,ijac,Ijcb->IL', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('Liab,jiac,Ijbc->IL', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('Liab,jiac,Ijcb->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Lijk,Ijab,ikab->IL', v_cccc, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += einsum('Lijk,Ijab,ikba->IL', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('iabj,Iiac,Ljbc->IL', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('iabj,Iiac,Ljcb->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('iabj,Iica,Ljbc->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= 4 * einsum('iabj,Iica,Ljcb->IL', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('ijab,Iibc,Ljac->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('ijab,Iibc,Ljca->IL', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] -= einsum('ijab,Iicb,Ljac->IL', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[:,vir_list,:,vir_list] += 2 * einsum('ijab,Iicb,Ljca->IL', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)

        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iADj,jkab,ikab->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('iADj,jkab,ikba->DA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 4 * einsum('iAaj,jkab,ikDb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iAaj,jkab,kiDb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iAaj,jkba,ikDb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('iAaj,jkba,kiDb->DA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 4 * einsum('iDaj,jkab,ikAb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iDaj,jkab,kiAb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iDaj,jkba,ikAb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('iDaj,jkba,kiAb->DA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('iabj,ikAa,jkDb->DA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iabj,ikAa,kjDb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('iabj,kiAa,jkDb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 4 * einsum('iabj,kiAa,kjDb->DA', v_ceec, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 4 * einsum('ijAD,ikab,jkab->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijAD,ikab,jkba->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijAa,ikab,jkDb->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('ijAa,ikab,kjDb->DA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('ijAa,ikba,jkDb->DA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijAa,ikba,kjDb->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijDa,ikab,jkAb->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('ijDa,ikab,kjAb->DA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('ijDa,ikba,jkAb->DA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijDa,ikba,kjAb->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijab,jkAa,ikDb->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('ijab,jkAa,kiDb->DA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] -= einsum('ijab,kjAa,ikDb->DA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab[occ_list,:,occ_list,:] += 2 * einsum('ijab,kjAa,kiDb->DA', v_ccee, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijkl,ikAa,jlDa->DA', v_cccc, t1_ccee, t1_ccee, optimize =
        einsum_type)
        M_ab[occ_list,:,occ_list,:] += einsum('ijkl,ikAa,ljDa->DA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)

        M_ab += 2 * einsum('IiDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab -= einsum('IiDa,iAaL->IDLA', t2_ccee, v_cece, optimize = einsum_type)

        M_ab += 2 * einsum('LiAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)

        M_ab -= einsum('LiAa,iDaI->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab -= einsum('iA,IDiL->IDLA', t2_ce, v_cecc, optimize = einsum_type)
        M_ab -= einsum('iD,LAiI->IDLA', t2_ce, v_cecc, optimize = einsum_type)
        M_ab -= einsum('iIDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab -= einsum('iLAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_ab += einsum('A,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('A,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('A,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('IDai,iLAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += einsum('D,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('IDai,LiAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('IDai,LiAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('I,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iDaI,LiAa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('L,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('L,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('LAai,IiDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab -= 1/2*einsum('LAai,IiDa->IDLA',v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('L,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iAaL,IiDa->IDLA', v_ceec, t2_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('a,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('a,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('a,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('a,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('i,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += einsum('i,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('i,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += einsum('i,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab += einsum('i,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_ab -= einsum('IDaL,ijab,ijAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('IDaL,ijab,jiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('IDai,ijab,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IDai,ijab,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IDai,ijba,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('IDai,ijba,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iabi,ijDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Iabi,jiDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('IiAa,Ljab,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('IiAa,Ljab,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('IiAa,Ljba,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Iiab,ijDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iiab,ijDa,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Iiab,jiDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('IijL,jkAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IijL,jkAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('IijL,kjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('Iijk,LjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Iijk,LjAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Iijk,jLAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LADi,ijab,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('LADi,ijab,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LAaI,ijab,ijDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('LAaI,ijab,jiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('LAai,ijab,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LAai,ijab,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('LAai,ijba,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('LAai,ijba,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Labi,ijAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Labi,jiAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('LiDa,Ijab,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('LiDa,Ijab,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('LiDa,Ijba,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('Liab,ijAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Liab,ijAa,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('Liab,jiAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 2 * einsum('Lijk,IjDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Lijk,IjDa,kiAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('Lijk,jIDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iADI,ijab,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iADI,ijab,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iADj,Ljab,Iiba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iAaI,jiDb,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iAaj,Ljab,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iAaj,Ljba,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iDaL,jiAb,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iDaj,Ijab,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iDaj,Ijba,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iIDa,ijab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iIDa,ijab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iIDa,ijba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= einsum('iLAa,ijab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iLAa,ijab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 1/2 * einsum('iLAa,ijba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += 4 * einsum('iabj,LiAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iabj,LiAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('iabj,iLAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('iabj,iLAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ijAa,Liab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijAa,Liab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijAa,Liba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ijDa,Iiab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijDa,Iiab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijDa,Iiba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab -= 2 * einsum('ijab,LjAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijab,LjAa,iIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_ab += einsum('ijab,jLAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)

    M_ab = M_ab.reshape(n_singles, n_singles)

    return M_ab


def get_diag(adc,M_ab=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ = adc.get_imds()

    M_ = M_ab

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles
    diag = np.zeros(dim)

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None]+e_occ
    d_ab = e_vir[:,None]+e_vir

    D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))
    diag[s2:f2] = D_ijab.reshape(-1)

    diag[s1:f1] = np.diagonal(M_)

    return diag


def matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_  = adc.get_imds()

    M_ = M_ab

    if eris is None:
        eris = adc.transform_integrals()

    einsum = lib.einsum
    einsum_type = True
    t1 = adc.t1
    t2 = adc.t2
    t1_ccee = t2[0][:]
    t2_ce = t1[0]
    v_ccce = eris.ovoo

    v_ccee = eris.oovv
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    d_ij = e_occ[:,None]+e_occ
    d_ab = e_vir[:,None]+e_vir

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    e_core = adc.mo_energy[:nocc].copy()
    e_extern = adc.mo_energy[nocc:].copy()

    #Calculate sigma vector

    def sigma_(r):

        r1 = r[s1:f1]

        Y = r1.reshape(nocc, nvir)

        r2 = r[s2:f2].reshape(nocc,nocc,nvir,nvir).copy()

        s = np.zeros(dim)

        s[s1:f1] = lib.einsum('ab,b->a',M_,r1, optimize = True)

        D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))
        s[s2:f2] = (D_ijab.reshape(-1))*r[s2:f2]
        del D_ijab

        if isinstance(eris.ovvv, type(None)):
            M_11Y0 = np.zeros((nocc,nocc,nvir,nvir))
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nocc,chnk_size):
                v_ceee = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir,nvir,nvir)
                M_11Y0[:,a:b,:,:] += einsum('Ia,JDaC->IJCD', Y, v_ceee, optimize = einsum_type)
                M_11Y0[a:b,:,:,:] += einsum('Ja,ICaD->IJCD', Y, v_ceee, optimize = einsum_type)

                s[s1:f1] += -einsum('Iiab,iabD->ID', r2[:,a:b,:,:], v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += 2*einsum('Iiab,ibDa->ID', r2[:,a:b,:,:], v_ceee, optimize = einsum_type).reshape(-1)
                del v_ceee
            s[s2:f2] += M_11Y0.reshape(-1)
            del M_11Y0
        else:
            v_ceee = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            M_11Y0 = einsum('Ia,JDaC->IJCD', Y, v_ceee, optimize = einsum_type)
            M_11Y0 += einsum('Ja,ICaD->IJCD', Y, v_ceee, optimize = einsum_type)
            s[s2:f2] += M_11Y0.reshape(-1)

            M_01Y1 = -einsum('Iiab,iabD->ID', r2, v_ceee, optimize = einsum_type)
            M_01Y1 += 2*einsum('Iiab,ibDa->ID', r2, v_ceee, optimize = einsum_type)
            s[s1:f1] += M_01Y1.reshape(-1)
            del M_11Y0
            del M_01Y1

        s[s2:f2] -= einsum('iC,JDIi->IJCD', Y, v_cecc, optimize = einsum_type).reshape(-1)
        s[s2:f2] -= einsum('iD,ICJi->IJCD', Y, v_cecc, optimize = einsum_type).reshape(-1)

        s[s1:f1] -= 2*einsum('ijDa,jaiI->ID', r2, v_cecc, optimize = einsum_type).reshape(-1)
        s[s1:f1] += einsum('ijDa,iajI->ID', r2, v_cecc, optimize = einsum_type).reshape(-1)

        if (adc.method == "adc(2)-x") or (adc.method == "adc(3)"):
            Y = r2

            if isinstance(eris.vvvv, type(None)):
                s[s2:f2] += radc_amplitudes.contract_ladder(adc,Y,eris.Lvv).reshape(-1)
            elif isinstance(eris.vvvv, list):
                s[s2:f2] += radc_amplitudes.contract_ladder(adc,Y,eris.vvvv).reshape(-1)
            else:
                temp = Y.reshape(nocc*nocc,nvir*nvir)
                s[s2:f2] += np.dot(temp, eris.vvvv).reshape(-1)
                del temp

            s[s2:f2] += 2 * einsum('IiCa,JDai->IJCD', Y, v_ceec, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= einsum('IiCa,iJDa->IJCD', Y, v_ccee, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= einsum('IiaC,JDai->IJCD', Y, v_ceec, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= einsum('IiaD,iJCa->IJCD', Y, v_ccee, optimize = einsum_type).reshape(-1)
            s[s2:f2] += 2 * einsum('JiDa,ICai->IJCD', Y, v_ceec, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= einsum('JiDa,iICa->IJCD', Y, v_ccee, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= einsum('JiaC,iIDa->IJCD', Y, v_ccee, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= einsum('JiaD,ICai->IJCD', Y, v_ceec, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('ijCD,IiJj->IJCD', Y, v_cccc, optimize = einsum_type).reshape(-1)

        if (adc.method == "adc(3)"):
            s[s1:f1] += 2 * einsum('IiDa,a,ia->ID', Y, e_extern, t2_ce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= 2 * einsum('IiDa,i,ia->ID', Y, e_core, t2_ce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('IiaD,a,ia->ID', Y, e_extern, t2_ce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += einsum('IiaD,i,ia->ID', Y, e_core, t2_ce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= 4 * einsum('IiDa,jkab,kbji->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('IiDa,jkab,jbki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('IiaD,jkab,kbji->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('IiaD,jkab,jbki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('Iiab,jkab,kDji->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('Iiab,jkab,jDki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('ijDa,ikab,kbIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('ijDa,ikab,Ibkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('ijDa,ikba,kbIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('ijDa,ikba,Ibkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= 4 * einsum('ijDa,jkab,kbIi->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('ijDa,jkab,Ibki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('ijDa,jkba,kbIi->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('ijDa,jkba,Ibki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += 2 * einsum('ijab,ikab,kDIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= 3 * einsum('ijab,ikab,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('ijab,ikba,kDIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += einsum('ijab,ikba,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] -= einsum('ijab,ikab,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s1:f1] += einsum('ijab,ikba,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)

            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
                temp = np.zeros((nocc,nvir))
                for a,b in lib.prange(0,nocc,chnk_size):
                    v_ceee = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir,nvir,nvir)
                    temp  += -2 * einsum('IiDa,ijbc,jbac->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += 4 * einsum('IiDa,ijbc,jcab->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += einsum('IiaD,ijbc,jbac->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= 2 * einsum('IiaD,ijbc,jcab->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += einsum('Iiab,ijac,jDbc->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= 2 * einsum('Iiab,ijac,jcbD->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= 2 * einsum('Iiab,ijbc,jDac->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += 4 * einsum('Iiab,ijbc,jcaD->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= 2 * einsum('Iiab,ijca,jDbc->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += einsum('Iiab,ijca,jcbD->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += einsum('Iiab,ijcb,jDac->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= 2 * einsum('Iiab,ijcb,jcaD->ID', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp[a:b,:] -= 2 * einsum('ijDa,ijbc,Ibac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] += einsum('ijDa,ijbc,Icab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] += 3 * einsum('ijab,ijac,IDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] -= 2 * einsum('ijab,ijac,IcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] -= einsum('ijab,ijbc,IDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] += einsum('ijab,ijbc,IcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] += einsum('ijab,ijac,IDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                    temp[a:b,:] -= einsum('ijab,ijbc,IDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
                s[s1:f1] +=temp.reshape(-1)
                del temp
                del v_ceee
            else:
                s[s1:f1] -= 2 * einsum('IiDa,ijbc,jbac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += 4 * einsum('IiDa,ijbc,jcab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('IiaD,ijbc,jbac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('IiaD,ijbc,jcab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('Iiab,ijac,jDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('Iiab,ijac,jcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('Iiab,ijbc,jDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += 4 * einsum('Iiab,ijbc,jcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('Iiab,ijca,jDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('Iiab,ijca,jcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('Iiab,ijcb,jDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('Iiab,ijcb,jcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('ijDa,ijbc,Ibac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('ijDa,ijbc,Icab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += 3 * einsum('ijab,ijac,IDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= 2 * einsum('ijab,ijac,IcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= einsum('ijab,ijbc,IDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('ijab,ijbc,IcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += einsum('ijab,ijac,IDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] -= einsum('ijab,ijbc,IDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type).reshape(-1)

            Y = r1.reshape(nocc, nvir)

            int_1 = einsum('ijAb,LA->ijLb', t1_ccee, Y, optimize = einsum_type)
            int_2 = einsum('ijDa,ijLb->LbDa', t1_ccee, int_1, optimize = einsum_type)
            s[s1:f1] += 2 * einsum('ILab,LbDa->ID', v_ccee, int_2, optimize = einsum_type).reshape(-1)
            del int_1
            del int_2

            int_1 = einsum('jiAb,LA->jiLb', t1_ccee, Y, optimize = einsum_type)
            int_2 = einsum('ijDa,jiLb->LbDa',t1_ccee, int_1, optimize = einsum_type)
            s[s1:f1] -= einsum('ILab,LbDa->ID', v_ccee, int_2, optimize = einsum_type).reshape(-1)
            del int_1
            del int_2

            int_1 = einsum('ijAa,LA->ijLa', t1_ccee, Y, optimize = einsum_type)
            int_2 = einsum('ijDb,ijLa->LaDb', t1_ccee, int_1, optimize = einsum_type)
            s[s1:f1] -= einsum('IabL,LaDb->ID', v_ceec, int_2, optimize = einsum_type).reshape(-1)
            del int_1
            del int_2

            int_1 = einsum('jiAa,LA->jiLa', t1_ccee, Y, optimize = einsum_type)
            int_2 = einsum('ijDb,jiLa->LaDb',t1_ccee, int_1, optimize = einsum_type)
            s[s1:f1] += einsum('IabL,LaDb->ID', v_ceec, int_2, optimize = einsum_type).reshape(-1)
            del int_1
            del int_2

            int_1 = einsum('jiAa,LA->jiLa', t1_ccee, Y, optimize = einsum_type)
            int_2 = einsum('ijDb,jiLa->LaDb', t1_ccee, int_1, optimize = einsum_type)
            s[s1:f1] += einsum('IabL,LaDb->ID', v_ceec, int_2, optimize = einsum_type).reshape(-1)
            del int_1
            del int_2

            s[s2:f2] += einsum('Ia,ijCD,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('Ja,ijCD,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= 2 * einsum('iC,JjDa,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('iC,JjDa,iajI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('iC,jIDa,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('iC,jJDa,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= 2 * einsum('iD,IjCa,jaiJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('iD,IjCa,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('iD,jICa,jaiJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('iD,jJCa,iajI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('ia,IjCD,jaiJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= 2 * einsum('ia,IjCD,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] += einsum('ia,jJCD,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)
            s[s2:f2] -= 2 * einsum('ia,jJCD,iajI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type).reshape(-1)

            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
                temp = np.zeros((nocc,nocc,nvir,nvir))
                for a,b in lib.prange(0,nocc,chnk_size):
                    v_ceee = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir,nvir,nvir)
                    temp += -einsum('Ia,JiDb,iaCb->IJCD', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += 2 * einsum('Ia,JiDb,ibCa->IJCD', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= einsum('Ia,iJCb,iaDb->IJCD', Y, t1_ccee[a:b,:,:,:], v_ceee, optimize = einsum_type)
                    temp -= einsum('Ia,iJDb,ibCa->IJCD', Y, t1_ccee[a:b,:,:,:], v_ceee, optimize = einsum_type)
                    temp -= einsum('Ja,IiCb,iaDb->IJCD', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp += 2 * einsum('Ja,IiCb,ibDa->IJCD', Y, t1_ccee[:,a:b,:,:], v_ceee, optimize = einsum_type)
                    temp -= einsum('Ja,iICb,ibDa->IJCD', Y, t1_ccee[a:b,:,:,:], v_ceee, optimize = einsum_type)
                    temp -= einsum('Ja,iIDb,iaCb->IJCD', Y, t1_ccee[a:b,:,:,:], v_ceee, optimize = einsum_type)
                    temp -= einsum('iC,IJab,iaDb->IJCD', Y[a:b,:], t1_ccee, v_ceee, optimize = einsum_type)
                    temp -= einsum('iD,IJab,ibCa->IJCD', Y[a:b,:], t1_ccee, v_ceee, optimize = einsum_type)
                    temp += 2 * einsum('ia,IJCb,iaDb->IJCD', Y[a:b,:], t1_ccee, v_ceee, optimize = einsum_type)
                    temp -= einsum('ia,IJCb,ibDa->IJCD', Y[a:b,:], t1_ccee, v_ceee, optimize = einsum_type)
                    temp += 2 * einsum('ia,JIDb,iaCb->IJCD', Y[a:b,:], t1_ccee, v_ceee, optimize = einsum_type)
                    temp -= einsum('ia,JIDb,ibCa->IJCD', Y[a:b,:], t1_ccee, v_ceee, optimize = einsum_type)
                s[s2:f2] += temp.reshape(-1)
                del temp
                del v_ceee
            else:
                M_12Y0_ab  = -einsum('Ia,JiDb,iaCb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab += 2 * einsum('Ia,JiDb,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('Ia,iJCb,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('Ia,iJDb,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('Ja,IiCb,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab += 2 * einsum('Ja,IiCb,ibDa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('Ja,iICb,ibDa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('Ja,iIDb,iaCb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('iC,IJab,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('iD,IJab,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab += 2 * einsum('ia,IJCb,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('ia,IJCb,ibDa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab += 2 * einsum('ia,JIDb,iaCb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                M_12Y0_ab -= einsum('ia,JIDb,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
                s[s2:f2] += M_12Y0_ab.reshape(-1)

            del Y
        return s

    return sigma_

def analyze(myadc):

    if myadc.compute_properties:

        header = (
            "\n*************************************************************"
            "\n          Spectroscopic amplitude analysis summary"
            "\n*************************************************************")
        logger.info(myadc, header)

class RADCEE(radc.RADC):
    '''restricted ADC for EE energies and spectroscopic amplitudes

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
            Convergence threshold for Davidson iterations.  Default is 1e-8.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative
            diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcee = adc.RADCEE(myadc).run()

    Saved results

        e_ee : float or list of floats
            EE energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ee : array
            Eigenvectors for each EE transition.
        p_ee : array
            Oscillator strength for each EE transition (currently not implemented, returns None).
        x_ee : array
            Spectroscopic amplitudes for each EE transition (currently not implemented, returns None).
    '''

    _keys = {
        'tol_residual', 'conv_tol', 'e_corr', 'method',
        'method_type', 'mo_coeff', 'mo_energy', 'max_memory',
        't1', 't2', 'max_space', 'max_cycle',
        'nocc', 'nvir', 'nmo', 'mol', 'transform_integrals',
        'with_df', 'spec_factor_print_tol', 'evec_print_tol',
        'compute_properties', 'approx_trans_moments', 'E', 'U', 'P', 'X',
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
        self.E = None
        self.U = None
        self.P = None
        self.X = None
        self.evec_print_tol = adc.evec_print_tol
        self.spec_factor_print_tol = adc.spec_factor_print_tol

    kernel = radc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag

    analyze = analyze

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
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
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag
