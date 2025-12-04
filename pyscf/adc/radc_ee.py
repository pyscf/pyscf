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
#         Ning-Yuan Chen <cny003@outlook.com>
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
from pyscf import symm
from pyscf.data.nist import HARTREE2EV


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


def get_trans_moments(adc):

    U = renormalize_eigenvectors(adc)

    U = U.T.copy()

    nocc = adc._nocc
    nvir = adc._nvir

    nmo = nocc + nvir

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    t1 = adc.t1
    t2 = adc.t2

    t1_ccee = t2[0][:]

    if adc.t1[0] is not None:
        t2_ce = t1[0]
    else:
        t2_ce = np.zeros((nocc, nvir))

    if adc.t2[1] is not None:
        t2_ccee = t2[1][:]
    else:
        t2_ccee = np.zeros((nocc, nocc, nvir, nvir))

    einsum = lib.einsum
    einsum_type = True

    TY = np.zeros((nmo, nmo))

    TY_ = []

    for r in range(U.shape[0]):

        r1 = U[r][s1:f1]

        Y = r1.reshape(nocc, nvir).copy()
        r2 = U[r][s2:f2].reshape(nocc,nocc,nvir,nvir).copy()

        TY[:nocc, nocc:]  = einsum('IC->IC', Y, optimize = einsum_type).copy()

        TY[nocc:,:nocc]  = einsum('ia,LiAa->AL', Y, t1_ccee, optimize = einsum_type)
        TY[nocc:,:nocc] -= einsum('ia,iLAa->AL', Y, t1_ccee, optimize = einsum_type)

        TY[nocc:,:nocc] += einsum('ia,LiAa->AL', Y, t1_ccee, optimize = einsum_type)

        TY[:nocc, :nocc] =- einsum('Ia,La->IL', Y, t2_ce, optimize = einsum_type)
        TY[nocc:,nocc:]  = einsum('iC,iA->AC', Y, t2_ce, optimize = einsum_type)

        TY[:nocc,nocc:] +=- einsum('Ia,ijab,ijCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('Ia,ijab,jiCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= einsum('iC,ijab,Ijab->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('iC,ijab,Ijba->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += einsum('ia,ijab,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijab,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijba,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('ia,ijba,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)

        TY[nocc:,:nocc] += einsum('ia,LiAa->AL', Y, t2_ccee, optimize = einsum_type)
        TY[nocc:,:nocc] -= einsum('ia,iLAa->AL', Y, t2_ccee, optimize = einsum_type)

        TY[:nocc,nocc:] += einsum('ia,ijab,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijab,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijba,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)

        TY[nocc:,:nocc]  += einsum('ia,LiAa->AL', Y, t2_ccee, optimize = einsum_type)

        TY[:nocc,:nocc] +=- 2 * einsum('Iiab,Liab->IL', r2, t1_ccee, optimize = einsum_type)
        TY[:nocc,:nocc] += einsum('Iiab,Liba->IL', r2, t1_ccee, optimize = einsum_type)

        TY[nocc:,nocc:] += 2 * einsum('ijCa,ijAa->AC', r2, t1_ccee, optimize = einsum_type)
        TY[nocc:,nocc:] -=   einsum('ijCa,jiAa->AC', r2, t1_ccee, optimize = einsum_type)

        if (adc.method == "adc(2)-x") or (adc.method == "adc(3)"):

            TY[:nocc, :nocc] +=- 2 * einsum('Iiab,Jiab->IJ', r2, t2_ccee, optimize = einsum_type)
            TY[:nocc, :nocc] += einsum('Iiab,Jiba->IJ', r2, t2_ccee, optimize = einsum_type)

            TY[nocc:, nocc:] += 2 * einsum('ijBa,ijAa->AB', r2, t2_ccee, optimize = einsum_type)
            TY[nocc:, nocc:] -= einsum('ijBa,jiAa->AB', r2, t2_ccee, optimize = einsum_type)

        if (adc.method == "adc(3)"):

            if adc.t1[1] is not None:
                t3_ce = adc.t1[1]
            else:
                t3_ce = np.zeros((nocc, nvir))

            TY[:nocc,:nocc] -= einsum('Ia,La->IL', Y, t3_ce, optimize = einsum_type)
            TY[:nocc,:nocc] -= einsum('Ia,Liab,ib->IL', Y, t1_ccee, t2_ce, optimize = einsum_type)
            TY[:nocc,:nocc] += 1/2 * einsum('Ia,Liba,ib->IL', Y, t1_ccee, t2_ce, optimize = einsum_type)
            TY[:nocc,:nocc] += einsum('ia,Liab,Ib->IL', Y, t1_ccee, t2_ce, optimize = einsum_type)
            TY[:nocc,:nocc] -= einsum('ia,Liba,Ib->IL', Y, t1_ccee, t2_ce, optimize = einsum_type)

            TY[nocc:,nocc:] += einsum('iC,iA->AC', Y, t3_ce, optimize = einsum_type)
            TY[nocc:,nocc:] += einsum('iC,ijAa,ja->AC', Y, t1_ccee, t2_ce, optimize = einsum_type)
            TY[nocc:,nocc:] -= 1/2 * einsum('iC,jiAa,ja->AC', Y, t1_ccee, t2_ce, optimize = einsum_type)
            TY[nocc:,nocc:] -= einsum('ia,ijAa,jC->AC', Y, t1_ccee, t2_ce, optimize = einsum_type)
            TY[nocc:,nocc:] += einsum('ia,jiAa,jC->AC', Y, t1_ccee, t2_ce, optimize = einsum_type)

            TY[:nocc,nocc:] +=- einsum('Ia,ijCb,ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += 1/2 * einsum('Ia,ijCb,jiab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= einsum('Ia,ijab,ijCb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += 1/2 * einsum('Ia,ijab,jiCb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= einsum('iC,Ijab,ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += 1/2 * einsum('iC,Ijab,ijba->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= einsum('iC,ijab,Ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += 1/2 * einsum('iC,ijab,Ijba->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += einsum('ia,IjCb,ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,IjCb,ijba->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += einsum('ia,ijab,IjCb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijab,jICb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijba,IjCb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += 1/2 * einsum('ia,ijba,jICb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,jICb,ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += 1/2 * einsum('ia,jICb,ijba->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)

            TY[:nocc,:nocc] +=- einsum('ia,Liba,Ib->IL', Y, t1_ccee, t2_ce, optimize = einsum_type)

            TY[nocc:,nocc:] += einsum('ia,jiAa,jC->AC', Y, t1_ccee, t2_ce, optimize = einsum_type)

            TY[:nocc,nocc:] += einsum('ia,IjCb,ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,IjCb,ijba->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] += einsum('ia,ijab,IjCb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijab,jICb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijba,IjCb->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)
            TY[:nocc,nocc:] -= 1/2 * einsum('ia,jICb,ijab->IC', Y, t1_ccee, t2_ccee, optimize = einsum_type)

            #TY[nocc:,:nocc] += einsum('ia,LiAa->AL', Y, t3, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,LiAb,jkac,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,LiAb,jkac,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 4/3 * einsum('ia,Liba,jkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,Liba,jkAc,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,Libc,jkAa,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,ijab,LkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijab,LkAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijab,kLAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijab,kLAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijba,LkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijba,LkAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijba,kLAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijbc,LkAa,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijbc,LkAa,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 4/3 * einsum('ia,jiAa,Lkbc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,jiAa,Lkbc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,jiAb,Lkca,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)

            #TY[nocc:,:nocc] += einsum('ia,LiAa->AL', Y, t3, optimize = einsum_type)
            #TY[nocc:,:nocc] -= einsum('ia,LiaA->AL', Y, t3, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,LiAb,jkac,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,LiAb,jkac,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 4/3 * einsum('ia,Liab,jkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 2/3 * einsum('ia,Liab,jkAc,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 4/3 * einsum('ia,Liba,jkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,Liba,jkAc,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,Libc,jkAa,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 2/3 * einsum('ia,Libc,jkAa,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/3 * einsum('ia,iLAb,jkac,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/6 * einsum('ia,iLAb,jkac,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 4/3 * einsum('ia,ijAa,Lkbc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 2/3 * einsum('ia,ijAa,Lkbc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 8/3 * einsum('ia,ijAb,Lkac,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 4/3 * einsum('ia,ijAb,Lkac,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 4/3 * einsum('ia,ijAb,Lkca,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 2/3 * einsum('ia,ijAb,Lkca,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,ijab,LkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijab,LkAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijab,kLAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijab,kLAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijba,LkAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijba,LkAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijba,kLAc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/6 * einsum('ia,ijba,kLAc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/6 * einsum('ia,ijbc,LkAa,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/3 * einsum('ia,ijbc,LkAa,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 1/6 * einsum('ia,ijbc,kLAa,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 1/3 * einsum('ia,ijbc,kLAa,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 4/3 * einsum('ia,jiAa,Lkbc,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,jiAa,Lkbc,jkcb->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 4/3 * einsum('ia,jiAb,Lkac,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 2/3 * einsum('ia,jiAb,Lkac,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] -= 2/3 * einsum('ia,jiAb,Lkca,jkbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
            TY[nocc:,:nocc] += 2/3 * einsum('ia,jiAb,Lkca,kjbc->AL', Y,
                                                    t1_ccee, t1_ccee, t1_ccee, optimize = einsum_type)

        TY_ = np.append(TY_,TY)

    return TY_


def analyze_eigenvector(adc):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc*nvir
    evec_print_tol = adc.evec_print_tol
    U = adc.U

    logger.info(adc, "Number of occupied orbitals = %d", nocc)
    logger.info(adc, "Number of virtual orbitals =  %d", nvir)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nocc,nvir,nvir)
        U1dotU1 = np.dot(U1, U1)
        U2dotU2 = 1.*np.dot(U2.ravel(), U2.ravel()) - \
            .5*np.dot(U2.ravel(), U2.transpose(0,1,3,2).ravel())

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
                i_idx = orb_idx//nvir
                a_idx = orb_idx % nvir
                singles_idx.append((i_idx + 1, a_idx + 1))
                singles_val.append(U_sorted[iter_num])

            if orb_idx >= n_singles:
                ijab_idx = orb_idx - n_singles
                ab_rem = ijab_idx % (nvir*nvir)
                ij_idx = ijab_idx//(nvir*nvir)
                a_idx = ab_rem//nvir
                b_idx = ab_rem % nvir
                i_idx = ij_idx//nocc
                j_idx = ij_idx % nocc
                doubles_idx.append((i_idx + 1, j_idx + 1, a_idx + 1 + nocc, b_idx + 1 + nocc))
                doubles_val.append(U_sorted[iter_num])

            iter_num += 1

        logger.info(adc,'%s | root %d | Energy (eV) = %12.8f | norm(1p1h)  = %6.4f | norm(2p2h) = %6.4f ',
                    adc.method, I, adc.E[I]*HARTREE2EV, U1dotU1, U2dotU2)

        if singles_val:
            logger.info(adc, "\n1p1h block: ")
            logger.info(adc, "     i     a     U(i,a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_idx):
                logger.info(adc, '  %4d   %4d   %7.4f', print_singles[0], print_singles[1], singles_val[idx])

        if doubles_val:
            logger.info(adc, "\n2p2h block: ")
            logger.info(adc, "     i     j     a     b     U(i,j,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_idx):
                logger.info(adc, '  %4d  %4d  %4d  %4d  %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], print_doubles[3], doubles_val[idx])

        logger.info(adc,
            "***************************************************************************************\n")


def analyze_spec_factor(adc):

    X = adc.X
    nroots = X.shape[0]
    X = X.reshape(nroots, -1)

    X_2 = (X.copy()**2)*2
    thresh = adc.spec_factor_print_tol

    nmo = adc._nmo

    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)

    for i in range(X_2.shape[0]):

        sort = np.argsort(-X_2[i,:])
        X_2_row = X_2[i,:]
        X_2_row = X_2_row[sort]

        if not adc.mol.symmetry:
            sym = np.repeat(['A'], nmo)
        else:
            sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff.orbsym]
            sym = np.array(sym)

            sym = sym[sort]

        spec_Contribution = X_2_row[X_2_row > thresh]

        if np.sum(spec_Contribution) == 0.0:
            continue

        logger.info(adc, '%s | root %d | Energy (eV) = %12.8f \n',
                adc.method, i, adc.E[i]*HARTREE2EV)
        logger.info(adc, "     Hole_MO     Particle_MO     Spec. Contribution     Orbital symmetry")
        logger.info(adc, "-----------------------------------------------------------")

        for hp in range(spec_Contribution.shape[0]):
            P = ((sort[hp]) % nmo)
            H = ((sort[hp]) // nmo)

            logger.info(adc, '     %3.d             %3.d          %10.8f                %s -> %s',
                        (H+1), (P+1), spec_Contribution[hp], sym[H], sym[P])

        logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
        logger.info(adc,
        "***********************************************************\n")


def renormalize_eigenvectors(adc, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc*nvir

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 1.*np.dot(U2.ravel(), U2.ravel()) - \
            .5*np.dot(U2.ravel(), U2.transpose(0,1,3,2).ravel())
        U[:,I] /= np.sqrt(UdotU)

    return U


def get_properties(adc, nroots=1):

    TY  = adc.get_trans_moments()

    nocc = adc._nocc
    nvir = adc._nvir

    nmo = nocc + nvir

    dm = adc.dip_mom

    X = TY.reshape(nroots, nmo, nmo)
    DX = lib.einsum("xqp,nqp->xn", dm, X, optimize=True)

    spec_intensity = np.conj(DX[0]) * DX[0]
    spec_intensity += np.conj(DX[1]) * DX[1]
    spec_intensity += np.conj(DX[2]) * DX[2]

    P = 2*spec_intensity*adc.E*(2.0/3.0)

    return P,X


def analyze(myadc):

    header = ("\n*************************************************************"
              "\n           Eigenvector analysis summary"
              "\n*************************************************************")
    logger.info(myadc, header)

    myadc.analyze_eigenvector()

    if myadc.compute_properties:

        header = (
            "\n*************************************************************"
            "\n          Spectroscopic amplitude analysis summary"
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

    L = np.array(L).ravel()
    R = np.array(R).ravel()

    t1_ccee = adc.t2[0][:]

    einsum = lib.einsum
    einsum_type = True

    nocc = adc._nocc
    nvir = adc._nvir
    nmo = nocc + nvir
    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    if adc.t1[0] is not None:
        t2_ce = adc.t1[0]
    else:
        t2_ce = np.zeros((nocc, nvir))

    occ_list = range(nocc)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1  = np.zeros((nmo,nmo))

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L1 = L1.reshape(nocc,nvir)
    R1 = R1.reshape(nocc,nvir)
    L2 = L2.reshape(nocc,nocc,nvir,nvir)
    R2 = R2.reshape(nocc,nocc,nvir,nvir)

############# block- ij
    ### 000 ###
    rdm1[:nocc, :nocc] =- einsum('Ja,Ia->IJ', L1, R1, optimize = einsum_type)
    rdm1[occ_list, occ_list] += 2 * einsum('ia,ia->', L1, R1, optimize = einsum_type)

    ### 101 ###
    rdm1[:nocc, :nocc] -= 2 * einsum('Jiab,Iiab->IJ', L2, R2, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('Jiab,Iiba->IJ', L2, R2, optimize = einsum_type)
    rdm1[occ_list, occ_list] += 2 * einsum('ijab,ijab->', L2, R2, optimize = einsum_type)
    rdm1[occ_list, occ_list] -= einsum('ijab,ijba->', L2, R2, optimize = einsum_type)

    ### 020 ###
    rdm1[:nocc, :nocc] += einsum('Ja,ia,ijbc,Ijbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 1/2 * einsum('Ja,ia,ijbc,Ijcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 2 * einsum('Ja,ib,Ijac,ijbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('Ja,ib,Ijac,ijcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('Ja,ib,Ijca,ijbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 1/2 * einsum('Ja,ib,Ijca,ijcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('ia,Ia,ijbc,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 1/2 * einsum('ia,Ia,ijbc,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 2 * einsum('ia,Ib,ijac,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('ia,Ib,ijac,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += einsum('ia,Ib,ijca,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 1/2 * einsum('ia,Ib,ijca,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 4 * einsum('ia,ia,Ijbc,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('ia,ia,Ijbc,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('ia,ib,Ijac,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= einsum('ia,ib,Ijac,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= einsum('ia,ib,Ijca,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('ia,ib,Ijca,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('ia,ja,Iibc,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= einsum('ia,ja,Iibc,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= einsum('ia,jb,Iiac,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('ia,jb,Iiac,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] += 2 * einsum('ia,jb,Iica,Jjbc->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, :nocc] -= 4 * einsum('ia,jb,Iica,Jjcb->IJ', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)


############# block- ab
    ### 000 ###
    rdm1[nocc:, nocc:]  = einsum('iA,iB->AB', L1, R1, optimize = einsum_type)

    ### 101 ###
    rdm1[nocc:, nocc:] += 2 * einsum('ijAa,ijBa->AB', L2, R2, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('ijAa,ijaB->AB', L2, R2, optimize = einsum_type)

    ### 020 ###
    rdm1[nocc:, nocc:] -= einsum('iA,ia,jkab,jkBb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 1/2 * einsum('iA,ia,jkab,kjBb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 2 * einsum('iA,ja,ikBb,jkab->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('iA,ja,ikBb,jkba->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('iA,ja,kiBb,jkab->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 1/2 * einsum('iA,ja,kiBb,jkba->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('ia,iB,jkab,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 1/2 * einsum('ia,iB,jkab,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 4 * einsum('ia,ia,jkAb,jkBb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= 2 * einsum('ia,ia,jkAb,kjBb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= 2 * einsum('ia,ib,jkBa,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += einsum('ia,ib,jkBa,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 2 * einsum('ia,jB,ikab,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('ia,jB,ikab,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= einsum('ia,jB,ikba,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 1/2 * einsum('ia,jB,ikba,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= 2 * einsum('ia,ja,ikBb,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += einsum('ia,ja,ikBb,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += einsum('ia,ja,kiBb,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= 2 * einsum('ia,ja,kiBb,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += einsum('ia,jb,ikBa,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= 2 * einsum('ia,jb,ikBa,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] -= 2 * einsum('ia,jb,kiBa,jkAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)
    rdm1[nocc:, nocc:] += 4 * einsum('ia,jb,kiBa,kjAb->AB', L1, R1, t1_ccee, t1_ccee, optimize = einsum_type)


############# block- ia
    ### 100 & 001 ###
    rdm1[:nocc, nocc:]  = 2 * einsum('ia,IiAa->IA', L1, R2, optimize = einsum_type)
    rdm1[:nocc, nocc:] -= einsum('ia,IiaA->IA', L1, R2, optimize = einsum_type)

    ### 110 & 011 ###
    rdm1[:nocc, nocc:] -= 2 * einsum('ijab,Ia,ijAb->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] += einsum('ijab,Ia,jiAb->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] -= 2 * einsum('ijab,iA,Ijab->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] += einsum('ijab,iA,Ijba->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] += 4 * einsum('ijab,ia,IjAb->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] -= 2 * einsum('ijab,ia,jIAb->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] -= 2 * einsum('ijab,ib,IjAa->IA', L2, R1, t1_ccee, optimize = einsum_type)
    rdm1[:nocc, nocc:] += einsum('ijab,ib,jIAa->IA', L2, R1, t1_ccee, optimize = einsum_type)

    ### 020 ###
    rdm1[:nocc, nocc:] -= einsum('ia,Ia,iA->IA', L1, R1, t2_ce, optimize = einsum_type)
    rdm1[:nocc, nocc:] -= einsum('ia,iA,Ia->IA', L1, R1, t2_ce, optimize = einsum_type)
    rdm1[:nocc, nocc:] += 2 * einsum('ia,ia,IA->IA', L1, R1, t2_ce, optimize = einsum_type)

############# block- ai
    rdm1[nocc:,:nocc] = rdm1[:nocc,nocc:].T

    ####### ADC(3) SPIN ADAPTED EXCITED STATE OPDM WITH SQA ################
    if adc.method == "adc(3)":
        ### Redudant Variables used for names from SQA
        einsum_type = True
        t2_ccee = adc.t2[1][:]

        if adc.t1[1] is not None:
            t3_ce = adc.t1[1]
        else:
            t3_ce = np.zeros((nocc, nvir))
        ###################################################

############# block- ij
        ### 030 ###
        rdm1[:nocc, :nocc] += einsum('Ja,ia,Ijbc,ijbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('Ja,ia,Ijbc,ijcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Ja,ia,ijbc,Ijbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('Ja,ia,ijbc,Ijcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 2 * einsum('Ja,ib,Ijac,ijbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Ja,ib,Ijac,ijcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Ja,ib,Ijca,ijbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('Ja,ib,Ijca,ijcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 2 * einsum('Ja,ib,ijbc,Ijac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Ja,ib,ijbc,Ijca->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Ja,ib,ijcb,Ijac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('Ja,ib,ijcb,Ijca->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('ia,Ia,Jjbc,ijbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('ia,Ia,Jjbc,ijcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('ia,Ia,ijbc,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('ia,Ia,ijbc,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 2 * einsum('ia,Ib,Jjbc,ijac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('ia,Ib,Jjbc,ijca->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('ia,Ib,Jjcb,ijac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('ia,Ib,Jjcb,ijca->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 2 * einsum('ia,Ib,ijac,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('ia,Ib,ijac,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('ia,Ib,ijca,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 1/2 * einsum('ia,Ib,ijca,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 4 * einsum('ia,ia,Ijbc,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ia,Ijbc,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 4 * einsum('ia,ia,Jjbc,Ijbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ia,Jjbc,Ijcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ib,Ijac,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,ib,Ijac,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,ib,Ijca,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ib,Ijca,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ib,Jjbc,Ijac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,ib,Jjbc,Ijca->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,ib,Jjcb,Ijac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ib,Jjcb,Ijca->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ja,Iibc,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,ja,Iibc,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,ja,Jjbc,Iibc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,ja,Jjbc,Iicb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,jb,Iiac,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,jb,Iiac,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,jb,Iica,Jjbc->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 4 * einsum('ia,jb,Iica,Jjcb->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= einsum('ia,jb,Jjbc,Iiac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,jb,Jjbc,Iica->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] += 2 * einsum('ia,jb,Jjcb,Iiac->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 4 * einsum('ia,jb,Jjcb,Iica->IJ', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)

        ### 021 & 120 ###
        rdm1[:nocc, :nocc] += einsum('ia,Iiab,Jb->IJ', L1, R2, t2_ce, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 2 * einsum('ia,Iiba,Jb->IJ', L1, R2, t2_ce, optimize = einsum_type)
        rdm1[:nocc, :nocc] += einsum('Jiab,ia,Ib->IJ', L2, R1, t2_ce, optimize = einsum_type)
        rdm1[:nocc, :nocc] -= 2 * einsum('Jiab,ib,Ia->IJ', L2, R1, t2_ce, optimize = einsum_type)

        #----------------------------------------------------------------------------------------------------------#

############# block- ab
        ### 030 ###
        rdm1[nocc:, nocc:] -= einsum('iA,ia,jkBb,jkab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('iA,ia,jkBb,kjab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('iA,ia,jkab,jkBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('iA,ia,jkab,kjBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('iA,ja,ikBb,jkab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('iA,ja,ikBb,jkba->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('iA,ja,jkab,ikBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('iA,ja,jkab,kiBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('iA,ja,jkba,ikBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('iA,ja,jkba,kiBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('iA,ja,kiBb,jkab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('iA,ja,kiBb,jkba->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ia,iB,jkAb,jkab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('ia,iB,jkAb,kjab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ia,iB,jkab,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('ia,iB,jkab,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 4 * einsum('ia,ia,jkAb,jkBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ia,jkAb,kjBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 4 * einsum('ia,ia,jkBb,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ia,jkBb,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ib,jkAb,jkBa->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,ib,jkAb,kjBa->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ib,jkBa,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,ib,jkBa,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('ia,jB,ikab,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ia,jB,ikab,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ia,jB,ikba,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('ia,jB,ikba,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('ia,jB,jkAb,ikab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ia,jB,jkAb,ikba->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ia,jB,kjAb,ikab->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 1/2 * einsum('ia,jB,kjAb,ikba->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ja,ikBb,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,ja,ikBb,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ja,jkAb,ikBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,ja,jkAb,kiBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,ja,kiBb,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ja,kiBb,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,ja,kjAb,ikBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,ja,kjAb,kiBb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,jb,ikBa,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,jb,ikBa,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += einsum('ia,jb,jkAb,ikBa->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,jb,jkAb,kiBa->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,jb,kiBa,jkAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 4 * einsum('ia,jb,kiBa,kjAb->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= 2 * einsum('ia,jb,kjAb,ikBa->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 4 * einsum('ia,jb,kjAb,kiBa->AB', L1, R1, t1_ccee, t2_ccee, optimize = einsum_type)

        ### 021 & 120 ###
        rdm1[nocc:, nocc:] -= einsum('ia,ijBa,jA->AB', L1, R2, t2_ce, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('ia,ijaB,jA->AB', L1, R2, t2_ce, optimize = einsum_type)
        rdm1[nocc:, nocc:] -= einsum('ijAa,ia,jB->AB', L2, R1, t2_ce, optimize = einsum_type)
        rdm1[nocc:, nocc:] += 2 * einsum('ijAa,ja,iB->AB', L2, R1, t2_ce, optimize = einsum_type)
        #----------------------------------------------------------------------------------------------------------#
############# block- ia
        ### 030 ###
        rdm1[:nocc, nocc:] -= einsum('ia,Ia,iA->IA', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,iA,Ia->IA', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 2 * einsum('ia,ia,IA->IA', L1, R1, t3_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,Ia,ijAb,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('ia,Ia,jiAb,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('ia,Ib,ijAa,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,Ib,jiAa,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,iA,Ijab,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('ia,iA,Ijba,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 2 * einsum('ia,ia,IjAb,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,ia,jIAb,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,ib,IjAa,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('ia,ib,jIAa,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('ia,jA,Iiab,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,jA,Iiba,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= einsum('ia,ja,IiAb,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 1/2 * einsum('ia,ja,iIAb,jb->IA', L1, R1, t1_ccee, t2_ce, optimize = einsum_type)

        ### 021 & 120 ###
        rdm1[:nocc, nocc:] -= 2 * einsum('ijab,Ia,ijAb->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ijab,Ia,jiAb->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ijab,iA,Ijab->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ijab,iA,Ijba->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 4 * einsum('ijab,ia,IjAb->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ijab,ia,jIAb->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ijab,ib,IjAa->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ijab,ib,jIAa->IA', L2, R1, t2_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Iiab,jkbc,jkAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,Iiab,jkbc,kjAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,Iiba,jkbc,jkAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Iiba,jkbc,kjAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Iibc,jkAa,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,Iibc,jkAa,kjbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,Ijab,ikAc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Ijab,ikAc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Ijab,kiAc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,Ijab,kiAc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Ijba,ikAc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,Ijba,ikAc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,Ijba,kiAc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Ijba,kiAc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,Ijbc,ikAa,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Ijbc,ikAa,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,Ijbc,kiAa,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,Ijbc,kiAa,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijAa,jkbc,Ikbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,ijAa,jkbc,Ikcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,ijAb,Ikac,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijAb,Ikac,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijAb,Ikca,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,ijAb,Ikca,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,ijaA,jkbc,Ikbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijaA,jkbc,Ikcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += 4 * einsum('ia,ijab,jkbc,IkAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,ijab,jkbc,kIAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,ijab,jkcb,IkAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijab,jkcb,kIAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijbA,Ikac,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,ijbA,Ikac,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,ijbA,Ikca,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijbA,Ikca,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,ijba,jkbc,IkAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijba,jkbc,kIAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijba,jkcb,IkAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,ijba,jkcb,kIAc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijbc,IkAa,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,ijbc,IkAa,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,ijbc,kIAa,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,ijbc,kIAa,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,jkAa,Iibc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,jkAa,Iibc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,jkAb,Iiac,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,jkAb,Iiac,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,jkAb,Iica,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,jkAb,Iica,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,jkab,IiAc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 2 * einsum('ia,jkab,IiAc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] -= 1/2 * einsum('ia,jkab,iIAc,jkbc->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)
        rdm1[:nocc, nocc:] += einsum('ia,jkab,iIAc,jkcb->IA', L1, R2, t1_ccee, t1_ccee, optimize = einsum_type)

        #----------------------------------------------------------------------------------------------------------#
############# block- ai
        rdm1[nocc:,:nocc] = rdm1[:nocc,nocc:].T

    return rdm1


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
        'method_type', 'mo_coeff', 'mo_coeff_hf', 'mo_energy', 'max_memory',
        't1', 't2', 'max_space', 'max_cycle',
        'nocc', 'nvir', 'nmo', 'mol', 'transform_integrals',
        'with_df', 'dip_mom','spec_factor_print_tol', 'evec_print_tol',
        'compute_properties', 'approx_trans_moments', 'E', 'U', 'P', 'X',
        '_make_rdm1', 'frozen', 'mo_occ'
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
        self.mo_coeff_hf = adc.mo_coeff_hf
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.dip_mom = adc.dip_mom
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
        self.frozen = adc.frozen
        self.mo_occ = adc.mo_occ
        self._adc_es = self

    kernel = radc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag
    get_trans_moments = get_trans_moments

    renormalize_eigenvectors = renormalize_eigenvectors
    analyze = analyze
    _make_rdm1 = make_rdm1
    analyze_eigenvector = analyze_eigenvector
    analyze_spec_factor = analyze_spec_factor
    get_properties = get_properties

    def get_init_guess(self, nroots=1, diag=None, ascending=True, type=None, eris=None, ini=None):
        def der_sig(n_singles,nocc,nvir,dim,M_ab):
            def sigma_(r):
                r1 = r[0:n_singles]
                s = np.zeros(dim)
                s[0:n_singles] = lib.einsum('ab,b->a',M_ab,r1, optimize = True)
                return s
            return sigma_

        if (type=="cis"):
            logger.info(self, "Generating CIS initial guess for eigenvector")

            ncore = self._nocc
            nextern = self._nvir
            n_singles = ncore * nextern
            n_doubles = ncore * ncore * nextern * nextern
            dim = n_singles + n_doubles

            einsum = lib.einsum
            einsum_type = True

            if eris is None:
                eris = self.transform_integrals()

            v_ccee = eris.oovv
            v_ceec = eris.ovvo

            M_ab = np.zeros((ncore*nextern, ncore*nextern))

            ####000#####################
            d_ai_a = self.mo_energy[ncore:][:,None] - self.mo_energy[:ncore]
            np.fill_diagonal(M_ab, d_ai_a.transpose().reshape(-1))
            M_ab = M_ab.reshape(ncore,nextern,ncore,nextern).copy()

            ####010#####################
            M_ab -= einsum('ILAD->IDLA', v_ccee, optimize = einsum_type).copy()
            M_ab += 2 * einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()

            M_ab = M_ab.reshape(n_singles, n_singles)
            guess = self.get_init_guess(nroots, diag, ascending = True)
            sigma = der_sig(n_singles,ncore,nextern,dim,M_ab)
            conv, uu, g = lib.linalg_helper.davidson_nosym1(
                lambda xs : [sigma(x) for x in xs],
                guess, diag, nroots=nroots, verbose=self.verbose, tol=self.conv_tol, max_memory=self.max_memory,
                max_cycle=self.max_cycle, max_space=self.max_space, tol_residual=self.tol_residual)
            nfalse = np.shape(conv)[0] - np.sum(conv)
            if nfalse >= 1:
                logger.warn(self, "ADC1 Davidson iterations for " + str(nfalse) + " root(s) did not converge!!!")
            g = np.array(g).T
            if not ascending:
                g = g[:, ::-1]
        elif (type=="read"):
            logger.info(self, "obtain initial guess from input variable")
            ncore = self._nocc
            nextern = self._nvir
            n_singles = ncore * nextern
            n_doubles = ncore * ncore * nextern * nextern
            dim = n_singles + n_doubles
            if isinstance(ini, list):
                g = np.array(ini)
            else:
                g = ini
            if g.shape[0] != dim or g.shape[1] != nroots:
                raise ValueError(f"Shape of guess should be ({dim},{nroots})")
        else:
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
