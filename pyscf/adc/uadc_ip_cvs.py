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
from pyscf import lib, symm
from pyscf.lib import logger
from pyscf.adc import uadc
from pyscf.adc import uadc_ao2mo
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc

def get_imds(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2 = adc.t2

    ncvs = adc.ncvs

    # We assume that the number of ionized alpha and beta electrons is the same
    e_occ_a = adc.mo_energy_a[:ncvs]
    e_occ_b = adc.mo_energy_b[:ncvs]

    idn_occ = np.identity(ncvs)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = np.array(eris.ovvo).transpose(0,1,3,2)
    eris_OVOV = np.array(eris.OVVO).transpose(0,1,3,2)
    eris_ovOV = np.array(eris.OVvo).transpose(3,2,0,1)
    eris_ceoe = eris_ovov[:ncvs,:,:,:].copy()
    eris_CEOE = eris_OVOV[:ncvs,:,:,:].copy()
    eris_ceOE = eris_ovOV[:ncvs,:,:,:].copy()
    eris_oeCE = eris_ovOV[:,:,:ncvs,:].copy()

    # i-j block
    # Zeroth-order terms

    M_ij_a = lib.einsum('ij,j->ij', idn_occ ,e_occ_a)
    M_ij_b = lib.einsum('ij,j->ij', idn_occ ,e_occ_b)

    # Second-order terms

    t2_1_a = t2[0][0][:]
    t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
    M_ij_a += 0.5 * 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_a_coee, eris_ceoe, optimize=True)
    M_ij_a -= 0.5 * 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_a_coee, eris_ceoe, optimize=True)
    M_ij_a += 0.5 * 0.5 *  lib.einsum('jlde,idle->ij',t2_1_a_coee, eris_ceoe, optimize=True)
    M_ij_a -= 0.5 * 0.5 *  lib.einsum('jlde,ield->ij',t2_1_a_coee, eris_ceoe, optimize=True)

    t2_1_b = t2[0][2][:]
    t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
    M_ij_b += 0.5 * 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_b_coee, eris_CEOE, optimize=True)
    M_ij_b -= 0.5 * 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_b_coee, eris_CEOE, optimize=True)
    M_ij_b += 0.5 * 0.5 *  lib.einsum('jlde,idle->ij',t2_1_b_coee, eris_CEOE, optimize=True)
    M_ij_b -= 0.5 * 0.5 *  lib.einsum('jlde,ield->ij',t2_1_b_coee, eris_CEOE, optimize=True)

    t2_1_ab = t2[0][1][:]
    t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
    t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
    M_ij_a += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ab_coee, eris_ceOE, optimize=True)
    M_ij_b += 0.5 * lib.einsum('lied,lejd->ij',t2_1_ab_ocee, eris_oeCE, optimize=True)
    M_ij_a += 0.5 * lib.einsum('jlde,idle->ij',t2_1_ab_coee, eris_ceOE, optimize=True)
    M_ij_b += 0.5 * lib.einsum('ljed,leid->ij',t2_1_ab_ocee, eris_oeCE, optimize=True)

    del t2_1_a
    del t2_1_a_coee
    del t2_1_b
    del t2_1_b_coee
    del t2_1_ab
    del t2_1_ab_coee
    del t2_1_ab_ocee

    # Third-order terms

    if (method == "adc(3)"):

        eris_ccee = eris.oovv[:ncvs,:ncvs,:,:].copy()
        eris_CCEE = eris.OOVV[:ncvs,:ncvs,:,:].copy()
        eris_ccEE = eris.ooVV[:ncvs,:ncvs,:,:].copy()
        eris_CCee = eris.OOvv[:ncvs,:ncvs,:,:].copy()
        eris_oecc = eris.ovoo[:,:,:ncvs,:ncvs].copy()
        eris_OECC = eris.OVOO[:,:,:ncvs,:ncvs].copy()
        eris_oeCC = eris.ovOO[:,:,:ncvs,:ncvs].copy()
        eris_OEcc = eris.OVoo[:,:,:ncvs,:ncvs].copy()
        eris_cooo = eris.oooo[:ncvs,:,:,:].copy()
        eris_ccoo = eris.oooo[:ncvs,:ncvs,:,:].copy()
        eris_COOO = eris.OOOO[:ncvs,:,:,:].copy()
        eris_CCOO = eris.OOOO[:ncvs,:ncvs,:,:].copy()
        eris_ccOO = eris.ooOO[:ncvs,:ncvs,:,:].copy()
        eris_ooCC = eris.ooOO[:,:,:ncvs,:ncvs].copy()
        eris_coOO = eris.ooOO[:ncvs,:,:,:].copy()

        eris_cece = eris_ovov[:ncvs,:,:ncvs,:].copy()
        eris_CECE = eris_OVOV[:ncvs,:,:ncvs,:].copy()
        eris_coee = eris.oovv[:ncvs,:,:,:].copy()
        eris_COEE = eris.OOVV[:ncvs,:,:,:].copy()
        eris_coEE = eris.ooVV[:ncvs,:,:,:].copy()
        eris_COee = eris.OOvv[:ncvs,:,:,:].copy()
        eris_ceco = eris.ovoo[:ncvs,:,:ncvs,:].copy()
        eris_CECO = eris.OVOO[:ncvs,:,:ncvs,:].copy()
        eris_coco = eris.oooo[:ncvs,:,:ncvs,:].copy()
        eris_COCO = eris.OOOO[:ncvs,:,:ncvs,:].copy()
        eris_ooCO = eris.ooOO[:,:,:ncvs,:].copy()

        t1 = adc.t1
        t1_2_a, t1_2_b = t1[0]

        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_a, eris_oecc, optimize=True)
        M_ij_a -= lib.einsum('ld,jdil->ij',t1_2_a, eris_ceco, optimize=True)
        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_b, eris_OEcc, optimize=True)

        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_b, eris_OECC, optimize=True)
        M_ij_b -= lib.einsum('ld,jdil->ij',t1_2_b, eris_CECO, optimize=True)
        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_a, eris_oeCC, optimize=True)

        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_a, eris_oecc, optimize=True)
        M_ij_a -= lib.einsum('ld,idjl->ij',t1_2_a, eris_ceco, optimize=True)
        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_b, eris_OEcc, optimize=True)

        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_b, eris_OECC, optimize=True)
        M_ij_b -= lib.einsum('ld,idjl->ij',t1_2_b, eris_CECO, optimize=True)
        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_a, eris_oeCC, optimize=True)

        del t1_2_a
        del t1_2_b

        t2_1_a = t2[0][0][:]
        t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
        t2_2_a = t2[1][0][:]
        t2_2_a_coee = t2_2_a[:ncvs,:,:,:].copy()

        M_ij_a += 0.5 * 0.5* lib.einsum('ilde,jdle->ij',t2_2_a_coee, eris_ceoe, optimize=True)
        M_ij_a -= 0.5 * 0.5* lib.einsum('ilde,jeld->ij',t2_2_a_coee, eris_ceoe, optimize=True)

        M_ij_a += 0.5 * 0.5* lib.einsum('jlde,idle->ij',t2_2_a_coee, eris_ceoe, optimize=True)
        M_ij_a -= 0.5 * 0.5* lib.einsum('jlde,ield->ij',t2_2_a_coee, eris_ceoe, optimize=True)

        t2_1_b = t2[0][2][:]
        t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
        t2_2_b = t2[1][2][:]
        t2_2_b_coee = t2_2_b[:ncvs,:,:,:].copy()

        M_ij_b += 0.5 * 0.5* lib.einsum('ilde,jdle->ij',t2_2_b_coee, eris_CEOE, optimize=True)
        M_ij_b -= 0.5 * 0.5* lib.einsum('ilde,jeld->ij',t2_2_b_coee, eris_CEOE, optimize=True)
        M_ij_b += 0.5 * 0.5* lib.einsum('jlde,idle->ij',t2_2_b_coee, eris_CEOE, optimize=True)
        M_ij_b -= 0.5 * 0.5* lib.einsum('jlde,ield->ij',t2_2_b_coee, eris_CEOE, optimize=True)

        t2_1_ab = t2[0][1][:]
        t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
        t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
        t2_2_ab = t2[1][1][:]
        t2_2_ab_coee = t2_2_ab[:ncvs,:,:,:].copy()
        t2_2_ab_ocee = t2_2_ab[:,:ncvs,:,:].copy()

        M_ij_a += 0.5 * lib.einsum('ilde,jdle->ij',t2_2_ab_coee, eris_ceOE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('lied,lejd->ij',t2_2_ab_ocee, eris_oeCE, optimize=True)
        M_ij_a += 0.5 * lib.einsum('jlde,idle->ij',t2_2_ab_coee, eris_ceOE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('ljed,leid->ij',t2_2_ab_ocee, eris_oeCE, optimize=True)

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_a, t2_1_a_coee, optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mejf,ifme->ij',M_ij_t, eris_ceoe, optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mejf,ifme->ji',M_ij_t, eris_ceoe, optimize=True)
        M_ij_a += 0.5 * lib.einsum('mejf,imfe->ij',M_ij_t, eris_coee, optimize=True)
        M_ij_a += 0.5 * lib.einsum('mejf,imfe->ji',M_ij_t, eris_coee, optimize=True)
        del M_ij_t

        M_ij_a += 0.5 * 0.25*lib.einsum('lmde,jnde,ilmn->ij',t2_1_a,
                                        t2_1_a_coee,eris_cooo, optimize=True)
        M_ij_a -= 0.5 * 0.25*lib.einsum('lmde,jnde,imnl->ij',t2_1_a,
                                        t2_1_a_coee,eris_cooo, optimize=True)

        M_ij_a += 0.5 *0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a_coee,
                                       t2_1_a,eris_cooo, optimize=True)
        M_ij_a -= 0.5 *0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a_coee,
                                       t2_1_a,eris_cooo, optimize=True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_ccee, optimize=True)
        M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfie->ij',t2_1_a, t2_1_a, eris_cece, optimize=True)
        M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_CCee , optimize=True)

        M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_ccoo, optimize=True)
        M_ij_a += 0.5*lib.einsum('lnde,lmde,jmin->ij',t2_1_a, t2_1_a, eris_coco, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooCC, optimize=True)

        t2_1_ab = t2[0][1][:]
        t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
        t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
        M_ij_a -= 0.5 * lib.einsum('lmde,jldf,ifme->ij',t2_1_ab,
                                   t2_1_a_coee, eris_ceOE,optimize=True)
        M_ij_b += 0.5 * lib.einsum('lmde,ljdf,meif->ij',t2_1_a,
                                   t2_1_ab_ocee, eris_oeCE ,optimize=True)
        M_ij_a -= 0.5 * lib.einsum('lmde,ildf,jfme->ij',t2_1_ab,
                                   t2_1_a_coee, eris_ceOE,optimize=True)
        M_ij_b += 0.5 * lib.einsum('lmde,lidf,mejf->ij',t2_1_a,
                                   t2_1_ab_ocee, eris_oeCE ,optimize=True)
        del t2_1_a
        del t2_1_a_coee

        M_ij_a += 0.5 * lib.einsum('lmde,jlfd,ifme->ij',t2_1_b,
                                   t2_1_ab_coee, eris_ceOE ,optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mled,jldf,meif->ij',t2_1_ab,
                                   t2_1_b_coee, eris_oeCE,optimize=True)
        M_ij_a += 0.5 * lib.einsum('lmde,ilfd,jfme->ij',t2_1_b,
                                   t2_1_ab_coee, eris_ceOE ,optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mled,ildf,mejf->ij',t2_1_ab,
                                   t2_1_b_coee, eris_oeCE,optimize=True)

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_b, t2_1_b_coee, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mejf,ifme->ij',M_ij_t, eris_CEOE, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mejf,ifme->ji',M_ij_t, eris_CEOE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mejf,imfe->ij',M_ij_t, eris_COEE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mejf,imfe->ji',M_ij_t, eris_COEE, optimize=True)
        del M_ij_t

        M_ij_b += 0.5 * 0.25*lib.einsum('lmde,jnde,ilmn->ij',t2_1_b,
                                        t2_1_b_coee,eris_COOO, optimize=True)
        M_ij_b -= 0.5 * 0.25*lib.einsum('lmde,jnde,imnl->ij',t2_1_b,
                                        t2_1_b_coee,eris_COOO, optimize=True)

        M_ij_b += 0.5 * 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_b_coee,
                                        t2_1_b,eris_COOO, optimize=True)
        M_ij_b -= 0.5 * 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_b_coee,
                                        t2_1_b,eris_COOO, optimize=True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ccEE , optimize=True)
        M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_CCEE , optimize=True)
        M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfie->ij',t2_1_b, t2_1_b, eris_CECE , optimize=True)

        M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ccOO, optimize=True)
        M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_CCOO, optimize=True)
        M_ij_b += 0.5*lib.einsum('lnde,lmde,jmin->ij',t2_1_b, t2_1_b, eris_COCO, optimize=True)
        del t2_1_b
        del t2_1_b_coee

        M_ij_a += 0.5 * lib.einsum('mled,jlfd,ifme->ij',t2_1_ab,
                                   t2_1_ab_coee, eris_ceoe ,optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mled,jlfd,imfe->ij',t2_1_ab,
                                   t2_1_ab_coee, eris_coee ,optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mlde,jldf,imfe->ij',t2_1_ab,
                                   t2_1_ab_coee, eris_coEE ,optimize=True)

        M_ij_b += 0.5 * lib.einsum('lmde,ljdf,ifme->ij',t2_1_ab,
                                   t2_1_ab_ocee, eris_CEOE,optimize=True)
        M_ij_b -= 0.5 * lib.einsum('lmde,ljdf,imfe->ij',t2_1_ab,
                                   t2_1_ab_ocee, eris_COEE,optimize=True)
        M_ij_b -= 0.5 * lib.einsum('lmed,ljfd,imfe->ij',t2_1_ab,
                                   t2_1_ab_ocee, eris_COee ,optimize=True)

        M_ij_a += 0.5 * lib.einsum('mled,ilfd,jfme->ij',t2_1_ab,
                                   t2_1_ab_coee, eris_ceoe ,optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mled,ilfd,jmfe->ij',t2_1_ab,
                                   t2_1_ab_coee, eris_coee ,optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mlde,ildf,jmfe->ij',t2_1_ab,
                                   t2_1_ab_coee, eris_coEE ,optimize=True)

        M_ij_b += 0.5 * lib.einsum('lmde,lidf,jfme->ij',t2_1_ab,
                                   t2_1_ab_ocee, eris_CEOE ,optimize=True)
        M_ij_b -= 0.5 * lib.einsum('lmde,lidf,jmfe->ij',t2_1_ab,
                                   t2_1_ab_ocee, eris_COEE ,optimize=True)
        M_ij_b -= 0.5 * lib.einsum('lmed,lifd,jmfe->ij',t2_1_ab,
                                   t2_1_ab_ocee, eris_COee ,optimize=True)

        M_ij_a += 0.5 * lib.einsum('lmde,jnde,ilmn->ij',t2_1_ab ,
                                   t2_1_ab_coee,eris_coOO, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mled,njed,mnil->ij',t2_1_ab ,
                                   t2_1_ab_ocee,eris_ooCO, optimize=True)

        M_ij_a +=0.5 * lib.einsum('inde,lmde,jlnm->ij',t2_1_ab_coee,
                                  t2_1_ab,eris_coOO, optimize=True)

        M_ij_b +=0.5 * lib.einsum('nied,mled,nmjl->ij',t2_1_ab_ocee,
                                  t2_1_ab,eris_ooCO, optimize=True)

        M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_ccee , optimize=True)
        M_ij_a -=lib.einsum('mlfd,mled,jfie->ij',t2_1_ab, t2_1_ab, eris_cece , optimize=True)
        M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ccEE , optimize=True)
        M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_CCEE , optimize=True)
        M_ij_b -=lib.einsum('lmdf,lmde,jfie->ij',t2_1_ab, t2_1_ab, eris_CECE , optimize=True)
        M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_CCee , optimize=True)

        M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_ccoo, optimize=True)
        M_ij_a += lib.einsum('nled,mled,jmin->ij',t2_1_ab, t2_1_ab, eris_coco, optimize=True)
        M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ccOO, optimize=True)
        M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_CCOO, optimize=True)
        M_ij_b += lib.einsum('lnde,lmde,jmin->ij',t2_1_ab, t2_1_ab, eris_COCO, optimize=True)
        M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooCC, optimize=True)

        del t2_1_ab
        del t2_1_ab_coee
        del t2_1_ab_ocee

    M_ij = (M_ij_a, M_ij_b)
    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

    return M_ij


def get_diag(adc,M_ij=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    M_ij_a, M_ij_b = M_ij[0], M_ij[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = nvir_a * ncvs * (ncvs - 1) // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a
    n_doubles_bba_ecc = nvir_b * ncvs * ncvs
    n_doubles_bba_ecv = nvir_b * ncvs * nval_a
    n_doubles_bba_evc = nvir_b * nval_b * ncvs
    n_doubles_aab_ecc = nvir_a * ncvs * ncvs
    n_doubles_aab_ecv = nvir_a * ncvs * nval_b
    n_doubles_aab_evc = nvir_a * nval_a * ncvs
    n_doubles_bbb_ecc = nvir_b * ncvs * (ncvs - 1) // 2
    n_doubles_bbb_ecv = nvir_b * ncvs * nval_b

    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_aaa_ecv + n_doubles_bba_ecc + n_doubles_bba_ecv + \
        n_doubles_bba_evc + n_doubles_aab_ecc + n_doubles_aab_ecv + \
            n_doubles_aab_evc + n_doubles_bbb_ecc + n_doubles_bbb_ecv

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bba_ecc = f_aaa_ecv
    f_bba_ecc = s_bba_ecc + n_doubles_bba_ecc
    s_bba_ecv = f_bba_ecc
    f_bba_ecv = s_bba_ecv + n_doubles_bba_ecv
    s_bba_evc = f_bba_ecv
    f_bba_evc = s_bba_evc + n_doubles_bba_evc
    s_aab_ecc = f_bba_evc
    f_aab_ecc = s_aab_ecc + n_doubles_aab_ecc
    s_aab_ecv = f_aab_ecc
    f_aab_ecv = s_aab_ecv + n_doubles_aab_ecv
    s_aab_evc = f_aab_ecv
    f_aab_evc = s_aab_evc + n_doubles_aab_evc
    s_bbb_ecc = f_aab_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_n_a_ecc = D_n_a[:, :ncvs, :ncvs].copy()
    D_aij_a_ecv = D_n_a[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_a_ecc = D_n_a_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_n_b_ecc = D_n_b[:, :ncvs, :ncvs].copy()
    D_aij_b_ecv = D_n_b[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_b_ecc = D_n_b_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_ba = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bba = -d_a_b + d_ij_ba.reshape(-1)
    D_n_bba = D_n_bba.reshape((nvir_b,nocc_b,nocc_a))
    D_aij_bba_ecc = D_n_bba[:, :ncvs, :ncvs].reshape(-1)
    D_aij_bba_ecv = D_n_bba[:, :ncvs, ncvs:].reshape(-1)
    D_aij_bba_evc = D_n_bba[:, ncvs:, :ncvs].reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aab = -d_a_a + d_ij_ab.reshape(-1)
    #D_n_aba = D_n_aba.reshape(-1)
    D_n_aab = D_n_aab.reshape((nvir_a,nocc_a,nocc_b))
    D_aij_aab_ecc = D_n_aab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aab_ecv = D_n_aab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aab_evc = D_n_aab[:, ncvs:, :ncvs].reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    diag[s_a:f_a] = M_ij_a_diag.copy()
    diag[s_b:f_b] = M_ij_b_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s_aaa_ecc:f_aaa_ecc] = D_aij_a_ecc.copy()
    diag[s_aaa_ecv:f_aaa_ecv] = D_aij_a_ecv.copy()
    diag[s_bba_ecc:f_bba_ecc] = D_aij_bba_ecc.copy()
    diag[s_bba_ecv:f_bba_ecv] = D_aij_bba_ecv.copy()
    diag[s_bba_evc:f_bba_evc] = D_aij_bba_evc.copy()
    diag[s_aab_ecc:f_aab_ecc] = D_aij_aab_ecc.copy()
    diag[s_aab_ecv:f_aab_ecv] = D_aij_aab_ecv.copy()
    diag[s_aab_evc:f_aab_evc] = D_aij_aab_evc.copy()
    diag[s_bbb_ecc:f_bbb_ecc] = D_aij_b_ecc.copy()
    diag[s_bbb_ecv:f_bbb_ecv] = D_aij_b_ecv.copy()

    diag = -diag

    return diag

def matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs

    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = nvir_a * ncvs * (ncvs - 1) // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a
    n_doubles_bba_ecc = nvir_b * ncvs * ncvs
    n_doubles_bba_ecv = nvir_b * ncvs * nval_a
    n_doubles_bba_evc = nvir_b * nval_b * ncvs
    n_doubles_aab_ecc = nvir_a * ncvs * ncvs
    n_doubles_aab_ecv = nvir_a * ncvs * nval_b
    n_doubles_aab_evc = nvir_a * nval_a * ncvs
    n_doubles_bbb_ecc = nvir_b * ncvs * (ncvs - 1) // 2
    n_doubles_bbb_ecv = nvir_b * ncvs * nval_b
    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_aaa_ecv + n_doubles_bba_ecc + n_doubles_bba_ecv + \
        n_doubles_bba_evc + n_doubles_aab_ecc + n_doubles_aab_ecv + \
            n_doubles_aab_evc + n_doubles_bbb_ecc + n_doubles_bbb_ecv

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    if eris is None:
        eris = adc.transform_integrals()

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bba_ecc = f_aaa_ecv
    f_bba_ecc = s_bba_ecc + n_doubles_bba_ecc
    s_bba_ecv = f_bba_ecc
    f_bba_ecv = s_bba_ecv + n_doubles_bba_ecv
    s_bba_evc = f_bba_ecv
    f_bba_evc = s_bba_evc + n_doubles_bba_evc
    s_aab_ecc = f_bba_evc
    f_aab_ecc = s_aab_ecc + n_doubles_aab_ecc
    s_aab_ecv = f_aab_ecc
    f_aab_ecv = s_aab_ecv + n_doubles_aab_ecv
    s_aab_evc = f_aab_ecv
    f_aab_evc = s_aab_evc + n_doubles_aab_evc
    s_bbb_ecc = f_aab_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_n_a_ecc = D_n_a[:, :ncvs, :ncvs].copy()
    D_aij_a_ecv = D_n_a[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_a_ecc = D_n_a_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_n_b_ecc = D_n_b[:, :ncvs, :ncvs].copy()
    D_aij_b_ecv = D_n_b[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_b_ecc = D_n_b_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_ba = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bba = -d_a_b + d_ij_ba.reshape(-1)
    D_n_bba = D_n_bba.reshape((nvir_b,nocc_b,nocc_a))
    D_aij_bba_ecc = D_n_bba[:, :ncvs, :ncvs].reshape(-1)
    D_aij_bba_ecv = D_n_bba[:, :ncvs, ncvs:].reshape(-1)
    D_aij_bba_evc = D_n_bba[:, ncvs:, :ncvs].reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aab = -d_a_a + d_ij_ab.reshape(-1)
    D_n_aab = D_n_aab.reshape((nvir_a,nocc_a,nocc_b))
    D_aij_aab_ecc = D_n_aab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aab_ecv = D_n_aab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aab_evc = D_n_aab[:, ncvs:, :ncvs].reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij

    def sigma_(r):

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa_ecc = r[s_aaa_ecc:f_aaa_ecc]
        r_aaa_ecv = r[s_aaa_ecv:f_aaa_ecv]
        r_bba_ecc = r[s_bba_ecc:f_bba_ecc]
        r_bba_ecv = r[s_bba_ecv:f_bba_ecv]
        r_bba_evc = r[s_bba_evc:f_bba_evc]
        r_aab_ecc = r[s_aab_ecc:f_aab_ecc]
        r_aab_ecv = r[s_aab_ecv:f_aab_ecv]
        r_aab_evc = r[s_aab_evc:f_aab_evc]
        r_bbb_ecc = r[s_bbb_ecc:f_bbb_ecc]
        r_bbb_ecv = r[s_bbb_ecv:f_bbb_ecv]

        r_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)
        r_bbb_ecc = r_bbb_ecc.reshape(nvir_b,-1)

        r_aaa_ecc_u = None
        r_aaa_ecc_u = np.zeros((nvir_a,ncvs,ncvs))
        r_aaa_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_aaa_ecc.copy()
        r_aaa_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_aaa_ecc.copy()

        r_bbb_ecc_u = None
        r_bbb_ecc_u = np.zeros((nvir_b,ncvs,ncvs))
        r_bbb_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_bbb_ecc.copy()
        r_bbb_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_bbb_ecc.copy()

        r_aaa_ecv = r_aaa_ecv.reshape(nvir_a,ncvs,nval_a).copy()
        r_bba_ecc = r_bba_ecc.reshape(nvir_b,ncvs,ncvs)  .copy()
        r_bba_ecv = r_bba_ecv.reshape(nvir_b,ncvs,nval_a).copy()
        r_bba_evc = r_bba_evc.reshape(nvir_b,nval_b,ncvs).copy()
        r_aab_ecc = r_aab_ecc.reshape(nvir_a,ncvs,ncvs)  .copy()
        r_aab_ecv = r_aab_ecv.reshape(nvir_a,ncvs,nval_b).copy()
        r_aab_evc = r_aab_evc.reshape(nvir_a,nval_a,ncvs).copy()
        r_bbb_ecv = r_bbb_ecv.reshape(nvir_b,ncvs,nval_b).copy()

        eris_cecc = eris.ovoo[:ncvs,:,:ncvs,:ncvs].copy()
        eris_vecc = eris.ovoo[ncvs:,:,:ncvs,:ncvs].copy()
        eris_CECC = eris.OVOO[:ncvs,:,:ncvs,:ncvs].copy()
        eris_VECC = eris.OVOO[ncvs:,:,:ncvs,:ncvs].copy()
        eris_ceCC = eris.ovOO[:ncvs,:,:ncvs,:ncvs].copy()
        eris_veCC = eris.ovOO[ncvs:,:,:ncvs,:ncvs].copy()
        eris_CEcc = eris.OVoo[:ncvs,:,:ncvs,:ncvs].copy()
        eris_VEcc = eris.OVoo[ncvs:,:,:ncvs,:ncvs].copy()

        eris_cecv = eris.ovoo[:ncvs,:,:ncvs,ncvs:].copy()
        eris_CECV = eris.OVOO[:ncvs,:,:ncvs,ncvs:].copy()
        eris_ceCV = eris.ovOO[:ncvs,:,:ncvs,ncvs:].copy()
        eris_CEcv = eris.OVoo[:ncvs,:,:ncvs,ncvs:].copy()

############ ADC(2) ij block ############################

        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)

############# ADC(2) i - kja block #########################

        s[s_a:f_a] += 0.5*lib.einsum('JaKI,aJK->I', eris_cecc, r_aaa_ecc_u, optimize=True)
        s[s_a:f_a] -= 0.5*lib.einsum('KaJI,aJK->I', eris_cecc, r_aaa_ecc_u, optimize=True)
        s[s_a:f_a] += lib.einsum('JaIk,aJk->I', eris_cecv, r_aaa_ecv, optimize=True)
        s[s_a:f_a] -= lib.einsum('kaJI,aJk->I', eris_vecc, r_aaa_ecv, optimize=True)
        s[s_a:f_a] += lib.einsum('JaKI,aJK->I', eris_CEcc, r_bba_ecc, optimize=True)
        s[s_a:f_a] += lib.einsum('JaIk,aJk->I', eris_CEcv, r_bba_ecv, optimize=True)
        s[s_a:f_a] += lib.einsum('jaKI,ajK->I', eris_VEcc, r_bba_evc, optimize=True)

        s[s_b:f_b] += 0.5*lib.einsum('JaKI,aJK->I', eris_CECC, r_bbb_ecc_u, optimize=True)
        s[s_b:f_b] -= 0.5*lib.einsum('KaJI,aJK->I', eris_CECC, r_bbb_ecc_u, optimize=True)
        s[s_b:f_b] += lib.einsum('JaIk,aJk->I', eris_CECV, r_bbb_ecv, optimize=True)
        s[s_b:f_b] -= lib.einsum('kaJI,aJk->I', eris_VECC, r_bbb_ecv, optimize=True)
        s[s_b:f_b] += lib.einsum('JaKI,aJK->I', eris_ceCC, r_aab_ecc, optimize=True)
        s[s_b:f_b] += lib.einsum('JaIk,aJk->I', eris_ceCV, r_aab_ecv, optimize=True)
        s[s_b:f_b] += lib.einsum('jaKI,ajK->I', eris_veCC, r_aab_evc, optimize=True)

############## ADC(2) ajk - i block ############################

        temp_aaa_ecc = lib.einsum('JaKI,I->aJK', eris_cecc, r_a, optimize=True)
        temp_aaa_ecc -= lib.einsum('KaJI,I->aJK', eris_cecc, r_a, optimize=True)
        s[s_aaa_ecc:f_aaa_ecc] += temp_aaa_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] += lib.einsum('JaIk,I->aJk',
                                             eris_cecv, r_a, optimize=True).reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] -= lib.einsum('kaJI,I->aJk',
                                             eris_vecc, r_a, optimize=True).reshape(-1)
        s[s_bba_ecc:f_bba_ecc] += lib.einsum('JaIK,I->aJK',
                                             eris_CEcc, r_a, optimize=True).reshape(-1)
        s[s_bba_ecv:f_bba_ecv] += lib.einsum('JaIk,I->aJk',
                                             eris_CEcv, r_a, optimize=True).reshape(-1)
        s[s_bba_evc:f_bba_evc] += lib.einsum('jaIK,I->ajK',
                                             eris_VEcc, r_a, optimize=True).reshape(-1)
        s[s_aab_ecc:f_aab_ecc] += lib.einsum('JaKI,I->aJK',
                                             eris_ceCC, r_b, optimize=True).reshape(-1)
        s[s_aab_ecv:f_aab_ecv] += lib.einsum('JaIk,I->aJk',
                                             eris_ceCV, r_b, optimize=True).reshape(-1)
        s[s_aab_evc:f_aab_evc] += lib.einsum('jaKI,I->ajK',
                                             eris_veCC, r_b, optimize=True).reshape(-1)
        temp_bbb_ecc = lib.einsum('JaKI,I->aJK', eris_CECC, r_b, optimize=True)
        temp_bbb_ecc -= lib.einsum('KaJI,I->aJK', eris_CECC, r_b, optimize=True)
        s[s_bbb_ecc:f_bbb_ecc] += temp_bbb_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] += lib.einsum('JaIk,I->aJk',
                                             eris_CECV, r_b, optimize=True).reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] -= lib.einsum('kaJI,I->aJk',
                                             eris_VECC, r_b, optimize=True).reshape(-1)

############ ADC(2) ajk - bil block ############################

        r_aaa_ecc = r_aaa_ecc.reshape(-1)
        r_bbb_ecc = r_bbb_ecc.reshape(-1)

        s[s_aaa_ecc:f_aaa_ecc] += D_aij_a_ecc * r_aaa_ecc
        s[s_aaa_ecv:f_aaa_ecv] += D_aij_a_ecv * r_aaa_ecv.reshape(-1)
        s[s_bba_ecc:f_bba_ecc] += D_aij_bba_ecc * r_bba_ecc.reshape(-1)
        s[s_bba_ecv:f_bba_ecv] += D_aij_bba_ecv * r_bba_ecv.reshape(-1)
        s[s_bba_evc:f_bba_evc] += D_aij_bba_evc * r_bba_evc.reshape(-1)
        s[s_aab_ecc:f_aab_ecc] += D_aij_aab_ecc * r_aab_ecc.reshape(-1)
        s[s_aab_ecv:f_aab_ecv] += D_aij_aab_ecv * r_aab_ecv.reshape(-1)
        s[s_aab_evc:f_aab_evc] += D_aij_aab_evc * r_aab_evc.reshape(-1)
        s[s_bbb_ecc:f_bbb_ecc] += D_aij_b_ecc * r_bbb_ecc
        s[s_bbb_ecv:f_bbb_ecv] += D_aij_b_ecv * r_bbb_ecv.reshape(-1)

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_ovov = np.array(eris.ovvo).transpose(0,1,3,2)
            eris_OVOV = np.array(eris.OVVO).transpose(0,1,3,2)
            eris_OVov = np.array(eris.OVvo).transpose(0,1,3,2)

            eris_cccc = eris.oooo[:ncvs,:ncvs,:ncvs,:ncvs].copy()
            eris_cccv = eris.oooo[:ncvs,:ncvs,:ncvs,ncvs:].copy()
            eris_ccvv = eris.oooo[:ncvs,:ncvs,ncvs:,ncvs:].copy()
            eris_CCCC = eris.OOOO[:ncvs,:ncvs,:ncvs,:ncvs].copy()
            eris_CCCV = eris.OOOO[:ncvs,:ncvs,:ncvs,ncvs:].copy()
            eris_CCVV = eris.OOOO[:ncvs,:ncvs,ncvs:,ncvs:].copy()
            eris_ccCC = eris.ooOO[:ncvs,:ncvs,:ncvs,:ncvs].copy()
            eris_ccCV = eris.ooOO[:ncvs,:ncvs,:ncvs,ncvs:].copy()
            eris_vvCC = eris.ooOO[ncvs:,ncvs:,:ncvs,:ncvs].copy()
            eris_ccVV = eris.ooOO[:ncvs,:ncvs,ncvs:,ncvs:].copy()
            eris_ccee = eris.oovv[:ncvs,:ncvs,:,:].copy()
            eris_vvee = eris.oovv[ncvs:,ncvs:,:,:].copy()
            eris_CCEE = eris.OOVV[:ncvs,:ncvs,:,:].copy()
            eris_VVEE = eris.OOVV[ncvs:,ncvs:,:,:].copy()
            eris_ccEE = eris.ooVV[:ncvs,:ncvs,:,:].copy()
            eris_vvEE = eris.ooVV[ncvs:,ncvs:,:,:].copy()
            eris_CCee = eris.OOvv[:ncvs,:ncvs,:,:].copy()
            eris_VVee = eris.OOvv[ncvs:,ncvs:,:,:].copy()

            eris_cvcv = eris.oooo[:ncvs,ncvs:,:ncvs,ncvs:].copy()
            eris_CVCV = eris.OOOO[:ncvs,ncvs:,:ncvs,ncvs:].copy()
            eris_cvCC = eris.ooOO[:ncvs,ncvs:,:ncvs,:ncvs].copy()
            eris_cvCV = eris.ooOO[:ncvs,ncvs:,:ncvs,ncvs:].copy()
            eris_cece = eris_ovov[:ncvs,:,:ncvs,:].copy()
            eris_vece = eris_ovov[ncvs:,:,:ncvs,:].copy()
            eris_veve = eris_ovov[ncvs:,:,ncvs:,:].copy()
            eris_CECE = eris_OVOV[:ncvs,:,:ncvs,:].copy()
            eris_VECE = eris_OVOV[ncvs:,:,:ncvs,:].copy()
            eris_VEVE = eris_OVOV[ncvs:,:,ncvs:,:].copy()
            eris_CEce = eris_OVov[:ncvs,:,:ncvs,:].copy()
            eris_VEce = eris_OVov[ncvs:,:,:ncvs,:].copy()
            eris_CEve = eris_OVov[:ncvs,:,ncvs:,:].copy()
            eris_VEve = eris_OVov[ncvs:,:,ncvs:,:].copy()
            eris_cvee = eris.oovv[:ncvs,ncvs:,:,:].copy()
            eris_CVEE = eris.OOVV[:ncvs,ncvs:,:,:].copy()
            eris_cvEE = eris.ooVV[:ncvs,ncvs:,:,:].copy()
            eris_CVee = eris.OOvv[:ncvs,ncvs:,:,:].copy()

            temp_ecc =  0.5*lib.einsum('JLKI,aIL->aJK',eris_cccc,r_aaa_ecc_u ,optimize=True)
            temp_ecc -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_cccc,r_aaa_ecc_u ,optimize=True)
            temp_ecc +=     lib.einsum('KIJl,aIl->aJK',eris_cccv,r_aaa_ecv ,optimize=True)
            temp_ecc -=     lib.einsum('JIKl,aIl->aJK',eris_cccv,r_aaa_ecv ,optimize=True)
            temp_ecv =  0.5*lib.einsum('JLIk,aIL->aJk',eris_cccv,r_aaa_ecc_u ,optimize=True)
            temp_ecv -= 0.5*lib.einsum('JILk,aIL->aJk',eris_cccv,r_aaa_ecc_u ,optimize=True)
            temp_ecv +=     lib.einsum('JlIk,aIl->aJk',eris_cvcv,r_aaa_ecv ,optimize=True)
            temp_ecv -=     lib.einsum('JIkl,aIl->aJk',eris_ccvv,r_aaa_ecv ,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

            temp_ecc =  0.5*lib.einsum('JLKI,aIL->aJK',eris_CCCC,r_bbb_ecc_u ,optimize=True)
            temp_ecc -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_CCCC,r_bbb_ecc_u ,optimize=True)
            temp_ecc +=     lib.einsum('KIJl,aIl->aJK',eris_CCCV,r_bbb_ecv ,optimize=True)
            temp_ecc -=     lib.einsum('JIKl,aIl->aJK',eris_CCCV,r_bbb_ecv ,optimize=True)
            temp_ecv =  0.5*lib.einsum('JLIk,aIL->aJk',eris_CCCV,r_bbb_ecc_u ,optimize=True)
            temp_ecv -= 0.5*lib.einsum('JILk,aIL->aJk',eris_CCCV,r_bbb_ecc_u ,optimize=True)
            temp_ecv +=     lib.einsum('JlIk,aIl->aJk',eris_CVCV,r_bbb_ecv ,optimize=True)
            temp_ecv -=     lib.einsum('JIkl,aIl->aJk',eris_CCVV,r_bbb_ecv ,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

            s[s_bba_ecc:f_bba_ecc] -= lib.einsum('KLJI,aIL->aJK',
                                                 eris_ccCC,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -= lib.einsum('KIJl,alI->aJK',
                                                 eris_ccCV,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -= lib.einsum('KlJI,aIl->aJK',
                                                 eris_cvCC,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -= lib.einsum('LkJI,aIL->aJk',
                                                 eris_cvCC,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -= lib.einsum('IkJl,alI->aJk',
                                                 eris_cvCV,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -= lib.einsum('klJI,aIl->aJk',
                                                 eris_vvCC,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -= lib.einsum('KLIj,aIL->ajK',
                                                 eris_ccCV,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -= lib.einsum('KIjl,alI->ajK',
                                                 eris_ccVV,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -= lib.einsum('KlIj,aIl->ajK',
                                                 eris_cvCV,r_bba_ecv,optimize=True).reshape(-1)

            s[s_aab_ecc:f_aab_ecc] -= lib.einsum('JIKL,aIL->aJK',
                                                 eris_ccCC,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= lib.einsum('JlKI,alI->aJK',
                                                 eris_cvCC,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= lib.einsum('JIKl,aIl->aJK',
                                                 eris_ccCV,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= lib.einsum('JILk,aIL->aJk',
                                                 eris_ccCV,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= lib.einsum('JlIk,alI->aJk',
                                                 eris_cvCV,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= lib.einsum('JIkl,aIl->aJk',
                                                 eris_ccVV,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= lib.einsum('IjKL,aIL->ajK',
                                                 eris_cvCC,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= lib.einsum('jlKI,alI->ajK',
                                                 eris_vvCC,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= lib.einsum('IjKl,aIl->ajK',
                                                 eris_cvCV,r_aab_ecv,optimize=True).reshape(-1)

            temp_ecc =  0.5*lib.einsum('KLba,bJL->aJK',eris_ccee,r_aaa_ecc_u,optimize=True)
            temp_ecc -= 0.5*lib.einsum('KaLb,bJL->aJK',eris_cece,r_aaa_ecc_u,optimize=True)
            temp_ecc += 0.5*lib.einsum('LbKa,bLJ->aJK',eris_CEce,r_bba_ecc,optimize=True)
            temp_ecc += 0.5*lib.einsum('Klba,bJl->aJK',eris_cvee,r_aaa_ecv,optimize=True)
            temp_ecc -= 0.5*lib.einsum('lbKa,bJl->aJK',eris_vece,r_aaa_ecv,optimize=True)
            temp_ecc += 0.5*lib.einsum('lbKa,blJ->aJK',eris_VEce,r_bba_evc,optimize=True)
            temp_ecv =  0.5*lib.einsum('Lkba,bJL->aJk',eris_cvee,r_aaa_ecc_u,optimize=True)
            temp_ecv -= 0.5*lib.einsum('kaLb,bJL->aJk',eris_vece,r_aaa_ecc_u,optimize=True)
            temp_ecv += 0.5*lib.einsum('Lbka,bLJ->aJk',eris_CEve,r_bba_ecc,optimize=True)
            temp_ecv += 0.5*lib.einsum('klba,bJl->aJk',eris_vvee,r_aaa_ecv,optimize=True)
            temp_ecv -= 0.5*lib.einsum('kalb,bJl->aJk',eris_veve,r_aaa_ecv,optimize=True)
            temp_ecv += 0.5*lib.einsum('lbka,blJ->aJk',eris_VEve,r_bba_evc,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

            s[s_bba_ecc:f_bba_ecc] += 0.5* \
                lib.einsum('KLba,bJL->aJK',eris_ccEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] += 0.5* \
                lib.einsum('Klba,bJl->aJK',eris_cvEE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += 0.5* \
                lib.einsum('Lkba,bJL->aJk',eris_cvEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += 0.5* \
                lib.einsum('klba,bJl->aJk',eris_vvEE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] += 0.5* \
                lib.einsum('KLba,bjL->ajK',eris_ccEE,r_bba_evc,optimize=True).reshape(-1)

            temp_1_ecc =  0.5*lib.einsum('KLba,bJL->aJK',eris_CCEE,r_bbb_ecc_u,optimize=True)
            temp_1_ecc -= 0.5*lib.einsum('KaLb,bJL->aJK',eris_CECE,r_bbb_ecc_u,optimize=True)
            temp_1_ecc += 0.5*lib.einsum('KaLb,bLJ->aJK',eris_CEce,r_aab_ecc,optimize=True)
            temp_1_ecc += 0.5*lib.einsum('Klba,bJl->aJK',eris_CVEE,r_bbb_ecv,optimize=True)
            temp_1_ecc -= 0.5*lib.einsum('lbKa,bJl->aJK',eris_VECE,r_bbb_ecv,optimize=True)
            temp_1_ecc += 0.5*lib.einsum('Kalb,blJ->aJK',eris_CEve,r_aab_evc,optimize=True)
            temp_1_ecv =  0.5*lib.einsum('Lkba,bJL->aJk',eris_CVEE,r_bbb_ecc_u,optimize=True)
            temp_1_ecv -= 0.5*lib.einsum('kaLb,bJL->aJk',eris_VECE,r_bbb_ecc_u,optimize=True)
            temp_1_ecv += 0.5*lib.einsum('kaLb,bLJ->aJk',eris_VEce,r_aab_ecc,optimize=True)
            temp_1_ecv += 0.5*lib.einsum('klba,bJl->aJk',eris_VVEE,r_bbb_ecv,optimize=True)
            temp_1_ecv -= 0.5*lib.einsum('kalb,bJl->aJk',eris_VEVE,r_bbb_ecv,optimize=True)
            temp_1_ecv += 0.5*lib.einsum('kalb,blJ->aJk',eris_VEve,r_aab_evc,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_1_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_1_ecv.reshape(-1)


            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('KLba,bJL->aJK',eris_CCee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('Klba,bJl->aJK',eris_CVee,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += 0.5* \
                lib.einsum('Lkba,bJL->aJk',eris_CVee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += 0.5* \
                lib.einsum('klba,bJl->aJk',eris_VVee,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('KLba,bjL->ajK',eris_CCee,r_aab_evc,optimize=True).reshape(-1)

            temp_ecc =  -0.5*lib.einsum('JLba,bKL->aJK',eris_ccee,r_aaa_ecc_u,optimize=True)
            temp_ecc +=  0.5*lib.einsum('JaLb,bKL->aJK',eris_cece,r_aaa_ecc_u,optimize=True)
            temp_ecc -=  0.5*lib.einsum('LbJa,bLK->aJK',eris_CEce,r_bba_ecc,optimize=True)
            temp_ecc += -0.5*lib.einsum('Jlba,bKl->aJK',eris_cvee,r_aaa_ecv,optimize=True)
            temp_ecc +=  0.5*lib.einsum('lbJa,bKl->aJK',eris_vece,r_aaa_ecv,optimize=True)
            temp_ecc -=  0.5*lib.einsum('lbJa,blK->aJK',eris_VEce,r_bba_evc,optimize=True)
            temp_ecv =   0.5*lib.einsum('JLba,bLk->aJk',eris_ccee,r_aaa_ecv,optimize=True)
            temp_ecv -=  0.5*lib.einsum('JaLb,bLk->aJk',eris_cece,r_aaa_ecv,optimize=True)
            temp_ecv -=  0.5*lib.einsum('LbJa,bLk->aJk',eris_CEce,r_bba_ecv,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)


            s[s_bba_ecc:f_bba_ecc] +=  0.5* \
                lib.einsum('JaLb,bKL->aJK',eris_CEce,r_aaa_ecc_u,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] +=  0.5* \
                lib.einsum('JLba,bLK->aJK',eris_CCEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -=  0.5* \
                lib.einsum('JaLb,bLK->aJK',eris_CECE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] +=  0.5* \
                lib.einsum('Jalb,bKl->aJK',eris_CEve,r_aaa_ecv,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] +=  0.5* \
                lib.einsum('Jlba,blK->aJK',eris_CVEE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -=  0.5* \
                lib.einsum('lbJa,blK->aJK',eris_VECE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += -0.5* \
                lib.einsum('JaLb,bLk->aJk',eris_CEce,r_aaa_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] +=  0.5* \
                lib.einsum('JLba,bLk->aJk',eris_CCEE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -=  0.5* \
                lib.einsum('JaLb,bLk->aJk',eris_CECE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] +=  0.5* \
                lib.einsum('jaLb,bKL->ajK',eris_VEce,r_aaa_ecc_u,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] +=  0.5* \
                lib.einsum('Ljba,bLK->ajK',eris_CVEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -=  0.5* \
                lib.einsum('jaLb,bLK->ajK',eris_VECE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] +=  0.5* \
                lib.einsum('jalb,bKl->ajK',eris_VEve,r_aaa_ecv,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] +=  0.5* \
                lib.einsum('jlba,blK->ajK',eris_VVEE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -=  0.5* \
                lib.einsum('jalb,blK->ajK',eris_VEVE,r_bba_evc,optimize=True).reshape(-1)

            temp_ecc = -0.5*lib.einsum('JLba,bKL->aJK',eris_CCEE,r_bbb_ecc_u,optimize=True)
            temp_ecc += 0.5*lib.einsum('JaLb,bKL->aJK',eris_CECE,r_bbb_ecc_u,optimize=True)
            temp_ecc -= 0.5*lib.einsum('JaLb,bLK->aJK',eris_CEce,r_aab_ecc,optimize=True)
            temp_ecc +=-0.5*lib.einsum('Jlba,bKl->aJK',eris_CVEE,r_bbb_ecv,optimize=True)
            temp_ecc += 0.5*lib.einsum('lbJa,bKl->aJK',eris_VECE,r_bbb_ecv,optimize=True)
            temp_ecc -= 0.5*lib.einsum('Jalb,blK->aJK',eris_CEve,r_aab_evc,optimize=True)
            temp_ecv =  0.5*lib.einsum('JLba,bLk->aJk',eris_CCEE,r_bbb_ecv,optimize=True)
            temp_ecv +=-0.5*lib.einsum('JaLb,bLk->aJk',eris_CECE,r_bbb_ecv,optimize=True)
            temp_ecv -= 0.5*lib.einsum('JaLb,bLk->aJk',eris_CEce,r_aab_ecv,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('JLba,bLK->aJK',eris_ccee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= 0.5* \
                lib.einsum('JaLb,bLK->aJK',eris_cece,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('LbJa,bKL->aJK',eris_CEce,r_bbb_ecc_u,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('Jlba,blK->aJK',eris_cvee,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= 0.5* \
                lib.einsum('lbJa,blK->aJK',eris_vece,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('lbJa,bKl->aJK',eris_VEce,r_bbb_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += 0.5* \
                lib.einsum('JLba,bLk->aJk',eris_ccee,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= 0.5* \
                lib.einsum('JaLb,bLk->aJk',eris_cece,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] +=-0.5* \
                lib.einsum('LbJa,bLk->aJk',eris_CEce,r_bbb_ecv,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('Ljba,bLK->ajK',eris_cvee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= 0.5* \
                lib.einsum('jaLb,bLK->ajK',eris_vece,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('Lbja,bKL->ajK',eris_CEve,r_bbb_ecc_u,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('jlba,blK->ajK',eris_vvee,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= 0.5* \
                lib.einsum('jalb,blK->ajK',eris_veve,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('lbja,bKl->ajK',eris_VEve,r_bbb_ecv,optimize=True).reshape(-1)

            temp_ecc = -0.5*lib.einsum('KIba,bIJ->aJK',eris_ccee,r_aaa_ecc_u,optimize=True)
            temp_ecc += 0.5*lib.einsum('KaIb,bIJ->aJK',eris_cece,r_aaa_ecc_u,optimize=True)
            temp_ecc += 0.5*lib.einsum('IbKa,bIJ->aJK',eris_CEce,r_bba_ecc,optimize=True)
            temp_ecc += 0.5*lib.einsum('Kiba,bJi->aJK',eris_cvee,r_aaa_ecv,optimize=True)
            temp_ecc +=-0.5*lib.einsum('ibKa,bJi->aJK',eris_vece,r_aaa_ecv,optimize=True)
            temp_ecc += 0.5*lib.einsum('ibKa,biJ->aJK',eris_VEce,r_bba_evc,optimize=True)
            temp_ecv = -0.5*lib.einsum('Ikba,bIJ->aJk',eris_cvee,r_aaa_ecc_u,optimize=True)
            temp_ecv += 0.5*lib.einsum('kaIb,bIJ->aJk',eris_vece,r_aaa_ecc_u,optimize=True)
            temp_ecv += 0.5*lib.einsum('Ibka,bIJ->aJk',eris_CEve,r_bba_ecc,optimize=True)
            temp_ecv += 0.5*lib.einsum('kiba,bJi->aJk',eris_vvee,r_aaa_ecv,optimize=True)
            temp_ecv +=-0.5*lib.einsum('kaib,bJi->aJk',eris_veve,r_aaa_ecv,optimize=True)
            temp_ecv += 0.5*lib.einsum('ibka,biJ->aJk',eris_VEve,r_bba_evc,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

            s[s_bba_ecc:f_bba_ecc] += 0.5* \
                lib.einsum('KIba,bJI->aJK',eris_ccEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] += 0.5* \
                lib.einsum('Kiba,bJi->aJK',eris_cvEE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += 0.5* \
                lib.einsum('Ikba,bJI->aJk',eris_cvEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += 0.5* \
                lib.einsum('kiba,bJi->aJk',eris_vvEE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] += 0.5* \
                lib.einsum('KIba,bjI->ajK',eris_ccEE,r_bba_evc,optimize=True).reshape(-1)

            temp_ecc = -0.5*lib.einsum('KIba,bIJ->aJK',eris_CCEE,r_bbb_ecc_u,optimize=True)
            temp_ecc += 0.5*lib.einsum('KaIb,bIJ->aJK',eris_CECE,r_bbb_ecc_u,optimize=True)
            temp_ecc += 0.5*lib.einsum('KaIb,bIJ->aJK',eris_CEce,r_aab_ecc,optimize=True)
            temp_ecc += 0.5*lib.einsum('Kiba,bJi->aJK',eris_CVEE,r_bbb_ecv,optimize=True)
            temp_ecc +=-0.5*lib.einsum('ibKa,bJi->aJK',eris_VECE,r_bbb_ecv,optimize=True)
            temp_ecc += 0.5*lib.einsum('Kaib,biJ->aJK',eris_CEve,r_aab_evc,optimize=True)
            temp_ecv = -0.5*lib.einsum('Ikba,bIJ->aJk',eris_CVEE,r_bbb_ecc_u,optimize=True)
            temp_ecv += 0.5*lib.einsum('kaIb,bIJ->aJk',eris_VECE,r_bbb_ecc_u,optimize=True)
            temp_ecv += 0.5*lib.einsum('kaIb,bIJ->aJk',eris_VEce,r_aab_ecc,optimize=True)
            temp_ecv += 0.5*lib.einsum('kiba,bJi->aJk',eris_VVEE,r_bbb_ecv,optimize=True)
            temp_ecv +=-0.5*lib.einsum('kaib,bJi->aJk',eris_VEVE,r_bbb_ecv,optimize=True)
            temp_ecv += 0.5*lib.einsum('kaib,biJ->aJk',eris_VEve,r_aab_evc,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('KIba,bJI->aJK',eris_CCee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('Kiba,bJi->aJK',eris_CVee,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += 0.5* \
                lib.einsum('Ikba,bJI->aJk',eris_CVee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += 0.5* \
                lib.einsum('kiba,bJi->aJk',eris_VVee,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('KIba,bjI->ajK',eris_CCee,r_aab_evc,optimize=True).reshape(-1)

            temp_ecc =   0.5*lib.einsum('JIba,bIK->aJK',eris_ccee,r_aaa_ecc_u,optimize=True)
            temp_ecc -=  0.5*lib.einsum('JaIb,bIK->aJK',eris_cece,r_aaa_ecc_u,optimize=True)
            temp_ecc -=  0.5*lib.einsum('IbJa,bIK->aJK',eris_CEce,r_bba_ecc,optimize=True)
            temp_ecc += -0.5*lib.einsum('Jiba,bKi->aJK',eris_cvee,r_aaa_ecv,optimize=True)
            temp_ecc -= -0.5*lib.einsum('ibJa,bKi->aJK',eris_vece,r_aaa_ecv,optimize=True)
            temp_ecc -=  0.5*lib.einsum('ibJa,biK->aJK',eris_VEce,r_bba_evc,optimize=True)
            temp_ecv =   0.5*lib.einsum('JIba,bIk->aJk',eris_ccee,r_aaa_ecv,optimize=True)
            temp_ecv -=  0.5*lib.einsum('JaIb,bIk->aJk',eris_cece,r_aaa_ecv,optimize=True)
            temp_ecv -=  0.5*lib.einsum('IbJa,bIk->aJk',eris_CEce,r_bba_ecv,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

            s[s_bba_ecc:f_bba_ecc] += 0.5* \
                lib.einsum('JIba,bIK->aJK',eris_CCEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -= 0.5* \
                lib.einsum('JaIb,bIK->aJK',eris_CECE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -= 0.5* \
                lib.einsum('JaIb,bIK->aJK',eris_CEce,r_aaa_ecc_u,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] += 0.5* \
                lib.einsum('Jiba,biK->aJK',eris_CVEE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -= 0.5* \
                lib.einsum('ibJa,biK->aJK',eris_VECE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_ecc:f_bba_ecc] -=-0.5* \
                lib.einsum('Jaib,bKi->aJK',eris_CEve,r_aaa_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += 0.5* \
                lib.einsum('JIba,bIk->aJk',eris_CCEE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -= 0.5* \
                lib.einsum('JaIb,bIk->aJk',eris_CECE,r_bba_ecv,optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -= 0.5* \
                lib.einsum('JaIb,bIk->aJk',eris_CEce,r_aaa_ecv,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] += 0.5* \
                lib.einsum('Ijba,bIK->ajK',eris_CVEE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -= 0.5* \
                lib.einsum('jaIb,bIK->ajK',eris_VECE,r_bba_ecc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -= 0.5* \
                lib.einsum('jaIb,bIK->ajK',eris_VEce,r_aaa_ecc_u,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] += 0.5* \
                lib.einsum('jiba,biK->ajK',eris_VVEE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -= 0.5* \
                lib.einsum('jaib,biK->ajK',eris_VEVE,r_bba_evc,optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] -=-0.5* \
                lib.einsum('jaib,bKi->ajK',eris_VEve,r_aaa_ecv,optimize=True).reshape(-1)

            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('JIba,bIK->aJK',eris_ccee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= 0.5* \
                lib.einsum('JaIb,bIK->aJK',eris_cece,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= 0.5* \
                lib.einsum('IbJa,bIK->aJK',eris_CEce,r_bbb_ecc_u,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] += 0.5* \
                lib.einsum('Jiba,biK->aJK',eris_cvee,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -= 0.5* \
                lib.einsum('ibJa,biK->aJK',eris_vece,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_ecc:f_aab_ecc] -=-0.5* \
                lib.einsum('ibJa,bKi->aJK',eris_VEce,r_bbb_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += 0.5* \
                lib.einsum('JIba,bIk->aJk',eris_ccee,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= 0.5* \
                lib.einsum('JaIb,bIk->aJk',eris_cece,r_aab_ecv,optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= 0.5* \
                lib.einsum('IbJa,bIk->aJk',eris_CEce,r_bbb_ecv,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('Ijba,bIK->ajK',eris_cvee,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= 0.5* \
                lib.einsum('jaIb,bIK->ajK',eris_vece,r_aab_ecc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= 0.5* \
                lib.einsum('Ibja,bIK->ajK',eris_CEve,r_bbb_ecc_u,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += 0.5* \
                lib.einsum('jiba,biK->ajK',eris_vvee,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -= 0.5* \
                lib.einsum('jaib,biK->ajK',eris_veve,r_aab_evc,optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] -=-0.5* \
                lib.einsum('ibja,bKi->ajK',eris_VEve,r_bbb_ecv,optimize=True).reshape(-1)

            temp_ecc =   0.5*lib.einsum('JIba,bIK->aJK',eris_CCEE,r_bbb_ecc_u,optimize=True)
            temp_ecc -=  0.5*lib.einsum('JaIb,bIK->aJK',eris_CECE,r_bbb_ecc_u,optimize=True)
            temp_ecc -=  0.5*lib.einsum('JaIb,bIK->aJK',eris_CEce,r_aab_ecc,optimize=True)
            temp_ecc += -0.5*lib.einsum('Jiba,bKi->aJK',eris_CVEE,r_bbb_ecv,optimize=True)
            temp_ecc -= -0.5*lib.einsum('ibJa,bKi->aJK',eris_VECE,r_bbb_ecv,optimize=True)
            temp_ecc -=  0.5*lib.einsum('Jaib,biK->aJK',eris_CEve,r_aab_evc,optimize=True)
            temp_ecv =   0.5*lib.einsum('JIba,bIk->aJk',eris_CCEE,r_bbb_ecv,optimize=True)
            temp_ecv -=  0.5*lib.einsum('JaIb,bIk->aJk',eris_CECE,r_bbb_ecv,optimize=True)
            temp_ecv -=  0.5*lib.einsum('JaIb,bIk->aJk',eris_CEce,r_aab_ecv,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

        if (method == "adc(3)"):

            eris_oecc = eris.ovoo[:,:,:ncvs,:ncvs].copy()
            eris_oecv = eris.ovoo[:,:,:ncvs,ncvs:].copy()
            eris_OECC = eris.OVOO[:,:,:ncvs,:ncvs].copy()
            eris_OECV = eris.OVOO[:,:,:ncvs,ncvs:].copy()
            eris_OEcc = eris.OVoo[:,:,:ncvs,:ncvs].copy()
            eris_OEcv = eris.OVoo[:,:,:ncvs,ncvs:].copy()
            eris_oeCC = eris.ovOO[:,:,:ncvs,:ncvs].copy()
            eris_oeCV = eris.ovOO[:,:,:ncvs,ncvs:].copy()

            eris_ceco = eris.ovoo[:ncvs,:,:ncvs,:].copy()
            eris_cevo = eris.ovoo[:ncvs,:,ncvs:,:].copy()
            eris_CECO = eris.OVOO[:ncvs,:,:ncvs,:].copy()
            eris_CEVO = eris.OVOO[:ncvs,:,ncvs:,:].copy()
            eris_CEco = eris.OVoo[:ncvs,:,:ncvs,:].copy()
            eris_CEvo = eris.OVoo[:ncvs,:,ncvs:,:].copy()
            eris_ceCO = eris.ovOO[:ncvs,:,:ncvs,:].copy()
            eris_ceVO = eris.ovOO[:ncvs,:,ncvs:,:].copy()

################ ADC(3) i - kja and ajk - i block ############################
            t2_1_a = adc.t2[0][0][:]
            t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
            t2_1_a_voee = t2_1_a[ncvs:,:,:,:].copy()
            t2_1_a_cvee = t2_1_a[:ncvs,ncvs:,:,:].copy()
            t2_1_a_ccee = t2_1_a[:ncvs,:ncvs,:,:].copy()
            t2_1_a_ccee_t = t2_1_a_ccee[ij_ind_ncvs[0],ij_ind_ncvs[1],:,:]

            if isinstance(eris.ovvv, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            else :
                chnk_size = ncvs

            a = 0
            temp_singles = np.zeros((ncvs))
            temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a))
            r_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)


            temp_1_ecc = lib.einsum('Pbc,aP->abc',t2_1_a_ccee_t,r_aaa_ecc, optimize=True)
            temp_1_ecv = lib.einsum('Pqbc,aPq->abc',t2_1_a_cvee,r_aaa_ecv, optimize=True)
            for p in range(0,ncvs,chnk_size):
                if getattr(adc, 'with_df', None):

                    eris_ceee = dfadc.get_ovvv_spin_df(
                        adc, eris.Lce, eris.Lee, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                else :
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                    eris_ceee = eris_ovvv[:ncvs,:,:,:].copy()
                k = eris_ceee.shape[0]
                temp_singles[a:a+k] += 0.5* \
                    lib.einsum('abc,Icab->I',temp_1_ecc, eris_ceee, optimize=True)
                temp_singles[a:a+k] -= 0.5* \
                    lib.einsum('abc,Ibac->I',temp_1_ecc, eris_ceee, optimize=True)
                temp_singles[a:a+k] += 0.5* \
                    lib.einsum('abc,Icab->I',temp_1_ecv, eris_ceee, optimize=True)
                temp_singles[a:a+k] -= 0.5* \
                    lib.einsum('abc,Ibac->I',temp_1_ecv, eris_ceee, optimize=True)

                temp_doubles += lib.einsum('I,Icab->bca',r_a[a:a+k],eris_ceee,optimize=True)
                temp_doubles -= lib.einsum('I,Ibac->bca',r_a[a:a+k],eris_ceee,optimize=True)
                del eris_ceee
                a += k

            s[s_a:f_a] += temp_singles
            s[s_aaa_ecc:f_aaa_ecc] += 0.5* \
                lib.einsum('bca,Pbc->aP',temp_doubles,t2_1_a_ccee_t,optimize=True).reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += 0.5* \
                lib.einsum('bca,Pqbc->aPq',temp_doubles,t2_1_a_cvee,optimize=True).reshape(-1)
            del temp_singles
            del temp_doubles

            temp_ecc = lib.einsum('Jlab,aJK->blK',t2_1_a_coee,r_aaa_ecc_u,optimize=True)
            temp_ecv_1 = lib.einsum('Jlab,aJk->blk',t2_1_a_coee,r_aaa_ecv,optimize=True)
            temp_ecv_2 = -lib.einsum('jlab,aKj->blK',t2_1_a_voee,r_aaa_ecv,optimize=True)

            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_oecc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_ceco,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv_1,eris_oecv,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv_1,eris_cevo,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_ecv_2,eris_oecc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecv_2,eris_ceco,optimize=True)

            temp_1_ecc = lib.einsum('Jlab,aJK->blK',t2_1_a_coee,r_aab_ecc,optimize=True)
            temp_1_ecv = lib.einsum('Jlab,aJk->blk',t2_1_a_coee,r_aab_ecv,optimize=True)
            temp_1_evc = lib.einsum('jlab,ajK->blK',t2_1_a_voee,r_aab_evc,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_oeCC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_1_ecv,eris_oeCV,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_evc,eris_oeCC,optimize=True)

            temp_ecc = -lib.einsum('klab,akj->blj',t2_1_a_coee,r_aaa_ecc_u,optimize=True)
            temp_ecv_1 = -lib.einsum('klab,akj->blj',t2_1_a_coee,r_aaa_ecv,optimize=True)
            temp_ecv_2 = lib.einsum('klab,ajk->blj',t2_1_a_voee,r_aaa_ecv,optimize=True)

            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_oecc,optimize=True)
            s[s_a:f_a] +=0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_ceco,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv_1,eris_oecv,optimize=True)
            s[s_a:f_a] +=0.5*lib.einsum('blj,Ibjl->I',temp_ecv_1,eris_cevo,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecv_2,eris_oecc,optimize=True)
            s[s_a:f_a] +=0.5*lib.einsum('blJ,IbJl->I',temp_ecv_2,eris_ceco,optimize=True)

            temp_1_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_a_coee,r_aab_ecc,optimize=True)
            temp_1_ecv = -lib.einsum('Klab,aKj->blj',t2_1_a_coee,r_aab_ecv,optimize=True)
            temp_1_evc = -lib.einsum('klab,akJ->blJ',t2_1_a_voee,r_aab_evc,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_oeCC,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv,eris_oeCV,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_evc,eris_oeCC,optimize=True)

            temp_1 = lib.einsum('I,lbIK->Kbl',r_a, eris_oecc)
            temp_1 -= lib.einsum('I,IbKl->Kbl',r_a, eris_ceco)
            temp_2 = lib.einsum('I,lbIk->kbl',r_a, eris_oecv)
            temp_2 -= lib.einsum('I,Ibkl->kbl',r_a, eris_cevo)

            temp_ecc  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_a_coee,optimize=True)
            temp_ecv  = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_a_coee,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

            temp_3  = lib.einsum('I,lbIK->Kbl',r_b,eris_oeCC)
            temp_4  = lib.einsum('I,lbIk->kbl',r_b,eris_oeCV)
            temp_ecc = lib.einsum('Kbl,Jlab->aJK',temp_3,t2_1_a_coee,optimize=True)
            temp_ecv = lib.einsum('kbl,Jlab->aJk',temp_4,t2_1_a_coee,optimize=True)
            temp_evc = lib.einsum('Kbl,jlab->ajK',temp_3,t2_1_a_voee,optimize=True)
            s[s_aab_ecc:f_aab_ecc] += temp_ecc.reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += temp_ecv.reshape(-1)
            s[s_aab_evc:f_aab_evc] += temp_evc.reshape(-1)

            temp_1 = lib.einsum('I,lbIJ->Jbl',r_a, eris_oecc)
            temp_1 -= lib.einsum('I,IbJl->Jbl',r_a, eris_ceco)

            temp_ecc  = lib.einsum('Jbl,Klab->aJK',temp_1,t2_1_a_coee,optimize=True)
            temp_ecv  = lib.einsum('Jbl,klab->aJk',temp_1,t2_1_a_voee,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] -= temp_ecv.reshape(-1)

            del t2_1_a_coee
            del t2_1_a_voee

            t2_1_b = adc.t2[0][2][:]
            t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
            t2_1_b_voee = t2_1_b[ncvs:,:,:,:].copy()
            t2_1_b_cvee = t2_1_b[:ncvs,ncvs:,:,:].copy()
            t2_1_b_ccee = t2_1_b[:ncvs,:ncvs,:,:].copy()
            t2_1_b_ccee_t = t2_1_b_ccee[ij_ind_ncvs[0],ij_ind_ncvs[1],:,:]
            t2_1_ab = adc.t2[0][1][:]
            t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
            t2_1_ab_voee = t2_1_ab[ncvs:,:,:,:].copy()
            t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
            t2_1_ab_ovee = t2_1_ab[:,ncvs:,:,:].copy()
            t2_1_ab_ccee = t2_1_ab[:ncvs,:ncvs,:,:].copy()
            t2_1_ab_cvee = t2_1_ab[:ncvs,ncvs:,:,:].copy()
            t2_1_ab_vcee = t2_1_ab[ncvs:,:ncvs,:,:].copy()

            if isinstance(eris.OVVV, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            else :
                chnk_size = ncvs
            a = 0
            temp_singles = np.zeros((ncvs))
            temp_doubles = np.zeros((nvir_b, nvir_b, nvir_b))
            r_bbb_ecc = r_bbb_ecc.reshape(nvir_b,-1)
            temp_1_ecc = lib.einsum('Pbc,aP->abc',t2_1_b_ccee_t,r_bbb_ecc, optimize=True)
            temp_1_ecv = lib.einsum('Pqbc,aPq->abc',t2_1_b_cvee,r_bbb_ecv, optimize=True)
            for p in range(0,ncvs,chnk_size):
                if getattr(adc, 'with_df', None):
                    eris_CEEE = dfadc.get_ovvv_spin_df(
                        adc, eris.LCE, eris.LEE, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                else :
                    eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                    eris_CEEE = eris_OVVV[:ncvs,:,:,:].copy()
                k = eris_CEEE.shape[0]
                temp_singles[a:a+k] += 0.5* \
                    lib.einsum('abc,Icab->I',temp_1_ecc, eris_CEEE, optimize=True)
                temp_singles[a:a+k] -= 0.5* \
                    lib.einsum('abc,Ibac->I',temp_1_ecc, eris_CEEE, optimize=True)
                temp_singles[a:a+k] += 0.5* \
                    lib.einsum('abc,Icab->I',temp_1_ecv, eris_CEEE, optimize=True)
                temp_singles[a:a+k] -= 0.5* \
                    lib.einsum('abc,Ibac->I',temp_1_ecv, eris_CEEE, optimize=True)
                temp_doubles += lib.einsum('I,Icab->bca',r_b[a:a+k],eris_CEEE,optimize=True)
                temp_doubles -= lib.einsum('I,Ibac->bca',r_b[a:a+k],eris_CEEE,optimize=True)
                del eris_CEEE
                a += k

            s[s_b:f_b] += temp_singles
            s[s_bbb_ecc:f_bbb_ecc] += 0.5* \
                lib.einsum('bca,Pbc->aP',temp_doubles,t2_1_b_ccee_t,optimize=True).reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += 0.5* \
                lib.einsum('bca,Pqbc->aPq',temp_doubles,t2_1_b_cvee,optimize=True).reshape(-1)
            del temp_singles
            del temp_doubles

            temp_1_ecc = lib.einsum('Jlab,aJK->blK',t2_1_b_coee,r_bba_ecc,optimize=True)
            temp_1_ecv = lib.einsum('Jlab,aJk->blk',t2_1_b_coee,r_bba_ecv,optimize=True)
            temp_1_evc = lib.einsum('jlab,ajK->blK',t2_1_b_voee,r_bba_evc,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_OEcc,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_1_ecv,eris_OEcv,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_evc,eris_OEcc,optimize=True)

            temp_ecc = lib.einsum('Jlab,aJK->blK',t2_1_b_coee,r_bbb_ecc_u,optimize=True)
            temp_ecv_1 = lib.einsum('Jlab,aJk->blk',t2_1_b_coee,r_bbb_ecv,optimize=True)
            temp_ecv_2 = -lib.einsum('jlab,aKj->blK',t2_1_b_voee,r_bbb_ecv,optimize=True)

            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_OECC,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_CECO,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv_1,eris_OECV,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv_1,eris_CEVO,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_ecv_2,eris_OECC,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecv_2,eris_CECO,optimize=True)

            temp_1_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_b_coee,r_bba_ecc,optimize=True)
            temp_1_ecv = -lib.einsum('Klab,aKj->blj',t2_1_b_coee,r_bba_ecv,optimize=True)
            temp_1_evc = -lib.einsum('klab,akJ->blJ',t2_1_b_voee,r_bba_evc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_OEcc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv,eris_OEcv,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_evc,eris_OEcc,optimize=True)

            temp_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_b_coee,r_bbb_ecc_u,optimize=True)
            temp_ecv_1 = -lib.einsum('Klab,aKj->blj',t2_1_b_coee,r_bbb_ecv,optimize=True)
            temp_ecv_2 = lib.einsum('klab,aJk->blJ',t2_1_b_voee,r_bbb_ecv,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_OECC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_CECO,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv_1,eris_OECV,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blj,Ibjl->I',temp_ecv_1,eris_CEVO,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecv_2,eris_OECC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecv_2,eris_CECO,optimize=True)

            temp_2  = lib.einsum('I,lbIK->Kbl',r_a,eris_OEcc)
            temp_3  = lib.einsum('I,lbIk->kbl',r_a,eris_OEcv)
            temp_ecc = lib.einsum('Kbl,Jlab->aJK',temp_2,t2_1_b_coee,optimize=True)
            temp_ecv = lib.einsum('kbl,Jlab->aJk',temp_3,t2_1_b_coee,optimize=True)
            temp_evc = lib.einsum('Kbl,jlab->ajK',temp_2,t2_1_b_voee,optimize=True)
            s[s_bba_ecc:f_bba_ecc] += temp_ecc.reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += temp_ecv.reshape(-1)
            s[s_bba_evc:f_bba_evc] += temp_evc.reshape(-1)

            temp_1 = lib.einsum('I,lbIK->Kbl',r_b, eris_OECC)
            temp_1 -= lib.einsum('I,IbKl->Kbl',r_b, eris_CECO)
            temp_2 = lib.einsum('I,lbIk->kbl',r_b, eris_OECV)
            temp_2 -= lib.einsum('I,Ibkl->kbl',r_b, eris_CEVO)

            temp_ecc  = lib.einsum('Kbl,Jlab->aJK',temp_1,t2_1_b_coee,optimize=True)
            temp_ecv  = lib.einsum('kbl,Jlab->aJk',temp_2,t2_1_b_coee,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

            temp_1 = lib.einsum('I,lbIJ->Jbl',r_b, eris_OECC)
            temp_1 -= lib.einsum('I,IbJl->Jbl',r_b, eris_CECO)

            temp_ecc  = lib.einsum('Jbl,Klab->aJK',temp_1,t2_1_b_coee,optimize=True)
            temp_ecv  = lib.einsum('Jbl,klab->aJk',temp_1,t2_1_b_voee,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] -= temp_ecv.reshape(-1)
            del t2_1_b_coee
            del t2_1_b_voee

            if isinstance(eris.ovVV, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            else :
                chnk_size = ncvs
            a = 0
            temp_1_ecc = lib.einsum('KJcb,aJK->abc',t2_1_ab_ccee,r_bba_ecc, optimize=True)
            temp_1_ecv = lib.einsum('kJcb,aJk->abc',t2_1_ab_vcee,r_bba_ecv, optimize=True)
            temp_1_evc = lib.einsum('Kjcb,ajK->abc',t2_1_ab_cvee,r_bba_evc, optimize=True)
            temp_2 = np.zeros((nvir_a, nvir_b, nvir_b))
            for p in range(0,ncvs,chnk_size):
                if getattr(adc, 'with_df', None):
                    eris_ceEE = dfadc.get_ovvv_spin_df(
                        adc, eris.Lce, eris.LEE, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                else :
                    eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                    eris_ceEE = eris_ovVV[:ncvs,:,:,:].copy()
                k = eris_ceEE.shape[0]

                s[s_a:f_a][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecc, eris_ceEE, optimize=True)
                s[s_a:f_a][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecv, eris_ceEE, optimize=True)
                s[s_a:f_a][a:a+k] += lib.einsum('abc,Icab->I',temp_1_evc, eris_ceEE, optimize=True)

                temp_2 += lib.einsum('I,Icab->cba',r_a[a:a+k],eris_ceEE,optimize=True)
                del eris_ceEE
                a += k

            s[s_bba_ecc:f_bba_ecc] += lib.einsum('cba,KJcb->aJK',
                                                 temp_2, t2_1_ab_ccee, optimize=True).reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += lib.einsum('cba,kJcb->aJk',
                                                 temp_2, t2_1_ab_vcee, optimize=True).reshape(-1)
            s[s_bba_evc:f_bba_evc] += lib.einsum('cba,Kjcb->ajK',
                                                 temp_2, t2_1_ab_cvee, optimize=True).reshape(-1)
            del temp_1_ecc
            del temp_1_ecv
            del temp_1_evc
            del temp_2

            if isinstance(eris.OVvv, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            else :
                chnk_size = ncvs

            a = 0
            temp_1_ecc = lib.einsum('JKbc,aJK->abc',t2_1_ab_ccee,r_aab_ecc, optimize=True)
            temp_1_ecv = lib.einsum('Jkbc,aJk->abc',t2_1_ab_cvee,r_aab_ecv, optimize=True)
            temp_1_evc = lib.einsum('jKbc,ajK->abc',t2_1_ab_vcee,r_aab_evc, optimize=True)
            temp_2 = np.zeros((nvir_a, nvir_b, nvir_a))
            for p in range(0,ncvs,chnk_size):
                if getattr(adc, 'with_df', None):
                    eris_CEee = dfadc.get_ovvv_spin_df(
                        adc, eris.LCE, eris.Lee, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                else :
                    eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                    eris_CEee = eris_OVvv[:ncvs,:,:,:].copy()
                k = eris_CEee.shape[0]
                s[s_b:f_b][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecc, eris_CEee, optimize=True)
                s[s_b:f_b][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecv, eris_CEee, optimize=True)
                s[s_b:f_b][a:a+k] += lib.einsum('abc,Icab->I',temp_1_evc, eris_CEee, optimize=True)

                temp_2 += lib.einsum('I,Icab->bca',r_b[a:a+k],eris_CEee,optimize=True)
                del eris_CEee
                a += k

            s[s_aab_ecc:f_aab_ecc] += lib.einsum('bca,JKbc->aJK',
                                                 temp_2, t2_1_ab_ccee, optimize=True).reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += lib.einsum('bca,Jkbc->aJk',
                                                 temp_2, t2_1_ab_cvee, optimize=True).reshape(-1)
            s[s_aab_evc:f_aab_evc] += lib.einsum('bca,jKbc->ajK',
                                                 temp_2, t2_1_ab_vcee, optimize=True).reshape(-1)
            del temp_1_ecc
            del temp_1_ecv
            del temp_2

            temp_ecc = lib.einsum('lJba,aJK->blK',t2_1_ab_ocee,r_bba_ecc,optimize=True)
            temp_ecv = lib.einsum('lJba,aJk->blk',t2_1_ab_ocee,r_bba_ecv,optimize=True)
            temp_evc = lib.einsum('ljba,ajK->blK',t2_1_ab_ovee,r_bba_evc,optimize=True)
            temp_1_ecc = lib.einsum('Jlab,aJK->blK',t2_1_ab_coee,r_aaa_ecc_u,optimize=True)
            temp_1_ecv = lib.einsum('Jlab,aJk->blk',t2_1_ab_coee,r_aaa_ecv,optimize=True)
            temp_1_evc = -lib.einsum('jlab,aKj->blK',t2_1_ab_voee,r_aaa_ecv,optimize=True)
            temp_2_ecc = lib.einsum('Jlba,aKJ->blK',t2_1_ab_coee,r_bba_ecc, optimize=True)
            temp_2_ecv = lib.einsum('jlba,aKj->blK',t2_1_ab_voee,r_bba_ecv, optimize=True)
            temp_2_evc = lib.einsum('Jlba,akJ->blk',t2_1_ab_coee,r_bba_evc,optimize=True)

            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_oecc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_ceco,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv,eris_oecv,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv,eris_cevo,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_evc,eris_oecc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_evc,eris_ceco,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_OEcc,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_1_ecv,eris_OEcv,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_evc,eris_OEcc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecc,eris_ceCO,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecv,eris_ceCO,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_evc,eris_ceVO,optimize=True)

            temp_ecc = lib.einsum('Jlab,aJK->blK',t2_1_ab_coee,r_aab_ecc,optimize=True)
            temp_ecv = lib.einsum('Jlab,aJk->blk',t2_1_ab_coee,r_aab_ecv,optimize=True)
            temp_evc = lib.einsum('jlab,ajK->blK',t2_1_ab_voee,r_aab_evc,optimize=True)
            temp_1_ecc = lib.einsum('lJba,aJK->blK',t2_1_ab_ocee,r_bbb_ecc_u,optimize=True)
            temp_1_evc = lib.einsum('lJba,aJk->blk',t2_1_ab_ocee,r_bbb_ecv,optimize=True)
            temp_1_ecv = -lib.einsum('ljba,aKj->blK',t2_1_ab_ovee,r_bbb_ecv,optimize=True)
            temp_2_ecc = lib.einsum('lJab,aKJ->blK',t2_1_ab_ocee,r_aab_ecc,optimize=True)
            temp_2_ecv = lib.einsum('ljab,aKj->blK',t2_1_ab_ovee,r_aab_ecv,optimize=True)
            temp_2_evc = lib.einsum('lJab,akJ->blk',t2_1_ab_ocee,r_aab_evc,optimize=True)

            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_OECC,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_CECO,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv,eris_OECV,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv,eris_CEVO,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_evc,eris_OECC,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_evc,eris_CECO,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_oeCC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecv,eris_oeCC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_1_evc,eris_oeCV,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecc,eris_CEco,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecv,eris_CEco,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blk,Ibkl->I',temp_2_evc,eris_CEvo,optimize=True)

            temp_ecc = -lib.einsum('lKba,aKJ->blJ',t2_1_ab_ocee,r_bba_ecc,optimize=True)
            temp_ecv = -lib.einsum('lKba,aKj->blj',t2_1_ab_ocee,r_bba_ecv,optimize=True)
            temp_evc = -lib.einsum('lkba,akJ->blJ',t2_1_ab_ovee,r_bba_evc,optimize=True)
            temp_1_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_ab_coee,r_aaa_ecc_u,optimize=True)
            temp_1_ecv_1 = -lib.einsum('Klab,aKj->blj',t2_1_ab_coee,r_aaa_ecv,optimize=True)
            temp_1_ecv_2 = lib.einsum('klab,aJk->blJ',t2_1_ab_voee,r_aaa_ecv,optimize=True)
            temp_2_ecc = -lib.einsum('Klba,aJK->blJ',t2_1_ab_coee,r_bba_ecc,optimize=True)
            temp_2_ecv = -lib.einsum('klba,aJk->blJ',t2_1_ab_voee,r_bba_ecv,optimize=True)
            temp_2_evc = -lib.einsum('Klba,ajK->blj',t2_1_ab_coee,r_bba_evc,optimize=True)

            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_oecc,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_ceco,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv,eris_oecv,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blj,Ibjl->I',temp_ecv,eris_cevo,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_evc,eris_oecc,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_evc,eris_ceco,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_OEcc,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv_1,eris_OEcv,optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecv_2,eris_OEcc,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecc,eris_ceCO,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecv,eris_ceCO,optimize=True)
            s[s_a:f_a] += 0.5*lib.einsum('blj,Ibjl->I',temp_2_evc,eris_ceVO,optimize=True)

            temp_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_ab_coee,r_aab_ecc,optimize=True)
            temp_ecv = -lib.einsum('Klab,aKj->blj',t2_1_ab_coee,r_aab_ecv,optimize=True)
            temp_evc = -lib.einsum('klab,akJ->blJ',t2_1_ab_voee,r_aab_evc,optimize=True)
            temp_1_ecc = -lib.einsum('lKba,aKJ->blJ',t2_1_ab_ocee,r_bbb_ecc_u,optimize=True)
            temp_1_ecv_1 = -lib.einsum('lKba,aKj->blj',t2_1_ab_ocee,r_bbb_ecv,optimize=True)
            temp_1_ecv_2 = lib.einsum('lkba,aJk->blJ',t2_1_ab_ovee,r_bbb_ecv,optimize=True)
            temp_2_ecc = -lib.einsum('lKab,aJK->blJ',t2_1_ab_ocee,r_aab_ecc,optimize=True)
            temp_2_ecv = -lib.einsum('lkab,aJk->blJ',t2_1_ab_ovee,r_aab_ecv,optimize=True)
            temp_2_evc = -lib.einsum('lKab,ajK->blj',t2_1_ab_ocee,r_aab_evc,optimize=True)

            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_OECC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_CECO,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv,eris_OECV,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blj,Ibjl->I',temp_ecv,eris_CEVO,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_evc,eris_OECC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_evc,eris_CECO,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_oeCC,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv_1,eris_oeCV,optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecv_2,eris_oeCC,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecc,eris_CEco,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecv,eris_CEco,optimize=True)
            s[s_b:f_b] += 0.5*lib.einsum('blj,Ibjl->I',temp_2_evc,eris_CEvo,optimize=True)

            temp_2 = lib.einsum('I,lbIK->Kbl',r_a, eris_OEcc)
            temp_3 = lib.einsum('I,lbIk->kbl',r_a, eris_OEcv)
            temp_ecc = lib.einsum('Kbl,Jlab->aJK',temp_2, t2_1_ab_coee,optimize=True)
            temp_ecv = lib.einsum('kbl,Jlab->aJk',temp_3, t2_1_ab_coee,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

            temp_2  = lib.einsum('I,lbIK->Kbl',r_a,eris_oecc)
            temp_2  -= lib.einsum('I,IbKl->Kbl',r_a,eris_ceco)
            temp_3  = lib.einsum('I,lbIk->kbl',r_a,eris_oecv)
            temp_3  -= lib.einsum('I,Ibkl->kbl',r_a,eris_cevo)

            temp_ecc  = lib.einsum('Kbl,lJba->aJK',temp_2,t2_1_ab_ocee,optimize=True)
            temp_ecv  = lib.einsum('kbl,lJba->aJk',temp_3,t2_1_ab_ocee,optimize=True)
            temp_evc  = lib.einsum('Kbl,ljba->ajK',temp_2,t2_1_ab_ovee,optimize=True)
            s[s_bba_ecc:f_bba_ecc] += temp_ecc.reshape(-1)
            s[s_bba_ecv:f_bba_ecv] += temp_ecv.reshape(-1)
            s[s_bba_evc:f_bba_evc] += temp_evc.reshape(-1)

            temp_4 = lib.einsum('I,lbIK->Kbl',r_b, eris_oeCC)
            temp_5 = lib.einsum('I,lbIk->kbl',r_b, eris_oeCV)

            temp_ecc = lib.einsum('Kbl,lJba->aJK',temp_4,t2_1_ab_ocee,optimize=True)
            temp_ecv = lib.einsum('kbl,lJba->aJk',temp_5,t2_1_ab_ocee,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

            temp_1  = lib.einsum('I,lbIK->Kbl',r_b,eris_OECC)
            temp_1  -= lib.einsum('I,IbKl->Kbl',r_b,eris_CECO)
            temp_2  = lib.einsum('I,lbIk->kbl',r_b,eris_OECV)
            temp_2  -= lib.einsum('I,Ibkl->kbl',r_b,eris_CEVO)

            temp_ecc  = lib.einsum('Kbl,Jlab->aJK',temp_1,t2_1_ab_coee,optimize=True)
            temp_ecv  = lib.einsum('kbl,Jlab->aJk',temp_2,t2_1_ab_coee,optimize=True)
            temp_evc  = lib.einsum('Kbl,jlab->ajK',temp_1,t2_1_ab_voee,optimize=True)
            s[s_aab_ecc:f_aab_ecc] += temp_ecc.reshape(-1)
            s[s_aab_ecv:f_aab_ecv] += temp_ecv.reshape(-1)
            s[s_aab_evc:f_aab_evc] += temp_evc.reshape(-1)

            temp_2 = lib.einsum('I,lbIJ->Jbl',r_a, eris_OEcc)

            temp_ecc = lib.einsum('Jbl,Klab->aJK',temp_2,t2_1_ab_coee,optimize=True)
            temp_ecv = lib.einsum('Jbl,klab->aJk',temp_2,t2_1_ab_voee,optimize=True)
            s[s_aaa_ecc:f_aaa_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_aaa_ecv:f_aaa_ecv] -= temp_ecv.reshape(-1)

            temp_1  = -lib.einsum('I,IbJl->Jbl',r_a,eris_ceCO,optimize=True)
            temp_2  = -lib.einsum('I,Ibjl->jbl',r_a,eris_ceVO,optimize=True)
            temp_1_ecc = -lib.einsum('Jbl,Klba->aJK',temp_1,t2_1_ab_coee,optimize=True)
            temp_1_ecv = -lib.einsum('Jbl,klba->aJk',temp_1,t2_1_ab_voee,optimize=True)
            temp_1_evc = -lib.einsum('jbl,Klba->ajK',temp_2,t2_1_ab_coee,optimize=True)
            s[s_bba_ecc:f_bba_ecc] -= temp_1_ecc.reshape(-1)
            s[s_bba_ecv:f_bba_ecv] -= temp_1_ecv.reshape(-1)
            s[s_bba_evc:f_bba_evc] -= temp_1_evc.reshape(-1)

            temp_2 = lib.einsum('I,lbIJ->Jbl',r_b, eris_oeCC)
            temp_ecc = lib.einsum('Jbl,lKba->aJK',temp_2,t2_1_ab_ocee,optimize=True)
            temp_ecv = lib.einsum('Jbl,lkba->aJk',temp_2,t2_1_ab_ovee,optimize=True)
            s[s_bbb_ecc:f_bbb_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
            s[s_bbb_ecv:f_bbb_ecv] -= temp_ecv.reshape(-1)

            temp_3  = -lib.einsum('I,IbJl->Jbl',r_b,eris_CEco,optimize=True)
            temp_4  = -lib.einsum('I,Ibjl->jbl',r_b,eris_CEvo,optimize=True)
            temp_1_ecc = -lib.einsum('Jbl,lKab->aJK',temp_3,t2_1_ab_ocee,optimize=True)
            temp_1_ecv = -lib.einsum('Jbl,lkab->aJk',temp_3,t2_1_ab_ovee,optimize=True)
            temp_1_evc = -lib.einsum('jbl,lKab->ajK',temp_4,t2_1_ab_ocee,optimize=True)
            s[s_aab_ecc:f_aab_ecc] -= temp_1_ecc.reshape(-1)
            s[s_aab_ecv:f_aab_ecv] -= temp_1_ecv.reshape(-1)
            s[s_aab_evc:f_aab_evc] -= temp_1_evc.reshape(-1)

            del t2_1_ab_coee
            del t2_1_ab_voee
            del t2_1_ab_ocee
            del t2_1_ab_ovee
            del t2_1_ab_ccee
            del t2_1_ab_cvee
            del t2_1_ab_vcee

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0
        return s

    return sigma_

def get_trans_moments(adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b

    T_a = []
    T_b = []

    for orb in range(nmo_a):
        T_aa = get_trans_moments_orbital(adc,orb, spin="alpha")
        T_a.append(T_aa)

    for orb in range(nmo_b):
        T_bb = get_trans_moments_orbital(adc,orb, spin="beta")
        T_b.append(T_bb)

    cput0 = log.timer_debug1("completed spec vector calc in ADC(3) calculation", *cput0)
    return (T_a, T_b)


def get_trans_moments_orbital(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        t1_2_a, t1_2_b = adc.t1[0]

    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs

    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = nvir_a * ncvs * (ncvs - 1) // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a
    n_doubles_bba_ecc = nvir_b * ncvs * ncvs
    n_doubles_bba_ecv = nvir_b * ncvs * nval_a
    n_doubles_bba_evc = nvir_b * nval_b * ncvs
    n_doubles_aab_ecc = nvir_a * ncvs * ncvs
    n_doubles_aab_ecv = nvir_a * ncvs * nval_b
    n_doubles_aab_evc = nvir_a * nval_a * ncvs
    n_doubles_bbb_ecc = nvir_b * ncvs * (ncvs - 1) // 2
    n_doubles_bbb_ecv = nvir_b * ncvs * nval_b
    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_aaa_ecv + n_doubles_bba_ecc + n_doubles_bba_ecv + \
        n_doubles_bba_evc + n_doubles_aab_ecc + n_doubles_aab_ecv + \
            n_doubles_aab_evc + n_doubles_bbb_ecc + n_doubles_bbb_ecv

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bba_ecc = f_aaa_ecv
    f_bba_ecc = s_bba_ecc + n_doubles_bba_ecc
    s_bba_ecv = f_bba_ecc
    f_bba_ecv = s_bba_ecv + n_doubles_bba_ecv
    s_bba_evc = f_bba_ecv
    f_bba_evc = s_bba_evc + n_doubles_bba_evc
    s_aab_ecc = f_bba_evc
    f_aab_ecc = s_aab_ecc + n_doubles_aab_ecc
    s_aab_ecv = f_aab_ecc
    f_aab_ecv = s_aab_ecv + n_doubles_aab_ecv
    s_aab_evc = f_aab_ecv
    f_aab_evc = s_aab_evc + n_doubles_aab_evc
    s_bbb_ecc = f_aab_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

    T = np.zeros((dim))

######## spin = alpha  ############################################
    if spin=="alpha":
        ######## ADC(2) 1h part  ############################################

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
        t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :ncvs]
            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:],
                                          t2_1_a_coee, optimize=True)
            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:],
                                          t2_1_ab_coee, optimize=True)
            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:],
                                          t2_1_ab_coee, optimize=True)
        else :
            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_a:f_a] += t1_2_a[:ncvs,(orb-nocc_a)]

######## ADC(2) 2h-1p  part  ############################################

            t2_1_a_t = t2_1_a.transpose(2,3,1,0)
            t2_1_ab_t = t2_1_ab.transpose(2,3,1,0)
            t2_1_a_eecc = t2_1_a_t[:,:,:ncvs,:ncvs].copy()
            t2_1_a_eecv = t2_1_a_t[:,:,:ncvs,ncvs:].copy()
            t2_1_a_ecc = t2_1_a_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
            t2_1_ab_eecc = t2_1_ab_t[:,:,:ncvs,:ncvs].copy()
            t2_1_ab_eecv = t2_1_ab_t[:,:,:ncvs,ncvs:].copy()
            t2_1_ab_eevc = t2_1_ab_t[:,:,ncvs:,:ncvs].copy()

            T[s_aaa_ecc:f_aaa_ecc] = t2_1_a_ecc[(orb-nocc_a),:,:].reshape(-1)
            T[s_aaa_ecv:f_aaa_ecv] = t2_1_a_eecv[(orb-nocc_a),:,:,:].reshape(-1)
            T[s_bba_ecc:f_bba_ecc] = t2_1_ab_eecc[(orb-nocc_a),:,:,:].reshape(-1)
            T[s_bba_ecv:f_bba_ecv] = t2_1_ab_eecv[(orb-nocc_a),:,:,:].reshape(-1)
            T[s_bba_evc:f_bba_evc] = t2_1_ab_eevc[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb >= nocc_a:

                t2_2_a_t = t2_2_a.transpose(2,3,1,0)
                t2_2_ab_t = t2_2_ab.transpose(2,3,1,0)
                t2_2_a_eecc = t2_2_a_t[:,:,:ncvs,:ncvs].copy()
                t2_2_a_eecv = t2_2_a_t[:,:,:ncvs,ncvs:].copy()
                t2_2_a_ecc = t2_2_a_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
                t2_2_ab_eecc = t2_2_ab_t[:,:,:ncvs,:ncvs].copy()
                t2_2_ab_eecv = t2_2_ab_t[:,:,:ncvs,ncvs:].copy()
                t2_2_ab_eevc = t2_2_ab_t[:,:,ncvs:,:ncvs].copy()

                T[s_aaa_ecc:f_aaa_ecc] += t2_2_a_ecc[(orb-nocc_a),:,:].reshape(-1)
                T[s_aaa_ecv:f_aaa_ecv] += t2_2_a_eecv[(orb-nocc_a),:,:,:].reshape(-1)
                T[s_bba_ecc:f_bba_ecc] += t2_2_ab_eecc[(orb-nocc_a),:,:,:].reshape(-1)
                T[s_bba_ecv:f_bba_ecv] += t2_2_ab_eecv[(orb-nocc_a),:,:,:].reshape(-1)
                T[s_bba_evc:f_bba_evc] += t2_2_ab_eevc[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:

                t2_1_a_tmp = np.ascontiguousarray(t2_1_a[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[orb,:,:,:])
                t2_2_a_coee = t2_2_a[:ncvs,:,:,:].copy()
                t2_2_ab_coee = t2_2_ab[:ncvs,:,:,:].copy()

                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a_tmp, t2_2_a_coee, optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab_tmp,
                                              t2_2_ab_coee, optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab_tmp,
                                              t2_2_ab_coee, optimize=True)

                del t2_1_a_tmp, t2_1_ab_tmp

                t2_2_a_tmp = np.ascontiguousarray(t2_2_a[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a_coee,  t2_2_a_tmp,optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab_coee, t2_2_ab_tmp,optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab_coee, t2_2_ab_tmp,optimize=True)

                del t2_2_a_tmp, t2_2_ab_tmp
            else:
                t2_1_a_tmp =  np.ascontiguousarray(t2_1_a[:ncvs,:,(orb-nocc_a),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:ncvs,:,(orb-nocc_a),:])

                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a_tmp, t1_2_a,optimize=True)
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab_tmp, t1_2_b,optimize=True)
                if (adc.approx_trans_moments is False):
                    T[s_a:f_a] += t1_3_a[:ncvs,(orb-nocc_a)]
                del t2_1_a_tmp, t2_1_ab_tmp

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab
######## spin = beta  ############################################
    else:
        ######## ADC(2) 1h part  ############################################

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
        t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
        if orb < nocc_b:

            t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
            t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

            T[s_b:f_b] = idn_occ_b[orb, :ncvs]
            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_1_b_coee, optimize=True)
            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_1_ab_ocee, optimize=True)
            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_1_ab_ocee, optimize=True)
            del t2_1_b_tmp, t2_1_ab_tmp
        else :
            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_b:f_b] += t1_2_b[:ncvs,(orb-nocc_b)]

######## ADC(2) 2h-1p part  ############################################

            t2_1_b_t = t2_1_b.transpose(2,3,1,0)
            t2_1_ab_t = t2_1_ab.transpose(2,3,0,1)
            t2_1_b_eecc = t2_1_b_t[:,:,:ncvs,:ncvs].copy()
            t2_1_b_eecv = t2_1_b_t[:,:,:ncvs,ncvs:].copy()
            t2_1_b_ecc = t2_1_b_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
            t2_1_ab_eecc = t2_1_ab_t[:,:,:ncvs,:ncvs].copy()
            t2_1_ab_eecv = t2_1_ab_t[:,:,:ncvs,ncvs:].copy()
            t2_1_ab_eevc = t2_1_ab_t[:,:,ncvs:,:ncvs].copy()

            T[s_bbb_ecc:f_bbb_ecc] = t2_1_b_ecc[(orb-nocc_b),:,:].reshape(-1)
            T[s_bbb_ecv:f_bbb_ecv] = t2_1_b_eecv[(orb-nocc_b),:,:,:].reshape(-1)
            T[s_aab_ecc:f_aab_ecc] = t2_1_ab_eecc[:,(orb-nocc_b),:,:].reshape(-1)
            T[s_aab_ecv:f_aab_ecv] = t2_1_ab_eecv[:,(orb-nocc_b),:,:].reshape(-1)
            T[s_aab_evc:f_aab_evc] = t2_1_ab_eevc[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 2h-1p part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb >= nocc_b:

                t2_2_b_t = t2_2_b.transpose(2,3,1,0)
                t2_2_ab_t = t2_2_ab.transpose(2,3,0,1)
                t2_2_b_eecc = t2_2_b_t[:,:,:ncvs,:ncvs].copy()
                t2_2_b_eecv = t2_2_b_t[:,:,:ncvs,ncvs:].copy()
                t2_2_b_ecc = t2_2_b_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
                t2_2_ab_eecc = t2_2_ab_t[:,:,:ncvs,:ncvs].copy()
                t2_2_ab_eecv = t2_2_ab_t[:,:,:ncvs,ncvs:].copy()
                t2_2_ab_eevc = t2_2_ab_t[:,:,ncvs:,:ncvs].copy()

                T[s_bbb_ecc:f_bbb_ecc] += t2_2_b_ecc[(orb-nocc_b),:,:].reshape(-1)
                T[s_bbb_ecv:f_bbb_ecv] += t2_2_b_eecv[(orb-nocc_b),:,:,:].reshape(-1)
                T[s_aab_ecc:f_aab_ecc] += t2_2_ab_eecc[:,(orb-nocc_b),:,:].reshape(-1)
                T[s_aab_ecv:f_aab_ecv] += t2_2_ab_eecv[:,(orb-nocc_b),:,:].reshape(-1)
                T[s_aab_evc:f_aab_evc] += t2_2_ab_eevc[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])
                t2_2_b_coee = t2_2_b[:ncvs,:,:,:].copy()
                t2_2_ab_ocee = t2_2_ab[:,:ncvs,:,:].copy()


                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_2_b_coee, optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp,
                                              t2_2_ab_ocee, optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp,
                                              t2_2_ab_ocee, optimize=True)

                del t2_1_b_tmp, t2_1_ab_tmp

                t2_2_b_tmp = np.ascontiguousarray(t2_2_b[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b_coee,  t2_2_b_tmp ,optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab_ocee, t2_2_ab_tmp,optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab_ocee, t2_2_ab_tmp,optimize=True)

                del t2_2_b_tmp, t2_2_ab_tmp

            else:
                t2_1_b_tmp  = np.ascontiguousarray(t2_1_b[:ncvs,:,(orb-nocc_b),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:ncvs,:,(orb-nocc_b)])

                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b_tmp, t1_2_b,optimize=True)
                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab_tmp, t1_2_a,optimize=True)
                if (adc.approx_trans_moments is False):
                    T[s_b:f_b] += t1_3_b[:ncvs,(orb-nocc_b)]
                del t2_1_b_tmp, t2_1_ab_tmp
                del t2_2_b
                del t2_2_ab

        del t2_1_b
        del t2_1_ab

    return T

def analyze_spec_factor(adc):

    X_a = adc.X[0]
    X_b = adc.X[1]

    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)

    X_tot = (X_a, X_b)

    for iter_idx, X in enumerate(X_tot):
        if iter_idx == 0:
            spin = "alpha"
        else:
            spin = "beta"

        X_2 = (X.copy()**2)

        thresh = adc.spec_factor_print_tol

        for i in range(X_2.shape[1]):

            sort = np.argsort(-X_2[:,i])
            X_2_row = X_2[:,i]

            X_2_row = X_2_row[sort]

            if not adc.mol.symmetry:
                sym = np.repeat(['A'], X_2_row.shape[0])
            else:
                if spin == "alpha":
                    sym = [symm.irrep_id2name(adc.mol.groupname, x)
                                              for x in adc._scf.mo_coeff[0].orbsym]
                    sym = np.array(sym)
                else:
                    sym = [symm.irrep_id2name(adc.mol.groupname, x)
                                              for x in adc._scf.mo_coeff[1].orbsym]
                    sym = np.array(sym)

                sym = sym[sort]

            spec_Contribution = X_2_row[X_2_row > thresh]
            index_mo = sort[X_2_row > thresh]+1

            if np.sum(spec_Contribution) == 0.0:
                continue

            logger.info(adc, '%s | root %d %s\n', adc.method, i, spin)
            logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
            logger.info(adc, "-----------------------------------------------------------")

            for c in range(index_mo.shape[0]):
                logger.info(adc, '     %3.d          %10.8f                %s',
                            index_mo[c], spec_Contribution[c], sym[c])

            logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
            logger.info(adc, "\n*************************************************************\n")


def get_properties(adc, nroots=1):

    #Transition moments
    T = adc.get_trans_moments()

    T_a = T[0]
    T_b = T[1]

    T_a = np.array(T_a)
    T_b = np.array(T_b)

    U = adc.U

    #Spectroscopic amplitudes
    X_a = np.dot(T_a, U).reshape(-1,nroots)
    X_b = np.dot(T_b, U).reshape(-1,nroots)

    X = (X_a,X_b)

    #Spectroscopic factors
    P = lib.einsum("pi,pi->i", X_a, X_a)
    P += lib.einsum("pi,pi->i", X_b, X_b)

    return P, X


def analyze(myadc):

    #TODO: Implement eigenvector analysis for CVS-UADC:
    #header = ("\n*************************************************************"
    #          "\n           Eigenvector analysis summary"
    #          "\n*************************************************************")
    #logger.info(myadc, header)

    #myadc.analyze_eigenvector()

    if myadc.compute_properties:

        header = ("\n*************************************************************"
                  "\n            Spectroscopic factors analysis summary"
                  "\n*************************************************************")
        logger.info(myadc, header)

        myadc.analyze_spec_factor()


def compute_dyson_mo(myadc):

    X_a = myadc.X[0]
    X_b = myadc.X[1]

    if X_a is None:
        nroots = myadc.U.shape[1]
        P,X_a,X_b = myadc.get_properties(nroots)

    nroots = X_a.shape[1]
    dyson_mo_a = np.dot(myadc.mo_coeff[0],X_a)
    dyson_mo_b = np.dot(myadc.mo_coeff[1],X_b)

    dyson_mo = (dyson_mo_a,dyson_mo_b)

    return dyson_mo


class UADCIPCVS(uadc.UADC):
    '''unrestricted ADC for IP-CVS energies and spectroscopic amplitudes

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

            >>> myadc = adc.UADC(mf).run()
            >>> myadcip = adc.UADC(myadc).run()

    Saved results

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method',
        'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory',
        't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'nocc_a', 'nocc_b', 'nvir_a', 'nvir_b', 'mo_coeff', 'mo_energy_a',
        'mo_energy_b', 'nmo_a', 'nmo_b', 'mol', 'transform_integrals',
        'with_df', 'spec_factor_print_tol', 'evec_print_tol', 'ncvs',
        'compute_properties', 'approx_trans_moments', 'E', 'U', 'P', 'X',
        'compute_spin_square'
    }

    def __init__(self, adc):
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
        self.nocc_a = adc._nocc[0]
        self.nocc_b = adc._nocc[1]
        self.nvir_a = adc._nvir[0]
        self.nvir_b = adc._nvir[1]
        self.mo_coeff = adc.mo_coeff
        self.mo_energy_a = adc.mo_energy_a
        self.mo_energy_b = adc.mo_energy_b
        self.nmo_a = adc._nmo[0]
        self.nmo_b = adc._nmo[1]
        self.mol = adc.mol
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol
        self.ncvs = adc.ncvs

        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments

        self.compute_spin_square = False

        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

    kernel = uadc.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    get_trans_moments = get_trans_moments
    get_properties = get_properties

    analyze_spec_factor = analyze_spec_factor
    #analyze_eigenvector = analyze_eigenvector
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo

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
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag
