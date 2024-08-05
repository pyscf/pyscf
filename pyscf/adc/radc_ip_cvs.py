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

    ncvs = adc.ncvs

    e_cvs = adc.mo_energy[:ncvs]
    idn_cvs = np.identity(ncvs)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo

    # i-j block
    # Zeroth-order terms

    M_ij = lib.einsum('ij,j->ij', idn_cvs ,e_cvs)

    # Second-order terms
    t2_1 = t2[0][:]
    t2_1_coee = t2_1[:ncvs,:,:,:].copy()
    t2_1_ocee = t2_1[:,:ncvs,:,:].copy()
    eris_ceeo = eris_ovvo[:ncvs,:,:,:].copy()

    M_ij += 0.5 * 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_coee, eris_ceeo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('lide,jdel->ij',t2_1_ocee, eris_ceeo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_coee, eris_ceeo,optimize=True)
    M_ij += 0.5 * 0.5 *  lib.einsum('lide,jedl->ij',t2_1_ocee, eris_ceeo,optimize=True)
    M_ij += 0.5 * lib.einsum('ilde,jdel->ij',t2_1_coee, eris_ceeo,optimize=True)

    M_ij += 0.5 * 0.5 *  lib.einsum('jlde,idel->ij',t2_1_coee, eris_ceeo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('ljde,idel->ij',t2_1_ocee, eris_ceeo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('jlde,iedl->ij',t2_1_coee, eris_ceeo,optimize=True)
    M_ij += 0.5 * 0.5 *  lib.einsum('ljde,iedl->ij',t2_1_ocee, eris_ceeo,optimize=True)
    M_ij += 0.5 * lib.einsum('jlde,idel->ij',t2_1_coee, eris_ceeo,optimize=True)

    del t2_1

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    # Third-order terms

    if (method == "adc(3)"):

        eris_oovv = eris.oovv
        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo

        eris_oecc = eris_ovoo[:,:,:ncvs,:ncvs].copy()
        eris_ceoc = eris_ovoo[:ncvs,:,:,:ncvs].copy()
        eris_coee = eris_oovv[:ncvs,:,:,:].copy()
        eris_ccee = eris_coee[:,:ncvs,:,:].copy()
        eris_ceec = eris_ceeo[:,:,:,:ncvs].copy()
        eris_cooo = eris_oooo[:ncvs,:,:,:].copy()
        eris_cooc = eris_cooo[:,:,:,:ncvs].copy()
        eris_ccoo = eris_cooo[:,:ncvs,:,:].copy()

        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_oecc,optimize=True)
        M_ij -= lib.einsum('ld,jdli->ij',t1_2, eris_ceoc,optimize=True)
        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_oecc,optimize=True)

        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_oecc,optimize=True)
        M_ij -= lib.einsum('ld,idlj->ij',t1_2, eris_ceoc,optimize=True)
        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_oecc,optimize=True)
        t2_2 = t2[1][:]
        t2_2_coee = t2_2[:ncvs,:,:,:].copy()
        t2_2_ocee = t2_2[:,:ncvs,:,:].copy()

        M_ij += 0.5 * 0.5* lib.einsum('ilde,jdel->ij',t2_2_coee, eris_ceeo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('lide,jdel->ij',t2_2_ocee, eris_ceeo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('ilde,jedl->ij',t2_2_coee, eris_ceeo,optimize=True)
        M_ij += 0.5 * 0.5* lib.einsum('lide,jedl->ij',t2_2_ocee, eris_ceeo,optimize=True)
        M_ij += 0.5 * lib.einsum('ilde,jdel->ij',t2_2_coee, eris_ceeo,optimize=True)

        M_ij += 0.5 * 0.5* lib.einsum('jlde,idel->ij',t2_2_coee, eris_ceeo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('ljde,idel->ij',t2_2_ocee, eris_ceeo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('jlde,iedl->ij',t2_2_coee, eris_ceeo,optimize=True)
        M_ij += 0.5 * 0.5* lib.einsum('ljde,iedl->ij',t2_2_ocee, eris_ceeo,optimize=True)
        M_ij += 0.5 * lib.einsum('jlde,idel->ij',t2_2_coee, eris_ceeo,optimize=True)
        t2_1 = t2[0][:]

        log.timer_debug1("Starting the small integrals  calculation")

        temp_t2_v_1_oece = lib.einsum('lmde,jldf->mejf',t2_1, t2_1_coee,optimize=True)
        temp_t2_v_1_ceoe = lib.einsum('lmde,jldf->mejf',t2_1_ocee, t2_1,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('mejf,ifem->ij',temp_t2_v_1_oece, eris_ceeo,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('jfme,ifem->ij',temp_t2_v_1_ceoe, eris_ceeo,optimize=True)
        M_ij +=  0.5 * lib.einsum('mejf,imfe->ij',temp_t2_v_1_oece, eris_coee,optimize=True)
        M_ij +=  0.5 * lib.einsum('jfme,imfe->ij',temp_t2_v_1_ceoe, eris_coee,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('meif,jfem->ij',temp_t2_v_1_oece, eris_ceeo ,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('ifme,jfem->ij',temp_t2_v_1_ceoe, eris_ceeo ,optimize=True)
        M_ij +=  0.5 * lib.einsum('meif,jmfe->ij',temp_t2_v_1_oece, eris_coee ,optimize=True)
        M_ij +=  0.5 * lib.einsum('ifme,jmfe->ij',temp_t2_v_1_ceoe, eris_coee ,optimize=True)
        del temp_t2_v_1_oece
        del temp_t2_v_1_ceoe

        temp_t2_v_2 = lib.einsum('lmde,ljdf->mejf',t2_1, t2_1_ocee,optimize=True)
        M_ij +=  0.5 * 4 * lib.einsum('mejf,ifem->ij',temp_t2_v_2, eris_ceeo,optimize=True)
        M_ij +=  0.5 * 4 * lib.einsum('meif,jfem->ij',temp_t2_v_2, eris_ceeo,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('meif,jmfe->ij',temp_t2_v_2, eris_coee,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('mejf,imfe->ij',temp_t2_v_2, eris_coee,optimize=True)
        del temp_t2_v_2

        temp_t2_v_3 = lib.einsum('mlde,jldf->mejf',t2_1, t2_1_coee,optimize=True)
        M_ij += 0.5 * lib.einsum('mejf,ifem->ij',temp_t2_v_3, eris_ceeo,optimize=True)
        M_ij += 0.5 * lib.einsum('meif,jfem->ij',temp_t2_v_3, eris_ceeo,optimize=True)
        M_ij -= 0.5 * 2 *lib.einsum('meif,jmfe->ij',temp_t2_v_3, eris_coee,optimize=True)
        M_ij -= 0.5 * 2 * lib.einsum('mejf,imfe->ij',temp_t2_v_3, eris_coee,optimize=True)
        del temp_t2_v_3

        temp_t2_v_8 = lib.einsum('lmdf,lmde->fe',t2_1, t2_1,optimize=True)
        M_ij += 3 *lib.einsum('fe,jief->ij',temp_t2_v_8, eris_ccee, optimize=True)
        M_ij -= 1.5 *lib.einsum('fe,jfei->ij',temp_t2_v_8, eris_ceec, optimize=True)
        M_ij += lib.einsum('ef,jief->ij',temp_t2_v_8, eris_ccee, optimize=True)
        M_ij -= 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_8, eris_ceec, optimize=True)
        del temp_t2_v_8

        temp_t2_v_9 = lib.einsum('lmdf,mlde->fe',t2_1, t2_1,optimize=True)
        M_ij -= 1.0 * lib.einsum('fe,jief->ij',temp_t2_v_9, eris_ccee, optimize=True)
        M_ij -= 1.0 * lib.einsum('ef,jief->ij',temp_t2_v_9, eris_ccee, optimize=True)
        M_ij += 0.5 * lib.einsum('fe,jfei->ij',temp_t2_v_9, eris_ceec, optimize=True)
        M_ij += 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_9, eris_ceec, optimize=True)
        del temp_t2_v_9

        temp_t2_v_10 = lib.einsum('lnde,lmde->nm',t2_1, t2_1,optimize=True)
        M_ij -= 3.0 * lib.einsum('nm,jinm->ij',temp_t2_v_10, eris_ccoo, optimize=True)
        M_ij -= 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_10, eris_ccoo, optimize=True)
        M_ij += 1.5 * lib.einsum('nm,jmni->ij',temp_t2_v_10, eris_cooc, optimize=True)
        M_ij += 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_10, eris_cooc, optimize=True)
        del temp_t2_v_10

        temp_t2_v_11 = lib.einsum('lnde,mlde->nm',t2_1, t2_1,optimize=True)
        M_ij += 1.0 * lib.einsum('nm,jinm->ij',temp_t2_v_11, eris_ccoo, optimize=True)
        M_ij -= 0.5 * lib.einsum('nm,jmni->ij',temp_t2_v_11, eris_cooc, optimize=True)
        M_ij -= 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_11, eris_cooc, optimize=True)
        M_ij += 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_11, eris_ccoo, optimize=True)
        del temp_t2_v_11

        temp_t2_v_12_cooo = lib.einsum('inde,lmde->inlm',t2_1_coee, t2_1,optimize=True)
        temp_t2_v_12_ooco = lib.einsum('inde,lmde->inlm',t2_1, t2_1_coee,optimize=True)
        M_ij += 0.5 * 1.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12_cooo, eris_cooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('lmin,jlnm->ij',temp_t2_v_12_ooco, eris_cooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12_cooo, eris_cooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('lmin,jmnl->ij',temp_t2_v_12_ooco, eris_cooo, optimize=True)

        M_ij += 0.5 * 0.25 * lib.einsum('inlm,jlnm->ji',temp_t2_v_12_cooo, eris_cooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('inlm,jmnl->ji',temp_t2_v_12_cooo, eris_cooo, optimize=True)
        M_ij += 0.5 * 1.00 * lib.einsum('inlm,jlmn->ji',temp_t2_v_12_cooo, eris_cooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('lmin,jmnl->ji',temp_t2_v_12_ooco, eris_cooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('lmin,jlmn->ji',temp_t2_v_12_ooco, eris_cooo, optimize=True)
        del temp_t2_v_12_cooo
        del temp_t2_v_12_ooco

        temp_t2_v_13_cooo = lib.einsum('inde,mlde->inml',t2_1_coee, t2_1,optimize=True)
        temp_t2_v_13_ooco = lib.einsum('inde,mlde->inml',t2_1, t2_1_coee,optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('inml,jlnm->ij',temp_t2_v_13_cooo, eris_cooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('mlin,jlnm->ij',temp_t2_v_13_ooco, eris_cooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('inml,jmnl->ij',temp_t2_v_13_cooo, eris_cooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('mlin,jmnl->ij',temp_t2_v_13_ooco, eris_cooo, optimize=True)

        M_ij -= 0.5 * 0.25 * lib.einsum('inml,jlnm->ji',temp_t2_v_13_cooo, eris_cooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('inml,jmnl->ji',temp_t2_v_13_cooo, eris_cooo, optimize=True)

        M_ij -= 0.5 * 0.25 * lib.einsum('inml,jlmn->ji',temp_t2_v_13_cooo, eris_cooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('inml,jmnl->ji',temp_t2_v_13_cooo, eris_cooo, optimize=True)
        del temp_t2_v_13_cooo
        del temp_t2_v_13_ooco
        del t2_1

    cput0 = log.timer_debug1("Completed CVS M_ij ADC(n) calculation", *cput0)

    return M_ij

def get_diag(adc,M_ij=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    nocc = adc._nocc
    nvir = adc._nvir
    ncvs = adc.ncvs
    nval = nocc - ncvs

    n_singles = ncvs
    n_doubles_ecc = nvir * ncvs * ncvs
    n_doubles_ecv =  nvir * ncvs * nval

    dim = n_singles + n_doubles_ecc + 2 * n_doubles_ecv

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2_ecc = f1
    f2_ecc = s2_ecc + n_doubles_ecc
    s2_ecv = f2_ecc
    f2_ecv = s2_ecv + n_doubles_ecv
    s2_evc = f2_ecv
    f2_evc = s2_evc + n_doubles_ecv

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)

    D_aij = D_n.reshape(-1)
    D_aij = D_n.reshape(nvir,nocc,nocc)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij)
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block
    diag[s2_ecc:f2_ecc] = D_aij[:,:ncvs,:ncvs].reshape(-1)
    diag[s2_ecv:f2_ecv] = D_aij[:,:ncvs,ncvs:].reshape(-1)
    diag[s2_evc:f2_evc] = D_aij[:,ncvs:,:ncvs].reshape(-1)

    diag = -diag

    return diag

def matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc = adc._nocc
    nvir = adc._nvir
    ncvs = adc.ncvs
    nval = nocc - ncvs

    n_singles = ncvs
    n_doubles_ecc = nvir * ncvs * ncvs
    n_doubles_ecv =  nvir * ncvs * nval

    dim = n_singles + n_doubles_ecc + 2 * n_doubles_ecv

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2_ecc = f1
    f2_ecc = s2_ecc + n_doubles_ecc
    s2_ecv = f2_ecc
    f2_ecv = s2_ecv + n_doubles_ecv
    s2_evc = f2_ecv
    f2_evc = s2_evc + n_doubles_ecv

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)
    D_aij = D_n.reshape(nvir,nocc,nocc)

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2_ecc = r[s2_ecc:f2_ecc]
        r2_ecv = r[s2_ecv:f2_ecv]
        r2_evc = r[s2_evc:f2_evc]

        r2_ecc = r2_ecc.reshape(nvir,ncvs,ncvs)
        r2_ecv = r2_ecv.reshape(nvir,ncvs,nval)
        r2_evc = r2_evc.reshape(nvir,nval,ncvs)

        eris_ovoo = eris.ovoo

        eris_cecc = eris_ovoo[:ncvs,:,:ncvs,:ncvs].copy()
        eris_cecv = eris_ovoo[:ncvs,:,:ncvs,ncvs:].copy()
        eris_vecc = eris_ovoo[ncvs:,:,:ncvs,:ncvs].copy()

        del eris_ovoo

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('IJ,J->I',M_ij,r1)

############ ADC(2) i - kja block #########################

        s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_cecc, r2_ecc, optimize=True)
        s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_cecc, r2_ecc, optimize=True)
        s[s1:f1] += 2. * lib.einsum('jaik,ajk->i', eris_cecv, r2_ecv, optimize=True)
        s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_vecc, r2_ecv, optimize=True)
        s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_vecc, r2_evc, optimize=True)
        s[s1:f1] -= lib.einsum('kaij,ajk->i', eris_cecv, r2_evc, optimize=True)

############## ADC(2) ajk - i block ############################

        s[s2_ecc:f2_ecc] += lib.einsum('jaki,i->ajk', eris_cecc, r1, optimize=True).reshape(-1)
        s[s2_ecv:f2_ecv] += lib.einsum('jaik,i->ajk', eris_cecv, r1, optimize=True).reshape(-1)
        s[s2_evc:f2_evc] += lib.einsum('jaki,i->ajk', eris_vecc, r1, optimize=True).reshape(-1)

        del eris_cecc, eris_cecv, eris_vecc

################ ADC(2) ajk - bil block ############################

        temp_ecc = D_aij[:,:ncvs,:ncvs].reshape(-1)
        s[s2_ecc:f2_ecc] += temp_ecc*r2_ecc.reshape(-1)
        temp_ecv = D_aij[:,:ncvs,ncvs:].reshape(-1)
        s[s2_ecv:f2_ecv] += temp_ecv*r2_ecv.reshape(-1)
        temp_evc = D_aij[:,ncvs:,:ncvs].reshape(-1)
        s[s2_evc:f2_evc] += temp_evc*r2_evc.reshape(-1)

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            eris_cccc = eris_oooo[:ncvs,:ncvs,:ncvs,:ncvs].copy()
            eris_cccv = eris_oooo[:ncvs,:ncvs,:ncvs,ncvs:].copy()
            eris_cvcv = eris_oooo[:ncvs,ncvs:,:ncvs,ncvs:].copy()
            eris_ccvv = eris_oooo[:ncvs,:ncvs,ncvs:,ncvs:].copy()
            eris_ceec = eris_ovvo[:ncvs,:,:,:ncvs].copy()
            eris_ceev = eris_ovvo[:ncvs,:,:,ncvs:].copy()
            eris_veev = eris_ovvo[ncvs:,:,:,ncvs:].copy()
            eris_ccee = eris_oovv[:ncvs,:ncvs,:,:].copy()
            eris_cvee = eris_oovv[:ncvs,ncvs:,:,:].copy()
            eris_vvee = eris_oovv[ncvs:,ncvs:,:,:].copy()

            del eris_oooo
            del eris_oovv
            del eris_ovvo

            #eris_veev, eris_vvee
            s[s2_ecv:f2_ecv] += lib.einsum('klba,bJl->aJk',eris_vvee,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] += lib.einsum('jlba,blK->ajK',eris_vvee,
                                           r2_evc,optimize=True).reshape(-1)
            del eris_vvee
            s[s2_evc:f2_evc] += lib.einsum('jabl,bKl->ajK',eris_veev,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] -= 2*lib.einsum('jabl,blK->ajK',eris_veev,
                                             r2_evc,optimize=True).reshape(-1)
            del eris_veev
            #eris_ceev, eris_cvee
            s[s2_ecc:f2_ecc] += lib.einsum('Klba,bJl->aJK',eris_cvee,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_ecc:f2_ecc] += lib.einsum('Jlba,blK->aJK',eris_cvee,
                                           r2_evc,optimize=True).reshape(-1)
            s[s2_ecv:f2_ecv] += lib.einsum('Lkba,bJL->aJk',eris_cvee,
                                           r2_ecc,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] += lib.einsum('Ljba,bLK->ajK',eris_cvee,
                                           r2_ecc,optimize=True).reshape(-1)
            del eris_cvee
            s[s2_ecc:f2_ecc] += lib.einsum('Jabl,bKl->aJK',eris_ceev,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_ecc:f2_ecc] -= 2*lib.einsum('Jabl,blK->aJK',eris_ceev,
                                             r2_evc,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] += lib.einsum('Lbaj,bKL->ajK',eris_ceev,
                                           r2_ecc,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] -= 2*lib.einsum('Lbaj,bLK->ajK',eris_ceev,
                                             r2_ecc,optimize=True).reshape(-1)
            del eris_ceev
            #eris_ceec, eris_ccee
            s[s2_ecc:f2_ecc] += lib.einsum('KLba,bJL->aJK',eris_ccee,
                                           r2_ecc,optimize=True).reshape(-1)
            s[s2_ecc:f2_ecc] += lib.einsum('JLba,bLK->aJK',eris_ccee,
                                           r2_ecc,optimize=True).reshape(-1)
            s[s2_ecv:f2_ecv] += lib.einsum('JLba,bLk->aJk',eris_ccee,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] += lib.einsum('KLba,bjL->ajK',eris_ccee,
                                           r2_evc,optimize=True).reshape(-1)
            del eris_ccee
            s[s2_ecc:f2_ecc] += lib.einsum('JabL,bKL->aJK',eris_ceec,
                                           r2_ecc,optimize=True).reshape(-1)
            s[s2_ecc:f2_ecc] -= 2*lib.einsum('JabL,bLK->aJK',eris_ceec,
                                             r2_ecc,optimize=True).reshape(-1)
            s[s2_ecv:f2_ecv] -= 2*lib.einsum('JabL,bLk->aJk',eris_ceec,
                                             r2_ecv,optimize=True).reshape(-1)
            s[s2_ecv:f2_ecv] += lib.einsum('JabL,bkL->aJk',eris_ceec,
                                           r2_evc,optimize=True).reshape(-1)
            del eris_ceec
            #eris_cvcv, eris_ccvv
            s[s2_ecv:f2_ecv] -= lib.einsum('JIkl,aIl->aJk',eris_ccvv,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] -= lib.einsum('KIjl,alI->ajK',eris_ccvv,
                                           r2_evc,optimize=True).reshape(-1)
            del eris_ccvv
            s[s2_ecv:f2_ecv] -= lib.einsum('IkJl,alI->aJk',eris_cvcv,
                                           r2_evc,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] -= lib.einsum('KlIj,aIl->ajK',eris_cvcv,
                                           r2_ecv,optimize=True).reshape(-1)
            del eris_cvcv
            #eris_cccv
            s[s2_ecc:f2_ecc] -= lib.einsum('KIJl,alI->aJK',eris_cccv,
                                           r2_evc,optimize=True).reshape(-1)
            s[s2_ecc:f2_ecc] -= lib.einsum('JIKl,aIl->aJK',eris_cccv,
                                           r2_ecv,optimize=True).reshape(-1)
            s[s2_ecv:f2_ecv] -= lib.einsum('JILk,aIL->aJk',eris_cccv,
                                           r2_ecc,optimize=True).reshape(-1)
            s[s2_evc:f2_evc] -= lib.einsum('KLIj,aIL->ajK',eris_cccv,
                                           r2_ecc,optimize=True).reshape(-1)
            del eris_cccv
            #eris_cccc
            s[s2_ecc:f2_ecc] -= lib.einsum('KLJI,aIL->aJK',eris_cccc,
                                           r2_ecc,optimize=True).reshape(-1)
            del eris_cccc

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo
            t2_1 = adc.t2[0]
            t2_1_ccee = t2_1[:ncvs,:ncvs,:,:].copy()
            t2_1_cvee = t2_1[:ncvs,ncvs:,:,:].copy()
            t2_1_vcee = t2_1[ncvs:,:ncvs,:,:].copy()
            del t2_1
################ ADC(3) i - kja block and ajk - i ############################

            eris_oecc = eris_ovoo[:,:,:ncvs,:ncvs].copy()
            eris_oecv = eris_ovoo[:,:,:ncvs,ncvs:].copy()
            eris_ceco = eris_ovoo[:ncvs,:,:ncvs,:].copy()
            eris_cevo = eris_ovoo[:ncvs,:,ncvs:,:].copy()

            temp_1_ecc = lib.einsum('ijbc,aij->abc',t2_1_ccee, r2_ecc, optimize=True)
            temp_ecc = 0.25 * temp_1_ecc
            temp_ecc -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1_ccee, r2_ecc, optimize=True)
            temp_ecc -= 0.25 * lib.einsum('jibc,aij->abc',t2_1_ccee, r2_ecc, optimize=True)
            temp_ecc += 0.25 * lib.einsum('jibc,aji->abc',t2_1_ccee, r2_ecc, optimize=True)
            temp_1_ecv = lib.einsum('ijbc,aij->abc',t2_1_cvee, r2_ecv, optimize=True)
            temp_ecv = 0.25 * temp_1_ecv
            temp_ecv -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1_vcee, r2_ecv, optimize=True)
            temp_ecv -= 0.25 * lib.einsum('jibc,aij->abc',t2_1_vcee, r2_ecv, optimize=True)
            temp_ecv += 0.25 * lib.einsum('jibc,aji->abc',t2_1_cvee, r2_ecv, optimize=True)
            temp_1_evc = lib.einsum('ijbc,aij->abc',t2_1_vcee, r2_evc, optimize=True)
            temp_evc = 0.25 * temp_1_evc
            temp_evc -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1_cvee, r2_evc, optimize=True)
            temp_evc -= 0.25 * lib.einsum('jibc,aij->abc',t2_1_cvee, r2_evc, optimize=True)
            temp_evc += 0.25 * lib.einsum('jibc,aji->abc',t2_1_vcee, r2_evc, optimize=True)

            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            else:
                chnk_size = nocc
            a = 0
            temp_singles = np.zeros((ncvs))
            temp_doubles = np.zeros((nvir,nvir,nvir))
            for p in range(0,ncvs,chnk_size):
                if getattr(adc, 'with_df', None):
                    eris_ceee = dfadc.get_ovvv_df(
                        adc, eris.Lce, eris.Lee, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                else:
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                    eris_ceee = eris_ovvv[:ncvs,:,:,:].copy()
                    del eris_ovvv
                k = eris_ceee.shape[0]

                temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_ecc, eris_ceee, optimize=True)
                temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp_ecc, eris_ceee, optimize=True)
                temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_ecv, eris_ceee, optimize=True)
                temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp_ecv, eris_ceee, optimize=True)
                temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_evc, eris_ceee, optimize=True)
                temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp_evc, eris_ceee, optimize=True)
                temp_singles[a:a+k] += lib.einsum('abc,icab->i',
                                                  temp_1_ecc, eris_ceee, optimize=True)
                temp_singles[a:a+k] += lib.einsum('abc,icab->i',
                                                  temp_1_ecv, eris_ceee, optimize=True)
                temp_singles[a:a+k] += lib.einsum('abc,icab->i',
                                                  temp_1_evc, eris_ceee, optimize=True)
                temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ceee,optimize=True)
                s[s1:f1] += temp_singles
                s[s2_ecc:f2_ecc] += lib.einsum('cba,kjcb->ajk',
                                               temp_doubles, t2_1_ccee, optimize=True).reshape(-1)
                s[s2_ecv:f2_ecv] += lib.einsum('cba,kjcb->ajk',
                                               temp_doubles, t2_1_vcee, optimize=True).reshape(-1)
                s[s2_evc:f2_evc] += lib.einsum('cba,kjcb->ajk',
                                               temp_doubles, t2_1_cvee, optimize=True).reshape(-1)
                del eris_ceee, temp_singles, temp_doubles
                a += k

            # ADC(3) jka-i
            temp_1_c_a  = lib.einsum('I,lbIK->Kbl',r1,eris_oecc, optimize=True)
            temp_1_c_b  = -lib.einsum('I,IbKl->Kbl',r1,eris_ceco, optimize=True)
            temp_1_c = temp_1_c_a + temp_1_c_b
            temp_1_v_a  = lib.einsum('I,lbIk->kbl',r1,eris_oecv, optimize=True)
            temp_1_v_b  = -lib.einsum('I,Ibkl->kbl',r1,eris_cevo, optimize=True)
            temp_1_v = temp_1_v_a + temp_1_v_b

            t2_1 = adc.t2[0]
            t2_1_coee = t2_1[:ncvs,:,:,:].copy()
            t2_1_ocee = t2_1[:,:ncvs,:,:].copy()
            t2_1_voee = t2_1[ncvs:,:,:,:].copy()
            t2_1_ovee = t2_1[:,ncvs:,:,:].copy()
            del t2_1

            temp_ecc  = lib.einsum('Kbl,lJba->aJK',temp_1_c,t2_1_ocee,optimize=True)
            temp_ecc += lib.einsum('Kbl,Jlab->aJK',temp_1_c_a,t2_1_coee,optimize=True)
            temp_ecc -= lib.einsum('Kbl,lJab->aJK',temp_1_c_a,t2_1_ocee,optimize=True)
            temp_ecc += lib.einsum('Jbl,Klba->aJK',temp_1_c_b,t2_1_coee,optimize=True)
            temp_ecv  = lib.einsum('kbl,lJba->aJk',temp_1_v,t2_1_ocee,optimize=True)
            temp_ecv += lib.einsum('kbl,Jlab->aJk',temp_1_v_a,t2_1_coee,optimize=True)
            temp_ecv -= lib.einsum('kbl,lJab->aJk',temp_1_v_a,t2_1_ocee,optimize=True)
            temp_ecv += lib.einsum('Jbl,klba->aJk',temp_1_c_b,t2_1_voee,optimize=True)
            temp_evc  = lib.einsum('Kbl,ljba->ajK',temp_1_c,t2_1_ovee,optimize=True)
            temp_evc += lib.einsum('Kbl,jlab->ajK',temp_1_c_a,t2_1_voee,optimize=True)
            temp_evc -= lib.einsum('Kbl,ljab->ajK',temp_1_c_a,t2_1_ovee,optimize=True)
            temp_evc += lib.einsum('jbl,Klba->ajK',temp_1_v_b,t2_1_coee,optimize=True)
            s[s2_ecc:f2_ecc] += temp_ecc.reshape(-1)
            s[s2_ecv:f2_ecv] += temp_ecv.reshape(-1)
            s[s2_evc:f2_evc] += temp_evc.reshape(-1)

            del temp_1_c_a, temp_1_c_b, temp_1_c , temp_1_v_a, temp_1_v_b, temp_1_v

            # ADC(3) i-jka
            temp_a = lib.einsum('Jlab,aJK->blK',t2_1_coee,r2_ecc,optimize=True)
            temp_b = -lib.einsum('Jlab,aKJ->blK',t2_1_coee,r2_ecc,optimize=True)
            temp_c = -lib.einsum('Jlba,aJK->blK',t2_1_coee,r2_ecc,optimize=True)
            temp_d = lib.einsum('Jlba,aKJ->blK',t2_1_coee,r2_ecc,optimize=True)
            temp_ecc = temp_a + temp_b + temp_c + temp_d
            temp_1_ecc = -temp_a - temp_c
            temp_2_ecc = -temp_a
            temp_3_ecc = -temp_a - temp_b
            temp_4_ecc = -temp_d

            s[s1:f1] += lib.einsum('blK,lbIK->I',temp_ecc,eris_oecc,optimize=True)
            s[s1:f1] -= lib.einsum('blK,IbKl->I',temp_ecc,eris_ceco,optimize=True)
            s[s1:f1] -= lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_oecc,optimize=True)
            s[s1:f1] -= lib.einsum('blJ,lbIJ->I',temp_2_ecc,eris_oecc,optimize=True)
            s[s1:f1] += lib.einsum('blJ,IbJl->I',temp_2_ecc,eris_ceco,optimize=True)
            s[s1:f1] -= lib.einsum('blJ,lbIJ->I',temp_3_ecc,eris_oecc,optimize=True)
            s[s1:f1] += lib.einsum('blJ,IbJl->I',temp_4_ecc,eris_ceco,optimize=True)

            del temp_a, temp_b, temp_c, temp_d, temp_ecc, temp_1_ecc, temp_2_ecc, temp_3_ecc, temp_4_ecc

            temp_a = lib.einsum('Jlab,aJk->blk',t2_1_coee,r2_ecv,optimize=True)
            temp_b = -lib.einsum('lJab,aJk->blk',t2_1_ocee,r2_ecv,optimize=True)
            temp_c = -lib.einsum('Jlab,akJ->blk',t2_1_coee,r2_evc,optimize=True)
            temp_d = lib.einsum('lJab,akJ->blk',t2_1_ocee,r2_evc,optimize=True)
            temp_ecv = temp_a + temp_b + temp_c + temp_d
            temp_1_ecv = -temp_a - temp_b
            temp_2_ecv = -temp_a
            temp_3_ecv = -temp_a - temp_c
            temp_4_ecv = -lib.einsum('klba,aJk->blJ',t2_1_voee,r2_ecv,optimize=True)

            s[s1:f1] += lib.einsum('blk,lbIk->I',temp_ecv,eris_oecv,optimize=True)
            s[s1:f1] -= lib.einsum('blk,Ibkl->I',temp_ecv,eris_cevo,optimize=True)
            s[s1:f1] -= lib.einsum('blj,lbIj->I',temp_1_ecv,eris_oecv,optimize=True)
            s[s1:f1] -= lib.einsum('blj,lbIj->I',temp_2_ecv,eris_oecv,optimize=True)
            s[s1:f1] += lib.einsum('blj,Ibjl->I',temp_2_ecv,eris_cevo,optimize=True)
            s[s1:f1] -= lib.einsum('blj,lbIj->I',temp_3_ecv,eris_oecv,optimize=True)
            s[s1:f1] += lib.einsum('blJ,IbJl->I',temp_4_ecv,eris_ceco,optimize=True)

            del temp_a, temp_b, temp_c, temp_d, temp_ecv, temp_1_ecv, temp_2_ecv, temp_3_ecv, temp_4_ecv

            temp_a = -lib.einsum('jlab,aKj->blK',t2_1_voee,r2_ecv,optimize=True)
            temp_b = lib.einsum('ljab,aKj->blK',t2_1_ovee,r2_ecv,optimize=True)
            temp_c = lib.einsum('jlab,ajK->blK',t2_1_voee,r2_evc,optimize=True)
            temp_d = -lib.einsum('ljab,ajK->blK',t2_1_ovee,r2_evc,optimize=True)
            temp_evc = temp_a + temp_b + temp_c + temp_d
            temp_1_evc = -temp_c - temp_d
            temp_2_evc = -temp_c
            temp_3_evc = -temp_a - temp_c
            temp_4_evc = -lib.einsum('Klba,ajK->blj',t2_1_coee,r2_evc,optimize=True)

            s[s1:f1] += lib.einsum('blK,lbIK->I',temp_evc,eris_oecc,optimize=True)
            s[s1:f1] -= lib.einsum('blK,IbKl->I',temp_evc,eris_ceco,optimize=True)
            s[s1:f1] -= lib.einsum('blJ,lbIJ->I',temp_1_evc,eris_oecc,optimize=True)
            s[s1:f1] -= lib.einsum('blJ,lbIJ->I',temp_2_evc,eris_oecc,optimize=True)
            s[s1:f1] += lib.einsum('blJ,IbJl->I',temp_2_evc,eris_ceco,optimize=True)
            s[s1:f1] -= lib.einsum('blJ,lbIJ->I',temp_3_evc,eris_oecc,optimize=True)
            s[s1:f1] += lib.einsum('blj,Ibjl->I',temp_4_evc,eris_cevo,optimize=True)

            del temp_a, temp_b, temp_c, temp_d, temp_evc, temp_1_evc, temp_2_evc, temp_3_evc, temp_4_evc

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

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

    nocc = adc._nocc
    nvir = adc._nvir
    ncvs = adc.ncvs
    nval = nocc - ncvs

    t2_1 = adc.t2[0][:]
    t2_1_coee = t2_1[:ncvs,:,:,:].copy()
    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        t1_2 = adc.t1[0][:]
        t1_2_ce = t1_2[:ncvs,:].copy()

    n_singles = ncvs
    n_doubles_ecc = nvir * ncvs * ncvs
    n_doubles_ecv =  nvir * ncvs * nval

    dim = n_singles + n_doubles_ecc + 2 * n_doubles_ecv

    idn_occ= np.identity(nocc)

    s1 = 0
    f1 = n_singles
    s2_ecc = f1
    f2_ecc = s2_ecc + n_doubles_ecc
    s2_ecv = f2_ecc
    f2_ecv = s2_ecv + n_doubles_ecv
    s2_evc = f2_ecv
    f2_evc = s2_evc + n_doubles_ecv

    T = np.zeros((dim))

######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T[s1:f1] = idn_occ[orb, :ncvs]
        T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_1_coee, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_1_coee, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_1_coee, optimize=True)
        T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_1_coee, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_1_coee, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_1_coee, optimize=True)
    else :
        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            T[s1:f1] += t1_2_ce[:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        t2_1_t = t2_1.transpose(2,3,1,0)
        t2_1_t_eecc = t2_1_t[:,:,:ncvs,:ncvs].copy()
        t2_1_t_eecv = t2_1_t[:,:,:ncvs,ncvs:].copy()
        t2_1_t_eevc = t2_1_t[:,:,ncvs:,:ncvs].copy()

        T[s2_ecc:f2_ecc] = t2_1_t_eecc[(orb-nocc),:,:,:].reshape(-1)
        T[s2_ecv:f2_ecv] = t2_1_t_eecv[(orb-nocc),:,:,:].reshape(-1)
        T[s2_evc:f2_evc] = t2_1_t_eevc[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1][:]

        if orb >= nocc:
            t2_2_t = t2_2.transpose(2,3,1,0)
            t2_2_t_eecc = t2_2_t[:,:,:ncvs,:ncvs].copy()
            t2_2_t_eecv = t2_2_t[:,:,:ncvs,ncvs:].copy()
            t2_2_t_eevc = t2_2_t[:,:,ncvs:,:ncvs].copy()

            T[s2_ecc:f2_ecc] += t2_2_t_eecc[(orb-nocc),:,:,:].reshape(-1)
            T[s2_ecv:f2_ecv] += t2_2_t_eecv[(orb-nocc),:,:,:].reshape(-1)
            T[s2_evc:f2_evc] += t2_2_t_eevc[(orb-nocc),:,:,:].reshape(-1)

            del t2_2, t2_2_t_eecc, t2_2_t_eecv, t2_2_t_eevc

######### ADC(3) 1h part  ############################################

    if(method=='adc(3)'):
        t2_2 = adc.t2[1][:]
        t2_2_coee = t2_2[:ncvs,:,:,:].copy()
        if (adc.approx_trans_moments is False):
            t1_3 = adc.t1[1]
            t1_3_ce = t1_3[:ncvs,:].copy()
        t2_1_ocee = t2_1[:,:ncvs,:,:].copy()

        if orb < ncvs:
            T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_2_coee, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_2_coee, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_2_coee, optimize=True)
            T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_2_coee, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_2_coee, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_2_coee, optimize=True)

            T[s1:f1] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_coee, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kdc->i',t2_1_coee, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kcd->i',t2_1_coee, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] += 0.25*lib.einsum('ikcd,kcd->i',t2_1_coee, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_coee, t2_2[orb,:,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_coee, t2_2[orb,:,:,:],optimize=True)
        else:
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1_coee[:,:,(orb-nocc),:], t1_2,optimize=True)
            T[s1:f1] -= 0.5*lib.einsum('kic,kc->i',t2_1_ocee[:,:,(orb-nocc),:], t1_2,optimize=True)
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1_coee[:,:,(orb-nocc),:], t1_2,optimize=True)
            if (adc.approx_trans_moments is False):
                T[s1:f1] += t1_3_ce[:,(orb-nocc)]

        del t2_2
    del t2_1

    T_aaa_ecc = T[s2_ecc:f2_ecc].reshape(nvir,ncvs,ncvs).copy()
    T_aaa_ecv = T[s2_ecv:f2_ecv].reshape(nvir,ncvs,nval).copy()
    T_aaa_evc = T[s2_evc:f2_evc].reshape(nvir,nval,ncvs).copy()
    T_aaa_ecc_asym = T_aaa_ecc - T_aaa_ecc.transpose(0,2,1)
    T_aaa_ecv_asym = T_aaa_ecv - T_aaa_evc.transpose(0,2,1)
    T_aaa_evc_asym = T_aaa_evc - T_aaa_ecv.transpose(0,2,1)

    T[s2_ecc:f2_ecc] += T_aaa_ecc_asym.reshape(-1)
    T[s2_ecv:f2_ecv] += T_aaa_ecv_asym.reshape(-1)
    T[s2_evc:f2_evc] += T_aaa_evc_asym.reshape(-1)
    return T

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
    ncvs = adc.ncvs
    nval = nocc - ncvs

    n_singles = ncvs
    n_doubles_ecc = nvir * ncvs * ncvs
    n_doubles_ecv =  nvir * ncvs * nval

    f1 = n_singles
    s2_ecc = f1
    f2_ecc = s2_ecc + n_doubles_ecc
    s2_ecv = f2_ecc
    f2_ecv = s2_ecv + n_doubles_ecv
    s2_evc = f2_ecv
    f2_evc = s2_evc + n_doubles_ecv

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2_ecc = U[s2_ecc:f2_ecc,I].reshape(nvir,ncvs,ncvs)
        U2_ecv = U[s2_ecv:f2_ecv,I].reshape(nvir,ncvs,nval)
        U2_evc = U[s2_evc:f2_evc,I].reshape(nvir,nval,ncvs)

        UdotU = np.dot(U1, U1)
        UdotU += 2.*np.dot(U2_ecc.ravel(), U2_ecc.ravel()) - \
                           np.dot(U2_ecc.ravel(), U2_ecc.transpose(0,2,1).ravel())
        UdotU += 2.*np.dot(U2_ecv.ravel(), U2_ecv.ravel()) - \
                           np.dot(U2_ecv.ravel(), U2_evc.transpose(0,2,1).ravel())
        UdotU += 2.*np.dot(U2_evc.ravel(), U2_evc.ravel()) - \
                           np.dot(U2_evc.ravel(), U2_ecv.transpose(0,2,1).ravel())
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

    #TODO: Implement eigenvector analysis for CVS-RADC
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

    X = myadc.X

    if X is None:
        nroots = myadc.U.shape[1]
        P,X = myadc.get_properties(nroots)

    nroots = X.shape[1]
    dyson_mo = np.dot(myadc.mo_coeff,X)

    return dyson_mo


class RADCIPCVS(radc.RADC):
    '''restricted ADC for IP-CVS energies and spectroscopic amplitudes

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
            IP energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
        'mo_energy_b', 't1', 'mo_energy_a',
        'max_space', 't2', 'max_cycle',
        'nmo', 'transform_integrals', 'with_df', 'compute_properties',
        'approx_trans_moments', 'E', 'U', 'P', 'X',
        'evec_print_tol', 'spec_factor_print_tol', 'ncvs',
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
        self.ncvs = adc.ncvs

    kernel = radc.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
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
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag
