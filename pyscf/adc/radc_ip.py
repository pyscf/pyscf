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

    nocc = adc._nocc

    e_occ = adc.mo_energy[:nocc]

    idn_occ = np.identity(nocc)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo
    M_ij = np.zeros((nocc,nocc))

    # i-j block
    # Zeroth-order terms

    M_ij = lib.einsum('ij,j->ij', idn_occ ,e_occ)

    # Second-order terms
    t2_1 = t2[0][:]

    M_ij += 0.5 * 0.5 *  lib.einsum('ilde,jdel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('lide,jdel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('ilde,jedl->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += 0.5 * 0.5 *  lib.einsum('lide,jedl->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += 0.5 * lib.einsum('ilde,jdel->ij',t2_1, eris_ovvo,optimize=True)

    M_ij += 0.5 * 0.5 *  lib.einsum('jlde,idel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('ljde,idel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 * 0.5 *  lib.einsum('jlde,ldei->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += 0.5 * 0.5 *  lib.einsum('ljde,ldei->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += 0.5 * lib.einsum('jlde,idel->ij',t2_1, eris_ovvo,optimize=True)

    del t2_1
    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    # Third-order terms

    if (method == "adc(3)"):

        eris_oovv = eris.oovv
        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo

        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,jdli->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)

        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,idlj->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)
        t2_2 = t2[1][:]

        M_ij += 0.5 * 0.5* lib.einsum('ilde,jdel->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('lide,jdel->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('ilde,jedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += 0.5 * 0.5* lib.einsum('lide,jedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += 0.5 * lib.einsum('ilde,jdel->ij',t2_2, eris_ovvo,optimize=True)

        M_ij += 0.5 * 0.5* lib.einsum('jlde,ledi->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('ljde,ledi->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5 * 0.5* lib.einsum('jlde,iedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += 0.5 * 0.5* lib.einsum('ljde,iedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += 0.5 * lib.einsum('jlde,ledi->ij',t2_2, eris_ovvo,optimize=True)
        t2_1 = t2[0][:]

        log.timer_debug1("Starting the small integrals  calculation")

        temp_t2_v_1 = lib.einsum('lmde,jldf->mejf',t2_1, t2_1,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('mejf,mefi->ij',temp_t2_v_1, eris_ovvo,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('jfme,mefi->ij',temp_t2_v_1, eris_ovvo,optimize=True)
        M_ij +=  0.5 * lib.einsum('mejf,mife->ij',temp_t2_v_1, eris_oovv,optimize=True)
        M_ij +=  0.5 * lib.einsum('jfme,mife->ij',temp_t2_v_1, eris_oovv,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('meif,mefj->ij',temp_t2_v_1, eris_ovvo ,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('ifme,mefj->ij',temp_t2_v_1, eris_ovvo ,optimize=True)
        M_ij +=  0.5 * lib.einsum('meif,mjfe->ij',temp_t2_v_1, eris_oovv ,optimize=True)
        M_ij +=  0.5 * lib.einsum('ifme,mjfe->ij',temp_t2_v_1, eris_oovv ,optimize=True)
        del temp_t2_v_1

        temp_t2_v_2 = lib.einsum('lmde,ljdf->mejf',t2_1, t2_1,optimize=True)
        M_ij +=  0.5 * 4 * lib.einsum('mejf,mefi->ij',temp_t2_v_2, eris_ovvo,optimize=True)
        M_ij +=  0.5 * 4 * lib.einsum('meif,mefj->ij',temp_t2_v_2, eris_ovvo,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('meif,mjfe->ij',temp_t2_v_2, eris_oovv,optimize=True)
        M_ij -=  0.5 * 2 * lib.einsum('mejf,mife->ij',temp_t2_v_2, eris_oovv,optimize=True)
        del temp_t2_v_2

        temp_t2_v_3 = lib.einsum('mlde,jldf->mejf',t2_1, t2_1,optimize=True)
        M_ij += 0.5 * lib.einsum('mejf,mefi->ij',temp_t2_v_3, eris_ovvo,optimize=True)
        M_ij += 0.5 * lib.einsum('meif,mefj->ij',temp_t2_v_3, eris_ovvo,optimize=True)
        M_ij -= 0.5 * 2 *lib.einsum('meif,mjfe->ij',temp_t2_v_3, eris_oovv,optimize=True)
        M_ij -= 0.5 * 2 * lib.einsum('mejf,mife->ij',temp_t2_v_3, eris_oovv,optimize=True)
        del temp_t2_v_3

        temp_t2_v_8 = lib.einsum('lmdf,lmde->fe',t2_1, t2_1,optimize=True)
        M_ij += 3 *lib.einsum('fe,jief->ij',temp_t2_v_8, eris_oovv, optimize=True)
        M_ij -= 1.5 *lib.einsum('fe,jfei->ij',temp_t2_v_8, eris_ovvo, optimize=True)
        M_ij +=   lib.einsum('ef,jief->ij',temp_t2_v_8, eris_oovv, optimize=True)
        M_ij -= 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_8, eris_ovvo, optimize=True)
        del temp_t2_v_8

        temp_t2_v_9 = lib.einsum('lmdf,mlde->fe',t2_1, t2_1,optimize=True)
        M_ij -= 1.0 * lib.einsum('fe,jief->ij',temp_t2_v_9, eris_oovv, optimize=True)
        M_ij -= 1.0 * lib.einsum('ef,jief->ij',temp_t2_v_9, eris_oovv, optimize=True)
        M_ij += 0.5 * lib.einsum('fe,jfei->ij',temp_t2_v_9, eris_ovvo, optimize=True)
        M_ij += 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_9, eris_ovvo, optimize=True)
        del temp_t2_v_9

        temp_t2_v_10 = lib.einsum('lnde,lmde->nm',t2_1, t2_1,optimize=True)
        M_ij -= 3.0 * lib.einsum('nm,jinm->ij',temp_t2_v_10, eris_oooo, optimize=True)
        M_ij -= 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_10, eris_oooo, optimize=True)
        M_ij += 1.5 * lib.einsum('nm,jmni->ij',temp_t2_v_10, eris_oooo, optimize=True)
        M_ij += 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_10, eris_oooo, optimize=True)
        del temp_t2_v_10

        temp_t2_v_11 = lib.einsum('lnde,mlde->nm',t2_1, t2_1,optimize=True)
        M_ij += 1.0 * lib.einsum('nm,jinm->ij',temp_t2_v_11, eris_oooo, optimize=True)
        M_ij -= 0.5 * lib.einsum('nm,jmni->ij',temp_t2_v_11, eris_oooo, optimize=True)
        M_ij -= 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_11, eris_oooo, optimize=True)
        M_ij += 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_11, eris_oooo, optimize=True)
        del temp_t2_v_11

        temp_t2_v_12 = lib.einsum('inde,lmde->inlm',t2_1, t2_1,optimize=True)
        M_ij += 0.5 * 1.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('lmin,jlnm->ij',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('lmin,jmnl->ij',temp_t2_v_12, eris_oooo, optimize=True)

        M_ij += 0.5 * 0.25 * lib.einsum('inlm,jlnm->ji',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('inlm,lnmj->ji',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij += 0.5 * 1.00 * lib.einsum('inlm,ljmn->ji',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('lmin,lnmj->ji',temp_t2_v_12, eris_oooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('lmin,ljmn->ji',temp_t2_v_12, eris_oooo, optimize=True)
        del temp_t2_v_12

        temp_t2_v_13 = lib.einsum('inde,mlde->inml',t2_1, t2_1,optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('inml,jlnm->ij',temp_t2_v_13, eris_oooo, optimize=True)
        M_ij -= 0.5 * 0.25 * lib.einsum('mlin,jlnm->ij',temp_t2_v_13, eris_oooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('inml,jmnl->ij',temp_t2_v_13, eris_oooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('mlin,jmnl->ij',temp_t2_v_13, eris_oooo, optimize=True)

        M_ij -= 0.5 * 0.25 * lib.einsum('inml,jlnm->ji',temp_t2_v_13, eris_oooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13, eris_oooo, optimize=True)

        M_ij -= 0.5 * 0.25 * lib.einsum('inml,ljmn->ji',temp_t2_v_13, eris_oooo, optimize=True)
        M_ij += 0.5 * 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13, eris_oooo, optimize=True)
        del temp_t2_v_13
        del t2_1

    cput0 = log.timer_debug1("Completed M_ij ADC(n) calculation", *cput0)

    return M_ij


def get_diag(adc,M_ij=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij)
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] = D_aij.copy()

#    ###### Additional terms for the preconditioner ####
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv, np.ndarray):
#
#            eris_oooo = eris.oooo
#            eris_oovv = eris.oovv
#            eris_ovvo = eris.ovvo
#
#            eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
#            eris_oooo_p = eris_oooo_p.reshape(nocc*nocc, nocc*nocc)
#
#            temp = np.zeros((nvir, eris_oooo_p.shape[0]))
#            temp[:] += np.diag(eris_oooo_p)
#            diag[s2:f2] += -temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nocc, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s2:f2] += temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nocc, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s2:f2] += temp.reshape(-1)
#        else:
#            raise Exception("Precond not available for out-of-core and density-fitted algo")

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

    return diag


def matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method


    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nvir,nocc,nocc)

        eris_ovoo = eris.ovoo

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ij,j->i',M_ij,r1)

############ ADC(2) i - kja block #########################

        s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo, r2, optimize=True)
        s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo, r2, optimize=True)

########## ###### ADC(2) ajk - i block ############################

        temp = lib.einsum('jaki,i->ajk', eris_ovoo, r1, optimize=True).reshape(-1)
        s[s2:f2] += temp.reshape(-1)

################# ADC(2) ajk - bil block ############################

        s[s2:f2] += D_aij * r2.reshape(-1)

################ ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize=True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize=True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize=True).reshape(-1)

            s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize=True).reshape(-1)
            s[s2:f2] -=  lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize=True).reshape(-1)
            s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize=True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize=True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize=True).reshape(-1)
            s[s2:f2] -= lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize=True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize=True).reshape(-1)

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo
            t2_1 = adc.t2[0]

################ ADC(3) i - kja block and ajk - i ############################

            temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
            temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
            temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
            temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)

            temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)

            temp_singles = np.zeros((nocc))
            temp_doubles = np.zeros((nvir,nvir,nvir))

            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
                a = 0
                for p in range(0,nocc,chnk_size):
                    eris_ovvv = dfadc.get_ovvv_df(
                        adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                    k = eris_ovvv.shape[0]
                    temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
                    temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
                    temp_singles[a:a+k] += lib.einsum('abc,icab->i',
                                                      temp_1, eris_ovvv, optimize=True)
                    temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv,optimize=True)
                    s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles,
                                           t2_1, optimize=True).reshape(-1)
                    del eris_ovvv
                    del temp_doubles
                    a += k
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)

                temp_singles += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
                temp_singles -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
                temp_singles += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                temp_doubles = lib.einsum('i,icab->cba',r1,eris_ovvv,optimize=True)
                s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles,
                                       t2_1, optimize=True).reshape(-1)
                del eris_ovvv
                del temp_doubles

            s[s1:f1] += temp_singles

            temp = np.zeros_like(r2)
            temp =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
            temp -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
            temp -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
            temp += lib.einsum('ljab,akj->blk',t2_1,r2,optimize=True)
            temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1 =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
            temp_1 -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
            temp_1 += lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
            temp_1 -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)

            temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)

            s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)
            del temp
            del temp_1
            del temp_2

            temp = np.zeros_like(r2)
            temp = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
            temp += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
            temp += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
            temp -= lib.einsum('lkab,ajk->blj',t2_1,r2,optimize=True)
            temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1  = -2 * lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
            temp_1 += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
            temp_1 += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)

            temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)

            s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)

            del temp
            del temp_1
            del temp_2

            temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
            temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
            temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)

            temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
            temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1,optimize=True)
            temp -= lib.einsum('kbl,ljab->ajk',temp_2,t2_1,optimize=True)
            s[s2:f2] += temp.reshape(-1)

            temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
            temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
            s[s2:f2] -= temp_1.reshape(-1)

            del temp
            del temp_1
            del temp_2
            del t2_1

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

    t2_1 = adc.t2[0][:]
    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        t1_2 = adc.t1[0][:]

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    idn_occ = np.identity(nocc)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T = np.zeros((dim))

######## ADC(2) 1h part  ############################################
    if orb < nocc:
        T[s1:f1]  = idn_occ[orb, :]
        T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_1, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_1, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_1, optimize=True)
        T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_1, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_1, optimize=True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_1, optimize=True)
    else :
        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            T[s1:f1] += t1_2[:,(orb-nocc)]

######### ADC(2) 2h-1p  part  ############################################

        t2_1_t = t2_1.transpose(2,3,1,0)

        T[s2:f2] += t2_1_t[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1][:]

        if orb >= nocc:
            t2_2_t = t2_2.transpose(2,3,1,0)

            T[s2:f2] += t2_2_t[(orb-nocc),:,:,:].reshape(-1)
        del t2_2
######### ADC(3) 1h part  ############################################

    if(method=='adc(3)'):
        t2_2 = adc.t2[1][:]
        if (adc.approx_trans_moments is False):
            t1_3 = adc.t1[1]
        if orb < nocc:
            T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_2, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_2, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_2, optimize=True)
            T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_2, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_2, optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_2, optimize=True)

            T[s1:f1] += 0.25*lib.einsum('ikdc,kdc->i',t2_1, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kdc->i',t2_1, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kcd->i',t2_1, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] += 0.25*lib.einsum('ikcd,kcd->i',t2_1, t2_2[:,orb,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1, t2_2[orb,:,:,:],optimize=True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1, t2_2[orb,:,:,:],optimize=True)
        else:
            T[s1:f1] += 0.5 * lib.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize=True)
            T[s1:f1] -= 0.5*lib.einsum('kic,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize=True)
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize=True)
            if (adc.approx_trans_moments is False):
                T[s1:f1] += t1_3[:,(orb-nocc)]

        del t2_2
    del t2_1

    T_aaa = T[n_singles:].reshape(nvir,nocc,nocc).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)


    return T


def analyze_eigenvector(adc):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    evec_print_tol = adc.evec_print_tol
    U = adc.U

    logger.info(adc, "Number of occupied orbitals = %d", nocc)
    logger.info(adc, "Number of virtual orbitals =  %d", nvir)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nvir,nocc,nocc)
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
                i_idx = orb_idx + 1
                singles_idx.append(i_idx)
                singles_val.append(U_sorted[iter_num])

            if orb_idx >= n_singles:
                aij_idx = orb_idx - n_singles
                ij_rem = aij_idx % (nocc*nocc)
                a_idx = aij_idx//(nocc*nocc)
                i_idx = ij_rem//nocc
                j_idx = ij_rem % nocc
                doubles_idx.append((a_idx + 1 + n_singles, i_idx + 1, j_idx + 1))
                doubles_val.append(U_sorted[iter_num])

            iter_num += 1

        logger.info(adc, '%s | root %d | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',
                    adc.method ,I, U1dotU1, U2dotU2)

        if singles_val:
            logger.info(adc, "\n1h block: ")
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_val[idx])

        if doubles_val:
            logger.info(adc, "\n2h1p block: ")
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_val[idx])

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

    n_singles = nocc

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nvir,nocc,nocc)
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


def compute_dyson_mo(myadc):

    X = myadc.X

    if X is None:
        nroots = myadc.U.shape[1]
        P,X = myadc.get_properties(nroots)

    nroots = X.shape[1]
    dyson_mo = np.dot(myadc.mo_coeff,X)

    return dyson_mo

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

    t2_1 = adc.t2[0][:]
    t1_2 = adc.t1[0][:]

    nocc = adc._nocc
    nvir = adc._nvir
    nmo = nocc + nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

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

    L2 = L2.reshape(nvir,nocc,nocc)
    R2 = R2.reshape(nvir,nocc,nocc)

#####G^000#### block- ij
    rdm1[:nocc,:nocc] =  2*np.einsum('ij,m,m->ij',kd_oc,L1,R1,optimize=True)
    rdm1[:nocc,:nocc] -= np.einsum('i,j->ij',L1,R1,optimize=True)

    rdm1[:nocc,:nocc] += 4*np.einsum('ij,etu,etu->ij',kd_oc,L2,R2,optimize=True)
    rdm1[:nocc,:nocc] -= np.einsum('ij,etu,eut->ij',kd_oc,L2,R2,optimize=True)
    rdm1[:nocc,:nocc] -= np.einsum('ij,eut,etu->ij',kd_oc,L2,R2,optimize=True)
    rdm1[:nocc,:nocc] -= 4*np.einsum('eti,etj->ij',L2,R2,optimize=True)
    rdm1[:nocc,:nocc] += np.einsum('eti,ejt->ij',L2,R2,optimize=True)
    rdm1[:nocc,:nocc] += np.einsum('eit,etj->ij',L2,R2,optimize=True)

    rdm1[:nocc,:nocc] -= 2*np.einsum('g,g,hjcd,hicd->ij', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[:nocc,:nocc] += 2*np.einsum('g,g,jhcd,hicd->ij', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[:nocc,:nocc] -= 2*np.einsum('g,g,jhcd,ihcd->ij', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[:nocc,:nocc] += 2*np.einsum('g,h,jgcd,ihcd->ij', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[:nocc,:nocc] -= np.einsum('g,h,gjcd,ihcd->ij', L1,R1,t2_1,t2_1,optimize=True)
    temp_ij = np.einsum('g,j,ghcd,ihcd->ij',L1,R1,t2_1,t2_1,optimize=True)
    temp_ij -= 0.25*np.einsum('g,j,hgcd,ihcd->ij',L1,R1,t2_1,t2_1,optimize=True)
    temp_ij -= 0.25*np.einsum('g,j,ghcd,hicd->ij',L1,R1,t2_1,t2_1,optimize=True)
    rdm1[:nocc,:nocc] += temp_ij
    rdm1[:nocc,:nocc] += temp_ij.T

########## block- ab
    rdm1[nocc:,nocc:] = np.einsum('atu,btu->ab', L2,R2,optimize=True)
    rdm1[nocc:,nocc:] -= 0.5*np.einsum('aut,btu->ab', L2,R2,optimize=True)
    rdm1[nocc:,nocc:] -= 0.5*np.einsum('atu,but->ab', L2,R2,optimize=True)
    rdm1[nocc:,nocc:] += np.einsum('atu,btu->ab', L2,R2,optimize=True)

    rdm1[nocc:,nocc:] += 4*np.einsum('g,g,hmbc,hmac->ab', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[nocc:,nocc:] -= np.einsum('g,g,mhbc,hmac->ab', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[nocc:,nocc:] -= np.einsum('g,g,hmbc,mhac->ab', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[nocc:,nocc:] -= 4*np.einsum('g,h,hmbc,gmac->ab', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[nocc:,nocc:] += np.einsum('g,h,mhbc,gmac->ab', L1,R1,t2_1,t2_1,optimize=True)
    rdm1[nocc:,nocc:] += np.einsum('g,h,hmbc,mgac->ab', L1,R1,t2_1,t2_1,optimize=True)

########### block- ia
    rdm1[:nocc,nocc:] = -np.einsum('n,ani->ia', R1,L2,optimize=True)
    rdm1[:nocc,nocc:] += 2*np.einsum('n,ain->ia', R1,L2,optimize=True)

    rdm1[:nocc,nocc:] -= 2*np.einsum('g,cgh,ihac->ia', L1,R2,t2_1,optimize=True)
    rdm1[:nocc,nocc:] += np.einsum('g,cgh,hiac->ia', L1,R2,t2_1,optimize=True)
    rdm1[:nocc,nocc:] += 4*np.einsum('g,chg,ihac->ia', L1,R2,t2_1,optimize=True)
    rdm1[:nocc,nocc:] -= np.einsum('g,chg,hiac->ia', L1,R2,t2_1,optimize=True)
    rdm1[:nocc,nocc:] -= np.einsum('g,cgh,ihac->ia', L1,R2,t2_1,optimize=True)
    rdm1[:nocc,nocc:] += np.einsum('i,cgh,ghac->ia', L1,R2,t2_1,optimize=True)
    rdm1[:nocc,nocc:] -= 2*np.einsum('i,chg,ghac->ia', L1,R2,t2_1,optimize=True)

    rdm1[:nocc,nocc:] += np.einsum('g,g,ia->ia', L1,R1,t1_2,optimize=True)
    rdm1[:nocc,nocc:] += np.einsum('g,g,ia->ia', L1,R1,t1_2,optimize=True)
    rdm1[:nocc,nocc:] -= np.einsum('g,i,ga->ia', R1,L1,t1_2,optimize=True)

############ block- ai
    rdm1[nocc:,:nocc] = rdm1[:nocc,nocc:].T

    return rdm1

class RADCIP(radc.RADC):
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
            IP energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''

    _keys = set((
        'tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
        'mo_energy_b', 't1', 'mo_energy_a',
        'max_space', 't2', 'max_cycle',
        'nmo', 'transform_integrals', 'with_df', 'compute_properties',
        'approx_trans_moments', 'E', 'U', 'P', 'X',
        'evec_print_tol', 'spec_factor_print_tol',
    ))

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
    get_diag = get_diag
    matvec = matvec
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
    get_properties = get_properties
    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector
    analyze = analyze
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
