# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

import itertools

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import mp
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_ao2mo
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx,_padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa

def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

    kptlist = [0]

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
       raise NotImplementedError(adc.method)

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    #if adc.verbose >= logger.WARN:
    #    adc.check_sanity()
    #adc.dump_flags()

    if eris is None:
        eris = adc.transform_integrals()

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc
    size = n_singles + n_doubles
    nroots = min(nroots,size)

    if kptlist is None:
        kptlist = range(nkpts)

    #if dtype is None:
    dtype = np.result_type(adc.t2)

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    convs = np.zeros((len(kptlist),nroots), dtype)

    imds = adc.get_imds(eris)

    for k, kshift in enumerate(kptlist):
        matvec, diag = adc.gen_matvec(kshift, imds, eris)
        guess = adc.get_init_guess(nroots, diag, ascending = True)
        conv_k,evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space,tol_residual=adc.tol_residual)
    
        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        convs[k] = conv_k
#    adc.U = np.array(U).T.copy()
#
#    if adc.compute_properties:
#        adc.P,adc.X = adc.get_properties(nroots)
#
#    nfalse = np.shape(conv)[0] - np.sum(conv)
#
#    str = ("\n*************************************************************"
#           "\n            ADC calculation summary"
#           "\n*************************************************************")
#    logger.info(adc, str)
#
#    if nfalse >= 1:
#        logger.warn(adc, "Davidson iterations for " + str(nfalse) + " root(s) not converged\n")
#
#    for n in range(nroots):
#        print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' % (adc.method, n, adc.E[n], adc.E[n]*27.2114))
#        if adc.compute_properties:
#            print_string += ("|  Spec factors = %10.8f  " % adc.P[n])
#        print_string += ("|  conv = %s" % conv[n])
#        logger.info(adc, print_string)
#
#    log.timer('ADC', *cput0)
#
    return convs, evals, evecs
    #return evals


def compute_amplitudes_energy(myadc, eris, verbose=None):

    t2 = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t2, eris)

    return e_corr, t2


def compute_amplitudes(myadc, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    #if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
    #    raise NotImplementedError(myadc.method)

    nmo = myadc.nmo
    nocc = myadc.nocc
    nvir = nmo - nocc
    nkpts = myadc.nkpts

    # Compute first-order doubles t2 (tijab)
    t2_1 = np.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)

    mo_energy =  myadc.mo_energy
    mo_coeff =  myadc.mo_coeff
    mo_coeff, mo_energy = _add_padding(myadc, mo_coeff, mo_energy)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(myadc, kind="split")
   
    eris_oovv = eris.oovv.copy()

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

    t2 = t2_1
    return t2

#
#    e = myadc.mo_energy
#    d_ij = e[:nocc][:,None] + e[:nocc]
#    d_ab = e[nocc:][:,None] + e[nocc:]
#
#    D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
#    D2 = D2.reshape((nocc,nocc,nvir,nvir))
#
#    D1 = e[:nocc][:None].reshape(-1,1) - e[nocc:].reshape(-1)
#    D1 = D1.reshape((nocc,nvir))
#
#    t2_1 = v2e_oovv/D2
#    if not isinstance(eris.oooo, np.ndarray):
#        t2_1 = radc_ao2mo.write_dataset(t2_1)
#        
#    del v2e_oovv
#    del D2
#
#    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

#    # Compute second-order singles t1 (tij)
#
#    if isinstance(eris.ovvv, type(None)):
#        chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
#    else:
#        chnk_size = nocc
#    a = 0
#    t1_2 = np.zeros((nocc,nvir))
#
#    for p in range(0,nocc,chnk_size):
#        if getattr(myadc, 'with_df', None):
#            eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
#        else :
#            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
#        k = eris_ovvv.shape[0]
       
#        t1_2 += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
#        t1_2 -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)
#        t1_2 -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
#        t1_2 += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)
#
#        t1_2 += lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
#        del eris_ovvv
#        a += k
#
#    t1_2 -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1[:],optimize=True)
#    t1_2 += 0.5*lib.einsum('lcki,lkac->ia',eris_ovoo,t2_1[:],optimize=True)
#    t1_2 -= 0.5*lib.einsum('kcli,lkac->ia',eris_ovoo,t2_1[:],optimize=True)
#    t1_2 += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1[:],optimize=True)
#    t1_2 -= lib.einsum('lcki,klac->ia',eris_ovoo,t2_1[:],optimize=True)
#
#    t1_2 = t1_2/D1
#
#    cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)
#
#    t2_2 = None
#    t1_3 = None
#    t2_1_vvvv = None
#
#    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
#
#    # Compute second-order doubles t2 (tijab)
#
#        eris_oooo = eris.oooo
#        eris_ovvo = eris.ovvo
#
#        if isinstance(eris.vvvv, np.ndarray):
#            eris_vvvv = eris.vvvv
#            temp = t2_1.reshape(nocc*nocc,nvir*nvir)
#            t2_1_vvvv = np.dot(temp,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
#        elif isinstance(eris.vvvv, list):
#            t2_1_vvvv = contract_ladder(myadc,t2_1[:],eris.vvvv)
#        else:
#            t2_1_vvvv = contract_ladder(myadc,t2_1[:],eris.Lvv)
#
#        if not isinstance(eris.oooo, np.ndarray):
#            t2_1_vvvv = radc_ao2mo.write_dataset(t2_1_vvvv)
#
#        t2_2 = t2_1_vvvv[:].copy()
#
#        t2_2 += lib.einsum('kilj,klab->ijab',eris_oooo,t2_1[:],optimize=True)
#        t2_2 += lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1[:],optimize=True)
#        t2_2 -= lib.einsum('kcbj,ikca->ijab',eris_ovvo,t2_1[:],optimize=True)
#        t2_2 += lib.einsum('kcbj,ikac->ijab',eris_ovvo,t2_1[:],optimize=True)
#        t2_2 -= lib.einsum('kjbc,ikac->ijab',eris_oovv,t2_1[:],optimize=True)
#        t2_2 -= lib.einsum('kibc,kjac->ijab',eris_oovv,t2_1[:],optimize=True)
#        t2_2 -= lib.einsum('kjac,ikcb->ijab',eris_oovv,t2_1[:],optimize=True)
#        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1[:],optimize=True)
#        t2_2 -= lib.einsum('kcai,jkcb->ijab',eris_ovvo,t2_1[:],optimize=True)
#        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1[:],optimize=True)
#        t2_2 -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1[:],optimize=True)
#
#        D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
#        D2 = D2.reshape((nocc,nocc,nvir,nvir))
#
#        t2_2 = t2_2/D2
#        if not isinstance(eris.oooo, np.ndarray):
#            t2_2 = radc_ao2mo.write_dataset(t2_2)
#        del D2
#
#    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)
#        
#    if (myadc.method == "adc(3)"):
#
#        eris_ovoo = eris.ovoo
#
#        t1_3 =  lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)
#        t1_3 -= lib.einsum('d,liad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)
#        t1_3 += lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)
# 
#        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
#        t1_3 += lib.einsum('l,liad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
#        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1[:],t1_2,optimize=True)
# 
#        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1[:], t1_2,optimize=True)
#        t1_3 -= 0.5*lib.einsum('a,liad,ld->ia',e[nocc:],t2_1[:], t1_2,optimize=True)
#        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)
# 
#        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
#        t1_3 += 0.5*lib.einsum('i,liad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
#        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1[:],t1_2,optimize=True)
# 
#        t1_3 += lib.einsum('ld,iadl->ia',t1_2,eris_ovvo,optimize=True)
#        t1_3 -= lib.einsum('ld,ladi->ia',t1_2,eris_ovvo,optimize=True)
#        t1_3 += lib.einsum('ld,iadl->ia',t1_2,eris_ovvo,optimize=True)
# 
#        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo ,optimize=True)
#        t1_3 -= lib.einsum('ld,liad->ia',t1_2,eris_oovv ,optimize=True)
#        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo,optimize=True)
# 
#        t1_3 -= 0.5*lib.einsum('lmad,mdli->ia',t2_2[:],eris_ovoo,optimize=True)
#        t1_3 += 0.5*lib.einsum('mlad,mdli->ia',t2_2[:],eris_ovoo,optimize=True)
#        t1_3 += 0.5*lib.einsum('lmad,ldmi->ia',t2_2[:],eris_ovoo,optimize=True)
#        t1_3 -= 0.5*lib.einsum('mlad,ldmi->ia',t2_2[:],eris_ovoo,optimize=True)
#        t1_3 -=     lib.einsum('lmad,mdli->ia',t2_2[:],eris_ovoo,optimize=True)
# 
#        if isinstance(eris.ovvv, type(None)):
#            chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
#        else :
#            chnk_size = nocc
#        a = 0
#
#        for p in range(0,nocc,chnk_size):
#            if getattr(myadc, 'with_df', None):
#                eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
#            else :
#                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
#            k = eris_ovvv.shape[0]
#
#            t1_3 += 0.5*lib.einsum('ilde,lead->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
#            t1_3 -= 0.5*lib.einsum('lide,lead->ia', t2_2[a:a+k],eris_ovvv,optimize=True)
#
#            t1_3 -= 0.5*lib.einsum('ilde,ldae->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
#            t1_3 += 0.5*lib.einsum('lide,ldae->ia', t2_2[a:a+k],eris_ovvv,optimize=True)
#
#            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
#            t1_3 += lib.einsum('ildf,mefa,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
#            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
#            t1_3 -= lib.einsum('lidf,mefa,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
#
#            t1_3 += lib.einsum('ildf,mafe,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
#            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
#            t1_3 -= lib.einsum('lidf,mafe,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
#            t1_3 += lib.einsum('lidf,mafe,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
#
#            t1_3 += lib.einsum('ilfd,mefa,mled->ia',  t2_1[:],eris_ovvv, t2_1[a:a+k],optimize=True)
#            t1_3 -= lib.einsum('ilfd,mafe,mled->ia',  t2_1[:],eris_ovvv, t2_1[a:a+k],optimize=True)
#
#            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#            t1_3 -= 0.5*lib.einsum('liaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 += 0.5*lib.einsum('liaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#
#            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#            t1_3 += 0.5*lib.einsum('liaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 -= 0.5*lib.einsum('liaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#
#            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,ifea,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.5*lib.einsum('mldf,ifea,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,ifea,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= lib.einsum('mlfd,ifea,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.25*lib.einsum('lmef,iedf,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.25*lib.einsum('mlef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.25*lib.einsum('mlef,iedf,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.25*lib.einsum('lmef,ifde,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.25*lib.einsum('mlef,ifde,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.25*lib.einsum('mlef,ifde,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#
#            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#
#            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#            t1_3 += lib.einsum('ilaf,mefd,mled->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
#
#            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#            t1_3[a:a+k] -= lib.einsum('lmef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
#
#            t1_3 += lib.einsum('ilde,lead->ia',t2_2[:,a:a+k],eris_ovvv,optimize=True)
#
#            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1[:],eris_ovvv, t2_1[:,a:a+k],optimize=True)
#            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1[:],eris_ovvv, t2_1[:,a:a+k],optimize=True)
#
#            t1_3 += lib.einsum('ilfd,mefa,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k] ,optimize=True)
#            t1_3 -= lib.einsum('ilfd,mefa,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k] ,optimize=True)
#
#            t1_3 += lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#            t1_3 -= lib.einsum('liaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
#
#            del eris_ovvv
#            a += k
#
#        t1_3 += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.25*lib.einsum('inde,lamn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.25*lib.einsum('nide,lamn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.25*lib.einsum('nide,lamn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.25*lib.einsum('inde,maln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.25*lib.einsum('nide,maln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.25*lib.einsum('nide,maln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('inde,lamn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
# 
#        t1_3 += 0.5 * lib.einsum('inad,lemn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('niad,lemn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('niad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('inad,meln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('niad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('niad,meln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('niad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('niad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5 * lib.einsum('inad,lemn,lmed->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('inad,meln,mled->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 += 0.5 * lib.einsum('inad,lemn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('inad,meln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5 * lib.einsum('lnde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('lnde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('nlde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('nlde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 += 0.5 * lib.einsum('lnde,naim,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('lnde,naim,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5 * lib.einsum('nlde,naim,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5 * lib.einsum('nlde,naim,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= lib.einsum('nled,ianm,mled->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('nled,naim,mled->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5*lib.einsum('lnde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += 0.5*lib.einsum('nlde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= 0.5*lib.einsum('nlde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= lib.einsum('lnde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= lib.einsum('lnde,ienm,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('lnde,ienm,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('nlde,ienm,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= lib.einsum('nlde,ienm,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= lib.einsum('nlde,neim,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('nlde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 += lib.einsum('nled,ienm,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 -= lib.einsum('nled,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('lned,ienm,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#        t1_3 += lib.einsum('nlde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
#
#        t1_3 = t1_3/D1
#
#    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)
#
#    t1 = (t1_2, t1_3)
#    t2 = (t2_1, t2_2)


#
#    return t1, t2, t2_1_vvvv
#    return t2_1
#
#

def compute_energy(myadc, t2, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)
    nkpts = myadc.nkpts

    emp2 = 0.0 

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):

        emp2 += 0.5 * lib.einsum('ijab,iajb',  t2[ki,kj,ka], eris.ovov[ki,ka,kj],optimize=True)
        emp2 -= 0.5 * lib.einsum('ijab,jaib', t2[ki,kj,ka], eris.ovov[kj,ka,ki],optimize=True)
        emp2 -= 0.5 * lib.einsum('jiab,iajb', t2[kj,ki,ka], eris.ovov[ki,ka,kj],optimize=True)
        emp2 += 0.5 * lib.einsum('jiab,jaib', t2[kj,ki,ka], eris.ovov[kj,ka,ki],optimize=True)
        emp2 += lib.einsum('ijab,iajb', t2[ki,kj,ka], eris.ovov[ki,ka,kj],optimize=True)
  
    emp2 = emp2.real / nkpts
    return emp2


class RADC(pyscf.adc.radc.RADC):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        assert (isinstance(mf, scf.khf.KSCF))

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
      
        self._scf = mf
        self.kpts = self._scf.kpts
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.method = "adc(2)"
        self.method_type = "ea"

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.cell = self._scf.cell
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.frozen = frozen

        self._nocc = None
        self._nmo = None
        self._nvir = None

        self.t2 = None
        self.e_corr = None

        ##################################################
        # don't modify the following attributes, unless you know what you are doing
#        self.keep_exxdiv = False
#
#        keys = set(['kpts', 'khelper', 'ip_partition',
#                    'ea_partition', 'max_space', 'direct'])
#        self._keys = self._keys.union(keys)
#        self.__imds__ = None
        self.mo_energy = mf.mo_energy

    transform_integrals = kadc_ao2mo.transform_integrals_incore
    compute_amplitudes = compute_amplitudes
    compute_energy = compute_energy
    compute_amplitudes_energy = compute_amplitudes_energy

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    get_nocc = get_nocc
    get_nmo = get_nmo

    def kernel_gs(self):

        eris = self.transform_integrals()
        self.e_corr,self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)

        return self.e_corr, self.t2

    def kernel(self, nroots=1, guess=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
    
        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)
    
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()
    
#        nmo = self._nmo
#        nao = self.mo_coeff.shape[0]
#        nmo_pair = nmo * (nmo+1) // 2
#        nao_pair = nao * (nao+1) // 2
#        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
#        mem_now = lib.current_memory()[0]
#
#        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
#           if getattr(self, 'with_df', None): 
#               self.with_df = self.with_df
#           else :
#               self.with_df = self._scf.with_df
#
#           def df_transform():
#              return radc_ao2mo.transform_integrals_df(self)
#           self.transform_integrals = df_transform
#        elif (self._scf._eri is None or
#            (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
#           def outcore_transform():
#               return radc_ao2mo.transform_integrals_outcore(self)
#           self.transform_integrals = outcore_transform

        eris = self.transform_integrals() 
            
#        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self.e_corr, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
#        self._finalize()
#
        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
             con, e, v = self.ea_adc(nroots=nroots, guess=guess, eris=eris)
#            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)
#
        elif(self.method_type == "ip"):
             #e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(nroots=nroots, guess=guess, eris=eris)
             con, e, v = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)
#        self._adc_es = adc_es
#        return e_exc, v_exc, spec_fac, x
        return con, e, v

    def ip_adc(self, nroots=1, guess=None, eris=None):
        adc_es = RADCIP(self)
        con, e, v = adc_es.kernel(nroots, guess, eris)
        #e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        #return e_exc, v_exc, spec_fac, x, adc_es
        return con, e, v

    def ea_adc(self, nroots=1, guess=None, eris=None):
        adc_es = RADCEA(self)
        con, e, v = adc_es.kernel(nroots, guess, eris)
        #e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        #return e_exc, v_exc, spec_fac, x, adc_es
        return con, e, v

def get_imds_ea(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

#    t1 = adc.t1
    t2_1 = adc.t2
    #mymp = mp.KMP2(adc._scf)
    #emp2, t2_1 = mymp.kernel()
    #t2_1 = mp.kmp2.kernel()[1]
#    t1_2 = t1[0]
#
#    nocc = adc._nocc
#    nvir = adc._nvir
#
    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nmo =  adc.get_nmo
    nocc = adc.t2.shape[3]
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_occ = np.identity(nocc)
    M_ij = np.empty((nkpts,nocc,nocc),dtype=t2_1.dtype)

#    if eris is None:
#        eris = adc.transform_integrals()
#
    eris_ovov = eris.ovov

    # i-j block
    # Zeroth-order terms

    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ , e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -=  lib.einsum('d,ilde,ljde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] -=  lib.einsum('d,lide,jlde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,lide,ljde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,iled,jled->ij',e_vir[kd],t2_1[ki,kl,ke], t2_1[kj,kl,ke].conj(), optimize=True)

                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)

                M_ij_t = lib.einsum('ilde,jlde->ij', t2_1[ki,kl,kd],t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= lib.einsum('i,ij->ij',e_occ[ki],M_ij_t, optimize=True)
                M_ij[ki] -= lib.einsum('j,ij->ij',e_occ[kj],M_ij_t, optimize=True)
                del M_ij_t
                
                M_ij_t = lib.einsum('ilde,ljde->ij', t2_1[ki,kl,kd],t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t, optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('j,ij->ij',e_occ[kj],M_ij_t, optimize=True)
                del M_ij_t

                M_ij[ki] += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1[ki,kl,kd], eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('lide,jdle->ij',t2_1[kl,ki,kd], eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1[ki,kl,kd], eris_ovov[kj,ke,kl],optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('lide,jeld->ij',t2_1[kl,ki,kd], eris_ovov[kj,ke,kl],optimize=True)
                M_ij[ki] += lib.einsum('ilde,jdle->ij',t2_1[ki,kl,kd], eris_ovov[kj,kd,kl],optimize=True)
                
                M_ij[ki] += 0.5 *  lib.einsum('jlde,idle->ij',t2_1[kj,kl,kd].conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('ljde,idle->ij',t2_1[kl,kj,kd].conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1[kj,kl,kd].conj(), eris_ovov[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('ljde,ldie->ij',t2_1[kl,kj,kd].conj(), eris_ovov[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] += lib.einsum('jlde,idle->ij',t2_1[kj,kl,kd].conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
    
    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)

    return M_ij

def get_imds_ip(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

#    t1 = adc.t1
    t2_1 = adc.t2
    #mymp = mp.KMP2(adc._scf)
    #emp2, t2_1 = mymp.kernel()
    #t2_1 = mp.kmp2.kernel()[1]
#    t1_2 = t1[0]
#
#    nocc = adc._nocc
#    nvir = adc._nvir
#
    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nmo =  adc.get_nmo
    nocc = adc.t2.shape[3]
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_occ = np.identity(nocc)
    M_ij = np.empty((nkpts,nocc,nocc),dtype=t2_1.dtype)

#    if eris is None:
#        eris = adc.transform_integrals()
#
    eris_ovov = eris.ovov

    # i-j block
    # Zeroth-order terms

    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ , e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -=  lib.einsum('d,ilde,ljde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] -=  lib.einsum('d,lide,jlde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,lide,ljde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,iled,jled->ij',e_vir[kd],t2_1[ki,kl,ke], t2_1[kj,kl,ke].conj(), optimize=True)

                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_1[kj,kl,kd].conj(), optimize=True)

                M_ij_t = lib.einsum('ilde,jlde->ij', t2_1[ki,kl,kd],t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= lib.einsum('i,ij->ij',e_occ[ki],M_ij_t, optimize=True)
                M_ij[ki] -= lib.einsum('j,ij->ij',e_occ[kj],M_ij_t, optimize=True)
                del M_ij_t
                
                M_ij_t = lib.einsum('ilde,ljde->ij', t2_1[ki,kl,kd],t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t, optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('j,ij->ij',e_occ[kj],M_ij_t, optimize=True)
                del M_ij_t

                M_ij[ki] += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1[ki,kl,kd], eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('lide,jdle->ij',t2_1[kl,ki,kd], eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1[ki,kl,kd], eris_ovov[kj,ke,kl],optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('lide,jeld->ij',t2_1[kl,ki,kd], eris_ovov[kj,ke,kl],optimize=True)
                M_ij[ki] += lib.einsum('ilde,jdle->ij',t2_1[ki,kl,kd], eris_ovov[kj,kd,kl],optimize=True)
                
                M_ij[ki] += 0.5 *  lib.einsum('jlde,idle->ij',t2_1[kj,kl,kd].conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('ljde,idle->ij',t2_1[kl,kj,kd].conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1[kj,kl,kd].conj(), eris_ovov[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('ljde,ldie->ij',t2_1[kl,kj,kd].conj(), eris_ovov[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] += lib.einsum('jlde,idle->ij',t2_1[kj,kl,kd].conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
    
    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)

    return M_ij


def ea_adc_diag(adc,kshift,M_ij=None,eris=None):
   
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ij is None:
        M_ij = adc.get_imds()

    #nocc = adc._nocc
    #nvir = adc._nvir
    nkpts = adc.nkpts
    t2 = adc.t2
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

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.t2.shape[3]
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_ij = np.zeros((nkpts,nocc,nkpts,nocc), dtype=t2.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    e_ij = e_occ[:,:,None,None] + e_occ[None,None,:,:]
    e_ija = - e_vir[:,:,None,None,None,None] + e_ij[None,None,:,:,:,:] 
    e_ija = e_ija.transpose(0,2,4,1,3,5).copy()

    diag = np.zeros((dim), dtype=np.complex)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij[kshift])
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] += e_ija[kshift].reshape(-1) 

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

    return diag

def ip_adc_diag(adc,kshift,M_ij=None,eris=None):
   
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ij is None:
        M_ij = adc.get_imds()

    #nocc = adc._nocc
    #nvir = adc._nvir
    nkpts = adc.nkpts
    t2 = adc.t2
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

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.t2.shape[3]
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_ij = np.zeros((nkpts,nocc,nkpts,nocc), dtype=t2.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    e_ij = e_occ[:,:,None,None] + e_occ[None,None,:,:]
    e_ija = - e_vir[:,:,None,None,None,None] + e_ij[None,None,:,:,:,:] 
    e_ija = e_ija.transpose(0,2,4,1,3,5).copy()

    diag = np.zeros((dim), dtype=np.complex)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij[kshift])
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] += e_ija[kshift].reshape(-1) 

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

    return diag

def ea_adc_matvec(adc, kshift, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    t2 = adc.t2
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.t2.shape[3]
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_ij = np.zeros((nkpts,nocc,nkpts,nocc), dtype=t2.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    e_ij = e_occ[:,:,None,None] + e_occ[None,None,:,:]
    e_ija = - e_vir[:,:,None,None,None,None] + e_ij[None,None,:,:,:,:] 
    e_ija = e_ija.transpose(0,2,4,1,3,5).copy()

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim), dtype=np.complex)

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nkpts,nkpts,nvir,nocc,nocc)
        temp = np.zeros_like((r2),dtype=np.complex)
        temp_aij = np.zeros_like((r2),dtype=np.complex)
        
        eris_ovoo = eris.ovoo
        eris_vooo = eris.vooo

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ij,j->i',M_ij[kshift],r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]

                s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo[kj,ka,kk], r2[ka,kj], optimize = True)
                s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo[kk,ka,kj], r2[ka,kj], optimize = True)

################### ADC(2) ajk - i block ############################

                temp[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk].conj(), r1, optimize = True)

################# ADC(2) ajk - bil block ############################

       # for kj in range(nkpts):
       #     for kk in range(nkpts):
       #         ka = kconserv[kk, kshift, kj]
                #temp_aij[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]
                temp[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]

        s[s2:f2] += temp.reshape(-1)
        #s[s2:f2] += temp_aij.reshape(-1)

################ ADC(3) ajk - bil block ############################
#
#        if (method == "adc(2)-x" or method == "adc(3)"):
#        
#               eris_oooo = eris.oooo
#               eris_oovv = eris.oovv
#               eris_ovvo = eris.ovvo
#               
#               s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
#               s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)
#               
#               s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               
#               s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               
#               s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               
#               s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               
#        if (method == "adc(3)"):
#
#               eris_ovoo = eris.ovoo
#               t2_1 = adc.t2[0]
#
################# ADC(3) i - kja block and ajk - i ############################
#
#               temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
#               temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
#               temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
#               temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)
#
#               temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)
#
#               if isinstance(eris.ovvv, type(None)):
#                   chnk_size = radc_ao2mo.calculate_chunk_size(adc)
#               else :
#                   chnk_size = nocc
#               a = 0
#               temp_singles = np.zeros((nocc))
#               temp_doubles = np.zeros((nvir,nvir,nvir))
#               for p in range(0,nocc,chnk_size):
#                   if getattr(adc, 'with_df', None):
#                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
#                   else :
#                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
#                   k = eris_ovvv.shape[0]
#
#                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
#                   temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
#                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
#                   temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv,optimize=True)
#                   s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles, t2_1, optimize=True).reshape(-1)
#                   del eris_ovvv
#                   del temp_doubles
#                   a += k
#
#               s[s1:f1] += temp_singles
#               temp = np.zeros_like(r2)
#               temp =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
#               temp -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
#               temp -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
#               temp += lib.einsum('ljab,akj->blk',t2_1,r2,optimize=True)
#               temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)
#
#               temp_1 = np.zeros_like(r2)
#               temp_1 =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
#               temp_1 -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
#               temp_1 += lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
#               temp_1 -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
#
#               temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)
#
#               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
#               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)
#               del temp
#               del temp_1
#               del temp_2
#
#               temp = np.zeros_like(r2)
#               temp = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
#               temp += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
#               temp += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
#               temp -= lib.einsum('lkab,ajk->blj',t2_1,r2,optimize=True)
#               temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)
#
#               temp_1 = np.zeros_like(r2)
#               temp_1  = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
#               temp_1 += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
#               temp_1 -= lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
#               temp_1 += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
#
#               temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)
#
#               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
#               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)
#               
#               del temp
#               del temp_1
#               del temp_2
#
#               temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
#               temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
#               temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
#
#               temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
#               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1,optimize=True)
#               temp -= lib.einsum('kbl,ljab->ajk',temp_2,t2_1,optimize=True)
#               s[s2:f2] += temp.reshape(-1)
#
#               temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
#               temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
#               s[s2:f2] -= temp_1.reshape(-1)
#
#               del temp
#               del temp_1
#               del temp_2
#               del t2_1

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s
    return sigma_

def ip_adc_matvec(adc, kshift, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    t2 = adc.t2
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.t2.shape[3]
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_ij = np.zeros((nkpts,nocc,nkpts,nocc), dtype=t2.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    e_ij = e_occ[:,:,None,None] + e_occ[None,None,:,:]
    e_ija = - e_vir[:,:,None,None,None,None] + e_ij[None,None,:,:,:,:] 
    e_ija = e_ija.transpose(0,2,4,1,3,5).copy()

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim), dtype=np.complex)

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nkpts,nkpts,nvir,nocc,nocc)
        temp = np.zeros_like((r2),dtype=np.complex)
        temp_aij = np.zeros_like((r2),dtype=np.complex)
        
        eris_ovoo = eris.ovoo
        eris_vooo = eris.vooo

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ij,j->i',M_ij[kshift],r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]

                s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo[kj,ka,kk], r2[ka,kj], optimize = True)
                s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo[kk,ka,kj], r2[ka,kj], optimize = True)

################### ADC(2) ajk - i block ############################

                temp[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk].conj(), r1, optimize = True)

################# ADC(2) ajk - bil block ############################

       # for kj in range(nkpts):
       #     for kk in range(nkpts):
       #         ka = kconserv[kk, kshift, kj]
                #temp_aij[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]
                temp[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]

        s[s2:f2] += temp.reshape(-1)
        #s[s2:f2] += temp_aij.reshape(-1)

################ ADC(3) ajk - bil block ############################
#
#        if (method == "adc(2)-x" or method == "adc(3)"):
#        
#               eris_oooo = eris.oooo
#               eris_oovv = eris.oovv
#               eris_ovvo = eris.ovvo
#               
#               s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
#               s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)
#               
#               s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               
#               s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               
#               s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               
#               s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
#               s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
#               
#        if (method == "adc(3)"):
#
#               eris_ovoo = eris.ovoo
#               t2_1 = adc.t2[0]
#
################# ADC(3) i - kja block and ajk - i ############################
#
#               temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
#               temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
#               temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
#               temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)
#
#               temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)
#
#               if isinstance(eris.ovvv, type(None)):
#                   chnk_size = radc_ao2mo.calculate_chunk_size(adc)
#               else :
#                   chnk_size = nocc
#               a = 0
#               temp_singles = np.zeros((nocc))
#               temp_doubles = np.zeros((nvir,nvir,nvir))
#               for p in range(0,nocc,chnk_size):
#                   if getattr(adc, 'with_df', None):
#                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
#                   else :
#                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
#                   k = eris_ovvv.shape[0]
#
#                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
#                   temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
#                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
#                   temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv,optimize=True)
#                   s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles, t2_1, optimize=True).reshape(-1)
#                   del eris_ovvv
#                   del temp_doubles
#                   a += k
#
#               s[s1:f1] += temp_singles
#               temp = np.zeros_like(r2)
#               temp =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
#               temp -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
#               temp -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
#               temp += lib.einsum('ljab,akj->blk',t2_1,r2,optimize=True)
#               temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)
#
#               temp_1 = np.zeros_like(r2)
#               temp_1 =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
#               temp_1 -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
#               temp_1 += lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
#               temp_1 -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
#
#               temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)
#
#               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
#               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)
#               del temp
#               del temp_1
#               del temp_2
#
#               temp = np.zeros_like(r2)
#               temp = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
#               temp += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
#               temp += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
#               temp -= lib.einsum('lkab,ajk->blj',t2_1,r2,optimize=True)
#               temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)
#
#               temp_1 = np.zeros_like(r2)
#               temp_1  = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
#               temp_1 += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
#               temp_1 -= lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
#               temp_1 += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
#
#               temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)
#
#               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
#               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
#               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)
#               
#               del temp
#               del temp_1
#               del temp_2
#
#               temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
#               temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
#               temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
#
#               temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
#               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1,optimize=True)
#               temp -= lib.einsum('kbl,ljab->ajk',temp_2,t2_1,optimize=True)
#               s[s2:f2] += temp.reshape(-1)
#
#               temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
#               temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
#               s[s2:f2] -= temp_1.reshape(-1)
#
#               del temp
#               del temp_1
#               del temp_2
#               del t2_1

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s
    return sigma_


class RADCEA(RADC):
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
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
	nroots : int
	    Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcip = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ea : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''
    def __init__(self, adc):
        #self.mol = adc.mol
        #self.verbose = adc.verbose
        #self.stdout = adc.stdout
        #self.max_memory = adc.max_memory
        self.max_space = 50
        self.max_cycle = 200
        self.conv_tol  = 1e-12
        self.tol_residual  = 1e-6
        #self.t1 = adc.t1
        self.t2 = adc.t2
        #self.imds = adc.imds
        #self.e_corr = adc.e_corr
        #self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        #self._nvir = adc._nvir
        self._nmo = adc._nmo
        #self.mo_coeff = adc.mo_coeff
        #self.mo_energy = adc.mo_energy
        #self.nmo = adc._nmo
        #self.transform_integrals = adc.transform_integrals
        #self.with_df = adc.with_df
        #self.compute_properties = adc.compute_properties
        #self.E = None
        #self.U = None
        #self.P = None
        #self.X = None
        #self.evec_print_tol = adc.evec_print_tol
        #self.spec_factor_print_tol = adc.spec_factor_print_tol

        self.kpts = adc._scf.kpts
        self.verbose = adc.verbose
        self.max_memory = adc.max_memory
        self.method = adc.method

        self.khelper = adc.khelper
        self.cell = adc.cell
        self.mo_coeff = adc.mo_coeff
        self.mo_occ = adc.mo_occ
        self.frozen = adc.frozen

        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.mo_energy = adc.mo_energy

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ea
    get_diag = ea_adc_diag
    matvec = ea_adc_matvec
    #compute_trans_moments = ip_compute_trans_moments
    #get_trans_moments = get_trans_moments
    #renormalize_eigenvectors = renormalize_eigenvectors_ip
    #get_properties = get_properties
    #analyze_spec_factor = analyze_spec_factor
    #analyze_eigenvector = analyze_eigenvector_ip
    #analyze = analyze
    #compute_dyson_mo = compute_dyson_mo

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.ea_adc_diag()
        idx = None
        dtype = getattr(diag, 'dtype', np.complex)
        if ascending:
            idx = np.argsort(diag)
        else:
            idx = np.argsort(diag)[::-1]
        guess = np.zeros((diag.shape[0], nroots), dtype=dtype)
        min_shape = min(diag.shape[0], nroots)
        guess[:min_shape,:min_shape] = np.identity(min_shape)
        g = np.zeros((diag.shape[0], nroots), dtype=dtype)
        g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess

    def gen_matvec(self,kshift,imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(kshift,imds,eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag
        #return diag

class RADCIP(RADC):
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
            IP energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''
    def __init__(self, adc):
        #self.mol = adc.mol
        #self.verbose = adc.verbose
        #self.stdout = adc.stdout
        #self.max_memory = adc.max_memory
        self.max_space = 50
        self.max_cycle = 200
        self.conv_tol  = 1e-12
        self.tol_residual  = 1e-6
        #self.t1 = adc.t1
        self.t2 = adc.t2
        #self.imds = adc.imds
        #self.e_corr = adc.e_corr
        #self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        #self._nvir = adc._nvir
        self._nmo = adc._nmo
        #self.mo_coeff = adc.mo_coeff
        #self.mo_energy = adc.mo_energy
        #self.nmo = adc._nmo
        #self.transform_integrals = adc.transform_integrals
        #self.with_df = adc.with_df
        #self.compute_properties = adc.compute_properties
        #self.E = None
        #self.U = None
        #self.P = None
        #self.X = None
        #self.evec_print_tol = adc.evec_print_tol
        #self.spec_factor_print_tol = adc.spec_factor_print_tol

        self.kpts = adc._scf.kpts
        self.verbose = adc.verbose
        self.max_memory = adc.max_memory
        self.method = adc.method

        self.khelper = adc.khelper
        self.cell = adc.cell
        self.mo_coeff = adc.mo_coeff
        self.mo_occ = adc.mo_occ
        self.frozen = adc.frozen

        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.mo_energy = adc.mo_energy

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    #compute_trans_moments = ip_compute_trans_moments
    #get_trans_moments = get_trans_moments
    #renormalize_eigenvectors = renormalize_eigenvectors_ip
    #get_properties = get_properties
    #analyze_spec_factor = analyze_spec_factor
    #analyze_eigenvector = analyze_eigenvector_ip
    #analyze = analyze
    #compute_dyson_mo = compute_dyson_mo

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.ip_adc_diag()
        idx = None
        dtype = getattr(diag, 'dtype', np.complex)
        if ascending:
            idx = np.argsort(diag)
        else:
            idx = np.argsort(diag)[::-1]
        guess = np.zeros((diag.shape[0], nroots), dtype=dtype)
        min_shape = min(diag.shape[0], nroots)
        guess[:min_shape,:min_shape] = np.identity(min_shape)
        g = np.zeros((diag.shape[0], nroots), dtype=dtype)
        g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess

    def gen_matvec(self,kshift,imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(kshift,imds,eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag
        #return diag
