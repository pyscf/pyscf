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

def kernel(adc, nroots=1, guess=None, eris=None, kptlist=None, verbose=None):

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

    size = adc.vector_size()
    nroots = min(nroots,size)
    nkpts = adc.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    #if dtype is None:
    dtype = np.result_type(adc.t2[0])

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

    adc.U = np.array(evecs).T.copy()

    P,X = adc.get_properties(nroots)

    print (evals)
    print (P)
    exit()

#    adc.U = np.array(U).T.copy()
#
#     T = adc.get_trans_moments
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

    t1,t2,myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t2, eris)

    return e_corr, t1, t2


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
    t1_3 = None
    t2_1_vvvv = None
    t2_2 = None
    t1_2 = np.zeros((nkpts,nocc,nvir), dtype=t2_1.dtype)
    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)
    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

#    # Compute second-order singles t1 (tij)
#
#    if isinstance(eris.ovvv, type(None)):
#        chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
#    else:
#        chnk_size = nocc
#    a = 0
        t1_2 = np.zeros((nkpts,nocc,nvir), dtype=t2_1.dtype)

        eris_ovvv = eris.ovvv
        eris_ovoo = eris.ovoo
        eris_oovo = eris.oovo
        for ki in range (nkpts):
            ka = ki
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = kconserv[ki, kc, kk]
                    ka = kconserv[kc, kk, kd]

                    t1_2[ki] += 1.5*lib.einsum('kdac,ikcd->ia',eris_ovvv[kk,kd,ka].conj(),t2_1[ki,kk,kc].conj(),optimize=True)
                    t1_2[ki] -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv[kk,kd,ka].conj(),t2_1[kk,ki,kc].conj(),optimize=True)
                    t1_2[ki] -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv[kk,kc,ka].conj(),t2_1[ki,kk,kc].conj(),optimize=True)
                    t1_2[ki] += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv[kk,kc,ka].conj(),t2_1[kk,ki,kc].conj(),optimize=True)


                for kl in range(nkpts):
                    kc = kconserv[kk, ki, kl]
                    ka = kconserv[kl, kc, kk]
 
                    t1_2[ki] -= 1.5*lib.einsum('lcki,klac->ia',eris_ovoo[kl,kc,kk].conj(),t2_1[kk,kl,ka].conj(),optimize=True)
                    t1_2[ki] += 0.5*lib.einsum('lcki,lkac->ia',eris_ovoo[kl,kc,kk].conj(),t2_1[kl,kk,ka].conj(),optimize=True)
                    t1_2[ki] -= 0.5*lib.einsum('kcli,lkac->ia',eris_ovoo[kk,kc,kl].conj(),t2_1[kl,kk,ka].conj(),optimize=True)
                    t1_2[ki] += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo[kk,kc,kl].conj(),t2_1[kk,kl,ka].conj(),optimize=True)


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
        t2_1_vvvv = np.zeros_like((t2_1))


    # Compute second-order doubles t2 (tijab)

        eris_oooo = eris.oooo
        eris_ovov = eris.ovov
        eris_ovvo = eris.ovvo

        if isinstance(eris.vvvv, np.ndarray):
            eris_vvvv = eris.vvvv
            for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
                kd = kconserv[ka, kc, kb]
                for ki in range(nkpts):
                    kj = kconserv[ka, ki, kb]
                    t2_1_vvvv[ki, kj, ka] += lib.einsum('cdab,ijcd->ijab', eris_vvvv[kc,kd,ka].conj(), t2_1[ki,kj,kc])

        #elif isinstance(eris.vvvv, list):
        #    t2_1_vvvv = contract_ladder(myadc,t2_1[:],eris.vvvv)
        #else:
        #    t2_1_vvvv = contract_ladder(myadc,t2_1[:],eris.Lvv)

        #if not isinstance(eris.oooo, np.ndarray):
        #    t2_1_vvvv = radc_ao2mo.write_dataset(t2_1_vvvv)
         #t2_1_vvvv[ki, kj, ka] += lib.einsum('cadb,ijcd->ijab', eris_vvvv[kc,ka,kd].conj(), tau)
                    #t2_1_vvvv[ki, kj, ka] += lib.einsum('abcd,ijcd->ijab', eris_vvvv[ka,kb,kc], tau)

        t2_2 = np.zeros_like((t2_1))

        t2_2 = t2_1_vvvv.copy()
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            for kk in range(nkpts):

                    kc = kconserv[ki,ka,kk]
                    kb = kconserv[kj,kc,kk]
                    t2_2[ki,kj,ka] -= lib.einsum('jkcb,ikac->ijab',eris_oovv[kj,kk,kc].conj(),t2_1[ki,kk,ka],optimize=True)
                    kc = kconserv[kb,kk,ki]
                    t2_2[ki,kj,ka] -= lib.einsum('ikcb,kjac->ijab',eris_oovv[ki,kk,kc].conj(),t2_1[kk,kj,ka],optimize=True)
                    kc = kconserv[ka,kk,kj]
                    t2_2[ki,kj,ka] -= lib.einsum('jkca,ikcb->ijab',eris_oovv[kj,kk,kc].conj(),t2_1[ki,kk,kc],optimize=True)
                    kc = kconserv[ka,kk,ki]
                    t2_2[ki,kj,ka] -= lib.einsum('ikca,kjcb->ijab',eris_oovv[ki,kk,kc].conj(),t2_1[kk,kj,kc],optimize=True)


            for kl in range(nkpts):
                    kk = kconserv[kj,kl,ki]
                    t2_2[ki,kj,ka] += lib.einsum('ikjl,klab->ijab',eris_oooo[ki,kk,kj].conj(),t2_1[kk,kl,ka],optimize=True)


            for kk in range(nkpts):
                    kc = kconserv[ka,kk,ki]
                    kb = kconserv[kk,kj,kc]

                    t2_2[ki,kj,ka] += 2 * lib.einsum('jbck,kica->ijab',eris_ovvo[kj,kb,kc].conj(),t2_1[kk,ki,kc],optimize=True)
                    t2_2[ki,kj,ka] -= lib.einsum('jbck,ikca->ijab',eris_ovvo[kj,kb,kc].conj(),t2_1[ki,kk,kc],optimize=True)
                    t2_2[ki,kj,ka] += 2 * lib.einsum('iack,kjcb->ijab',eris_ovvo[ki,ka,kc].conj(),t2_1[kk,kj,kc],optimize=True)
                    t2_2[ki,kj,ka] -= lib.einsum('iack,jkcb->ijab',eris_ovvo[ki,ka,kc].conj(),t2_1[kj,kk,kc],optimize=True)

            #for kc in range(nkpts):
            #        #kc = kconserv[ka,kk,ki]
            #        kk = kconserv[ka,ki,kc]
            #        kb = kconserv[kc,kj,kk]

            #        t2_2[ki,kj,ka] += 2 * lib.einsum('jbkc,kica->ijab',eris_ovov[kj,kb,kk].conj(),t2_1[kk,ki,kc],optimize=True)
                    #t2_2[ki,kj,ka] -= lib.einsum('jbkc,ikca->ijab',eris_ovov[kj,kb,kk].conj(),t2_1[ki,kk,kc],optimize=True)
                    #t2_2[ki,kj,ka] += 2 * lib.einsum('iakc,kjcb->ijab',eris_ovov[ki,ka,kk].conj(),t2_1[kk,kj,kc],optimize=True)
                    #t2_2[ki,kj,ka] -= lib.einsum('iakc,jkcb->ijab',eris_ovov[ki,ka,kk].conj(),t2_1[kj,kk,kc],optimize=True)

                    #kc = kconserv[ki,ka,kk]
                    #kb = kconserv[kc,kk,kj]
                    #t2_2[ki,kj,ka] -= lib.einsum('kjbc,ikac->ijab',eris_oovv[kk,kj,kb],t2_1[ki,kk,ka],optimize=True)
                    #t2_2[ki,kj,ka] -= lib.einsum('kibc,kjac->ijab',eris_oovv[kk,ki,kb].conj(),t2_1[kk,kj,ka],optimize=True)
                    #kc = kconserv[kk,kj,ka]
                    #t2_2[ki,kj,ka] -= lib.einsum('kjac,ikcb->ijab',eris_oovv[kk,kj,ka].conj(),t2_1[ki,kk,kc],optimize=True)
                    #kc = kconserv[kk,ki,ka]
                    #t2_2[ki,kj,ka] -= lib.einsum('kiac,kjcb->ijab',eris_oovv[kk,ki,ka].conj(),t2_1[kk,kj,kc],optimize=True)

#
#            for kk in range(nkpts):
#                    kc = kconserv[ka,kk,ki]
#                    kb = kconserv[kj,kk,kc]
#
#                    t2_2[ki,kj,ka] += lib.einsum('kcbj,kica->ijab',eris_ovvo[kk,kc,kb].conj(),t2_1[kk,ki,kc],optimize=True)
#                    t2_2[ki,kj,ka] -= lib.einsum('kcbj,ikca->ijab',eris_ovvo[kk,kc,kb].conj(),t2_1[ki,kk,kc],optimize=True)
#                    t2_2[ki,kj,ka] += lib.einsum('kcbj,ikac->ijab',eris_ovvo[kk,kc,kb].conj(),t2_1[ki,kk,ka],optimize=True)
#                    t2_2[ki,kj,ka] += lib.einsum('kcai,kjcb->ijab',eris_ovvo[kk,kc,ka].conj(),t2_1[kk,kj,kc],optimize=True)
#                    t2_2[ki,kj,ka] -= lib.einsum('kcai,jkcb->ijab',eris_ovvo[kk,kc,ka].conj(),t2_1[kj,kk,kc],optimize=True)
#                    t2_2[ki,kj,ka] += lib.einsum('kcai,kjcb->ijab',eris_ovvo[kk,kc,ka].conj(),t2_1[kk,kj,kc],optimize=True)

            #for kk in range(nkpts):
            #        kc = kconserv[ka,kk,ki]
            #        kb = kconserv[kj,kk,kc]

            #        t2_2[ki,kj,ka] += 2 * lib.einsum('kcbj,kica->ijab',eris_ovvo[kk,kc,kb],t2_1[kk,ki,kc],optimize=True)
            #        t2_2[ki,kj,ka] -= lib.einsum('kcbj,ikca->ijab',eris_ovvo[kk,kc,kb],t2_1[ki,kk,kc],optimize=True)
            #        t2_2[ki,kj,ka] += 2 * lib.einsum('kcai,kjcb->ijab',eris_ovvo[kk,kc,ka],t2_1[kk,kj,kc],optimize=True)
            #        t2_2[ki,kj,ka] -= lib.einsum('kcai,jkcb->ijab',eris_ovvo[kk,kc,ka],t2_1[kj,kk,kc],optimize=True)






#        for ki, kj, kb in kpts_helper.loop_kkk(nkpts):
#            for kk in range(nkpts):
#                    ka = kconserv[kj,kb,ki]
#                    #kc = kconserv[ka,kk,ki]
#                    kc = kconserv[ki,ka,kk]
#                    #kc = kconserv[kk,ki,ka]
#                    #kk = kconserv[kc,kj,kb]
#                    #kc = kconserv[kj,kb,kk]
#
#            #        t2_2[ki,kj,ka] += lib.einsum('kcjb,kica->ijab',eris_ovov[kk,kc,kj],t2_1[kk,ki,kc],optimize=True)
#            ##        t2_2[ki,kj,ka] -= lib.einsum('kcjb,ikca->ijab',eris_ovov[kk,kc,kj],t2_1[ki,kk,kc],optimize=True)
#            ##        t2_2[ki,kj,ka] += lib.einsum('kcjb,ikac->ijab',eris_ovov[kk,kc,kj],t2_1[ki,kk,ka],optimize=True)
#                    t2_2[ki,kj,ka] += lib.einsum('kcia,kjcb->ijab',eris_ovov[kk,kc,ki],t2_1[kk,kj,kc],optimize=True)
#            ##        t2_2[ki,kj,ka] -= lib.einsum('kcia,jkcb->ijab',eris_ovov[kk,kc,ki],t2_1[kj,kk,kc],optimize=True)
#            ##        t2_2[ki,kj,ka] += lib.einsum('kcia,kjcb->ijab',eris_ovov[kk,kc,ki],t2_1[kk,kj,kc],optimize=True)
#
#            #        #t2_2[ki,kj,ka] =  t2_2[ki,kj,ka] / eijab
#
#            #t2_2 += lib.einsum('xyzkcbj,xyzkica->xyzijab',eris_ovvo,t2_1,optimize=True)
#
#            #temp = lib.einsum('xyzkcbj,xwykica->xwzijab',eris_ovvo,t2_1,optimize=True)
#            #temp1 = lib.einsum('xywkcjb,xwykica->xwyijab',eris_ovov,t2_1,optimize=True)
#
#            #temp = lib.einsum('xyziabj,xyzijab',eris_ovvo,t2_1,optimize=True)
#            #temp1 = lib.einsum('xyz,xyzjiab',eris_ovov,t2_1,optimize=True)


        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki, ka, kj]
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])

            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                           [0,nvir,kb,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            t2_2[ki,kj,ka] /= eijab

#        D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
#        D2 = D2.reshape((nocc,nocc,nvir,nvir))
#
#        t2_2 = t2_2/D2
#        if not isinstance(eris.oooo, np.ndarray):
#            t2_2 = radc_ao2mo.write_dataset(t2_2)
#        del D2

    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)
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
    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2, t2_1_vvvv
#    return t2_1
#    return t1,t2
#
#

def compute_energy(myadc, t2, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)
    nkpts = myadc.nkpts

    emp2 = 0.0
    eris_ovov = eris.ovov
    t2_amp = t2[0].copy()     
 
    if (myadc.method == "adc(3)"):
        t2_amp += t2[1]

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):

        emp2 += 2 * lib.einsum('ijab,iajb', t2_amp[ki,kj,ka], eris_ovov[ki,ka,kj],optimize=True)
        emp2 -= 1 * lib.einsum('ijab,jaib', t2_amp[ki,kj,ka], eris_ovov[kj,ka,ki],optimize=True)
        #emp2 -= 0.5 * lib.einsum('jiab,iajb', t2_amp[kj,ki,ka], eris_ovov[ki,ka,kj],optimize=True)
        #emp2 += 0.5 * lib.einsum('jiab,jaib', t2_amp[kj,ki,ka], eris_ovov[kj,ka,ki],optimize=True)
  
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
        self.method_type = "ip"

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.cell = self._scf.cell
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.frozen = frozen

        self._nocc = None
        self._nmo = None
        self._nvir = None

        self.t1 = None
        self.t2 = None
        self.e_corr = None
        self.imds = lambda:None

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
        self.e_corr,self.t1,self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)

        return self.e_corr, self.t1,self.t2

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
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
            
#       self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        print ("MP2:",self.e_corr)
#       self._finalize()
#
        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
             con, e, v = self.ea_adc(nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)
#            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)
#
        elif(self.method_type == "ip"):
             #e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(nroots=nroots, guess=guess, eris=eris)
             con, e, v = self.ip_adc(nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        else:
            raise NotImplementedError(self.method_type)
#        self._adc_es = adc_es
#        return e_exc, v_exc, spec_fac, x
        return con, e, v

    def ip_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        adc_es = RADCIP(self)
        con, e, v = adc_es.kernel(nroots, guess, eris, kptlist)
        #e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        #return e_exc, v_exc, spec_fac, x, adc_es
        return con, e, v

    def ea_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        adc_es = RADCEA(self)
        con, e, v = adc_es.kernel(nroots, guess, eris, kptlist)
        #e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        #return e_exc, v_exc, spec_fac, x, adc_es
        return con, e, v


def ea_vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc

    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir
    size = n_singles + n_doubles

    return size

def ip_vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc
    size = n_singles + n_doubles

    return size

def get_imds_ea(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    t2_1 = adc.t2[0]
    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nmo =  adc.get_nmo
    nocc = adc.t2[0].shape[3]
    nvir = adc.nmo - adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_vir = np.identity(nvir)
##    if eris is None:
##        eris = adc.transform_integrals()

    eris_ovov = eris.ovov

    # a-b block
    # Zeroth-order terms
    M_ab = np.empty((nkpts,nvir,nvir),dtype=t2_1.dtype)

    for ka in range(nkpts):
        kb = ka
        M_ab[ka] = lib.einsum('ab,a->ab', idn_vir, e_vir[ka])
        for kl in range(nkpts):
            for km in range(nkpts):

                kd = kconserv[kl,ka,km]
                # Second-order terms

                M_ab[ka] +=  lib.einsum('l,lmad,lmbd->ab',e_occ[kl] ,t2_1[kl,km,ka], t2_1[kl,km,kb].conj(),optimize=True)
                M_ab[ka] -=  lib.einsum('l,lmad,mlbd->ab',e_occ[kl] ,t2_1[kl,km,ka], t2_1[km,kl,kb].conj(),optimize=True)
                M_ab[ka] -=  lib.einsum('l,mlad,lmbd->ab',e_occ[kl] ,t2_1[km,kl,ka], t2_1[kl,km,kb].conj(),optimize=True)
                M_ab[ka] +=  lib.einsum('l,mlad,mlbd->ab',e_occ[kl] ,t2_1[km,kl,ka], t2_1[km,kl,kb].conj(),optimize=True)
                M_ab[ka] +=  lib.einsum('l,lmad,lmbd->ab',e_occ[kl] ,t2_1[kl,km,ka], t2_1[kl,km,kb].conj(),optimize=True)
                M_ab[ka] +=  lib.einsum('l,mlad,mlbd->ab',e_occ[kl] ,t2_1[km,kl,ka], t2_1[km,kl,kb].conj(),optimize=True)

                M_ab[ka] -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir[kd], t2_1[kl,km,ka], t2_1[kl,km,kb].conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('d,lmad,mlbd->ab',e_vir[kd], t2_1[kl,km,ka], t2_1[km,kl,kb].conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('d,mlad,lmbd->ab',e_vir[kd], t2_1[km,kl,ka], t2_1[kl,km,kb].conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir[kd], t2_1[km,kl,ka], t2_1[km,kl,kb].conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir[kd], t2_1[kl,km,ka], t2_1[kl,km,kb].conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir[kd], t2_1[km,kl,ka], t2_1[km,kl,kb].conj(),optimize=True)

                M_ab_t = lib.einsum('lmad,lmbd->ab', t2_1[kl,km,ka],t2_1[kl,km,kb].conj(), optimize=True)
                M_ab[ka] -= 1 *  lib.einsum('a,ab->ab',e_vir[ka],M_ab_t,optimize=True)
                M_ab[ka] -= 1 *  lib.einsum('b,ab->ab',e_vir[kb],M_ab_t,optimize=True)
                del M_ab_t

                M_ab_t = lib.einsum('lmad,mlbd->ab', t2_1[kl,km,ka],t2_1[km,kl,kb].conj(), optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('a,ab->ab',e_vir[ka],M_ab_t,optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('b,ab->ab',e_vir[kb],M_ab_t,optimize=True)
                del M_ab_t

                M_ab[ka] -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_1[kl,km,ka], eris_ovov[kl,kb,km],optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('mlad,lbmd->ab',t2_1[km,kl,ka], eris_ovov[kl,kb,km],optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_1[kl,km,ka], eris_ovov[kl,kd,km],optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('mlad,ldmb->ab',t2_1[km,kl,ka], eris_ovov[kl,kd,km],optimize=True)
                M_ab[ka] -=        lib.einsum('lmad,lbmd->ab',t2_1[kl,km,ka], eris_ovov[kl,kb,km],optimize=True)

                M_ab[ka] -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_1[kl,km,kb].conj(), eris_ovov[kl,ka,km].conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('mlbd,lamd->ab',t2_1[km,kl,kb].conj(), eris_ovov[kl,ka,km].conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_1[kl,km,kb].conj(), eris_ovov[kl,kd,km].conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('mlbd,ldma->ab',t2_1[km,kl,kb].conj(), eris_ovov[kl,kd,km].conj(),optimize=True)
                M_ab[ka] -=        lib.einsum('lmbd,lamd->ab',t2_1[kl,km,kb].conj(), eris_ovov[kl,ka,km].conj(),optimize=True)

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
 
    return M_ab

def get_imds_ip(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

#    t1 = adc.t1
    t2_1 = adc.t2[0]
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
    nocc = adc.t2[0].shape[3]
    nvir = adc.nmo - adc.nocc
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
    if (method == "adc(3)"):
        t1_2 = adc.t1[0]
        t2_2 = adc.t2[1]

        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv
        eris_voov = eris.voov

        for ki in range(nkpts):
            kj = ki
            for kl in range(nkpts):
                kd = kconserv[ki,kl,kj]
                M_ij[ki] += lib.einsum('ld,ldji->ij',t1_2[kl], eris_ovoo[kl,kd,kj].conj(),optimize=True)
                M_ij[ki] -= lib.einsum('ld,jdli->ij',t1_2[kl], eris_ovoo[kj,kd,kl].conj(),optimize=True)
                M_ij[ki] += lib.einsum('ld,ldji->ij',t1_2[kl], eris_ovoo[kl,kd,kj].conj(),optimize=True)

                M_ij[ki] += lib.einsum('ld,ldij->ij',t1_2[kl], eris_ovoo[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] -= lib.einsum('ld,idlj->ij',t1_2[kl], eris_ovoo[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] += lib.einsum('ld,ldij->ij',t1_2[kl], eris_ovoo[kl,kd,ki].conj(),optimize=True)

                for kd in range(nkpts):
                    ke = kconserv[kj,kd,kl]

                    M_ij[ki] += 0.5* lib.einsum('ilde,jdle->ij',t2_2[ki,kl,kd], eris_ovov[kj,kd,kl],optimize=True)
                    M_ij[ki] -= 0.5* lib.einsum('lide,jdle->ij',t2_2[kl,ki,kd], eris_ovov[kj,kd,kl],optimize=True)
                    M_ij[ki] -= 0.5* lib.einsum('ilde,jeld->ij',t2_2[ki,kl,kd], eris_ovov[kj,ke,kl],optimize=True)
                    M_ij[ki] += 0.5* lib.einsum('lide,jeld->ij',t2_2[kl,ki,kd], eris_ovov[kj,ke,kl],optimize=True)
                    M_ij[ki] += lib.einsum('ilde,jdle->ij',t2_2[ki,kl,kd], eris_ovov[kj,kd,kl],optimize=True)

                    M_ij[ki] += 0.5* lib.einsum('jlde,leid->ij',t2_2[kj,kl,kd].conj(), eris_ovov[kl,ke,ki].conj(),optimize=True)
                    M_ij[ki] -= 0.5* lib.einsum('ljde,leid->ij',t2_2[kl,kj,kd].conj(), eris_ovov[kl,ke,ki].conj(),optimize=True)
                    M_ij[ki] -= 0.5* lib.einsum('jlde,ield->ij',t2_2[kj,kl,kd].conj(), eris_ovov[ki,ke,kl].conj(),optimize=True)
                    M_ij[ki] += 0.5* lib.einsum('ljde,ield->ij',t2_2[kl,kj,kd].conj(), eris_ovov[ki,ke,kl].conj(),optimize=True)
                    M_ij[ki] += lib.einsum('jlde,leid->ij',t2_2[kj,kl,kd].conj(), eris_ovov[kl,ke,ki].conj(),optimize=True)

                    M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_2[kj,kl,kd].conj(),optimize=True)
                    M_ij[ki] -=  lib.einsum('d,ilde,ljde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_2[kl,kj,kd].conj(),optimize=True)
                    M_ij[ki] -=  lib.einsum('d,lide,jlde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_2[kj,kl,kd].conj(),optimize=True)
                    M_ij[ki] +=  lib.einsum('d,lide,ljde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_2[kl,kj,kd].conj(),optimize=True)
                    M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_2[kj,kl,kd].conj(),optimize=True)
                    M_ij[ki] +=  lib.einsum('d,iled,jled->ij',e_vir[kd],t2_1[ki,kl,ke], t2_2[kj,kl,ke].conj(),optimize=True)

                    M_ij[ki] +=  lib.einsum('d,jlde,ilde->ij',e_vir[kd],t2_1[kj,kl,kd].conj(), t2_2[ki,kl,kd],optimize=True)
                    M_ij[ki] -=  lib.einsum('d,jlde,lide->ij',e_vir[kd],t2_1[kj,kl,kd].conj(), t2_2[kl,ki,kd],optimize=True)
                    M_ij[ki] -=  lib.einsum('d,ljde,ilde->ij',e_vir[kd],t2_1[kl,kj,kd].conj(), t2_2[ki,kl,kd],optimize=True)
                    M_ij[ki] +=  lib.einsum('d,ljde,lide->ij',e_vir[kd],t2_1[kl,kj,kd].conj(), t2_2[kl,ki,kd],optimize=True)
                    M_ij[ki] +=  lib.einsum('d,jlde,ilde->ij',e_vir[kd],t2_1[kj,kl,kd].conj(), t2_2[ki,kl,kd],optimize=True)
                    M_ij[ki] +=  lib.einsum('d,jled,iled->ij',e_vir[kd],t2_1[kj,kl,ke].conj(), t2_2[ki,kl,ke],optimize=True)

                    M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2[kj,kl,kd].conj(),optimize=True)
                    M_ij[ki] += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2[kl,kj,kd].conj(),optimize=True)
                    M_ij[ki] += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_2[kj,kl,kd].conj(),optimize=True)
                    M_ij[ki] -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_2[kl,kj,kd].conj(),optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2[kj,kl,kd].conj(),optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2[kj,kl,kd].conj(),optimize=True)

                    M_ij[ki] -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2[ki,kl,kd],optimize=True)
                    M_ij[ki] += 0.5 *  lib.einsum('l,jlde,lide->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2[kl,ki,kd],optimize=True)
                    M_ij[ki] += 0.5 *  lib.einsum('l,ljde,ilde->ij',e_occ[kl],t2_1[kl,kj,kd].conj(), t2_2[ki,kl,kd],optimize=True)
                    M_ij[ki] -= 0.5 *  lib.einsum('l,ljde,lide->ij',e_occ[kl],t2_1[kl,kj,kd].conj(), t2_2[kl,ki,kd],optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2[ki,kl,kd],optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2[ki,kl,kd],optimize=True)

                    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1[ki,kl,kd],t2_2[kj,kl,kd].conj(), optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t, optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('i,ji->ij',e_occ[ki], M_ij_t, optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('j,ij->ij',e_occ[kj], M_ij_t, optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('j,ji->ij',e_occ[kj], M_ij_t, optimize=True)
                    del M_ij_t

                    M_ij_t_1 = lib.einsum('ilde,ljde->ij', t2_1[ki,kl,kd],t2_2[kl,kj,kd].conj(), optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t_1, optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('i,ji->ij',e_occ[ki], M_ij_t_1, optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('j,ij->ij',e_occ[kj], M_ij_t_1, optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('j,ji->ij',e_occ[kj], M_ij_t_1, optimize=True)
                    del M_ij_t_1

                    temp_t2_vvvv = adc.imds.t2_1_vvvv[:]
                    M_ij[ki] += 0.25*lib.einsum('ilde,jlde->ij',t2_1[ki,kl,kd], temp_t2_vvvv[kj,kl,kd].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('ilde,ljde->ij',t2_1[ki,kl,kd], temp_t2_vvvv[kl,kj,kd].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('lide,jlde->ij',t2_1[kl,ki,kd], temp_t2_vvvv[kj,kl,kd].conj(), optimize = True)
                    M_ij[ki] += 0.25*lib.einsum('lide,ljde->ij',t2_1[kl,ki,kd], temp_t2_vvvv[kl,kj,kd].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('ilde,jled->ij',t2_1[ki,kl,kd], temp_t2_vvvv[kj,kl,ke].conj(), optimize = True)
                    M_ij[ki] += 0.25*lib.einsum('ilde,ljed->ij',t2_1[ki,kl,kd], temp_t2_vvvv[kl,kj,ke].conj(), optimize = True)
                    M_ij[ki] += 0.25*lib.einsum('lide,jled->ij',t2_1[kl,ki,kd], temp_t2_vvvv[kj,kl,ke].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('lide,ljed->ij',t2_1[kl,ki,kd], temp_t2_vvvv[kl,kj,ke].conj(), optimize = True)
                    M_ij[ki] +=lib.einsum('ilde,jlde->ij',t2_1[ki,kl,kd], temp_t2_vvvv[kj,kl,kd].conj(), optimize = True)
                    del temp_t2_vvvv

                    log.timer_debug1("Starting the small integrals  calculation")

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
         
                kf = kconserv[km,kj,ke]
                kl = kconserv[kf,kj,kd]
                temp_t2_v_1 = np.zeros_like((eris_ovov))
                temp_t2_v_1[km,ke,kj] += lib.einsum('lmde,jldf->mejf',t2_1[kl,km,kd], t2_1[kj,kl,kd].conj(),optimize=True)
                M_ij[ki] -=  2 * lib.einsum('mejf,mefi->ij',temp_t2_v_1[km,ke,kj].conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=  lib.einsum('mejf,mife->ij',temp_t2_v_1[km,ke,kj].conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('meif,mefj->ij',temp_t2_v_1[km,ke,ki].conj(), eris_ovvo[km,ke,kf].conj() ,optimize = True)
                M_ij[ki] +=  lib.einsum('meif,mjfe->ij',temp_t2_v_1[km,ke,ki].conj(), eris_oovv[km,kj,kf].conj() ,optimize = True)
                del temp_t2_v_1        

                temp_t2_v_new = np.zeros_like((eris_ovov))
                temp_t2_v_new[km,ke,kj] += lib.einsum('mlde,ljdf->mejf',t2_1[km,kl,kd], t2_1[kl,kj,kd].conj(),optimize=True)
                M_ij[ki] -=  2 * lib.einsum('mejf,mefi->ij',temp_t2_v_new[km,ke,kj].conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=  lib.einsum('mejf,mife->ij',temp_t2_v_new[km,ke,kj].conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('meif,mefj->ij',temp_t2_v_new[km,ke,ki].conj(), eris_ovvo[km,ke,kf].conj() ,optimize = True)
                M_ij[ki] +=  lib.einsum('meif,mjfe->ij',temp_t2_v_new[km,ke,ki].conj(), eris_oovv[km,kj,kf].conj() ,optimize = True)
                del temp_t2_v_new       


                kf = kconserv[km,kj,ke]
                temp_t2_v_2 = np.zeros_like((eris_ovov))
                temp_t2_v_2[km,ke,kj] += lib.einsum('lmde,ljdf->mejf',t2_1[kl,km,kd], t2_1[kl,kj,kd].conj(),optimize=True)
                M_ij[ki] +=  4 * lib.einsum('mejf,mefi->ij',temp_t2_v_2[km,ke,kj].conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=  4 * lib.einsum('meif,mefj->ij',temp_t2_v_2[km,ke,ki].conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('meif,mjfe->ij',temp_t2_v_2[km,ke,ki].conj(), eris_oovv[km,kj,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('mejf,mife->ij',temp_t2_v_2[km,ke,kj].conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                del temp_t2_v_2        

                temp_t2_v_3 = np.zeros_like((eris_ovov))
                temp_t2_v_3[km,ke,kj] += lib.einsum('mlde,jldf->mejf',t2_1[km,kl,kd], t2_1[kj,kl,kd].conj(),optimize=True)
                M_ij[ki] += lib.einsum('mejf,mefi->ij',temp_t2_v_3[km,ke,kj].conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] += lib.einsum('meif,mefj->ij',temp_t2_v_3[km,ke,ki].conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] -= 2 * lib.einsum('meif,mjfe->ij',temp_t2_v_3[km,ke,ki].conj(), eris_oovv[km,kj,kf].conj(),optimize = True)
                M_ij[ki] -= 2 * lib.einsum('mejf,mife->ij',temp_t2_v_3[km,ke,kj].conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                del temp_t2_v_3        


                kl = kconserv[kd,ke,ki]
                kf = kconserv[ki,kd,km]
                temp_t2_v_4 = np.zeros_like((eris_ovov))
                temp_t2_v_4[ki,kd,km] += lib.einsum('ilde,lmfe->idmf',t2_1[ki,kl,kd].conj(), eris_oovv[kl,km,kf].conj(),optimize=True)
                M_ij[ki] -= 2 * lib.einsum('idmf,jmdf->ij',temp_t2_v_4[ki,kd,km].conj(), t2_1[kj,km,kd].conj(), optimize = True)
                M_ij[ki] += lib.einsum('idmf,mjdf->ij',temp_t2_v_4[ki,kd,km].conj(), t2_1[km,kj,kd].conj(), optimize = True)
                del temp_t2_v_4


                temp_t2_v_5 = np.zeros_like((eris_ovov))
                temp_t2_v_5[ki,kd,km] += lib.einsum('lide,lmfe->idmf',t2_1[kl,ki,kd].conj(), eris_oovv[kl,km,kf].conj(),optimize=True)
                M_ij[ki] += lib.einsum('idmf,jmdf->ij',temp_t2_v_5[ki,kd,km].conj(), t2_1[kj,km,kd].conj(), optimize = True)
                M_ij[ki] -= 2 * lib.einsum('idmf,mjdf->ij',temp_t2_v_5[ki,kd,km].conj(), t2_1[km,kj,kd].conj(), optimize = True)
                del temp_t2_v_5

                temp_t2_v_6 = np.zeros_like((eris_ovvo))
                temp_t2_v_6[ki,kd,kf] += lib.einsum('ilde,lefm->idfm',t2_1[ki,kl,kd].conj(), eris_ovvo[kl,ke,kf].conj(),optimize=True)
                M_ij[ki] += 4 * lib.einsum('idfm,jmdf->ij',temp_t2_v_6[ki,kd,kf].conj(), t2_1[kj,km,kd].conj(),optimize = True)
                M_ij[ki] -= 2 * lib.einsum('idfm,mjdf->ij',temp_t2_v_6[ki,kd,kf].conj(), t2_1[km,kj,kd].conj(),optimize = True)
                del temp_t2_v_6

                temp_t2_v_7 = np.zeros_like((eris_ovvo))
                temp_t2_v_7[ki,kd,kf] = lib.einsum('lide,lefm->idfm',t2_1[kl,ki,kd].conj(), eris_ovvo[kl,ke,kf].conj(),optimize=True)
                M_ij[ki] -= 2 * lib.einsum('idfm,jmdf->ij',temp_t2_v_7[ki,kd,kf].conj(), t2_1[kj,km,kd].conj(),optimize = True)
                M_ij[ki] += lib.einsum('idfm,mjdf->ij',temp_t2_v_7[ki,kd,kf].conj(), t2_1[km,kj,kd].conj(),optimize = True)
                del temp_t2_v_7

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
            
                kf = ke
                kl = kconserv[km,kd,kf]
                temp_t2_v_8 = np.zeros((nkpts,nvir,nvir),dtype=t2_1.dtype)
                temp_t2_v_8[kf] += lib.einsum('lmdf,lmde->fe',t2_1[kl,km,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] += 3 *lib.einsum('fe,jief->ij',temp_t2_v_8[kf], eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] -= 1.5 *lib.einsum('fe,jfei->ij',temp_t2_v_8[kf], eris_ovvo[kj,kf,ke], optimize = True)
                M_ij[ki] +=   lib.einsum('ef,jief->ij',temp_t2_v_8[ke].T, eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] -= 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_8[ke].T, eris_ovvo[kj,kf,ke], optimize = True)
                del temp_t2_v_8

                temp_t2_v_9 = np.zeros((nkpts,nvir,nvir),dtype=t2_1.dtype)
                temp_t2_v_9[kf] += lib.einsum('lmdf,mlde->fe',t2_1[kl,km,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('fe,jief->ij',temp_t2_v_9[kf], eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] -= 1.0 * lib.einsum('ef,jief->ij',temp_t2_v_9[ke].T, eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] += 0.5 * lib.einsum('fe,jfei->ij',temp_t2_v_9[kf], eris_ovvo[kj,kf,ke], optimize = True)
                M_ij[ki] += 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_9[ke].T, eris_ovvo[kj,kf,ke], optimize = True)
                del temp_t2_v_9

                kn = km
                kl = kconserv[kn,kd,ke]
                temp_t2_v_10 = np.zeros((nkpts,nocc,nocc),dtype=t2_1.dtype)
                temp_t2_v_10[kn] += lib.einsum('lnde,lmde->nm',t2_1[kl,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] -= 3.0 * lib.einsum('nm,jinm->ij',temp_t2_v_10[kn], eris_oooo[kj,ki,kn], optimize = True)
                M_ij[ki] -= 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_10[km].T, eris_oooo[kj,ki,kn], optimize = True)
                M_ij[ki] += 1.5 * lib.einsum('nm,jmni->ij',temp_t2_v_10[kn], eris_oooo[kj,km,kn], optimize = True)
                M_ij[ki] += 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_10[km].T, eris_oooo[kj,km,kn], optimize = True)
                del temp_t2_v_10

                temp_t2_v_11 = np.zeros((nkpts,nocc,nocc),dtype=t2_1.dtype)
                temp_t2_v_11[kn] += lib.einsum('lnde,mlde->nm',t2_1[kl,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] += 1.0 * lib.einsum('nm,jinm->ij',temp_t2_v_11[kn], eris_oooo[kj,ki,kn], optimize = True)
                M_ij[ki] -= 0.5 * lib.einsum('nm,jmni->ij',temp_t2_v_11[kn], eris_oooo[kj,km,kn], optimize = True)
                M_ij[ki] -= 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_11[km].T, eris_oooo[kj,km,kn], optimize = True)
                M_ij[ki] += 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_11[km].T, eris_oooo[kj,ki,kn], optimize = True)
                del temp_t2_v_11

            for km, kn, kd in kpts_helper.loop_kkk(nkpts):
                kl = kconserv[km,ki,kn]
                temp_t2_v_12 = np.zeros_like((eris_oooo))
                temp_t2_v_12[ki,kn,kl] += lib.einsum('inde,lmde->inlm',t2_1[ki,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                kl = kconserv[km,ki,kn]
                M_ij[ki] += 1.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12[ki,kn,kl], eris_oooo[kj,kl,kn], optimize = True)
                M_ij[ki] -= 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12[ki,kn,kl], eris_oooo[kj,km,kn], optimize = True)
 
                M_ij[kj] += 0.25 * lib.einsum('inlm,jlnm->ji',temp_t2_v_12[ki,kn,kl].conj(), eris_oooo[kj,kl,kn].conj(), optimize = True)
                M_ij[kj] -= 0.25 * lib.einsum('inlm,lnmj->ji',temp_t2_v_12[ki,kn,kl], eris_oooo[kl,kn,km].conj(), optimize = True)
                M_ij[kj] += 1.00 * lib.einsum('inlm,ljmn->ji',temp_t2_v_12[ki,kn,kl], eris_oooo[kl,kj,km].conj(), optimize = True)
                del temp_t2_v_12

                temp_t2_v_12_1 = np.zeros_like((eris_oooo))
                temp_t2_v_12_1[ki,kn,kl] += lib.einsum('nide,mlde->inlm',t2_1[kn,ki,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] += 0.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12_1[ki,kn,kl].conj(), eris_oooo[kj,kl,kn].conj(), optimize = True)
                M_ij[ki] -= 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12_1[ki,kn,kl].conj(), eris_oooo[kj,km,kn].conj(), optimize = True)
                M_ij[kj] -= 0.25 * lib.einsum('inlm,lnmj->ji',temp_t2_v_12_1[ki,kn,kl].conj(), eris_oooo[kl,kn,km], optimize = True)
                M_ij[kj] += 0.25 * lib.einsum('inlm,ljmn->ji',temp_t2_v_12_1[ki,kn,kl].conj(), eris_oooo[kl,kj,km], optimize = True)

                temp_t2_v_13 = np.zeros_like((eris_oooo))
                temp_t2_v_13[ki,kn,km] += lib.einsum('inde,mlde->inml',t2_1[ki,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] -= 0.25 * lib.einsum('inml,jlnm->ij',temp_t2_v_13[ki,kn,km], eris_oooo[kj,kl,kn], optimize = True)
                M_ij[ki] += 0.25 * lib.einsum('inml,jmnl->ij',temp_t2_v_13[ki,kn,km], eris_oooo[kj,km,kn], optimize = True)

                M_ij[kj] -= 0.25 * lib.einsum('inml,jlnm->ji',temp_t2_v_13[ki,kn,km], eris_oooo[kj,kl,kn], optimize = True)
                M_ij[kj] += 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13[ki,kn,km], eris_oooo[kl,kn,km].conj(), optimize = True)

                M_ij[kj] -= 0.25 * lib.einsum('inml,ljmn->ji',temp_t2_v_13[ki,kn,km], eris_oooo[kl,kj,km].conj(), optimize = True)
                M_ij[kj] += 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13[ki,kn,km], eris_oooo[kl,kn,km].conj(), optimize = True)
                del temp_t2_v_13

                temp_t2_v_13_1 = np.zeros_like((eris_oooo))
                temp_t2_v_13_1[ki,kn,kl] += lib.einsum('nide,lmde->inlm',t2_1[kn,ki,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] -= 0.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_13_1[ki,kn,kl].conj(), eris_oooo[kj,kl,kn].conj(), optimize = True)
                M_ij[ki] += 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_13_1[ki,kn,kl].conj(), eris_oooo[kj,km,kn].conj(), optimize = True)
                del temp_t2_v_13_1

    return M_ij


def ea_adc_diag(adc,kshift,M_ab=None,eris=None):
   
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ab is None:
        M_ab = adc.get_imds()

    nkpts = adc.nkpts
    t2 = adc.t2[0]
    kconserv = adc.khelper.kconserv

    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.t2[0].shape[3]
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_ab = np.zeros((nkpts,nvir,nkpts,nvir), dtype=t2.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    e_ab = e_vir[:,:,None,None] + e_vir[None,None,:,:]
    e_iab = - e_occ[:,:,None,None,None,None] + e_ab[None,None,:,:,:,:] 
    e_iab = e_iab.transpose(0,2,4,1,3,5).copy()

    e_iab_ones = np.ones_like(e_iab)

    diag = np.zeros((dim), dtype=np.complex)

    # Compute precond in h1-h1 block
    M_ab_diag = np.diagonal(M_ab[kshift])
    diag[s1:f1] = M_ab_diag.copy()

    # Compute precond in 2p1h-2p1h block

    #diag[s2:f2] += e_iab[:,kshift].reshape(-1) 
    #diag[s2:f2] += 1e13 * e_iab[:,kshift].reshape(-1) 
    diag[s2:f2] += e_iab[:,kshift].reshape(-1) 

    diag = diag
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
    t2 = adc.t2[0]
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
    nocc = adc.t2[0].shape[3]
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

def ea_adc_matvec(adc, kshift, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    t2 = adc.t2[0]
    nvir = adc.nmo - adc.nocc
    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.t2[0].shape[3]
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_ab = np.zeros((nkpts,nvir,nkpts,nvir), dtype=t2.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    e_ab = e_vir[:,:,None,None] + e_vir[None,None,:,:]
    e_iab = - e_occ[:,:,None,None,None,None] + e_ab[None,None,:,:,:,:] 
    e_iab = e_iab.transpose(0,2,4,1,3,5).copy()

    if M_ab is None:
        M_ab = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim), dtype=np.complex)

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nkpts,nkpts,nocc,nvir,nvir)
        temp = np.zeros_like((r2),dtype=np.complex)
        temp_doubles = np.zeros_like((r2),dtype=np.complex)
        #temp_aij = np.zeros_like((r2),dtype=np.complex)
        
        eris_ovoo = eris.ovoo
        eris_ovvv = eris.ovvv

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ab,b->a',M_ab[kshift],r1)

########### ADC(2) coupling blocks #########################

        for kb in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kb,kshift, kc]
                s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv[ki,kc,kshift], r2[ki,kb], optimize = True)
                s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv[ki,kb,kshift], r2[ki,kb], optimize = True)
   
                temp_doubles[ki,kb] += lib.einsum('icab,a->ibc', eris_ovvv[ki,kc,kshift].conj(), r1, optimize = True)

        #s[s2:f2] +=  temp_doubles.reshape(-1)
################## ADC(2) ajk - bil block ############################

        #for kb in range(nkpts):
        #    for kc in range(nkpts):
        #        ki = kconserv[kc,kshift, kb]
        #        #temp_aij[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]
                temp_doubles[ki,kb] += e_iab[ki,kb,kc] *  r2[ki,kb]

        #s[s2:f2] = 1e13 * r2.reshape(-1)
        s[s2:f2] += temp_doubles.reshape(-1)

################ ADC(3) ajk - bil block ############################

#        if (method == "adc(2)-x" or method == "adc(3)"):
#
#               eris_oovv = eris.oovv
#               eris_ovvo = eris.ovvo
#
#               temp_doubles = np.zeros_like((r2),dtype=np.complex)
#
#               for kx in range(nkpts):
#                   for ky in range(nkpts):
#                       ki = kconserv[ky,kshift, kx]
#
#                       if isinstance(eris.vvvv, np.ndarray):
#                           eris_vvvv = eris.vvvv
#                           kj = kconserv[kx, ky, ki]
#                           for kv in range(nkpts):
#                               kw = kconserv[kv, kx, ky]
#                               temp_doubles[ki, kx] += lib.einsum('xywv,iwv->ixy', eris_vvvv[kx,ky,kw], r2[ki,kw],optimize=True)
#
#                       for kj in range(nkpts):
#                           kz = kconserv[ky,ki,kj]
#                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo[kj,kz,ky],r2[kj,kz],optimize = True)
#                           temp_doubles[ki,kx] += lib.einsum('jzyi,jxz->ixy',eris_ovvo[kj,kz,ky],r2[kj,kx],optimize = True)
#                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv[kj,ki,ky],r2[kj,kx],optimize = True)
#
#                           kz = kconserv[kj,ki,kx]
#                           temp_doubles[ki,kx] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_oovv[kj,ki,kx],r2[kj,kz],optimize = True)
#
#                           kw = kconserv[kj,ki,kx]
#                           temp_doubles[ki,kx] -=  0.5*lib.einsum('jixw,jwy->ixy',eris_oovv[kj,ki,kx],r2[kj,kw],optimize = True)
#                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv[kj,ki,ky],r2[kj,kx],optimize = True)
#
#                           kw = kconserv[ky,ki,kj]
#                           temp_doubles[ki,kx] += lib.einsum('jwyi,jxw->ixy',eris_ovvo[kj,kw,ky],r2[kj,kx],optimize = True)
#                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jwyi,jwx->ixy',eris_ovvo[kj,kw,ky],r2[kj,kw],optimize = True)
#
#               s[s2:f2] += temp_doubles.reshape(-1)
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

        return s
    return sigma_

def ip_adc_matvec(adc, kshift, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    t2 = adc.t2[0]
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
    nocc = adc.t2[0].shape[3]
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

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ij,j->i',M_ij[kshift],r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]

                #s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo[kj,ka,kk], r2[ka,kj], optimize = True)
                #s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo[kk,ka,kj], r2[ka,kj], optimize = True)

                s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo[kj,ka,kk].conj(), r2[ka,kj], optimize = True)
                s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo[kk,ka,kj].conj(), r2[ka,kj], optimize = True)
#################### ADC(2) ajk - i block ############################

                #temp[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk].conj(), r1, optimize = True)
                temp[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk], r1, optimize = True)

################# ADC(2) ajk - bil block ############################

                temp[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]

        s[s2:f2] += temp.reshape(-1)

################ ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):
        
               eris_oooo = eris.oooo
               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo
               eris_ovov = eris.ovov

               temp = np.zeros_like((r2),dtype=np.complex)
               for kj in range(nkpts):
                   for kk in range(nkpts):
                          ka = kconserv[kk, kshift, kj]
                          for kl in range(nkpts):
                              ki = kconserv[kj, kl, kk]

                              temp[ka,kj] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[kk,ki,kj], r2[ka,kl], optimize = True)
                              temp[ka,kj] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo[kk,kl,kj],r2[ka,ki], optimize = True)
                          
                          for kl in range(nkpts):
                              kb = kconserv[ka, kk, kl]
                              temp[ka,kj] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[kk,kl,kb],r2[kb,kj],optimize = True)

                              kb = kconserv[kl, kj, ka]
                              temp[ka,kj] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[kj,ka,kb],r2[kb,kk],optimize = True)
                              temp[ka,kj] -=  lib.einsum('jabl,blk->ajk',eris_ovvo[kj,ka,kb],r2[kb,kl],optimize = True)
                              kb = kconserv[ka, kj, kl]
                              temp[ka,kj] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv[kj,kl,kb],r2[kb,kl],optimize = True)

                          for ki in range(nkpts):
                               kb = kconserv[ka, kk, ki]
                               temp[ka,kj] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[kk,ki,kb],r2[kb,kj],optimize = True)

                               kb = kconserv[ka, kj, ki]
                               temp[ka,kj] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[kj,ki,kb],r2[kb,ki],optimize = True)
                               temp[ka,kj] -= lib.einsum('jabi,bik->ajk',eris_ovvo[kj,ka,kb],r2[kb,ki],optimize = True)
                               kb = kconserv[ki, kj, ka]
                               temp[ka,kj] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[kj,ka,kb],r2[kb,kk],optimize = True)

               s[s2:f2] += temp.reshape(-1)
               
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo
               eris_oovo = eris.oovo
               eris_ovvv = eris.ovvv
               t2_1 = adc.t2[0]

################ ADC(3) i - kja block and ajk - i ############################

               for kb in range(nkpts):
                   for kc in range(nkpts):
                       ka = kconserv[kc,kshift,kb]
                         
                       for kk in range(nkpts): 
                           temp = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=t2_1.dtype)
                           temp_1 = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=t2_1.dtype)

                           kj = kconserv[kk, kb, kc]
                           temp_1[ka,kb] +=  lib.einsum('jkbc,ajk->abc',    t2_1[kj,kk,kb], r2[ka,kj], optimize=True)
                           temp[ka,kb] += 0.25 * lib.einsum('jkbc,ajk->abc',t2_1[kj,kk,kb], r2[ka,kj], optimize=True)
                           temp[ka,kb] -= 0.25 * lib.einsum('jkbc,akj->abc',t2_1[kj,kk,kb], r2[ka,kk], optimize=True)
                           temp[ka,kb] -= 0.25 * lib.einsum('kjbc,ajk->abc',t2_1[kk,kj,kb], r2[ka,kj], optimize=True)
                           temp[ka,kb] += 0.25 * lib.einsum('kjbc,akj->abc',t2_1[kk,kj,kb], r2[ka,kk], optimize=True)

                           ki = kconserv[kc,ka,kb]
                           s[s1:f1] += lib.einsum('abc,icab->i',temp_1[ka,kb].conj(), eris_ovvv[ki,kc,ka].conj(), optimize=True)
                           s[s1:f1] += lib.einsum('abc,icab->i',temp[ka,kb].conj(),   eris_ovvv[ki,kc,ka].conj(), optimize=True)
                           s[s1:f1] -= lib.einsum('abc,ibac->i',temp[ka,kb].conj(),   eris_ovvv[ki,kb,ka].conj(), optimize=True)

               temp_doubles = np.zeros_like((r2))
               for kj in range(nkpts):
                   for kk in range(nkpts):
                          ka = kconserv[kk, kshift, kj]
                          for kc in range(nkpts):
                              temp = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=t2_1.dtype)
                              kb = kconserv[ka, kshift, kc]
                              ki = kconserv[kc,ka,kb]
                              temp[kc,kb] += lib.einsum('i,icab->cba',r1,eris_ovvv[ki,kc,ka].conj(), optimize=True)
                              kb = kconserv[kk,kj,kc]
                              temp_doubles[ka,kj] += lib.einsum('cba,kjcb->ajk',temp[kc,kb], t2_1[kk,kj,kc].conj(), optimize=True)
               s[s2:f2] += temp_doubles.reshape(-1) 


               for kl in range(nkpts):
                   for kk in range(nkpts):
                          kb = kconserv[kk, kshift, kl]
                          for kj in range(nkpts):
                              ka = kconserv[kb, kj, kl]
                              temp = np.zeros_like(r2)
                              temp[kb,kl] += lib.einsum('jlab,ajk->blk',t2_1[kj,kl,ka],r2[ka,kj],optimize=True)
                              temp[kb,kl] -= lib.einsum('jlab,akj->blk',t2_1[kj,kl,ka],r2[ka,kk],optimize=True)
                              temp[kb,kl] -= lib.einsum('ljab,ajk->blk',t2_1[kl,kj,ka],r2[ka,kj],optimize=True)
                              temp[kb,kl] += lib.einsum('ljab,akj->blk',t2_1[kl,kj,ka],r2[ka,kk],optimize=True)
                              temp[kb,kl] += lib.einsum('ljba,ajk->blk',t2_1[kl,kj,kb],r2[ka,kj],optimize=True)

                              temp_1 = np.zeros_like(r2)
                              temp_1[kb,kl] += lib.einsum('jlab,ajk->blk',t2_1[kj,kl,ka],r2[ka,kj],optimize=True)
                              temp_1[kb,kl] -= lib.einsum('jlab,akj->blk',t2_1[kj,kl,ka],r2[ka,kk],optimize=True)
                              temp_1[kb,kl] += lib.einsum('jlab,ajk->blk',t2_1[kj,kl,ka],r2[ka,kj],optimize=True)
                              temp_1[kb,kl] -= lib.einsum('ljab,ajk->blk',t2_1[kl,kj,ka],r2[ka,kj],optimize=True)

                              temp_2 = np.zeros_like(r2)
                              temp_2[kb,kl] = lib.einsum('jlba,akj->blk',t2_1[kj,kl,kb],r2[ka,kk], optimize=True)

                              ki = kconserv[kk, kl, kb]
                              s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp[kb,kl].conj(),  eris_ovoo[kl,kb,ki].conj(),optimize=True)
                              s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp[kb,kl].conj(),  eris_ovoo[ki,kb,kl].conj(),optimize=True)
                              s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1[kb,kl].conj(),eris_ovoo[kl,kb,ki].conj(),optimize=True)
                              s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2[kb,kl].conj(),eris_ovoo[ki,kb,kl].conj(),optimize=True)
                              del temp
                              del temp_1
                              del temp_2

                   for kj in range(nkpts):
                          kb = kconserv[kj, kshift, kl]
                          for kk in range(nkpts):
                              ka = kconserv[kb, kk, kl]
                              temp = np.zeros_like(r2)
                              temp[kb,kl] -= lib.einsum('klab,akj->blj',t2_1[kk,kl,ka],r2[ka,kk],optimize=True)
                              temp[kb,kl] += lib.einsum('klab,ajk->blj',t2_1[kk,kl,ka],r2[ka,kj],optimize=True)
                              temp[kb,kl] += lib.einsum('lkab,akj->blj',t2_1[kl,kk,ka],r2[ka,kk],optimize=True)
                              temp[kb,kl] -= lib.einsum('lkab,ajk->blj',t2_1[kl,kk,ka],r2[ka,kj],optimize=True)
                              temp[kb,kl] -= lib.einsum('lkba,akj->blj',t2_1[kl,kk,kb],r2[ka,kk],optimize=True)

                              temp_1 = np.zeros_like(r2)
                              temp_1[kb,kl] -= 2.0 * lib.einsum('klab,akj->blj',t2_1[kk,kl,ka],r2[ka,kk],optimize=True)
                              temp_1[kb,kl] += lib.einsum('klab,ajk->blj',t2_1[kk,kl,ka],r2[ka,kj],optimize=True)
                              #temp_1[kb,kl] -= lib.einsum('klab,akj->blj',t2_1[kl,ka,kb],r2[ka,kk],optimize=True)
                              temp_1[kb,kl] += lib.einsum('lkab,akj->blj',t2_1[kl,kk,ka],r2[ka,kk],optimize=True)

                              temp_2 = np.zeros_like(r2)
                              temp_2[kb,kl] = -lib.einsum('klba,ajk->blj',t2_1[kk,kl,kb],r2[ka,kj],optimize=True)

                              ki = kconserv[kj, kl, kb]
                              s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp[kb,kl].conj(),  eris_ovoo[kl,kb,ki].conj(),optimize=True)
                              s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp[kb,kl].conj(),  eris_ovoo[ki,kb,kl].conj(),optimize=True)
                              s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1[kb,kl].conj(),eris_ovoo[kl,kb,ki].conj(),optimize=True)
                              s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2[kb,kl].conj(),eris_ovoo[ki,kb,kl].conj(),optimize=True)
               
               del temp
               del temp_1
               del temp_2

               temp_doubles = np.zeros_like((r2))
               for kj in range(nkpts):
                   for kk in range(nkpts):
                          ka = kconserv[kk, kshift, kj]
                          for kl in range(nkpts):
                              temp = np.zeros((nkpts,nkpts,nocc,nvir,nocc),dtype=t2_1.dtype)
                              temp_1 = np.zeros((nkpts,nkpts,nocc,nvir,nocc),dtype=t2_1.dtype)
                              temp_2 = np.zeros((nkpts,nkpts,nocc,nvir,nocc),dtype=t2_1.dtype)
                              kb = kconserv[kshift, kl, kk]
                              ki = kconserv[kk,kl,kb]
                              temp_1[kk,kb] += lib.einsum('i,lbik->kbl',r1,eris_ovoo[kl,kb,ki].conj(), optimize=True)
                              temp[kk,kb] += lib.einsum('i,lbik->kbl',r1,eris_ovoo[kl,kb,ki].conj(), optimize=True)
                              temp[kk,kb] -= lib.einsum('i,iblk->kbl',r1,eris_ovoo[ki,kb,kl].conj(), optimize=True)
                              kb = kconserv[ka, kl, kj]
                              temp_doubles[ka,kj] += lib.einsum('kbl,ljba->ajk',temp[kk,kb], t2_1[kl,kj,kb].conj(), optimize=True)
                              temp_doubles[ka,kj] += lib.einsum('kbl,jlab->ajk',temp_1[kk,kb], t2_1[kj,kl,ka].conj(), optimize=True)
                              temp_doubles[ka,kj] -= lib.einsum('kbl,ljab->ajk',temp_1[kk,kb], t2_1[kl,kj,ka].conj(), optimize=True)

                              kb = kconserv[kshift, kl, kj]
                              ki = kconserv[kb,kl,kj]
                              temp_2[kj,kb] -= lib.einsum('i,iblj->jbl',r1,eris_ovoo[ki,kb,kl].conj(), optimize=True)
                              kb = kconserv[ka, kk, kl]
                              temp_doubles[ka,kj] += lib.einsum('jbl,klba->ajk',temp_2[kj,kb], t2_1[kk,kl,kb].conj(), optimize=True)

               s[s2:f2] += temp_doubles.reshape(-1) 

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s
    return sigma_

def ip_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    T = np.zeros((dim),dtype=np.complex)

######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T[s1:f1]  = idn_occ[orb, :]
        T[s1:f1] += 0.25*lib.einsum('pqrkdc,pqrikdc->i',t2_1[:,:,:,:,orb,:,:], t2_1, optimize = True)
        #T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        #T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        #T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        #T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        #T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_1, optimize = True)
        #T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_1, optimize = True)
    else :
        for ki in range (nkpts):
            T[s1:f1] += t1_2[ki][:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        #t2_1_t = t2_1.transpose(2,3,1,0)

        #T[s2:f2] = t2_1_t[(orb-nocc),:,:,:].reshape(-1)

    return T

def get_trans_moments(adc):

    nmo  = adc.nmo
    T = []
    for orb in range(nmo):

            T_a = adc.compute_trans_moments(orb)
            T.append(T_a)

    T = np.array(T)
    return T

def renormalize_eigenvectors_ip(adc, nroots=1):

    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    t2 = adc.t2[0]
    t1_2 = adc.t1[0]
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    dim = n_singles + n_doubles

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nkpts,nkpts,nvir,nocc,nocc)
        #UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        UdotU = np.dot(U1.ravel(), U1.ravel()) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,1,2,4,3).ravel())
        U[:,I] /= np.sqrt(UdotU)

    U = U.reshape(-1,nroots)

    return U

def get_properties(adc, nroots=1):

    #Transition moments
    T = adc.get_trans_moments()
    
    #Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(nroots)
    X = np.dot(T, U).reshape(-1, nroots)
    
    #Spectroscopic factors
    P = 2.0*lib.einsum("pi,pi->i", X, X)
    P = P.real

    return P,X



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
        self.t1 = adc.t1
        self.t2 = adc.t2
        #self.imds = adc.imds
        #self.e_corr = adc.e_corr
        self.method = adc.method
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
        self.imds = adc.imds

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ea
    get_diag = ea_adc_diag
    matvec = ea_adc_matvec
    vector_size = ea_vector_size
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors_ip
    get_properties = get_properties
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
        self.max_space = 100
        self.max_cycle = 300
        self.conv_tol  = 1e-8
        self.tol_residual  = 1e-6
        self.t1 = adc.t1
        self.t2 = adc.t2
        #self.imds = adc.imds
        #self.e_corr = adc.e_corr
        self.method = adc.method
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
        self.imds = adc.imds

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    vector_size = ip_vector_size
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors_ip
    get_properties = get_properties
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
