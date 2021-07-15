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
from pyscf.adc import radc_ao2mo

import itertools

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
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

        ##guess = adc.get_init_guess(kshift,imds,nroots, diag, ascending = True)
        #kop_npick = [0,1,2,3]
        #nkop_chk = False
        #Eh2ev = 27.211386245988        

        #if (nkop_chk is True) or (kop_npick is not False):
        #        kop_w, kop_v = np.linalg.eigh(imds[kshift])
        #        kop_v = kop_v
        #        kop_w = kop_w
        #        dim_kop = kop_v.shape[1]
        #       
        #
        #        for kop_root, w in enumerate(kop_w):
        #            print("Koopman State #", kop_root, " Energy [Eh]: ", w, "  Energy [ev]: ", w*Eh2ev ) ## Only works for IP; need another print statement for EA
        #
        #        if nkop_chk is True:
        #           print("Initial Koopman's state checkpoint... exiting calculation")
        #           exit()
        #
        #        if kop_npick is not False:
        #            kroots, dim_guess = np.array(guess).shape
        #            len_kop_npick = len(kop_npick)
        #            nroots = len_kop_npick
        #            guess = np.zeros((dim_guess,kroots))
        #            for idx_guess, npick in enumerate(kop_npick):
        #               print (guess[:dim_kop,idx_guess].shape)
        #               print (kop_v[:,npick].shape)
        #               print (idx_guess)
        #               print (np.array(guess).shape)
        #               guess[:dim_kop,idx_guess] = kop_v[:,npick] 

        conv_k,evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space,tol_residual=adc.tol_residual)
    
        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        convs[k] = conv_k
        print (conv_k)

    adc.U = np.array(evecs).T.copy()

    for k, kshift in enumerate(kptlist):
        P,X = adc.get_properties(kshift,nroots)

    print (evals)
    print (P)
#    exit()
#
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

##@profile
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
   
    #eris_oovv = eris.oovv[:].copy()

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

    if not isinstance(eris.oooo, np.ndarray):
        t2_1 = radc_ao2mo.write_dataset(t2_1)

    t1_2 = np.zeros((nkpts,nocc,nvir), dtype=t2_1.dtype)
    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    # Compute second-order singles t1 (tij)

    t1_2 = np.zeros((nkpts,nocc,nvir), dtype=t2_1.dtype)
    eris_ovoo = eris.ovoo

    for ki in range (nkpts):
        ka = ki
        for kk in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ki, kc, kk]
                ka = kconserv[kc, kk, kd]

                if isinstance(eris.ovvv, type(None)):
                    chnk_size = myadc.chnk_size
                    #chnk_size = kadc_ao2mo.calculate_chunk_size(myadc)
                    #chnk_size = kadc_ao2mo.calculate_chunk_size(myadc)
                    if chnk_size > nocc:
                        chnk_size = nocc
                    a = 0
                    for p in range(0,nocc,chnk_size):
                     #if getattr(eris, 'Lov', None) is not None:
                         eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov[kk,kd], eris.Lvv[ka,kc], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                         k = eris_ovvv.shape[0]
                         t1_2[ki] += 1.5*lib.einsum('kdac,ikcd->ia',eris_ovvv.conj(),t2_1[ki,kk,kc,:,a:a+k].conj(),optimize=True)
                         t1_2[ki] -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv.conj(),t2_1[kk,ki,kc,a:a+k,:].conj(),optimize=True)
                         del eris_ovvv
                         eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov[kk,kc], eris.Lvv[ka,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                         t1_2[ki] -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv.conj(),t2_1[ki,kk,kc,:,a:a+k].conj(),optimize=True)
                         t1_2[ki] += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv.conj(),t2_1[kk,ki,kc,a:a+k,:].conj(),optimize=True)
                         del eris_ovvv
                         a += k
                else:
                    eris_ovvv = eris.ovvv[:]
                    t1_2[ki] += 1.5*lib.einsum('kdac,ikcd->ia',eris_ovvv[kk,kd,ka].conj(),t2_1[ki,kk,kc].conj(),optimize=True)
                    t1_2[ki] -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv[kk,kd,ka].conj(),t2_1[kk,ki,kc].conj(),optimize=True)
                    t1_2[ki] -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv[kk,kc,ka].conj(),t2_1[ki,kk,kc].conj(),optimize=True)
                    t1_2[ki] += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv[kk,kc,ka].conj(),t2_1[kk,ki,kc].conj(),optimize=True)
                    del eris_ovvv

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

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
    # Compute second-order doubles t2 (tijab)
        t2_1_vvvv = np.zeros_like((t2_1))

        eris_oooo = eris.oooo
        eris_ovov = eris.ovov
        eris_ovvo = eris.ovvo

        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka, kc, kb]
            for ki in range(nkpts):
                kj = kconserv[ka, ki, kb]
                if isinstance(eris.vvvv, np.ndarray):
                    eris_vvvv = eris.vvvv.reshape(nkpts,nkpts,nkpts,nvir*nvir,nvir*nvir)
                    t2_1_a = t2_1.reshape(nkpts,nkpts,nkpts,nocc*nocc,nvir*nvir)
                    t2_1_vvvv[ki, kj, ka] += np.dot(t2_1_a[ki,kj,kc], eris_vvvv[kc,kd,ka].conj()).reshape(nocc,nocc,nvir,nvir)
                elif isinstance(eris.vvvv, type(None)):
                    t2_1_vvvv[ki,kj,ka] += contract_ladder(myadc,t2_1[ki,kj,kc],eris.Lvv,ka,kb,kc) 
                else : 
                    t2_1_vvvv[ki,kj,ka] += contract_ladder(myadc,t2_1[ki,kj,kc],eris.vvvv,kc,kd,ka) 

        t2_2 = np.zeros_like((t2_1))

        t2_2 = t2_1_vvvv.copy()
        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv = radc_ao2mo.write_dataset(t2_1_vvvv)
        
        eris_oovv = eris.oovv
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

        if not isinstance(eris.oooo, np.ndarray):
            t2_2 = radc_ao2mo.write_dataset(t2_2)


    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)
        
    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2, t2_1_vvvv

def compute_energy(myadc, t2, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)
    nkpts = myadc.nkpts

    emp2 = 0.0
    eris_ovov = eris.ovov
    t2_amp = t2[0][:].copy()     
 
    if (myadc.method == "adc(3)"):
        t2_amp += t2[1]

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):

        emp2 += 2 * lib.einsum('ijab,iajb', t2_amp[ki,kj,ka], eris_ovov[ki,ka,kj],optimize=True)
        emp2 -= 1 * lib.einsum('ijab,jaib', t2_amp[ki,kj,ka], eris_ovov[kj,ka,ki],optimize=True)
        #emp2 -= 0.5 * lib.einsum('jiab,iajb', t2_amp[kj,ki,ka], eris_ovov[ki,ka,kj],optimize=True)
        #emp2 += 0.5 * lib.einsum('jiab,jaib', t2_amp[kj,ki,ka], eris_ovov[kj,ka,ki],optimize=True)
  
    del  t2_amp
    emp2 = emp2.real / nkpts
    return emp2

def contract_ladder(myadc,t_amp,vvvv,ka,kb,kc):

    log = logger.Logger(myadc.stdout, myadc.verbose)
    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc
    nkpts = myadc.nkpts
    kconserv = myadc.khelper.kconserv

    kd = kconserv[ka, kc, kb]
    #t_amp = np.ascontiguousarray(t_amp.reshape(nocc*nocc,nvir*nvir).T)
    t_amp = np.ascontiguousarray(t_amp.reshape(nocc*nocc,nvir*nvir))
    #t = np.zeros((nvir,nvir, nocc*nocc),dtype=t_amp.dtype)
    t = np.zeros((nocc,nocc, nvir, nvir),dtype=t_amp.dtype)
    #chnk_size = kadc_ao2mo.calculate_chunk_size(myadc)
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
            #t = np.dot(vvvv_p.conj(),t_amp[a:a+k]).reshape(nvir,nvir,nocc*nocc)
            t += np.dot(t_amp[:,a:a+k],vvvv_p.conj()).reshape(nocc,nocc,nvir,nvir)
            #t += np.einsum('ijcd,cdab->ijab',t_amp[:,:,a:a+k],vvvv_p.conj())
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

    #transform_integrals = kadc_ao2mo.transform_integrals_outcore
    #transform_integrals = kadc_ao2mo.transform_integrals_incore
    transform_integrals = kadc_ao2mo.transform_integrals_df
    compute_amplitudes = compute_amplitudes
    compute_energy = compute_energy
    compute_amplitudes_energy = compute_amplitudes_energy
    get_chnk_size = kadc_ao2mo.calculate_chunk_size

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

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
           if getattr(self, 'with_df', None): 
               self.with_df = self.with_df
           else :
               self.with_df = self._scf.with_df
           self.chnk_size = self.get_chnk_size()
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
        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
           if getattr(self, 'with_df', None): 
               self.with_df = self.with_df
           else :
               self.with_df = self._scf.with_df
           self.chnk_size = self.get_chnk_size()
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

##@profile
def get_imds_ea(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nmo =  adc.get_nmo
    nocc = adc.nocc
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
    M_ab = np.empty((nkpts,nvir,nvir),dtype=mo_coeff.dtype)

    for ka in range(nkpts):
        kb = ka
        M_ab[ka] = lib.einsum('ab,a->ab', idn_vir, e_vir[ka])
        for kl in range(nkpts):
            for km in range(nkpts):

                kd = kconserv[kl,ka,km]
                # Second-order terms
                t2_1_lmb = adc.t2[0][kl,km,kb]
                t2_1_lma = adc.t2[0][kl,km,ka]

                M_ab[ka] +=  lib.einsum('l,lmad,lmbd->ab',e_occ[kl] , t2_1_lma, t2_1_lmb.conj(),optimize=True)
                M_ab[ka] +=  lib.einsum('l,lmad,lmbd->ab',e_occ[kl] , t2_1_lma, t2_1_lmb.conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir[kd], t2_1_lma, t2_1_lmb.conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir[kd], t2_1_lma, t2_1_lmb.conj(),optimize=True)

                M_ab_t = lib.einsum('lmad,lmbd->ab', t2_1_lma,t2_1_lmb.conj(), optimize=True)
                M_ab[ka] -= 1 *  lib.einsum('a,ab->ab',e_vir[ka],M_ab_t,optimize=True)
                M_ab[ka] -= 1 *  lib.einsum('b,ab->ab',e_vir[kb],M_ab_t,optimize=True)
                del M_ab_t
                del t2_1_lma

                t2_1_mla = adc.t2[0][km,kl,ka]
                M_ab[ka] -=  lib.einsum('l,mlad,lmbd->ab',e_occ[kl] , t2_1_mla, t2_1_lmb.conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('d,mlad,lmbd->ab',e_vir[kd], t2_1_mla, t2_1_lmb.conj(),optimize=True)
                del t2_1_lmb

                M_ab[ka] += 0.5 *  lib.einsum('mlad,lbmd->ab',t2_1_mla, eris_ovov[kl,kb,km],optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('mlad,ldmb->ab',t2_1_mla, eris_ovov[kl,kd,km],optimize=True)

                t2_1_mlb = adc.t2[0][km,kl,kb]
                M_ab[ka] +=        lib.einsum('l,mlad,mlbd->ab',e_occ[kl] ,t2_1_mla, t2_1_mlb.conj(),optimize=True)
                M_ab[ka] +=        lib.einsum('l,mlad,mlbd->ab',e_occ[kl] ,t2_1_mla, t2_1_mlb.conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir[kd], t2_1_mla, t2_1_mlb.conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir[kd], t2_1_mla, t2_1_mlb.conj(),optimize=True)
                del t2_1_mla

                t2_1_lma = adc.t2[0][kl,km,ka]
                M_ab[ka] -=        lib.einsum('l,lmad,mlbd->ab',e_occ[kl] ,t2_1_lma, t2_1_mlb.conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('d,lmad,mlbd->ab',e_vir[kd], t2_1_lma, t2_1_mlb.conj(),optimize=True)
                M_ab_t =           lib.einsum('lmad,mlbd->ab', t2_1_lma,t2_1_mlb.conj(), optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('a,ab->ab',e_vir[ka],M_ab_t,optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('b,ab->ab',e_vir[kb],M_ab_t,optimize=True)
                del M_ab_t
                del t2_1_mlb

                M_ab[ka] -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_1_lma, eris_ovov[kl,kb,km],optimize=True)
                M_ab[ka] -=        lib.einsum('lmad,lbmd->ab',t2_1_lma, eris_ovov[kl,kb,km],optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_1_lma, eris_ovov[kl,kd,km],optimize=True)
                del t2_1_lma

                t2_1_lmb = adc.t2[0][kl,km,kb]
                M_ab[ka] -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_1_lmb.conj(), eris_ovov[kl,ka,km].conj(),optimize=True)
                M_ab[ka] -=        lib.einsum('lmbd,lamd->ab',t2_1_lmb.conj(), eris_ovov[kl,ka,km].conj(),optimize=True)
                M_ab[ka] += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_1_lmb.conj(), eris_ovov[kl,kd,km].conj(),optimize=True)
                del t2_1_lmb

                t2_1_mlb = adc.t2[0][km,kl,kb]
                M_ab[ka] += 0.5 *  lib.einsum('mlbd,lamd->ab',t2_1_mlb.conj(), eris_ovov[kl,ka,km].conj(),optimize=True)
                M_ab[ka] -= 0.5 *  lib.einsum('mlbd,ldma->ab',t2_1_mlb.conj(), eris_ovov[kl,kd,km].conj(),optimize=True)
                del t2_1_mlb

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    if(method =='adc(3)'):

       eris_oovv = eris.oovv 
       eris_ovvo = eris.ovvo
       eris_oooo = eris.oooo
  
       t1_2 = adc.t1[0]

       for ka in range(nkpts):
           kb = ka
           for kl in range(nkpts):
               kd = kconserv[ka,kb,kl]
               #kd = kconserv[ka,kl,kb]

               if isinstance(eris.ovvv, type(None)):
                   #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                   chnk_size = adc.chnk_size
                   if chnk_size > nocc:
                       chnk_size = nocc
                   a = 0
                   for p in range(0,nocc,chnk_size):
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kd], eris.Lvv[ka,kb], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                       k = eris_ovvv.shape[0]
                       M_ab[ka] += 2. * lib.einsum('ld,ldab->ab',t1_2[kl][a:a+k].conj(), eris_ovvv,optimize=True)
                       del eris_ovvv
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kd], eris.Lvv[kb,ka], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                       M_ab[ka] += 2. * lib.einsum('ld,ldba->ab',t1_2[kl][a:a+k], eris_ovvv.conj(),optimize=True)
                       del eris_ovvv
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kb], eris.Lvv[ka,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                       M_ab[ka] -= lib.einsum('ld,lbad->ab',t1_2[kl][a:a+k].conj(), eris_ovvv,optimize=True)
                       del eris_ovvv
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,ka], eris.Lvv[kb,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                       M_ab[ka] -= lib.einsum('ld,labd->ab',t1_2[kl,a:a+k], eris_ovvv.conj(),optimize=True)
                       del eris_ovvv
                       a += k

               else :
                   eris_ovvv = eris.ovvv
                   M_ab[ka] += 2. * lib.einsum('ld,ldab->ab',t1_2[kl], eris_ovvv[kl,kd,ka].conj(),optimize=True)
                   M_ab[ka] += 2. * lib.einsum('ld,ldba->ab',t1_2[kl], eris_ovvv[kl,kd,kb].conj(),optimize=True)
                   M_ab[ka] -=  lib.einsum('ld,lbad->ab',t1_2[kl], eris_ovvv[kl,kb,ka].conj(),optimize=True)
                   M_ab[ka] -= lib.einsum('ld,labd->ab',t1_2[kl], eris_ovvv[kl,ka,kb].conj(),optimize=True)

       cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

       for ka in range(nkpts):
           kb = ka
           for kl in range(nkpts):
                for km in range(nkpts):
                    kd = kconserv[kl,km,ka]

                    t2_1 = adc.t2[0]
                    t2_2_lma = adc.t2[1][kl,km,ka]
                    M_ab[ka] -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_2_lma, eris_ovov[kl,kb,km],optimize=True)
                    M_ab[ka] += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_2_lma, eris_ovov[kl,kd,km],optimize=True)
                    M_ab[ka] -=    lib.einsum('lmad,lbmd->ab',t2_2_lma, eris_ovov[kl,kb,km],optimize=True)

                    M_ab[ka] +=  2. *  lib.einsum('l,lmbd,lmad->ab',e_occ[kl], t2_1[kl,km,kb].conj(), t2_2_lma, optimize=True)
                    M_ab[ka] -=        lib.einsum('l,mlbd,lmad->ab',e_occ[kl], t2_1[km,kl,kb].conj(), t2_2_lma, optimize=True)

                    M_ab[ka] -= 1.0 *lib.einsum('d,lmbd,lmad->ab', e_vir[kd], t2_1[kl,km,kb].conj(), t2_2_lma, optimize=True)
                    M_ab[ka] += 0.5 *lib.einsum('d,mlbd,lmad->ab', e_vir[kd], t2_1[km,kl,kb].conj(), t2_2_lma, optimize=True)

                    M_ab_t = lib.einsum('lmbd,lmad->ab', t2_1[kl,km,kb].conj(),t2_2_lma, optimize=True)
                    del t2_2_lma
                    M_ab[ka] -= 1. * lib.einsum('a,ab->ab',e_vir[ka], M_ab_t, optimize=True)
                    M_ab[ka] -= 1. * lib.einsum('a,ba->ab',e_vir[ka], M_ab_t.conj(), optimize=True)
                    M_ab[ka] -= 1. * lib.einsum('b,ab->ab',e_vir[kb], M_ab_t, optimize=True)
                    M_ab[ka] -= 1. * lib.einsum('b,ba->ab',e_vir[kb], M_ab_t.conj(), optimize=True)
                    del M_ab_t

                    t2_2_mla = adc.t2[1][km,kl,ka]
                    M_ab[ka] += 0.5 *  lib.einsum('mlad,lbmd->ab',t2_2_mla, eris_ovov[kl,kb,km],optimize=True)
                    M_ab[ka] -= 0.5 *  lib.einsum('mlad,ldmb->ab',t2_2_mla, eris_ovov[kl,kd,km],optimize=True)

                    M_ab[ka] -=        lib.einsum('l,lmbd,mlad->ab',e_occ[kl], t2_1[kl,km,kb].conj(), t2_2_mla, optimize=True)
                    M_ab[ka] +=  2. *  lib.einsum('l,mlbd,mlad->ab',e_occ[kl], t2_1[km,kl,kb].conj(), t2_2_mla, optimize=True)

                    M_ab[ka] += 0.5 *lib.einsum('d,lmbd,mlad->ab', e_vir[kd], t2_1[kl,km,kb].conj(), t2_2_mla, optimize=True)
                    M_ab[ka] -= 1.0 *lib.einsum('d,mlbd,mlad->ab', e_vir[kd], t2_1[km,kl,kb].conj(), t2_2_mla, optimize=True)

                    M_ab_t_1 = lib.einsum('lmbd,mlad->ab', t2_1[kl,km,kb].conj(),t2_2_mla, optimize=True)
                    del t2_2_mla
                    M_ab[ka] += 0.5 * lib.einsum('a,ab->ab',e_vir[ka], M_ab_t_1, optimize=True)
                    M_ab[ka] += 0.5 * lib.einsum('a,ba->ab',e_vir[ka], M_ab_t_1.conj(), optimize=True)
                    M_ab[ka] += 0.5 * lib.einsum('b,ab->ab',e_vir[kb], M_ab_t_1, optimize=True)
                    M_ab[ka] += 0.5 * lib.einsum('b,ba->ab',e_vir[kb], M_ab_t_1.conj(), optimize=True)
                    del M_ab_t_1

                    kd = kconserv[kl,km,kb]
                    t2_2_lmb = adc.t2[1][kl,km,kb]
                    M_ab[ka] -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_2_lmb.conj(), eris_ovov[kl,ka,km].conj(),optimize=True)
                    M_ab[ka] += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_2_lmb.conj(), eris_ovov[kl,kd,km].conj(),optimize=True)
                    M_ab[ka] -= lib.einsum('lmbd,lamd->ab',t2_2_lmb.conj(), eris_ovov[kl,ka,km].conj(),optimize=True)

                    M_ab[ka] +=  2. *  lib.einsum('l,lmad,lmbd->ab',e_occ[kl], t2_1[kl,km,ka], t2_2_lmb.conj(), optimize=True)
                    M_ab[ka] -=        lib.einsum('l,mlad,lmbd->ab',e_occ[kl], t2_1[km,kl,ka], t2_2_lmb.conj(), optimize=True)

                    M_ab[ka] -= 1.0 * lib.einsum('d,lmad,lmbd->ab', e_vir[kd], t2_1[kl,km,ka], t2_2_lmb.conj(), optimize=True)
                    M_ab[ka] += 0.5 * lib.einsum('d,mlad,lmbd->ab', e_vir[kd], t2_1[km,kl,ka], t2_2_lmb.conj(), optimize=True)
                    del t2_2_lmb

                    t2_2_mlb = adc.t2[1][km,kl,kb]
                    M_ab[ka] += 0.5 *  lib.einsum('mlbd,mdla->ab',t2_2_mlb.conj(), eris_ovov[km,kd,kl].conj(),optimize=True)
                    M_ab[ka] -= 0.5 *  lib.einsum('mlbd,ldma->ab',t2_2_mlb.conj(), eris_ovov[kl,kd,km].conj(),optimize=True)

                    M_ab[ka] -=        lib.einsum('l,lmad,mlbd->ab',e_occ[kl], t2_1[kl,km,ka], t2_2_mlb.conj(), optimize=True)
                    M_ab[ka] +=  2. *  lib.einsum('l,mlad,mlbd->ab',e_occ[kl], t2_1[km,kl,ka], t2_2_mlb.conj(), optimize=True)

                    M_ab[ka] += 0.5 * lib.einsum('d,lmad,mlbd->ab', e_vir[kd], t2_1[kl,km,ka], t2_2_mlb.conj(), optimize=True)
                    M_ab[ka] -= 1.0 * lib.einsum('d,mlad,mlbd->ab', e_vir[kd], t2_1[km,kl,ka], t2_2_mlb.conj(), optimize=True)

                    del t2_2_mlb

           #log.timer_debug1("Starting the small integrals  calculation")
           for kn, ke, kd in kpts_helper.loop_kkk(nkpts):
           
               km = kconserv[kb,kn,ke]
               kl = kconserv[kn,ke,kd]
               #temp_t2_v_1 = np.zeros_like((eris_ovov))
               #temp_t2_v_1[kn,ke,km] = lib.einsum('lned,mlbd->nemb',t2_1[kl,kn,ke], t2_1[km,kl,kb].conj(),optimize=True)
               temp_t2_v_1 = lib.einsum('lned,mlbd->nemb',t2_1[kl,kn,ke], t2_1[km,kl,kb].conj(),optimize=True)
               M_ab[ka] -=      lib.einsum('nemb,nmae->ab',temp_t2_v_1, eris_oovv[kn,km,ka], optimize=True)
               M_ab[ka] += 2. * lib.einsum('nemb,maen->ab',temp_t2_v_1, eris_ovvo[km,ka,ke].conj(), optimize=True)
               M_ab[ka] += 2. * lib.einsum('nema,mben->ab',temp_t2_v_1.conj(), eris_ovvo[km,kb,ke], optimize=True)
               M_ab[ka] -=      lib.einsum('nema,mneb->ab',temp_t2_v_1.conj(), eris_oovv[km,kn,ke], optimize=True)
               del temp_t2_v_1

               #temp_t2_v_1 = np.zeros_like((eris_ovov))
               #temp_t2_v_1[km,kb,kn] = lib.einsum('nled,lmbd->mbne',t2_1[kn,kl,ke], t2_1[kl,km,kb].conj(),optimize=True)
               temp_t2_v_1 = lib.einsum('nled,lmbd->mbne',t2_1[kn,kl,ke], t2_1[kl,km,kb].conj(),optimize=True)
               M_ab[ka] -=      lib.einsum('mbne,nmae->ab',temp_t2_v_1, eris_oovv[kn,km,ka], optimize=True)
               M_ab[ka] += 2. * lib.einsum('mbne,maen->ab',temp_t2_v_1, eris_ovvo[km,ka,ke].conj(), optimize=True)
               M_ab[ka] -=      lib.einsum('mane,mneb->ab',temp_t2_v_1.conj(), eris_oovv[km,kn,ke], optimize=True)
               M_ab[ka] += 2. * lib.einsum('mane,mben->ab',temp_t2_v_1.conj(), eris_ovvo[km,kb,ke], optimize=True)
               del temp_t2_v_1

               #temp_t2_v_2 = np.zeros_like((eris_ovov))
               #temp_t2_v_2[kn,ke,km] = lib.einsum('nled,mlbd->nemb',t2_1[kn,kl,ke], t2_1[km,kl,kb].conj(),optimize=True)
               temp_t2_v_2 = lib.einsum('nled,mlbd->nemb',t2_1[kn,kl,ke], t2_1[km,kl,kb].conj(),optimize=True)
               M_ab[ka] += 2. * lib.einsum('nemb,nmae->ab',temp_t2_v_2, eris_oovv[kn,km,ka], optimize=True)
               M_ab[ka] -= 4. * lib.einsum('nemb,maen->ab',temp_t2_v_2, eris_ovvo[km,ka,ke].conj(), optimize=True)
               M_ab[ka] += 2. * lib.einsum('nema,mneb->ab',temp_t2_v_2.conj(), eris_oovv[km,kn,ke], optimize=True)
               M_ab[ka] -= 4. * lib.einsum('nema,mben->ab',temp_t2_v_2.conj(), eris_ovvo[km,kb,ke], optimize=True)
               del temp_t2_v_2

               #temp_t2_v_3 = np.zeros_like((eris_ovov))
               #temp_t2_v_3[kn,ke,km] = lib.einsum('lned,lmbd->nemb',t2_1[kl,kn,ke], t2_1[kl,km,kb].conj(),optimize=True)
               temp_t2_v_3 = lib.einsum('lned,lmbd->nemb',t2_1[kl,kn,ke], t2_1[kl,km,kb].conj(),optimize=True)
               M_ab[ka] -=      lib.einsum('nemb,maen->ab',temp_t2_v_3, eris_ovvo[km,ka,ke].conj(), optimize=True)
               M_ab[ka] += 2. * lib.einsum('nemb,nmae->ab',temp_t2_v_3, eris_oovv[kn,km,ka], optimize=True)
               M_ab[ka] += 2. * lib.einsum('nema,mneb->ab',temp_t2_v_3.conj(), eris_oovv[km,kn,ke], optimize=True)
               M_ab[ka] -=      lib.einsum('nema,mben->ab',temp_t2_v_3.conj(), eris_ovvo[km,kb,ke], optimize=True)
               del temp_t2_v_3

               kl = kconserv[kn,ka,ke]
               km = kconserv[ka,kd,kl]
               #temp_t2_v_4 = np.zeros_like((eris_oovv))
               #temp_t2_v_4[kl,km,ka] = lib.einsum('lnae,nmde->lmad',t2_1[kl,kn,ka], eris_oovv[kn,km,kd],optimize=True)
               temp_t2_v_4 = lib.einsum('lnae,nmde->lmad',t2_1[kl,kn,ka], eris_oovv[kn,km,kd],optimize=True)
               M_ab[ka] -=   lib.einsum('mlbd,lmad->ab',t2_1[km,kl,kb].conj(), temp_t2_v_4,optimize=True)
               M_ab[ka] += 2. * lib.einsum('lmbd,lmad->ab',t2_1[kl,km,kb].conj(), temp_t2_v_4,optimize=True)
               del temp_t2_v_4

               #temp_t2_v_5 = np.zeros_like((eris_ovov))
               #temp_t2_v_5[kl,ka,km] = lib.einsum('nlae,nmde->lamd',t2_1[kn,kl,ka], eris_oovv[kn,km,kd],optimize=True)
               temp_t2_v_5 = lib.einsum('nlae,nmde->lamd',t2_1[kn,kl,ka], eris_oovv[kn,km,kd],optimize=True)
               M_ab[ka] += 2. * lib.einsum('mlbd,lamd->ab',t2_1[km,kl,kb].conj(), temp_t2_v_5, optimize=True)
               M_ab[ka] -= lib.einsum('lmbd,lamd->ab',t2_1[kl,km,kb].conj(), temp_t2_v_5, optimize=True)
               del temp_t2_v_5

               #temp_t2_v_6 = np.zeros_like((eris_ovvo))
               #temp_t2_v_6[kl,ka,kd] = lib.einsum('lnae,nedm->ladm',t2_1[kl,kn,ka], eris_ovvo[kn,ke,kd],optimize=True)
               temp_t2_v_6 = lib.einsum('lnae,nedm->ladm',t2_1[kl,kn,ka], eris_ovvo[kn,ke,kd],optimize=True)
               M_ab[ka] += 2. * lib.einsum('mlbd,ladm->ab',t2_1[km,kl,kb].conj(), temp_t2_v_6, optimize=True)
               M_ab[ka] -= 4. * lib.einsum('lmbd,ladm->ab',t2_1[kl,km,kb].conj(), temp_t2_v_6, optimize=True)
               del temp_t2_v_6

               #temp_t2_v_7 = np.zeros_like((eris_ovvo))
               #temp_t2_v_7[kl,ka,kd] = lib.einsum('nlae,nedm->ladm',t2_1[kn,kl,ka], eris_ovvo[kn,ke,kd],optimize=True)
               temp_t2_v_7 = lib.einsum('nlae,nedm->ladm',t2_1[kn,kl,ka], eris_ovvo[kn,ke,kd],optimize=True)
               M_ab[ka] -=      lib.einsum('mlbd,ladm->ab',t2_1[km,kl,kb].conj(), temp_t2_v_7, optimize=True)
               M_ab[ka] += 2. * lib.einsum('lmbd,ladm->ab',t2_1[kl,km,kb].conj(), temp_t2_v_7, optimize=True)
               del temp_t2_v_7

               km = kn
               kl = kconserv[kn,ke,kd]
               #temp_t2_v_8 = np.zeros((nkpts,nocc,nocc),dtype=t2_1.dtype)
               #temp_t2_v_8[km] = lib.einsum('lned,mled->mn',t2_1[kl,kn,ke], t2_1[km,kl,ke].conj(),optimize=True)
               temp_t2_v_8 = lib.einsum('lned,mled->mn',t2_1[kl,kn,ke], t2_1[km,kl,ke].conj(),optimize=True)
               M_ab[ka] += 2.* lib.einsum('mn,nmab->ab',temp_t2_v_8, eris_oovv[kn,km,ka], optimize=True)
               M_ab[ka] -= lib.einsum('mn,nbam->ab', temp_t2_v_8, eris_ovvo[kn,kb,ka], optimize=True)
               del temp_t2_v_8

               #temp_t2_v_9 = np.zeros((nkpts,nocc,nocc),dtype=t2_1.dtype)
               #temp_t2_v_9[km] = lib.einsum('nled,mled->mn',t2_1[kn,kl,ke], t2_1[km,kl,ke].conj(),optimize=True)
               temp_t2_v_9 = lib.einsum('nled,mled->mn',t2_1[kn,kl,ke], t2_1[km,kl,ke].conj(),optimize=True)
               M_ab[ka] -= 4. * lib.einsum('mn,nmab->ab',temp_t2_v_9, eris_oovv[kn,km,ka], optimize=True)
               M_ab[ka] += 2. * lib.einsum('mn,nbam->ab',temp_t2_v_9, eris_ovvo[kn,kb,ka], optimize=True)
               del temp_t2_v_9

           for km, kl, kn in kpts_helper.loop_kkk(nkpts):
 
               kd = kconserv[km,kl,ka]
               ko = kconserv[ka,kd,kn]
               #temp_t2_v_10 = np.zeros_like((eris_oovv))
               #temp_t2_v_10[km,kl,ka] = lib.einsum('noad,nmol->mlad',t2_1[kn,ko,ka], eris_oooo[kn,km,ko],optimize=True)
               temp_t2_v_10 = lib.einsum('noad,nmol->mlad',t2_1[kn,ko,ka], eris_oooo[kn,km,ko],optimize=True)
               M_ab[ka] -= 1.25*lib.einsum('mlbd,mlad->ab',t2_1[km,kl,kb].conj(), temp_t2_v_10, optimize=True)
               M_ab[ka] += 0.25*lib.einsum('lmbd,mlad->ab',t2_1[kl,km,kb].conj(), temp_t2_v_10, optimize=True)
               M_ab[ka] += 0.25*lib.einsum('mldb,mlad->ab',t2_1[km,kl,kd].conj(), temp_t2_v_10, optimize=True)
               M_ab[ka] -= 0.25*lib.einsum('lmdb,mlad->ab',t2_1[kl,km,kd].conj(), temp_t2_v_10, optimize=True)
               del temp_t2_v_10

               #temp_t2_v_11 = np.zeros_like((eris_oovv))
               #temp_t2_v_11[km,kl,ka] = lib.einsum('onad,nmol->mlad',t2_1[ko,kn,ka], eris_oooo[kn,km,ko],optimize=True)
               temp_t2_v_11 = lib.einsum('onad,nmol->mlad',t2_1[ko,kn,ka], eris_oooo[kn,km,ko],optimize=True)
               M_ab[ka] += 0.25*lib.einsum('mlbd,mlad->ab',t2_1[km,kl,kb].conj(), temp_t2_v_11, optimize=True)
               M_ab[ka] -= 0.25*lib.einsum('lmbd,mlad->ab',t2_1[kl,km,kb].conj(), temp_t2_v_11, optimize=True)
               M_ab[ka] -= 0.25*lib.einsum('mldb,mlad->ab',t2_1[km,kl,kd].conj(), temp_t2_v_11, optimize=True)
               M_ab[ka] += 0.25*lib.einsum('lmdb,mlad->ab',t2_1[kl,km,kd].conj(), temp_t2_v_11, optimize=True)
               del temp_t2_v_11

           #log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

           #log.timer_debug1("Starting M_ab vvvv ADC(3) calculation")

           for km in range(nkpts):
               for kl in range(nkpts):
                   kf = kconserv[km,kl,ka]
                   #if isinstance(eris.vvvv, np.ndarray):
                   temp_t2 = adc.imds.t2_1_vvvv
                   t2_1_mla = adc.t2[0][km,kl,ka]
                   M_ab[ka] -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_mla, temp_t2[km,kl,kb].conj(), optimize=True)
                   M_ab[ka] += 0.25*lib.einsum('mlaf,lmbf->ab',t2_1_mla, temp_t2[kl,km,kb].conj(), optimize=True)
                   M_ab[ka] += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1_mla, temp_t2[km,kl,kf].conj(), optimize=True)
                   M_ab[ka] -= 0.25*lib.einsum('mlaf,lmfb->ab',t2_1_mla, temp_t2[kl,km,kf].conj(), optimize=True)
                   M_ab[ka] -=      lib.einsum('mlaf,mlbf->ab',t2_1_mla, temp_t2[km,kl,kb].conj(), optimize=True)
                   del t2_1_mla

                   t2_1_lma = adc.t2[0][kl,km,ka]
                   M_ab[ka] += 0.25*lib.einsum('lmaf,mlbf->ab',t2_1_lma, temp_t2[km,kl,kb].conj(), optimize=True)
                   M_ab[ka] -= 0.25*lib.einsum('lmaf,lmbf->ab',t2_1_lma, temp_t2[kl,km,kb].conj(), optimize=True)
                   M_ab[ka] -= 0.25*lib.einsum('lmaf,mlfb->ab',t2_1_lma, temp_t2[km,kl,kf].conj(), optimize=True)
                   M_ab[ka] += 0.25*lib.einsum('lmaf,lmfb->ab',t2_1_lma, temp_t2[kl,km,kf].conj(), optimize=True)
                   del t2_1_lma

                   kd = kconserv[km,kl,ka]
                   t2_1_mlb = adc.t2[0][km,kl,kb]
                   M_ab[ka] -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2[km,kl,ka], t2_1_mlb.conj(), optimize=True)
                   M_ab[ka] += 0.25*lib.einsum('lmad,mlbd->ab', temp_t2[kl,km,ka], t2_1_mlb.conj(), optimize=True)
                   M_ab[ka] -=      lib.einsum('mlad,mlbd->ab', temp_t2[km,kl,ka], t2_1_mlb.conj(), optimize=True)

                   M_ab[ka] += 0.25*lib.einsum('lmad,mlbd->ab', temp_t2[kl,km,ka], t2_1_mlb.conj(), optimize=True)
                   M_ab[ka] -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2[km,kl,ka], t2_1_mlb.conj(), optimize=True)
                   del t2_1_mlb

                   t2_1_lmb = adc.t2[0][kl,km,kb]
                   M_ab[ka] += 0.25*lib.einsum('mlad,lmbd->ab', temp_t2[km,kl,ka], t2_1_lmb.conj(), optimize=True)
                   M_ab[ka] -= 0.25*lib.einsum('lmad,lmbd->ab', temp_t2[kl,km,ka], t2_1_lmb.conj(), optimize=True)

                   M_ab[ka] -= 0.25*lib.einsum('lmad,lmbd->ab', temp_t2[kl,km,ka], t2_1_lmb.conj(), optimize=True)
                   M_ab[ka] += 0.25*lib.einsum('mlad,lmbd->ab', temp_t2[km,kl,ka], t2_1_lmb.conj(), optimize=True)
                   del temp_t2
                   del t2_1_lmb

                   for kd in range(nkpts):
                       kf = kconserv[km,kl,kd]
                       ke = kconserv[kb,kf,ka]
                       if isinstance(eris.vvvv, type(None)):
                           #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                           chnk_size = adc.chnk_size
                           if chnk_size > nvir:
                               chnk_size = nvir
                           a = 0
                           for p in range(0,nvir,chnk_size):
                               eris_vvvv = dfadc.get_vvvv_df(adc, eris.Lvv[ka,kb], eris.Lvv[ke,kf], p, chnk_size)/nkpts
                               k = eris_vvvv.shape[0]
                               M_ab[ka,a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1[km,kl,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1[km,kl,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1[kl,km,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1[kl,km,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1[km,kl,kf], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               del eris_vvvv

                               eris_vvvv = dfadc.get_vvvv_df(adc, eris.Lvv[ka,kf], eris.Lvv[ke,kb], p, chnk_size)/nkpts
                               M_ab[ka,a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1[km,kl,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1[km,kl,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1[kl,km,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1[kl,km,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1[km,kl,kf], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               del eris_vvvv
                               a += k
                       #else :
                       elif isinstance(eris.vvvv, np.ndarray):
                           eris_vvvv =  eris.vvvv
                           M_ab[ka] -= lib.einsum('mldf,mled,aebf->ab',t2_1[km,kl,kd], t2_1[km,kl,ke].conj(), eris_vvvv[ka,ke,kb], optimize=True)
                           M_ab[ka] += lib.einsum('mldf,lmed,aebf->ab',t2_1[km,kl,kd], t2_1[kl,km,ke].conj(), eris_vvvv[ka,ke,kb], optimize=True)
                           M_ab[ka] += lib.einsum('lmdf,mled,aebf->ab',t2_1[kl,km,kd], t2_1[km,kl,ke].conj(), eris_vvvv[ka,ke,kb], optimize=True)
                           M_ab[ka] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1[kl,km,kd], t2_1[kl,km,ke].conj(), eris_vvvv[ka,ke,kb], optimize=True)
                           M_ab[ka] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1[km,kl,kf], t2_1[km,kl,ke].conj(), eris_vvvv[ka,ke,kb], optimize=True)
                           M_ab[ka] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1[km,kl,kd], t2_1[km,kl,ke].conj(), eris_vvvv[ka,ke,kf], optimize=True)
                           M_ab[ka] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1[km,kl,kd], t2_1[kl,km,ke].conj(), eris_vvvv[ka,ke,kf], optimize=True)
                           M_ab[ka] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1[kl,km,kd], t2_1[km,kl,ke].conj(), eris_vvvv[ka,ke,kf], optimize=True)
                           M_ab[ka] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1[kl,km,kd], t2_1[kl,km,ke].conj(), eris_vvvv[ka,ke,kf], optimize=True)
                           M_ab[ka] -= lib.einsum('mlfd,mled,aefb->ab',t2_1[km,kl,kf], t2_1[km,kl,ke].conj(), eris_vvvv[ka,ke,kf], optimize=True)
                       else :
                           #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                           chnk_size = adc.chnk_size
                           if chnk_size > nvir:
                               chnk_size = nvir
                           a = 0
                           for p in range(0,nvir,chnk_size):
                               eris_vvvv = eris.vvvv[ka,ke,kb,p:p+chnk_size]
                               k = eris_vvvv.shape[0]
                               M_ab[ka,a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1[km,kl,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1[km,kl,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1[kl,km,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1[kl,km,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1[km,kl,kf], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               del eris_vvvv
                               eris_vvvv = eris.vvvv[ka,ke,kf,p:p+chnk_size]
                               M_ab[ka,a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1[km,kl,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1[km,kl,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1[kl,km,kd], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1[kl,km,kd], t2_1[kl,km,ke].conj(), eris_vvvv, optimize=True)
                               M_ab[ka,a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1[km,kl,kf], t2_1[km,kl,ke].conj(), eris_vvvv, optimize=True)
                               del eris_vvvv
                               a += k

    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
 
    return M_ab

#@profile
def get_imds_ip(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nmo =  adc.get_nmo
    nocc =  adc.nocc
    nvir = adc.nmo - adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_occ = np.identity(nocc)
    M_ij = np.empty((nkpts,nocc,nocc),dtype=mo_coeff.dtype)

#    if eris is None:
#        eris = adc.transform_integrals()
#
    # i-j block
    # Zeroth-order terms

    eris_ovov = eris.ovov
    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ , e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                t2_1 = adc.t2[0]
                t2_1_ild = adc.t2[0][ki,kl,kd]

                M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1_ild, t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -=  lib.einsum('d,ilde,ljde->ij',e_vir[kd],t2_1_ild, t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1_ild, t2_1[kj,kl,kd].conj(), optimize=True)

                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1_ild, t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ[kl],t2_1_ild, t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1_ild, t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1_ild, t2_1[kj,kl,kd].conj(), optimize=True)

                M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_ild,t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= lib.einsum('i,ij->ij',e_occ[ki],M_ij_t, optimize=True)
                M_ij[ki] -= lib.einsum('j,ij->ij',e_occ[kj],M_ij_t, optimize=True)
                del M_ij_t
                
                M_ij_t = lib.einsum('ilde,ljde->ij', t2_1_ild,t2_1[kl,kj,kd].conj(), optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t, optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('j,ij->ij',e_occ[kj],M_ij_t, optimize=True)
                del M_ij_t

                M_ij[ki] += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_ild, eris_ovov[kj,ke,kl],optimize=True)
                M_ij[ki] += lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov[kj,kd,kl],optimize=True)
                del t2_1_ild

                t2_1_lid = adc.t2[0][kl,ki,kd]
                M_ij[ki] -=  lib.einsum('d,lide,jlde->ij',e_vir[kd],t2_1_lid, t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] +=  lib.einsum('d,lide,ljde->ij',e_vir[kd],t2_1_lid, t2_1[kl,kj,kd].conj(), optimize=True)

                M_ij[ki] += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ[kl],t2_1_lid, t2_1[kj,kl,kd].conj(), optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ[kl],t2_1_lid, t2_1[kl,kj,kd].conj(), optimize=True)

                M_ij[ki] -= 0.5 *  lib.einsum('lide,jdle->ij',t2_1_lid, eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('lide,jeld->ij',t2_1_lid, eris_ovov[kj,ke,kl],optimize=True)
                del t2_1_lid

                t2_1_jld = adc.t2[0][kj,kl,kd]
                M_ij[ki] += 0.5 *  lib.einsum('jlde,idle->ij',t2_1_jld.conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1_jld.conj(), eris_ovov[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] += lib.einsum('jlde,idle->ij',t2_1_jld.conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                del t2_1_jld

                t2_1_ljd = adc.t2[0][kl,kj,kd]
                M_ij[ki] -= 0.5 *  lib.einsum('ljde,idle->ij',t2_1_ljd.conj(), eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] += 0.5 *  lib.einsum('ljde,ldie->ij',t2_1_ljd.conj(), eris_ovov[kl,kd,ki].conj(),optimize=True)
                del t2_1_ljd

                M_ij[ki] +=  lib.einsum('d,iled,jled->ij',e_vir[kd],t2_1[ki,kl,ke], t2_1[kj,kl,ke].conj(), optimize=True)
                del t2_1    

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    if (method == "adc(3)"):
        t1_2 = adc.t1[0]
        eris_ovoo = eris.ovoo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv
        eris_oooo = eris.oooo

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
                    t2_1 = adc.t2[0]
                    t2_2_ild = adc.t2[1][ki,kl,kd]

                    M_ij[ki] += 0.5* lib.einsum('ilde,jdle->ij',t2_2_ild, eris_ovov[kj,kd,kl],optimize=True)
                    M_ij[ki] -= 0.5* lib.einsum('ilde,jeld->ij',t2_2_ild, eris_ovov[kj,ke,kl],optimize=True)
                    M_ij[ki] += lib.einsum('ilde,jdle->ij',t2_2_ild, eris_ovov[kj,kd,kl],optimize=True)

                    M_ij[ki] +=  lib.einsum('d,jlde,ilde->ij',e_vir[kd],t2_1[kj,kl,kd].conj(), t2_2_ild,optimize=True)
                    M_ij[ki] -=  lib.einsum('d,ljde,ilde->ij',e_vir[kd],t2_1[kl,kj,kd].conj(), t2_2_ild,optimize=True)
                    M_ij[ki] +=  lib.einsum('d,jlde,ilde->ij',e_vir[kd],t2_1[kj,kl,kd].conj(), t2_2_ild,optimize=True)

                    M_ij[ki] -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2_ild,optimize=True)
                    M_ij[ki] += 0.5 *  lib.einsum('l,ljde,ilde->ij',e_occ[kl],t2_1[kl,kj,kd].conj(), t2_2_ild,optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2_ild,optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2_ild,optimize=True)
                    del t2_2_ild

                    t2_2 = adc.t2[1]
                    M_ij[ki] +=  lib.einsum('d,jled,iled->ij',e_vir[kd],t2_1[kj,kl,ke].conj(), t2_2[ki,kl,ke],optimize=True)
                    M_ij[ki] +=  lib.einsum('d,iled,jled->ij',e_vir[kd],t2_1[ki,kl,ke], t2_2[kj,kl,ke].conj(),optimize=True)
                    del t2_2

                    t2_2_lid = adc.t2[1][kl,ki,kd]

                    M_ij[ki] -= 0.5* lib.einsum('lide,jdle->ij',t2_2_lid, eris_ovov[kj,kd,kl],optimize=True)
                    M_ij[ki] += 0.5* lib.einsum('lide,jeld->ij',t2_2_lid, eris_ovov[kj,ke,kl],optimize=True)

                    M_ij[ki] -=  lib.einsum('d,jlde,lide->ij',e_vir[kd],t2_1[kj,kl,kd].conj(), t2_2_lid,optimize=True)
                    M_ij[ki] +=  lib.einsum('d,ljde,lide->ij',e_vir[kd],t2_1[kl,kj,kd].conj(), t2_2_lid,optimize=True)

                    M_ij[ki] += 0.5 *  lib.einsum('l,jlde,lide->ij',e_occ[kl],t2_1[kj,kl,kd].conj(), t2_2_lid,optimize=True)
                    M_ij[ki] -= 0.5 *  lib.einsum('l,ljde,lide->ij',e_occ[kl],t2_1[kl,kj,kd].conj(), t2_2_lid,optimize=True)
                    del t2_2_lid
                        
                    t2_2_jld = adc.t2[1][kj,kl,kd]
                    M_ij[ki] += 0.5* lib.einsum('jlde,leid->ij',t2_2_jld.conj(), eris_ovov[kl,ke,ki].conj(),optimize=True)
                    M_ij[ki] -= 0.5* lib.einsum('jlde,ield->ij',t2_2_jld.conj(), eris_ovov[ki,ke,kl].conj(),optimize=True)
                    M_ij[ki] +=      lib.einsum('jlde,leid->ij',t2_2_jld.conj(), eris_ovov[kl,ke,ki].conj(),optimize=True)

                    M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_2_jld.conj(),optimize=True)
                    M_ij[ki] -=  lib.einsum('d,lide,jlde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_2_jld.conj(),optimize=True)
                    M_ij[ki] +=  lib.einsum('d,ilde,jlde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_2_jld.conj(),optimize=True)

                    M_ij[ki] -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2_jld.conj(),optimize=True)
                    M_ij[ki] += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_2_jld.conj(),optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2_jld.conj(),optimize=True)
                    M_ij[ki] -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2_jld.conj(),optimize=True)

                    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1[ki,kl,kd],t2_2_jld.conj(), optimize=True)
                    del t2_2_jld
                    M_ij[ki] -= 1. * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t, optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('i,ji->ij',e_occ[ki], M_ij_t, optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('j,ij->ij',e_occ[kj], M_ij_t, optimize=True)
                    M_ij[ki] -= 1. * lib.einsum('j,ji->ij',e_occ[kj], M_ij_t, optimize=True)
                    del M_ij_t

                    t2_2_ljd = adc.t2[1][kl,kj,kd]
                    M_ij[ki] -= 0.5* lib.einsum('ljde,leid->ij',t2_2_ljd.conj(), eris_ovov[kl,ke,ki].conj(),optimize=True)
                    M_ij[ki] += 0.5* lib.einsum('ljde,ield->ij',t2_2_ljd.conj(), eris_ovov[ki,ke,kl].conj(),optimize=True)

                    M_ij[ki] -=  lib.einsum('d,ilde,ljde->ij',e_vir[kd],t2_1[ki,kl,kd], t2_2_ljd.conj(),optimize=True)
                    M_ij[ki] +=  lib.einsum('d,lide,ljde->ij',e_vir[kd],t2_1[kl,ki,kd], t2_2_ljd.conj(),optimize=True)

                    M_ij[ki] += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ[kl],t2_1[ki,kl,kd], t2_2_ljd.conj(),optimize=True)
                    M_ij[ki] -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ[kl],t2_1[kl,ki,kd], t2_2_ljd.conj(),optimize=True)

                    M_ij_t_1 = lib.einsum('ilde,ljde->ij', t2_1[ki,kl,kd],t2_2_ljd.conj(), optimize=True)
                    del t2_2_ljd
                    M_ij[ki] += 0.5 * lib.einsum('i,ij->ij',e_occ[ki], M_ij_t_1, optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('i,ji->ij',e_occ[ki], M_ij_t_1, optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('j,ij->ij',e_occ[kj], M_ij_t_1, optimize=True)
                    M_ij[ki] += 0.5 * lib.einsum('j,ji->ij',e_occ[kj], M_ij_t_1, optimize=True)
                    del M_ij_t_1

                    temp_t2_vvvv = adc.imds.t2_1_vvvv
                    t2_1_ild = adc.t2[0][ki,kl,kd]
                    M_ij[ki] += 0.25*lib.einsum('ilde,jlde->ij',t2_1_ild, temp_t2_vvvv[kj,kl,kd].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('ilde,ljde->ij',t2_1_ild, temp_t2_vvvv[kl,kj,kd].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('ilde,jled->ij',t2_1_ild, temp_t2_vvvv[kj,kl,ke].conj(), optimize = True)
                    M_ij[ki] += 0.25*lib.einsum('ilde,ljed->ij',t2_1_ild, temp_t2_vvvv[kl,kj,ke].conj(), optimize = True)
                    M_ij[ki] +=      lib.einsum('ilde,jlde->ij',t2_1_ild, temp_t2_vvvv[kj,kl,kd].conj(), optimize = True)
                    del t2_1_ild

                    t2_1_lid = adc.t2[0][kl,ki,kd]
                    M_ij[ki] -= 0.25*lib.einsum('lide,jlde->ij',t2_1_lid, temp_t2_vvvv[kj,kl,kd].conj(), optimize = True)
                    M_ij[ki] += 0.25*lib.einsum('lide,ljde->ij',t2_1_lid, temp_t2_vvvv[kl,kj,kd].conj(), optimize = True)
                    M_ij[ki] += 0.25*lib.einsum('lide,jled->ij',t2_1_lid, temp_t2_vvvv[kj,kl,ke].conj(), optimize = True)
                    M_ij[ki] -= 0.25*lib.einsum('lide,ljed->ij',t2_1_lid, temp_t2_vvvv[kl,kj,ke].conj(), optimize = True)
                    del t2_1_lid
                    del temp_t2_vvvv

                    log.timer_debug1("Starting the small integrals  calculation")

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
         
                kf = kconserv[km,kj,ke]
                kl = kconserv[kf,kj,kd]
                #temp_t2_v_1 = np.zeros_like((eris_ovov))
                #temp_t2_v_1[km,ke,kj] += lib.einsum('lmde,jldf->mejf',t2_1[kl,km,kd], t2_1[kj,kl,kd].conj(),optimize=True)
                temp_t2_v_1 = lib.einsum('lmde,jldf->mejf',t2_1[kl,km,kd], t2_1[kj,kl,kd].conj(),optimize=True)
                M_ij[ki] -=  2 * lib.einsum('mejf,mefi->ij',temp_t2_v_1.conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=      lib.einsum('mejf,mife->ij',temp_t2_v_1.conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('meif,mefj->ij',temp_t2_v_1.conj(), eris_ovvo[km,ke,kf].conj() ,optimize = True)
                M_ij[ki] +=      lib.einsum('meif,mjfe->ij',temp_t2_v_1.conj(), eris_oovv[km,kj,kf].conj() ,optimize = True)
                del temp_t2_v_1        

                #temp_t2_v_new = np.zeros_like((eris_ovov))
                #temp_t2_v_new[km,ke,kj] += lib.einsum('mlde,ljdf->mejf',t2_1[km,kl,kd], t2_1[kl,kj,kd].conj(),optimize=True)
                temp_t2_v_new = lib.einsum('mlde,ljdf->mejf',t2_1[km,kl,kd], t2_1[kl,kj,kd].conj(),optimize=True)
                M_ij[ki] -=  2 * lib.einsum('mejf,mefi->ij',temp_t2_v_new.conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=      lib.einsum('mejf,mife->ij',temp_t2_v_new.conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('meif,mefj->ij',temp_t2_v_new.conj(), eris_ovvo[km,ke,kf].conj() ,optimize = True)
                M_ij[ki] +=      lib.einsum('meif,mjfe->ij',temp_t2_v_new.conj(), eris_oovv[km,kj,kf].conj() ,optimize = True)
                del temp_t2_v_new       


                kf = kconserv[km,kj,ke]
                #temp_t2_v_2 = np.zeros_like((eris_ovov))
                #temp_t2_v_2[km,ke,kj] += lib.einsum('lmde,ljdf->mejf',t2_1[kl,km,kd], t2_1[kl,kj,kd].conj(),optimize=True)
                temp_t2_v_2 = lib.einsum('lmde,ljdf->mejf',t2_1[kl,km,kd], t2_1[kl,kj,kd].conj(),optimize=True)
                M_ij[ki] +=  4 * lib.einsum('mejf,mefi->ij',temp_t2_v_2.conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=  4 * lib.einsum('meif,mefj->ij',temp_t2_v_2.conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('meif,mjfe->ij',temp_t2_v_2.conj(), eris_oovv[km,kj,kf].conj(),optimize = True)
                M_ij[ki] -=  2 * lib.einsum('mejf,mife->ij',temp_t2_v_2.conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                del temp_t2_v_2        

                #temp_t2_v_3 = np.zeros_like((eris_ovov))
                #temp_t2_v_3[km,ke,kj] += lib.einsum('mlde,jldf->mejf',t2_1[km,kl,kd], t2_1[kj,kl,kd].conj(),optimize=True)
                temp_t2_v_3 = lib.einsum('mlde,jldf->mejf',t2_1[km,kl,kd], t2_1[kj,kl,kd].conj(),optimize=True)
                M_ij[ki] +=     lib.einsum('mejf,mefi->ij',temp_t2_v_3.conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] +=     lib.einsum('meif,mefj->ij',temp_t2_v_3.conj(), eris_ovvo[km,ke,kf].conj(),optimize = True)
                M_ij[ki] -= 2 * lib.einsum('meif,mjfe->ij',temp_t2_v_3.conj(), eris_oovv[km,kj,kf].conj(),optimize = True)
                M_ij[ki] -= 2 * lib.einsum('mejf,mife->ij',temp_t2_v_3.conj(), eris_oovv[km,ki,kf].conj(),optimize = True)
                del temp_t2_v_3        


                kl = kconserv[kd,ke,ki]
                kf = kconserv[ki,kd,km]
                #temp_t2_v_4 = np.zeros_like((eris_ovov))
                #temp_t2_v_4[ki,kd,km] += lib.einsum('ilde,lmfe->idmf',t2_1[ki,kl,kd].conj(), eris_oovv[kl,km,kf].conj(),optimize=True)
                temp_t2_v_4 = lib.einsum('ilde,lmfe->idmf',t2_1[ki,kl,kd].conj(), eris_oovv[kl,km,kf].conj(),optimize=True)
                M_ij[ki] -= 2 * lib.einsum('idmf,jmdf->ij',temp_t2_v_4.conj(), t2_1[kj,km,kd].conj(), optimize = True)
                M_ij[ki] +=     lib.einsum('idmf,mjdf->ij',temp_t2_v_4.conj(), t2_1[km,kj,kd].conj(), optimize = True)
                del temp_t2_v_4


                #temp_t2_v_5 = np.zeros_like((eris_ovov))
                #temp_t2_v_5[ki,kd,km] += lib.einsum('lide,lmfe->idmf',t2_1[kl,ki,kd].conj(), eris_oovv[kl,km,kf].conj(),optimize=True)
                temp_t2_v_5 = lib.einsum('lide,lmfe->idmf',t2_1[kl,ki,kd].conj(), eris_oovv[kl,km,kf].conj(),optimize=True)
                M_ij[ki] +=     lib.einsum('idmf,jmdf->ij',temp_t2_v_5.conj(), t2_1[kj,km,kd].conj(), optimize = True)
                M_ij[ki] -= 2 * lib.einsum('idmf,mjdf->ij',temp_t2_v_5.conj(), t2_1[km,kj,kd].conj(), optimize = True)
                del temp_t2_v_5

                #temp_t2_v_6 = np.zeros_like((eris_ovvo))
                #temp_t2_v_6[ki,kd,kf] += lib.einsum('ilde,lefm->idfm',t2_1[ki,kl,kd].conj(), eris_ovvo[kl,ke,kf].conj(),optimize=True)
                temp_t2_v_6 = lib.einsum('ilde,lefm->idfm',t2_1[ki,kl,kd].conj(), eris_ovvo[kl,ke,kf].conj(),optimize=True)
                M_ij[ki] += 4 * lib.einsum('idfm,jmdf->ij',temp_t2_v_6.conj(), t2_1[kj,km,kd].conj(),optimize = True)
                M_ij[ki] -= 2 * lib.einsum('idfm,mjdf->ij',temp_t2_v_6.conj(), t2_1[km,kj,kd].conj(),optimize = True)
                del temp_t2_v_6

                #temp_t2_v_7 = np.zeros_like((eris_ovvo))
                #temp_t2_v_7[ki,kd,kf] = lib.einsum('lide,lefm->idfm',t2_1[kl,ki,kd].conj(), eris_ovvo[kl,ke,kf].conj(),optimize=True)
                temp_t2_v_7 = lib.einsum('lide,lefm->idfm',t2_1[kl,ki,kd].conj(), eris_ovvo[kl,ke,kf].conj(),optimize=True)
                M_ij[ki] -= 2 * lib.einsum('idfm,jmdf->ij',temp_t2_v_7.conj(), t2_1[kj,km,kd].conj(),optimize = True)
                M_ij[ki] +=     lib.einsum('idfm,mjdf->ij',temp_t2_v_7.conj(), t2_1[km,kj,kd].conj(),optimize = True)
                del temp_t2_v_7

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
            
                kf = ke
                kl = kconserv[km,kd,kf]
                #temp_t2_v_8 = np.zeros((nkpts,nvir,nvir),dtype=t2_1.dtype)
                #temp_t2_v_8[kf] += lib.einsum('lmdf,lmde->fe',t2_1[kl,km,kd], t2_1[kl,km,kd].conj(),optimize=True)
                temp_t2_v_8 = lib.einsum('lmdf,lmde->fe',t2_1[kl,km,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] += 3.0 * lib.einsum('fe,jief->ij',temp_t2_v_8, eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] -= 1.5 * lib.einsum('fe,jfei->ij',temp_t2_v_8, eris_ovvo[kj,kf,ke], optimize = True)
                M_ij[ki] +=       lib.einsum('ef,jief->ij',temp_t2_v_8.T, eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] -= 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_8.T, eris_ovvo[kj,kf,ke], optimize = True)
                del temp_t2_v_8

                #temp_t2_v_9 = np.zeros((nkpts,nvir,nvir),dtype=t2_1.dtype)
                #temp_t2_v_9[kf] += lib.einsum('lmdf,mlde->fe',t2_1[kl,km,kd], t2_1[km,kl,kd].conj(),optimize=True)
                temp_t2_v_9 = lib.einsum('lmdf,mlde->fe',t2_1[kl,km,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('fe,jief->ij',temp_t2_v_9, eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] -= 1.0 * lib.einsum('ef,jief->ij',temp_t2_v_9.T, eris_oovv[kj,ki,ke], optimize = True)
                M_ij[ki] += 0.5 * lib.einsum('fe,jfei->ij',temp_t2_v_9, eris_ovvo[kj,kf,ke], optimize = True)
                M_ij[ki] += 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_9.T, eris_ovvo[kj,kf,ke], optimize = True)
                del temp_t2_v_9

                kn = km
                kl = kconserv[kn,kd,ke]
                #temp_t2_v_10 = np.zeros((nkpts,nocc,nocc),dtype=t2_1.dtype)
                #temp_t2_v_10[kn] += lib.einsum('lnde,lmde->nm',t2_1[kl,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                temp_t2_v_10 = lib.einsum('lnde,lmde->nm',t2_1[kl,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] -= 3.0 * lib.einsum('nm,jinm->ij',temp_t2_v_10, eris_oooo[kj,ki,kn], optimize = True)
                M_ij[ki] -= 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_10.T, eris_oooo[kj,ki,kn], optimize = True)
                M_ij[ki] += 1.5 * lib.einsum('nm,jmni->ij',temp_t2_v_10, eris_oooo[kj,km,kn], optimize = True)
                M_ij[ki] += 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_10.T, eris_oooo[kj,km,kn], optimize = True)
                del temp_t2_v_10

                #temp_t2_v_11 = np.zeros((nkpts,nocc,nocc),dtype=t2_1.dtype)
                #temp_t2_v_11[kn] += lib.einsum('lnde,mlde->nm',t2_1[kl,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                temp_t2_v_11 = lib.einsum('lnde,mlde->nm',t2_1[kl,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] += 1.0 * lib.einsum('nm,jinm->ij',temp_t2_v_11, eris_oooo[kj,ki,kn], optimize = True)
                M_ij[ki] -= 0.5 * lib.einsum('nm,jmni->ij',temp_t2_v_11, eris_oooo[kj,km,kn], optimize = True)
                M_ij[ki] -= 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_11.T, eris_oooo[kj,km,kn], optimize = True)
                M_ij[ki] += 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_11.T, eris_oooo[kj,ki,kn], optimize = True)
                del temp_t2_v_11

            for km, kn, kd in kpts_helper.loop_kkk(nkpts):
                kl = kconserv[km,ki,kn]
                t2_1 = adc.t2[0]
                #temp_t2_v_12 = np.zeros_like((eris_oooo))
                #temp_t2_v_12[ki,kn,kl] += lib.einsum('inde,lmde->inlm',t2_1[ki,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                temp_t2_v_12 = lib.einsum('inde,lmde->inlm',t2_1[ki,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                kl = kconserv[km,ki,kn]
                M_ij[ki] += 1.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12, eris_oooo[kj,kl,kn], optimize = True)
                M_ij[ki] -= 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12, eris_oooo[kj,km,kn], optimize = True)
 
                M_ij[kj] += 0.25 * lib.einsum('inlm,jlnm->ji',temp_t2_v_12.conj(), eris_oooo[kj,kl,kn].conj(), optimize = True)
                M_ij[kj] -= 0.25 * lib.einsum('inlm,lnmj->ji',temp_t2_v_12, eris_oooo[kl,kn,km].conj(), optimize = True)
                M_ij[kj] += 1.00 * lib.einsum('inlm,ljmn->ji',temp_t2_v_12, eris_oooo[kl,kj,km].conj(), optimize = True)
                del temp_t2_v_12

                #temp_t2_v_12_1 = np.zeros_like((eris_oooo))
                #temp_t2_v_12_1[ki,kn,kl] += lib.einsum('nide,mlde->inlm',t2_1[kn,ki,kd], t2_1[km,kl,kd].conj(),optimize=True)
                temp_t2_v_12_1 = lib.einsum('nide,mlde->inlm',t2_1[kn,ki,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] += 0.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12_1.conj(), eris_oooo[kj,kl,kn].conj(), optimize = True)
                M_ij[ki] -= 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12_1.conj(), eris_oooo[kj,km,kn].conj(), optimize = True)
                M_ij[kj] -= 0.25 * lib.einsum('inlm,lnmj->ji',temp_t2_v_12_1.conj(), eris_oooo[kl,kn,km], optimize = True)
                M_ij[kj] += 0.25 * lib.einsum('inlm,ljmn->ji',temp_t2_v_12_1.conj(), eris_oooo[kl,kj,km], optimize = True)

                #temp_t2_v_13 = np.zeros_like((eris_oooo))
                #temp_t2_v_13[ki,kn,km] += lib.einsum('inde,mlde->inml',t2_1[ki,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                temp_t2_v_13 = lib.einsum('inde,mlde->inml',t2_1[ki,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] -= 0.25 * lib.einsum('inml,jlnm->ij',temp_t2_v_13, eris_oooo[kj,kl,kn], optimize = True)
                M_ij[ki] += 0.25 * lib.einsum('inml,jmnl->ij',temp_t2_v_13, eris_oooo[kj,km,kn], optimize = True)

                M_ij[kj] -= 0.25 * lib.einsum('inml,jlnm->ji',temp_t2_v_13, eris_oooo[kj,kl,kn], optimize = True)
                M_ij[kj] += 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13, eris_oooo[kl,kn,km].conj(), optimize = True)

                M_ij[kj] -= 0.25 * lib.einsum('inml,ljmn->ji',temp_t2_v_13, eris_oooo[kl,kj,km].conj(), optimize = True)
                M_ij[kj] += 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13, eris_oooo[kl,kn,km].conj(), optimize = True)
                del temp_t2_v_13

                #temp_t2_v_13_1 = np.zeros_like((eris_oooo))
                #temp_t2_v_13_1[ki,kn,kl] += lib.einsum('nide,lmde->inlm',t2_1[kn,ki,kd], t2_1[kl,km,kd].conj(),optimize=True)
                temp_t2_v_13_1 = lib.einsum('nide,lmde->inlm',t2_1[kn,ki,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] -= 0.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_13_1.conj(), eris_oooo[kj,kl,kn].conj(), optimize = True)
                M_ij[ki] += 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_13_1.conj(), eris_oooo[kj,km,kn].conj(), optimize = True)
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
    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    diag = np.zeros((dim), dtype=np.complex)
    doubles = np.zeros((nkpts,nkpts,nocc*nvir*nvir),dtype=np.complex)

    # Compute precond in h1-h1 block
    M_ab_diag = np.diagonal(M_ab[kshift])
    diag[s1:f1] = M_ab_diag.copy()

    # Compute precond in 2p1h-2p1h block
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift,ka,kj]
            d_ab = e_vir[ka][:,None] + e_vir[kb]
            d_i = -e_occ[kj][:,None]
            D_n = -d_i + d_ab.reshape(-1)
            doubles[kj,ka] += D_n.reshape(-1)

    diag[s2:f2] = doubles.reshape(-1)
    log.timer_debug1("Completed ea_diag calculation")

    return diag

def ip_adc_diag(adc,kshift,M_ij=None,eris=None):
   
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ij is None:
        M_ij = adc.get_imds()

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
    nocc = adc.nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    diag = np.zeros((dim), dtype=np.complex)
    doubles = np.zeros((nkpts,nkpts,nvir*nocc*nocc),dtype=np.complex)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij[kshift])
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    #diag[s2:f2] += e_ija[kshift].reshape(-1) 
    for ka in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[kshift,ki,ka]
            d_ij = e_occ[ki][:,None] + e_occ[kj]
            d_a = e_vir[ka][:,None]
            D_n = -d_a + d_ij.reshape(-1)
            doubles[ka,ki] += D_n.reshape(-1)

    diag[s2:f2] = doubles.reshape(-1)

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

    return diag

def ea_contract_r_vvvv(adc,r2,vvvv,ka,kb,kc):

    log = logger.Logger(adc.stdout, adc.verbose)
    #nocc = myadc.nocc
    #nmo = myadc.nmo
    #nvir = nmo - nocc
    nocc = r2.shape[0]
    nvir = r2.shape[1]
    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    kd = kconserv[ka, kc, kb]
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))
    r2_vvvv = np.zeros((nvir,nvir,nocc),dtype=r2.dtype)
    #chnk_size = kadc_ao2mo.calculate_chunk_size(myadc)
    chnk_size = adc.chnk_size
    if chnk_size > nvir:
       chnk_size = nvir

    a = 0
    if isinstance(vvvv, np.ndarray):
        vv1 = vvvv[ka,kc] 
        vv2 = vvvv[kb,kd] 
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(adc, vv1, vv2, p, chnk_size)/nkpts
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[a:a+k] += np.dot(vvvv_p,r2.T).reshape(-1,nvir,nocc)
            del vvvv_p
            a += k
    else :
        for p in range(0,nvir,chnk_size):
            vvvv_p = vvvv[ka,kb,kc][p:p+chnk_size].reshape(-1,nvir*nvir)
            k = vvvv_p.shape[0]
            r2_vvvv[a:a+k] += np.dot(vvvv_p,r2.T).reshape(-1,nvir,nocc)
            del vvvv_p
            a += k

    r2_vvvv = np.ascontiguousarray(r2_vvvv.transpose(2,0,1))

    return r2_vvvv


def ea_adc_matvec(adc, kshift, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
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
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    if M_ab is None:
        M_ab = adc.get_imds()

    #Calculate sigma vector
    #@profile
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim), dtype=np.complex)

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nkpts,nkpts,nocc,nvir,nvir)
        temp_doubles = np.zeros((nkpts,nkpts,nocc,nvir,nvir), dtype=np.complex)

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ab,b->a',M_ab[kshift],r1)

########### ADC(2) coupling blocks #########################

        for kb in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kb,kshift, kc]
                if isinstance(eris.ovvv, type(None)):
                    chnk_size = adc.chnk_size
                    if chnk_size > nocc:
                        chnk_size = nocc
                    a = 0
                    for p in range(0,nocc,chnk_size):
                        eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[ki,kc], eris.Lvv[kshift,kb], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                        k = eris_ovvv.shape[0]
                        s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv, r2[ki,kb,a:a+k], optimize = True)
                        temp_doubles[ki,kb,a:a+k] += lib.einsum('icab,a->ibc', eris_ovvv.conj(), r1, optimize = True)
                        del eris_ovvv

                        eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[ki,kb], eris.Lvv[kshift,kc], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                        s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv, r2[ki,kb,a:a+k], optimize = True)
                        del eris_ovvv
                        a += k
                else :
                    eris_ovvv = eris.ovvv[:]
                    s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv[ki,kc,kshift], r2[ki,kb], optimize = True)
                    temp_doubles[ki,kb] += lib.einsum('icab,a->ibc', eris_ovvv[ki,kc,kshift].conj(), r1, optimize = True)
                    del eris_ovvv


########    ########## ADC(2) ajk - bil block ############################

                temp_doubles[ki, kb] -= lib.einsum('i,ibc->ibc', e_occ[ki], r2[ki, kb])
                temp_doubles[ki, kb] += lib.einsum('b,ibc->ibc', e_vir[kb], r2[ki, kb])
                temp_doubles[ki, kb] += lib.einsum('c,ibc->ibc', e_vir[kc], r2[ki, kb])

        s[s2:f2] += temp_doubles.reshape(-1)
        del temp_doubles

################ ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo

               temp_doubles = np.zeros_like((r2),dtype=np.complex)

               for kx in range(nkpts):
                   for ky in range(nkpts):
                       ki = kconserv[ky,kshift, kx]
                       kj = kconserv[kx, ky, ki]
                       for kv in range(nkpts):
                           kw = kconserv[kv, kx, ky]

                           if isinstance(eris.vvvv, np.ndarray):
                               eris_vvvv = eris.vvvv.reshape(nkpts,nkpts,nkpts,nvir*nvir,nvir*nvir)
                               r2_1 = r2.reshape(nkpts,nkpts,nocc,nvir*nvir)
                               temp_doubles[ki, kx] += np.dot(r2_1[ki,kw],eris_vvvv[kx,ky,kw].T).reshape(nocc,nvir,nvir)
                           elif isinstance(eris.vvvv, type(None)):
                               temp_doubles[ki,kx] += ea_contract_r_vvvv(adc,r2[ki,kw],eris.Lvv,kx,ky,kw)
                           else : 
                               temp_doubles[ki,kx] += ea_contract_r_vvvv(adc,r2[ki,kw],eris.vvvv,kx,ky,kw)

                       for kj in range(nkpts):
                           kz = kconserv[ky,ki,kj]
                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo[kj,kz,ky],r2[kj,kz],optimize = True)
                           temp_doubles[ki,kx] += lib.einsum('jzyi,jxz->ixy',eris_ovvo[kj,kz,ky],r2[kj,kx],optimize = True)
                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv[kj,ki,ky],r2[kj,kx],optimize = True)

                           kz = kconserv[kj,ki,kx]
                           temp_doubles[ki,kx] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_oovv[kj,ki,kx],r2[kj,kz],optimize = True)

                           kw = kconserv[kj,ki,kx]
                           temp_doubles[ki,kx] -=  0.5*lib.einsum('jixw,jwy->ixy',eris_oovv[kj,ki,kx],r2[kj,kw],optimize = True)
                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv[kj,ki,ky],r2[kj,kx],optimize = True)

                           kw = kconserv[ky,ki,kj]
                           temp_doubles[ki,kx] += lib.einsum('jwyi,jxw->ixy',eris_ovvo[kj,kw,ky],r2[kj,kx],optimize = True)
                           temp_doubles[ki,kx] -= 0.5*lib.einsum('jwyi,jwx->ixy',eris_ovvo[kj,kw,ky],r2[kj,kw],optimize = True)

               s[s2:f2] += temp_doubles.reshape(-1)
               del temp_doubles
               
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################
               
               t2_1 = adc.t2[0]

               for kl in range(nkpts):
                   for km in range(nkpts):
                          kj = kconserv[kl, kshift, km]

                          for kw in range(nkpts):
                              kz = kconserv[kw, kl, km]
                              t2_1_lmz = adc.t2[0][kl,km,kz]
                              ka = kconserv[km, kj, kl]
                              temp_1 =       lib.einsum('lmzw,jzw->jlm',t2_1_lmz.conj(),r2[kj,kz])
                              temp = 0.25 * lib.einsum('lmzw,jzw->jlm',t2_1_lmz.conj(),r2[kj,kz])
                              temp -= 0.25 * lib.einsum('lmzw,jwz->jlm',t2_1_lmz.conj(),r2[kj,kw])
                              del t2_1_lmz

                              t2_1_mlz = adc.t2[0][km,kl,kz]
                              temp -= 0.25 * lib.einsum('mlzw,jzw->jlm',t2_1_mlz.conj(),r2[kj,kz])
                              temp += 0.25 * lib.einsum('mlzw,jwz->jlm',t2_1_mlz.conj(),r2[kj,kw])
                              del t2_1_mlz

                              s[s1:f1] += lib.einsum('jlm,lamj->a',temp,   eris_ovoo[kl,ka,km].conj(), optimize=True)
                              s[s1:f1] -= lib.einsum('jlm,malj->a',temp,   eris_ovoo[km,ka,kl].conj(), optimize=True)
                              s[s1:f1] += lib.einsum('jlm,lamj->a',temp_1, eris_ovoo[kl,ka,km].conj(), optimize=True)

                              del temp
                              del temp_1

               temp_doubles = np.zeros_like((r2))
               for kx in range(nkpts):
                   for ky in range(nkpts):
                          ki = kconserv[ky, kshift, kx]
                          for kl in range(nkpts):
                              km = kconserv[ki, kshift, kl]
                              kb = kconserv[km,ki,kl]
                              temp = lib.einsum('b,lbmi->lmi',r1,eris_ovoo[kl,kb,km], optimize=True)
                              km = kconserv[kx,ky,kl]
                              temp_doubles[ki,kx] += lib.einsum('lmi,lmxy->ixy',temp, t2_1[kl,km,kx], optimize=True)

                              del temp
               s[s2:f2] += temp_doubles.reshape(-1) 
               del temp_doubles
 
               for kd in range(nkpts):
                   for kz in range(nkpts):
                          kl = kconserv[kd, kshift, kz]
                          for kj in range(nkpts):
                              kw = kconserv[kd, kl, kj]
                              t2_1_jlw = adc.t2[0][kj,kl,kw]

                              temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1_jlw.conj(),r2[kj,kz],optimize=True)
                              temp_s_a -= lib.einsum('jlwd,jwz->lzd',t2_1_jlw.conj(),r2[kj,kw],optimize=True)

                              temp_t2_r2_1 = lib.einsum('jlwd,jzw->lzd',t2_1_jlw.conj(),r2[kj,kz],optimize=True)
                              temp_t2_r2_1 -= lib.einsum('jlwd,jwz->lzd',t2_1_jlw.conj(),r2[kj,kw],optimize=True)
                              temp_t2_r2_1 += lib.einsum('jlwd,jzw->lzd',t2_1_jlw.conj(),r2[kj,kz],optimize=True)
                              del t2_1_jlw

                              t2_1_ljw = adc.t2[0][kl,kj,kw]
                              temp_s_a -= lib.einsum('ljwd,jzw->lzd',t2_1_ljw.conj(),r2[kj,kz],optimize=True)
                              temp_s_a += lib.einsum('ljwd,jwz->lzd',t2_1_ljw.conj(),r2[kj,kw],optimize=True)
                              temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1[kl,kj,kd].conj(),r2[kj,kz],optimize=True)

                              temp_t2_r2_1 -= lib.einsum('ljwd,jzw->lzd',t2_1_ljw.conj(),r2[kj,kz],optimize=True)
                              del t2_1_ljw
                              temp_t2_r2_4 = lib.einsum('jldw,jwz->lzd',t2_1[kj,kl,kd].conj(),r2[kj,kw], optimize=True)

                              #if getattr(eris, 'Lov', None) is not None:
                              if isinstance(eris.ovvv, type(None)):
                                  #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                                  chnk_size = adc.chnk_size
                                  if chnk_size > nocc:
                                      chnk_size = nocc
                                  a = 0
                                  for p in range(0,nocc,chnk_size):

                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kd], eris.Lvv[kz,ka], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      k = eris_ovvv.shape[0]

                                      s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a[a:a+k],eris_ovvv.conj(),optimize=True)
                                      s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_t2_r2_1[a:a+k],eris_ovvv.conj(),optimize=True)
                                      del eris_ovvv
                                      
                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,ka], eris.Lvv[kz,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a[a:a+k],eris_ovvv.conj(),optimize=True)
                                      s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_t2_r2_4[a:a+k],eris_ovvv.conj(),optimize=True)
                                      del eris_ovvv
                                      a += k

                              else :
                                  eris_ovvv = eris.ovvv[:]
                                  s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a,eris_ovvv[kl,kd,kz].conj(),optimize=True)
                                  s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_t2_r2_1,eris_ovvv[kl,kd,kz].conj(),optimize=True)

                                  s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a,eris_ovvv[kl,ka,kz].conj(),optimize=True)
                                  s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_t2_r2_4,eris_ovvv[kl,ka,kz].conj(),optimize=True)
                                  del eris_ovvv
                              del temp_s_a
                              del temp_t2_r2_1
                              del temp_t2_r2_4

                   for kw in range(nkpts):
                          kl = kconserv[kd, kshift, kw]
                          for kj in range(nkpts):
                              kz = kconserv[kd, kl, kj]
                              t2_1_jlz = adc.t2[0][kj,kl,kz]

                              temp_s_a_1  = -lib.einsum('jlzd,jwz->lwd',t2_1_jlz.conj(),r2[kj,kw],optimize=True)
                              temp_s_a_1 += lib.einsum('jlzd,jzw->lwd',t2_1_jlz.conj(),r2[kj,kz],optimize=True)
                              temp_t2_r2_2  = -lib.einsum('jlzd,jwz->lwd',t2_1_jlz.conj(),r2[kj,kw],optimize=True)
                              temp_t2_r2_2 += lib.einsum('jlzd,jzw->lwd',t2_1_jlz.conj(),r2[kj,kz],optimize=True)
                              temp_t2_r2_2 -= lib.einsum('jlzd,jwz->lwd',t2_1_jlz.conj(),r2[kj,kw],optimize=True)
                              del t2_1_jlz

                              t2_1_ljz = adc.t2[0][kl,kj,kz]
                              temp_s_a_1 += lib.einsum('ljzd,jwz->lwd',t2_1_ljz.conj(),r2[kj,kw],optimize=True)
                              temp_s_a_1 -= lib.einsum('ljzd,jzw->lwd',t2_1_ljz.conj(),r2[kj,kz],optimize=True)
                              temp_s_a_1 -= lib.einsum('ljdz,jwz->lwd',t2_1[kl,kj,kd].conj(),r2[kj,kw],optimize=True)

                              temp_t2_r2_2 += lib.einsum('ljzd,jwz->lwd',t2_1_ljz.conj(),r2[kj,kw],optimize=True)

                              temp_t2_r2_3 = -lib.einsum('ljzd,jzw->lwd',t2_1_ljz.conj(),r2[kj,kz],optimize=True)
                              del t2_1_ljz

                              #if getattr(eris, 'Lov', None) is not None:
                              if isinstance(eris.ovvv, type(None)):
                                  #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                                  chnk_size = adc.chnk_size
                                  if chnk_size > nocc:
                                      chnk_size = nocc
                                  a = 0
                                  for p in range(0,nocc,chnk_size):

                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kd], eris.Lvv[kw,ka], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      k = eris_ovvv.shape[0]
                                      s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1[a:a+k],eris_ovvv.conj(),optimize=True)
                                      s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_t2_r2_2[a:a+k],eris_ovvv.conj(),optimize=True)
                                      del eris_ovvv

                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,ka], eris.Lvv[kw,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1[a:a+k],eris_ovvv.conj(),optimize=True)
                                      s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_t2_r2_3[a:a+k],eris_ovvv.conj(),optimize=True)
                                      del eris_ovvv
                                      a += k
                                
                              else :
                                  eris_ovvv = eris.ovvv[:]
                                  s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1,eris_ovvv[kl,kd,kw].conj(),optimize=True)
                                  s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1,eris_ovvv[kl,ka,kw].conj(),optimize=True)

                                  s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_t2_r2_2,eris_ovvv[kl,kd,kw].conj(),optimize=True)
                                  s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_t2_r2_3,eris_ovvv[kl,ka,kw].conj(),optimize=True)
                                  del eris_ovvv

                              del temp_s_a_1
                              del temp_t2_r2_2
                              del temp_t2_r2_3

               temp_doubles = np.zeros_like((r2))
               for kx in range(nkpts):
                   for ky in range(nkpts):
                          ki = kconserv[kshift,kx,ky]
                          for kl in range(nkpts):
                              temp = np.zeros((nocc,nvir,nvir),dtype=eris.oooo.dtype)
                              temp_1_1 = np.zeros((nocc,nvir,nvir),dtype=eris.oooo.dtype)
                              temp_2_1 = np.zeros((nocc,nvir,nvir),dtype=eris.oooo.dtype)
                              #if getattr(eris, 'Lov', None) is not None:
                              if isinstance(eris.ovvv, type(None)):
                                  #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                                  chnk_size = adc.chnk_size
                                  if chnk_size > nocc:
                                      chnk_size = nocc
                                  a = 0
                                  for p in range(0,nocc,chnk_size):
                                      kd = kconserv[kshift, kl, kx]
                                      kb = kconserv[kl,kd,kx]

                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kd], eris.Lvv[kx,kb], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      k = eris_ovvv.shape[0]
                                      temp_1_1[a:a+k] += lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
                                      temp_2_1[a:a+k] += lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
                                      del eris_ovvv

                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kb], eris.Lvv[kx,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      temp_1_1[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r1,optimize=True)
                                      del eris_ovvv

                                      kd = kconserv[kshift,kl,ky]
                                      kb = kconserv[ky,kd,kl]
                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[kl,kb], eris.Lvv[ky,kd], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      temp[a:a+k] -= lib.einsum('lbyd,b->lyd', eris_ovvv,r1,optimize=True)
                                      del eris_ovvv
                                      a += k
                              else :
                                  kd = kconserv[kshift, kl, kx]
                                  kb = kconserv[kl,kd,kx]
                                  eris_ovvv = eris.ovvv[:]
                                  temp_1_1 += lib.einsum('ldxb,b->lxd', eris_ovvv[kl,kd,kx],r1,optimize=True)
                                  temp_1_1 -= lib.einsum('lbxd,b->lxd', eris_ovvv[kl,kb,kx],r1,optimize=True)
                                  temp_2_1 += lib.einsum('ldxb,b->lxd', eris_ovvv[kl,kd,kx],r1,optimize=True)

                                  kd = kconserv[kshift,kl,ky]
                                  kb = kconserv[ky,kd,kl]
                                  temp -= lib.einsum('lbyd,b->lyd', eris_ovvv[kl,kb,ky],r1,optimize=True)
                                  del eris_ovvv

                              kd = kconserv[ky, kl, ki]
                              temp_doubles[ki,kx]  += lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1[kl,ki,kd],optimize=True)
                              temp_doubles[ki,kx]  += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1[ki,kl,ky],optimize=True)
                              temp_doubles[ki,kx]  -= lib.einsum('lxd,ildy->ixy',temp_2_1,t2_1[ki,kl,kd],optimize=True)

                              kd = kconserv[kl,ki,kx]
                              temp_doubles[ki,kx]  += lib.einsum('lyd,lixd->ixy',temp,t2_1[kl,ki,kx],optimize=True)
                              del temp
                              del temp_1_1
                              del temp_2_1

               s[s2:f2] += temp_doubles.reshape(-1)
               del temp_doubles
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
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    #e_ij = np.zeros((nkpts,nocc,nkpts,nocc), dtype=mo_coeff.dtype)

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)
        
    #e_ij = e_occ[:,:,None,None] + e_occ[None,None,:,:]
    #e_ija = - e_vir[:,:,None,None,None,None] + e_ij[None,None,:,:,:,:] 
    #e_ija = e_ija.transpose(0,2,4,1,3,5).copy()

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    #@profile
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim), dtype=mo_coeff.dtype)

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nkpts,nkpts,nvir,nocc,nocc)
        temp_doubles = np.zeros_like((r2),dtype=mo_coeff.dtype)
        temp_aij = np.zeros_like((r2),dtype=mo_coeff.dtype)
        
        eris_ovoo = eris.ovoo

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ij,j->i',M_ij[kshift],r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]

                s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo[kj,ka,kk].conj(), r2[ka,kj], optimize = True)
                s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo[kk,ka,kj].conj(), r2[ka,kj], optimize = True)
#################### ADC(2) ajk - i block ############################

                temp_doubles[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk], r1, optimize = True)

################# ADC(2) ajk - bil block ############################

                #temp_doubles[ka,kj] += e_ija[ka,kj,kk] *  r2[ka,kj]
                temp_doubles[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2[ka, kj])
                temp_doubles[ka, kj] += lib.einsum('j,ajk->ajk', e_occ[kj], r2[ka, kj])
                temp_doubles[ka, kj] += lib.einsum('k,ajk->ajk', e_occ[kk], r2[ka, kj])

        s[s2:f2] += temp_doubles.reshape(-1)
        del temp_doubles

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):
        
               eris_oooo = eris.oooo
               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo
               eris_ovov = eris.ovov

               temp_doubles = np.zeros_like((r2),dtype=np.complex)
               for kj in range(nkpts):
                   for kk in range(nkpts):
                          ka = kconserv[kk, kshift, kj]
                          for kl in range(nkpts):
                              ki = kconserv[kj, kl, kk]

                              temp_doubles[ka,kj] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[kk,ki,kj], r2[ka,kl], optimize = True)
                              temp_doubles[ka,kj] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo[kk,kl,kj],r2[ka,ki], optimize = True)
                          
                          for kl in range(nkpts):
                              kb = kconserv[ka, kk, kl]
                              temp_doubles[ka,kj] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[kk,kl,kb],r2[kb,kj],optimize = True)

                              kb = kconserv[kl, kj, ka]
                              temp_doubles[ka,kj] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[kj,ka,kb],r2[kb,kk],optimize = True)
                              temp_doubles[ka,kj] -=  lib.einsum('jabl,blk->ajk',eris_ovvo[kj,ka,kb],r2[kb,kl],optimize = True)
                              kb = kconserv[ka, kj, kl]
                              temp_doubles[ka,kj] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv[kj,kl,kb],r2[kb,kl],optimize = True)

                          for ki in range(nkpts):
                               kb = kconserv[ka, kk, ki]
                               temp_doubles[ka,kj] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[kk,ki,kb],r2[kb,kj],optimize = True)

                               kb = kconserv[ka, kj, ki]
                               temp_doubles[ka,kj] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[kj,ki,kb],r2[kb,ki],optimize = True)
                               temp_doubles[ka,kj] -= lib.einsum('jabi,bik->ajk',eris_ovvo[kj,ka,kb],r2[kb,ki],optimize = True)
                               kb = kconserv[ki, kj, ka]
                               temp_doubles[ka,kj] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[kj,ka,kb],r2[kb,kk],optimize = True)

               s[s2:f2] += temp_doubles.reshape(-1)
               del temp_doubles
               
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo
               #eris_ovvv = eris.ovvv

################ ADC(3) i - kja block and ajk - i ############################

               temp_singles = np.zeros_like((r1))

               for kb in range(nkpts):
                   for kc in range(nkpts):
                       ka = kconserv[kc,kshift,kb]
                         
                       for kk in range(nkpts): 
                           t2_1 = adc.t2[0]
                           #temp = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=t2_1.dtype)
                           #temp_1 = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=t2_1.dtype)
                           kj = kconserv[kk, kb, kc]
                           temp_1 =       lib.einsum('jkbc,ajk->abc',t2_1[kj,kk,kb], r2[ka,kj], optimize=True)
                           temp  = 0.25 * lib.einsum('jkbc,ajk->abc',t2_1[kj,kk,kb], r2[ka,kj], optimize=True)
                           temp -= 0.25 * lib.einsum('jkbc,akj->abc',t2_1[kj,kk,kb], r2[ka,kk], optimize=True)
                           temp -= 0.25 * lib.einsum('kjbc,ajk->abc',t2_1[kk,kj,kb], r2[ka,kj], optimize=True)
                           temp += 0.25 * lib.einsum('kjbc,akj->abc',t2_1[kk,kj,kb], r2[ka,kk], optimize=True)
                           ki = kconserv[kc,ka,kb]
                           #if getattr(eris, 'Lov', None) is not None:
                           if isinstance(eris.ovvv, type(None)):
                               #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                               chnk_size = adc.chnk_size
                               if chnk_size > nocc:
                                   chnk_size = nocc
                               a = 0
                               for p in range(0,nocc,chnk_size):
                                   eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[ki,kc], eris.Lvv[ka,kb], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                   k = eris_ovvv.shape[0]
                                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp,   eris_ovvv, optimize=True)
                                   del eris_ovvv
                                   eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[ki,kb], eris.Lvv[ka,kc], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                   temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp,   eris_ovvv, optimize=True)
                                   del eris_ovvv
                                   a += k
                           else :
                               eris_ovvv = eris.ovvv[:]
                               temp_singles += lib.einsum('abc,icab->i',temp_1, eris_ovvv[ki,kc,ka], optimize=True)
                               temp_singles += lib.einsum('abc,icab->i',temp,   eris_ovvv[ki,kc,ka], optimize=True)
                               temp_singles -= lib.einsum('abc,ibac->i',temp,   eris_ovvv[ki,kb,ka], optimize=True)
                               del eris_ovvv
               s[s1:f1] += temp_singles 
               del temp 
               del temp_1 

               del temp_singles 
               temp_doubles = np.zeros_like((r2))
               for kj in range(nkpts):
                   for kk in range(nkpts):
                          ka = kconserv[kk, kshift, kj]
                          for kc in range(nkpts):
                              #temp = np.zeros((nkpts,nkpts,nvir,nvir,nvir),dtype=t2_1.dtype)
                              kb = kconserv[ka, kshift, kc]
                              ki = kconserv[kc,ka,kb]
                              #if getattr(eris, 'Lov', None) is not None:
                              if isinstance(eris.ovvv, type(None)):
                                  #chnk_size = kadc_ao2mo.calculate_chunk_size(adc)
                                  chnk_size = adc.chnk_size
                                  if chnk_size > nocc:
                                      chnk_size = nocc
                                  a = 0
                                  for p in range(0,nocc,chnk_size):
                
                                      eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov[ki,kc], eris.Lvv[ka,kb], p, chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                      k = eris_ovvv.shape[0]
                                      temp = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv.conj(), optimize=True)
                                      del eris_ovvv
                                      a += k
                              else :                                  
                                  eris_ovvv = eris.ovvv[:]
                                  temp = lib.einsum('i,icab->cba',r1,eris_ovvv[ki,kc,ka].conj(), optimize=True)
                                  del eris_ovvv
                              kb = kconserv[kk,kj,kc]
                              temp_doubles[ka,kj] += lib.einsum('cba,kjcb->ajk',temp, t2_1[kk,kj,kc].conj(), optimize=True)
               s[s2:f2] += temp_doubles.reshape(-1) 
               del temp
               del temp_doubles

               for kl in range(nkpts):
                   for kk in range(nkpts):
                          kb = kconserv[kk, kshift, kl]
                          for kj in range(nkpts):
                              ka = kconserv[kb, kj, kl]
                              #temp = np.zeros_like(r2)
                              #temp_1 = np.zeros_like(r2)
                              #temp_2 = np.zeros_like(r2)

                              t2_1 = adc.t2[0]
                              temp = lib.einsum('ljba,ajk->blk',t2_1[kl,kj,kb],r2[ka,kj],optimize=True)
                              temp_2 = lib.einsum('jlba,akj->blk',t2_1[kj,kl,kb],r2[ka,kk], optimize=True)
                              del t2_1 

                              t2_1_jla = adc.t2[0][kj,kl,ka]
                              temp += lib.einsum('jlab,ajk->blk',t2_1_jla,r2[ka,kj],optimize=True)
                              temp -= lib.einsum('jlab,akj->blk',t2_1_jla,r2[ka,kk],optimize=True)

                              temp_1  = lib.einsum('jlab,ajk->blk',t2_1_jla,r2[ka,kj],optimize=True)
                              temp_1 -= lib.einsum('jlab,akj->blk',t2_1_jla,r2[ka,kk],optimize=True)
                              temp_1 += lib.einsum('jlab,ajk->blk',t2_1_jla,r2[ka,kj],optimize=True)
                              del t2_1_jla
                                  
                              t2_1_lja = adc.t2[0][kl,kj,ka]
                              temp -= lib.einsum('ljab,ajk->blk',t2_1_lja,r2[ka,kj],optimize=True)
                              temp += lib.einsum('ljab,akj->blk',t2_1_lja,r2[ka,kk],optimize=True)

                              temp_1 -= lib.einsum('ljab,ajk->blk',t2_1_lja,r2[ka,kj],optimize=True)
                              del t2_1_lja                              

                              ki = kconserv[kk, kl, kb]
                              s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,  eris_ovoo[kl,kb,ki],optimize=True)
                              s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,  eris_ovoo[ki,kb,kl],optimize=True)
                              s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo[kl,kb,ki],optimize=True)
                              s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo[ki,kb,kl],optimize=True)
                              del temp
                              del temp_1
                              del temp_2

                   for kj in range(nkpts):
                          kb = kconserv[kj, kshift, kl]
                          for kk in range(nkpts):
                              ka = kconserv[kb, kk, kl]
                              #temp = np.zeros_like(r2)
                              #temp_1 = np.zeros_like(r2)
                              #temp_2 = np.zeros_like(r2)

                              t2_1 = adc.t2[0]

                              temp = -lib.einsum('lkba,akj->blj',t2_1[kl,kk,kb],r2[ka,kk],optimize=True)
                              temp_2 = -lib.einsum('klba,ajk->blj',t2_1[kk,kl,kb],r2[ka,kj],optimize=True)
                              del t2_1

                              t2_1_kla = adc.t2[0][kk,kl,ka]
                              temp -= lib.einsum('klab,akj->blj',t2_1_kla,r2[ka,kk],optimize=True)
                              temp += lib.einsum('klab,ajk->blj',t2_1_kla,r2[ka,kj],optimize=True)
                              temp_1  = -2.0 * lib.einsum('klab,akj->blj',t2_1_kla,r2[ka,kk],optimize=True)
                              temp_1 += lib.einsum('klab,ajk->blj',t2_1_kla,r2[ka,kj],optimize=True)
                              del t2_1_kla

                              t2_1_lka = adc.t2[0][kl,kk,ka]
                              temp += lib.einsum('lkab,akj->blj',t2_1_lka,r2[ka,kk],optimize=True)
                              temp -= lib.einsum('lkab,ajk->blj',t2_1_lka,r2[ka,kj],optimize=True)
                              temp_1 += lib.einsum('lkab,akj->blj',t2_1_lka,r2[ka,kk],optimize=True)
                              del t2_1_lka

                              ki = kconserv[kj, kl, kb]
                              s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,  eris_ovoo[kl,kb,ki],optimize=True)
                              s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,  eris_ovoo[ki,kb,kl],optimize=True)
                              s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo[kl,kb,ki],optimize=True)
                              s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo[ki,kb,kl],optimize=True)
               
                              del temp
                              del temp_1
                              del temp_2

               temp_doubles = np.zeros_like((r2))
               for kj in range(nkpts):
                   for kk in range(nkpts):
                          ka = kconserv[kk, kshift, kj]
                          for kl in range(nkpts):
                              #temp = np.zeros((nkpts,nkpts,nocc,nvir,nocc),dtype=eris.oooo.dtype)
                              #temp_1 = np.zeros((nkpts,nkpts,nocc,nvir,nocc),dtype=eris.oooo.dtype)
                              #temp_2 = np.zeros((nkpts,nkpts,nocc,nvir,nocc),dtype=eris.oooo.dtype)
                              kb = kconserv[kshift, kl, kk]
                              ki = kconserv[kk,kl,kb]
                              temp_1 = lib.einsum('i,lbik->kbl',r1,eris_ovoo[kl,kb,ki].conj(), optimize=True)
                              temp  = lib.einsum('i,lbik->kbl',r1,eris_ovoo[kl,kb,ki].conj(), optimize=True)
                              temp -= lib.einsum('i,iblk->kbl',r1,eris_ovoo[ki,kb,kl].conj(), optimize=True)
                              kb = kconserv[ka, kl, kj]
                              
                              t2_1 = adc.t2[0]
                              temp_doubles[ka,kj] += lib.einsum('kbl,ljba->ajk',temp, t2_1[kl,kj,kb].conj(), optimize=True)
                              temp_doubles[ka,kj] += lib.einsum('kbl,jlab->ajk',temp_1, t2_1[kj,kl,ka].conj(), optimize=True)
                              temp_doubles[ka,kj] -= lib.einsum('kbl,ljab->ajk',temp_1, t2_1[kl,kj,ka].conj(), optimize=True)

                              kb = kconserv[kshift, kl, kj]
                              ki = kconserv[kb,kl,kj]
                              temp_2 = -lib.einsum('i,iblj->jbl',r1,eris_ovoo[ki,kb,kl].conj(), optimize=True)
                              kb = kconserv[ka, kk, kl]
                              temp_doubles[ka,kj] += lib.einsum('jbl,klba->ajk',temp_2, t2_1[kk,kl,kb].conj(), optimize=True)
                              del t2_1
               s[s2:f2] += temp_doubles.reshape(-1) 
               del temp_doubles
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s
    return sigma_


def ea_compute_trans_moments(adc, orb, kshift):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts
    nocc = adc.nocc
    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]
    nvir = adc.nmo - adc.nocc
    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    T = np.zeros((dim),dtype=np.complex)
    T_doub = np.zeros((n_doubles),dtype=t2_1.dtype).reshape(nkpts,nkpts,nocc,nvir,nvir)
    
######## ADC(2) 1h part  ############################################

    if orb < nocc:

        T[s1:f1] -= t1_2[kshift][orb,:]

        t2_1_t = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir),dtype=np.complex)

        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = adc.khelper.kconserv[kj, kshift, ka]
                ki = adc.khelper.kconserv[kj, ka, kb]

                t2_1_t[kj,ki,ka] = t2_1[ki,kj,ka].conj().transpose(1,0,2,3)
        
                T_doub[kj,ka] -= t2_1_t[kj,kshift,ka][:,orb,:,:]

        T[s2:f2] += T_doub.reshape(-1)

    else:

        T[s1:f1] += idn_vir[(orb-nocc), :]
        for kk in range(nkpts):
            for kc in range(nkpts):
                kl = adc.khelper.kconserv[kc, kk, kshift]
                ka = adc.khelper.kconserv[kc, kl, kk]
                T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,(orb-nocc),:], t2_1[kk,kl,ka].conj(), optimize = True)
                T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,(orb-nocc),:], t2_1[kl,kk,ka].conj(), optimize = True)

                T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,(orb-nocc),:], t2_1[kk,kl,ka].conj(), optimize = True)
                T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[kl,kk,kshift][:,:,(orb-nocc),:], t2_1[kk,kl,ka].conj(), optimize = True)
                T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[kk,kl,kshift][:,:,(orb-nocc),:], t2_1[kl,kk,ka].conj(), optimize = True)
                T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,(orb-nocc),:], t2_1[kl,kk,ka].conj(), optimize = True)

######### ADC(3) 2p-1h  part  ############################################

    if(method=="adc(2)-x"or adc.method=="adc(3)"):

        t2_2 = adc.t2[1]
        t2_2_t = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir),dtype=np.complex)

        if orb < nocc:

            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = adc.khelper.kconserv[kj, kshift, ka]
                    ki = adc.khelper.kconserv[kj, ka, kb]

                    t2_2_t[kj,ki,ka] = t2_2[ki,kj,ka].conj().transpose(1,0,2,3)
            
                    T_doub[kj,ka] -= t2_2_t[kj,kshift,ka][:,orb,:,:]

            T[s2:f2] += T_doub.reshape(-1)

########### ADC(3) 1p part  ############################################

    if(adc.method=="adc(3)"):


        if orb < nocc:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    ka = adc.khelper.kconserv[kc, kk, kshift]
                    T[s1:f1] += 0.5*lib.einsum('kac,ck->a',t2_1[kk,kshift,kc][:,orb,:,:], t1_2[kc].T,optimize = True)
                    T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[kshift,kk,ka][orb,:,:,:], t1_2[kc].T,optimize = True)
                    T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[kshift,kk,ka][orb,:,:,:], t1_2[kc].T,optimize = True)

        else:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kl = adc.khelper.kconserv[kc, kk, kshift]
                    ka = adc.khelper.kconserv[kc, kl, kk]

                    T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,(orb-nocc),:], t2_2[kk,kl,ka].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,(orb-nocc),:], t2_2[kl,kk,ka].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,(orb-nocc),:], t2_2[kk,kl,ka].conj(), optimize = True)
                    T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[kk,kl,kshift][:,:,(orb-nocc),:], t2_2[kl,kk,ka].conj(), optimize = True)
                    T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[kl,kk,kshift][:,:,(orb-nocc),:], t2_2[kk,kl,ka].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,(orb-nocc),:], t2_2[kl,kk,ka].conj(), optimize = True)

                    T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1[kk,kl,ka].conj(), t2_2[kk,kl,kshift][:,:,(orb-nocc),:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1[kl,kk,ka].conj(), t2_2[kl,kk,kshift][:,:,(orb-nocc),:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1[kk,kl,ka].conj(), t2_2[kk,kl,kshift][:,:,(orb-nocc),:],optimize = True)
                    T[s1:f1] += 0.25*lib.einsum('klac,lkc->a',t2_1[kk,kl,ka].conj(), t2_2[kl,kk,kshift][:,:,(orb-nocc),:],optimize = True)
                    T[s1:f1] += 0.25*lib.einsum('lkac,klc->a',t2_1[kl,kk,ka].conj(), t2_2[kk,kl,kshift][:,:,(orb-nocc),:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1[kl,kk,ka].conj(), t2_2[kl,kk,kshift][:,:,(orb-nocc),:],optimize = True)

        del t2_2
    del t2_1

    T_aaa = T[n_singles:].reshape(nkpts,nkpts,nocc,nvir,nvir).copy()
    for ka in range(nkpts):
        for kb in range(nkpts):
            ki = adc.khelper.kconserv[kb,kshift, ka]
            T_aaa[ki,ka] = T_aaa[ki,ka] - T_aaa[ki,kb].transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def ip_compute_trans_moments(adc, orb, kshift):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts
    nocc = adc.nocc
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
    T_doub = np.zeros((n_doubles),dtype=t2_1.dtype).reshape(nkpts,nkpts,nvir,nocc,nocc)
    
######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T[s1:f1] += idn_occ[orb, :]
        for kk in range(nkpts):
            for kc in range(nkpts):
                kd = adc.khelper.kconserv[kc, kk, kshift]
                ki = adc.khelper.kconserv[kk, kd, kc]
                T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[kk,kshift,kd][:,orb,:,:], t2_1[ki,kk,kd].conj(), optimize = True)
                T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[kk,kshift,kc][:,orb,:,:], t2_1[ki,kk,kd].conj(), optimize = True)
                T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[kk,kshift,kd][:,orb,:,:], t2_1[ki,kk,kc].conj(), optimize = True)
                T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[kk,kshift,kc][:,orb,:,:], t2_1[ki,kk,kc].conj(), optimize = True)
                T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[kshift,kk,kd][orb,:,:,:], t2_1[ki,kk,kd].conj(), optimize = True)
                T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[kshift,kk,kc][orb,:,:,:], t2_1[ki,kk,kc].conj(), optimize = True)
    else :
        t2_1_t = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc),dtype=np.complex)
        T[s1:f1] += t1_2[kshift][:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = adc.khelper.kconserv[kj, kshift, ki]
                kb = adc.khelper.kconserv[ki, kj, ka]

                t2_1_t[ka,kb,ki] = t2_1[ki,kj,ka].conj().transpose(2,3,0,1)
                T_doub[ka,ki] += t2_1_t[ka,kshift,ki][:,(orb-nocc),:,:]

        T[s2:f2] += T_doub.reshape(-1)
        del t2_1_t
####### ADC(3) 2h-1p  part  ############################################

    if(method=='adc(2)-x'or method=='adc(3)'):

        t2_2 = adc.t2[1]
        t2_2_t = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nocc,nocc),dtype=np.complex)

        if orb >= nocc:
            for ki in range(nkpts):
                for kj in range(nkpts):
                    ka = adc.khelper.kconserv[kj, kshift, ki]
                    kb = adc.khelper.kconserv[ki, kj, ka]

                    t2_2_t[ka,kb,ki] = t2_2[ki,kj,ka].conj().transpose(2,3,0,1)
                    T_doub[ka,ki] += t2_2_t[ka,kshift,ki][:,(orb-nocc),:,:]

            T[s2:f2] += T_doub.reshape(-1)

######### ADC(3) 1h part  ############################################

    if(method=='adc(3)'):
        if orb < nocc:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = adc.khelper.kconserv[kc, kk, kshift]
                    ki = adc.khelper.kconserv[kk, kd, kc]
                    T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[kk,ki,kd][:,orb,:,:], t2_2[ki,kk,kd].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[kk,ki,kc][:,orb,:,:], t2_2[ki,kk,kd].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[kk,ki,kd][:,orb,:,:], t2_2[ki,kk,kc].conj(), optimize = True)
                    T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[kk,ki,kc][:,orb,:,:], t2_2[ki,kk,kc].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[ki,kk,kd][orb,:,:,:], t2_2[ki,kk,kd].conj(), optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[ki,kk,kc][orb,:,:,:], t2_2[ki,kk,kc].conj(), optimize = True)

                    T[s1:f1] += 0.25*lib.einsum('ikdc,kdc->i',t2_1[ki,kk,kd].conj(), t2_2[kk,ki,kd][:,orb,:,:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('ikcd,kdc->i',t2_1[ki,kk,kc].conj(), t2_2[kk,ki,kd][:,orb,:,:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('ikdc,kcd->i',t2_1[ki,kk,kd].conj(), t2_2[kk,ki,kc][:,orb,:,:],optimize = True)
                    T[s1:f1] += 0.25*lib.einsum('ikcd,kcd->i',t2_1[ki,kk,kc].conj(), t2_2[kk,ki,kc][:,orb,:,:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1[ki,kk,kc].conj(), t2_2[ki,kk,kc][orb,:,:,:],optimize = True)
                    T[s1:f1] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1[ki,kk,kd].conj(), t2_2[ki,kk,kd][orb,:,:,:],optimize = True)
        else:

            for kk in range(nkpts):
                for kc in range(nkpts):
                    ki = adc.khelper.kconserv[kk,kshift,kc]

                    T[s1:f1] += 0.5 * lib.einsum('kic,kc->i',t2_1[kk,ki,kc][:,:,:,(orb-nocc)], t1_2[kk],optimize = True)
                    T[s1:f1] -= 0.5*lib.einsum('ikc,kc->i',t2_1[ki,kk,kc][:,:,:,(orb-nocc)], t1_2[kk],optimize = True)
                    T[s1:f1] += 0.5*lib.einsum('kic,kc->i',t2_1[kk,ki,kc][:,:,:,(orb-nocc)], t1_2[kk],optimize = True)

        del t2_2
    del t2_1

    T_aaa = T[n_singles:].reshape(nkpts,nkpts,nvir,nocc,nocc).copy()
    for ki in range(nkpts):
        for kj in range(nkpts):
            ka = adc.khelper.kconserv[kj,kshift, ki]
            T_aaa[ka,ki] = T_aaa[ka,ki] - T_aaa[ka,kj].transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T

def get_trans_moments(adc,kshift):

    nmo  = adc.nmo
    T = []
    for orb in range(nmo):
        T_a = adc.compute_trans_moments(orb,kshift)
        T.append(T_a)

    T = np.array(T)
    return T

def renormalize_eigenvectors_ea(adc, kshift, nroots=1):

    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    t2 = adc.t2[0]
    t1_2 = adc.t1[0]
    nvir = adc.nmo - adc.nocc
    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    dim = n_singles + n_doubles

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nkpts,nkpts,nocc,nvir,nvir)
        UdotU = np.dot(U1.conj().ravel(),U1.ravel())
        for ka in range(nkpts):
            for kb in range(nkpts):
                ki = adc.khelper.kconserv[kb,kshift, ka]
                UdotU +=  2.*np.dot(U2[ki,ka].conj().ravel(), U2[ki,ka].ravel()) - np.dot(U2[ki,ka].conj().ravel(), U2[ki,kb].transpose(0,2,1).ravel())
        #UdotU = np.dot(U1.ravel(), U1.ravel()) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[:,I] /= np.sqrt(UdotU)

    U = U.reshape(-1,nroots)

    return U

def renormalize_eigenvectors_ip(adc, kshift, nroots=1):

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
        UdotU = np.dot(U1.conj().ravel(),U1.ravel())
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = adc.khelper.kconserv[kj, kshift, kk]
                UdotU +=  2.*np.dot(U2[ka,kj].conj().ravel(), U2[ka,kj].ravel()) - np.dot(U2[ka,kj].conj().ravel(), U2[ka,kk].transpose(0,2,1).ravel())
        #UdotU = np.dot(U1.ravel(), U1.ravel()) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[:,I] /= np.sqrt(UdotU)

    U = U.reshape(-1,nroots)

    return U

def get_properties(adc, kshift, nroots=1):

    #Transition moments
    T = adc.get_trans_moments(kshift)
    
    #Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(kshift,nroots)
    X = np.dot(T, U).reshape(-1, nroots)
    
    #Spectroscopic factors
    P = 2.0*lib.einsum("pi,pi->i", X.conj(), X)
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
        self.max_space = 60
        self.max_cycle = 200
        self.conv_tol  = 1e-7
        self.tol_residual =1e-3
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
        self.with_df = adc.with_df
        self.chnk_size = adc.chnk_size
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
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors_ea
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
        self.max_space = 60
        self.max_cycle = 100
        self.conv_tol  = 1e-7
        self.tol_residual  = 1e-3
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
        self.with_df = adc.with_df
        self.chnk_size = adc.chnk_size
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

    #def get_init_guess(self, kshift, imds, nroots=1, diag=None, ascending = True):
    #    if diag is None :
    #        diag = self.ip_adc_diag()
    #    idx = None
    #    dtype = getattr(diag, 'dtype', np.complex)
    #    if ascending:
    #        idx = np.argsort(diag)
    #    else:
    #        idx = np.argsort(diag)[::-1]
    #    guess = np.zeros((diag.shape[0], nroots), dtype=dtype)
    #    #min_shape = min(diag.shape[0], nroots)
    #    #guess[:min_shape,:min_shape] = np.identity(min_shape)
    #    e,v = np.linalg.eigh(imds[kshift])
    #    print (v)
    #    exit()
    #    guess[:self.nocc,:] = v.reshape(-1)
    #    exit()
    #    g = np.zeros((diag.shape[0], nroots), dtype=dtype)
    #    g[idx] = guess.copy()
    #    guess = []
    #    for p in range(g.shape[1]):
    #        guess.append(g[:,p])
    #    return guess

    def gen_matvec(self,kshift,imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(kshift,imds,eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag
        #return diag
