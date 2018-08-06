#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Authors: James D. McClain
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#          Alec White
#

import time
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import uccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf.pbc.cc import kintermediates_uhf
from pyscf import __config__

einsum = lib.einsum

def update_amps(cc, t1, t2, eris):
    from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    Ht1a = np.zeros_like(t1a)
    Ht1b = np.zeros_like(t1b)
    Ht2aa = np.zeros_like(t2aa)
    Ht2ab = np.zeros_like(t2ab)
    Ht2bb = np.zeros_like(t2bb)

    nocca, nvira = t1a.shape[1:]
    noccb, nvirb = t1b.shape[1:]
    fvv_ = eris.fock[0][:,nocca:,nocca:]
    fVV_ = eris.fock[1][:,noccb:,noccb:]
    foo_ = eris.fock[0][:,:nocca,:nocca]
    fOO_ = eris.fock[1][:,:noccb,:noccb]
    fov_ = eris.fock[0][:,:nocca,nocca:]
    fOV_ = eris.fock[1][:,:noccb,noccb:]

    Fvv_, FVV_ = kintermediates_uhf.cc_Fvv(cc, t1, t2, eris)
    Foo_, FOO_ = kintermediates_uhf.cc_Foo(cc, t1, t2, eris)
    Fov_, FOV_ = kintermediates_uhf.cc_Fov(cc, t1, t2, eris)

    # Move energy terms to the other side
    for k in range(nkpts):
        Fvv_[k] -= np.diag(np.diag(fvv_[k]))
        FVV_[k] -= np.diag(np.diag(fVV_[k]))
        Foo_[k] -= np.diag(np.diag(foo_[k]))
        FOO_[k] -= np.diag(np.diag(fOO_[k]))

    # Get the momentum conservation array
    kconserv = cc.khelper.kconserv

    # T1 equation
    P = kintermediates_uhf.kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    Ht1a += fov_.conj()
    Ht1b += fOV_.conj()
    Ht1a += einsum('xyximae,yme->xia', t2aa, Fov_)
    Ht1a += einsum('xyximae,yme->xia', t2ab, FOV_)
    Ht1b += einsum('xyximae,yme->xia', t2bb, FOV_)
    Ht1b += einsum('yxymiea,yme->xia', t2ab, Fov_)
    Ht1a -= np.einsum('xyzmnae,xzymine,xyzw->zia', t2aa, eris.ooov, P)
    Ht1a -= np.einsum('xyzmNaE,xzymiNE,xyzw->zia', t2ab, eris.ooOV, P)
    Ht1b -= np.einsum('xyzmnae,xzymine,xyzw->zia', t2bb, eris.OOOV, P)
    Ht1b -= np.einsum('yxwnmea,xzymine,xyzw->zia', t2ab, eris.OOov, P)

    for ka in range(nkpts):
        Ht1a[ka] += einsum('ie,ae->ia', t1a[ka], Fvv_[ka])
        Ht1b[ka] += einsum('ie,ae->ia', t1b[ka], FVV_[ka])
        Ht1a[ka] -= einsum('ma,mi->ia', t1a[ka], Foo_[ka])
        Ht1b[ka] -= einsum('ma,mi->ia', t1b[ka], FOO_[ka])

        for km in range(nkpts):
            # ka == ki; km == kf == km
            # <ma||if> = [mi|af] - [mf|ai]
            #         => [mi|af] - [fm|ia]
            Ht1a[ka] += einsum('mf,aimf->ia', t1a[km], eris.voov[ka, ka, km])
            Ht1a[ka] -= einsum('mf,miaf->ia', t1a[km], eris.oovv[km, ka, ka])
            Ht1a[ka] += einsum('MF,aiMF->ia', t1b[km], eris.voOV[ka, ka, km])

            # miaf - mfai => miaf - fmia
            Ht1b[ka] += einsum('MF,AIMF->IA', t1b[km], eris.VOOV[ka, ka, km])
            Ht1b[ka] -= einsum('MF,MIAF->IA', t1b[km], eris.OOVV[km, ka, ka])
            Ht1b[ka] += einsum('mf,fmIA->IA', t1a[km], eris.voOV[km, km, ka].conj())

            for kf in range(nkpts):
                ki = ka
                ke = kconserv[ki, kf, km]
                Ht1a[ka] += einsum('imef,fmea->ia', t2aa[ki,km,ke], eris.vovv[kf,km,ke].conj())
                Ht1a[ka] += einsum('iMeF,FMea->ia', t2ab[ki,km,ke], eris.VOvv[kf,km,ke].conj())
                Ht1b[ka] += einsum('IMEF,FMEA->IA', t2bb[ki,km,ke], eris.VOVV[kf,km,ke].conj())
                Ht1b[ka] += einsum('mIfE,fmEA->IA', t2ab[km,ki,kf], eris.voVV[kf,km,ke].conj())

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]

        # Fvv equation
        Ftmpa_kb = Fvv_[kb] - 0.5 * einsum('mb,me->be', t1a[kb], Fov_[kb])
        Ftmpb_kb = FVV_[kb] - 0.5 * einsum('MB,ME->BE', t1b[kb], FOV_[kb])

        Ftmpa_ka = Fvv_[ka] - 0.5 * einsum('mb,me->be', t1a[ka], Fov_[ka])
        Ftmpb_ka = FVV_[ka] - 0.5 * einsum('MB,ME->BE', t1b[ka], FOV_[ka])

        tmp = einsum('ijae,be->ijab', t2aa[ki, kj, ka], Ftmpa_kb)
        Ht2aa[ki, kj, ka] += tmp

        tmp = einsum('IJAE,BE->IJAB', t2bb[ki, kj, ka], Ftmpb_kb)
        Ht2bb[ki, kj, ka] += tmp

        tmp = einsum('iJaE,BE->iJaB', t2ab[ki, kj, ka], Ftmpb_kb)
        Ht2ab[ki, kj, ka] += tmp

        tmp = einsum('iJeB,ae->iJaB', t2ab[ki, kj, ka], Ftmpa_ka)
        Ht2ab[ki, kj, ka] += tmp

        #P(ab)
        tmp = einsum('ijbe,ae->ijab', t2aa[ki, kj, kb], Ftmpa_ka)
        Ht2aa[ki, kj, ka] -= tmp

        tmp = einsum('IJBE,AE->IJAB', t2bb[ki, kj, kb], Ftmpb_ka)
        Ht2bb[ki, kj, ka] -= tmp

        # Foo equation
        Ftmpa_kj = Foo_[kj] + 0.5 * einsum('je,me->mj', t1a[kj], Fov_[kj])
        Ftmpb_kj = FOO_[kj] + 0.5 * einsum('JE,ME->MJ', t1b[kj], FOV_[kj])

        Ftmpa_ki = Foo_[ki] + 0.5 * einsum('je,me->mj', t1a[ki], Fov_[ki])
        Ftmpb_ki = FOO_[ki] + 0.5 * einsum('JE,ME->MJ', t1b[ki], FOV_[ki])

        tmp = einsum('imab,mj->ijab', t2aa[ki, kj, ka], Ftmpa_kj)
        Ht2aa[ki, kj, ka] -= tmp

        tmp = einsum('IMAB,MJ->IJAB', t2bb[ki, kj, ka], Ftmpb_kj)
        Ht2bb[ki, kj, ka] -= tmp

        tmp = einsum('iMaB,MJ->iJaB', t2ab[ki, kj, ka], Ftmpb_kj)
        Ht2ab[ki, kj, ka] -= tmp

        tmp = einsum('mJaB,mi->iJaB', t2ab[ki, kj, ka], Ftmpa_ki)
        Ht2ab[ki, kj, ka] -= tmp

        #P(ij)
        tmp = einsum('jmab,mi->ijab', t2aa[kj, ki, ka], Ftmpa_ki)
        Ht2aa[ki, kj, ka] += tmp

        tmp = einsum('JMAB,MI->IJAB', t2bb[kj, ki, ka], Ftmpb_ki)
        Ht2bb[ki, kj, ka] += tmp

    # T2 equation
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    Ht2aa += (eris_ovov.transpose(0,2,1,3,5,4,6) - eris_ovov.transpose(2,0,1,5,3,4,6)).conj()
    Ht2bb += (eris_OVOV.transpose(0,2,1,3,5,4,6) - eris_OVOV.transpose(2,0,1,5,3,4,6)).conj()
    Ht2ab += eris_ovOV.transpose(0,2,1,3,5,4,6).conj()

    tauaa, tauab, taubb = kintermediates_uhf.make_tau(cc, t2, t1, t1)
    Woooo, WooOO, WOOOO = kintermediates_uhf.cc_Woooo(cc, t1, t2, eris)
    # Add the contributions from Wvvvv
    Woooo += .5 * np.einsum('xwymenf,uvwijef,xywz,uvwz->xuyminj', eris_ovov, tauaa, P, P)
    WOOOO += .5 * np.einsum('xwymenf,uvwijef,xywz,uvwz->xuyminj', eris_OVOV, taubb, P, P)
    WooOO += .5 * np.einsum('xwymeNF,uvwiJeF,xywz,uvwz->xuymiNJ', eris_ovOV, tauab, P, P)
    Ht2aa += np.einsum('xuyminj,xywmnab,xyuv->uvwijab', Woooo, tauaa, P) * .5
    Ht2bb += np.einsum('xuyminj,xywmnab,xyuv->uvwijab', WOOOO, taubb, P) * .5
    Ht2ab += np.einsum('xuymiNJ,xywmNaB,xyuv->uvwiJaB', WooOO, tauab, P)

    add_vvvv_(cc, (Ht2aa, Ht2ab, Ht2bb), t1, t2, eris)

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = \
            kintermediates_uhf.cc_Wovvo(cc, t1, t2, eris)

    #:Ht2ab += einsum('xwzimae,wvumeBJ,xwzv,wuvy->xyziJaB', t2aa, WovVO, P, P)
    #:Ht2ab += einsum('xwziMaE,wvuMEBJ,xwzv,wuvy->xyziJaB', t2ab, WOVVO, P, P)
    #:Ht2ab -= einsum('xie,zma,uwzBJme,zuwx,xyzu->xyziJaB', t1a, t1a, eris.VOov, P, P)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            Ht2ab[kx, ky, kz] += lib.einsum('imae,mebj->ijab', t2aa[kx,kw,kz], WovVO[kw,kv,ku])
            Ht2ab[kx, ky, kz] += lib.einsum('imae,mebj->ijab', t2ab[kx,kw,kz], WOVVO[kw,kv,ku])

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz,kw,ku]
        ky = kconserv[kz,kx,ku]
        Ht2ab[kx, ky, kz] -= lib.einsum('ie, ma, emjb->ijab', t1a[kx], t1a[kz], eris.voOV[kx,kz,kw].conj())

    #:Ht2ab += einsum('wxvmIeA,wvumebj,xwzv,wuvy->yxujIbA', t2ab, Wovvo, P, P)
    #:Ht2ab += einsum('wxvMIEA,wvuMEbj,xwzv,wuvy->yxujIbA', t2bb, WOVvo, P, P)
    #:Ht2ab -= einsum('xIE,zMA,uwzbjME,zuwx,xyzu->yxujIbA', t1b, t1b, eris.voOV, P, P)

    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            Ht2ab[ky,kx,ku] += lib.einsum('miea, mebj-> jiba', t2ab[kw,kx,kv], Wovvo[kw,kv,ku])
            Ht2ab[ky,kx,ku] += lib.einsum('miea, mebj-> jiba', t2bb[kw,kx,kv], WOVvo[kw,kv,ku])

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz, kw, ku]
        ky = kconserv[kz, kx, ku]
        Ht2ab[ky,kx,ku] -= lib.einsum('ie, ma, bjme->jiba', t1b[kx], t1b[kz], eris.voOV[ku,kw,kz])


    #:Ht2ab += einsum('xwviMeA,wvuMebJ,xwzv,wuvy->xyuiJbA', t2ab, WOvvO, P, P)
    #:Ht2ab -= einsum('xie,zMA,zwuMJbe,zuwx,xyzu->xyuiJbA', t1a, t1b, eris.OOvv, P, P)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            Ht2ab[kx,ky,ku] += lib.einsum('imea,mebj->ijba', t2ab[kx,kw,kv],WOvvO[kw,kv,ku])

    for kz,ku,kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz, kw, ku]
        ky = kconserv[kz, kx, ku]
        Ht2ab[kx,ky,ku] -= lib.einsum('ie, ma, mjbe->ijba', t1a[kx], t1b[kz], eris.OOvv[kz, kw, ku])

    #:Ht2ab += einsum('wxzmIaE,wvumEBj,xwzv,wuvy->yxzjIaB', t2ab, WoVVo, P, P)
    #:Ht2ab -= einsum('xIE,zma,zwumjBE,zuwx,xyzu->yxzjIaB', t1b, t1a, eris.ooVV, P, P)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            Ht2ab[ky, kx, kz] += lib.einsum('miae,mebj->jiab', t2ab[kw,kx,kz], WoVVo[kw,kv,ku])

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz,kw,ku]
        ky = kconserv[kz,kx,ku]
        Ht2ab[ky,kx,kz] -= lib.einsum('ie, ma, mjbe->jiab', t1b[kx], t1a[kz], eris.ooVV[kz,kw,ku])

    #:u2aa  = einsum('xwzimae,wvumebj,xwzv,wuvy->xyzijab', t2aa, Wovvo, P, P)
    #:u2aa += einsum('xwziMaE,wvuMEbj,xwzv,wuvy->xyzijab', t2ab, WOVvo, P, P)
    #Left this in to keep proper shape, need to replace later
    u2aa  = np.zeros_like(t2aa)
    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw, kv, ku]
            u2aa[kx,ky,kz] += lib.einsum('imae, mebj->ijab', t2aa[kx,kw,kz], Wovvo[kw,kv,ku])
            u2aa[kx,ky,kz] += lib.einsum('imae, mebj->ijab', t2ab[kx,kw,kz], WOVvo[kw,kv,ku])

    #:u2aa += einsum('xie,zma,zwumjbe,zuwx,xyzu->xyzijab', t1a, t1a, eris.oovv, P, P)
    #:u2aa -= einsum('xie,zma,uwzbjme,zuwx,xyzu->xyzijab', t1a, t1a, eris.voov, P, P)

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz,kw,ku]
        ky = kconserv[kz,kx,ku]
        u2aa[kx,ky,kz] += lib.einsum('ie,ma,mjbe->ijab',t1a[kx],t1a[kz],eris.oovv[kz,kw,ku])
        u2aa[kx,ky,kz] -= lib.einsum('ie,ma,bjme->ijab',t1a[kx],t1a[kz],eris.voov[ku,kw,kz])


    #:u2aa += np.einsum('xie,uyzbjae,uzyx->xyzijab', t1a, eris.vovv, P)
    #:u2aa -= np.einsum('zma,xzyimjb->xyzijab', t1a, eris.ooov.conj())

    for ky, kx, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[ky, ku, kx]
        u2aa[kx, ky, kz] += lib.einsum('ie, bjae->ijab', t1a[kx], eris.vovv[ku,ky,kz])
        u2aa[kx, ky, kz] -= lib.einsum('ma, imjb->ijab', t1a[kz], eris.ooov[kx,kz,ky].conj())

    u2aa = u2aa - u2aa.transpose(1,0,2,4,3,5,6)
    u2aa = u2aa - np.einsum('xyzijab,xyzu->xyuijba', u2aa, P)
    Ht2aa += u2aa

    #:u2bb  = einsum('xwzimae,wvumebj,xwzv,wuvy->xyzijab', t2bb, WOVVO, P, P)
    #:u2bb += einsum('wxvMiEa,wvuMEbj,xwzv,wuvy->xyzijab', t2ab, WovVO, P, P)
    #:u2bb += einsum('xie,zma,zwumjbe,zuwx,xyzu->xyzijab', t1b, t1b, eris.OOVV, P, P)
    #:u2bb -= einsum('xie,zma,uwzbjme,zuwx,xyzu->xyzijab', t1b, t1b, eris.VOOV, P, P)

    u2bb = np.zeros_like(t2bb)

    for kx, kw, kz in kpts_helper.loop_kkk(nkpts):
        kv = kconserv[kx, kz, kw]
        for ku in range(nkpts):
            ky = kconserv[kw,kv, ku]
            u2bb[kx, ky, kz] += lib.einsum('imae,mebj->ijab', t2bb[kx,kw,kz], WOVVO[kw,kv,ku])
            u2bb[kx, ky, kz] += lib.einsum('miea, mebj-> ijab', t2ab[kw,kx,kv],WovVO[kw,kv,ku])

    for kz, ku, kw in kpts_helper.loop_kkk(nkpts):
        kx = kconserv[kz, kw, ku]
        ky = kconserv[kz, kx, ku]
        u2bb[kx, ky, kz] += lib.einsum('ie, ma, mjbe->ijab',t1b[kx],t1b[kz],eris.OOVV[kz,kw,ku])
        u2bb[kx, ky, kz] -= lib.einsum('ie, ma, bjme->ijab', t1b[kx], t1b[kz],eris.VOOV[ku,kw,kz])

    #:u2bb += np.einsum('xie,uzybjae,uzyx->xyzijab', t1b, eris.VOVV, P)
    #:u2bb -= np.einsum('zma,xzyimjb->xyzijab', t1b, eris.OOOV.conj())

    for ky, kx, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[ky, ku, kx]
        u2bb[kx,ky,kz] += lib.einsum('ie,bjae->ijab', t1b[kx], eris.VOVV[ku,ky,kz])

    for kx, kz, ky in kpts_helper.loop_kkk(nkpts):
        u2bb[kx,ky,kz] -= lib.einsum('ma, imjb-> ijab', t1b[kz], eris.OOOV[kx,kz,ky].conj())

    u2bb = u2bb - u2bb.transpose(1,0,2,4,3,5,6)
    u2bb = u2bb - np.einsum('xyzijab,xyzu->xyuijba', u2bb, P)
    Ht2bb += u2bb

    #:Ht2ab += np.einsum('xie,uyzBJae,uzyx->xyziJaB', t1a, eris.VOvv, P)
    #:Ht2ab += np.einsum('yJE,zxuaiBE,zuxy->xyziJaB', t1b, eris.voVV, P)
    #:Ht2ab -= np.einsum('zma,xzyimjb->xyzijab', t1a, eris.ooOV.conj())
    #:Ht2ab -= np.einsum('umb,yuxjmia,xyuz->xyzijab', t1b, eris.OOov.conj(), P)
    for ky, kx, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[ky,ku,kx]
        Ht2ab[kx,ky,kz] += lib.einsum('ie, bjae-> ijab', t1a[kx], eris.VOvv[ku,ky,kz])
        Ht2ab[kx,ky,kz] += lib.einsum('je, aibe-> ijab', t1b[ky], eris.voVV[kz,kx,ku])

    for kx, kz, ky in kpts_helper.loop_kkk(nkpts):
        Ht2ab[kx,ky,kz] -= lib.einsum('ma, imjb->ijab', t1a[kz], eris.ooOV[kx,kz,ky].conj())

    for kx, ky, ku in kpts_helper.loop_kkk(nkpts):
        kz = kconserv[kx, ku, ky]
        Ht2ab[kx,ky,kz] -= lib.einsum('mb,jmia->ijab',t1b[ku],eris.OOov[ky,ku,kx].conj())

    mo_ea_v = [fvv_[k].diagonal() for k in range(nkpts)]
    mo_eb_v = [fVV_[k].diagonal() for k in range(nkpts)]
    mo_ea_o = [foo_[k].diagonal() for k in range(nkpts)]
    mo_eb_o = [fOO_[k].diagonal() for k in range(nkpts)]

    eia = []
    eIA = []
    for ki in range(nkpts):
        eia.append([mo_ea_o[ki][:,None] - mo_ea_v[ka] for ka in range(nkpts)])
        eIA.append([mo_eb_o[ki][:,None] - mo_eb_v[ka] for ka in range(nkpts)])

    for ki in range(nkpts):
        Ht1a[ki] /= eia[ki][ki]
        Ht1b[ki] /= eIA[ki][ki]

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eijab = eia[ki][ka][:,None,:,None] + eia[kj][kb][:,None,:]
        eijab[abs(eijab) < LOOSE_ZERO_TOL] = LARGE_DENOM
        Ht2aa[ki,kj,ka] /= eijab

        eijab = eia[ki][ka][:,None,:,None] + eIA[kj][kb][:,None,:]
        eijab[abs(eijab) < LOOSE_ZERO_TOL] = LARGE_DENOM
        Ht2ab[ki,kj,ka] /= eijab

        eijab = eIA[ki][ka][:,None,:,None] + eIA[kj][kb][:,None,:]
        eijab[abs(eijab) < LOOSE_ZERO_TOL] = LARGE_DENOM
        Ht2bb[ki,kj,ka] /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)
    return (Ht1a, Ht1b), (Ht2aa, Ht2ab, Ht2bb)


def get_normt_diff(cc, t1, t2, t1new, t2new):
    '''Calculates norm(t1 - t1new) + norm(t2 - t2new).'''
    return (np.linalg.norm(t1new[0] - t1[0])**2 +
            np.linalg.norm(t1new[1] - t1[1])**2 +
            np.linalg.norm(t2new[0] - t2[0])**2 +
            np.linalg.norm(t2new[1] - t2[1])**2 +
            np.linalg.norm(t2new[2] - t2[2])**2) ** .5


def energy(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    kka, noa, nva = t1a.shape
    kkb, nob, nvb = t1b.shape
    assert(kka == kkb)
    nkbps = kka
    s = 0.0 + 0j
    fa, fb = eris.fock
    for ki in range(nkpts):
        s += einsum('ia,ia', fa[ki, :noa, noa:], t1a[ki, :, :])
        s += einsum('ia,ia', fb[ki, :nob, nob:], t1b[ki, :, :])
    t1t1aa = np.zeros(shape=t2aa.shape, dtype=t2aa.dtype)
    t1t1ab = np.zeros(shape=t2ab.shape, dtype=t2ab.dtype)
    t1t1bb = np.zeros(shape=t2bb.shape, dtype=t2bb.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            t1t1aa[ki, kj, ka, :, :, :, :] = einsum('ia,jb->ijab', t1a[ki, :, :], t1a[kj, :, :])
            t1t1ab[ki, kj, ka, :, :, :, :] = einsum('ia,jb->ijab', t1a[ki, :, :], t1b[kj, :, :])
            t1t1bb[ki, kj, ka, :, :, :, :] = einsum('ia,jb->ijab', t1b[ki, :, :], t1b[kj, :, :])
    tauaa = t2aa + 2*t1t1aa
    tauab = t2ab + t1t1ab
    taubb = t2bb + 2*t1t1bb
    d = 0.0 + 0.j
    d += 0.25*(einsum('xzyiajb,xyzijab->',eris.ovov,tauaa)
            - einsum('yzxjaib,xyzijab->',eris.ovov,tauaa))
    d += einsum('xzyiajb,xyzijab->',eris.ovOV,tauab)
    d += 0.25*(einsum('xzyiajb,xyzijab->',eris.OVOV,taubb)
            - einsum('yzxjaib,xyzijab->',eris.OVOV,taubb))
    e = s + d
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KCCSD energy %s', e)
    return e.real


def get_nocc(cc, per_kpoint=False):
    '''See also function get_nocc in pyscf/pbc/mp2/kmp2.py'''
    if cc._nocc is not None:
        return cc._nocc

    assert(cc.frozen == 0)

    if isinstance(cc.frozen, (int, np.integer)):
        nocca = [(np.count_nonzero(cc.mo_occ[0][k] > 0) - cc.frozen) for k in range(cc.nkpts)]
        noccb = [(np.count_nonzero(cc.mo_occ[1][k] > 0) - cc.frozen) for k in range(cc.nkpts)]

    else:
        raise NotImplementedError

    if not per_kpoint:
        nocca = np.amax(nocca)
        noccb = np.amax(noccb)
    return nocca, noccb

def get_nmo(cc, per_kpoint=False):
    '''See also function get_nmo in pyscf/pbc/mp2/kmp2.py'''
    if cc._nmo is not None:
        return cc._nmo

    assert(cc.frozen == 0)

    if isinstance(cc.frozen, (int, np.integer)):
        nmoa = [(cc.mo_occ[0][k].size - cc.frozen) for k in range(cc.nkpts)]
        nmob = [(cc.mo_occ[1][k].size - cc.frozen) for k in range(cc.nkpts)]

    else:
        raise NotImplementedError

    if not per_kpoint:
        nmoa = np.amax(nmoa)
        nmob = np.amax(nmob)
    return nmoa, nmob

def get_frozen_mask(cc):
    '''See also get_frozen_mask function in pyscf/pbc/mp2/kmp2.py'''

    moidxa = [np.ones(x.size, dtype=np.bool) for x in cc.mo_occ[0]]
    moidxb = [np.ones(x.size, dtype=np.bool) for x in cc.mo_occ[1]]
    assert(cc.frozen == 0)

    if isinstance(cc.frozen, (int, np.integer)):
        for idx in moidxa:
            idx[:cc.frozen] = False
        for idx in moidxb:
            idx[:cc.frozen] = False
    else:
        raise NotImplementedError

    return moidxa, moisxb

def amplitudes_to_vector(t1, t2):
    return np.hstack((t1[0].ravel(), t1[1].ravel(),
                      t2[0].ravel(), t2[1].ravel(), t2[2].ravel()))

def vector_to_amplitudes(vec, nmo, nocc, nkpts=1):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    sizes = (nkpts*nocca*nvira, nkpts*noccb*nvirb,
             nkpts**3*nocca**2*nvira**2, nkpts**3*nocca*noccb*nvira*nvirb,
             nkpts**3*noccb**2*nvirb**2)
    sections = np.cumsum(sizes[:-1])
    t1a, t1b, t2aa, t2ab, t2bb = np.split(vec, sections)

    t1a = t1a.reshape(nkpts,nocca,nvira)
    t1b = t1b.reshape(nkpts,noccb,nvirb)
    t2aa = t2aa.reshape(nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)
    t2ab = t2ab.reshape(nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)
    t2bb = t2bb.reshape(nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)
    return (t1a,t1b), (t2aa,t2ab,t2bb)

def add_vvvv_(cc, Ht2, t1, t2, eris):
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    Ht2aa, Ht2ab, Ht2bb = Ht2

    if cc.direct and hasattr(eris, 'Lpv'):
        def get_Wvvvv(ka, kc, kb):
            kd = kconserv[ka,kc,kb]

            Lbd = (eris.Lpv[kb,kd,:,nocca:] -
                   lib.einsum('Lkd,kb->Lbd', eris.Lpv[kb,kd,:,:nocca], t1a[kb]))
            Wvvvv = lib.einsum('Lac,Lbd->acbd', eris.Lpv[ka,kc,:,nocca:], Lbd)
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.Lpv[ka,kc,:,:nocca],
                              eris.Lpv[kb,kd,:,nocca:])
            Wvvvv -= lib.einsum('kcbd,ka->acbd', kcbd, t1a[ka])

            LBD = (eris.LPV[kb,kd,:,noccb:] -
                   lib.einsum('Lkd,kb->Lbd', eris.LPV[kb,kd,:,:noccb], t1b[kb]))

            WvvVV = lib.einsum('Lac,Lbd->acbd', eris.Lpv[ka,kc,:,nocca:], LBD)
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.Lpv[ka,kc,:,:nocca],
                              eris.LPV[kb,kd,:,noccb:])
            WvvVV -= lib.einsum('kcbd,ka->acbd', kcbd, t1a[ka])

            WVVVV = lib.einsum('Lac,Lbd->acbd', eris.LPV[ka,kc,:,noccb:], LBD)
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.LPV[ka,kc,:,:noccb],
                              eris.LPV[kb,kd,:,noccb:])
            WVVVV -= lib.einsum('kcbd,ka->acbd', kcbd, t1b[ka])

            Wvvvv *= (1./nkpts)
            WvvVV *= (1./nkpts)
            WVVVV *= (1./nkpts)
            return Wvvvv, WvvVV, WVVVV

        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka,kc,kb]
            Wvvvv, WvvVV, WVVVV = get_Wvvvv(ka, kc, kb)
            for ki in range(nkpts):
                kj = kconserv[ka,ki,kb]
                tauaa = t2aa[ki,kj,kc].copy()
                tauab = t2ab[ki,kj,kc].copy()
                taubb = t2bb[ki,kj,kc].copy()
                if ki == kc and kj == kd:
                    tauaa += np.einsum('ic,jd->ijcd', t1a[ki], t1a[kj])
                    tauab += np.einsum('ic,jd->ijcd', t1a[ki], t1b[kj])
                    taubb += np.einsum('ic,jd->ijcd', t1b[ki], t1b[kj])
                if ki == kd and kj == kc:
                    tauaa -= np.einsum('id,jc->ijcd', t1a[ki], t1a[kj])
                    taubb -= np.einsum('id,jc->ijcd', t1b[ki], t1b[kj])

                tmp = lib.einsum('acbd,ijcd->ijab', Wvvvv, tauaa) * .5
                Ht2aa[ki,kj,ka] += tmp
                Ht2aa[ki,kj,kb] -= tmp.transpose(0,1,3,2)

                tmp = lib.einsum('acbd,ijcd->ijab', WVVVV, taubb) * .5
                Ht2bb[ki,kj,ka] += tmp
                Ht2bb[ki,kj,kb] -= tmp.transpose(0,1,3,2)

                Ht2ab[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', WvvVV, tauab)
            Wvvvv = WvvVV = WVVVV = None
    else:
        _Wvvvv, _WvvVV, _WVVVV = kintermediates_uhf.cc_Wvvvv(cc, t1, t2, eris)
        def get_Wvvvv(ka, kc, kb):
            return _Wvvvv[ka,kc,kb], _WvvVV[ka,kc,kb], _WVVVV[ka,kc,kb]

        #:Ht2aa += np.einsum('xyuijef,zuwaebf,xyuv,zwuv->xyzijab', tauaa, _Wvvvv, P, P) * .5
        #:Ht2bb += np.einsum('xyuijef,zuwaebf,xyuv,zwuv->xyzijab', taubb, _WVVVV, P, P) * .5
        #:Ht2ab += np.einsum('xyuiJeF,zuwaeBF,xyuv,zwuv->xyziJaB', tauab, _WvvVV, P, P)
        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka,kc,kb]
            Wvvvv, WvvVV, WVVVV = get_Wvvvv(ka, kc, kb)
            for ki in range(nkpts):
                kj = kconserv[ka,ki,kb]
                tauaa = t2aa[ki,kj,kc].copy()
                tauab = t2ab[ki,kj,kc].copy()
                taubb = t2bb[ki,kj,kc].copy()
                if ki == kc and kj == kd:
                    tauaa += np.einsum('ic,jd->ijcd', t1a[ki], t1a[kj])
                    tauab += np.einsum('ic,jd->ijcd', t1a[ki], t1b[kj])
                    taubb += np.einsum('ic,jd->ijcd', t1b[ki], t1b[kj])
                if ki == kd and kj == kc:
                    tauaa -= np.einsum('id,jc->ijcd', t1a[ki], t1a[kj])
                    taubb -= np.einsum('id,jc->ijcd', t1b[ki], t1b[kj])

                Ht2aa[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', Wvvvv, tauaa) * .5
                Ht2bb[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', WVVVV, taubb) * .5
                Ht2ab[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', WvvVV, tauab)
        _Wvvvv = _WvvVV = _WVVVV = None

    # Contractions below are merged to Woooo intermediates
    # tauaa, tauab, taubb = kintermediates_uhf.make_tau(cc, t2, t1, t1)
    # P = kintermediates_uhf.kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    # minj = np.einsum('xwymenf,uvwijef,xywz,uvwz->xuyminj', eris.ovov, tauaa, P, P)
    # MINJ = np.einsum('xwymenf,uvwijef,xywz,uvwz->xuyminj', eris.OVOV, taubb, P, P)
    # miNJ = np.einsum('xwymeNF,uvwiJeF,xywz,uvwz->xuymiNJ', eris.ovOV, tauab, P, P)
    # Ht2aa += np.einsum('xuyminj,xywmnab,xyuv->uvwijab', minj, tauaa, P) * .25
    # Ht2bb += np.einsum('xuyminj,xywmnab,xyuv->uvwijab', MINJ, taubb, P) * .25
    # Ht2ab += np.einsum('xuymiNJ,xywmNaB,xyuv->uvwiJaB', miNJ, tauab, P) * .5
    return (Ht2aa, Ht2ab, Ht2bb)


class KUCCSD(uccsd.UCCSD):

    max_space = getattr(__config__, 'pbc_cc_kccsd_uhf_KUCCSD_max_space', 20)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.khf.KSCF))
        uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.khelper = kpts_helper.KptsHelper(mf.cell, self.kpts)
        self.direct = True  # If possible, use GDF to compute Wvvvv on-the-fly

        keys = set(['kpts', 'mo_energy', 'khelper', 'max_space', 'direct'])
        self._keys = self._keys.union(keys)

    @property
    def nkpts(self):
        return len(self.kpts)

    get_normt_diff = get_normt_diff
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    update_amps = update_amps
    energy = energy

    def dump_flags(self):
        return uccsd.UCCSD.dump_flags(self)

    def ao2mo(self, mo_coeff=None):
        nkpts = self.nkpts
        nmoa, nmob = self.nmo
        mem_incore = nkpts**3 * (nmoa**4 + nmob**4) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            return _make_eris_incore(self, mo_coeff)
        else:
            raise NotImplementedError

    def init_amps(self, eris):
        from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
        time0 = time.clock(), time.time()

        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb

        nocc = nocca + noccb
        nvir = nvira + nvirb

        nkpts = self.nkpts
        t1a = np.zeros((nkpts, nocca, nvira), dtype=np.complex128)
        t1b = np.zeros((nkpts, noccb, nvirb), dtype=np.complex128)
        t1 = (t1a, t1b)
        t2aa = np.zeros((nkpts, nkpts, nkpts, nocca, nocca, nvira, nvira), dtype=np.complex128)
        t2ab = np.zeros((nkpts, nkpts, nkpts, nocca, noccb, nvira, nvirb), dtype=np.complex128)
        t2bb = np.zeros((nkpts, nkpts, nkpts, noccb, noccb, nvirb, nvirb), dtype=np.complex128)
        fa, fb = eris.fock
        fooa = fa[:,:nocca,:nocca]
        foob = fb[:,:noccb,:noccb]
        fvva = fa[:,nocca:,nocca:]
        fvvb = fb[:,noccb:,noccb:]

        kconserv = kpts_helper.get_kconserv(self._scf.cell, self.kpts)
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki, ka, kj]
            Daa = (fooa[ki].diagonal()[:, None, None, None] + fooa[kj].diagonal()[None, :, None, None] -
                     fvva[ka].diagonal()[None, None, :, None] - fvva[kb].diagonal()[None, None, None, :])
            Dab = (fooa[ki].diagonal()[:, None, None, None] + foob[kj].diagonal()[None, :, None, None] -
                     fvva[ka].diagonal()[None, None, :, None] - fvvb[kb].diagonal()[None, None, None, :])
            Dbb = (foob[ki].diagonal()[:, None, None, None] + foob[kj].diagonal()[None, :, None, None] -
                     fvvb[ka].diagonal()[None, None, :, None] - fvvb[kb].diagonal()[None, None, None, :])

            # Due to padding; see above discussion concerning t1new in update_amps()
            idx = np.where(abs(Daa) < LOOSE_ZERO_TOL)[0]
            Daa[idx] = LARGE_DENOM
            idx = np.where(abs(Dab) < LOOSE_ZERO_TOL)[0]
            Dab[idx] = LARGE_DENOM
            idx = np.where(abs(Dbb) < LOOSE_ZERO_TOL)[0]
            Dbb[idx] = LARGE_DENOM

            t2aa[ki,kj,ka,:,:,:,:] = eris.ovov[ki,ka,kj,:,:,:,:].transpose((0,2,1,3))/Daa \
                    - eris.ovov[kj,ka,ki,:,:,:,:].transpose((2,0,1,3))/Daa
            t2ab[ki,kj,ka,:,:,:,:] = eris.ovOV[ki,ka,kj,:,:,:,:].transpose((0,2,1,3))/Dab
            t2bb[ki,kj,ka,:,:,:,:] = eris.OVOV[ki,ka,kj,:,:,:,:].transpose((0,2,1,3))/Dbb \
                    - eris.OVOV[kj,ka,ki,:,:,:,:].transpose((2,0,1,3))/Dbb

        t2aa = np.conj(t2aa)
        t2ab = np.conj(t2ab)
        t2bb = np.conj(t2bb)
        t2 = (t2aa,t2ab,t2bb)

        d = 0.0 + 0.j
        d += 0.25*(einsum('xzyiajb,xyzijab->',eris.ovov,t2aa)
                - einsum('yzxjaib,xyzijab->',eris.ovov,t2aa))
        d += einsum('xzyiajb,xyzijab->',eris.ovOV,t2ab)
        d += 0.25*(einsum('xzyiajb,xyzijab->',eris.OVOV,t2bb)
                - einsum('yzxjaib,xyzijab->',eris.OVOV,t2bb))
        self.emp2 = d/nkpts

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def amplitudes_to_vector(self, t1, t2):
        return amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None, nkpts=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes(vec, nmo, nocc, nkpts)

UCCSD = KUCCSD


def _make_eris_incore(cc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(cc)
    eris = uccsd._ChemistsERIs()
    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
    eris.mo_coeff = mo_coeff  # TODO: handle frozen orbitals
    eris.nocc = cc.nocc

    kpts = cc.kpts
    mo_a, mo_b = mo_coeff
    nkpts = cc.nkpts
    nocca, noccb = eris.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    # Re-make our fock MO matrix elements from density and fock AO
    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    hcore = cc._scf.get_hcore()
    vhf = cc._scf.get_veff(cc._scf.cell, dm)
    focka = [reduce(np.dot, (mo.conj().T, hcore[k]+vhf[0][k], mo))
             for k, mo in enumerate(mo_a)]
    fockb = [reduce(np.dot, (mo.conj().T, hcore[k]+vhf[1][k], mo))
             for k, mo in enumerate(mo_b)]
    eris.fock = (np.asarray(focka), np.asarray(fockb))

    # The momentum conservation array
    kconserv = cc.khelper.kconserv

    dtype = mo_coeff[0][0].dtype
    eris.oooo = np.empty((nkpts,nkpts,nkpts,nocca,nocca,nocca,nocca), dtype=dtype)
    eris.ooov = np.empty((nkpts,nkpts,nkpts,nocca,nocca,nocca,nvira), dtype=dtype)
    eris.oovv = np.empty((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira), dtype=dtype)
    eris.ovov = np.empty((nkpts,nkpts,nkpts,nocca,nvira,nocca,nvira), dtype=dtype)
    eris.voov = np.empty((nkpts,nkpts,nkpts,nvira,nocca,nocca,nvira), dtype=dtype)
    eris.vovv = np.empty((nkpts,nkpts,nkpts,nvira,nocca,nvira,nvira), dtype=dtype)
    eris.vvvv = np.empty((nkpts,nkpts,nkpts,nvira,nvira,nvira,nvira), dtype=dtype)
    fao2mo = cc._scf.with_df.ao2mo
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        eri_kpt = fao2mo((mo_a[kp],mo_a[kq],mo_a[kr],mo_a[ks]),
                         (kpts[kp],kpts[kq],kpts[kr],kpts[ks]), compact=False)
        eri_kpt = eri_kpt.reshape(nmoa,nmoa,nmoa,nmoa)

        eris.oooo[kp,kq,kr] = eri_kpt[:nocca,:nocca,:nocca,:nocca] / nkpts
        eris.ooov[kp,kq,kr] = eri_kpt[:nocca,:nocca,:nocca,nocca:] / nkpts
        eris.oovv[kp,kq,kr] = eri_kpt[:nocca,:nocca,nocca:,nocca:] / nkpts
        eris.ovov[kp,kq,kr] = eri_kpt[:nocca,nocca:,:nocca,nocca:] / nkpts
        eris.voov[kp,kq,kr] = eri_kpt[nocca:,:nocca,:nocca,nocca:] / nkpts
        eris.vovv[kp,kq,kr] = eri_kpt[nocca:,:nocca,nocca:,nocca:] / nkpts
        eris.vvvv[kp,kq,kr] = eri_kpt[nocca:,nocca:,nocca:,nocca:] / nkpts

    eris.OOOO = np.empty((nkpts,nkpts,nkpts,noccb,noccb,noccb,noccb), dtype=dtype)
    eris.OOOV = np.empty((nkpts,nkpts,nkpts,noccb,noccb,noccb,nvirb), dtype=dtype)
    eris.OOVV = np.empty((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb), dtype=dtype)
    eris.OVOV = np.empty((nkpts,nkpts,nkpts,noccb,nvirb,noccb,nvirb), dtype=dtype)
    eris.VOOV = np.empty((nkpts,nkpts,nkpts,nvirb,noccb,noccb,nvirb), dtype=dtype)
    eris.VOVV = np.empty((nkpts,nkpts,nkpts,nvirb,noccb,nvirb,nvirb), dtype=dtype)
    eris.VVVV = np.empty((nkpts,nkpts,nkpts,nvirb,nvirb,nvirb,nvirb), dtype=dtype)
    fao2mo = cc._scf.with_df.ao2mo
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        eri_kpt = fao2mo((mo_b[kp],mo_b[kq],mo_b[kr],mo_b[ks]),
                         (kpts[kp],kpts[kq],kpts[kr],kpts[ks]), compact=False)
        eri_kpt = eri_kpt.reshape(nmob,nmob,nmob,nmob)

        eris.OOOO[kp,kq,kr] = eri_kpt[:noccb,:noccb,:noccb,:noccb] / nkpts
        eris.OOOV[kp,kq,kr] = eri_kpt[:noccb,:noccb,:noccb,noccb:] / nkpts
        eris.OOVV[kp,kq,kr] = eri_kpt[:noccb,:noccb,noccb:,noccb:] / nkpts
        eris.OVOV[kp,kq,kr] = eri_kpt[:noccb,noccb:,:noccb,noccb:] / nkpts
        eris.VOOV[kp,kq,kr] = eri_kpt[noccb:,:noccb,:noccb,noccb:] / nkpts
        eris.VOVV[kp,kq,kr] = eri_kpt[noccb:,:noccb,noccb:,noccb:] / nkpts
        eris.VVVV[kp,kq,kr] = eri_kpt[noccb:,noccb:,noccb:,noccb:] / nkpts

    eris.ooOO = np.empty((nkpts,nkpts,nkpts,nocca,nocca,noccb,noccb), dtype=dtype)
    eris.ooOV = np.empty((nkpts,nkpts,nkpts,nocca,nocca,noccb,nvirb), dtype=dtype)
    eris.ooVV = np.empty((nkpts,nkpts,nkpts,nocca,nocca,nvirb,nvirb), dtype=dtype)
    eris.ovOV = np.empty((nkpts,nkpts,nkpts,nocca,nvira,noccb,nvirb), dtype=dtype)
    eris.voOV = np.empty((nkpts,nkpts,nkpts,nvira,nocca,noccb,nvirb), dtype=dtype)
    eris.voVV = np.empty((nkpts,nkpts,nkpts,nvira,nocca,nvirb,nvirb), dtype=dtype)
    eris.vvVV = np.empty((nkpts,nkpts,nkpts,nvira,nvira,nvirb,nvirb), dtype=dtype)
    fao2mo = cc._scf.with_df.ao2mo
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        eri_kpt = fao2mo((mo_a[kp],mo_a[kq],mo_b[kr],mo_b[ks]),
                         (kpts[kp],kpts[kq],kpts[kr],kpts[ks]), compact=False)
        eri_kpt = eri_kpt.reshape(nmoa,nmoa,nmob,nmob)

        eris.ooOO[kp,kq,kr] = eri_kpt[:nocca,:nocca,:noccb,:noccb] / nkpts
        eris.ooOV[kp,kq,kr] = eri_kpt[:nocca,:nocca,:noccb,noccb:] / nkpts
        eris.ooVV[kp,kq,kr] = eri_kpt[:nocca,:nocca,noccb:,noccb:] / nkpts
        eris.ovOV[kp,kq,kr] = eri_kpt[:nocca,nocca:,:noccb,noccb:] / nkpts
        eris.voOV[kp,kq,kr] = eri_kpt[nocca:,:nocca,:noccb,noccb:] / nkpts
        eris.voVV[kp,kq,kr] = eri_kpt[nocca:,:nocca,noccb:,noccb:] / nkpts
        eris.vvVV[kp,kq,kr] = eri_kpt[nocca:,nocca:,noccb:,noccb:] / nkpts

    eris.OOoo = np.empty((nkpts,nkpts,nkpts,noccb,noccb,nocca,nocca), dtype=dtype)
    eris.OOov = np.empty((nkpts,nkpts,nkpts,noccb,noccb,nocca,nvira), dtype=dtype)
    eris.OOvv = np.empty((nkpts,nkpts,nkpts,noccb,noccb,nvira,nvira), dtype=dtype)
    eris.OVov = np.empty((nkpts,nkpts,nkpts,noccb,nvirb,nocca,nvira), dtype=dtype)
    eris.VOov = np.empty((nkpts,nkpts,nkpts,nvirb,noccb,nocca,nvira), dtype=dtype)
    eris.VOvv = np.empty((nkpts,nkpts,nkpts,nvirb,noccb,nvira,nvira), dtype=dtype)
    eris.VVvv = np.empty((nkpts,nkpts,nkpts,nvirb,nvirb,nvira,nvira), dtype=dtype)
    fao2mo = cc._scf.with_df.ao2mo
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp,kq,kr]
        eri_kpt = fao2mo((mo_b[kp],mo_b[kq],mo_a[kr],mo_a[ks]),
                         (kpts[kp],kpts[kq],kpts[kr],kpts[ks]), compact=False)
        eri_kpt = eri_kpt.reshape(nmob,nmob,nmoa,nmoa)

        eris.OOoo[kp,kq,kr] = eri_kpt[:noccb,:noccb,:nocca,:nocca] / nkpts
        eris.OOov[kp,kq,kr] = eri_kpt[:noccb,:noccb,:nocca,nocca:] / nkpts
        eris.OOvv[kp,kq,kr] = eri_kpt[:noccb,:noccb,nocca:,nocca:] / nkpts
        eris.OVov[kp,kq,kr] = eri_kpt[:noccb,noccb:,:nocca,nocca:] / nkpts
        eris.VOov[kp,kq,kr] = eri_kpt[noccb:,:noccb,:nocca,nocca:] / nkpts
        eris.VOvv[kp,kq,kr] = eri_kpt[noccb:,:noccb,nocca:,nocca:] / nkpts
        eris.VVvv[kp,kq,kr] = eri_kpt[noccb:,noccb:,nocca:,nocca:] / nkpts

    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_df_eris(cc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo
    if cc._scf.with_df._cderi is None:
        cc._scf.with_df.build()

    eris = _make_eris_incore(cc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nao = cc._scf.cell.nao_nr()
    mo_kpts_a, mo_kpts_b = eris.mo_coeff

    kpts = cc.kpts
    nkpts = len(kpts)
    naux = cc._scf.with_df.get_naoaux()
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_kpts_a)
    eris.Lpv = np.empty((nkpts,nkpts,naux,nmoa,nvira), dtype=dtype)
    eris.LPV = np.empty((nkpts,nkpts,naux,nmob,nvirb), dtype=dtype)

    with h5py.File(cc._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'].value
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti,kptj))
                k_id = member(kpti_kptj, kptij_lst)
                if len(k_id) > 0:
                    Lpq = np.asarray(f['j3c/' + str(k_id[0])])
                else:
                    kptji = kpti_kptj[[1,0]]
                    k_id = member(kptji, kptij_lst)
                    Lpq = np.asarray(f['j3c/' + str(k_id[0])])
                    Lpq = lib.transpose(Lpq.reshape(naux,nao,nao), axes=(0,2,1))
                    Lpq = Lpq.conj()

                mo_a = np.hstack((mo_kpts_a[ki], mo_kpts_a[kj][:,nocca:]))
                mo_b = np.hstack((mo_kpts_b[ki], mo_kpts_b[kj][:,noccb:]))
                mo_a = np.asarray(mo_a, dtype=dtype, order='F')
                mo_b = np.asarray(mo_b, dtype=dtype, order='F')
                if dtype == np.double:
                    _ao2mo.nr_e2(Lpq, mo_a, (0, nmoa, nmoa, nmoa+nvira), aosym='s2',
                                 out=eris.Lpv[ki,kj])
                    _ao2mo.nr_e2(Lpq, mo_b, (0, nmob, nmob, nmob+nvirb), aosym='s2',
                                 out=eris.LPV[ki,kj])
                else:
                    if Lpq.size != naux*nao**2: # aosym = 's2'
                        Lpq = lib.unpack_tril(Lpq).astype(np.complex128)
                    _ao2mo.r_e2(Lpq, mo_a, (0, nmoa, nmoa, nmoa+nvira), tao, ao_loc,
                                out=eris.Lpv[ki,kj])
                    _ao2mo.r_e2(Lpq, mo_b, (0, nmob, nmob, nmob+nvirb), tao, ao_loc,
                                out=eris.LPV[ki,kj])
    return eris


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from pyscf import lo

    cell = gto.Cell()
    cell.atom='''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    #cell.basis = [[0, (1., 1.)], [1, (.5, 1.)]]
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.build()

    np.random.seed(2)
    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((2,3,nmo))
    kmf.mo_occ[0,:,:3] = 1
    kmf.mo_occ[1,:,:1] = 1
    kmf.mo_energy = np.arange(nmo) + np.random.random((2,3,nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2

    mo = (np.random.random((2,3,nmo,nmo)) +
          np.random.random((2,3,nmo,nmo))*1j - .5-.5j)
    s = kmf.get_ovlp()
    kmf.mo_coeff = np.empty_like(mo)
    nkpts = len(kmf.kpts)
    for k in range(nkpts):
        kmf.mo_coeff[0,k] = lo.orth.vec_lowdin(mo[0,k], s[k])
        kmf.mo_coeff[1,k] = lo.orth.vec_lowdin(mo[1,k], s[k])

    def rand_t1_t2(mycc):
        nkpts = mycc.nkpts
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        np.random.seed(1)
        t1a = (np.random.random((nkpts,nocca,nvira)) +
               np.random.random((nkpts,nocca,nvira))*1j - .5-.5j)
        t1b = (np.random.random((nkpts,noccb,nvirb)) +
               np.random.random((nkpts,noccb,nvirb))*1j - .5-.5j)
        t2aa = (np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira))*1j - .5-.5j)
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        t2aa = t2aa - t2aa.transpose(1,0,2,4,3,5,6)
        tmp = t2aa.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
        t2ab = (np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
        t2bb = (np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
        t2bb = t2bb - t2bb.transpose(1,0,2,4,3,5,6)
        tmp = t2bb.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2bb[ki,kj,kk] = t2bb[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)

        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        return t1, t2

    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1[0]) - (-1.2692088297292825-12.893074780897923j))
    print(lib.finger(Ht1[1]) - (-11.831413366451148+19.95758532598137j ))
    print(lib.finger(Ht2[0])*1e-2 - (0.97436765562779959 +0.16548728742427826j ))
    print(lib.finger(Ht2[1])*1e-2 - (-1.7752605990115735 +4.2106261874056212j  ))
    print(lib.finger(Ht2[2])*1e-3 - (-0.52223406190978494-0.91888685193234421j))

    kmf.mo_occ[:] = 0
    kmf.mo_occ[:,:,:2] = 1
    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1[0]) - (3.7571382837650931+3.6719235677672519j))
    print(lib.finger(Ht1[1])*1e-2 - (-0.42270622344333642+0.65025799860663025j))
    print(lib.finger(Ht2[0])*1e-2 - (2.5124103335695689  -1.3180553113575906j ))
    print(lib.finger(Ht2[1])*1e-2 - (-2.4427382960124304 +0.15329780363467621j))
    print(lib.finger(Ht2[2])*1e-2 - (3.0683780903085842  +2.580910132273615j  ))

    from pyscf.pbc.cc import kccsd
    kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(kmf))
    kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
    r1 = kgcc.spatial2spin(t1)
    r2 = kgcc.spatial2spin(t2)
    ge = kccsd.energy(kgcc, r1, r2, kccsd_eris)
    r1, r2 = kgcc.update_amps(r1, r2, kccsd_eris)
    ue = energy(mycc, t1, t2, eris)
    print(abs(ge - ue))
    print(abs(r1 - kgcc.spatial2spin(Ht1)).max())
    print(abs(r2 - kgcc.spatial2spin(Ht2)).max())

    kmf = kmf.density_fit(auxbasis=[[0, (1., 1.)], [0, (.5, 1.)]])
    mycc = KUCCSD(kmf)
    eris = _make_df_eris(mycc, mycc.mo_coeff)
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1[0]) - (3.6569734813260473 +3.8092774902489754j))
    print(lib.finger(Ht1[1]) - (-105.8651917884019 +219.86020519421155j))
    print(lib.finger(Ht2[0]) - (-265.25767382882208+215.41888861285341j))
    print(lib.finger(Ht2[1]) - (-115.13953446128346-49.303887916188629j))
    print(lib.finger(Ht2[2]) - (122.51835547779413 +33.85757422327751j ))

    print(all([abs(lib.finger(eris.oooo) - (-0.18290712163391809-0.13839081039521306j)  )<1e-12,
               abs(lib.finger(eris.ooOO) - (-0.084752145202964035-0.28496525042110676j) )<1e-12,
               #abs(lib.finger(eris.OOoo) - (0.43054922768629345-0.27990237216969871j)   )<1e-12,
               abs(lib.finger(eris.OOOO) - (-0.2941475969103261-0.047247498899840978j)  )<1e-12,
               abs(lib.finger(eris.ooov) - (0.23381463349517045-0.11703340936984277j)   )<1e-12,
               abs(lib.finger(eris.ooOV) - (-0.052655392703214066+0.69533309442418556j) )<1e-12,
               abs(lib.finger(eris.OOov) - (-0.2111361247200903+0.85087916975274647j)   )<1e-12,
               abs(lib.finger(eris.OOOV) - (-0.36995992208047412-0.18887278030885621j)  )<1e-12,
               abs(lib.finger(eris.oovv) - (0.21107397525051516+0.0048714991438174871j) )<1e-12,
               abs(lib.finger(eris.ooVV) - (-0.076411225687065987+0.11080438166425896j) )<1e-12,
               abs(lib.finger(eris.OOvv) - (-0.17880337626095003-0.24174716216954206j)  )<1e-12,
               abs(lib.finger(eris.OOVV) - (0.059186286356424908+0.68433866387500164j)  )<1e-12,
               abs(lib.finger(eris.ovov) - (0.15402983765151051+0.064359681685222214j)  )<1e-12,
               abs(lib.finger(eris.ovOV) - (-0.10697649196044598+0.30351249676253234j)  )<1e-12,
               #abs(lib.finger(eris.OVov) - (-0.17619329728836752-0.56585020976035816j)  )<1e-12,
               abs(lib.finger(eris.OVOV) - (-0.63963235318492118+0.69863219317718828j)  )<1e-12,
               abs(lib.finger(eris.voov) - (-0.24137641647339092+0.18676684336011531j)  )<1e-12,
               abs(lib.finger(eris.voOV) - (0.19257709151227204+0.38929027819406414j)   )<1e-12,
               #abs(lib.finger(eris.VOov) - (0.07632606729926053-0.70350947950650355j)   )<1e-12,
               abs(lib.finger(eris.VOOV) - (-0.47970203195500816+0.46735207193861927j)  )<1e-12,
               abs(lib.finger(eris.vovv) - (-0.1342049915673903-0.23391327821719513j)   )<1e-12,
               abs(lib.finger(eris.voVV) - (-0.28989635223866056+0.9644368822688475j)   )<1e-12,
               abs(lib.finger(eris.VOvv) - (-0.32428269235420271+0.0029847254383674748j))<1e-12,
               abs(lib.finger(eris.VOVV) - (0.45031779746222456-0.36858577475752041j)   )<1e-12,
               abs(lib.finger(eris.vvvv) - (-0.080512851258903173-0.2868384266725581j)  )<1e-12,
               abs(lib.finger(eris.vvVV) - (-0.5137063762484736+1.1036785801263898j)    )<1e-12,
               #abs(lib.finger(eris.VVvv) - (0.16468487082491939+0.25730725586992997j)   )<1e-12,
               abs(lib.finger(eris.VVVV) - (-0.56714875196802295+0.058636785679170501j) )<1e-12]))
