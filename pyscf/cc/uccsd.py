#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
UCCSD with spatial integrals
'''


from functools import reduce
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.ao2mo import _ao2mo
from pyscf.mp import ump2
from pyscf import scf
from pyscf import __config__

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

# This is unrestricted (U)CCSD, in spatial-orbital form.

def update_amps(cc, t1, t2, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + cc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + cc.level_shift
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    u2aa, u2ab, u2bb = cc._add_vvvv(None, (tauaa,tauab,taubb), eris)
    u2aa *= .5
    u2bb *= .5

    Fooa =  .5 * lib.einsum('me,ie->mi', fova, t1a)
    Foob =  .5 * lib.einsum('me,ie->mi', fovb, t1b)
    Fvva = -.5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb = -.5 * lib.einsum('me,ma->ae', fovb, t1b)
    Fooa += eris.focka[:nocca,:nocca] - np.diag(mo_ea_o)
    Foob += eris.fockb[:noccb,:noccb] - np.diag(mo_eb_o)
    Fvva += eris.focka[nocca:,nocca:] - np.diag(mo_ea_v)
    Fvvb += eris.fockb[noccb:,noccb:] - np.diag(mo_eb_v)
    dtype = u2aa.dtype
    wovvo = np.zeros((nocca,nvira,nvira,nocca), dtype=dtype)
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb), dtype=dtype)
    woVvO = np.zeros((nocca,nvirb,nvira,noccb), dtype=dtype)
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca), dtype=dtype)
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca), dtype=dtype)
    wOvvO = np.zeros((noccb,nvira,nvira,noccb), dtype=dtype)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now - u2aa.size*8e-6)
    if nvira > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira**3*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] += lib.einsum('jf,mebf->mbej', t1a, ovvv)
            u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
            u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
            tmp1aa = lib.einsum('ijef,mebf->ijmb', tauaa, ovvv)
            u2aa -= lib.einsum('ijmb,ma->ijab', tmp1aa, t1a[p0:p1]*.5)
            ovvv = tmp1aa = None

    if nvirb > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb**3*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVVV = eris.get_OVVV(slice(p0,p1))  # OVVV = eris.OVVV[p0:p1]
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
            u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
            tmp1bb = lib.einsum('ijef,mebf->ijmb', taubb, OVVV)
            u2bb -= lib.einsum('ijmb,ma->ijab', tmp1bb, t1b[p0:p1]*.5)
            OVVV = tmp1bb = None

    if nvirb > 0 and nocca > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvira*nvirb**2*3+1)))
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = eris.get_ovVV(slice(p0,p1))  # ovVV = eris.ovVV[p0:p1]
            Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
            u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
            tmp1ab = lib.einsum('iJeF,meBF->iJmB', tauab, ovVV)
            u2ab -= lib.einsum('iJmB,ma->iJaB', tmp1ab, t1a[p0:p1])
            ovVV = tmp1ab = None

    if nvira > 0 and noccb > 0:
        blksize = max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvirb*nvira**2*3+1)))
        for p0,p1 in lib.prange(0, noccb, blksize):
            OVvv = eris.get_OVvv(slice(p0,p1))  # OVvv = eris.OVvv[p0:p1]
            Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
            u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
            tmp1abba = lib.einsum('iJeF,MFbe->iJbM', tauab, OVvv)
            u2ab -= lib.einsum('iJbM,MA->iJbA', tmp1abba, t1b[p0:p1])
            OVvv = tmp1abba = None

    eris_ovov = np.asarray(eris.ovov)
    eris_ovoo = np.asarray(eris.ovoo)
    Woooo = lib.einsum('je,nemi->mnij', t1a, eris_ovoo)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    Fooa += np.einsum('ne,nemi->mi', t1a, ovoo)
    u1a += 0.5*lib.einsum('mnae,meni->ia', t2aa, ovoo)
    wovvo += lib.einsum('nb,nemj->mbej', t1a, ovoo)
    ovoo = eris_ovoo = None

    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * lib.einsum('mnaf,menf->ae', tilaa, ovov)
    Fooa += .5 * lib.einsum('inef,menf->mi', tilaa, ovov)
    Fova = np.einsum('nf,menf->me',t1a, ovov)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    wovvo -= 0.5*lib.einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*lib.einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = lib.einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= lib.einsum('nb,mnej->mbej', t1a, tmpaa)
    eris_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OVOO = np.asarray(eris.OVOO)
    WOOOO = lib.einsum('je,nemi->mnij', t1b, eris_OVOO)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    Foob += np.einsum('ne,nemi->mi', t1b, OVOO)
    u1b += 0.5*lib.einsum('mnae,meni->ia', t2bb, OVOO)
    wOVVO += lib.einsum('nb,nemj->mbej', t1b, OVOO)
    OVOO = eris_OVOO = None

    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * lib.einsum('MNAF,MENF->AE', tilbb, OVOV)
    Foob += .5 * lib.einsum('inef,menf->mi', tilbb, OVOV)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    wOVVO -= 0.5*lib.einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*lib.einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = lib.einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= lib.einsum('nb,mnej->mbej', t1b, tmpbb)
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_OVoo = np.asarray(eris.OVoo)
    eris_ovOO = np.asarray(eris.ovOO)
    Fooa += np.einsum('NE,NEmi->mi', t1b, eris_OVoo)
    u1a -= lib.einsum('nMaE,MEni->ia', t2ab, eris_OVoo)
    wOvVo -= lib.einsum('nb,MEnj->MbEj', t1a, eris_OVoo)
    woVVo += lib.einsum('NB,NEmj->mBEj', t1b, eris_OVoo)
    Foob += np.einsum('ne,neMI->MI', t1a, eris_ovOO)
    u1b -= lib.einsum('mNeA,meNI->IA', t2ab, eris_ovOO)
    woVvO -= lib.einsum('NB,meNJ->mBeJ', t1b, eris_ovOO)
    wOvvO += lib.einsum('nb,neMJ->MbeJ', t1a, eris_ovOO)
    WoOoO = lib.einsum('JE,NEmi->mNiJ', t1b, eris_OVoo)
    WoOoO+= lib.einsum('je,neMI->nMjI', t1a, eris_ovOO)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_OVoo = eris_ovOO = None

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= lib.einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= lib.einsum('nMfA,nfME->AE', tilab, eris_ovOV)
    Fooa += lib.einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += lib.einsum('nIfE,nfME->MI', tilab, eris_ovOV)
    Fova += np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb += np.einsum('nf,nfME->ME',t1a, eris_ovOV)
    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    wovvo += 0.5*lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*lib.einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*lib.einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = lib.einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = lib.einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= lib.einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= lib.einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += lib.einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += lib.einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = tilab = None

    Fova += fova
    Fovb += fovb
    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia', t1a, Fvva)
    u1a -= np.einsum('ma,mi->ia', t1a, Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia', t1b, Fvvb)
    u1b -= np.einsum('ma,mi->ia', t1b, Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a, oovv)
    tmp1aa = lib.einsum('ie,mjbe->mbij', t1a, oovv)
    u2aa += 2*lib.einsum('ma,mbij->ijab', t1a, tmp1aa)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b, OOVV)
    tmp1bb = lib.einsum('ie,mjbe->mbij', t1b, OOVV)
    u2bb += 2*lib.einsum('ma,mbij->ijab', t1b, tmp1bb)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    tmp1ab = lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    tmp1ab+= lib.einsum('IE,mjBE->mBjI', t1b, eris_ooVV)
    u2ab -= lib.einsum('ma,mBiJ->iJaB', t1a, tmp1ab)
    eris_ooVV = eris_ovVO = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    tmp1ba = lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    tmp1ba+= lib.einsum('ie,MJbe->MbJi', t1a, eris_OOvv)
    u2ab -= lib.einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    eris_OOvv = eris_OVvo = tmp1ba = None

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    Ftmpa = Fvva - .5*lib.einsum('mb,me->be', t1a, Fova)
    Ftmpb = Fvvb - .5*lib.einsum('mb,me->be', t1b, Fovb)
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa + 0.5*lib.einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + 0.5*lib.einsum('je,me->mj', t1b, Fovb)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    eris_ovoo = np.asarray(eris.ovoo).conj()
    eris_OVOO = np.asarray(eris.OVOO).conj()
    eris_OVoo = np.asarray(eris.OVoo).conj()
    eris_ovOO = np.asarray(eris.ovOO).conj()
    ovoo = eris_ovoo - eris_ovoo.transpose(2,1,0,3)
    OVOO = eris_OVOO - eris_OVOO.transpose(2,1,0,3)
    u2aa -= lib.einsum('ma,jbim->ijab', t1a, ovoo)
    u2bb -= lib.einsum('ma,jbim->ijab', t1b, OVOO)
    u2ab -= lib.einsum('ma,JBim->iJaB', t1a, eris_OVoo)
    u2ab -= lib.einsum('MA,ibJM->iJbA', t1b, eris_ovOO)
    eris_ovoo = eris_OVoo = eris_OVOO = eris_ovOO = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    u1a /= eia_a
    u1b /= eia_b

    u2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    u2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = u1a, u1b
    t2new = u2aa, u2ab, u2bb
    return t1new, t2new


def energy(cc, t1=None, t2=None, eris=None):
    '''UCCSD correlation energy'''
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo()

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    e  = np.einsum('ia,ia', fova, t1a)
    e += np.einsum('ia,ia', fovb, t1b)
    e += 0.25*np.einsum('ijab,iajb',t2aa, eris_ovov)
    e -= 0.25*np.einsum('ijab,ibja',t2aa, eris_ovov)
    e += 0.25*np.einsum('ijab,iajb',t2bb, eris_OVOV)
    e -= 0.25*np.einsum('ijab,ibja',t2bb, eris_OVOV)
    e +=      np.einsum('iJaB,iaJB',t2ab, eris_ovOV)
    e += 0.5*lib.einsum('ia,jb,iajb',t1a, t1a, eris_ovov)
    e -= 0.5*lib.einsum('ia,jb,ibja',t1a, t1a, eris_ovov)
    e += 0.5*lib.einsum('ia,jb,iajb',t1b, t1b, eris_OVOV)
    e -= 0.5*lib.einsum('ia,jb,ibja',t1b, t1b, eris_OVOV)
    e +=     lib.einsum('ia,jb,iajb',t1a, t1b, eris_ovOV)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in UCCSD energy %s', e)
    return e.real


get_nocc = ump2.get_nocc
get_nmo = ump2.get_nmo
get_frozen_mask = ump2.get_frozen_mask


def amplitudes_to_vector(t1, t2, out=None):
    nocca, nvira = t1[0].shape
    noccb, nvirb = t1[1].shape
    sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
    sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
    sizeab = nocca * noccb * nvira * nvirb
    vector = np.ndarray(sizea+sizeb+sizeab, t2[0].dtype, buffer=out)
    ccsd.amplitudes_to_vector_s4(t1[0], t2[0], out=vector[:sizea])
    ccsd.amplitudes_to_vector_s4(t1[1], t2[2], out=vector[sizea:])
    vector[sizea+sizeb:] = t2[1].ravel()
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nocc = nocca + noccb
    nvir = nvira + nvirb
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
    sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
    if vector.size == size and sizea > 0 and sizeb > 0:
        #return ccsd.vector_to_amplitudes_s4(vector, nmo, nocc)
        raise RuntimeError('Input vector is GCCSD vector')
    else:
        sections = np.cumsum([sizea, sizeb])
        veca, vecb, t2ab = np.split(vector, sections)
        t1a, t2aa = ccsd.vector_to_amplitudes_s4(veca, nmoa, nocca)
        t1b, t2bb = ccsd.vector_to_amplitudes_s4(vecb, nmob, noccb)
        t2ab = t2ab.copy().reshape(nocca,noccb,nvira,nvirb)
        return (t1a,t1b), (t2aa,t2ab,t2bb)

def amplitudes_from_rccsd(t1, t2):
    t2aa = t2 - t2.transpose(0,1,3,2)
    return (t1,t1), (t2aa,t2,t2aa)


def _add_vvVV(mycc, t1, t2ab, eris, out=None):
    '''Ht2 = np.einsum('iJcD,acBD->iJaB', t2ab, vvVV)
    without using symmetry in t2ab or Ht2
    '''
    time0 = logger.process_clock(), logger.perf_counter()
    if t2ab.size == 0:
        return np.zeros_like(t2ab)
    if t1 is not None:
        t2ab = make_tau_ab(t2ab, t1, t1)

    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocca, noccb, nvira, nvirb = t2ab.shape

    if mycc.direct:  # AO direct CCSD
        if getattr(eris, 'mo_coeff', None) is not None:
            mo_a, mo_b = eris.mo_coeff
        else:
            moidxa, moidxb = mycc.get_frozen_mask()
            mo_a = mycc.mo_coeff[0][:,moidxa]
            mo_b = mycc.mo_coeff[1][:,moidxb]
        # Note tensor t2ab may be t2bbab from eom_uccsd code.  In that
        # particular case, nocca, noccb do not equal to the actual number of
        # alpha/beta occupied orbitals. orbva and orbvb cannot be indexed as
        # mo_a[:,nocca:] and mo_b[:,noccb:]
        orbva = mo_a[:,-nvira:]
        orbvb = mo_b[:,-nvirb:]
        tau = lib.einsum('ijab,pa->ijpb', t2ab, orbva)
        tau = lib.einsum('ijab,pb->ijap', tau, orbvb)
        time0 = logger.timer_debug1(mycc, 'vvvv-tau mo2ao', *time0)
        buf = eris._contract_vvVV_t2(mycc, tau, mycc.direct, out, log)
        mo = np.asarray(np.hstack((orbva, orbvb)), order='F')
        Ht2 = _ao2mo.nr_e2(buf.reshape(nocca*noccb,-1), mo.conj(),
                           (0,nvira,nvira,nvira+nvirb), 's1', 's1')
        return Ht2.reshape(t2ab.shape)
    else:
        return eris._contract_vvVV_t2(mycc, t2ab, mycc.direct, out, log)

def _add_vvvv(mycc, t1, t2, eris, out=None, with_ovvv=False, t2sym=None):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    if t1 is None:
        t2aa, t2ab, t2bb = t2
    else:
        t2aa, t2ab, t2bb = make_tau(t2, t1, t1)
    nocca, nvira = t2aa.shape[1:3]
    noccb, nvirb = t2bb.shape[1:3]

    if mycc.direct:
        assert (t2sym is None)
        if with_ovvv:
            raise NotImplementedError
        if getattr(eris, 'mo_coeff', None) is not None:
            mo_a, mo_b = eris.mo_coeff
        else:
            moidxa, moidxb = mycc.get_frozen_mask()
            mo_a = mycc.mo_coeff[0][:,moidxa]
            mo_b = mycc.mo_coeff[1][:,moidxb]
        nao = mo_a.shape[0]
        otrila = np.tril_indices(nocca,-1)
        otrilb = np.tril_indices(noccb,-1)
        if nocca > 1:
            tauaa = lib.einsum('xab,pa->xpb', t2aa[otrila], mo_a[:,nocca:])
            tauaa = lib.einsum('xab,pb->xap', tauaa, mo_a[:,nocca:])
        else:
            tauaa = np.zeros((0,nao,nao))
        if noccb > 1:
            taubb = lib.einsum('xab,pa->xpb', t2bb[otrilb], mo_b[:,noccb:])
            taubb = lib.einsum('xab,pb->xap', taubb, mo_b[:,noccb:])
        else:
            taubb = np.zeros((0,nao,nao))
        tauab = lib.einsum('ijab,pa->ijpb', t2ab, mo_a[:,nocca:])
        tauab = lib.einsum('ijab,pb->ijap', tauab, mo_b[:,noccb:])
        tau = np.vstack((tauaa, taubb, tauab.reshape(nocca*noccb,nao,nao)))
        tauaa = taubb = tauab = None
        time0 = log.timer_debug1('vvvv-tau', *time0)

        buf = ccsd._contract_vvvv_t2(mycc, mycc.mol, None, tau, out, log)

        mo = np.asarray(np.hstack((mo_a[:,nocca:], mo_b[:,noccb:])), order='F')
        u2aa = np.zeros_like(t2aa)
        if nocca > 1:
            u2tril = buf[:otrila[0].size]
            u2tril = _ao2mo.nr_e2(u2tril.reshape(-1,nao**2), mo.conj(),
                                  (0,nvira,0,nvira), 's1', 's1')
            u2tril = u2tril.reshape(otrila[0].size,nvira,nvira)
            u2aa[otrila[1],otrila[0]] = u2tril.transpose(0,2,1)
            u2aa[otrila] = u2tril

        u2bb = np.zeros_like(t2bb)
        if noccb > 1:
            u2tril = buf[otrila[0].size:otrila[0].size+otrilb[0].size]
            u2tril = _ao2mo.nr_e2(u2tril.reshape(-1,nao**2), mo.conj(),
                                  (nvira,nvira+nvirb,nvira,nvira+nvirb), 's1', 's1')
            u2tril = u2tril.reshape(otrilb[0].size,nvirb,nvirb)
            u2bb[otrilb[1],otrilb[0]] = u2tril.transpose(0,2,1)
            u2bb[otrilb] = u2tril

        if nocca*noccb > 0:
            u2ab = _ao2mo.nr_e2(buf[-nocca*noccb:].reshape(nocca*noccb,nao**2), mo,
                                (0,nvira,nvira,nvira+nvirb), 's1', 's1')
            u2ab = u2ab.reshape(t2ab.shape)
        else:
            u2ab = np.zeros_like(t2ab)

    else:
        assert (not with_ovvv)
        if t2sym is None:
            tmp = eris._contract_vvvv_t2(mycc, t2aa[np.tril_indices(nocca)],
                                         mycc.direct, None)
            u2aa = ccsd._unpack_t2_tril(tmp, nocca, nvira, None, 'jiba')
            tmp = eris._contract_VVVV_t2(mycc, t2bb[np.tril_indices(noccb)],
                                         mycc.direct, None)
            u2bb = ccsd._unpack_t2_tril(tmp, noccb, nvirb, None, 'jiba')
            u2ab = eris._contract_vvVV_t2(mycc, t2ab, mycc.direct, None)
        else:
            u2aa = eris._contract_vvvv_t2(mycc, t2aa, mycc.direct, None)
            u2bb = eris._contract_VVVV_t2(mycc, t2bb, mycc.direct, None)
            u2ab = eris._contract_vvVV_t2(mycc, t2ab, mycc.direct, None)

    return u2aa,u2ab,u2bb


class UCCSD(ccsd.CCSDBase):

    conv_tol = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_uccsd_UCCSD_conv_tol_normt', 1e-6)

# Attribute frozen can be
# * An integer : The same number of inner-most alpha and beta orbitals are frozen
# * One list : Same alpha and beta orbital indices to be frozen
# * A pair of list : First list is the orbital indices to be frozen for alpha
#       orbitals, second list is for beta orbitals
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def init_amps(self, eris=None):
        time0 = logger.process_clock(), logger.perf_counter()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocca, noccb = self.nocc

        fova = eris.focka[:nocca,nocca:]
        fovb = eris.fockb[:noccb,noccb:]
        mo_ea_o = eris.mo_energy[0][:nocca]
        mo_ea_v = eris.mo_energy[0][nocca:]
        mo_eb_o = eris.mo_energy[1][:noccb]
        mo_eb_v = eris.mo_energy[1][noccb:]
        eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
        eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

        t1a = fova.conj() / eia_a
        t1b = fovb.conj() / eia_b

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        t2aa = eris_ovov.transpose(0,2,1,3).conj() / lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
        t2ab = eris_ovOV.transpose(0,2,1,3).conj() / lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
        t2bb = eris_OVOV.transpose(0,2,1,3).conj() / lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
        t2aa = t2aa - t2aa.transpose(0,1,3,2)
        t2bb = t2bb - t2bb.transpose(0,1,3,2)
        e  =      np.einsum('iJaB,iaJB', t2ab, eris_ovOV)
        e += 0.25*np.einsum('ijab,iajb', t2aa, eris_ovov)
        e -= 0.25*np.einsum('ijab,ibja', t2aa, eris_ovov)
        e += 0.25*np.einsum('ijab,iajb', t2bb, eris_OVOV)
        e -= 0.25*np.einsum('ijab,ibja', t2bb, eris_OVOV)
        self.emp2 = e.real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, (t1a,t1b), (t2aa,t2ab,t2bb)

    energy = energy
    update_amps = update_amps
    _add_vvvv = _add_vvvv
    _add_vvVV = _add_vvVV

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unrestricted (U)CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if mbpt2:
            pt = ump2.UMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            t2ab = self.t2[1]
            nocca, noccb, nvira, nvirb = t2ab.shape
            self.t1 = (np.zeros((nocca,nvira)), np.zeros((noccb,nvirb)))
            return self.e_corr, self.t1, self.t2

        return ccsd.CCSDBase.ccsd(self, t1, t2, eris)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import uccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                uccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import uccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return uccsd_t.kernel(self, eris, t1, t2, self.verbose)
    uccsd_t = ccsd_t

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_mf=True):
        '''Un-relaxed 1-particle density matrix in MO space

        Returns:
            dm1a, dm1b
        '''
        from pyscf.cc import uccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return uccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                   with_frozen=with_frozen, with_mf=with_mf)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_dm1=True):
        '''2-particle density matrix in spin-orbital basis.
        '''
        from pyscf.cc import uccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return uccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                   with_frozen=with_frozen, with_dm1=with_dm1)

    def spin_square(self, mo_coeff=None, s=None):
        from pyscf.fci.spin_op import spin_square_general
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if s is None:
            s = self._scf.get_ovlp()

        dma,dmb        = self.make_rdm1()
        dmaa,dmab,dmbb = self.make_rdm2()

        return spin_square_general(dma,dmb,dmaa,dmab,dmbb,mo_coeff,s)

    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.get_nmo()
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmoa * (nmoa+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'UCCSD detected DF being used in the HF object. '
                       'MO integrals are computed based on the DF 3-index tensors.\n'
                       'It\'s recommended to use dfuccsd.UCCSD for the '
                       'DF-UCCSD calculations')
            return _make_df_eris_outcore(self, mo_coeff)
        else:
            return _make_eris_outcore(self, mo_coeff)

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    def eomee_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMEESpinKeep(self).kernel(nroots, koopmans, guess, eris)

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMEESpinFlip(self).kernel(nroots, koopmans, guess, eris)

    def eomip_method(self):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMIP(self)

    def eomea_method(self):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMEA(self)

    def eomee_method(self):
        from pyscf.cc import eom_uccsd
        return eom_uccsd.EOMEE(self)

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.cc import dfuccsd
        mycc = dfuccsd.UCCSD(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mycc.with_df = with_df
        if mycc.with_df.auxbasis != auxbasis:
            mycc.with_df = mycc.with_df.copy()
            mycc.with_df.auxbasis = auxbasis
        return mycc

    def nuc_grad_method(self):
        from pyscf.grad import uccsd
        return uccsd.Gradients(self)

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vector, nmo, nocc)

    def vector_size(self, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nocca, noccb = nocc
        nmoa, nmob = nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
        sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
        sizeab = nocca * noccb * nvira * nvirb
        return sizea + sizeb + sizeab

    def amplitudes_from_rccsd(self, t1, t2):
        return amplitudes_from_rccsd(t1, t2)

    to_gpu = lib.to_gpu

CCSD = UCCSD


class _ChemistsERIs(ccsd._ChemistsERIs):
    def __init__(self, mol=None):
        ccsd._ChemistsERIs.__init__(self, mol)
        self.OOOO = None
        self.OVOO = None
        self.OVOV = None
        self.OOVV = None
        self.OVVO = None
        self.OVVV = None
        self.VVVV = None

        self.ooOO = None
        self.ovOO = None
        self.ovOV = None
        self.ooVV = None
        self.ovVO = None
        self.ovVV = None
        self.vvVV = None

        self.OVoo = None
        self.OOvv = None
        self.OVvo = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,mo_idx[0]], mo_coeff[1][:,mo_idx[1]])
        # Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.focka = reduce(np.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(np.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
        self.fock = (self.focka, self.fockb)

        nocca, noccb = self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_ea = self.focka.diagonal().real
        mo_eb = self.fockb.diagonal().real
        self.mo_energy = (mo_ea, mo_eb)
        gap_a = abs(mo_ea[:nocca,None] - mo_ea[None,nocca:])
        gap_b = abs(mo_eb[:noccb,None] - mo_eb[None,noccb:])
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD',
                        gap_a, gap_b)
        return self

    def get_ovvv(self, *slices):
        return _get_ovvv_base(self.ovvv, *slices)

    def get_ovVV(self, *slices):
        return _get_ovvv_base(self.ovVV, *slices)

    def get_OVvv(self, *slices):
        return _get_ovvv_base(self.OVvv, *slices)

    def get_OVVV(self, *slices):
        return _get_ovvv_base(self.OVVV, *slices)

    def _contract_VVVV_t2(self, mycc, t2, vvvv_or_direct=False, out=None, verbose=None):
        if isinstance(vvvv_or_direct, np.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:
            vvvv = None
        else:
            vvvv = self.VVVV
        return ccsd._contract_vvvv_t2(mycc, self.mol, vvvv, t2, out, verbose)

    def _contract_vvVV_t2(self, mycc, t2, vvvv_or_direct=False, out=None, verbose=None):
        if isinstance(vvvv_or_direct, np.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:
            vvvv = None
        else:
            vvvv = self.vvVV
        return ccsd._contract_vvvv_t2(mycc, self.mol, vvvv, t2, out, verbose)

def _get_ovvv_base(ovvv, *slices):
    if ovvv.ndim == 3:
        ovw = np.asarray(ovvv[slices])
        nocc, nvir, nvir_pair = ovw.shape
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
        nvir1 = ovvv.shape[2]
        return ovvv.reshape(nocc,nvir,nvir1,nvir1)
    elif slices:
        return ovvv[slices]
    else:
        return ovvv

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    else:
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    if not callable(ao2mofn):
        ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
        vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
        idxa = np.tril_indices(nvira)
        idxb = np.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

        OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    return eris

def _make_df_eris_outcore(mycc, mo_coeff=None):
    assert mycc._scf.istype('UHF')
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nvira_pair = nvira*(nvira+1)//2
    nvirb_pair = nvirb*(nvirb+1)//2
    naux = mycc._scf.with_df.get_naoaux()

    # --- Three-center integrals
    # (L|aa)
    Loo = np.empty((naux,nocca,nocca))
    Lov = np.empty((naux,nocca,nvira))
    Lvo = np.empty((naux,nvira,nocca))
    Lvv = np.empty((naux,nvira_pair))
    # (L|bb)
    LOO = np.empty((naux,noccb,noccb))
    LOV = np.empty((naux,noccb,nvirb))
    LVO = np.empty((naux,nvirb,noccb))
    LVV = np.empty((naux,nvirb_pair))
    p1 = 0
    oa, va = np.s_[:nocca], np.s_[nocca:]
    ob, vb = np.s_[:noccb], np.s_[noccb:]
    # Transform three-center integrals to MO basis
    einsum = lib.einsum
    for eri1 in mycc._scf.with_df.loop():
        eri1 = lib.unpack_tril(eri1).reshape(-1,nao,nao)
        # (L|aa)
        Lpq = einsum('Lab,ap,bq->Lpq', eri1, moa, moa)
        p0, p1 = p1, p1 + Lpq.shape[0]
        blk = np.s_[p0:p1]
        Loo[blk] = Lpq[:,oa,oa]
        Lov[blk] = Lpq[:,oa,va]
        Lvo[blk] = Lpq[:,va,oa]
        Lvv[blk] = lib.pack_tril(Lpq[:,va,va].reshape(-1,nvira,nvira))
        # (L|bb)
        Lpq = einsum('Lab,ap,bq->Lpq', eri1, mob, mob)
        LOO[blk] = Lpq[:,ob,ob]
        LOV[blk] = Lpq[:,ob,vb]
        LVO[blk] = Lpq[:,vb,ob]
        LVV[blk] = lib.pack_tril(Lpq[:,vb,vb].reshape(-1,nvirb,nvirb))
    Loo = Loo.reshape(naux,nocca*nocca)
    Lov = Lov.reshape(naux,nocca*nvira)
    Lvo = Lvo.reshape(naux,nocca*nvira)
    LOO = LOO.reshape(naux,noccb*noccb)
    LOV = LOV.reshape(naux,noccb*nvirb)
    LVO = LVO.reshape(naux,noccb*nvirb)

    # --- Four-center integrals
    dot = lib.ddot
    eris.feri1 = lib.H5TmpFile()
    # (aa|aa)
    eris.oooo = eris.feri1.create_dataset('oooo', (nocca,nocca,nocca,nocca), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocca,nocca,nvira,nvira), 'f8', chunks=(nocca,nocca,1,nvira))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocca,nvira,nocca,nocca), 'f8', chunks=(nocca,1,nocca,nocca))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocca,nvira,nvira,nocca), 'f8', chunks=(nocca,1,nvira,nocca))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocca,nvira,nocca,nvira), 'f8', chunks=(nocca,1,nocca,nvira))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocca,nvira,nvira_pair), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvira_pair,nvira_pair), 'f8')
    eris.oooo[:] = dot(Loo.T, Loo).reshape(nocca,nocca,nocca,nocca)
    eris.ovoo[:] = dot(Lov.T, Loo).reshape(nocca,nvira,nocca,nocca)
    eris.oovv[:] = lib.unpack_tril(dot(Loo.T, Lvv)).reshape(nocca,nocca,nvira,nvira)
    eris.ovvo[:] = dot(Lov.T, Lvo).reshape(nocca,nvira,nvira,nocca)
    eris.ovov[:] = dot(Lov.T, Lov).reshape(nocca,nvira,nocca,nvira)
    eris.ovvv[:] = dot(Lov.T, Lvv).reshape(nocca,nvira,nvira_pair)
    eris.vvvv[:] = dot(Lvv.T, Lvv)
    # (bb|bb)
    eris.OOOO = eris.feri1.create_dataset('OOOO', (noccb,noccb,noccb,noccb), 'f8')
    eris.OOVV = eris.feri1.create_dataset('OOVV', (noccb,noccb,nvirb,nvirb), 'f8', chunks=(noccb,noccb,1,nvirb))
    eris.OVOO = eris.feri1.create_dataset('OVOO', (noccb,nvirb,noccb,noccb), 'f8', chunks=(noccb,1,noccb,noccb))
    eris.OVVO = eris.feri1.create_dataset('OVVO', (noccb,nvirb,nvirb,noccb), 'f8', chunks=(noccb,1,nvirb,noccb))
    eris.OVOV = eris.feri1.create_dataset('OVOV', (noccb,nvirb,noccb,nvirb), 'f8', chunks=(noccb,1,noccb,nvirb))
    eris.OVVV = eris.feri1.create_dataset('OVVV', (noccb,nvirb,nvirb_pair), 'f8')
    eris.VVVV = eris.feri1.create_dataset('VVVV', (nvirb_pair,nvirb_pair), 'f8')
    eris.OOOO[:] = dot(LOO.T, LOO).reshape(noccb,noccb,noccb,noccb)
    eris.OVOO[:] = dot(LOV.T, LOO).reshape(noccb,nvirb,noccb,noccb)
    eris.OOVV[:] = lib.unpack_tril(dot(LOO.T, LVV)).reshape(noccb,noccb,nvirb,nvirb)
    eris.OVVO[:] = dot(LOV.T, LVO).reshape(noccb,nvirb,nvirb,noccb)
    eris.OVOV[:] = dot(LOV.T, LOV).reshape(noccb,nvirb,noccb,nvirb)
    eris.OVVV[:] = dot(LOV.T, LVV).reshape(noccb,nvirb,nvirb_pair)
    eris.VVVV[:] = dot(LVV.T, LVV)
    # (aa|bb)
    eris.ooOO = eris.feri1.create_dataset('ooOO', (nocca,nocca,noccb,noccb), 'f8')
    eris.ooVV = eris.feri1.create_dataset('ooVV', (nocca,nocca,nvirb,nvirb), 'f8', chunks=(nocca,nocca,1,nvirb))
    eris.ovOO = eris.feri1.create_dataset('ovOO', (nocca,nvira,noccb,noccb), 'f8', chunks=(nocca,1,noccb,noccb))
    eris.ovVO = eris.feri1.create_dataset('ovVO', (nocca,nvira,nvirb,noccb), 'f8', chunks=(nocca,1,nvirb,noccb))
    eris.ovOV = eris.feri1.create_dataset('ovOV', (nocca,nvira,noccb,nvirb), 'f8', chunks=(nocca,1,noccb,nvirb))
    eris.ovVV = eris.feri1.create_dataset('ovVV', (nocca,nvira,nvirb_pair), 'f8')
    eris.vvVV = eris.feri1.create_dataset('vvVV', (nvira_pair,nvirb_pair), 'f8')
    eris.ooOO[:] = dot(Loo.T, LOO).reshape(nocca,nocca,noccb,noccb)
    eris.ovOO[:] = dot(Lov.T, LOO).reshape(nocca,nvira,noccb,noccb)
    eris.ooVV[:] = lib.unpack_tril(dot(Loo.T, LVV)).reshape(nocca,nocca,nvirb,nvirb)
    eris.ovVO[:] = dot(Lov.T, LVO).reshape(nocca,nvira,nvirb,noccb)
    eris.ovOV[:] = dot(Lov.T, LOV).reshape(nocca,nvira,noccb,nvirb)
    eris.ovVV[:] = dot(Lov.T, LVV).reshape(nocca,nvira,nvirb_pair)
    eris.vvVV[:] = dot(Lvv.T, LVV)
    # (bb|aa)
    eris.OOvv = eris.feri1.create_dataset('OOvv', (noccb,noccb,nvira,nvira), 'f8', chunks=(noccb,noccb,1,nvira))
    eris.OVoo = eris.feri1.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), 'f8', chunks=(noccb,1,nocca,nocca))
    eris.OVvo = eris.feri1.create_dataset('OVvo', (noccb,nvirb,nvira,nocca), 'f8', chunks=(noccb,1,nvira,nocca))
    eris.OVvv = eris.feri1.create_dataset('OVvv', (noccb,nvirb,nvira_pair), 'f8')
    eris.OVoo[:] = dot(LOV.T, Loo).reshape(noccb,nvirb,nocca,nocca)
    eris.OOvv[:] = lib.unpack_tril(dot(LOO.T, Lvv)).reshape(noccb,noccb,nvira,nvira)
    eris.OVvo[:] = dot(LOV.T, Lvo).reshape(noccb,nvirb,nvira,nocca)
    eris.OVvv[:] = dot(LOV.T, Lvv).reshape(noccb,nvirb,nvira_pair)

    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    from pyscf.scf.uhf import UHF
    assert isinstance(mycc._scf, UHF)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    orboa = moa[:,:nocca]
    orbob = mob[:,:noccb]
    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]
    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocca,nocca,nocca,nocca), 'f8')
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocca,nvira,nocca,nocca), 'f8')
    eris.ovov = eris.feri.create_dataset('ovov', (nocca,nvira,nocca,nvira), 'f8')
    eris.oovv = eris.feri.create_dataset('oovv', (nocca,nocca,nvira,nvira), 'f8')
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocca,nvira,nvira,nocca), 'f8')
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocca,nvira,nvira*(nvira+1)//2), 'f8')
    #eris.vvvv = eris.feri.create_dataset('vvvv', (nvira,nvira,nvira,nvira), 'f8')
    eris.OOOO = eris.feri.create_dataset('OOOO', (noccb,noccb,noccb,noccb), 'f8')
    eris.OVOO = eris.feri.create_dataset('OVOO', (noccb,nvirb,noccb,noccb), 'f8')
    eris.OVOV = eris.feri.create_dataset('OVOV', (noccb,nvirb,noccb,nvirb), 'f8')
    eris.OOVV = eris.feri.create_dataset('OOVV', (noccb,noccb,nvirb,nvirb), 'f8')
    eris.OVVO = eris.feri.create_dataset('OVVO', (noccb,nvirb,nvirb,noccb), 'f8')
    eris.OVVV = eris.feri.create_dataset('OVVV', (noccb,nvirb,nvirb*(nvirb+1)//2), 'f8')
    #eris.VVVV = eris.feri.create_dataset('VVVV', (nvirb,nvirb,nvirb,nvirb), 'f8')
    eris.ooOO = eris.feri.create_dataset('ooOO', (nocca,nocca,noccb,noccb), 'f8')
    eris.ovOO = eris.feri.create_dataset('ovOO', (nocca,nvira,noccb,noccb), 'f8')
    eris.ovOV = eris.feri.create_dataset('ovOV', (nocca,nvira,noccb,nvirb), 'f8')
    eris.ooVV = eris.feri.create_dataset('ooVV', (nocca,nocca,nvirb,nvirb), 'f8')
    eris.ovVO = eris.feri.create_dataset('ovVO', (nocca,nvira,nvirb,noccb), 'f8')
    eris.ovVV = eris.feri.create_dataset('ovVV', (nocca,nvira,nvirb*(nvirb+1)//2), 'f8')
    #eris.vvVV = eris.feri.create_dataset('vvVV', (nvira,nvira,nvirb,nvirb), 'f8')
    eris.OVoo = eris.feri.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), 'f8')
    eris.OOvv = eris.feri.create_dataset('OOvv', (noccb,noccb,nvira,nvira), 'f8')
    eris.OVvo = eris.feri.create_dataset('OVvo', (noccb,nvirb,nvira,nocca), 'f8')
    eris.OVvv = eris.feri.create_dataset('OVvv', (noccb,nvirb,nvira*(nvira+1)//2), 'f8')

    cput1 = logger.process_clock(), logger.perf_counter()
    mol = mycc.mol
    # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
    tmpf = lib.H5TmpFile()
    if nocca > 0:
        ao2mo.general(mol, (orboa,moa,moa,moa), tmpf, 'aa')
        buf = np.empty((nmoa,nmoa,nmoa))
        for i in range(nocca):
            lib.unpack_tril(tmpf['aa'][i*nmoa:(i+1)*nmoa], out=buf)
            eris.oooo[i] = buf[:nocca,:nocca,:nocca]
            eris.ovoo[i] = buf[nocca:,:nocca,:nocca]
            eris.ovov[i] = buf[nocca:,:nocca,nocca:]
            eris.oovv[i] = buf[:nocca,nocca:,nocca:]
            eris.ovvo[i] = buf[nocca:,nocca:,:nocca]
            eris.ovvv[i] = lib.pack_tril(buf[nocca:,nocca:,nocca:])
        del (tmpf['aa'])

    if noccb > 0:
        buf = np.empty((nmob,nmob,nmob))
        ao2mo.general(mol, (orbob,mob,mob,mob), tmpf, 'bb')
        for i in range(noccb):
            lib.unpack_tril(tmpf['bb'][i*nmob:(i+1)*nmob], out=buf)
            eris.OOOO[i] = buf[:noccb,:noccb,:noccb]
            eris.OVOO[i] = buf[noccb:,:noccb,:noccb]
            eris.OVOV[i] = buf[noccb:,:noccb,noccb:]
            eris.OOVV[i] = buf[:noccb,noccb:,noccb:]
            eris.OVVO[i] = buf[noccb:,noccb:,:noccb]
            eris.OVVV[i] = lib.pack_tril(buf[noccb:,noccb:,noccb:])
        del (tmpf['bb'])

    if nocca > 0:
        buf = np.empty((nmoa,nmob,nmob))
        ao2mo.general(mol, (orboa,moa,mob,mob), tmpf, 'ab')
        for i in range(nocca):
            lib.unpack_tril(tmpf['ab'][i*nmoa:(i+1)*nmoa], out=buf)
            eris.ooOO[i] = buf[:nocca,:noccb,:noccb]
            eris.ovOO[i] = buf[nocca:,:noccb,:noccb]
            eris.ovOV[i] = buf[nocca:,:noccb,noccb:]
            eris.ooVV[i] = buf[:nocca,noccb:,noccb:]
            eris.ovVO[i] = buf[nocca:,noccb:,:noccb]
            eris.ovVV[i] = lib.pack_tril(buf[nocca:,noccb:,noccb:])
        del (tmpf['ab'])

    if noccb > 0:
        buf = np.empty((nmob,nmoa,nmoa))
        ao2mo.general(mol, (orbob,mob,moa,moa), tmpf, 'ba')
        for i in range(noccb):
            lib.unpack_tril(tmpf['ba'][i*nmob:(i+1)*nmob], out=buf)
            eris.OVoo[i] = buf[noccb:,:nocca,:nocca]
            eris.OOvv[i] = buf[:noccb,nocca:,nocca:]
            eris.OVvo[i] = buf[noccb:,nocca:,:nocca]
            eris.OVvv[i] = lib.pack_tril(buf[noccb:,nocca:,nocca:])
        del (tmpf['ba'])
    buf = None
    cput1 = logger.timer_debug1(mycc, 'transforming oopq, ovpq', *cput1)

    if not mycc.direct:
        ao2mo.full(mol, orbva, eris.feri, dataname='vvvv')
        ao2mo.full(mol, orbvb, eris.feri, dataname='VVVV')
        ao2mo.general(mol, (orbva,orbva,orbvb,orbvb), eris.feri, dataname='vvVV')
        eris.vvvv = eris.feri['vvvv']
        eris.VVVV = eris.feri['VVVV']
        eris.vvVV = eris.feri['vvVV']
        cput1 = logger.timer_debug1(mycc, 'transforming vvvv', *cput1)

    return eris


def make_tau(t2, t1, r1, fac=1, out=None):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1aa = make_tau_aa(t2[0], t1a, r1a, fac, out)
    tau1bb = make_tau_aa(t2[2], t1b, r1b, fac, out)
    tau1ab = make_tau_ab(t2[1], t1, r1, fac, out)
    return tau1aa, tau1ab, tau1bb

def make_tau_aa(t2aa, t1a, r1a, fac=1, out=None):
    tau1aa = np.einsum('ia,jb->ijab', t1a, r1a)
    tau1aa-= np.einsum('ia,jb->jiab', t1a, r1a)
    tau1aa = tau1aa - tau1aa.transpose(0,1,3,2)
    tau1aa *= fac * .5
    tau1aa += t2aa
    return tau1aa

def make_tau_ab(t2ab, t1, r1, fac=1, out=None):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1ab = np.einsum('ia,jb->ijab', t1a, r1b)
    tau1ab+= np.einsum('ia,jb->ijab', r1a, t1b)
    tau1ab *= fac * .5
    tau1ab += t2ab
    return tau1ab

def _flops(nocc, nmo):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    # vvvv
    fp  = nocca**2 * nvira**4
    fp += noccb**2 * nvirb**4
    fp += nocca    * nvira**2 * noccb    * nvirb**2 * 2

    # ovvv
    fp += nocca**2 * nvira**3 * 2
    fp += nocca**2 * nvira**3 * 2
    fp += nocca**2 * nvira**3 * 2
    fp += nocca**3 * nvira**3 * 2
    fp += nocca**3 * nvira**2 * 2

    # OVVV
    fp += noccb**2 * nvirb**3 * 2
    fp += noccb**2 * nvirb**3 * 2
    fp += noccb**2 * nvirb**3 * 2
    fp += noccb**3 * nvirb**3 * 2
    fp += noccb**3 * nvirb**2 * 2

    # ovVV
    fp += nocca    * nvira * noccb    * nvirb**2 * 2
    fp += nocca**2 * nvira            * nvirb**2 * 2
    fp += nocca    * nvira * noccb    * nvirb**2 * 2
    fp += nocca    * nvira * noccb    * nvirb**2 * 2
    fp += nocca**2 * nvira * noccb    * nvirb**2 * 2
    fp += nocca**2 * nvira * noccb    * nvirb    * 2

    # OVvv
    fp += nocca    * nvira**2 * noccb    * nvirb * 2
    fp +=            nvira**2 * noccb**2 * nvirb * 2
    fp += nocca    * nvira**2 * noccb    * nvirb * 2
    fp += nocca    * nvira**2 * noccb    * nvirb * 2
    fp += nocca    * nvira**2 * noccb**2 * nvirb * 2
    fp += nocca    * nvira    * noccb**2 * nvirb * 2

    fp += nocca**4 * nvira    * 2
    fp += nocca**4 * nvira**2 * 2
    fp += nocca**4 * nvira**2 * 2
    fp += nocca**3 * nvira**2 * 2
    fp += nocca**3 * nvira**2 * 2
    fp += nocca**2 * nvira**3 * 2
    fp += nocca**3 * nvira**2 * 2
    fp += nocca**3 * nvira**3 * 2
    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += nocca**3 * nvira**2 * 2
    fp += nocca**3 * nvira**2 * 2

    fp += noccb**4 * nvirb    * 2
    fp += noccb**4 * nvirb**2 * 2
    fp += noccb**4 * nvirb**2 * 2
    fp += noccb**3 * nvirb**2 * 2
    fp += noccb**3 * nvirb**2 * 2
    fp += noccb**2 * nvirb**3 * 2
    fp += noccb**3 * nvirb**2 * 2
    fp += noccb**3 * nvirb**3 * 2
    fp += noccb**2 * nvirb**2 * nocca    * nvira    * 2
    fp += noccb**3 * nvirb**2 * 2
    fp += noccb**3 * nvirb**2 * 2

    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += nocca**2            * noccb    * nvirb**2 * 2
    fp += noccb**2 * nvirb    * nocca    * nvira    * 2
    fp += noccb**2 * nvirb    * nocca    * nvira    * 2
    fp += noccb**2            * nocca    * nvira**2 * 2

    fp += nocca**2            * noccb**2 * nvirb    * 2
    fp += nocca**2 * nvira    * noccb**2            * 2
    fp += nocca**2 * nvira    * noccb**2 * nvirb    * 2
    fp += nocca**2 * nvira    * noccb**2 * nvirb    * 2

    fp += nocca    * nvira**2 * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb    * nvirb**2 * 2
    fp += nocca    * nvira**2 * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb    * nvirb**2 * 2

    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb**2 * nvirb**2 * 2
    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb**2 * nvirb**2 * 2
    fp += nocca**2 * nvira    * noccb    * nvirb**2 * 2
    fp += nocca    * nvira**2 * noccb**2 * nvirb    * 2

    fp += nocca    * nvira    * noccb**2 * nvirb    * 2
    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb**2 * nvirb    * 2
    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += nocca**2            * noccb    * nvirb**2 * 2
    fp += nocca    * nvira**2 * noccb**2            * 2

    fp += nocca**3 * nvira**2 * 2
    fp += nocca**3 * nvira**2 * 2
    fp += noccb**3 * nvirb**2 * 2
    fp += noccb**3 * nvirb**2 * 2

    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += nocca**2            * noccb    * nvirb**2 * 2
    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += noccb**2 * nvirb    * nocca    * nvira    * 2
    fp += noccb**2            * nocca    * nvira**2 * 2
    fp += noccb**2 * nvirb    * nocca    * nvira    * 2

    fp += nocca**3 * nvira**3 * 2
    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += noccb**3 * nvirb**3 * 2
    fp += noccb**2 * nvirb**2 * nocca    * nvira    * 2

    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb**2 * nvirb**2 * 2
    fp += nocca    * nvira**2 * noccb**2 * nvirb    * 2
    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += nocca**2 * nvira**2 * noccb    * nvirb    * 2
    fp += nocca**2 * nvira    * noccb    * nvirb**2 * 2

    fp += nocca**2 * nvira**3 * 2
    fp += noccb**2 * nvirb**3 * 2
    fp += nocca    * nvira    * noccb    * nvirb**2 * 2
    fp += nocca    * nvira**2 * noccb    * nvirb    * 2
    fp += nocca**3 * nvira**2 * 2
    fp += noccb**3 * nvirb**2 * 2
    fp += nocca    * nvira    * noccb**2 * nvirb    * 2
    fp += nocca**2 * nvira    * noccb    * nvirb    * 2

    fp += nocca**2 * nvira**3 * 2
    fp += noccb**2 * nvirb**3 * 2
    fp += nocca**2 * nvira    * noccb    * nvirb    * 2
    fp += nocca    * nvira    * noccb**2 * nvirb    * 2
    return fp


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    # Freeze 1s electrons
    # also acceptable
    #frozen = 4 or [2,2]
    frozen = [[0,1], [0,1]]
    ucc = UCCSD(mf, frozen=frozen)
    eris = ucc.ao2mo()
    ecc, t1, t2 = ucc.kernel(eris=eris)
    print(ecc - -0.3486987472235819)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol).run()

    mycc = UCCSD(mf)
    mycc.direct = True
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    print(mycc.ccsd_t() - -0.003060021865720902)

    e,v = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    e,v = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)

    e,v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
