#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Yu Jin <yjin@flatironinstitute.org>
#         Huanchen Zhai <hczhai.ok@gmail.com>
#

'''
UHF-CCSDT with full T3 storage.

T2 amplitudes are stored as t2aa, t2ab, and t2bb, where t2ab has the shape (nocca, nvira, noccb, nvirb).
This differs from the convention in `pyscf.cc.uccsd`, where t2ab is stored as (nocca, noccb, nvira, nvirb).

Equations derived from the GCCSDT equations in Shavitt and Bartlett, Many-Body Methods in Chemistry and Physics:
MBPT and Coupled-Cluster Theory, Cambridge University Press (2009). DOI: 10.1017/CBO9780511596834.
'''

import numpy as np
import numpy
import functools
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.mp2 import get_e_hf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.cc import uccsdt
from pyscf.cc.rccsdt import _einsum, run_diis, _finalize
from pyscf.cc.uccsdt import (update_t1_fock_eris_uhf, intermediates_t1t2_uhf, compute_r1r2_uhf,
                            antisymmetrize_r2_uhf_, r1r2_divide_e_uhf_, intermediates_t3_uhf, _PhysicistsERIs, _IMDS)
from pyscf import __config__


def r1r2_add_t3_uhf_(mycc, imds, r1, r2, t3):
    '''Add the T3 contributions to r1 and r2. T3 amplitudes are stored in full form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    t1_focka, t1_fockb = imds.t1_fock
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris
    t3aaa, t3aab, t3bba, t3bbb = t3

    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2
    einsum('mnef,imnaef->ia', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t3aaa, out=r1a, alpha=0.25, beta=1.0)
    einsum('nmfe,inafme->ia', t1_erisab[:nocca, :noccb, nocca:, noccb:], t3aab, out=r1a, alpha=1.0, beta=1.0)
    einsum('mnef,mnefia->ia', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t3bba, out=r1a, alpha=0.25, beta=1.0)

    einsum('mnef,imnaef->ia', t1_erisbb[:noccb, :noccb, noccb:, noccb:], t3bbb, out=r1b, alpha=0.25, beta=1.0)
    einsum('mnef,inafme->ia', t1_erisab[:nocca, :noccb, nocca:, noccb:], t3bba, out=r1b, alpha=1.0, beta=1.0)
    einsum('mnef,mnefia->ia', t1_erisaa[:nocca, :nocca, nocca:, nocca:], t3aab, out=r1b, alpha=0.25, beta=1.0)

    einsum("me,ijmabe->ijab", t1_focka[:nocca, nocca:], t3aaa, out=r2aa, alpha=0.25, beta=1.0)
    einsum("me,ijabme->ijab", t1_fockb[:noccb, noccb:], t3aab, out=r2aa, alpha=0.25, beta=1.0)
    einsum("bmef,ijmaef->ijab", t1_erisaa[nocca:, :nocca, nocca:, nocca:], t3aaa, out=r2aa, alpha=0.25, beta=1.0)
    einsum("bmef,ijaemf->ijab", t1_erisab[nocca:, :noccb, nocca:, noccb:], t3aab, out=r2aa, alpha=0.5, beta=1.0)
    einsum("mnje,imnabe->ijab", t1_erisaa[:nocca, :nocca, :nocca, nocca:], t3aaa, out=r2aa, alpha=-0.25, beta=1.0)
    einsum("mnje,imabne->ijab", t1_erisab[:nocca, :noccb, :nocca, noccb:], t3aab, out=r2aa, alpha=-0.5, beta=1.0)

    einsum("me,imaejb->iajb", t1_focka[:nocca, nocca:], t3aab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("me,jmbeia->iajb", t1_fockb[:noccb, noccb:], t3bba, out=r2ab, alpha=1.0, beta=1.0)
    einsum("mbfe,imafje->iajb", t1_erisab[:nocca, noccb:, nocca:, noccb:], t3aab, out=r2ab, alpha=1.0, beta=1.0)
    einsum("bmef,jmefia->iajb", t1_erisbb[noccb:, :noccb, noccb:, noccb:], t3bba, out=r2ab, alpha=0.5, beta=1.0)
    einsum("amef,imefjb->iajb", t1_erisaa[nocca:, :nocca, nocca:, nocca:], t3aab, out=r2ab, alpha=0.5, beta=1.0)
    einsum("amef,jmbfie->iajb", t1_erisab[nocca:, :noccb, nocca:, noccb:], t3bba, out=r2ab, alpha=1.0, beta=1.0)
    einsum("nmej,inaemb->iajb", t1_erisab[:nocca, :noccb, nocca:, :noccb], t3aab, out=r2ab, alpha=-1.0, beta=1.0)
    einsum("mnje,mnbeia->iajb", t1_erisbb[:noccb, :noccb, :noccb, noccb:], t3bba, out=r2ab, alpha=-0.5, beta=1.0)
    einsum("mnie,mnaejb->iajb", t1_erisaa[:nocca, :nocca, :nocca, nocca:], t3aab, out=r2ab, alpha=-0.5, beta=1.0)
    einsum("mnie,jnbema->iajb", t1_erisab[:nocca, :noccb, :nocca, noccb:], t3bba, out=r2ab, alpha=-1.0, beta=1.0)

    einsum("me,ijmabe->ijab", t1_fockb[:noccb, noccb:], t3bbb, out=r2bb, alpha=0.25, beta=1.0)
    einsum("me,ijabme->ijab", t1_focka[:nocca, nocca:], t3bba, out=r2bb, alpha=0.25, beta=1.0)
    einsum("bmef,ijmaef->ijab", t1_erisbb[noccb:, :noccb, noccb:, noccb:], t3bbb, out=r2bb, alpha=0.25, beta=1.0)
    einsum("mbfe,ijaemf->ijab", t1_erisab[:nocca, noccb:, nocca:, noccb:], t3bba, out=r2bb, alpha=0.5, beta=1.0)
    einsum("mnje,imnabe->ijab", t1_erisbb[:noccb, :noccb, :noccb, noccb:], t3bbb, out=r2bb, alpha=-0.25, beta=1.0)
    einsum("nmej,imabne->ijab", t1_erisab[:nocca, :noccb, nocca:, :noccb], t3bba, out=r2bb, alpha=-0.5, beta=1.0)
    return [r1a, r1b], [r2aa, r2ab, r2bb]

def intermediates_t3_add_t3_uhf(mycc, imds, t3):
    '''Add the T3-dependent contributions to the T3 intermediates, with T3 stored in full form.'''
    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    nocca, noccb = mycc.nocc
    t1_erisaa, t1_erisab, t1_erisbb = imds.t1_eris
    t3aaa, t3aab, t3bba, t3bbb = t3

    W_ovoo, W_oVoO, W_OVOO = imds.W_ovoo, imds.W_oVoO, imds.W_OVOO
    W_vvvo, W_vVvO, W_VVVO = imds.W_vvvo, imds.W_vVvO, imds.W_VVVO
    W_vOoO, W_vVoV = imds.W_vOoO, imds.W_vVoV

    oaoavava = (slice(None, nocca), slice(None, nocca), slice(nocca, None), slice(nocca, None))
    oaobvavb = (slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None))
    obobvbvb = (slice(None, noccb), slice(None, noccb), slice(noccb, None), slice(noccb, None))
    einsum('lmde,lmkbec->bcdk', t1_erisaa[oaoavava], t3aaa, out=W_vvvo, alpha=-0.5, beta=1.0)
    einsum('lmde,lkbcme->bcdk', t1_erisab[oaobvavb], t3aab, out=W_vvvo, alpha=-1.0, beta=1.0)
    einsum('lmde,jmkdec->lcjk', t1_erisaa[oaoavava], t3aaa, out=W_ovoo, alpha=0.5, beta=1.0)
    einsum('lmde,jkdcme->lcjk', t1_erisab[oaobvavb], t3aab, out=W_ovoo, alpha=1.0, beta=1.0)
    einsum('lmde,lmkbec->bcdk', t1_erisbb[obobvbvb], t3bbb, out=W_VVVO, alpha=-0.5, beta=1.0)
    einsum('mled,lkbcme->bcdk', t1_erisab[oaobvavb], t3bba, out=W_VVVO, alpha=-1.0, beta=1.0)
    einsum('lmde,jmkdec->lcjk', t1_erisbb[obobvbvb], t3bbb, out=W_OVOO, alpha=0.5, beta=1.0)
    einsum('mled,jkdcme->lcjk', t1_erisab[oaobvavb], t3bba, out=W_OVOO, alpha=1.0, beta=1.0)
    einsum('lmde,lmbekc->bcdk', t1_erisaa[oaoavava], t3aab, out=W_vVvO, alpha=-0.5, beta=1.0)
    einsum('lmde,mkeclb->bcdk', t1_erisab[oaobvavb], t3bba, out=W_vVvO, alpha=-1.0, beta=1.0)
    einsum('lmde,jmdekc->lcjk', t1_erisaa[oaoavava], t3aab, out=W_oVoO, alpha=0.5, beta=1.0)
    einsum('lmde,mkecjd->lcjk', t1_erisab[oaobvavb], t3bba, out=W_oVoO, alpha=1.0, beta=1.0)
    einsum('mled,jmbelc->bcjd', t1_erisab[oaobvavb], t3aab, out=W_vVoV, alpha=-1.0, beta=1.0)
    einsum('lmde,mlecjb->bcjd', t1_erisbb[obobvbvb], t3bba, out=W_vVoV, alpha=-0.5, beta=1.0)
    einsum('mled,jmaekd->aljk', t1_erisab[oaobvavb], t3aab, out=W_vOoO, alpha=1.0, beta=1.0)
    einsum('lmde,mkedja->aljk', t1_erisbb[obobvbvb], t3bba, out=W_vOoO, alpha=0.5, beta=1.0)
    return imds

def compute_r3_uhf(mycc, imds, t2, t3):
    '''Compute r3 with full T3 amplitudes; r3 is returned in full form as well.'''
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    backend = mycc.einsum_backend
    einsum = functools.partial(_einsum, backend)
    t2aa, t2ab, t2bb = t2
    t3aaa, t3aab, t3bba, t3bbb = t3

    F_oo, F_OO, F_vv, F_VV = imds.F_oo, imds.F_OO, imds.F_vv, imds.F_VV
    W_oooo, W_oOoO, W_OOOO = imds.W_oooo, imds.W_oOoO, imds.W_OOOO
    W_ovoo, W_oVoO, W_OVOO = imds.W_ovoo, imds.W_oVoO, imds.W_OVOO
    W_vOoO, W_oVoV, W_vOvO, W_vVoV = imds.W_vOoO, imds.W_oVoV, imds.W_vOvO, imds.W_vVoV
    W_voov, W_vOoV, W_VoOv, W_VOOV = imds.W_voov, imds.W_vOoV, imds.W_VoOv, imds.W_VOOV
    W_vvvo, W_vVvO, W_VVVO = imds.W_vvvo, imds.W_vVvO, imds.W_VVVO
    W_vvvv, W_vVvV, W_VVVV = imds.W_vvvv, imds.W_vVvV, imds.W_VVVV

    r3aaa = np.empty_like(t3aaa)
    einsum("bcdk,ijad->ijkabc", W_vvvo, t2aa, out=r3aaa, alpha=0.25, beta=0.0)
    einsum("lcjk,ilab->ijkabc", W_ovoo, t2aa, out=r3aaa, alpha=-0.25, beta=1.0)
    einsum("cd,ijkabd->ijkabc", F_vv, t3aaa, out=r3aaa, alpha=1.0 / 12.0, beta=1.0)
    einsum("lk,ijlabc->ijkabc", F_oo, t3aaa, out=r3aaa, alpha=-1.0 / 12.0, beta=1.0)
    einsum("abde,ijkdec->ijkabc", W_vvvv, t3aaa, out=r3aaa, alpha=1.0 / 24.0, beta=1.0)
    einsum("lmij,lmkabc->ijkabc", W_oooo, t3aaa, out=r3aaa, alpha=1.0 / 24.0, beta=1.0)
    einsum("alid,ljkdbc->ijkabc", W_voov, t3aaa, out=r3aaa, alpha=0.25, beta=1.0)
    einsum("alid,jkbcld->ijkabc", W_vOoV, t3aab, out=r3aaa, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t3: r3aaa', *time1)

    r3bbb = np.empty_like(t3bbb)
    einsum("bcdk,ijad->ijkabc", W_VVVO, t2bb, out=r3bbb, alpha=0.25, beta=0.0)
    einsum("lcjk,ilab->ijkabc", W_OVOO, t2bb, out=r3bbb, alpha=-0.25, beta=1.0)
    einsum("cd,ijkabd->ijkabc", F_VV, t3bbb, out=r3bbb, alpha=1.0 / 12.0, beta=1.0)
    einsum("lk,ijlabc->ijkabc", F_OO, t3bbb, out=r3bbb, alpha=-1.0 / 12.0, beta=1.0)
    einsum("abde,ijkdec->ijkabc", W_VVVV, t3bbb, out=r3bbb, alpha=1.0 / 24.0, beta=1.0)
    einsum("lmij,lmkabc->ijkabc", W_OOOO, t3bbb, out=r3bbb, alpha=1.0 / 24.0, beta=1.0)
    einsum("alid,ljkdbc->ijkabc", W_VOOV, t3bbb, out=r3bbb, alpha=0.25, beta=1.0)
    einsum("alid,jkbcld->ijkabc", W_VoOv, t3bba, out=r3bbb, alpha=0.25, beta=1.0)
    time1 = log.timer_debug1('t3: r3bbb', *time1)

    r3aab = np.empty_like(t3aab)
    einsum("bcdk,ijad->ijabkc", W_vVvO, t2aa, out=r3aab, alpha=0.5, beta=0.0)
    einsum("bcjd,iakd->ijabkc", W_vVoV, t2ab, out=r3aab, alpha=1.0, beta=1.0)
    einsum("abdi,jdkc->ijabkc", W_vvvo, t2ab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum("lcjk,ilab->ijabkc", W_oVoO, t2aa, out=r3aab, alpha=-0.5, beta=1.0)
    einsum("aljk,iblc->ijabkc", W_vOoO, t2ab, out=r3aab, alpha=1.0, beta=1.0)
    einsum("laij,lbkc->ijabkc", W_ovoo, t2ab, out=r3aab, alpha=0.5, beta=1.0)
    einsum("cd,ijabkd->ijabkc", F_VV, t3aab, out=r3aab, alpha=0.25, beta=1.0)
    einsum("ad,ijbdkc->ijabkc", F_vv, t3aab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum("lk,ijablc->ijabkc", F_OO, t3aab, out=r3aab, alpha=-0.25, beta=1.0)
    einsum("li,jlabkc->ijabkc", F_oo, t3aab, out=r3aab, alpha=0.5, beta=1.0)
    einsum("abde,ijdekc->ijabkc", W_vvvv, t3aab, out=r3aab, alpha=0.125, beta=1.0)
    einsum("bced,ijaekd->ijabkc", W_vVvV, t3aab, out=r3aab, alpha=0.5, beta=1.0)
    einsum("lmij,lmabkc->ijabkc", W_oooo, t3aab, out=r3aab, alpha=0.125, beta=1.0)
    einsum("lmik,ljabmc->ijabkc", W_oOoO, t3aab, out=r3aab, alpha=0.5, beta=1.0)
    einsum("alid,ljdbkc->ijabkc", W_voov, t3aab, out=r3aab, alpha=1.0, beta=1.0)
    einsum("alid,lkdcjb->ijabkc", W_vOoV, t3bba, out=r3aab, alpha=1.0, beta=1.0)
    einsum("lcid,ljabkd->ijabkc", W_oVoV, t3aab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum("aldk,ijdblc->ijabkc", W_vOvO, t3aab, out=r3aab, alpha=-0.5, beta=1.0)
    einsum("clkd,ijlabd->ijabkc", W_VoOv, t3aaa, out=r3aab, alpha=0.25, beta=1.0)
    einsum("clkd,ijabld->ijabkc", W_VOOV, t3aab, out=r3aab, alpha=0.25, beta=1.0)
    W_vvvo = imds.W_vvvo = None
    W_ovoo = imds.W_ovoo = None
    W_vvvv = imds.W_vvvv = None
    W_oooo = imds.W_oooo = None
    time1 = log.timer_debug1('t3: r3aab', *time1)

    r3bba = np.empty_like(t3bba)
    einsum("cbkd,ijad->ijabkc", W_vVoV, t2bb, out=r3bba, alpha=0.5, beta=0.0)
    einsum("cbdj,kdia->ijabkc", W_vVvO, t2ab, out=r3bba, alpha=1.0, beta=1.0)
    einsum("abdi,kcjd->ijabkc", W_VVVO, t2ab, out=r3bba, alpha=-0.5, beta=1.0)
    einsum("clkj,ilab->ijabkc", W_vOoO, t2bb, out=r3bba, alpha=-0.5, beta=1.0)
    einsum("lakj,lcib->ijabkc", W_oVoO, t2ab, out=r3bba, alpha=1.0, beta=1.0)
    einsum("laij,kclb->ijabkc", W_OVOO, t2ab, out=r3bba, alpha=0.5, beta=1.0)
    einsum("cd,ijabkd->ijabkc", F_vv, t3bba, out=r3bba, alpha=0.25, beta=1.0)
    einsum("ad,ijbdkc->ijabkc", F_VV, t3bba, out=r3bba, alpha=-0.5, beta=1.0)
    einsum("lk,ijablc->ijabkc", F_oo, t3bba, out=r3bba, alpha=-0.25, beta=1.0)
    einsum("li,jlabkc->ijabkc", F_OO, t3bba, out=r3bba, alpha=0.5, beta=1.0)
    einsum("abde,ijdekc->ijabkc", W_VVVV, t3bba, out=r3bba, alpha=0.125, beta=1.0)
    einsum("cbde,ijaekd->ijabkc", W_vVvV, t3bba, out=r3bba, alpha=0.5, beta=1.0)
    einsum("lmij,lmabkc->ijabkc", W_OOOO, t3bba, out=r3bba, alpha=0.125, beta=1.0)
    einsum("mlki,ljabmc->ijabkc", W_oOoO, t3bba, out=r3bba, alpha=0.5, beta=1.0)
    einsum("alid,ljdbkc->ijabkc", W_VOOV, t3bba, out=r3bba, alpha=1.0, beta=1.0)
    einsum("alid,lkdcjb->ijabkc", W_VoOv, t3aab, out=r3bba, alpha=1.0, beta=1.0)
    einsum("cldi,ljabkd->ijabkc", W_vOvO, t3bba, out=r3bba, alpha=-0.5, beta=1.0)
    einsum("lakd,ijdblc->ijabkc", W_oVoV, t3bba, out=r3bba, alpha=-0.5, beta=1.0)
    einsum("clkd,ijlabd->ijabkc", W_vOoV, t3bbb, out=r3bba, alpha=0.25, beta=1.0)
    einsum("clkd,ijabld->ijabkc", W_voov, t3bba, out=r3bba, alpha=0.25, beta=1.0)
    F_oo = imds.F_oo = None
    F_OO = imds.F_OO = None
    F_vv = imds.F_vv = None
    F_VV = imds.F_VV = None
    W_vVoV = imds.W_vVoV = None
    W_vVvO = imds.W_vVvO = None
    W_voov = imds.W_voov = None
    W_vOoV = imds.W_vOoV = None
    W_oVoO = imds.W_oVoO = None
    W_vOoO = imds.W_vOoO = None
    W_vVvV = imds.W_vVvV = None
    W_oOoO = imds.W_oOoO = None
    W_oVoV = imds.W_oVoV = None
    W_vOvO = imds.W_vOvO = None
    W_VoOv = imds.W_VoOv = None
    W_VOOV = imds.W_VOOV = None
    W_VVVV = imds.W_VVVV = None
    W_VVVO = imds.W_VVVO = None
    W_OVOO = imds.W_OVOO = None
    W_OOOO = imds.W_OOOO = None
    time1 = log.timer_debug1('t3: r3bba', *time1)
    return [r3aaa, r3aab, r3bba, r3bbb]

def antisymmetrize_r3_uhf_(r3):
    r3[0] = (r3[0] - r3[0].transpose(1, 0, 2, 3, 4, 5) - r3[0].transpose(0, 2, 1, 3, 4, 5)
    - r3[0].transpose(2, 1, 0, 3, 4, 5) + r3[0].transpose(1, 2, 0, 3, 4, 5) + r3[0].transpose(2, 0, 1, 3, 4, 5))
    r3[0] = (r3[0] - r3[0].transpose(0, 1, 2, 4, 3, 5) - r3[0].transpose(0, 1, 2, 3, 5, 4)
    - r3[0].transpose(0, 1, 2, 5, 4, 3) + r3[0].transpose(0, 1, 2, 4, 5, 3) + r3[0].transpose(0, 1, 2, 5, 3, 4))
    r3[1] -= r3[1].transpose(1, 0, 2, 3, 4, 5)
    r3[1] -= r3[1].transpose(0, 1, 3, 2, 4, 5)
    r3[2] -= r3[2].transpose(1, 0, 2, 3, 4, 5)
    r3[2] -= r3[2].transpose(0, 1, 3, 2, 4, 5)
    r3[3] = (r3[3] - r3[3].transpose(1, 0, 2, 3, 4, 5) - r3[3].transpose(0, 2, 1, 3, 4, 5)
    - r3[3].transpose(2, 1, 0, 3, 4, 5) + r3[3].transpose(1, 2, 0, 3, 4, 5) + r3[3].transpose(2, 0, 1, 3, 4, 5))
    r3[3] = (r3[3] - r3[3].transpose(0, 1, 2, 4, 3, 5) - r3[3].transpose(0, 1, 2, 3, 5, 4)
    - r3[3].transpose(0, 1, 2, 5, 4, 3) + r3[3].transpose(0, 1, 2, 4, 5, 3) + r3[3].transpose(0, 1, 2, 5, 3, 4))
    return r3

def r3_divide_e_uhf_(mycc, r3, mo_energy):
    nocca, noccb = r3[0].shape[0], r3[-1].shape[0]
    eia_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:] - mycc.level_shift
    eia_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:] - mycc.level_shift

    eijkabc_aaa = (eia_a[:, None, None, :, None, None] + eia_a[None, :, None, None, :, None]
                    + eia_a[None, None, :, None, None, :])
    r3[0] /= eijkabc_aaa
    eijkabc_aaa = None
    eijkabc_aab = (eia_a[:, None, :, None, None, None] + eia_a[None, :, None, :, None,  None]
                    + eia_b[None, None, None, None, :, :])
    r3[1] /= eijkabc_aab
    eijkabc_aab = None
    eijkabc_bba = (eia_b[:, None, :, None, None, None] + eia_b[None, :, None, :, None, None]
                    + eia_a[None, None, None, None, :, :])
    r3[2] /= eijkabc_bba
    eijkabc_bba = None
    eijkabc_bbb = (eia_b[:, None, None, :, None, None] + eia_b[None, :, None, None, :, None]
                    + eia_b[None, None, :, None, None, :])
    r3[3] /= eijkabc_bbb
    eijkabc_bbb = None
    return r3

def update_amps_uccsdt_(mycc, tamps, eris):
    '''Update UCCSDT amplitudes in place, with T3 amplitudes stored in full form.'''
    assert (isinstance(eris, _PhysicistsERIs))
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1, t2, t3 = tamps
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t3aaa, t3aab, t3bba, t3bbb = t3
    mo_energy = eris.mo_energy[:]

    imds = _IMDS()

    # t1 t2
    update_t1_fock_eris_uhf(mycc, imds, t1, eris)
    time1 = log.timer_debug1('t1t2: update fock and eris', *time0)
    intermediates_t1t2_uhf(mycc, imds, t2)
    time1 = log.timer_debug1('t1t2: update intermediates', *time1)
    r1, r2 = compute_r1r2_uhf(mycc, imds, t2)
    r1r2_add_t3_uhf_(mycc, imds, r1, r2, t3)
    time1 = log.timer_debug1('t1t2: compute r1 & r2', *time1)
    # antisymmetrization
    antisymmetrize_r2_uhf_(r2)
    time1 = log.timer_debug1('t1t2: antisymmetrize r2', *time1)
    # divide by eijkabc
    r1r2_divide_e_uhf_(mycc, r1, r2, mo_energy)
    (r1a, r1b), (r2aa, r2ab, r2bb) = r1, r2
    time1 = log.timer_debug1('t1t2: divide r1 & r2 by eia & eijab', *time1)

    res_norm = [np.linalg.norm(r1a), np.linalg.norm(r1b),
                np.linalg.norm(r2aa), np.linalg.norm(r2ab), np.linalg.norm(r2bb)]

    t1a += r1a
    t1b += r1b
    t2aa += r2aa
    t2ab += r2ab
    t2bb += r2bb
    time1 = log.timer_debug1('t1t2: update t1 & t2', *time1)
    time0 = log.timer_debug1('t1t2 total', *time0)

    # t3
    intermediates_t3_uhf(mycc, imds, t2)
    intermediates_t3_add_t3_uhf(mycc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    time1 = log.timer_debug1('t3: update intermediates', *time0)
    r3 = compute_r3_uhf(mycc, imds, t2, t3)
    imds = None
    time1 = log.timer_debug1('t3: compute r3', *time1)
    # antisymmetrization
    antisymmetrize_r3_uhf_(r3)
    time1 = log.timer_debug1('t3: antisymmetrize r3', *time1)
    # divide by eijkabc
    r3_divide_e_uhf_(mycc, r3, mo_energy)
    r3aaa, r3aab, r3bba, r3bbb = r3
    time1 = log.timer_debug1('t3: divide r3 by eijkabc', *time1)

    res_norm += [np.linalg.norm(r3aaa), np.linalg.norm(r3aab), np.linalg.norm(r3bba), np.linalg.norm(r3bbb)]

    t3aaa += r3aaa
    r3aaa = None
    t3bbb += r3bbb
    r3bbb = None
    t3aab += r3aab
    r3aab = None
    t3bba += r3bba
    r3bba = None
    t3 = (t3aaa, t3aab, t3bba, t3bbb)
    time1 = log.timer_debug1('t3: update t3', *time1)
    time0 = log.timer_debug1('t3 total', *time0)

    tamps = [t1, t2, t3]
    return res_norm


class UCCSDT(uccsdt.UCCSDT):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        uccsdt.UCCSDT.__init__(self, mf, frozen, mo_coeff, mo_occ)

    do_tri_max_t = property(lambda self: False)

    update_amps_ = update_amps_uccsdt_


if __name__ == "__main__":

    from pyscf import gto, scf, df

    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis='631g', verbose=0, spin=2)
    mf = scf.UHF(mol)
    mf.level_shift = 0.0
    mf.conv_tol = 1e-14
    mf.max_cycle = 1000
    mf.kernel()

    ref_ecorr = -0.2413923134881862
    mycc = UCCSDT(mf, frozen=0)
    mycc.set_einsum_backend('numpy')
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.max_cycle = 100
    mycc.verbose = 5
    mycc.do_diis_max_t = True
    mycc.incore_complete = True
    ecorr, tamps = mycc.kernel()
    print("E_corr: % .10f    Ref: % .10f    Diff: % .10e"%(mycc.e_corr, ref_ecorr, mycc.e_corr - ref_ecorr))
    print()
