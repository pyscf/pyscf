#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Huanchen Zhai <hczhai.ok@gmail.com>
#

"""
Spin-adapted internally contracted Complete Active Space Second-Order Perturbation Theory (IC-CASPT2).

Reference:

K. Andersson and B. O. Roos, "Multiconfigurational Second-Order Perturbation Theory,"
in Modern Electronic Structure Theory, Part I, 1995, 55-109.

"""

from pyscf import lib, fci
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.mcscf import mc_ao2mo
import numpy as np, scipy, h5py
import functools, time

def _einsum(einsum_backend, script, *tensors, alpha=1.0, beta=0.0, out=None):
    assert einsum_backend in ['pytblis', 'numpy']
    if einsum_backend == 'pytblis':
        import pytblis
        ctr_func = (pytblis.contract, pytblis.transpose_add)[len(tensors) == 1]
        if out is None:
            out = ctr_func(script, *tensors, alpha=alpha)
        else:
            ctr_func(script, *tensors, out=out, alpha=alpha, beta=beta)
    elif einsum_backend == 'numpy':
        if out is None:
            out = alpha * np.einsum(script, *tensors, optimize='optimal')
        elif beta == 0.0:
            out[...] = alpha * np.einsum(script, *tensors, optimize='optimal')
        else:
            out[...] = alpha * np.einsum(script, *tensors, optimize='optimal') + beta * out
    else:
        raise ValueError(f"Unknown einsum_backend: {einsum_backend}")
    return out

def compute_lhs_amplitudes(ncore, ncas, nvir, tamps, rdms, fock, backend, max_memory, io_blksize=None):
    einsum = functools.partial(_einsum, backend)

    t_ccvv, t_aaav, t_acaa, t_aavv, t_ccaa, t_acav, t_acva, t_acvv, t_ccav = tamps

    t_aavv = t_aavv + t_aavv.transpose(1, 0, 3, 2)
    t_ccaa = t_ccaa + t_ccaa.transpose(1, 0, 3, 2)

    fcc, faa = fock[:ncore, :ncore], fock[ncore:ncore + ncas, ncore:ncore + ncas]
    fvv, fav = fock[ncore + ncas:, ncore + ncas:], fock[ncore:ncore + ncas, ncore + ncas:]
    fca, fcv = fock[:ncore, ncore:ncore + ncas], fock[:ncore, ncore + ncas:]
    assert ncore == 0 or np.max(np.abs(fcc - np.diag(np.diag(fcc)))) < 1e-8
    assert nvir == 0 or np.max(np.abs(fvv - np.diag(np.diag(fvv)))) < 1e-8

    newt_aaav = np.zeros(t_aaav.shape, dtype=t_aaav.dtype)
    newt_aavv = np.zeros(t_aavv.shape, dtype=t_aavv.dtype)
    newt_acaa = np.zeros(t_acaa.shape, dtype=t_acaa.dtype)
    newt_acvv = np.zeros(t_acvv.shape, dtype=t_acvv.dtype)
    newt_ccaa = np.zeros(t_ccaa.shape, dtype=t_ccaa.dtype)
    newt_ccav = np.zeros(t_ccav.shape, dtype=t_ccav.dtype)
    newt_acav = np.zeros(t_acav.shape, dtype=t_acav.dtype)
    newt_acva = np.zeros(t_acva.shape, dtype=t_acva.dtype)

    e_fock_ref = np.dot(rdms[0].ravel(), faa.ravel())

    # acav / acva / acvv -> aaav
    ht_av = einsum('iu,viua->va', fca, t_acav)
    einsum('iu,viau->va', fca, t_acva, alpha=-2.0, beta=1.0, out=ht_av)
    einsum('ia,uiba->ub', fcv, t_acvv, alpha=-2.0, beta=1.0, out=ht_av)
    einsum('ia,uiab->ub', fcv, t_acvv, alpha=1.0, beta=1.0, out=ht_av)
    einsum('va,svrt->rtsa', ht_av, rdms[1], alpha=1.0, beta=1.0, out=newt_aaav)
    ht_av = None

    # ccaa -> acaa // ccav -> acaa
    ht_ca = einsum('ju,ijur->ir', fca, t_ccaa)
    einsum('ir,ijrs->js', fca, t_ccaa, alpha=-2.0, beta=1.0, out=ht_ca)
    einsum('ja,jira->ir', fcv, t_ccav, alpha=1.0, beta=1.0, out=ht_ca)
    einsum('ia,jira->jr', fcv, t_ccav, alpha=-2.0, beta=1.0, out=ht_ca)
    einsum('js,ut->tjsu', ht_ca, rdms[0], alpha=-1.0, beta=1.0, out=newt_acaa)
    einsum('it,sr->rist', ht_ca, rdms[0], alpha=2.0, beta=1.0, out=newt_acaa)
    einsum('jv,surv->rjsu', ht_ca, rdms[1], alpha=-1.0, beta=1.0, out=newt_acaa)
    ht_ca = None

    # aavv / acav / acva / aaav -> aaav
    ht_aaav = einsum('iv,tira->tvra', fca, t_acav)
    einsum('tb,svab->vsta', fav, t_aavv, alpha=-1.0, beta=1.0, out=ht_aaav)
    einsum('iv,siat->vsta', fca, t_acva, alpha=1.0, beta=1.0, out=ht_aaav)
    einsum('tv,swva->swta', faa, t_aaav, alpha=-1.0, beta=1.0, out=ht_aaav)
    einsum('a,vtra->vtra', np.diag(fvv), t_aaav, alpha=-1.0, beta=1.0, out=ht_aaav)
    einsum('vtra->vtra', t_aaav, alpha=-1.0 * -e_fock_ref, beta=1.0, out=ht_aaav)
    einsum('tvra,tvsu->sura', ht_aaav, rdms[1], alpha=1.0, beta=1.0, out=newt_aaav)
    einsum('twua,stwruv->rvsa', ht_aaav, rdms[2], alpha=1.0, beta=1.0, out=newt_aaav)
    ht_aaav = None

    # aaav / acvv / aavv -> aavv
    ht_aavv = einsum('ub,vsua->svab', fav, t_aaav)
    einsum('iu,siab->suab', fca, t_acvv, alpha=-1.0, beta=1.0, out=ht_aavv)
    einsum('b,suab->suab', np.diag(fvv), t_aavv, alpha=1.0, beta=1.0, out=ht_aavv)
    einsum('suab->suab', t_aavv, alpha=0.5 * -e_fock_ref, beta=1.0, out=ht_aavv)
    einsum('svab,svrt->rtab', ht_aavv, rdms[1], alpha=-1.0, beta=1.0, out=newt_aavv)
    ht_aavv = None

    # acaa -> acaa // acav -> acaa // acva -> acaa // ccaa -> acaa
    ht_acaa = einsum('rs,vits->vitr', faa, t_acaa)
    einsum('ru,vius->virs', faa, t_acaa, alpha=1.0, beta=1.0, out=ht_acaa)
    einsum('i,uirs->uirs', np.diag(fcc), t_acaa, alpha=-1.0, beta=1.0, out=ht_acaa)
    einsum('uirs->uirs', t_acaa, alpha=-e_fock_ref, beta=1.0, out=ht_acaa)
    einsum('ra,uisa->uisr', fav, t_acav, alpha=1.0, beta=1.0, out=ht_acaa)
    einsum('ra,uias->uirs', fav, t_acva, alpha=1.0, beta=1.0, out=ht_acaa)
    einsum('ju,ijrs->uisr', fca, t_ccaa, alpha=-1.0, beta=1.0, out=ht_acaa)
    einsum('vitr,vu->uitr', ht_acaa, rdms[0], alpha=-2.0, beta=1.0, out=newt_acaa)
    einsum('visr,vt->tirs', ht_acaa, rdms[0], alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('rist,rvsu->uitv', ht_acaa, rdms[1], alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('witv,swrv->rist', ht_acaa, rdms[1], alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('tirw,tusw->siru', ht_acaa, rdms[1], alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('tiuv,stru->risv', ht_acaa, rdms[1], alpha=-2.0, beta=1.0, out=newt_acaa)
    einsum('tiux,stvrux->risv', ht_acaa, rdms[2], alpha=1.0, beta=1.0, out=newt_acaa)
    ht_acaa = None

    # acav -> acav // acva -> acav // acaa -> acav // acvv -> acav // ccav -> acav
    ht_acav = einsum('rt,uita->uira', faa, t_acav, alpha=-2.0)
    einsum('a,tira->tira', np.diag(fvv), t_acav, alpha=-2.0, beta=1.0, out=ht_acav)
    einsum('i,tira->tira', np.diag(fcc), t_acav, alpha=2.0, beta=1.0, out=ht_acav)
    einsum('uira->uira', t_acav, alpha=-2.0 * -e_fock_ref, beta=1.0, out=ht_acav)
    einsum('uiar->uira', t_acva, alpha=-e_fock_ref, beta=1.0, out=ht_acav)
    einsum('rt,uiat->uira', faa, t_acva, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('a,tiar->tira', np.diag(fvv), t_acva, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('i,tiar->tira', np.diag(fcc), t_acva, alpha=-1.0, beta=1.0, out=ht_acav)
    einsum('ra,uisr->uisa', fav, t_acaa, alpha=-2.0, beta=1.0, out=ht_acav)
    einsum('ta,uitr->uira', fav, t_acaa, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('rb,tiba->tira', fav, t_acvv, alpha=-2.0, beta=1.0, out=ht_acav)
    einsum('rb,tiab->tira', fav, t_acvv, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('jt,jira->tira', fca, t_ccav, alpha=2.0, beta=1.0, out=ht_acav)
    einsum('jt,ijra->tira', fca, t_ccav, alpha=-1.0, beta=1.0, out=ht_acav)
    einsum('uira,us->sira', ht_acav, rdms[0], alpha=1.0, beta=1.0, out=newt_acav)
    einsum('viua,svru->risa', ht_acav, rdms[1], alpha=1.0, beta=1.0, out=newt_acav)
    ht_acav = None

    # t_acav / acaa / acvv / ccav -> acva
    ht_acav = einsum('rt,uita->uira', faa, t_acav)
    einsum('a,tira->tira', np.diag(fvv), t_acav, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('i,tira->tira', np.diag(fcc), t_acav, alpha=-1.0, beta=1.0, out=ht_acav)
    einsum('uira->uira', t_acav, alpha=-e_fock_ref, beta=1.0, out=ht_acav)
    einsum('ra,uisr->uisa', fav, t_acaa, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('rb,tiba->tira', fav, t_acvv, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('jt,jira->tira', fca, t_ccav, alpha=-1.0, beta=1.0, out=ht_acav)
    einsum('uira,us->siar', ht_acav, rdms[0], alpha=1.0, beta=1.0, out=newt_acva)
    einsum('viua,svru->rias', ht_acav, rdms[1], alpha=1.0, beta=1.0, out=newt_acva)
    ht_acav = None

    # t_acva / acaa / acvv / ccav -> acva
    ht_acav = einsum('rt,uiat->uira', faa, t_acva)
    einsum('a,tiar->tira', np.diag(fvv), t_acva, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('i,tiar->tira', np.diag(fcc), t_acva, alpha=-1.0, beta=1.0, out=ht_acav)
    einsum('uiar->uira', t_acva, alpha=-e_fock_ref, beta=1.0, out=ht_acav)
    einsum('ta,uitr->uira', fav, t_acaa, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('rb,tiab->tira', fav, t_acvv, alpha=1.0, beta=1.0, out=ht_acav)
    einsum('jt,ijra->tira', fca, t_ccav, alpha=-1.0, beta=1.0, out=ht_acav)
    einsum('uira,us->siar', ht_acav, rdms[0], alpha=-2.0, beta=1.0, out=newt_acva)
    einsum('siva,strv->riat', ht_acav, rdms[1], alpha=1.0, beta=1.0, out=newt_acva)
    ht_acav = None

    # ccaa -> ccaa
    ht_ccaa = einsum('st,ijrt->ijrs', faa, t_ccaa)
    einsum('ijrs->ijrs', t_ccaa, alpha=0.5 * -e_fock_ref, beta=1.0, out=ht_ccaa)
    einsum('ijsr->ijrs', ht_ccaa.copy(), alpha=-1.0, beta=2.0, out=ht_ccaa)
    einsum('ijrs->ijrs', ht_ccaa, alpha=-2.0, beta=1.0, out=newt_ccaa)
    einsum('ijrt,st->ijrs', ht_ccaa, rdms[0], alpha=1.0, beta=1.0, out=newt_ccaa)
    einsum('ijst,rs->ijrt', ht_ccaa, rdms[0], alpha=1.0, beta=1.0, out=newt_ccaa)
    ht_ccaa = None

    # ccaa -> ccaa
    einsum('ijrs->ijrs', t_ccaa, alpha=-2.0 * e_fock_ref, beta=1.0, out=newt_ccaa)
    einsum('ijsr->ijrs', t_ccaa, alpha=e_fock_ref, beta=1.0, out=newt_ccaa)

    # ccaa / ccav -> ccaa
    ht_ccaa = -einsum('sa,ijra->ijrs', fav, t_ccav)
    einsum('ijsu,rtsu->ijrt', ht_ccaa, rdms[1], alpha=1.0, beta=1.0, out=newt_ccaa)
    einsum('j,ijrs->ijrs', np.diag(fcc), t_ccaa, alpha=1.0, beta=1.0, out=ht_ccaa)
    einsum('jirs->ijrs', ht_ccaa.copy(), alpha=-1.0, beta=2.0, out=ht_ccaa)
    einsum('ijrs->ijrs', ht_ccaa, alpha=2.0, beta=1.0, out=newt_ccaa)
    einsum('ijrt,st->ijrs', ht_ccaa, rdms[0], alpha=-1.0, beta=1.0, out=newt_ccaa)
    einsum('ijst,rs->ijrt', ht_ccaa, rdms[0], alpha=-1.0, beta=1.0, out=newt_ccaa)
    ht_ccaa = None

    # ccaa -> ccaa
    ht_ccaa = einsum('ijrs->ijrs', t_ccaa, alpha=0.5 * -e_fock_ref)
    einsum('vu,ijsu->ijsv', faa, t_ccaa, alpha=1.0, beta=1.0, out=ht_ccaa)
    einsum('j,ijsu->ijsu', np.diag(fcc), t_ccaa, alpha=-1.0, beta=1.0, out=ht_ccaa)
    einsum('ijsu,rtsu->ijrt', ht_ccaa, rdms[1], alpha=-1.0, beta=1.0, out=newt_ccaa)
    ht_ccaa = None

    # ccaa / ccav -> ccav
    ht_ccav = einsum('a,ijra->ijra', np.diag(fvv), t_ccav)
    einsum('j,ijra->ijra', np.diag(fcc), t_ccav, alpha=-1.0, beta=1.0, out=ht_ccav)
    einsum('j,jira->jira', np.diag(fcc), t_ccav, alpha=-1.0, beta=1.0, out=ht_ccav)
    einsum('rs,jisa->jira', faa, t_ccav, alpha=1.0, beta=1.0, out=ht_ccav)
    einsum('jira->jira', ht_ccav, alpha=-4.0, beta=1.0, out=newt_ccav)
    einsum('ijra->jira', ht_ccav, alpha=2.0, beta=1.0, out=newt_ccav)
    einsum('ta,ijst->ijsa', fav, t_ccaa, alpha=1.0, beta=1.0, out=ht_ccav)
    einsum('ijsa,rs->ijra', ht_ccav, rdms[0], alpha=2.0, beta=1.0, out=newt_ccav)
    einsum('jisa,rs->ijra', ht_ccav, rdms[0], alpha=-1.0, beta=1.0, out=newt_ccav)
    ht_ccav = None

    # aaav -> aaav // acaa -> acaa
    hp_aaaaaa = einsum('vx,ruxstw->ruvstw', faa, rdms[2])
    einsum('tiuw,stvruw->risv', t_acaa, hp_aaaaaa, alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('vw,stwyruvx->styrux', faa, rdms[3], alpha=1.0, beta=1.0, out=hp_aaaaaa)
    einsum('rusa,ruvstw->wtva', t_aaav, hp_aaaaaa, alpha=-1.0, beta=1.0, out=newt_aaav)
    einsum('risu,rtvsuw->wivt', t_acaa, hp_aaaaaa, alpha=1.0, beta=1.0, out=newt_acaa)
    hp_aaaaaa = None

    # aaav -> aaav // aavv -> aavv // acaa -> acaa // acav -> acav // acva -> acav
    hp_aaaa = einsum('rs,suxrtw->uxtw', faa, rdms[2])
    einsum('xuva,uxtw->wtva', t_aaav, hp_aaaa, alpha=-1.0, beta=1.0, out=newt_aaav)
    einsum('uwab,uwtv->tvab', t_aavv, hp_aaaa, alpha=-0.5, beta=1.0, out=newt_aavv)
    einsum('uw,rwsv->rusv', faa, rdms[1], alpha=1.0, beta=1.0, out=hp_aaaa)
    einsum('uivt,usvr->rist', t_acaa, hp_aaaa, alpha=-2.0, beta=1.0, out=newt_acaa)
    einsum('risv,rxsw->wivx', t_acaa, hp_aaaa, alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('wivx,wsxr->risv', t_acaa, hp_aaaa, alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('xivu,xtwu->wivt', t_acaa, hp_aaaa, alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('risa,rtsu->uita', t_acav, hp_aaaa, alpha=-2.0, beta=1.0, out=newt_acav)
    einsum('uiav,urvs->sira', t_acva, hp_aaaa, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('risa,rtsu->uiat', t_acav, hp_aaaa, alpha=1.0, beta=1.0, out=newt_acva)
    einsum('siau,stru->riat', t_acva, hp_aaaa, alpha=1.0, beta=1.0, out=newt_acva)
    einsum('tv,rvsu->rtsu', faa, rdms[1], alpha=1.0, beta=1.0, out=hp_aaaa)
    einsum('ijuw,tvuw->ijtv', t_ccaa, hp_aaaa, alpha=-0.5, beta=1.0, out=newt_ccaa)
    hp_aaaa = None

    # acaa -> acaa // acav / acva -> acav / acva // acvv -> acvv // ccav -> ccav
    hp_aa = einsum('rs,swrv->wv', faa, rdms[1])
    einsum('witu,wv->viut', t_acaa, hp_aa, alpha=1.0, beta=1.0, out=newt_acaa)
    einsum('wiur,wv->viur', t_acaa, hp_aa, alpha=-2.0, beta=1.0, out=newt_acaa)
    einsum('vita,vu->uita', t_acav, hp_aa, alpha=-2.0, beta=1.0, out=newt_acav)
    einsum('viat,vu->uita', t_acva, hp_aa, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('vita,vu->uiat', t_acav, hp_aa, alpha=1.0, beta=1.0, out=newt_acva)
    einsum('viar,vu->uiar', t_acva, hp_aa, alpha=-2.0, beta=1.0, out=newt_acva)
    einsum('uiab,ut->tiba', t_acvv, hp_aa, alpha=1.0, beta=1.0, out=newt_acvv)
    einsum('uiba,ut->tiba', t_acvv, hp_aa, alpha=-2.0, beta=1.0, out=newt_acvv)
    einsum('rt,ts->rs', faa, rdms[0], alpha=1.0, beta=1.0, out=hp_aa)
    einsum('rs->rs', rdms[0], alpha=-e_fock_ref, beta=1.0, out=hp_aa)
    einsum('jiua,tu->jita', t_ccav, hp_aa, alpha=2.0, beta=1.0, out=newt_ccav)
    einsum('ijua,tu->jita', t_ccav, hp_aa, alpha=-1.0, beta=1.0, out=newt_ccav)
    hp_aa = None

    # ccaa -> ccaa
    hp_aa = einsum('rt,ts->rs', faa, rdms[0])
    einsum('st,tusv->uv', faa, rdms[1], alpha=1.0, beta=1.0, out=hp_aa)
    einsum('ijrt,st->ijrs', t_ccaa, hp_aa, alpha=2.0, beta=1.0, out=newt_ccaa)
    einsum('ijtr,st->ijrs', t_ccaa, hp_aa, alpha=-1.0, beta=1.0, out=newt_ccaa)
    hp_aa = None

    # acaa -> acav // acav / acva -> acvv
    hp_aaav = einsum('va,rvsu->rsua', fav, rdms[1])
    einsum('rist,rsua->uita', t_acaa, hp_aaav, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('tiru,tsua->sira', t_acaa, hp_aaav, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('sitr,stua->uiar', t_acaa, hp_aaav, alpha=-2.0, beta=1.0, out=newt_acva)
    einsum('uitv,uvra->riat', t_acaa, hp_aaav, alpha=1.0, beta=1.0, out=newt_acva)
    einsum('risa,rstb->tiab', t_acav, hp_aaav, alpha=1.0, beta=1.0, out=newt_acvv)
    einsum('risa,rstb->tiba', t_acav, hp_aaav, alpha=-2.0, beta=1.0, out=newt_acvv)
    einsum('siat,srtb->riab', t_acva, hp_aaav, alpha=1.0, beta=1.0, out=newt_acvv)
    einsum('tibu,tura->riab', t_acva, hp_aaav, alpha=1.0, beta=1.0, out=newt_acvv)
    hp_aaav = None

    # aavv -> acvv
    hp_caaa = einsum('iu,stru->istr', fca, rdms[1])
    einsum('stab,istr->riab', t_aavv, hp_caaa, alpha=1.0, beta=1.0, out=newt_acvv)
    hp_caaa = None

    # ccaa -> ccav
    hp_av = einsum('ra->ra', fav, alpha=2.0)
    einsum('sb,rs->rb', fav, rdms[0], alpha=-1.0, beta=1.0, out=hp_av)
    einsum('sa,ijrs->ijra', hp_av, t_ccaa, alpha=-2.0, beta=1.0, out=newt_ccav)
    einsum('sa,ijrs->jira', hp_av, t_ccaa, alpha=1.0, beta=1.0, out=newt_ccav)
    hp_av = None

    # aaav -> acav / acva / acvv
    tp_av = einsum('rvsa,rvsu->ua', t_aaav, rdms[1])
    einsum('it,ua->uita', fca, tp_av, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('it,ua->uiat', fca, tp_av, alpha=-2.0, beta=1.0, out=newt_acva)
    einsum('ib,ta->tiba', fcv, tp_av, alpha=1.0, beta=1.0, out=newt_acvv)
    einsum('ia,tb->tiba', fcv, tp_av, alpha=-2.0, beta=1.0, out=newt_acvv)
    tp_av = None

    # acav / acva -> ccav
    tp_cv = einsum('sita,st->ia', t_acav, rdms[0], alpha=-2.0)
    einsum('sjat,st->ja', t_acva, rdms[0], alpha=1.0, beta=1.0, out=tp_cv)
    einsum('ja,ir->ijra', tp_cv, fca, alpha=2.0, beta=1.0, out=newt_ccav)
    einsum('ia,jr->ijra', tp_cv, fca, alpha=-1.0, beta=1.0, out=newt_ccav)
    tp_cv = None

    # acaa -> ccav / ccaa
    tp_ca = einsum('rist,rs->it', t_acaa, rdms[0], alpha=2.0)
    einsum('sirt,st->ir', t_acaa, rdms[0], alpha=-1.0, beta=1.0, out=tp_ca)
    einsum('rjsu,rtsu->jt', t_acaa, rdms[1], alpha=-1.0, beta=1.0, out=tp_ca)
    einsum('ir,ja->ijra', tp_ca, fcv, alpha=-2.0, beta=1.0, out=newt_ccav)
    einsum('it,ja->jita', tp_ca, fcv, alpha=1.0, beta=1.0, out=newt_ccav)
    einsum('ju,ir->ijru', tp_ca, fca, alpha=-2.0, beta=1.0, out=newt_ccaa)
    einsum('iu,jr->ijru', tp_ca, fca, alpha=1.0, beta=1.0, out=newt_ccaa)
    tp_ca = None

    # aaav -> aavv / acav
    tp_aaav = einsum('rusa,ruwstv->wtva', t_aaav, rdms[2])
    einsum('wtva,wb->tvab', tp_aaav, fav, alpha=-1.0, beta=1.0, out=newt_aavv)
    einsum('tura,tusv->rvsa', t_aaav, rdms[1], alpha=1.0, beta=1.0, out=tp_aaav)
    einsum('iw,swra->risa', fca, tp_aaav, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('iv,trva->riat', fca, tp_aaav, alpha=1.0, beta=1.0, out=newt_acva)
    tp_aaav = None

    # acaa -> acav / ccaa
    tp_acaa = einsum('tiuv,stwruv->risw', t_acaa, rdms[2])
    einsum('risw,wa->risa', tp_acaa, fav, alpha=1.0, beta=1.0, out=newt_acav)
    einsum('riws,wa->rias', tp_acaa, fav, alpha=1.0, beta=1.0, out=newt_acva)
    einsum('rist,rusv->vitu', t_acaa, rdms[1], alpha=1.0, beta=1.0, out=tp_acaa)
    einsum('sitr,sutv->viur', t_acaa, rdms[1], alpha=-2.0, beta=1.0, out=tp_acaa)
    einsum('uits,rusv->vitr', t_acaa, rdms[1], alpha=1.0, beta=1.0, out=tp_acaa)
    einsum('tirs,tu->uisr', t_acaa, rdms[0], alpha=1.0, beta=1.0, out=tp_acaa)
    einsum('tisr,tu->uisr', t_acaa, rdms[0], alpha=-2.0, beta=1.0, out=tp_acaa)
    einsum('wivt,jw->ijtv', tp_acaa, fca, alpha=-1.0, beta=1.0, out=newt_ccaa)
    tp_acaa = None

    # acaa -> ccaa
    tp_acaa = einsum('ujtv,rusv->tjrs', t_acaa, rdms[1])
    einsum('tjrs,is->ijrt', tp_acaa, fca, alpha=-1.0, beta=1.0, out=newt_ccaa)
    tp_acaa = None

    # ccaa -> ccav
    tp_ccaa = einsum('ijst,rust->ijru', t_ccaa, rdms[1])
    einsum('ijru,ua->ijra', tp_ccaa, fav, alpha=-1.0, beta=1.0, out=newt_ccav)
    tp_ccaa = None

    # acav / acva -> ccav
    tp_acav = einsum('sira,st->tira', t_acav, rdms[0])
    einsum('risa,rtsu->uita', t_acav, rdms[1], alpha=1.0, beta=1.0, out=tp_acav)
    einsum('uita,ju->jita', tp_acav, fca, alpha=2.0, beta=1.0, out=newt_ccav)
    einsum('uita,ju->ijta', tp_acav, fca, alpha=-1.0, beta=1.0, out=newt_ccav)
    tp_acav = einsum('siar,st->tira', t_acva, rdms[0])
    einsum('tira,jt->ijra', tp_acav, fca, alpha=2.0, beta=1.0, out=newt_ccav)
    einsum('tira,jt->jira', tp_acav, fca, alpha=-1.0, beta=1.0, out=newt_ccav)
    tp_acav = einsum('tias,rtsu->uira', t_acva, rdms[1])
    einsum('uira,ju->ijra', tp_acav, fca, alpha=-1.0, beta=1.0, out=newt_ccav)
    tp_acav = einsum('tias,rtus->uira', t_acva, rdms[1])
    einsum('sjra,is->ijra', tp_acav, fca, alpha=-1.0, beta=1.0, out=newt_ccav)
    tp_acav = None

    newt_aavv = newt_aavv + newt_aavv.transpose(1, 0, 3, 2)
    newt_ccaa = newt_ccaa + newt_ccaa.transpose(1, 0, 3, 2)

    # ccav / ccvv -> acav / acva
    ht_cv = einsum('jt,jita->ia', fca, t_ccav, alpha=2.0)
    einsum('jt,ijta->ia', fca, t_ccav, alpha=-1.0, beta=1.0, out=ht_cv)

    # acvv / acav / acva / ccvv -> acvv
    ht_acvv = einsum('sa,tisb->tiab', fav, t_acav)
    einsum('ra,tibr->tiba', fav, t_acva, alpha=1.0, beta=1.0, out=ht_acvv)
    einsum('tiba->tiba', t_acvv, alpha=-e_fock_ref, beta=1.0, out=ht_acvv)
    einsum('a,sica->sica', np.diag(fvv), t_acvv, alpha=1.0, beta=1.0, out=ht_acvv)
    einsum('b,siba->siba', np.diag(fvv), t_acvv, alpha=1.0, beta=1.0, out=ht_acvv)
    einsum('i,siba->siba', np.diag(fcc), t_acvv, alpha=-1.0, beta=1.0, out=ht_acvv)

    # acvv -> acvv / ccvv
    hp_acvv = einsum('riba,rs->siab', t_acvv, rdms[0], alpha=2.0)
    einsum('riab,rs->siab', t_acvv, rdms[0], alpha=-1.0, beta=1.0, out=hp_acvv)

    # acav / acva -> ccvv
    hp_cv = einsum('rias,rs->ia', t_acva, rdms[0])
    einsum('risa,rs->ia', t_acav, rdms[0], alpha=-2.0, beta=1.0, out=hp_cv)

    # ccav -> ccvv // ccvv -> ccav
    hp_av = einsum('ra->ra', fav, alpha=2.0)
    einsum('sb,rs->rb', fav, rdms[0], alpha=-1.0, beta=1.0, out=hp_av)

    if isinstance(t_ccvv, np.ndarray):

        t_ccvv = t_ccvv + t_ccvv.transpose(1, 0, 3, 2)
        newt_ccvv = np.zeros(t_ccvv.shape, dtype=t_ccvv.dtype)

        einsum('jb,ijab->ia', fcv, t_ccvv, alpha=2.0, beta=1.0, out=ht_cv)
        einsum('jb,ijba->ia', fcv, t_ccvv, alpha=-1.0, beta=1.0, out=ht_cv)

        einsum('js,ijab->siba', fca, t_ccvv, alpha=-1.0, beta=1.0, out=ht_acvv)

        # ccav -> ccvv // ccvv -> ccav
        einsum('rb,ijab->jira', hp_av, t_ccvv, alpha=-2.0, beta=1.0, out=newt_ccav)
        einsum('rb,jiab->jira', hp_av, t_ccvv, alpha=1.0, beta=1.0, out=newt_ccav)
        einsum('rb,jira->ijab', hp_av, t_ccav, alpha=-2.0, beta=1.0, out=newt_ccvv)
        einsum('ra,jirb->ijab', hp_av, t_ccav, alpha=1.0, beta=1.0, out=newt_ccvv)

        # acvv -> acvv / ccvv
        einsum('siab,js->ijab', hp_acvv, fca, alpha=1.0, beta=1.0, out=newt_ccvv)

        # acav / acva -> ccvv
        einsum('ia,jb->ijab', hp_cv, fcv, alpha=2.0, beta=1.0, out=newt_ccvv)
        einsum('ia,jb->ijba', hp_cv, fcv, alpha=-1.0, beta=1.0, out=newt_ccvv)

        # ccvv -> ccvv
        einsum('b,ijab->ijab', np.diag(fvv), t_ccvv, alpha=-4.0, beta=1.0, out=newt_ccvv)
        einsum('a,ijba->ijab', np.diag(fvv), t_ccvv, alpha=2.0, beta=1.0, out=newt_ccvv)
        einsum('j,ijab->ijab', np.diag(fcc), t_ccvv, alpha=4.0, beta=1.0, out=newt_ccvv)
        einsum('i,ijba->ijab', np.diag(fcc), t_ccvv, alpha=-2.0, beta=1.0, out=newt_ccvv)

        newt_ccvv = newt_ccvv + newt_ccvv.transpose(1, 0, 3, 2)
    else:
        blksize = max(1, io_blksize if io_blksize is not None else int((max(2000, max_memory - lib.current_memory()[0])
            * 0.95e6 / 8 / max(1, nvir) ** 2 / 8) ** 0.5))
        io_file = t_ccvv
        with lib.H5TmpFile(io_file.name, 'r+') as f5:
            ft_cvcv, fnewt_cvcv = f5['i_cvcv'], f5['cvcv']
            for ia, ib in ((i, min(ncore, i + blksize)) for i in range(0, ncore, blksize)):
                for ja, jb in ((j, min(ncore, j + blksize)) for j in range(ia, ncore, blksize)):
                    tmp = ft_cvcv[ia * nvir:ib * nvir, ja * nvir:jb * nvir].reshape(ib - ia, nvir, jb - ja, nvir)
                    tmp2 = tmp if ia == ja else ft_cvcv[ja * nvir:jb * nvir, ia * nvir:ib * nvir].reshape(jb - ja,
                        nvir, ib - ia, nvir)
                    t_ccvv = (tmp + tmp2.transpose(2, 3, 0, 1)).transpose(0, 2, 1, 3)
                    tmp, tmp2 = None, None
                    newt_ccvv = np.zeros(t_ccvv.shape, dtype=t_ccvv.dtype)

                    einsum('jb,ijab->ia', fcv[ja:jb], t_ccvv, alpha=2.0, beta=1.0, out=ht_cv[ia:ib])
                    einsum('jb,ijba->ia', fcv[ja:jb], t_ccvv, alpha=-1.0, beta=1.0, out=ht_cv[ia:ib])
                    einsum('js,ijab->siba', fca[ja:jb], t_ccvv, alpha=-1.0, beta=1.0, out=ht_acvv[:, ia:ib])
                    if ia != ja:
                        einsum('jb,jiba->ia', fcv[ia:ib], t_ccvv, alpha=2.0, beta=1.0, out=ht_cv[ja:jb])
                        einsum('jb,jiab->ia', fcv[ia:ib], t_ccvv, alpha=-1.0, beta=1.0, out=ht_cv[ja:jb])
                        einsum('js,jiba->siba', fca[ia:ib], t_ccvv, alpha=-1.0, beta=1.0, out=ht_acvv[:, ja:jb])

                    # ccav -> ccvv // ccvv -> ccav
                    einsum('rb,ijab->jira', hp_av, t_ccvv, alpha=-2.0, beta=1.0, out=newt_ccav[ja:jb, ia:ib])
                    einsum('rb,jiab->jira', hp_av, t_ccvv, alpha=1.0, beta=1.0, out=newt_ccav[ia:ib, ja:jb])
                    einsum('rb,jira->ijab', hp_av, t_ccav[ja:jb, ia:ib], alpha=-2.0, beta=1.0, out=newt_ccvv)
                    einsum('ra,jirb->ijab', hp_av, t_ccav[ja:jb, ia:ib], alpha=1.0, beta=1.0, out=newt_ccvv)
                    if ia != ja:
                        einsum('rb,jiba->jira', hp_av, t_ccvv, alpha=-2.0, beta=1.0, out=newt_ccav[ia:ib, ja:jb])
                        einsum('rb,ijba->jira', hp_av, t_ccvv, alpha=1.0, beta=1.0, out=newt_ccav[ja:jb, ia:ib])
                        einsum('rb,ijra->ijba', hp_av, t_ccav[ia:ib, ja:jb], alpha=-2.0, beta=1.0, out=newt_ccvv)
                        einsum('ra,ijrb->ijba', hp_av, t_ccav[ia:ib, ja:jb], alpha=1.0, beta=1.0, out=newt_ccvv)

                    # acvv / acav / acva -> ccvv
                    einsum('siab,js->ijab', hp_acvv[:, ia:ib], fca[ja:jb], alpha=1.0, beta=1.0, out=newt_ccvv)
                    einsum('ia,jb->ijab', hp_cv[ia:ib], fcv[ja:jb], alpha=2.0, beta=1.0, out=newt_ccvv)
                    einsum('ia,jb->ijba', hp_cv[ia:ib], fcv[ja:jb], alpha=-1.0, beta=1.0, out=newt_ccvv)
                    if ia != ja:
                        einsum('sjab,is->ijba', hp_acvv[:, ja:jb], fca[ia:ib], alpha=1.0, beta=1.0, out=newt_ccvv)
                        einsum('ja,ib->ijba', hp_cv[ja:jb], fcv[ia:ib], alpha=2.0, beta=1.0, out=newt_ccvv)
                        einsum('ja,ib->ijab', hp_cv[ja:jb], fcv[ia:ib], alpha=-1.0, beta=1.0, out=newt_ccvv)

                    # ccvv -> ccvv
                    einsum('b,ijab->ijab', np.diag(fvv), t_ccvv, alpha=-4.0, beta=1.0, out=newt_ccvv)
                    einsum('a,ijba->ijab', np.diag(fvv), t_ccvv, alpha=2.0, beta=1.0, out=newt_ccvv)
                    einsum('j,ijab->ijab', np.diag(fcc)[ja:jb], t_ccvv, alpha=4.0, beta=1.0, out=newt_ccvv)
                    einsum('i,ijba->ijab', np.diag(fcc)[ia:ib], t_ccvv, alpha=-2.0, beta=1.0, out=newt_ccvv)
                    if ia != ja:
                        einsum('a,ijab->ijab', np.diag(fvv), t_ccvv, alpha=-4.0, beta=1.0, out=newt_ccvv)
                        einsum('b,ijba->ijab', np.diag(fvv), t_ccvv, alpha=2.0, beta=1.0, out=newt_ccvv)
                        einsum('i,ijab->ijab', np.diag(fcc)[ia:ib], t_ccvv, alpha=4.0, beta=1.0, out=newt_ccvv)
                        einsum('j,ijba->ijab', np.diag(fcc)[ja:jb], t_ccvv, alpha=-2.0, beta=1.0, out=newt_ccvv)
                    if ia == ja:
                        newt_ccvv = newt_ccvv + newt_ccvv.transpose(1, 0, 3, 2)

                    newt_cvcv = newt_ccvv.transpose(0, 2, 1, 3).reshape((ib - ia) * nvir, (jb - ja) * nvir)
                    newt_ccvv = None

                    fnewt_cvcv[ia * nvir:ib * nvir, ja * nvir:jb * nvir] = newt_cvcv
                    if ia != ja:
                        fnewt_cvcv[ja * nvir:jb * nvir, ia * nvir:ib * nvir] = newt_cvcv.T
        newt_ccvv = io_file

    hp_acvv, hp_cv, hp_av = None, None, None

    # ccav / ccvv -> acav / acva
    einsum('ia,sr->risa', ht_cv, rdms[0], alpha=-2.0, beta=1.0, out=newt_acav)
    einsum('ia,sr->rias', ht_cv, rdms[0], alpha=1.0, beta=1.0, out=newt_acva)
    ht_cv = None

    # acvv / acav / acva / ccvv -> acvv
    einsum('tiab,tr->riab', ht_acvv, rdms[0], alpha=-2.0, beta=1.0, out=newt_acvv)
    einsum('tiba,tr->riab', ht_acvv, rdms[0], alpha=1.0, beta=1.0, out=newt_acvv)
    ht_acvv = None

    return newt_ccvv, newt_aaav, newt_acaa, newt_aavv, newt_ccaa, newt_acav, newt_acva, newt_acvv, newt_ccav

def compute_rhs_amplitudes(ncore, ncas, nvir, rdms, h1e, eris, backend):
    einsum = functools.partial(_einsum, backend)

    papa, ppaa, pcav, ccvv = eris
    hca, hcv, hav = h1e[:ncore, ncore:ncore + ncas], h1e[:ncore, ncore + ncas:], h1e[ncore:ncore + ncas, ncore + ncas:]
    g_aaav = ppaa[ncore:ncore + ncas, ncore + ncas:].transpose(2, 3, 0, 1)
    g_aavv = ppaa[ncore + ncas:, ncore + ncas:].transpose(2, 3, 0, 1)
    g_acaa = ppaa[ncore:ncore + ncas, :ncore]
    g_acav = papa[:ncore, :, ncore + ncas:].transpose(1, 0, 3, 2)
    g_acva = ppaa[ncore + ncas:, :ncore].transpose(2, 1, 0, 3)
    g_acvv = pcav[ncore + ncas:].transpose(2, 1, 0, 3)
    g_ccaa, g_ccav = ppaa[:ncore, :ncore], pcav[:ncore]

    t_aaav = einsum('twua,stwruv->rvsa', g_aaav, rdms[2])
    t_aaav = einsum('vsta,svru->urta', g_aaav, rdms[1], alpha=1.0, beta=1.0, out=t_aaav)
    t_aaav = einsum('ua,surt->rtsa', hav, rdms[1], alpha=1.0, beta=1.0, out=t_aaav)
    t_aavv = einsum('suab,surt->rtab', g_aavv, rdms[1], alpha=0.5)
    t_acaa = einsum('uirs,ut->tirs', g_acaa, rdms[0])
    t_acaa = einsum('ir,ts->sitr', hca, rdms[0], alpha=1.0, beta=1.0, out=t_acaa)
    t_acaa = einsum('uivt,surv->rist', g_acaa, rdms[1], alpha=1.0, beta=1.0, out=t_acaa)
    t_acaa = 2 * t_acaa - t_acaa.transpose(0, 1, 3, 2)
    t_acaa = einsum('uitv,surv->rist', g_acaa, rdms[1], alpha=-1.0, beta=1.0, out=t_acaa)
    t_acaa = einsum('uitv,suvr->rits', g_acaa, rdms[1], alpha=-1.0, beta=1.0, out=t_acaa)
    t_acaa = einsum('tiuw,stvruw->risv', g_acaa, rdms[2], alpha=-1.0, beta=1.0, out=t_acaa)
    t_acaa = einsum('iu,stru->rist', hca, rdms[1], alpha=-1.0, beta=1.0, out=t_acaa)
    t_acav = einsum('ia,sr->risa', hcv, rdms[0], alpha=2.0)
    tmp = 2 * g_acav - g_acva.transpose(0, 1, 3, 2)
    t_acav = einsum('tira,ts->sira', tmp, rdms[0], alpha=1.0, beta=1.0, out=t_acav)
    t_acav = einsum('tiua,stru->risa', tmp, rdms[1], alpha=1.0, beta=1.0, out=t_acav)
    tmp = None
    t_acva = einsum('ia,sr->rias', hcv, rdms[0], alpha=-1.0)
    t_acva = einsum('tiar,ts->siar', 2 * g_acva - g_acav.transpose(0, 1, 3, 2), rdms[0], beta=1.0, out=t_acva)
    t_acva = einsum('risa,rust->tiau', g_acav, rdms[1], alpha=-1.0, beta=1.0, out=t_acva)
    t_acva = einsum('rias,urst->tiau', g_acva, rdms[1], alpha=-1.0, beta=1.0, out=t_acva)
    t_acvv = einsum('siba,sr->riba', 2 * g_acvv - g_acvv.transpose(0, 1, 3, 2), rdms[0])
    t_ccaa = 2 * g_ccaa - g_ccaa.transpose(0, 1, 3, 2)
    t_ccaa = einsum('ijrt,st->ijrs', t_ccaa.copy(), rdms[0], alpha=-1.0, beta=1.0, out=t_ccaa)
    t_ccaa = einsum('ijsu,rtsu->ijrt', g_ccaa, rdms[1], alpha=0.5, beta=1.0, out=t_ccaa)
    t_ccav = 2 * g_ccav - g_ccav.transpose(1, 0, 2, 3)
    t_ccav = einsum('jisa,rs->jira', t_ccav.copy(), rdms[0], alpha=-1.0, beta=2.0, out=t_ccav)

    t_aavv = t_aavv + t_aavv.transpose(1, 0, 3, 2)
    t_ccaa = t_ccaa + t_ccaa.transpose(1, 0, 3, 2)

    if isinstance(ccvv, np.ndarray):
        t_ccvv = 4 * ccvv - 2 * ccvv.transpose(0, 1, 3, 2)
    else:
        with lib.H5TmpFile(ccvv.name, 'r+') as ft:
            g_cvcv = ft['cvcv']
            for i in range(ncore):
                tmp = g_cvcv[i * nvir:(i + 1) * nvir].reshape(nvir, ncore, nvir)
                g_cvcv[i * nvir:(i + 1) * nvir] = (4 * tmp - 2 * tmp.transpose(2, 1, 0)).reshape(nvir, ncore * nvir)
        t_ccvv = ccvv

    return t_ccvv, t_aaav, t_acaa, t_aavv, t_ccaa, t_acav, t_acva, t_acvv, t_ccav

def compute_ipea_shift(ncas, tamps, rdms, rdm_mats, ipea_shift):
    i = np.ogrid[:ncas]

    _, t_aaav, t_acaa, t_aavv, t_ccaa, t_acav, t_acva, t_acvv, t_ccav = tamps

    t_aavv = t_aavv + t_aavv.transpose(1, 0, 3, 2)
    t_ccaa = t_ccaa + t_ccaa.transpose(1, 0, 3, 2)

    dpdm1, dhdm1 = np.diag(rdms[0]), 2 - np.diag(rdms[0])
    ipea_p = 0.5 * ipea_shift * (dpdm1[:, None] + dpdm1[None, :])
    ipea_h = 0.5 * ipea_shift * (dhdm1[:, None] + dhdm1[None, :])
    ipea_pp, ipea_hh = ipea_p * (2.0 - np.identity(ncas)), ipea_h * (2.0 - np.identity(ncas))
    mat_acaa, mat_ccaa = rdm_mats

    mat = rdms[2].transpose(3, 5, 0, 1, 2, 4).copy()
    mat[:, :, i, :, :, i] += rdms[1].transpose(3, 2, 1, 0)
    s_aaav = np.diag(mat.reshape((ncas ** 3,) * 2)).reshape(ncas, ncas, ncas)
    newt_aaav = (0.5 * ipea_shift * dpdm1[None, None, :] * s_aaav)[:, :, :, None] * t_aaav
    newt_aaav += (ipea_h[:, :, None] * s_aaav)[:, :, :, None] * t_aaav

    s_acaa = np.diag(mat_acaa.reshape((ncas ** 3,) * 2)).reshape(ncas, ncas, ncas)
    newt_acaa = (0.5 * ipea_shift * dhdm1[:, None, None] * s_acaa)[:, None, :, :] * t_acaa
    newt_acaa += (ipea_p[None, :, :] * s_acaa)[:, None, :, :] * t_acaa

    mat = rdms[1].transpose(2, 3, 0, 1)
    s_aavv_p = np.diag((mat + mat.transpose(0, 1, 3, 2)).reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
    s_aavv_m = np.diag((mat - mat.transpose(0, 1, 3, 2)).reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
    newt_aavv = (ipea_hh * s_aavv_p)[:, :, None, None] * (t_aavv + t_aavv.transpose(1, 0, 2, 3))
    newt_aavv += (ipea_hh * s_aavv_m)[:, :, None, None] * (t_aavv - t_aavv.transpose(1, 0, 2, 3))
    s_aavv_p, s_aavv_m = None, None

    t_acvv = 2 * t_acvv - t_acvv.transpose(0, 1, 3, 2)
    newt_acvv = (0.5 * ipea_shift * dhdm1 * np.diag(rdms[0]))[:, None, None, None] * t_acvv

    s_ccaa_p = np.diag((mat_ccaa + mat_ccaa.transpose(1, 0, 2, 3)).reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
    s_ccaa_m = np.diag((mat_ccaa - mat_ccaa.transpose(1, 0, 2, 3)).reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
    newt_ccaa = (ipea_pp * s_ccaa_p)[None, None] * (t_ccaa + t_ccaa.transpose(0, 1, 3, 2))
    newt_ccaa += (ipea_pp * s_ccaa_m)[None, None] * (t_ccaa - t_ccaa.transpose(0, 1, 3, 2))
    s_ccaa_p, s_ccaa_m = None, None

    t_ccav = 2 * t_ccav - t_ccav.transpose(1, 0, 2, 3)
    newt_ccav = (0.5 * ipea_shift * dpdm1 * (2 - np.diag(rdms[0])))[None, None, :, None] * t_ccav

    mat = 2 * rdms[1].transpose(2, 0, 1, 3)
    mat[:, i, :, i] += 2 * rdms[0]
    s_acav = np.diag(mat.reshape((ncas ** 2,) * 2)).reshape(ncas, ncas)
    newt_acav = (0.5 * ipea_shift * (dhdm1[:, None] + dpdm1[None, :]) * s_acav)[:, None, :, None] * t_acav

    mat = -rdms[1].transpose(3, 0, 1, 2)
    mat[:, i, :, i] += 2 * rdms[0]
    s_acva = np.diag(mat.reshape((ncas ** 2,) * 2)).reshape(ncas, ncas)
    newt_acva = (0.5 * ipea_shift * (dhdm1[:, None] + dpdm1[None, :]) * s_acva)[:, None, None, :] * t_acva

    newt_ccvv = None

    return newt_ccvv, newt_aaav, newt_acaa, newt_aavv, newt_ccaa, newt_acav, newt_acva, newt_acvv, newt_ccav

def apply_lhs_preconditioner(ncore, vector, shapes, diag_es, diag_rs, io_file=None, vname=None, zname=None):
    (e_c, e_vv_p, e_vv_m, e_aaav, e_acaa, e_aavv_p, e_aavv_m, e_ccaa_p, e_ccaa_m, e_acav, e_acvv_p,
        e_acvv_m, e_ccav_p, e_ccav_m) = diag_es
    r_aaav, r_acaa, r_aavv_p, r_aavv_m, r_ccaa_p, r_ccaa_m, r_acav, r_acvv, r_ccav = diag_rs
    vectors = [v.reshape(s) for v, s in zip(np.split(vector, np.cumsum([a * b for a, b in shapes])[:-1]), shapes)]
    v_ccvv_p, v_ccvv_m, v_aaav, v_acaa, v_aavv_p, v_aavv_m, v_ccaa_p = vectors[0:7]
    v_ccaa_m, v_acav, v_acvv_p, v_acvv_m, v_ccav_p, v_ccav_m = vectors[7:13]

    if io_file is None or isinstance(io_file, np.ndarray):
        z_ccvv_p, z_ccvv_m = np.empty_like(v_ccvv_p), np.empty_like(v_ccvv_m)
        ip, im = 0, 0
        for i in range(ncore):
            z_ccvv_p[ip:ip + ncore - i] = v_ccvv_p[ip:ip + ncore - i] / (e_vv_p[None] - e_c[i] - e_c[i:, None])
            z_ccvv_m[im:im + i] = v_ccvv_m[im:im + i] / (e_vv_m[None] - e_c[i] - e_c[:i, None])
            ip, im = ip + ncore - i, im + i
    else:
        with lib.H5TmpFile(io_file.name, 'r+') as f5:
            v_ccvv_p, v_ccvv_m = f5[vname + '_p'], f5[vname + '_m']
            z_ccvv_p, z_ccvv_m = f5[zname + '_p'], f5[zname + '_m']
            ip, im = 0, 0
            for i in range(ncore):
                z_ccvv_p[ip:ip + ncore - i] = v_ccvv_p[ip:ip + ncore - i] / (e_vv_p[None] - e_c[i] - e_c[i:, None])
                z_ccvv_m[im:im + i] = v_ccvv_m[im:im + i] / (e_vv_m[None] - e_c[i] - e_c[:i, None])
                ip, im = ip + ncore - i, im + i
        v_ccvv_p, v_ccvv_m, z_ccvv_p, z_ccvv_m = None, None, np.zeros((0, )), np.zeros((0, ))

    z_aaav = (v_aaav @ r_aaav / e_aaav) @ r_aaav.T
    z_acaa = (v_acaa @ r_acaa / e_acaa) @ r_acaa.T
    z_aavv_p = (v_aavv_p @ r_aavv_p / e_aavv_p) @ r_aavv_p.T
    z_aavv_m = (v_aavv_m @ r_aavv_m / e_aavv_m) @ r_aavv_m.T
    z_ccaa_p = (v_ccaa_p @ r_ccaa_p / e_ccaa_p) @ r_ccaa_p.T
    z_ccaa_m = (v_ccaa_m @ r_ccaa_m / e_ccaa_m) @ r_ccaa_m.T
    z_acav = (v_acav @ r_acav / e_acav) @ r_acav.T
    z_acvv_p = (v_acvv_p @ r_acvv / e_acvv_p) @ r_acvv.T
    z_acvv_m = (v_acvv_m @ r_acvv / e_acvv_m) @ r_acvv.T
    z_ccav_p = (v_ccav_p @ r_ccav / e_ccav_p) @ r_ccav.T
    z_ccav_m = (v_ccav_m @ r_ccav / e_ccav_m) @ r_ccav.T

    return np.concatenate([v.ravel() for v in [z_ccvv_p, z_ccvv_m, z_aaav, z_acaa, z_aavv_p, z_aavv_m, z_ccaa_p,
        z_ccaa_m, z_acav, z_acvv_p, z_acvv_m, z_ccav_p, z_ccav_m]], axis=0)

def prepare_outcore_files(ncore, nvir, tmpdir=None):
    tmpdir = lib.param.TMPDIR if tmpdir is None else tmpdir
    io_file = lib.NamedTemporaryFile(dir=tmpdir)
    with lib.H5TmpFile(io_file.name, 'w') as f5:
        f5.create_dataset('cvcv', (ncore * nvir, ncore * nvir), 'f8')
        f5.create_dataset('i_cvcv', (ncore * nvir, ncore * nvir), 'f8')
        for name in ['b', 'x', 'r', 'p', 'z']:
            f5.create_dataset(name + '_p', (ncore * (ncore + 1) // 2, nvir * (nvir + 1) // 2), 'f8')
            f5.create_dataset(name + '_m', (ncore * (ncore - 1) // 2, nvir * (nvir - 1) // 2), 'f8')
    return io_file

def trans_e1_incore(ncore, ncas, nvir, mo, eri_ao):
    eri1 = ao2mo.incore.half_e1(eri_ao, (mo[:, :ncore + ncas], mo[:, ncore:]), compact=False)
    load_buf = lambda r0, r1: eri1[r0 * (ncas + nvir):r1 * (ncas + nvir)]
    ppaa, papa, pacv, cvcv = _trans(ncore, ncas, nvir, mo, load_buf)
    return ppaa, papa, pacv, cvcv.reshape((ncore, nvir) * 2)

def trans_e1_outcore(ncore, ncas, nvir, mo, mol, max_memory, cvcv_file, ioblk_size=256, tmpdir=None):
    tmpdir = lib.param.TMPDIR if tmpdir is None else tmpdir
    swapfile = lib.NamedTemporaryFile(dir=tmpdir)
    ao2mo.outcore.half_e1(mol, (mo[:, :ncore + ncas], mo[:, ncore:]), swapfile.name,
        max_memory=max_memory, ioblk_size=ioblk_size, compact=False)
    nao_pair = mo.shape[0] * (mo.shape[0] + 1) // 2
    fswap = h5py.File(swapfile.name, 'r')
    def load_buf(r0, r1):
        buf = np.empty(((r1 - r0) * (ncas + nvir), nao_pair))
        col0 = 0
        for ic in range(len(fswap['0'])):
            dat = fswap['0/%d' % ic]
            buf[:, col0:col0 + dat.shape[1]] = dat[r0 * (ncas + nvir):r1 * (ncas + nvir)]
            col0 += dat.shape[1]
        return buf
    ao_loc = np.array(mol.ao_loc_nr(), dtype=np.int32)
    with lib.H5TmpFile(cvcv_file.name, 'r+') as f5:
        ppaa, papa, pacv = _trans(ncore, ncas, nvir, mo, load_buf, f5['cvcv'], ao_loc)[:3]
    fswap.close()
    return ppaa, papa, pacv, cvcv_file

def _trans(ncore, ncas, nvir, mo, fload, cvcv=None, ao_loc=None):
    nocc, nav, nmo = ncore + ncas, ncas + nvir, mo.shape[1]
    if cvcv is None:
        cvcv = np.zeros((ncore * nvir, ncore * nvir))
    pacv = np.empty((nmo, ncas, ncore * nvir))
    aapp = np.empty((ncas, ncas, nmo * nmo))
    papa = np.empty((nmo, ncas, nmo * ncas))
    vcv = np.empty((nav, ncore * nvir))
    vpa = np.empty((nav, nmo * ncas))
    for i in range(ncore):
        buf = fload(i, i+1)
        _ao2mo.nr_e2(buf, mo, (0, ncore, nocc, nmo), aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        cvcv[i * nvir:(i + 1) * nvir], pacv[i] = vcv[ncas:], vcv[:ncas]
        _ao2mo.nr_e2(buf[:ncas], mo, (0, nmo, ncore, nocc), aosym='s4', mosym='s1', out=papa[i], ao_loc=ao_loc)
    for i in range(ncas):
        buf = fload(ncore + i, ncore + i + 1)
        _ao2mo.nr_e2(buf, mo, (0, ncore, nocc, nmo), aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        pacv[ncore:, i] = vcv
        _ao2mo.nr_e2(buf, mo, (0, nmo, ncore, nocc), aosym='s4', mosym='s1', out=vpa, ao_loc=ao_loc)
        papa[ncore:, i] = vpa
        _ao2mo.nr_e2(buf[:ncas], mo, (0, nmo, 0, nmo), aosym='s4', mosym='s1', out=aapp[i], ao_loc=ao_loc)
    ppaa = lib.transpose(aapp.reshape(ncas ** 2, nmo ** 2)).reshape(nmo, nmo, ncas, ncas)
    return ppaa, papa.reshape(nmo, ncas, nmo, ncas), pacv.reshape(nmo, ncas, ncore, nvir), cvcv

def trans_df_incore(ncore, ncas, nvir, mo, with_df):
    moa, moc, mov = mo[:, ncore:ncore + ncas], mo[:, :ncore], mo[:, ncore + ncas:]
    nmo = ncore + ncas + nvir
    ppaa = with_df.ao2mo([mo, mo, moa, moa], compact=False).reshape(nmo, nmo, ncas, ncas)
    papa = with_df.ao2mo([mo, moa, mo, moa], compact=False).reshape(nmo, ncas, nmo, ncas)
    pacv = with_df.ao2mo([mo, moa, moc, mov], compact=False).reshape(nmo, ncas, ncore, nvir)
    cvcv = with_df.ao2mo([moc, mov, moc, mov], compact=False).reshape(ncore, nvir, ncore, nvir)
    return ppaa, papa, pacv, cvcv

def trans_df_outcore(ncore, ncas, nvir, mo, with_df, max_memory, cvcv_file):
    naoaux = with_df.get_naoaux()
    moa, moc, mov = mo[:, ncore:ncore + ncas], mo[:, :ncore], mo[:, ncore + ncas:]
    nao, nmo = mo.shape
    with lib.H5TmpFile(cvcv_file.name, 'r+') as f5:
        cvcv = f5['cvcv']
        blksize = max(4, int(min(with_df.blockdim, (max_memory * 0.95e6 / 8 - naoaux * nmo * ncas) / 3 / nmo ** 2)))
        for k, eri1 in enumerate(with_df.loop(blksize)):
            eri1 = lib.unpack_tril(eri1).reshape(eri1.shape[0], nao, nao)
            lcv = np.einsum('xab,ac,bv->xcv', eri1, moc, mov, optimize='optimal')
            for i in range(ncore):
                vcv = np.einsum('xv,xcw->vcw', lcv[:, i], lcv, optimize='optimal').reshape((nvir, ncore * nvir))
                if k == 0:
                    cvcv[i * nvir:(i + 1) * nvir] = vcv
                else:
                    cvcv[i * nvir:(i + 1) * nvir] += vcv
    ppaa = with_df.ao2mo([mo, mo, moa, moa], compact=False).reshape(nmo, nmo, ncas, ncas)
    papa = with_df.ao2mo([mo, moa, mo, moa], compact=False).reshape(nmo, ncas, nmo, ncas)
    pacv = with_df.ao2mo([mo, moa, moc, mov], compact=False).reshape(nmo, ncas, ncore, nvir)
    return ppaa, papa, pacv, cvcv_file

def trans_integrals(mc, mo, rdms, method='incore', frozen=0):
    ncore, ncas, nvir = mc.ncore - frozen, mc.ncas, mo.shape[1] - mc.ncore - mc.ncas
    mo_nf = mo[:, frozen:]
    nmo = mo_nf.shape[1]
    mem_incore, mem_now = mc_ao2mo._mem_usage(ncore, ncas, nmo)[0], lib.current_memory()[0]
    do_incore = mem_incore + mem_now < mc.max_memory * 0.9 or mc.mol.incore_anyway
    max_mem = max(2000, mc.max_memory - mem_now)
    if method == 'incore' and getattr(mc, 'with_df', None) and mc._scf._eri is None and do_incore:
        ppaa, papa, pacv, cvcv = trans_df_incore(ncore, ncas, nvir, mo_nf, mc.with_df)
    elif method == 'incore' and mc._scf._eri is not None and do_incore:
        ppaa, papa, pacv, cvcv = trans_e1_incore(ncore, ncas, nvir, mo_nf, mc._scf._eri)
    else:
        io_file = prepare_outcore_files(ncore, nvir, tmpdir=None)
        if getattr(mc, 'with_df', None):
            ppaa, papa, pacv, cvcv = trans_df_outcore(ncore, ncas, nvir, mo_nf, mc.with_df, max_mem, io_file)
        else:
            ppaa, papa, pacv, cvcv = trans_e1_outcore(ncore, ncas, nvir, mo_nf, mc.mol, max_mem, io_file)
    papa, ppaa, pcav = [x.transpose(0, 2, 1, 3) for x in [ppaa, papa, pacv]]
    ccvv = cvcv.transpose(0, 2, 1, 3) if isinstance(cvcv, np.ndarray) else cvcv

    dm_core = np.dot(mo[:,  :ncore + frozen], mo[:,:ncore + frozen].T)
    vj, vk = mc._scf.get_jk(mc.mol, dm_core)
    h1e = mo_nf.T.conj() @ (mc._scf.get_hcore() + 2.0 * vj - vk) @ mo_nf
    fock = h1e.copy()
    fock += (papa.transpose(0, 2, 1, 3).reshape(nmo * nmo, ncas * ncas) @ rdms[0].ravel()).reshape(nmo, nmo)
    fock -= (ppaa.reshape(nmo * nmo, ncas * ncas) @ (0.5 * rdms[0].ravel())).reshape(nmo, nmo)
    e_ref = mc.energy_nuc() + np.dot(dm_core.ravel(), (2.0 * mc._scf.get_hcore() + 2.0 * vj - vk).T.ravel())
    e_ref += np.dot(rdms[0].ravel(), h1e[ncore:ncore + ncas, ncore:ncore + ncas].ravel())
    e_ref += 0.5 * np.dot(rdms[1].ravel(), ppaa[ncore:ncore + ncas, ncore:ncore + ncas].ravel())
    return e_ref, fock, h1e, (papa, ppaa, pcav, ccvv)

def disjoint_eigh(mat):
    nconn, conns = scipy.sparse.csgraph.connected_components(np.abs(mat) > 1E-14, directed=False)
    if nconn == 1:
        return np.linalg.eigh(mat)
    xw, xu = np.zeros(len(mat)), np.zeros_like(mat)
    for icn in range(nconn):
        midx = np.ogrid[:len(mat)][conns == icn]
        zw, zu = np.linalg.eigh(mat[midx[:, None], midx[None]])
        xw[midx], xu[midx[:, None], midx[None]] = zw, zu
    idx = np.argsort(xw)
    return xw[idx], xu[:, idx]

def compute_lhs_preconditioner(ncore, ncas, nvir, ortho_us, rdms, rdm_mats, ipea_shift, fock, backend):
    einsum = functools.partial(_einsum, backend)

    nvu, nvl, ncu, ncl = [x for nx in [nvir, ncore] for x in [nx * (nx + 1) // 2, nx * (nx - 1) // 2]]
    uca, ucb, uva, uvb = *np.triu_indices(ncore), *np.triu_indices(nvir)
    lca, lcb, lva, lvb = *np.tril_indices(ncore, k=-1), *np.tril_indices(nvir, k=-1)

    u_aaav, u_acaa, u_aavv_p, u_aavv_m, u_ccaa_p, u_ccaa_m, u_acav, u_acvv, u_ccav = ortho_us

    fcc, faa = fock[:ncore, :ncore], fock[ncore:ncore + ncas, ncore:ncore + ncas]
    fvv = fock[ncore + ncas:, ncore + ncas:]

    newv_aaav = np.zeros((ncas, ncas, ncas, u_aaav.shape[1]), dtype=u_aaav.dtype)
    newv_acaa = np.zeros((ncas, u_acaa.shape[1], ncas, ncas), dtype=u_acaa.dtype)
    newv_aavv = np.zeros((ncas, ncas, u_aavv_p.shape[1] + u_aavv_m.shape[1]), dtype=u_aavv_p.dtype)
    newv_ccaa = np.zeros((u_ccaa_p.shape[1] + u_ccaa_m.shape[1], ncas, ncas), dtype=u_ccaa_p.dtype)
    newv_acav = np.zeros((ncas, u_acav.shape[1], ncas), dtype=u_acav.dtype)
    newv_acva = np.zeros((ncas, u_acav.shape[1], ncas), dtype=u_acav.dtype)
    newv_acvv = np.zeros((ncas, u_acvv.shape[1]), dtype=u_acvv.dtype)
    newv_ccav = np.zeros((ncas, u_ccav.shape[1]), dtype=u_ccav.dtype)

    v_aaav = u_aaav.reshape(ncas, ncas, ncas, u_aaav.shape[1])
    v_acaa = u_acaa.reshape(ncas, ncas, ncas, u_acaa.shape[1]).transpose(0, 3, 1, 2)
    v_aavv_p = u_aavv_p.reshape(ncas, ncas, u_aavv_p.shape[1])
    v_aavv_m = u_aavv_m.reshape(ncas, ncas, u_aavv_m.shape[1])
    v_aavv = np.concatenate([v_aavv_p + v_aavv_p.transpose(1, 0, 2), v_aavv_m - v_aavv_m.transpose(1, 0, 2)], axis=2)
    v_ccaa_p = u_ccaa_p.reshape(ncas, ncas, u_ccaa_p.shape[1]).transpose(2, 0, 1)
    v_ccaa_m = u_ccaa_m.reshape(ncas, ncas, u_ccaa_m.shape[1]).transpose(2, 0, 1)
    v_ccaa = np.concatenate([v_ccaa_p + v_ccaa_p.transpose(0, 2, 1), v_ccaa_m - v_ccaa_m.transpose(0, 2, 1)], axis=0)
    v_acav = u_acav[:ncas ** 2].reshape(ncas, ncas, u_acav.shape[1]).transpose(0, 2, 1)
    v_acva = u_acav[ncas ** 2:].reshape(ncas, ncas, u_acav.shape[1]).transpose(0, 2, 1)
    v_acvv = u_acvv.reshape(ncas, u_acvv.shape[1])
    v_ccav = u_ccav.reshape(ncas, u_ccav.shape[1])

    e_fock_ref = np.dot(rdms[0].ravel(), faa.ravel())

    hv_acaa = einsum('rs,vits->vitr', faa, v_acaa)
    einsum('ru,vius->virs', faa, v_acaa, alpha=1.0, beta=1.0, out=hv_acaa)
    einsum('uirs->uirs', v_acaa, alpha=-e_fock_ref, beta=1.0, out=hv_acaa)
    einsum('vitr,vu->uitr', hv_acaa, rdms[0], alpha=-2.0, beta=1.0, out=newv_acaa)
    einsum('visr,vt->tirs', hv_acaa, rdms[0], alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('rist,rvsu->uitv', hv_acaa, rdms[1], alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('witv,swrv->rist', hv_acaa, rdms[1], alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('tirw,tusw->siru', hv_acaa, rdms[1], alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('tiuv,stru->risv', hv_acaa, rdms[1], alpha=-2.0, beta=1.0, out=newv_acaa)
    einsum('tiux,stvrux->risv', hv_acaa, rdms[2], alpha=1.0, beta=1.0, out=newv_acaa)
    hv_acaa = None

    hv_acav = einsum('rt,uit->uir', faa, v_acav, alpha=-2.0)
    einsum('uir->uir', v_acav, alpha=-2.0 * -e_fock_ref, beta=1.0, out=hv_acav)
    einsum('uir->uir', v_acva, alpha=-e_fock_ref, beta=1.0, out=hv_acav)
    einsum('rt,uit->uir', faa, v_acva, alpha=1.0, beta=1.0, out=hv_acav)
    einsum('uir,us->sir', hv_acav, rdms[0], alpha=1.0, beta=1.0, out=newv_acav)
    einsum('viu,svru->ris', hv_acav, rdms[1], alpha=1.0, beta=1.0, out=newv_acav)
    hv_acav = None

    hv_acav = einsum('rt,uit->uir', faa, v_acav)
    einsum('uir->uir', v_acav, alpha=-e_fock_ref, beta=1.0, out=hv_acav)
    einsum('uir,us->sir', hv_acav, rdms[0], alpha=1.0, beta=1.0, out=newv_acva)
    einsum('viu,svru->ris', hv_acav, rdms[1], alpha=1.0, beta=1.0, out=newv_acva)
    hv_acav = None

    hv_acav = einsum('rt,uit->uir', faa, v_acva)
    einsum('uir->uir', v_acva, alpha=-e_fock_ref, beta=1.0, out=hv_acav)
    einsum('uir,us->sir', hv_acav, rdms[0], alpha=-2.0, beta=1.0, out=newv_acva)
    einsum('siv,strv->rit', hv_acav, rdms[1], alpha=1.0, beta=1.0, out=newv_acva)
    hv_acav = None

    hv_acvv = einsum('ti->ti', v_acvv, alpha=-e_fock_ref)
    einsum('ti,tr->ri', hv_acvv, rdms[0], alpha=-1.0, beta=1.0, out=newv_acvv)
    hv_acvv = None

    hv_aaav = einsum('tv,swva->swta', faa, v_aaav, alpha=-1.0)
    einsum('vtra->vtra', v_aaav, alpha=-1.0 * -e_fock_ref, beta=1.0, out=hv_aaav)
    einsum('tvra,tvsu->sura', hv_aaav, rdms[1], alpha=1.0, beta=1.0, out=newv_aaav)
    einsum('twua,stwruv->rvsa', hv_aaav, rdms[2], alpha=1.0, beta=1.0, out=newv_aaav)
    hv_aaav = None

    hv_aavv = einsum('sua->sua', v_aavv, alpha=0.5 * -e_fock_ref)
    einsum('sva,svrt->rta', hv_aavv, rdms[1], alpha=-1.0, beta=1.0, out=newv_aavv)
    hv_aavv = None

    hv_ccaa = einsum('st,irt->irs', faa, v_ccaa)
    einsum('irs->irs', v_ccaa, alpha=0.5 * -e_fock_ref, beta=1.0, out=hv_ccaa)
    einsum('isr->irs', hv_ccaa.copy(), alpha=-1.0, beta=2.0, out=hv_ccaa)
    einsum('irs->irs', hv_ccaa, alpha=-2.0, beta=1.0, out=newv_ccaa)
    einsum('irt,st->irs', hv_ccaa, rdms[0], alpha=1.0, beta=1.0, out=newv_ccaa)
    einsum('ist,rs->irt', hv_ccaa, rdms[0], alpha=1.0, beta=1.0, out=newv_ccaa)
    hv_ccaa = None

    einsum('irs->irs', v_ccaa, alpha=-2.0 * e_fock_ref, beta=1.0, out=newv_ccaa)
    einsum('isr->irs', v_ccaa, alpha=e_fock_ref, beta=1.0, out=newv_ccaa)

    hv_ccaa = einsum('irs->irs', v_ccaa, alpha=0.5 * -e_fock_ref)
    einsum('vu,isu->isv', faa, v_ccaa, alpha=1.0, beta=1.0, out=hv_ccaa)
    einsum('isu,rtsu->irt', hv_ccaa, rdms[1], alpha=-1.0, beta=1.0, out=newv_ccaa)
    hv_ccaa = None

    hv_ccav = einsum('rs,sa->ra', faa, v_ccav)
    einsum('ra->ra', hv_ccav, alpha=-2.0, beta=1.0, out=newv_ccav)
    einsum('sa,rs->ra', hv_ccav, rdms[0], alpha=1.0, beta=1.0, out=newv_ccav)
    hv_ccav = None

    hp_aaaaaa = einsum('vx,ruxstw->ruvstw', faa, rdms[2])
    einsum('tiuw,stvruw->risv', v_acaa, hp_aaaaaa, alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('vw,stwyruvx->styrux', faa, rdms[3], alpha=1.0, beta=1.0, out=hp_aaaaaa)
    einsum('rusa,ruvstw->wtva', v_aaav, hp_aaaaaa, alpha=-1.0, beta=1.0, out=newv_aaav)
    einsum('risu,rtvsuw->wivt', v_acaa, hp_aaaaaa, alpha=1.0, beta=1.0, out=newv_acaa)
    hp_aaaaaa = None

    hp_aaaa = einsum('rs,suxrtw->uxtw', faa, rdms[2])
    einsum('xuva,uxtw->wtva', v_aaav, hp_aaaa, alpha=-1.0, beta=1.0, out=newv_aaav)
    einsum('uwa,uwtv->tva', v_aavv, hp_aaaa, alpha=-0.5, beta=1.0, out=newv_aavv)
    einsum('uw,rwsv->rusv', faa, rdms[1], alpha=1.0, beta=1.0, out=hp_aaaa)
    einsum('uivt,usvr->rist', v_acaa, hp_aaaa, alpha=-2.0, beta=1.0, out=newv_acaa)
    einsum('risv,rxsw->wivx', v_acaa, hp_aaaa, alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('wivx,wsxr->risv', v_acaa, hp_aaaa, alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('xivu,xtwu->wivt', v_acaa, hp_aaaa, alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('ris,rtsu->uit', v_acav, hp_aaaa, alpha=-2.0, beta=1.0, out=newv_acav)
    einsum('uiv,urvs->sir', v_acva, hp_aaaa, alpha=1.0, beta=1.0, out=newv_acav)
    einsum('ris,rtsu->uit', v_acav, hp_aaaa, alpha=1.0, beta=1.0, out=newv_acva)
    einsum('siu,stru->rit', v_acva, hp_aaaa, alpha=1.0, beta=1.0, out=newv_acva)
    einsum('tv,rvsu->rtsu', faa, rdms[1], alpha=1.0, beta=1.0, out=hp_aaaa)
    einsum('iuw,tvuw->itv', v_ccaa, hp_aaaa, alpha=-0.5, beta=1.0, out=newv_ccaa)
    hp_aaaa = None

    hp_aa = einsum('rs,swrv->wv', faa, rdms[1])
    einsum('witu,wv->viut', v_acaa, hp_aa, alpha=1.0, beta=1.0, out=newv_acaa)
    einsum('wiur,wv->viur', v_acaa, hp_aa, alpha=-2.0, beta=1.0, out=newv_acaa)
    einsum('vit,vu->uit', v_acav, hp_aa, alpha=-2.0, beta=1.0, out=newv_acav)
    einsum('vit,vu->uit', v_acva, hp_aa, alpha=1.0, beta=1.0, out=newv_acav)
    einsum('vit,vu->uit', v_acav, hp_aa, alpha=1.0, beta=1.0, out=newv_acva)
    einsum('vir,vu->uir', v_acva, hp_aa, alpha=-2.0, beta=1.0, out=newv_acva)
    einsum('ui,ut->ti', v_acvv, hp_aa, alpha=-1.0, beta=1.0, out=newv_acvv)
    einsum('rt,ts->rs', faa, rdms[0], alpha=1.0, beta=1.0, out=hp_aa)
    einsum('rs->rs', rdms[0], alpha=-e_fock_ref, beta=1.0, out=hp_aa)
    einsum('ua,tu->ta', v_ccav, hp_aa, alpha=1.0, beta=1.0, out=newv_ccav)
    hp_aa = None

    hp_aa = einsum('rt,ts->rs', faa, rdms[0])
    einsum('st,tusv->uv', faa, rdms[1], alpha=1.0, beta=1.0, out=hp_aa)
    einsum('irt,st->irs', v_ccaa, hp_aa, alpha=2.0, beta=1.0, out=newv_ccaa)
    einsum('itr,st->irs', v_ccaa, hp_aa, alpha=-1.0, beta=1.0, out=newv_ccaa)
    hp_aa = None

    if ipea_shift != 0.0:
        i = np.ogrid[:ncas]
        dpdm1, dhdm1 = np.diag(rdms[0]), 2 - np.diag(rdms[0])
        ipea_p = 0.5 * ipea_shift * (dpdm1[:, None] + dpdm1[None, :])
        ipea_h = 0.5 * ipea_shift * (dhdm1[:, None] + dhdm1[None, :])
        ipea_pp, ipea_hh = ipea_p * (2.0 - np.identity(ncas)), ipea_h * (2.0 - np.identity(ncas))
        mat_acaa, mat_ccaa = rdm_mats

        mat = rdms[2].transpose(3, 5, 0, 1, 2, 4).copy()
        mat[:, :, i, :, :, i] += rdms[1].transpose(3, 2, 1, 0)
        s_aaav = np.diag(mat.reshape((ncas ** 3,) * 2)).reshape(ncas, ncas, ncas)
        newv_aaav -= (0.5 * ipea_shift * dpdm1[None, None, :] * s_aaav)[:, :, :, None] * v_aaav
        newv_aaav -= (ipea_h[:, :, None] * s_aaav)[:, :, :, None] * v_aaav

        s_acaa = np.diag(mat_acaa.reshape((ncas ** 3,) * 2)).reshape(ncas, ncas, ncas)
        newv_acaa -= (0.5 * ipea_shift * dhdm1[:, None, None] * s_acaa)[:, None, :, :] * v_acaa
        newv_acaa -= (ipea_p[None, :, :] * s_acaa)[:, None, :, :] * v_acaa

        mat = rdms[1].transpose(2, 3, 0, 1)
        s_aavv_p = np.diag((mat + mat.transpose(0, 1, 3, 2)).reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
        s_aavv_m = np.diag((mat - mat.transpose(0, 1, 3, 2)).reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
        v_aavv_p, v_aavv_m = v_aavv[:, :, :u_aavv_p.shape[1]], v_aavv[:, :, u_aavv_p.shape[1]:]
        newv_aavv[..., :u_aavv_p.shape[1]] -= (ipea_hh * s_aavv_p)[..., None] * (v_aavv_p + v_aavv_p.transpose(1, 0, 2))
        newv_aavv[..., u_aavv_p.shape[1]:] -= (ipea_hh * s_aavv_m)[..., None] * (v_aavv_m - v_aavv_m.transpose(1, 0, 2))
        s_aavv_p, s_aavv_m, v_aavv_p, v_aavv_m = None, None, None, None

        newv_acvv -= (0.5 * ipea_shift * dhdm1 * np.diag(rdms[0]))[:, None] * v_acvv

        s_ccaa_p, s_ccaa_m = mat_ccaa + mat_ccaa.transpose(1, 0, 2, 3), mat_ccaa - mat_ccaa.transpose(1, 0, 2, 3)
        s_ccaa_p = np.diag(s_ccaa_p.reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
        s_ccaa_m = np.diag(s_ccaa_m.reshape((ncas ** 2,) * 2)).reshape(ncas, ncas) / 4
        v_ccaa_p, v_ccaa_m = v_ccaa[:u_ccaa_p.shape[1]], v_ccaa[u_ccaa_p.shape[1]:]
        newv_ccaa[:u_ccaa_p.shape[1]] -= (ipea_pp * s_ccaa_p)[None] * (v_ccaa_p + v_ccaa_p.transpose(0, 2, 1))
        newv_ccaa[u_ccaa_p.shape[1]:] -= (ipea_pp * s_ccaa_m)[None] * (v_ccaa_m - v_ccaa_m.transpose(0, 2, 1))
        s_ccaa_p, s_ccaa_m, v_ccaa_p, v_ccaa_m = None, None, None, None

        newv_ccav -= (0.5 * ipea_shift * dpdm1 * (2 - np.diag(rdms[0])))[:, None] * v_ccav

        mat = 2 * rdms[1].transpose(2, 0, 1, 3)
        mat[:, i, :, i] += 2 * rdms[0]
        s_acav = np.diag(mat.reshape((ncas ** 2,) * 2)).reshape(ncas, ncas)
        newv_acav -= (0.5 * ipea_shift * (dhdm1[:, None] + dpdm1[None, :]) * s_acav)[:, None, :] * v_acav

        mat = -rdms[1].transpose(3, 0, 1, 2)
        mat[:, i, :, i] += 2 * rdms[0]
        s_acva = np.diag(mat.reshape((ncas ** 2,) * 2)).reshape(ncas, ncas)
        newv_acva -= (0.5 * ipea_shift * (dhdm1[:, None] + dpdm1[None, :]) * s_acva)[:, None, :] * v_acva

    e_aaav, r_aaav = disjoint_eigh(newv_aaav.reshape(ncas ** 3, u_aaav.shape[1]).T @ u_aaav)
    e_aaav = np.diag(fvv)[:, None] - e_aaav[None, :]

    e_acaa, r_acaa = disjoint_eigh(newv_acaa.transpose(0, 2, 3, 1).reshape(ncas ** 3, u_acaa.shape[1]).T @ u_acaa)
    e_acaa = -np.diag(fcc)[:, None] - e_acaa[None, :]

    newv_aavv_p = newv_aavv[:, :, :u_aavv_p.shape[1]] + newv_aavv[:, :, :u_aavv_p.shape[1]].transpose(1, 0, 2)
    newv_aavv_m = newv_aavv[:, :, u_aavv_p.shape[1]:] - newv_aavv[:, :, u_aavv_p.shape[1]:].transpose(1, 0, 2)
    e_aavv_p, r_aavv_p = disjoint_eigh(newv_aavv_p.reshape(ncas ** 2, u_aavv_p.shape[1]).T @ u_aavv_p)
    e_aavv_p = (np.diag(fvv)[uva] + np.diag(fvv)[uvb])[:, None] - e_aavv_p[None, :]
    e_aavv_m, r_aavv_m = disjoint_eigh(newv_aavv_m.reshape(ncas ** 2, u_aavv_m.shape[1]).T @ u_aavv_m)
    e_aavv_m = (np.diag(fvv)[lva] + np.diag(fvv)[lvb])[:, None] - e_aavv_m[None, :]

    newv_ccaa_p = newv_ccaa[:u_ccaa_p.shape[1]] + newv_ccaa[:u_ccaa_p.shape[1]].transpose(0, 2, 1)
    newv_ccaa_m = newv_ccaa[u_ccaa_p.shape[1]:] - newv_ccaa[u_ccaa_p.shape[1]:].transpose(0, 2, 1)
    e_ccaa_p, r_ccaa_p = disjoint_eigh(newv_ccaa_p.reshape(u_ccaa_p.shape[1], ncas ** 2) @ u_ccaa_p)
    e_ccaa_p = -(np.diag(fcc)[uca] + np.diag(fcc)[ucb])[:, None] - e_ccaa_p[None, :]
    e_ccaa_m, r_ccaa_m = disjoint_eigh(newv_ccaa_m.reshape(u_ccaa_m.shape[1], ncas ** 2) @ u_ccaa_m)
    e_ccaa_m = -(np.diag(fcc)[lca] + np.diag(fcc)[lcb])[:, None] - e_ccaa_m[None, :]

    e_acav, r_acav = disjoint_eigh(np.concatenate([newv_acav.transpose(0, 2, 1).reshape(ncas ** 2, u_acav.shape[1]),
        newv_acva.transpose(0, 2, 1).reshape(ncas ** 2, u_acav.shape[1])], axis=0).T @ u_acav)
    e_acav = (np.diag(fvv)[None, :] - np.diag(fcc)[:, None]).reshape(ncore * nvir, 1) - e_acav[None, :]

    e_acvv, r_acvv = disjoint_eigh(newv_acvv.T @ u_acvv)
    e_acvv_p = ((np.diag(fvv)[uva] + np.diag(fvv)[uvb])[:, None] - np.diag(fcc)[None, :]).reshape(nvu * ncore, 1)
    e_acvv_m = ((np.diag(fvv)[lva] + np.diag(fvv)[lvb])[:, None] - np.diag(fcc)[None, :]).reshape(nvl * ncore, 1)
    e_acvv_p, e_acvv_m = e_acvv_p - e_acvv[None, :], e_acvv_m - e_acvv[None, :]

    mat = newv_ccav.T @ u_ccav
    e_ccav, r_ccav = disjoint_eigh(mat)
    e_ccav_p = (np.diag(fvv)[None, :] - (np.diag(fcc)[uca] + np.diag(fcc)[ucb])[:, None]).reshape(ncu * nvir, 1)
    e_ccav_m = (np.diag(fvv)[None, :] - (np.diag(fcc)[lca] + np.diag(fcc)[lcb])[:, None]).reshape(ncl * nvir, 1)
    e_ccav_p, e_ccav_m = e_ccav_p - e_ccav[None, :], e_ccav_m - e_ccav[None, :]

    e_vv_p = np.diag(fvv)[uva] + np.diag(fvv)[uvb]
    e_vv_m = np.diag(fvv)[lva] + np.diag(fvv)[lvb]

    diag_es = (np.diag(fcc), e_vv_p, e_vv_m, e_aaav, e_acaa, e_aavv_p, e_aavv_m, e_ccaa_p, e_ccaa_m, e_acav,
        e_acvv_p, e_acvv_m, e_ccav_p, e_ccav_m)
    diag_rs = r_aaav, r_acaa, r_aavv_p, r_aavv_m, r_ccaa_p, r_ccaa_m, r_acav, r_acvv, r_ccav

    return diag_es, diag_rs

def basis_orthogonalize(ncore, ncas, nvir, rdms, ortho_thrds=1E-10):
    i, j, k = np.ogrid[:ncas], *np.ogrid[:ncas, :ncas]
    trunc_f = lambda m, tol: [v[:, x] * w[x] ** -0.5 for w, v in [disjoint_eigh(m)] for x in [w > tol]][0]

    mat = rdms[2].transpose(3, 5, 0, 1, 2, 4).copy()
    mat[:, :, i, :, :, i] += rdms[1].transpose(3, 2, 1, 0)
    u_aaav = trunc_f(mat.reshape((ncas ** 3,) * 2), ortho_thrds)
    mat_acaa = -1 * rdms[2].transpose(3, 0, 2, 1, 4, 5)
    mat_acaa[:, i, :, :, :, i] -= rdms[1].transpose(3, 1, 0, 2)
    mat_acaa[:, i, :, :, i, :] -= rdms[1].transpose(3, 0, 1, 2)
    mat_acaa[:, :, i, :, i, :] -= rdms[1].transpose(2, 0, 1, 3)
    mat_acaa[:, :, i, :, :, i] += 2 * rdms[1].transpose(2, 0, 1, 3)
    mat_acaa[:, j, k, :, k, j] -= rdms[0]
    mat_acaa[:, j, k, :, j, k] += 2 * rdms[0]
    u_acaa = trunc_f(mat_acaa.reshape((ncas ** 3,) * 2), ortho_thrds)
    mat = rdms[1].transpose(2, 3, 0, 1)
    u_aavv_p = trunc_f((mat + mat.transpose(0, 1, 3, 2)).reshape((ncas ** 2,) * 2), ortho_thrds)
    u_aavv_m = trunc_f((mat - mat.transpose(0, 1, 3, 2)).reshape((ncas ** 2,) * 2), ortho_thrds)
    u_acvv = trunc_f(rdms[0], ortho_thrds)
    mat_ccaa = 0.5 * rdms[1]
    mat_ccaa[i, :, i, :] -= 2 * rdms[0]
    mat_ccaa[i, :, :, i] += rdms[0]
    mat_ccaa[j, k, k, j] -= 1
    mat_ccaa[j, k, j, k] += 2
    mat_ccaa = mat_ccaa + mat_ccaa.transpose(1, 0, 3, 2)
    u_ccaa_p = trunc_f((mat_ccaa + mat_ccaa.transpose(1, 0, 2, 3)).reshape((ncas ** 2,) * 2), ortho_thrds)
    u_ccaa_m = trunc_f((mat_ccaa - mat_ccaa.transpose(1, 0, 2, 3)).reshape((ncas ** 2,) * 2), ortho_thrds)
    u_ccav = trunc_f((2 * np.identity(ncas) - rdms[0]), ortho_thrds)
    mat = np.zeros((2, ncas, ncas, 2, ncas, ncas))
    mat[0, :, :, 0, :, :] += 2 * rdms[1].transpose(2, 0, 1, 3)
    mat[0, :, :, 1, :, :] -= rdms[1].transpose(2, 0, 1, 3)
    mat[1, :, :, 0, :, :] -= rdms[1].transpose(3, 1, 0, 2)
    mat[1, :, :, 1, :, :] -= rdms[1].transpose(3, 0, 1, 2)
    mat[:, :, i, :, :, i] += np.array([[2, -1], [-1, 2]])[:, None, :, None] * rdms[0][None, :, None, :]
    u_acav = trunc_f(mat.reshape((2 * ncas ** 2,) * 2), ortho_thrds)

    nvu, nvl, ncu, ncl = [x for nx in [nvir, ncore] for x in [nx * (nx + 1) // 2, nx * (nx - 1) // 2]]
    ortho_us = u_aaav, u_acaa, u_aavv_p, u_aavv_m, u_ccaa_p, u_ccaa_m, u_acav, u_acvv, u_ccav
    shapes = [(ncu, nvu), (ncl, nvl), (nvir, u_aaav.shape[1]), (ncore, u_acaa.shape[1]), (nvu, u_aavv_p.shape[1]),
        (nvl, u_aavv_m.shape[1]), (ncu, u_ccaa_p.shape[1]), (ncl, u_ccaa_m.shape[1]), (ncore * nvir, u_acav.shape[1]),
        (nvu * ncore, u_acvv.shape[1]), (nvl * ncore, u_acvv.shape[1]), (ncu * nvir, u_ccav.shape[1]),
        (ncl * nvir, u_ccav.shape[1])]

    return (mat_acaa, mat_ccaa), ortho_us, shapes

def amplitudes_to_vector(ncore, ncas, nvir, tamps, ortho_us, vname=None):
    uca, ucb, uva, uvb = *np.triu_indices(ncore), *np.triu_indices(nvir)
    lca, lcb, lva, lvb = *np.tril_indices(ncore, k=-1), *np.tril_indices(nvir, k=-1)
    nvl, ncl = nvir * (nvir + 1) // 2, ncore * (ncore + 1) // 2

    t_ccvv, t_aaav, t_acaa, t_aavv, t_ccaa, t_acav, t_acva, t_acvv, t_ccav = tamps
    u_aaav, u_acaa, u_aavv_p, u_aavv_m, u_ccaa_p, u_ccaa_m, u_acav, u_acvv, u_ccav = ortho_us

    if isinstance(t_ccvv, np.ndarray):
        v_ccvv_p = (t_ccvv + t_ccvv.transpose(0, 1, 3, 2))[uca[:, None], ucb[:, None], uva[None, :], uvb[None, :]] / 4
        v_ccvv_p[uca != ucb] *= 2 ** 0.5
        v_ccvv_p[:, uva != uvb] *= 2 ** 0.5
        v_ccvv_m = (t_ccvv - t_ccvv.transpose(0, 1, 3, 2))[lca[:, None], lcb[:, None], lva[None, :], lvb[None, :]]
        v_ccvv_m *= (2 / 3 ** 0.5) / 4
    else:
        with lib.H5TmpFile(t_ccvv.name, 'r+') as f5:
            t_cvcv, fv_ccvv_p, fv_ccvv_m = f5['cvcv'], f5[vname + '_p'], f5[vname + '_m']
            ip, im = 0, 0
            for i in range(ncore):
                tmp = t_cvcv[i * nvir:(i + 1) * nvir].reshape(nvir, ncore, nvir)
                fv_ccvv_m[im:im + i] = (tmp - tmp.transpose(2, 1, 0))[lva, :i, lvb].T * ((2 / 3 ** 0.5) / 4)
                tmp = (tmp + tmp.transpose(2, 1, 0))[uva, i:, uvb].T / 4
                tmp[1:] *= 2 ** 0.5
                tmp[:, uva != uvb] *= 2 ** 0.5
                fv_ccvv_p[ip:ip + ncore - i] = tmp
                ip, im = ip + ncore - i, im + i
            t_cvcv, fv_ccvv_p, fv_ccvv_m = None, None, None
        v_ccvv_p, v_ccvv_m = np.zeros((0, )), np.zeros((0, ))

    v_aaav = t_aaav.reshape((ncas ** 3, nvir)).T @ u_aaav.conj()
    v_acaa = t_acaa.transpose(1, 0, 2, 3).reshape(ncore, ncas ** 3) @ u_acaa.conj()
    v_aavv_p, v_aavv_m = t_aavv + t_aavv.transpose(0, 1, 3, 2), t_aavv - t_aavv.transpose(0, 1, 3, 2)
    v_aavv_p = (v_aavv_p.reshape(ncas ** 2, nvir, nvir)[:, uva, uvb].T @ u_aavv_p.conj()) / 2
    v_aavv_m = (v_aavv_m.reshape(ncas ** 2, nvir, nvir)[:, lva, lvb].T @ u_aavv_m.conj()) / 2 ** 0.5
    v_aavv_p[uva != uvb] *= 2 ** 0.5
    v_ccaa_p, v_ccaa_m = t_ccaa + t_ccaa.transpose(1, 0, 2, 3), t_ccaa - t_ccaa.transpose(1, 0, 2, 3)
    v_ccaa_p = (v_ccaa_p.reshape(ncore, ncore, ncas ** 2)[uca, ucb, :] @ u_ccaa_p.conj()) / 2
    v_ccaa_p[uca != ucb] *= 2 ** 0.5
    v_ccaa_m = (v_ccaa_m.reshape(ncore, ncore, ncas ** 2)[lca, lcb, :] @ u_ccaa_m.conj()) / 2 ** 0.5
    v_acav = np.concatenate([t_acav.transpose(0, 2, 1, 3).reshape(ncas ** 2, ncore * nvir),
        t_acva.transpose(0, 3, 1, 2).reshape(ncas ** 2, ncore * nvir)], axis=0).T @ u_acav.conj()
    v_acvv_p = (t_acvv + t_acvv.transpose(0, 1, 3, 2)).reshape(ncas, ncore, nvir, nvir)[..., uva, uvb].T
    v_acvv_p = (v_acvv_p.reshape(nvl * ncore, ncas) @ u_acvv.conj()).reshape(nvl, ncore, u_acvv.shape[1]) / 2
    v_acvv_p[uva != uvb] *= 2 ** 0.5
    v_acvv_m = (t_acvv - t_acvv.transpose(0, 1, 3, 2)).reshape(ncas, ncore, nvir, nvir)[..., lva, lvb].T
    v_acvv_m = (v_acvv_m.reshape(nvir * (nvir - 1) // 2 * ncore, ncas) @ u_acvv.conj()) / 6 ** 0.5
    v_ccav_p = (t_ccav + t_ccav.transpose(1, 0, 2, 3)).reshape(ncore, ncore, ncas, nvir)[uca, ucb].transpose(0, 2, 1)
    v_ccav_p = (v_ccav_p.reshape(ncl * nvir, ncas) @ u_ccav.conj()).reshape(ncl, nvir, u_ccav.shape[1]) / 2
    v_ccav_p[uca != ucb] *= 2 ** 0.5
    v_ccav_m = (t_ccav - t_ccav.transpose(1, 0, 2, 3)).reshape(ncore, ncore, ncas, nvir)[lca, lcb].transpose(0, 2, 1)
    v_ccav_m = (v_ccav_m.reshape(ncore * (ncore - 1) // 2 * nvir, ncas) @ u_ccav.conj()) / 6 ** 0.5

    return np.concatenate([v.ravel() for v in [v_ccvv_p, v_ccvv_m, v_aaav, v_acaa, v_aavv_p, v_aavv_m, v_ccaa_p,
        v_ccaa_m, v_acav, v_acvv_p, v_acvv_m, v_ccav_p, v_ccav_m]], axis=0)

def vector_to_amplitudes(ncore, ncas, nvir, vector, ortho_us, shapes, io_file=None, vname=None):
    nvu, nvl, ncu, ncl = [x for nx in [nvir, ncore] for x in [nx * (nx + 1) // 2, nx * (nx - 1) // 2]]
    uca, ucb, uva, uvb = *np.triu_indices(ncore), *np.triu_indices(nvir)
    lca, lcb, lva, lvb = *np.tril_indices(ncore, k=-1), *np.tril_indices(nvir, k=-1)
    ngc, ngv = np.ogrid[:ncore], np.ogrid[:nvir]
    mc = -1 * (ngc[:, None] < ngc[None]) + 1 * (ngc[:, None] > ngc[None]) + (2 ** 0.5 - 1) * (ngc[:, None] == ngc[None])
    mv = -1 * (ngv[:, None] < ngv[None]) + 1 * (ngv[:, None] > ngv[None]) + (2 ** 0.5 - 1) * (ngv[:, None] == ngv[None])

    u_aaav, u_acaa, u_aavv_p, u_aavv_m, u_ccaa_p, u_ccaa_m, u_acav, u_acvv, u_ccav = ortho_us
    vectors = [v.reshape(s) for v, s in zip(np.split(vector, np.cumsum([a * b for a, b in shapes])[:-1]), shapes)]
    v_ccvv_p, v_ccvv_m, v_aaav, v_acaa, v_aavv_p, v_aavv_m, v_ccaa_p = vectors[0:7]
    v_ccaa_m, v_acav, v_acvv_p, v_acvv_m, v_ccav_p, v_ccav_m = vectors[7:13]

    if io_file is None or isinstance(io_file, np.ndarray):
        t_ccvv = np.zeros((ncore, ncore, nvir, nvir), dtype=vector.dtype)
        factor = (1 + (uca != ucb)[:, None]) * (1 + (uva != uvb)[None, :])
        t_ccvv[uca[:, None], ucb[:, None], uva[None, :], uvb[None, :]] = v_ccvv_p * factor ** 0.5 / 4
        t_ccvv[lca[:, None], lcb[:, None], lva[None, :], lvb[None, :]] = v_ccvv_m / 3 ** 0.5 / 2
        factor = -1 * (ngv[:, None] < ngv[None, :]) + 1 * (ngv[:, None] >= ngv[None, :])
        t_ccvv += factor[None, None, :, :] * t_ccvv.transpose(0, 1, 3, 2)
    else:
        with lib.H5TmpFile(io_file.name, 'r+') as f5:
            t_cvcv, fv_ccvv_p, fv_ccvv_m = f5['i_cvcv'], f5[vname + '_p'], f5[vname + '_m']
            ip, im = 0, 0
            t_factor = (1 + (np.ogrid[:ncore] != 0)[:, None]) * (1 + (uva != uvb)[None, :])
            factor = -1 * (ngv[:, None] < ngv[None, :]) + 1 * (ngv[:, None] >= ngv[None, :])
            for i in range(ncore):
                tmp = np.zeros((ncore, nvir, nvir), dtype=t_cvcv.dtype)
                tmp[i:, uva, uvb] = fv_ccvv_p[ip:ip + ncore - i] * t_factor[:ncore - i] ** 0.5 / 4
                tmp[:i, lva, lvb] = fv_ccvv_m[im:im + i] / 3 ** 0.5 / 2
                tmp += factor[None, :, :] * tmp.transpose(0, 2, 1)
                t_cvcv[i * nvir:(i + 1) * nvir] = tmp.transpose(1, 0, 2).reshape(nvir, ncore * nvir)
                ip, im = ip + ncore - i, im + i
            t_cvcv, fv_ccvv_p, fv_ccvv_m, t_factor, factor = None, None, None, None, None
        t_ccvv = io_file

    t_aaav = (u_aaav @ v_aaav.T).reshape(ncas, ncas, ncas, nvir)
    t_acaa = (u_acaa @ v_acaa.T).reshape(ncas, ncas, ncas, ncore).transpose(0, 3, 1, 2)
    t_aavv = np.zeros((ncas, ncas, nvir, nvir), dtype=vector.dtype)
    t_aavv[:, :, uva, uvb] = (u_aavv_p @ v_aavv_p.T).reshape(ncas, ncas, nvu) / 2 ** 0.5
    t_aavv[:, :, lva, lvb] = (u_aavv_m @ v_aavv_m.T).reshape(ncas, ncas, nvl) / 2 ** 0.5
    t_aavv = t_aavv + mv[None, None] * t_aavv.transpose(0, 1, 3, 2)
    t_ccaa = np.zeros((ncore, ncore, ncas, ncas), dtype=vector.dtype)
    t_ccaa[uca, ucb, :, :] = (u_ccaa_p @ v_ccaa_p.T).reshape(ncas, ncas, ncu).transpose(2, 0, 1) / 2 ** 0.5
    t_ccaa[lca, lcb, :, :] = (u_ccaa_m @ v_ccaa_m.T).reshape(ncas, ncas, ncl).transpose(2, 0, 1) / 2 ** 0.5
    t_ccaa = t_ccaa + mc[:, :, None, None] * t_ccaa.transpose(1, 0, 2, 3)
    t_acav = (u_acav[:ncas * ncas] @ v_acav.T).reshape(ncas, ncas, ncore, nvir).transpose(0, 2, 1, 3)
    t_acva = (u_acav[ncas * ncas:] @ v_acav.T).reshape(ncas, ncas, ncore, nvir).transpose(0, 2, 3, 1)
    t_acvv = np.empty((ncas, ncore, nvir, nvir), dtype=vector.dtype)
    t_acvv[:, :, uva, uvb] = (u_acvv @ v_acvv_p.T).reshape(ncas, nvu, ncore).transpose(0, 2, 1) / 2 ** 0.5
    t_acvv[:, :, lva, lvb] = (u_acvv @ v_acvv_m.T).reshape(ncas, nvl, ncore).transpose(0, 2, 1) / 6 ** 0.5
    t_acvv = t_acvv + mv[None, None] * t_acvv.transpose(0, 1, 3, 2)
    t_ccav = np.empty((ncore, ncore, ncas, nvir), dtype=vector.dtype)
    t_ccav[uca, ucb, :, :] = (u_ccav @ v_ccav_p.T).reshape(ncas, ncu, nvir).transpose(1, 0, 2) / 2 ** 0.5
    t_ccav[lca, lcb, :, :] = (u_ccav @ v_ccav_m.T).reshape(ncas, ncl, nvir).transpose(1, 0, 2) / 6 ** 0.5
    t_ccav = t_ccav + mc[:, :, None, None] * t_ccav.transpose(1, 0, 2, 3)

    return t_ccvv, t_aaav, t_acaa, t_aavv, t_ccaa, t_acav, t_acva, t_acvv, t_ccav

def canonicalize_active_orbitals(ncore, ncas, rdms, fock, h1e, eris):
    a, x = slice(ncore, ncore + ncas), slice(None)
    if np.linalg.norm(fock[a, a] - np.diag(np.diag(fock[a, a]))) >= 1E-10:
        rdms, fock, h1e, uaa = list(rdms), fock.copy(), h1e.copy(), disjoint_eigh(fock[a, a])[1]
        fock[:, a], h1e[:, a] = fock[:, a] @ uaa, h1e[:, a] @ uaa
        fock[a, :], h1e[a, :] = uaa.T.conj() @ fock[a, :], uaa.T.conj() @ h1e[a, :]
        for i in range(len(rdms)):
            for u in (uaa.conj(), ) * (i + 1) + (uaa, ) * (i + 1):
                rdms[i] = np.tensordot(u, rdms[i], axes=(0, 0)).transpose(*range(1, rdms[i].ndim), 0)
        papa, ppaa, pcav, ccvv = eris[0].copy(), eris[1].copy(), eris[2].copy(), eris[3]
        for u, s, t in zip([uaa.conj(), uaa.conj(), uaa, uaa], [a, x, a, x], [a, a, x, x]):
            papa[s], ppaa[t] = np.tensordot(u, papa[s], axes=(0, 0)), np.tensordot(u, ppaa[t], axes=(0, 0))
            papa, ppaa = papa.transpose(1, 2, 3, 0), ppaa.transpose(1, 2, 3, 0)
        pcav[a] = np.tensordot(uaa.conj(), pcav[a], axes=(0, 0))
        pcav = np.tensordot(uaa, pcav.transpose(2, 3, 0, 1), axes=(0, 0)).transpose(2, 3, 0, 1)
        rdms, eris = tuple(rdms), (papa, ppaa, pcav, ccvv)
    assert ncas == 0 or np.linalg.norm(fock[a, a] - np.diag(np.diag(fock[a, a]))) < 1E-10
    return rdms, fock, h1e, eris

def caspt2_kernel(mc, ortho_thrds, frozen, ipea_shift, log, backend, canonicalized=False, root=0,
                  max_cycle=50, tol=1E-8, tolnormt=1E-4, precond=True, io_method='incore', io_blksize=None):
    ncore, ncas, nvir, nelec = mc.ncore - frozen, mc.ncas, len(mc._scf.mo_energy) - mc.ncore - mc.ncas, mc.nelecas
    name = 'CASPT2'
    ci = mc.ci if mc.fcisolver.nroots == 1 else mc.ci[root]
    fci_base = mc.fcisolver.undo_state_average() if hasattr(mc.fcisolver, 'undo_state_average') else mc.fcisolver
    t0 = time.perf_counter()
    assert io_method in ('incore', 'outcore')

    cput0 = (logger.process_clock(), logger.perf_counter())

    if not canonicalized:
        dm1 = fci_base.make_rdm1(ci, ncas, nelec) if ncas != 0 else np.zeros((0, 0))
        mo_coeff, ci, _ = mc.canonicalize(mc.mo_coeff, ci=ci, casdm1=dm1, cas_natorb=False, verbose=mc.verbose)
        cput0 = log.timer(name + ' orbital canonicalization', *cput0)
    else:
        mo_coeff = mc.mo_coeff

    if ncas == 0:
        rdms = [np.zeros((0, ) * (i + 1) * 2) for i in range(4)]
    elif hasattr(fci_base, 'make_rdm1234'):
        rdms = fci_base.make_rdm1234(ci, ncas, nelec)
    else:
        rdms = fci.rdm.reorder_dm123(*fci_base._make_dm123(ci, ncas, nelec), inplace=True)
        rdms = (*rdms, fci_base.make_rdm4(ci, ncas, nelec, restart=True).transpose(0, 4, 1, 5, 2, 6, 3, 7))
    rdms = [rdm.transpose(*range(0, rdm.ndim, 2), *range(1, rdm.ndim, 2)) for rdm in rdms]
    cput0 = log.timer(name + ' active space rdms computation', *cput0)

    e_ref, fock, h1e, eris = trans_integrals(mc, mo_coeff, rdms, method=io_method, frozen=frozen)
    cput0 = log.timer(name + ' integral transformation', *cput0)

    io_method = 'incore' if isinstance(eris[3], np.ndarray) else 'outcore'
    log.info('Init E_ref = %22.15f  E_corr(%s) = %22.15f  I/O = %s', e_ref, name, 0.0, io_method)

    if precond or ipea_shift != 0.0:
        rdms, fock, h1e, eris = canonicalize_active_orbitals(ncore, ncas, rdms, fock, h1e, eris)
        cput0 = log.timer(name + ' active orbtials canonicalization', *cput0)

    rdm_mats, ortho_us, shapes = basis_orthogonalize(ncore, ncas, nvir, rdms, ortho_thrds)
    shapes = shapes if io_method == 'incore' else [(0, 0), (0, 0)] + shapes[2:]
    cput0 = log.timer(name + ' basis orthogonalization', *cput0)

    bamps = compute_rhs_amplitudes(ncore, ncas, nvir, rdms, h1e, eris, backend=backend)
    h1e = eris = None
    io_file = bamps[0] if io_method != 'incore' else None
    b = amplitudes_to_vector(ncore, ncas, nvir, bamps, ortho_us, vname='b')
    cput0 = log.timer(name + ' rhs amplitudes', *cput0)

    if precond:
        diag_es, diag_rs = compute_lhs_preconditioner(ncore, ncas, nvir, ortho_us, rdms, rdm_mats,
            ipea_shift, fock, backend)
        cput0 = log.timer(name + ' lhs preconditioner', *cput0)

    def hop_f(xvec, xname, rname):
        cput1 = (logger.process_clock(), logger.perf_counter())
        tamps = vector_to_amplitudes(ncore, ncas, nvir, xvec, ortho_us, shapes, io_file, vname=xname)
        new_tamps = compute_lhs_amplitudes(ncore, ncas, nvir, tamps, rdms, fock, backend, mc.max_memory, io_blksize)
        if ipea_shift != 0.0:
            new_tamps = tuple(xt - dt if dt is not None else xt for xt, dt in zip(new_tamps,
                compute_ipea_shift(ncas, tamps, rdms, rdm_mats, ipea_shift)))
        new_xvec = amplitudes_to_vector(ncore, ncas, nvir, new_tamps, ortho_us, vname=rname)
        log.timer(name + ' lhs amplitudes', *cput1)
        return new_xvec

    def copy_f(xvec, xname, rname):
        rvec = xvec.copy()
        if io_method != 'incore':
            with lib.H5TmpFile(io_file.name, 'r+') as f5:
                del f5[rname + '_p'], f5[rname + '_m']
                f5.copy(f5[xname + '_p'], f5, name=rname + '_p')
                f5.copy(f5[xname + '_m'], f5, name=rname + '_m')
        return rvec

    def precond_f(xvec, xname, rname):
        return copy_f(xvec, xname, rname) if not precond else apply_lhs_preconditioner(ncore, xvec, shapes, diag_es,
            diag_rs, io_file=io_file, vname=xname, zname=rname)

    def outcore_dot_f(aname, bname):
        x = 0.0
        with lib.H5TmpFile(io_file.name, 'r+') as f5:
            ap, am, bp, bm = f5[aname + '_p'], f5[aname + '_m'], f5[bname + '_p'], f5[bname + '_m']
            ip, im = 0, 0
            for i in range(ncore):
                x += np.dot(ap[ip:ip + ncore - i].ravel(), bp[ip:ip + ncore - i].ravel())
                x += np.dot(am[im:im + i].ravel(), bm[im:im + i].ravel())
                ip, im = ip + ncore - i, im + i
        return x

    # a = alpha * b + beta * a
    def outcore_iadd_f(aname, bname, alpha=1.0, beta=1.0):
        with lib.H5TmpFile(io_file.name, 'r+') as f5:
            ap, am, bp, bm = f5[aname + '_p'], f5[aname + '_m'], f5[bname + '_p'], f5[bname + '_m']
            ip, im = 0, 0
            for i in range(ncore):
                if beta == 1.0:
                    ap[ip:ip + ncore - i] += alpha * bp[ip:ip + ncore - i]
                    am[im:im + i] += alpha * bm[im:im + i]
                else:
                    ap[ip:ip + ncore - i] = alpha * bp[ip:ip + ncore - i] + beta * ap[ip:ip + ncore - i]
                    am[im:im + i] = alpha * bm[im:im + i] + beta * am[im:im + i]
                ip, im = ip + ncore - i, im + i

    def dot_f(avec, bvec, aname, bname):
        return np.dot(avec, bvec) if io_method == 'incore' else np.dot(avec, bvec) + outcore_dot_f(aname, bname)

    def iadd_f(avec, bvec, aname, bname, alpha=1.0, beta=1.0):
        if beta == 1.0:
            avec += alpha * bvec
        else:
            avec[:] = alpha * bvec + beta * avec
        if io_method != 'incore':
            outcore_iadd_f(aname, bname, alpha=alpha, beta=beta)

    conv, e_corr, z = False, 0.0, None
    t1 = time.perf_counter()
    x = np.zeros_like(b)
    r = copy_f(b, 'b', 'r')
    p = precond_f(r, 'r', 'p')
    error = dot_f(p, r, 'p', 'r')
    if np.sqrt(np.abs(error)) < tolnormt:
        conv = True
        log.info("cycle =%3d  E_corr(%s) = %22.15f  dE = %10.3E  norm(res) = %10.3E T = %12.3f",
            1, name, e_corr, 0.0, np.sqrt(np.abs(error)), time.perf_counter() - t1)
    else:
        old_error = error
        xiter = 0
        while xiter < max_cycle:
            xiter += 1
            z = hop_f(p, 'p', 'z')
            alpha = old_error / dot_f(p, z, 'p', 'z')
            iadd_f(x, p, 'x', 'p', alpha=alpha) # x += alpha * p
            iadd_f(r, z, 'r', 'z', alpha=-alpha) # r -= alpha * z
            z = precond_f(r, 'r', 'z')
            error, e_corr, e_corr_old = dot_f(z, r, 'z', 'r'), dot_f(x, b, 'x', 'b'), e_corr
            log.info("cycle =%3d  E_corr(%s) = %22.15f  dE = %10.3E  norm(res) = %10.3E T = %10.3f",
                xiter, name, e_corr, e_corr - e_corr_old, np.sqrt(np.abs(error)), time.perf_counter() - t1)
            if abs(e_corr - e_corr_old) < tol and np.sqrt(np.abs(error)) < tolnormt:
                conv = True
                break
            else:
                beta, old_error = error / old_error, error
                iadd_f(p, z, 'p', 'z', beta=beta) # p = beta * p + z
            t1 = time.perf_counter()
    r, p, z = None, None, None
    b_vecs, x_vecs = [[v for v in np.split(z, np.cumsum([a * b for a, b in shapes])[:-1])] for z in [b, x]]
    decomp_e_corrs = [np.dot(bb, xx) for bb, xx in zip(b_vecs, x_vecs)]
    outcore_e_corr = 0.0 if io_method == 'incore' else outcore_dot_f('b', 'x')
    log.info("%s %sconverged", name, "" if conv else "not ")
    log.info("Decomposed CASPT2 correlation energies:")
    log.info("  E_corr[ijab] = %22.15f", decomp_e_corrs[0] + decomp_e_corrs[1] + outcore_e_corr)
    log.info("  E_corr[iab]  = %22.15f", decomp_e_corrs[9] + decomp_e_corrs[10])
    log.info("  E_corr[ija]  = %22.15f", decomp_e_corrs[11] + decomp_e_corrs[12])
    log.info("  E_corr[ab]   = %22.15f", decomp_e_corrs[4] + decomp_e_corrs[5])
    log.info("  E_corr[ij]   = %22.15f", decomp_e_corrs[6] + decomp_e_corrs[7])
    log.info("  E_corr[ia]   = %22.15f", decomp_e_corrs[8])
    log.info("  E_corr[a]    = %22.15f", decomp_e_corrs[2])
    log.info("  E_corr[i]    = %22.15f", decomp_e_corrs[3])
    b_vecs, x_vecs = None, None
    e_tot = e_ref + e_corr
    log.note("E(%s) = %22.15f  E_corr = %22.15f", name, e_tot, e_corr)
    log.info("Time = %12.3f", time.perf_counter() - t0)
    return conv, e_tot, e_corr

class ICCASPT2(lib.StreamObject):
    '''Internally contracted CASPT2.

    Attributes:
        root : int
            Which state to compute if multiple roots or state-average
            wfn were calculated in CASCI/CASSCF. Default is zero.
        frozen : int
            Number of frozen core orbitals. Default is zero.
        ipea_shift : float
            IPEA shift. Default is zero.
        ortho_thrds : float
            Threshold for basis orthogonalization. Default is 1E-10.
        precond : bool
            Whether preconditioner should be computed for convergence
            acceleration. Default is True.
        io_method : str ('incore' or 'outcore')
            Indicating incore or outcore storage for the ccvv amplitudes.
            Default is 'incore', which uses the incore method whenever possible.

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M('N 0 0 0; N 0 0 1.4', basis='6-31g')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 4, 4).run()
    >>> CASPT2(mc).kernel()
    '''

    max_cycle = getattr(__config__, 'mrpt_caspt2_ICCASPT2_max_cycle', 50)
    ortho_thrds = getattr(__config__, 'mrpt_caspt2_ICCASPT2_ortho_thrds', 1e-10)
    conv_tol = getattr(__config__, 'mrpt_caspt2_ICCASPT2_conv_tol', 1e-8)
    conv_tol_normt = getattr(__config__, 'mrpt_caspt2_ICCASPT2_conv_tol_normt', 1e-6)
    einsum_backend = getattr(__config__, 'mrpt_caspt2_ICCASPT2_einsum_backend', 'numpy')
    precond = getattr(__config__, 'mrpt_caspt2_ICCASPT2_precond', True)
    io_method = getattr(__config__, 'mrpt_caspt2_ICCASPT2_io_method', 'incore')
    io_blksize = getattr(__config__, 'mrpt_caspt2_ICCASPT2_io_blksize', None)

    _keys = {'root', 'frozen', 'ortho_thrds', 'max_cycle', 'conv_tol', 'conv_tol_normt', 'io_method', 'io_blksize',
        'converged', 'e_tot', 'e_corr', 'einsum_backend', 'precond', 'ipea_shift', 'canonicalized'}

    def __init__(self, mc, root=0, frozen=0, ipea_shift=0.0):
        self._mc = mc
        self.root = root
        self.frozen = frozen
        self.ipea_shift = ipea_shift
        self.verbose = mc.verbose
        self.canonicalized = False

    def set_einsum_backend(self, backend):
        self.einsum_backend = backend

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore, ncas = self._mc.ncore - self.frozen, self._mc.ncas
        nvir = len(self._mc._scf.mo_energy) - ncore - ncas - self.frozen
        log.info('IC-CASPT2 nfrozen = %d ncore = %d ncas = %d nvir = %d', self.frozen, ncore, ncas, nvir)
        log.info('root = %d', self.root)
        log.info('ipea_shift = %g', self.ipea_shift)
        log.info('canonicalized = %s', self.canonicalized)
        log.info('ortho_thrds = %g', self.ortho_thrds)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %g', self.conv_tol_normt)
        log.info('precond = %s', self.precond)
        log.info('io_method = %s', self.io_method)
        log.info('io_blksize = %s', self.io_blksize)
        log.info('verbose = %s', self.verbose)
        if self.einsum_backend is not None:
            log.info('einsum_backend: %s', self.einsum_backend)
        return self

    def kernel(self):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        log = logger.new_logger(self, self.verbose)
        self.converged, self.e_tot, self.e_corr = caspt2_kernel(self._mc, self.ortho_thrds, self.frozen,
            self.ipea_shift, log, self.einsum_backend, canonicalized=self.canonicalized, root=self.root,
            max_cycle=self.max_cycle, tol=self.conv_tol, tolnormt=self.conv_tol_normt, precond=self.precond,
            io_method=self.io_method, io_blksize=self.io_blksize)
        return self.e_tot, self.e_corr

CASPT2 = ICCASPT2

if __name__ == "__main__":

    from pyscf import gto, scf, mcscf

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, symmetry='d2h', verbose=3)
    mf = scf.RHF(mol).run(conv_tol=1E-16)

    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-12
    mc.canonicalization = True
    mc.natorb = True
    mc.kernel()
    assert abs(mc.e_tot - -149.708657771235039) < 1E-8

    mr = CASPT2(mc, frozen=0, ipea_shift=0.0, root=0)
    mr.einsum_backend = 'numpy'
    mr.verbose = 4
    mr.max_cycle = 20
    mr.kernel()
    assert abs(mr.e_tot - -149.98063465606262) < 1E-8

    mr.io_method = 'outcore'
    mr.io_blksize = 3
    mr.kernel()
    assert abs(mr.e_tot - -149.98063465606262) < 1E-8
