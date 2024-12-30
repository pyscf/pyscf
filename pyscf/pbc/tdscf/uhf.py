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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy as np
from pyscf import lib
from pyscf.tdscf import uhf
from pyscf.pbc.tdscf import rhf as td_rhf
from pyscf.pbc.tdscf.rhf import TDBase


def get_ab(mf):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Spin symmetry is considered in the returned A, B lists.  List A has three
    items: (A_aaaa, A_aabb, A_bbbb). A_bbaa = A_aabb.transpose(2,3,0,1).
    B has three items: (B_aaaa, B_aabb, B_bbbb).
    B_bbaa = B_aabb.transpose(2,3,0,1).
    '''
    cell = mf.cell
    nao = cell.nao_nr()
    mo_energy = scf.addons.mo_energy_with_exxdiv_none(mf)
    mo = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    kpt = mf.kpt

    occidx_a = np.where(mo_occ[0]==1)[0]
    viridx_a = np.where(mo_occ[0]==0)[0]
    occidx_b = np.where(mo_occ[1]==1)[0]
    viridx_b = np.where(mo_occ[1]==0)[0]
    orbo_a = mo[0][:,occidx_a]
    orbv_a = mo[0][:,viridx_a]
    orbo_b = mo[1][:,occidx_b]
    orbv_b = mo[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]
    mo_a = np.hstack((orbo_a,orbv_a))
    mo_b = np.hstack((orbo_b,orbv_b))
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    e_ia_a = mo_energy[0][viridx_a] - mo_energy[0][occidx_a,None]
    e_ia_b = mo_energy[1][viridx_b] - mo_energy[1][occidx_b,None]
    a_aa = np.diag(e_ia_a.ravel()).reshape(nocc_a,nvir_a,nocc_a,nvir_a)
    a_bb = np.diag(e_ia_b.ravel()).reshape(nocc_b,nvir_b,nocc_b,nvir_b)
    a_ab = np.zeros((nocc_a,nvir_a,nocc_b,nvir_b))
    b_aa = np.zeros_like(a_aa)
    b_ab = np.zeros_like(a_ab)
    b_bb = np.zeros_like(a_bb)
    a = (a_aa, a_ab, a_bb)
    b = (b_aa, b_ab, b_bb)

    def add_hf_(a, b, hyb=1):
        eri_aa = mf.with_df.ao2mo([orbo_a,mo_a,mo_a,mo_a], kpt, compact=False)
        eri_ab = mf.with_df.ao2mo([orbo_a,mo_a,mo_b,mo_b], kpt, compact=False)
        eri_bb = mf.with_df.ao2mo([orbo_b,mo_b,mo_b,mo_b], kpt, compact=False)
        eri_aa = eri_aa.reshape(nocc_a,nmo_a,nmo_a,nmo_a)
        eri_ab = eri_ab.reshape(nocc_a,nmo_a,nmo_b,nmo_b)
        eri_bb = eri_bb.reshape(nocc_b,nmo_b,nmo_b,nmo_b)
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b

        a_aa += np.einsum('iabj->iajb', eri_aa[:nocc_a,nocc_a:,nocc_a:,:nocc_a])
        a_aa -= np.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:]) * hyb
        b_aa += np.einsum('iajb->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:])
        b_aa -= np.einsum('jaib->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:]) * hyb

        a_bb += np.einsum('iabj->iajb', eri_bb[:nocc_b,nocc_b:,nocc_b:,:nocc_b])
        a_bb -= np.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:]) * hyb
        b_bb += np.einsum('iajb->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:])
        b_bb -= np.einsum('jaib->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:]) * hyb

        a_ab += np.einsum('iabj->iajb', eri_ab[:nocc_a,nocc_a:,nocc_b:,:nocc_b])
        b_ab += np.einsum('iajb->iajb', eri_ab[:nocc_a,nocc_a:,:nocc_b,nocc_b:])

    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)

        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            with mf.with_df.range_coulomb(omega) as rsh_df:
                eri_aa = rsh_df.ao2mo([orbo_a,mo_a,mo_a,mo_a], kpt, compact=False)
                eri_ab = rsh_df.ao2mo([orbo_a,mo_a,mo_b,mo_b], kpt, compact=False)
                eri_bb = rsh_df.ao2mo([orbo_b,mo_b,mo_b,mo_b], kpt, compact=False)
                eri_aa = eri_aa.reshape(nocc_a,nmo_a,nmo_a,nmo_a)
                eri_ab = eri_ab.reshape(nocc_a,nmo_a,nmo_b,nmo_b)
                eri_bb = eri_bb.reshape(nocc_b,nmo_b,nmo_b,nmo_b)
                a_aa, a_ab, a_bb = a
                b_aa, b_ab, b_bb = b
                k_fac = alpha - hyb
                a_aa -= np.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:]) * k_fac
                b_aa -= np.einsum('jaib->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:]) * k_fac
                a_bb -= np.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:]) * k_fac
                b_bb -= np.einsum('jaib->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:]) * k_fac

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo, mo_occ)
        make_rho = ni._gen_rho_evaluator(cell, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpt, None, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[:,0,:,0] * weight

                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_a = np.einsum('ri,ra->ria', rho_o_a, rho_v_a)
                rho_ov_b = np.einsum('ri,ra->ria', rho_o_b, rho_v_b)
                rho_vo_a = rho_ov_a.conj()
                rho_vo_b = rho_ov_b.conj()
                w_vo_aa = np.einsum('ria,r->ria', rho_vo_a, wfxc[0,0])
                w_vo_ab = np.einsum('ria,r->ria', rho_vo_a, wfxc[0,1])
                w_vo_bb = np.einsum('ria,r->ria', rho_vo_b, wfxc[1,1])

                a_aa += lib.einsum('ria,rjb->iajb', w_vo_aa, rho_ov_a)
                b_aa += lib.einsum('ria,rjb->iajb', w_vo_aa, rho_vo_a)

                a_ab += lib.einsum('ria,rjb->iajb', w_vo_ab, rho_ov_b)
                b_ab += lib.einsum('ria,rjb->iajb', w_vo_ab, rho_vo_b)

                a_bb += lib.einsum('ria,rjb->iajb', w_vo_bb, rho_ov_b)
                b_bb += lib.einsum('ria,rjb->iajb', w_vo_bb, rho_vo_b)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpt, None, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_a = np.einsum('xri,ra->xria', rho_o_a, rho_v_a[0])
                rho_ov_b = np.einsum('xri,ra->xria', rho_o_b, rho_v_b[0])
                rho_ov_a[1:4] += np.einsum('ri,xra->xria', rho_o_a[0], rho_v_a[1:4])
                rho_ov_b[1:4] += np.einsum('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                rho_vo_a = rho_ov_a.conj()
                rho_vo_b = rho_ov_b.conj()
                w_vo_aa = np.einsum('xyr,xria->yria', wfxc[0,:,0], rho_vo_a)
                w_vo_ab = np.einsum('xyr,xria->yria', wfxc[0,:,1], rho_vo_a)
                w_vo_bb = np.einsum('xyr,xria->yria', wfxc[1,:,1], rho_vo_b)

                a_aa += lib.einsum('xria,xrjb->iajb', w_vo_aa, rho_ov_a)
                b_aa += lib.einsum('xria,xrjb->iajb', w_vo_aa, rho_vo_a)

                a_ab += lib.einsum('xria,xrjb->iajb', w_vo_ab, rho_ov_b)
                b_ab += lib.einsum('xria,xrjb->iajb', w_vo_ab, rho_vo_b)

                a_bb += lib.einsum('xria,xrjb->iajb', w_vo_bb, rho_ov_b)
                b_bb += lib.einsum('xria,xrjb->iajb', w_vo_bb, rho_vo_b)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpt, None, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_oa = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_a = np.einsum('xri,ra->xria', rho_oa, rho_va[0])
                rho_ov_b = np.einsum('xri,ra->xria', rho_ob, rho_vb[0])
                rho_ov_a[1:4] += np.einsum('ri,xra->xria', rho_oa[0], rho_va[1:4])
                rho_ov_b[1:4] += np.einsum('ri,xra->xria', rho_ob[0], rho_vb[1:4])
                tau_ov_a = np.einsum('xri,xra->ria', rho_oa[1:4], rho_va[1:4]) * .5
                tau_ov_b = np.einsum('xri,xra->ria', rho_ob[1:4], rho_vb[1:4]) * .5
                rho_ov_a = np.vstack([rho_ov_a, tau_ov_a[np.newaxis]])
                rho_ov_b = np.vstack([rho_ov_b, tau_ov_b[np.newaxis]])
                rho_vo_a = rho_ov_a.conj()
                rho_vo_b = rho_ov_b.conj()
                w_vo_aa = np.einsum('xyr,xria->yria', wfxc[0,:,0], rho_vo_a)
                w_vo_ab = np.einsum('xyr,xria->yria', wfxc[0,:,1], rho_vo_a)
                w_vo_bb = np.einsum('xyr,xria->yria', wfxc[1,:,1], rho_vo_b)

                a_aa += lib.einsum('xria,xrjb->iajb', w_vo_aa, rho_ov_a)
                b_aa += lib.einsum('xria,xrjb->iajb', w_vo_aa, rho_vo_a)

                a_ab += lib.einsum('xria,xrjb->iajb', w_vo_ab, rho_ov_b)
                b_ab += lib.einsum('xria,xrjb->iajb', w_vo_ab, rho_vo_b)

                a_bb += lib.einsum('xria,xrjb->iajb', w_vo_bb, rho_ov_b)
                b_bb += lib.einsum('xria,xrjb->iajb', w_vo_bb, rho_vo_b)

    else:
        add_hf_(a, b)

    return a, b

class TDA(TDBase):

    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    singlet = None

    init_guess = uhf.TDA.init_guess
    kernel = uhf.TDA.kernel
    _gen_vind = uhf.TDA.gen_vind
    gen_vind = td_rhf.TDA.gen_vind

CIS = TDA


class TDHF(TDBase):

    get_ab = TDA.get_ab

    singlet = None

    init_guess = uhf.TDHF.init_guess
    kernel = uhf.TDHF.kernel
    _gen_vind = uhf.TDHF.gen_vind
    gen_vind = td_rhf.TDA.gen_vind

RPA = TDUHF = TDHF


from pyscf.pbc import scf
scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)
