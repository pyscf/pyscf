#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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


import numpy
from pyscf import lib
from pyscf import scf
from pyscf import symm
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf.scf import uhf_symm
from pyscf.scf import _response_functions
from pyscf.data import nist
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def gen_tda_operation(mf, fock_ao=None, wfnsym=None):
    '''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
    hdiag = e_ia
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = mf.gen_response(hermi=0, max_memory=max_memory)

    def vind(zs):
        zs = numpy.asarray(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        za = zs[:,:nocca*nvira].reshape(-1,nocca,nvira)
        zb = zs[:,nocca*nvira:].reshape(-1,noccb,nvirb)
        dmova = lib.einsum('xov,qv,po->xpq', za, orbva.conj(), orboa)
        dmovb = lib.einsum('xov,qv,po->xpq', zb, orbvb.conj(), orbob)

        v1ao = vresp(numpy.asarray((dmova,dmovb)))

        v1a = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        v1b = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        v1a += numpy.einsum('xia,ia->xia', za, e_ia_a)
        v1b += numpy.einsum('xia,ia->xia', zb, e_ia_b)

        nz = zs.shape[0]
        hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        if wfnsym is not None and mol.symmetry:
            hx[:,sym_forbid] = 0
        return hx

    return vind, hdiag
gen_tda_hop = gen_tda_operation

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    Spin symmetry is considered in the returned A, B lists.  List A has three
    items: (A_aaaa, A_aabb, A_bbbb). A_bbaa = A_aabb.transpose(2,3,0,1).
    B has three items: (B_aaaa, B_aabb, B_bbbb).
    B_bbaa = B_aabb.transpose(2,3,0,1).
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    mol = mf.mol
    nao = mol.nao_nr()
    occidx_a = numpy.where(mo_occ[0]==1)[0]
    viridx_a = numpy.where(mo_occ[0]==0)[0]
    occidx_b = numpy.where(mo_occ[1]==1)[0]
    viridx_b = numpy.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]
    mo_a = numpy.hstack((orbo_a,orbv_a))
    mo_b = numpy.hstack((orbo_b,orbv_b))
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    e_ia_a = (mo_energy[0][viridx_a,None] - mo_energy[0][occidx_a]).T
    e_ia_b = (mo_energy[1][viridx_b,None] - mo_energy[1][occidx_b]).T
    a_aa = numpy.diag(e_ia_a.ravel()).reshape(nocc_a,nvir_a,nocc_a,nvir_a)
    a_bb = numpy.diag(e_ia_b.ravel()).reshape(nocc_b,nvir_b,nocc_b,nvir_b)
    a_ab = numpy.zeros((nocc_a,nvir_a,nocc_b,nvir_b))
    b_aa = numpy.zeros_like(a_aa)
    b_ab = numpy.zeros_like(a_ab)
    b_bb = numpy.zeros_like(a_bb)
    a = (a_aa, a_ab, a_bb)
    b = (b_aa, b_ab, b_bb)

    def add_hf_(a, b, hyb=1):
        eri_aa = ao2mo.general(mol, [orbo_a,mo_a,mo_a,mo_a], compact=False)
        eri_ab = ao2mo.general(mol, [orbo_a,mo_a,mo_b,mo_b], compact=False)
        eri_bb = ao2mo.general(mol, [orbo_b,mo_b,mo_b,mo_b], compact=False)
        eri_aa = eri_aa.reshape(nocc_a,nmo_a,nmo_a,nmo_a)
        eri_ab = eri_ab.reshape(nocc_a,nmo_a,nmo_b,nmo_b)
        eri_bb = eri_bb.reshape(nocc_b,nmo_b,nmo_b,nmo_b)
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b

        a_aa += numpy.einsum('iabj->iajb', eri_aa[:nocc_a,nocc_a:,nocc_a:,:nocc_a])
        a_aa -= numpy.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:]) * hyb
        b_aa += numpy.einsum('iajb->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:])
        b_aa -= numpy.einsum('jaib->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:]) * hyb

        a_bb += numpy.einsum('iabj->iajb', eri_bb[:nocc_b,nocc_b:,nocc_b:,:nocc_b])
        a_bb -= numpy.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:]) * hyb
        b_bb += numpy.einsum('iajb->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:])
        b_bb -= numpy.einsum('jaib->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:]) * hyb

        a_ab += numpy.einsum('iabj->iajb', eri_ab[:nocc_a,nocc_a:,nocc_b:,:nocc_b])
        b_ab += numpy.einsum('iajb->iajb', eri_ab[:nocc_a,nocc_a:,:nocc_b,nocc_b:])

    if isinstance(mf, scf.hf.KohnShamDFT):
        from pyscf.dft import xc_deriv
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[:,0,:,0] * weight

                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo_a)
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv_a)
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo_b)
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv_b)
                rho_ov_a = numpy.einsum('ri,ra->ria', rho_o_a, rho_v_a)
                rho_ov_b = numpy.einsum('ri,ra->ria', rho_o_b, rho_v_b)

                w_ov = numpy.einsum('ria,r->ria', rho_ov_a, wfxc[0,0])
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                a_aa += iajb
                b_aa += iajb

                w_ov = numpy.einsum('ria,r->ria', rho_ov_b, wfxc[0,1])
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov)
                a_ab += iajb
                b_ab += iajb

                w_ov = numpy.einsum('ria,r->ria', rho_ov_b, wfxc[1,1])
                iajb = lib.einsum('ria,rjb->iajb', rho_ov_b, w_ov)
                a_bb += iajb
                b_bb += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o_a = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_v_a = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_o_b = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_v_b = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_a = numpy.einsum('xri,ra->xria', rho_o_a, rho_v_a[0])
                rho_ov_b = numpy.einsum('xri,ra->xria', rho_o_b, rho_v_b[0])
                rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', rho_o_a[0], rho_v_a[1:4])
                rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                w_ov_aa = numpy.einsum('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = numpy.einsum('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = numpy.einsum('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                a_aa += iajb
                b_aa += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                a_bb += iajb
                b_bb += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                a_ab += iajb
                b_ab += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_oa = lib.einsum('xrp,pi->xri', ao, orbo_a)
                rho_ob = lib.einsum('xrp,pi->xri', ao, orbo_b)
                rho_va = lib.einsum('xrp,pi->xri', ao, orbv_a)
                rho_vb = lib.einsum('xrp,pi->xri', ao, orbv_b)
                rho_ov_a = numpy.einsum('xri,ra->xria', rho_oa, rho_va[0])
                rho_ov_b = numpy.einsum('xri,ra->xria', rho_ob, rho_vb[0])
                rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', rho_oa[0], rho_va[1:4])
                rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', rho_ob[0], rho_vb[1:4])
                tau_ov_a = numpy.einsum('xri,xra->ria', rho_oa[1:4], rho_va[1:4]) * .5
                tau_ov_b = numpy.einsum('xri,xra->ria', rho_ob[1:4], rho_vb[1:4]) * .5
                rho_ov_a = numpy.vstack([rho_ov_a, tau_ov_a[numpy.newaxis]])
                rho_ov_b = numpy.vstack([rho_ov_b, tau_ov_b[numpy.newaxis]])
                w_ov_aa = numpy.einsum('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = numpy.einsum('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = numpy.einsum('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                a_aa += iajb
                b_aa += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                a_bb += iajb
                b_bb += iajb

                iajb = lib.einsum('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                a_ab += iajb
                b_ab += iajb

    else:
        add_hf_(a, b)

    return a, b

def get_nto(tdobj, state=1, threshold=OUTPUT_THRESHOLD, verbose=None):
    r'''
    Natural transition orbital analysis.

    The natural transition density matrix between ground state and excited
    state :math:`Tia = \langle \Psi_{ex} | i a^\dagger | \Psi_0 \rangle` can
    be transformed to diagonal form through SVD
    :math:`T = O \sqrt{\lambda} V^\dagger`. O and V are occupied and virtual
    natural transition orbitals. The diagonal elements :math:`\lambda` are the
    weights of the occupied-virtual orbital pair in the excitation.

    Ref: Martin, R. L., JCP, 118, 4775-4777

    Note in the TDHF/TDDFT calculations, the excitation part (X) is
    interpreted as the CIS coefficients and normalized to 1. The de-excitation
    part (Y) is ignored.

    Args:
        state : int
            Excited state ID.  state = 1 means the first excited state.
            If state < 0, state ID is counted from the last excited state.

    Kwargs:
        threshold : float
            Above which the NTO coefficients will be printed in the output.

    Returns:
        A list (weights, NTOs).  NTOs are natural orbitals represented in AO
        basis. The first N_occ NTOs are occupied NTOs and the rest are virtual
        NTOs.
    '''
    if state == 0:
        logger.warn(tdobj, 'Excited state starts from 1. '
                    'Set state=1 for first excited state.')
        state_id = state
    elif state < 0:
        state_id = state
    else:
        state_id = state - 1

    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo_a = mo_coeff[0][:,mo_occ[0]==1]
    orbv_a = mo_coeff[0][:,mo_occ[0]==0]
    orbo_b = mo_coeff[1][:,mo_occ[1]==1]
    orbv_b = mo_coeff[1][:,mo_occ[1]==0]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]

    cis_t1a, cis_t1b = tdobj.xy[state_id][0]
    norm = numpy.linalg.norm(cis_t1a)**2 + numpy.linalg.norm(cis_t1b)**2
    cis_t1a *= 1. / norm
    cis_t1b *= 1. / norm

    if mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        o_sym_a = orbsyma[mo_occ[0]==1]
        v_sym_a = orbsyma[mo_occ[0]==0]
        o_sym_b = orbsymb[mo_occ[1]==1]
        v_sym_b = orbsymb[mo_occ[1]==0]
        nto_o_a = numpy.eye(nocc_a)
        nto_v_a = numpy.eye(nvir_a)
        nto_o_b = numpy.eye(nocc_b)
        nto_v_b = numpy.eye(nvir_b)
        weights_o_a = numpy.zeros(nocc_a)
        weights_v_a = numpy.zeros(nvir_a)
        weights_o_b = numpy.zeros(nocc_b)
        weights_v_b = numpy.zeros(nvir_b)

        for ir in set(orbsyma):
            idx = numpy.where(o_sym_a == ir)[0]
            if idx.size > 0:
                dm_oo = numpy.dot(cis_t1a[idx], cis_t1a[idx].T)
                weights_o_a[idx], nto_o_a[idx[:,None],idx] = numpy.linalg.eigh(dm_oo)

            idx = numpy.where(v_sym_a == ir)[0]
            if idx.size > 0:
                dm_vv = numpy.dot(cis_t1a[:,idx].T, cis_t1a[:,idx])
                weights_v_a[idx], nto_v_a[idx[:,None],idx] = numpy.linalg.eigh(dm_vv)

        for ir in set(orbsymb):
            idx = numpy.where(o_sym_b == ir)[0]
            if idx.size > 0:
                dm_oo = numpy.dot(cis_t1b[idx], cis_t1b[idx].T)
                weights_o_b[idx], nto_o_b[idx[:,None],idx] = numpy.linalg.eigh(dm_oo)

            idx = numpy.where(v_sym_b == ir)[0]
            if idx.size > 0:
                dm_vv = numpy.dot(cis_t1b[:,idx].T, cis_t1b[:,idx])
                weights_v_b[idx], nto_v_b[idx[:,None],idx] = numpy.linalg.eigh(dm_vv)

        def sort(weights, nto, sym):
            # weights in descending order
            idx = numpy.argsort(-weights)
            weights = weights[idx]
            nto = nto[:,idx]
            sym = sym[idx]
            return weights, nto, sym

        weights_o_a, nto_o_a, o_sym_a = sort(weights_o_a, nto_o_a, o_sym_a)
        weights_v_a, nto_v_a, v_sym_a = sort(weights_v_a, nto_v_a, v_sym_a)
        weights_o_b, nto_o_b, o_sym_b = sort(weights_o_b, nto_o_b, o_sym_b)
        weights_v_b, nto_v_b, v_sym_b = sort(weights_v_b, nto_v_b, v_sym_b)

        nto_orbsyma = numpy.hstack((o_sym_a, v_sym_a))
        nto_orbsymb = numpy.hstack((o_sym_b, v_sym_b))

        if nocc_a < nvir_a:
            weights_a = weights_o_a
        else:
            weights_a = weights_v_a
        if nocc_b < nvir_b:
            weights_b = weights_o_b
        else:
            weights_b = weights_v_b

    else:
        nto_o_a, w_a, nto_v_aT = numpy.linalg.svd(cis_t1a)
        nto_o_b, w_b, nto_v_bT = numpy.linalg.svd(cis_t1b)
        nto_v_a = nto_v_aT.conj().T
        nto_v_b = nto_v_bT.conj().T
        weights_a = w_a**2
        weights_b = w_b**2
        nto_orbsyma = nto_orbsymb = None

    def _set_phase_(c):
        idx = numpy.argmax(abs(c.real), axis=0)
        c[:,c[idx,numpy.arange(c.shape[1])].real<0] *= -1
    _set_phase_(nto_o_a)
    _set_phase_(nto_o_b)
    _set_phase_(nto_v_a)
    _set_phase_(nto_v_b)

    occupied_nto_a = numpy.dot(orbo_a, nto_o_a)
    occupied_nto_b = numpy.dot(orbo_b, nto_o_b)
    virtual_nto_a = numpy.dot(orbv_a, nto_v_a)
    virtual_nto_b = numpy.dot(orbv_b, nto_v_b)
    nto_coeff = (numpy.hstack((occupied_nto_a, virtual_nto_a)),
                 numpy.hstack((occupied_nto_b, virtual_nto_b)))

    if mol.symmetry:
        nto_coeff = (lib.tag_array(nto_coeff[0], orbsym=nto_orbsyma),
                     lib.tag_array(nto_coeff[1], orbsym=nto_orbsymb))

    log = logger.new_logger(tdobj, verbose)
    if log.verbose >= logger.INFO:
        log.info('State %d: %g eV  NTO largest component %s',
                 state_id+1, tdobj.e[state_id]*nist.HARTREE2EV,
                 weights_a[0]+weights_b[0])
        fmt = '%' + str(lib.param.OUTPUT_DIGITS) + 'f (MO #%d)'
        o_idx_a = numpy.where(abs(nto_o_a[:,0]) > threshold)[0]
        v_idx_a = numpy.where(abs(nto_v_a[:,0]) > threshold)[0]
        o_idx_b = numpy.where(abs(nto_o_b[:,0]) > threshold)[0]
        v_idx_b = numpy.where(abs(nto_v_b[:,0]) > threshold)[0]
        log.info('    alpha occ-NTO: ' +
                 ' + '.join([(fmt % (nto_o_a[i,0], i+MO_BASE))
                             for i in o_idx_a]))
        log.info('    alpha vir-NTO: ' +
                 ' + '.join([(fmt % (nto_v_a[i,0], i+MO_BASE+nocc_a))
                             for i in v_idx_a]))
        log.info('    beta occ-NTO: ' +
                 ' + '.join([(fmt % (nto_o_b[i,0], i+MO_BASE))
                             for i in o_idx_b]))
        log.info('    beta vir-NTO: ' +
                 ' + '.join([(fmt % (nto_v_b[i,0], i+MO_BASE+nocc_b))
                             for i in v_idx_b]))
    return (weights_a, weights_b), nto_coeff


def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocc_a = numpy.count_nonzero(mo_occ[0] == 1)
    nocc_b = numpy.count_nonzero(mo_occ[1] == 1)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    log.note('\n** Excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        x_syma = symm.direct_prod(orbsyma[mo_occ[0]==1], orbsyma[mo_occ[0]==0], mol.groupname)
        x_symb = symm.direct_prod(orbsymb[mo_occ[1]==1], orbsymb[mo_occ[1]==0], mol.groupname)
    else:
        x_syma = None

    f_oscillator = tdobj.oscillator_strength()
    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        if x_syma is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        else:
            wfnsyma = rhf.analyze_wfnsym(tdobj, x_syma, x[0])
            wfnsymb = rhf.analyze_wfnsym(tdobj, x_symb, x[1])
            if wfnsyma == wfnsymb:
                wfnsym = wfnsyma
            else:
                wfnsym = '???'
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])

        if log.verbose >= logger.INFO:
            for o, v in zip(* numpy.where(abs(x[0]) > 0.1)):
                log.info('    %4da -> %4da %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc_a, x[0][o,v])
            for o, v in zip(* numpy.where(abs(x[1]) > 0.1)):
                log.info('    %4db -> %4db %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc_b, x[1][o,v])

    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_dip = tdobj.transition_dipole()
        for i, ei in enumerate(tdobj.e):
            dip = trans_dip[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, dip[0], dip[1], dip[2], numpy.dot(dip, dip),
                     f_oscillator[i])

        log.info('\n** Transition velocity dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_v = tdobj.transition_velocity_dipole()
        f_v = tdobj.oscillator_strength(gauge='velocity', order=0)
        for i, ei in enumerate(tdobj.e):
            v = trans_v[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, v[0], v[1], v[2], numpy.dot(v, v), f_v[i])

        log.info('\n** Transition magnetic dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z')
        trans_m = tdobj.transition_magnetic_dipole()
        for i, ei in enumerate(tdobj.e):
            m = trans_m[i]
            log.info('%3d    %11.4f %11.4f %11.4f',
                     i+1, m[0], m[1], m[2])
    return tdobj


def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    if xy is None: xy = tdobj.xy
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo_a = mo_coeff[0][:,mo_occ[0]==1]
    orbv_a = mo_coeff[0][:,mo_occ[0]==0]
    orbo_b = mo_coeff[1][:,mo_occ[1]==1]
    orbv_b = mo_coeff[1][:,mo_occ[1]==0]

    ints_a = numpy.einsum('...pq,pi,qj->...ij', ints, orbo_a.conj(), orbv_a)
    ints_b = numpy.einsum('...pq,pi,qj->...ij', ints, orbo_b.conj(), orbv_b)
    pol = [(numpy.einsum('...ij,ij->...', ints_a, x[0]) +
            numpy.einsum('...ij,ij->...', ints_b, x[1])) for x,y in xy]
    pol = numpy.array(pol)
    y = xy[0][1]
    if isinstance(y[0], numpy.ndarray):
        pol_y = [(numpy.einsum('...ij,ij->...', ints_a, y[0]) +
                  numpy.einsum('...ij,ij->...', ints_b, y[1])) for x,y in xy]
        if hermi:
            pol += pol_y
        else:  # anti-Hermitian
            pol -= pol_y
    return pol


class TDBase(rhf.TDBase):

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    analyze = analyze
    get_nto = get_nto
    _contract_multipole = _contract_multipole  # needed by transition dipoles

    def nuc_grad_method(self):
        from pyscf.grad import tduhf
        return tduhf.Gradients(self)


@lib.with_doc(rhf.TDA.__doc__)
class TDA(TDBase):

    singlet = None

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_hop(mf, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mol = mf.mol
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        e_ia_a = (mo_energy[0][viridxa,None] - mo_energy[0][occidxa]).T
        e_ia_b = (mo_energy[1][viridxb,None] - mo_energy[1][occidxb]).T

        if wfnsym is not None and mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
            orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mf.mo_coeff)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
            orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
            e_ia_a[(orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym] = 1e99
            e_ia_b[(orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym] = 1e99

        e_ia = numpy.hstack((e_ia_a.ravel(), e_ia_b.ravel()))
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = numpy.sort(e_ia)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [((xi[:nocca*nvira].reshape(nocca,nvira),  # X_alpha
                     xi[nocca*nvira:].reshape(noccb,nvirb)), # X_beta
                    (0, 0))  # (Y_alpha, Y_beta)
                   for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

CIS = TDA


def gen_tdhf_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert (mo_coeff[0].dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff[0].shape
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
        orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
        sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
        sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    e_ia = hdiag = numpy.hstack((e_ia_a.ravel(), e_ia_b.ravel()))
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag, -hdiag))

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = mf.gen_response(hermi=0, max_memory=max_memory)

    def vind(xys):
        nz = len(xys)
        xys = numpy.asarray(xys).reshape(nz,2,-1)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,-1): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2)
        xa = xs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        xb = xs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        ya = ys[:,:nocca*nvira].reshape(nz,nocca,nvira)
        yb = ys[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        # dms = AX + BY
        dmsa  = lib.einsum('xov,qv,po->xpq', xa, orbva.conj(), orboa)
        dmsb  = lib.einsum('xov,qv,po->xpq', xb, orbvb.conj(), orbob)
        dmsa += lib.einsum('xov,pv,qo->xpq', ya, orbva, orboa.conj())
        dmsb += lib.einsum('xov,pv,qo->xpq', yb, orbvb, orbob.conj())

        v1ao = vresp(numpy.asarray((dmsa,dmsb)))

        v1aov = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        v1avo = lib.einsum('xpq,qo,pv->xov', v1ao[0], orboa, orbva.conj())
        v1bov = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        v1bvo = lib.einsum('xpq,qo,pv->xov', v1ao[1], orbob, orbvb.conj())

        v1ov = xs * e_ia  # AX
        v1vo = ys * e_ia  # AY
        v1ov[:,:nocca*nvira] += v1aov.reshape(nz,-1)
        v1vo[:,:nocca*nvira] += v1avo.reshape(nz,-1)
        v1ov[:,nocca*nvira:] += v1bov.reshape(nz,-1)
        v1vo[:,nocca*nvira:] += v1bvo.reshape(nz,-1)
        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0
        hx = numpy.hstack((v1ov, -v1vo))
        return hx

    return vind, hdiag


class TDHF(TDA):

    singlet = None

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(mf, singlet=self.singlet, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        x0 = TDA.init_guess(self, mf, nstates, wfnsym)
        y0 = numpy.zeros_like(x0)
        return numpy.asarray(numpy.block([[x0, y0], [y0, x0.conj()]]))

    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                      real_eigenvectors=True)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_cycle=self.max_cycle,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca = (self._scf.mo_occ[0]>0).sum()
        noccb = (self._scf.mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        e = []
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                e.append(w[i])
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.e = numpy.array(e)
        self.xy = xy

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

RPA = TDUHF = TDHF

scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)

del (OUTPUT_THRESHOLD)
