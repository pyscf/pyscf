#!/usr/bin/env python
# Copyright 2021-2022 The PySCF Developers. All Rights Reserved.
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
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
# Recent Advances in Density Functional Methods, Chapter 5, M. E. Casida
#


from functools import reduce
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from pyscf.scf import ghf_symm
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)


def gen_tda_operation(mf, fock_ao=None, wfnsym=None, with_nlc=True):
    '''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    td = TDA(mf)
    td.exclude_nlc = not with_nlc
    return _gen_tda_operation(td, fock_ao, wfnsym)
gen_tda_hop = gen_tda_operation

def _gen_tda_operation(td, fock_ao=None, wfnsym=None):
    assert fock_ao is None
    mf = td._scf
    mol = mf.mol
    mask = td.get_frozen_mask()
    mo_coeff = mf.mo_coeff[:, mask]
    mo_energy = mf.mo_energy[mask]
    mo_occ = mf.mo_occ[mask]
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ == 1)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        sym_forbid = _get_x_sym_table(td) != wfnsym

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel().real

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = td.gen_response(hermi=0)

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo.conj())
        v1ao = vresp(dms)
        v1mo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        v1mo += numpy.einsum('xia,ia->xia', zs, e_ia)
        if wfnsym is not None and mol.symmetry:
            v1mo[:,sym_forbid] = 0
        return v1mo.reshape(v1mo.shape[0],-1)

    return vind, hdiag

def _get_x_sym_table(td):
    '''Irrep (up to D2h symmetry) of each coefficient in X[nocc,nvir]'''
    mf = td._scf
    mol = mf.mol
    mask = td.get_frozen_mask()
    mo_occ = mf.mo_occ[mask]
    orbsym = ghf_symm.get_orbsym(mol, mf.mo_coeff[:, mask])
    orbsym = numpy.asarray(orbsym) % 10  # convert to D2h irreps
    return orbsym[mo_occ==1,None] ^ orbsym[mo_occ==0]

def get_ab(mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    mo_coeff0 = numpy.copy(mo_coeff)
    mo_occ0 = numpy.copy(mo_occ)

    if frozen is not None:
        # see get_frozen_mask()
        moidx = numpy.ones(mf.mo_occ.size, dtype=bool)
        if isinstance(frozen, (int, numpy.integer)):
            moidx[:frozen] = False
        elif hasattr(frozen, '__len__'):
            moidx[list(frozen)] = False
        else:
            raise NotImplementedError
        mo_energy = mo_energy[moidx]
        mo_coeff = mo_coeff[:, moidx]
        mo_occ = mo_occ[moidx]

    mol = mf.mol
    nmo = mo_occ.size
    nao = mol.nao
    occidx = numpy.where(mo_occ==1)[0]
    viridx = numpy.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))
    moa = mo[:nao].copy()
    mob = mo[nao:].copy()
    orboa = orbo[:nao]
    orbob = orbo[nao:]
    nmo = nocc + nvir

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir).astype(mo_coeff.dtype)
    b = numpy.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        if mo_coeff.dtype == numpy.double:
            eri_mo  = ao2mo.general(mol, [orboa,moa,moa,moa], compact=False)
            eri_mo += ao2mo.general(mol, [orbob,mob,mob,mob], compact=False)
            eri_mo += ao2mo.general(mol, [orboa,moa,mob,mob], compact=False)
            eri_mo += ao2mo.general(mol, [orbob,mob,moa,moa], compact=False)
            eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        else:
            eri_ao = mol.intor('int2e').reshape([nao]*4)
            eri_mo_a = lib.einsum('pqrs,pi,qj->ijrs', eri_ao, orboa.conj(), moa)
            eri_mo_a+= lib.einsum('pqrs,pi,qj->ijrs', eri_ao, orbob.conj(), mob)
            eri_mo = lib.einsum('ijrs,rk,sl->ijkl', eri_mo_a, moa.conj(), moa)
            eri_mo+= lib.einsum('ijrs,rk,sl->ijkl', eri_mo_a, mob.conj(), mob)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc].conj())
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:].conj()) * hyb
        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:].conj())
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:].conj()) * hyb
        return a, b

    if isinstance(mf, scf.hf.KohnShamDFT):
        from pyscf.dft import xc_deriv
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.do_nlc():
            raise NotImplementedError('DKS-TDDFT NLC functional')

        if not mf.collinear:
            raise NotImplementedError

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        assert omega == 0.

        a, b = add_hf_(a, b, hyb)

        if ni.collinear[0] == 'm':  # mcol
            a = a.astype(numpy.complex128)
            b = b.astype(numpy.complex128)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff0, mo_occ0)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        def get_mo_value(ao):
            if ao.ndim == 2:
                mo_a = lib.einsum('rp,pi->ri', ao, moa)
                mo_b = lib.einsum('rp,pi->ri', ao, mob)
                return mo_a[:,:nocc], mo_a[:,nocc:], mo_b[:,:nocc], mo_b[:,nocc:]
            else:
                mo_a = lib.einsum('xrp,pi->xri', ao, moa)
                mo_b = lib.einsum('xrp,pi->xri', ao, mob)
                return mo_a[:,:,:nocc], mo_a[:,:,nocc:], mo_b[:,:,:nocc], mo_b[:,:,nocc:]

        def ud2tm(aa, ab, ba, bb):
            return numpy.stack([aa + bb,        # rho
                                ba + ab,        # mx
                                (ba - ab) * 1j, # my
                                aa - bb])       # mz

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
                    fxc = eval_xc(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = weight * fxc.reshape(4,4,-1)
                    wr, wmx, wmy, wmz = weight * fxc.reshape(4,4,-1)
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_aa = numpy.einsum('ri,ra->ria', mo_oa.conj(), mo_va)
                    rho_ov_ab = numpy.einsum('ri,ra->ria', mo_oa.conj(), mo_vb)
                    rho_ov_ba = numpy.einsum('ri,ra->ria', mo_ob.conj(), mo_va)
                    rho_ov_bb = numpy.einsum('ri,ra->ria', mo_ob.conj(), mo_vb)
                    rho_ov = ud2tm(rho_ov_aa, rho_ov_ab, rho_ov_ba, rho_ov_bb)
                    rho_vo = rho_ov.conj()
                    w_vo = numpy.einsum('tsr,tria->sria', wfxc, rho_vo)
                    a += lib.einsum('sria,srjb->iajb', w_vo, rho_ov)
                    b += lib.einsum('sria,srjb->iajb', w_vo, rho_vo)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2)[2]
                    wv_a, wv_b = weight * fxc.reshape(2,2,-1)
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('ri,ra->ria', mo_oa.conj(), mo_va)
                    rho_ov_b = numpy.einsum('ri,ra->ria', mo_ob.conj(), mo_vb)
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_vo  = wv_a[:,:,None,None] * rho_vo_a
                    w_vo += wv_b[:,:,None,None] * rho_vo_b
                    wa_vo, wb_vo = w_vo
                    a += lib.einsum('ria,rjb->iajb', wa_vo, rho_ov_a)
                    a += lib.einsum('ria,rjb->iajb', wb_vo, rho_ov_b)
                    b += lib.einsum('ria,rjb->iajb', wa_vo, rho_vo_a)
                    b += lib.einsum('ria,rjb->iajb', wb_vo, rho_vo_b)
                else:
                    raise NotImplementedError(ni.collinear)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
                    fxc = eval_xc(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = weight * fxc
                    wr, wmx, wmy, wmz = weight * fxc
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_aa = numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_va)
                    rho_ov_ab = numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_vb)
                    rho_ov_ba = numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_va)
                    rho_ov_bb = numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_vb)
                    rho_ov_aa[1:4] += numpy.einsum('xri,ra->xria', mo_oa[1:4].conj(), mo_va[0])
                    rho_ov_ab[1:4] += numpy.einsum('xri,ra->xria', mo_oa[1:4].conj(), mo_vb[0])
                    rho_ov_ba[1:4] += numpy.einsum('xri,ra->xria', mo_ob[1:4].conj(), mo_va[0])
                    rho_ov_bb[1:4] += numpy.einsum('xri,ra->xria', mo_ob[1:4].conj(), mo_vb[0])
                    rho_ov = ud2tm(rho_ov_aa, rho_ov_ab, rho_ov_ba, rho_ov_bb)
                    rho_vo = rho_ov.conj()
                    w_vo = numpy.einsum('txsyr,txria->syria', wfxc, rho_vo)
                    a += lib.einsum('syria,syrjb->iajb', w_vo, rho_ov)
                    b += lib.einsum('syria,syrjb->iajb', w_vo, rho_vo)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2)[2]
                    wv_a, wv_b = weight * fxc
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('xri,ra->xria', mo_oa.conj(), mo_va[0])
                    rho_ov_b = numpy.einsum('xri,ra->xria', mo_ob.conj(), mo_vb[0])
                    rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_va[1:4])
                    rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_vb[1:4])
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_vo  = numpy.einsum('xsyr,xria->syria', wv_a, rho_vo_a)
                    w_vo += numpy.einsum('xsyr,xria->syria', wv_b, rho_vo_b)
                    wa_vo, wb_vo = w_vo
                    a += lib.einsum('xria,xrjb->iajb', wa_vo, rho_ov_a)
                    a += lib.einsum('xria,xrjb->iajb', wb_vo, rho_ov_b)
                    b += lib.einsum('xria,xrjb->iajb', wa_vo, rho_vo_a)
                    b += lib.einsum('xria,xrjb->iajb', wb_vo, rho_vo_b)
                else:
                    raise NotImplementedError(ni.collinear)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
                    fxc = eval_xc(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = weight * fxc
                    wr, wmx, wmy, wmz = weight * fxc
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_aa = numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_va)
                    rho_ov_ab = numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_vb)
                    rho_ov_ba = numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_va)
                    rho_ov_bb = numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_vb)
                    rho_ov_aa[1:4] += numpy.einsum('xri,ra->xria', mo_oa[1:4].conj(), mo_va[0])
                    rho_ov_ab[1:4] += numpy.einsum('xri,ra->xria', mo_oa[1:4].conj(), mo_vb[0])
                    rho_ov_ba[1:4] += numpy.einsum('xri,ra->xria', mo_ob[1:4].conj(), mo_va[0])
                    rho_ov_bb[1:4] += numpy.einsum('xri,ra->xria', mo_ob[1:4].conj(), mo_vb[0])
                    tau_ov_aa = numpy.einsum('xri,xra->ria', mo_oa[1:4].conj(), mo_va[1:4]) * .5
                    tau_ov_ab = numpy.einsum('xri,xra->ria', mo_oa[1:4].conj(), mo_vb[1:4]) * .5
                    tau_ov_ba = numpy.einsum('xri,xra->ria', mo_ob[1:4].conj(), mo_va[1:4]) * .5
                    tau_ov_bb = numpy.einsum('xri,xra->ria', mo_ob[1:4].conj(), mo_vb[1:4]) * .5
                    rho_ov_aa = numpy.vstack([rho_ov_aa, tau_ov_aa[numpy.newaxis]])
                    rho_ov_ab = numpy.vstack([rho_ov_ab, tau_ov_ab[numpy.newaxis]])
                    rho_ov_ba = numpy.vstack([rho_ov_ba, tau_ov_ba[numpy.newaxis]])
                    rho_ov_bb = numpy.vstack([rho_ov_bb, tau_ov_bb[numpy.newaxis]])
                    rho_ov = ud2tm(rho_ov_aa, rho_ov_ab, rho_ov_ba, rho_ov_bb)
                    rho_vo = rho_ov.conj()
                    w_vo = numpy.einsum('txsyr,txria->syria', wfxc, rho_vo)
                    a += lib.einsum('syria,syrjb->iajb', w_vo, rho_ov)
                    b += lib.einsum('syria,syrjb->iajb', w_vo, rho_vo)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2)[2]
                    wv_a, wv_b = weight * fxc
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('xri,ra->xria', mo_oa.conj(), mo_va[0])
                    rho_ov_b = numpy.einsum('xri,ra->xria', mo_ob.conj(), mo_vb[0])
                    rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_va[1:4])
                    rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_vb[1:4])
                    tau_ov_a = numpy.einsum('xri,xra->ria', mo_oa[1:4].conj(), mo_va[1:4]) * .5
                    tau_ov_b = numpy.einsum('xri,xra->ria', mo_ob[1:4].conj(), mo_vb[1:4]) * .5
                    rho_ov_a = numpy.vstack([rho_ov_a, tau_ov_a[numpy.newaxis]])
                    rho_ov_b = numpy.vstack([rho_ov_b, tau_ov_b[numpy.newaxis]])
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_vo  = numpy.einsum('xsyr,xria->syria', wv_a, rho_vo_a)
                    w_vo += numpy.einsum('xsyr,xria->syria', wv_b, rho_vo_b)
                    wa_vo, wb_vo = w_vo
                    a += lib.einsum('xria,xrjb->iajb', wa_vo, rho_ov_a)
                    a += lib.einsum('xria,xrjb->iajb', wb_vo, rho_ov_b)
                    b += lib.einsum('xria,xrjb->iajb', wa_vo, rho_vo_a)
                    b += lib.einsum('xria,xrjb->iajb', wb_vo, rho_vo_b)
                else:
                    raise NotImplementedError(ni.collinear)

    else:
        a, b = add_hf_(a, b)

    return a, b

def get_nto(tdobj, state=1, threshold=OUTPUT_THRESHOLD, verbose=None):
    raise NotImplementedError('get_nto')

def analyze(tdobj, verbose=None):
    raise NotImplementedError('analyze')

def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    raise NotImplementedError


class TDBase(rhf.TDBase):

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    analyze = analyze
    get_nto = get_nto
    _contract_multipole = _contract_multipole  # needed by transition dipoles


@lib.with_doc(rhf.TDA.__doc__)
class TDA(TDBase):

    singlet = None

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        assert mf is None or mf is self._scf
        return _gen_tda_operation(self, wfnsym=self.wfnsym)

    def get_init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mask = self.get_frozen_mask()
        mo_energy = mf.mo_energy[mask]
        mo_occ = mf.mo_occ[mask]
        occidx = numpy.where(mo_occ==1)[0]
        viridx = numpy.where(mo_occ==0)[0]
        e_ia = (mo_energy[viridx] - mo_energy[occidx,None]).ravel()
        nov = e_ia.size
        nstates = min(nstates, nov)

        if (wfnsym is not None or return_symmetry) and mf.mol.symmetry:
            x_sym = _get_x_sym_table(self).ravel()
            if wfnsym is not None:
                if isinstance(wfnsym, str):
                    wfnsym = symm.irrep_name2id(mf.mol.groupname, wfnsym)
                wfnsym = wfnsym % 10  # convert to D2h subgroup
                e_ia[x_sym != wfnsym] = 1e99
                nov_allowed = numpy.count_nonzero(x_sym == wfnsym)
                nstates = min(nstates, nov_allowed)

        e_threshold = numpy.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations

        if return_symmetry:
            if mf.mol.symmetry:
                x0sym = x_sym[idx]
            else:
                x0sym = None
            return x0, x0sym
        else:
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
        mol = self.mol

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0, x0sym = self.get_init_guess(
                self._scf, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym = _get_x_sym_table(self).ravel()
            x0sym = [rhf._guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mo_occ = self._scf.mo_occ[self.get_frozen_mask()]
        nocc = (mo_occ>0).sum()
        nmo = mo_occ.size
        nvir = nmo - nocc
        self.xy = [(xi.reshape(nocc,nvir), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

CIS = TDA


def gen_tdhf_operation(mf, fock_ao=None, wfnsym=None, with_nlc=True):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    td = TDHF(mf)
    td.exclude_nlc = not with_nlc
    return _gen_tdhf_operation(td, fock_ao, wfnsym)

def _gen_tdhf_operation(td, fock_ao=None, wfnsym=None):
    mf = td._scf
    mol = mf.mol
    mask = td.get_frozen_mask()
    mo_coeff = mf.mo_coeff[:, mask]
    mo_energy = mf.mo_energy[mask]
    mo_occ = mf.mo_occ[mask]
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ == 1)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        sym_forbid = _get_x_sym_table(td) != wfnsym

    assert fock_ao is None

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel())).real

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = td.gen_response(hermi=0)

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2,3)
        dms  = lib.einsum('xov,pv,qo->xpq', xs, orbv, orbo.conj())
        dms += lib.einsum('xov,qv,po->xpq', ys, orbv.conj(), orbo)
        v1ao = vresp(dms) # = <mj||nb> Xjb + <mb||nj> Yjb
        # A ~= <aj||ib>, B = <ab||ij>
        # AX + BY
        # = <aj||ib> Xjb + <ab||ij> Yjb
        # = (<mj||nb> Xjb + <mb||nj> Yjb) Cma* Cni
        v1_top = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        # (B*)X + (A*)Y
        # = <ij||ab> Xjb + <ib||aj> Yjb
        # = (<mj||nb> Xjb + <mb||nj> Yjb) Cmi* Cna
        v1_bot = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1_top += numpy.einsum('xia,ia->xia', xs, e_ia)  # AX
        v1_bot += numpy.einsum('xia,ia->xia', ys, e_ia)  # (A*)Y

        if wfnsym is not None and mol.symmetry:
            v1_top[:,sym_forbid] = 0
            v1_bot[:,sym_forbid] = 0

        # (AX, (-A*)Y)
        nz = xys.shape[0]
        hx = numpy.hstack((v1_top.reshape(nz,-1), -v1_bot.reshape(nz,-1)))
        return hx

    return vind, hdiag


class TDHF(TDBase):

    singlet = None

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        assert mf is None or mf is self._scf
        return _gen_tdhf_operation(self, wfnsym=self.wfnsym)

    def get_init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        if return_symmetry:
            x0, x0sym = TDA.get_init_guess(self, mf, nstates, wfnsym, return_symmetry)
            y0 = numpy.zeros_like(x0)
            return numpy.hstack([x0, y0]), x0sym
        else:
            x0 = TDA.get_init_guess(self, mf, nstates, wfnsym, return_symmetry)
            y0 = numpy.zeros_like(x0)
            return numpy.hstack([x0, y0])

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
        mol = self.mol

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        ensure_real = self._scf.mo_coeff.dtype == numpy.double
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            # FIXME: Should the amplitudes be real? It also affects x2c-tdscf
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, ensure_real)

        x0sym = None
        if x0 is None:
            x0, x0sym = self.get_init_guess(
                self._scf, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym = y_sym = _get_x_sym_table(self).ravel()
            x_sym = numpy.append(x_sym, y_sym)
            x0sym = [rhf._guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, w, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mo_occ = self._scf.mo_occ[self.get_frozen_mask()]
        nocc = (mo_occ>0).sum()
        nmo = mo_occ.size
        nvir = nmo - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm < 0:
                log.warn('TDDFT amplitudes |X| smaller than |Y|')
            norm = abs(norm)**-.5
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

RPA = TDGHF = TDHF

scf.ghf.GHF.TDA = lib.class_as_method(TDA)
scf.ghf.GHF.TDHF = lib.class_as_method(TDHF)
