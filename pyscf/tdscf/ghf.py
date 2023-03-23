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
from pyscf import dft
from pyscf.dft import numint
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf.scf import ghf_symm
from pyscf.data import nist
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)


def gen_tda_operation(mf, fock_ao=None, wfnsym=None):
    '''A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
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
        orbsym = ghf_symm.get_orbsym(mol, mo_coeff)
        orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
        sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

    if fock_ao is None:
        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

    hdiag = fvv.diagonal() - foo.diagonal()[:,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel().real

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(hermi=0)

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs)
            zs[:,sym_forbid] = 0

        dmov = lib.einsum('xov,qv,po->xpq', zs, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1ov += lib.einsum('xqs,sp->xqp', zs, fvv)
        v1ov -= lib.einsum('xpr,sp->xsr', zs, foo)
        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

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

    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
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
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb
        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb
        return a, b

    if isinstance(mf, dft.KohnShamDFT):
        from pyscf.dft import xc_deriv
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            raise NotImplementedError('DKS-TDDFT NLC functional')

        if not mf.collinear:
            raise NotImplementedError

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        a, b = add_hf_(a, b, hyb)

        if ni.collinear[0] == 'm':  # mcol
            a = a.astype(numpy.complex128)
            b = b.astype(numpy.complex128)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
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
                    w_ov = numpy.einsum('tsr,tria->sria', wfxc, rho_ov)
                    a += lib.einsum('sria,srjb->iajb', w_ov, rho_vo)
                    b += lib.einsum('sria,srjb->iajb', w_ov, rho_ov)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2)[2]
                    wv_a, wv_b = weight * fxc.reshape(2,2,-1)
                    mo_oa, mo_va, mo_ob, mo_vb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('ri,ra->ria', mo_oa.conj(), mo_va)
                    rho_ov_b = numpy.einsum('ri,ra->ria', mo_ob.conj(), mo_vb)
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_ov  = wv_a[:,:,None,None] * rho_ov_a
                    w_ov += wv_b[:,:,None,None] * rho_ov_b
                    wa_ov, wb_ov = w_ov
                    a += lib.einsum('ria,rjb->iajb', wa_ov, rho_vo_a)
                    a += lib.einsum('ria,rjb->iajb', wb_ov, rho_vo_b)
                    b += lib.einsum('ria,rjb->iajb', wa_ov, rho_ov_a)
                    b += lib.einsum('ria,rjb->iajb', wb_ov, rho_ov_b)
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
                    w_ov = numpy.einsum('txsyr,txria->syria', wfxc, rho_ov)
                    a += lib.einsum('syria,syrjb->iajb', w_ov, rho_vo)
                    b += lib.einsum('syria,syrjb->iajb', w_ov, rho_ov)
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
                    w_ov  = numpy.einsum('xsyr,xria->syria', wv_a, rho_ov_a)
                    w_ov += numpy.einsum('xsyr,xria->syria', wv_b, rho_ov_b)
                    wa_ov, wb_ov = w_ov
                    a += lib.einsum('xria,xrjb->iajb', wa_ov, rho_vo_a)
                    a += lib.einsum('xria,xrjb->iajb', wb_ov, rho_vo_b)
                    b += lib.einsum('xria,xrjb->iajb', wa_ov, rho_ov_a)
                    b += lib.einsum('xria,xrjb->iajb', wb_ov, rho_ov_b)
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
                    w_ov = numpy.einsum('txsyr,txria->syria', wfxc, rho_ov)
                    a += lib.einsum('syria,syrjb->iajb', w_ov, rho_vo)
                    b += lib.einsum('syria,syrjb->iajb', w_ov, rho_ov)
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
                    w_ov  = numpy.einsum('xsyr,xria->syria', wv_a, rho_ov_a)
                    w_ov += numpy.einsum('xsyr,xria->syria', wv_b, rho_ov_b)
                    wa_ov, wb_ov = w_ov
                    a += lib.einsum('xria,xrjb->iajb', wa_ov, rho_vo_a)
                    a += lib.einsum('xria,xrjb->iajb', wb_ov, rho_vo_b)
                    b += lib.einsum('xria,xrjb->iajb', wa_ov, rho_ov_a)
                    b += lib.einsum('xria,xrjb->iajb', wb_ov, rho_ov_b)
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


class TDMixin(rhf.TDMixin):

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    analyze = analyze
    get_nto = get_nto
    _contract_multipole = _contract_multipole  # needed by transition dipoles

    def nuc_grad_method(self):
        raise NotImplementedError


@lib.with_doc(rhf.TDA.__doc__)
class TDA(TDMixin):

    singlet = None

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_hop(mf, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = numpy.where(mo_occ==1)[0]
        viridx = numpy.where(mo_occ==0)[0]
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia_max = e_ia.max()

        if wfnsym is not None and mf.mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mf.mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsym = ghf_symm.get_orbsym(mf.mol, mf.mo_coeff)
            orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
            e_ia[(orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym] = 1e99

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_ia = e_ia.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6

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

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        # FIXME: Is it correct to call davidson1 for complex integrals
        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        self.xy = [(xi.reshape(nocc,nvir), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

CIS = TDA


def gen_tdhf_operation(mf, fock_ao=None, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
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
        orbsym = ghf_symm.get_orbsym(mol, mo_coeff)
        orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
        sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

    #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
    #fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
    #foo = fock[occidx[:,None],occidx]
    #fvv = fock[viridx[:,None],viridx]
    foo = numpy.diag(mo_energy[occidx])
    fvv = numpy.diag(mo_energy[viridx])

    hdiag = fvv.diagonal() - foo.diagonal()[:,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel())).real

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(hermi=0)

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys)
            xys[:,:,sym_forbid] = 0

        xs, ys = xys.transpose(1,0,2,3)
        # dms = AX + BY
        dms  = lib.einsum('xov,qv,po->xpq', xs, orbv.conj(), orbo)
        dms += lib.einsum('xov,pv,qo->xpq', ys, orbv, orbo.conj())

        v1ao = vresp(dms)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        v1ov += lib.einsum('xqs,sp->xqp', xs, fvv)  # AX
        v1ov -= lib.einsum('xpr,sp->xsr', xs, foo)  # AX
        v1vo += lib.einsum('xqs,sp->xqp', ys, fvv.conj())  # (A*)Y
        v1vo -= lib.einsum('xpr,sp->xsr', ys, foo.conj())  # (A*)Y

        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
            v1vo[:,sym_forbid] = 0

        # (AX, (-A*)Y)
        nz = xys.shape[0]
        hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx

    return vind, hdiag


class TDHF(TDMixin):

    singlet = None

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(mf, wfnsym=self.wfnsym)

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
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        ensure_real = self._scf.mo_coeff.dtype == numpy.double
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            # FIXME: Should the amplitudes be real? It also affects x2c-tdscf
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, ensure_real)

        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_cycle=self.max_cycle,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(1./norm)
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

RPA = TDGHF = TDHF

from pyscf import scf
scf.ghf.GHF.TDA = lib.class_as_method(TDA)
scf.ghf.GHF.TDHF = lib.class_as_method(TDHF)
