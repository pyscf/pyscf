#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
import numpy
from pyscf import lib
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf import ghf, gks
from pyscf.x2c import x2c
from pyscf.x2c import dft as x2c_dft
# To ensure .gen_response() methods are registered
from pyscf.x2c import _response_functions  # noqa
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_uhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_uhf_TDDFT_pick_eig_threshold', 1e-4)

def gen_tda_operation(mf, fock_ao=None):
    '''A x
    '''
    return ghf.gen_tda_operation(mf, fock_ao, None)
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
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ == 1)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    nmo = nocc + nvir
    mo = numpy.hstack((orbo, orbv))

    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        eri_mo = ao2mo.kernel(mol, [orbo, mo, mo, mo], intor='int2e_spinor')
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)

        a = a + numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
        a = a - numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb
        b = b + numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b = b - numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb
        return a, b

    if isinstance(mf, dft.KohnShamDFT):
        from pyscf.dft import xc_deriv
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            raise NotImplementedError('X2C-TDDFT for NLC functionals')

        if not mf.collinear:
            raise NotImplementedError

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        a, b = add_hf_(a, b, hyb)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        def get_mo_value(ao):
            ao_a, ao_b = ao
            if ao_a.ndim == 2:
                mo_a = lib.einsum('rp,pi->ri', ao_a, mo)
                mo_b = lib.einsum('rp,pi->ri', ao_b, mo)
                return mo_a[:,:nocc], mo_a[:,nocc:], mo_b[:,:nocc], mo_b[:,nocc:]
            else:
                mo_a = lib.einsum('xrp,pi->xri', ao_a, mo)
                mo_b = lib.einsum('xrp,pi->xri', ao_b, mo)
                return mo_a[:,:,:nocc], mo_a[:,:,nocc:], mo_b[:,:,:nocc], mo_b[:,:,nocc:]

        def ud2tm(aa, ab, ba, bb):
            return numpy.stack([aa + bb,        # rho
                                ba + ab,        # mx
                                (ba - ab) * 1j, # my
                                aa - bb])       # mz

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory,
                                     with_s=False):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
                    fxc = eval_xc(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = weight * fxc.reshape(4,4,-1)
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
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1)
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
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory,
                                     with_s=False):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
                    fxc = eval_xc(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = weight * fxc
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
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1)
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
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory,
                                     with_s=False):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    eval_xc = ni.mcfun_eval_xc_adapter(mf.xc)
                    fxc = eval_xc(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = weight * fxc
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


class TDMixin(ghf.TDMixin):

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    analyze = analyze
    get_nto = get_nto
    _contract_multipole = _contract_multipole  # needed by transition dipoles

    def nuc_grad_method(self):
        raise NotImplementedError


class TDA(TDMixin, ghf.TDA):
    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_hop(mf)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        assert self.wfnsym is None
        return ghf.TDA.init_guess(self, mf, nstates, None)

    kernel = ghf.TDA.kernel


def gen_tdhf_operation(mf, fock_ao=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]
    '''
    return ghf.gen_tdhf_operation(mf, fock_ao, None)


class TDHF(TDMixin, ghf.TDHF):
    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(mf)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        assert self.wfnsym is None
        return ghf.TDHF.init_guess(self, mf, nstates, None)

    kernel = ghf.TDHF.kernel

TDDFT = TDHF

x2c.RHF.TDA  = x2c.UHF.TDA  = lib.class_as_method(TDA)
x2c.RHF.TDHF = x2c.UHF.TDHF = lib.class_as_method(TDHF)

x2c_dft.RKS.TDA   = x2c_dft.UKS.TDA   = lib.class_as_method(TDA)
x2c_dft.RKS.TDHF  = x2c_dft.UKS.TDHF  = None
x2c_dft.RKS.TDDFT = x2c_dft.UKS.TDDFT = lib.class_as_method(TDDFT)
