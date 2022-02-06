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

# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_uhf_TDDFT_positive_eig_threshold', 1e-3)

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
    assert(mo_coeff.dtype == numpy.double)

    mol = mf.mol
    nao, nmo = mo_coeff.shape
    n2c = nmo // 2
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
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

        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb
        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb

    if isinstance(mf, dft.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            raise NotImplementedError

        if not mf.collinear:
            raise NotImplementedError

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc(mf.xc, rho, 1, deriv=2)[2]
                u_u, u_d, d_d = fxc[0].T

                aoa, aob = ao
                mo_oa = lib.einsum('rp,pi->ri', aoa, orbo)
                mo_va = lib.einsum('rp,pi->ri', aoa, orbv)
                mo_ob = lib.einsum('rp,pi->ri', aob, orbo)
                mo_vb = lib.einsum('rp,pi->ri', aob, orbv)
                rho_ov_a = numpy.einsum('ri,ra->ria', mo_oa.conj(), mo_va)
                rho_ov_b = numpy.einsum('ri,ra->ria', mo_ob.conj(), mo_vb)
                rho_vo_a = rho_ov_a.conj()
                rho_vo_b = rho_ov_b.conj()

                w_ov_a = numpy.einsum('ria,r->ria', u_u * rho_ov_a + u_d * rho_ov_b, weight)
                a += lib.einsum('ria,rjb->iajb', w_ov_a, rho_vo_a)
                b += lib.einsum('ria,rjb->iajb', w_ov_a, rho_ov_a)

                w_ov_b = numpy.einsum('ria,r->ria', u_d * rho_ov_a + d_d * rho_ov_b, weight)
                a += lib.einsum('ria,rjb->iajb', w_ov_b, rho_vo_b)
                b += lib.einsum('ria,rjb->iajb', w_ov_b, rho_ov_b)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0 = make_rho(0, ao, mask, xctype)
                vxc, fxc = ni.eval_xc(mf.xc, rho0, 1, deriv=2)[1:3]
                r0, (mx0, my0, mz0) = rho0
                rho0a = (r0 + mz0) * .5
                rho0b = (r0 - mz0) * .5
                uu, ud, dd = vxc[1].T
                u_u, u_d, d_d = fxc[0].T
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T

                aoa, aob = ao
                mo_oa = lib.einsum('xrp,pi->xri', aoa, orbo)
                mo_va = lib.einsum('xrp,pi->xri', aoa, orbv)
                mo_ob = lib.einsum('xrp,pi->xri', aob, orbo)
                mo_vb = lib.einsum('xrp,pi->xri', aob, orbv)
                rho_ov_a = numpy.einsum('xri,ra->xria', mo_oa.conj(), mo_va[0])
                rho_ov_b = numpy.einsum('xri,ra->xria', mo_ob.conj(), mo_vb[0])
                rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', mo_oa[0].conj(), mo_va[1:4])
                rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', mo_ob[0].conj(), mo_vb[1:4])
                rho_vo_a = rho_ov_a.conj()
                rho_vo_b = rho_vo_a.conj()
                # sigma1 ~ \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
                a0a1 = numpy.einsum('xr,xria->ria', rho0a[1:4], rho_ov_a[1:4])
                a0b1 = numpy.einsum('xr,xria->ria', rho0a[1:4], rho_ov_b[1:4])
                b0a1 = numpy.einsum('xr,xria->ria', rho0b[1:4], rho_ov_a[1:4])
                b0b1 = numpy.einsum('xr,xria->ria', rho0b[1:4], rho_ov_b[1:4])

                # aaaa
                w_ov = numpy.empty_like(rho_ov_a)
                w_ov[0]  = numpy.einsum('r,ria->ria', u_u, rho_ov_a[0])
                w_ov[0] += numpy.einsum('r,ria->ria', 2*u_uu, a0a1)
                w_ov[0] += numpy.einsum('r,ria->ria',   u_ud, b0a1)
                f_ov_a  = numpy.einsum('r,ria->ria', 4*uu_uu, a0a1)
                f_ov_b  = numpy.einsum('r,ria->ria', 2*uu_ud, a0a1)
                f_ov_a += numpy.einsum('r,ria->ria', 2*uu_ud, b0a1)
                f_ov_b += numpy.einsum('r,ria->ria',   ud_ud, b0a1)
                f_ov_a += numpy.einsum('r,ria->ria', 2*u_uu, rho_ov_a[0])
                f_ov_b += numpy.einsum('r,ria->ria',   u_ud, rho_ov_a[0])
                w_ov[1:] = numpy.einsum('ria,xr->xria', f_ov_a, rho0a[1:4])
                w_ov[1:]+= numpy.einsum('ria,xr->xria', f_ov_b, rho0b[1:4])
                w_ov[1:]+= numpy.einsum('r,xria->xria', 2*uu, rho_ov_a[1:4])
                w_ov *= weight[:,None,None]
                a += lib.einsum('xria,xrjb->iajb', w_ov, rho_vo_a)
                b += lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a)

                # bbbb
                w_ov = numpy.empty_like(rho_ov_b)
                w_ov[0]  = numpy.einsum('r,ria->ria', d_d, rho_ov_b[0])
                w_ov[0] += numpy.einsum('r,ria->ria', 2*d_dd, b0b1)
                w_ov[0] += numpy.einsum('r,ria->ria',   d_ud, a0b1)
                f_ov_b  = numpy.einsum('r,ria->ria', 4*dd_dd, b0b1)
                f_ov_a  = numpy.einsum('r,ria->ria', 2*ud_dd, b0b1)
                f_ov_b += numpy.einsum('r,ria->ria', 2*ud_dd, a0b1)
                f_ov_a += numpy.einsum('r,ria->ria',   ud_ud, a0b1)
                f_ov_b += numpy.einsum('r,ria->ria', 2*d_dd, rho_ov_b[0])
                f_ov_a += numpy.einsum('r,ria->ria',   d_ud, rho_ov_b[0])
                w_ov[1:] = numpy.einsum('ria,xr->xria', f_ov_a, rho0a[1:4])
                w_ov[1:]+= numpy.einsum('ria,xr->xria', f_ov_b, rho0b[1:4])
                w_ov[1:]+= numpy.einsum('r,xria->xria', 2*dd, rho_ov_b[1:4])
                w_ov *= weight[:,None,None]
                a += lib.einsum('xria,xrjb->iajb', w_ov, rho_vo_b)
                b += lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_b)

                # aabb and bbaa
                w_ov = numpy.empty_like(rho_ov_a)
                w_ov[0]  = numpy.einsum('r,ria->ria', u_d, rho_ov_b[0])
                w_ov[0] += numpy.einsum('r,ria->ria', 2*u_dd, b0b1)
                w_ov[0] += numpy.einsum('r,ria->ria',   u_ud, a0b1)
                f_ov_a  = numpy.einsum('r,ria->ria', 4*uu_dd, b0b1)
                f_ov_b  = numpy.einsum('r,ria->ria', 2*ud_dd, b0b1)
                f_ov_a += numpy.einsum('r,ria->ria', 2*uu_ud, a0b1)
                f_ov_b += numpy.einsum('r,ria->ria',   ud_ud, a0b1)
                f_ov_a += numpy.einsum('r,ria->ria', 2*d_uu, rho_ov_b[0])
                f_ov_b += numpy.einsum('r,ria->ria',   d_ud, rho_ov_b[0])
                w_ov[1:] = numpy.einsum('ria,xr->xria', f_ov_a, rho0a[1:4])
                w_ov[1:]+= numpy.einsum('ria,xr->xria', f_ov_b, rho0b[1:4])
                w_ov[1:]+= numpy.einsum('r,xria->xria', ud, rho_ov_b[1:4])
                w_ov *= weight[:,None,None]
                a_iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_vo_a)
                b_iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov_a)
                a += a_iajb
                a += a_iajb.conj().transpose(2,3,0,1)
                b += b_iajb * 2

        elif xctype == 'HF':
            pass
        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        elif xctype == 'MGGA':
            raise NotImplementedError('meta-GGA')

    else:
        add_hf_(a, b)

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
    def gen_vind(self, mf):
        '''Compute Ax'''
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
    def gen_vind(self, mf):
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
