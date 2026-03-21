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
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

import numpy as np
from pyscf import lib
from pyscf.tdscf import rhf
from pyscf.pbc import scf
from pyscf.pbc.gto.pseudo.ppnl_velgauge import get_gth_pp_nl_velgauge_commutator
from pyscf import __config__

def get_ab(mf):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454
    '''
    cell = mf.cell
    mo_energy = scf.addons.mo_energy_with_exxdiv_none(mf)
    mo = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    kpt = mf.kpt
    nao, nmo = mo.shape
    nocc = np.count_nonzero(mo_occ==2)
    nvir = nmo - nocc
    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]

    e_ia = mo_energy[nocc:] - mo_energy[:nocc,None]
    a = np.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir).astype(mo.dtype)
    b = np.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        eri = mf.with_df.ao2mo([orbo,mo,mo,mo], kpt, compact=False)
        eri = eri.reshape(nocc,nmo,nmo,nmo)
        a += np.einsum('iabj->iajb', eri[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= np.einsum('ijba->iajb', eri[:nocc,:nocc,nocc:,nocc:]) * hyb
        b += np.einsum('iajb->iajb', eri[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= np.einsum('ibja->iajb', eri[:nocc,nocc:,:nocc,nocc:]) * hyb

    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)

        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            with mf.with_df.range_coulomb(omega) as rsh_df:
                eri = rsh_df.ao2mo([orbo,mo,mo,mo], kpt, compact=False)
                eri = eri.reshape(nocc,nmo,nmo,nmo)
                k_fac = alpha - hyb
                a -= np.einsum('ijba->iajb', eri[:nocc,:nocc,nocc:,nocc:]) * k_fac
                b -= np.einsum('ibja->iajb', eri[:nocc,nocc:,:nocc,nocc:]) * k_fac

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo, mo_occ)
        make_rho = ni._gen_rho_evaluator(cell, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpt, None, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[0,0] * weight

                rho_o = lib.einsum('rp,pi->ri', ao, orbo)
                rho_v = lib.einsum('rp,pi->ri', ao, orbv)
                rho_ov = np.einsum('ri,ra->ria', rho_o, rho_v)
                rho_vo = rho_ov.conj()
                w_vo = np.einsum('ria,r->ria', rho_vo, wfxc) * 2
                a += lib.einsum('ria,rjb->iajb', w_vo, rho_ov)
                b += lib.einsum('ria,rjb->iajb', w_vo, rho_vo)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpt, None, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = np.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += np.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                rho_vo = rho_ov.conj()
                w_vo = np.einsum('xyr,xria->yria', wfxc, rho_vo) * 2
                a += lib.einsum('xria,xrjb->iajb', w_vo, rho_ov)
                b += lib.einsum('xria,xrjb->iajb', w_vo, rho_vo)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpt, None, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = np.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += np.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                tau_ov = np.einsum('xri,xra->ria', rho_o[1:4], rho_v[1:4]) * .5
                rho_ov = np.vstack([rho_ov, tau_ov[np.newaxis]])
                rho_vo = rho_ov.conj()
                w_vo = np.einsum('xyr,xria->yria', wfxc, rho_vo) * 2
                a += lib.einsum('xria,xrjb->iajb', w_vo, rho_ov)
                b += lib.einsum('xria,xrjb->iajb', w_vo, rho_vo)
    else:
        add_hf_(a, b)

    return a, b

def transition_velocity_dipole(tdobj, xy=None):
    '''Transition dipole moments in the velocity gauge (imaginary part only)
    '''
    kpt = tdobj._scf.kpt
    ints_p = tdobj.cell.pbc_intor('int1e_ipovlp', comp=3, hermi=0, kpts=kpt)
    if tdobj.cell.pseudo:
        r_vnl_commutator = get_gth_pp_nl_velgauge_commutator(tdobj.cell, q=np.zeros(3), kpts=kpt)
    else:
        r_vnl_commutator = 0.0
    # velocity operator = p - i[r, V_nl]
    # Because int1e_ipovlp is ( nabla \| ) = ( \| -nabla ) = p / i, we have
    # Im [ vel. ] = int1e_ipovlp - [ r, V_nl ].
    # Note that the matrix of [ r_a, V_nl ] (a = 1, 2, 3) is real and anti-Hermitian.
    velocity_operator = ints_p - r_vnl_commutator
    v = tdobj._contract_multipole(velocity_operator, hermi=0, xy=xy)
    return -v

def oscillator_strength(tdobj, e=None, xy=None, gauge='velocity', order=0):
    if e is None: e = tdobj.e

    if gauge == 'length':
        raise NotImplementedError

    elif gauge == 'velocity':
        trans_dip = tdobj.transition_velocity_dipole(xy)
        f = 2./3. * np.einsum('s,sx,sx->s', 1./e, trans_dip, trans_dip.conj()).real

        if order > 0:
            raise NotImplementedError
    else:
        raise ValueError(f'Unknown gauge {gauge}')

    lib.logger.warn(tdobj, 'You requested oscillator strengths for a periodic system.'
                           ' Please note that they are only intended for comparison'
                           ' with molecular calculations.')

    return f

class TDBase(rhf.TDBase):
    _keys = {'cell'}

    def __init__(self, mf, frozen=None):
        rhf.TDBase.__init__(self, mf, frozen)
        self.cell = mf.cell

    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    def nuc_grad_method(self):
        raise NotImplementedError

    get_nto = rhf.TDBase.get_nto
    transition_velocity_dipole     = transition_velocity_dipole
    oscillator_strength            = oscillator_strength
    analyze                        = lib.invalid_method('analyze')
    transition_dipole              = lib.invalid_method('transition_dipole')
    transition_quadrupole          = lib.invalid_method('transition_quadrupole')
    transition_octupole            = lib.invalid_method('transition_octupole')
    transition_velocity_quadrupole = lib.invalid_method('transition_velocity_quadrupole')
    transition_velocity_octupole   = lib.invalid_method('transition_velocity_octupole')
    transition_magnetic_dipole     = lib.invalid_method('transition_magnetic_dipole')
    transition_magnetic_quadrupole = lib.invalid_method('transition_magnetic_quadrupole')
    photoabsorption_cross_section  = lib.invalid_method('photoabsorption_cross_section')


class TDA(TDBase):

    get_init_guess = rhf.TDA.get_init_guess
    kernel = rhf.TDA.kernel
    _gen_vind = rhf.TDA.gen_vind

    def gen_vind(self, mf=None):
        if mf is None: mf = self._scf
        moe = scf.addons.mo_energy_with_exxdiv_none(mf)
        with lib.temporary_env(mf, mo_energy=moe):
            vind, hdiag = self._gen_vind(mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

CIS = TDA


class TDHF(TDBase):

    get_init_guess = rhf.TDHF.get_init_guess
    kernel = rhf.TDHF.kernel
    _gen_vind = rhf.TDHF.gen_vind
    gen_vind = TDA.gen_vind

RPA = TDRHF = TDHF


scf.hf.RHF.TDA = lib.class_as_method(TDA)
scf.hf.RHF.TDHF = lib.class_as_method(TDHF)
scf.rohf.ROHF.TDA = None
scf.rohf.ROHF.TDHF = None
