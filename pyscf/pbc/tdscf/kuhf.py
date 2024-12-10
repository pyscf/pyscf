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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.tdscf import uhf
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from pyscf.pbc import scf
from pyscf.pbc.tdscf.krhf import KTDBase, _get_e_ia
from pyscf.pbc.lib.kpts_helper import is_gamma_point, get_kconserv_ria, conj_mapping
from pyscf.pbc.scf import _response_functions  # noqa
from pyscf import __config__

REAL_EIG_THRESHOLD = getattr(__config__, 'pbc_tdscf_uhf_TDDFT_pick_eig_threshold', 1e-3)

def get_ab(mf, kshift=0):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454

    Kwargs:
        kshift : integer
            The index of the k-point that represents the transition between
            k-points in the excitation coefficients.
    '''
    cell = mf.cell
    mo_energy = scf.addons.mo_energy_with_exxdiv_none(mf)
    mo_a, mo_b = mo = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    kpts = mf.kpts
    nkpts, nao, nmo = mo_a.shape
    noccs = np.count_nonzero(mo_occ!=0, axis=2)
    nocc = noccs[0,0]
    nvir = nmo - nocc
    assert np.all(noccs == nocc)
    nocc_a = nocc_b = nocc
    nvir_a = nvir_b = nvir
    orbo_a = mo_a[:,:,:nocc]
    orbo_b = mo_b[:,:,:nocc]
    orbv_a = mo_a[:,:,nocc:]
    orbv_b = mo_b[:,:,nocc:]

    kconserv = get_kconserv_ria(cell, kpts)[kshift]
    e_ia_a = np.asarray(_get_e_ia(mo_energy[0], mo_occ[0], kconserv)).astype(mo.dtype)
    e_ia_b = np.asarray(_get_e_ia(mo_energy[1], mo_occ[1], kconserv)).astype(mo.dtype)
    a_aa = np.diag(e_ia_a.ravel()).reshape(nkpts,nocc_a,nvir_a,nkpts,nocc_a,nvir_a)
    a_bb = np.diag(e_ia_b.ravel()).reshape(nkpts,nocc_b,nvir_b,nkpts,nocc_b,nvir_b)
    a_ab = np.zeros((nkpts,nocc_a,nvir_a,nkpts,nocc_b,nvir_b), dtype=a_aa.dtype)
    b_aa = np.zeros_like(a_aa)
    b_bb = np.zeros_like(a_bb)
    b_ab = np.zeros_like(a_ab)
    a = (a_aa, a_ab, a_bb)
    b = (b_aa, b_ab, b_bb)
    weight = 1./nkpts

    def add_hf_(a, b, hyb=1):
        eri_aa = mf.with_df.ao2mo_7d([mo_a,orbo_a,mo_a,mo_a], kpts)
        eri_ab = mf.with_df.ao2mo_7d([mo_a,orbo_a,mo_b,mo_b], kpts)
        eri_bb = mf.with_df.ao2mo_7d([mo_b,orbo_b,mo_b,mo_b], kpts)
        eri_aa *= weight
        eri_ab *= weight
        eri_bb *= weight
        eri_aa.reshape(nkpts,nkpts,nkpts,nmo,nocc_a,nmo,nmo)
        eri_ab.reshape(nkpts,nkpts,nkpts,nmo,nocc_a,nmo,nmo)
        eri_bb.reshape(nkpts,nkpts,nkpts,nmo,nocc_b,nmo,nmo)
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b

        for ki, ka in enumerate(kconserv):
            for kj, kb in enumerate(kconserv):
                a_aa[ki,:,:,kj] += np.einsum('aijb->iajb', eri_aa[ka,ki,kj,nocc_a:,:,:nocc_a,nocc_a:])
                a_aa[ki,:,:,kj] -= np.einsum('jiab->iajb', eri_aa[kj,ki,ka,:nocc_a,:,nocc_a:,nocc_a:]) * hyb
                a_bb[ki,:,:,kj] += np.einsum('aijb->iajb', eri_bb[ka,ki,kj,nocc_b:,:,:nocc_b,nocc_b:])
                a_bb[ki,:,:,kj] -= np.einsum('jiab->iajb', eri_bb[kj,ki,ka,:nocc_b,:,nocc_b:,nocc_b:]) * hyb
                a_ab[ki,:,:,kj] += np.einsum('aijb->iajb', eri_ab[ka,ki,kj,nocc_a:,:,:nocc_b,nocc_b:])

            for kb, kj in enumerate(kconserv):
                b_aa[ki,:,:,kj] += np.einsum('aibj->iajb', eri_aa[ka,ki,kb,nocc_a:,:,nocc_a:,:nocc_a])
                b_aa[ki,:,:,kj] -= np.einsum('ajbi->iajb', eri_aa[ka,kj,kb,nocc_a:,:,nocc_a:,:nocc_a]) * hyb
                b_bb[ki,:,:,kj] += np.einsum('aibj->iajb', eri_bb[ka,ki,kb,nocc_b:,:,nocc_b:,:nocc_b])
                b_bb[ki,:,:,kj] -= np.einsum('ajbi->iajb', eri_bb[ka,kj,kb,nocc_b:,:,nocc_b:,:nocc_b]) * hyb
                b_ab[ki,:,:,kj] += np.einsum('aibj->iajb', eri_ab[ka,ki,kb,nocc_a:,:,nocc_b:,:nocc_b])

    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)

        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            raise NotImplementedError

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo, mo_occ)
        make_rho = ni._gen_rho_evaluator(cell, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)
        cmap = conj_mapping(cell, kpts)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpts, None, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[0,0] * weight

                rho_o_a = lib.einsum('krp,kpi->kri', ao, orbo_a)
                rho_v_a = lib.einsum('krp,kpi->kri', ao, orbv_a)
                rho_o_b = lib.einsum('krp,kpi->kri', ao, orbo_b)
                rho_v_b = lib.einsum('krp,kpi->kri', ao, orbv_b)
                rho_ov_a = np.einsum('kri,kra->kria', rho_o_a, rho_v_a)
                rho_ov_b = np.einsum('kri,kra->kria', rho_o_b, rho_v_b)
                rho_vo_a = rho_ov_a.conj()[cmap]
                rho_vo_b = rho_ov_b.conj()[cmap]
                w_vo_aa = np.einsum('kria,r->kria', rho_vo_a, wfxc[0,0]) * (1/nkpts)
                w_vo_ab = np.einsum('kria,r->kria', rho_vo_a, wfxc[0,1]) * (1/nkpts)
                w_vo_bb = np.einsum('kria,r->kria', rho_vo_b, wfxc[1,1]) * (1/nkpts)

                a_aa += lib.einsum('kria,lrjb->kialjb', w_vo_aa, rho_ov_a)
                b_aa += lib.einsum('kria,lrjb->kialjb', w_vo_aa, rho_vo_a)

                a_ab += lib.einsum('kria,lrjb->kialjb', w_vo_ab, rho_ov_b)
                b_ab += lib.einsum('kria,lrjb->kialjb', w_vo_ab, rho_vo_b)

                a_bb += lib.einsum('kria,lrjb->kialjb', w_vo_bb, rho_ov_b)
                b_bb += lib.einsum('kria,lrjb->kialjb', w_vo_bb, rho_vo_b)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpts, None, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight

                rho_o_a = lib.einsum('kxrp,kpi->kxri', ao, orbo_a)
                rho_v_a = lib.einsum('kxrp,kpi->kxri', ao, orbv_a)
                rho_o_b = lib.einsum('kxrp,kpi->kxri', ao, orbo_b)
                rho_v_b = lib.einsum('kxrp,kpi->kxri', ao, orbv_b)
                rho_ov_a = np.einsum('kxri,kra->kxria', rho_o_a, rho_v_a[:,0])
                rho_ov_b = np.einsum('kxri,kra->kxria', rho_o_b, rho_v_b[:,0])
                rho_ov_a[:,1:4] += np.einsum('kri,kxra->kxria', rho_o_a[:,0], rho_v_a[:,1:4])
                rho_ov_b[:,1:4] += np.einsum('kri,kxra->kxria', rho_o_b[:,0], rho_v_b[:,1:4])
                rho_vo_a = rho_ov_a.conj()[cmap]
                rho_vo_b = rho_ov_b.conj()[cmap]
                w_vo_aa = np.einsum('xyr,kxria->kyria', wfxc[0,:,0], rho_vo_a) * (1/nkpts)
                w_vo_ab = np.einsum('xyr,kxria->kyria', wfxc[0,:,1], rho_vo_a) * (1/nkpts)
                w_vo_bb = np.einsum('xyr,kxria->kyria', wfxc[1,:,1], rho_vo_b) * (1/nkpts)

                a_aa += lib.einsum('kxria,lxrjb->kialjb', w_vo_aa, rho_ov_a)
                b_aa += lib.einsum('kxria,lxrjb->kialjb', w_vo_aa, rho_vo_a)

                a_ab += lib.einsum('kxria,lxrjb->kialjb', w_vo_ab, rho_ov_b)
                b_ab += lib.einsum('kxria,lxrjb->kialjb', w_vo_ab, rho_vo_b)

                a_bb += lib.einsum('kxria,lxrjb->kialjb', w_vo_bb, rho_ov_b)
                b_bb += lib.einsum('kxria,lxrjb->kialjb', w_vo_bb, rho_vo_b)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpts, None, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho = (rho0a, rho0b)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight

                rho_o_a = lib.einsum('kxrp,kpi->kxri', ao, orbo_a)
                rho_o_b = lib.einsum('kxrp,kpi->kxri', ao, orbo_b)
                rho_v_a = lib.einsum('kxrp,kpi->kxri', ao, orbv_a)
                rho_v_b = lib.einsum('kxrp,kpi->kxri', ao, orbv_b)
                rho_ov_a = np.einsum('kxri,kra->kxria', rho_o_a, rho_v_a[:,0])
                rho_ov_b = np.einsum('kxri,kra->kxria', rho_o_b, rho_v_b[:,0])
                rho_ov_a[:,1:4] += np.einsum('ri,xra->xria', rho_o_a[:,0], rho_v_a[:,1:4])
                rho_ov_b[:,1:4] += np.einsum('ri,xra->xria', rho_o_b[:,0], rho_v_b[:,1:4])
                tau_ov_a = np.einsum('kxri,kxra->kria', rho_o_a[:,1:4], rho_v_a[:,1:4]) * .5
                tau_ov_b = np.einsum('kxri,kxra->kria', rho_o_b[:,1:4], rho_v_b[:,1:4]) * .5
                rho_ov_a = np.vstack([rho_ov_a, tau_ov_a[:,np.newaxis]])
                rho_ov_b = np.vstack([rho_ov_b, tau_ov_b[:,np.newaxis]])
                rho_vo_a = rho_ov_a.conj()[cmap]
                rho_vo_b = rho_ov_b.conj()[cmap]
                w_vo_aa = np.einsum('xyr,kxria->kyria', wfxc[0,:,0], rho_vo_a) * (1/nkpts)
                w_vo_ab = np.einsum('xyr,kxria->kyria', wfxc[0,:,1], rho_vo_a) * (1/nkpts)
                w_vo_bb = np.einsum('xyr,kxria->kyria', wfxc[1,:,1], rho_vo_b) * (1/nkpts)

                a_aa += lib.einsum('kxria,lxrjb->kilajb', w_vo_aa, rho_ov_a)
                b_aa += lib.einsum('kxria,lxrjb->kilajb', w_vo_aa, rho_vo_a)

                a_ab += lib.einsum('kxria,lxrjb->kilajb', w_vo_ab, rho_ov_b)
                b_ab += lib.einsum('kxria,lxrjb->kilajb', w_vo_ab, rho_vo_b)

                a_bb += lib.einsum('kxria,lxrjb->kilajb', w_vo_bb, rho_ov_b)
                b_bb += lib.einsum('kxria,lxrjb->kilajb', w_vo_bb, rho_vo_b)
    else:
        add_hf_(a, b)

    return a, b

class TDA(KTDBase):

    def get_ab(self, mf=None, kshift=0):
        if mf is None: mf = self._scf
        return get_ab(mf, kshift)

    def gen_vind(self, mf, kshift=0):
        '''Compute Ax

        Kwargs:
            kshift : integer
                The index of the k-point that represents the transition between
                k-points in the excitation coefficients.
        '''
        kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ[0])
        nao, nmo = mo_coeff[0][0].shape
        occidxa = [mo_occ[0][k]> 0 for k in range(nkpts)]
        occidxb = [mo_occ[1][k]> 0 for k in range(nkpts)]
        viridxa = [mo_occ[0][k]==0 for k in range(nkpts)]
        viridxb = [mo_occ[1][k]==0 for k in range(nkpts)]
        orboa = [mo_coeff[0][k][:,occidxa[k]] for k in range(nkpts)]
        orbob = [mo_coeff[1][k][:,occidxb[k]] for k in range(nkpts)]
        orbva = [mo_coeff[0][k][:,viridxa[k]] for k in range(nkpts)]
        orbvb = [mo_coeff[1][k][:,viridxb[k]] for k in range(nkpts)]
        dtype = np.result_type(*mo_coeff[0])

        moe = scf.addons.mo_energy_with_exxdiv_none(mf)
        e_ia_a = _get_e_ia(moe[0], mo_occ[0], kconserv)
        e_ia_b = _get_e_ia(moe[1], mo_occ[1], kconserv)
        hdiag = np.hstack([x.ravel() for x in (e_ia_a + e_ia_b)])

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = mf.gen_response(hermi=0, max_memory=max_memory)

        def vind(zs):
            nz = len(zs)
            zs = [_unpack(z, mo_occ, kconserv) for z in zs]
            dms = np.empty((2,nz,nkpts,nao,nao), dtype=dtype)
            for i in range(nz):
                dm1a, dm1b = zs[i]
                for k, kp in enumerate(kconserv):
                    dms[0,i,kp] = lib.einsum('ov,pv,qo->pq', dm1a[k], orbva[kp], orboa[k].conj())
                    dms[1,i,kp] = lib.einsum('ov,pv,qo->pq', dm1b[k], orbvb[kp], orbob[k].conj())

            with lib.temporary_env(mf, exxdiv=None):
                v1ao = vresp(dms, kshift)
                v1ao = v1ao.reshape(2,nz,nkpts,nao,nao)

            v1s = []
            for i in range(nz):
                dm1a, dm1b = zs[i]
                v1as = [None] * nkpts
                v1bs = [None] * nkpts
                for k, kp in enumerate(kconserv):
                    v1a = lib.einsum('pq,qo,pv->ov', v1ao[0,i,kp], orboa[k], orbva[kp].conj())
                    v1b = lib.einsum('pq,qo,pv->ov', v1ao[1,i,kp], orbob[k], orbvb[kp].conj())
                    v1a += e_ia_a[k] * dm1a[k]
                    v1b += e_ia_b[k] * dm1b[k]
                    v1as[k] = v1a.ravel()
                    v1bs[k] = v1b.ravel()
                v1s.append( np.concatenate(v1as + v1bs) )
            return np.stack(v1s)

        return vind, hdiag

    def init_guess(self, mf, kshift, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]
        e_ia_a = _get_e_ia(mo_energy[0], mo_occ[0], kconserv)
        e_ia_b = _get_e_ia(mo_energy[1], mo_occ[1], kconserv)
        e_ia = np.hstack([x.ravel() for x in (e_ia_a + e_ia_b)])

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None):
        '''TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()

        log = logger.new_logger(self)

        mf = self._scf
        mo_occ = mf.mo_occ

        def pickeig(w, v, nroots, envs):
            idx = np.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        log = logger.Logger(self.stdout, self.verbose)

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, e, x1 = lr_eigh(
                vind, x0k, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, pick=pickeig, max_cycle=self.max_cycle,
                max_memory=self.max_memory, verbose=log)
            self.converged.append( converged )
            self.e.append( e )
            self.xy.append( [(_unpack(xi, mo_occ, kconserv),  # (X_alpha, X_beta)
                        (0, 0))  # (Y_alpha, Y_beta)
                       for xi in x1] )
        #TODO: analyze CIS wfn point group symmetry
        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
CIS = KTDA = TDA


class TDHF(KTDBase):

    get_ab = TDA.get_ab

    def gen_vind(self, mf, kshift=0):
        assert kshift == 0

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ[0])
        nao, nmo = mo_coeff[0][0].shape
        occidxa = [mo_occ[0][k]> 0 for k in range(nkpts)]
        occidxb = [mo_occ[1][k]> 0 for k in range(nkpts)]
        viridxa = [mo_occ[0][k]==0 for k in range(nkpts)]
        viridxb = [mo_occ[1][k]==0 for k in range(nkpts)]
        orboa = [mo_coeff[0][k][:,occidxa[k]] for k in range(nkpts)]
        orbob = [mo_coeff[1][k][:,occidxb[k]] for k in range(nkpts)]
        orbva = [mo_coeff[0][k][:,viridxa[k]] for k in range(nkpts)]
        orbvb = [mo_coeff[1][k][:,viridxb[k]] for k in range(nkpts)]
        dtype = np.result_type(*mo_coeff[0])

        kconserv = np.arange(nkpts)
        moe = scf.addons.mo_energy_with_exxdiv_none(mf)
        e_ia_a = _get_e_ia(moe[0], mo_occ[0], kconserv)
        e_ia_b = _get_e_ia(moe[1], mo_occ[1], kconserv)
        hdiag = np.hstack([x.ravel() for x in (e_ia_a + e_ia_b)])
        tot_x = hdiag.size
        hdiag = np.hstack((hdiag, -hdiag))

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = mf.gen_response(hermi=0, max_memory=max_memory)

        def vind(xys):
            nz = len(xys)
            x1s = [_unpack(x[:tot_x], mo_occ, kconserv) for x in xys]
            y1s = [_unpack(x[tot_x:], mo_occ, kconserv) for x in xys]
            dms = np.empty((2,nz,nkpts,nao,nao), dtype=dtype)
            for i in range(nz):
                xa, xb = x1s[i]
                ya, yb = y1s[i]
                for k in range(nkpts):
                    dms[0,i,k]  = lib.einsum('ov,pv,qo->pq', xa[k], orbva[k], orboa[k].conj())
                    dms[1,i,k]  = lib.einsum('ov,pv,qo->pq', xb[k], orbvb[k], orbob[k].conj())
                    dms[0,i,k] += lib.einsum('ov,qv,po->pq', ya[k], orbva[k].conj(), orboa[k])
                    dms[1,i,k] += lib.einsum('ov,qv,po->pq', yb[k], orbvb[k].conj(), orbob[k])

            with lib.temporary_env(mf, exxdiv=None):
                v1ao = vresp(dms, kshift)
                v1ao = v1ao.reshape(2,nz,nkpts,nao,nao)

            v1s = []
            for i in range(nz):
                xa, xb = x1s[i]
                ya, yb = y1s[i]
                v1xsa = [0] * nkpts
                v1xsb = [0] * nkpts
                v1ysa = [0] * nkpts
                v1ysb = [0] * nkpts
                for k in range(nkpts):
                    v1xa = lib.einsum('pq,qo,pv->ov', v1ao[0,i,k], orboa[k], orbva[k].conj())
                    v1xb = lib.einsum('pq,qo,pv->ov', v1ao[1,i,k], orbob[k], orbvb[k].conj())
                    v1ya = lib.einsum('pq,po,qv->ov', v1ao[0,i,k], orboa[k].conj(), orbva[k])
                    v1yb = lib.einsum('pq,po,qv->ov', v1ao[1,i,k], orbob[k].conj(), orbvb[k])
                    v1xa += e_ia_a[k] * xa[k]
                    v1xb += e_ia_b[k] * xb[k]
                    v1ya += e_ia_a[k] * ya[k]
                    v1yb += e_ia_b[k] * yb[k]
                    v1xsa[k] += v1xa.ravel()
                    v1xsb[k] += v1xb.ravel()
                    v1ysa[k] -= v1ya.ravel()
                    v1ysb[k] -= v1yb.ravel()
                v1s.append( np.concatenate(v1xsa + v1xsb + v1ysa + v1ysb) )
            return np.stack(v1s)

        return vind, hdiag

    def init_guess(self, mf, kshift, nstates=None, wfnsym=None):
        x0 = TDA.init_guess(self, mf, kshift, nstates)
        y0 = np.zeros_like(x0)
        return np.hstack([x0, y0])

    get_precond = uhf.TDHF.get_precond

    def kernel(self, x0=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()

        log = logger.new_logger(self)

        mf = self._scf
        mo_occ = mf.mo_occ

        real_system = (is_gamma_point(self._scf.kpts) and
                       self._scf.mo_coeff[0][0].dtype == np.double)

        if any(k != 0 for k in self.kshift_lst):
            raise RuntimeError('kshift != 0 for TDHF')

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, w, x1 = lr_eig(
                vind, x0k, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, pick=pickeig, max_cycle=self.max_cycle,
                max_memory=self.max_memory, verbose=log)
            self.converged.append( converged )

            e = []
            xy = []
            for i, z in enumerate(x1):
                xs, ys = z.reshape(2,-1)
                norm = lib.norm(xs)**2 - lib.norm(ys)**2
                if norm < 0:
                    log.warn('TDDFT amplitudes |X| smaller than |Y|')
                norm = abs(norm)**-.5
                xs *= norm
                ys *= norm
                e.append(w[i])
                xy.append((_unpack(xs, mo_occ, kconserv), _unpack(ys, mo_occ, kconserv)))
            self.e.append( np.array(e) )
            self.xy.append( xy )

        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
RPA = KTDHF = TDHF

def _unpack(vo, mo_occ, kconserv):
    za = []
    zb = []
    p1 = 0
    no_a_kpts = [np.count_nonzero(occ) for occ in mo_occ[0]]
    no_b_kpts = [np.count_nonzero(occ) for occ in mo_occ[1]]
    for k, occ in enumerate(mo_occ[0]):
        no = no_a_kpts[k]
        nv = occ.size - no_a_kpts[kconserv[k]]
        p0, p1 = p1, p1 + no * nv
        za.append(vo[p0:p1].reshape(no,nv))

    for k, occ in enumerate(mo_occ[1]):
        no = no_b_kpts[k]
        nv = occ.size - no_b_kpts[kconserv[k]]
        p0, p1 = p1, p1 + no * nv
        zb.append(vo[p0:p1].reshape(no,nv))
    return za, zb


scf.kuhf.KUHF.TDA  = lib.class_as_method(KTDA)
scf.kuhf.KUHF.TDHF = lib.class_as_method(KTDHF)
