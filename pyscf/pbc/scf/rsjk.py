#!/usr/bin/env python
# Copyright 2020-2021 The PySCF Developers. All Rights Reserved.
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Range separation JK builder

Ref:
    Q. Sun, arXiv:2012.07929
'''

import copy
import ctypes
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df import aft, rsdf_builder
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.df_jk import (zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0,
                                _format_dms, _format_kpts_band, _format_jks)
from pyscf.pbc.df.incore import _get_cache_size
from pyscf.pbc.lib.kpts_helper import (is_zero, unique_with_wrap_around,
                                       group_by_conj_pairs)
from pyscf import __config__

# Threshold of steep bases and local bases
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.0)
# kecut=10 can rougly converge GTO with alpha=0.5
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)

libpbc = lib.load_library('libpbc')

class RangeSeparatedJKBuilder(object):
    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = None
        self.kpts = np.reshape(kpts, (-1, 3))
        self.purify = True

        if cell.omega != 0:
            # Initialize omega to cell.omega for HF exchange of short range
            # int2e in RSH functionals
            self.omega = abs(cell.omega)
        else:
            self.omega = None
        self.rs_cell = None
        # Born-von Karman supercell
        self.bvk_kmesh = None
        self.supmol_sr = None
        self.supmol_ft = None
        self.supmol_d = None
        # For shells in bvkcell, use overlap mask to remove d-d block
        self.ovlp_mask = None
        # which shells are located in the first primitive cell
        self.cell0_basis_mask = None
        self.ke_cutoff = None
        self.vhfopt = None
        # Use fully uncontracted basis for jk_sr part
        self.fully_uncontracted = False

        self._keys = set(self.__dict__.keys())

    def has_long_range(self):
        '''Whether to add the long-range part computed with AFT/FFT integrals'''
        return self.omega is None or abs(self.cell.omega) < self.omega

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'omega = %s', self.omega)
        logger.info(self, 'purify = %s', self.purify)
        logger.info(self, 'bvk_kmesh = %s', self.bvk_kmesh)
        logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
        logger.info(self, 'fully_uncontracted = %s', self.fully_uncontracted)
        logger.info(self, 'has_long_range = %s', self.has_long_range())
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.rs_cell = None
        self.supmol_sr = None
        self.supmol_ft = None
        return self

    def build(self, omega=None, direct_scf_tol=None, intor='int2e'):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        if omega is not None:
            self.omega = omega

        if self.omega is None:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            self.ke_cutoff = estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), self.ke_cutoff)
            if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
                self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]

        log.info('omega = %.15g  ke_cutoff = %s  mesh = %s',
                 self.omega, self.ke_cutoff, self.mesh)

        if direct_scf_tol is None:
            cell_exp = np.hstack(cell.bas_exps()).min()
            theta = 1./(1./cell_exp + self.omega**-2)
            lattice_sum_factor = 2*np.pi*cell.rcut / theta
            direct_scf_tol = cell.precision / lattice_sum_factor
        log.debug('Set direct_scf_tol %g', direct_scf_tol)

        rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)
        self.rs_cell = rs_cell

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        # FIXME: tests the extra requirements on precision is caused by the
        # triple lattice sum
        precision = cell.precision * 1e-1
        rcut_sr = estimate_rcut(rs_cell, self.omega, precision)

        if self.fully_uncontracted:
            log.debug('make supmol from fully uncontracted cell basis')
            pcell, contr_coeff = rs_cell.decontract_basis(to_cart=True)
            self.contr_coeff = scipy.linalg.block_diag(*contr_coeff)
            supmol_sr = ft_ao.ExtendedMole.from_cell(
                pcell, kmesh, rcut_sr.max(), log)
        else:
            log.debug('make supmol from partially uncontracted cell basis')
            supmol_sr = ft_ao.ExtendedMole.from_cell(
                rs_cell, kmesh, rcut_sr.max(), log)
        supmol_sr.omega = -self.omega
        self.supmol_sr = supmol_sr.strip_basis(rcut_sr)
        log.debug('supmol nbas = %d cGTO = %d pGTO = %d',
                  supmol_sr.nbas, supmol_sr.nao, supmol_sr.npgto_nr())

        if self.has_long_range():
            rcut = rsdf_builder.estimate_ft_rcut(rs_cell, exclude_dd_block=True)
            supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(
                rs_cell, kmesh, rcut.max(), log)
            supmol_ft.exclude_dd_block = True
            self.supmol_ft = supmol_ft.strip_basis(rcut)
            log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                      supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())

        log.timer_debug1('initializing supmol', *cpu0)

        # Intialize vhfopt
        with supmol_sr.with_integral_screen(direct_scf_tol**2):
            vhfopt = _vhf.VHFOpt(supmol_sr, cell._add_suffix(intor),
                                 qcondname=libpbc.PBCVHFsetnr_direct_scf)
        self.vhfopt = vhfopt
        vhfopt.direct_scf_tol = direct_scf_tol
        log.timer('initializing vhfopt', *cpu0)

        # Remove the smooth-smooth basis block.
        # Modify the contents of vhfopt.q_cond inplace
        q_cond = self.get_q_cond()
        smooth_idx = supmol_sr.bas_type_to_indices(ft_ao.SMOOTH_BASIS)
        q_cond[smooth_idx[:,None], smooth_idx] = 1e-200

        sh_loc = supmol_sr.sh_loc
        bvk_q_cond = lib.condense('NP_absmax', q_cond, sh_loc, sh_loc)
        self.ovlp_mask = (bvk_q_cond > direct_scf_tol).astype(np.int8)
        return self

    def get_q_cond(self):
        supmol = self.supmol_sr
        q_cond = self.vhfopt.get_q_cond((supmol.nbas, supmol.nbas))
        return q_cond

    def _get_jk_sr(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
                   with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

        cpu0 = logger.process_clock(), logger.perf_counter()
        if self.supmol_sr is None:
            self.build()

        comp = 1
        nkpts = kpts.shape[0]
        vhfopt = self.vhfopt
        supmol = self.supmol_sr
        cell = self.cell
        nao = cell.nao
        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        nbands = bvk_ncells

        # A map to assign each basis of supmol._bas the index in
        # [bvk_cell-id, bas-id, image-id]
        bas_map = np.where(supmol.bas_mask.ravel())[0].astype(np.int32)

        if dm_kpts.ndim != 4:
            dm = dm_kpts.reshape(-1, nkpts, nao, nao)
        else:
            dm = dm_kpts
        n_dm = dm.shape[0]

        if self.fully_uncontracted:
            # parameters for decontracted basis are different
            c = self.contr_coeff
            dm = lib.einsum('nkij,pi,qj->nkpq', dm, c, c)
            decontracted_cell = supmol.rs_cell
            nbasp = decontracted_cell.nbas
            cell0_ao_loc = decontracted_cell.ao_loc
        else:
            nbasp = cell.nbas  # The number of shells in the primitive cell
            cell0_ao_loc = cell.ao_loc
        nao = dm.shape[-1]

        weight = 1. / nkpts
        expRk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts.T))
        # Utilized symmetry sc_dm[R,S] = sc_dm[S-R] = sc_dm[(S-R)%N]
        #:phase = expRk / nkpts**.5
        #:sc_dm = lib.einsum('Rk,nkuv,Sk->nRuSv', phase, sc_dm, phase.conj())
        sc_dm = lib.einsum('k,Sk,nkuv->nSuv', expRk[0]*weight, expRk.conj(), dm)
        dm_translation = k2gamma.double_translation_indices(self.bvk_kmesh).astype(np.int32)
        dm_imag_max = abs(sc_dm.imag).max()
        is_complex_dm = dm_imag_max > 1e-6
        if is_complex_dm:
            if dm_imag_max < 1e-2:
                logger.warn(self, 'DM in (BvK) cell has small imaginary part.  '
                            'It may be a signal of symmetry broken in k-point symmetry')
            sc_dm = np.vstack([sc_dm.real, sc_dm.imag])
        else:
            sc_dm = sc_dm.real
        sc_dm = np.asarray(sc_dm.reshape(-1, bvk_ncells, nao, nao), order='C')
        n_sc_dm = sc_dm.shape[0]

        # * sparse_ao_loc has dimension (Nk,nbas), corresponding to the
        # bvkcell with all basis
        sparse_ao_loc = nao * np.arange(bvk_ncells)[:,None] + cell0_ao_loc[:-1]
        sparse_ao_loc = np.append(sparse_ao_loc.ravel(), nao * bvk_ncells)
        dm_cond = [lib.condense('NP_absmax', d, sparse_ao_loc, cell0_ao_loc)
                   for d in sc_dm]
        dm_cond = np.asarray(np.max(dm_cond, axis=0), order='C')
        libpbc.CVHFset_dm_cond(vhfopt._this,
                               dm_cond.ctypes.data_as(ctypes.c_void_p), dm_cond.size)
        dm_cond = None

        bvk_nbas = nbasp * bvk_ncells
        shls_slice = (0, nbasp, 0, bvk_nbas, 0, bvk_nbas, 0, bvk_nbas)

        cache_size = _get_cache_size(cell, 'int2e_sph')
        cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
        cache_size += cell0_dims.max()**4 * comp * 2

        if hermi:
            fdot_suffix = 's2kl'
        else:
            fdot_suffix = 's1'
        if with_j and with_k:
            fdot = 'PBCVHF_contract_jk_' + fdot_suffix
            vs = np.zeros((2, n_sc_dm, nao, nbands, nao))
        elif with_j:
            fdot = 'PBCVHF_contract_j_' + fdot_suffix
            vs = np.zeros((1, n_sc_dm, nao, nbands, nao))
        else:  # with_k
            fdot = 'PBCVHF_contract_k_' + fdot_suffix
            vs = np.zeros((1, n_sc_dm, nao, nbands, nao))

        if supmol.cart:
            intor = 'PBCint2e_cart'
        else:
            intor = 'PBCint2e_sph'

        drv = libpbc.PBCVHF_direct_drv
        drv(getattr(libpbc, fdot), getattr(libpbc, intor),
            vs.ctypes.data_as(ctypes.c_void_p),
            sc_dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(vs.size), ctypes.c_int(n_dm),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs),
            ctypes.c_int(nkpts), ctypes.c_int(nbands),
            ctypes.c_int(nbasp), ctypes.c_int(comp),
            supmol.sh_loc.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*8)(*shls_slice),
            dm_translation.ctypes.data_as(ctypes.c_void_p),
            self.ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            bas_map.ctypes.data_as(ctypes.c_void_p),
            vhfopt._cintopt, vhfopt._this, ctypes.c_int(cache_size),
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))

        if is_complex_dm:
            vs = vs[:,:n_dm] + vs[:,n_dm:] * 1j

        if self.fully_uncontracted:
            c = self.contr_coeff
            vs = lib.einsum('snpkq,pi,qj->snikj', vs, c, c)

        if kpts_band is None:
            vs = lib.einsum('snpRq,Rk->snkpq', vs, expRk)
        else:
            logger.warn(self, 'Approximate J/K matrices at kpts_band '
                        'with the bvk-cell dervied from kpts')
            kpts_band = np.reshape(kpts_band, (-1, 3))
            vs = lib.einsum('snpRq,Rk->snkpq', vs,
                            np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts_band.T)))

        logger.timer_debug1(self, 'short range part vj and vk', *cpu0)
        return vs

    def get_jk(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            if omega > 0:  # Long-range part only, call AFTDF
                cell = self.cell
                dfobj = aft.AFTDF(cell, self.kpts)
                ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
                dfobj.mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)
                return dfobj.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                    with_j, with_k, omega, exxdiv)
            elif omega < 0:  # Short-range part only
                if self.omega is not None and self.omega != omega:
                    raise RuntimeError(f'omega = {omega}, self.omega = {self.omega}')
                raise NotImplementedError

        # Does not support to specify arbitrary kpts
        if kpts is not None and abs(kpts-self.kpts).max() > 1e-7:
            raise RuntimeError('kpts error. kpts cannot be modified in RSJK')
        kpts = self.kpts

        vs = self._get_jk_sr(dm_kpts, hermi, kpts, kpts_band,
                             with_j, with_k, omega, exxdiv)
        if with_j and with_k:
            vj, vk = vs
        elif with_j:
            vj, vk = vs[0], None
        else:
            vj, vk = None, vs[0]

        if self.purify and kpts_band is None:
            phase = np.exp(1j*np.dot(self.supmol_sr.bvkmesh_Ls, kpts.T))
            phase /= np.sqrt(len(kpts))
        else:
            phase = None

        if with_j:
            if self.has_long_range():
                vj += self._get_lr_j_kpts(dm_kpts, hermi, kpts, kpts_band)
            if hermi:
                vj = (vj + vj.conj().transpose(0,1,3,2)) * .5
            if self.purify and kpts_band is None:
                vj = _purify(vj, phase)
            vj = _format_jks(vj, dm_kpts, kpts_band, kpts)
            if is_zero(kpts) and dm_kpts.dtype == np.double:
                vj = vj.real.copy()

        if with_k:
            # The AFT-FFT mixed implementation may use a large amount of memory
            # vk += self._get_lr_k_kpts1(dm_kpts, hermi, kpts, kpts_band, exxdiv)
            if self.has_long_range():
                vk += self._get_lr_k_kpts(dm_kpts, hermi, kpts, kpts_band, exxdiv)
            if hermi:
                vk = (vk + vk.conj().transpose(0,1,3,2)) * .5
            if self.purify and kpts_band is None:
                vk = _purify(vk, phase)
            vk = _format_jks(vk, dm_kpts, kpts_band, kpts)
            if is_zero(kpts) and dm_kpts.dtype == np.double:
                vk = vk.real.copy()

        return vj, vk

    weighted_coulG = aft.weighted_coulG

    def weighted_coulG_LR(self, kpt=np.zeros(3), exx=False, mesh=None):
        # The long range part Coulomb kernel has to be computed as the
        # difference between coulG(cell.omega) - coulG(self.omega). It allows this
        # module to handle the SR- and regular integrals in the same framework
        return (self.weighted_coulG(kpt, exx, mesh) -
                self.weighted_coulG_SR(kpt, exx, mesh))

    def weighted_coulG_SR(self, kpt=np.zeros(3), exx=False, mesh=None):
        return self.weighted_coulG(kpt, False, mesh, -self.omega)


    def _get_lr_j_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        '''
        Long-range part of J matrix

        C ~ compact basis, D ~ diffused basis

        Compute J matrix with coulG_LR:
        (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)

        Compute J matrix with full coulG:
        (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
        '''
        if kpts_band is not None:
            return self._get_lr_j_for_bands(dm_kpts, hermi, kpts, kpts_band)

        if len(kpts) == 1 and not is_zero(kpts):
            raise NotImplementedError('Single k-point get-j')

        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        cell_d = rs_cell.smooth_basis_cell()
        kpts = np.asarray(kpts.reshape(-1, 3), order='C')
        dms = _format_dms(dm_kpts, kpts)
        n_dm, nkpts, nao = dms.shape[:3]
        naod = cell_d.nao
        if naod > 0:
            mesh = cell_d.mesh
        else:
            mesh = self.mesh
        ngrids = np.prod(mesh)

        vj_kpts = np.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)

        # TODO: aosym == 's2'
        aosym = 's1'
        ft_kern = self.supmol_ft.gen_ft_kernel(
            aosym, return_complex=True, verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        weight = 1./nkpts

        kpt_allow = np.zeros(3)
        coulG = self.weighted_coulG(kpt_allow, False, mesh)
        coulG_SR = self.weighted_coulG_SR(kpt_allow, False, mesh)
        if cell.dimension >= 2:
            G0_idx = 0  # due to np.fft.fftfreq convension
            G0_weight = kws[0] if isinstance(kws, np.ndarray) else kws
            coulG_SR[G0_idx] += np.pi/self.omega**2 * G0_weight
        # Long-range part is calculated as the difference
        # coulG(cell.omega) - coulG(self.omega) . It can support both regular
        # integrals and LR integrals.
        coulG_LR = coulG - coulG_SR

        if naod > 0:
            aoR_ks, aoI_ks = rsdf_builder._eval_gto(cell_d, mesh, kpts)
            smooth_bas_mask = rs_cell.bas_type == ft_ao.SMOOTH_BASIS
            smooth_bas_idx = rs_cell.bas_map[smooth_bas_mask]
            smooth_ao_idx = rs_cell.get_ao_indices(smooth_bas_idx, cell.ao_loc)

            # rho = einsum('nkji,kig,kjg->ng', dm, ao.conj(), ao)
            rho = np.zeros((n_dm, ngrids))
            tmpR = np.empty((naod, ngrids))
            tmpI = np.empty((naod, ngrids))
            dmR_dd = np.asarray(dms.real[:,:,smooth_ao_idx[:,None],smooth_ao_idx], order='C')
            dmI_dd = np.asarray(dms.imag[:,:,smooth_ao_idx[:,None],smooth_ao_idx], order='C')
            # vG = einsum('ij,gji->g', dm_dd[k], aoao[k]) * coulG
            for i in range(n_dm):
                for k in range(nkpts):
                    zdotNN(dmR_dd[i,k].T, dmI_dd[i,k].T, aoR_ks[k], aoI_ks[k], 1, tmpR, tmpI)
                    rho[i] += np.einsum('ig,ig->g', aoR_ks[k], tmpR)
                    rho[i] += np.einsum('ig,ig->g', aoI_ks[k], tmpI)
            vG_dd = pbctools.ifft(rho, mesh) * cell.vol * coulG
            vG_dd *= weight
            tmpR = tmpI = dmR_dd = dmI_dd = None
            cpu1 = log.timer_debug1('get_lr_j_kpts dd block', *cpu0)

            mem_now = lib.current_memory()[0]
            max_memory = (self.max_memory - mem_now) * .9
            log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)
            Gblksize = max(72, int(max_memory*1e6/16/nao**2/(nkpts+1)))
            log.debug1('Gblksize = %d', Gblksize)
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts)
                #: aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG[p0:p1], weight)
                #: aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG_SR[p0:p1], -weight)
                for i in range(n_dm):
                    rho = np.einsum('kij,kgij->g', dms[i].conj(), Gpq).conj()
                    # NOTE: vG_dd are updated inplace. It stores the full vG then
                    vG = vG_dd[i,p0:p1]
                    vG += coulG[p0:p1] * weight * rho
                    vG_SR = coulG_SR[p0:p1] * weight * rho
                    # vG_LR contains full vG of dd-block and vG_LR of rest blocks
                    vG_LR = vG - vG_SR
                    vj_kpts[i] += np.einsum('g,kgij->kij', vG_LR, Gpq)
                Gpq = None
            log.timer_debug1('get_lr_j_kpts ft_aopair', *cpu1)

            vR = pbctools.fft(vG_dd, mesh).real * (cell.vol/ngrids)
            vjR_dd = np.empty((naod, naod))
            vjI_dd = np.empty((naod, naod))
            for i in range(n_dm):
                for k in range(nkpts):
                    tmpR = aoR_ks[k] * vR[i]
                    tmpI = aoI_ks[k] * vR[i]
                    zdotCN(aoR_ks[k], aoI_ks[k], tmpR.T, tmpI.T, 1, vjR_dd, vjI_dd)
                    lib.takebak_2d(vj_kpts[i,k], vjR_dd + vjI_dd * 1j,
                                   smooth_ao_idx, smooth_ao_idx)

        else:
            max_memory = (self.max_memory - lib.current_memory()[0]) * .9
            Gblksize = max(16, int(max_memory*1e6/16/nao**2/(nkpts+1)))
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts)
                for i in range(n_dm):
                    rho = np.einsum('kij,kgij->g', dms[i].conj(), Gpq).conj()
                    vG_LR = coulG_LR[p0:p1] * weight * rho
                    vj_kpts[i] += np.einsum('g,kgij->kij', vG_LR, Gpq)
                Gpq = None

        log.timer_debug1('get_lr_j_kpts', *cpu0)
        return vj_kpts

    def _get_lr_j_for_bands(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        raise NotImplementedError

    def _get_lr_k_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
                       exxdiv=None):
        '''
        Long-range part of K matrix

        C ~ compact basis, D ~ diffused basis

        Compute K matrix with coulG_LR:
        (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)

        Compute K matrix with full coulG:
        (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
        '''
        assert kpts_band is None
        cpu0 = cpu1 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        cell_d = rs_cell.smooth_basis_cell()

        mesh = self.mesh
        ngrids = np.prod(mesh)
        dm_kpts = lib.asarray(dm_kpts, order='C')
        dms = _format_dms(dm_kpts, kpts)
        nset, nkpts, nao = dms.shape[:3]
        naod = cell_d.nao

        kpts_band = _format_kpts_band(kpts_band, kpts)
        nband = len(kpts_band)
        vkR = np.zeros((nset,nband,nao,nao))
        vkI = np.zeros((nset,nband,nao,nao))
        dmsR = np.asarray(dms.real, order='C')
        dmsI = np.asarray(dms.imag, order='C')
        vk = [vkR, vkI]
        dm = [dmsR, dmsI]
        weight = 1. / nkpts

        aosym = 's1'
        ft_kern = self.supmol_ft.gen_ft_kernel(
            aosym, return_complex=False, verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

        uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
            cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
        scaled_uniq_kpts = cell.get_scaled_kpts(uniq_kpts).round(5)
        log.debug('Num uniq kpts %d', len(uniq_kpts))
        log.debug2('Scaled unique kpts %s', scaled_uniq_kpts)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now))
        log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)

        cache_size = (naod*(naod+1)*ngrids*(nkpts+1))*16e-6
        log.debug1('naod = %d cache_size = %d', naod, cache_size)

        if naod > 0:
            # TODO: less number of planewaves are needed with AFT(cell_d).
            # Test if the dd_block should be computed with FFTDF
            ao_loc = cell.ao_loc
            smooth_bas_mask = rs_cell.bas_type == ft_ao.SMOOTH_BASIS
            smooth_bas_idx = rs_cell.bas_map[smooth_bas_mask]
            smooth_ao_idx = rs_cell.get_ao_indices(smooth_bas_idx, ao_loc)

            # compute the dd blocks with fft_aopair_dd is more efficient than
            # aft_aopair_dd. But it requires a large mount of memory
            if max_memory * .8 > cache_size:
                log.debug1('merge_dd with cached fft_aopair_dd')
                aoR_ks, aoI_ks = rsdf_builder._eval_gto(cell_d, mesh, kpts)
                coords = cell_d.get_uniform_grids(mesh)
                max_memory -= cache_size

                def fft_aopair_dd(ki, kj, expmikr):
                    # einsum('g,ig,jg->ijg', expmikr, ao_ki.conj(), ao_kj)
                    pqG_ddR = np.empty((naod**2, ngrids))
                    pqG_ddI = np.empty((naod**2, ngrids))
                    expmikrR, expmikrI = expmikr
                    libpbc.PBC_zjoin_fCN_s1(
                        pqG_ddR.ctypes.data_as(ctypes.c_void_p),
                        pqG_ddI.ctypes.data_as(ctypes.c_void_p),
                        expmikrR.ctypes.data_as(ctypes.c_void_p),
                        expmikrI.ctypes.data_as(ctypes.c_void_p),
                        aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
                        aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
                        aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
                        aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naod), ctypes.c_int(naod), ctypes.c_int(ngrids))
                    pqG_dd = pqG_ddR + pqG_ddI * 1j
                    pqG_ddR = pqG_ddI = None
                    pqG_dd *= cell.vol / ngrids
                    pqG_dd = pbctools.fft(pqG_dd, mesh)
                    return pqG_dd.reshape(naod, naod, ngrids)

                def merge_dd(Gpq, p0, p1, ki_lst, kj_lst, cache):
                    '''Merge diffused basis block into ao-pair tensor inplace'''
                    expmikr = np.exp(-1j * np.dot(coords, kpts[kj_lst[0]]-kpts[ki_lst[0]]))
                    expmikrR = np.asarray(expmikr.real, order='C')
                    expmikrI = np.asarray(expmikr.imag, order='C')
                    GpqR, GpqI = Gpq
                    # Gpq should be an array of (nkpts,ni,nj,ngrids) in C order
                    if not GpqR[0].flags.c_contiguous:
                        assert GpqR[0].strides[0] == 8  # stride for grids
                    for k, (ki, kj) in enumerate(zip(ki_lst, kj_lst)):
                        if (ki, kj) not in cache:
                            log.debug3('cache dd block (%d, %d)', ki, kj)
                            cache[ki, kj] = fft_aopair_dd(ki, kj, (expmikrR, expmikrI))

                        pqG_dd = cache[ki, kj]
                        libpbc.PBC_ft_zfuse_dd_s1(
                            GpqR[k].ctypes.data_as(ctypes.c_void_p),
                            GpqI[k].ctypes.data_as(ctypes.c_void_p),
                            pqG_dd.ctypes.data_as(ctypes.c_void_p),
                            smooth_ao_idx.ctypes.data_as(ctypes.c_void_p),
                            (ctypes.c_int*2)(p0, p1), ctypes.c_int(nao),
                            ctypes.c_int(naod), ctypes.c_int(ngrids))
                    return (GpqR, GpqI)
            else:
                log.debug1('merge_dd with aft_aopair_dd')

                if self.supmol_d is None:
                    rcut = ft_ao.estimate_rcut(cell_d)
                    supmol_d = ft_ao.ExtendedMole.from_cell(cell_d, kmesh, rcut.max(), log)
                    self.supmol_d = supmol_d.strip_basis(rcut)
                    log.debug('supmol_d nbas = %d cGTO = %d',
                              self.supmol_d.nbas, self.supmol_d.nao)
                aft_aopair_dd = self.supmol_d.gen_ft_kernel(
                    aosym, return_complex=False, verbose=log)

                def merge_dd(Gpq, p0, p1, ki_lst, kj_lst, cache):
                    '''Merge diffused basis block into ao-pair tensor inplace'''
                    kpt = kpts[kj_lst[0]] - kpts[ki_lst[0]]
                    kptjs = kpts[kptj_idx]
                    GpqR, GpqI = Gpq
                    pqG_ddR, pqG_ddI = aft_aopair_dd(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, kptjs)
                    nkpts = len(kptjs)
                    # Gpq should be an array of (nkpts,ni,nj,ngrids) in C order
                    if not GpqR[0].flags.c_contiguous:
                        assert GpqR[0].strides[0] == 8  # stride for grids
                    for k in range(nkpts):
                        libpbc.PBC_ft_fuse_dd_s1(
                            GpqR[k].ctypes.data_as(ctypes.c_void_p),
                            GpqI[k].ctypes.data_as(ctypes.c_void_p),
                            pqG_ddR[k].ctypes.data_as(ctypes.c_void_p),
                            pqG_ddI[k].ctypes.data_as(ctypes.c_void_p),
                            smooth_ao_idx.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(nao), ctypes.c_int(naod), ctypes.c_int(p1-p0))
                    return (GpqR, GpqI)

            cpu1 = log.timer_debug1('get_lr_k_kpts dd block', *cpu1)

        for group_id, (k, k_conj) in enumerate(group_by_conj_pairs(cell, uniq_kpts)[0]):
            kpt_ij_idx = np.where(uniq_inverse == k)[0]
            kpti_idx = kpt_ij_idx // nkpts
            kptj_idx = kpt_ij_idx % nkpts
            nkptj = len(kptj_idx)
            kptjs = kpts[kptj_idx]
            kpt = uniq_kpts[k]
            log.debug1('ft_ao_pair for kpt = %s', kpt)
            log.debug2('ft_ao_pair for kpti_idx = %s', kpti_idx)
            log.debug2('ft_ao_pair for kptj_idx = %s', kptj_idx)
            swap_2e = k_conj is not None and k != k_conj

            coulG = self.weighted_coulG(kpt, exxdiv, mesh)
            coulG_SR = self.weighted_coulG_SR(kpt, False, mesh)

            # G=0 associated to 2e integrals in real-space
            if cell.dimension >= 2 and is_zero(uniq_kpts[k]):
                G0_idx = 0
                G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
                coulG_SR[G0_idx] += np.pi/self.omega**2 * G0_weight
            coulG_LR = coulG - coulG_SR

            Gblksize = max(56, int(max_memory*1e6/16/nao**2/(nkptj+max(nkptj,3))))
            Gblksize = min(Gblksize, ngrids, 200000)
            log.debug1('Gblksize = %d', Gblksize)
            if naod > 0:
                cache = {}
                vkcoulG = self.weighted_coulG(kpt, exxdiv, mesh)
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    log.debug3('_update_vk_ [%s:%s]', p0, p1)
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, kptjs)
                    _update_vk_(vk, Gpq, dm, coulG_SR[p0:p1],
                                -weight, kpti_idx, kptj_idx, swap_2e)
                    Gpq = merge_dd(Gpq, p0, p1, kpti_idx, kptj_idx, cache)
                    _update_vk_(vk, Gpq, dm, vkcoulG[p0:p1],
                                weight, kpti_idx, kptj_idx, swap_2e)
                    Gpq = None
            else:
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    log.debug3('_update_vk_ [%s:%s]', p0, p1)
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, kptjs)
                    _update_vk_(vk, Gpq, dm, coulG_LR[p0:p1],
                                weight, kpti_idx, kptj_idx, swap_2e)
                    Gpq = None
            cpu1 = log.timer_debug1('ft_aopair group %d'%group_id, *cpu1)

        if (is_zero(kpts) and is_zero(kpts_band) and
            not np.iscomplexobj(dm_kpts)):
            vk_kpts = vkR
        else:
            vk_kpts = vkR + vkI * 1j

        # Add ewald_exxdiv contribution because G=0 was not included in the
        # non-uniform grids
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            _ewald_exxdiv_for_G0(cell, kpts_band, dms, vk_kpts, kpts_band)

        log.timer_debug1('get_lr_k_kpts', *cpu0)
        return vk_kpts

RangeSeparationJKBuilder = RangeSeparatedJKBuilder

def _purify(mat_kpts, phase):
    #:mat_bvk = np.einsum('Rk,nkij,Sk->nRSij', phase, mat_kpts, phase.conj())
    #:return np.einsum('Rk,nRSij,Sk->nkij', phase.conj(), mat_bvk.real, phase)
    nkpts = phase.shape[1]
    mat_bvk = lib.einsum('k,Sk,nkuv->nSuv', phase[0], phase.conj(), mat_kpts)
    return lib.einsum('S,Sk,nSuv->nkuv', nkpts*phase[:,0].conj(), phase, mat_bvk.real)


def estimate_rcut(rs_cell, omega, precision=None):
    '''Estimate rcut for 2e SR-integrals'''
    if precision is None:
        precision = rs_cell.precision

    rs_cell = rs_cell
    exps = np.array([e.min() for e in rs_cell.bas_exps()])
    ls = rs_cell._bas[:,gto.ANG_OF]
    cs = gto.gto_norm(ls, exps)
    exp_min_idx = exps.argmin()
    ak_idx = exp_min_idx
    compact_mask = rs_cell.bas_type != ft_ao.SMOOTH_BASIS
    exps_c = exps[compact_mask]
    if exps_c.size > 0:
        ak_idx = exps_c.argmin()
    ak = exps[ak_idx]
    lk = rs_cell._bas[ak_idx,gto.ANG_OF]
    ck = cs[ak_idx]
    aj = exps
    lj = ls
    cj = cs
    ai = al = exps[exp_min_idx]
    li = ll = rs_cell._bas[exp_min_idx,gto.ANG_OF]
    ci = cl = cs[exp_min_idx]

    aij = ai + aj
    akl = ak + al
    lij = li + lj
    lkl = lk + ll
    l4 = lij + lkl
    norm_ang = ((2*li+1)*(2*lj+1)*(2*lk+1)*(2*ll+1)/(4*np.pi)**4)**.5
    c1 = ci * cj * ck * cl * norm_ang
    theta = 1./(1./aij + 1./akl + omega**-2)
    sfac = aij*akl*aj*al/(aij*aj*ak*theta + akl*ai*al*theta + aij*akl*aj*al)
    fl = 2
    fac = 2**(li+lk+1)*np.pi**3.5*c1 * theta**(l4-1.5)
    fac /= aij**(lij+1.5) / akl**(lkl+1.5)
    fac *= (1 + ai/aj)**lj * (1 + ak/al)**ll * fl / precision

    r0 = rs_cell.rcut
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
    rcut = r0

    if 0 < exps_c.size < rs_cell.nbas:
        smooth_mask = ~compact_mask
        ai, li, ci = ak, lk, ck
        aj = exps[smooth_mask]
        lj = ls[smooth_mask]
        cj = cs[smooth_mask]
        aij = ai + aj
        lij = li + lj
        l4 = lij + lkl
        norm_ang = ((2*li+1)*(2*lj+1)*(2*lk+1)*(2*ll+1)/(4*np.pi)**4)**.5
        c1 = ci * cj * ck * cl * norm_ang
        theta = 1./(1./aij + 1./akl + omega**-2)
        sfac = aij*akl*aj*al/(aij*aj*ak*theta + akl*ai*al*theta + aij*akl*aj*al)
        fl = 2
        fac = 2**(li+lk+1)*np.pi**3.5*c1 * theta**(l4-1.5)
        fac /= aij**(lij+1.5) / akl**(lkl+1.5)
        fac *= (1 + ai/aj)**lj * (1 + ak/al)**ll * fl / precision

        r0 = rs_cell.rcut
        r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
        r0 = (np.log(fac * r0 * (sfac*r0)**(l4-2) + 1.) / (sfac*theta))**.5
        rcut[smooth_mask] = r0
    return rcut

def _guess_omega(cell, kpts, mesh=None):
    precision = cell.precision
    a = cell.lattice_vectors()
    naop = cell.npgto_nr()
    nao = cell.nao
    nkpts = len(kpts)
    if mesh is None:
        rcut = cell.rcut
        omega_min = 0.25
        omega_min = 0.75 * (-np.log(precision * np.pi**.5 * rcut**2 * omega_min))**.5 / rcut
        ke_min = estimate_ke_cutoff_for_omega(cell, omega_min)
        mesh_min = pbctools.cutoff_to_mesh(a, ke_min)
        # FIXME: balance the two workloads
        # int2e integrals ~ naop*(cell.rcut**3/cell.vol*naop)**3
        # ft_ao integrals ~ nkpts*naop*(cell.rcut**3/cell.vol*naop)*mesh**3
        #                   nkpts**2*naop**3*mesh**3
        nimgs = (cell.rcut**3 / cell.vol) ** (cell.dimension / 3)
        # mesh = [max(4, int((nimgs * naop**2 / nkpts**.5) ** (1./3) * 0.5))] * 3
        # mesh = [max(4, int((nimgs**1.5 * naop**2 / nkpts**.5) ** (1./3) * 0.2))] * 3
        # mesh = [max(4, int((nimgs**2 * naop**2 / nkpts**.5) ** (1./3) * 0.125))] * 3
        # mesh = [max(4, int((nimgs**1.5 * naop**1.5 / nkpts**.5) ** (1./3) * 0.5))] * 3
        # mesh = [max(4, int((nimgs * naop / nkpts**.5) ** (1./3) * 1.5))] * 3
        # mesh = [max(4, int((nimgs * naop / nkpts**.5) ** (1./3) * 1.5))] * 3
        # mesh = [max(4, int((nimgs * naop / nkpts**(1./3)) ** (1./3) * 1.5))] * 3
        nimgs = 8 * nimgs
        mesh = (nimgs**2*naop**2 / (nkpts**.5 * naop * nimgs * 2e3 +
                                    nkpts**2*nao**2))**(1./3) * 8 + 1
        mesh = int(min((cell.max_memory*1e6/32/(.7*nao)**2)**(1./3), mesh))
        mesh = rsdf_builder._round_off_to_odd_mesh(mesh)
        mesh = np.max([mesh_min, [mesh] * 3], axis=0)
        ke_cutoff = pbctools.mesh_to_cutoff(a, mesh)
        ke_cutoff = ke_cutoff[:cell.dimension].min()
        if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
            mesh[cell.dimension:] = cell.mesh[cell.dimension:]
        elif cell.dimension == 2:
            mesh = pbctools.cutoff_to_mesh(a, ke_cutoff)
    else:
        ke_cutoff = min(pbctools.mesh_to_cutoff(a, mesh)[:cell.dimension])
    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision)
    return omega, mesh, ke_cutoff

def _update_vk_(vk, Gpq, dms, coulG, weight, kpti_idx, kptj_idx, swap_2e):
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmsR, dmsI = dms
    nG = len(coulG)
    n_dm = vkR.shape[0]
    nao = vkR.shape[-1]
    bufR = np.empty((nG*nao**2))
    bufI = np.empty((nG*nao**2))
    buf1R = np.empty((nG*nao**2))
    buf1I = np.empty((nG*nao**2))

    for k, (ki, kj) in enumerate(zip(kpti_idx, kptj_idx)):
        # case 1: k_pq = (pi|iq)
        #:v4 = np.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,jk->il', v4, dm)
        pLqR = np.ndarray((nao,nG,nao), buffer=bufR)
        pLqI = np.ndarray((nao,nG,nao), buffer=bufI)
        pLqR[:] = GpqR[k].transpose(1,0,2)
        pLqI[:] = GpqI[k].transpose(1,0,2)
        iLkR = np.ndarray((nao,nG,nao), buffer=buf1R)
        iLkI = np.ndarray((nao,nG,nao), buffer=buf1I)
        for i in range(n_dm):
            zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                   dmsR[i,kj], dmsI[i,kj], 1,
                   iLkR.reshape(-1,nao), iLkI.reshape(-1,nao))
            iLkR *= coulG.reshape(1,nG,1)
            iLkI *= coulG.reshape(1,nG,1)
            zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                   pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                   weight, vkR[i,ki], vkI[i,ki], 1)

        # case 2: k_pq = (iq|pi)
        #:v4 = np.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,li->kj', v4, dm)
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        if swap_2e:
            for i in range(n_dm):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1,
                       iLkR.reshape(nao,-1), iLkI.reshape(nao,-1))
                iLkR *= coulG.reshape(1,nG,1)
                iLkI *= coulG.reshape(1,nG,1)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                       weight, vkR[i,kj], vkI[i,kj], 1)

def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
    '''Energy cutoff for FFTDF to converge attenuated Coulomb in moment space
    '''
    if precision is None:
        precision = cell.precision
    ai = np.hstack(cell.bas_exps()).max()
    theta = 1./(1./ai + omega**-2)
    fac = 32*np.pi**2 * theta / precision
    Ecut = 20.
    Ecut = np.log(fac / (2*Ecut) + 1.) * 2*theta
    Ecut = np.log(fac / (2*Ecut) + 1.) * 2*theta
    return Ecut

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulombl given energy cutoff
    '''
    if precision is None:
        precision = cell.precision
#    # esitimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
#    # enough to converge the 2-electron integrals. A penalty term here is to
#    # reduce the error in integrals
#    precision *= 1e-2
#    kmax = (ke_cutoff*2)**.5
#    log_rest = np.log(precision / (16*np.pi**2 * kmax**lmax))
#    omega = (-.5 * ke_cutoff / log_rest)**.5
#    return omega

    ai = np.hstack(cell.bas_exps()).max()
    aij = ai * 2
    fac = 32*np.pi**2 / precision
    omega = 2.
    theta = 1./(1./ai + omega**-2)
    omega2 = 1./(np.log(fac * theta/ (2*ke_cutoff) + 1.)*2/ke_cutoff - 1./aij)
    if omega2 < 0:
        omega = 2
    else:
        omega = min(omega2, 4.)**.5
    return omega
