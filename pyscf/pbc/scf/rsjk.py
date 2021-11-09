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
from pyscf.pbc.df import aft, aft_jk
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf import __config__

# Threshold of steep bases and local bases
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 3.2)


libpbc = lib.load_library('libpbc')

class RangeSeparationJKBuilder(object):
    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = None
        self.kpts = np.reshape(kpts, (-1, 3))
        self.purify = True

        self.omega = None
        self.cell_rs = None
        # Born-von Karman supercell
        self.bvkcell = None
        self.bvkmesh_Ls = None
        self.bvk_kmesh = None
        self.phase = None
        self.supmol = None
        self.supmol_Ls = None
        # bvk_bas_mask[bvk_cell_id, bas_id]: if basis in bvkcell presented in supmol
        self.bvk_bas_mask = None
        # For shells in the supmol, cell0_shl_id is the shell ID in cell0
        self.cell0_shl_id = None
        # bvk_cell_id is the Id of image in BvK supercell
        self.bvk_cell_id = None
        # For shells in bvkcell, use overlap mask to remove d-d block
        self.ovlp_mask = None
        self.lr_aft = None
        self.ke_cutoff = None
        self.vhfopt = None

        # to mimic molecular DF object
        self.blockdim = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'omega = %s', self.omega)
        #logger.info(self, 'len(kpts) = %d', len(self.kpts))
        #logger.debug1(self, '    kpts = %s', self.kpts)
        if self.lr_aft is not None:
            self.lr_aft.dump_flags(verbose)
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.supmol = None
        return self

    def build(self, omega=None, direct_scf_tol=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        cell = self.cell
        kpts = self.kpts

        k_scaled = cell.get_scaled_kpts(kpts).sum(axis=0)
        k_mod_to_half = k_scaled * 2 - (k_scaled * 2).round(0)
        if abs(k_mod_to_half).sum() > 1e-5:
            raise NotImplementedError('k-points must be symmetryic')

        if omega is not None:
            self.omega = omega

        if self.omega is None:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            self.ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), self.ke_cutoff)

        logger.info(self, 'omega = %.15g  ke_cutoff = %s  mesh = %s',
                    self.omega, self.ke_cutoff, self.mesh)

        if direct_scf_tol is None:
            direct_scf_tol = cell.precision**1.5
            logger.debug(self, 'Set direct_scf_tol %g', direct_scf_tol)

        self.cell_rs = cell_rs = _re_contract_cell(cell, self.ke_cutoff)
        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell_rs, kpts)
        bvkcell, phase = k2gamma.get_phase(cell_rs, kpts, kmesh)
        self.bvkmesh_Ls = Ks = k2gamma.translation_vectors_for_kmesh(cell_rs, kmesh)
        self.bvkcell = bvkcell
        self.phase = phase

        # Given ke_cutoff, eta corresponds to the most steep Gaussian basis
        # of which the Coulomb integrals can be accurately computed in moment
        # space.
        eta = aft.estimate_eta_for_ke_cutoff(cell, self.ke_cutoff,
                                             precision=cell.precision)
        # * Assuming the most steep function in smooth basis has exponent eta,
        # with attenuation parameter omega, rcut_sr is the distance of which
        # the value of attenuated Coulomb integrals of four shells |eta> is
        # smaller than the required precision.
        # * The attenuated coulomb integrals between four s-type Gaussians
        # (2*a/pi)^{3/4}exp(-a*r^2) is
        #   (erfc(omega*a^0.5/(omega^2+a)^0.5*R) - erfc(a^0.5*R)) / R
        # if two Gaussians on one center and the other two on another center
        # and the distance between the two centers are R.
        # * The attenuated coulomb integrals between two spherical charge
        # distributions is
        #   ~(pi/eta)^3/2 (erfc(tau*(eta/2)^0.5*R) - erfc((eta/2)^0.5*R)) / R
        #       tau = omega/sqrt(omega^2 + eta/2)
        # if the spherical charge distribution is the product of above s-type
        # Gaussian with exponent eta and a very smooth function.
        # When R is large, the attenuated Coulomb integral is
        #   ~= (pi/eta)^3/2 erfc(tau*(eta/2)^0.5*R) / R
        #   ~= pi/(tau*eta^2*R^2) exp(-tau^2*eta*R^2/2)
        tau = self.omega / (self.omega**2 + eta/2)**.5
        rcut_sr = 10  # initial guess
        rcut_sr = (-np.log(direct_scf_tol * tau * (eta * rcut_sr)**2/np.pi) / (tau**2*eta/2))**.5
        logger.debug(self, 'eta = %g  rcut_sr = %g', eta, rcut_sr)

        # Ls is the translation vectors to mimic periodicity of a cell
        Ls = bvkcell.get_lattice_Ls(rcut=cell.rcut+rcut_sr)
        self.supmol_Ls = Ls = Ls[np.linalg.norm(Ls, axis=1).argsort()]

        supmol = _make_extended_mole(cell_rs, Ls, Ks, self.omega, direct_scf_tol)
        self.supmol = supmol

        nkpts = len(self.bvkmesh_Ls)
        nbas = cell_rs.nbas
        n_steep, n_local, n_diffused = cell_rs._nbas_each_set
        n_compact = n_steep + n_local
        bas_mask = supmol._bas_mask

        self.bvk_bas_mask = bvk_bas_mask = bas_mask.any(axis=2)
        # Some basis in bvk-cell are not presented in the supmol. They can be
        # skipped when computing SR integrals
        self.bvkcell._bas = bvkcell._bas[bvk_bas_mask.ravel()]

        # Record the mapping between the dense bvkcell basis and the
        # original sparse bvkcell basis
        bvk_cell_idx = np.repeat(np.arange(nkpts)[:,None], nbas, axis=1)
        self.bvk_cell_id = bvk_cell_idx[bvk_bas_mask].astype(np.int32)
        cell0_shl_idx = np.repeat(np.arange(nbas)[None,:], nkpts, axis=0)
        self.cell0_shl_id = cell0_shl_idx[bvk_bas_mask].astype(np.int32)

        logger.timer_debug1(self, 'initializing supmol', *cpu0)
        logger.info(self, 'sup-mol nbas = %d cGTO = %d pGTO = %d',
                    supmol.nbas, supmol.nao, supmol.npgto_nr())

        supmol.omega = -self.omega  # Set short range coulomb
        with supmol.with_integral_screen(direct_scf_tol**2):
            vhfopt = _vhf.VHFOpt(supmol, 'int2e_sph',
                                 qcondname=libpbc.PBCVHFsetnr_direct_scf)
        vhfopt.direct_scf_tol = direct_scf_tol
        self.vhfopt = vhfopt
        logger.timer(self, 'initializing vhfopt', *cpu0)

        q_cond = vhfopt.get_q_cond((supmol.nbas, supmol.nbas))
        idx = supmol._images_loc
        bvk_q_cond = lib.condense('NP_absmax', q_cond, idx, idx)
        ovlp_mask = bvk_q_cond > direct_scf_tol
        # Remove diffused-diffused block
        if n_diffused > 0:
            diffused_mask = np.zeros_like(bvk_bas_mask)
            diffused_mask[:,n_compact:] = True
            diffused_mask = diffused_mask[bvk_bas_mask]
            ovlp_mask[diffused_mask[:,None] & diffused_mask] = False
        self.ovlp_mask = ovlp_mask.astype(np.int8)

        # mute rcut_threshold, divide basis into two sets only
        cell_lr_aft = _re_contract_cell(cell, self.ke_cutoff, -1, verbose=0)
        self.lr_aft = lr_aft = _LongRangeAFT(cell_lr_aft, kpts,
                                             self.omega, self.bvk_kmesh)
        lr_aft.ke_cutoff = self.ke_cutoff
        lr_aft.mesh = self.mesh
        lr_aft.eta = eta
        return self

    def get_jk(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            # TODO: call AFTDF.get_jk function
            raise NotImplementedError

        # Does not support to specify arbitrary kpts
        if kpts is not None and abs(kpts-self.kpts).max() > 1e-7:
            raise RuntimeError('kpts error')
        kpts = self.kpts

        if kpts_band is not None:
            raise NotImplementedError

        cpu0 = logger.process_clock(), logger.perf_counter()
        if self.supmol is None:
            self.build()

        nkpts = kpts.shape[0]
        vhfopt = self.vhfopt
        supmol = self.supmol
        bvkcell = self.bvkcell
        phase = self.phase
        cell = self.cell_rs
        nao = cell.nao
        orig_nao = self.cell.nao

        # * dense_bvk_ao_loc are the AOs which appear in supmol (some basis
        # are removed)
        # * sparse_ao_loc has dimension (Nk,nbas), corresponding to the
        # bvkcell with all basis
        dense_bvk_ao_loc = bvkcell.ao_loc
        sparse_ao_loc = nao * np.arange(nkpts)[:,None] + cell.ao_loc[:-1]
        sparse_ao_loc = np.append(sparse_ao_loc.ravel(), nao * nkpts)
        nbands = nkpts

        if dm_kpts.ndim != 4:
            dm = dm_kpts.reshape(-1, nkpts, orig_nao, orig_nao)
        else:
            dm = dm_kpts
        n_dm = dm.shape[0]

        rs_c_coeff = cell._contr_coeff
        sc_dm = lib.einsum('nkij,pi,qj->nkpq', dm, rs_c_coeff, rs_c_coeff)
        # Utilized symmetry sc_dm[R,S] = sc_dm[S-R] = sc_dm[(S-R)%N]
        #:sc_dm = lib.einsum('Rk,nkuv,Sk->nRuSv', phase, sc_dm, phase.conj())
        sc_dm = lib.einsum('k,Sk,nkuv->nSuv', phase[0], phase.conj(), sc_dm)
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
        sc_dm = np.asarray(sc_dm.reshape(-1, nkpts, nao, nao), order='C')
        n_sc_dm = sc_dm.shape[0]

        dm_cond = [lib.condense('NP_absmax', d, sparse_ao_loc, sparse_ao_loc[:cell.nbas+1])
                   for d in sc_dm]
        dm_cond = np.asarray(np.max(dm_cond, axis=0), order='C')
        libpbc.CVHFset_dm_cond(vhfopt._this,
                               dm_cond.ctypes.data_as(ctypes.c_void_p), dm_cond.size)
        dm_cond = None

        bvk_nbas = bvkcell.nbas
        shls_slice = (0, cell.nbas, 0, bvk_nbas, 0, bvk_nbas, 0, bvk_nbas)

        if hermi:
            fdot_suffix = 's2kl'
        else:
            fdot_suffix = 's1'
        if with_j and with_k:
            fdot = 'PBCVHF_contract_jk_' + fdot_suffix
            vs = np.zeros((2, n_sc_dm, nao, nkpts, nao))
        elif with_j:
            fdot = 'PBCVHF_contract_j_' + fdot_suffix
            vs = np.zeros((1, n_sc_dm, nao, nkpts, nao))
        else:  # with_k
            fdot = 'PBCVHF_contract_k_' + fdot_suffix
            vs = np.zeros((1, n_sc_dm, nao, nkpts, nao))

        drv = libpbc.PBCVHF_direct_drv
        drv(getattr(libpbc, fdot), vs.ctypes.data_as(ctypes.c_void_p),
            sc_dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
            ctypes.c_int(nkpts), ctypes.c_int(nbands), ctypes.c_int(cell.nbas),
            self.ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            self.bvk_cell_id.ctypes.data_as(ctypes.c_void_p),
            self.cell0_shl_id.ctypes.data_as(ctypes.c_void_p),
            supmol._images_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*8)(*shls_slice),
            dense_bvk_ao_loc.ctypes.data_as(ctypes.c_void_p),
            dm_translation.ctypes.data_as(ctypes.c_void_p),
            vhfopt._cintopt, vhfopt._this,
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))

        if is_complex_dm:
            vs = vs[:,:n_dm] + vs[:,n_dm:] * 1j

        if with_j and with_k:
            vj, vk = vs
        elif with_j:
            vj, vk = vs[0], None
        else:
            vj, vk = None, vs[0]
        cpu1 = logger.timer(self, 'short range part vj and vk', *cpu0)

        lr_c_coeff = self.lr_aft.cell._contr_coeff
        lr_dm = lib.einsum('nkij,pi,qj->nkpq', dm, lr_c_coeff, lr_c_coeff)
        # For rho product other than diffused-diffused block, construct LR
        # parts in terms of full ERIs and SR ERIs
        vj1, vk1 = self.lr_aft.get_jk(lr_dm, hermi, kpts, kpts_band,
                                      with_j, with_k, exxdiv=exxdiv)
        cpu1 = logger.timer(self, 'AFT-vj and AFT-vk', *cpu1)

        # expRk is almost the same to phase, except a normalization factor
        expRk = np.exp(1j*np.dot(self.bvkmesh_Ls, kpts.T))

        if with_j:
            vj = lib.einsum('npRq,pi,qj,Rk->nkij', vj, rs_c_coeff, rs_c_coeff, expRk)
            vj += lib.einsum('nkpq,pi,qj->nkij', vj1, lr_c_coeff, lr_c_coeff)
            if self.purify and kpts_band is None:
                vj = _purify(vj, phase)
            if gamma_point(kpts) and dm_kpts.dtype == np.double:
                vj = vj.real
            if hermi:
                vj = (vj + vj.conj().transpose(0,1,3,2)) * .5
            vj = vj.reshape(dm_kpts.shape)

        if with_k:
            vk = lib.einsum('npRq,pi,qj,Rk->nkij', vk, rs_c_coeff, rs_c_coeff, expRk)
            vk += lib.einsum('nkpq,pi,qj->nkij', vk1, lr_c_coeff, lr_c_coeff)
            if self.purify and kpts_band is None:
                vk = _purify(vk, phase)
            if gamma_point(kpts) and dm_kpts.dtype == np.double:
                vk = vk.real
            if hermi:
                vk = (vk + vk.conj().transpose(0,1,3,2)) * .5
            vk = vk.reshape(dm_kpts.shape)

        return vj, vk

def _purify(mat_kpts, phase):
    #:mat_bvk = np.einsum('Rk,nkij,Sk->nRSij', phase, mat_kpts, phase.conj())
    #:return np.einsum('Rk,nRSij,Sk->nkij', phase.conj(), mat_bvk.real, phase)
    nkpts = phase.shape[1]
    mat_bvk = lib.einsum('k,Sk,nkuv->nSuv', phase[0], phase.conj(), mat_kpts)
    return lib.einsum('S,Sk,nSuv->nkuv', nkpts*phase[:,0].conj(), phase, mat_bvk.real)

def _make_extended_mole(cell, Ls, bvkmesh_Ls, omega, precision=None, verbose=None):
    from pyscf.pbc.df.ft_ao import _estimate_overlap
    if precision is None:
        precision = cell.precision * 1e-4
    LKs = Ls[:,None,:] + bvkmesh_Ls
    nimgs, nk = LKs.shape[:2]
    nbas = cell.nbas

    supmol = cell.to_mol()
    supmol = pbctools.pbc._build_supcell_(supmol, cell, LKs.reshape(nimgs*nk, 3))

    n_steep, n_local, n_diffused = cell._nbas_each_set
    n_compact = n_steep + n_local

    exps = np.array([cell.bas_exp(ib).min() for ib in range(nbas)])
    aij = exps[:,None] * exps / (exps[:,None] + exps)
    s0 = _estimate_overlap(cell, LKs.reshape(nimgs*nk, 3))
    # (bas_i, bas_j, image_id, bvk_cell_for_j)
    s0max = (2 * (aij[:,:,None]/np.pi)**.5 * s0).max(axis=0)

    if n_diffused > 0 and n_compact > 0:
        # For remotely separated smooth functions |i> and |l>, the eri-type
        # (ij|kl) can have small contributions if |j> and |k> are steep functions
        # and close in real space. Since we are focusing on the SR attenuated
        # Coulomb integrals, we don't have to consider the steep functions j and k
        # remotely separated since SR integrals decays exponently wrt their
        # separation.
        atom_coords = cell.atom_coords()
        bas_coords = atom_coords[cell._bas[:,gto.ATOM_OF]]
        exps_d = exps[n_compact:]  # diffused
        exps_c_min = exps[:n_compact].min()
        aij_min = aij[:n_compact,n_compact:].min(axis=0)
        # The product of two diffused functions
        rij_dd = bas_coords[n_compact:,None,:] - bas_coords[n_compact:]
        # The product of two diffused functions from different cells
        dijL = np.linalg.norm(rij_dd[:,:,None,:] - LKs.reshape(nimgs*nk,3), axis=-1)
        fac = 16/(2*np.pi)**.5/(exps_d + exps_c_min) * aij_min**1.5
        # Estimation of (DC|DC) from four s-type shells
        eri_dcdc = fac[:,None,None] * np.exp(-.5*aij_min[:,None,None] * dijL**2)
        s0max[n_compact:] = eri_dcdc.max(axis=1)

    if n_diffused == 0:
        # When diffused functions exist, ranges of compact functions are
        # determined by the diffused functions. Without diffused functions,
        # assuming two charge distributions are well separated, their
        # interactions are erfc(omega*r12)/r12
        exps_c = exps[:n_compact]
        r2_LKs = np.einsum('lkx,lkx->lk', LKs, LKs).ravel()
        # Exclude cell 0
        assert r2_LKs[0] == 0, 'Ls not sorted'
        r2_LKs = r2_LKs[1:]
        fac = -omega**2*exps_c/(2*omega**2+exps_c)
        upper_bounds = np.exp(fac[:,None] * r2_LKs) / (2*omega*r2_LKs)
        s0max[:n_compact,1:] = np.max([s0max[:n_compact,1:], upper_bounds], axis=0)

    # (bas_id, image_id, bvk_cell_id) -> (bvk_cell_id, bas_id, image_id)
    s0max = s0max.reshape(nbas, nimgs, nk).transpose(2,0,1)
    bas_mask = s0max > precision
    _bas_reordered = supmol._bas.reshape(nimgs,nk,nbas,gto.BAS_SLOTS).transpose(1,2,0,3)
    supmol._bas = np.asarray(_bas_reordered[bas_mask], dtype=np.int32, order='C')

    images_count = np.count_nonzero(bas_mask, axis=2)
    # Some bases are completely local inside the bvk-cell. Exclude them from
    # lattice sum.
    images_loc = np.append(0, np.cumsum(images_count[images_count != 0]))
    supmol._images_loc = images_loc.astype(np.int32)
    supmol._bas_mask = bas_mask

    log = logger.new_logger(cell, verbose)
    log.debug('Steep basis in sup-mol %d', np.count_nonzero(bas_mask[:,:n_steep,:]))
    log.debug('Local basis in sup-mol %d', np.count_nonzero(bas_mask[:,n_steep:n_compact,:]))
    log.debug('Diffused basis in sup-mol %d', np.count_nonzero(bas_mask[:,n_compact:,:]))
    return supmol

def _re_contract_cell(cell, ke_cut_threshold, rcut_threshold=RCUT_THRESHOLD, verbose=None):
    from pyscf.gto import NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, ATOM_OF
    from pyscf.pbc.dft.multigrid import _primitive_gto_cutoff
    log = logger.new_logger(cell, verbose)

    # Split shells based on rcut
    rcuts, kecuts = _primitive_gto_cutoff(cell, cell.precision)

    def transform_(cint_coeff, orig_bas, bas_to_append, env, pexp, pcoeff):
        np1, nc1 = cint_coeff.shape
        l = orig_bas[gto.ANG_OF]
        if cell.cart:
            degen = (l + 1) * (l + 2) // 2
        else:
            degen = 2 * l + 1
        if np1 >= nc1:
            bas = orig_bas.copy()
            bas[NPRIM_OF] = np1
            bas[PTR_EXP] = pexp
            bas[PTR_COEFF] = pcoeff
            bas_to_append.append(bas)
            coeff = np.eye(nc1 * degen)
        else:
            bas = np.repeat(orig_bas.copy()[None,:], np1, axis=0)
            bas[:,NPRIM_OF] = 1
            bas[:,NCTR_OF] = 1
            bas[:,PTR_EXP] = pexp + np.arange(np1)
            bas[:,PTR_COEFF] = pcoeff + np.arange(np1)
            bas_to_append.extend(bas)

            exps = _env[pexp:pexp+np1]
            cs = gto.gto_norm(l, exps)
            _env[pcoeff:pcoeff+np1] = cs
            unit_coeff = np.eye(degen)
            coeff = np.einsum('p,mn,pc->pmcn', 1/cs, unit_coeff, cint_coeff)
            coeff = coeff.reshape(np1*degen, nc1*degen)
        return coeff

    ao_loc = cell.ao_loc_nr()
    _env = cell._env.copy()
    steep_bas = []
    local_bas = []
    smooth_bas = []
    steep_p2c = []
    local_p2c = []
    smooth_p2c = []
    ke_cutoff = 0
    for ib, orig_bas in enumerate(cell._bas):
        nprim = orig_bas[NPRIM_OF]
        nctr = orig_bas[NCTR_OF]
        ke = kecuts[ib]

        smooth_mask = ke < ke_cut_threshold
        steep_mask = (~smooth_mask) & (rcuts[ib] < rcut_threshold)
        local_mask = (~steep_mask) & (~smooth_mask)
        if log.verbose >= logger.DEBUG3:
            log.debug3('bas %d rcuts %s', ib, rcuts)
            log.debug3('bas %d kecuts %s', ib, kecuts)
            log.debug3('steep %s, local %s, smooth', np.where(steep_mask),
                       np.where(local_mask), np.where(smooth_mask))

        pexp = orig_bas[PTR_EXP]
        pcoeff = orig_bas[PTR_COEFF]
        es = cell.bas_exp(ib)
        cs = cell._libcint_ctr_coeff(ib)

        c_steep = cs[steep_mask]
        c_local = cs[local_mask]
        c_smooth = cs[smooth_mask]
        _env[pcoeff:pcoeff+nprim*nctr] = np.hstack([
            c_steep.T.ravel(),
            c_local.T.ravel(),
            c_smooth.T.ravel(),
        ])
        _env[pexp:pexp+nprim] = np.hstack([
            es[steep_mask],
            es[local_mask],
            es[smooth_mask],
        ])

        if c_steep.size > 0:
            steep_p2c.append(transform_(c_steep, orig_bas, steep_bas, _env,
                                        pexp, pcoeff))
        else:
            steep_p2c.append(np.zeros((0, ao_loc[ib+1]-ao_loc[ib])))

        if c_local.size > 0:
            local_p2c.append(transform_(c_local, orig_bas, local_bas, _env,
                                        pexp + c_steep.shape[0],
                                        pcoeff + c_steep.size))
        else:
            local_p2c.append(np.zeros((0, ao_loc[ib+1]-ao_loc[ib])))

        if c_smooth.size > 0:
            smooth_p2c.append(transform_(c_smooth, orig_bas, smooth_bas, _env,
                                         pexp + c_steep.shape[0] + c_local.shape[0],
                                         pcoeff + c_steep.size + c_local.size))
            ke_cutoff = max(ke_cutoff, ke[smooth_mask].max())
        else:
            smooth_p2c.append(np.zeros((0, ao_loc[ib+1]-ao_loc[ib])))

    cell_rs = copy.copy(cell)
    cell_rs._bas = np.asarray(steep_bas + local_bas + smooth_bas,
                              dtype=np.int32, order='C').reshape(-1, gto.BAS_SLOTS)
    cell_rs._env = _env
    cell_rs._contr_coeff = np.vstack([scipy.linalg.block_diag(*steep_p2c),
                                      scipy.linalg.block_diag(*local_p2c),
                                      scipy.linalg.block_diag(*smooth_p2c)])
    cell_rs._nbas_each_set = (len(steep_bas), len(local_bas), len(smooth_bas))
    log.debug('No. steep_bas %d', len(steep_bas))
    log.debug('No. local_bas %d', len(local_bas))
    log.debug('No. smooth_bas %d', len(smooth_bas))
    return cell_rs

def _guess_omega(cell, kpts, mesh=None):
    nao = cell.npgto_nr()
    nkpts = len(kpts)
    nkk = nkpts**(1./3) * 2 - 1
    if mesh is None:
        mesh = [max(5, int(cell.rcut * nao ** (1./3) / nkk + 1))] * 3
        mesh = np.min([cell.mesh, mesh], axis=0)
    ke_cutoff = min(pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh[:cell.dimension]))
    omega = aft.estimate_omega_for_ke_cutoff(cell, ke_cutoff)
    return omega, mesh, ke_cutoff

class _LongRangeAFT(aft.AFTDF):
    '''
    Regular Coulomb metric for (DD|DD), (DD|CD), (DD|CC)
    Short-range Coulomb metric for (CD|CD), (CD|CC), (CC|CC)
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)), omega=None, bvk_kmesh=None):
        self.omega = omega
        self.bvk_kmesh = bvk_kmesh
        aft.AFTDF.__init__(self, cell, kpts)

    def weighted_coulG_LR(self, kpt=np.zeros(3), exx=False, mesh=None):
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if cell.omega != 0:
            raise RuntimeError('RangeSeparationJKBuilder cannot be used to evaluate '
                               'the long-range HF exchange in RSH functional')
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = pbctools.get_coulG(cell, kpt, False, self, mesh, Gv,
                                   omega=self.omega)
        coulG *= kws
        return coulG

    def weighted_coulG_SR(self, kpt=np.zeros(3), exx=False, mesh=None):
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if cell.omega != 0:
            raise RuntimeError('RangeSeparationJKBuilder cannot be used to evaluate '
                               'the long-range HF exchange in RSH functional')
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = pbctools.get_coulG(cell, kpt, False, self, mesh, Gv,
                                   omega=-self.omega)
        coulG *= kws
        return coulG

    def ft_loop(self, mesh=None, q=np.zeros(3), kpts=None, shls_slice=None,
                max_memory=4000, aosym='s1', intor='GTO_ft_ovlp', comp=1):
        for dat, p0, p1 in aft.AFTDF.ft_loop(self, mesh, q, kpts, shls_slice,
                                             max_memory, aosym, intor, comp,
                                             bvk_kmesh=self.bvk_kmesh):
            yield dat, p0, p1

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

        if kpts is None:
            if np.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = np.zeros(3)
            else:
                kpts = self.kpts
        kpts = np.asarray(kpts)

        is_single_kpt = kpts.ndim == 1
        if is_single_kpt:
            kpts = kpts.reshape(1,3)

        vj = vk = None
        if with_k:
            vk = self.get_k_kpts(dm, hermi, kpts, kpts_band, exxdiv)
            if is_single_kpt:
                vk = vk[...,0,:,:]
        if with_j:
            vj = self.get_j_kpts(dm, hermi, kpts, kpts_band)
            if is_single_kpt:
                vj = vj[...,0,:,:]
        return vj, vk

    def get_j_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        '''
        C ~ compact basis, D ~ diffused basis

        Compute J matrix with coulG_LR:
        (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)

        Compute J matrix with full coulG:
        (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
        '''
        if kpts_band is not None:
            return self.get_j_for_bands(dm_kpts, hermi, kpts, kpts_band)

        if len(kpts) == 1 and not is_zero(kpts):
            raise NotImplementedError('Single k-point get-j')

        cell = self.cell
        dm_kpts = lib.asarray(dm_kpts, order='C')
        dms = _format_dms(dm_kpts, kpts)
        n_dm, nkpts, nao = dms.shape[:3]

        n_diffused = cell._nbas_each_set[2]
        nao_compact = cell.ao_loc[cell.nbas-n_diffused]

        vj_kpts = np.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)
        kpt_allow = np.zeros(3)
        mesh = self.mesh
        coulG = self.weighted_coulG(kpt_allow, False, mesh)
        coulG_LR = self.weighted_coulG_LR(kpt_allow, False, mesh)
        coulG_SR = coulG - coulG_LR
        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
        weight = 1./len(kpts)
        for aoaoks, p0, p1 in self.ft_loop(mesh, kpt_allow, kpts, max_memory=max_memory):
            if nao_compact < nao:
                aoaoks = [aoao.reshape(-1,nao,nao) for aoao in aoaoks]
                aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG[p0:p1], weight)
                for aoao in aoaoks:
                    aoao[:,nao_compact:,nao_compact:] = 0
                aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG_SR[p0:p1], -weight)
            else:
                aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG_LR[p0:p1], weight)
            aoao = aoaoks = p0 = p1 = None

        # G=0 contribution, associated to 2e integrals in real-space
        if cell.dimension >= 2:
            ovlp = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
            if nao_compact < nao:
                ovlp[:,nao_compact:,nao_compact:] = 0
            kws = cell.get_Gv_weights(mesh)[2]
            G0_weight = kws[0] if isinstance(kws, np.ndarray) else kws
            vj_G0 = lib.einsum('kpq,nkqp,lrs->nlrs', ovlp, dm_kpts, ovlp)
            vj_kpts -= np.pi/self.omega**2 * weight * G0_weight * vj_G0

        if gamma_point(kpts):
            vj_kpts = vj_kpts.real.copy()
        return _format_jks(vj_kpts, dm_kpts, kpts_band, kpts)

    def get_j_for_bands(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        log = logger.Logger(self.stdout, self.verbose)
        t1 = logger.process_clock(), logger.perf_counter()

        cell = self.cell
        dm_kpts = lib.asarray(dm_kpts, order='C')
        dms = _format_dms(dm_kpts, kpts)
        n_dm, nkpts, nao = dms.shape[:3]

        n_diffused = cell._nbas_each_set[2]
        nao_compact = cell.ao_loc[cell.nbas-n_diffused]

        kpt_allow = np.zeros(3)
        mesh = self.mesh
        coulG = self.weighted_coulG(kpt_allow, False, mesh)
        coulG_LR = self.weighted_coulG_LR(kpt_allow, False, mesh)
        coulG_SR = coulG - coulG_LR
        ngrids = len(coulG)
        vG = np.zeros((n_dm,ngrids), dtype=np.complex128)
        vG_SR = np.zeros((n_dm,ngrids), dtype=np.complex128)
        max_memory = (self.max_memory - lib.current_memory()[0]) * .8

        for aoaoks, p0, p1 in self.ft_loop(mesh, kpt_allow, kpts, max_memory=max_memory):
            #:rho = np.einsum('lkL,lk->L', pqk.conj(), dm)
            for k, aoao in enumerate(aoaoks):
                aoao = aoao.reshape(-1,nao,nao)
                if nao_compact < nao:
                    for i in range(n_dm):
                        rho = np.einsum('ij,Lji->L', dms[i,k], aoao.conj())
                        vG[i,p0:p1] += rho * coulG[p0:p1]
                    aoao[:,nao_compact:,nao_compact:] = 0
                    for i in range(n_dm):
                        rho = np.einsum('ij,Lji->L', dms[i,k], aoao.conj())
                        vG_SR[i,p0:p1] += rho * coulG_SR[p0:p1]
                else:
                    for i in range(n_dm):
                        rho = np.einsum('ij,Lji->L', dms[i,k], aoao.conj())
                        vG[i,p0:p1] += rho * coulG_LR[p0:p1]
            aoao = aoaoks = p0 = p1 = None
        weight = 1./len(kpts)
        vG *= weight
        vG_SR *= weight
        t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

        kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
        nband = len(kpts_band)
        vj_kpts = np.zeros((n_dm,nband,nao,nao), dtype=np.complex128)
        for aoaoks, p0, p1 in self.ft_loop(mesh, kpt_allow, kpts_band,
                                            max_memory=max_memory):
            for k, aoao in enumerate(aoaoks):
                aoao = aoao.reshape(-1,nao,nao)
                if nao_compact < nao:
                    for i in range(n_dm):
                        vj_kpts[i,k] += np.einsum('L,Lij->ij', vG[i,p0:p1], aoao)
                    aoao[:,nao_compact:,nao_compact:] = 0
                    for i in range(n_dm):
                        vj_kpts[i,k] -= np.einsum('L,Lij->ij', vG_SR[i,p0:p1], aoao)
                else:
                    for i in range(n_dm):
                        vj_kpts[i,k] += np.einsum('L,Lij->ij', vG[i,p0:p1], aoao)
            aoao = aoaoks = p0 = p1 = None

        # G=0 contribution, associated to 2e integrals in real-space
        if cell.dimension >= 2:
            ovlp = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
            ovlp[:,nao_compact:,nao_compact:] = 0
            ovlp_b = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts_band))
            ovlp_b[:,nao_compact:,nao_compact:] = 0
            kws = cell.get_Gv_weights(mesh)[2]
            G0_weight = kws[0] if isinstance(kws, np.ndarray) else kws
            vj_G0 = lib.einsum('kpq,nkqp,lrs->nlrs', ovlp, dm_kpts, ovlp_b)
            vj_kpts -= np.pi/self.omega**2 * weight * G0_weight * vj_G0

        if gamma_point(kpts_band):
            vj_kpts = vj_kpts.real.copy()
        t1 = log.timer_debug1('get_j pass 2', *t1)
        return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

    def get_k_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
                   exxdiv=None):
        '''
        C ~ compact basis, D ~ diffused basis

        Compute K matrix with coulG_LR:
        (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)

        Compute K matrix with full coulG:
        (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
        '''
        cell = self.cell
        log = logger.Logger(self.stdout, self.verbose)
        t1 = logger.process_clock(), logger.perf_counter()

        mesh = self.mesh
        dm_kpts = lib.asarray(dm_kpts, order='C')
        dms = _format_dms(dm_kpts, kpts)
        nset, nkpts, nao = dms.shape[:3]

        swap_2e = (kpts_band is None)
        kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
        nband = len(kpts_band)
        kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
        kk_todo = np.ones(kk_table.shape[:2], dtype=bool)
        vkR = np.zeros((nset,nband,nao,nao))
        vkI = np.zeros((nset,nband,nao,nao))
        dmsR = np.asarray(dms.real, order='C')
        dmsI = np.asarray(dms.imag, order='C')
        weight = 1. / nkpts

        n_diffused = cell._nbas_each_set[2]
        nao_compact = cell.ao_loc[cell.nbas-n_diffused]

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now)) * .8
        log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
        # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
        def make_kpt(kpt):  # kpt = kptj - kpti
            # search for all possible ki and kj that has ki-kj+kpt=0
            kk_match = np.einsum('ijx->ij', abs(kk_table + kpt)) < 1e-9
            kpti_idx, kptj_idx = np.where(kk_todo & kk_match)
            nkptj = len(kptj_idx)
            log.debug1('kpt = %s', kpt)
            log.debug2('kpti_idx = %s', kpti_idx)
            log.debug2('kptj_idx = %s', kptj_idx)
            kk_todo[kpti_idx,kptj_idx] = False
            if swap_2e and not is_zero(kpt):
                kk_todo[kptj_idx,kpti_idx] = False

            max_memory1 = max_memory * (nkptj+1)/(nkptj+5)
            #blksize = max(int(max_memory1*4e6/(nkptj+5)/16/nao**2), 16)

            #bufR = np.empty((blksize*nao**2))
            #bufI = np.empty((blksize*nao**2))
            # Use DF object to mimic KRHF/KUHF object in function get_coulG
            vkcoulG = self.weighted_coulG(kpt, exxdiv, mesh)
            coulG_SR = self.weighted_coulG_SR(kpt, False, mesh)
            coulG_LR = vkcoulG - coulG_SR
            kptjs = kpts[kptj_idx]
            perm_sym = swap_2e and not is_zero(kpt)
            for aoaoks, p0, p1 in self.ft_loop(mesh, kpt, kptjs, max_memory=max_memory1):
                if nao_compact < nao:
                    aoaoks = [aoao.reshape(-1,nao,nao) for aoao in aoaoks]
                    aft_jk._update_vk_((vkR, vkI), aoaoks, (dmsR, dmsI), vkcoulG[p0:p1],
                                       weight, kpti_idx, kptj_idx, perm_sym)
                    for aoao in aoaoks:
                        aoao[:,nao_compact:,nao_compact:] = 0
                    aft_jk._update_vk_((vkR, vkI), aoaoks, (dmsR, dmsI), coulG_SR[p0:p1],
                                       -weight, kpti_idx, kptj_idx, perm_sym)
                else:
                    aft_jk._update_vk_((vkR, vkI), aoaoks, (dmsR, dmsI), coulG_LR[p0:p1],
                                       weight, kpti_idx, kptj_idx, perm_sym)

        for ki, kpti in enumerate(kpts_band):
            for kj, kptj in enumerate(kpts):
                if kk_todo[ki,kj]:
                    make_kpt(kptj-kpti)
            t1 = log.timer_debug1('get_k_kpts: make_kpt (%d,*)'%ki, *t1)

        if (gamma_point(kpts) and gamma_point(kpts_band) and
            not np.iscomplexobj(dm_kpts)):
            vk_kpts = vkR
        else:
            vk_kpts = vkR + vkI * 1j

        # G=0 associated to 2e integrals in real-space
        if cell.dimension >= 2:
            ovlp = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
            ovlp[:,nao_compact:,nao_compact:] = 0
            kws = cell.get_Gv_weights(mesh)[2]
            G0_weight = kws[0] if isinstance(kws, np.ndarray) else kws
            vk_G0 = lib.einsum('kpq,nkqr,krs->nkps', ovlp, dm_kpts, ovlp)
            vk_kpts -= np.pi/self.omega**2 * weight * G0_weight * vk_G0

        # Add ewald_exxdiv contribution because G=0 was not included in the
        # non-uniform grids
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            _ewald_exxdiv_for_G0(cell, kpts_band, dms, vk_kpts, kpts_band)

        return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

if __name__ == '__main__':
    from pyscf.pbc.gto import Cell
    cells = []

    cell = Cell()
    cell.a = np.eye(3)*1.8
    cell.atom = '''#He     0.      0.      0.
                   He     0.4917  0.4917  0.4917'''
    cell.basis = {'He': [[0, [2.5, 1]]]}
    cell.build()
    cells.append(cell)

    if 1:
        cell = Cell()
        cell.a = np.eye(3)*2.4
        cell.atom = '''He     0.      0.      0.
                       He     0.4917  0.4917  0.4917'''
        cell.basis = {'He': [[0, [4.1, 1, -.2],
                                 [0.5, .2, .5],
                                 [0.15, .5, .5]],
                             #[1, [1.5, 1]],
                             [1, [0.3, 1]],]}
        cell.build()
        cell.verbose = 6
        cells.append(cell)

    if 1:
        cell = Cell().build(a = '''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329
                            ''',
                            unit = 'Ang',
                            atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391''',
                            basis='''
C S
4.3362376436      0.1490797872
1.2881838513      -0.0292640031
0.4037767149      -0.688204051
0.1187877657      -0.3964426906
C P
4.3362376436      -0.0878123619
1.2881838513      -0.27755603
0.4037767149      -0.4712295093
0.1187877657      -0.4058039291
''')
        cell.verbose = 6
        cells.append(cell)

    for cell in cells:
        kpts = cell.make_kpts([3,1,1])
        mf = cell.KRHF(kpts=kpts)#.run()
        #dm = mf.make_rdm1()
        np.random.seed(1)
        dm = (np.random.rand(len(kpts), cell.nao, cell.nao) +
              np.random.rand(len(kpts), cell.nao, cell.nao) * 1j)
        dm = dm + dm.transpose(0,2,1).conj()
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        phase = k2gamma.get_phase(cell, kpts, kmesh)[1]
        dm = lib.einsum('Rk,kuv,Sk->RSuv', phase.conj().T, dm, phase.T)
        dm = lib.einsum('Rk,RSuv,Sk->kuv', phase, dm.real, phase.conj())

        jref, kref = mf.get_jk(cell, dm, kpts=kpts)
        ej = np.einsum('kij,kji->', jref, dm)
        ek = np.einsum('kij,kji->', kref, dm) * .5

        jk_builder = RangeSeparationJKBuilder(cell, kpts)
        jk_builder.build(omega=0.5)
        #jk_builder.mesh = [6,6,6]
        #print(jk_builder.omega, jk_builder.mesh)
        vj, vk = jk_builder.get_jk(dm, kpts=kpts, exxdiv=mf.exxdiv)
        print(abs(vj - jref).max())
        print(abs(vk - kref).max())
        print('ej_ref', ej, 'ek_ref', ek)
        print('ej', np.einsum('kij,kji->', vj, dm).real,
              'ek', np.einsum('kij,kji->', vk, dm).real * .5)
