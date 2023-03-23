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
Build GDF tensor with range-separation technique.

rsdf.py is another version of RS-GDF using a different algorithm to compute
3c2e integrals. Note both modules CANNOT handle the long-range operator
erf(omega*r12)/r12. It has to be computed with the gdf_builder module.

Ref:
    Q. Sun, arXiv:2012.07929
'''

import os
import ctypes
import tempfile
import numpy as np
import scipy.linalg
import h5py
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger, zdotCN
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.gto import ANG_OF
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df import aft
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.incore import libpbc, _Int3cBuilder
from pyscf.pbc.lib.kpts_helper import (is_zero, member, unique_with_wrap_around,
                                       group_by_conj_pairs)
from pyscf import __config__

LINEAR_DEP_THR = getattr(__config__, 'pbc_df_df_DF_lindep', 1e-9)
# Threshold of steep bases and local bases
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.0)


class _RSGDFBuilder(_Int3cBuilder):
    '''
    Use the range-separated algorithm to build Gaussian density fitting 3-center tensor
    '''

    # In real-space 3c2e integrals exclude smooth-smooth block (*|DD)
    fft_dd_block = True
    # In real-space 3c2e integrals exclude smooth auxiliary basis (D|**)
    exclude_d_aux = True
    # If both exclude_d_aux and fft_dd_block are enabled,
    # evaluate only the (C|CC), (C|CD), (C|DC) blocks.

    # set True to force calculating j2c^(-1/2) using eigenvalue
    # decomposition (ED); otherwise, Cholesky decomposition (CD) is used
    # first, and ED is called only if CD fails.
    j2c_eig_always = False
    linear_dep_threshold = LINEAR_DEP_THR

    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        self.mesh = None
        if cell.omega != 0:
            # Initialize omega to cell.omega for HF exchange of short range
            # int2e in RSH functionals
            self.omega = abs(cell.omega)
        else:
            self.omega = None
        self.rs_auxcell = None

        _Int3cBuilder.__init__(self, cell, auxcell, kpts)

    @property
    def exclude_dd_block(self):
        cell = self.cell
        return (self.fft_dd_block and
                cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum')

    def has_long_range(self):
        '''Whether to add the long-range part computed with AFT integrals'''
        # If self.exclude_d_aux is set, the block (D|**) will not be computed in
        # outcore_auxe2. It has to be computed by AFT code.
        return self.omega is None or abs(self.cell.omega) < self.omega or self.exclude_d_aux

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
        logger.info(self, 'omega = %s', self.omega)
        logger.info(self, 'exclude_d_aux = %s', self.exclude_d_aux)
        logger.info(self, 'exclude_dd_block = %s', self.exclude_dd_block)
        logger.info(self, 'j2c_eig_always = %s', self.j2c_eig_always)
        logger.info(self, 'has_long_range = %s', self.has_long_range())
        return self

    def build(self, omega=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        auxcell = self.auxcell
        kpts = self.kpts

        if self.cell.dimension == 0:
            log.warn('_RSGDFBuilder for cell.dimension=0 may have larger error '
                     'than _CCGDFBuilder')

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if omega is not None:
            self.omega = omega

        if self.omega is None:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(auxcell, kpts, self.mesh)
        elif self.mesh is None:
            self.ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, self.omega)
            mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), self.ke_cutoff)
            self.mesh = _round_off_to_odd_mesh(mesh)
        elif self.ke_cutoff is None:
            ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            self.ke_cutoff = ke_cutoff[:cell.dimension].min()

        self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)
        self.rs_auxcell = rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(
            auxcell, self.ke_cutoff, verbose=log)

        # Estimate rcut to generate Ls. rcut (and the translation vectors Ls)
        # here needs to cover all possible shells to converge int3c2e.
        # cell.rcut cannot be used here because it only converge the GTOval.
        smooth_bas_mask = rs_cell.bas_type == ft_ao.SMOOTH_BASIS
        cell_exps = rs_cell.bas_exps()
        aux_exps = rs_auxcell.bas_exps()
        exps_d = [cell_exps[ib] for ib in range(rs_cell.nbas) if smooth_bas_mask[ib]]
        exps_c = [cell_exps[ib] for ib in range(rs_cell.nbas) if not smooth_bas_mask[ib]]

        if self.exclude_d_aux:
            compact_aux_idx = np.where(rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS)[0]
            if len(compact_aux_idx) > 0:
                exp_aux_min = np.hstack([aux_exps[ib] for ib in compact_aux_idx]).min()
            else:
                exp_aux_min = np.hstack(aux_exps).max()
        else:
            exp_aux_min = np.hstack(aux_exps).min()

        if not exps_c: # Only smooth functions
            rcut_sr = cell.rcut
        else:
            # Estimation with the assumption self.exclude_dd_block = True
            # Is rcut enough if exclude_dd_block = False?
            if not exps_d:  # Only compact functions
                exp_d_min = exp_c_min = np.hstack(exps_c).min()
                aij = exp_c_min * 2
                eij = exp_c_min / 2
            else:  # both smooth and compact functions exist
                exp_d_min = np.hstack(exps_d).min()
                exp_c_min = np.hstack(exps_c).min()
                aij = exp_d_min + exp_c_min
                eij = exp_d_min * exp_c_min / aij
            theta = 1/(self.omega**-2 + 1./aij + 1./exp_aux_min)
            fac = ((8*np.pi*exp_d_min*exp_c_min/(aij*exp_aux_min)**2)**.75
                   / (theta * np.pi)**.5)
            # x = rcut * x_ratio for the distance between compact function
            # and smooth function (smooth function in the far end)
            # fac*erfc(\sqrt(theta)|rcut - x|) for the asymptotic value of short-range eri
            x_ratio = 1. / (exp_c_min/aij + exp_d_min/theta)
            exp_fac = eij * x_ratio**2 + theta * (1 - exp_c_min/aij*x_ratio)**2

            rcut_sr = cell.rcut  # initial guess
            rcut_sr = ((-np.log(cell.precision
                                * rcut_sr / (2*np.pi*fac)) / exp_fac)**.5
                       + pbcgto.cell._rcut_penalty(cell))
            log.debug1('exp_d_min = %g, exp_c_min = %g, exp_aux_min = %g, rcut_sr = %g',
                       exp_d_min, exp_c_min, exp_aux_min, rcut_sr)

        supmol = _ExtendedMoleSR.from_cell(rs_cell, kmesh, self.omega, rcut_sr, log)
        self.supmol = _strip_basis(supmol, self.omega, exp_aux_min, self.exclude_dd_block)
        log.timer_debug1('initializing supmol', *cpu0)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())
        return self

    weighted_coulG = aft.weighted_coulG

    def weighted_coulG_LR(self, kpt=np.zeros(3), exx=False, mesh=None):
        # The long range part Coulomb kernel has to be computed as the
        # difference between coulG(cell.omega) - coulG(df.omega). It allows this
        # module to handle the SR- and regular integrals in the same framework
        return (self.weighted_coulG(kpt, exx, mesh) -
                self.weighted_coulG_SR(kpt, exx, mesh))

    def weighted_coulG_SR(self, kpt=np.zeros(3), exx=False, mesh=None):
        return self.weighted_coulG(kpt, False, mesh, -self.omega)

    def get_q_cond(self):
        '''Integral screening condition max(sqrt((ij|ij))) inside the supmol'''
        supmol = self.supmol
        intor = 'int2e_sph'
        cintopt = lib.c_null_ptr()
        nbas = supmol.nbas
        q_cond = np.empty((nbas, nbas))
        with supmol.with_integral_screen(supmol.precision**2):
            ao_loc = gto.moleintor.make_loc(supmol._bas, intor)
            libpbc.CVHFset_int2e_q_cond(
                getattr(libpbc, intor), cintopt,
                q_cond.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
                supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
                supmol._env.ctypes.data_as(ctypes.c_void_p))

        # Remove d-d block in supmol q_cond
        if self.exclude_dd_block:
            smooth_idx = supmol.bas_type_to_indices(ft_ao.SMOOTH_BASIS)
            q_cond[smooth_idx[:,None], smooth_idx] = 1e-200
        return q_cond

    def get_q_cond_aux(self, auxcell=None, supmol=None):
        '''max(sqrt((k|ii))) between the auxcell and the supmol'''
        if supmol is None:
            supmol = self.supmol
        auxcell_s = self.rs_auxcell.copy()
        auxcell_s._bas[:,ANG_OF] = 0
        intor = 'int3c2e_sph'
        cintopt = lib.c_null_ptr()
        nbas = supmol.nbas
        q_cond_aux = np.empty((auxcell_s.nbas, nbas))
        with supmol.with_integral_screen(supmol.precision**2):
            atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                         auxcell_s._atm, auxcell_s._bas, auxcell_s._env)
            ao_loc = gto.moleintor.make_loc(bas, intor)
            shls_slice = (0, supmol.nbas, supmol.nbas, len(bas))
            libpbc.PBC_nr3c_q_cond(
                getattr(libpbc, intor), cintopt,
                q_cond_aux.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
                env.ctypes.data_as(ctypes.c_void_p))

        if self.exclude_d_aux:
            # Assign a very small value to q_cond_aux to avoid dividing 0 error
            q_cond_aux[self.rs_auxcell.bas_type == ft_ao.SMOOTH_BASIS] = 1e-200
        return q_cond_aux

    def get_bas_map(self, auxcell=None, supmol=None):
        '''bas_map is to assign each basis of supmol._bas the index in
        [bvk_cell-id, bas-id, image-id]
        '''
        if supmol is None:
            supmol = self.supmol
        if self.exclude_d_aux:
            # Use aux_mask to skip smooth auxiliary basis and handle them in AFT part.
            aux_mask = (self.rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS).astype(np.int32)
        else:
            aux_mask = np.ones(self.rs_auxcell.nbas, dtype=np.int32)

        # Append aux_mask to bas_map as a temporary solution for function
        # _assemble3c in fill_ints.c
        bas_map = np.where(supmol.bas_mask.ravel())[0].astype(np.int32)
        bas_map = np.asarray(np.append(bas_map, aux_mask), dtype=np.int32)
        return bas_map

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        if supmol is None:
            supmol = self.supmol
        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        nbasp = self.cell.nbas  # The number of shells in the primitive cell
        ovlp_mask = self.get_q_cond() > cutoff
        bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, supmol.sh_loc)
        cell0_ovlp_mask = bvk_ovlp_mask.reshape(
            bvk_ncells, nbasp, bvk_ncells, nbasp).any(axis=2).any(axis=0)
        ovlp_mask = ovlp_mask.astype(np.int8)
        cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)
        return ovlp_mask, cell0_ovlp_mask

    def decompose_j2c(self, j2c):
        j2c = np.asarray(j2c)
        if self.j2c_eig_always:
            return self.eigenvalue_decomposed_metric(j2c)
        else:
            return self.cholesky_decomposed_metric(j2c)

    def cholesky_decomposed_metric(self, j2c):
        try:
            j2c_negative = None
            j2ctag = 'CD'
            j2c = scipy.linalg.cholesky(j2c, lower=True)
        except scipy.linalg.LinAlgError:
            j2c, j2c_negative, j2ctag = self.eigenvalue_decomposed_metric(j2c)
        return j2c, j2c_negative, j2ctag

    def eigenvalue_decomposed_metric(self, j2c):
        cell = self.cell
        j2c_negative = None
        w, v = scipy.linalg.eigh(j2c)
        logger.debug(self, 'cond = %.4g, drop %d bfns',
                     w[-1]/w[0], np.count_nonzero(w<self.linear_dep_threshold))
        v1 = v[:,w>self.linear_dep_threshold].conj().T
        v1 /= np.sqrt(w[w>self.linear_dep_threshold]).reshape(-1,1)
        j2c = v1
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            idx = np.where(w < -self.linear_dep_threshold)[0]
            if len(idx) > 0:
                j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
        j2ctag = 'ED'
        return j2c, j2c_negative, j2ctag

    def get_2c2e(self, uniq_kpts):
        # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
        auxcell = self.auxcell
        if not self.has_long_range():
            omega = auxcell.omega
            with lib.temporary_env(auxcell):
                j2c = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
            if auxcell.dimension == 3 and auxcell.low_dim_ft_type != 'inf_vacuum':
                gamma_point_idx = member(np.zeros(3), uniq_kpts)
                if len(gamma_point_idx) > 0:
                    # Add G=0 contribution
                    g0_fac = np.pi / omega**2 / auxcell.vol
                    aux_chg = _gaussian_int(auxcell)
                    j2c[gamma_point_idx[0]] -= g0_fac * aux_chg[:,None] * aux_chg
            return j2c

        precision = auxcell.precision**2
        omega = self.omega
        rs_auxcell = self.rs_auxcell
        auxcell_c = rs_auxcell.compact_basis_cell()
        if auxcell_c.nbas > 0:
            rcut_sr = (-np.log(precision * auxcell_c.rcut**2 * omega))**.5 / omega
            auxcell_c.rcut = rcut_sr
            logger.debug1(self, 'auxcell_c  rcut_sr = %g', rcut_sr)
            with auxcell_c.with_short_range_coulomb(omega):
                sr_j2c = list(auxcell_c.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts))
            recontract_1d = rs_auxcell.recontract()

            compact_bas_idx = np.where(rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS)[0]
            compact_ao_idx = rs_auxcell.get_ao_indices(compact_bas_idx)
            ao_map = auxcell.get_ao_indices(rs_auxcell.bas_map[compact_bas_idx])

            def recontract_2d(j2c, j2c_cc):
                return lib.takebak_2d(j2c, j2c_cc, ao_map, ao_map, thread_safe=False)
        else:
            sr_j2c = None

        # 2c2e integrals the metric can easily cause errors in cderi tensor.
        # self.mesh may not be enough to produce required accuracy.
        # mesh = self.mesh
        ke = aft.estimate_ke_cutoff_for_omega(auxcell, omega, precision)
        mesh = pbctools.cutoff_to_mesh(auxcell.lattice_vectors(), ke)
        if auxcell.dimension < 2 or auxcell.low_dim_ft_type == 'inf_vacuum':
            mesh[auxcell.dimension:] = self.mesh[auxcell.dimension:]
        mesh = self.cell.symmetrize_mesh(mesh)
        logger.debug(self, 'Set 2c2e integrals precision %g, mesh %s', precision, mesh)

        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        b = auxcell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]
        naux_rs = rs_auxcell.nao
        naux = auxcell.nao
        max_memory = max(1000, self.max_memory - lib.current_memory()[0])
        blksize = min(ngrids, int(max_memory*.4e6/16/naux_rs), 200000)
        logger.debug2(self, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)
        j2c = []
        for k, kpt in enumerate(uniq_kpts):
            coulG = self.weighted_coulG(kpt, False, mesh)
            if is_zero(kpt):  # kpti == kptj
                j2c_k = np.zeros((naux, naux))
            else:
                j2c_k = np.zeros((naux, naux), dtype=np.complex128)

            if sr_j2c is None:
                for p0, p1 in lib.prange(0, ngrids, blksize):
                    auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                    if is_zero(kpt):  # kpti == kptj
                        j2c_k += lib.dot(auxG.conj() * coulG[p0:p1], auxG.T).real
                    else:
                        #j2cR, j2cI = zdotCN(LkR*coulG[p0:p1],
                        #                    LkI*coulG[p0:p1], LkR.T, LkI.T)
                        j2c_k += lib.dot(auxG.conj() * coulG[p0:p1], auxG.T)
                    auxG = None
            else:
                # coulG_sr here to first remove the FT-SR-2c2e for compact basis
                # from the analytical 2c2e integrals. The FT-SR-2c2e for compact
                # basis is added back in j2c_k.
                coulG_sr = self.weighted_coulG_SR(kpt, False, mesh)
                if auxcell.dimension == 3 and is_zero(kpt):
                    G0_idx = 0  # due to np.fft.fftfreq convention
                    G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
                    coulG_sr[G0_idx] += np.pi/omega**2 * G0_weight

                for p0, p1 in lib.prange(0, ngrids, blksize):
                    auxG = ft_ao.ft_ao(rs_auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                    auxG_sr = auxG[compact_ao_idx]
                    if is_zero(kpt):
                        sr_j2c[k] -= lib.dot(auxG_sr.conj() * coulG_sr[p0:p1], auxG_sr.T).real
                    else:
                        sr_j2c[k] -= lib.dot(auxG_sr.conj() * coulG_sr[p0:p1], auxG_sr.T)
                    auxG = recontract_1d(auxG)
                    if is_zero(kpt):  # kpti == kptj
                        j2c_k += lib.dot(auxG.conj() * coulG[p0:p1], auxG.T).real
                    else:
                        j2c_k += lib.dot(auxG.conj() * coulG[p0:p1], auxG.T)
                    auxG = auxG_sr = None

                j2c_k = recontract_2d(j2c_k, sr_j2c[k])
                sr_j2c[k] = None

            j2c.append(j2c_k)
        return j2c

    def outcore_auxe2(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                      j_only=False, dataname='j3c', shls_slice=None,
                      fft_dd_block=None):
        r'''The SR part of 3-center integrals (ij|L) with double lattice sum.

        Kwargs:
            shls_slice :
                Indicate the shell slices in the primitive cell
        '''
        # The ideal way to hold the temporary integrals is to store them in the
        # cderi_file and overwrite them inplace in the second pass.  The current
        # HDF5 library does not have an efficient way to manage free space in
        # overwriting.  It often leads to the cderi_file ~2 times larger than the
        # necessary size.  For now, dumping the DF integral intermediates to a
        # separated temporary file can avoid this issue.  The DF intermediates may
        # be terribly huge. The temporary file should be placed in the same disk
        # as cderi_file.
        swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
        fswap = lib.H5TmpFile(swapfile.name)
        # Unlink swapfile to avoid trash files
        swapfile = None

        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        auxcell = self.auxcell
        naux = auxcell.nao
        kpts = self.kpts
        nkpts = kpts.shape[0]

        gamma_point_only = is_zero(kpts)
        if gamma_point_only:
            assert nkpts == 1
            j_only = True

        intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

        if fft_dd_block is None:
            fft_dd_block = self.exclude_dd_block

        if fft_dd_block:
            self._outcore_dd_block(fswap, intor, aosym, comp, j_only,
                                   dataname, shls_slice)

        # int3c may be the regular int3c2e, LR-int3c2e or SR-int3c2e, depending
        # on how self.supmol is initialized
        int3c = self.gen_int3c_kernel(intor, aosym, comp, j_only,
                                      rs_auxcell=self.rs_auxcell)

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

        ao_loc = cell.ao_loc
        aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
        ish0, ish1, jsh0, jsh1, ksh0, ksh1 = shls_slice
        i0, i1, j0, j1 = ao_loc[list(shls_slice[:4])]
        k0, k1 = aux_loc[[ksh0, ksh1]]
        if aosym == 's1':
            nao_pair = (i1 - i0) * (j1 - j0)
        else:
            nao_pair = i1*(i1+1)//2 - i0*(i0+1)//2
        naux = k1 - k0

        if fft_dd_block and np.any(rs_cell.bas_type == ft_ao.SMOOTH_BASIS):
            merge_dd = rs_cell.merge_diffused_block(aosym)
        else:
            merge_dd = None

        # TODO: shape = (comp, nao_pair, naux)
        shape = (nao_pair, naux)
        if j_only or nkpts == 1:
            for k in range(nkpts):
                fswap.create_dataset(f'{dataname}R/{k*nkpts+k}', shape, 'f8')
                # exclude imaginary part for gamma point
                if not is_zero(kpts[k]):
                    fswap.create_dataset(f'{dataname}I/{k*nkpts+k}', shape, 'f8')
            nkpts_ij = nkpts
            kikj_idx = [k*nkpts+k for k in range(nkpts)]
        else:
            for ki in range(nkpts):
                for kj in range(nkpts):
                    fswap.create_dataset(f'{dataname}R/{ki*nkpts+kj}', shape, 'f8')
                    fswap.create_dataset(f'{dataname}I/{ki*nkpts+kj}', shape, 'f8')
                # exclude imaginary part for gamma point
                if is_zero(kpts[ki]):
                    del fswap[f'{dataname}I/{ki*nkpts+ki}']
            nkpts_ij = nkpts * nkpts
            kikj_idx = range(nkpts_ij)
            if merge_dd:
                uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
                    cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
                kpt_ij_pairs = group_by_conj_pairs(cell, uniq_kpts)[0]

        if naux == 0:
            return fswap

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, self.max_memory-mem_now)

        # split the 3-center tensor (nkpts_ij, i, j, aux) along shell i.
        # plus 1 to ensure the intermediates in libpbc do not overflow
        buflen = min(max(int(max_memory*.9e6/16/naux/(nkpts_ij+1)), 1), nao_pair)
        # lower triangle part
        sh_ranges = _guess_shell_ranges(cell, buflen, aosym, start=ish0, stop=ish1)
        max_buflen = max([x[2] for x in sh_ranges])
        if max_buflen > buflen:
            log.warn('memory usage of outcore_auxe2 may be %.2f times over max_memory',
                     (max_buflen/buflen - 1))

        bufR = np.empty((nkpts_ij, comp, max_buflen, naux))
        bufI = np.empty_like(bufR)
        cpu0 = logger.process_clock(), logger.perf_counter()
        nsteps = len(sh_ranges)
        row1 = 0
        for istep, (sh_start, sh_end, nrow) in enumerate(sh_ranges):
            if aosym == 's2':
                shls_slice = (sh_start, sh_end, jsh0, sh_end, ksh0, ksh1)
            else:
                shls_slice = (sh_start, sh_end, jsh0, jsh1, ksh0, ksh1)
            outR, outI = int3c(shls_slice, bufR, bufI)
            log.debug2('      step [%d/%d], shell range [%d:%d], len(buf) = %d',
                       istep+1, nsteps, sh_start, sh_end, nrow)
            cpu0 = log.timer_debug1(f'outcore_auxe2 [{istep+1}/{nsteps}]', *cpu0)

            shls_slice = (sh_start, sh_end, 0, cell.nbas)
            row0, row1 = row1, row1 + nrow
            if merge_dd is not None:
                if gamma_point_only:
                    merge_dd(outR[0], fswap[f'{dataname}R-dd/0'], shls_slice)
                elif j_only or nkpts == 1:
                    for k in range(nkpts):
                        merge_dd(outR[k], fswap[f'{dataname}R-dd/{k*nkpts+k}'], shls_slice)
                        merge_dd(outI[k], fswap[f'{dataname}I-dd/{k*nkpts+k}'], shls_slice)
                else:
                    for k, k_conj in kpt_ij_pairs:
                        kpt_ij_idx = np.where(uniq_inverse == k)[0]
                        if k_conj is None or k == k_conj:
                            for ij_idx in kpt_ij_idx:
                                merge_dd(outR[ij_idx], fswap[f'{dataname}R-dd/{ij_idx}'], shls_slice)
                                merge_dd(outI[ij_idx], fswap[f'{dataname}I-dd/{ij_idx}'], shls_slice)
                        else:
                            ki_lst = kpt_ij_idx // nkpts
                            kj_lst = kpt_ij_idx % nkpts
                            kpt_ji_idx = kj_lst * nkpts + ki_lst
                            for ij_idx, ji_idx in zip(kpt_ij_idx, kpt_ji_idx):
                                j3cR_dd = np.asarray(fswap[f'{dataname}R-dd/{ij_idx}'])
                                merge_dd(outR[ij_idx], j3cR_dd, shls_slice)
                                merge_dd(outR[ji_idx], j3cR_dd.transpose(1,0,2), shls_slice)
                                j3cI_dd = np.asarray(fswap[f'{dataname}I-dd/{ij_idx}'])
                                merge_dd(outI[ij_idx], j3cI_dd, shls_slice)
                                merge_dd(outI[ji_idx],-j3cI_dd.transpose(1,0,2), shls_slice)

            for k, kk_idx in enumerate(kikj_idx):
                fswap[f'{dataname}R/{kk_idx}'][row0:row1] = outR[k]
                if f'{dataname}I/{kk_idx}' in fswap:
                    fswap[f'{dataname}I/{kk_idx}'][row0:row1] = outI[k]
            outR = outI = None
        bufR = bufI = None
        return fswap

    def _outcore_dd_block(self, h5group, intor='int3c2e', aosym='s2', comp=None,
                          j_only=False, dataname='j3c', shls_slice=None):
        '''
        The block of smooth AO basis in i and j of (ij|L) with full Coulomb kernel
        '''
        if intor not in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart'):
            raise NotImplementedError

        if shls_slice is not None:
            raise NotImplementedError

        log = logger.new_logger(self)
        cell = self.cell
        cell_d = self.rs_cell.smooth_basis_cell()
        auxcell = self.auxcell
        nao = cell_d.nao
        naux = auxcell.nao
        kpts = self.kpts
        nkpts = kpts.shape[0]
        if nao == 0 or naux == 0:
            log.debug2('Not found diffused basis. Skip outcore_smooth_block')
            return

        mesh = cell_d.mesh
        aoR_ks, aoI_ks = _eval_gto(cell_d, mesh, kpts)
        coords = cell_d.get_uniform_grids(mesh)

        # TODO check if max_memory is enough
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        b = cell_d.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]

        def get_Vaux(kpt):
            # int3c2e = fft(ao.conj()*ao*exp(-1j*coords.dot(kpt))) * coulG *
            #           (cell.vol/ngrids) * fft(aux*exp(-1j*coords.dot(-kpt)))
            #         = fft(ao.conj()*ao*exp(-1j*coords.dot(kpt))) * coulG *
            #           ft_ao(aux, -kpt)
            #         = ao.conj()*ao*exp(-1j*coords.dot(kpt)) *
            #           ifft(coulG * ft_ao(aux, -kpt))
            #         = ao.conj()*ao*Vaux
            # where
            # Vaux = ao*exp(-1j*coords.dot(kpt)) * ifft(coulG * ft_ao(aux, -kpt))
            auxG = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, -kpt).T
            if self.has_long_range():
                auxG *= pbctools.get_coulG(cell, -kpt, False, None, mesh, Gv,
                                           omega=cell.omega)
            else:
                auxG *= pbctools.get_coulG(cell, -kpt, False, None, mesh, Gv,
                                           omega=-self.omega)

            max_memory = (self.max_memory - lib.current_memory()[0])
            blksize = max(8, int(max_memory*.95e6/16/2/ngrids))
            log.debug2('Block size for IFFT(Vaux) %d', blksize)
            # Reuse auxG to reduce memory footprint
            Vaux = auxG
            for p0, p1 in lib.prange(0, naux, blksize):
                Vaux[p0:p1] = pbctools.ifft(auxG[p0:p1], mesh)
            Vaux *= np.exp(-1j * coords.dot(kpt))
            return Vaux

        def join_R(ki, kj):
            #:aopair = np.einsum('ig,jg->ijg', aoR_ks[ki], aoR_ks[kj])
            #:aopair+= np.einsum('ig,jg->ijg', aoI_ks[ki], aoI_ks[kj])
            aopair = np.empty((nao**2, ngrids))
            libpbc.PBC_zjoinR_CN_s1(
                aopair.ctypes.data_as(ctypes.c_void_p),
                aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
                aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
                aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
                aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
            return aopair

        def join_I(ki, kj):
            #:aopair = np.einsum('ig,jg->ijg', aoR_ks[ki], aoI_ks[kj])
            #:aopair-= np.einsum('ig,jg->ijg', aoI_ks[ki], aoR_ks[kj])
            aopair = np.empty((nao**2, ngrids))
            libpbc.PBC_zjoinI_CN_s1(
                aopair.ctypes.data_as(ctypes.c_void_p),
                aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
                aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
                aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
                aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
            return aopair

        gamma_point_only = is_zero(kpts)
        if j_only or nkpts == 1:
            Vaux = np.asarray(get_Vaux(np.zeros(3)).real, order='C')
            if gamma_point_only:
                #:aopair = np.einsum('ig,jg->ijg', aoR_ks[0], aoR_ks[0])
                aopair = np.empty((nao**2, ngrids))
                libpbc.PBC_djoin_NN_s1(
                    aopair.ctypes.data_as(ctypes.c_void_p),
                    aoR_ks[0].ctypes.data_as(ctypes.c_void_p),
                    aoR_ks[0].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
                j3c = lib.ddot(aopair.reshape(nao**2, ngrids), Vaux.T)
                h5group[f'{dataname}R-dd/0'] = j3c.reshape(nao, nao, naux)
                aopair = j3c = None

            else:
                #:for k in range(nkpts):
                #:    h5group[f'{dataname}R-dd/{k*nkpts+k}'] = lib.ddot(join_R(k, k), Vaux.T)
                #:    h5group[f'{dataname}I-dd/{k*nkpts+k}'] = lib.ddot(join_I(k, k), Vaux.T)
                k_idx = np.arange(nkpts, dtype=np.int32)
                kpt_ij_idx = k_idx * nkpts + k_idx
                j3cR = np.empty((nkpts, nao, nao, naux))
                j3cI = np.empty((nkpts, nao, nao, naux))
                libpbc.PBC_kzdot_CNN_s1(j3cR.ctypes.data_as(ctypes.c_void_p),
                                        j3cI.ctypes.data_as(ctypes.c_void_p),
                                        aoR_ks.ctypes.data_as(ctypes.c_void_p),
                                        aoI_ks.ctypes.data_as(ctypes.c_void_p),
                                        Vaux.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
                                        kpt_ij_idx.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(nao), ctypes.c_int(nao),
                                        ctypes.c_int(naux), ctypes.c_int(ngrids),
                                        ctypes.c_int(nkpts), ctypes.c_int(nkpts))
                for k, kk_idx in enumerate(kpt_ij_idx):
                    h5group[f'{dataname}R-dd/{kk_idx}'] = j3cR[k]
                    h5group[f'{dataname}I-dd/{kk_idx}'] = j3cI[k]

        else:
            uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
                cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
            scaled_uniq_kpts = cell_d.get_scaled_kpts(uniq_kpts).round(5)
            log.debug('Num uniq kpts %d', len(uniq_kpts))
            log.debug2('Scaled unique kpts %s', scaled_uniq_kpts)
            for k, k_conj in group_by_conj_pairs(cell, uniq_kpts)[0]:
                # Find ki's and kj's that satisfy k_aux = kj - ki
                kpt_ij_idx = np.asarray(np.where(uniq_inverse == k)[0], dtype=np.int32)
                nkptij = len(kpt_ij_idx)

                Vaux = get_Vaux(uniq_kpts[k])
                VauxR = np.asarray(Vaux.real, order='C')
                VauxI = np.asarray(Vaux.imag, order='C')
                Vaux = None
                #:for kk_idx in kpt_ij_idx:
                #:    ki = kk_idx // nkpts
                #:    kj = kk_idx % nkpts
                #:    aopair = join_R(ki, kj, exp(-i*k dot r))
                #:    j3cR = lib.ddot(aopair.reshape(nao**2, ngrids), VauxR.T)
                #:    j3cI = lib.ddot(aopair.reshape(nao**2, ngrids), VauxI.T)
                #:    aopair = join_I(ki, kj, exp(-i*k dot r))
                #:    j3cR = lib.ddot(aopair.reshape(nao**2, ngrids), VauxI.T,-1, j3cR, 1)
                #:    j3cI = lib.ddot(aopair.reshape(nao**2, ngrids), VauxR.T, 1, j3cI, 1)
                j3cR = np.empty((nkptij, nao, nao, naux))
                j3cI = np.empty((nkptij, nao, nao, naux))
                libpbc.PBC_kzdot_CNN_s1(j3cR.ctypes.data_as(ctypes.c_void_p),
                                        j3cI.ctypes.data_as(ctypes.c_void_p),
                                        aoR_ks.ctypes.data_as(ctypes.c_void_p),
                                        aoI_ks.ctypes.data_as(ctypes.c_void_p),
                                        VauxR.ctypes.data_as(ctypes.c_void_p),
                                        VauxI.ctypes.data_as(ctypes.c_void_p),
                                        kpt_ij_idx.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(nao), ctypes.c_int(nao),
                                        ctypes.c_int(naux), ctypes.c_int(ngrids),
                                        ctypes.c_int(nkptij), ctypes.c_int(nkpts))
                for k, kk_idx in enumerate(kpt_ij_idx):
                    h5group[f'{dataname}R-dd/{kk_idx}'] = j3cR[k]
                    h5group[f'{dataname}I-dd/{kk_idx}'] = j3cI[k]
                j3cR = j3cI = VauxR = VauxI = None

    def weighted_ft_ao(self, kpt):
        '''exp(-i*(G + k) dot r) * Coulomb_kernel'''
        rs_cell = self.rs_cell
        Gv, Gvbase, kws = rs_cell.get_Gv_weights(self.mesh)
        b = rs_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        coulG = self.weighted_coulG(kpt, False, self.mesh)
        coulG_LR = coulG - self.weighted_coulG_SR(kpt, False, self.mesh)

        shls_slice = None
        if self.exclude_d_aux:
            # The smooth basis in auxcell was excluded in outcore_auxe2.
            # Full Coulomb kernel needs to be applied for the smooth basis
            rs_auxcell = self.rs_auxcell
            smooth_aux_mask = rs_auxcell.get_ao_type() == ft_ao.SMOOTH_BASIS
            auxG = ft_ao.ft_ao(rs_auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
            auxG[smooth_aux_mask] *= coulG
            auxG[~smooth_aux_mask] *= coulG_LR
            auxG = rs_auxcell.recontract_1d(auxG)
        else:
            auxcell = self.auxcell
            auxG = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
            auxG *= coulG_LR
        Gaux = lib.transpose(auxG)
        GauxR = np.asarray(Gaux.real, order='C')
        GauxI = np.asarray(Gaux.imag, order='C')
        return GauxR, GauxI

    def gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, aosym):
        cell = self.cell
        naux = self.auxcell.nao
        vbar = None
        # Explicitly add the G0 contributions here because FT will not be
        # applied to the j3c integrals for short range integrals.
        if cell.dimension == 3 and is_zero(kpt):
            if self.exclude_d_aux:
                rs_auxcell = self.rs_auxcell
                aux_chg = _gaussian_int(rs_auxcell)
                smooth_ao_idx = rs_auxcell.get_ao_type() == ft_ao.SMOOTH_BASIS
                aux_chg[smooth_ao_idx] = 0
                aux_chg = rs_auxcell.recontract_1d(aux_chg[:,None]).ravel()
            else:
                aux_chg = _gaussian_int(self.auxcell)

            if self.exclude_dd_block:
                rs_cell = self.rs_cell
                ovlp = rs_cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts)
                smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
                for s in ovlp:
                    s[smooth_ao_idx[:,None] & smooth_ao_idx] = 0
                recontract_2d = rs_cell.recontract(dim=2)
                ovlp = [recontract_2d(s) for s in ovlp]
            else:
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts)

            if aosym == 's2':
                ovlp = [lib.pack_tril(s) for s in ovlp]
            else:
                ovlp = [s.ravel() for s in ovlp]

            vbar = np.pi / self.omega**2 / cell.vol * aux_chg
            vbar_idx = np.where(vbar != 0)[0]
            if len(vbar_idx) == 0:
                vbar = None
            nkpts = len(self.kpts)

        def load_j3c(col0, col1):
            j3cR = []
            j3cI = []
            for kk in kpt_ij_idx:
                vR = h5group[f'j3cR/{kk}'][col0:col1].reshape(-1, naux)
                if f'j3cI/{kk}' in h5group:
                    vI = h5group[f'j3cI/{kk}'][col0:col1].reshape(-1, naux)
                else:
                    vI = None
                if vbar is not None:
                    kj = kk % nkpts
                    vmod = ovlp[kj][col0:col1,None] * vbar[vbar_idx]
                    vR[:,vbar_idx] -= vmod.real
                    if vI is not None:
                        vI[:,vbar_idx] -= vmod.imag
                j3cR.append(vR)
                j3cI.append(vI)
            return j3cR, j3cI

        return load_j3c

    def add_ft_j3c(self, j3c, Gpq, Gaux, p0, p1):
        j3cR, j3cI = j3c
        GauxR = Gaux[0][p0:p1]
        GauxI = Gaux[1][p0:p1]
        nG = p1 - p0
        for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
            GpqR = GpqR.reshape(nG, -1)
            GpqI = GpqI.reshape(nG, -1)
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            lib.ddot(GpqR.T, GauxR, 1, j3cR[k], 1)
            lib.ddot(GpqI.T, GauxI, 1, j3cR[k], 1)
            if j3cI[k] is not None:
                lib.ddot(GpqI.T, GauxR,  1, j3cI[k], 1)
                lib.ddot(GpqR.T, GauxI, -1, j3cI[k], 1)

    def solve_cderi(self, cd_j2c, j3cR, j3cI):
        j2c, j2c_negative, j2ctag = cd_j2c
        if j3cI is None:
            j3c = j3cR.T
        else:
            j3c = (j3cR + j3cI * 1j).T

        cderi_negative = None
        if j2ctag == 'CD':
            cderi = scipy.linalg.solve_triangular(j2c, j3c, lower=True, overwrite_b=True)
        else:
            cderi = lib.dot(j2c, j3c)
            if j2c_negative is not None:
                # for low-dimension systems
                cderi_negative = lib.dot(j2c_negative, j3c)
        return cderi, cderi_negative

    def gen_uniq_kpts_groups(self, j_only, h5swap):
        '''Group (kpti,kptj) pairs
        '''
        cpu1 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        if j_only or nkpts == 1:
            uniq_kpts = np.zeros((1,3))
            j2c = self.get_2c2e(uniq_kpts)[0]
            cpu1 = log.timer('int2c2e', *cpu1)
            cd_j2c = self.decompose_j2c(j2c)
            j2c = None
            ki = np.arange(nkpts)
            kpt_ii_idx = ki * nkpts + ki
            yield uniq_kpts[0], kpt_ii_idx, cd_j2c

        else:
            uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
                cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
            scaled_uniq_kpts = cell.get_scaled_kpts(uniq_kpts).round(5)
            log.debug('Num uniq kpts %d', len(uniq_kpts))
            log.debug2('scaled unique kpts %s', scaled_uniq_kpts)

            kpts_idx_pairs = group_by_conj_pairs(cell, uniq_kpts)[0]
            j2c_uniq_kpts = uniq_kpts[[k for k, _ in kpts_idx_pairs]]
            for k, j2c in enumerate(self.get_2c2e(j2c_uniq_kpts)):
                h5swap[f'j2c/{k}'] = j2c
                j2c = None
            cpu1 = log.timer('int2c2e', *cpu1)

            for j2c_idx, (k, k_conj) in enumerate(kpts_idx_pairs):
                # Find ki's and kj's that satisfy k_aux = kj - ki
                log.debug1('Cholesky decomposition for j2c at kpt %s %s',
                           k, scaled_uniq_kpts[k])
                j2c = h5swap[f'j2c/{j2c_idx}']
                if k == k_conj:
                    # DF metric for self-conjugated k-point should be real
                    j2c = np.asarray(j2c).real
                cd_j2c = self.decompose_j2c(j2c)
                j2c = None
                kpt_ij_idx = np.where(uniq_inverse == k)[0]
                yield uniq_kpts[k], kpt_ij_idx, cd_j2c

                if k_conj is None or k == k_conj:
                    continue

                # Swap ki, kj for the conjugated case
                log.debug1('Cholesky decomposition for the conjugated kpt %s %s',
                           k_conj, scaled_uniq_kpts[k_conj])
                kpt_ji_idx = np.where(uniq_inverse == k_conj)[0]
                # If self.mesh is not enough to converge compensated charge or
                # SR-coulG, the conj symmetry between j2c[k] and j2c[k_conj]
                # (j2c[k] == conj(j2c[k_conj]) may not be strictly held.
                # Decomposing j2c[k] and j2c[k_conj] may lead to different
                # dimension in cderi tensor. Certain df_ao2mo requires
                # contraction for cderi of k and cderi of k_conj. By using the
                # conj(j2c[k]) and -uniq_kpts[k] (instead of j2c[k_conj] and
                # uniq_kpts[k_conj]), conj-symmetry in j2c is imposed.
                yield -uniq_kpts[k], kpt_ji_idx, _conj_j2c(cd_j2c)

    def make_j3c(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                 j_only=False, dataname='j3c', shls_slice=None):
        if self.rs_cell is None:
            self.build()
        log = logger.new_logger(self)
        cpu0 = logger.process_clock(), logger.perf_counter()

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao
        naux = self.auxcell.nao
        if shls_slice is None:
            ish0, ish1 = 0, cell.nbas
        else:
            ish0, ish1 = shls_slice[:2]

        fswap = self.outcore_auxe2(cderi_file, intor, aosym, comp, j_only,
                                   dataname, shls_slice)
        cpu1 = log.timer('pass1: real space int3c2e', *cpu0)

        feri = h5py.File(cderi_file, 'w')
        feri['kpts'] = kpts
        feri['aosym'] = aosym

        if aosym == 's2':
            nao_pair = nao*(nao+1)//2
        else:
            nao_pair = nao**2

        if self.has_long_range():
            supmol_ft = _ExtendedMoleFT.from_cell(self.rs_cell, self.bvk_kmesh, verbose=log)
            supmol_ft.exclude_dd_block = self.exclude_dd_block
            supmol_ft = supmol_ft.strip_basis()
            ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False, verbose=log)

        Gv, Gvbase, kws = cell.get_Gv_weights(self.mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]

        def make_cderi(kpt, kpt_ij_idx, j2c):
            log.debug1('make_cderi for %s', kpt)
            log.debug1('kpt_ij_idx = %s', kpt_ij_idx)
            kptjs = kpts[kpt_ij_idx % nkpts]
            nkptj = len(kptjs)
            if self.has_long_range():
                Gaux = self.weighted_ft_ao(kpt)

            mem_now = lib.current_memory()[0]
            log.debug2('memory = %s', mem_now)
            max_memory = max(1000, self.max_memory - mem_now)
            # nkptj for 3c-coulomb arrays plus 1 Lpq array
            buflen = min(max(int(max_memory*.3e6/16/naux/(nkptj+1)), 1), nao_pair)
            sh_ranges = _guess_shell_ranges(cell, buflen, aosym, start=ish0, stop=ish1)
            buflen = max([x[2] for x in sh_ranges])
            # * 2 for the buffer used in preload
            max_memory -= buflen * naux * (nkptj+1) * 16e-6 * 2

            # +1 for a pqkbuf
            Gblksize = max(16, int(max_memory*1e6/16/buflen/(nkptj+1)))
            Gblksize = min(Gblksize, ngrids, 200000)

            load = self.gen_j3c_loader(fswap, kpt, kpt_ij_idx, aosym)

            cols = [sh_range[2] for sh_range in sh_ranges]
            locs = np.append(0, np.cumsum(cols))
            # buf for ft_aopair
            buf = np.empty(nkptj*buflen*Gblksize, dtype=np.complex128)
            for istep, j3c in enumerate(lib.map_with_prefetch(load, locs[:-1], locs[1:])):
                bstart, bend, ncol = sh_ranges[istep]
                log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                           istep+1, len(sh_ranges), bstart, bend, ncol)
                if aosym == 's2':
                    shls_slice = (bstart, bend, 0, bend)
                else:
                    shls_slice = (bstart, bend, 0, cell.nbas)

                if self.has_long_range():
                    for p0, p1 in lib.prange(0, ngrids, Gblksize):
                        # shape of Gpq (nkpts, nGv, ni, nj)
                        Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt,
                                      kptjs, shls_slice, out=buf)
                        self.add_ft_j3c(j3c, Gpq, Gaux, p0, p1)
                        Gpq = None

                j3cR, j3cI = j3c
                for k, kk_idx in enumerate(kpt_ij_idx):
                    cderi, cderi_negative = self.solve_cderi(j2c, j3cR[k], j3cI[k])
                    feri[f'{dataname}/{kk_idx}/{istep}'] = cderi
                    if cderi_negative is not None:
                        # for low-dimension systems
                        feri[f'{dataname}-/{kk_idx}/{istep}'] = cderi_negative
                j3cR = j3cI = j3c = cderi = None

        for kpt, kpt_ij_idx, cd_j2c in self.gen_uniq_kpts_groups(j_only, fswap):
            make_cderi(kpt, kpt_ij_idx, cd_j2c)

        feri.close()
        cpu1 = log.timer('pass2: AFT int3c2e', *cpu1)
        return self


def _strip_basis(supmol, omega, exp_aux_min=None, exclude_dd_block=False):
    '''Remove redundant remote basis'''
    rs_cell = supmol.rs_cell
    bas_mask = supmol.bas_mask
    compact_bas_mask = rs_cell.bas_type != ft_ao.SMOOTH_BASIS
    exps = np.array([e.min() for e in rs_cell.bas_exps()])
    exps_c = exps[compact_bas_mask]
    if exps_c.size > 0:
        exp_min = exps.min()
        # compact_aux_idx = np.where(rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS)[0]
        # exp_aux_min = min([rs_auxcell.bas_exp(ib).min() for ib in compact_aux_idx])
        # Is the exact exp_aux_min needed here?
        if exp_aux_min is None:
            exp_aux_min = exp_min
        aij = exp_min + exps_c
        eij = exp_min * exps_c / aij
        theta = 1./(omega**-2 + 1./aij + 1./exp_aux_min)
        LKs = supmol.Ls[:,None,:] + supmol.bvkmesh_Ls

        # For basis on the boundary of a cell, boundary_penalty can adjust
        # the LKs to get proper distance between basis
        shifts = lib.cartesian_prod([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        a_off_bond = rs_cell.lattice_vectors() - 1.
        boundary_penalty = np.einsum('te,ex->tx', shifts, a_off_bond)
        rLK = np.linalg.norm(LKs + boundary_penalty[:,None,None], axis=-1).min(axis=0)
        rLK[rLK < 1e-2] = 1e-2  # avoid singularity in upper_bounds
        rr = rLK ** 2

        # x = rcut * x_ratio for the distance between compact function
        # and smooth function (compact function in the far end)
        # fac*erfc(\sqrt(theta)|rcut - x|) for the asymptotic value of short-range eri
        x_ratio = 1. / (exp_min/aij + exps_c/theta)
        exp_fac = eij * x_ratio**2 + theta * (1 - exp_min/aij*x_ratio)**2
        fac = ((8*np.pi*exp_min*exps_c/(aij*exp_aux_min)**2)**.75
               / (theta * np.pi)**.5)
        # upper_bounds are the maximum values int3c2e can reach for each
        # basis in each repeated image. shape (bas_id, image_id, bvk_cell_id)
        upper_bounds = np.einsum('i,lk,ilk->kil', fac, 2*np.pi/rr,
                                 np.exp(-exp_fac[:,None,None]*rr))
        # The cutoff here is most critical parameter that impacts the
        # accuracy of DF integrals
        bas_mask[:,compact_bas_mask] = upper_bounds > supmol.precision

        # determine rcut boundary for diffused functions
        exps_d = exps[~compact_bas_mask]
        if exps_d.size > 0:
            if exclude_dd_block:
                # Just needs to estimate the upper bounds of (C,D|aux)
                # otherwise, we need exp_min = exp_d_min for the (D,D|aux)
                # upper bound estimation
                exp_min = exps_c.min()
            aij = exp_min + exps_d
            eij = exp_min * exps_d / aij
            theta = 1./(omega**-2 + 1./aij + 1./exp_aux_min)

            x_ratio = 1. / (exps_d/aij + exp_min/theta)
            exp_fac = eij * x_ratio**2 + theta * (1 - exps_d/aij*x_ratio)**2
            fac = ((8*np.pi*exps_d*exp_min/(aij*exp_aux_min)**2)**.75
                   / (theta * np.pi)**.5)
            # upper_bounds are the maximum values int3c2e can reach for each
            # basis in each repeated image. shape (bas_id, image_id, bvk_cell_id)
            upper_bounds = np.einsum('i,lk,ilk->kil', fac, 2*np.pi/rr,
                                     np.exp(-exp_fac[:,None,None]*rr))
            bas_mask[:,~compact_bas_mask] = upper_bounds > supmol.precision

        bas_mask[0,:,0] = True

    nbas0 = supmol._bas.shape[0]
    supmol._bas = np.asarray(supmol._bas[bas_mask.ravel()], dtype=np.int32, order='C')
    nbas1 = supmol._bas.shape[0]
    logger.debug1(supmol, 'strip_basis %d to %d ', nbas0, nbas1)
    supmol.sh_loc = supmol.bas_mask_to_sh_loc(rs_cell, bas_mask)
    supmol.bas_mask = bas_mask
    return supmol

class _ExtendedMoleSR(ft_ao._ExtendedMole):
    '''Extended Mole for short-range ERIs without dd-blocks'''

    @classmethod
    def from_cell(cls, cell, kmesh, omega, rcut=None, verbose=None):
        supmol = super(_ExtendedMoleSR, cls).from_cell(cell, kmesh, rcut, verbose)
        supmol.omega = -omega
        return supmol

class _ExtendedMoleFT(ft_ao._ExtendedMole):
    '''Extended Mole for Fourier Transform without dd-blocks'''

    def __init__(self):
        self.exclude_dd_block = True
        ft_ao._ExtendedMole.__init__(self)

    def get_ovlp_mask(self, cutoff=None):
        '''integral screening mask for basis product between cell and supmol.
        The diffused-diffused basis block are removed
        '''
        ovlp_mask = super().get_ovlp_mask(cutoff)
        if self.exclude_dd_block:
            rs_cell = self.rs_cell
            cell0_smooth_idx = np.where(rs_cell.bas_type == ft_ao.SMOOTH_BASIS)[0]
            smooth_idx = self.bas_type_to_indices(ft_ao.SMOOTH_BASIS)
            ovlp_mask[cell0_smooth_idx[:,None], smooth_idx] = 0
        return ovlp_mask

# ngrids ~= 8*naux = prod(mesh)
def _guess_omega(cell, kpts, mesh=None):
    if cell.dimension == 0:
        if mesh is None:
            mesh = cell.mesh
        ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh).min()
        omega = aft.estimate_omega_for_ke_cutoff(cell, ke_cutoff, cell.precision)
        return omega, mesh, ke_cutoff

    a = cell.lattice_vectors()
    # requiring Coulomb potential < cell.precision at rcut is often not
    # enough to truncate the interaction.
    omega_min = aft.estimate_omega(cell, cell.precision*1e-2)
    ke_min = aft.estimate_ke_cutoff_for_omega(cell, omega_min, cell.precision)
    mesh_min = _round_off_to_odd_mesh(pbctools.cutoff_to_mesh(a, ke_min))

    if mesh is None:
        nao = cell.npgto_nr()
        nkpts = len(kpts)
        # FIXME: balance the two workloads
        # int3c2e integrals ~ nao*(cell.rcut**3/cell.vol*nao)**2
        # ft_ao integrals ~ nkpts*nao*(cell.rcut**3/cell.vol*nao)*mesh**3
        nimgs = (8 * cell.rcut**3 / cell.vol) ** (cell.dimension / 3)
        mesh = (nimgs**2*nao / (nkpts**.5*nimgs**.5 * 1e2 + nkpts**2*nao))**(1./3) + 2
        mesh = int(min((1e8/nao)**(1./3), mesh))
        mesh = np.max([mesh_min, [mesh] * 3], axis=0)
        ke_cutoff = pbctools.mesh_to_cutoff(a, mesh-1)
        ke_cutoff = ke_cutoff[:cell.dimension].min()
        if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
            mesh[cell.dimension:] = cell.mesh[cell.dimension:]
        elif cell.dimension == 2:
            mesh = pbctools.cutoff_to_mesh(a, ke_cutoff)
        mesh = _round_off_to_odd_mesh(mesh)
    else:
        if np.any(mesh[:cell.dimension] < mesh_min[:cell.dimension]):
            logger.warn(cell, 'mesh %s is not enough to converge to the required '
                        'integral precision %g.\nRecommended mesh is %s.',
                        mesh, cell.precision, mesh_min)
        ke_cutoff = pbctools.mesh_to_cutoff(a, np.asarray(mesh)-1)
        ke_cutoff = ke_cutoff[:cell.dimension].min()
    omega = aft.estimate_omega_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    return omega, mesh, ke_cutoff

def _eval_gto(cell, mesh, kpts):
    coords = cell.get_uniform_grids(mesh)
    nkpts = len(kpts)
    nao = cell.nao
    ngrids = len(coords)

    ao_ks = cell.pbc_eval_gto('GTOval', coords, kpts=kpts)

    aoR_ks = np.empty((nkpts, nao, ngrids))
    aoI_ks = np.empty((nkpts, nao, ngrids))
    for k, dat in enumerate(ao_ks):
        aoR_ks[k] = dat.real.T
        aoI_ks[k] = dat.imag.T
    return aoR_ks, aoI_ks

def _conj_j2c(cd_j2c):
    j2c, j2c_negative, j2ctag = cd_j2c
    if j2c_negative is None:
        return j2c.conj(), None, j2ctag
    else:
        return j2c.conj(), j2c_negative.conj(), j2ctag

def _gaussian_int(cell):
    r'''Regular gaussian integral \int g(r) dr^3'''
    return ft_ao.ft_ao(cell, np.zeros((1,3)))[0].real

def _round_off_to_odd_mesh(mesh):
    # Round off mesh to the nearest odd numbers.
    # Odd number of grids is preferred because even number of grids may break
    # the conjugation symmetry between the k-points k and -k.
    # When building the DF integral tensor in function _make_j3c, the symmetry
    # between k and -k is used (function conj_j2c) to overcome the error
    # caused by auxiliary basis linear dependency. More detalis of this
    # problem can be found in function _make_j3c.
    return (np.asarray(mesh) // 2) * 2 + 1
