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
import warnings
import tempfile
import numpy as np
import scipy.linalg
import h5py
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger, zdotCN
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df import aft
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.incore import libpbc, Int3cBuilder
from pyscf.pbc.lib.kpts_helper import (is_zero, member, kk_adapted_iter,
                                       members_with_wrap_around, KPT_DIFF_TOL)
from pyscf import __config__

OMEGA_MIN = 0.08
INDEX_MIN = -10000
LINEAR_DEP_THR = getattr(__config__, 'pbc_df_df_DF_lindep', 1e-10)
# Threshold of steep bases and local bases
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 1.0)


class _RSGDFBuilder(Int3cBuilder):
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

    _keys = {
        'mesh', 'omega', 'rs_auxcell', 'supmol_ft'
    }

    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        self.mesh = None
        if cell.omega == 0:
            self.omega = None
        elif cell.omega < 0:
            # Initialize omega to cell.omega for HF exchange of short range
            # int2e in RSH functionals
            self.omega = -cell.omega
        else:
            raise RuntimeError('RSDF does not support LR integrals')
        self.rs_auxcell = None
        self.supmol_ft = None

        Int3cBuilder.__init__(self, cell, auxcell, kpts)

    @property
    def exclude_dd_block(self):
        cell = self.cell
        return (self.fft_dd_block and
                cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum')

    @exclude_dd_block.setter
    def exclude_dd_block(self, x):
        self.fft_dd_block = x
        self.reset()

    def has_long_range(self):
        '''Whether to add the long-range part computed with AFT integrals'''
        # If self.exclude_d_aux is set, the block (D|**) will not be computed in
        # outcore_auxe2. It has to be computed by AFT code.
        cell = self.cell
        return (cell.dimension > 0 and
                (self.omega is None or abs(cell.omega) < self.omega or self.exclude_d_aux))

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

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if omega is not None:
            self.omega = omega

        if self.omega is None or self.omega == 0:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(auxcell, kpts, self.mesh)
        elif self.mesh is None:
            self.ke_cutoff = estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = cell.cutoff_to_mesh(self.ke_cutoff)
        elif self.ke_cutoff is None:
            ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            self.ke_cutoff = ke_cutoff[:cell.dimension].min()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            self.mesh[2] = _estimate_meshz(cell)
        elif cell.dimension < 2:
            self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]
        self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        exp_min = np.hstack(cell.bas_exps()).min()
        # For each basis i in (ij|, small integrals accumulated by the lattice
        # sum for j are not negligible. (2*cell.rcut)**3/vol is roughly the
        # number of basis i and 1./exp_min for the non-negligible basis j.
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff
        log.debug('Set _RSGDFBuilder.direct_scf_tol to %g', cutoff)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)
        self.rs_auxcell = rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(
            auxcell, self.ke_cutoff, verbose=log)

        rcut_sr = estimate_rcut(rs_cell, rs_auxcell, self.omega,
                                exclude_dd_block=self.exclude_dd_block,
                                exclude_d_aux=self.exclude_d_aux and cell.dimension > 0)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_sr.max(), log)
        supmol.omega = -self.omega
        self.supmol = supmol.strip_basis(rcut_sr)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        if self.has_long_range():
            rcut = estimate_ft_rcut(rs_cell, exclude_dd_block=self.exclude_dd_block)
            supmol_ft = _ExtendedMoleFT.from_cell(rs_cell, kmesh, rcut.max(), log)
            supmol_ft.exclude_dd_block = self.exclude_dd_block
            self.supmol_ft = supmol_ft.strip_basis(rcut)
            log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                      supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        log.timer_debug1('initializing supmol', *cpu0)
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

    def get_q_cond(self, supmol=None):
        '''Integral screening condition max(sqrt((ij|ij))) inside the supmol'''
        q_cond = Int3cBuilder.get_q_cond(self, supmol)

        # Remove d-d block in supmol q_cond
        if self.exclude_dd_block and self.cell.dimension > 0:
            smooth_idx = supmol.bas_type_to_indices(ft_ao.SMOOTH_BASIS)
            q_cond[smooth_idx[:,None], smooth_idx] = INDEX_MIN
        return q_cond

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
        mask = w > self.linear_dep_threshold
        logger.debug(self, 'cond = %.4g, drop %d bfns',
                     w[-1]/w[0], w.size-np.count_nonzero(mask))
        v1 = v[:,mask].conj().T
        v1 /= np.sqrt(w[mask, None])
        j2c = v1
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            idx = np.where(w < -self.linear_dep_threshold)[0]
            if len(idx) > 0:
                j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
        j2ctag = 'ED'
        return j2c, j2c_negative, j2ctag

    def get_2c2e(self, uniq_kpts):
        # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
        cell = self.cell
        auxcell = self.auxcell
        if auxcell.dimension == 0:
            return [auxcell.intor('int2c2e', hermi=1)]

        if not self.has_long_range():
            omega = auxcell.omega
            j2c = auxcell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
            if (auxcell.dimension >= 2 and omega != 0 and
                auxcell.low_dim_ft_type != 'inf_vacuum'):
                gamma_point_idx = member(np.zeros(3), uniq_kpts)
                if len(gamma_point_idx) > 0:
                    # Add G=0 contribution
                    g0_fac = np.pi / omega**2 / auxcell.vol
                    aux_chg = _gaussian_int(auxcell)
                    j2c[gamma_point_idx[0]] -= g0_fac * aux_chg[:,None] * aux_chg
            return j2c

        precision = auxcell.precision**1.5
        omega = self.omega
        rs_auxcell = self.rs_auxcell
        auxcell_c = rs_auxcell.compact_basis_cell()
        if auxcell_c.nbas > 0:
            aux_exp = np.hstack(auxcell_c.bas_exps()).min()
            if omega == 0:
                theta = aux_exp / 2
            else:
                theta = 1./(2./aux_exp + omega**-2)
            fac = 2*np.pi**3.5/auxcell.vol * aux_exp**-3 * theta**-1.5
            rcut_sr = (np.log(fac / auxcell_c.rcut / precision + 1.) / theta)**.5
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
        ke = estimate_ke_cutoff_for_omega(auxcell, omega, precision)
        mesh = auxcell.cutoff_to_mesh(ke)
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            mesh[2] = _estimate_meshz(auxcell)
        elif cell.dimension < 2:
            mesh[cell.dimension:] = cell.mesh[cell.dimension:]
        mesh = cell.symmetrize_mesh(mesh)
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
                    auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                                       Gvbase, kpt).T
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
                if (cell.dimension == 3 or
                    (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
                    with lib.temporary_env(cell, dimension=3):
                        coulG_sr = self.weighted_coulG_SR(kpt, False, mesh)
                    if omega != 0 and is_zero(kpt):
                        G0_idx = 0  # due to np.fft.fftfreq convention
                        coulG_SR_at_G0 = np.pi/omega**2 * kws
                        coulG_sr[G0_idx] += coulG_SR_at_G0
                else:
                    coulG_sr = self.weighted_coulG_SR(kpt, False, mesh)

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
                      fft_dd_block=None, kk_idx=None):
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
        fswap = lib.H5TmpFile(dir=os.path.dirname(cderi_file), prefix='.outcore_auxe2_swap')
        # Unlink swapfile to avoid trash files
        os.unlink(fswap.filename)

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

        if self.exclude_d_aux and cell.dimension > 0:
            rs_auxcell = self.rs_auxcell.compact_basis_cell()
        else:
            rs_auxcell = self.rs_auxcell
        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

        ao_loc = cell.ao_loc
        aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
        ish0, ish1, jsh0, jsh1, ksh0, ksh1 = shls_slice
        i0, i1, j0, j1 = ao_loc[list(shls_slice[:4])].astype(np.int64)
        k0, k1 = aux_loc[[ksh0, ksh1]].astype(np.int64)
        if aosym == 's1':
            nao_pair = (i1 - i0) * (j1 - j0)
        else:
            nao_pair = i1*(i1+1)//2 - i0*(i0+1)//2
        naux = k1 - k0

        if fft_dd_block and np.any(rs_cell.bas_type == ft_ao.SMOOTH_BASIS):
            merge_dd = rs_cell.merge_diffused_block(aosym)
        else:
            merge_dd = None

        reindex_k = None
        # TODO: shape = (comp, nao_pair, naux)
        shape = (nao_pair, naux)
        if j_only or nkpts == 1:
            nkpts_ij = nkpts
            ks = np.arange(nkpts, dtype=np.int32)
            kikj_idx = ks * nkpts + ks
            if kk_idx is not None:
                # Ensure kk_idx is a subset of all possible ki-kj paris
                assert np.all(np.isin(kk_idx, kikj_idx))
                kikj_idx = kk_idx
            reindex_k = kikj_idx // nkpts
        else:
            nkpts_ij = nkpts * nkpts
            if kk_idx is None:
                kikj_idx = np.arange(nkpts_ij, dtype=np.int32)
            else:
                kikj_idx = kk_idx
            reindex_k = kikj_idx
            if merge_dd and kk_idx is None:
                kpt_ij_iters = list(kk_adapted_iter(cell, kpts))

        for idx in kikj_idx:
            fswap.create_dataset(f'{dataname}R/{idx}', shape, 'f8')
            fswap.create_dataset(f'{dataname}I/{idx}', shape, 'f8')
        # exclude imaginary part for gamma point
        for k in np.where(abs(kpts).max(axis=1) < KPT_DIFF_TOL)[0]:
            if f'{dataname}I/{k*nkpts+k}' in fswap:
                del fswap[f'{dataname}I/{k*nkpts+k}']

        if naux == 0:
            return fswap

        if fft_dd_block:
            self._outcore_dd_block(fswap, intor, aosym, comp, j_only,
                                   dataname, kk_idx=kk_idx)

        # int3c may be the regular int3c2e, LR-int3c2e or SR-int3c2e, depending
        # on how self.supmol is initialized
        # TODO: call gen_int3c_kernel(reindex_k=kikj_idx) for a subset of kpts
        int3c = self.gen_int3c_kernel(intor, aosym, comp, j_only,
                                      reindex_k=reindex_k, rs_auxcell=rs_auxcell)

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
                    for k, idx in enumerate(kikj_idx):
                        merge_dd(outR[k], fswap[f'{dataname}R-dd/{idx}'], shls_slice)
                        merge_dd(outI[k], fswap[f'{dataname}I-dd/{idx}'], shls_slice)
                elif kk_idx is None:
                    for _, ki_idx, kj_idx, self_conj in kpt_ij_iters:
                        kpt_ij_idx = ki_idx * nkpts + kj_idx
                        if self_conj:
                            for ij_idx in kpt_ij_idx:
                                merge_dd(outR[ij_idx], fswap[f'{dataname}R-dd/{ij_idx}'], shls_slice)
                                merge_dd(outI[ij_idx], fswap[f'{dataname}I-dd/{ij_idx}'], shls_slice)
                        else:
                            kpt_ji_idx = kj_idx * nkpts + ki_idx
                            for ij_idx, ji_idx in zip(kpt_ij_idx, kpt_ji_idx):
                                j3cR_dd = np.asarray(fswap[f'{dataname}R-dd/{ij_idx}'])
                                merge_dd(outR[ij_idx], j3cR_dd, shls_slice)
                                merge_dd(outR[ji_idx], j3cR_dd.transpose(1,0,2), shls_slice)
                                j3cI_dd = np.asarray(fswap[f'{dataname}I-dd/{ij_idx}'])
                                merge_dd(outI[ij_idx], j3cI_dd, shls_slice)
                                merge_dd(outI[ji_idx],-j3cI_dd.transpose(1,0,2), shls_slice)
                else:
                    for k, idx in enumerate(kikj_idx):
                        merge_dd(outR[k], fswap[f'{dataname}R-dd/{idx}'], shls_slice)
                        merge_dd(outI[k], fswap[f'{dataname}I-dd/{idx}'], shls_slice)

            for k, idx in enumerate(kikj_idx):
                fswap[f'{dataname}R/{idx}'][row0:row1] = outR[k]
                if f'{dataname}I/{idx}' in fswap:
                    fswap[f'{dataname}I/{idx}'][row0:row1] = outI[k]
            outR = outI = None
        bufR = bufI = None
        return fswap

    def _outcore_dd_block(self, h5group, intor='int3c2e', aosym='s2', comp=None,
                          j_only=False, dataname='j3c', shls_slice=None,
                          kk_idx=None):
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
        assert cell_d.low_dim_ft_type != 'inf_vacuum'
        assert cell_d.dimension > 1

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
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
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

        #:def join_R(ki, kj):
        #:    #:aopair = np.einsum('ig,jg->ijg', aoR_ks[ki], aoR_ks[kj])
        #:    #:aopair+= np.einsum('ig,jg->ijg', aoI_ks[ki], aoI_ks[kj])
        #:    aopair = np.empty((nao**2, ngrids))
        #:    libpbc.PBC_zjoinR_CN_s1(
        #:        aopair.ctypes.data_as(ctypes.c_void_p),
        #:        aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
        #:        aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
        #:        aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
        #:        aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
        #:        ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
        #:    return aopair

        #:def join_I(ki, kj):
        #:    #:aopair = np.einsum('ig,jg->ijg', aoR_ks[ki], aoI_ks[kj])
        #:    #:aopair-= np.einsum('ig,jg->ijg', aoI_ks[ki], aoR_ks[kj])
        #:    aopair = np.empty((nao**2, ngrids))
        #:    libpbc.PBC_zjoinI_CN_s1(
        #:        aopair.ctypes.data_as(ctypes.c_void_p),
        #:        aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
        #:        aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
        #:        aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
        #:        aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
        #:        ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
        #:    return aopair

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
                if kk_idx is None:
                    ks = np.arange(nkpts, dtype=np.int32)
                    kpt_ij_idx = ks * nkpts + ks
                else:
                    kpt_ij_idx = np.asarray(kk_idx, dtype=np.int32)
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
                for k, idx in enumerate(kpt_ij_idx):
                    h5group[f'{dataname}R-dd/{idx}'] = j3cR[k]
                    h5group[f'{dataname}I-dd/{idx}'] = j3cI[k]

        else:
            enable_t_rev_sym = kk_idx is None
            for kpt, ki_idx, kj_idx, self_conj \
                    in kk_adapted_iter(cell, kpts, kk_idx, enable_t_rev_sym):
                kpt_ij_idx = np.asarray(ki_idx * nkpts + kj_idx, dtype=np.int32)
                nkptij = len(kpt_ij_idx)
                Vaux = get_Vaux(kpt)
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
                for k, idx in enumerate(kpt_ij_idx):
                    h5group[f'{dataname}R-dd/{idx}'] = j3cR[k]
                    h5group[f'{dataname}I-dd/{idx}'] = j3cI[k]
                j3cR = j3cI = VauxR = VauxI = None

    def weighted_ft_ao(self, kpt):
        '''exp(-i*(G + k) dot r) * Coulomb_kernel'''
        cell = self.cell
        rs_cell = self.rs_cell
        Gv, Gvbase, kws = rs_cell.get_Gv_weights(self.mesh)
        b = rs_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        coulG = self.weighted_coulG(kpt, False, self.mesh)
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            with lib.temporary_env(cell, dimension=3):
                coulG_SR = self.weighted_coulG_SR(kpt, False, self.mesh)
        else:
            coulG_SR = self.weighted_coulG_SR(kpt, False, self.mesh)
        coulG_LR = coulG - coulG_SR

        shls_slice = None
        if self.exclude_d_aux and rs_cell.dimension > 0:
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
        if (is_zero(kpt) and self.omega != 0 and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            if self.exclude_d_aux and cell.dimension > 0:
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

    def gen_uniq_kpts_groups(self, j_only, h5swap, kk_idx=None):
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
            if kk_idx is None:
                ki = np.arange(nkpts, dtype=np.int32)
                kpt_ii_idx = ki * nkpts + ki
            else:
                kpt_ii_idx = np.asarray(kk_idx, dtype=np.int32)
            yield uniq_kpts[0], kpt_ii_idx, cd_j2c

        else:
            enable_t_rev_sym = kk_idx is None
            kpt_ij_iters = list(kk_adapted_iter(cell, kpts, kk_idx, enable_t_rev_sym))
            j2c_uniq_kpts = np.asarray([s[0] for s in kpt_ij_iters])
            for k, j2c in enumerate(self.get_2c2e(j2c_uniq_kpts)):
                h5swap[f'j2c/{k}'] = j2c
                j2c = None
            cpu1 = log.timer('int2c2e', *cpu1)

            for j2c_idx, (kpt, ki_idx, kj_idx, self_conj) \
                    in enumerate(kpt_ij_iters):
                # Find ki's and kj's that satisfy k_aux = kj - ki
                log.debug1('Cholesky decomposition for j2c %d', j2c_idx)
                j2c = h5swap[f'j2c/{j2c_idx}']
                if self_conj:
                    # DF metric for self-conjugated k-point should be real
                    j2c = np.asarray(j2c).real
                cd_j2c = self.decompose_j2c(j2c)
                j2c = None

                kpt_ij_idx = ki_idx * nkpts + kj_idx
                yield kpt, kpt_ij_idx, cd_j2c

                if self_conj or not enable_t_rev_sym:
                    continue

                # Swap ki, kj for the conjugated case
                kpt_ji_idx = kj_idx * nkpts + ki_idx
                # If self.mesh is not enough to converge compensated charge or
                # SR-coulG, the conj symmetry between j2c[k] and j2c[k_conj]
                # (j2c[k] == conj(j2c[k_conj]) may not be strictly held.
                # Decomposing j2c[k] and j2c[k_conj] may lead to different
                # dimension in cderi tensor. Certain df_ao2mo requires
                # contraction for cderi of k and cderi of k_conj. By using the
                # conj(j2c[k]) and -uniq_kpts[k] (instead of j2c[k_conj] and
                # uniq_kpts[k_conj]), conj-symmetry in j2c is imposed.
                yield -kpt, kpt_ji_idx, _conj_j2c(cd_j2c)

    def make_j3c(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                 j_only=False, dataname='j3c', shls_slice=None, kptij_lst=None):
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

        if kptij_lst is not None:
            if aosym == 's2':
                warnings.warn('rsdf_builder does not support aosym="s2" for '
                              'custom kptij_lst')
                aosym = 's1'
            ki_idx = members_with_wrap_around(cell, kptij_lst[:,0], kpts)
            kj_idx = members_with_wrap_around(cell, kptij_lst[:,1], kpts)
            if ki_idx.size != len(kptij_lst) or kj_idx.size != len(kptij_lst):
                msg = f'some k-points in kptij_lst are not found in {self}.kpts'
                raise RuntimeError(msg)
            kk_idx = ki_idx * nkpts + kj_idx
        else:
            kk_idx = None

        if h5py.is_hdf5(cderi_file):
            feri = lib.H5FileWrap(cderi_file, 'a')
            if 'kpts' in feri:
                del feri['kpts']
                del feri['aosym']
            if dataname in feri:
                log.warn(f'Overwritting {dataname} in {cderi_file}.')
                del feri[dataname]
        else:
            feri = lib.H5FileWrap(cderi_file, 'w')
        feri['kpts'] = kpts
        feri['aosym'] = aosym

        fswap = self.outcore_auxe2(cderi_file, intor, aosym, comp, j_only,
                                   'j3c', shls_slice, kk_idx=kk_idx)
        cpu1 = log.timer('pass1: real space int3c2e', *cpu0)

        if aosym == 's2':
            nao_pair = nao*(nao+1)//2
        else:
            nao_pair = nao**2

        if self.has_long_range():
            ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                                   verbose=log)

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
            Gblksize = max(16, int(max_memory*1e6/16/buflen/(nkptj+1))//8*8)
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
                for k, idx in enumerate(kpt_ij_idx):
                    cderi, cderi_negative = self.solve_cderi(j2c, j3cR[k], j3cI[k])
                    feri[f'{dataname}/{idx}/{istep}'] = cderi
                    if cderi_negative is not None:
                        # for low-dimension systems
                        feri[f'{dataname}-/{idx}/{istep}'] = cderi_negative
                j3cR = j3cI = j3c = cderi = None

        for kpt, kpt_ij_idx, cd_j2c \
                in self.gen_uniq_kpts_groups(j_only, fswap, kk_idx=kk_idx):
            make_cderi(kpt, kpt_ij_idx, cd_j2c)

        feri.close()
        cpu1 = log.timer('pass2: AFT int3c2e', *cpu1)
        return self


def get_nuc(nuc_builder):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    nuc = nuc_builder.get_pp_loc_part1(with_pseudo=False)
    logger.timer(nuc_builder, 'get_nuc', *t0)
    return nuc

def get_pp(nuc_builder):
    '''get the periodic pseudopotential nuc-el ao matrix, with g=0 removed.

    kwargs:
        mesh: custom mesh grids. by default mesh is determined by the
        function _guess_eta from module pbc.df.gdf_builder.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    cell = nuc_builder.cell
    vpp = nuc_builder.get_pp_loc_part1()
    t1 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part1', *t0)
    pp2builder = aft._IntPPBuilder(cell, nuc_builder.kpts)
    vpp += pp2builder.get_pp_loc_part2()
    t1 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part2', *t1)
    vpp += pseudo.pp_int.get_pp_nl(cell, nuc_builder.kpts)
    t1 = logger.timer_debug1(nuc_builder, 'get_pp_nl', *t1)
    logger.timer(nuc_builder, 'get_pp', *t0)
    return vpp

def _int_dd_block(dfbuilder, fakenuc, intor='int3c2e', comp=None):
    '''
    The block of smooth AO basis in i and j of (ij|L) with full Coulomb kernel
    '''
    if intor not in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart'):
        raise NotImplementedError

    t0 = (logger.process_clock(), logger.perf_counter())
    cell = dfbuilder.cell
    cell_d = dfbuilder.rs_cell.smooth_basis_cell()
    assert cell_d.low_dim_ft_type != 'inf_vacuum'
    assert cell_d.dimension > 1

    nao = cell_d.nao
    kpts = dfbuilder.kpts
    nkpts = kpts.shape[0]
    if nao == 0 or fakenuc.natm == 0:
        if is_zero(kpts):
            return np.zeros((nao,nao,1))
        else:
            return np.zeros((2,nkpts,nao,nao,1))

    mesh = cell_d.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell_d.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

    kpt_allow = np.zeros(3)
    charges = -cell.atom_charges()
    #:rhoG = np.dot(charges, SI)
    aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
    rhoG = np.einsum('i,xi->x', charges, aoaux)
    coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
    vG = rhoG * coulG
    if (cell.dimension == 3 or
        (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
        vG[0] -= charges.dot(np.pi/np.hstack(fakenuc.bas_exps()))

    vR = pbctools.ifft(vG, mesh).real

    coords = cell_d.get_uniform_grids(mesh)
    if is_zero(kpts):
        ao_ks = cell_d.pbc_eval_gto('GTOval', coords)
        j3c = lib.dot(ao_ks.T * vR, ao_ks).reshape(nao,nao,1)

    else:
        ao_ks = cell_d.pbc_eval_gto('GTOval', coords, kpts=kpts)
        j3cR = np.empty((nkpts, nao, nao))
        j3cI = np.empty((nkpts, nao, nao))
        for k in range(nkpts):
            v = lib.dot(ao_ks[k].conj().T * vR, ao_ks[k])
            j3cR[k] = v.real
            j3cI[k] = v.imag
        j3c = j3cR.reshape(nkpts,nao,nao,1), j3cI.reshape(nkpts,nao,nao,1)
    t0 = logger.timer_debug1(dfbuilder, 'FFT smooth basis', *t0)
    return j3c


class _RSNucBuilder(_RSGDFBuilder):

    exclude_d_aux = False

    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.mesh = None
        self.omega = None
        self.auxcell = self.rs_auxcell = None
        Int3cBuilder.__init__(self, cell, self.auxcell, kpts)

    def build(self, omega=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        fakenuc = aft._fake_nuc(cell, with_pseudo=True)
        kpts = self.kpts
        nkpts = len(kpts)

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if cell.dimension == 0:
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            if omega is None:
                omega = 1./(1.+nkpts**(1./9))
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
            self.ke_cutoff = min(pbctools.mesh_to_cutoff(
                cell.lattice_vectors(), self.mesh)[:cell.dimension])
            self.omega = estimate_omega_for_ke_cutoff(cell, self.ke_cutoff)
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                self.mesh[2] = _estimate_meshz(cell)
            elif cell.dimension < 2:
                self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]
            self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        exp_min = np.hstack(cell.bas_exps()).min()
        # For each basis i in (ij|, small integrals accumulated by the lattice
        # sum for j are not negligible.
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff / cell.atom_charges().max()
        log.debug('Set _RSNucBuilder.direct_scf_tol to %g', cutoff)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)
        rcut_sr = estimate_rcut(rs_cell, fakenuc, self.omega,
                                exclude_dd_block=self.exclude_dd_block)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_sr.max(), log)
        supmol.omega = -self.omega
        self.supmol = supmol.strip_basis(rcut_sr)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        rcut = estimate_ft_rcut(rs_cell, exclude_dd_block=self.exclude_dd_block)
        supmol_ft = _ExtendedMoleFT.from_cell(rs_cell, kmesh, rcut.max(), log)
        supmol_ft.exclude_dd_block = self.exclude_dd_block
        self.supmol_ft = supmol_ft.strip_basis(rcut)
        log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                  supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        log.timer_debug1('initializing supmol', *cpu0)
        return self

    def _int_nuc_vloc(self, fakenuc, intor='int3c2e', aosym='s2', comp=None):
        '''SR-Vnuc
        '''
        logger.debug2(self, 'Real space integrals %s for SR-Vnuc', intor)

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc)
        bufR, bufI = int3c()

        charge = -cell.atom_charges()
        if is_zero(kpts):
            mat = np.einsum('k...z,z->k...', bufR, charge)
        else:
            mat = (np.einsum('k...z,z->k...', bufR, charge) +
                   np.einsum('k...z,z->k...', bufI, charge) * 1j)

        # G = 0 contributions to SR integrals
        if (self.omega != 0 and
            (intor in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart')) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            logger.debug2(self, 'G=0 part for %s', intor)
            nucbar = np.pi / self.omega**2 / cell.vol * charge.sum()
            if self.exclude_dd_block:
                rs_cell = self.rs_cell
                ovlp = rs_cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
                smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
                for s in ovlp:
                    s[smooth_ao_idx[:,None] & smooth_ao_idx] = 0
                recontract_2d = rs_cell.recontract(dim=2)
                ovlp = [recontract_2d(s) for s in ovlp]
            else:
                ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)

            for k in range(nkpts):
                if aosym == 's1':
                    mat[k] -= nucbar * ovlp[k].ravel()
                else:
                    mat[k] -= nucbar * lib.pack_tril(ovlp[k])
        return mat

    _int_dd_block = _int_dd_block

    def get_pp_loc_part1(self, mesh=None, with_pseudo=True):
        log = logger.Logger(self.stdout, self.verbose)
        t0 = t1 = (logger.process_clock(), logger.perf_counter())
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        nao_pair = nao * (nao+1) // 2
        mesh = self.mesh

        fakenuc = aft._fake_nuc(cell, with_pseudo=with_pseudo)
        vj = self._int_nuc_vloc(fakenuc)
        if cell.dimension == 0:
            return lib.unpack_tril(vj)

        if self.exclude_dd_block:
            cell_d = self.rs_cell.smooth_basis_cell()
            if cell_d.nao > 0 and fakenuc.natm > 0:
                merge_dd = self.rs_cell.merge_diffused_block(aosym)
                if is_zero(kpts):
                    vj_dd = self._int_dd_block(fakenuc)
                    merge_dd(vj, vj_dd)
                else:
                    vj_ddR, vj_ddI = self._int_dd_block(fakenuc)
                    for k in range(nkpts):
                        outR = vj[k].real.copy()
                        outI = vj[k].imag.copy()
                        merge_dd(outR, vj_ddR[k])
                        merge_dd(outI, vj_ddI[k])
                        vj[k] = outR + outI * 1j
        t0 = t1 = log.timer_debug1('vnuc pass1: analytic int', *t0)

        kpt_allow = np.zeros(3)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        b = cell.reciprocal_vectors()
        aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
        charges = -cell.atom_charges()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            with lib.temporary_env(cell, dimension=3):
                coulG_SR = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                                              omega=-self.omega)
            coulG_LR = coulG - coulG_SR
        else:
            coulG_LR = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                                          omega=self.omega)
        wcoulG = coulG_LR * kws
        vG = np.einsum('i,xi,x->x', charges, aoaux, wcoulG)

        # contributions due to pseudo.pp_int.get_gth_vlocG_part1
        if (cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
            G0_idx = 0
            exps = np.hstack(fakenuc.bas_exps())
            vG[G0_idx] -= charges.dot(np.pi/exps) * kws

        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                               kpts=kpts, verbose=log)
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        Gblksize = max(16, int(max_memory*.8e6/16/(nao_pair*nkpts))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        vGR = vG.real
        vGI = vG.imag
        log.debug1('max_memory = %s  Gblksize = %s  ngrids = %s',
                   max_memory, Gblksize, ngrids)

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # shape of Gpq (nkpts, nGv, nao_pair)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # rho_ij(G) nuc(-G) / G^2
                # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
                vR = np.einsum('k,kx->x', vGR[p0:p1], GpqR)
                vR+= np.einsum('k,kx->x', vGI[p0:p1], GpqI)
                vj[k] += vR
                if not is_zero(kpts[k]):
                    vI = np.einsum('k,kx->x', vGR[p0:p1], GpqI)
                    vI-= np.einsum('k,kx->x', vGI[p0:p1], GpqR)
                    vj[k].imag += vI
            t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
        log.timer_debug1('contracting Vnuc', *t0)

        vj_kpts = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_kpts.append(lib.unpack_tril(vj[k].real))
            else:
                vj_kpts.append(lib.unpack_tril(vj[k]))
        return np.asarray(vj_kpts)

    get_pp = get_pp
    get_nuc = get_nuc


class _ExtendedMoleFT(ft_ao.ExtendedMole):
    '''Extended Mole for Fourier Transform without dd-blocks'''

    exclude_dd_block = False

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
        return 0, mesh, ke_cutoff

    # requiring Coulomb potential < cell.precision at rcut is often not
    # enough to truncate the interaction.
    # omega_min = estimate_omega_min(cell, cell.precision*1e-2)
    omega_min = OMEGA_MIN
    ke_min = estimate_ke_cutoff_for_omega(cell, omega_min)
    a = cell.lattice_vectors()

    if mesh is None:
        nkpts = len(kpts)
        ke_cutoff = 20. * (cell.nao/25 * nkpts)**(-1./3)
        ke_cutoff = max(ke_cutoff, ke_min)
        # avoid large omega since numerical issues were found in Rys
        # polynomials when computing SR integrals with nroots > 3
        exps = [e for l, e in zip(cell._bas[:,gto.ANG_OF], cell.bas_exps()) if l != 0]
        if exps:
            omega_max = np.hstack(exps).min()**.5 * 2
            ke_max = estimate_ke_cutoff_for_omega(cell, omega_max)
            ke_cutoff = min(ke_cutoff, ke_max)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
    else:
        mesh = np.asarray(mesh)
        mesh_min = cell.cutoff_to_mesh(ke_min)
        if np.any(mesh[:cell.dimension] < mesh_min[:cell.dimension]):
            logger.warn(cell, 'mesh %s is not enough to converge to the required '
                        'integral precision %g.\nRecommended mesh is %s.',
                        mesh, cell.precision, mesh_min)
    ke_cutoff = min(pbctools.mesh_to_cutoff(a, mesh)[:cell.dimension])
    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    return omega, mesh, ke_cutoff

def _estimate_meshz(cell, precision=None):
    '''For 2D with truncated Coulomb, estimate the necessary mesh size
    that can converge the Gaussian function to the required precision.
    '''
    if precision is None:
        precision = cell.precision
    e = np.hstack(cell.bas_exps()).max()
    ke_cut = -np.log(precision) * 2 * e
    meshz = cell.cutoff_to_mesh(ke_cut)[2]
    logger.debug2(cell, '_estimate_meshz %d', meshz)
    return max(meshz, cell.mesh[2])

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
    # caused by auxiliary basis linear dependency. More details of this
    # problem can be found in function _make_j3c.
    if isinstance(mesh, (int, np.integer)):
        return (mesh // 2) * 2 + 1
    else:
        return (np.asarray(mesh) // 2) * 2 + 1

def estimate_rcut(rs_cell, rs_auxcell, omega, precision=None,
                  exclude_dd_block=False, exclude_d_aux=False):
    '''Estimate rcut for 3c2e SR-integrals'''
    if precision is None:
        # Adjust precision a little bit as errors are found slightly larger than cell.precision.
        precision = rs_cell.precision * 1e-1

    if rs_cell.nbas == 0 or rs_auxcell.nbas == 0:
        return np.zeros(1)

    if omega == 0:
        # No SR integrals in int3c2e if omega=0
        assert rs_cell.dimension == 0
        return np.zeros(1)

    cell_exps, cs = pbcgto.cell._extract_pgto_params(rs_cell, 'min')
    ls = rs_cell._bas[:,gto.ANG_OF]

    aux_exps = np.array([e.min() for e in rs_auxcell.bas_exps()])
    aux_min_idx = aux_exps.argmin()
    if exclude_d_aux:
        compact_aux_idx = np.where(rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS)[0]
        if compact_aux_idx.size > 0:
            aux_min_idx = compact_aux_idx[aux_exps[compact_aux_idx].argmin()]
    ak = aux_exps[aux_min_idx]
    lk = rs_auxcell._bas[aux_min_idx,gto.ANG_OF]

    ai_idx = cell_exps.argmin()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    li = rs_cell._bas[ai_idx,gto.ANG_OF]
    lj = ls

    ci = cs[ai_idx]
    cj = cs
    # Note ck normalizes the auxiliary basis \int \chi_k dr to 1
    ck = 1./(4*np.pi) / gto.gaussian_int(lk+2, ak)

    aij = ai + aj
    lij = li + lj
    l3 = lij + lk
    theta = 1./(omega**-2 + 1./aij + 1./ak)
    norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
    c1 = ci * cj * ck * norm_ang
    sfac = aij*aj/(aij*aj + ai*theta)
    fl = 2
    fac = 2**li*np.pi**2.5*c1 * theta**(l3-.5)
    fac *= 2*np.pi/rs_cell.vol/theta
    fac /= aij**(li+1.5) * ak**(lk+1.5) * aj**lj
    fac *= fl / precision

    r0 = rs_cell.rcut  # initial guess
    r0 = (np.log(fac * r0 * (sfac*r0)**(l3-1) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l3-1) + 1.) / (sfac*theta))**.5
    rcut = r0

    if exclude_dd_block:
        compact_mask = rs_cell.bas_type != ft_ao.SMOOTH_BASIS
        compact_idx = np.where(compact_mask)[0]
        if 0 < compact_idx.size < rs_cell.nbas:
            compact_idx = compact_idx[cell_exps[compact_idx].argmin()]
            smooth_mask = ~compact_mask
            ai = cell_exps[compact_idx]
            li = ls[compact_idx]
            ci = cs[compact_idx]
            aj = cell_exps[smooth_mask]
            lj = ls[smooth_mask]
            cj = cs[smooth_mask]

            aij = ai + aj
            lij = li + lj
            l3 = lij + lk
            theta = 1./(omega**-2 + 1./aij + 1./ak)
            norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
            c1 = ci * cj * ck * norm_ang
            sfac = aij*aj/(aij*aj + ai*theta)
            fl = 2
            fac = 2**li*np.pi**2.5*c1 * theta**(l3-.5)
            fac *= 2*np.pi/rs_cell.vol/theta
            fac /= aij**(li+1.5) * ak**(lk+1.5) * aj**lj
            fac *= fl / precision

            r0 = rs_cell.rcut
            r0 = (np.log(fac * r0 * (sfac*r0)**(l3-1) + 1.) / (sfac*theta))**.5
            r0 = (np.log(fac * r0 * (sfac*r0)**(l3-1) + 1.) / (sfac*theta))**.5
            rcut[smooth_mask] = r0
    return rcut

def estimate_ft_rcut(rs_cell, precision=None, exclude_dd_block=False):
    '''Remove less important basis based on Schwarz inequality
    Q_ij ~ S_ij * (sqrt(2aij/pi) * aij**(lij*2) * (4*lij-1)!!)**.5
    '''
    if precision is None:
        # Similar to ft_ao.estimate_rcut, adjusts precision to improve hermitian
        # symmetry of MO integrals for post-HF.
        precision = rs_cell.precision * 1e-2

    # consider only the most diffused component of a basis
    exps, cs = pbcgto.cell._extract_pgto_params(rs_cell, 'min')
    ls = rs_cell._bas[:,gto.ANG_OF]
    ai_idx = exps.argmin()
    ai = exps[ai_idx]
    li = ls[ai_idx]
    ci = cs[ai_idx]
    aj = exps
    lj = ls
    cj = cs
    aij = ai + aj
    lij = li + lj
    norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
    c1 = ci * cj * norm_ang
    theta = ai * aj / aij
    aij1 = aij**-.5
    fac = np.pi**1.5*c1 * aij1**(lij+3) * (2*aij/np.pi)**.25 * aij**lij
    fac /= precision

    r0 = rs_cell.rcut
    dri = aj*aij1 * r0 + 1.
    drj = ai*aij1 * r0 + 1.
    fl = 2*np.pi*r0/theta + 1.
    r0 = (np.log(fac * dri**li * drj**lj * fl + 1.) / theta)**.5

    dri = aj*aij1 * r0 + 1.
    drj = ai*aij1 * r0 + 1.
    fl = 2*np.pi/rs_cell.vol*r0/theta
    r0 = (np.log(fac * dri**li * drj**lj * fl + 1.) / theta)**.5
    rcut = r0

    if exclude_dd_block:
        compact_mask = rs_cell.bas_type != ft_ao.SMOOTH_BASIS
        compact_idx = np.where(compact_mask)[0]
        if 0 < compact_idx.size < rs_cell.nbas:
            compact_idx = compact_idx[exps[compact_idx].argmin()]
            smooth_mask = ~compact_mask
            ai = exps[compact_idx]
            li = ls[compact_idx]
            ci = cs[compact_idx]
            aj = exps[smooth_mask]
            lj = ls[smooth_mask]
            cj = cs[smooth_mask]
            aij = ai + aj
            lij = li + lj
            norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
            c1 = ci * cj * norm_ang
            theta = ai * aj / aij
            aij1 = aij**-.5
            fac = np.pi**1.5*c1 * aij1**(lij+3) * (2*aij/np.pi)**.25 * aij**lij
            fac /= precision

            r0 = rs_cell.rcut
            dri = aj*aij1 * r0 + 1.
            drj = ai*aij1 * r0 + 1.
            fl = 2*np.pi/rs_cell.vol*r0/theta
            r0 = (np.log(fac * dri**li * drj**lj * fl + 1.) / theta)**.5

            dri = aj*aij1 * r0 + 1.
            drj = ai*aij1 * r0 + 1.
            fl = 2*np.pi*r0/theta + 1.
            r0 = (np.log(fac * dri**li * drj**lj * fl + 1.) / theta)**.5
            rcut[smooth_mask] = r0
    return rcut

def estimate_omega_min(cell, precision=None):
    '''Given cell.rcut the boundary of repeated images of the cell, estimates
    the minimal omega for the attenuated Coulomb interactions, requiring that at
    boundary the Coulomb potential of a point charge < cutoff
    '''
    if precision is None:
        precision = cell.precision
    # erfc(z) = 2/\sqrt(pi) int_z^infty exp(-t^2) dt < exp(-z^2)/(z\sqrt(pi))
    # erfc(omega*rcut)/rcut < cutoff
    # ~ exp(-(omega*rcut)**2) / (omega*rcut**2*pi**.5) < cutoff
    rcut = cell.rcut
    omega = OMEGA_MIN
    omega = max((-np.log(precision * rcut**2 * omega))**.5 / rcut, OMEGA_MIN)
    return omega

def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
    '''Energy cutoff for AFTDF to converge attenuated Coulomb in moment space
    '''
    if precision is None:
        precision = cell.precision
    exps, cs = pbcgto.cell._extract_pgto_params(cell, 'max')
    ls = cell._bas[:,gto.ANG_OF]
    cs = gto.gto_norm(ls, exps)
    Ecut = aft._estimate_ke_cutoff(exps, ls, cs, precision, omega)
    return Ecut.max()

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulomb given energy cutoff
    '''
    if precision is None:
        precision = cell.precision
    # estimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
    # enough to converge the 2-electron integrals. A penalty term here is to
    # reduce the error in integrals
    precision *= 1e-2
    # Consider l>0 basis here to increate Ecut for slightly better accuracy
    lmax = np.max(cell._bas[:,gto.ANG_OF])
    kmax = (ke_cutoff*2)**.5
    log_rest = np.log(precision / (16*np.pi**2 * kmax**lmax))
    omega = (-.5 * ke_cutoff / log_rest)**.5
    return omega
