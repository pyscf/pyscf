#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
Build GDF tensor with compensated charges

This algorithm can handle the LR-, SR- and regular density fitting integrals
with the same framework. The RSGDF algorithms (rsdf.py rsdf_builder.py) are good
for regular density fitting and SR-integral density fitting only.
'''

import os
import copy
import ctypes
import tempfile
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger, zdotNN, zdotCN, zdotNC
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import aft
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df.incore import libpbc, Int3cBuilder
from pyscf.pbc.lib.kpts_helper import is_zero, kk_adapted_iter, KPT_DIFF_TOL
from pyscf import __config__

ETA_MIN = getattr(__config__, 'pbc_df_aft_estimate_eta_min', 0.1)

class _CCGDFBuilder(rsdf_builder._RSGDFBuilder):
    '''
    Use the compensated-charge algorithm to build Gaussian density fitting 3-center tensor
    '''
    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        self.eta = None
        self.mesh = None
        self.fused_cell = None
        self.fuse: callable = None
        self.rs_fused_cell = None
        self.supmol_ft = None

        Int3cBuilder.__init__(self, cell, auxcell, kpts)

    def has_long_range(self):
        '''Whether to add the long-range part computed with AFT integrals'''
        return self.cell.dimension > 0

    def reset(self, cell=None):
        Int3cBuilder.reset(self, cell)
        self.fused_cell = None
        self.fuse = None

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
        logger.info(self, 'eta = %s', self.eta)
        logger.info(self, 'j2c_eig_always = %s', self.j2c_eig_always)
        return self

    def build(self):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        auxcell = self.auxcell
        kpts = self.kpts

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if self.eta is None:
            self.eta, self.mesh, self.ke_cutoff = _guess_eta(auxcell, kpts, self.mesh)
        elif self.mesh is None:
            self.ke_cutoff = estimate_ke_cutoff_for_eta(cell, self.eta)
            self.mesh = cell.cutoff_to_mesh(self.ke_cutoff)
        elif self.ke_cutoff is None:
            ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            self.ke_cutoff = ke_cutoff.min()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            self.mesh[2] = rsdf_builder._estimate_meshz(cell)
        elif cell.dimension < 2:
            self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]
        self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        exp_min = np.hstack(cell.bas_exps()).min()
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff
        log.debug('Set _CCGDFBuilder.direct_scf_tol to %g', cutoff)

        self.fused_cell, self.fuse = fuse_auxcell(auxcell, self.eta)
        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, rsdf_builder.RCUT_THRESHOLD, verbose=log)
        rcut = estimate_rcut(rs_cell, self.fused_cell, rs_cell.precision,
                             self.exclude_dd_block)
        rcut_max = rcut.max()
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_max, log)
        supmol.exclude_dd_block = self.exclude_dd_block
        self.supmol = supmol.strip_basis(rcut)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        if self.has_long_range():
            rcut = rsdf_builder.estimate_ft_rcut(rs_cell, cell.precision,
                                                 self.exclude_dd_block)
            supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(rs_cell, kmesh,
                                                               rcut.max(), log)
            supmol_ft.exclude_dd_block = self.exclude_dd_block
            self.supmol_ft = supmol_ft.strip_basis(rcut)
            log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                      supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        log.timer_debug1('initializing supmol', *cpu0)
        return self

    weighted_coulG = aft.weighted_coulG

    def get_2c2e(self, uniq_kpts):
        fused_cell = self.fused_cell
        auxcell = self.auxcell
        naux = auxcell.nao
        if auxcell.dimension == 0:
            return [auxcell.intor('int2c2e', hermi=1)]

        # j2c ~ (-kpt_ji | kpt_ji)
        # Generally speaking, the int2c2e integrals with lattice sum applied on
        # |j> are not necessary hermitian because int2c2e cannot be made converged
        # with regular lattice sum unless the lattice sum vectors (from
        # cell.get_lattice_Ls) are symmetric. After adding the planewaves
        # contributions and fuse(fuse(j2c)), the output matrix is hermitian.
        j2c = list(fused_cell.pbc_intor('int2c2e', hermi=0, kpts=uniq_kpts))

        # 2c2e integrals the metric can easily cause errors in cderi tensor.
        # self.mesh may not be enough to produce required accuracy.
        # mesh = self.mesh
        precision = auxcell.precision**2
        ke = estimate_ke_cutoff_for_eta(auxcell, self.eta, precision)
        mesh = auxcell.cutoff_to_mesh(ke)
        if auxcell.dimension < 2 or auxcell.low_dim_ft_type == 'inf_vacuum':
            mesh[auxcell.dimension:] = self.mesh[auxcell.dimension:]
        mesh = self.cell.symmetrize_mesh(mesh)
        logger.debug(self, 'Set 2c2e integrals precision %g, mesh %s', precision, mesh)

        Gv, Gvbase, kws = fused_cell.get_Gv_weights(mesh)
        b = fused_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory - lib.current_memory()[0])
        blksize = max(2048, int(max_memory*.4e6/16/fused_cell.nao_nr()))
        logger.debug2(self, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)
        for k, kpt in enumerate(uniq_kpts):
            coulG = self.weighted_coulG(kpt, False, mesh)
            for p0, p1 in lib.prange(0, ngrids, blksize):
                auxG = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                auxGR = np.asarray(auxG.real, order='C')
                auxGI = np.asarray(auxG.imag, order='C')
                auxG = None

                if is_zero(kpt):  # kpti == kptj
                    j2c_p  = lib.ddot(auxGR[naux:]*coulG[p0:p1], auxGR.T)
                    j2c_p += lib.ddot(auxGI[naux:]*coulG[p0:p1], auxGI.T)
                else:
                    j2cR, j2cI = zdotCN(auxGR[naux:]*coulG[p0:p1],
                                        auxGI[naux:]*coulG[p0:p1], auxGR.T, auxGI.T)
                    j2c_p = j2cR + j2cI * 1j
                j2c[k][naux:] -= j2c_p
                j2c[k][:naux,naux:] -= j2c_p[:,:naux].conj().T
                auxGR = auxGI = j2c_p = j2cR = j2cI = None
            # Symmetrizing the matrix is not must if the integrals converged.
            # Since symmetry cannot be enforced in the pbc_intor('int2c2e'),
            # the aggregated j2c here may have error in hermitian if the range of
            # lattice sum is not big enough.
            j2c[k] = (j2c[k] + j2c[k].conj().T) * .5
            j2c[k] = self.fuse(self.fuse(j2c[k]), axis=1)
        return j2c

    def outcore_auxe2(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                      j_only=False, dataname='j3c', shls_slice=None,
                      fft_dd_block=None, kk_idx=None):
        r'''The 3-center integrals (ij|L) in real space with double lattice sum.

        Kwargs:
            shls_slice :
                Indicate the shell slices in the primitive cell
        '''
        swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
        fswap = lib.H5TmpFile(swapfile.name)
        swapfile = None

        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        fused_cell = self.fused_cell
        naux = self.auxcell.nao
        kpts = self.kpts
        nkpts = kpts.shape[0]

        gamma_point_only = is_zero(kpts)
        if gamma_point_only:
            assert nkpts == 1
            j_only = True

        intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

        if fft_dd_block is None:
            fft_dd_block = self.exclude_dd_block

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas, 0, fused_cell.nbas)
        assert shls_slice[4] == 0 and shls_slice[5] == fused_cell.nbas

        ao_loc = cell.ao_loc
        ish0, ish1, jsh0, jsh1, ksh0, ksh1 = shls_slice
        i0, i1, j0, j1 = ao_loc[list(shls_slice[:4])]
        if aosym == 's1':
            nao_pair = (i1 - i0) * (j1 - j0)
        else:
            nao_pair = i1*(i1+1)//2 - i0*(i0+1)//2

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

        # int3c2e for (cell, cell | fused_cell)
        int3c = self.gen_int3c_kernel(intor, aosym, comp, j_only,
                                      reindex_k=reindex_k, auxcell=self.fused_cell)

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

        cpu0 = logger.process_clock(), logger.perf_counter()
        nsteps = len(sh_ranges)
        row1 = 0
        for istep, (sh_start, sh_end, nrow) in enumerate(sh_ranges):
            if aosym == 's2':
                shls_slice = (sh_start, sh_end, jsh0, sh_end, ksh0, ksh1)
            else:
                shls_slice = (sh_start, sh_end, jsh0, jsh1, ksh0, ksh1)
            outR, outI = int3c(shls_slice)
            log.debug2('      step [%d/%d], shell range [%d:%d], len(buf) = %d',
                       istep+1, nsteps, sh_start, sh_end, nrow)
            cpu0 = log.timer_debug1(f'outcore_auxe2 [{istep+1}/{nsteps}]', *cpu0)

            outR = list(outR)
            if outI is not None:
                outI = list(outI)
            for k, idx in enumerate(kikj_idx):
                outR[k] = self.fuse(outR[k], axis=1)
                if f'{dataname}I/{idx}' in fswap and outI[k] is not None:
                    outI[k] = self.fuse(outI[k], axis=1)

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
        return fswap

    def weighted_ft_ao(self, kpt):
        '''exp(-i*(G + k) dot r) * Coulomb_kernel for the basis of model charge'''
        cell = self.cell
        fused_cell = self.fused_cell
        mesh = self.mesh
        Gv, Gvbase, kws = fused_cell.get_Gv_weights(mesh)
        b = fused_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            auxG = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
            naux = self.auxcell.nao
            coulG = self.weighted_coulG(kpt, False, mesh)
            with lib.temporary_env(cell, dimension=3):
                coulG_full = self.weighted_coulG(kpt, False, mesh)
            # For compensated basis, add_ft_j3c will remove its full Coulomb
            # interactions
            auxG[naux:] *= coulG_full
            # For auxbasis, in truncated Coulomb treatments, coulG_full - coulG
            # gives the trunc-Coul completion (interactions beyond truncation
            # length). add_ft_j3c function will remove this part
            auxG[:naux] *= coulG_full - coulG
        else:
            # FT for the compensated charge basis only
            shls_slice = (self.auxcell.nbas, fused_cell.nbas)
            auxG = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
            auxG *= self.weighted_coulG(kpt, False, mesh)
        Gaux = lib.transpose(auxG)
        GauxR = np.asarray(Gaux.real, order='C')
        GauxI = np.asarray(Gaux.imag, order='C')
        return GauxR, GauxI

    def gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, aosym):
        cell = self.cell
        naux = self.auxcell.nao
        nauxc = self.fused_cell.nao

        # vbar is the interaction between the background charge
        # and the auxiliary basis.  0D, 1D, 2D do not have vbar.
        vbar = None
        if (is_zero(kpt) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
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

            vbar = self.fuse(auxbar(self.fused_cell))
            vbar_idx = np.where(vbar != 0)[0]
            if len(vbar_idx) == 0:
                vbar = None
            nkpts = len(self.kpts)

        def load_j3c(col0, col1):
            j3cR = []
            j3cI = []
            ncol = col1 - col0
            for k, kk in enumerate(kpt_ij_idx):
                vR = np.empty((nauxc, ncol))
                vR[naux:] = 0
                lib.transpose(h5group[f'j3cR/{kk}'][col0:col1], out=vR)
                if f'j3cI/{kk}' in h5group:
                    vI = np.empty((nauxc, ncol))
                    vI[naux:] = 0
                    lib.transpose(h5group[f'j3cI/{kk}'][col0:col1], out=vI)
                else:
                    vI = None
                if vbar is not None:
                    kj = kk % nkpts
                    vmod = vbar[vbar_idx,None] * ovlp[kj][col0:col1]
                    vR[vbar_idx] -= vmod.real
                    if vI is not None:
                        vI[vbar_idx] -= vmod.imag
                j3cR.append(vR)
                j3cI.append(vI)
            return j3cR, j3cI

        return load_j3c

    def add_ft_j3c(self, j3c, Gpq, Gaux, p0, p1):
        cell = self.cell
        j3cR, j3cI = j3c
        GchgR = Gaux[0][p0:p1]
        GchgI = Gaux[1][p0:p1]
        nG = p1 - p0
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                GpqR = GpqR.reshape(nG, -1)
                GpqI = GpqI.reshape(nG, -1)
                lib.ddot(GchgR.T, GpqR, -1, j3cR[k], 1)
                lib.ddot(GchgI.T, GpqI, -1, j3cR[k], 1)
                if j3cI[k] is not None:
                    lib.ddot(GchgR.T, GpqI, -1, j3cI[k], 1)
                    lib.ddot(GchgI.T, GpqR,  1, j3cI[k], 1)
        else:
            naux = j3cR[0].shape[0] - GchgR.shape[1]
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                GpqR = GpqR.reshape(nG, -1)
                GpqI = GpqI.reshape(nG, -1)
                # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
                # functions |P> are assumed to be real
                lib.ddot(GchgR.T, GpqR, -1, j3cR[k][naux:], 1)
                lib.ddot(GchgI.T, GpqI, -1, j3cR[k][naux:], 1)
                if j3cI[k] is not None:
                    lib.ddot(GchgR.T, GpqI, -1, j3cI[k][naux:], 1)
                    lib.ddot(GchgI.T, GpqR,  1, j3cI[k][naux:], 1)

    def solve_cderi(self, cd_j2c, j3cR, j3cI):
        j2c, j2c_negative, j2ctag = cd_j2c
        if j3cI is None:
            j3c = self.fuse(j3cR)
        else:
            j3c = self.fuse(j3cR + j3cI * 1j)

        cderi_negative = None
        if j2ctag == 'CD':
            cderi = scipy.linalg.solve_triangular(j2c, j3c, lower=True, overwrite_b=True)
        else:
            cderi = lib.dot(j2c, j3c)
            if j2c_negative is not None:
                # for low-dimension systems
                cderi_negative = lib.dot(j2c_negative, j3c)
        return cderi, cderi_negative


class _CCNucBuilder(_CCGDFBuilder):

    exclude_dd_block = True

    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.mesh = None
        self.fused_cell = None
        self.modchg_cell = None
        self.auxcell = self.rs_auxcell = None
        Int3cBuilder.__init__(self, cell, self.auxcell, kpts)

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
        logger.info(self, 'eta = %s', self.eta)
        logger.info(self, 'j2c_eig_always = %s', self.j2c_eig_always)
        return self

    def build(self, eta=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if cell.dimension == 0:
            self.eta, self.mesh, self.ke_cutoff = _guess_eta(cell, kpts, self.mesh)
        else:
            if eta is None:
                eta = max(.5/(.5+nkpts**(1./9)), ETA_MIN)
            ke_cutoff = estimate_ke_cutoff_for_eta(cell, eta)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
            self.ke_cutoff = min(pbctools.mesh_to_cutoff(
                cell.lattice_vectors(), self.mesh)[:cell.dimension])
            self.eta = estimate_eta_for_ke_cutoff(cell, self.ke_cutoff)
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                self.mesh[2] = rsdf_builder._estimate_meshz(cell)
            elif cell.dimension < 2:
                self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]
            self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        self.modchg_cell = _compensate_nuccell(cell, self.eta)
        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, rsdf_builder.RCUT_THRESHOLD, verbose=log)
        rcut = estimate_rcut(rs_cell, self.modchg_cell,
                             exclude_dd_block=self.exclude_dd_block)
        rcut_max = rcut.max()
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_max, log)
        supmol.exclude_dd_block = self.exclude_dd_block
        self.supmol = supmol.strip_basis(rcut)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        exp_min = np.hstack(cell.bas_exps()).min()
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff / cell.atom_charges().max()
        log.debug('Set _CCNucBuilder.direct_scf_tol to %g', cutoff)

        rcut = rsdf_builder.estimate_ft_rcut(rs_cell, cell.precision,
                                             self.exclude_dd_block)
        supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(rs_cell, kmesh,
                                                           rcut.max(), log)
        supmol_ft.exclude_dd_block = self.exclude_dd_block
        self.supmol_ft = supmol_ft.strip_basis(rcut)
        log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                  supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        log.timer_debug1('initializing supmol', *cpu0)
        return self

    def _int_nuc_vloc(self, fakenuc, intor='int3c2e', aosym='s2', comp=None,
                      supmol=None):
        '''Vnuc - Vloc.
        '''
        logger.debug2(self, 'Real space integrals %s for Vnuc - Vloc', intor)

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        charge = -cell.atom_charges()
        if cell.dimension > 0:
            mod_cell = self.modchg_cell
            fakenuc = copy.copy(fakenuc)
            fakenuc._atm, fakenuc._bas, fakenuc._env = \
                    gto.conc_env(mod_cell._atm, mod_cell._bas, mod_cell._env,
                                 fakenuc._atm, fakenuc._bas, fakenuc._env)
            charge = np.append(-charge, charge)

        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc, supmol=supmol)
        bufR, bufI = int3c()

        if is_zero(kpts):
            mat = np.einsum('k...z,z->k...', bufR, charge)
        else:
            mat = (np.einsum('k...z,z->k...', bufR, charge) +
                   np.einsum('k...z,z->k...', bufI, charge) * 1j)

        # vbar is the interaction between the background charge
        # and the compensating function.  0D, 1D, 2D do not have vbar.
        if ((intor in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart')) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            logger.debug2(self, 'G=0 part for %s', intor)

            # Note only need to remove the G=0 for mod_cell. when fakenuc is
            # constructed for pseudo potentail, don't remove its G=0 contribution
            charge = -cell.atom_charges()
            nucbar = (charge / np.hstack(mod_cell.bas_exps())).sum()
            nucbar *= np.pi/cell.vol
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

    _int_dd_block = rsdf_builder._int_dd_block

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
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        charges = -cell.atom_charges()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            with lib.temporary_env(cell, dimension=3):
                coulG_full = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            aoaux = ft_ao.ft_ao(self.modchg_cell, Gv, None, b, gxyz, Gvbase)
            vG = np.einsum('i,xi,x->x', charges, aoaux, coulG_full * kws)
            aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
            vG += np.einsum('i,xi,x->x', charges, aoaux, (coulG-coulG_full)*kws)
        else:
            aoaux = ft_ao.ft_ao(self.modchg_cell, Gv, None, b, gxyz, Gvbase)
            coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            vG = np.einsum('i,xi,x->x', charges, aoaux, coulG * kws)

        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                               verbose=log)
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
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, out=buf)
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # rho_ij(G) nuc(-G) / G^2
                # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
                vR = np.einsum('k,kx->x', vGR[p0:p1], GpqR)
                vR+= np.einsum('k,kx->x', vGI[p0:p1], GpqI)
                vj[k] += vR
                if not is_zero(kpts[k]):
                    vI = np.einsum('k,kx->x', vGR[p0:p1], GpqI)
                    vI-= np.einsum('k,kx->x', vGI[p0:p1], GpqR)
                    vj[k] += vI * 1j
            t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
        log.timer_debug1('contracting Vnuc', *t0)

        vj_kpts = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_kpts.append(lib.unpack_tril(vj[k].real))
            else:
                vj_kpts.append(lib.unpack_tril(vj[k]))
        return np.asarray(vj_kpts)

    get_nuc = rsdf_builder.get_nuc
    get_pp = rsdf_builder.get_pp


def auxbar(fused_cell):
    r'''
    Potential average = \sum_L V_L*Lpq

    The coulomb energy is computed with chargeless density
    \int (rho-C) V,  C = (\int rho) / vol = Tr(gamma,S)/vol
    It is equivalent to removing the averaged potential from the short range V
    vs = vs - (\int V)/vol * S
    '''
    aux_loc = fused_cell.ao_loc_nr()
    naux = aux_loc[-1]
    vbar = np.zeros(naux)
    # SR ERI should not have contributions from backgound charge
    if fused_cell.dimension < 2 or fused_cell.omega < 0:
        return vbar

    half_sph_norm = .5/np.sqrt(np.pi)
    for i in range(fused_cell.nbas):
        l = fused_cell.bas_angular(i)
        if l == 0:
            es = fused_cell.bas_exp(i)
            if es.size == 1:
                vbar[aux_loc[i]] = -1/es[0]
            else:
                # Remove the normalization to get the primitive contraction coeffcients
                norms = half_sph_norm/gto.gaussian_int(2, es)
                cs = np.einsum('i,ij->ij', 1/norms, fused_cell._libcint_ctr_coeff(i))
                vbar[aux_loc[i]:aux_loc[i+1]] = np.einsum('in,i->n', cs, -1/es)
    # TODO: fused_cell.cart and l%2 == 0: # 6d 10f ...
    # Normalization coefficients are different in the same shell for cartesian
    # basis. E.g. the d-type functions, the 5 d-type orbitals are normalized wrt
    # the integral \int r^2 * r^2 e^{-a r^2} dr.  The s-type 3s orbital should be
    # normalized wrt the integral \int r^0 * r^2 e^{-a r^2} dr. The different
    # normalization was not built in the basis.
    vbar *= np.pi/fused_cell.vol
    return vbar

def make_modchg_basis(auxcell, smooth_eta):
    # * chgcell defines smooth gaussian functions for each angular momentum for
    #   auxcell. The smooth functions may be used to carry the charge
    chgcell = copy.copy(auxcell)  # smooth model density for coulomb integral to carry charge
    half_sph_norm = .5/np.sqrt(np.pi)
    chg_bas = []
    chg_env = [smooth_eta]
    ptr_eta = auxcell._env.size
    ptr = ptr_eta + 1
    l_max = auxcell._bas[:,gto.ANG_OF].max()
# gaussian_int(l*2+2) for multipole integral:
# \int (r^l e^{-ar^2} * Y_{lm}) (r^l Y_{lm}) r^2 dr d\Omega
    norms = [half_sph_norm/gto.gaussian_int(l*2+2, smooth_eta)
             for l in range(l_max+1)]
    for ia in range(auxcell.natm):
        for l in set(auxcell._bas[auxcell._bas[:,gto.ATOM_OF]==ia, gto.ANG_OF]):
            chg_bas.append([ia, l, 1, 1, 0, ptr_eta, ptr, 0])
            chg_env.append(norms[l])
            ptr += 1

    chgcell._atm = auxcell._atm
    chgcell._bas = np.asarray(chg_bas, dtype=np.int32).reshape(-1,gto.BAS_SLOTS)
    chgcell._env = np.hstack((auxcell._env, chg_env))

    # chgcell.rcut needs to ensure the model charges are well separated such
    # that the Coulomb interaction between the compensated auxiliary basis can
    # be calculated as 1/Rcut.
    # _estimate_rcut based on the integral overlap
    chgcell.rcut = pbcgto.cell._estimate_rcut(smooth_eta, l_max, 1., auxcell.precision)

    logger.debug1(auxcell, 'make compensating basis, num shells = %d, num cGTOs = %d',
                  chgcell.nbas, chgcell.nao_nr())
    logger.debug1(auxcell, 'chgcell.rcut %s', chgcell.rcut)
    return chgcell

def fuse_auxcell(auxcell, eta):
    if auxcell.dimension == 0:
        def fuse(Lpq, axis=0):
            return Lpq
        return auxcell, fuse

    chgcell = make_modchg_basis(auxcell, eta)
    fused_cell = copy.copy(auxcell)
    fused_cell._atm, fused_cell._bas, fused_cell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)
    fused_cell.rcut = max(auxcell.rcut, chgcell.rcut)

    aux_loc = auxcell.ao_loc_nr()
    naux = aux_loc[-1]
    modchg_offset = -np.ones((chgcell.natm,8), dtype=int)
    smooth_loc = chgcell.ao_loc_nr()
    for i in range(chgcell.nbas):
        ia = chgcell.bas_atom(i)
        l  = chgcell.bas_angular(i)
        modchg_offset[ia,l] = smooth_loc[i]

    if auxcell.cart:
        # Normalization coefficients are different in the same shell for cartesian
        # basis. E.g. the d-type functions, the 5 d-type orbitals are normalized wrt
        # the integral \int r^2 * r^2 e^{-a r^2} dr.  The s-type 3s orbital should be
        # normalized wrt the integral \int r^0 * r^2 e^{-a r^2} dr. The different
        # normalization was not built in the basis.  There two ways to surmount this
        # problem.  First is to transform the cartesian basis and scale the 3s (for
        # d functions), 4p (for f functions) ... then transform back. The second is to
        # remove the 3s, 4p functions. The function below is the second solution
        c2s_fn = gto.moleintor.libcgto.CINTc2s_ket_sph
        aux_loc_sph = auxcell.ao_loc_nr(cart=False)
        naux_sph = aux_loc_sph[-1]
        def fuse(Lpq, axis=0):
            if axis == 1 and Lpq.ndim == 2:
                Lpq = lib.transpose(Lpq)
            Lpq, chgLpq = Lpq[:naux], Lpq[naux:]
            if Lpq.ndim == 1:
                npq = 1
                Lpq_sph = np.empty(naux_sph, dtype=Lpq.dtype)
            else:
                npq = Lpq.shape[1]
                Lpq_sph = np.empty((naux_sph,npq), dtype=Lpq.dtype)
            if Lpq.dtype == np.complex128:
                npq *= 2  # c2s_fn supports double only, *2 to handle complex
            for i in range(auxcell.nbas):
                l  = auxcell.bas_angular(i)
                ia = auxcell.bas_atom(i)
                p0 = modchg_offset[ia,l]
                if p0 >= 0:
                    nd = (l+1) * (l+2) // 2
                    c0, c1 = aux_loc[i], aux_loc[i+1]
                    s0, s1 = aux_loc_sph[i], aux_loc_sph[i+1]
                    for i0, i1 in lib.prange(c0, c1, nd):
                        Lpq[i0:i1] -= chgLpq[p0:p0+nd]

                    if l < 2:
                        Lpq_sph[s0:s1] = Lpq[c0:c1]
                    else:
                        Lpq_cart = np.asarray(Lpq[c0:c1], order='C')
                        c2s_fn(Lpq_sph[s0:s1].ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_int(npq * auxcell.bas_nctr(i)),
                               Lpq_cart.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_int(l))
            if axis == 1 and Lpq.ndim == 2:
                Lpq_sph = lib.transpose(Lpq_sph)
            return Lpq_sph
    else:
        def fuse(Lpq, axis=0):
            if axis == 1 and Lpq.ndim == 2:
                Lpq = lib.transpose(Lpq)
            Lpq, chgLpq = Lpq[:naux], Lpq[naux:]
            for i in range(auxcell.nbas):
                l  = auxcell.bas_angular(i)
                ia = auxcell.bas_atom(i)
                p0 = modchg_offset[ia,l]
                if p0 >= 0:
                    nd = l * 2 + 1
                    for i0, i1 in lib.prange(aux_loc[i], aux_loc[i+1], nd):
                        Lpq[i0:i1] -= chgLpq[p0:p0+nd]
            if axis == 1 and Lpq.ndim == 2:
                Lpq = lib.transpose(Lpq)
            return np.asarray(Lpq, order='A')
    return fused_cell, fuse

def _guess_eta(cell, kpts=None, mesh=None):
    '''Search for optimal eta and mesh'''
    if cell.dimension == 0:
        if mesh is None:
            mesh = cell.mesh
        ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh).min()
        eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
        return eta, mesh, ke_cutoff

    # eta_min = estimate_eta_min(cell, cell.precision*1e-2)
    eta_min = ETA_MIN
    ke_min = estimate_ke_cutoff_for_eta(cell, eta_min, cell.precision)
    a = cell.lattice_vectors()

    if mesh is None:
        nkpts = len(kpts)
        ke_cutoff = 30. * nkpts**(-1./3)
        ke_cutoff = max(ke_cutoff, ke_min)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
    else:
        mesh = np.asarray(mesh)
        mesh_min = cell.cutoff_to_mesh(ke_min)
        if np.any(mesh[:cell.dimension] < mesh_min[:cell.dimension]):
            logger.warn(cell, 'mesh %s is not enough to converge to the required '
                        'integral precision %g.\nRecommended mesh is %s.',
                        mesh, cell.precision, mesh_min)
    ke_cutoff = min(pbctools.mesh_to_cutoff(a, mesh)[:cell.dimension])
    eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    return eta, mesh, ke_cutoff

def _compensate_nuccell(cell, eta):
    '''A cell of the compensated Gaussian charges for nucleus'''
    modchg_cell = copy.copy(cell)
    half_sph_norm = .5/np.sqrt(np.pi)
    norm = half_sph_norm/gto.gaussian_int(2, eta)
    chg_env = [eta, norm]
    ptr_eta = cell._env.size
    ptr_norm = ptr_eta + 1
    chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
    modchg_cell._atm = cell._atm
    modchg_cell._bas = np.asarray(chg_bas, dtype=np.int32)
    modchg_cell._env = np.hstack((cell._env, chg_env))
    return modchg_cell

def estimate_rcut(rs_cell, auxcell, precision=None, exclude_dd_block=False):
    '''Estimate rcut for 3c2e integrals'''
    if precision is None:
        precision = rs_cell.precision

    if rs_cell.nbas == 0 or auxcell.nbas == 0:
        return np.zeros(1)

    cell_exps, cs = pbcgto.cell._extract_pgto_params(rs_cell, 'min')
    ls = rs_cell._bas[:,gto.ANG_OF]

    aux_exps = np.array([e.min() for e in auxcell.bas_exps()])
    ai_idx = cell_exps.argmin()
    ak_idx = aux_exps.argmin()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    ak = aux_exps[ak_idx]
    li = rs_cell._bas[ai_idx,gto.ANG_OF]
    lj = ls
    lk = auxcell._bas[ak_idx,gto.ANG_OF]

    ci = cs[ai_idx]
    cj = cs
    # Note ck normalizes the auxiliary basis \int \chi_k dr to 1
    ck = 1./(4*np.pi) / gto.gaussian_int(lk+2, ak)

    aij = ai + aj
    lij = li + lj
    l3 = lij + lk
    theta = 1./(1./aij + 1./ak)
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
            theta = 1./(1./aij + 1./ak)
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

def estimate_eta_min(cell, precision=None):
    '''Given rcut the boundary of repeated images of the cell, estimates the
    minimal exponent of the smooth compensated gaussian model charge, requiring
    that at boundary, density ~ 4pi rmax^2 exp(-eta/2*rmax^2) < precision
    '''
    if precision is None:
        precision = cell.precision
    lmax = min(np.max(cell._bas[:,gto.ANG_OF]), 4)
    # If lmax=3 (r^5 for radial part), this expression guarantees at least up
    # to f shell the convergence at boundary
    rcut = cell.rcut
    eta = max(np.log(4*np.pi*rcut**(lmax+2)/precision)/rcut**2, ETA_MIN)
    return eta

def estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''Given ke_cutoff, the upper bound of eta to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    if precision is None:
        precision = cell.precision
    ai = np.hstack(cell.bas_exps()).max()
    aij = ai * 2
    ci = gto.gto_norm(0, ai)
    norm_ang = (4*np.pi)**-1.5
    c1 = ci**2 * norm_ang
    fac = 64*np.pi**5*c1 * (aij*ke_cutoff*2)**-.5 / precision

    eta = 4.
    eta = 1./(np.log(fac * eta**-1.5)*2 / ke_cutoff - 1./aij)
    if eta < 0:
        eta = 4.
    else:
        eta = min(4., eta)
    return eta

def estimate_ke_cutoff_for_eta(cell, eta, precision=None):
    '''Given eta, the lower bound of ke_cutoff to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    if precision is None:
        precision = cell.precision
    ai = np.hstack(cell.bas_exps()).max()
    aij = ai * 2
    ci = gto.gto_norm(0, ai)
    ck = gto.gto_norm(0, eta)
    theta = 1./(1./aij + 1./eta)
    Norm_ang = (4*np.pi)**-1.5
    fac = 32*np.pi**5 * ci**2*ck*Norm_ang * (2*aij) / (aij*eta)**1.5
    fac /= precision

    Ecut = 20.
    Ecut = np.log(fac * (Ecut*2)**(-.5)) * 2*theta
    Ecut = np.log(fac * (Ecut*2)**(-.5)) * 2*theta
    return Ecut
