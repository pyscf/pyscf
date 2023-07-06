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

'''
Gaussian and planewaves mixed density fitting
Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import tempfile
import numpy as np
import h5py
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger, zdotCN
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import gto
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df
from pyscf.pbc.df import aft
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.incore import libpbc, make_auxcell
from pyscf.pbc.lib.kpts_helper import is_zero, member, unique
from pyscf.pbc.df import mdf_jk
from pyscf.pbc.df import mdf_ao2mo
from pyscf.pbc.df.aft import _sub_df_jk_
from pyscf import __config__


class MDF(df.GDF):
    '''Gaussian and planewaves mixed density fitting
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts  # default is gamma point
        self.kpts_band = None
        self._auxbasis = None

        # In MDF, fitting PWs (self.mesh), and parameters eta and exp_to_discard
        # are related to each other. The compensated function does not need to
        # be very smooth. It just needs to be expanded by the specified PWs
        # (self.mesh). self.eta is estimated on the fly based on the value of
        # self.mesh.
        self.eta = None
        self.mesh = None

        # Any functions which are more diffused than the compensated Gaussian
        # are linearly dependent to the PWs. They can be removed from the
        # auxiliary set without affecting the accuracy of MDF. exp_to_discard
        # can be set to the value of self.eta
        self.exp_to_discard = None

        # tends to call _CCMDFBuilder if applicable
        self._prefer_ccdf = False

        # TODO: More tests are needed
        self.time_reversal_symmetry = False

        # The following attributes are not input options.
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.blockdim = getattr(__config__, 'df_df_DF_blockdim', 240)
        self.linear_dep_threshold = df.LINEAR_DEP_THR
        self._j_only = False
# If _cderi_to_save is specified, the 3C-integral tensor will be saved in this file.
        self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
# If _cderi is specified, the 3C-integral tensor will be read from this file
        self._cderi = None
        self._rsh_df = {}  # Range separated Coulomb DF objects
        self._keys = set(self.__dict__.keys())

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        df.GDF.build(self, j_only, with_j3c, kpts_band)
        cell = self.cell
        if any(x % 2 == 0 for x in self.mesh[:cell.dimension]):
            # Even number in mesh can produce planewaves without couterparts
            # (see np.fft.fftfreq). MDF mesh is typically not enough to capture
            # all basis. The singular planewaves can break the symmetry in
            # potential (leads to non-real density) and thereby break the
            # hermitian of J and K matrices
            logger.warn(self, 'MDF with even number in mesh may have significant errors')
        return self

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None, cderi_file=None):
        if cell is None: cell = self.cell
        if auxcell is None: auxcell = self.auxcell
        if cderi_file is None: cderi_file = self._cderi_to_save

        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack([self.kpts, self.kpts_band]))[0]

        if self._prefer_ccdf or cell.omega > 0:
            # For long-range integrals _CCMDFBuilder is the only option
            dfbuilder = _CCMDFBuilder(cell, auxcell, kpts_union)
        else:
            dfbuilder = _RSMDFBuilder(cell, auxcell, kpts_union)
            dfbuilder.eta = self.eta
        dfbuilder.mesh = self.mesh
        dfbuilder.linear_dep_threshold = self.linear_dep_threshold
        j_only = self._j_only or len(kpts_union) == 1
        dfbuilder.make_j3c(cderi_file, j_only=j_only, dataname=self._dataname,
                           kptij_lst=kptij_lst)

        # mdf.mesh must be the mesh to generate cderi
        self.mesh = dfbuilder.mesh

    get_pp = df.GDF.get_pp
    get_nuc = df.GDF.get_nuc

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            cell = self.cell
            # * AFT is computationally more efficient than MDF if the Coulomb
            #   attenuation tends to the long-range role (i.e. small omega).
            # * Note: changing to AFT integrator may cause small difference to
            #   the MDF integrator. If a very strict MDF result is desired,
            #   we can disable this trick by setting
            #   LONGRANGE_AFT_TURNOVER_THRESHOLD to 0.
            # * The sparse mesh is not appropriate for low dimensional systems
            #   with infinity vacuum since the ERI may require large mesh to
            #   sample density in vacuum.
            if (omega < df.LONGRANGE_AFT_TURNOVER_THRESHOLD and
                cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum'):
                mydf = aft.AFTDF(cell, self.kpts)
                ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = cell.cutoff_to_mesh(ke_cutoff)
            else:
                mydf = self
            with mydf.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            return mdf_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
                                 with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = mdf_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = mdf_ao2mo.get_eri
    ao2mo = get_mo_eri = mdf_ao2mo.general
    ao2mo_7d = mdf_ao2mo.ao2mo_7d

    def update_mp(self):
        raise NotImplementedError

    def update_cc(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

################################################################################
# With this function to mimic the molecular DF.loop function, the pbc gamma
# point DF object can be used in the molecular code
    def loop(self, blksize=None):
        for dat in aft.AFTDF.loop(self, blksize):
            yield dat
        for dat in df.DF.loop(self, blksize):
            yield dat

    def get_naoaux(self):
        return df.DF.get_naoaux(self) + aft.AFTDF.get_naoaux(self)


class _RSMDFBuilder(_RSGDFBuilder):
    '''
    Use the range-separated algorithm to build mixed density fitting 3-center tensor
    '''
    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        _RSGDFBuilder.__init__(self, cell, auxcell, kpts)

        # For MDF, large difference may be found in results between the CD/ED
        # treatments. In some systems, small integral errors can lead to a
        # differnece in the total energy/orbital energy around 4th decimal
        # place. Abandon CD treatment for better numerical stability
        self.j2c_eig_always = True

    def has_long_range(self):
        return True

    def get_2c2e(self, uniq_kpts):
        # The basis for MDF are planewaves {G} and orthogonal gaussians
        # {|g> - |G><G|g>}. computing j2c for orthogonal gaussians here:
        #    <g|g> - 2 <g|G><G|g> + <g|G><G|G><G|g> = <g|g> - <g|G><G|g>
        auxcell = self.auxcell
        omega = self.omega
        rs_auxcell = self.rs_auxcell
        auxcell_c = rs_auxcell.compact_basis_cell()
        if auxcell_c.nbas > 0:
            with auxcell_c.with_short_range_coulomb(omega):
                sr_j2c = list(auxcell_c.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts))

            compact_bas_idx = np.where(rs_auxcell.bas_type != ft_ao.SMOOTH_BASIS)[0]
            ao_map = auxcell.get_ao_indices(rs_auxcell.bas_map[compact_bas_idx])

            def recontract_2d(j2c, j2c_cc):
                return lib.takebak_2d(j2c, j2c_cc, ao_map, ao_map, thread_safe=False)
        else:
            sr_j2c = None

        mesh = self.mesh
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
            if is_zero(kpt):  # kpti == kptj
                j2c_k = np.zeros((naux, naux))
            else:
                j2c_k = np.zeros((naux, naux), dtype=np.complex128)

            if sr_j2c is not None:
                # coulG_sr here to first remove the FT-SR-2c2e for compact basis
                # from the analytical 2c2e integrals. The FT-SR-2c2e for compact
                # basis is added back in j2c_k.
                coulG_sr = self.weighted_coulG_SR(kpt, False, mesh)
                if auxcell.dimension == 3 and is_zero(kpt):
                    G0_idx = 0  # due to np.fft.fftfreq convention
                    G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
                    coulG_sr[G0_idx] += np.pi/omega**2 * G0_weight

                for p0, p1 in lib.prange(0, ngrids, blksize):
                    auxG_sr = ft_ao.ft_ao(auxcell_c, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                    if is_zero(kpt):
                        sr_j2c[k] -= lib.dot(auxG_sr.conj() * coulG_sr[p0:p1], auxG_sr.T).real
                    else:
                        sr_j2c[k] -= lib.dot(auxG_sr.conj() * coulG_sr[p0:p1], auxG_sr.T)
                    auxG_sr = None

                j2c_k = recontract_2d(j2c_k, sr_j2c[k])
                sr_j2c[k] = None

            j2c.append(j2c_k)
        return j2c

    def outcore_auxe2(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                      j_only=False, dataname='j3c', shls_slice=None,
                      fft_dd_block=False, kk_idx=None):
        # dd_block from real-space integrals will be cancelled by AFT part
        # anyway. It's safe to omit dd_block when computing real-space int3c2e
        return super().outcore_auxe2(cderi_file, intor, aosym, comp, j_only,
                                     dataname, shls_slice, fft_dd_block, kk_idx)

    def weighted_ft_ao(self, kpt):
        '''exp(-i*(G + k) dot r) * Coulomb_kernel'''
        rs_cell = self.rs_cell
        Gv, Gvbase, kws = rs_cell.get_Gv_weights(self.mesh)
        b = rs_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        coulG_SR = self.weighted_coulG_SR(kpt, False, self.mesh)

        if self.exclude_d_aux:
            # The smooth basis in auxcell was excluded in outcore_auxe2.
            # Full Coulomb kernel needs to be applied for the smooth basis
            rs_auxcell = self.rs_auxcell
            smooth_aux_mask = rs_auxcell.get_ao_type() == ft_ao.SMOOTH_BASIS
            auxG = ft_ao.ft_ao(rs_auxcell, Gv, None, b, gxyz, Gvbase, kpt).T
            auxG[smooth_aux_mask] = 0
            auxG[~smooth_aux_mask] *= -coulG_SR
            auxG = rs_auxcell.recontract_1d(auxG)
        else:
            auxcell = self.auxcell
            auxG = ft_ao.ft_ao(auxcell, Gv, None, b, gxyz, Gvbase, kpt).T
            auxG *= -coulG_SR
        Gaux = lib.transpose(auxG)
        GauxR = np.asarray(Gaux.real, order='C')
        GauxI = np.asarray(Gaux.imag, order='C')
        return GauxR, GauxI


class _CCMDFBuilder(_CCGDFBuilder):
    '''
    Use the compensated-charge algorithm to build mixed density fitting 3-center tensor
    '''
    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        _CCGDFBuilder.__init__(self, cell, auxcell, kpts)

        # For MDF, large difference may be found in results between the CD/ED
        # treatments. In some systems, small integral errors can lead to a
        # differnece in the total energy/orbital energy around 4th decimal
        # place. Abandon CD treatment for better numerical stability
        self.j2c_eig_always = True

    def has_long_range(self):
        return True

    def get_2c2e(self, uniq_kpts):
        # The basis for MDF are planewaves {G} and orthogonal gaussians
        # {|g> - |G><G|g>}. computing j2c for orthogonal gaussians here:
        #    <g|g> - 2 <g|G><G|g> + <g|G><G|G><G|g> = <g|g> - <g|G><G|g>
        fused_cell = self.fused_cell

        # j2c ~ (-kpt_ji | kpt_ji)
        j2c = list(fused_cell.pbc_intor('int2c2e', hermi=0, kpts=uniq_kpts))

        Gv, Gvbase, kws = fused_cell.get_Gv_weights(self.mesh)
        b = fused_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory - lib.current_memory()[0])
        blksize = max(2048, int(max_memory*.4e6/16/fused_cell.nao_nr()))
        logger.debug2(self, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)
        for k, kpt in enumerate(uniq_kpts):
            j2c_k = self.fuse(self.fuse(j2c[k]), axis=1)
            j2c_k = np.asarray((j2c_k + j2c_k.conj().T) * .5, order='C')

            coulG = self.weighted_coulG(kpt, False, self.mesh)
            for p0, p1 in lib.prange(0, ngrids, blksize):
                auxG = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                auxG = self.fuse(auxG)
                if is_zero(kpt):  # kpti == kptj
                    j2c_k -= lib.dot(auxG.conj()*coulG[p0:p1], auxG.T).real
                else:
                    j2c_k -= lib.dot(auxG.conj()*coulG[p0:p1], auxG.T)
                auxG = None
            j2c[k] = j2c_k
        return j2c


    def outcore_auxe2(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                      j_only=False, dataname='j3c', shls_slice=None,
                      fft_dd_block=False, kk_idx=None):
        return super().outcore_auxe2(cderi_file, intor, aosym, comp, j_only,
                                     dataname, shls_slice, fft_dd_block, kk_idx)

    def weighted_ft_ao(self, kpt):
        fused_cell = self.fused_cell
        Gv, Gvbase, kws = fused_cell.get_Gv_weights(self.mesh)
        b = fused_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        auxG = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        auxG = self.fuse(auxG)
        auxG *= self.weighted_coulG(kpt, False, self.mesh)
        Gaux = lib.transpose(auxG)
        GauxR = np.asarray(Gaux.real, order='C')
        GauxI = np.asarray(Gaux.imag, order='C')
        return GauxR, GauxI

    def gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, aosym):
        gdf_load = _CCGDFBuilder.gen_j3c_loader(self, h5group, kpt, kpt_ij_idx, aosym)
        naux = self.auxcell.nao
        def load_j3c(col0, col1):
            j3cR, j3cI = gdf_load(col0, col1)
            j3cR = [vR[:naux] for vR in j3cR]
            j3cI = [vI[:naux] if vI is not None else None for vI in j3cI]
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
            lib.ddot(GauxR.T, GpqR, -1, j3cR[k], 1)
            lib.ddot(GauxI.T, GpqI, -1, j3cR[k], 1)
            if j3cI[k] is not None:
                lib.ddot(GauxR.T, GpqI, -1, j3cI[k], 1)
                lib.ddot(GauxI.T, GpqR,  1, j3cI[k], 1)

    def solve_cderi(self, cd_j2c, j3cR, j3cI):
        j2c, j2c_negative, j2ctag = cd_j2c
        if j3cI is None:
            j3c = j3cR
        else:
            j3c = j3cR + j3cI * 1j

        cderi = lib.dot(j2c, j3c)
        if j2c_negative is not None:
            # for low-dimension systems
            cderi_negative = lib.dot(j2c_negative, j3c)
        else:
            cderi_negative = None
        return cderi, cderi_negative
