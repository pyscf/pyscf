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

import os

import tempfile
import numpy as np
import h5py
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger, zdotCN
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df
from pyscf.pbc.df import aft
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder, _round_off_to_odd_mesh
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.incore import libpbc, make_auxcell, _Int3cBuilder, _ExtendedMole
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
        self.mesh = _mesh_for_valence(cell)

        # In MDF, fitting PWs (self.mesh), and parameters eta and exp_to_discard
        # are related to each other. The compensated function does not need to
        # be very smooth. It just needs to be expanded by the specified PWs
        # (self.mesh). self.eta is estimated on the fly based on the value of
        # self.mesh.
        self.eta = None

        # Any functions which are more diffused than the compensated Gaussian
        # are linearly dependent to the PWs. They can be removed from the
        # auxiliary set without affecting the accuracy of MDF. exp_to_discard
        # can be set to the value of self.eta
        self.exp_to_discard = None

        # tends to call _CCMDFBuilder if applicable
        self._prefer_ccdf = False

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

    @property
    def eta(self):
        if self._eta is not None:
            return self._eta
        else:
            cell = self.cell
            if cell.dimension == 0:
                return 0.2
            ke_cutoff = tools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            ke_cutoff = ke_cutoff[:cell.dimension].min()
            return aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    @eta.setter
    def eta(self, x):
        self._eta = x

    @property
    def exp_to_discard(self):
        if self._exp_to_discard is not None:
            return self._exp_to_discard
        else:
            return self.eta
    @exp_to_discard.setter
    def exp_to_discard(self, x):
        self._exp_to_discard = x

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
            dfbuilder = _CCMDFBuilder(cell, auxcell, kpts_union).set(
                mesh=self.mesh,
                linear_dep_threshold=self.linear_dep_threshold,
            )
        else:
            dfbuilder = _RSMDFBuilder(cell, auxcell, kpts_union).set(
                mesh=self.mesh,
                linear_dep_threshold=self.linear_dep_threshold,
            )

        if len(kpts_union) == 1 or self._j_only:
            dfbuilder.make_j3c(cderi_file, aosym='s2', j_only=self._j_only)
        else:
            dfbuilder.make_j3c(cderi_file, aosym='s1', j_only=self._j_only)

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
                mydf.ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), mydf.ke_cutoff)
            else:
                mydf = self
            return _sub_df_jk_(mydf, dm, hermi, kpts, kpts_band,
                               with_j, with_k, omega, exxdiv)

        if kpts is None:
            if np.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = np.zeros(3)
            else:
                kpts = self.kpts
        kpts = np.asarray(kpts)

        if kpts.shape == (3,):
            return mdf_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
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


# valence_exp = 1. are typically the Gaussians in the valence
VALENCE_EXP = getattr(__config__, 'pbc_df_mdf_valence_exp', 1.0)
def _mesh_for_valence(cell, valence_exp=VALENCE_EXP):
    '''Energy cutoff estimation'''
    precision = cell.precision * 10
    Ecut_max = 0
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i).copy()
        es[es>valence_exp] = valence_exp
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        ke_guess = gto.cell._estimate_ke_cutoff(es, l, cs, precision)
        Ecut_max = max(Ecut_max, ke_guess.max())
    mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), Ecut_max)
    mesh = np.min((mesh, cell.mesh), axis=0)
    if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
        mesh[cell.dimension:] = cell.mesh[cell.dimension:]
    return _round_off_to_odd_mesh(mesh)
del(VALENCE_EXP)


def _outcore_auxe2(dfbuilder, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                   j_only=False, dataname='j3c', shls_slice=None):
    r'''The SR part of 3-center integrals (ij|L) with double lattice sum.

    Kwargs:
        shls_slice :
            Indicate the shell slices in the primitive cell
    '''
    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash files
    swapfile = None

    log = logger.new_logger(dfbuilder)
    cell = dfbuilder.cell
    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
    int3c = dfbuilder.gen_int3c_kernel(intor, aosym, comp, j_only)

    auxcell = dfbuilder.auxcell
    naux = auxcell.nao
    kpts = dfbuilder.kpts
    nkpts = kpts.shape[0]

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    i0, i1, j0, j1 = [ao_loc[i] for i in shls_slice[:4]]
    k0, k1 = aux_loc[shls_slice[4]],  aux_loc[shls_slice[5]]
    if aosym == 's1':
        nao_pair = (i1 - i0) * (j1 - j0)
    else:
        nao_pair = i1*(i1+1)//2 - i0*(i0+1)//2
    naux = k1 - k0

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

    if naux == 0:
        return fswap

    mem_now = lib.current_memory()[0]
    log.debug2('memory = %s', mem_now)
    max_memory = max(2000, dfbuilder.max_memory-mem_now)

    # split the 3-center tensor (nkpts_ij, i, j, aux) along shell i.
    # plus 1 to ensure the intermediates in libpbc do not overflow
    buflen = min(max(int(max_memory*.9e6/16/naux/(nkpts_ij+1)), 1), nao_pair)
    # lower triangle part
    sh_ranges = _guess_shell_ranges(cell, buflen, aosym,
                                    start=shls_slice[0], stop=shls_slice[1])
    max_buflen = max([x[2] for x in sh_ranges])
    if max_buflen > buflen:
        log.warn('memory usage of outcore_auxe2 may be '
                 f'{(max_buflen/buflen - 1):.2%} over max_memory')

    bufR = np.empty((nkpts_ij, comp, max_buflen, naux))
    bufI = np.empty_like(bufR)
    cpu0 = logger.process_clock(), logger.perf_counter()
    nsteps = len(sh_ranges)
    row1 = 0
    for istep, (sh_start, sh_end, nrow) in enumerate(sh_ranges):
        outR, outI = int3c(shls_slice, bufR, bufI)
        log.debug2('      step [%d/%d], shell range [%d:%d], len(buf) = %d',
                   istep+1, nsteps, sh_start, sh_end, nrow)
        cpu0 = log.timer_debug1(f'outcore_auxe2 [{istep+1}/{nsteps}]', *cpu0)

        shls_slice = (sh_start, sh_end, 0, cell.nbas)
        row0, row1 = row1, row1 + nrow

        for k, kk_idx in enumerate(kikj_idx):
            fswap[f'{dataname}R/{kk_idx}'][row0:row1] = outR[k]
            if f'{dataname}I/{kk_idx}' in fswap:
                fswap[f'{dataname}I/{kk_idx}'][row0:row1] = outI[k]
        outR = outI = None
    bufR = bufI = None
    return fswap


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
            rcut_sr = auxcell_c.rcut
            rcut_sr = (-2*np.log(
                .225*self.cell.precision * omega**4 * rcut_sr**2))**.5 / omega
            auxcell_c.rcut = rcut_sr
            logger.debug1(self, 'auxcell_c  rcut_sr = %g', rcut_sr)
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
                if auxcell.dimension >= 2 and is_zero(kpt):
                    G0_idx = 0  # due to np.fft.fftfreq convention
                    G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
                    coulG_sr[G0_idx] += np.pi/omega**2 * G0_weight

                for p0, p1 in lib.prange(0, ngrids, blksize):
                    auxG_sr = ft_ao.ft_ao(auxcell_c, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                    if is_zero(kpt):
                        sr_j2c[k] -= lib.dot(auxG_sr.conj() * coulG_sr, auxG_sr.T).real
                    else:
                        sr_j2c[k] -= lib.dot(auxG_sr.conj() * coulG_sr, auxG_sr.T)
                    auxG_sr = None

                j2c_k = recontract_2d(j2c_k, sr_j2c[k])
                sr_j2c[k] = None

            j2c.append(j2c_k)
        return j2c

    outcore_auxe2 = _outcore_auxe2

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

    @lib.with_doc(_outcore_auxe2.__doc__)
    def outcore_auxe2(self, cderi_file, intor='int3c2e', aosym='s2', comp=None,
                      j_only=False, dataname='j3c', shls_slice=None):
        assert not self.exclude_d_aux
        with lib.temporary_env(self, auxcell=self.fused_cell,
                               rs_auxcell=self.rs_fused_cell):
            return _outcore_auxe2(self, cderi_file, intor, aosym, comp,
                                  j_only, dataname, shls_slice)

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
        def load_j3c(col0, col1):
            j3cR, j3cI = gdf_load(col0, col1)
            j3cR = [self.fuse(vR) for vR in j3cR]
            j3cI = [self.fuse(vI) if vI is not None else None for vI in j3cI]
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
