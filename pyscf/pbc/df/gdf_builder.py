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
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc.df import aft
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df.rsdf_builder import _round_off_to_odd_mesh
from pyscf.pbc.df.incore import libpbc, _Int3cBuilder, _strip_basis
from pyscf.pbc.lib.kpts_helper import (is_zero, unique_with_wrap_around,
                                       group_by_conj_pairs)


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

        _Int3cBuilder.__init__(self, cell, auxcell, kpts)

    def has_long_range(self):
        '''Whether to add the long-range part computed with AFT integrals'''
        return True

    def reset(self, cell=None):
        _Int3cBuilder.reset(self, cell)
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
            self.ke_cutoff = aft.estimate_ke_cutoff_for_eta(cell, self.eta)
            mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), self.ke_cutoff)
            self.mesh = _round_off_to_odd_mesh(mesh)
        elif self.ke_cutoff is None:
            ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            self.ke_cutoff = ke_cutoff.min()

        self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        self.fused_cell, self.fuse = fuse_auxcell(auxcell, self.eta)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, rsdf_builder.RCUT_THRESHOLD, verbose=log)

        supmol = ft_ao._ExtendedMole.from_cell(rs_cell, kmesh)
        self.supmol = supmol = _strip_basis(supmol, self.eta)

        log.timer_debug1('initializing supmol', *cpu0)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())
        return self

    weighted_coulG = aft.weighted_coulG
    get_q_cond_aux = _Int3cBuilder.get_q_cond_aux
    get_bas_map = _Int3cBuilder.get_bas_map

    def get_2c2e(self, uniq_kpts):
        fused_cell = self.fused_cell
        auxcell = self.auxcell
        naux = auxcell.nao

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
        ke = aft.estimate_ke_cutoff_for_eta(auxcell, self.eta, precision)
        mesh = pbctools.cutoff_to_mesh(auxcell.lattice_vectors(), ke)
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
                      fft_dd_block=None):
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

        if fft_dd_block:
            self._outcore_dd_block(fswap, intor, aosym, comp, j_only,
                                   dataname, shls_slice)

        # int3c2e for (cell, cell | fused_cell)
        with lib.temporary_env(self, auxcell=self.fused_cell):
            int3c = self.gen_int3c_kernel(intor, aosym, comp, j_only)

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
            for k, kk_idx in enumerate(kikj_idx):
                outR[k] = self.fuse(outR[k], axis=1)
                if f'{dataname}I/{kk_idx}' in fswap:
                    outI[k] = self.fuse(outI[k], axis=1)

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
        return fswap

    def weighted_ft_ao(self, kpt):
        '''exp(-i*(G + k) dot r) * Coulomb_kernel for the basis of model charge'''
        fused_cell = self.fused_cell
        mesh = self.mesh
        Gv, Gvbase, kws = fused_cell.get_Gv_weights(mesh)
        b = fused_cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
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
        if cell.dimension == 3 and is_zero(kpt):
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
        j3cR, j3cI = j3c
        GchgR = Gaux[0][p0:p1]
        GchgI = Gaux[1][p0:p1]
        naux = j3cR[0].shape[0] - GchgR.shape[1]
        nG = p1 - p0
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
    # SR ERI does not have the contributions from backgound charge
    if fused_cell.dimension != 3 or fused_cell.omega < 0:
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
    chgcell.rcut = _estimate_rcut(smooth_eta, l_max, 1., auxcell.precision)

    logger.debug1(auxcell, 'make compensating basis, num shells = %d, num cGTOs = %d',
                  chgcell.nbas, chgcell.nao_nr())
    logger.debug1(auxcell, 'chgcell.rcut %s', chgcell.rcut)
    return chgcell

def fuse_auxcell(auxcell, eta):
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
            return Lpq
    return fused_cell, fuse

def _guess_eta(cell, kpts=None, mesh=None):
    '''Search for optimal eta and mesh'''
    if cell.dimension == 0:
        if mesh is None:
            mesh = cell.mesh
        ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh).min()
        eta = aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
        return eta, mesh, ke_cutoff

    a = cell.lattice_vectors()
    eta_min = aft.estimate_eta(cell, cell.precision*1e-2)
    ke_min = aft.estimate_ke_cutoff_for_eta(cell, eta_min, cell.precision)
    mesh_min = _round_off_to_odd_mesh(pbctools.cutoff_to_mesh(a, ke_min))

    if mesh is None:
        nimgs = (8 * cell.rcut**3 / cell.vol) ** (cell.dimension / 3)
        nkpts = len(kpts)
        nao = cell.nao
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
        ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), np.asarray(mesh)-1)
        ke_cutoff = ke_cutoff[:cell.dimension].min()
    eta = aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
    return eta, mesh, ke_cutoff
