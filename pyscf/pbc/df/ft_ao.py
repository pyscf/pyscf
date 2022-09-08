#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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
Analytical Fourier transformation AO-pair product for PBC
'''

import ctypes
import copy
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import gto as pbcgto
from pyscf.gto.ft_ao import ft_ao as mol_ft_ao
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf import __config__

RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.0)
# kecut=10 can rougly converge GTO with alpha=0.5
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)

# cutoff penalty due to lattice summation
LATTICE_SUM_PENALTY = 1e-2

STEEP_BASIS = 0
LOCAL_BASIS = 1
SMOOTH_BASIS = 2

libpbc = lib.load_library('libpbc')

#
# \int mu*nu*exp(-ik*r) dr
#
def ft_aopair(cell, Gv, shls_slice=None, aosym='s1',
              b=None, gxyz=None, Gvbase=None, kpti_kptj=np.zeros((2,3)),
              q=None, intor='GTO_ft_ovlp', comp=1, verbose=None):
    r'''
    Fourier transform AO pair for a pair of k-points
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3
    '''
    kpti, kptj = kpti_kptj
    if q is None:
        q = kptj - kpti
    val = ft_aopair_kpts(cell, Gv, shls_slice, aosym, b, gxyz, Gvbase,
                         q, kptj.reshape(1,3), intor, comp)
    return val[0]


def ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                   b=None, gxyz=None, Gvbase=None, q=np.zeros(3),
                   kptjs=np.zeros((1,3)), intor='GTO_ft_ovlp', comp=1,
                   bvk_kmesh=None, out=None):
    r'''
    Fourier transform AO pair for a group of k-points
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The return array holds the AO pair
    corresponding to the kpoints given by kptjs
    '''
    log = logger.new_logger(cell)
    kptjs = np.asarray(kptjs, order='C').reshape(-1,3)

    rs_cell = _RangeSeparatedCell.from_cell(cell, KECUT_THRESHOLD, RCUT_THRESHOLD, log)
    if bvk_kmesh is None:
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kptjs)
        log.debug2('Set bvk_kmesh = %s', bvk_kmesh)
    supmol = _ExtendedMole.from_cell(rs_cell, bvk_kmesh, verbose=log)
    supmol = supmol.strip_basis()

    ft_kern = supmol.gen_ft_kernel(aosym, intor=intor, comp=comp,
                                   return_complex=True, verbose=log)

    return ft_kern(Gv, gxyz, Gvbase, q, kptjs, shls_slice)


@lib.with_doc(mol_ft_ao.__doc__)
def ft_ao(mol, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=np.zeros(3), verbose=None):
    if gamma_point(kpt):
        return mol_ft_ao(mol, Gv, shls_slice, b, gxyz, Gvbase, verbose)
    else:
        kG = Gv + kpt
        return mol_ft_ao(mol, kG, shls_slice, None, None, None, verbose)


def gen_ft_kernel(supmol, aosym='s1', intor='GTO_ft_ovlp', comp=1,
                  return_complex=False, verbose=None):
    r'''
    Generate the analytical fourier transform kernel for AO products

    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3
    '''
    log = logger.new_logger(supmol)
    cput0 = logger.process_clock(), logger.perf_counter()
    rs_cell = supmol.rs_cell
    assert isinstance(rs_cell, _RangeSeparatedCell)

    # The number of basis in the original cell
    nbasp = rs_cell.ref_cell.nbas
    cell0_ao_loc = rs_cell.ref_cell.ao_loc
    bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape

    ovlp_mask = supmol.get_ovlp_mask()
    bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, rs_cell.sh_loc, supmol.sh_loc)
    cell0_ovlp_mask = bvk_ovlp_mask.reshape(nbasp, bvk_ncells, nbasp).any(axis=1)
    ovlp_mask = ovlp_mask.astype(np.int8)
    cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)

    # A map to assign each basis of supmol._bas the index in
    # [bvk_cell-id, bas-id, image-id]
    bas_map = np.where(supmol.bas_mask.ravel())[0].astype(np.int32)

    b = rs_cell.reciprocal_vectors()

    log.timer_debug1('ft_ao kernel initialization', *cput0)

    # TODO: use Gv = b * gxyz + q in c code
    # TODO: add zfill
    def ft_kernel(Gv, gxyz=None, Gvbase=None, q=np.zeros(3), kptjs=np.zeros((1, 3)),
                  shls_slice=None, aosym=aosym, out=None):
        '''
        Analytical FT for orbital products. The output tensor has the shape [nGv, nao, nao]
        '''
        cput0 = logger.process_clock(), logger.perf_counter()
        q = np.reshape(q, 3)
        kptjs = np.asarray(kptjs, order='C').reshape(-1,3)
        nkpts = len(kptjs)

        expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kptjs.T))
        expLkR = np.asarray(expLk.real, order='C')
        expLkI = np.asarray(expLk.imag, order='C')
        expLk = None

        GvT = np.asarray(Gv.T, order='C') + q.reshape(-1,1)
        nGv = GvT.shape[1]

        if shls_slice is None:
            shls_slice = (0, nbasp, 0, nbasp)
        ni = cell0_ao_loc[shls_slice[1]] - cell0_ao_loc[shls_slice[0]]
        nj = cell0_ao_loc[shls_slice[3]] - cell0_ao_loc[shls_slice[2]]
        shape = (nkpts, comp, ni, nj, nGv)

        aosym = aosym[:2]
        if aosym == 's1hermi':
            # Gamma point only
            assert is_zero(q) and is_zero(kptjs) and ni == nj
            # Theoretically, hermitian symmetry can be also found for kpti == kptj != 0:
            #       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
            # hermi operation needs to reorder axis-0.  It is inefficient.
        elif aosym == 's2':
            i0 = cell0_ao_loc[shls_slice[0]]
            i1 = cell0_ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
            shape = (nkpts, comp, nij, nGv)

        if gxyz is None or b is None or Gvbase is None or (abs(q).sum() > 1e-9):
            p_gxyzT = lib.c_null_ptr()
            p_mesh = (ctypes.c_int*3)(0,0,0)
            p_b = (ctypes.c_double*1)(0)
            eval_gz = 'GTO_Gv_general'
        else:
            if abs(b-np.diag(b.diagonal())).sum() < 1e-8:
                eval_gz = 'GTO_Gv_orth'
            else:
                eval_gz = 'GTO_Gv_nonorth'
            gxyzT = np.asarray(gxyz.T, order='C', dtype=np.int32)
            p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
            bqGv = np.hstack((b.ravel(), q) + Gvbase)
            p_b = bqGv.ctypes.data_as(ctypes.c_void_p)
            p_mesh = (ctypes.c_int*3)(*[len(x) for x in Gvbase])

        drv = libpbc.PBC_ft_bvk_drv
        cintor = getattr(libpbc, rs_cell._add_suffix(intor))
        eval_gz = getattr(libpbc, eval_gz)
        if nkpts == 1:
            fill = getattr(libpbc, 'PBC_ft_bvk_nk1'+aosym)
        else:
            fill = getattr(libpbc, 'PBC_ft_bvk_k'+aosym)

        if return_complex:
            fsort = getattr(libpbc, 'PBC_ft_zsort_' + aosym)
            out = np.ndarray(shape, dtype=np.complex128, buffer=out)
        else:
            fsort = getattr(libpbc, 'PBC_ft_dsort_' + aosym)
            out = np.ndarray((2,) + shape, buffer=out)
        drv(cintor, eval_gz, fill, fsort,
            out.ctypes.data_as(ctypes.c_void_p),
            expLkR.ctypes.data_as(ctypes.c_void_p),
            expLkI.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs),
            ctypes.c_int(nkpts), ctypes.c_int(nbasp), ctypes.c_int(comp),
            supmol.sh_loc.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*shls_slice),
            ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            bas_map.ctypes.data_as(ctypes.c_void_p),
            GvT.ctypes.data_as(ctypes.c_void_p), p_b, p_gxyzT, p_mesh, ctypes.c_int(nGv),
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))

        log.timer_debug1(f'ft_ao intor {intor}', *cput0)

        if return_complex:
            if aosym == 's1hermi':
                for i in range(1, ni):
                    out[:,:,:i,i] = out[:,:,i,:i]
            out = np.rollaxis(out, -1, 2)
            if comp == 1:
                out = out[:,0]
            return out
        else:
            if aosym == 's1hermi':
                for i in range(1, ni):
                    out[:,:,:,:i,i] = out[:,:,:,i,:i]
            out = np.rollaxis(out, -1, 3)
            if comp == 1:
                out = out[:,:,0]
            outR = out[0]
            outI = out[1]
            return outR, outI

    return ft_kernel


class _RangeSeparatedCell(pbcgto.Cell):
    '''Cell with partially de-contracted basis'''
    def __init__(self):
        # ref_cell is the original cell of which the basis to be de-contracted
        self.ref_cell = None
        # For each de-contracted basis, the shell Id in the original cell
        self.bas_map = None
        # Type of each de-contracted basis
        self.bas_type = None
        # For each shell in the original cell, the first basis in the rs-cell.
        # sh_loc indicates how the rs-cell basis can be aggregated to restore
        # the contracted basis in the original cell
        self.sh_loc = None

    @classmethod
    def from_cell(cls, cell, ke_cut_threshold=KECUT_THRESHOLD,
                  rcut_threshold=None, verbose=None):
        from pyscf.pbc.dft.multigrid import _primitive_gto_cutoff
        log = logger.new_logger(cell, verbose)
        if ke_cut_threshold is None:
            ke_cut_threshold = KECUT_THRESHOLD
        if not isinstance(ke_cut_threshold, float):
            ke_cut_threshold = np.min(ke_cut_threshold)

        # rcut and energy cutoff for eash shell
        rcuts, kecuts = _primitive_gto_cutoff(cell, cell.precision)

        # preserves all environments defined in cell (e.g. omega, gauge origin)
        _env = cell._env.copy()
        decontracted_bas = []
        bas_type = []
        # For each basis of rs_cell, bas_map gives the basis in cell
        bas_map = []
        # For each basis of cell, bas_loc gives the first basis in rs_cell
        bas_loc = [0]

        def _append_to_decontracted_bas(orig_id, e_offset, nprim, btype):
            new_bas = cell._bas[orig_id].copy()
            new_bas[gto.PTR_EXP] += e_offset
            new_bas[gto.PTR_COEFF] += e_offset * new_bas[gto.NCTR_OF]
            new_bas[gto.NPRIM_OF] = nprim
            decontracted_bas.append(new_bas)
            bas_type.append(btype)
            bas_map.append(orig_id)

        # Split shells based on rcut
        for ib, orig_bas in enumerate(cell._bas):
            nprim = orig_bas[gto.NPRIM_OF]
            nctr = orig_bas[gto.NCTR_OF]
            ke = kecuts[ib]

            smooth_mask = ke < ke_cut_threshold
            if rcut_threshold is None:
                local_mask = ~smooth_mask
                steep_mask = np.zeros_like(local_mask)
            else:
                steep_mask = (~smooth_mask) & (rcuts[ib] < rcut_threshold)
                local_mask = (~steep_mask) & (~smooth_mask)

            pexp = orig_bas[gto.PTR_EXP]
            pcoeff = orig_bas[gto.PTR_COEFF]
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
            if log.verbose >= logger.DEBUG2:
                log.debug2('bas %d rcuts %s  kecuts %s', ib, rcuts[ib], ke)
                log.debug2('steep %s, %s', np.where(steep_mask)[0], es[steep_mask])
                log.debug2('local %s, %s', np.where(local_mask)[0], es[local_mask])
                log.debug2('smooth %s, %s', np.where(smooth_mask)[0], es[smooth_mask])

            nprim_steep = c_steep.shape[0]
            nprim_local = c_local.shape[0]
            nprim_smooth = c_smooth.shape[0]
            if nprim_steep > 0:
                _append_to_decontracted_bas(ib, 0, nprim_steep, STEEP_BASIS)

            if nprim_local > 0:
                _append_to_decontracted_bas(ib, nprim_steep, nprim_local, LOCAL_BASIS)

            if nprim_smooth > 0:
                _append_to_decontracted_bas(ib, nprim_steep+nprim_local,
                                            nprim_smooth, SMOOTH_BASIS)

            bas_loc.append(len(decontracted_bas))

        rs_cell = cls()
        rs_cell.__dict__.update(cell.__dict__)
        rs_cell.ref_cell = cell
        rs_cell._bas = np.asarray(decontracted_bas, dtype=np.int32, order='C')
        # rs_cell._bas might be of size (0, BAS_SLOTS)
        rs_cell._bas = rs_cell._bas.reshape(-1, gto.BAS_SLOTS)
        rs_cell._env = _env
        rs_cell.bas_map = np.asarray(bas_map, dtype=np.int32)
        rs_cell.bas_type = np.asarray(bas_type, dtype=np.int32)
        rs_cell.sh_loc = np.asarray(bas_loc, dtype=np.int32)
        rs_cell.ke_cutoff = ke_cut_threshold
        if log.verbose >= logger.DEBUG:
            bas_type = rs_cell.bas_type
            log.debug1('No. steep_bas in rs_cell %d', np.count_nonzero(bas_type == STEEP_BASIS))
            log.debug1('No. local_bas in rs_cell %d', np.count_nonzero(bas_type == LOCAL_BASIS))
            log.debug('No. smooth_bas in rs_cell %d', np.count_nonzero(bas_type == SMOOTH_BASIS))
            map_bas = rs_cell._reverse_bas_map(rs_cell.bas_map)
            log.debug2('bas_map from cell to rs_cell %s', map_bas)
            assert np.array_equiv(map_bas, bas_loc)
        log.debug2('%s.bas_type %s', cls, rs_cell.bas_type)
        log.debug2('%s.sh_loc %s', cls, rs_cell.sh_loc)
        return rs_cell

    @staticmethod
    def _reverse_bas_map(bas_map):
        '''Map basis between the original cell and the derived rs-cell.
        For each shell in the original cell, the first basis Id of the
        de-contracted basis in the rs-cell'''
        uniq_bas, map_bas = np.unique(bas_map, return_index=True)
        assert uniq_bas[-1] == len(uniq_bas) - 1
        return np.append(map_bas, len(bas_map)).astype(np.int32)

    def smooth_basis_cell(self):
        '''Construct a cell with only the smooth part of the AO basis'''
        cell_d = self.view(pbcgto.Cell)
        mask = self.bas_type == SMOOTH_BASIS
        cell_d._bas = self._bas[mask]
        precision = self.precision
        cell_d.rcut = pbcgto.estimate_rcut(cell_d, precision=precision)

        # Update mesh
        ke_cutoff = pbcgto.estimate_ke_cutoff(cell_d, precision)
        mesh = pbctools.cutoff_to_mesh(cell_d.lattice_vectors(), ke_cutoff)
        if (cell_d.dimension < 2 or
            (cell_d.dimension == 2 and cell_d.low_dim_ft_type == 'inf_vacuum')):
            mesh[cell_d.dimension:] = pbcgto.cell._mesh_inf_vaccum(cell_d)
        logger.debug1(self, 'cell_d rcut %g ke_cutoff %g, mesh %s',
                      cell_d.rcut, ke_cutoff, mesh)
        cell_d.mesh = mesh
        return cell_d

    def compact_basis_cell(self):
        '''Construct a cell with only the smooth part of the AO basis'''
        cell_c = copy.copy(self)
        mask = self.bas_type != SMOOTH_BASIS
        cell_c._bas = self._bas[mask]
        cell_c.bas_map = cell_c.bas_map[mask]
        cell_c.bas_type = cell_c.bas_type[mask]
        cell_c.sh_loc = None
        cell_c.rcut = pbcgto.estimate_rcut(cell_c, self.precision)
        return cell_c

    def merge_diffused_block(self, aosym='s1'):
        '''For AO pair that are evaluated in blocks with using the basis
        partitioning self.compact_basis_cell() and self.smooth_basis_cell(),
        merge the DD block into the CC, CD, DC blocks (D ~ compact basis,
        D ~ diffused basis)
        '''
        ao_loc = self.ref_cell.ao_loc
        smooth_bas_idx = self.bas_map[self.bas_type == SMOOTH_BASIS]
        smooth_ao_idx = self.get_ao_indices(smooth_bas_idx, ao_loc)
        nao = ao_loc[-1]
        naod = smooth_ao_idx.size
        drv = getattr(libpbc, f'PBCnr3c_fuse_dd_{aosym}')

        def merge(j3c, j3c_dd, shls_slice):
            if j3c_dd.size == 0:
                return j3c
            # The AO index in the original cell
            slice_in_cell = ao_loc[list(shls_slice[:4])]
            # Then search the corresponding index in the diffused block
            slice_in_cell_d = np.searchsorted(smooth_ao_idx, slice_in_cell)

            # j3c_dd may be an h5 object. Load j3c_dd to memory
            d0, d1 = slice_in_cell_d[:2]
            j3c_dd = np.asarray(j3c_dd[d0:d1], order='C')
            naux = j3c_dd.shape[-1]

            drv(j3c.ctypes.data_as(ctypes.c_void_p),
                j3c_dd.ctypes.data_as(ctypes.c_void_p),
                smooth_ao_idx.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*4)(*slice_in_cell),
                (ctypes.c_int*4)(*slice_in_cell_d),
                ctypes.c_int(nao), ctypes.c_int(naod), ctypes.c_int(naux))
            return j3c
        return merge

    def recontract(self, dim=1):
        '''Recontract the vector evaluated with the RS-cell to the vector
        associated to the basis of reference cell
        '''
        ao_loc = self.ref_cell.ao_loc
        ao_map = self.get_ao_indices(self.bas_map, ao_loc)
        nao = ao_loc[-1]

        if dim == 1:

            def recontractor(a):
                assert a.ndim == 2
                a = np.asarray(a, order='C')
                ngrids = a.shape[1]
                out = np.zeros((nao, ngrids), dtype=a.dtype)
                idx = np.arange(ngrids, dtype=np.int32)
                return lib.takebak_2d(out, a, ao_map, idx, thread_safe=False)

        elif dim == 2:

            def recontractor(a):
                assert a.ndim == 2
                a = np.asarray(a, order='C')
                out = np.zeros((nao, nao), dtype=a.dtype)
                return lib.takebak_2d(out, a, ao_map, ao_map, thread_safe=False)

        else:
            raise NotImplementedError(f'dim = {dim}')

        return recontractor

    def recontract_1d(self, vec):
        '''Recontract the vector evaluated with the RS-cell to the vector
        associated to the basis of reference cell
        '''
        return self.recontract()(vec)

    def get_ao_type(self):
        '''Assign a label (STEEP_BASIS, LOCAL_BASIS, SMOOTH_BASIS) to each AO function'''
        ao_loc = self.ao_loc
        nao = ao_loc[-1]
        ao_type = np.empty(nao, dtype=int)

        def assign(type_code):
            ao_idx = self.get_ao_indices(self.bas_type == type_code, ao_loc)
            ao_type[ao_idx] = type_code

        assign(STEEP_BASIS)
        assign(LOCAL_BASIS)
        assign(SMOOTH_BASIS)
        return ao_type

    def decontract_basis(self, to_cart=True):
        pcell, ctr_coeff = self.ref_cell.decontract_basis(to_cart=to_cart)
        pcell = pcell.view(self.__class__)
        pcell.ref_cell = None

        # Set bas_type labels for the primitive basis of decontracted cell
        smooth_mask = self.bas_type == SMOOTH_BASIS
        smooth_exp_thresholds = {}
        for ia, (ib0, ib1) in enumerate(self.aoslice_by_atom()[:,:2]):
            smooth_bas_ids = ib0 + np.where(smooth_mask[ib0:ib1])[0]
            for ib in smooth_bas_ids:
                l = self._bas[ib,gto.ANG_OF]
                nprim = self._bas[ib,gto.NPRIM_OF]
                pexp = self._bas[ib,gto.PTR_EXP]
                smooth_exp_thresholds[(ia, l)] = max(
                    self._env[pexp:pexp+nprim].max(),
                    smooth_exp_thresholds.get((ia, l), 0))

        pcell_ls = pcell._bas[:,gto.ANG_OF]
        pcell_exps = pcell._env[pcell._bas[:,gto.PTR_EXP]]
        pcell_ao_slices = pcell.aoslice_by_atom()
        pcell.bas_type = np.empty(pcell.nbas, dtype=np.int32)
        pcell.bas_type[:] = LOCAL_BASIS
        for (ia, l), exp_cut in smooth_exp_thresholds.items():
            ib0, ib1 = pcell_ao_slices[ia,:2]
            smooth_mask = ((pcell_exps[ib0:ib1] <= exp_cut+1e-8) &
                           (pcell_ls[ib0:ib1] == l))
            pcell.bas_type[ib0:ib1][smooth_mask] = SMOOTH_BASIS

        pcell.bas_map = np.arange(pcell.nbas, dtype=np.int32)
        pcell.sh_loc = np.append(np.arange(pcell.nbas), pcell.nbas).astype(np.int32)
        logger.debug3(pcell, 'decontracted cell bas_type %s', pcell.bas_type)
        logger.debug3(pcell, 'decontracted cell sh_loc %s', pcell.sh_loc)
        return pcell, ctr_coeff

class _ExtendedMole(gto.Mole):
    '''An extended Mole object to mimic periodicity'''
    def __init__(self):
        # The cell which used to generate the supmole
        self.rs_cell: _RangeSeparatedCell = None
        self.bvk_kmesh = None
        self.Ls = None
        self.bvkmesh_Ls = None
        # For each shell in the bvk cell, the first basis in the supmole.
        # sh_loc indicates the boundary of the batches in supmol. These batches
        # can be aggregated to restore the original contracted basis in the bvk-cell
        self.sh_loc = None
        # whether the basis bas_mask[bvk-cell-id, basis-id, image-id] is
        # needed to reproduce the periodicity
        self.bas_mask = None
        self.precision = None

    @classmethod
    def from_cell(cls, cell, kmesh, rcut=None, verbose=None):
        if rcut is None: rcut = cell.rcut
        log = logger.new_logger(cell, verbose)

        if not isinstance(cell, _RangeSeparatedCell):
            rs_cell = cell.view(_RangeSeparatedCell)
            rs_cell.ref_cell = cell
            rs_cell.bas_map = np.arange(cell.nbas, dtype=np.int32)
            rs_cell.bas_type = np.empty(cell.nbas, dtype=np.int32)
            rs_cell.bas_type[:] = LOCAL_BASIS
            rs_cell.sh_loc = np.append(np.arange(cell.nbas), cell.nbas).astype(np.int32)
            cell = rs_cell

        bvkcell = pbctools.super_cell(cell, kmesh, wrap_around=True)
        Ls = bvkcell.get_lattice_Ls(rcut=rcut)
        Ls = Ls[np.linalg.norm(Ls, axis=1).argsort()]
        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
        LKs = Ls[:,None,:] + bvkmesh_Ls
        nimgs, bvk_ncells = LKs.shape[:2]
        log.debug1('Generate supmol with nimgs = %d bvk_ncells = %d',
                   nimgs, bvk_ncells)

        supmol = cls()
        supmol.__dict__.update(cell.__dict__)
        supmol = pbctools._build_supcell_(supmol, cell, LKs.reshape(nimgs*bvk_ncells, 3))
        supmol.rs_cell = cell
        supmol.bvk_kmesh = kmesh
        supmol.Ls = Ls
        supmol.bvkmesh_Ls = bvkmesh_Ls
        bas_mask = np.ones((bvk_ncells, cell.nbas, nimgs), dtype=bool)
        supmol.sh_loc = supmol.bas_mask_to_sh_loc(cell, bas_mask, verbose)
        supmol.bas_mask = bas_mask
        supmol.precision = cell.precision * LATTICE_SUM_PENALTY

        _bas_reordered = supmol._bas.reshape(
            nimgs, bvk_ncells, cell.nbas, gto.BAS_SLOTS).transpose(1,2,0,3)
        supmol._bas = np.asarray(_bas_reordered.reshape(-1, gto.BAS_SLOTS),
                                 dtype=np.int32, order='C')
        return supmol

    def strip_basis(self, cutoff=None, inplace=True):
        '''Remove remote basis if they do not contribute to the FT of basis product'''
        ovlp_mask = self.get_ovlp_mask(cutoff).any(axis=0)
        if inplace:
            supmol = self
        else:
            supmol = copy.copy(self)
        if ovlp_mask.size == supmol.bas_mask.size:
            supmol.bas_mask = ovlp_mask.reshape(supmol.bas_mask.shape)
        else:
            supmol.bas_mask = supmol.bas_mask.copy()
            supmol.bas_mask[supmol.bas_mask] = ovlp_mask
        supmol._bas = supmol._bas[ovlp_mask]
        supmol.sh_loc = supmol.bas_mask_to_sh_loc(supmol.rs_cell, supmol.bas_mask)
        return supmol

    def get_ovlp_mask(self, cutoff=None):
        '''integral screening mask for basis product between cell and supmol'''
        if cutoff is None:
            cutoff = self.precision
        rs_cell = self.rs_cell
        supmol = self
        # consider only the most diffused component of a basis
        cell_exps = np.array([e.min() for e in rs_cell.bas_exps()])
        cell_l = rs_cell._bas[:,gto.ANG_OF]
        cell_bas_coords = rs_cell.atom_coords()[rs_cell._bas[:,gto.ATOM_OF]]

        supmol_exps = np.array([e.min() for e in supmol.bas_exps()])
        supmol_bas_coords = supmol.atom_coords()[supmol._bas[:,gto.ATOM_OF]]
        supmol_l = supmol._bas[:,gto.ANG_OF]

        # Removing the angular part of each basis and turning basis to s-type
        # functions, ovlp is an estimation of the upper bound of radial part
        # overlap integrals
        aij = cell_exps[:,None] + supmol_exps
        a1 = cell_exps[:,None] * supmol_exps / aij
        dr = np.linalg.norm(cell_bas_coords[:,None,:] - supmol_bas_coords, axis=2)
        # dri is the distance between the center of gaussian product to basis_i
        # drj is the distance between the center of gaussian product to basis_j
        dri = supmol_exps/aij * dr + 1
        drj = cell_exps[:,None]/aij * dr + 1
        ovlp = ((4 * a1 / aij)**.75 * np.exp(-a1*dr**2) *
                dri**cell_l[:,None] * drj**supmol_l)
        return ovlp > cutoff

    @staticmethod
    def bas_mask_to_sh_loc(rs_cell, bas_mask, verbose=None):
        '''
        bas_mask shape [bvk_ncells, nbas, nimgs]
        '''
        log = logger.new_logger(rs_cell, verbose)
        bvk_ncells, cell_rs_nbas, nimgs = bas_mask.shape
        images_count = np.count_nonzero(bas_mask, axis=2)
        if log.verbose >= logger.DEBUG:
            steep_mask = rs_cell.bas_type == STEEP_BASIS
            local_mask = rs_cell.bas_type == LOCAL_BASIS
            diffused_mask = rs_cell.bas_type == SMOOTH_BASIS
            log.debug1('No. steep basis in sup-mol %d', images_count[:,steep_mask].sum())
            log.debug1('No. local basis in sup-mol %d', images_count[:,local_mask].sum())
            log.debug1('No. diffused basis in sup-mol %d', images_count[:,diffused_mask].sum())

        cell0_sh_loc = rs_cell.sh_loc
        cell_nbas = len(cell0_sh_loc) - 1
        condensed_images_count = np.empty((bvk_ncells, cell_nbas), dtype=np.int32)
        for ib, (i0, i1) in enumerate(zip(cell0_sh_loc[:-1], cell0_sh_loc[1:])):
            condensed_images_count[:,ib] = images_count[:,i0:i1].sum(axis=1)

        # sh_loc maps from shell Id in bvk-cell to shell Id of supmol
        sh_loc = np.append(0, np.cumsum(condensed_images_count.ravel()))
        log.debug3('sup-mol sh_loc %s', sh_loc)
        # TODO: Some bases are completely inside the bvk-cell. Exclude them from
        # lattice sum. ??
        # sh_loc = np.append(0, np.cumsum(condensed_images_count[condensed_images_count != 0]))
        return sh_loc.astype(np.int32)

    def bas_type_to_indices(self, type_code=SMOOTH_BASIS):
        '''Return the basis indices of required bas_type'''
        cell0_mask = self.rs_cell.bas_type == type_code
        if np.any(cell0_mask):
            # (bvk_ncells, rs_cell.nbas, nimgs)
            bas_type_mask = np.empty_like(self.bas_mask)
            bas_type_mask[:] = cell0_mask[None,:,None]
            bas_type_mask = bas_type_mask[self.bas_mask]
            return np.where(bas_type_mask)[0]
        else:
            return np.arange(0)

    gen_ft_kernel = gen_ft_kernel
