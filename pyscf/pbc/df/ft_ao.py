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

RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 1.0)
# kecut=10 can roughly converge GTO with alpha=0.5
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)

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

    rs_cell = _RangeSeparatedCell.from_cell(cell, KECUT_THRESHOLD,
                                            RCUT_THRESHOLD, log)
    if bvk_kmesh is None:
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kptjs)
        log.debug2('Set bvk_kmesh = %s', bvk_kmesh)
    rcut = estimate_rcut(rs_cell)
    supmol = ExtendedMole.from_cell(rs_cell, bvk_kmesh, rcut.max(), log)
    supmol = supmol.strip_basis(rcut)

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
                  return_complex=False, kpts=None, verbose=None):
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

    if kpts is not None:
        expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts.T))
        expLkR = np.asarray(expLk.real, order='C')
        expLkI = np.asarray(expLk.imag, order='C')
        _expLk = (expLkR, expLkI)
    else:
        _expLk = None

    b = rs_cell.reciprocal_vectors()
    if abs(b-np.diag(b.diagonal())).sum() < 1e-8:
        _eval_gz = 'GTO_Gv_orth'
    else:
        _eval_gz = 'GTO_Gv_nonorth'
    drv = libpbc.PBC_ft_bvk_drv
    cintor = getattr(libpbc, rs_cell._add_suffix(intor))

    log.timer_debug1('ft_ao kernel initialization', *cput0)

    # TODO: use Gv = b * gxyz + q in c code
    # TODO: add zfill
    def ft_kernel(Gv, gxyz=None, Gvbase=None, q=np.zeros(3), kptjs=None,
                  shls_slice=None, aosym=aosym, out=None):
        '''
        Analytical FT for orbital products. The output tensor has the shape [nGv, nao, nao]
        '''
        cput0 = logger.process_clock(), logger.perf_counter()
        assert q.ndim == 1
        if kptjs is None:
            if _expLk is None:
                expLkR = np.ones((nimgs,1))
                expLkI = np.zeros((nimgs,1))
            else:
                expLkR, expLkI = _expLk
        else:
            kptjs = np.asarray(kptjs, order='C').reshape(-1,3)
            expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kptjs.T))
            expLkR = np.asarray(expLk.real, order='C')
            expLkI = np.asarray(expLk.imag, order='C')
            expLk = None

        nkpts = expLkR.shape[1]
        GvT = np.asarray(Gv.T + q[:,None], order='C')
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

        if gxyz is None or Gvbase is None or (abs(q).sum() > 1e-9):
            p_gxyzT = lib.c_null_ptr()
            p_mesh = (ctypes.c_int*3)(0,0,0)
            p_b = (ctypes.c_double*1)(0)
            eval_gz = 'GTO_Gv_general'
        else:
            eval_gz = _eval_gz
            gxyzT = np.asarray(gxyz.T, order='C', dtype=np.int32)
            p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
            bqGv = np.hstack((b.ravel(), q) + Gvbase)
            p_b = bqGv.ctypes.data_as(ctypes.c_void_p)
            p_mesh = (ctypes.c_int*3)(*[len(x) for x in Gvbase])

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

        if nGv > 0:
            drv(cintor, eval_gz, fill, fsort,
                out.ctypes.data_as(ctypes.c_void_p),
                expLkR.ctypes.data_as(ctypes.c_void_p),
                expLkI.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs),
                ctypes.c_int(nkpts), ctypes.c_int(nbasp), ctypes.c_int(comp),
                supmol.seg_loc.ctypes.data_as(ctypes.c_void_p),
                supmol.seg2sh.ctypes.data_as(ctypes.c_void_p),
                cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*4)(*shls_slice),
                ovlp_mask.ctypes.data_as(ctypes.c_void_p),
                cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
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
            return out

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
        # Each shell in the original cell can have several segments in the rs-cell.
        # sh_loc indicates the shell Id in rs-cell for each shell in cell.
        self.sh_loc = None

    @classmethod
    def from_cell(cls, cell, ke_cut_threshold=None, rcut_threshold=None,
                  in_rsjk=False, verbose=None):
        from pyscf.pbc.df import aft
        rs_cell = cls()
        rs_cell.__dict__.update(cell.__dict__)
        rs_cell.ref_cell = cell

        if ke_cut_threshold is None:
            rs_cell.bas_map = np.arange(cell.nbas, dtype=np.int32)
            rs_cell.bas_type = np.empty(cell.nbas, dtype=np.int32)
            rs_cell.bas_type[:] = LOCAL_BASIS
            rs_cell.sh_loc = np.arange(cell.nbas + 1, dtype=np.int32)
            return rs_cell

        log = logger.new_logger(cell, verbose)
        if not isinstance(ke_cut_threshold, float):
            ke_cut_threshold = np.min(ke_cut_threshold)

        precision = cell.precision

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

        for ib, orig_bas in enumerate(cell._bas):
            nprim = orig_bas[gto.NPRIM_OF]
            nctr = orig_bas[gto.NCTR_OF]
            l = orig_bas[gto.ANG_OF]
            es = cell.bas_exp(ib)
            # Sort exponents because integral screening of rsjk method relies on
            # the dscending order in exponents
            es_idx = es.argsort()[::-1]
            es = es[es_idx]
            cs = cell._libcint_ctr_coeff(ib)[es_idx]
            abs_cs = abs(cs).max(axis=1)

            # aft._estimate_ke_cutoff is accurate for 4c2e integrals
            # For other integrals such as nuclear attraction.
            # aft._estimate_ke_cutoff may put some primitive GTOs of large es
            # and small cs in the group SMOOTH_BASIS.  These GTOs requires a
            # large ke_cutoff (mesh) in _RSNucBuilder or _CCNucBuilder.
            if in_rsjk:
                ke = aft._estimate_ke_cutoff(es, l, abs_cs, precision)
            else:
                ke = pbcgto.cell._estimate_ke_cutoff(es, l, abs_cs, precision)

            smooth_mask = ke < ke_cut_threshold
            if rcut_threshold is None:
                local_mask = ~smooth_mask
                steep_mask = np.zeros_like(local_mask)
                rcut = None
            else:
                norm_ang = ((2*l+1)/(4*np.pi))**.5
                fac = 2*np.pi*abs_cs/cell.vol * norm_ang/es / precision
                rcut = cell.rcut
                rcut = (np.log(fac * rcut**(l+1) + 1.) / es)**.5
                rcut = (np.log(fac * rcut**(l+1) + 1.) / es)**.5
                steep_mask = (~smooth_mask) & (rcut < rcut_threshold)
                local_mask = (~steep_mask) & (~smooth_mask)

            pexp = orig_bas[gto.PTR_EXP]
            pcoeff = orig_bas[gto.PTR_COEFF]

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
                log.debug2('bas %d rcut %s  kecut %s', ib, rcut, ke)
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
            log.debug('rs_cell.nbas %d nao %d', rs_cell.nbas, rs_cell.nao)
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
        segs = np.zeros(self.ref_cell.nbas)
        segs[self.bas_map[mask]] = 1
        cell_d.sh_loc = np.append(0, np.cumsum(segs)).astype(np.int32)
        logger.debug1(self, 'cell_d.nbas %d', cell_d.nbas)
        if cell_d.nbas == 0:
            return cell_d

        cell_d.ke_cutoff = ke_cutoff = pbcgto.estimate_ke_cutoff(cell_d)
        cell_d.mesh = cell_d.cutoff_to_mesh(ke_cutoff)
        logger.debug1(self, 'cell_d rcut %g ke_cutoff %g, mesh %s',
                      cell_d.rcut, ke_cutoff, cell_d.mesh)
        return cell_d

    def compact_basis_cell(self):
        '''Construct a cell with only the smooth part of the AO basis'''
        cell_c = self.copy(deep=False)
        mask = self.bas_type != SMOOTH_BASIS
        cell_c._bas = self._bas[mask]
        cell_c.bas_map = cell_c.bas_map[mask]
        cell_c.bas_type = cell_c.bas_type[mask]
        segs = self.sh_loc[1:] - self.sh_loc[:-1]
        segs[self.bas_map[~mask]] -= 1
        cell_c.sh_loc = np.append(0, np.cumsum(segs)).astype(np.int32)
        cell_c.rcut = pbcgto.estimate_rcut(cell_c, self.precision)
        return cell_c

    def merge_diffused_block(self, aosym='s1'):
        '''For AO pair that are evaluated in blocks with using the basis
        partitioning self.compact_basis_cell() and self.smooth_basis_cell(),
        merge the DD block into the CC, CD, DC blocks (C ~ compact basis,
        D ~ diffused basis)
        '''
        ao_loc = self.ref_cell.ao_loc
        smooth_bas_idx = self.bas_map[self.bas_type == SMOOTH_BASIS]
        smooth_ao_idx = self.get_ao_indices(smooth_bas_idx, ao_loc)
        nao = ao_loc[-1]
        naod = smooth_ao_idx.size
        drv = getattr(libpbc, f'PBCnr3c_fuse_dd_{aosym}')

        def merge(j3c, j3c_dd, shls_slice=None):
            if j3c_dd.size == 0:
                return j3c
            # The AO index in the original cell
            if shls_slice is None:
                slice_in_cell = (0, nao, 0, nao)
            else:
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

    def decontract_basis(self, to_cart=True, aggregate=False):
        pcell, ctr_coeff = self.ref_cell.decontract_basis(to_cart=to_cart,
                                                          aggregate=aggregate)
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

class ExtendedMole(gto.Mole):
    '''An extended Mole object to mimic periodicity'''
    def __init__(self):
        # The cell which used to generate the supmole
        self.rs_cell: _RangeSeparatedCell = None
        self.bvk_kmesh = None
        self.Ls = None
        self.bvkmesh_Ls = None
        # seg_loc maps the shell Id in bvk cell to shell Id in bvk rs-cell.
        # seg2sh maps the shell Id in bvk rs-cell to the shell Id in supmol.
        # Lattice sum range for each bvk cell shell can be obtained
        # (seg2sh[n+1] - seg2sh[n])
        self.seg_loc = None
        self.seg2sh = None
        # whether the basis bas_mask[bvk-cell-id, basis-id, image-id] is
        # needed to reproduce the periodicity
        self.bas_mask = None
        self.precision = None

    @property
    def sh_loc(self):
        # A map for shell in bvk cell to shell Id in supmol
        return self.seg2sh[self.seg_loc]

    @property
    def bas_map(self):
        # A map to assign each basis of supmol._bas the index in
        # [bvk_cell-id, bas-id, image-id]
        return np.where(self.bas_mask.ravel())[0].astype(np.int32)

    @classmethod
    def from_cell(cls, cell, kmesh, rcut=None, verbose=None):
        if rcut is None: rcut = cell.rcut
        log = logger.new_logger(cell, verbose)

        if not isinstance(cell, _RangeSeparatedCell):
            cell = _RangeSeparatedCell.from_cell(cell)

        bvkcell = pbctools.super_cell(cell, kmesh, wrap_around=True)
        Ls = bvkcell.get_lattice_Ls(rcut=rcut)
        Ls = Ls[np.linalg.norm(Ls, axis=1).argsort()]
        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
        LKs = Ls[:,None,:] + bvkmesh_Ls
        nimgs, bvk_ncells = LKs.shape[:2]
        log.debug1('Generate supmol with rcut = %g nimgs = %d bvk_ncells = %d',
                   rcut, nimgs, bvk_ncells)

        supmol = cls()
        supmol.__dict__.update(cell.__dict__)
        supmol = pbctools._build_supcell_(supmol, cell, LKs.reshape(nimgs*bvk_ncells, 3))
        supmol.rs_cell = cell
        supmol.bvk_kmesh = kmesh
        supmol.Ls = Ls
        supmol.bvkmesh_Ls = bvkmesh_Ls
        bas_mask = np.ones((bvk_ncells, cell.nbas, nimgs), dtype=bool)
        supmol.seg_loc, supmol.seg2sh = supmol.bas_mask_to_segment(cell, bas_mask, verbose)
        supmol.bas_mask = bas_mask
        supmol.precision = cell.precision
        supmol._env[gto.PTR_EXPCUTOFF] = -np.log(cell.precision*1e-4)

        _bas_reordered = supmol._bas.reshape(
            nimgs, bvk_ncells, cell.nbas, gto.BAS_SLOTS).transpose(1,2,0,3)
        supmol._bas = np.asarray(_bas_reordered.reshape(-1, gto.BAS_SLOTS),
                                 dtype=np.int32, order='C')
        return supmol

    def strip_basis(self, rcut):
        rs_cell = self.rs_cell
        dim = rs_cell.dimension
        if dim == 0:
            return self

        # Search the shortest distance to the reference cell for each atom in the supercell.
        atom_coords = self.atom_coords()
        d = np.linalg.norm(atom_coords[:,None] - rs_cell.atom_coords(), axis=2)
        shortest_dist = np.min(d, axis=1)
        bas_dist = shortest_dist[self._bas[:,gto.ATOM_OF]]

        # filter _bas
        nbas0 = self._bas.shape[0]
        if bas_dist.size == self.bas_mask.size:
            bas_dist = bas_dist.reshape(self.bas_mask.shape)
            self.bas_mask = bas_mask = bas_dist < rcut[:,None]
            self._bas = self._bas[bas_mask.ravel()]
        else:
            dr = np.empty(self.bas_mask.shape)
            dr[:] = 1e9
            dr[self.bas_mask] = bas_dist
            bas_mask = dr < rcut[:,None]
            self._bas = self._bas[bas_mask[self.bas_mask]]
            self.bas_mask = bas_mask

        # filter _atm
        atm_ids = np.unique(self._bas[:,gto.ATOM_OF])
        atm_mapping = np.zeros(self._atm.shape[0], dtype=np.int32)
        atm_mapping[atm_ids] = np.arange(atm_ids.size)
        self._atm = self._atm[atm_ids]
        self._bas[:,gto.ATOM_OF] = atm_mapping[self._bas[:,gto.ATOM_OF]]

        nbas1 = self._bas.shape[0]
        logger.debug1(self, 'strip_basis %d to %d ', nbas0, nbas1)
        self.seg_loc, self.seg2sh = self.bas_mask_to_segment(rs_cell, self.bas_mask)
        return self

    def get_ovlp_mask(self, cutoff=None):
        '''integral screening mask for basis product between cell and supmol'''
        rs_cell = self.rs_cell
        supmol = self
        # consider only the most diffused component of a basis
        cell_exps, cell_cs = pbcgto.cell._extract_pgto_params(rs_cell, 'min')
        cell_l = rs_cell._bas[:,gto.ANG_OF]
        cell_bas_coords = rs_cell.atom_coords()[rs_cell._bas[:,gto.ATOM_OF]]

        if cutoff is None:
            theta_ij = cell_exps.min() / 2
            vol = rs_cell.vol
            lattice_sum_factor = max(2*np.pi*rs_cell.rcut/(vol*theta_ij), 1)
            cutoff = rs_cell.precision/lattice_sum_factor * .1
            logger.debug(self, 'Set ft_ao cutoff to %g', cutoff)

        supmol_exps, supmol_cs = pbcgto.cell._extract_pgto_params(supmol, 'min')
        supmol_bas_coords = supmol.atom_coords()[supmol._bas[:,gto.ATOM_OF]]
        supmol_l = supmol._bas[:,gto.ANG_OF]

        aij = cell_exps[:,None] + supmol_exps
        theta = cell_exps[:,None] * supmol_exps / aij
        dr = np.linalg.norm(cell_bas_coords[:,None,:] - supmol_bas_coords, axis=2)

        aij1 = 1./aij
        aij2 = aij**-.5
        dri = supmol_exps*aij1 * dr + aij2
        drj = cell_exps[:,None]*aij1 * dr + aij2
        norm_i = cell_cs * ((2*cell_l+1)/(4*np.pi))**.5
        norm_j = supmol_cs * ((2*supmol_l+1)/(4*np.pi))**.5
        fl = 2*np.pi/rs_cell.vol*dr/theta + 1.
        ovlp = (np.pi**1.5 * norm_i[:,None]*norm_j * np.exp(-theta*dr**2) *
                dri**cell_l[:,None] * drj**supmol_l * aij1**1.5 * fl)
        return ovlp > cutoff

    @staticmethod
    def bas_mask_to_segment(rs_cell, bas_mask, verbose=None):
        '''
        bas_mask shape [bvk_ncells, nbas, nimgs]
        '''
        log = logger.new_logger(rs_cell, verbose)
        bvk_ncells, cell_rs_nbas, nimgs = bas_mask.shape
        images_count = np.count_nonzero(bas_mask, axis=2)

        # seg_loc maps shell Id in bvk-cell to segment Id in supmol
        # seg2sh maps the segment Id to shell Id of supmol
        seg_loc = np.arange(bvk_ncells)[:,None] * cell_rs_nbas + rs_cell.sh_loc[:-1]
        seg_loc = np.append(seg_loc.ravel(), bvk_ncells * cell_rs_nbas)
        seg2sh = np.append(0, np.cumsum(images_count.ravel()))

        if log.verbose > logger.DEBUG:
            steep_mask = rs_cell.bas_type == STEEP_BASIS
            local_mask = rs_cell.bas_type == LOCAL_BASIS
            diffused_mask = rs_cell.bas_type == SMOOTH_BASIS
            log.debug1('No. steep basis in sup-mol %d', images_count[:,steep_mask].sum())
            log.debug1('No. local basis in sup-mol %d', images_count[:,local_mask].sum())
            log.debug1('No. diffused basis in sup-mol %d', images_count[:,diffused_mask].sum())
            log.debug3('sup-mol seg_loc %s', seg_loc)
            log.debug3('sup-mol seg2sh %s', seg2sh)
        return seg_loc.astype(np.int32), seg2sh.astype(np.int32)

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

def estimate_rcut(cell, precision=None):
    '''Estimate rcut for each basis based on Schwarz inequality
    Q_ij ~ S_ij * (sqrt(2aij/pi) * aij**(lij*2) * (4*lij-1)!!)**.5
    '''
    if precision is None:
        # The rcut estimated with this function is sufficient to converge
        # the integrals to the required precision. Errors around the required
        # precision is found when checking hermitian symmetry of the integrals.
        # The discrepancy in hermitian symmetry may cause issues in post-HF
        # methods which assume the hermitian symmetry in MO integrals.
        # Therefore precision is adjusted to ensure hermitian symmetry.
        precision = cell.precision * 1e-2

    if cell.nbas == 0:
        return np.zeros(1)

    # consider only the most diffused component of a basis
    exps, cs = pbcgto.cell._extract_pgto_params(cell, 'min')
    ls = cell._bas[:,gto.ANG_OF]

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

    r0 = cell.rcut
    dri = aj*aij1 * r0 + 1.
    drj = ai*aij1 * r0 + 1.
    fl = 2*np.pi/cell.vol * r0/theta
    r0 = (np.log(fac * dri**li * drj**lj * fl + 1.) / theta)**.5

    dri = aj*aij1 * r0 + 1.
    drj = ai*aij1 * r0 + 1.
    fl = 2*np.pi/cell.vol * r0/theta
    r0 = (np.log(fac * dri**li * drj**lj * fl + 1.) / theta)**.5
    return r0
