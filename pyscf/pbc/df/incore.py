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

import ctypes
import copy
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
import pyscf.df
from pyscf.scf import _vhf
from pyscf.pbc.df import ft_ao
from pyscf.pbc.gto import estimate_rcut
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_zero, unique
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import pbc as pbctools
from pyscf import __config__

RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.5)
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)

libpbc = lib.load_library('libpbc')

def make_auxmol(cell, auxbasis=None):
    '''
    See pyscf.df.addons.make_auxmol
    '''
    auxcell = pyscf.df.addons.make_auxmol(cell, auxbasis)
    auxcell.rcut = estimate_rcut(auxcell, cell.precision)
    return auxcell

make_auxcell = make_auxmol

def format_aux_basis(cell, auxbasis='weigend+etb'):
    '''For backward compatibility'''
    return make_auxmol(cell, auxbasis)

def aux_e2(cell, auxcell_or_auxbasis, intor='int3c2e', aosym='s1', comp=None,
           kptij_lst=np.zeros((1,2,3)), shls_slice=None, **kwargs):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.

    Returns:
        (nao_pair, naux) array
    '''
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

# For some unkown reasons, the pre-decontracted basis 'is slower than
## Slighly decontract basis. The decontracted basis has better locality.
## The locality can be used in the lattice sum to reduce cost.
#    if shls_slice is None and cell.nao_nr() < 200:
#        cell, contr_coeff = pbcgto.cell._split_basis(cell)
#    else:
#        contr_coeff = None

    int3c = wrap_int3c(cell, auxcell, intor, aosym, comp, kptij_lst, **kwargs)
    out = int3c(shls_slice)

#    if contr_coeff is not None:
#        if aosym == 's2':
#            tmp = out.reshape(nkptij,comp,ni,ni,naux)
#            idx, idy = np.tril_indices(ni)
#            tmp[:,:,idy,idx] = out.conj()
#            tmp[:,:,idx,idy] = out
#            out, tmp = tmp, None
#            out = lib.einsum('kcpql,pi->kciql', out, contr_coeff)
#            out = lib.einsum('kciql,qj->kcijl', out, contr_coeff)
#            idx, idy = np.tril_indices(contr_coeff.shape[1])
#            out = out[:,:,idx,idy]
#        else:
#            out = out.reshape(nkptij,comp,ni,nj,naux)
#            out = lib.einsum('kcpql,pi->kciql', out, contr_coeff)
#            out = lib.einsum('kciql,qj->kcijl', out, contr_coeff)
#            out = out.reshape(nkptij,comp,-1,naux)

    if len(kptij_lst) == 1:
        out = out[0]
    return out

def wrap_int3c(cell, auxcell, intor='int3c2e', aosym='s1', comp=1,
               kptij_lst=np.zeros((1,2,3)), cintopt=None, pbcopt=None):
    '''Generate a 3-center integral kernel which can be called repeatedly in
    the incore or outcore driver. The kernel function has a simple function
    signature f(shls_slice)
    '''
    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    j_only = is_zero(kpti - kptj)
    if j_only:
        kpts = kpti
        nkpts = len(kpts)
        kptij_idx = np.arange(nkpts)
    else:
        kpts, _, kptij_idx = unique(np.vstack([kpti,kptj]))
        wherei = kptij_idx[:len(kpti)]
        wherej = kptij_idx[len(kpti):]
        nkpts = len(kpts)
        kptij_idx = wherei * nkpts + wherej
    dfbuilder = _Int3cBuilder(cell, auxcell, kpts).build()
    # Reduce the size of supmol based on the 3-center overlaps. Otherwise the
    # dfbuilder.supmol oftens two large for the calculations.
    # TODO: strip supmol basis based on the intor type.
    eta = np.hstack(auxcell.bas_exps()).min()
    dfbuilder.supmol = _strip_basis(dfbuilder.supmol, eta)
    int3c = dfbuilder.gen_int3c_kernel(intor, aosym, comp, j_only,
                                       kptij_idx, return_complex=True)
    return int3c


def fill_2c2e(cell, auxcell_or_auxbasis, intor='int2c2e', hermi=0, kpt=np.zeros(3)):
    '''2-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    if hermi != 0:
        hermi = pyscf.lib.HERMITIAN
# pbcopt use the value of AO-pair to prescreening PBC integrals in the lattice
# summation.  Pass NULL pointer to pbcopt to prevent the prescreening
    return auxcell.pbc_intor(intor, 1, hermi, kpt, pbcopt=lib.c_null_ptr())


class _Int3cBuilder(lib.StreamObject):
    '''helper functions to compute 3-center integral tensor with double-lattice sum
    '''
    def __init__(self, cell, auxcell, kpts=None):
        self.cell = cell
        self.auxcell = auxcell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        if kpts is None:
            self.kpts = np.zeros((1, 3))
        else:
            self.kpts = np.reshape(kpts, (-1, 3))

        self.rs_cell = None
        # mesh to generate Born-von Karman supercell
        self.bvk_kmesh = None
        self.supmol = None
        self.ke_cutoff = None

        self._keys = set(self.__dict__.keys())

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.rs_cell = None
        self.supmol = None
        return self

    def build(self):
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)

        supmol = ft_ao._ExtendedMole.from_cell(rs_cell, kmesh, cell.rcut,
                                               verbose=log)
        if self.auxcell is not None:
            eta = np.hstack(self.auxcell.bas_exps()).min()
            supmol = _strip_basis(supmol, eta)
        self.supmol = supmol
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())
        return self

    def gen_int3c_kernel(self, intor='int3c2e', aosym='s2', comp=None,
                         j_only=False, reindex_k=None, rs_auxcell=None,
                         auxcell=None, supmol=None, return_complex=False):
        '''Generate function to compute int3c2e with double lattice-sum

        rs_auxcell: range-separated auxcell for gdf/rsdf module

        reindex_k: an index array to sort the order of k-points in output
        '''
        log = logger.new_logger(self)
        cput0 = logger.process_clock(), logger.perf_counter()
        if self.rs_cell is None:
            self.build()
        if auxcell is None:
            auxcell = self.auxcell
        if supmol is None:
            supmol = self.supmol
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
        nbasp = cell.nbas  # The number of shells in the primitive cell

        # integral mask for supmol
        log.debug1('int3c_kernel integral cutoff %g', supmol.precision)

        def _conc_locs(cell_loc, auxcell_loc):
            '''auxiliary basis was appended to regular AO basis when calling int3c2e
            integrals. Composite loc combines locs from regular AO basis and auxiliary
            basis accordingly.'''
            comp_loc = np.append(cell_loc[:-1], cell_loc[-1] + auxcell_loc)
            return np.asarray(comp_loc, dtype=np.int32)

        if rs_auxcell is None:
            atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                         auxcell._atm, auxcell._bas, auxcell._env)
            sh_loc = _conc_locs(supmol.sh_loc, np.arange(auxcell.nbas + 1))
        else:
            atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                         rs_auxcell._atm, rs_auxcell._bas, rs_auxcell._env)
            sh_loc = _conc_locs(supmol.sh_loc, rs_auxcell.sh_loc)

        cell0_ao_loc = _conc_locs(cell.ao_loc, auxcell.ao_loc)

        if 'ECP' in intor:
            q_cond_aux = None
            env[gto.AS_ECPBAS_OFFSET] = supmol.nbas + 1
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            q_cond_aux = self.get_q_cond_aux()
            cintopt = _vhf.make_cintopt(supmol._atm, supmol._bas, supmol._env, intor)

        ovlp_mask, cell0_ovlp_mask = self.get_ovlp_mask(supmol.precision, supmol, cintopt)
        bas_map = self.get_bas_map(auxcell, supmol)

        # Estimate the buffer size required by PBCfill_nr3c functions
        cache_size = max(_get_cache_size(cell, intor),
                         _get_cache_size(auxcell, intor))
        cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
        dijk = cell0_dims[:nbasp].max()**2 * cell0_dims[nbasp:].max() * comp

        aosym = aosym[:2]
        gamma_point_only = is_zero(kpts)
        if gamma_point_only:
            assert nkpts == 1
            fill = f'PBCfill_nr3c_g{aosym}'
            nkpts_ij = 1
            cache_size += dijk
        elif nkpts == 1:
            fill = f'PBCfill_nr3c_nk1{aosym}'
            nkpts_ij = 1
            cache_size += dijk * 3
        elif j_only:
            fill = f'PBCfill_nr3c_k{aosym}'
            # sort kpts then reindex_k in sort_k can be skipped
            if reindex_k is not None:
                kpts = kpts[reindex_k]
                nkpts = len(kpts)
            nkpts_ij = nkpts
            cache_size = (dijk * bvk_ncells + dijk * nkpts * 2 +
                          max(dijk * nkpts * 2, cache_size))
        else:
            fill = f'PBCfill_nr3c_kk{aosym}'
            nkpts_ij = nkpts * nkpts
            cache_size = (max(dijk * bvk_ncells**2 + cache_size, dijk * nkpts**2 * 2) +
                          dijk * bvk_ncells * nkpts * 2)
            if aosym == 's2' and reindex_k is not None:
                kk_mask = np.zeros((nkpts*nkpts), dtype=bool)
                kk_mask[reindex_k] = True
                kk_mask = kk_mask.reshape(nkpts, nkpts)
                if not np.all(kk_mask == kk_mask.T):
                    log.warn('aosym=s2 not found in required kpts pairs')

        expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts.T))
        expLkR = np.asarray(expLk.real, order='C')
        expLkI = np.asarray(expLk.imag, order='C')
        expLk = None

        if reindex_k is None:
            reindex_k = np.arange(nkpts_ij, dtype=np.int32)
        else:
            reindex_k = np.asarray(reindex_k, dtype=np.int32)
            nkpts_ij = reindex_k.size

        drv = libpbc.PBCfill_nr3c_drv

        # is_pbcintor controls whether to use memory efficient functions
        # Only suppots int3c2e_sph, int3c2e_cart in current C library
        is_pbcintor = intor in ('int3c2e_sph', 'int3c2e_cart')
        if is_pbcintor and not intor.startswith('PBC'):
            intor = 'PBC' + intor
        log.debug1('is_pbcintor = %d, intor = %s', is_pbcintor, intor)

        log.timer_debug1('int3c kernel initialization', *cput0)

        def int3c(shls_slice=None, outR=None, outI=None):
            cput0 = logger.process_clock(), logger.perf_counter()
            if shls_slice is None:
                shls_slice = [0, nbasp, 0, nbasp, nbasp, nbasp + auxcell.nbas]
            else:
                ksh0 = nbasp + shls_slice[4]
                ksh1 = nbasp + shls_slice[5]
                shls_slice = list(shls_slice[:4]) + [ksh0, ksh1]
            i0, i1, j0, j1, k0, k1 = cell0_ao_loc[shls_slice]
            if aosym == 's1':
                nrow = (i1-i0)*(j1-j0)
            else:
                nrow = i1*(i1+1)//2 - i0*(i0+1)//2
            if comp == 1:
                shape = (nkpts_ij, nrow, k1-k0)
            else:
                shape = (nkpts_ij, comp, nrow, k1-k0)
            # output has to be filled with zero first because certain integrals
            # may be skipped by fill_ints driver
            outR = np.ndarray(shape, buffer=outR)
            outR[:] = 0
            if gamma_point_only:
                outI = np.zeros(0)
            else:
                outI = np.ndarray(shape, buffer=outI)
                outI[:] = 0

            if q_cond_aux is None:
                q_cond_aux_ptr = lib.c_null_ptr()
            else:
                q_cond_aux_ptr = q_cond_aux.ctypes.data_as(ctypes.c_void_p)

            drv(getattr(libpbc, intor), getattr(libpbc, fill),
                ctypes.c_int(is_pbcintor),
                outR.ctypes.data_as(ctypes.c_void_p),
                outI.ctypes.data_as(ctypes.c_void_p),
                expLkR.ctypes.data_as(ctypes.c_void_p),
                expLkI.ctypes.data_as(ctypes.c_void_p),
                reindex_k.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts_ij),
                ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs),
                ctypes.c_int(nkpts), ctypes.c_int(nbasp), ctypes.c_int(comp),
                sh_loc.ctypes.data_as(ctypes.c_void_p),
                cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*6)(*shls_slice),
                ovlp_mask.ctypes.data_as(ctypes.c_void_p),
                cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
                bas_map.ctypes.data_as(ctypes.c_void_p),
                q_cond_aux_ptr, ctypes.c_double(supmol.precision),
                cintopt, ctypes.c_int(cache_size),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
                env.ctypes.data_as(ctypes.c_void_p))

            log.timer_debug1(f'pbc integral {intor}', *cput0)
            if return_complex:
                if gamma_point_only:
                    return outR
                else:
                    return outR + outI * 1j
            else:
                if gamma_point_only:
                    return outR, None
                else:
                    return outR, outI
        return int3c

    def get_q_cond_aux(self):
        '''To compute Schwarz inequality for auxiliary basis'''
        return None

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        '''integral screening mask between two sup-mols'''
        if cutoff is None:
            cutoff = self.cell.precision
        if supmol is None:
            supmol = self.supmol
        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        nbasp = self.cell.nbas  # The number of shells in the primitive cell
        nbas = supmol.nbas
        ovlp_mask = np.empty((nbas, nbas), dtype=np.int8)
        libpbc.PBCsupmol_ovlp_mask(
            ovlp_mask.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(cutoff),
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))

        bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, supmol.sh_loc)
        cell0_ovlp_mask = bvk_ovlp_mask.reshape(
            bvk_ncells, nbasp, bvk_ncells, nbasp).any(axis=2).any(axis=0)
        ovlp_mask = ovlp_mask.astype(np.int8)
        cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)
        return ovlp_mask, cell0_ovlp_mask

    def get_bas_map(self, auxcell=None, supmol=None):
        '''bas_map is to assign each basis of supmol._bas the index in
        [bvk_cell-id, bas-id, image-id]
        '''
        # Append aux_mask to bas_map as a temporary solution for function
        # _assemble3c in fill_ints.c
        if auxcell is None:
            auxcell = self.auxcell
        if supmol is None:
            supmol = self.supmol
        aux_mask = np.ones(auxcell.nbas, dtype=np.int32)
        bas_map = np.where(supmol.bas_mask.ravel())[0].astype(np.int32)
        bas_map = np.asarray(np.append(bas_map, aux_mask), dtype=np.int32)
        return bas_map


libpbc.GTOmax_cache_size.restype = ctypes.c_int
def _get_cache_size(cell, intor):
    '''Cache size for libcint integrals. Cache size cannot be accurately
    estimated in function PBC_ft_bvk_drv
    '''
    cache_size = libpbc.GTOmax_cache_size(
        getattr(libpbc, intor), (ctypes.c_int*2)(0, cell.nbas), ctypes.c_int(1),
        cell._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
        cell._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
        cell._env.ctypes.data_as(ctypes.c_void_p))
    return cache_size


class _IntNucBuilder(_Int3cBuilder):
    '''In this builder, ovlp_mask can be reused for different types of intor
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        # cache ovlp_mask
        self._supmol = None
        self._ovlp_mask = None
        self._cell0_ovlp_mask = None
        _Int3cBuilder.__init__(self, cell, None, kpts)

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        if self._ovlp_mask is None or supmol is not self._supmol:
            self._ovlp_mask, self._cell0_ovlp_mask = \
                    _Int3cBuilder.get_ovlp_mask(self, cutoff, supmol, cintopt)
            self._supmol = supmol
        return self._ovlp_mask, self._cell0_ovlp_mask

    def _int_nuc_vloc(self, nuccell, intor='int3c2e', aosym='s2', comp=None,
                      with_pseudo=True, supmol=None):
        '''Vnuc - Vloc. nuccell is the cell for model charges
        '''
        logger.debug2(self, 'Real space integrals %s for Vnuc - Vloc', intor)

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        # Use the 3c2e code with steep s gaussians to mimic nuclear density
        fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
        fakenuc._atm, fakenuc._bas, fakenuc._env = \
                gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                             fakenuc._atm, fakenuc._bas, fakenuc._env)

        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc, supmol=supmol)
        bufR, bufI = int3c()

        charge = cell.atom_charges()
        charge = np.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
        if is_zero(kpts):
            mat = np.einsum('k...z,z->k...', bufR, charge)
        else:
            mat = (np.einsum('k...z,z->k...', bufR, charge) +
                   np.einsum('k...z,z->k...', bufI, charge) * 1j)

        # vbar is the interaction between the background charge
        # and the compensating function.  0D, 1D, 2D do not have vbar.
        if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                             'int3c2e_cart'):
            logger.debug2(self, 'G=0 part for %s', intor)
            charge = -cell.atom_charges()

            nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar *= np.pi/cell.vol

            ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
            for k in range(nkpts):
                if aosym == 's1':
                    mat[k] -= nucbar * ovlp[k].ravel()
                else:
                    mat[k] -= nucbar * lib.pack_tril(ovlp[k])
        return mat

    def _get_nuc(self, mesh=None, with_pseudo=False):
        from pyscf.pbc.df.gdf_builder import _guess_eta
        log = logger.Logger(self.stdout, self.verbose)
        t0 = t1 = (logger.process_clock(), logger.perf_counter())
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        nao_pair = nao * (nao+1) // 2

        kpt_allow = np.zeros(3)
        eta, mesh, ke_cutoff = _guess_eta(cell, kpts, mesh)
        log.debug1('get_nuc/get_pp_loc_part1 eta = %s mesh = %s', eta, mesh)

        # Initialize self.supmol
        if self.rs_cell is None:
            self.build()
        self.supmol = supmol = _strip_basis(self.supmol, eta)
        log.debug('stripped supmol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        modchg_cell = _compensate_nuccell(cell, eta)
        vj = self._int_nuc_vloc(modchg_cell, with_pseudo=with_pseudo,
                                supmol=supmol)
        t0 = t1 = log.timer_debug1('vnuc pass1: analytic int', *t0)

        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
        aoaux = ft_ao.ft_ao(modchg_cell, Gv)
        charges = cell.atom_charges()
        vG = np.einsum('i,xi->x', -charges, aoaux) * coulG

        supmol_ft = ft_ao._ExtendedMole.from_cell(self.rs_cell, self.bvk_kmesh, verbose=log)
        supmol_ft = supmol_ft.strip_basis()
        ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False, verbose=log)

        Gv, Gvbase, kws = modchg_cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        Gblksize = max(16, int(max_memory*1e6/16/nao_pair/nkpts))
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

    def get_nuc(self, mesh=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.

        Kwargs:
            mesh: custom mesh grids. By default mesh is determined by the
            function _guess_eta from module pbc.df.gdf_builder.
        '''
        t0 = (logger.process_clock(), logger.perf_counter())
        nuc = self._get_nuc(mesh, with_pseudo=False)
        logger.timer(self, 'get_nuc', *t0)
        return nuc

    def get_pp_loc_part1(self, mesh=None):
        return self._get_nuc(mesh, with_pseudo=True)

    def get_pp_loc_part2(self):
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
                  'int3c1e_r4_origk', 'int3c1e_r6_origk')
        fake_cells = [pp_int.fake_cell_vloc(cell, cn) for cn in range(1, 5)]

        supmol = self.supmol
        if supmol.nbas == supmol.bas_mask.size:  # supmol not stripped
            eta = min([np.hstack(c.bas_exps()).min()
                       for c in fake_cells if c.nbas > 0])
            supmol = _strip_basis(supmol, eta, inplace=False)

        bufR = 0
        bufI = 0
        for cn, fake_cell in enumerate(fake_cells, start=1):
            if fake_cell.nbas > 0:
                int3c = self.gen_int3c_kernel(intors[cn], 's2', comp=1, j_only=True,
                                              auxcell=fake_cell, supmol=supmol)
                vR, vI = int3c()
                bufR += np.einsum('...i->...', vR)
                if vI is not None:
                    bufI += np.einsum('...i->...', vI)

        if isinstance(bufR, int):
            if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
                pass
            else:
                lib.logger.warn(cell, 'cell.pseudo was specified but its elements %s '
                                 'were not found in the system.', cell._pseudo.keys())
            vpploc = [0] * nkpts
        else:
            buf = (bufR + bufI * 1j).reshape(nkpts,-1)
            vpploc = []
            for k, kpt in enumerate(kpts):
                v = lib.unpack_tril(buf[k])
                if abs(kpt).sum() < 1e-9:  # gamma_point:
                    v = v.real
                vpploc.append(v)
        return vpploc

    def get_pp(self, mesh=None):
        '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.

        Kwargs:
            mesh: custom mesh grids. By default mesh is determined by the
            function _guess_eta from module pbc.df.gdf_builder.
        '''
        t0 = (logger.process_clock(), logger.perf_counter())
        vloc1 = self.get_pp_loc_part1(mesh)
        t1 = logger.timer_debug1(self, 'get_pp_loc_part1', *t0)
        vloc2 = self.get_pp_loc_part2()
        t1 = logger.timer_debug1(self, 'get_pp_loc_part2', *t1)
        vpp = pp_int.get_pp_nl(self.cell, self.kpts)
        nkpts = len(self.kpts)
        for k in range(nkpts):
            vpp[k] += vloc1[k] + vloc2[k]
        t1 = logger.timer_debug1(self, 'get_pp_nl', *t1)
        logger.timer(self, 'get_pp', *t0)
        return vpp

# Since the real-space lattice-sum for nuclear attraction is not implemented,
# use the 3c2e code with steep gaussians to mimic nuclear density
def _fake_nuc(cell, with_pseudo=True):
    fakenuc = copy.copy(cell)
    fakenuc._atm = cell._atm.copy()
    fakenuc._atm[:,gto.PTR_COORD] = np.arange(gto.PTR_ENV_START,
                                              gto.PTR_ENV_START+cell.natm*3,3)
    _bas = []
    _env = [0]*gto.PTR_ENV_START + [cell.atom_coords().ravel()]
    ptr = gto.PTR_ENV_START + cell.natm * 3
    half_sph_norm = .5/np.sqrt(np.pi)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if with_pseudo and symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            eta = .5 / rloc**2
        else:
            eta = 1e16
        norm = half_sph_norm/gto.gaussian_int(2, eta)
        _env.extend([eta, norm])
        _bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        ptr += 2
    fakenuc._bas = np.asarray(_bas, dtype=np.int32)
    fakenuc._env = np.asarray(np.hstack(_env), dtype=np.double)
    fakenuc.rcut = cell.rcut
    return fakenuc

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

def _strip_basis(supmol, eta, cutoff=None, inplace=True):
    rs_cell = supmol.rs_cell

    supmol_exps = np.array([e.min() for e in supmol.bas_exps()])
    supmol_bas_coords = supmol.atom_coords()[supmol._bas[:,gto.ATOM_OF]]

    dim = rs_cell.dimension
    if dim == 0:
        bas_mask = np.ones(supmol.nbas, dtype=bool)
    else:
        if cutoff is None:
            cutoff = supmol.precision
        # estimation based on 3-center gaussian overlap integrals
        # ejk = minimize (ai*aj*|ri-rj|^2+ai*eta*ri^2+aj*eta*rj^2)/(ai+aj+eta)
        ajk = supmol_exps + eta
        ejk = supmol_exps * eta / ajk
        rb = np.linalg.norm(supmol_bas_coords[:,:dim], axis=1)
        a = rs_cell.lattice_vectors()
        dr = rb - np.linalg.norm(a[:dim])
        dr[dr < 0] = 0
        ovlp = ejk * dr**2
        bas_mask = ovlp < -np.log(cutoff)

    if not inplace:
        supmol = copy.copy(supmol)
    if bas_mask.size == supmol.bas_mask.size:
        supmol.bas_mask = bas_mask.reshape(supmol.bas_mask.shape)
    else:
        supmol.bas_mask = supmol.bas_mask.copy()
        supmol.bas_mask[supmol.bas_mask] = bas_mask
    supmol._bas = supmol._bas[bas_mask]
    supmol.sh_loc = supmol.bas_mask_to_sh_loc(rs_cell, supmol.bas_mask)
    return supmol
