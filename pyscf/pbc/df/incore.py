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
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
import pyscf.df
from pyscf.scf import _vhf
from pyscf.pbc.df import ft_ao
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_zero, unique
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import pbc as pbctools
from pyscf import __config__
from pyscf.pbc.gto import _pbcintor

RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.5)
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)
LOG_ADJUST = 32

libpbc = lib.load_library('libpbc')

def make_auxcell(cell, auxbasis=None):
    '''
    See pyscf.df.addons.make_auxmol
    '''
    auxcell = pyscf.df.addons.make_auxmol(cell, auxbasis)
    auxcell.rcut = pbcgto.estimate_rcut(auxcell)
    ke_cutoff = pbcgto.estimate_ke_cutoff(auxcell)
    a = cell.lattice_vectors()
    mesh = pbctools.cutoff_to_mesh(a, ke_cutoff)
    dimension = cell.dimension
    if dimension < 2 or (dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'):
        mesh[dimension:] = cell.mesh[dimension:]
    auxcell.mesh = mesh
    return auxcell

make_auxmol = make_auxcell

def format_aux_basis(cell, auxbasis='weigend+etb'):
    '''For backward compatibility'''
    return make_auxcell(cell, auxbasis)

def aux_e2(cell, auxcell_or_auxbasis, intor='int3c2e', aosym='s1', comp=None,
           kptij_lst=np.zeros((1,2,3)), shls_slice=None, **kwargs):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.

    Returns:
        (nao_pair, naux) array
    '''
    if isinstance(auxcell_or_auxbasis, gto.MoleBase):
        auxcell = auxcell_or_auxbasis
    else:
        assert isinstance(auxcell_or_auxbasis, str)
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
    dfbuilder = Int3cBuilder(cell, auxcell, kpts).build()
    int3c = dfbuilder.gen_int3c_kernel(intor, aosym, comp, j_only,
                                       kptij_idx, return_complex=True)
    return int3c


def fill_2c2e(cell, auxcell_or_auxbasis, intor='int2c2e', hermi=0, kpt=np.zeros(3)):
    '''2-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    if isinstance(auxcell_or_auxbasis, gto.MoleBase):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    if hermi != 0:
        hermi = pyscf.lib.HERMITIAN
# pbcopt use the value of AO-pair to prescreening PBC integrals in the lattice
# summation.  Pass NULL pointer to pbcopt to prevent the prescreening
    return auxcell.pbc_intor(intor, 1, hermi, kpt, pbcopt=lib.c_null_ptr())


class Int3cBuilder(lib.StreamObject):
    '''helper functions to compute 3-center integral tensor with double-lattice sum
    '''

    _keys = {
        'cell', 'auxcell', 'kpts', 'rs_cell', 'bvk_kmesh',
        'supmol', 'ke_cutoff', 'direct_scf_tol',
    }

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
        self.direct_scf_tol = None

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.rs_cell = None
        self.supmol = None
        self.direct_scf_tol = None
        return self

    def build(self):
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)

        # TODO: strip supmol basis based on the intor type.
        rcut = estimate_rcut(cell, self.auxcell)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max(), log)
        self.supmol = supmol.strip_basis(rcut)
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
        if rs_auxcell is None:
            rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(auxcell)
        elif not isinstance(rs_auxcell, ft_ao._RangeSeparatedCell):
            rs_auxcell = ft_ao._RangeSeparatedCell.from_cell(rs_auxcell)
        if supmol is None:
            supmol = self.supmol
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
        nbasp = cell.nbas  # The number of shells in the primitive cell

        if self.direct_scf_tol is None:
            omega = supmol.omega
            aux_exp = np.hstack(auxcell.bas_exps()).min()
            cell_exp = np.hstack(cell.bas_exps()).min()
            if omega == 0:
                theta = 1./(1./cell_exp + 1./aux_exp)
            else:
                theta = 1./(1./cell_exp + 1./aux_exp + omega**-2)
            lattice_sum_factor = max(2*np.pi*cell.rcut/(cell.vol*theta), 1)
            cutoff = cell.precision / lattice_sum_factor**2 * .1
            log.debug1('int3c_kernel integral omega=%g theta=%g cutoff=%g',
                       omega, theta, cutoff)
        else:
            cutoff = self.direct_scf_tol
        log_cutoff = int(np.log(cutoff) * LOG_ADJUST)

        atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                     rs_auxcell._atm, rs_auxcell._bas, rs_auxcell._env)
        cell0_ao_loc = _conc_locs(cell.ao_loc, auxcell.ao_loc)
        seg_loc = _conc_locs(supmol.seg_loc, rs_auxcell.sh_loc)
        # mimic the lattice sum with range 1
        aux_seg2sh = np.arange(rs_auxcell.nbas + 1)
        seg2sh = _conc_locs(supmol.seg2sh, aux_seg2sh)

        if 'ECP' in intor:
            # rs_auxcell is a placeholder only to represent the ecpbas.
            # Ensure the ECPBAS_OFFSET be consistent with the treatment in pbc.gto.ecp
            env[gto.AS_ECPBAS_OFFSET] = len(bas)
            bas = np.asarray(np.vstack([bas, cell._ecpbas]), dtype=np.int32)
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
            # sindex may not be accurate enough to screen ECP integral.
            # Add penalty 1e-2 to reduce the screening error
            log_cutoff = int(np.log(cutoff*1e-2) * LOG_ADJUST)
        else:
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)

        sindex = self.get_q_cond(supmol)
        ovlp_mask = sindex > log_cutoff
        bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, supmol.sh_loc)
        cell0_ovlp_mask = bvk_ovlp_mask.reshape(
            bvk_ncells, nbasp, bvk_ncells, nbasp).any(axis=2).any(axis=0)
        cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)
        ovlp_mask = None

        # Estimate the buffer size required by PBCfill_nr3c functions
        cache_size = max(_get_cache_size(cell, intor),
                         _get_cache_size(rs_auxcell, intor))
        cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
        dijk = int(cell0_dims[:nbasp].max())**2 * int(cell0_dims[nbasp:].max()) * comp

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
            assert nkpts < 45000
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
        # Only supports int3c2e_sph, int3c2e_cart in current C library
        is_pbcintor = intor in ('int3c2e_sph', 'int3c2e_cart') or intor[:3] == 'ECP'
        if is_pbcintor and not intor.startswith('PBC'):
            intor = 'PBC' + intor
        log.debug1('is_pbcintor = %d, intor = %s', is_pbcintor, intor)

        log.timer('int3c kernel initialization', *cput0)

        def int3c(shls_slice=None, outR=None, outI=None):
            cput0 = logger.process_clock(), logger.perf_counter()
            if shls_slice is None:
                shls_slice = [0, nbasp, 0, nbasp, nbasp, len(cell0_dims)]
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

            if sindex is None:
                sindex_ptr = lib.c_null_ptr()
            else:
                sindex_ptr = sindex.ctypes.data_as(ctypes.c_void_p)

            drv(getattr(libpbc, intor), getattr(libpbc, fill),
                ctypes.c_int(is_pbcintor),
                outR.ctypes.data_as(ctypes.c_void_p),
                outI.ctypes.data_as(ctypes.c_void_p),
                expLkR.ctypes.data_as(ctypes.c_void_p),
                expLkI.ctypes.data_as(ctypes.c_void_p),
                reindex_k.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts_ij),
                ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs),
                ctypes.c_int(nkpts), ctypes.c_int(nbasp), ctypes.c_int(comp),
                seg_loc.ctypes.data_as(ctypes.c_void_p),
                seg2sh.ctypes.data_as(ctypes.c_void_p),
                cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*6)(*shls_slice),
                cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
                sindex_ptr, ctypes.c_int(log_cutoff),
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

    def get_q_cond(self, supmol=None):
        '''Integral screening condition max(sqrt((ij|ij))) inside the supmol'''
        if supmol is None:
            supmol = self.supmol
        nbas = supmol.nbas
        sindex = np.empty((nbas,nbas), dtype=np.int16)
        libpbc.PBCVHFsetnr_sindex(
            sindex.ctypes.data_as(ctypes.c_void_p),
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))
        return sindex


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

def estimate_rcut(cell, auxcell, precision=None):
    '''Estimate rcut for 3c2e integrals'''
    if precision is None:
        precision = cell.precision

    cell_exps = np.array([e.min() for e in cell.bas_exps()])
    aux_exps = np.array([e.min() for e in auxcell.bas_exps()])
    if cell_exps.size == 0 or aux_exps.size == 0:
        return np.zeros(1)

    ls = cell._bas[:,gto.ANG_OF]
    cs = gto.gto_norm(ls, cell_exps)

    ai_idx = cell_exps.argmin()
    ak_idx = aux_exps.argmin()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    ak = aux_exps[ak_idx]
    li = cell._bas[ai_idx,gto.ANG_OF]
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
    fac = 2**(li+1)*np.pi**3.5*c1 * theta**(l3-1.5) / aij**(lij+1.5) / ak**(lk+1.5)
    fac *= (1 + ai/aj)**lj * fl / precision

    r0 = cell.rcut
    r0 = (np.log(fac * r0 * (sfac*r0)**(l3-2) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l3-2) + 1.) / (sfac*theta))**.5
    return r0

def _conc_locs(ao_loc1, ao_loc2):
    '''auxiliary basis was appended to regular AO basis when calling int3c2e
    integrals. Composite loc combines locs from regular AO basis and auxiliary
    basis accordingly.'''
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return np.asarray(comp_loc, dtype=np.int32)

# The following functions use pre-constructed shell pair list
def aux_e2_sum_auxbas(cell, auxcell_or_auxbasis, intor='int3c2e', aosym='s1', comp=None,
                      kptij_lst=np.zeros((1,2,3)), shls_slice=None, **kwargs):
    r'''Compute :math:`\sum_{L} (ij|L)` on the fly.

    Returns:
        out : (nao_pair,) array
    '''
    if isinstance(auxcell_or_auxbasis, gto.MoleBase):
        auxcell = auxcell_or_auxbasis
    else:
        assert isinstance(auxcell_or_auxbasis, str)
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

    int3c = wrap_int3c_sum_auxbas(cell, auxcell, intor, aosym, comp, kptij_lst, **kwargs)
    out = int3c(shls_slice)
    return out

def wrap_int3c_sum_auxbas(cell, auxcell, intor='int3c2e', aosym='s1', comp=None,
                          kptij_lst=np.zeros((1,2,3)), cintopt=None, pbcopt=None,
                          neighbor_list=None):
    if neighbor_list is None:
        raise KeyError('Neighbor list is not initialized.')

    log = logger.new_logger(cell)

    nkptij = len(kptij_lst)
    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    j_only = is_zero(kpti - kptj)
    if j_only:
        kpts = kpti
        nkpts = len(kpts)
        kptij_idx = np.arange(nkpts, dtype=np.int32)
    else:
        raise NotImplementedError

    intor = cell._add_suffix(intor)
    intor, comp = gto.moleintor._get_intor_and_comp(intor, comp)

    pcell = cell.copy()
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                         cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr()
    ao_loc = np.asarray(np.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                        dtype=np.int32)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)

    Ls = cell.get_lattice_Ls()
    nimgs = len(Ls)
    nbas = cell.nbas

    gamma_point_only = is_zero(kpts)
    if gamma_point_only:
        assert nkpts == 1
        kk_type = 'g'
        expkL = np.ones(1, dtype=np.complex128)
        out_dtype = np.double
    else:
        raise NotImplementedError

    fill = 'PBCnr3c_screened_sum_auxbas_fill_%s%s' % (kk_type, aosym[:2])
    drv = libpbc.PBCnr3c_screened_sum_auxbas_drv

    if cintopt is None:
        if nbas > 0:
            env[gto.PTR_EXPCUTOFF] = abs(np.log(cell.precision))
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)
    if pbcopt is None:
        pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell)
    if isinstance(pbcopt, _pbcintor.PBCOpt):
        cpbcopt = pbcopt._this
    else:
        cpbcopt = lib.c_null_ptr()

    def int3c(shls_slice=None, out=None):
        t0 = (logger.process_clock(), logger.perf_counter())
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, 0, auxcell.nbas)
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas+shls_slice[2], nbas+shls_slice[3],
                      nbas*2+shls_slice[4], nbas*2+shls_slice[5])
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]

        if aosym[:2] == 's2':
            assert ni == nj
            nao_pair = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
                        ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
        else:
            nao_pair = ni * nj

        if out is None:
            out = np.empty((nkptij,comp,nao_pair), dtype=out_dtype)

        drv(getattr(libpbc, intor), getattr(libpbc, fill),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkptij), ctypes.c_int(nkpts),
            ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            expkL.ctypes.data_as(ctypes.c_void_p),
            kptij_idx.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cpbcopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCnr3c_drv
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
            ctypes.byref(neighbor_list))

        log.timer_debug1(f'pbc integral {intor}', *t0)

        if comp == 1:
            out = out[:,0]
        if nkptij == 1:
            out = out[0]
        return out

    return int3c

def int3c1e_nuc_grad(cell, auxcell, dm, intor='int3c1e', aosym='s1', comp=3,
                     kptij_lst=np.zeros((1,2,3)), shls_slice=None, **kwargs):
    '''Compute the nuclear gradient contribution
    to the 2nd local part of PP on the fly.
    See `pbc.gto.pseudo.pp_int.vpploc_part2_nuc_grad`.

    Returns:
        out : (natm,comp) array
    '''
    if comp != 3:
        raise NotImplementedError
    if aosym != 's1':
        raise NotImplementedError

    int3c = wrap_int3c1e_nuc_grad(cell, auxcell, dm, intor, aosym, comp, kptij_lst, **kwargs)
    out = int3c(shls_slice)
    return out

def wrap_int3c1e_nuc_grad(cell, auxcell, dm, intor='int3c1e', aosym='s1', comp=3,
                          kptij_lst=np.zeros((1,2,3)), cintopt=None, pbcopt=None,
                          neighbor_list=None):
    if neighbor_list is None:
        raise KeyError('Neighbor list is not initialized.')

    log = logger.new_logger(cell)

    nkptij = len(kptij_lst)
    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    j_only = is_zero(kpti - kptj)
    if j_only:
        kpts = kpti
        nkpts = len(kpts)
        kptij_idx = np.arange(nkpts, dtype=np.int32)
    else:
        raise NotImplementedError

    intor = cell._add_suffix(intor)
    intor, comp = gto.moleintor._get_intor_and_comp(intor, comp)

    pcell = cell.copy()
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                         cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr()
    ao_loc = np.asarray(np.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                        dtype=np.int32)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)

    Ls = cell.get_lattice_Ls()
    nimgs = len(Ls)
    nbas = cell.nbas

    gamma_point_only = is_zero(kpts)
    if gamma_point_only:
        assert nkpts == 1
        kk_type = 'g'
        expkL = np.ones(1, dtype=np.complex128)
        dm = np.asarray(dm, order="C", dtype=np.double)
    else:
        raise NotImplementedError

    fill = 'PBCnr3c1e_screened_nuc_grad_fill_%s%s' % (kk_type, aosym[:2])
    drv = libpbc.PBCnr3c1e_screened_nuc_grad_drv

    if cintopt is None:
        if nbas > 0:
            env[gto.PTR_EXPCUTOFF] = abs(np.log(cell.precision))
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)
    if pbcopt is None:
        pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell)
    if isinstance(pbcopt, _pbcintor.PBCOpt):
        cpbcopt = pbcopt._this
    else:
        cpbcopt = lib.c_null_ptr()

    def int3c(shls_slice=None, out=None):
        t0 = (logger.process_clock(), logger.perf_counter())
        if shls_slice is None:
            shls_slice = (0, nbas, 0, nbas, 0, auxcell.nbas)
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas+shls_slice[2], nbas+shls_slice[3],
                      nbas*2+shls_slice[4], nbas*2+shls_slice[5])

        if out is None:
            out = np.zeros((nkptij,cell.natm,comp), dtype=np.double)

        drv(getattr(libpbc, intor), getattr(libpbc, fill),
            out.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkptij), ctypes.c_int(nkpts),
            ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            expkL.ctypes.data_as(ctypes.c_void_p),
            kptij_idx.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cpbcopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
            ctypes.c_int(cell.nao), ctypes.byref(neighbor_list))

        log.timer_debug1(f'pbc integral {intor}', *t0)

        if nkptij == 1:
            out = out[0]
        return out

    return int3c
