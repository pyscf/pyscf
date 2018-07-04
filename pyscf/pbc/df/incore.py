#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
import numpy
from pyscf import lib
from pyscf import gto
import pyscf.df
from pyscf.scf import _vhf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL

libpbc = lib.load_library('libpbc')

#try:
### Moderate speedup by caching eval_ao
#    from joblib import Memory
#    memory = Memory(cachedir='./tmp/', mmap_mode='r', verbose=0)
#    def memory_cache(f):
#        g = memory.cache(f)
#        def maybe_cache(*args, **kwargs):
#            if pyscf.pbc.DEBUG:
#                return g(*args, **kwargs)
#            else:
#                return f(*args, **kwargs)
#        return maybe_cache
#except:
#    memory_cache = lambda f: f

def make_auxmol(cell, auxbasis=None):
    '''
    See pyscf.df.addons.make_auxmol
    '''
    auxcell = pyscf.df.addons.make_auxmol(cell, auxbasis)
    auxcell.rcut = max([auxcell.bas_rcut(ib, cell.precision)
                        for ib in range(auxcell.nbas)])
    return auxcell
def format_aux_basis(cell, auxbasis='weigend+etb'):
    '''For backward compatibility'''
    return make_auxmol(cell, auxbasis)

#@memory_cache
def aux_e2(cell, auxcell, intor='int3c2e', aosym='s1', comp=None,
           kptij_lst=numpy.zeros((1,2,3)), shls_slice=None, **kwargs):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.

    Returns:
        (nao_pair, naux) array
    '''
# For some unkown reasons, the pre-decontracted basis 'is slower than
#    if shls_slice is None and cell.nao_nr() < 200:
## Slighly decontract basis. The decontracted basis has better locality.
## The locality can be used in the lattice sum to reduce cost.
#        cell, contr_coeff = pbcgto.cell._split_basis(cell)
#    else:
#        contr_coeff = None

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    naux = aux_loc[shls_slice[5]] - aux_loc[shls_slice[4]]

    nkptij = len(kptij_lst)

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    j_only = is_zero(kpti-kptj)

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
                    ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    else:
        nao_pair = ni * nj

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    int3c = wrap_int3c(cell, auxcell, intor, aosym, comp, kptij_lst, **kwargs)
    out = numpy.empty((nkptij,comp,nao_pair,naux), dtype=dtype)
    out = int3c(shls_slice, out)

#    if contr_coeff is not None:
#        if aosym == 's2':
#            tmp = out.reshape(nkptij,comp,ni,ni,naux)
#            idx, idy = numpy.tril_indices(ni)
#            tmp[:,:,idy,idx] = out.conj()
#            tmp[:,:,idx,idy] = out
#            out, tmp = tmp, None
#            out = lib.einsum('kcpql,pi->kciql', out, contr_coeff)
#            out = lib.einsum('kciql,qj->kcijl', out, contr_coeff)
#            idx, idy = numpy.tril_indices(contr_coeff.shape[1])
#            out = out[:,:,idx,idy]
#        else:
#            out = out.reshape(nkptij,comp,ni,nj,naux)
#            out = lib.einsum('kcpql,pi->kciql', out, contr_coeff)
#            out = lib.einsum('kciql,qj->kcijl', out, contr_coeff)
#            out = out.reshape(nkptij,comp,-1,naux)

    if comp == 1:
        out = out[:,0]
    if nkptij == 1:
        out = out[0]
    return out

def wrap_int3c(cell, auxcell, intor='int3c2e', aosym='s1', comp=1,
               kptij_lst=numpy.zeros((1,2,3)), cintopt=None, pbcopt=None):
    intor = cell._add_suffix(intor)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    ao_loc = numpy.asarray(numpy.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                           dtype=numpy.int32)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)
    Ls = cell.get_lattice_Ls()
    nimgs = len(Ls)
    nbas = cell.nbas

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    if gamma_point(kptij_lst):
        kk_type = 'g'
        dtype = numpy.double
        nkpts = nkptij = 1
        kptij_idx = numpy.array([0], dtype=numpy.int32)
        expkL = numpy.ones(1)
    elif is_zero(kpti-kptj):  # j_only
        kk_type = 'k'
        dtype = numpy.complex128
        kpts = kptij_idx = numpy.asarray(kpti, order='C')
        expkL = numpy.exp(1j * numpy.dot(kpts, Ls.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = numpy.complex128
        kpts = unique(numpy.vstack([kpti,kptj]))[0]
        expkL = numpy.exp(1j * numpy.dot(kpts, Ls.T))
        wherei = numpy.where(abs(kpti.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        wherej = numpy.where(abs(kptj.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = numpy.asarray(wherei*nkpts+wherej, dtype=numpy.int32)
        nkptij = len(kptij_lst)

    fill = 'PBCnr3c_fill_%s%s' % (kk_type, aosym[:2])
    drv = libpbc.PBCnr3c_drv
    if cintopt is None:
        if nbas > 0:
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
# Remove the precomputed pair data because the pair data corresponds to the
# integral of cell #0 while the lattice sum moves shls to all repeated images.
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)
    if pbcopt is None:
        pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell)
    if isinstance(pbcopt, _pbcintor.PBCOpt):
        cpbcopt = pbcopt._this
    else:
        cpbcopt = lib.c_null_ptr()

    def int3c(shls_slice, out):
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas+shls_slice[2], nbas+shls_slice[3],
                      nbas*2+shls_slice[4], nbas*2+shls_slice[5])
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
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        return out
    return int3c


def fill_2c2e(cell, auxcell, intor='int2c2e', hermi=0, kpt=numpy.zeros(3)):
    '''2-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    if hermi != 0:
        hermi = pyscf.lib.HERMITIAN
# pbcopt use the value of AO-pair to prescreening PBC integrals in the lattice
# summation.  Pass NULL pointer to pbcopt to prevent the prescreening
    return auxcell.pbc_intor(intor, 1, hermi, kpt, pbcopt=lib.c_null_ptr())

