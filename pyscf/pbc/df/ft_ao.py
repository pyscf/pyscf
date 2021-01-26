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

'''
Analytic Fourier transformation AO-pair value for PBC
'''

import ctypes
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.gto.ft_ao import ft_ao as mol_ft_ao
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

libpbc = lib.load_library('libpbc')

#
# \int mu*nu*exp(-ik*r) dr
#
def ft_aopair(cell, Gv, shls_slice=None, aosym='s1',
              b=None, gxyz=None, Gvbase=None, kpti_kptj=numpy.zeros((2,3)),
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

# NOTE buffer out must be initialized to 0
# gxyz is the index for Gvbase
def ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                   b=None, gxyz=None, Gvbase=None, q=numpy.zeros(3),
                   kptjs=numpy.zeros((1,3)), intor='GTO_ft_ovlp', comp=1,
                   bvk_kmesh=None, out=None):
    r'''
    Fourier transform AO pair for a group of k-points
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The return array holds the AO pair
    corresponding to the kpoints given by kptjs
    '''

    intor = cell._add_suffix(intor)

    q = numpy.reshape(q, 3)
    kptjs = numpy.asarray(kptjs, order='C').reshape(-1,3)
    Gv = numpy.asarray(Gv, order='C').reshape(-1,3)
    nGv = Gv.shape[0]
    GvT = numpy.asarray(Gv.T, order='C')
    GvT += q.reshape(-1,1)

    if (gxyz is None or b is None or Gvbase is None or (abs(q).sum() > 1e-9)
        # backward compatibility for pyscf-1.2, in which the argument Gvbase is gs
        or (Gvbase is not None and isinstance(Gvbase[0], (int, numpy.integer)))):
        p_gxyzT = lib.c_null_ptr()
        p_mesh = (ctypes.c_int*3)(0,0,0)
        p_b = (ctypes.c_double*1)(0)
        eval_gz = 'GTO_Gv_general'
    else:
        if abs(b-numpy.diag(b.diagonal())).sum() < 1e-8:
            eval_gz = 'GTO_Gv_orth'
        else:
            eval_gz = 'GTO_Gv_nonorth'
        gxyzT = numpy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        b = numpy.hstack((b.ravel(), q) + Gvbase)
        p_b = b.ctypes.data_as(ctypes.c_void_p)
        p_mesh = (ctypes.c_int*3)(*[len(x) for x in Gvbase])

    Ls = cell.get_lattice_Ls()
    Ls = Ls[numpy.linalg.norm(Ls, axis=1).argsort()]
    nkpts = len(kptjs)
    nimgs = len(Ls)
    nbas = cell.nbas

    if bvk_kmesh is None:
        expkL = numpy.exp(1j * numpy.dot(kptjs, Ls.T))
    else:
        ovlp_mask = _estimate_overlap(cell, Ls) > cell.precision
        ovlp_mask = numpy.asarray(ovlp_mask, dtype=numpy.int8, order='C')

        # Using Ls = translations.dot(a)
        translations = numpy.linalg.solve(cell.lattice_vectors().T, Ls.T)
        # t_mod is the translations inside the BvK cell
        t_mod = translations.round(3).astype(int) % numpy.asarray(bvk_kmesh)[:,None]
        cell_loc_bvk = numpy.ravel_multi_index(t_mod, bvk_kmesh).astype(numpy.int32)

        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh)
        expkL = numpy.exp(1j * numpy.dot(kptjs, bvkmesh_Ls.T))

    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    if shls_slice is None:
        shls_slice = (0, nbas, nbas, nbas*2)
    else:
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas+shls_slice[2], nbas+shls_slice[3])
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    shape = (nkpts, comp, ni, nj, nGv)

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# hermi operation needs reordering the axis-0.  It is inefficient.
    if aosym == 's1hermi': # Symmetry for Gamma point
        assert(is_zero(q) and is_zero(kptjs) and ni == nj)
    elif aosym == 's2':
        i0 = ao_loc[shls_slice[0]]
        i1 = ao_loc[shls_slice[1]]
        nij = i1*(i1+1)//2 - i0*(i0+1)//2
        shape = (nkpts, comp, nij, nGv)

    cintor = getattr(libpbc, intor)
    eval_gz = getattr(libpbc, eval_gz)

    out = numpy.ndarray(shape, dtype=numpy.complex128, buffer=out)

    if bvk_kmesh is None:
        if nkpts == 1:
            fill = getattr(libpbc, 'PBC_ft_fill_nk1'+aosym)
        else:
            fill = getattr(libpbc, 'PBC_ft_fill_k'+aosym)
        drv = libpbc.PBC_ft_latsum_drv
        drv(cintor, eval_gz, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p), expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
            GvT.ctypes.data_as(ctypes.c_void_p), p_b, p_gxyzT, p_mesh, ctypes.c_int(nGv),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
            env.ctypes.data_as(ctypes.c_void_p))
    else:
        if nkpts == 1:
            fill = getattr(libpbc, 'PBC_ft_bvk_nk1'+aosym)
        else:
            fill = getattr(libpbc, 'PBC_ft_bvk_k'+aosym)
        drv = libpbc.PBC_ft_bvk_drv
        drv(cintor, eval_gz, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp),
            ctypes.c_int(nimgs), ctypes.c_int(expkL.shape[1]),
            Ls.ctypes.data_as(ctypes.c_void_p), expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
            cell_loc_bvk.ctypes.data_as(ctypes.c_void_p),
            ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            GvT.ctypes.data_as(ctypes.c_void_p), p_b, p_gxyzT, p_mesh, ctypes.c_int(nGv),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
            env.ctypes.data_as(ctypes.c_void_p))

    if aosym == 's1hermi':
        for i in range(1,ni):
            out[:,:,:i,i] = out[:,:,i,:i]
    out = numpy.rollaxis(out, -1, 2)
    if comp == 1:
        out = out[:,0]
    return out

@lib.with_doc(mol_ft_ao.__doc__)
def ft_ao(mol, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=numpy.zeros(3), verbose=None):
    if gamma_point(kpt):
        return mol_ft_ao(mol, Gv, shls_slice, b, gxyz, Gvbase, verbose)
    else:
        kG = Gv + kpt
        return mol_ft_ao(mol, kG, shls_slice, None, None, None, verbose)

def _estimate_overlap(cell, Ls):
    '''Consider the lattice sum in overlap when estimating the ss-type overlap
    integrals for each traslation vector
    '''
    exps = numpy.array([cell.bas_exp(ib).min() for ib in range(cell.nbas)])
    atom_coords = cell.atom_coords()
    bas_coords = atom_coords[cell._bas[:,gto.ATOM_OF]]
    aij = exps[:,None] * exps / (exps[:,None] + exps)
    rij = bas_coords[:,None,:] - bas_coords
    dijL = numpy.linalg.norm(rij[:,:,None,:] - Ls, axis=-1)
    vol = cell.vol
    vol_rad = vol**(1./3)
    fac = (4 * aij / (exps[:,None] + exps))**.75
    s = fac[:,:,None] * numpy.exp(-aij[:,:,None] * dijL**2)
    s_cum = fac[:,:,None] * numpy.exp(-aij[:,:,None] * (dijL - vol_rad/2)**2)
    fac = 2*numpy.pi/vol / aij[:,:,None] * abs(dijL - vol_rad/2)
    return numpy.max([fac * s_cum, s], axis=0)

if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    from pyscf.pbc import tools

    L = 5.
    n = 20
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''C    1.3    .2       .3
                   C     .1    .1      1.1
                   '''
    cell.basis = 'ccpvdz'
    #cell.basis = {'C': [[0, (2.4, .1, .6), (1.0,.8, .4)], [1, (1.1, 1)]]}
    #cell.basis = {'C': [[1, (1.1, 1)]]}
    cell.unit = 'B'
    #cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([2,2,2])
    Gv, Gvbase, kws = cell.get_Gv_weights()
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)

    ref = ft_aopair_kpts(cell, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase, kptjs=kpts)
    aopair = ft_aopair_kpts(cell, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase,
                            kptjs=kpts, bvk_kmesh=bvk_kmesh)
    print(abs(ref - aopair).max())
    print(lib.fp(aopair) - (11.437194884312916-6.14510793811694j))
