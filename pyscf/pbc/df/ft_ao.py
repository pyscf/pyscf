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
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

libpbc = lib.load_library('libpbc')

#
# \int mu*nu*exp(-ik*r) dr
#
def ft_aopair(cell, Gv, shls_slice=None, aosym='s1',
              b=None, gxyz=None, Gvbase=None, kpti_kptj=numpy.zeros((2,3)),
              q=None, intor='GTO_ft_ovlp', comp=1, verbose=None):
    r'''
    FT transform AO pair
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3
    '''
    kpti, kptj = kpti_kptj
    if q is None:
        q = kptj - kpti
    val = _ft_aopair_kpts(cell, Gv, shls_slice, aosym, b, gxyz, Gvbase,
                          q, kptj.reshape(1,3), intor, comp)
    return val[0]

# NOTE buffer out must be initialized to 0
# gxyz is the index for Gvbase
def _ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                    b=None, gxyz=None, Gvbase=None, q=numpy.zeros(3),
                    kptjs=numpy.zeros((1,3)), intor='GTO_ft_ovlp', comp=1,
                    out=None):
    r'''
    FT transform AO pair
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
    expkL = numpy.exp(1j * numpy.dot(kptjs, Ls.T))

    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    if shls_slice is None:
        shls_slice = (0, cell.nbas, cell.nbas, cell.nbas*2)
    else:
        shls_slice = (shls_slice[0], shls_slice[1],
                      cell.nbas+shls_slice[2], cell.nbas+shls_slice[3])
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nkpts = len(kptjs)
    nimgs = len(Ls)
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

    drv = libpbc.PBC_ft_latsum_drv
    cintor = getattr(libpbc, intor)
    eval_gz = getattr(libpbc, eval_gz)
    if nkpts == 1:
        fill = getattr(libpbc, 'PBC_ft_fill_nk1'+aosym)
    else:
        fill = getattr(libpbc, 'PBC_ft_fill_k'+aosym)
    out = numpy.ndarray(shape, dtype=numpy.complex128, buffer=out)

    drv(cintor, eval_gz, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
        Ls.ctypes.data_as(ctypes.c_void_p), expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
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

def ft_ao(mol, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=numpy.zeros(3), verbose=None):
    if gamma_point(kpt):
        return mol_ft_ao(mol, Gv, shls_slice, b, gxyz, Gvbase, verbose)
    else:
        kG = Gv + kpt
        return mol_ft_ao(mol, kG, shls_slice, None, None, None, verbose)

if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.dft.numint
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
    #cell.basis = {'C': [[0, (2.4, 1)]]}
    cell.unit = 'B'
    #cell.verbose = 4
    cell.build(0,0)
    #cell.nimgs = (2,2,2)

    ao2 = ft_aopair(cell, cell.Gv)
    nao = cell.nao_nr()
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = cell.pbc_eval_gto('GTOval', coords)
    aoR2 = numpy.einsum('ki,kj->kij', aoR.conj(), aoR)
    ngrids = aoR.shape[0]

    for i in range(nao):
        for j in range(nao):
            ao2ref = tools.fft(aoR2[:,i,j], cell.mesh) * cell.vol/ngrids
            print(i, j, numpy.linalg.norm(ao2ref - ao2[:,i,j]))

    aoG = ft_ao(cell, cell.Gv)
    for i in range(nao):
        aoref = tools.fft(aoR[:,i], cell.mesh) * cell.vol/ngrids
        print(i, numpy.linalg.norm(aoref - aoG[:,i]))

