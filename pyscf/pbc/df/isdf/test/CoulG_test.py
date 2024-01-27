#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import copy
from functools import reduce
import numpy as np
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk

import sys
import ctypes
import _ctypes

from multiprocessing import Pool

import dask.array as da
from dask import delayed

from memory_profiler import profile

libpbc = lib.load_library('libpbc')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, name))

if __name__ == '__main__':

    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

    cell.atom = '''
                   C     0.      0.      0.
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751
                '''

    cell.basis   = 'gth-szv'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
    # cell.ke_cutoff = 32
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 1])

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
    df_tmp = MultiGridFFTDF2(cell)
    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    mesh   = grids.mesh

    coulG = tools.get_coulG(cell, mesh=mesh)
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)
    ngrids_real = coulG_real.shape[0]
    mesh_real = np.array([mesh[0], mesh[1], mesh[2]//2+1])
    print("coulG_real.shape = ", coulG_real.shape)

    nAux = 33
    ngrids = coords.shape[0]
    aux_basis = np.random.rand(nAux, ngrids).reshape(-1, *mesh)

    ############## test the construction of V ################

    # bench mark

    V = (np.fft.ifftn((np.fft.fftn(aux_basis, axes=(1,2,3)).reshape(-1, ngrids) *
         coulG[None,:]).reshape(-1, *mesh), axes=(1,2,3)).real).reshape(-1, ngrids)
    print("V.shape = ", V.shape)

    V_Real = (np.fft.irfftn((np.fft.rfftn(aux_basis, axes=(1,2,3)).reshape(-1, ngrids_real) *
              coulG_real[None,:]).reshape(-1, *mesh_real), axes=(1,2,3)).real).reshape(-1, ngrids)  # this will reduce half of the memory and cost
    print("V_Real.shape = ", V_Real.shape)

    print("np.allclose(V, V_Real) = ", np.allclose(V, V_Real))



    # test the construction of V in C

    fn = getattr(libpbc, "_construct_V", None)
    assert(fn is not None)

    V_C = np.zeros((nAux, ngrids), dtype=np.float64)
    nThread = lib.num_threads()
    # bunchsize = nAux // nThread
    bunchsize = 1

    bufsize = bunchsize * coulG_real.shape[0] * 2
    bufsize = (bufsize + 15) // 16 * 16
    bufsize = bufsize * nThread

    print("bufsize = ", bufsize)

    bufsize_per_thread = bufsize // nThread

    buf = np.empty(bufsize, dtype=np.float64)
    bufsize_per_thread = bufsize // nThread

    mesh_int32 = np.array(mesh, dtype=np.int32)

    fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nAux),
       aux_basis.ctypes.data_as(ctypes.c_void_p),
       coulG_real.ctypes.data_as(ctypes.c_void_p),
       V_C.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(bunchsize),
       buf.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(bufsize_per_thread))

    print("np.allclose(V, V_C) = ", np.allclose(V, V_C))