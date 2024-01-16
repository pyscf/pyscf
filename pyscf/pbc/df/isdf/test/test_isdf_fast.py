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

import sys
import ctypes
import _ctypes

libpbc = lib.load_library('libpbc')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, name))

'''
/// the following variables are input variables
    int nao;
    int natm;
    int ngrids;
    double cutoff_aoValue;
    const int *ao2atomID;
    const double *aoG;
    double cutoff_QR;
/// the following variables are output variables
    int *voronoi_partition;
    int *ao_sparse_rep_row;
    int *ao_sparse_rep_col;
    double *ao_sparse_rep_val;
    int naux;
    int *IP_index;
    double *auxiliary_basis;

'''
class _PBC_ISDF(ctypes.Structure):
    _fields_ = [('nao', ctypes.c_int),
                ('natm', ctypes.c_int),
                ('ngrids', ctypes.c_int),
                ('cutoff_aoValue', ctypes.c_int),
                ('ao2atomID', ctypes.c_void_p),
                ('aoG', ctypes.c_void_p),
                ('cutoff_QR', ctypes.c_double),
                ('voronoi_partition', ctypes.c_void_p),
                ('ao_sparse_rep_row', ctypes.c_void_p),
                ('ao_sparse_rep_col', ctypes.c_void_p),
                ('ao_sparse_rep_val', ctypes.c_void_p),
                ('naux', ctypes.c_int),
                ('IP_index', ctypes.c_void_p),
                ('auxiliary_basis', ctypes.c_void_p)
                ]

class PBC_ISDF_Info(object):
    def __init__(self, mol:Cell, aoR: np.ndarray,
                 cutoff_aoValue: float = 1e-12,
                 cutoff_QR: float = 1e-8):

        self._this = ctypes.POINTER(_PBC_ISDF)()

        nao = ctypes.c_int(mol.nao_nr())
        natm = ctypes.c_int(mol.natm)
        ngrids = ctypes.c_int(aoR.shape[1])
        _cutoff_aoValue = ctypes.c_double(cutoff_aoValue)
        _cutoff_QR = ctypes.c_double(cutoff_QR)

        assert nao.value == aoR.shape[0]

        ao2atomID = np.zeros(nao.value, dtype=np.int32)

        # only valid for spherical GTO

        ao_loc = 0
        for i in range(mol._bas.shape[0]):
            atm_id = mol._bas[i, ATOM_OF]
            nctr   = mol._bas[i, NCTR_OF]
            angl   = mol._bas[i, ANG_OF]
            nao_now = nctr * (2 * angl + 1)
            ao2atomID[ao_loc:ao_loc+nao_now] = atm_id
            ao_loc += nao_now

        print("ao2atomID = ", ao2atomID)


        libpbc.PBC_ISDF_init(ctypes.byref(self._this),
                                nao, natm, ngrids,
                                _cutoff_aoValue,
                                ao2atomID.ctypes.data_as(ctypes.c_void_p),
                                aoR.ctypes.data_as(ctypes.c_void_p),
                                _cutoff_QR)

    def __del__(self):
        try:
            libpbc.PBC_ISDF_del(ctypes.byref(self._this))
        except AttributeError:
            pass

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

    # cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 2]) 

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR)