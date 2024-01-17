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

libpbc = lib.load_library('libpbc')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, name))

# python version colpilot_qr() function

def colpivot_qr(A, max_rank=None):
    m, n = A.shape
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    AA = A.copy()
    pivot = np.arange(n)

    if max_rank is None:
        max_rank = min(m, n)

    for j in range(min(m, n, max_rank)):
        # Find the column with the largest norm
        
        norms = np.linalg.norm(AA[:, j:], axis=0)
        p = np.argmax(norms) + j

        # Swap columns j and p

        AA[:, [j, p]] = AA[:, [p, j]]
        R[:, [j, p]] = R[:, [p, j]]
        pivot[[j, p]] = pivot[[p, j]]

        # perform Shimdt orthogonalization

        R[j, j] = np.linalg.norm(AA[:, j])
        Q[:, j] = AA[:, j] / R[j, j]

        R[j, j + 1:] = np.dot(Q[:, j].T, AA[:, j + 1:])
        AA[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])

    return Q, R, pivot

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
                ('cutoff_aoValue', ctypes.c_double),
                ('cutoff_QR', ctypes.c_double),
                ('naux', ctypes.c_int),
                ('ao2atomID', ctypes.c_void_p),
                ('aoG', ctypes.c_void_p),
                ('voronoi_partition', ctypes.c_void_p),
                ('ao_sparse_rep_row', ctypes.c_void_p),
                ('ao_sparse_rep_col', ctypes.c_void_p),
                ('ao_sparse_rep_val', ctypes.c_void_p),
                ('IP_index', ctypes.c_void_p),
                ('auxiliary_basis', ctypes.c_void_p)
                ]

from pyscf.pbc import df

class PBC_ISDF_Info(df.fft.FFTDF):
    def __init__(self, mol:Cell, aoR: np.ndarray,
                 cutoff_aoValue: float = 1e-12,
                 cutoff_QR: float = 1e-8):

        super().__init__(cell=cell)

        self._this = ctypes.POINTER(_PBC_ISDF)()

        ## the following variables are used in build_sandeep

        self.IP_ID     = None
        self.aux_basis = None
        self.c         = None
        self.naux      = None
        self.W         = None 
        self.aoRg      = None 
        self.aoR       = None
        self.V_R       = None
        self.cell      = mol

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

    def build(self):
        libpbc.PBC_ISDF_build(self._this)

    def build_only_partition(self):
        libpbc.PBC_ISDF_build_onlyVoronoiPartition(self._this)

    def build_IP_Sandeep(self, c=5, m=5, ao_value_cutoff=1e-8, debug=True):

        # build partition

        libpbc.PBC_ISDF_build_onlyVoronoiPartition(self._this)
        ao2atomID = self.get_ao2atomID()
        partition = self.get_partition()
        aoR  = self.get_aoG()
        natm = self._this.contents.natm
        nao  = self._this.contents.nao

        nao_per_atm = np.zeros(natm, dtype=np.int32)
        for i in range(self._this.contents.nao):
            atm_id = ao2atomID[i]
            nao_per_atm[atm_id] += 1

        # for each atm

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        possible_IP = []

        for atm_id in range(natm):
            # find partition for this atm
            grid_ID = np.where(partition == atm_id)[0]
            # get aoR for this atm
            aoR_atm = aoR[:, grid_ID]
            nao_atm = nao_per_atm[atm_id]
            npt_find = c * nao_atm + 10
            naux_tmp = int(np.sqrt(c*nao_atm)) + m
            # generate to random orthogonal matrix of size (naux_tmp, nao), do not assume sparsity here
            if npt_find > nao:
                aoR_atm1 = aoR_atm
                aoR_atm2 = aoR_atm
            else:
                G1 = np.random.rand(nao, naux_tmp)
                G1, _ = numpy.linalg.qr(G1)
                G2 = np.random.rand(nao, naux_tmp)
                G2, _ = numpy.linalg.qr(G2)
                aoR_atm1 = G1.T @ aoR_atm
                aoR_atm2 = G2.T @ aoR_atm
            aoPair = np.einsum('ik,jk->ijk', aoR_atm1, aoR_atm2).reshape(-1, grid_ID.shape[0])
            _, R, pivot = colpivot_qr(aoPair, max_rank=npt_find)
            pivot_ID = grid_ID[pivot[:npt_find]]
            possible_IP.extend(pivot_ID.tolist())

            print("atm_id = ", atm_id)
            for i in range(npt_find):
                print("R[%3d] = %15.8e" % (i, R[i, i]))

        # sort the possible_IP

        possible_IP.sort()
        possible_IP = np.array(possible_IP)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_IP_Atm")
        t1 = t2

        # a final global QRCP

        aoR_IP = aoR[:, possible_IP]
        naux_tmp = int(np.sqrt(c*nao)) + m
        if naux_tmp > nao:
            aoR1 = aoR_IP
            aoR2 = aoR_IP
        else:
            G1 = np.random.rand(nao, naux_tmp)
            G1, _ = numpy.linalg.qr(G1)
            G2 = np.random.rand(nao, naux_tmp)
            G2, _ = numpy.linalg.qr(G2)
            aoR1 = G1.T @ aoR_IP
            aoR2 = G2.T @ aoR_IP
        aoPair = np.einsum('ik,jk->ijk', aoR1, aoR2).reshape(-1, possible_IP.shape[0])
        npt_find = c * nao
        _, R, pivot = colpivot_qr(aoPair, max_rank=npt_find)

        print("global QRCP")
        for i in range(npt_find):
            print("R[%3d] = %15.8e" % (i, R[i, i]))

        IP_ID = possible_IP[pivot[:npt_find]]
        IP_ID.sort()
        print("IP_ID = ", IP_ID)

        self.IP_ID = IP_ID

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_IP_Global")
        t1 = t2

        # build the auxiliary basis

        aoRg = aoR[:, IP_ID]
        naux = IP_ID.shape[0]
        A = np.asarray(lib.dot(aoRg.T, aoRg), order='C')
        A = A ** 2
        print("A.shape = ", A.shape)

        B = np.asarray(lib.dot(aoRg.T, aoR), order='C')
        B = B ** 2

        try:
            self.aux_basis = scipy.linalg.solve(A, B, assume_a='sym')
        except np.linalg.LinAlgError:
            # catch singular matrix error
            e, h = np.linalg.eigh(A)
            # remove those eigenvalues that are too small
            where = np.where(abs(e) > 1e-12)[0]
            e = e[where]
            h = h[:, where]
            self.aux_basis = h @ np.diag(1/e) @ h.T @ B

        # construct the auxiliary basis

        # self.aux_basis = np.empty((naux,ngrids))
        # blksize = int(10*1e9/8/naux)
        # for p0, p1 in lib.prange(0, ngrids, blksize):
        #     # B = numpy.dot(aoRg, aoR[p0:p1].T) ** 2
        #     B = np.asarray(lib.dot(aoRg.T, aoR[:, p0:p1]), order='C')
        #     B = B ** 2
        #     self.aux_basis[:,p0:p1] = scipy.linalg.lstsq(A, B)[0]
        #     B = None
        # A = None

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_basis")
        
        self.c    = c
        self.naux = naux
        self.aoRg = aoRg
        self.aoR  = aoR

    def build_auxiliary_Coulomb(self, cell:Cell, mesh, debug=True):
        
        print("mesh = ", mesh)

        ngrids = self._this.contents.ngrids
        naux   = self.naux

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        V_R   = np.zeros((naux, ngrids))
        coulG = tools.get_coulG(cell, mesh=mesh)
        
        blksize1 = int(5*1e9/8/ngrids)
        for p0, p1 in lib.prange(0, naux, blksize1):
            X_freq     = numpy.fft.fftn(self.aux_basis[p0:p1].reshape(-1,*mesh), axes=(1,2,3)).reshape(-1,ngrids)
            V_G        = X_freq * coulG[None,:]
            X_freq     = None
            V_R[p0:p1] = numpy.fft.ifftn(V_G.reshape(-1,*mesh), axes=(1,2,3)).real.reshape(-1,ngrids)
            V_G        = None
        coulG = None

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_V_R")
        t1 = t2

        W = np.zeros((naux,naux))
        for p0, p1 in lib.prange(0, ngrids, blksize1*2):
            W += numpy.dot(self.aux_basis[:,p0:p1], V_R[:,p0:p1].T)

        # for i in range(naux):
        #     for j in range(i):
        #         print("W[%5d, %5d] = %15.8e" % (i, j, W[i,j]))

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_W")
        
        self.V_R  = V_R
        self.W    = W
        self.mesh = mesh

    def check_AOPairError(self):
        assert(self.IP_ID is not None)
        assert(self.aux_basis is not None)

        aoR = self.aoR
        aoRg = aoR[:, self.IP_ID]


        nao = self._this.contents.nao

        print("In check_AOPairError")

        for i in range(nao):
            coeff = numpy.einsum('k,jk->jk', aoRg[i, :], aoRg).reshape(-1, self.IP_ID.shape[0])
            aoPair = numpy.einsum('k,jk->jk', aoR[i, :], aoR).reshape(-1, aoR.shape[1])
            aoPair_approx = coeff @ self.aux_basis

            diff = aoPair - aoPair_approx
            diff_pair_abs_max = np.max(np.abs(diff), axis=1)
            # print("diff_pair_abs_max = ", diff_pair_abs_max)
            # print("diff_pair_abs_max.shape = ", diff_pair_abs_max.shape)

            for j in range(diff_pair_abs_max.shape[0]):
                # print("i = %5d, j = %5d diff_pair_abs_max = %15.8e" % (i, j, diff_pair_abs_max[j]))
                print("(%5d, %5d, %15.8e)" % (i, j, diff_pair_abs_max[j]))


    def get_partition(self):
        shape = (self._this.contents.ngrids,)
        print("shape = ", shape)
        data = ctypes.cast(self._this.contents.voronoi_partition,
                           ctypes.POINTER(ctypes.c_int))
        # print("data = ", data)
        return numpy.ctypeslib.as_array(data, shape=shape)
        # pass

    def get_ao2atomID(self):
        shape = (self._this.contents.nao,)
        data = ctypes.cast(self._this.contents.ao2atomID,
                           ctypes.POINTER(ctypes.c_int))
        return numpy.ctypeslib.as_array(data, shape=shape)

    def get_aoG(self):
        shape = (self._this.contents.nao, self._this.contents.ngrids)
        data = ctypes.cast(self._this.contents.aoG,
                           ctypes.POINTER(ctypes.c_double))
        return numpy.ctypeslib.as_array(data, shape=shape)

    def get_auxiliary_basis(self):
        shape = (self._this.contents.naux, self._this.contents.ngrids)
        print("shape = ", shape)
        data = ctypes.cast(self._this.contents.auxiliary_basis,
                           ctypes.POINTER(ctypes.c_double))
        return numpy.ctypeslib.as_array(data, shape=shape)

    def __del__(self):
        try:
            libpbc.PBC_ISDF_del(ctypes.byref(self._this))
        except AttributeError:
            pass

    ##### functions defined in isdf_ao2mo.py #####

    get_eri = get_ao_eri = isdf_ao2mo.get_eri
    ao2mo = get_mo_eri = isdf_ao2mo.general
    ao2mo_7d = isdf_ao2mo.ao2mo_7d  # seems to be only called in kadc and kccsd, NOT implemented!

    ##### functions defined in isdf_jk.py #####

    get_jk = isdf_jk.get_jk_dm

if __name__ == '__main__':

    # Test the function

    A = np.random.rand(5, 5)
    Q, R, pivot = colpivot_qr(A)

    print("A = ", A)
    print("Q = ", Q)
    print("R = ", R)
    print("Q@R = ", Q@R)
    print("A * pivot = ", A[:, pivot])
    print("pivot = ", pivot)
    print("inverse P = ", np.argsort(pivot))
    print("Q * R * inverse P = ", Q@R[:, np.argsort(pivot)])
    print("diff = ", Q@R[:, np.argsort(pivot)] - A)
    print("Q^T * Q = ", Q.T @ Q)

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

    # cell.atom = '''
    #                C     0.8917  0.8917  0.8917
    #                C     2.6751  2.6751  0.8917
    #                C     2.6751  0.8917  2.6751
    #                C     0.8917  2.6751  2.6751
    #             '''

    cell.basis   = 'gth-szv'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 1])

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR, cutoff_aoValue=1e-6, cutoff_QR=1e-3)
    pbc_isdf_info.build_IP_Sandeep(c=20)
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)
    pbc_isdf_info.check_AOPairError()

    ### check eri ###

    # mydf_eri = df.FFTDF(cell)
    # eri = mydf_eri.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
    # print("eri.shape  = ", eri.shape)
    # eri_isdf = pbc_isdf_info.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
    # print("eri_isdf.shape  = ", eri_isdf.shape)
    # for i in range(cell.nao):
    #     for j in range(cell.nao):
    #         for k in range(cell.nao):
    #             for l in range(cell.nao):
    #                 if abs(eri[i,j,k,l] - eri_isdf[i,j,k,l]) > 1e-6:
    #                     print("eri[{}, {}, {}, {}] = {} != {}".format(i,j,k,l,eri[i,j,k,l], eri_isdf[i,j,k,l]),
    #                           "ration = ", eri[i,j,k,l]/eri_isdf[i,j,k,l])

    ### perform scf ###

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.kernel()

    mf = scf.RHF(cell)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.kernel()

