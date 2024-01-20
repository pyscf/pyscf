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

BASIS_CUTOFF = 1e-18  # too small may lead to numerical instability
CRITERION_CALL_PARALLEL_QR = 256

# python version colpilot_qr() function

@delayed
def _vec_norm(vec):
    return np.linalg.norm(vec)
@delayed
def _daxpy(a, x, y):
    return y + a * x

def _colpivot_qr_parallel(A, max_rank=None, cutoff=1e-14):
    m, n = A.shape
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    AA = A.T.copy()  # cache friendly
    pivot = np.arange(n)

    if max_rank is None:
        max_rank = min(m, n)

    npt_find = 0

    for j in range(min(m, n, max_rank)):
        # Find the column with the largest norm

        # norms = np.linalg.norm(AA[j:, :], axis=1)
        task_norm = []
        for i in range(j, n):
            task_norm.append(_vec_norm(AA[i, :]))
        norms = da.compute(*task_norm)
        norms = np.asarray(norms)
        p = np.argmax(norms) + j

        # Swap columns j and p

        AA[[j, p], :] = AA[[p, j], :]
        R[:, [j, p]] = R[:, [p, j]]
        pivot[[j, p]] = pivot[[p, j]]

        # perform Shimdt orthogonalization

        R[j, j] = np.linalg.norm(AA[j, :])
        if R[j, j] < cutoff:
            break
        npt_find += 1
        Q[j, :] = AA[j, :] / R[j, j]

        R[j, j + 1:] = np.dot(AA[j + 1:, :], Q[j, :].T)
        # AA[j + 1:, :] -= np.outer(R[j, j + 1:], Q[j, :])
        task_daxpy = []
        for i in range(j + 1, n):
            task_daxpy.append(_daxpy(-R[j, i], Q[j, :], AA[i, :]))
        if len(task_daxpy) > 0:
            res = da.compute(*task_daxpy)
            AA[j + 1:, :] = np.concatenate(da.compute(res), axis=0)

    return Q.T, R, pivot, npt_find

def colpivot_qr(A, max_rank=None, cutoff=1e-14):
    '''
    we do not need Q
    '''
    m, n = A.shape
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    AA = A.T.copy()  # cache friendly
    pivot = np.arange(n)

    if max_rank is None:
        max_rank = min(m, n)

    npt_find = 0

    for j in range(min(m, n, max_rank)):
        # Find the column with the largest norm

        # norms = np.linalg.norm(AA[:, j:], axis=0)
        norms = np.linalg.norm(AA[j:, :], axis=1)
        p = np.argmax(norms) + j

        # Swap columns j and p

        # AA[:, [j, p]] = AA[:, [p, j]]
        AA[[j, p], :] = AA[[p, j], :]
        R[:, [j, p]] = R[:, [p, j]]
        pivot[[j, p]] = pivot[[p, j]]

        # perform Shimdt orthogonalization

        # R[j, j] = np.linalg.norm(AA[:, j])
        R[j, j] = np.linalg.norm(AA[j, :])
        if R[j, j] < cutoff:
            break
        npt_find += 1
        # Q[:, j] = AA[:, j] / R[j, j]
        Q[j, :] = AA[j, :] / R[j, j]

        # R[j, j + 1:] = np.dot(Q[:, j].T, AA[:, j + 1:])
        R[j, j + 1:] = np.dot(AA[j + 1:, :], Q[j, :].T)
        # AA[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
        AA[j + 1:, :] -= np.outer(R[j, j + 1:], Q[j, :])

    return Q.T, R, pivot, npt_find

@delayed
def atm_IP_task(taskinfo:tuple):
    grid_ID, aoR_atm, nao, nao_atm, c, m = taskinfo

    npt_find = c * nao_atm + 10
    naux_tmp = int(np.sqrt(c*nao_atm)) + m
    # generate to random orthogonal matrix of size (naux_tmp, nao), do not assume sparsity here
    if naux_tmp > nao:
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
    _, R, pivot, npt_find = colpivot_qr(aoPair, max_rank=npt_find)
    # npt_find = min(R.shape[0], R.shape[1])
    pivot_ID = grid_ID[pivot[:npt_find]]  # the global ID
    # pack res
    return pivot_ID, pivot[:npt_find], R[:npt_find, :npt_find], npt_find

@delayed
def partition_IP_task(taskinfo:tuple):
    grid_ID, aoR_atm, nao, naux, m = taskinfo

    npt_find = naux
    naux_tmp = int(np.sqrt(naux)) + m
    # generate to random orthogonal matrix of size (naux_tmp, nao), do not assume sparsity here
    if naux_tmp > nao:
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
    _, R, pivot, npt_find = colpivot_qr(aoPair, max_rank=npt_find)
    # npt_find = min(R.shape[0], R.shape[1])
    pivot_ID = grid_ID[pivot[:npt_find]]  # the global ID
    # pack res
    return pivot_ID, pivot[:npt_find], R[:npt_find, :npt_find], npt_find

@delayed
def construct_local_basis(taskinfo:tuple):
    # IP_local_ID, aoR_atm, naoatm, c = taskinfo
    IP_local_ID, aoR_atm, naux = taskinfo

    # naux = naoatm * c
    assert IP_local_ID.shape[0] >= naux
    IP_local_ID = IP_local_ID[:naux]

    IP_local_ID.sort()
    aoRg = aoR_atm[:, IP_local_ID]
    A = np.asarray(lib.dot(aoRg.T, aoRg), order='C')
    A = A ** 2
    B = np.asarray(lib.dot(aoRg.T, aoR_atm), order='C')
    B = B ** 2

    e, h = np.linalg.eigh(A)
    # remove those eigenvalues that are too small
    where = np.where(abs(e) > BASIS_CUTOFF)[0]
    e = e[where]
    h = h[:, where]
    aux_basis = np.asarray(lib.dot(h.T, B), order='C')
    aux_basis = (1.0/e).reshape(-1, 1) * aux_basis
    aux_basis = np.asarray(lib.dot(h, aux_basis), order='C')

    return IP_local_ID, aux_basis

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

        super().__init__(cell=mol)
        super().__init__(cell=mol)

        self._this = ctypes.POINTER(_PBC_ISDF)()

        ## the following variables are used in build_sandeep

        self.IP_ID     = None
        self.aux_basis = None
        self.c         = None
        self.naux      = None
        self.W         = None
        self.aoRg      = None
        self.aoR       = aoR
        self.aoR       = aoR
        self.V_R       = None
        self.cell      = mol

        self.partition = None

        self.natm = mol.natm
        self.nao = mol.nao_nr()
        self.ngrids = aoR.shape[1]

        assert self.nao == aoR.shape[0]
        self.natm = mol.natm
        self.nao = mol.nao_nr()
        self.ngrids = aoR.shape[1]

        self.jk_buffer = None
        self.ddot_buf  = None

        assert self.nao == aoR.shape[0]

        ao2atomID = np.zeros(self.nao, dtype=np.int32)
        ao2atomID = np.zeros(self.nao, dtype=np.int32)

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

        self.ao2atomID = ao2atomID
        self.ao2atomID = ao2atomID

        # libpbc.PBC_ISDF_init(ctypes.byref(self._this),
        #                         nao, natm, ngrids,
        #                         _cutoff_aoValue,
        #                         ao2atomID.ctypes.data_as(ctypes.c_void_p),
        #                         aoR.ctypes.data_as(ctypes.c_void_p),
        #                         _cutoff_QR)

        # given aoG, determine at given grid point, which ao has the maximal abs value

        self.partition = np.argmax(np.abs(aoR), axis=0)
        print("partition = ", self.partition.shape)
        # map aoID to atomID
        self.partition = np.asarray([ao2atomID[x] for x in self.partition])
        # for i in range(self.partition.shape[0]):
        #     print("i = %5d, partition = %5d" % (i, self.partition[i]))

    # @profile
    def _allocate_jk_buffer(self, datatype):

        if self.jk_buffer is None:

            nao    = self.nao
            ngrids = self.ngrids
            naux   = self.naux

            buffersize_k = nao * ngrids + naux * ngrids + naux * naux + nao * nao
            buffersize_j = nao * ngrids + ngrids + nao * naux + naux + naux + nao * nao

            self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),), dtype=datatype)
            # self.jk_buffer[-1] = 0.0 # memory allocate, well, you cannot cheat python in this way

            print("address of self.jk_buffer = ", id(self.jk_buffer))

            nThreadsOMP = lib.num_threads()
            print("nThreadsOMP = ", nThreadsOMP)
            self.ddot_buf = np.zeros((nThreadsOMP,(naux*naux)+2), dtype=datatype)
            # self.ddot_buf[nThreadsOMP-1, (naux*naux)+1] = 0.0 # memory allocate, well, you cannot cheat python in this way

        else:
            assert self.jk_buffer.dtype == datatype
            assert self.ddot_buf.dtype == datatype

    def build(self):
        # libpbc.PBC_ISDF_build(self._this)
        print("warning: not implemented yet")
        # libpbc.PBC_ISDF_build(self._this)
        print("warning: not implemented yet")

    def build_only_partition(self):
        # libpbc.PBC_ISDF_build_onlyVoronoiPartition(self._this)
        pass
        # libpbc.PBC_ISDF_build_onlyVoronoiPartition(self._this)
        pass

    # @profile
    def build_IP_Sandeep(self, c=5, m=5,
                         atomic_partition=True,
                         ratio=0.8,
                         global_IP_selection=True,
                         build_global_basis=True, debug=True):

        # build partition

        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao

        nao_per_atm = np.zeros(natm, dtype=np.int32)
        for i in range(self.nao):
            atm_id = ao2atomID[i]
            nao_per_atm[atm_id] += 1

        # for each atm

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        possible_IP = []

        # pack input info for each process

        grid_partition = []
        taskinfo = []

        if atomic_partition:
            for atm_id in range(natm):
                # find partition for this atm
                grid_ID = np.where(partition == atm_id)[0]
                grid_partition.append(grid_ID)
                # get aoR for this atm
                aoR_atm = aoR[:, grid_ID]
                nao_atm = nao_per_atm[atm_id]
                taskinfo.append(atm_IP_task((grid_ID, aoR_atm, nao, nao_atm, c, m)))
        else:
            raise NotImplementedError

            from clever_partition import _clever_partition

            atomID2AO = []
            for atm_id in range(natm):
                atomID2AO.append(np.where(ao2atomID == atm_id)[0])

            partition = _clever_partition(aoR, atomID2AO, ratio=ratio)

            naux_tot = nao * c

            for key in partition.keys():
                grid_ID = partition[key]
                grid_partition.append(grid_ID)
                aoR_now = aoR[:, grid_ID]
                naux_now = naux_tot * (float)(len(grid_ID)) / self.ngrids
                # find the cloest integer
                naux_now = int(naux_now + 0.5)
                if naux_now == 0:
                    naux_now = 1
                print("naux_now = ", naux_now)
                taskinfo.append(partition_IP_task((grid_ID, aoR_now, nao, naux_now, m)))

        results = da.compute(*taskinfo)

        if build_global_basis:

            # collect results

            for atm_id, result in enumerate(results):
                pivot_ID, _, R, npt_find = result

                if global_IP_selection == False:
                    nao_atm  = nao_per_atm[atm_id]
                    naux_now = c * nao_atm
                    pivot_ID = pivot_ID[:naux_now]
                    npt_find = naux_now
                possible_IP.extend(pivot_ID.tolist())


                print("atm_id = ", atm_id)
                print("npt_find = ", npt_find)
                # npt_find = min(R.shape[0], R.shape[1])
                for i in range(npt_find):
                    try:
                        print("R[%3d] = %15.8e" % (i, R[i, i]))
                    except:
                        break

            # sort the possible_IP

            possible_IP.sort()
            possible_IP = np.array(possible_IP)

            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
            if debug:
                _benchmark_time(t1, t2, "build_IP_Atm")
            t1 = t2

            # a final global QRCP， which is not needed!

            if global_IP_selection:
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
                    # aoR1 = G1.T @ aoR_IP
                    # aoR2 = G2.T @ aoR_IP
                    aoR1 = np.asarray(lib.dot(G1.T, aoR_IP), order='C')
                    aoR2 = np.asarray(lib.dot(G2.T, aoR_IP), order='C')
                aoPair = np.einsum('ik,jk->ijk', aoR1, aoR2).reshape(-1, possible_IP.shape[0])
                npt_find = c * nao

                _, R, pivot, npt_find = colpivot_qr(aoPair, max_rank=npt_find)

                print("global QRCP")
                print("npt_find = ", npt_find)
                # npt_find = min(R.shape[0], R.shape[1]) # may be smaller than c*nao
                for i in range(npt_find):
                    print("R[%3d] = %15.8e" % (i, R[i, i]))

                IP_ID = possible_IP[pivot[:npt_find]]
            else:
                IP_ID = possible_IP

            IP_ID.sort()
            print("IP_ID = ", IP_ID)
            self.IP_ID = IP_ID

            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
            if debug:
                _benchmark_time(t1, t2, "build_IP_Global")
            t1 = t2

            # build the auxiliary basis

            # allocate memory for the auxiliary basis

            naux = IP_ID.shape[0]
            self.naux = naux
            self._allocate_jk_buffer(datatype=np.double)
            buffer1 = np.ndarray((self.naux , self.naux), dtype=np.double, buffer=self.jk_buffer, offset=0)
            buffer2 = np.ndarray((self.naux , self.ngrids), dtype=np.double, buffer=self.jk_buffer,
                                 offset=self.naux * self.naux * self.jk_buffer.dtype.itemsize)

            ## TODO: optimize this code so that the memory allocation is minimal!

            aoRg = numpy.empty((nao, IP_ID.shape[0]))
            lib.dslice(aoR, IP_ID, out=aoRg)
            # aoRg = aoR[:, IP_ID]
            A = np.asarray(lib.ddot(aoRg.T, aoRg, c=buffer1), order='C')  # buffer 1 size = naux * naux
            # A = A ** 2
            lib.square_inPlace(A)
            print("A.shape = ", A.shape)

            self.aux_basis = np.asarray(lib.ddot(aoRg.T, aoR), order='C')   # buffer 2 size = naux * ngrids
            # B = B ** 2
            lib.square_inPlace(self.aux_basis)

            # try:
            # self.aux_basis = scipy.linalg.solve(A, B, assume_a='sym') # single thread too slow
            # except np.linalg.LinAlgError:
            # catch singular matrix error

            # use diagonalization instead

            e, h = np.linalg.eigh(A)  # single thread, but should not be slow, it should not be the bottleneck
            # remove those eigenvalues that are too small
            where = np.where(abs(e) > BASIS_CUTOFF)[0]
            e = e[where]
            h = h[:, where]
            print("e.shape = ", e.shape)
            # self.aux_basis = h @ np.diag(1/e) @ h.T @ B
            # self.aux_basis = np.asarray(lib.dot(h.T, B), order='C')  # maximal size = naux * ngrids
            B = np.asarray(lib.ddot(h.T, self.aux_basis, c=buffer2), order='C')
            # self.aux_basis = (1.0/e).reshape(-1, 1) * self.aux_basis
            # B = (1.0/e).reshape(-1, 1) * B
            lib.d_i_ij_ij(1.0/e, B, out=B)
            np.asarray(lib.ddot(h, B, c=self.aux_basis), order='C')

            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
            if debug:
                _benchmark_time(t1, t2, "build_auxiliary_basis")
        else:

            raise NotImplementedError

            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

            # build info for the next task

            # atmgridID_2_grid_ID = []

            taskinfo = []
            for atm_id, result in enumerate(results):

                # grid_ID = np.where(partition == atm_id)[0]
                # atmgridID_2_grid_ID.append(grid_ID)
                grid_ID = grid_partition[atm_id]

                _, pivot_local_ID, R, npt_find = result

                print("atm_id = ", atm_id)
                print("npt_find = ", npt_find)

                for i in range(npt_find):
                    try:
                        print("R[%3d] = %15.8e" % (i, R[i, i]))
                    except:
                        break

                # print("pivot_local_ID = ", pivot_local_ID)

                if atomic_partition:
                    taskinfo.append(construct_local_basis((pivot_local_ID, aoR[:, grid_ID], nao_per_atm[atm_id] * c)))
                else:
                    taskinfo.append(construct_local_basis((pivot_local_ID, aoR[:, grid_ID], npt_find)))

            results = da.compute(*taskinfo)

            # naux = c * nao
            naux = 0
            IP_ID = []

            aux_basis_packed = []

            for atm_id, result in enumerate(results):
                IP_local_ID, aux_basis = result
                print("atm_id = ", atm_id)
                # print("IP_local_ID = ", IP_local_ID)

                # sort IP_local_ID

                IP_ID.extend(grid_partition[atm_id][IP_local_ID])

                naux_atm = IP_local_ID.shape[0]
                aux_full = np.zeros((naux_atm, self.ngrids))
                aux_full[:, grid_partition[atm_id]] = aux_basis
                aux_basis_packed.append(aux_full)

                naux += naux_atm

            print("naux = ", naux)

            self.aux_basis = np.concatenate(aux_basis_packed, axis=0)
            self.IP_ID = np.array(IP_ID)

            # aoRg = aoR[:, IP_ID]
            aoRg = numpy.zeros((nao, IP_ID.shape[0]))
            lib.dslice(aoR, IP_ID, out=aoRg)

            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

            if debug:
                _benchmark_time(t1, t2, "build_auxiliary_basis")

        self.c    = c
        self.naux = naux
        self.aoRg = aoRg
        self.aoR  = aoR

    # @profile
    def build_auxiliary_Coulomb(self, cell:Cell, mesh, debug=True):

        # build the ddot buffer

        ngrids = self.ngrids
        ngrids = self.ngrids
        naux   = self.naux

        @delayed
        def construct_V(input:np.ndarray, ngrids, mesh, coul_G, axes=None):
            return (np.fft.ifftn((np.fft.fftn(input, axes=axes).reshape(-1, ngrids) * coul_G[None,:]).reshape(*mesh), axes=axes).real).reshape(ngrids)
            # res = (np.fft.ifftn((np.fft.fftn(input, axes=axes).reshape(-1, ngrids) * coul_G[None,:]).reshape(*mesh), axes=axes).real).reshape(ngrids)

        # print("mesh = ", mesh)

        # ngrids = self.ngrids
        # ngrids = self.ngrids
        # naux   = self.naux

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        # V_R   = np.zeros((naux, ngrids))
        coulG = tools.get_coulG(cell, mesh=mesh)

        # blksize1 = int(5*1e9/8/ngrids)
        # for p0, p1 in lib.prange(0, naux, blksize1):
        #     tmp1 = self.aux_basis[p0:p1].reshape(-1,*mesh)
        #     task = []
        #     for i in range(tmp1.shape[0]):
        #         task.append(construct_V(tmp1[i], ngrids, mesh, coulG))
        #     res = da.compute(*task)
        #     V_R[p0:p1] = np.asarray(res).reshape(-1, ngrids)
        #     # X_freq     = numpy.fft.fftn(self.aux_basis[p0:p1].reshape(-1,*mesh), axes=(1,2,3)).reshape(-1,ngrids)
        #     # V_G        = X_freq * coulG[None,:]
        #     # X_freq     = None
        #     # V_R[p0:p1] = numpy.fft.ifftn(V_G.reshape(-1,*mesh), axes=(1,2,3)).real.reshape(-1,ngrids)
        #     # V_G        = None

        task = []
        for i in range(naux):
            task.append(construct_V(self.aux_basis[i].reshape(-1,*mesh), ngrids, mesh, coulG))
        V_R = np.concatenate(da.compute(*task)).reshape(-1,ngrids)

        del task
        coulG = None

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_V_R")
        t1 = t2

        W = np.zeros((naux,naux))
        lib.ddot_withbuffer(a=self.aux_basis, b=V_R.T, buf=self.ddot_buf, c=W, beta=1.0)
        # lib.ddot(self.aux_basis, V_R.T, c=W, beta=1.0)

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
        nao = self.nao

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

import tracemalloc

if __name__ == '__main__':

    # Test the function

    A = np.random.rand(16, 16)
    Q, R, pivot, _ = _colpivot_qr_parallel(A)

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

    # exit(1)

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
    cell.ke_cutoff = 256
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
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=15, global_IP_selection=True)
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

    # mf = scf.RHF(cell)
    # mf.max_cycle = 100
    # mf.conv_tol = 1e-8
    # mf.kernel()

