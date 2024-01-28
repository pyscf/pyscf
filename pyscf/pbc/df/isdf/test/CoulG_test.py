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

from pyscf.pbc.df.isdf.test.test_isdf_fast import PBC_ISDF_Info

from multiprocessing import Pool

import dask.array as da
from dask import delayed

from memory_profiler import profile

libpbc = lib.load_library('libpbc')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, name))


class GMRES:
    '''
    GMRES solver to solve Ax = b, where x and b is a matrix ! 
    '''

    '''
    Args:
    
    '''

    def __init__(self,
                 func_Ax,
                 func_vec_scal,
                 func_vec_plus,
                 func_vec_empty,
                 func_InnerProd,
                 max_micro_iter=16,
                 max_macro_iter=128,
                 cnvg_criterion=1e-6,  # the differece between two consecutive residuals
                 proj_L_to_R=None
                 ):

        # function pointer

        self.func_Ax        = func_Ax
        self.func_InnerProd = func_InnerProd  # the norm is defined accordingly!

        # linear algebra op

        self.func_vec_scal  = func_vec_scal
        self.func_vec_plus  = func_vec_plus
        self.func_vec_empty = func_vec_empty

        self.proj_L_to_R = proj_L_to_R

        if proj_L_to_R is None:
            def identity(x, buf=None):
                if buf is None:
                    buf = np.zeros_like(x)
                buf = x
                return buf
            self.proj_L_to_R = identity

        # data member

        self.B = None

        # data member

        self.Q = None
        self.b = None

        # buffer

        self.buf_L = None
        self.buf_L2 = None
        self.buf_R = None
        self.buf_R2 = None
        self.basis_R = []
        self.basis_L = []

        self.X0 = None
        self.R0 = None
        self.Norm_R0 = None

        # control parameters

        self.max_micro_iter = max_micro_iter
        self.max_macro_iter = max_macro_iter
        self.cnvg_criterion = cnvg_criterion
        self.converged      = False

    def _first_iteration(self, B, X0=None):
        if X0 is None:
            X0 = self.func_vec_empty(B)  # to check

        self.B  = B
        self.X0 = X0

        # print("self.B.shape = ", self.B.shape)
        # print("self.X0.shape = ", self.X0.shape)

        # implement self.R0 = B - self.func_Ax(X0)

        # print("self.B = ", self.B)

        self.buf_L   = self.func_Ax(X0, buf=self.buf_L)
        self.R0      = self.func_vec_plus(1.0, B, -1.0, self.buf_L, buf=self.R0)
        self.Norm_R0 = np.sqrt(self.func_InnerProd(self.R0, self.R0))

        print("self.Norm_R0 = ", self.Norm_R0)

        if len(self.basis_L) == 0:
            self.basis_L.append(self.func_vec_scal(1.0/self.Norm_R0, self.R0))
        else:
            self.basis_L[0] = self.func_vec_scal(1.0/self.Norm_R0, self.R0, buf=self.basis_L[0])

        self.buf_R = self.proj_L_to_R(self.basis_L[0], buf=self.buf_R)
        norm       = self.func_InnerProd(self.buf_R, self.buf_R)
        if len(self.basis_R) == 0:
            self.basis_R.append(self.func_vec_scal(1.0/np.sqrt(norm), self.buf_R))
        else:
            self.basis_R[0] = self.func_vec_scal(1.0/np.sqrt(norm), self.buf_R, buf=self.basis_R[0])

        self.b = np.zeros((self.max_micro_iter+1), dtype=np.double)
        self.Q = np.zeros((self.max_micro_iter+1, self.max_micro_iter), dtype=np.double)

        self.b[0] = self.Norm_R0

        # print("self.b[0] = ", self.b[0])
        # print(np.dot(self.basis_L[0], self.R0))

        self.buf_L  = self.func_Ax(self.basis_R[0], buf=self.buf_L)
        self.Q[0,0] = self.func_InnerProd(self.basis_L[0], self.buf_L)
        # print("self.Q[0,0] = ", self.Q[0,0])

    def _micro_iteration(self):
        for i in range(self.max_micro_iter):

            # self.buf_L store the result of self.func_Ax(self.basis_L[i]) or self.func_Ax(self.basis_R[i])
            # this is a new basis vector in the Krylov subspace

            self.buf_R = self.proj_L_to_R(self.basis_L[i], buf=self.buf_R)

            # project out and normalization of the new basis vector

            for j in range(i+1):
                self.Q[j,i] = self.func_InnerProd(self.basis_L[j], self.buf_L)
                # print("i = ", i, "j = ", j, "self.Q[j,i] = ", self.Q[j,i])

            for j in range(i+1):
                self.buf_L = self.func_vec_plus(1.0, self.buf_L, -self.Q[j,i], self.basis_L[j], buf=self.buf_L)

            norm = np.sqrt(self.func_InnerProd(self.buf_L, self.buf_L))
            if len(self.basis_L) < i+2:
                self.basis_L.append(self.func_vec_scal(1.0/norm, self.buf_L))
            else:
                self.basis_L[i+1] = self.func_vec_scal(1.0/norm, self.buf_L, buf=self.basis_L[i+1])

            # print(self.basis_L[0])
            # print(self.basis_L[1])
            # print(np.linalg.norm(self.basis_L[0]))
            # print(np.linalg.norm(self.basis_L[1]))
            # print(np.dot(self.basis_L[0], self.basis_L[1]))

            # orhtogonalization of the new basis vector for the R

            self.buf_R = self.proj_L_to_R(self.basis_L[i+1], buf=self.buf_R)
            for j in range(i+1):
                tmp = self.func_InnerProd(self.basis_R[j], self.buf_R)
                self.buf_R = self.func_vec_plus(1.0, self.buf_R, -tmp, self.basis_R[j], buf=self.buf_R)
            norm = np.sqrt(self.func_InnerProd(self.buf_R, self.buf_R))
            if len(self.basis_R) < i+2:
                self.basis_R.append(self.func_vec_scal(1.0/norm, self.buf_R))
            else:
                self.basis_R[i+1] = self.func_vec_scal(1.0/norm, self.buf_R, buf=self.basis_R[i+1])

            # print(self.basis_R[0])
            # print(self.basis_R[1])
            # print(np.linalg.norm(self.basis_R[0]))
            # print(np.linalg.norm(self.basis_R[1]))
            # print(np.dot(self.basis_R[0], self.basis_R[1]))

            # construct b and judge convergence

            self.b[i+1] = self.func_InnerProd(self.basis_L[i+1], self.R0)

            norm_b = np.linalg.norm(self.b[:i+2])

            # next Ax

            self.buf_L = self.func_Ax(self.basis_R[i+1], buf=self.buf_L)

        self._restart(self.max_micro_iter)

        return

    def _restart(self, nMicroIter):

        # print("--------------------- RESTART --------------------- ")

        # finish the last micro iteration

        # print("nMicroIter = ", nMicroIter)

        # for j in range(nMicroIter+1):
        #     self.Q[j,nMicroIter] = self.func_InnerProd(self.basis_L[j], self.buf_L)
        #     print("j = ", j, "self.Q[j,nMicroIter] = ", self.Q[j,nMicroIter])

        # reconstruct Q , the number of basis is nMicroIter + 1

        for i in range(nMicroIter):
            basis_now = self.basis_R[i]
            self.buf_L = self.func_Ax(basis_now, buf=self.buf_L)
            for j in range(nMicroIter+1):
                self.Q[j, i] = self.func_InnerProd(self.basis_L[j], self.buf_L)
                # print("i = ", i, "j = ", j, "self.Q[j,i] = ", self.Q[j,i])

        Q = self.Q[:nMicroIter+1,:nMicroIter]
        b = self.b[:nMicroIter+1]

        # print("Q = ", Q)
        # print("b = ", b)

        # solve the least square problem

        # print("Q.shape = ", Q.shape)
        # print("Q = ", Q)

        y = np.linalg.lstsq(Q, b, rcond=None)[0]

        # print("y = ", y)

        # construct the solution

        for i in range(0, nMicroIter):
            self.X0 = self.func_vec_plus(1.0, self.X0, y[i], self.basis_R[i], buf=self.X0)

        self._first_iteration(self.B, X0=self.X0)

        # print("--------------------- END RESTART --------------------- ")

    def solve(self, b, X0=None):

        self._first_iteration(b, X0=X0)

        norm_R0_old = self.Norm_R0

        while(self.converged is False):

            self._micro_iteration()

            norm_R0_new = self.Norm_R0

            if abs(norm_R0_new - norm_R0_old) < self.cnvg_criterion or self.converged:
                self.converged = True
                break

            norm_R0_old = norm_R0_new

        return self.X0

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

    ############## test the construction of V ##############

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


    ############## test the construction of aux_basis ##############

    #### seems to be useless !

    N = 256

    # bench mark

    A = numpy.random.rand(N, N)
    A += A.T

    e, h = numpy.linalg.eigh(A)
    print("e = ", e)

    # make A positive definite

    e[e<0] = -e[e<0]

    A = h.dot(numpy.diag(e)).dot(h.T)

    B = numpy.random.rand(N, 2)

    X = numpy.linalg.solve(A, B)

    print("np.allclose(A.dot(X), B) = ", np.allclose(A.dot(X), B))

    e, h = numpy.linalg.eigh(A)

    print(e[-1]/e[0])

    ###### test GMRES ######

    def func_Ax(x, buf=None):
        if buf is None:
            buf = np.zeros_like(x)
        buf = A.dot(x, out=buf)
        return buf

    def func_InnerProd(x, y):
        assert(x.shape == y.shape)
        # return np.dot(x, y)

        if x.ndim == 1:
            return np.dot(x, y)
        else:
            return np.einsum('ij,ij->', x, y)

    def func_vec_scal(a, x, buf=None):
        if buf is None:
            buf = np.zeros_like(x)
        buf = a * x
        return buf

    def func_vec_plus(a, x, b, y, buf=None):
        if buf is None:
            buf = np.zeros_like(x)
        buf = a * x + b * y
        return buf

    def func_vec_empty(x):
        print("x.shape = ", x.shape)
        return np.zeros_like(x)

    def proj_L_to_R(x, buf=None):
        if buf is None:
            buf = np.zeros_like(x)
        buf = x
        return buf

    gmres = GMRES(func_Ax,
                  func_vec_scal,
                  func_vec_plus,
                  func_vec_empty,
                  func_InnerProd,
                  max_micro_iter=8,
                  proj_L_to_R=proj_L_to_R)

    # X0 = np.zeros((N), dtype=np.double)
    X0 = gmres.solve(B, X0=None)

    print("np.allclose(X, X0) = ", np.allclose(X, X0))

    print("X  = ", X)
    print("X0 = ", X0)

    ##################### check the preconditioner #####################

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx     = grids.mesh[0]
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    aoR    = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR   *= np.sqrt(cell.vol / ngrids)
    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR, cutoff_aoValue=1e-6, cutoff_QR=1e-3)
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=15, global_IP_selection=True)

    partition = pbc_isdf_info.partition
    IP_ID = pbc_isdf_info.IP_ID
    IP_2_atom = [partition[x] for x in IP_ID]

    # group IP_ID based on the atom index

    IP_ID_group = []

    for i in range(cell.natm):
        IP_ID_group.append([id for id, x in enumerate(IP_ID) if partition[x] == i])

    print("IP_ID_group = ", IP_ID_group)

    permutation = []

    for i in range(cell.natm):
        permutation.extend(IP_ID_group[i])

    A, B = pbc_isdf_info.get_A_B()

    M = np.zeros_like(A)

    loc = 0
    for i in range(cell.natm):
        TMP = A[:, IP_ID_group[i]]
        TMP = TMP[IP_ID_group[i], :]

        e, h = np.linalg.eigh(TMP)

        # print("e = ", e)

        print("e[-1]/e[0] = ", e[-1]/e[0])

        for id, _e in enumerate(e):
            print("id = ", id, "_e = ", _e)

        nrow = len(IP_ID_group[i])

        M[loc:loc+nrow, loc:loc+nrow] = h.dot(np.diag(1.0/e)).dot(h.T)
        loc += nrow

    # perform svd on B

    C = B @ B.T

    e, h = np.linalg.eigh(C)

    print("cond B = ", np.sqrt(e[-1]/e[0]))

    for id, _e in enumerate(e):
        print("id = ", id, "_e = ", np.sqrt(_e))

    # permutation first

    A2 = A[:, permutation]
    A2 = A2[permutation, :]
    B2 = B[permutation, :]

    A_New = M @ A2
    B_New = M @ B2

    # perform svd on A_New

    u, s, vh = np.linalg.svd(A_New)

    print("cond A_New = ", s[0]/s[-1])

    for id, _s in enumerate(s):
        print("id = ", id, "_s = ", _s)

    loc = 0
    for i in range(cell.natm):
        nrow = len(IP_ID_group[i])
        # print(A_New[loc:loc+nrow, loc:loc+nrow])
        loc_j = 0
        for j in range(cell.natm):
            print("i = ", i, "j = ", j)
            nrow_j = len(IP_ID_group[j])
            tmp = A2[loc:loc+nrow, loc_j:loc_j+nrow_j]
            u, s, vh = np.linalg.svd(tmp)
            print("cond tmp    = ", s[0]/s[-1])
            print("norm of tmp = ", np.linalg.norm(tmp))
            loc_j += nrow_j
        loc += nrow

    # exit(1)

    ##################### try GMRES to AX=B #####################

    inv_permutation = np.argsort(permutation)

    # construct a initial guess by diagonal preconditioner

    X0 = np.zeros_like(B)

    loc = 0
    for i in range(cell.natm):
        nrow = len(IP_ID_group[i])
        A1 = A2[loc:loc+nrow, loc:loc+nrow]
        B1 = B2[loc:loc+nrow, :]
        # diag A1
        e, h = np.linalg.eigh(A1)
        X0[loc:loc+nrow, :] = h.dot(np.diag(1.0/e)).dot(h.T).dot(B1)
        loc += nrow

    def func_Ax2(x, buf=None):
        if buf is None:
            buf = np.zeros_like(x)
        buf = A.dot(x, out=buf)
        return buf

    gmres2 = GMRES(func_Ax2,
                   func_vec_scal,
                   func_vec_plus,
                   func_vec_empty,
                   func_InnerProd,
                   max_micro_iter=8,
                   max_macro_iter=8,
                   proj_L_to_R=proj_L_to_R)

    # X = gmres2.solve(B, X0=X0)

    # pbc_isdf_info.aux_basis = X
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)
    # pbc_isdf_info.check_AOPairError()

    # solve AX = B via LLT

    L = np.linalg.cholesky(A)
    print(np.allclose(L.dot(L.T), A))

    # solve Ly = B

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    y = np.linalg.solve(L, B)
    x = np.linalg.solve(L.T, y)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    _benchmark_time(t1, t2, "LLT_Solve")  # too slow


    fn_cholesky = getattr(libpbc, "Cholesky", None)
    assert(fn_cholesky is not None)

    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    fn_cholesky(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(A.shape[0]),
    )

    nThread = lib.num_threads()
    fn_build_aux(
        ctypes.c_int(A.shape[0]),
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(B.shape[1]),
        ctypes.c_int(B.shape[1]//nThread)
    )

    # print(B[:, 0])
    # print(pbc_isdf_info.aux_basis[:, 0])

    np.allclose(B, pbc_isdf_info.aux_basis)

    # for i in range(B.shape[1]):
    #     print("i = ", i, "np.allclose(B[:,i], pbc_isdf_info.aux_basis[:,i]) = ", np.allclose(B[:,i], pbc_isdf_info.aux_basis[:,i]))

    exit(1)

    print(x[:, 0])
    print(pbc_isdf_info.aux_basis[:, 0])
    # print("np.allclose(x, X) = ", np.allclose(x, pbc_isdf_info.aux_basis))
    # for i in range(x.shape[0]):
    #     print("i = ", i, "np.allclose(x[:,i], pbc_isdf_info.aux_basis[:,i]) = ", np.allclose(x[:,i], pbc_isdf_info.aux_basis[:,i]))

    pbc_isdf_info.aux_basis = x
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)
    pbc_isdf_info.check_AOPairError()

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.kernel()  # the accuracy is not enough
