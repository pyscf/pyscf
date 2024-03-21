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
import pyscf.pbc.df.isdf.isdf_fast as isdf
import pyscf.pbc.df.isdf.isdf_outcore as isdf_outcore

from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, matrix_all2all_Col2Row, matrix_all2all_Row2Col, reduce

import ctypes
from multiprocessing import Pool
from memory_profiler import profile

libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

####### the following subroutines are used to get the sendbuf for all2all #######

def _get_sendbuf_Row2Col(Mat:np.ndarray, nRow, nCol):

    RowPad = (nRow % comm_size) != 0
    if RowPad:
        RowBunch = nRow // comm_size + 1
    else:
        RowBunch = nRow // comm_size
    ColPad = (nCol % comm_size) != 0
    if ColPad:
        ColBunch = nCol // comm_size + 1
    else:
        ColBunch = nCol // comm_size

    # if RowPad:
    #     sendbuf = numpy.zeros((comm_size * RowBunch, ColBunch), dtype=Mat.dtype)
    #     # print("sendbuf.shape = ", sendbuf.shape)
    #     # print("Mat.shape = ", Mat.shape)
    #     sendbuf[:Mat.shape[0], :Mat.shape[1]] = Mat
    # else:
    #     if ColPad:
    #         if rank == comm_size - 1:
    #             sendbuf = numpy.zeros((comm_size * RowBunch, ColBunch), dtype=Mat.dtype)
    #             sendbuf[:Mat.shape[0], :Mat.shape[1]] = Mat
    #         else:
    #             assert Mat.shape[1] == ColBunch
    #             sendbuf = Mat
    #     else:
    #         sendbuf = Mat
    
    if Mat.shape != (comm_size * RowBunch, ColBunch):
        sendbuf = numpy.zeros((comm_size * RowBunch, ColBunch), dtype=Mat.dtype)
        sendbuf[:Mat.shape[0], :Mat.shape[1]] = Mat
    else:
        sendbuf = Mat
    
    return sendbuf

def _get_sendbuf_Col2Row(Mat:np.ndarray, nRow, nCol, force_bunch_even=False):
    
    RowPad = (nRow % comm_size) != 0
    if RowPad:
        RowBunch = nRow // comm_size + 1
    else:
        RowBunch = nRow // comm_size
    ColPad = (nCol % comm_size) != 0
    if ColPad:
        ColBunch = nCol // comm_size + 1
    else:
        ColBunch = nCol // comm_size

    if force_bunch_even:
        if ColBunch % 2 == 1:
            ColBunch += 1

    # if ColPad:
    #     sendbuf = numpy.zeros((RowBunch, comm_size * ColBunch), dtype=Mat.dtype)
    #     sendbuf[:Mat.shape[0], :Mat.shape[1]] = Mat
    # else:
    #     if RowPad:
    #         if rank == comm_size - 1:
    #             sendbuf = numpy.zeros((RowBunch, comm_size * ColBunch), dtype=Mat.dtype)
    #             sendbuf[:Mat.shape[0], :Mat.shape[1]] = Mat
    #         else:
    #             assert Mat.shape[0] == RowBunch
    #             sendbuf = Mat
    #     else:
    #         sendbuf = Mat
    
    if Mat.shape != (RowBunch, comm_size * ColBunch):
        sendbuf = numpy.zeros((RowBunch, comm_size * ColBunch), dtype=Mat.dtype)
        sendbuf[:Mat.shape[0], :Mat.shape[1]] = Mat
    else:
        sendbuf = Mat
    
    return sendbuf
    
def _get_packed_mat_Row2Col(recvbuf:np.ndarray, nRow, nCol):
    
    RowPad = (nRow % comm_size) != 0
    if RowPad:
        RowBunch = nRow // comm_size + 1
    else:
        RowBunch = nRow // comm_size
    ColPad = (nCol % comm_size) != 0
    if ColPad:
        ColBunch = nCol // comm_size + 1
    else:
        ColBunch = nCol // comm_size
    
    print("RowBunch = ", RowBunch)
    print("ColBunch = ", ColBunch)
    print("recvbuf.shape = ", recvbuf.shape)
    print((RowBunch, ColBunch*comm_size))
    assert recvbuf.shape == (RowBunch, ColBunch*comm_size)
    
    row_id_begin = rank * RowBunch
    row_id_end = min(row_id_begin + RowBunch, nRow)

    if ColPad:
        return recvbuf[:row_id_end-row_id_begin, :nCol].copy()
    else:
        if row_id_end - row_id_begin == recvbuf.shape[0]:
            return recvbuf
        else:
            return recvbuf[:row_id_end-row_id_begin, :nCol].copy()

def _get_packed_mat_Col2Row(recvbuf:np.ndarray, nRow, nCol, force_bunch_even=False):
    
    RowPad = (nRow % comm_size) != 0
    if RowPad:
        RowBunch = nRow // comm_size + 1
    else:
        RowBunch = nRow // comm_size
    ColPad = (nCol % comm_size) != 0
    if ColPad:
        ColBunch = nCol // comm_size + 1
    else:
        ColBunch = nCol // comm_size
    
    if force_bunch_even:
        if ColBunch % 2 == 1:
            ColBunch += 1
            
    assert recvbuf.shape == (RowBunch*comm_size, ColBunch)
    
    col_id_begin = rank * ColBunch
    col_id_end = min(col_id_begin + ColBunch, nCol)

    if RowPad:
        return recvbuf[:nRow, :col_id_end-col_id_begin].copy()
    else:
        if col_id_end - col_id_begin == recvbuf.shape[1]:
            return recvbuf
        else:
            return recvbuf[:nRow, :col_id_end-col_id_begin].copy()

def build_partition(mydf:isdf.PBC_ISDF_Info):
        
    nGrids = mydf.ngrids
    if nGrids % comm_size == 0:
        grid_bunch = nGrids // comm_size
    else:
        grid_bunch = nGrids // comm_size + 1
    
    p0 = min(nGrids, rank * grid_bunch)
    p1 = min(nGrids, (rank+1) * grid_bunch)
    
    bufsize = min(mydf.IO_buf.size, 4*1e9/8) // 2
    bunchsize = int(bufsize / (mydf.nao))
    
    weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    
    coords_now = mydf.coords[p0:p1]
    
    mydf.grid_begin = p0
    mydf.grid_end = p1
    p0 = None
    p1 = None
    
    partition = np.zeros(coords_now.shape[0], dtype=np.int32)
    
    aoR = np.zeros((mydf.nao, coords_now.shape[0]), dtype=np.float64)
    
    for p0, p1 in lib.prange(0, coords_now.shape[0], bunchsize):
        AoR_Buf = np.ndarray((mydf.nao, p1-p0), dtype=np.complex128, buffer=mydf.IO_buf, offset=0)
        AoR_Buf = ISDF_eval_gto(mydf.cell, coords=coords_now[p0:p1], out=AoR_Buf)
        res = np.argmax(np.abs(AoR_Buf), axis=0)
        partition[p0:p1] = np.asarray([mydf.ao2atomID[x] for x in res])
        aoR[:, p0:p1] = AoR_Buf * weight
        AoR_Buf = None
            
    comm.Barrier()
    
    partition_new = allgather(partition)
    
    if mydf.verbose:
        print("partition_new.shape = ", partition_new.shape)
        print("partition_new = ", partition_new)
    
    mydf.partition = partition_new
    mydf.aoR = aoR

def build_aoRg(mydf:isdf.PBC_ISDF_Info):
    
    aoR = mydf.aoR
    assert aoR is not None

    weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    nIP = len(mydf.IP_ID)
    ngrids = mydf.ngrids
    nao = mydf.nao
    p0 = min(nIP, rank * nIP // comm_size)
    p1 = min(nIP, (rank+1) * nIP // comm_size)
    
    aoRg = ISDF_eval_gto(mydf.cell, coords=mydf.coords[mydf.IP_ID[p0:p1]]).real * weight
        
    mydf.aoRg = comm.allgather(aoRg)
    
    for i in range(len(mydf.aoRg)):
        mydf.aoRg[i] = mydf.aoRg[i].reshape(nao, -1)
    
    mydf.aoRg = np.hstack(mydf.aoRg)
    
    if rank == 0:
        print("aoRg.shape = ", mydf.aoRg.shape)
        assert mydf.aoRg.ndim == 2
        # print("aoRg = ", mydf.aoRg)
    
    IP_begin = None
    IP_end = None
    
    for ip_id in mydf.IP_ID:
        if ip_id < mydf.grid_end:
            if IP_begin is None:
                IP_begin = ip_id
            IP_end = ip_id + 1
        else:
            break
    
    comm.Barrier()

def build_auxiliary_Coulomb(mydf:isdf.PBC_ISDF_Info, debug=True):
    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = mydf.mesh

    mydf._allocate_jk_buffer(np.double)
    
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2
    
    def constrcuct_V_CCode(aux_basis:np.ndarray, mesh, coul_G):
        
        coulG_real         = coul_G.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)
        nThread            = lib.num_threads()
        bunchsize          = aux_basis.shape[0] // (2*nThread)
        bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
        bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
        nAux               = aux_basis.shape[0]
        ngrids             = aux_basis.shape[1]
        mesh_int32         = np.array(mesh, dtype=np.int32)

        V                  = np.zeros((nAux, ngrids), dtype=np.double)
        basis_fft          = np.zeros((nAux, ncomplex), dtype=np.double)
        CONSTRUCT_V = 1
        
        fn = getattr(libpbc, "_construct_V2", None)
        assert(fn is not None)

        # print("V.shape = ", V.shape)
        # print("aux_basis.shape = ", aux_basis.shape)
        # print("self.jk_buffer.size    = ", self.jk_buffer.size)
        # print("self.jk_buffer.shape   = ", self.jk_buffer.shape)

        fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(nAux),
           aux_basis.ctypes.data_as(ctypes.c_void_p),
           coulG_real.ctypes.data_as(ctypes.c_void_p),
           V.ctypes.data_as(ctypes.c_void_p),
           basis_fft.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(bunchsize),
           mydf.jk_buffer.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(bufsize_per_thread),
           ctypes.c_int(CONSTRUCT_V))

        return V, basis_fft

    coulG = tools.get_coulG(cell, mesh=mesh)
    
    ############### the first communication ##############
    
    t0_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    sendbuf = _get_sendbuf_Row2Col(mydf.aux_basis, mydf.naux, mydf.ngrids)
    nRow = sendbuf.shape[0]
    nCol = sendbuf.shape[1] * comm_size
    # print("sendbuf.shape = ", sendbuf.shape, " on rank = ", rank)
    aux_fullcol = matrix_all2all_Row2Col(comm, nRow, nCol, sendbuf)
    # print("aux_fullcol.shape = ", aux_fullcol.shape, " on rank = ", rank)
    aux_fullcol = _get_packed_mat_Row2Col(aux_fullcol, mydf.naux, mydf.ngrids)
    
    t1_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    t_comm = t1_comm[1] - t0_comm[1]

    #### construct V ####

    V, basis_fft = constrcuct_V_CCode(aux_fullcol, mesh, coulG)
    
    ############### the second communication ##############
    
    t0_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    sendbuf = _get_sendbuf_Col2Row(V, mydf.naux, mydf.ngrids)
    nRow = sendbuf.shape[0] * comm_size
    nCol = sendbuf.shape[1]
    V_fullrow = matrix_all2all_Col2Row(comm, nRow, nCol, sendbuf)
    # print("V_fullrow.shape = ", V_fullrow.shape, " on rank = ", rank)
    # comm.Barrier()
    V_fullrow = _get_packed_mat_Col2Row(V_fullrow, mydf.naux, mydf.ngrids)
    # print("V_fullrow.shape = ", V_fullrow.shape, " on rank = ", rank)
    mydf.V_R = V_fullrow
    V = None
    
    sendbuf = _get_sendbuf_Col2Row(basis_fft, mydf.naux, ncomplex, force_bunch_even=True)
    nRow = sendbuf.shape[0] * comm_size
    nCol = sendbuf.shape[1]
    basis_fft_fullrow = matrix_all2all_Col2Row(comm, nRow, nCol, sendbuf)
    # print("basis_fft_fullrow.shape = ", basis_fft_fullrow.shape, " on rank = ", rank)
    # comm.Barrier()
    basis_fft_fullrow = _get_packed_mat_Col2Row(basis_fft_fullrow, mydf.naux, ncomplex, force_bunch_even=True)
    # print("basis_fft_fullrow.shape = ", basis_fft_fullrow.shape, " on rank = ", rank)
    basis_fft = None
    basis_fft = basis_fft_fullrow
    
    # print("basis_fft = ", basis_fft)
    
    t1_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    t_comm += (t1_comm[1] - t0_comm[1])
    
    #### construct W ####
    
    fn = getattr(libpbc, "_construct_W_multiG", None)
    assert(fn is not None)

    # to account for the rfft

    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1]
    if mesh[2] % 2 == 0:
        coulG_real[:,:,1:-1] *= 2
    else:
        coulG_real[:,:,1:] *= 2
    coulG_real = coulG_real.reshape(-1)
    
    # col_bunchsize = ngrids // comm_size + 1
    # p0 = min(ngrids, rank * col_bunchsize)
    # p1 = min(ngrids, (rank+1) * col_bunchsize)

    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1]
    # if mesh[2] % 2 == 0:
    #     coulG_real[:,:,1:-1] *= 2
    # else:
    #     coulG_real[:,:,1:] *= 2
    coulG_real = coulG_real.reshape(-1) 
    
    basis_fft2 = basis_fft.copy()
    
    if ncomplex % comm_size == 0:
        ColBunch = ncomplex // comm_size
    else:
        ColBunch = ncomplex // comm_size + 1
    if ColBunch % 2 == 1:
        ColBunch += 1
    
    p0 = min(ncomplex, rank * ColBunch)
    p1 = min(ncomplex, (rank+1) * ColBunch)
    
    assert p1 - p0 == basis_fft2.shape[1]
    assert naux == basis_fft2.shape[0]
    
    # W = np.zeros((naux, naux), dtype=np.double)
    
    # print("ncomplex = ", ncomplex)
    # print("p0, p1 = ", p0, p1)
    # print("basis_fft2.shape = ", basis_fft2.shape)
    # print("coulG_real = ", coulG_real)
    # print("naux = ", naux)
    
    fn(ctypes.c_int(naux),
       ctypes.c_int(p0//2),
       ctypes.c_int(p1//2),
       basis_fft2.ctypes.data_as(ctypes.c_void_p),
       coulG_real.ctypes.data_as(ctypes.c_void_p))
    
    # print("buf_aux_basis_fft_copy = ", basis_fft2)
    # print("basis_fft = ", basis_fft)
    
    W = lib.ddot(basis_fft2, basis_fft.T)
    
    W = reduce(W, root=0)
    
    factor = 1.0 / np.prod(mesh)
    W *= factor
    
    W = bcast(W, root=0)
    
    mydf.W = W
    
    if hasattr(mydf, "t_comm"):
        mydf.t_comm += t_comm
    else:
        mydf.t_comm = t_comm
        
    if mydf.verbose:
        print("t_comm = ", t_comm)
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t0, t2, "build_auxiliary_Coulomb")
    
    return V, W
    
####### get_jk MPI ######

def get_jk_dm_mpi(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, 
           **kwargs):
    '''JK for given k-point'''
    
    # print("get_jk_dm_mpi")
    # sys.stdout.flush()
    
    comm.Barrier()
    
    dm = bcast(dm, root=0)
    
    # print("get_jk_dm_mpi")
    # sys.stdout.flush()
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    # print("get_jk_dm_mpi")
    # sys.stdout.flush()
        
    #### perform the calculation ####

    if mydf.jk_buffer is None:  # allocate the buffer for get jk
        mydf._allocate_jk_buffer(dm.dtype)

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
    else:
        exxdiv = None

    vj = vk = None

    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        raise NotImplementedError("ISDF does not support kpts_band != kpt")

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not np.iscomplexobj(dm)

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)

    if with_j:
        vj = isdf_jk._contract_j_dm(mydf, dm, mydf.with_robust_fitting, True)
    if with_k:
        vk = isdf_jk._contract_k_dm(mydf, dm, mydf.with_robust_fitting, True)
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    return vj, vk

class PBC_ISDF_Info_MPI(isdf.PBC_ISDF_Info):

    def __init__(self, mol:Cell,
                 with_robust_fitting=True,
                 Ls=None,
                 verbose = 1
                 ):
        if rank != 0:
            verbose = 0
        super().__init__(mol, None, with_robust_fitting, Ls, False, verbose)

        if rank != 0:
            self.cell.verbose = 0

        # self.partition = build_partition(self)

    def _allocate_jk_buffer(self, datatype):

        if self.jk_buffer is None:

            nao    = self.nao
            ngrids = self.ngrids
            naux   = self.naux

            buffersize_k = nao * ngrids + naux * ngrids + naux * naux + nao * nao
            buffersize_j = nao * ngrids + ngrids + nao * naux + naux + naux + nao * nao
            
            buffersize_k = buffersize_k//comm_size + 1 + naux * naux + nao * nao
            buufersize_j = buffersize_j//comm_size + 1 + naux + naux + nao * nao

            nThreadsOMP   = lib.num_threads()
            size_ddot_buf = max((naux*naux)+2, ngrids) * nThreadsOMP

            if hasattr(self, "IO_buf"):

                if self.IO_buf.size < (max(buffersize_k, buffersize_j) + size_ddot_buf):
                    self.IO_buf = np.zeros((max(buffersize_k, buffersize_j) + size_ddot_buf,), dtype=datatype)

                self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),),
                                            dtype=datatype, buffer=self.IO_buf, offset=0)
                offset         = max(buffersize_k, buffersize_j) * self.jk_buffer.dtype.itemsize
                self.ddot_buf  = np.ndarray((nThreadsOMP, max((naux*naux)+2, ngrids)),
                                            dtype=datatype, buffer=self.IO_buf, offset=offset)

            else:

                self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),), dtype=datatype)
                self.ddot_buf = np.zeros((nThreadsOMP, max((naux*naux)+2, ngrids)), dtype=datatype)


        else:
            assert self.jk_buffer.dtype == datatype
            assert self.ddot_buf.dtype == datatype
    
    def build_IP_Sandeep(self, c:int, m:int, first_natm=None, global_IP_selection=True, debug=True):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        build_partition(self)
        IP_ID = isdf._select_IP_direct(self, c, m, global_IP_selection=global_IP_selection, use_mpi=True)
        IP_ID.sort()
        IP_ID = np.array(IP_ID, dtype=np.int32)
        self.IP_ID = IP_ID
        build_aoRg(self)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug and self.verbose:
            print("IP_ID = ", IP_ID)
            _benchmark_time(t1, t2, "build_IP")
        t1 = t2
        
        self.c = c 
        
        isdf.build_aux_basis(self, use_mpi=True)

    def build_auxiliary_Coulomb(self, debug=True):
        return build_auxiliary_Coulomb(self, debug=debug)

    # from functools import partial
    get_jk = get_jk_dm_mpi

C = 10

if __name__ == '__main__':

    cell   = pbcgto.Cell()
    
    boxlen = 4.2
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    cell.atom = '''
Li 0.0   0.0   0.0
Li 2.1   2.1   0.0
Li 0.0   2.1   2.1
Li 2.1   0.0   2.1
H  0.0   0.0   2.1
H  0.0   2.1   0.0
H  2.1   0.0   0.0
H  2.1   2.1   2.1
'''

    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    
    if rank == 0:
        cell.verbose = 4
    else:
        cell.verbose = 0

    cell.ke_cutoff  = 70   # kinetic energy cutoff in a.u.
    # cell.ke_cutoff = 70
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()
    
    pbc_isdf_info = PBC_ISDF_Info_MPI(cell)
    # build_partition(pbc_isdf_info)        
    pbc_isdf_info.build_IP_Sandeep(C, 5, global_IP_selection=True, debug=True)
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    aux_bas = comm.gather(pbc_isdf_info.aux_basis, root=0)
    aoR = comm.gather(pbc_isdf_info.aoR, root=0)
    V_R = comm.gather(pbc_isdf_info.V_R, root=0)
    
    if rank == 0:
        for x in aux_bas:
            print("x.shape = ", x.shape) 
        aux_bas = np.hstack(aux_bas) 
        aoR  = np.hstack(aoR)
        V_R  = np.hstack(V_R)
    
    # exit(1)
    
    if rank == 0:
        
        ############### check aoRg ##############
    
        from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG
        df_tmp = MultiGridFFTDF2(cell)
        grids  = df_tmp.grids
        mesh   = grids.mesh
        ngrids = np.prod(mesh)
        coords = np.asarray(grids.coords).reshape(-1,3)
        assert ngrids == coords.shape[0]
        aoR_bench   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
        aoR_bench  *= np.sqrt(cell.vol / ngrids)
        
        pbc_isdf_info_benchmark = isdf.PBC_ISDF_Info(cell, aoR=aoR_bench)
        partition_bench = np.array(pbc_isdf_info_benchmark.partition, dtype=np.int32)
        print("partition_bench.shape = ", partition_bench.shape)
        loc_diff = []
        for loc, (a, b) in enumerate(zip(pbc_isdf_info.partition, partition_bench)):
            # print(a, b)
            if a!=b:
                # raise ValueError("partition not equal")
                # print("partition not equal at loc = ", loc)
                loc_diff.append(loc)
        print("n diff = ", len(loc_diff))
        # assert np.allclose(pbc_isdf_info.partition, partition_bench)
        pbc_isdf_info_benchmark.build_IP_Sandeep(C, 5, global_IP_selection=True, debug=True, IP_ID=pbc_isdf_info.IP_ID)
        pbc_isdf_info_benchmark.build_auxiliary_Coulomb(debug=True)
    
        aoRg1 = pbc_isdf_info.aoRg
        aoRg2 = pbc_isdf_info_benchmark.aoRg
        assert np.allclose(aoRg1, aoRg2)
    
        aoR1 = pbc_isdf_info_benchmark.aoR
        aoR2 = aoR_bench
        assert np.allclose(aoR1, aoR2)
        
        # check aux_bas 
        
        print("aux_bas.shape = ", aux_bas.shape)
        
        aux_bas_bench = pbc_isdf_info_benchmark.aux_basis   
        
        diff = np.linalg.norm(aux_bas - aux_bas_bench) / np.sqrt(aux_bas.size)
        print("diff = ", diff)
        # print("aux_bas = ", aux_bas[:5,:5])
        # print("aux_bas_bench = ", aux_bas_bench[:5,:5])
        # print(" / ", aux_bas[:5,:5] / aux_bas_bench[:5,:5])
        # assert np.allclose(aux_bas, aux_bas_bench, atol=1e-7)
        
        for i in range(aux_bas.shape[0]):
            for j in range(aux_bas.shape[1]):
                if abs(aux_bas[i,j] - aux_bas_bench[i,j]) > 1e-5:
                    print("i, j = ", i, j)
                    print(aux_bas[i,j], aux_bas_bench[i,j])
                    print(" / ", aux_bas[i,j] / aux_bas_bench[i,j])
                    # raise ValueError("aux_bas not equal")
        
        V_R_bench = pbc_isdf_info_benchmark.V_R 
        
        diff = np.linalg.norm(V_R - V_R_bench) / np.sqrt(V_R.size)
        
        print("diff VR = ", diff)
        
        W = pbc_isdf_info.W
        W_bench = pbc_isdf_info_benchmark.W
        
        diff = np.linalg.norm(W - W_bench) / np.sqrt(W.size)
        print("W = ", W[:5,:5])
        print("W_bench = ", W_bench[:5,:5])
        print("W/W_bench = ", W[:5,:5] / W_bench[:5,:5])
        print("diff W = ", diff)
        sys.stdout.flush()
        
        from pyscf.pbc import scf
    
        mf = scf.RHF(cell)
        # pbc_isdf_info.direct_scf = mf.direct_scf
        mf.with_df = pbc_isdf_info_benchmark
        mf.max_cycle = 100
        mf.conv_tol = 1e-7
        # print("mf.direct_scf = ", mf.direct_scf)
        mf.kernel()
        
        
    ### run scf ###
    
    comm.Barrier()
    
    dm = None 
    
    if rank == 0:
        dm = np.random.random((1, cell.nao, cell.nao))
    
    dm = bcast(dm, root=0)
    
    print("generate the dm")
    sys.stdout.flush()
    
    # res = pbc_isdf_info.get_jk(dm, with_j=True, with_k=True)
    
    # get_jk_dm_mpi(pbc_isdf_info, dm, with_j=True, with_k=True)
    
    print("finish the get_jk_dm_mpi")
    sys.stdout.flush()
    
    from pyscf.pbc import scf
    
    mf = scf.RHF(cell)
    # pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7
    # print("mf.direct_scf = ", mf.direct_scf)
    mf.kernel()