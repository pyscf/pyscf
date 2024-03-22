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
# import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk
import pyscf.pbc.df.isdf.isdf_fast as isdf
from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info
# import pyscf.pbc.df.isdf.isdf_outcore as isdf_outcore
from pyscf.pbc.df.isdf.isdf_k import build_supercell

from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch

# from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, matrix_all2all_Col2Row, matrix_all2all_Row2Col

# from mpi4pyscf.tools import mpi
# from mpi4pyscf.tools.mpi import allgather, bcast,  reduce

# comm = mpi.comm
# rank = mpi.rank
# comm_size = comm.Get_size()

import ctypes
from multiprocessing import Pool
from memory_profiler import profile

libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

# @mpi.parallel_call
def build_partition(mydf):
        
    nGrids = mydf.ngrids    
    grid_bunch = _comm_bunch(nGrids)
    
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

# @mpi.parallel_call
def build_aoRg(mydf):
    
    aoR = mydf.aoR
    assert aoR is not None

    weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    nIP = len(mydf.IP_ID)
    ngrids = mydf.ngrids
    nao = mydf.nao
    p0 = min(nIP, rank * nIP // comm_size)
    p1 = min(nIP, (rank+1) * nIP // comm_size)
    
    mydf.aoRg = ISDF_eval_gto(mydf.cell, coords=mydf.coords[mydf.IP_ID]).real * weight
            
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

# @mpi.parallel_call
def build_auxiliary_Coulomb(mydf, debug=True):
    
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
    
    mydf.coulG = coulG.copy()
    
    ############### the first communication ##############
    
    t0_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    comm_size_row = _comm_bunch(mydf.naux)
    sendbuf = []
    for i in range(comm_size):
        p0 = min(i*comm_size_row, mydf.naux)
        p1 = min((i+1)*comm_size_row, mydf.naux)
        sendbuf.append(mydf.aux_basis[p0:p1, :])
    aux_fullcol = np.hstack(alltoall(sendbuf, split_recvbuf=True))
    mydf.aux_basis = None
    sendbuf = None
    
    t1_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    t_comm = t1_comm[1] - t0_comm[1]

    #### construct V ####

    V, basis_fft = constrcuct_V_CCode(aux_fullcol, mesh, coulG)
    
    ############### the second communication ##############
    
    t0_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    comm_size_col = _comm_bunch(mydf.ngrids)
    sendbuf = []
    for i in range(comm_size):
        p0 = min(i*comm_size_col, mydf.ngrids)
        p1 = min((i+1)*comm_size_col, mydf.ngrids)
        sendbuf.append(V[:, p0:p1])
    V_fullrow = np.vstack(alltoall(sendbuf, split_recvbuf=True))
    sendbuf = None
    V = None
    
    mydf.V_R = V_fullrow
    
    comm_size_col = _comm_bunch(ncomplex, force_even=True)
    sendbuf = []
    for i in range(comm_size):
        p0 = min(i*comm_size_col, ncomplex)
        p1 = min((i+1)*comm_size_col, ncomplex)
        sendbuf.append(basis_fft[:, p0:p1])
    basis_fft_fullrow = np.vstack(alltoall(sendbuf, split_recvbuf=True))
    sendbuf = None
    basis_fft = None
    basis_fft = basis_fft_fullrow 
    
    # mydf.basis_fft = basis_fft_fullrow.copy()
    
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
    
    # mydf.coulG_real = coulG_real
    # mydf.coulG = coulG.copy()
    
    basis_fft2 = basis_fft.copy()

    ColBunch = comm_size_col
    
    p0 = min(ncomplex, rank * ColBunch)
    p1 = min(ncomplex, (rank+1) * ColBunch)
    
    print("p0 = ", p0)
    print("p1 = ", p1)  
    
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
    
    # W = reduce(W, root=0)
    
    factor = 1.0 / np.prod(mesh)
    W *= factor
    
    # W = bcast(W, root=0)
    
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
    
    # clean # 
    
    basis_fft_fullrow = None
    basis_fft = None
    basis_fft2 = None
    
    return V, W
    
####### get_jk MPI ######

def get_jk_dm_mpi(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, 
           **kwargs):
    '''JK for given k-point'''
    
    comm.Barrier()
    
    dm = bcast(dm, root=0)
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
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
        if mydf.with_robust_fitting:
            vj = isdf_jk._contract_j_dm_fast(mydf, dm, mydf.with_robust_fitting, True)
            # vj2 = isdf_jk._contract_j_dm(mydf, dm, mydf.with_robust_fitting, True)
            # print("vj = ", vj[0,-10:])
            # print("vj2 = ", vj2[0,-10:])
            # print("vj/vj2 = ", vj[0,-10:] / vj2[0,-10:])
        else:
            vj = isdf_jk._contract_j_dm(mydf, dm, mydf.with_robust_fitting, True)

    if with_k:
        vk = isdf_jk._contract_k_dm(mydf, dm, mydf.with_robust_fitting, True)
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    # if rank == 0:
    #     print("vj = ", vj[0,-10:])
    #     print("vk = ", vk[0,-10:])

    return vj * comm_size , vk * comm_size ### ? 

# @mpi.register_class
class PBC_ISDF_Info_MPI(PBC_ISDF_Info):

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

C = 12
KE_CUTOFF = 128

if __name__ == '__main__':

    # from pyscf.pbc.df.isdf import isdf_fast_mpi

#     atm = '''
# Li 0.0   0.0   0.0
# Li 2.1   2.1   0.0
# Li 0.0   2.1   2.1
# Li 2.1   0.0   2.1
# H  0.0   0.0   2.1
# H  0.0   2.1   0.0
# H  2.1   0.0   0.0
# H  2.1   2.1   2.1
# ''' 
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ]
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

    cell   = pbcgto.Cell()
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    cell.atom = atm
    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    
    if rank == 0:
        cell.verbose = 4
    else:
        cell.verbose = 0

    cell.ke_cutoff  = KE_CUTOFF   # kinetic energy cutoff in a.u.
    # cell.ke_cutoff = 70
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()
    
    prim_mesh = cell.mesh
    Ls = [1, 1, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    cell = build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
    pbc_isdf_info = PBC_ISDF_Info_MPI(cell, with_robust_fitting=True)
    # build_partition(pbc_isdf_info)        
    pbc_isdf_info.build_IP_Sandeep(C, 5, global_IP_selection=True, debug=True)
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    sys.stdout.flush()
    
    comm.Barrier()
    
    # aoR = pbc_isdf_info.aoR.copy()
    # V_R = pbc_isdf_info.V_R.copy()
    # W = pbc_isdf_info.W.copy()
    
    # aux_bas = comm.gather(pbc_isdf_info.aux_basis, root=0)
    # aoR = comm.gather(pbc_isdf_info.aoR, root=0)
    # V_R = comm.gather(pbc_isdf_info.V_R, root=0)
    # W = reduce(pbc_isdf_info.W, root=0)
    
    # aoR = comm.gather(aoR, root=0)
    # V_R = comm.gather(V_R, root=0)
    aoR = gather(pbc_isdf_info.aoR, split_recvbuf=True)
    V_R = gather(pbc_isdf_info.V_R, split_recvbuf=True)
    W = reduce(pbc_isdf_info.W, root=0)
    
    if rank == 0:
        # for x in aux_bas:
        #     print("x.shape = ", x.shape) 
        # aux_bas = np.hstack(aux_bas) 
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
        df_tmp = None
        
        pbc_isdf_info_benchmark = isdf.PBC_ISDF_Info(cell, aoR=aoR_bench, with_robust_fitting=True)
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
        
        # print("aux_bas.shape = ", aux_bas.shape)
        
        # aux_bas_bench = pbc_isdf_info_benchmark.aux_basis   
        
        # diff = np.linalg.norm(aux_bas - aux_bas_bench) / np.sqrt(aux_bas.size)
        # print("diff = ", diff)
        # print("aux_bas = ", aux_bas[:5,:5])
        # print("aux_bas_bench = ", aux_bas_bench[:5,:5])
        # print(" / ", aux_bas[:5,:5] / aux_bas_bench[:5,:5])
        # assert np.allclose(aux_bas, aux_bas_bench, atol=1e-7)
        
        # for i in range(aux_bas.shape[0]):
        #     for j in range(aux_bas.shape[1]):
        #         if abs(aux_bas[i,j] - aux_bas_bench[i,j]) > 1e-5:
        #             print("i, j = ", i, j)
        #             print(aux_bas[i,j], aux_bas_bench[i,j])
        #             print(" / ", aux_bas[i,j] / aux_bas_bench[i,j])
        #             # raise ValueError("aux_bas not equal")
        
        V_R_bench = pbc_isdf_info_benchmark.V_R 
        
        diff = np.linalg.norm(V_R - V_R_bench) / np.sqrt(V_R.size)
        
        print("diff VR = ", diff)
        
        for i in range(V_R.shape[0]):
            for j in range(V_R.shape[0]):
                if abs(V_R[i,j]-V_R_bench[i,j]) > 1e-6:
                    print("%2d %2d %15.8f %15.8f" % (i, j, V_R[i,j], V_R_bench[i,j]))
        
        # W = pbc_isdf_info.W
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