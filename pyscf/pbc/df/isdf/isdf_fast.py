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

import ctypes

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

BASIS_CUTOFF               = 1e-18  # too small may lead to numerical instability
CRITERION_CALL_PARALLEL_QR = 256

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

from pyscf.pbc.dft import multigrid

################### the MPI module ##########################

import mpi4py
from mpi4py import MPI
# from mpi4pyscf.tools.mpi import allgather

# from mpi4pyscf.tools import mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

## some tools copy from mpi4pyscf ##

INT_MAX = 2147483647
# INT_MAX = 65536
BLKSIZE = INT_MAX // 32 + 1

def _comm_bunch(size_of_comm, force_even=False):
    if size_of_comm % comm_size == 0:
        res = size_of_comm // comm_size
    else:
        res = (size_of_comm // comm_size) + 1
    if force_even:
        if res % 2 == 1 :
            res += 1
    return res

def _assert(condition):
    if not condition:
        import traceback
        sys.stderr.write(''.join(traceback.format_stack()[:-1]))
        comm.Abort()

def _segment_counts(counts, p0, p1):
    counts_seg = counts - p0
    counts_seg[counts<=p0] = 0
    counts_seg[counts> p1] = p1 - p0
    return counts_seg
     
def allgather(sendbuf, split_recvbuf=False):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape = sendbuf.shape
    attr = comm.allgather((shape, sendbuf.dtype.char))
    rshape = [x[0] for x in attr]
    counts = numpy.array([numpy.prod(x) for x in rshape])
    mpi_dtype = numpy.result_type(*[x[1] for x in attr]).char
    _assert(sendbuf.dtype.char == mpi_dtype or sendbuf.size == 0)

    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(sum(counts), dtype=mpi_dtype)

    sendbuf = sendbuf.ravel()

    size_of_recvbuf = recvbuf.size

    print("rank %d size recvbf %d" % (rank, size_of_recvbuf))

    if size_of_recvbuf >= INT_MAX:
        print("large data size go this branch")
        blk_size_small = min((INT_MAX // comm_size),BLKSIZE)
        recvbuf_small = numpy.empty(comm_size*blk_size_small, dtype=mpi_dtype)
        rdispls_small = numpy.arange(comm_size)*blk_size_small
        if rank == 0:
            print("blk_size_small = ", blk_size_small)
            print("rdispls_small = ", rdispls_small)
            sys.stdout.flush()
        for p0, p1 in prange(0, numpy.max(counts), blk_size_small):
            counts_seg = _segment_counts(counts, p0, p1)
            comm.Allgatherv([sendbuf[p0:p1], mpi_dtype],
                            [recvbuf_small, counts_seg, rdispls_small, mpi_dtype])
            # recvbuf[p0:p1] = recvbuf_small[:p1-p0]

            for i in range(comm_size):
                begin = displs[i]+p0
                end = begin + counts_seg[i]
                recvbuf[begin:end] = recvbuf_small[i*blk_size_small:i*blk_size_small+counts_seg[i]]

        del recvbuf_small
        del rdispls_small   
        
        if split_recvbuf:
            return [recvbuf[p0:p0+c].reshape(shape)
                    for p0,c,shape in zip(displs,counts,rshape)]
        else:
            return recvbuf
    else:
        print("small data size go this branch")
        print("maxcount = ", numpy.max(counts))
        end = numpy.max(counts)
        for p0, p1 in lib.prange(0, end, BLKSIZE):
            print("rank %d send p0 p1 %d %d"%(rank,p0,p1))
            counts_seg = _segment_counts(counts, p0, p1)
            comm.Allgatherv([sendbuf[p0:p1], mpi_dtype],
                            [recvbuf, counts_seg, displs+p0, mpi_dtype])
        print("rank %d finish all gather" % (rank))
        if split_recvbuf:
            return [recvbuf[p0:p0+c].reshape(shape)
                    for p0,c,shape in zip(displs,counts,rshape)]
        else:
            # try:
            #     return recvbuf.reshape((-1,) + shape[1:])
            # except ValueError:
            return recvbuf
            # raise ValueError("split_recvbuf is not supported")

def allgather_list(sendbuf):
    
    assert isinstance(sendbuf, list)
    for _data_ in sendbuf:
        assert isinstance(_data_, numpy.ndarray)
    
    shape = [x.shape for x in sendbuf]
    attr = comm.allgather(shape)
    attr_flat = []
    for x in attr:
        for y in x:
            attr_flat.append(y)

    if rank == 0:
        for x in attr_flat:
            print("x = ", x)

    print("rank %d get here 1" % (rank))
    sys.stdout.flush()

    size_tot = np.sum([x.size for x in sendbuf])
    sendbuf_flat = np.empty(size_tot, dtype=sendbuf[0].dtype)
    offset = 0
    for x in sendbuf:
        sendbuf_flat[offset:offset+x.size] = x.ravel()
        offset += x.size
    
    print("rank %d get here 2" % (rank))
    sys.stdout.flush()

    recvbuf_flat = allgather(sendbuf_flat)
    
    print("rank %d get here 3" % (rank))
    sys.stdout.flush()
    res = []
    
    offset = 0
    for x in attr_flat:
        res.append(recvbuf_flat[offset:offset+np.prod(x)].reshape(x))
        offset += np.prod(x)

    return res

def allgather_pickle(sendbuf):
    sendbuf_serialized = MPI.pickle.dumps(sendbuf)
    sendbuf_serialized = np.frombuffer(sendbuf_serialized, dtype=np.uint8)
    received = allgather(sendbuf_serialized, split_recvbuf=True)
    received = [MPI.pickle.loads(x.tobytes()) for x in received]
    del sendbuf_serialized 
    return received

def reduce(sendbuf, op=MPI.SUM, root=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char),root=root)
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    dtype = sendbuf.dtype.char
    recvbuf = numpy.zeros_like(sendbuf)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Reduce([send_seg[p0:p1], dtype],
                    [recv_seg[p0:p1], dtype], op, root)

    if rank == root:
        return recvbuf
    else:
        return sendbuf

def scatter(sendbuf, root=0):
    if rank == root:
        mpi_dtype = numpy.result_type(*sendbuf).char
        shape = comm.scatter([x.shape for x in sendbuf])
        counts = numpy.asarray([x.size for x in sendbuf])
        comm.bcast((mpi_dtype, counts))
        sendbuf = [numpy.asarray(x, mpi_dtype).ravel() for x in sendbuf]
        sendbuf = numpy.hstack(sendbuf)
    else:
        shape = comm.scatter(None)
        mpi_dtype, counts = comm.bcast(None)

    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(numpy.prod(shape), dtype=mpi_dtype)

    #DONOT use lib.prange. lib.prange may terminate early in some processes
    for p0, p1 in prange(comm, 0, numpy.max(counts), BLKSIZE):
        counts_seg = _segment_counts(counts, p0, p1)
        comm.Scatterv([sendbuf, counts_seg, displs+p0, mpi_dtype],
                      [recvbuf[p0:p1], mpi_dtype], root)
    return recvbuf.reshape(shape)

def bcast(buf, root=0):
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char), root=root)
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)

    dtype = buf.dtype.char
    buf_seg = numpy.ndarray(buf.size, dtype=buf.dtype, buffer=buf)
    for p0, p1 in lib.prange(0, buf.size, BLKSIZE):
        comm.Bcast([buf_seg[p0:p1], dtype], root)
    return buf

def bcast_pickel(buf, root=0):
    if rank == root:
        buf_serialized = MPI.pickle.dumps(buf)
        buf_serialized = np.frombuffer(buf_serialized, dtype=np.uint8)
    else:
        buf_serialized = None
    res = bcast(buf_serialized, root)
    res = MPI.pickle.loads(res.tobytes())
    return res

def gather(sendbuf, root=0, split_recvbuf=False):

    sendbuf = numpy.asarray(sendbuf, order='C')
    shape = sendbuf.shape
    size_dtype = comm.allgather((shape, sendbuf.dtype.char))
    # print(size_dtype)
    rshape = [x[0] for x in size_dtype]
    counts = numpy.array([numpy.prod(x) for x in rshape])

    mpi_dtype = numpy.result_type(*[x[1] for x in size_dtype]).char
    _assert(sendbuf.dtype == mpi_dtype or sendbuf.size == 0)

    if rank == root:
        displs = numpy.append(0, numpy.cumsum(counts[:-1]))
        recvbuf = numpy.empty(sum(counts), dtype=mpi_dtype)

        sendbuf = sendbuf.ravel()
        for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
            counts_seg = _segment_counts(counts, p0, p1)
            comm.Gatherv([sendbuf[p0:p1], mpi_dtype],
                         [recvbuf, counts_seg, displs+p0, mpi_dtype], root)
        if split_recvbuf:
            return [recvbuf[p0:p0+c].reshape(shape)
                    for p0,c,shape in zip(displs,counts,rshape)]
        else:
            try:
                return recvbuf.reshape((-1,) + shape[1:])
            except ValueError:
                return recvbuf
    else:
        send_seg = sendbuf.ravel()
        for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
            comm.Gatherv([send_seg[p0:p1], mpi_dtype], None, root)
        return sendbuf

def prange(start, stop, step):
    '''Similar to lib.prange. This function ensures that all processes have the
    same number of steps.  It is required by alltoall communication.
    '''
    nsteps = (stop - start + step - 1) // step
    nsteps = max(comm.allgather(nsteps))
    for i in range(nsteps):
        i0 = min(stop, start + i * step)
        i1 = min(stop, i0 + step)
        yield i0, i1
        
def alltoall(sendbuf, split_recvbuf=False):
    if isinstance(sendbuf, numpy.ndarray):
        raise NotImplementedError
        mpi_dtype = comm.bcast(sendbuf.dtype.char)
        sendbuf = numpy.asarray(sendbuf, mpi_dtype, 'C')
        nrow = sendbuf.shape[0]
        ncol = sendbuf.size // nrow
        segsize = (nrow+comm_size-1) // comm_size * ncol
        sdispls = numpy.arange(0, comm_size*segsize, segsize)
        sdispls[sdispls>sendbuf.size] = sendbuf.size
        scounts = numpy.append(sdispls[1:]-sdispls[:-1], sendbuf.size-sdispls[-1])
        rshape = comm.alltoall(scounts)
    else:
        _assert(len(sendbuf) == comm_size)
        mpi_dtype = comm.bcast(sendbuf[0].dtype.char)
        sendbuf = [numpy.asarray(x, mpi_dtype) for x in sendbuf]
        rshape = comm.alltoall([x.shape for x in sendbuf])
        scounts = numpy.asarray([x.size for x in sendbuf], dtype=np.int64)
        sdispls = numpy.append(0, numpy.cumsum(scounts[:-1]))
        sendbuf = numpy.hstack([x.ravel() for x in sendbuf])

    rcounts = numpy.asarray([numpy.prod(x) for x in rshape], dtype=np.int64)
    rdispls = numpy.append(0, numpy.cumsum(rcounts[:-1]))
    recvbuf = numpy.empty(sum(rcounts), dtype=mpi_dtype)

    if rank == 0:
        print("sdispls = ", sdispls)
        print("rcounts = ", rcounts)
        print("rdispls = ", rdispls)

    max_counts = max(numpy.max(scounts), numpy.max(rcounts))
    
    if rank == 0:
        print("max_counts = ", max_counts)
    
    sendbuf = sendbuf.ravel()
    #DONOT use lib.prange. lib.prange may terminate early in some processes
    
    size_of_sendbuf = sendbuf.size
    
    # if sdispls[-1] >= INT_MAX:
    if size_of_sendbuf >=INT_MAX:
        blk_size_small = min((INT_MAX // comm_size),BLKSIZE)
        sendbuf_small = numpy.empty(comm_size*blk_size_small, dtype=mpi_dtype)
        recvbuf_small = numpy.empty(comm_size*blk_size_small, dtype=mpi_dtype)
        sdispls_small = numpy.arange(comm_size)*blk_size_small
        if rank == 0:
            print("blk_size_small = ", blk_size_small)
            print("sdispls_small = ", sdispls_small)
            sys.stdout.flush()
        for p0, p1 in prange(0, max_counts, blk_size_small):
            scounts_seg = _segment_counts(scounts, p0, p1)
            rcounts_seg = _segment_counts(rcounts, p0, p1)
            
            # if rank == 0:
            #     print("p0 p1 = ", p0, p1)
            #     print("scounts_seg = ", scounts_seg)
            #     print("rcounts_seg = ", rcounts_seg)
            #     sys.stdout.flush()
            ### copy data to sendbuf_small
            for i in range(comm_size):
                begin = sdispls[i]+p0
                end = begin + scounts_seg[i]
                sendbuf_small[i*blk_size_small:i*blk_size_small+scounts_seg[i]] = sendbuf[begin:end]
            
            comm.Alltoallv([sendbuf_small, scounts_seg, sdispls_small, mpi_dtype],
                           [recvbuf_small, rcounts_seg, sdispls_small, mpi_dtype])
            
            for i in range(comm_size):
                begin = rdispls[i]+p0
                end = begin + rcounts_seg[i]
                recvbuf[begin:end] = recvbuf_small[i*blk_size_small:i*blk_size_small+rcounts_seg[i]]
                
        sendbuf_small = None
        recvbuf_small = None
    else:
        for p0, p1 in prange(0, max_counts, BLKSIZE):
            scounts_seg = _segment_counts(scounts, p0, p1)
            rcounts_seg = _segment_counts(rcounts, p0, p1)
            # if rank == 0:
            #     print("scounts_seg = ", scounts_seg)
            #     print("rcounts_seg = ", rcounts_seg)
            comm.Alltoallv([sendbuf, scounts_seg, sdispls+p0, mpi_dtype],
                           [recvbuf, rcounts_seg, rdispls+p0, mpi_dtype])

    # return None

    if split_recvbuf:
        return [recvbuf[p0:p0+c].reshape(shape)
                for p0,c,shape in zip(rdispls, rcounts, rshape)]
    else:
        return recvbuf
    
################### end of the MPI module ##########################

# python version colpilot_qr() function

def colpivot_qr(A, max_rank=None, cutoff=1e-14):  # python code, benchmark, but not efficient
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

def _select_IP_direct(mydf, c:int, m:int, first_natm=None, global_IP_selection=True, 
                      aoR_cutoff = None,
                      rela_cutoff = 0.0, 
                      no_retriction_on_nIP = False,
                      use_mpi=False):

    # bunchsize = lib.num_threads()

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())

    ### determine the largest grids point of one atm ###

    natm         = mydf.cell.natm
    nao          = mydf.nao
    naux_max     = 0

    nao_per_atm = np.zeros((natm), dtype=np.int32)
    for i in range(mydf.nao):
        atm_id = mydf.ao2atomID[i]
        nao_per_atm[atm_id] += 1

    for nao_atm in nao_per_atm:
        naux_max = max(naux_max, int(np.sqrt(c*nao_atm)) + m)

    nthread = lib.num_threads()

    buf_size_per_thread = mydf.get_buffer_size_in_IP_selection(c, m)
    buf_size            = buf_size_per_thread

    # print("nthread        = ", nthread)
    # print("buf_size       = ", buf_size)
    # print("buf_per_thread = ", buf_size_per_thread)

    if hasattr(mydf, "IO_buf"):
        buf = mydf.IO_buf
    else:
        buf = np.zeros((buf_size), dtype=np.float64)
        mydf.IO_buf = buf

    if buf.size < buf_size:
        # reallocate
        mydf.IO_buf = np.zeros((buf_size), dtype=np.float64)
        # print("reallocate buf of size = ", buf_size)
        buf = mydf.IO_buf
    buf_tmp = np.ndarray((buf_size), dtype=np.float64, buffer=buf)

    ### loop over atm ###

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(mydf.cell)
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    assert coords is not None

    results = []

    # fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    fn_colpivot_qr = getattr(libpbc, "ColPivotQRRelaCut", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])

    for p0, p1 in lib.prange(0, 1, 1):

        taskinfo = []

        # clear buffer

        if first_natm is None:
            first_natm = natm
    
        for atm_id in range(first_natm):
            
            if use_mpi:
                if atm_id % comm_size != rank:
                    continue

            buf_tmp[:buf_size_per_thread] = 0.0

            grid_ID = np.where(mydf.partition == atm_id)[0]

            offset  = 0
            aoR_atm = np.ndarray((nao, grid_ID.shape[0]), dtype=np.complex128, buffer=buf_tmp, offset=offset)
            aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[grid_ID], out=aoR_atm) * weight
            
            nao_tmp = nao
            
            if aoR_cutoff is not None:
                print("aoR_cutoff = ", aoR_cutoff)
                max_row = np.max(np.abs(aoR_atm), axis=1)
                where = np.where(max_row > mydf.aoR_cutoff)[0]
                print("before cutoff aoR_atm.shape = ", aoR_atm.shape)
                aoR_atm = aoR_atm[where]
                print("after  cutoff aoR_atm.shape = ", aoR_atm.shape)
                nao_tmp = aoR_atm.shape[0]

            # create buffer for this atm

            dtypesize = buf.dtype.itemsize

            offset += nao_tmp*grid_ID.shape[0] * dtypesize

            nao_atm  = nao_per_atm[atm_id]
            naux_now = int(np.sqrt(c*nao_atm)) + m
            naux2_now = naux_now * naux_now

            # R = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            R = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64)
            offset += naux2_now*grid_ID.shape[0] * dtypesize

            aoR_atm1 = np.ndarray((naux_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*grid_ID.shape[0] * dtypesize

            aoR_atm2 = np.ndarray((naux_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*grid_ID.shape[0] * dtypesize

            aoPairBuffer = np.ndarray(
                (naux_now*naux_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*naux_now*grid_ID.shape[0] * dtypesize

            G1 = np.random.rand(nao_tmp, naux_now)
            G1, _ = numpy.linalg.qr(G1)
            G1    = G1.T
            G2 = np.random.rand(nao_tmp, naux_now)
            G2, _ = numpy.linalg.qr(G2)
            G2    = G2.T

            lib.dot(G1, aoR_atm, c=aoR_atm1)
            lib.dot(G2, aoR_atm, c=aoR_atm2)

            fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                         aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                         aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux_now),
                         ctypes.c_int(naux_now),
                         ctypes.c_int(grid_ID.shape[0]))
            if global_IP_selection:
                if no_retriction_on_nIP:
                    max_rank = min(naux2_now, grid_ID.shape[0])
                else:
                    max_rank  = min(naux2_now, grid_ID.shape[0], nao_atm * c + m)
            else:
                if no_retriction_on_nIP:
                    max_rank = min(naux2_now, grid_ID.shape[0])
                else:
                    max_rank  = min(naux2_now, grid_ID.shape[0], nao_atm * c)
            npt_find      = ctypes.c_int(0)
            pivot         = np.arange(grid_ID.shape[0], dtype=np.int32)
            thread_buffer = np.ndarray((nthread+1, grid_ID.shape[0]+1), dtype=np.float64, buffer=buf_tmp, offset=offset)
            # thread_buffer = np.ndarray((nthread+1, grid_ID.shape[0]+1), dtype=np.float64)
            offset       += (nthread+1)*(grid_ID.shape[0]+1) * dtypesize
            global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            # global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64)
            offset       += grid_ID.shape[0] * dtypesize

            # print("thread_buffer.shape = ", thread_buffer.shape)
            fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(naux2_now),
                            ctypes.c_int(grid_ID.shape[0]),
                            ctypes.c_int(max_rank),
                            ctypes.c_double(1e-14),
                            ctypes.c_double(rela_cutoff),
                            pivot.ctypes.data_as(ctypes.c_void_p),
                            R.ctypes.data_as(ctypes.c_void_p),
                            ctypes.byref(npt_find),
                            thread_buffer.ctypes.data_as(ctypes.c_void_p),
                            global_buffer.ctypes.data_as(ctypes.c_void_p))
            npt_find = npt_find.value
            
            # aoPairBuffer, R, pivot, npt_find = colpivot_qr(aoPairBuffer, max_rank)
            
            cutoff   = abs(R[npt_find-1, npt_find-1])
            print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (grid_ID.shape[0], npt_find, cutoff))
            pivot = pivot[:npt_find]
            pivot.sort()
            results.extend(list(grid_ID[pivot]))

    # print("results = ", results)

    if use_mpi:
        comm.Barrier()
        results = allgather(results)
        # results.sort()
    results.sort()
    
    # print("results = ", results)

    ### global IP selection, we can use this step to avoid numerical issue ###

    if global_IP_selection and rank == 0:

        if mydf.verbose:
            print("global IP selection")

        bufsize = mydf.get_buffer_size_in_global_IP_selection(len(results), c, m)

        if buf.size < bufsize:
            mydf.IO_buf = np.zeros((bufsize), dtype=np.float64)
            buf = mydf.IO_buf
            if mydf.verbose:
                print("reallocate buf of size = ", bufsize)

        dtypesize = buf.dtype.itemsize

        buf_tmp = np.ndarray((bufsize), dtype=np.float64, buffer=buf)

        offset = 0
        aoRg   = np.ndarray((nao, len(results)), dtype=np.complex128, buffer=buf_tmp)
        aoRg   = ISDF_eval_gto(mydf.cell, coords=coords[results], out=aoRg) * weight

        offset += nao*len(results) * dtypesize

        naux_now  = int(np.sqrt(c*nao)) + m
        naux2_now = naux_now * naux_now

        # print("naux_now = ", naux_now)
        # print("naux2_now = ", naux2_now)

        # R = np.ndarray((naux2_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
        R = np.ndarray((naux2_now, len(results)), dtype=np.float64)
        offset += naux2_now*len(results) * dtypesize

        aoRg1 = np.ndarray((naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
        offset += naux_now*len(results) * dtypesize

        aoRg2 = np.ndarray((naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
        offset += naux_now*len(results) * dtypesize

        aoPairBuffer = np.ndarray(
            (naux_now*naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
        offset += naux_now*naux_now*len(results) * dtypesize

        G1 = np.random.rand(nao, naux_now)
        G1, _ = numpy.linalg.qr(G1)
        G1    = G1.T
        G2 = np.random.rand(nao, naux_now)
        G2, _ = numpy.linalg.qr(G2)
        G2    = G2.T

        lib.dot(G1, aoRg, c=aoRg1)
        lib.dot(G2, aoRg, c=aoRg2)

        fn_ik_jk_ijk(aoRg1.ctypes.data_as(ctypes.c_void_p),
                     aoRg2.ctypes.data_as(ctypes.c_void_p),
                     aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux_now),
                     ctypes.c_int(naux_now),
                     ctypes.c_int(len(results)))

        nao_first = np.sum(nao_per_atm[:first_natm])

        if no_retriction_on_nIP:
            max_rank = min(naux2_now, len(results))
        else:
            max_rank  = min(naux2_now, len(results), nao_first * c)

        # print("max_rank = ", max_rank)

        npt_find      = ctypes.c_int(0)
        pivot         = np.arange(len(results), dtype=np.int32)
        thread_buffer = np.ndarray((nthread+1, len(results)+1), dtype=np.float64, buffer=buf_tmp, offset=offset)
        # thread_buffer = np.ndarray((nthread+1, len(results)+1), dtype=np.float64)
        offset       += (nthread+1)*(len(results)+1) * dtypesize
        global_buffer = np.ndarray((1, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
        # global_buffer = np.ndarray((1, len(results)), dtype=np.float64)
        offset       += len(results) * dtypesize

        fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naux2_now),
                        ctypes.c_int(len(results)),
                        ctypes.c_int(max_rank),
                        ctypes.c_double(1e-14),
                        ctypes.c_double(rela_cutoff),
                        pivot.ctypes.data_as(ctypes.c_void_p),
                        R.ctypes.data_as(ctypes.c_void_p),
                        ctypes.byref(npt_find),
                        thread_buffer.ctypes.data_as(ctypes.c_void_p),
                        global_buffer.ctypes.data_as(ctypes.c_void_p))
        npt_find = npt_find.value
        
        # aoPairBuffer, R, pivot, npt_find = colpivot_qr(aoPairBuffer, max_rank)
        
        cutoff   = abs(R[npt_find-1, npt_find-1])
        print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (len(results), npt_find, cutoff))
        pivot = pivot[:npt_find]
        # print("pivot = ", pivot)

        pivot.sort()

        results = np.array(results, dtype=np.int32)
        results = list(results[pivot])

    if global_IP_selection and use_mpi:
        results = bcast(results)

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())

    return results

def build_aux_basis(mydf, debug=True, use_mpi=False):
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    # allocate memory for the auxiliary basis

    naux = mydf.IP_ID.shape[0]
    mydf.naux = naux
    mydf._allocate_jk_buffer(datatype=np.double)
    buffer1 = np.ndarray((mydf.naux , mydf.naux), dtype=np.double, buffer=mydf.jk_buffer, offset=0)
    
    nao = mydf.nao
    IP_ID = mydf.IP_ID
    aoR = mydf.aoR

    if not hasattr(mydf, "aoRg") or mydf.aoRg is None:
        aoRg = numpy.empty((mydf.nao, mydf.IP_ID.shape[0]))
        lib.dslice(aoR, IP_ID, out=aoRg)
    else:
        aoRg = mydf.aoRg
    
    # print("aoR = ", aoR)
    
    # A = None
    e = None
    h = None
    
    if not use_mpi or (use_mpi and rank == 0):
        A = np.asarray(lib.ddot(aoRg.T, aoRg, c=buffer1), order='C')  # buffer 1 size = naux * naux
        lib.square_inPlace(A)
        
        # fn_cholesky = getattr(libpbc, "Cholesky", None)
        # assert(fn_cholesky is not None)
        # fn_cholesky(
        #     A.ctypes.data_as(ctypes.c_void_p),
        #     ctypes.c_int(naux),
        # )
        
        t11 = (lib.logger.process_clock(), lib.logger.perf_counter())
        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
            e, h = scipy.linalg.eigh(A)
        t12 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t11, t12, "diag_A")
        print("condition number = ", e[-1]/e[0])
        where = np.where(e > e[-1]*1e-16)[0]
        # for id, val in enumerate(e):
        #     print("e[%5d] = %15.8e" % (id, val))
        e = e[where]
        h = h[:, where]
    # else:
    #     A = None
        
        
    if use_mpi:
        # A = bcast(A)
        e = bcast(e)
        h = bcast(h)
    
    mydf.aux_basis = np.asarray(lib.ddot(aoRg.T, aoR), order='C')   # buffer 2 size = naux * ngrids
    lib.square_inPlace(mydf.aux_basis)
    
    # print("mydf.aux_basis = ", mydf.aux_basis)

    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    nThread = lib.num_threads()
    nGrids  = aoR.shape[1]
    Bunchsize = nGrids // nThread
    
    # fn_build_aux(
    #     ctypes.c_int(naux),
    #     A.ctypes.data_as(ctypes.c_void_p),
    #     mydf.aux_basis.ctypes.data_as(ctypes.c_void_p),
    #     ctypes.c_int(nGrids),
    #     ctypes.c_int(Bunchsize)
    # )

    # use diagonalization instead, but too slow for large system
    # e, h = np.linalg.eigh(A)  # single thread, but should not be slow, it should not be the bottleneck
    # print("e[-1] = ", e[-1])
    # print("e[0]  = ", e[0])
    # print("condition number = ", e[-1]/e[0])
    # for id, val in enumerate(e):
    #     print("e[%5d] = %15.8e" % (id, val))
    # # remove those eigenvalues that are too small
    # where = np.where(abs(e) > BASIS_CUTOFF)[0]
    # e = e[where]
    # h = h[:, where]
    # print("e.shape = ", e.shape)
    # # self.aux_basis = h @ np.diag(1/e) @ h.T @ B
    # # self.aux_basis = np.asarray(lib.dot(h.T, B), order='C')  # maximal size = naux * ngrids
    
    buffer2 = np.ndarray((e.shape[0] , mydf.aux_basis.shape[1]), dtype=np.double, buffer=mydf.jk_buffer,
             offset=mydf.naux * mydf.naux * mydf.jk_buffer.dtype.itemsize)
    B = np.asarray(lib.ddot(h.T, mydf.aux_basis, c=buffer2), order='C')
    # self.aux_basis = (1.0/e).reshape(-1, 1) * self.aux_basis
    # B = (1.0/e).reshape(-1, 1) * B
    lib.d_i_ij_ij(1.0/e, B, out=B)
    np.asarray(lib.ddot(h, B, c=mydf.aux_basis), order='C')

    if use_mpi:
        comm.Barrier()

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    if debug and mydf.verbose:
        _benchmark_time(t1, t2, "build_auxiliary_basis")

    mydf.naux = naux
    mydf.aoRg = aoRg

from pyscf.pbc import df

class PBC_ISDF_Info(df.fft.FFTDF):

    def __init__(self, mol:Cell, aoR: np.ndarray = None,
                 # cutoff_aoValue: float = 1e-12,
                 # cutoff_QR: float = 1e-8
                 with_robust_fitting=True,
                 Ls=None,
                 get_partition=True,
                 verbose = 1
                 ):

        super().__init__(cell=mol)

        ## the following variables are used in build_sandeep

        self.with_robust_fitting = with_robust_fitting

        self.verbose   = verbose
        self.IP_ID     = None
        self.aux_basis = None
        self.c         = None
        self.naux      = None
        self.W         = None
        self.aoRg      = None
        self.aoR       = aoR
        self.grid_begin = 0
        if aoR is not None:
            self.aoRT  = aoR.T
        else:
            self.aoRT  = None
        self.V_R       = None
        self.cell      = mol
        self.mesh      = mol.mesh

        self.partition = None

        self.natm = mol.natm
        self.nao = mol.nao_nr()

        from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

        df_tmp = MultiGridFFTDF2(mol)
        if aoR is None:
            # df_tmp = MultiGridFFTDF2(mol)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3)
            self.ngrids = self.coords.shape[0]
        else:
            self.ngrids = aoR.shape[1]
            assert self.nao == aoR.shape[0]

        self.grid_end  = self.ngrids

        ## preallocated buffer for parallel calculation

        self.jk_buffer = None
        self.ddot_buf  = None

        ao2atomID = np.zeros(self.nao, dtype=np.int32)
        ao2atomID = np.zeros(self.nao, dtype=np.int32)

        # only valid for spherical GTO

        ao_loc = 0
        for i in range(mol._bas.shape[0]):
            atm_id = mol._bas[i, ATOM_OF]
            nctr   = mol._bas[i, NCTR_OF]
            angl   = mol._bas[i, ANG_OF]
            nao_now = nctr * (2 * angl + 1)  # NOTE: sph basis assumed!
            ao2atomID[ao_loc:ao_loc+nao_now] = atm_id
            ao_loc += nao_now

        # print("ao2atomID = ", ao2atomID)

        self.ao2atomID = ao2atomID
        self.ao2atomID = ao2atomID

        # given aoG, determine at given grid point, which ao has the maximal abs value

        if aoR is not None:
            self.partition = np.argmax(np.abs(aoR), axis=0)
            # print("partition = ", self.partition.shape)
            # map aoID to atomID
            self.partition = np.asarray([ao2atomID[x] for x in self.partition])
            # self.coords    = None
            # self._numints  = None
            grids   = df_tmp.grids
            self.coords  = np.asarray(grids.coords).reshape(-1,3)
            self._numints = df_tmp._numint
        else:
            grids   = df_tmp.grids
            coords  = np.asarray(grids.coords).reshape(-1,3)
            NumInts = df_tmp._numint

            coords_now = coords
            
            if Ls is not None:
                
                mesh = mol.mesh
                meshPrim = np.array(mesh, dtype=np.int32) // Ls
                # print("meshPrim = ", meshPrim)
                coords_now = coords_now.reshape(Ls[0], meshPrim[0], Ls[1], meshPrim[1], Ls[2], meshPrim[2], 3)
                coords_now = coords_now.transpose(0, 2, 4, 1, 3, 5, 6).reshape(-1, 3)
                coords_now = coords_now[:np.prod(meshPrim), :]

            self.partition = np.zeros(coords_now.shape[0], dtype=np.int32)

            from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

            if hasattr(self, "IO_buf"):
                if verbose:
                    print("IO_buf is already allocated")
            else:
                if verbose:
                    print("IO_buf is not allocated")
                MAX_MEMORY = 2 * 1e9  # 2 GB
                self.IO_buf = np.zeros((int(MAX_MEMORY//8),), dtype=np.double)

            if verbose:
                print("IO_buf.size = ", self.IO_buf.size)
                print("coords.shape[0] = ", coords_now.shape[0])
                print("self.nao = ", self.nao)

            bufsize = min(self.IO_buf.size, 4*1e9/8) // 2
            bunchsize = int(bufsize / (self.nao))

            # print("bunchsize = ", bunchsize)

            assert bunchsize > 0
            
            if get_partition and aoR is None:
                for p0, p1 in lib.prange(0, coords_now.shape[0], bunchsize):
                    # print("p0 = %d p1 = %d" % (p0, p1))
                    AoR_Buf = np.ndarray((self.nao, p1-p0), dtype=np.complex128, buffer=self.IO_buf, offset=0)
                    # res = NumInts.eval_ao(mol, coords[p0:p1], deriv=0, out=self.IO_buf[:bufsize])[0].T
                    AoR_Buf = ISDF_eval_gto(self.cell, coords=coords_now[p0:p1], out=AoR_Buf)
                    res = np.argmax(np.abs(AoR_Buf), axis=0)
                    # print("res = ", res)
                    self.partition[p0:p1] = np.asarray([ao2atomID[x] for x in res])
                    AoR_Buf = None
            else:
                self.partition = None
                
            res = None
            
            self.coords = coords
            self._numints = NumInts

    # @profile
    def _allocate_jk_buffer(self, datatype):

        if self.jk_buffer is None:

            nao    = self.nao
            ngrids = self.ngrids
            naux   = self.naux

            print("nao = %d, ngrids = %d, naux = %d" % (nao, ngrids, naux)) 
            buffersize_k = nao * ngrids + naux * ngrids + naux * naux + nao * nao           
            buffersize_j = nao * ngrids + ngrids + nao * naux + naux + naux + nao * nao

            nThreadsOMP   = lib.num_threads()
            size_ddot_buf = max((naux*naux)+2, ngrids) * nThreadsOMP

            if hasattr(self, "IO_buf"):

                if self.IO_buf.size < (max(buffersize_k, buffersize_j) + size_ddot_buf):
                    self.IO_buf = np.zeros((max(buffersize_k, buffersize_j) + size_ddot_buf,), dtype=datatype)

                self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),),
                                            dtype=datatype, buffer=self.IO_buf, offset=0)
                offset         = max(buffersize_k, buffersize_j) * self.jk_buffer.dtype.itemsize
                self.ddot_buf  = np.ndarray((nThreadsOMP, max((nao*nao)+2, ngrids)),
                                            dtype=datatype, buffer=self.IO_buf, offset=offset)

            else:

                self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),), dtype=datatype)
                self.ddot_buf = np.zeros((nThreadsOMP, max((nao*nao)+2, ngrids)), dtype=datatype)


        else:
            assert self.jk_buffer.dtype == datatype
            assert self.ddot_buf.dtype == datatype

    def build(self):
        raise NotImplementedError
        # print("warning: not implemented yet")

    def build_only_partition(self):
        raise NotImplementedError
        # print("warning: not implemented yet")

    def get_buffer_size_in_IP_selection(self, c, m=5):
        natm = self.cell.natm
        nao_per_atm = np.zeros((natm), dtype=np.int32)
        for i in range(self.nao):
            atm_id = self.ao2atomID[i]
            nao_per_atm[atm_id] += 1

        naux_max = 0
        for nao_atm in nao_per_atm:
            naux_max = max(naux_max, int(np.sqrt(c*nao_atm)) + m)

        ngrid_on_atm = np.zeros((self.cell.natm), dtype=np.int32)
        for atm_id in self.partition:
            ngrid_on_atm[atm_id] += 1

        naux_max2 = naux_max * naux_max

        ngrid_on_atm = np.max(ngrid_on_atm)

        nThread = lib.num_threads()

        buf_size  = self.nao*ngrid_on_atm                      # aoR_atm
        buf_size += naux_max2*ngrid_on_atm                     # R
        buf_size += naux_max*ngrid_on_atm*2                    # aoR_atm1, aoR_atm2
        buf_size += naux_max*naux_max*ngrid_on_atm             # aoPairBuffer
        buf_size += (nThread+1)*(ngrid_on_atm+1)
        buf_size += ngrid_on_atm

        return max(buf_size, 2*self.nao*ngrid_on_atm)

    def get_buffer_size_in_global_IP_selection(self, ngrids_possible, c, m=5):

        nao        = self.nao
        naux_max   = int(np.sqrt(c*nao)) + m
        ngrids_now = ngrids_possible
        naux_max2  = naux_max * naux_max

        nThread    = lib.num_threads()

        buf_size   = self.nao*ngrids_now                      # aoR_atm
        buf_size  += naux_max2*ngrids_now                     # R
        buf_size  += naux_max*ngrids_now*2                    # aoR_atm1, aoR_atm2
        buf_size  += naux_max*naux_max*ngrids_now             # aoPairBuffer
        buf_size  += (nThread+1)*(ngrids_now+1)
        buf_size  += ngrids_now

        return max(buf_size, 2*self.nao*ngrids_now)
    
    # @profile


    def get_A_B(self):

        aoR   = self.aoR
        IP_ID = self.IP_ID
        aoRG  = aoR[:, IP_ID]

        A = np.asarray(lib.dot(aoRG.T, aoRG), order='C')
        A = A ** 2
        B = np.asarray(lib.dot(aoRG.T, aoR), order='C')
        B = B ** 2

        return A, B


    def build_IP_Sandeep(self, c=5, m=5,
                         global_IP_selection=True,
                         build_global_basis=True,
                         IP_ID=None,
                         debug=True):

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

        # nao_per_atm = np.zeros(natm, dtype=np.int32)
        # for i in range(self.nao):
        #     atm_id = ao2atomID[i]
        #     nao_per_atm[atm_id] += 1

        # for each atm

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        if IP_ID is None:
            IP_ID = _select_IP_direct(self, c, m, global_IP_selection=global_IP_selection)
            IP_ID.sort()
            IP_ID = np.array(IP_ID, dtype=np.int32)
        self.IP_ID = np.array(IP_ID, dtype=np.int32)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug and rank == 0:
            _benchmark_time(t1, t2, "build_IP")
        t1 = t2

        # build the auxiliary basis

        self.c    = c

        build_aux_basis(self)
        # t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    # @profile
    def build_auxiliary_Coulomb(self, cell:Cell = None, mesh=None, debug=True):

        self._allocate_jk_buffer(datatype=np.double)

        # build the ddot buffer

        naux   = self.naux

        if cell is None:
            cell = self.cell
        if mesh is None:
            mesh = self.cell.mesh

        def constrcuct_V_CCode(aux_basis:np.ndarray, mesh, coul_G):
            coulG_real         = coul_G.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)
            nThread            = lib.num_threads()
            bunchsize          = naux // (2*nThread)
            bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
            bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
            nAux               = aux_basis.shape[0]
            ngrids             = aux_basis.shape[1]
            mesh_int32         = np.array(mesh, dtype=np.int32)

            V                  = np.zeros((nAux, ngrids), dtype=np.double)

            fn = getattr(libpbc, "_construct_V", None)
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
               ctypes.c_int(bunchsize),
               self.jk_buffer.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(bufsize_per_thread))

            return V

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        if cell is None:
            cell = self.cell
            print("cell.__class__ = ", cell.__class__)

        coulG = tools.get_coulG(cell, mesh=mesh)

        V_R = constrcuct_V_CCode(self.aux_basis, mesh, coulG)

        # del task
        # coulG = None

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_V_R")
        t1 = t2

        # W = np.zeros((naux,naux))
        # lib.ddot_withbuffer(a=self.aux_basis, b=V_R.T, buf=self.ddot_buf, c=W, beta=1.0) # allocate, just allocate!
        W = lib.ddot(a=self.aux_basis, b=V_R.T)

        # coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1]
        # if mesh[2] % 2 == 0:
        #     coulG_real[:,:,1:-1] *= 2
        # else:
        #     coulG_real[:,:,1:] *= 2
        # coulG_real = coulG_real.reshape(-1) 
        
        self.coulG = coulG.copy()

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_W")

        self.V_R  = V_R
        self.W    = W
        self.mesh = mesh

    def check_AOPairError(self):
        assert(self.aoR is not None)
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

            for j in range(diff_pair_abs_max.shape[0]):
                print("(%5d, %5d, %15.8e)" % (i, j, diff_pair_abs_max[j]))

    def __del__(self):
        return

    def get_pp(self, kpts=None):
        
        if hasattr(self, "PP") and self.PP is not None:
            return self.PP
        else:
            t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
            
            df_tmp = multigrid.MultiGridFFTDF2(self.cell)
            v_pp_loc2_nl = df_tmp.get_pp(max_memory=self.cell.max_memory)
            v_pp_loc1_G = df_tmp.vpplocG_part1
            v_pp_loc1 = multigrid.multigrid_pair._get_j_pass2(df_tmp, v_pp_loc1_G)
            self.PP = (v_pp_loc1 + v_pp_loc2_nl)[0]
            
            t1 = (lib.logger.process_clock(), lib.logger.perf_counter()) 
            
            if self.verbose:
                _benchmark_time(t0, t1, "get_pp")
            return self.PP
    
    ##### functions defined in isdf_ao2mo.py #####

    get_eri = get_ao_eri = isdf_ao2mo.get_eri
    ao2mo = get_mo_eri = isdf_ao2mo.general
    ao2mo_7d = isdf_ao2mo.ao2mo_7d  # seems to be only called in kadc and kccsd, NOT implemented!

    ##### functions defined in isdf_jk.py #####

    get_jk = isdf_jk.get_jk_dm

C = 15

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

#     boxlen = 4.2
#     cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
#     cell.atom = '''
# Li 0.0   0.0   0.0
# Li 2.1   2.1   0.0
# Li 0.0   2.1   2.1
# Li 2.1   0.0   2.1
# H  0.0   0.0   2.1
# H  0.0   2.1   0.0
# H  2.1   0.0   0.0
# H  2.1   2.1   2.1
# '''

    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 128   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 70
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 1])

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx = grids.mesh[0]

    # for i in range(coords.shape[0]):
    #     print(coords[i])
    # exit(1)

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR)
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=False)
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)
    # pbc_isdf_info.check_AOPairError()

    # exit(1)

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
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()

    exit(1)

    # without robust fitting 
    
    pbc_isdf_info.with_robust_fitting = False

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7
    mf.kernel()

    mf = scf.RHF(cell)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    # mf.kernel()
