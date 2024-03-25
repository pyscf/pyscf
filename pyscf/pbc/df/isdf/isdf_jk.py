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
from pyscf.lib import logger, zdotNN, zdotCN, zdotNC
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member

from mpi4py import MPI
import numpy
INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1

from memory_profiler import profile
import ctypes

libpbc = lib.load_library('libpbc')

##################################################
#
# only Gamma Point
#
##################################################

def _benchmark_time(t1, t2, label):
    print("%20s wall time: %12.6f CPU time: %12.6f" % (label, t2[1] - t1[1], t2[0] - t1[0]))

def _contract_j_mo(mydf, mo_coeffs):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    nao  = mo_coeffs.shape[0]
    # nocc = mo_coeffs.shape[1]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    V_R  = mydf.V_R
    # naux = aoRg.shape[1]

    #### step 1. get density value on real space grid and IPs

    #### TODO: make the following transformation linear-scaling

    moRg = np.asarray(lib.dot(mo_coeffs.T, aoRg), order='C')
    moR  = np.asarray(lib.dot(mo_coeffs.T, aoR), order='C')

    #### step 2. get J term1 and term2

    density_R  = np.sum(moR*moR, axis=0)
    density_Rg = np.sum(moRg*moRg, axis=0)

    ## TODO: remove the redundancy due to the symmetry

    rho_mu_nu_Rg = np.einsum('ij,kj->ikj', aoRg, aoRg)

    # J = np.asarray(lib.dot(V_R, density_R), order='C')
    J = np.dot(V_R, density_R)
    # J = np.asarray(lib.dot(rho_mu_nu_Rg, J), order='C')
    J = np.dot(rho_mu_nu_Rg, J)

    J2 = np.dot(V_R.T, density_Rg)
    J2 = np.einsum('ij,j->ij', aoR, J2)
    J += np.asarray(lib.dot(aoR, J2.T), order='C')

    #### step 3. get J term3

    # tmp = np.asarray(lib.dot(W, density_Rg), order='C')
    tmp = np.dot(W, density_Rg)
    # J -= np.asarray(lib.dot(rho_mu_nu_Rg, tmp), order='C')
    J -= np.dot(rho_mu_nu_Rg, tmp)
    # J = np.dot(rho_mu_nu_Rg, tmp)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_j_mo")

    return J * 2.0 * ngrid / vol  # 2.0 due to RHF

def _contract_k_mo(mydf, mo_coeffs):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    nao  = mo_coeffs.shape[0]
    nocc = mo_coeffs.shape[1]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    V_R  = mydf.V_R
    naux = aoRg.shape[1]

    #### step 1. get density value on real space grid and IPs

    moRg = np.asarray(lib.dot(mo_coeffs.T, aoRg), order='C')
    moR  = np.asarray(lib.dot(mo_coeffs.T, aoR), order='C')

    #### step 2. get K term1 and term2

    density_RgR = np.asarray(lib.dot(moRg.T, moR), order='C')
    tmp = V_R * density_RgR  # pointwise multiplication

    K = np.asarray(lib.dot(tmp, aoR.T), order='C')
    K = np.asarray(lib.dot(aoRg, K), order='C')  ### the order due to the fact that naux << ngrid
    K += K.T

    #### step 3. get K term3

    density_RgRg = np.asarray(lib.dot(moRg.T, moRg), order='C')
    tmp = W * density_RgRg  # pointwise multiplication
    tmp = np.asarray(lib.dot(tmp, aoRg.T), order='C')
    K -= np.asarray(lib.dot(aoRg, tmp), order='C')
    # K = np.asarray(lib.dot(aoRg, tmp), order='C')

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_k_mo")

    return K * 2.0 * ngrid / vol  # 2.0 due to RHF

def get_jk_mo(mydf, occ_mo_coeff, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''

    vj = vk = None

    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        raise NotImplementedError("ISDF does not support kpts_band != kpt")

    # cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not np.iscomplexobj(occ_mo_coeff)

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)

    if with_j:
        vj = _contract_j_mo(mydf, occ_mo_coeff)
    if with_k:
        vk = _contract_k_mo(mydf, occ_mo_coeff)
        if exxdiv == 'ewald':
            raise NotImplementedError("ISDF does not support ewald")


    t1 = log.timer('sr jk', *t1)
    return vj, vk

def _comm_bunch(size_of_comm, comm_size, force_even=False):
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
        import sys 
        import traceback
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        sys.stderr.write(''.join(traceback.format_stack()[:-1]))
        comm.Abort()
        
def mpi_reduce(sendbuf, comm, rank, op=MPI.SUM, root=0):
    
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
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

def mpi_bcast(buf, comm, rank, root=0):
    # from mpi4py import MPI
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)

    dtype = buf.dtype.char
    buf_seg = numpy.ndarray(buf.size, dtype=buf.dtype, buffer=buf)
    for p0, p1 in lib.prange(0, buf.size, BLKSIZE):
        comm.Bcast([buf_seg[p0:p1], dtype], root)
    return buf

def _segment_counts(counts, p0, p1):
    counts_seg = counts - p0
    counts_seg[counts<=p0] = 0
    counts_seg[counts> p1] = p1 - p0
    return counts_seg

def prange(comm, start, stop, step):
    '''Similar to lib.prange. This function ensures that all processes have the
    same number of steps.  It is required by alltoall communication.
    '''
    nsteps = (stop - start + step - 1) // step
    nsteps = max(comm.allgather(nsteps))
    for i in range(nsteps):
        i0 = min(stop, start + i * step)
        i1 = min(stop, i0 + step)
        yield i0, i1

def mpi_gather(sendbuf, comm, rank, root=0, split_recvbuf=False):

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

def mpi_scatter(sendbuf, comm, rank, root=0):
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

# @profile
def _contract_j_dm(mydf, dm, with_robust_fitting=True, use_mpi=False):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # print("rank = ", rank)
        # print("size = ", size)
        
    t1 = (logger.process_clock(), logger.perf_counter())

    # print("t1 = ", t1)  

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao  = dm.shape[0]

    cell = mydf.cell
    # print("cell.nao", cell.nao)
    # print("nao     ", nao)
    assert cell.nao == nao
    # ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    ngrid = aoR.shape[1]

    if hasattr(mydf, "V_R"):
        V_R  = mydf.V_R
    else:
        V_R = None
    naux = aoRg.shape[1]
    IP_ID = mydf.IP_ID
    
    #### step 2. get J term1 and term2

    # buffersize = nao * ngrid + ngrid + nao * naux + naux + naux + nao * nao
    # buffer = np.empty(buffersize, dtype=dm.dtype)
    
    buffer = mydf.jk_buffer
    buffer1 = np.ndarray((nao,ngrid), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer2 = np.ndarray((ngrid), dtype=dm.dtype, buffer=buffer, offset=nao * ngrid * dm.dtype.itemsize)
    buffer3 = np.ndarray((nao,naux), dtype=dm.dtype, buffer=buffer,
                         offset=(nao * ngrid + ngrid) * dm.dtype.itemsize)
    buffer4 = np.ndarray((naux), dtype=dm.dtype, buffer=buffer, offset=(nao *
                         ngrid + ngrid + nao * naux) * dm.dtype.itemsize)
    buffer5 = np.ndarray((naux), dtype=dm.dtype, buffer=buffer, offset=(nao *
                            ngrid + ngrid + nao * naux + naux) * dm.dtype.itemsize)
    buffer6 = np.ndarray((nao,nao), dtype=dm.dtype, buffer=buffer, offset=(nao *
                            ngrid + ngrid + nao * naux + naux + naux) * dm.dtype.itemsize)
    buffer7 = np.ndarray((nao,naux), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer8 = np.ndarray((naux), dtype=dm.dtype, buffer=buffer, offset=nao * ngrid * dm.dtype.itemsize)

    # print("address of mydf.jk_buffer = ", id(mydf.jk_buffer))
    # print("address of buffer         = ", id(buffer))
    # print("address of buffer1        = ", id(buffer1))
    # import sys
    # print("size    of buffer1        = ", sys.getsizeof(buffer1))
    # print(buffer.__array_interface__['data'][0])
    # print(buffer1.__array_interface__['data'][0])

    # print(buffer2.__array_interface__['data'][0])
    # print(buffer3.__array_interface__['data'][0])
    # print(buffer4.__array_interface__['data'][0])
    # print(buffer5.__array_interface__['data'][0])
    # print(buffer6.__array_interface__['data'][0])
    # print(buffer7.__array_interface__['data'][0])
    # print(buffer8.__array_interface__['data'][0])
    # print("begin_work")

    # ptr1 = buffer1.__array_interface__['data'][0]
    # ptr2 = buffer2.__array_interface__['data'][0]
    # ptr3 = buffer3.__array_interface__['data'][0]
    # ptr4 = buffer4.__array_interface__['data'][0]
    # ptr5 = buffer5.__array_interface__['data'][0]
    # ptr6 = buffer6.__array_interface__['data'][0]
    # ptr7 = buffer7.__array_interface__['data'][0]
    # ptr8 = buffer8.__array_interface__['data'][0]

    ## constract dm and aoR

    # need allocate memory, size = nao  * ngrid, (buffer 1)

    # tmp1 = np.asarray(lib.dot(dm, aoR, c=buffer1), order='C')
    # print('dm.flags: %s' % str(dm.flags))
    # print('aoR.flags: %s' % str(aoR.flags))
    # print("before calling ddot in _contract_j_dm")
    # print("dm.shape", dm.shape)
    # print("aoR.shape", aoR.shape)

    lib.ddot(dm, aoR, c=buffer1)  
    # print("after calling ddot in _contract_j_dm")
    tmp1 = buffer1
    # print(buffer1.__array_interface__['data'][0])
    # print(tmp1.__array_interface__['data'][0])
    # print("address of aoR      = ", aoR.__array_interface__['data'][0])

    # assert tmp1.__array_interface__['data'][0] == ptr1

    # need allocate memory, size = ngrid, (buffer 2)

    density_R = np.asarray(lib.multiply_sum_isdf(aoR, tmp1, out=buffer2), order='C')

    # if use_mpi and rank == 0:
    #    print("density_R = ", density_R[:16])

    # print("D1 = ", density_R)

    # assert density_R.__array_interface__['data'][0] == ptr2

    # need allocate memory, size = nao  * naux, (buffer 3)

    # lib.dslice(tmp1, IP_ID, buffer3)
    # tmp1 = buffer3
    
    tmp1 = lib.ddot(dm, aoRg)  

    # assert tmp1.__array_interface__['data'][0] == ptr3

    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg, tmp1, out=buffer4),
                            order='C')  # need allocate memory, size = naux, (buffer 4)

    # print("D3 = ", density_Rg[:16])

    # assert density_Rg.__array_interface__['data'][0] == ptr4

    # This should be the leading term of the computation cost in a single-thread mode.

    # need allocate memory, size = naux, (buffer 5)

    J = None

    if with_robust_fitting:
        J = np.asarray(lib.ddot_withbuffer(V_R, density_R.reshape(-1,1), c=buffer5.reshape(-1,1), buf=mydf.ddot_buf), order='C').reshape(-1)   # with buffer, size 
        # assert J.__array_interface__['data'][0] == ptr5

        # print("J = ", J[:16])

        # if use_mpi:
        # J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
            # J = comm.reduce(J, op=MPI.SUM, root=0)
        
        # do not need allocate memory, use buffer 3

        # J = np.einsum('ij,j->ij', aoRg, J)
        
        # if (use_mpi and rank == 0) or use_mpi == False:
        J = np.asarray(lib.d_ij_j_ij(aoRg, J, out=buffer3), order='C')

            # assert J.__array_interface__['data'][0] == ptr3

            # need allocate memory, size = nao  * nao, (buffer 6)

        J = np.asarray(lib.ddot_withbuffer(aoRg, J.T, c=buffer6, buf=mydf.ddot_buf), order='C')
            
            # assert J.__array_interface__['data'][0] == ptr6
        # else:
        #     J = np.zeros((nao, nao))

        # do not need allocate memory, use buffer 2

        J2 = np.asarray(lib.dot(V_R.T, density_Rg.reshape(-1,1), c=buffer2.reshape(-1,1)), order='C').reshape(-1)
        # assert J2.__array_interface__['data'][0] == ptr2

        # do not need allocate memory, use buffer 1

        # J2 = np.einsum('ij,j->ij', aoR, J2)
        J2 = np.asarray(lib.d_ij_j_ij(aoR, J2, out=buffer1), order='C')
        # assert J2.__array_interface__['data'][0] == ptr1

        # do not need allocate memory, use buffer 6

        # J += np.asarray(lib.dot(aoR, J2.T), order='C')
        lib.ddot_withbuffer(aoR, J2.T, c=J, beta=1, buf=mydf.ddot_buf)
        # assert J.__array_interface__['data'][0] == ptr6

        # print("J = ", J[0,:])

    # if use_mpi and J is not None:
        # J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
        # J = comm.reduce(J, op=MPI.SUM, root=0)

    # if (use_mpi and rank == 0) or use_mpi == False:
    #     print("J = ", J[0,-10:] * ngrid / vol)

    # return J * ngrid / vol

    #### step 3. get J term3

    # do not need allocate memory, use buffer 2

    # if (use_mpi and rank == 0) or use_mpi == False:

    tmp = np.asarray(lib.dot(W, density_Rg.reshape(-1,1), c=buffer8.reshape(-1,1)), order='C').reshape(-1)
    # assert tmp.__array_interface__['data'][0] == ptr8
    # do not need allocate memory, use buffer 1 but viewed as buffer 7
    # tmp = np.einsum('ij,j->ij', aoRg, tmp)
    tmp = np.asarray(lib.d_ij_j_ij(aoRg, tmp, out=buffer7), order='C')
    # assert tmp.__array_interface__['data'][0] == ptr7
    # do not need allocate memory, use buffer 6
    # J -= np.asarray(lib.dot(aoRg, tmp.T), order='C')
    if with_robust_fitting:
        # print("with robust fitting")
        lib.ddot_withbuffer(aoRg, -tmp.T, c=J, beta=1, buf=mydf.ddot_buf)
    else:
        # print("without robust fitting")
        J = buffer6
        lib.ddot_withbuffer(aoRg, tmp.T, c=J, beta=0, buf=mydf.ddot_buf)
    # assert J.__array_interface__['data'][0] == ptr6

    if use_mpi:
        J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
        # J = mpi_bcast(J, comm, rank, root=0)

    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm")

    # print("J = ", J[0,:])   

    # if use_mpi:
    #     comm.Barrier()

    return J * ngrid / vol

def _contract_j_dm_fast(mydf, dm, with_robust_fitting=True, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    
    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    nao  = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    assert ngrid == mydf.ngrids
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    ngrid = aoR.shape[1]
    if hasattr(mydf, "V_R"):
        V_R = mydf.V_R
    else:
        V_R = None
    # naux = aoRg.shape[1] 
    naux = mydf.naux
    IP_ID = mydf.IP_ID
    
    mesh = np.array(cell.mesh, dtype=np.int32)
    
    #### step 0. allocate buffer 
    
    buffer = mydf.jk_buffer
    buffer1 = np.ndarray((nao,ngrid), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer2 = np.ndarray((ngrid), dtype=dm.dtype, buffer=buffer, offset=nao * ngrid * dm.dtype.itemsize)
    buffer3 = np.ndarray((nao,naux), dtype=dm.dtype, buffer=buffer,
                         offset=(nao * ngrid + ngrid) * dm.dtype.itemsize)
    buffer4 = np.ndarray((naux), dtype=dm.dtype, buffer=buffer, offset=(nao *
                         ngrid + ngrid + nao * naux) * dm.dtype.itemsize)
    buffer5 = np.ndarray((naux), dtype=dm.dtype, buffer=buffer, offset=(nao *
                            ngrid + ngrid + nao * naux + naux) * dm.dtype.itemsize)
    buffer6 = np.ndarray((nao,nao), dtype=dm.dtype, buffer=buffer, offset=(nao *
                            ngrid + ngrid + nao * naux + naux + naux) * dm.dtype.itemsize)
    buffer7 = np.ndarray((nao,naux), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer8 = np.ndarray((naux), dtype=dm.dtype, buffer=buffer, offset=nao * ngrid * dm.dtype.itemsize)

    #### step 1. get density value on real space grid and IPs
    
    lib.ddot(dm, aoR, c=buffer1) 
    tmp1 = buffer1
    density_R = np.asarray(lib.multiply_sum_isdf(aoR, tmp1, out=buffer2), order='C')
    
    if use_mpi:
        density_R = np.hstack(mpi_gather(density_R, comm, rank, root=0, split_recvbuf=True))
    
    # if (use_mpi and rank == 0) or (use_mpi == False):
    #     print("density_R = ", density_R[:16])
    #     print("density_R = ", density_R[-16:])
    #     print("density_R.shape = ", density_R.shape)
    
    if hasattr(mydf, "grid_ID_ordered"):
        if (use_mpi and rank == 0) or (use_mpi == False):
            density_R_original = np.zeros_like(density_R)
            
            fn_order = getattr(libpbc, "_Reorder_Grid_to_Original_Grid", None)
            assert fn_order is not None
            
            fn_order(
                ctypes.c_int(density_R.size),
                mydf.grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
                density_R.ctypes.data_as(ctypes.c_void_p),
                density_R_original.ctypes.data_as(ctypes.c_void_p),
            )

            density_R = density_R_original.copy()
    
    J = None
    
    if (use_mpi and rank == 0) or (use_mpi == False):
    
        fn_J = getattr(libpbc, "_construct_J", None)
        assert(fn_J is not None)

        J = np.zeros_like(density_R)

        fn_J(
            mesh.ctypes.data_as(ctypes.c_void_p),
            density_R.ctypes.data_as(ctypes.c_void_p),
            mydf.coulG.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
        )
        
        if hasattr(mydf, "grid_ID_ordered"):
            
            J_ordered = np.zeros_like(J)

            fn_order = getattr(libpbc, "_Original_Grid_to_Reorder_Grid", None)
            assert fn_order is not None 
            
            fn_order(
                ctypes.c_int(J.size),
                mydf.grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
                J.ctypes.data_as(ctypes.c_void_p),
                J_ordered.ctypes.data_as(ctypes.c_void_p),
            )
            
            J = J_ordered.copy()
        
        # print("J = ", J[:16])   
    
    if use_mpi:
        
        if hasattr(mydf, "grid_segment"):
            grid_segment = mydf.grid_segment
            sendbuf = None
            if rank == 0:
                sendbuf = []
                for i in range(size):
                    p0 = grid_segment[i]
                    p1 = grid_segment[i+1]
                    sendbuf.append(J[p0:p1])
            J = mpi_scatter(sendbuf, comm, rank, root=0)
            del sendbuf
            sendbuf = None
        else:
            ngrid_global = np.prod(cell.mesh)
            comm_bunch = _comm_bunch(ngrid_global, size)
            sendbuf = None
            if rank == 0:
                sendbuf = []
                for i in range(size):
                    p0 = min(i * comm_bunch, ngrid_global)
                    p1 = min((i + 1) * comm_bunch, ngrid_global)
                    sendbuf.append(J[p0:p1])
            J = mpi_scatter(sendbuf, comm, rank, root=0)
            del sendbuf
            sendbuf = None
    
    #### step 3. get J 
    
    # if use_mpi:
    #     print("rank = ", rank, " J_shape = ", J.shape, " aoR.shape = ", aoR.shape)
    
    J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    if use_mpi:
        J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_fast")
    
    return J * ngrid / vol

def _contract_j_dm_wo_robust_fitting(mydf, dm, with_robust_fitting=False, use_mpi=False):
    
    assert with_robust_fitting == False
    
    if use_mpi:
        raise NotImplementedError("MPI is not supported in this function")

    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao  = dm.shape[0]
    
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(cell.mesh)

    W    = mydf.W
    aoRg = mydf.aoRg
    
    naux = aoRg.shape[1]
    # IP_ID = mydf.IP_ID
    
    tmp1 = lib.ddot(dm, aoRg)  
    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg, tmp1),
                            order='C') 
    tmp = np.asarray(lib.dot(W, density_Rg.reshape(-1,1)), order='C').reshape(-1)
    tmp = np.asarray(lib.d_ij_j_ij(aoRg, tmp), order='C')

    J = lib.ddot_withbuffer(aoRg, tmp.T, buf=mydf.ddot_buf)

    del tmp1 
    tmp1 = None
    del tmp 
    tmp = None
    del density_Rg
    density_Rg = None

    t2 = (logger.process_clock(), logger.perf_counter())

    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_wo_robust_fitting")
    
    return J * ngrid / vol

# @profile
def _contract_k_dm(mydf, dm, with_robust_fitting=True, use_mpi=False):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # dm = mpi_bcast(dm, comm, rank, root=0)

    # return numpy.zeros_like(dm)

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    nao  = dm.shape[0]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    assert ngrid == mydf.ngrids
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    ngrid = aoR.shape[1]
    if hasattr(mydf, "V_R"):
        V_R = mydf.V_R
    else:
        V_R = None
    # naux = aoRg.shape[1]
    naux = mydf.naux
    IP_ID = mydf.IP_ID

    buffer = mydf.jk_buffer
    buffer1 = np.ndarray((nao,ngrid), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer2 = np.ndarray((naux,ngrid), dtype=dm.dtype, buffer=buffer, offset=nao * ngrid * dm.dtype.itemsize)
    buffer3 = np.ndarray((naux,naux), dtype=dm.dtype, buffer=buffer,
                         offset=(nao * ngrid + naux * ngrid) * dm.dtype.itemsize)
    buffer4 = np.ndarray((nao,nao), dtype=dm.dtype, buffer=buffer, offset=(nao *
                         ngrid + naux * ngrid + naux * naux) * dm.dtype.itemsize)
    buffer5 = np.ndarray((naux,nao), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer6 = np.ndarray((naux,nao), dtype=dm.dtype, buffer=buffer, offset=nao * ngrid * dm.dtype.itemsize)

    # print("address of mydf.jk_buffer = ", id(mydf.jk_buffer))
    # print("address of buffer         = ", id(buffer))
    # print("address of buffer1        = ", id(buffer1))

    # ptr1 = buffer1.__array_interface__['data'][0]

    #### step 1. get density value on real space grid and IPs

    # need allocate memory, size = nao  * ngrid, this buffer does not need anymore  (buffer 1)

    density_RgR  = np.asarray(lib.dot(dm, aoR, c=buffer1), order='C')
    
    # print("buffer1.size", buffer1.size)
    # assert density_RgR.__array_interface__['data'][0] == ptr1

    # need allocate memory, size = naux * ngrid                                     (buffer 2)

    # density_RgR  = np.asarray(lib.dot(aoRg.T, density_RgR, c=buffer2), order='C')
    lib.ddot(aoRg.T, density_RgR, c=buffer2)
    density_RgR = buffer2
    # assert density_RgR.__array_interface__['data'] == buffer2.__array_interface__['data']

    # need allocate memory, size = naux * naux                                      (buffer 3)

    # density_RgRg = density_RgR[:, IP_ID]
    # lib.dslice(density_RgR, IP_ID, buffer3)
    # density_RgRg = buffer3

    density_RgRg = lib.ddot(dm, aoRg)
    density_RgRg = lib.ddot(aoRg.T, density_RgRg)

    # assert density_RgRg.__array_interface__['data'] == buffer3.__array_interface__['data']

    #### step 2. get K term1 and term2

    ### todo: optimize the following 4 lines, it seems that they may not parallize!

    # tmp = V_R * density_RgR  # pointwise multiplication, TODO: this term should be parallized
    # do not need allocate memory, size = naux * ngrid, (buffer 2)

    # tmp = np.asarray(lib.cwise_mul(V_R, density_RgR, out=buffer2), order='C')

    # lib.cwise_mul(V_R, density_RgR, out=buffer2)

    K = None

    if with_robust_fitting:
        lib.cwise_mul(V_R, density_RgR, out=buffer2)
        tmp = buffer2

        # assert tmp.__array_interface__['data'] == buffer2.__array_interface__['data']

        # do not need allocate memory, size = naux * nao,   (buffer 1, but viewed as buffer5)
    
        K = np.asarray(lib.ddot_withbuffer(tmp, aoR.T, c=buffer5, buf=mydf.ddot_buf), order='C')

        # assert K.__array_interface__['data'] == buffer5.__array_interface__['data']

        ### the order due to the fact that naux << ngrid  # need allocate memory, size = nao * nao,           (buffer 4)

        K  = np.asarray(lib.ddot_withbuffer(aoRg, K, c=buffer4, buf=mydf.ddot_buf), order='C')
        
        # if use_mpi:
        # K2 = mpi_reduce(K, comm, rank, op=MPI.SUM, root=0)
        # K = comm.reduce(K, op=MPI.SUM, root=0)
        # K = K2

        # if (use_mpi and rank == 0) or use_mpi == False:
        K += K.T
            
        # print("K = ", K[0,:])

    # assert K.__array_interface__['data'] == buffer4.__array_interface__['data']

    #### step 3. get K term3

    ### todo: optimize the following 4 lines, it seems that they may not parallize!
    # pointwise multiplication, do not need allocate memory, size = naux * naux, use buffer for (buffer 3)
    # tmp = W * density_RgRg

    # print("D5 = ", density_RgRg[0,:])
    
    # if (use_mpi and rank == 0) or use_mpi == False:
    
    lib.cwise_mul(W, density_RgRg, out=density_RgRg)
    tmp = density_RgRg

        # assert tmp.__array_interface__['data'] == buffer3.__array_interface__['data']

        # do not need allocate memory, size = naux * nao, use buffer 2 but viewed as buffer 6
    tmp = np.asarray(lib.dot(tmp, aoRg.T, c=buffer6), order='C')

        # assert tmp.__array_interface__['data'] == buffer6.__array_interface__['data']

        # K  -= np.asarray(lib.dot(aoRg, tmp, c=K, beta=1), order='C')     # do not need allocate memory, size = nao * nao, (buffer 4)
    
    if with_robust_fitting:
        # print("with robust fitting")
        lib.ddot_withbuffer(aoRg, -tmp, c=K, beta=1, buf=mydf.ddot_buf)
    else:
        # print("without robust fitting")
        K = buffer4
        lib.ddot_withbuffer(aoRg, tmp, c=K, beta=0, buf=mydf.ddot_buf)

    if use_mpi:
        # K = mpi_bcast(K, comm, rank, root=0)
        K = mpi_reduce(K, comm, rank, op=MPI.SUM, root=0)

    # assert K.__array_interface__['data'] == buffer4.__array_interface__['data']

    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm")

    # if use_mpi:
    #     comm.Barrier()

    if K is None:
        K = np.zeros((nao, nao))

    return K * ngrid / vol

def _contract_k_dm_wo_robust_fitting(mydf, dm, with_robust_fitting=False, use_mpi=False):
    
    assert with_robust_fitting == False
    
    if use_mpi:
        raise NotImplementedError("MPI is not supported in this function")

    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao  = dm.shape[0]
    
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(cell.mesh)

    W    = mydf.W
    aoRg = mydf.aoRg
    
    naux = aoRg.shape[1]
    
    density_RgRg = lib.ddot(dm, aoRg)
    density_RgRg = lib.ddot(aoRg.T, density_RgRg)
    
    lib.cwise_mul(W, density_RgRg, out=density_RgRg)
    tmp = density_RgRg
    tmp = np.asarray(lib.dot(tmp, aoRg.T), order='C')
    K = lib.ddot_withbuffer(aoRg, tmp, buf=mydf.ddot_buf)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm_wo_robust_fitting")
    
    del tmp
    tmp = None
    del density_RgRg
    density_RgRg = None
    
    return K * ngrid / vol

def get_jk_dm(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, 
           use_mpi = False, **kwargs):
    '''JK for given k-point'''
    
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
            vj = _contract_j_dm_fast(mydf, dm, mydf.with_robust_fitting, use_mpi)
        else:
            # vj = _contract_j_dm(mydf, dm, mydf.with_robust_fitting, use_mpi)
            vj = _contract_j_dm_wo_robust_fitting(mydf, dm, mydf.with_robust_fitting, use_mpi)
        # print("vj2 = ", vj2[0, :16])
        # print("vj  = ", vj[0, :16])
        # print("vj/vj2 = ", vj[0, :16] / vj2[0, :16])    
    if with_k:
        if mydf.with_robust_fitting:
            vk = _contract_k_dm(mydf, dm, mydf.with_robust_fitting, use_mpi)
        else:
            vk = _contract_k_dm_wo_robust_fitting(mydf, dm, mydf.with_robust_fitting, use_mpi)
        # print("vk = ", vk[0,:16])
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    return vj, vk