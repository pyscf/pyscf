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
# Author: Oliver Backhouse <olbackhouse@gmail.com>
#         George Booth <george.booth@kcl.ac.uk>
#

'''
MPI helper functions using mpi4py
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1

# attempt to successfully load and init the MPI, else assume 1 core:
try:
    from mpi4py import MPI as mpi
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except Exception:
    mpi = None
    comm = None
    size = 1
    rank = 0

SCALE_PRANGE_STEP = False


def bcast(buf, root=0):
    if size == 1:
        return buf

    is_array = isinstance(buf, np.ndarray)
    buf = np.asarray(buf, order='C')
    buf = buf.astype(buf.dtype.char)
    shape, mpi_dtype = comm.bcast((buf.shape, buf.dtype.char))

    if rank != root:
        buf = np.empty(shape, dtype=mpi_dtype)

    buf_seg = np.ndarray(buf.size, dtype=buf.dtype, buffer=buf)
    for p0, p1 in lib.prange(0, buf.size, BLKSIZE):
        comm.Bcast(buf_seg[p0:p1], root)

    return buf if is_array else buf.ravel()[0]


def bcast_dict(buf, root=0):
    if size == 1:
        return buf

    buf = comm.bcast(buf, root)

    return buf


def reduce(sendbuf, root=0, op=getattr(mpi, 'SUM', None)):
    if size == 1:
        return sendbuf

    is_array = isinstance(sendbuf, np.ndarray)
    sendbuf = np.asarray(sendbuf, order='C')
    sendbuf = sendbuf.astype(sendbuf.dtype.char)
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    assert sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype

    recvbuf = np.zeros_like(sendbuf)
    send_seg = np.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = np.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Reduce(send_seg[p0:p1], recv_seg[p0:p1], op, root)

    if rank == root:
        return recvbuf if is_array else recvbuf.ravel()[0]
    else:
        return sendbuf if is_array else sendbuf.ravel()[0]


def allreduce(sendbuf, root=0, op=getattr(mpi, 'SUM', None)):
    if size == 1:
        return sendbuf

    is_array = isinstance(sendbuf, np.ndarray)
    sendbuf = np.asarray(sendbuf, order='C')
    sendbuf = sendbuf.astype(sendbuf.dtype.char)
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    assert sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype

    recvbuf = np.zeros_like(sendbuf)
    send_seg = np.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = np.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Allreduce(send_seg[p0:p1], recv_seg[p0:p1], op)

    return recvbuf if is_array else recvbuf.ravel()[0]


def allreduce_safe_inplace(array):
    if size == 1:
        return array

    from pyscf.pbc.mpitools.mpi_helper import safeAllreduceInPlace

    safeAllreduceInPlace(comm, array)


def barrier():
    if comm is not None:
        comm.Barrier()


def nrange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start

    yield from range(start+rank, stop, step*size)


def prange(start, stop, step):
    ''' :func:`lib.prange` distributed over MPI processes. Returns
        the range for a single MPI rank.
    '''

    if size == 1:
        for p0, p1 in lib.prange(start, stop, step):
            yield p0, p1
    else:
        if SCALE_PRANGE_STEP:
            step //= size

        split = lambda x : x * (stop-start) // size

        start0 = split(rank)
        stop0 = stop if rank == (size-1) else split(rank+1)

        for p0, p1 in lib.prange(start0, stop0, step):
            yield p0, p1
