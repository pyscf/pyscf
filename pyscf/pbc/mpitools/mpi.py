#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import sys
import time
import threading
import traceback
import numpy
from mpi4py import MPI
from . import mpi_pool
from .mpi_pool import MPIPool

_registry = {}

if 'pool' not in _registry:
    import atexit
    pool = MPIPool(debug=False)
    _registry['pool'] = pool
    atexit.register(pool.close)

comm = pool.comm
rank = pool.rank

def static_partition(tasks):
    size = len(tasks)
    segsize = (size+pool.size-1) // pool.size
    start = pool.rank * segsize
    stop = min(size, start+segsize)
    return tasks[start:stop]

def work_balanced_partition(tasks, costs=None):
    if costs is None:
        costs = numpy.ones(tasks)
    if rank == 0:
        segsize = float(sum(costs)) / pool.size
        loads = []
        cum_costs = numpy.cumsum(costs)
        start_id = 0
        for k in range(pool.size):
            stop_id = numpy.argmin(abs(cum_costs - (k+1)*segsize)) + 1
            stop_id = max(stop_id, start_id+1)
            loads.append([start_id,stop_id])
            start_id = stop_id
        comm.bcast(loads)
    else:
        loads = comm.bcast()
    if rank < len(loads):
        start, stop = loads[rank]
        return tasks[start:stop]
    else:
        return tasks[:0]

INQUIRY = 50050
TASK = 50051
def work_share_partition(tasks, interval=.02, loadmin=1):
    loadmin = max(loadmin, len(tasks)//50//pool.size)
    rest_tasks = [x for x in tasks[loadmin*pool.size:]]
    tasks = tasks[loadmin*rank:loadmin*rank+loadmin]
    def distribute_task():
        while True:
            load = len(tasks)
            if rank == 0:
                for i in range(pool.size):
                    if i != 0:
                        load = comm.recv(source=i, tag=INQUIRY)
                    if rest_tasks:
                        if load <= loadmin:
                            task = rest_tasks.pop(0)
                            comm.send(task, i, tag=TASK)
                    else:
                        comm.send('OUT_OF_TASK', i, tag=TASK)
            else:
                comm.send(load, 0, tag=INQUIRY)
            if comm.Iprobe(source=0, tag=TASK):
                tasks.append(comm.recv(source=0, tag=TASK))
                if isinstance(tasks[-1], str) and tasks[-1] == 'OUT_OF_TASK':
                    return
            time.sleep(interval)

    tasks_handler = threading.Thread(target=distribute_task)
    tasks_handler.start()

    while True:
        if tasks:
            task = tasks.pop(0)
            if isinstance(task, str) and task == 'OUT_OF_TASK':
                tasks_handler.join()
                return
            yield task

def work_stealing_partition(tasks, interval=.0001):
    tasks = static_partition(tasks)
    out_of_task = [False]
    def task_daemon():
        while True:
            time.sleep(interval)
            while comm.Iprobe(source=MPI.ANY_SOURCE, tag=INQUIRY):
                src, req = comm.recv(source=MPI.ANY_SOURCE, tag=INQUIRY)
                if isinstance(req, str) and req == 'STOP_DAEMON':
                    return
                elif tasks:
                    comm.send(tasks.pop(), src, tag=TASK)
                elif src == 0 and isinstance(req, str) and req == 'ALL_DONE':
                    comm.send(out_of_task[0], src, tag=TASK)
                elif out_of_task[0]:
                    comm.send('OUT_OF_TASK', src, tag=TASK)
                else:
                    comm.send('BYPASS', src, tag=TASK)
    def prepare_to_stop():
        out_of_task[0] = True
        if rank == 0:
            while True:
                done = []
                for i in range(1, pool.size):
                    comm.send((0,'ALL_DONE'), i, tag=INQUIRY)
                    done.append(comm.recv(source=i, tag=TASK))
                if all(done):
                    break
                time.sleep(interval)
            for i in range(pool.size):
                comm.send((0,'STOP_DAEMON'), i, tag=INQUIRY)
        tasks_handler.join()

    if pool.size > 1:
        tasks_handler = threading.Thread(target=task_daemon)
        tasks_handler.start()

    while tasks:
        task = tasks.pop(0)
        yield task

    if pool.size > 1:
        def next_proc(proc):
            proc = (proc+1) % pool.size
            if proc == rank:
                proc = (proc+1) % pool.size
            return proc
        proc_last = (rank + 1) % pool.size
        proc = next_proc(proc_last)

        while True:
            comm.send((rank,None), proc, tag=INQUIRY)
            task = comm.recv(source=proc, tag=TASK)
            if isinstance(task, str) and task == 'OUT_OF_TASK':
                prepare_to_stop()
                return
            elif isinstance(task, str) and task == 'BYPASS':
                if proc == proc_last:
                    prepare_to_stop()
                    return
                else:
                    proc = next_proc(proc)
            else:
                if proc != proc_last:
                    proc_last, proc = proc, next_proc(proc)
                yield task

def bcast(buf, root=0):
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)
    comm.Bcast(buf, root)
    return buf

## Useful when sending large batches of arrays
#def safe_bcast(buf, root=0):
    

def reduce(sendbuf, op=MPI.SUM, root=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    recvbuf = numpy.zeros_like(sendbuf)
    comm.Reduce(sendbuf, recvbuf, op, root)
    if rank == root:
        return recvbuf
    else:
        return sendbuf

def allreduce(sendbuf, op=MPI.SUM):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    recvbuf = numpy.zeros_like(sendbuf)
    comm.Allreduce(sendbuf, recvbuf, op)
    return recvbuf

def gather(sendbuf, root=0):
#    if pool.debug:
#        if rank == 0:
#            res = [sendbuf]
#            for k in range(1, pool.size):
#                dat = comm.recv(source=k)
#                res.append(dat)
#            return numpy.vstack([x for x in res if len(x) > 0])
#        else:
#            comm.send(sendbuf, dest=0)
#            return sendbuf

    sendbuf = numpy.asarray(sendbuf, order='C')
    mpi_dtype = sendbuf.dtype.char
    if rank == root:
        size_dtype = comm.gather((sendbuf.size, mpi_dtype), root=root)
        _assert(all(x[1] == mpi_dtype for x in size_dtype if x[0] > 0))
        counts = numpy.array([x[0] for x in size_dtype])
        displs = numpy.append(0, numpy.cumsum(counts[:-1]))
        recvbuf = numpy.empty(sum(counts), dtype=sendbuf.dtype)
        comm.Gatherv([sendbuf.ravel(), mpi_dtype],
                     [recvbuf.ravel(), counts, displs, mpi_dtype], root)
        return recvbuf.reshape((-1,) + sendbuf[0].shape)
    else:
        comm.gather((sendbuf.size, mpi_dtype), root=root)
        comm.Gatherv([sendbuf.ravel(), mpi_dtype], None, root)
        return sendbuf

def allgather(sendbuf):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.dtype.char == mpi_dtype or sendbuf.size == 0)
    counts = numpy.array(comm.allgather(sendbuf.size))
    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(sum(counts), dtype=sendbuf.dtype)
    comm.Allgatherv([sendbuf.ravel(), mpi_dtype],
                    [recvbuf.ravel(), counts, displs, mpi_dtype])
    return recvbuf.reshape((-1,) + shape[1:])

def alltoall(sendbuf, split_recvbuf=False):
    if isinstance(sendbuf, numpy.ndarray):
        mpi_dtype = comm.bcast(sendbuf.dtype.char)
        sendbuf = numpy.asarray(sendbuf, mpi_dtype, 'C')
        nrow = sendbuf.shape[0]
        ncol = sendbuf.size // nrow
        segsize = (nrow+pool.size-1) // pool.size * ncol
        sdispls = numpy.arange(0, pool.size*segsize, segsize)
        sdispls[sdispls>sendbuf.size] = sendbuf.size
        scounts = numpy.append(sdispls[1:]-sdispls[:-1], sendbuf.size-sdispls[-1])
    else:
        assert(len(sendbuf) == pool.size)
        mpi_dtype = comm.bcast(sendbuf[0].dtype.char)
        sendbuf = [numpy.asarray(x, mpi_dtype).ravel() for x in sendbuf]
        scounts = numpy.asarray([x.size for x in sendbuf])
        sdispls = numpy.append(0, numpy.cumsum(scounts[:-1]))
        sendbuf = numpy.hstack(sendbuf)

    rcounts = numpy.asarray(comm.alltoall(scounts))
    rdispls = numpy.append(0, numpy.cumsum(rcounts[:-1]))

    recvbuf = numpy.empty(sum(rcounts), dtype=mpi_dtype)
    comm.Alltoallv([sendbuf.ravel(), scounts, sdispls, mpi_dtype],
                   [recvbuf.ravel(), rcounts, rdispls, mpi_dtype])
    if split_recvbuf:
        return [recvbuf[p0:p0+c] for p0,c in zip(rdispls,rcounts)]
    else:
        return recvbuf

def sendrecv(sendbuf, source=0, dest=0):
    if source == dest:
        return sendbuf

    if rank == source:
        sendbuf = numpy.asarray(sendbuf, order='C')
        comm.send((sendbuf.shape, sendbuf.dtype), dest=dest)
        comm.Send(sendbuf, dest=dest)
        return sendbuf
    elif rank == dest:
        shape, dtype = comm.recv(source=source)
        recvbuf = numpy.empty(shape, dtype=dtype)
        comm.Recv(recvbuf, source=source)
        return recvbuf

def _assert(condition):
    if not condition:
        sys.stderr.write(''.join(traceback.format_stack()[:-1]))
        comm.Abort()

def register_for(obj):
    global _registry
    key = id(obj)
    # Keep track of the object in a global registry.  On slave nodes, the
    # object can be accessed from global registry.
    _registry[key] = obj
    keys = comm.gather(key)
    if rank == 0:
        obj._reg_keys = keys
    return obj

def del_registry(reg_keys):
    if reg_keys:
        def f(reg_keys):
            from mpi4pyscf.tools import mpi
            mpi._registry.pop(reg_keys[mpi.rank])
        pool.apply(f, reg_keys, reg_keys)
    return []
