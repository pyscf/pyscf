################### the MPI module ##########################

from pyscf import lib
import mpi4py
from mpi4py import MPI
import numpy
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

## some tools copy from mpi4pyscf ##

INT_MAX = 2147483647
BLKSIZE = INT_MAX // 64 + 1

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