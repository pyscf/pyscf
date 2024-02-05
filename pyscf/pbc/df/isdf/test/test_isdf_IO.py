
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

import pyscf.pbc.df.isdf.test.test_isdf_fast as isdf_fast
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

import sys
import ctypes
import _ctypes

from multiprocessing import Pool

import dask.array as da
from dask import delayed

import memory_profiler

from memory_profiler import profile

libfft = lib.load_library('libfft')
libpbc = lib.load_library('libpbc')

AUX_BASIS_DATASET     = 'aux_basis'  # NOTE: indeed can be over written
AUX_BASIS_FFT_DATASET = 'aux_basis_fft'
V_DATASET             = 'V'
AOR_DATASET           = 'aoR'
MAX_BUNCHSIZE         = 4096
MAX_CHUNKSIZE         = int(3e9//8)  # 2GB
THREAD_BUF_PERCENTAGE = 0.15
CHUNK_COL_LIMITED     = 8192
CHUNK_ROW_LIMITED     = 1024

def _determine_bunchsize(nao, naux, mesh, buf_size):

    mesh         = np.asarray(mesh, dtype=np.int32)
    mesh_complex = np.asarray([mesh[0], mesh[1], mesh[2]//2+1], dtype=np.int32)
    ncomplex     = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2
    ngrids       = mesh[0] * mesh[1] * mesh[2]
    nThread      = lib.num_threads()

    #### 1. calculate the bunchsize for the construction of aux basis ####

    blksize_aux = (buf_size - naux*naux) // (4 * nao + 2 * naux)   # suppose that the memory is enough
    blksize_aux = min(blksize_aux, ngrids)
    blksize_aux = min(blksize_aux, MAX_CHUNKSIZE//naux)  # the maximal bunchsize in the construction of aux basis
    blksize_aux = (blksize_aux // nThread) * nThread

    #### 2. calculate the bunchsize for the construction of V and W ####

    ### you have to read the aux basis row by row ###

    bunch_size_IO = ((buf_size) // 7) // (ncomplex)  # ncomplex > ngrids
    if bunch_size_IO > naux:
        bunch_size_IO = naux
    if bunch_size_IO < nThread:
        print("WARNING: bunch_size_IO = %d < nThread = %d" % (bunch_size_IO, nThread))
    bunchsize          = bunch_size_IO // nThread
    bunch_size_IO      = (bunch_size_IO // (bunchsize * nThread)) * bunchsize * nThread
    if bunch_size_IO == 0:
        raise ValueError("IO_buf is not large enough, bunch_size_IO = %d, buf_size= %d" %
                            (bunch_size_IO, buf_size))
    # bufsize_per_thread = bunchsize * ncomplex
    # bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16

    chunk_row = bunch_size_IO  # this is the only restriction on the row size of chunk

    #### 3. calculate the bunchsize for the construction of W ####

    blksize_W = (buf_size - nThread * (naux * naux + 2)) // 3 // naux
    blksize_W = min(blksize_W, ncomplex)

    use_large_chunk = True
    if blksize_W < nThread * naux // 2:
        use_large_chunk = False

    if use_large_chunk == False:
        blksize_W = buf_size // 3 // naux
        blksize_W = min(blksize_W, ncomplex)
        blksize_W = min(blksize_W, 3*naux-4)  # do not need to read a large chunk
    blksize_W = blksize_W // 2 * 2     # make it even

    #### 4. calculate the bunchsize for read J,K ####

    offset = naux + ngrids + naux + ngrids + naux * naux + naux * nao
    bunchsize_readV = (buf_size - offset - naux * nao) // (naux + naux + nao + naux + nao*2 + nao + nao)
    bunchsize_readV = min(bunchsize_readV, MAX_BUNCHSIZE)
    # if naux < MAX_BUNCHSIZE: # NOTE: we cannot load too large chunk of grids
    #     bunchsize_readV = min(bunchsize_readV, MAX_BUNCHSIZE)
    # else:
    #     bunchsize_readV = min(bunchsize_readV, naux)

    offset += naux * bunchsize_readV
    offset += nao * bunchsize_readV
    nElmt_left = buf_size - offset
    grid_bunchsize = nElmt_left // nao // 3
    grid_bunchsize = min(grid_bunchsize, ngrids)
    grid_bunchsize = min(grid_bunchsize, MAX_BUNCHSIZE)
    # if nao < MAX_BUNCHSIZE: # NOTE: we cannot load too large chunk of grids
    #     grid_bunchsize = min(grid_bunchsize, MAX_BUNCHSIZE)
    # else:
    #     grid_bunchsize = min(grid_bunchsize, nao)

    print("bunch_size_IO   = ", bunch_size_IO)
    print("blksize_aux     = ", blksize_aux)
    print("bunchsize_readV = ", bunchsize_readV)
    print("grid_bunchsize  = ", grid_bunchsize)
    print("blksize_W       = ", blksize_W)

    chunk_col = min(bunchsize_readV, grid_bunchsize, blksize_W, blksize_aux)

    if chunk_row * chunk_col * 8 < 16000000:  # at least 16MB
        print("too small chunk, change the chunk row")
        chunk_row_new = 16000000 // (chunk_col * 8)
        chunk_row_new = (chunk_row_new // chunk_row + 1) * chunk_row
        chunk_row = chunk_row_new
        chunk_row = min(chunk_row, naux)

    if CHUNK_COL_LIMITED is not None:
        chunk_col = min(chunk_col, CHUNK_COL_LIMITED)
    if CHUNK_ROW_LIMITED is not None:
        chunk_row = min(chunk_row, CHUNK_ROW_LIMITED)

    # if chunk_row * chunk_col * 8 < 16000000: # at least 16MB
    #     print("too small chunk, change the chunk row")
    #     chunk_row_new = 16000000 // (chunk_col * 8)
    #     chunk_row_new = (chunk_row_new // chunk_row + 1) * chunk_row
    #     chunk_row = chunk_row_new
    #     chunk_row = min(chunk_row, naux)

    bunchsize_readV = (bunchsize_readV // chunk_col) * chunk_col
    grid_bunchsize  = (grid_bunchsize // chunk_col) * chunk_col
    blksize_W       = (blksize_W // chunk_col) * chunk_col
    blksize_aux     = (blksize_aux // chunk_col) * chunk_col
    blksize_aux     = (blksize_aux // nThread) * nThread

    print("blksize_aux = ", blksize_aux)
    print("bunchsize_readV = ", bunchsize_readV)
    print("grid_bunchsize = ", grid_bunchsize)
    print("blksize_W = ", blksize_W)
    print("chunk_row = ", chunk_row)
    print("chunk_col = ", chunk_col)

    return (chunk_row, chunk_col), bunch_size_IO, blksize_aux, bunchsize_readV, grid_bunchsize, blksize_W, use_large_chunk


def _construct_aux_basis_IO(mydf:isdf_fast.PBC_ISDF_Info, IO_File:str, IO_buf:np.ndarray):
    '''
    IO_buf_memory: seems to be redundant
    '''

    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_aux_basis = h5py.File(IO_File, 'a')
            if AUX_BASIS_DATASET in f_aux_basis:
                del (f_aux_basis[AUX_BASIS_DATASET])
            if AOR_DATASET in f_aux_basis:
                del (f_aux_basis[AOR_DATASET])
        else:
            f_aux_basis = h5py.File(IO_File, 'w')
    else:
        assert (isinstance(IO_File, h5py.Group))
        f_aux_basis = IO_File

    ### do the work ###

    IO_buf_memory = IO_buf.size * IO_buf.dtype.itemsize

    nao    = mydf.nao
    naux   = mydf.naux
    aoRg   = mydf.aoRg
    ngrids = mydf.ngrids

    blksize = (IO_buf_memory - naux*naux*IO_buf.dtype.itemsize) // ((4 * nao + 2 * naux)
               * IO_buf.dtype.itemsize)  # suppose that the memory is enough
    # chunks  = (IO_buf_memory // (8*2)//ngrids, ngrids)
    blksize = min(blksize, ngrids)
    blksize = min(blksize, MAX_CHUNKSIZE//naux)
    chunks = (naux, blksize)

    assert(blksize > 0)
    assert(chunks[0] > 0)

    if hasattr(mydf, 'blksize_aux'):
        blksize = mydf.blksize_aux

    print("IO_buf_memory = ", IO_buf_memory)
    print("blksize       = ", blksize)
    # print("chunks        = ", chunks)

    A = np.ndarray((naux, naux), dtype=IO_buf.dtype, buffer=IO_buf)
    A = np.asarray(lib.dot(aoRg.T, aoRg, c=A), order='C')  # no memory allocation here!
    lib.square_inPlace(A)

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    fn_cholesky = getattr(libpbc, "Cholesky", None)
    assert(fn_cholesky is not None)
    fn_cholesky(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux),
    )
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1,t2,"_aux_basis_IO.Cholesky")

    chunks1 = (naux, ngrids)
    chunks2 = (nao, ngrids)

    if hasattr(mydf, 'chunk_size'):
        chunks1 = mydf.chunk_size
        chunks2 = (min(mydf.chunk_size[0], nao), mydf.chunk_size[1])

    print("chunks1        = ", chunks1)
    print("chunks2        = ", chunks2)

    h5d_aux_basis = f_aux_basis.create_dataset(AUX_BASIS_DATASET, (naux, ngrids), 'f8', chunks=chunks1)
    h5d_aoR       = f_aux_basis.create_dataset(AOR_DATASET, (nao, ngrids), 'f8', chunks=chunks2)

    def save(col0, col1, buf:np.ndarray):
        dest_sel   = np.s_[:, col0:col1]
        source_sel = np.s_[:, :]
        h5d_aux_basis.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)

    def save_aoR(col0, col1, buf:np.ndarray):
        dest_sel   = np.s_[:, col0:col1]
        source_sel = np.s_[:, :]
        h5d_aoR.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)


    offset        = naux * naux * IO_buf.dtype.itemsize
    buf_calculate = np.ndarray((naux, blksize), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset       += naux*blksize*IO_buf.dtype.itemsize
    buf_write     = np.ndarray((naux, blksize), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset       += naux*blksize*IO_buf.dtype.itemsize
    offset_aoR1   = offset
    offset       += nao*blksize*IO_buf.dtype.itemsize*2  # complex
    offset_aoR2   = offset

    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    ## get the coord of grids ##

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
    df_tmp = MultiGridFFTDF2(mydf.cell)
    # NumInts = df_tmp._numint
    # grids   = df_tmp.grids
    # coords  = np.asarray(grids.coords).reshape(-1,3)
    coords  = mydf.coords
    assert coords is not None

    ## get the coords ##

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    nThread = lib.num_threads()
    weight  = np.sqrt(mydf.cell.vol / ngrids)

    with lib.call_in_background(save) as async_write:
        with lib.call_in_background(save_aoR) as async_write_aoR:

            for p0, p1 in lib.prange(0, ngrids, blksize):

                # build aux basis

                # aoR = NumInts.eval_ao(mydf.cell, coords[p0:p1])[0].T * weight  # TODO: write to disk

                AoR_Buf1 = np.ndarray((nao, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_aoR1)
                AoR_Buf2 = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_aoR2)

                AoR_Buf1 = ISDF_eval_gto(mydf.cell, coords=coords[p0:p1], out=AoR_Buf1) * weight
                aoR      = AoR_Buf1

                if p1!=p0 + blksize:
                    buf_calculate = np.ndarray((naux, p1-p0), dtype=IO_buf.dtype,
                                               buffer=buf_calculate)  # the last chunk

                lib.dot(aoRg.T, aoR, c=buf_calculate)
                lib.square_inPlace(buf_calculate)

                ngrid_tmp = p1 - p0
                Bunchsize = ngrid_tmp // nThread
                fn_build_aux(
                    ctypes.c_int(naux),
                    A.ctypes.data_as(ctypes.c_void_p),
                    buf_calculate.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ngrid_tmp),
                    ctypes.c_int(Bunchsize)
                )

                async_write(p0, p1, buf_calculate)
                async_write_aoR(p0, p1, AoR_Buf1)

                buf_write, buf_calculate = buf_calculate, buf_write
                AoR_Buf1, AoR_Buf2       = AoR_Buf2, AoR_Buf1
                offset_aoR1, offset_aoR2 = offset_aoR2, offset_aoR1

    ### close ###

    if isinstance(IO_File, str):
        f_aux_basis.close()

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1,t2,"_aux_basis_IO.build")

def _construct_V_W_IO(mydf:isdf_fast.PBC_ISDF_Info, mesh, IO_File:str, IO_buf:np.ndarray):

    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_dataset = h5py.File(IO_File, 'a')
            if V_DATASET in f_dataset:
                del (f_dataset[V_DATASET])
            assert (AUX_BASIS_DATASET in f_dataset)
        else:
            raise ValueError("IO_File should be a h5py.File object")
    else:
        assert (isinstance(IO_File, h5py.Group))
        f_dataset = IO_File

    ### do the work ###

    naux   = mydf.naux
    ngrids = mydf.ngrids

    mesh = np.asarray(mesh, dtype=np.int32)
    mesh_complex = np.asarray([mesh[0], mesh[1], mesh[2]//2+1], dtype=np.int32)

    ### the Coulomb kernel ###

    coulG      = tools.get_coulG(mydf.cell, mesh=mesh)
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)
    nThread    = lib.num_threads()

    ### buffer needed ###

    ### two to hold for read in aux_basis to construct V, two to hold another aux_basis to construct W ###
    ### one further buffer to hold the buf to construct V, suppose it consists of 5% of the total memory ###

    # suppose that nwork_per_thread , then the memory should be
    # buf size = nThreads * nwork_per_thread * mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2
    # suppose  V is of size (nThreads * nwork_per_thread) * mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2
    # then the memory needed is 5 * nThreads * nwork_per_thread * mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2
    # we set that the maximal nwork_per_thread is 16

    print("IO_buf.size = ", IO_buf.size)

    memory_thread      = int(THREAD_BUF_PERCENTAGE * IO_buf.size)

    bunchsize     = memory_thread // nThread // (mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2)
    if bunchsize > 16:
        bunchsize = 16
    if bunchsize == 0:
        bunchsize = 1

    if bunchsize * nThread > naux:
        bunchsize = naux // nThread  # force to use all the aux basis

    bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
    bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
    memory_thread      = bufsize_per_thread * nThread

    if memory_thread > IO_buf.size:
        raise ValueError("IO_buf is not large enough, memory_thread = %d, IO_buf.size = %d" %
                         (memory_thread, IO_buf.size))

    bunch_size_IO = ((IO_buf.size - memory_thread) // 5) // (ngrids)

    if bunch_size_IO > naux:
        bunch_size_IO = naux
    if bunch_size_IO == 0:
        raise ValueError("IO_buf is not large enough, bunch_size_IO = %d, IO_buf.size = %d" %
                         (bunch_size_IO, IO_buf.size))
    if bunch_size_IO < nThread:
        print("WARNING: bunch_size_IO = %d < nThread = %d" % (bunch_size_IO, nThread))

    bunch_size_IO      = (bunch_size_IO // (bunchsize * nThread)) * bunchsize * nThread
    bunch_size_IO      = min(bunch_size_IO, MAX_CHUNKSIZE//ngrids)
    bunchsize          = bunch_size_IO // nThread
    bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
    bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
    memory_thread      = bufsize_per_thread * nThread
    bunch_size_IO      = nThread * bunchsize

    print("bunchsize = ", bunchsize)
    print("bunch_size_IO = ", bunch_size_IO)
    print("memory_thread = ", memory_thread)

    ### allocate buffer ###

    buf_thread = np.ndarray((nThread, bufsize_per_thread), dtype=IO_buf.dtype, buffer=IO_buf)
    offset     = nThread * bufsize_per_thread * IO_buf.dtype.itemsize

    buf_V1  = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset += bunch_size_IO * ngrids * IO_buf.dtype.itemsize
    buf_V2  = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset += bunch_size_IO * ngrids * IO_buf.dtype.itemsize

    buf_aux_basis = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset += bunch_size_IO * ngrids * IO_buf.dtype.itemsize
    buf_aux_basis_bra = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset += bunch_size_IO * ngrids * IO_buf.dtype.itemsize
    buf_aux_basis_ket = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)

    if hasattr(mydf, 'ddot_buf') and mydf.ddot_buf is not None:
        if mydf.ddot_buf.size < (bunch_size_IO * bunch_size_IO + 2) * (nThread+1):
            mydf.ddot_buf = np.zeros(((bunch_size_IO * bunch_size_IO + 2) * (nThread+1)),
                                     dtype=np.float64)  # reallocate
    else:
        mydf.ddot_buf = np.zeros(((bunch_size_IO * bunch_size_IO + 2) * (nThread+1)), dtype=np.float64)

    ddot_buf = mydf.ddot_buf
    ddot_out = ddot_buf[(bunch_size_IO * bunch_size_IO + 2) * nThread:]

    ### create data set

    chunks = (bunch_size_IO, ngrids)

    h5d_V = f_dataset.create_dataset(V_DATASET, (naux, ngrids), 'f8', chunks=chunks)

    ### perform the task ###

    def save_V(row0, row1, buf:np.ndarray):
        # assert(buf.shape == (row1-row0, ngrids))
        if row0 < row1:
            h5d_V[row0:row1] = buf[:row1-row0]

    def load_aux_basis_async(row0, row1, buf:np.ndarray):
        # assert(buf.shape == (row1-row0, ngrids))
        if row0 < row1:
            buf[:row1-row0] = f_dataset[AUX_BASIS_DATASET][row0:row1]

    def load_aux_basis(row0, row1, buf:np.ndarray):
        # assert(buf.shape == (row1-row0, ngrids))
        if row0 < row1:
            buf[:row1-row0] = f_dataset[AUX_BASIS_DATASET][row0:row1]

    fn = getattr(libpbc, "_construct_V", None)
    assert(fn is not None)

    # mydf.W = np.zeros((naux, naux), dtype=np.float64)

    with lib.call_in_background(load_aux_basis_async) as prefetch:
        with lib.call_in_background(save_V) as async_write:

            load_aux_basis(0, min(bunch_size_IO, naux), buf_aux_basis_bra)  # force to load first bunch

            for p0, p1 in lib.prange(0, naux, bunch_size_IO):

                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

                # load aux basis

                buf_aux_basis, buf_aux_basis_bra = buf_aux_basis_bra, buf_aux_basis

                prefetch(p1, min(p1+bunch_size_IO, naux), buf_aux_basis_bra)
                prefetch(p1, min(p1+bunch_size_IO, naux), buf_aux_basis_ket)

                # construct V

                fn(mesh.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(bunch_size_IO),
                   buf_aux_basis.ctypes.data_as(ctypes.c_void_p),
                   coulG_real.ctypes.data_as(ctypes.c_void_p),
                   buf_V1.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(bunchsize),
                   buf_thread.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(bufsize_per_thread))

                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                print("construct V[%5d:%5d] wall time: %12.6f CPU time: %12.6f" %
                      (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

                async_write(p0, p1, buf_V1)

                t1 = t2

                # perform the current mvp

                _ddot_out = np.ndarray((p1-p0, p1-p0), dtype=ddot_out.dtype, buffer=ddot_out)
                lib.ddot_withbuffer(buf_V1[:p1-p0, :], buf_aux_basis[:p1-p0, :].T, buf=ddot_buf, c=_ddot_out)
                mydf.W[p0:p1, p0:p1] = _ddot_out

                # load aux basis one loop

                for q0, q1 in lib.prange(p1, naux, bunch_size_IO):

                    buf_aux_basis, buf_aux_basis_ket = buf_aux_basis_ket, buf_aux_basis
                    prefetch(q1, min(q1+bunch_size_IO, naux), buf_aux_basis_ket)

                    # perform the current mvp

                    _ddot_out = np.ndarray((p1-p0, q1-q0), dtype=ddot_out.dtype, buffer=ddot_out)
                    lib.ddot_withbuffer(buf_V1[:p1-p0, :], buf_aux_basis[:q1-q0,:].T, buf=ddot_buf, c=_ddot_out)
                    mydf.W[p0:p1, q0:q1] = _ddot_out
                    mydf.W[q0:q1, p0:p1] = _ddot_out.T

                buf_V1, buf_V2 = buf_V2, buf_V1

                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                print("construct W[%5d:%5d] wall time: %12.6f CPU time: %12.6f" %
                      (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))


def _copy_b_to_a(a:np.ndarray, b:np.ndarray, size:int):
    assert(a.shape == b.shape)
    a.ravel()[:size] = b.ravel()[:size]

def _construct_V_W_IO2(mydf:isdf_fast.PBC_ISDF_Info, mesh, IO_File:str, IO_buf:np.ndarray):
    '''
    this version only read aux_basis twice!
    '''

    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_dataset = h5py.File(IO_File, 'a')
            if V_DATASET in f_dataset:
                del (f_dataset[V_DATASET])
            if AUX_BASIS_FFT_DATASET in f_dataset:
                del (f_dataset[AUX_BASIS_FFT_DATASET])
            assert (AUX_BASIS_DATASET in f_dataset)
        else:
            raise ValueError("IO_File should be a h5py.File object")
    else:
        assert (isinstance(IO_File, h5py.Group))
        f_dataset = IO_File

    ### do the work ###

    naux   = mydf.naux
    ngrids = mydf.ngrids

    mesh = np.asarray(mesh, dtype=np.int32)
    mesh_complex = np.asarray([mesh[0], mesh[1], mesh[2]//2+1], dtype=np.int32)
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2

    ### the Coulomb kernel ###

    coulG      = tools.get_coulG(mydf.cell, mesh=mesh)
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)  # drop the minus frequency part
    nThread    = lib.num_threads()

    # print("mesh = ", mesh)
    # print(coulG.reshape(*mesh)[1,2,:])
    # print(coulG_real.reshape(*mesh_complex)[1,2,:])

    ### buffer needed ###

    ### two to hold for read in aux_basis to construct V, two to hold another aux_basis to construct W ###
    ### one further buffer to hold the buf to construct V, suppose it consists of 5% of the total memory ###

    # suppose that nwork_per_thread , then the memory should be
    # buf size = nThreads * nwork_per_thread * mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2
    # suppose  V is of size (nThreads * nwork_per_thread) * mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2
    # then the memory needed is 5 * nThreads * nwork_per_thread * mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2
    # we set that the maximal nwork_per_thread is 16

    # print("IO_buf.size = ", IO_buf.size)
    # memory_thread      = int(THREAD_BUF_PERCENTAGE * IO_buf.size)
    # bunchsize     = memory_thread // nThread // (mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2)
    # if bunchsize > 16:
    #     bunchsize = 16
    # if bunchsize == 0:
    #     bunchsize = 1
    # if bunchsize * nThread > naux:
    #     bunchsize = naux // nThread  # force to use all the aux basis
    # bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
    # bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
    # memory_thread      = bufsize_per_thread * nThread
    # if memory_thread > IO_buf.size:
    #     raise ValueError("IO_buf is not large enough, memory_thread = %d, IO_buf.size = %d" %
    #                      (memory_thread, IO_buf.size))
    # print("memory_thread = ", memory_thread)
    # bunch_size_IO = ((IO_buf.size - memory_thread) // 6) // (ncomplex) # ncomplex > ngrids
    # print("bunch_size_IO = ", bunch_size_IO)
    # if bunch_size_IO > naux:
    #     bunch_size_IO = naux
    # if bunch_size_IO == 0:
    #     raise ValueError("IO_buf is not large enough, bunch_size_IO = %d, IO_buf.size = %d" %
    #                      (bunch_size_IO, IO_buf.size))
    # bunchsize          = bunch_size_IO // nThread
    # bunch_size_IO      = (bunch_size_IO // (bunchsize * nThread)) * bunchsize * nThread
    # bunch_size_IO      = min(bunch_size_IO, MAX_CHUNKSIZE//ngrids)
    # if bunch_size_IO < nThread:
    #     print("WARNING: bunch_size_IO = %d < nThread = %d" % (bunch_size_IO, nThread))
    # bunchsize          = bunch_size_IO // nThread
    # bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
    # bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
    # memory_thread      = bufsize_per_thread * nThread
    # bunch_size_IO      = nThread * bunchsize

    bunch_size_IO = ((IO_buf.size) // 7) // (ncomplex)  # ncomplex > ngrids
    if bunch_size_IO > naux:
        bunch_size_IO = naux
    if bunch_size_IO < nThread:
        print("WARNING: bunch_size_IO = %d < nThread = %d" % (bunch_size_IO, nThread))
    bunchsize          = bunch_size_IO // nThread
    bunch_size_IO      = (bunch_size_IO // (bunchsize * nThread)) * bunchsize * nThread
    if bunch_size_IO == 0:
        raise ValueError("IO_buf is not large enough, bunch_size_IO = %d, IO_buf.size = %d" %
                            (bunch_size_IO, IO_buf.size))
    bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
    bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
    # memory_thread      = bufsize_per_thread * nThread
    # bunch_size_IO      = nThread * bunchsize

    if hasattr(mydf, 'nRow_IO_V'):
        bunch_size_IO = mydf.nRow_IO_V
        bunchsize     = bunch_size_IO // nThread
        bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
        bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16

    print("bunchsize = ", bunchsize)
    print("bunch_size_IO = ", bunch_size_IO)

    ### allocate buffer ###

    buf_thread = np.ndarray((nThread, bufsize_per_thread), dtype=IO_buf.dtype, buffer=IO_buf)
    offset     = nThread * bufsize_per_thread * IO_buf.dtype.itemsize

    buf_V1  = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset += bunch_size_IO * ngrids * IO_buf.dtype.itemsize
    buf_V2  = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset += bunch_size_IO * ngrids * IO_buf.dtype.itemsize

    buf_aux_basis_1 = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset         += bunch_size_IO * ngrids * IO_buf.dtype.itemsize
    buf_aux_basis_2 = np.ndarray((bunch_size_IO, ngrids), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset         += bunch_size_IO * ngrids * IO_buf.dtype.itemsize

    buf_aux_basis_fft_1 = np.ndarray((bunch_size_IO, ncomplex), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset             += bunch_size_IO * ncomplex * IO_buf.dtype.itemsize
    buf_aux_basis_fft_2 = np.ndarray((bunch_size_IO, ncomplex), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset             += bunch_size_IO * ncomplex * IO_buf.dtype.itemsize

    ### create data set

    chunks1 = (bunch_size_IO, ngrids)
    chunks2 = (bunch_size_IO, ncomplex)

    if hasattr(mydf, 'chunk_size'):
        chunks1 = mydf.chunk_size
        chunks2 = mydf.chunk_size

    h5d_V = f_dataset.create_dataset(V_DATASET, (naux, ngrids), 'f8', chunks=chunks1)
    h5d_aux_basis_fft = f_dataset.create_dataset(AUX_BASIS_FFT_DATASET, (naux, ncomplex), 'f8', chunks=chunks2)

    ### perform the task ###

    def save_V(row0, row1, buf:np.ndarray):
        if row0 < row1:
            dset_sel   = np.s_[row0:row1, :]
            source_sel = np.s_[:row1-row0, :]
            h5d_V.write_direct(buf, source_sel=source_sel, dest_sel=dset_sel)

    def save_auxbasisfft(row0, row1, buf:np.ndarray):
        if row0 < row1:
            dset_sel   = np.s_[row0:row1, :]
            source_sel = np.s_[:row1-row0, :]
            h5d_aux_basis_fft.write_direct(buf, source_sel=source_sel, dest_sel=dset_sel)

    def load_aux_basis_async(row0, row1, buf:np.ndarray):
        if row0 < row1:
            source   = np.s_[row0:row1, :]
            dest     = np.s_[:row1-row0, :]
            f_dataset[AUX_BASIS_DATASET].read_direct(buf, source_sel=source, dest_sel=dest)

    def load_aux_basis(row0, row1, buf:np.ndarray):
        if row0 < row1:
            source   = np.s_[row0:row1, :]
            dest     = np.s_[:row1-row0, :]
            f_dataset[AUX_BASIS_DATASET].read_direct(buf, source_sel=source, dest_sel=dest)

    fn = getattr(libpbc, "_construct_V2", None)
    assert(fn is not None)

    with lib.call_in_background(load_aux_basis_async) as prefetch:
        with lib.call_in_background(save_V) as async_write:
            with lib.call_in_background(save_auxbasisfft) as async_write_auxbasisfft:

                load_aux_basis(0, bunch_size_IO, buf_aux_basis_2)  # force to load first bunch

                for p0, p1 in lib.prange(0, naux, bunch_size_IO):

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    # load aux basis

                    buf_aux_basis_2, buf_aux_basis_1 = buf_aux_basis_1, buf_aux_basis_2

                    prefetch(p1, min(p1+bunch_size_IO, naux), buf_aux_basis_2)

                    # construct V

                    fn(mesh.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(bunch_size_IO),
                       buf_aux_basis_1.ctypes.data_as(ctypes.c_void_p),
                       coulG_real.ctypes.data_as(ctypes.c_void_p),
                       buf_V1.ctypes.data_as(ctypes.c_void_p),
                       buf_aux_basis_fft_1.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(bunchsize),
                       buf_thread.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(bufsize_per_thread))

                    async_write_auxbasisfft(p0, p1, buf_aux_basis_fft_1)
                    buf_aux_basis_fft_1, buf_aux_basis_fft_2 = buf_aux_basis_fft_2, buf_aux_basis_fft_1
                    async_write(p0, p1, buf_V1)
                    buf_V1, buf_V2 = buf_V2, buf_V1

                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    print("construct V[%5d:%5d] wall time: %12.6f CPU time: %12.6f" %
                        (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))


    #### another loop to construct W ####

    def load_aux_basis_fft_async(col0, col1, buf:np.ndarray):
        if col0 < col1:
            source   = np.s_[:, col0:col1]
            dest     = np.s_[:, :col1-col0]
            f_dataset[AUX_BASIS_FFT_DATASET].read_direct(buf, source_sel=source, dest_sel=dest)

    def load_aux_basis_fft(col0, col1, buf:np.ndarray):
        if col0 < col1:
            source   = np.s_[:, col0:col1]
            dest     = np.s_[:, :col1-col0]
            f_dataset[AUX_BASIS_FFT_DATASET].read_direct(buf, source_sel=source, dest_sel=dest)

    # first to check whether we can offer enough memory to hold the buffer for ddot

    blksize = (IO_buf.size - nThread * (naux * naux + 2)) // 3 // naux
    blksize = min(blksize, ncomplex)

    use_large_chunk = True
    if blksize < nThread * naux // 2:
        use_large_chunk = False

    if use_large_chunk == False:
        blksize = IO_buf.size // 3 // naux
        blksize = min(blksize, ncomplex)
        blksize = min(blksize, 3*naux-4)  # do not need to read a large chunk
    blksize = blksize // 2 * 2     # make it even

    if hasattr(mydf, 'use_large_chunk_W') and hasattr(mydf, 'blksize_W'):
        use_large_chunk = mydf.use_large_chunk_W
        blksize = mydf.blksize_W

    offset_buf1         = 0
    buf_aux_basis_fft_1 = np.ndarray((naux, blksize), dtype=IO_buf.dtype, buffer=IO_buf)
    offset              = blksize * naux * IO_buf.dtype.itemsize

    offset_buf2         = offset
    buf_aux_basis_fft_2 = np.ndarray((naux, blksize), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset)
    offset             += blksize * naux * IO_buf.dtype.itemsize

    offset_fft_copy = offset
    offset         += blksize * naux * IO_buf.dtype.itemsize
    offset_ddot     = offset

    if use_large_chunk == True:
        ddot_buffer = np.ndarray((naux * naux + 2) * (nThread), dtype=IO_buf.dtype, buffer=IO_buf, offset=offset_ddot)
    else:
        ddot_buffer = None

    fn = getattr(libpbc, "_construct_W_multiG", None)
    assert(fn is not None)

    # to account for the rfft

    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1]
    if mesh[2] % 2 == 0:
        coulG_real[:,:,1:-1] *= 2
    else:
        coulG_real[:,:,1:] *= 2
    coulG_real = coulG_real.reshape(-1)

    with lib.call_in_background(load_aux_basis_fft_async) as prefetch:
        load_aux_basis_fft(0, blksize, buf_aux_basis_fft_2)  # force to load first bunch

        for p0, p1 in lib.prange(0, ncomplex, blksize):

            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

            # load aux basis

            buf_aux_basis_fft_2, buf_aux_basis_fft_1 = buf_aux_basis_fft_1, buf_aux_basis_fft_2
            offset_buf1, offset_buf2                 = offset_buf2, offset_buf1

            buf_aux_basis_fft_2 = np.ndarray((naux, min(p1+blksize, ncomplex)-p1),
                                             dtype=IO_buf.dtype, buffer=IO_buf, offset=offset_buf2)
            prefetch(p1, min(p1+blksize, ncomplex), buf_aux_basis_fft_2)

            # copy buf1 to buf3

            buf_aux_basis_fft_copy = np.ndarray((naux, p1-p0), dtype=IO_buf.dtype,
                                                buffer=IO_buf, offset=offset_fft_copy)
            _copy_b_to_a(buf_aux_basis_fft_copy, buf_aux_basis_fft_1, buf_aux_basis_fft_1.size)

            # multiply buf_aux_basis_fft_copy by coulG_real

            fn(ctypes.c_int(naux),
               ctypes.c_int(p0//2),
               ctypes.c_int(p1//2),
               buf_aux_basis_fft_copy.ctypes.data_as(ctypes.c_void_p),
               coulG_real.ctypes.data_as(ctypes.c_void_p))

            ## ddot with buf_aux_basis_fft_2

            if use_large_chunk:
                lib.ddot_withbuffer(buf_aux_basis_fft_copy, buf_aux_basis_fft_1.T, buf=ddot_buffer, c=mydf.W, beta=1)
            else:
                lib.ddot(buf_aux_basis_fft_copy, buf_aux_basis_fft_1.T, c=mydf.W, beta=1)

            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

            print("construct W[%5d:%5d] wall time: %12.6f CPU time: %12.6f" % (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

    factor = 1.0 / np.prod(mesh)
    mydf.W *= factor

#### the following subroutines is designed for cache-friendly IP selection ####

# @profile
def _colpivot_qr(AA, Q, R, max_rank=None, cutoff=1e-14):
    '''
    we do not need Q
    '''
    n, m = AA.shape  # (ngrids, nao)
    # Q = np.zeros((m, m))
    # R = np.zeros((m, n))
    # AA = A.T.copy()
    pivot = np.arange(n)

    # print("AA.shape = ", AA.shape)
    # print("Q.shape  = ", Q.shape)

    if max_rank is None:
        max_rank = min(m, n)

    npt_find = 0

    for j in range(min(m, n, max_rank)):

        # Find the column with the largest norm

        norms = np.linalg.norm(AA[j:, :], axis=1)

        p = np.argmax(norms) + j

        # print("norms = ", norms)

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
        AA[j + 1:, :] -= np.outer(R[j, j + 1:], Q[j, :])
        AA[j, :] = Q[j, :]

    return pivot[:npt_find]

## this is a test func

def _colpivot_qr_C(B, max_rank=None, cutoff=1e-14):
    '''
    we do not need Q
    '''
    m, n = B.shape  # (nao, ngrids)
    pivot = np.arange(n, dtype=np.int32)

    if max_rank is None:
        max_rank = min(m, n)

    fn = getattr(libpbc, "ColPivotQR", None)
    assert(fn is not None)

    # Q = np.zeros((m, n))
    R = np.zeros((m, n))
    npt_find = ctypes.c_int(0)
    nThread  = lib.num_threads()
    thread_buffer = np.zeros((nThread+1, n+1), dtype=np.double)
    global_buffer = np.zeros((1, n), dtype=np.double)

    fn(B.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(m),
       ctypes.c_int(n),
       ctypes.c_int(max_rank),
       ctypes.c_double(cutoff),
       pivot.ctypes.data_as(ctypes.c_void_p),
       # Q.ctypes.data_as(ctypes.c_void_p),
       R.ctypes.data_as(ctypes.c_void_p),
       ctypes.byref(npt_find),
       thread_buffer.ctypes.data_as(ctypes.c_void_p),
       global_buffer.ctypes.data_as(ctypes.c_void_p))

    npt_find = npt_find.value

    return pivot[:npt_find]

@delayed
# @profile
def _atm_IP_task(taskinfo:tuple, buffer):
    grid_ID, aoR_atm, nao, nao_atm, c, m = taskinfo
    Q, R, aoR_atm1, aoR_atm2, aoPairBuffer = buffer

    npt_find = c * nao_atm
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
        lib.dot(aoR_atm, G1, c=aoR_atm1)
        lib.dot(aoR_atm, G2, c=aoR_atm2)
    aoPair = np.einsum('ki,kj->kij', aoR_atm1, aoR_atm2,
                       out=aoPairBuffer.reshape(aoR_atm.shape[0],naux_tmp,naux_tmp)).reshape(grid_ID.shape[0], -1)
    pivot = _colpivot_qr(aoPair, Q, R, max_rank=npt_find)
    pivot_ID = grid_ID[pivot]  # the global ID
    return pivot_ID

def _select_IP_direct2(mydf:isdf_fast.PBC_ISDF_Info, c:int, m:int):

    ''' 
    too complicated forget it
    '''

    ### determine the bunch size to loop grid points ###

    nao          = mydf.nao
    naux_random  = int(np.sqrt(c*nao)) + m

    G1 = np.random.rand(nao, naux_random)
    G1, _ = numpy.linalg.qr(G1)
    G1    = G1.T
    G2 = np.random.rand(nao, naux_random)
    G2, _ = numpy.linalg.qr(G2)
    G2    = G2.T

    naux_random2 = naux_random * naux_random
    naux         = c * nao

    nThread      = lib.num_threads()

    buf     = mydf.IO_buf
    bufsize = buf.size

    bunchsize = (bufsize - 5 * naux_random2 * naux - naux - (nThread + 1)*naux) // (2 * nao + 2 * naux_random +
                 naux_random2 + naux + 1 + nThread + 1 + 2 * naux_random2 + naux) - 1

    print("bufsize   = ", bufsize)
    print("bunchsize = ", bunchsize)

    if bunchsize < naux:
        print("_select_IP_direct2 warning: bunchsize = %d < naux = %d" % (bunchsize, naux))

    if bunchsize <= 0:
        raise ValueError("bufsize is not large enough")

    ### allocate buf ###

    offset = 0
    offset_aoPairRg = 0
    offset += naux_random2 * naux * IO_buf.dtype.itemsize

    offset_OrthoBasis = offset
    offset += naux_random2 * naux * IO_buf.dtype.itemsize


    offset_aoRg = offset
    offset += 2 * nao * bunchsize * IO_buf.dtype.itemsize  # 2 due to complex

    offset_aoG1 = offset
    offset += naux_random * bunchsize * IO_buf.dtype.itemsize

    offset_aoG2 = offset
    offset += naux_random * bunchsize * IO_buf.dtype.itemsize

    offset_aoPairBuffer = offset
    offset += naux_random2 * bunchsize * IO_buf.dtype.itemsize

    offset_InnerProd = offset
    offset          += naux * bunchsize * IO_buf.dtype.itemsize

    offset_thread_buf = offset
    offset           += (nThread + 1) * (bunchsize + naux + 1) * IO_buf.dtype.itemsize

    offset_global_buf = offset
    offset           += (bunchsize + naux) * IO_buf.dtype.itemsize

    offset_Q = offset
    offset  += naux_random2 * (bunchsize + naux) * IO_buf.dtype.itemsize

    offset_R = offset
    offset  += naux_random2 * (bunchsize + naux) * IO_buf.dtype.itemsize

    offset_SlicePack = offset

    ### fetch involved C function ###

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(mydf.cell)
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    assert coords is not None
    ngrids  = coords.shape[0]
    weight  = np.sqrt(mydf.cell.vol / ngrids)

    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    assert(fn_colpivot_qr is not None)

    fn_calculatenorm_remained = getattr(libpbc, "CalculateNormRemained", None)
    assert(fn_calculatenorm_remained is not None)

    fn_pack_slices = getattr(libpbc, "PackABwithSlice", None)
    assert(fn_pack_slices is not None)

    fn_packAB = getattr(libpbc, "PackAB", None)
    assert(fn_packAB is not None)

    fn_packAB_ABSlices = getattr(libpbc, "PackABwithABSlice", None)
    assert(fn_packAB_ABSlices is not None)

    nThread = lib.num_threads()

    ### parameter to control the selection ###


    IP_Now   = []
    max_rank = min(naux_random2, naux + bunchsize)
    cutoff   = 1e-12
    nPntAddedPotential = None
    nPntPotential = 0
    PntPotential = []

    p0 = 0
    p1 = min(bunchsize, ngrids)

    while True:

        if p0 < p1:
            AoR_Buf = np.ndarray((nao, p1-p0), dtype=np.complex128, buffer=buf, offset=offset_aoRg)
            aoR     = ISDF_eval_gto(mydf.cell, coords=coords[p0:p1], out=AoR_Buf) * weight
            aoRG1   = np.ndarray((naux_random, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_aoG1)
            aoRG2   = np.ndarray((naux_random, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_aoG2)

            lib.dot(G1, aoR, c=aoRG1)
            lib.dot(G2, aoR, c=aoRG2)

            aoPairBuffer = np.ndarray((naux_random2, p1-p0), dtype=IO_buf.dtype, buffer=buf,
                                    offset=offset_aoPairBuffer + nPntPotential*naux_random2*IO_buf.dtype.itemsize)

            fn_ik_jk_ijk(aoRG1.ctypes.data_as(ctypes.c_void_p),
                         aoRG2.ctypes.data_as(ctypes.c_void_p),
                         aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux_random),
                         ctypes.c_int(naux_random),
                         ctypes.c_int(p1-p0))

            grid_ID = np.arange(p0, p1, dtype=np.int32)

        if len(IP_Now) == 0:
            # just start
            Q = np.ndarray((naux_random2, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_Q)
            _copy_b_to_a(Q, aoPairBuffer, aoPairBuffer.size)
            R = np.ndarray((naux_random2, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_R)
            max_rank = min(naux_random2, p1-p0, naux)
            npt_find = ctypes.c_int(0)
            pivot = np.arange(p1-p0, dtype=np.int32)
            thread_buffer = np.ndarray((nThread+1, p1-p0+1), dtype=IO_buf.dtype, buffer=buf, offset=offset_thread_buf)
            global_buffer = np.ndarray((1, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_global_buf)
            fn_colpivot_qr(Q.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(naux_random2),
                            ctypes.c_int(p1-p0),
                            ctypes.c_int(max_rank),
                            ctypes.c_double(cutoff),
                            pivot.ctypes.data_as(ctypes.c_void_p),
                            R.ctypes.data_as(ctypes.c_void_p),
                            ctypes.byref(npt_find),
                            thread_buffer.ctypes.data_as(ctypes.c_void_p),
                            global_buffer.ctypes.data_as(ctypes.c_void_p))
            npt_find = npt_find.value
            # update parameter #
            cutoff   = abs(R[npt_find-1, npt_find-1])
            print(R[npt_find-1, npt_find-1:])
            nPntAddedPotential = None
            print("p0 = %d, p1 = %d, npt_find = %d, cutoff = %12.6e" % (p0, p1, npt_find, cutoff))
            slices = np.array(pivot[:npt_find], dtype=np.int32)
            aoPairRg = np.ndarray((naux_random2, npt_find), dtype=IO_buf.dtype, buffer=buf, offset=offset_aoPairRg)
            lib.dslice(aoPairBuffer, slices, out=aoPairRg)
            OrthoBasis = np.ndarray((naux_random2, npt_find), dtype=IO_buf.dtype, buffer=buf, offset=offset_OrthoBasis)
            lib.dslice(Q, slices, out=OrthoBasis)
            # check normalization
            tmp = np.dot(OrthoBasis.T, OrthoBasis)
            print("normalized = ", np.allclose(tmp, np.eye(npt_find)))
            print("tmp = ", tmp[:,0])
            IP_Now.extend(list(grid_ID[slices]))
            nPntPotential = 0
            # update the loop parameter #
            p0 = p1
            p1 = min(p0+bunchsize, ngrids)
        else:
            # calculate the inner product

            perform_QR = True

            if (p1-p0) > bunchsize // 5:

                print("dot not perform QR")

                perform_QR = False
                nIP = len(IP_Now)
                InnerProd = np.ndarray((nIP, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_InnerProd)
                lib.ddot(OrthoBasis.T, aoPairBuffer, c=InnerProd)
                # calculate the norm remained
                thread_buffer = np.ndarray((nThread, p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_thread_buf)
                NormRemained = np.ndarray((p1-p0), dtype=IO_buf.dtype, buffer=buf, offset=offset_global_buf)
                fn_calculatenorm_remained(InnerProd.ctypes.data_as(ctypes.c_void_p),
                                            ctypes.c_int(nIP),
                                            ctypes.c_int(p1-p0),
                                            aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                                            ctypes.c_int(naux_random2),
                                            thread_buffer.ctypes.data_as(ctypes.c_void_p),
                                            NormRemained.ctypes.data_as(ctypes.c_void_p))

                Pack_Needed = True
                if nPntPotential == 0:
                    aoPairPntPotential = aoPairBuffer
                    Pack_Needed = False
                    print("pack not needed")

                # find the grid points to be added

                PntPotential_tmp = np.where(NormRemained > cutoff)[0]

                print("norm remained = ", NormRemained)
                print("PntPotential_tmp = ", PntPotential_tmp)
                print("remained.shape = ", NormRemained.shape)
                print("PntPotential_tmp.shape = ", PntPotential_tmp.shape)

                PntPotentialID   = grid_ID[PntPotential_tmp]
                nPntAddedPotential = len(PntPotentialID)
                PntPotential.extend(list(PntPotentialID))
                nPntPotential = len(PntPotential)
                grid_ID = PntPotentialID

                # pack the info to be added

                if Pack_Needed:
                    SlicePackBuf = np.ndarray((naux_random2, nPntPotential), dtype=np.int32,
                                              buffer=buf, offset=offset_SlicePack)
                    aoPairPntPotentialNew = np.ndarray((naux_random2, nPntPotential),
                                                       dtype=IO_buf.dtype, buffer=buf, offset=offset_aoPairBuffer)

                    slices = np.array(PntPotential_tmp, dtype=np.int32)

                    fn_pack_slices(aoPairPntPotential.ctypes.data_as(ctypes.c_void_p),
                                   aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                                   aoPairPntPotentialNew.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(naux_random2),
                                    ctypes.c_int(aoPairPntPotential.shape[1]),
                                    ctypes.c_int(aoPairBuffer.shape[1]),
                                    slices.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(slices.shape[0]),
                                    SlicePackBuf.ctypes.data_as(ctypes.c_void_p))

                    aoPairPntPotential = aoPairPntPotentialNew

            # update the parameter

            left_pnt = bunchsize - nPntPotential

            print("nPntPotential = ", nPntPotential)
            print("left_pnt      = ", left_pnt)

            if left_pnt < naux // 2:
                perform_QR = True

            if p0 == p1 or p1 == ngrids:
                perform_QR = True  # the final one

            if perform_QR:

                # perform QR

                Q = np.ndarray(
                    (naux_random2, aoPairRg.shape[1] + aoPairPntPotential.shape[1]), dtype=IO_buf.dtype, buffer=buf, offset=offset_Q)
                R = np.ndarray(
                    (naux_random2, aoPairRg.shape[1] + aoPairPntPotential.shape[1]), dtype=IO_buf.dtype, buffer=buf, offset=offset_R)

                print(aoPairRg[0,:])
                print(aoPairPntPotential[0,:])

                fn_packAB(aoPairRg.ctypes.data_as(ctypes.c_void_p),
                        aoPairPntPotential.ctypes.data_as(ctypes.c_void_p),
                        Q.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naux_random2),
                        ctypes.c_int(aoPairRg.shape[1]),
                        ctypes.c_int(aoPairPntPotential.shape[1]))

                print(Q[0,:])
                print(Q[0,aoPairRg.shape[1]:])


                max_rank = min(naux_random2, aoPairRg.shape[1] + aoPairPntPotential.shape[1], naux)
                npt_find = ctypes.c_int(0)
                pivot = np.arange(aoPairRg.shape[1] + aoPairPntPotential.shape[1], dtype=np.int32)
                thread_buffer = np.ndarray(
                    (nThread+1, aoPairRg.shape[1] + aoPairPntPotential.shape[1] + 1), dtype=IO_buf.dtype, buffer=buf, offset=offset_thread_buf)
                global_buffer = np.ndarray(
                    (1, aoPairRg.shape[1] + aoPairPntPotential.shape[1]), dtype=IO_buf.dtype, buffer=buf, offset=offset_global_buf)
                fn_colpivot_qr(Q.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(naux_random2),
                                ctypes.c_int(aoPairRg.shape[1] + aoPairPntPotential.shape[1]),
                                ctypes.c_int(max_rank),
                                ctypes.c_double(cutoff),
                                pivot.ctypes.data_as(ctypes.c_void_p),
                                R.ctypes.data_as(ctypes.c_void_p),
                                ctypes.byref(npt_find),
                                thread_buffer.ctypes.data_as(ctypes.c_void_p),
                                global_buffer.ctypes.data_as(ctypes.c_void_p))

                npt_find = npt_find.value

                # update parameter #
                print(R[npt_find-1, npt_find-1:])
                cutoff = abs(R[npt_find-1, npt_find-1])

                print("p0 = %d, p1 = %d, npt_find = %d, cutoff = %12.6e" % (p0, p1, npt_find, cutoff))

                slices = np.array(pivot[:npt_find], dtype=np.int32)
                OrthoBasis = np.ndarray((naux_random2, npt_find), dtype=IO_buf.dtype,
                                        buffer=buf, offset=offset_OrthoBasis)
                lib.dslice(Q, slices, out=OrthoBasis)
                aoPairRg_new = np.ndarray((naux_random2, npt_find), dtype=IO_buf.dtype,
                                          buffer=buf, offset=offset_aoPairRg)

                IP_Now.extend(list(grid_ID))
                IP_Now = np.array(IP_Now, dtype=np.int32)
                IP_Now = IP_Now[slices]

                SlicePackBuf  = np.ndarray((naux_random2, npt_find), dtype=np.int32,
                                           buffer=buf, offset=offset_SlicePack)
                thread_buffer = np.ndarray((nThread, naux + bunchsize), dtype=IO_buf.dtype,
                                           buffer=buf, offset=offset_thread_buf)

                fn_packAB_ABSlices(
                    aoPairRg.ctypes.data_as(ctypes.c_void_p),
                    aoPairPntPotential.ctypes.data_as(ctypes.c_void_p),
                    aoPairRg_new.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux_random2),
                    ctypes.c_int(aoPairRg.shape[1]),
                    ctypes.c_int(aoPairPntPotential.shape[1]),
                    slices.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(slices.shape[0]),
                    SlicePackBuf.ctypes.data_as(ctypes.c_void_p),
                    thread_buffer.ctypes.data_as(ctypes.c_void_p))

                # reset the parameter

                nPntAddedPotential = None
                nPntPotential = 0
                PntPotential = []

                if p0 == ngrids:
                    break

                # update the loop parameter #
                p0 = p1
                p1 = min(p0+bunchsize, ngrids)

            else:

                if p0 == ngrids:
                    break

                p0 = p1
                p1 = min(p0+left_pnt, ngrids)

        # judge whether to break the loop

        if p0 == p1:
            break

    IP_Now.sort()

    return IP_Now

# @profile
def _select_IP_direct(mydf:isdf_fast.PBC_ISDF_Info, c:int, m:int):

    bunchsize = lib.num_threads()

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
    nthread = min(nthread, natm)

    # buf_size = (nao*ngrid_on_atm                        # aoR_atm
    #             + nao*nao                               # Q
    #             + nao*ngrid_on_atm                      # R
    #             + naux_max*ngrid_on_atm*2               # aoR_atm1, aoR_atm2
    #             + naux_max*naux_max*ngrid_on_atm        # aoPairBuffer
    #             ) * nthread

    buf_size_per_thread = mydf.get_buffer_size_in_IP_selection(c, m)
    # buf_size          = buf_size_per_thread * nthread
    buf_size            = buf_size_per_thread

    print("nthread        = ", nthread)
    print("buf_size       = ", buf_size)
    print("buf_per_thread = ", buf_size//nthread)

    buf = mydf.IO_buf

    if buf.size < buf_size:
        # reallocate
        mydf.IO_buf = np.zeros((buf_size), dtype=np.float64)
        print("reallocate buf of size = ", buf_size)
        buf = mydf.IO_buf
    buf_tmp = np.ndarray((buf_size), dtype=np.float64, buffer=buf)

    ### loop over atm ###

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(mydf.cell)
    NumInts = df_tmp._numint
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    # coords  = mydf.coords
    assert coords is not None

    results = []

    fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])

    # for p0, p1 in lib.prange(0, natm, nthread):
    for p0, p1 in lib.prange(0, 1, 1):

        taskinfo = []

        # clear buffer

        for atm_id in range(natm):

            buf_tmp[:buf_size_per_thread] = 0.0

            grid_ID = np.where(mydf.partition == atm_id)[0]

            offset  = 0
            aoR_atm = np.ndarray((nao, grid_ID.shape[0]), dtype=np.complex128, buffer=buf_tmp, offset=offset)
            # aoR_atm_tmp = NumInts.eval_ao(mydf.cell, coords[grid_ID])[0]
            aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[grid_ID], out=aoR_atm) * weight
            # print("aoR_atm.shape = ", aoR_atm.shape)

            # create buffer for this atm

            dtypesize = buf.dtype.itemsize

            # offset = buf_size_per_thread * (atm_id - p0) * dtypesize
            # aoR_atm = np.ndarray((grid_ID.shape[0], nao), dtype=np.float64, buffer=buf_tmp, offset=offset)
            # _copy_b_to_a(aoR_atm, aoR_atm_tmp, aoR_atm_tmp.size)
            # print(aoR_atm[0])

            offset += nao*grid_ID.shape[0] * dtypesize

            nao_atm  = nao_per_atm[atm_id]
            naux_now = int(np.sqrt(c*nao_atm)) + m
            naux2_now = naux_now * naux_now

            # Q = np.ndarray((naux2_now, naux2_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            # offset += naux2_now*naux2_now * dtypesize
            # Q = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            # offset += naux2_now*grid_ID.shape[0] * dtypesize

            # R = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            # offset += naux2_now*grid_ID.shape[0] * dtypesize
            R = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux2_now*grid_ID.shape[0] * dtypesize

            # aoR_atm1 = np.ndarray((grid_ID.shape[0], naux_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            aoR_atm1 = np.ndarray((naux_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*grid_ID.shape[0] * dtypesize

            # aoR_atm2 = np.ndarray((grid_ID.shape[0], naux_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            aoR_atm2 = np.ndarray((naux_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*grid_ID.shape[0] * dtypesize

            # aoPairBuffer = np.ndarray(
            #     (grid_ID.shape[0], naux_now*naux_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            aoPairBuffer = np.ndarray(
                (naux_now*naux_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*naux_now*grid_ID.shape[0] * dtypesize

            # task_now = (grid_ID, aoR_atm, nao, nao_atm, c, m)
            # task_buffer = (Q, R, aoR_atm1, aoR_atm2, aoPairBuffer)

            # taskinfo.append(_atm_IP_task(task_now, task_buffer))

            # print("nao_atm  = ", nao_atm)
            # print("naux_now = ", naux_now)

            G1 = np.random.rand(nao, naux_now)
            G1, _ = numpy.linalg.qr(G1)
            G1    = G1.T
            G2 = np.random.rand(nao, naux_now)
            G2, _ = numpy.linalg.qr(G2)
            G2    = G2.T

            # print("G1.shape = ", G1.shape)
            # print("G2.shape = ", G2.shape)
            # print("aoR_atm.shape = ", aoR_atm.shape)
            # print("aoR_atm1.shape = ", aoR_atm1.shape)
            # print("aoR_atm2.shape = ", aoR_atm2.shape)

            lib.dot(G1, aoR_atm, c=aoR_atm1)
            lib.dot(G2, aoR_atm, c=aoR_atm2)

            fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                         aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                         aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux_now),
                         ctypes.c_int(naux_now),
                         ctypes.c_int(grid_ID.shape[0]))

            max_rank      = min(naux2_now, grid_ID.shape[0], nao_atm * c)
            npt_find      = ctypes.c_int(0)
            pivot         = np.arange(grid_ID.shape[0], dtype=np.int32)
            thread_buffer = np.ndarray((nthread+1, grid_ID.shape[0]+1), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset       += (nthread+1)*(grid_ID.shape[0]+1) * dtypesize
            global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset       += grid_ID.shape[0] * dtypesize

            # Q_bench = np.ndarray((naux2_now, naux2_now), dtype=np.float64)
            # R_bench = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64)
            # aoPair_bench = (aoPairBuffer.T).copy()
            # IP_bench = _colpivot_qr(aoPair_bench, Q_bench, R_bench, max_rank=max_rank)
            # IP_bench = np.array(IP_bench, dtype=np.int32)

            fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(naux2_now),
                            ctypes.c_int(grid_ID.shape[0]),
                            ctypes.c_int(max_rank),
                            ctypes.c_double(1e-12),
                            pivot.ctypes.data_as(ctypes.c_void_p),
                            R.ctypes.data_as(ctypes.c_void_p),
                            ctypes.byref(npt_find),
                            thread_buffer.ctypes.data_as(ctypes.c_void_p),
                            global_buffer.ctypes.data_as(ctypes.c_void_p))
            npt_find = npt_find.value
            cutoff   = abs(R[npt_find-1, npt_find-1])
            # print(R[npt_find-1, npt_find-1:])
            print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (grid_ID.shape[0], npt_find, cutoff))
            pivot = pivot[:npt_find]

            # IP_bench.sort()
            pivot.sort()

            # print("IP_bench = ", IP_bench)
            # print("pivot    = ", pivot)

            results.extend(list(grid_ID[pivot]))

        # res_tmp = da.compute(*taskinfo, scheduler='threads')
        # for res in res_tmp:
            # print("res = ", res)
            # results.extend(res)

    # print("results = ", results)
    results.sort()

    return results

##################################### get_jk #####################################

def _get_jk_dm_IO(mydf, dm):
    '''
    all the intermediates and the calculation procedure is defined as follows:

    (D_1)_{\mu,\mathbf{R}} = D_{\mu \nu} \phi_{\mu \mathbf{R}}

    (D_2)_{\mathbf{R}}=\sum_{\mu}\phi_{\mu,\mathbf{R}}(D_1)_{\mu,\mathbf{R}}

    (D_3)_{\mathbf{R}_g}$, slices of $D_2

    (J_1)_{\mathbf{R}_g} = V_{\mathbf{R}_g,\mathbf{R}}(D_2)_{\mathbf{R}}

        (J_2)_{\mu,\mathbf{R}_g} = \phi_{\nu,\mathbf{R}_g}(J_1)_{\mathbf{R}_g}

        (J_3)_{\mu,\nu} = \phi_{\mu,\mathbf{R}_g}(J_2)_{\nu,\mathbf{R}_g}

        (J_4)_{\mathbf{R}} = V_{\mathbf{R}_g,\mathbf{R}}(D_3)_{\mathbf{R}_g}

        (J_5)_{\mu,\mathbf{R}} = \phi_{\nu,\mathbf{R}}(J_4)_{\mathbf{R}}

        (J_6)_{\mu,\nu} = \phi_{\mu,\mathbf{R}}(J_5)_{\nu,\mathbf{R}}

        (W_1)_{\mathbf{R}_g} = W_{\mathbf{R}_g,\mathbf{R}_g'} (D_3)_{\mathbf{R}_g'}

        (W_2)_{\mu,\mathbf{R}_g} = \phi_{\mu,\mathbf{R}_g} (W_1)_{\mathbf{R}_g}

        (W_3)_{\mu,\nu} = \phi_{\mu,\mathbf{R}_g} (W_2)_{\nu,\mathbf{R}_g}

    (D_4)_{\mathbf{R}_g,\mathbf{R}}=\phi_{\mu,\mathbf{R}_g}(D_1)_{\mu,\mathbf{R}}

        (D_5)_{\mathbf{R}_g,\mathbf{R}_g}$, a slice of $D_4

    (K_1)_{\mathbf{R}_g,\mathbf{R}} = V_{\mathbf{R}_g,\mathbf{R}}(D_4)_{\mathbf{R}_g,\mathbf{R}}

    (K_2)_{\mathbf{R}_g,\nu} = V_{\mathbf{R}_g,\mathbf{R}}\phi_{\nu,\mathbf{R}}

    (K_3)_{\mu,\nu} = (K_2)_{\mathbf{R}_g,\nu}\phi_{\mu,\mathbf{R}_g}

    (W_4)_{\mathbf{R}_g,\mathbf{R}_g}=(D_5)_{\mathbf{R}_g,\mathbf{R}_g} W_{\mathbf{R}_g,\mathbf{R}_g}

    (W_5)_{\mathbf{R}_g,\nu}

    (W_6)_{\mu,\nu}

    '''

    all_t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    J = np.zeros((dm.shape[0], dm.shape[1]), dtype=np.float64)
    K = np.zeros((dm.shape[0], dm.shape[1]), dtype=np.float64)

    # for i in range(dm.shape[0]):
    #     for j in range(dm.shape[1]):
    #         if abs(dm[i,j]) > 1e-6:
    #             print("dm[%d,%d] = %12.6f" % (i,j,dm[i,j]))

    nao   = dm.shape[0]

    cell  = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    assert ngrid == mydf.ngrids
    vol   = cell.vol

    W     = mydf.W
    aoRg  = mydf.aoRg
    aoR   = mydf.aoR
    V_R   = mydf.V_R
    naux  = aoRg.shape[1]
    IP_ID = mydf.IP_ID

    coords = mydf.coords
    NumInts = mydf._numint

    buf = mydf.IO_buf
    buf_size = buf.size

    print("buf_size = ", buf_size)

    weight = np.sqrt(vol / ngrid)

    ##### construct buffer #####

    buf_size_fixed = 2 * naux + 2 * ngrid + naux * naux  # to store the temporaries J1 J4 D3 and D2 D5

    if buf_size_fixed > buf_size:
        print("buf_size_fixed requires %d buf %d provide" % (buf_size_fixed, buf_size))
        raise RuntimeError

    offset = 0
    J1     = np.ndarray((naux,), dtype=np.float64, buffer=buf)

    offset += naux * buf.dtype.itemsize
    J4      = np.ndarray((ngrid,), dtype=np.float64, buffer=buf, offset=offset)

    offset += ngrid * buf.dtype.itemsize
    D3      = np.ndarray((naux,), dtype=np.float64, buffer=buf, offset=offset)

    offset += naux * buf.dtype.itemsize
    D2      = np.ndarray((ngrid,), dtype=np.float64, buffer=buf, offset=offset)

    offset += ngrid * buf.dtype.itemsize

    ### STEP 1 construct rhoR and prefetch V1 ###

    if isinstance(mydf.IO_FILE, str):
        if h5py.is_hdf5(mydf.IO_FILE):
            f_dataset = h5py.File(mydf.IO_FILE, 'r')
            assert (V_DATASET in f_dataset)
            assert (AOR_DATASET in f_dataset)
        else:
            raise ValueError("IO_File should be a h5py.File object")
    else:
        assert (isinstance(mydf.IO_FILE, h5py.Group))
        f_dataset = mydf.IO_FILE

    def load_V_async(col0, col1, buf:np.ndarray):
        if col0 < col1:
            source_sel   = np.s_[:, col0:col1]
            dset_sel     = np.s_[:, :col1-col0]
            f_dataset[V_DATASET].read_direct(buf, source_sel=source_sel, dest_sel=dset_sel)

    def load_aoR(col0, col1, buf:np.ndarray):  # loop over grids
        if col0 < col1:
            source_sel   = np.s_[:, col0:col1]
            dset_sel     = np.s_[:, :col1-col0]
            f_dataset[AOR_DATASET].read_direct(buf, source_sel=source_sel, dest_sel=dset_sel)

    def load_aoR_async(col0, col1, buf:np.ndarray):
        if col0 < col1:
            source_sel   = np.s_[:, col0:col1]
            dset_sel     = np.s_[:, :col1-col0]
            f_dataset[AOR_DATASET].read_direct(buf, source_sel=source_sel, dest_sel=dset_sel)

    offset_D5 = offset
    D5        = np.ndarray((naux,naux), dtype=np.float64, buffer=buf, offset=offset)
    offset   += naux * naux * buf.dtype.itemsize

    print("offset_D5 = ", offset_D5//8)

    if (buf_size - offset//8) < naux * nao:
        print("buf is not sufficient since (%d - %d) < %d * %d" % (buf_size, offset, naux, nao))

    offset_K2 = offset
    size_K2   = naux * nao
    K2      = np.ndarray((naux, nao), dtype=np.float64, buffer=buf, offset=offset)
    offset += size_K2 * buf.dtype.itemsize

    bunchsize_readV = (buf_size - offset//8 - size_K2//8) // (naux + naux + nao + naux + nao*2 + nao + nao)
    bunchsize_readV = min(bunchsize_readV, ngrid)
    if naux < MAX_BUNCHSIZE:  # NOTE: we cannot load too large chunk of grids
        bunchsize_readV = min(bunchsize_readV, MAX_BUNCHSIZE)
    else:
        bunchsize_readV = min(bunchsize_readV, naux)

    if hasattr(mydf, 'bunchsize_readV'):
        bunchsize_readV = mydf.bunchsize_readV

    print("bunchsize_readV = ", bunchsize_readV)

    offset_V1 = offset
    V1  = np.ndarray((naux, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset)
    offset += naux * bunchsize_readV * buf.dtype.itemsize

    offset_aoR1 = offset
    aoR1 = np.ndarray((nao, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset)
    offset += nao * bunchsize_readV * buf.dtype.itemsize

    nElmt_left     = buf_size - offset // buf.dtype.itemsize
    grid_bunchsize = nElmt_left // nao // 3
    grid_bunchsize = min(grid_bunchsize, ngrid)
    # if nao < MAX_BUNCHSIZE: # NOTE: we cannot load too large chunk of grids
    #     grid_bunchsize = min(grid_bunchsize, MAX_BUNCHSIZE)
    # else:
    #     grid_bunchsize = min(grid_bunchsize, nao)

    if hasattr(mydf, 'grid_bunchsize'):
        grid_bunchsize = mydf.grid_bunchsize
    print("grid_bunchsize = ", grid_bunchsize)

    offset_AoR1 = offset
    offset_AoR2 = offset + nao * grid_bunchsize * buf.dtype.itemsize
    offset_AoR3 = offset + 2 * nao * grid_bunchsize * buf.dtype.itemsize

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    with lib.call_in_background(load_V_async) as prefetch:
        with lib.call_in_background(load_aoR_async) as prefetch_aoR:
            prefetch(0, bunchsize_readV, V1)
            prefetch_aoR(0, bunchsize_readV, aoR1)

            AoR_Buf1 = np.ndarray((nao, grid_bunchsize), dtype=np.float64, buffer=buf, offset=offset_AoR1)
            load_aoR(0, grid_bunchsize, AoR_Buf1)

            for p0, p1 in lib.prange(0, ngrid, grid_bunchsize):

                AoR_Buf2 = np.ndarray((nao, min(p1+grid_bunchsize, ngrid)-p1),
                                      dtype=np.float64, buffer=buf, offset=offset_AoR2)
                prefetch_aoR(p1, min(p1+grid_bunchsize, ngrid), AoR_Buf2)

                AoR_Buf3 = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=buf, offset=offset_AoR3)

                lib.ddot(dm, AoR_Buf1, c=AoR_Buf3)
                lib.multiply_sum_isdf(AoR_Buf1, AoR_Buf3, out=D2[p0:p1])

                ## swap

                AoR_Buf1, AoR_Buf2 = AoR_Buf2, AoR_Buf1
                offset_AoR1, offset_AoR2 = offset_AoR2, offset_AoR1

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "construct D2")

    lib.dslice(D2, IP_ID, D3)
    # print("D3 = ", D3)
    # print("D2 = ", D2[naux:])

    ### STEP 2 Read V_R ###

    #### 2.1 determine the bunch size ####

    offset_V2 = offset
    V2        = np.ndarray((naux, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset_V2)
    offset   += naux * bunchsize_readV * buf.dtype.itemsize

    offset_D1  = offset
    D1         = np.ndarray((nao, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset_D1)
    offset    += nao * bunchsize_readV * buf.dtype.itemsize

    offset_D4  = offset
    D4         = np.ndarray((naux, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset_D4)
    offset    += naux * bunchsize_readV * buf.dtype.itemsize

    offset_aoR2  = offset
    aoR2         = np.ndarray((nao, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset_aoR2)
    offset      += nao * bunchsize_readV * aoR2.dtype.itemsize

    offset_aoR3  = offset
    aoR3         = np.ndarray((nao, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset_aoR3)  # used in copy
    offset      += nao * bunchsize_readV * aoR3.dtype.itemsize

    offset_J_tmp = offset
    J_tmp        = np.ndarray((nao, bunchsize_readV), dtype=np.float64, buffer=buf, offset=offset_J_tmp)

    #### 2.2 read V_R ####

    mydf.set_bunchsize_ReadV(bunchsize_readV)
    IP_partition = mydf.get_IP_partition()

    # print(IP_partition)

    print("bunchsize_readV = ", bunchsize_readV)

    t1_ReadV = (lib.logger.process_clock(), lib.logger.perf_counter())

    nThread = lib.num_threads()
    if hasattr(mydf, 'ddot_buf') and mydf.ddot_buf is not None:
        if mydf.ddot_buf.size < (naux * nao + 2) * nThread:
            print("reallocate ddot_buf of size = ", (naux * nao + 2) * nThread)
            mydf.ddot_buf = np.zeros(((naux * nao + 2) * nThread), dtype=np.float64)  # reallocate
    else:
        print("allocate ddot_buf of size = ", (naux * nao + 2) * nThread)
        mydf.ddot_buf = np.zeros(((naux * nao + 2) * nThread), dtype=np.float64)

    with lib.call_in_background(load_V_async) as prefetch:
        with lib.call_in_background(load_aoR_async) as prefetch_aoR:

            for ibunch, (p0, p1) in enumerate(lib.prange(0, ngrid, bunchsize_readV)):

                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

                V2   = np.ndarray((naux, min(p1+bunchsize_readV,ngrid)-p1),
                                  dtype=np.float64, buffer=buf, offset=offset_V2)
                aoR2 = np.ndarray((nao,  min(p1+bunchsize_readV,ngrid)-p1),
                                  dtype=np.float64, buffer=buf, offset=offset_aoR2)
                prefetch(p1, min(p1+bunchsize_readV,ngrid), V2)
                prefetch_aoR(p1, min(p1+bunchsize_readV,ngrid), aoR2)

                # construct D1

                D1 = np.ndarray((nao, p1-p0),  dtype=np.float64, buffer=buf, offset=offset_D1)
                D4 = np.ndarray((naux, p1-p0), dtype=np.float64, buffer=buf, offset=offset_D4)

                lib.ddot(dm, aoR1, c=D1)
                lib.ddot(aoRg.T, D1, c=D4)

                # add J1, J4

                beta = 1
                if ibunch == 0:
                    beta = 0

                lib.ddot_withbuffer(V1, D2[p0:p1].reshape(-1,1), c=J1.reshape(-1,1), beta=beta, buf=mydf.ddot_buf)
                # NOTE we do not need another loop for J4
                lib.ddot_withbuffer(D3.reshape(1,-1), V1, c=J4[p0:p1].reshape(1,-1), buf=mydf.ddot_buf)

                aoR3  = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=buf, offset=offset_aoR3)  # used in copy
                lib.copy(aoR1, out=aoR3)
                J_tmp = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=buf, offset=offset_J_tmp)
                lib.d_ij_j_ij(aoR3, J4[p0:p1], out=J_tmp)
                lib.ddot_withbuffer(aoR1, J_tmp.T, c=J, beta=beta, buf=mydf.ddot_buf)

                # pack D5

                offset = IP_partition[ibunch][0]
                slices = IP_partition[ibunch][1]
                lib.dslice_offset(D4, locs=slices, offset=offset, out=D5)

                # construct K1 inplace

                lib.cwise_mul(V1, D4, out=D4)
                K1 = D4
                lib.ddot_withbuffer(K1, aoR1.T, c=K2, buf=mydf.ddot_buf)
                lib.ddot_withbuffer(aoRg, K2, c=K, beta=beta, buf=mydf.ddot_buf)

                # switch

                V1, V2 = V2, V1
                offset_V1, offset_V2 = offset_V2, offset_V1

                aoR1, aoR2 = aoR2, aoR1
                offset_aoR2, offset_aoR1 = offset_aoR1, offset_aoR2

                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                print("ibunch %3d construct J[%8d:%8d] wall time: %12.6f CPU time: %12.6f" %
                      (ibunch, p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

    t2_ReadV = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1_ReadV, t2_ReadV, "ReadV")

    #### 2.3 construct K's W matrix ####

    K += K.T

    lib.cwise_mul(W, D5, out=D5)
    lib.ddot(D5, aoRg.T, c=K2)
    lib.ddot_withbuffer(aoRg, -K2, c=K, beta=1, buf=mydf.ddot_buf)

    #### 2.4 construct J ####

    J2 = np.ndarray((nao,naux), dtype=np.float64, buffer=buf, offset=offset_D5)
    lib.d_ij_j_ij(aoRg, J1, out=J2)
    lib.ddot_withbuffer(aoRg, J2.T, c=J, beta=1, buf=mydf.ddot_buf)

    #### 2.5 construct J's W matrix ####

    J_W_tmp1 = np.ndarray((naux,), dtype=np.float64, buffer=buf, offset=offset_K2)
    offset_2 = offset_K2 + naux * buf.dtype.itemsize
    J_W_tmp2 = np.ndarray((nao, naux), dtype=np.float64, buffer=buf, offset=offset_2)

    lib.ddot(W, D3.reshape(-1,1), c=J_W_tmp1.reshape(-1,1))
    lib.d_ij_j_ij(aoRg, J_W_tmp1, out=J_W_tmp2)
    lib.ddot_withbuffer(aoRg, -J_W_tmp2.T, c=J, beta=1, buf=mydf.ddot_buf)

    all_t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(all_t1, all_t2, "get_jk")

    return J * ngrid / vol, K * ngrid / vol

def get_jk_dm_IO(mydf, dm, hermi=1, kpt=np.zeros(3),
                 kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
    '''JK for given k-point'''

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    assert with_j is True and with_k is True

    #### explore the linearity of J K with respect to dm ####

    if mydf.direct_scf == False:
        if mydf._cached_dm is None:
            mydf._cached_dm = dm
            mydf._cached_j = None
            mydf._cached_k = None
        else:
            if mydf._cached_j is None and with_j == True or \
               mydf._cached_k is None and with_k == True:
                # recalculate the J or K
                mydf._cached_j = None
                mydf._cached_k = None
            else:
                assert(mydf._cached_dm.shape == dm.shape)
                dm = dm - mydf._cached_dm
                mydf._cached_dm += dm

    # if mydf._explore_sparsity:
    #     nNonZero = mydf.process_dm(dm)
    #     print("dm nNonZero   = %10.2f %%" % (nNonZero / dm.size * 100.0))
    #     # if small_comp == dm.size:
    #     if nNonZero == 0:
    #         # nothing to do
    #         assert mydf._cached_j is not None
    #         assert mydf._cached_k is not None
    #         return mydf._cached_j, mydf._cached_k
    #     if nNonZero < dm.size * 0.1:
    #         mydf.compress_dm(dm)
    #     else:
    #         mydf.dm_compressed = False

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

    # if with_j:
    #     vj = _contract_j_dm(mydf, dm)
    # if with_k:
    #     vk = _contract_k_dm(mydf, dm)
    #     if exxdiv == 'ewald':
    #         # raise NotImplemented("ISDF does not support ewald")
    #         print("WARNING: ISDF does not support ewald")

    vj, vk = _get_jk_dm_IO(mydf, dm)


    t1 = log.timer('sr jk', *t1)

    #### explore the linearity of J K with respect to dm ####

    if mydf.direct_scf == False:
        if with_j and mydf._cached_j is None:
            mydf._cached_j = vj
        elif with_j:
            mydf._cached_j += vj

        if with_k and mydf._cached_k is None:
            mydf._cached_k = vk
        elif with_k:
            mydf._cached_k += vk

    # return vj, vk

    return mydf._cached_j, mydf._cached_k

class PBC_ISDF_Info_IO(isdf_fast.PBC_ISDF_Info):
    def __init__(self, mol:Cell, max_buf_memory:int):
        super().__init__(mol=mol)

        self.max_buf_memory = max_buf_memory
        self.IO_buf         = np.zeros((max_buf_memory//8), dtype=np.float64)
        self.IO_FILE        = None

        self._cached_bunchsize_ReadV = None
        self._cached_IP_partition    = None

        self.mesh = mol.mesh

        print("self.mesh = ", self.mesh)

    def __del__(self):

        if self.IO_FILE is not None:
            os.system("rm %s" % (self.IO_FILE))

    def set_bunchsize_ReadV(self, input:int):
        if self._cached_bunchsize_ReadV is None or self._cached_bunchsize_ReadV != input:
            self._cached_bunchsize_ReadV = input
            self._cached_IP_partition = []

            nBunch = self.ngrids // self._cached_bunchsize_ReadV
            if self.ngrids % self._cached_bunchsize_ReadV != 0:
                nBunch += 1

            loc_IP_ID = 0

            for i in range(nBunch):
                p0 = i * self._cached_bunchsize_ReadV
                p1 = min(p0 + self._cached_bunchsize_ReadV, self.ngrids)
                # self._cached_IP_partition.append((p0, p1))

                offset    = loc_IP_ID
                partition = []

                while loc_IP_ID < self.naux and self.IP_ID[loc_IP_ID] < p1:
                    partition.append(self.IP_ID[loc_IP_ID]-p0)
                    loc_IP_ID += 1

                partition = np.array(partition, dtype=np.int32)

                self._cached_IP_partition.append((offset, partition))

    def get_IP_partition(self):
        assert (self._cached_IP_partition is not None)
        return self._cached_IP_partition

    def build_IP_Sandeep_IO(self, IO_File:str = None, c:int = 5, m:int = 5):

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.IP_ID = _select_IP_direct(self, c, m)
        self.IP_ID = np.asarray(self.IP_ID, dtype=np.int32)
        print("IP_ID = ", self.IP_ID)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "select IP")

        if IO_File is None:
            # generate a random file name start with tmp_
            import random
            import string
            IO_File = "tmp_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)) + ".hdf5"

        print("IO_File = ", IO_File)

        # construct aoR

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        coords_IP = self.coords[self.IP_ID]
        weight    = np.sqrt(self.cell.vol / self.ngrids)
        self.aoRg = self._numint.eval_ao(self.cell, coords_IP)[0].T * weight  # the T is important
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aoR")

        self.naux = self.aoRg.shape[1]
        self.c    = c

        print("naux = ", self.naux)

        self.chunk_size, self.nRow_IO_V, self.blksize_aux, self.bunchsize_readV, self.grid_bunchsize, self.blksize_W, self.use_large_chunk_W  = _determine_bunchsize(
            self.nao, self.naux, self.mesh, self.IO_buf.size)

        # construct aux basis

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _construct_aux_basis_IO(self, IO_File, self.IO_buf)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aux basis")

        self.IO_FILE = IO_File

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
        # buf_size += naux_max2*ngrid_on_atm                     # Q
        buf_size += naux_max2*ngrid_on_atm                     # R
        buf_size += naux_max*ngrid_on_atm*2                    # aoR_atm1, aoR_atm2
        buf_size += naux_max*naux_max*ngrid_on_atm             # aoPairBuffer
        buf_size += (nThread+1)*(ngrid_on_atm+1)
        buf_size += ngrid_on_atm

        return max(buf_size, 2*self.nao*ngrid_on_atm)

    def build_auxiliary_Coulomb_IO(self, mesh):

        mesh = np.asarray(mesh, dtype=np.int32)
        self.mesh = np.asarray(self.mesh, dtype=np.int32)

        print("mesh      = ", mesh)
        print("self.mesh = ", self.mesh)

        if mesh[0] != self.mesh[0] or mesh[1] != self.mesh[1] or mesh[2] != self.mesh[2]:
            print("warning: mesh is not consistent with the previous one")

        self.mesh = mesh
        self.W = np.zeros((self.naux, self.naux), dtype=np.float64)

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _construct_V_W_IO2(self, mesh, self.IO_FILE, self.IO_buf)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct V and W")

    get_jk = get_jk_dm_IO

C = 10

if __name__ == "__main__":

    import copy

    # test colqr

    # m = 512
    # n = 8192
    # MAX_BANK = 64
    # AA = np.random.rand(n,m)
    # AAA = np.zeros((m,n), dtype=np.double)
    # for i in range(m):
    #     for j in range(n):
    #         AAA[i,j] = AA[j,i]
    # Q = np.zeros((m, m))
    # R = np.zeros((m, n))
    # t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    # IP_bench = _colpivot_qr(AA, Q, R, max_rank=MAX_BANK)
    # t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    # _benchmark_time(t1, t2, "colpivot_qr")
    # t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    # IP_C = _colpivot_qr_C(AAA, max_rank=MAX_BANK)
    # t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    # _benchmark_time(t1, t2, "colpivot_qr_C")
    # IP_bench.sort()
    # IP_C.sort()
    # IP_bench = np.array(IP_bench, dtype=np.int32)
    # IP_C     = np.array(IP_C, dtype=np.int32)
    # print("IP_bench = ", IP_bench)
    # print("IP_C     = ", IP_C)
    # assert np.allclose(IP_bench, IP_C)

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
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 256   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 128
    # cell.ke_cutoff = 48
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

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("address of aoR = ", ctypes.addressof(aoR.ctypes.data_as(ctypes.c_void_p)))
    out = np.ndarray(aoR.shape, dtype=np.complex128)
    print("address of out = ", ctypes.addressof(out.ctypes.data_as(ctypes.c_void_p)))
    # out  = df_tmp._numint.eval_ao(cell, coords, out=out)[0]
    out = ISDF_eval_gto(cell, coords=coords, out=out)
    out *= np.sqrt(cell.vol / ngrids)
    print("address of out = ", ctypes.addressof(out.ctypes.data_as(ctypes.c_void_p)))
    # out  = out.T
    print(np.allclose(aoR, out))

    if np.allclose(aoR, out) == False:
        diff = aoR - out
        ## find where is the large difference
        idx = np.where(np.abs(diff) > 1e-8)
        ## print
        print("idx = ", idx)
        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])

    print("address of out = ", ctypes.addressof(out.ctypes.data_as(ctypes.c_void_p)))

    print("aoR.shape = ", aoR.shape)
    print("out.shape = ", out.shape)
    print("aoR.dtype = ", aoR.dtype)

    # np.real(aoR, out=aoR)

    buf_complex = np.zeros((20,20), dtype=np.complex128)
    print("address of buf_complex = ", ctypes.addressof(buf_complex.ctypes.data_as(ctypes.c_void_p)))
    buf_A       = np.ndarray((10,10), dtype=np.complex128, buffer=buf_complex, offset=0)

    print("address of buf_A       = ", ctypes.addressof(buf_A.ctypes.data_as(ctypes.c_void_p)))
    print("address of buf_complex = ", ctypes.addressof(buf_complex.ctypes.data_as(ctypes.c_void_p)))

    buf_A.real = np.random.rand(10,10)
    buf_A.imag = np.random.rand(10,10)
    print("address of buf_A       = ", ctypes.addressof(buf_A.ctypes.data_as(ctypes.c_void_p)))
    buf_A      = np.real(buf_A)
    print("address of buf_A       = ", ctypes.addressof(buf_A.ctypes.data_as(ctypes.c_void_p)))

    # exit(1)

    pbc_isdf_info = isdf_fast.PBC_ISDF_Info(cell, None, cutoff_aoValue=1e-6, cutoff_QR=1e-3)
    pbc_isdf_info.aoR = aoR
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=False)
    # pbc_isdf_info.check_AOPairError()

    IO_buf_memory = int(10e7)  # 100M
    IO_buf = np.zeros((IO_buf_memory//8), dtype=np.float64)
    IO_File = "test.h5"

    # exit(1)

    _construct_aux_basis_IO(pbc_isdf_info, IO_File, IO_buf)

    f_aux_basis = h5py.File(IO_File, 'r')
    aux_basis   = f_aux_basis[AUX_BASIS_DATASET]

    print("aux_basis.shape = ", aux_basis.shape)

    if np.allclose(aux_basis, pbc_isdf_info.aux_basis):
        print("PASS")
    else:
        ## print the difference
        diff = aux_basis - pbc_isdf_info.aux_basis
        ## find where is the large difference
        idx = np.where(np.abs(diff) > 1e-8)
        ## print
        print("idx = ", idx)
        # for i in range(idx[0].shape[0]):
        #     print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])

    # exit(1)

    # IO_File = None

    os.system("rm %s" % (IO_File))

    pbc_isdf_info_IO = PBC_ISDF_Info_IO(cell, max_buf_memory=IO_buf_memory)

    # IP_ID = _select_IP_direct2(pbc_isdf_info_IO, C, 5)
    # print(IP_ID)

    # exit(1)

    pbc_isdf_info_IO.build_IP_Sandeep_IO(c=C, IO_File=IO_File)
    pbc_isdf_info_IO.build_auxiliary_Coulomb_IO(mesh)

    f_aux_basis = h5py.File(IO_File, 'r')
    aux_basis   = f_aux_basis[AUX_BASIS_DATASET][:]
    print("aux_basis.shape = ", aux_basis.shape)
    pbc_isdf_info_IO.aux_basis = aux_basis
    pbc_isdf_info_IO.aoR       = aoR
    # pbc_isdf_info_IO.check_AOPairError()

    # exit(1)

    pbc_isdf_info.aoRg      = pbc_isdf_info_IO.aoRg
    pbc_isdf_info.IP_ID     = pbc_isdf_info_IO.IP_ID
    pbc_isdf_info.aux_basis = aux_basis
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)

    if np.allclose(pbc_isdf_info_IO.W, pbc_isdf_info.W):
        print("PASS")
    else:
        ## print the difference
        diff = pbc_isdf_info_IO.W - pbc_isdf_info.W
        ## find where is the large difference
        idx = np.where(np.abs(diff) > 1e-12)
        ## print
        print("idx = ", idx)
        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]],
                  pbc_isdf_info_IO.W[idx[0][i], idx[1][i]], pbc_isdf_info.W[idx[0][i], idx[1][i]])

    V = f_aux_basis[V_DATASET][:]
    V_bench = pbc_isdf_info.V_R

    if np.allclose(V, V_bench):
        print("PASS")
    else:
        ## print the difference
        diff = V - V_bench
        ## find where is the large difference
        idx = np.where(np.abs(diff) > 1e-12)
        ## print
        print("idx = ", idx)
        # for i in range(idx[0].shape[0]):
        #    print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])

    # exit(1)

    dm = np.random.rand(cell.nao, cell.nao)

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    J, K = _get_jk_dm_IO(pbc_isdf_info_IO, dm)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "get_jk_dm_IO")

    pbc_isdf_info.direct_scf = False
    J_bench, K_bench = pbc_isdf_info.get_jk(dm)

    if np.allclose(J, J_bench):
        print("PASS")
    else:
        ## print the difference
        diff = J - J_bench
        ## find where is the large difference
        idx = np.where(np.abs(diff) > 1e-12)
        ## print
        print("idx = ", idx)
        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])

    if np.allclose(K, K_bench):
        print("PASS")
    else:
        ## print the difference
        diff = K - K_bench
        ## find where is the large difference
        idx = np.where(np.abs(diff) > 1e-12)
        ## print
        print("idx = ", idx)
        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])


    ### perform scf ###

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    pbc_isdf_info_IO.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info_IO
    mf.max_cycle = 100
    mf.conv_tol = 1e-7

    # print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()

    mf = scf.RHF(cell)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.kernel()