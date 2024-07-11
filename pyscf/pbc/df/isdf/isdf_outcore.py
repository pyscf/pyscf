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

############ sys module ############

import copy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk
libfft = lib.load_library('libfft')
libpbc = lib.load_library('libpbc')

############ isdf utils ############

import pyscf.pbc.df.isdf.isdf_fast as isdf_fast
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

############ global variables ############

AUX_BASIS_DATASET     = 'aux_basis'  # NOTE: indeed can be over written
AUX_BASIS_FFT_DATASET = 'aux_basis_fft'
V_DATASET             = 'V'
AOR_DATASET           = 'aoR'
MAX_BUNCHSIZE         = 4096
MAX_CHUNKSIZE         = int(3e9//8)  # 2GB
THREAD_BUF_PERCENTAGE = 0.15
CHUNK_COL_LIMITED     = 8192
CHUNK_ROW_LIMITED     = 1024
CHUNK_MINIMAL         = 16000000  # 16MB

############ subroutines --- aux basis V W ############

def _determine_bunchsize(nao, naux, mesh, buf_size, with_robust_fitting=True):

    mesh         = np.asarray(mesh, dtype=np.int32)
    mesh_complex = np.asarray([mesh[0], mesh[1], mesh[2]//2+1], dtype=np.int32)
    ncomplex     = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2
    ngrids       = mesh[0] * mesh[1] * mesh[2]
    nThread      = lib.num_threads()

    #### 1. calculate the bunchsize for the construction of aux basis ####

    if with_robust_fitting:
        blksize_aux = (buf_size - naux*naux) // (4 * nao + 2 * naux)   # suppose that the memory is enough
    else:
        blksize_aux = (buf_size - naux*naux) // (2 * nao + 2 * naux)  # we still need one to hold AoR
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

    offset += naux * bunchsize_readV
    offset += nao * bunchsize_readV
    nElmt_left = buf_size - offset
    grid_bunchsize = nElmt_left // nao // 3
    grid_bunchsize = min(grid_bunchsize, ngrids)
    grid_bunchsize = min(grid_bunchsize, MAX_BUNCHSIZE)

    chunk_col = min(bunchsize_readV, grid_bunchsize, blksize_W, blksize_aux)

    if chunk_row * chunk_col * 8 < CHUNK_MINIMAL:  # at least 16MB
        #print("too small chunk, change the chunk col")

        chunk_col_new =  CHUNK_MINIMAL // (chunk_row * 8)
        chunk_col_new = (chunk_col_new // chunk_col + 1) * chunk_col
        chunk_col = chunk_col_new
        chunk_col = min(chunk_col, ngrids)

    if CHUNK_COL_LIMITED is not None:
        chunk_col = min(chunk_col, CHUNK_COL_LIMITED)
    if CHUNK_ROW_LIMITED is not None:
        chunk_row = min(chunk_row, CHUNK_ROW_LIMITED)

    if chunk_row * chunk_col * 8 < CHUNK_MINIMAL:  # at least 16MB
        #print("too small chunk, change the chunk row")
        chunk_row_new = CHUNK_MINIMAL // (chunk_col * 8)
        chunk_row_new = (chunk_row_new // chunk_row + 1) * chunk_row
        chunk_row = chunk_row_new
        chunk_row = min(chunk_row, naux)

    if bunchsize_readV > chunk_col:
        bunchsize_readV = (bunchsize_readV // chunk_col) * chunk_col
    if grid_bunchsize > chunk_col:
        grid_bunchsize  = (grid_bunchsize // chunk_col) * chunk_col
    if blksize_W > chunk_col:
        blksize_W       = (blksize_W // chunk_col) * chunk_col
    if blksize_aux > chunk_col:
        blksize_aux     = (blksize_aux // chunk_col) * chunk_col


    blksize_aux     = (blksize_aux // nThread) * nThread

    return (chunk_row, chunk_col), bunch_size_IO, blksize_aux, bunchsize_readV, grid_bunchsize, blksize_W, use_large_chunk

def _construct_aux_basis(mydf):

    naux = mydf.naux
    mydf._allocate_jk_buffer(datatype=np.double)
    buffer1 = np.ndarray((mydf.naux , mydf.naux), dtype=np.double, buffer=mydf.jk_buffer, offset=0)
    nao = mydf.nao
    aoR = mydf.aoR

    A = np.asarray(lib.ddot(mydf.aoRg.T, mydf.aoRg, c=buffer1), order='C')  # buffer 1 size = naux * naux
    lib.square_inPlace(A)

    mydf.aux_basis = np.asarray(lib.ddot(mydf.aoRg.T, aoR), order='C')   # buffer 2 size = naux * ngrids
    lib.square_inPlace(mydf.aux_basis)

    fn_cholesky = getattr(libpbc, "Cholesky", None)
    assert(fn_cholesky is not None)

    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    fn_cholesky(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux),
    )
    nThread = lib.num_threads()
    nGrids  = aoR.shape[1]
    Bunchsize = nGrids // nThread
    fn_build_aux(
        ctypes.c_int(naux),
        A.ctypes.data_as(ctypes.c_void_p),
        mydf.aux_basis.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nGrids),
        ctypes.c_int(Bunchsize)
    )

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
    chunks  = (IO_buf_memory // (8*2)//ngrids, ngrids)
    blksize = min(blksize, ngrids)
    blksize = min(blksize, MAX_CHUNKSIZE//naux)
    chunks = (naux, blksize)

    #print("blksize       = ", blksize)
    #print("IO_buf_memory = ", IO_buf_memory)
    #print("blksize       = ", blksize)

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
    _benchmark_time(t1,t2,"_aux_basis_IO.Cholesky", mydf)

    chunks1 = (naux, ngrids)
    chunks2 = (nao, ngrids)

    if hasattr(mydf, 'chunk_size'):
        chunks1 = mydf.chunk_size
        chunks2 = (min(mydf.chunk_size[0], nao), mydf.chunk_size[1])

    #print("chunks1        = ", chunks1)
    #print("chunks2        = ", chunks2)

    h5d_aux_basis = f_aux_basis.create_dataset(AUX_BASIS_DATASET, (naux, ngrids), 'f8', chunks=chunks1)

    if mydf.with_robust_fitting:
        h5d_aoR       = f_aux_basis.create_dataset(AOR_DATASET, (nao, ngrids), 'f8', chunks=chunks2)
    else:
        h5d_aoR       = None

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

    if mydf.with_robust_fitting:
        offset       += nao*blksize*IO_buf.dtype.itemsize*2  # complex
        offset_aoR2   = offset
    else:
        offset_aoR2 = None

    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    ## get the coord of grids ##

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
    df_tmp = MultiGridFFTDF2(mydf.cell)
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

                if mydf.with_robust_fitting:
                    AoR_Buf2 = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_aoR2)
                else:
                    AoR_Buf2 = None

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

                # if p0 == 0:
                #     print(buf_calculate[:10,:10])

                async_write(p0, p1, buf_calculate)

                if mydf.with_robust_fitting:
                    async_write_aoR(p0, p1, AoR_Buf1)
                    AoR_Buf1, AoR_Buf2       = AoR_Buf2, AoR_Buf1
                    offset_aoR1, offset_aoR2 = offset_aoR2, offset_aoR1

                buf_write, buf_calculate = buf_calculate, buf_write

    ### close ###

    if isinstance(IO_File, str):
        f_aux_basis.close()

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1,t2,"_aux_basis_IO.build", mydf)

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
    mydf.coulG = coulG.copy()
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)  # drop the minus frequency part
    nThread    = lib.num_threads()

    ### buffer needed ###

    ### two to hold for read in aux_basis to construct V, two to hold another aux_basis to construct W ###
    ### one further buffer to hold the buf to construct V, suppose it consists of 5% of the total memory ###

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

    if hasattr(mydf, 'nRow_IO_V'):
        bunch_size_IO = mydf.nRow_IO_V
        bunchsize     = bunch_size_IO // nThread
        bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
        bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16

    #print("bunchsize = ", bunchsize)
    #print("bunch_size_IO = ", bunch_size_IO)

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

    if mydf.with_robust_fitting:
        CONSTRUCT_V = 1
    else:
        CONSTRUCT_V = 0

    log = lib.logger.Logger(mydf.stdout, mydf.verbose)

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
                       ctypes.c_int(bufsize_per_thread),
                       ctypes.c_int(CONSTRUCT_V))

                    async_write_auxbasisfft(p0, p1, buf_aux_basis_fft_1)
                    buf_aux_basis_fft_1, buf_aux_basis_fft_2 = buf_aux_basis_fft_2, buf_aux_basis_fft_1
                    
                    if CONSTRUCT_V == 1:
                        async_write(p0, p1, buf_V1)
                        buf_V1, buf_V2 = buf_V2, buf_V1

                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    #print("construct V[%5d:%5d] wall time: %12.6f CPU time: %12.6f" %
                    #    (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

                    log.debug('construct V[%5d:%5d] wall time: %12.6f CPU time: %12.6f' %
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
    
    # mydf.coulG_real = coulG_real
    # mydf.coulG = coulG.copy()

    # print("coulG_real = ", coulG_real)

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

            # print("buf_aux_basis_fft_copy = ", buf_aux_basis_fft_copy)

            ## ddot with buf_aux_basis_fft_2

            if use_large_chunk:
                lib.ddot_withbuffer(buf_aux_basis_fft_copy, buf_aux_basis_fft_1.T, buf=ddot_buffer, c=mydf.W, beta=1)
            else:
                lib.ddot(buf_aux_basis_fft_copy, buf_aux_basis_fft_1.T, c=mydf.W, beta=1)

            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

            #print("construct W[%5d:%5d] wall time: %12.6f CPU time: %12.6f" % (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))
            log.debug('construct W[%5d:%5d] wall time: %12.6f CPU time: %12.6f' % (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

    factor = 1.0 / np.prod(mesh)
    mydf.W *= factor

############ subroutines --- get_jk ############

def _get_jk_dm_outcore(mydf, dm):
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

    log = logger.Logger(mydf.stdout, mydf.verbose)

    all_t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    J = np.zeros((dm.shape[0], dm.shape[1]), dtype=np.float64)
    K = np.zeros((dm.shape[0], dm.shape[1]), dtype=np.float64)

    nao   = dm.shape[0]

    cell  = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    assert ngrid == mydf.ngrids
    vol   = cell.vol

    W     = mydf.W
    aoRg  = mydf.aoRg
    naux  = aoRg.shape[1]
    IP_ID = mydf.IP_ID

    coords = mydf.coords
    NumInts = mydf._numint

    buf = mydf.IO_buf
    buf_size = buf.size

    #print("buf_size = ", buf_size)

    weight = np.sqrt(vol / ngrid)

    ##### construct buffer #####

    buf_size_fixed = 2 * naux + 2 * ngrid + naux * naux  # to store the temporaries J1 J4 D3 and D2 D5

    if buf_size_fixed > buf_size:
        print("buf_size_fixed requires %d buf %d provide" % (buf_size_fixed, buf_size))
        raise RuntimeError

    offset = 0
    J1     = np.ndarray((naux,), dtype=np.float64, buffer=buf)
    # J1      = np.zeros((naux,), dtype=np.float64)

    offset += naux * buf.dtype.itemsize
    J4      = np.ndarray((ngrid,), dtype=np.float64, buffer=buf, offset=offset)
    # J4      = np.zeros((ngrid,), dtype=np.float64)

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
    # D5 = np.zeros((naux,naux), dtype=np.float64)
    offset   += naux * naux * buf.dtype.itemsize

    #print("offset_D5 = ", offset_D5//8)

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

    #print("bunchsize_readV = ", bunchsize_readV)

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
    #print("grid_bunchsize = ", grid_bunchsize)

    offset_AoR1 = offset
    offset_AoR2 = offset + nao * grid_bunchsize * buf.dtype.itemsize
    offset_AoR3 = offset + 2 * nao * grid_bunchsize * buf.dtype.itemsize

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    with lib.call_in_background(load_V_async) as prefetch:
        with lib.call_in_background(load_aoR_async) as prefetch_aoR:
     
            prefetch = load_V_async
            prefetch_aoR = load_aoR_async
      
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
    _benchmark_time(t1, t2, "construct D2", mydf)

    lib.dslice(D2, IP_ID, D3)

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

    # print("bunchsize_readV = ", bunchsize_readV)

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
            
            prefetch = load_V_async
            prefetch_aoR = load_aoR_async
      
            for ibunch, (p0, p1) in enumerate(lib.prange(0, ngrid, bunchsize_readV)):

                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

                V2   = np.ndarray((naux, min(p1+bunchsize_readV,ngrid)-p1),
                                  dtype=np.float64, buffer=buf, offset=offset_V2)
                aoR2 = np.ndarray((nao,  min(p1+bunchsize_readV,ngrid)-p1),
                                  dtype=np.float64, buffer=buf, offset=offset_aoR2)
                prefetch(p1, min(p1+bunchsize_readV,ngrid), V2)
                prefetch_aoR(p1, min(p1+bunchsize_readV,ngrid), aoR2)

                # construct D1

                # D1 = np.ndarray((nao, p1-p0),  dtype=np.float64, buffer=buf, offset=offset_D1)
                # D4 = np.ndarray((naux, p1-p0), dtype=np.float64, buffer=buf, offset=offset_D4)
                D1 = np.zeros((nao, p1-p0),  dtype=np.float64)
                D4 = np.zeros((naux, p1-p0),  dtype=np.float64)
                lib.ddot(dm, aoR1, c=D1)
                lib.ddot(aoRg.T, D1, c=D4)

                # add J1, J4

                beta = 1
                if ibunch == 0:
                    beta = 0

                # lib.ddot_withbuffer(V1, D2[p0:p1].reshape(-1,1), c=J1.reshape(-1,1), beta=beta, buf=mydf.ddot_buf)
                lib.ddot(V1, D2[p0:p1].reshape(-1,1), c=J1.reshape(-1,1), beta=beta)
                # NOTE we do not need another loop for J4
                # lib.ddot_withbuffer(D3.reshape(1,-1), V1, c=J4[p0:p1].reshape(1,-1), buf=mydf.ddot_buf)
                lib.ddot(D3.reshape(1,-1), V1, c=J4[p0:p1].reshape(1,-1))

                aoR3  = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=buf, offset=offset_aoR3)  # used in copy
                lib.copy(aoR1, out=aoR3)
                J_tmp = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=buf, offset=offset_J_tmp)
                lib.d_ij_j_ij(aoR3, J4[p0:p1], out=J_tmp)
                # lib.ddot_withbuffer(aoR1, J_tmp.T, c=J, beta=beta, buf=mydf.ddot_buf)
                J += lib.ddot(aoR1, J_tmp.T, beta=beta)

                # pack D5

                offset = IP_partition[ibunch][0]
                slices = IP_partition[ibunch][1]
                lib.dslice_offset(D4, locs=slices, offset=offset, out=D5)

                # construct K1 inplace

                # lib.cwise_mul(V1, D4, out=D4)
                D4 = lib.cwise_mul(V1, D4)
                K1 = D4
                K2 = np.zeros((naux, nao), dtype=np.float64)
                # lib.ddot_withbuffer(K1, aoR1.T, c=K2, buf=mydf.ddot_buf)
                # lib.ddot_withbuffer(aoRg, K2, c=K, beta=beta, buf=mydf.ddot_buf)
                K2 = lib.ddot(K1, aoR1.T)
                K += lib.ddot(aoRg, K2, beta=beta)

                # switch

                V1, V2 = V2, V1
                offset_V1, offset_V2 = offset_V2, offset_V1

                aoR1, aoR2 = aoR2, aoR1
                offset_aoR2, offset_aoR1 = offset_aoR1, offset_aoR2

                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                #print("ibunch %3d construct J[%8d:%8d] wall time: %12.6f CPU time: %12.6f" %
                #      (ibunch, p0, p1, t2[1] - t1[1], t2[0] - t1[0]))
                log.debug("ibunch %3d construct J[%8d:%8d] wall time: %12.6f CPU time: %12.6f" %
                            (ibunch, p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

    t2_ReadV = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1_ReadV, t2_ReadV, "ReadV", mydf)

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
    _benchmark_time(all_t1, all_t2, "get_jk", mydf)

    return J * ngrid / vol, K * ngrid / vol

def _allocate_jk_buffer_outcore(mydf:isdf_fast.PBC_ISDF_Info, datatype):

    if mydf.jk_buffer is None:

        nao    = mydf.nao
        ngrids = mydf.ngrids
        naux   = mydf.naux

        buffersize_k = 2 * nao * naux + naux * naux
        buffersize_j = nao * naux + naux + naux + nao * nao

        nThreadsOMP   = lib.num_threads()
        size_ddot_buf = max((naux*naux)+2, ngrids) * nThreadsOMP

        if hasattr(mydf, "IO_buf"):
            if mydf.IO_buf.size < (max(buffersize_k, buffersize_j) + size_ddot_buf):
                mydf.IO_buf = np.zeros((max(buffersize_k, buffersize_j) + size_ddot_buf,), dtype=datatype)

            mydf.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),),
                                        dtype=datatype, buffer=mydf.IO_buf, offset=0)
            offset         = max(buffersize_k, buffersize_j) * mydf.jk_buffer.dtype.itemsize
            mydf.ddot_buf  = np.ndarray((nThreadsOMP, max((naux*naux)+2, ngrids)),
                                        dtype=datatype, buffer=mydf.IO_buf, offset=offset)
        else:
            mydf.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),), dtype=datatype)
            mydf.ddot_buf = np.zeros((nThreadsOMP, max((naux*naux)+2, ngrids)), dtype=datatype)


    else:
        assert mydf.jk_buffer.dtype == datatype
        assert mydf.ddot_buf.dtype == datatype
            
def _get_j_dm_wo_robust_fitting(mydf:isdf_fast.PBC_ISDF_Info, dm):
        
    mydf._allocate_jk_buffer(dm.dtype)
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    nao  = dm.shape[0]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    naux = aoRg.shape[1]

    buffer = mydf.jk_buffer
    buffer1 = np.ndarray((nao,naux),  dtype=dm.dtype, buffer=buffer, offset=0)
    buffer4 = np.ndarray((naux),      dtype=dm.dtype, buffer=buffer, offset=(nao * naux) * dm.dtype.itemsize)
    buffer8 = np.ndarray((naux),      dtype=dm.dtype, buffer=buffer, offset=(nao * naux + naux) * dm.dtype.itemsize)
    buffer6 = np.ndarray((nao,nao),   dtype=dm.dtype, buffer=buffer, offset=(nao * naux + naux + naux) * dm.dtype.itemsize)
    buffer7 = np.ndarray((nao,naux),  dtype=dm.dtype, buffer=buffer, offset=0)
    
    lib.ddot(dm, aoRg, c=buffer1)  
    tmp1       = buffer1
    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg, tmp1, out=buffer4),
                            order='C')  # need allocate memory, size = naux, (buffer 4)

    # print("density_Rg = ", density_Rg.shape)
    # print("buffer8    = ", buffer8.shape)

    tmp = np.asarray(lib.dot(W, density_Rg.reshape(-1,1), c=buffer8.reshape(-1,1)), order='C').reshape(-1)
    tmp = np.asarray(lib.d_ij_j_ij(aoRg, tmp, out=buffer7), order='C')
    
    J = buffer6
    lib.ddot_withbuffer(aoRg, tmp.T, c=J, beta=0, buf=mydf.ddot_buf)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_j_dm")

    return J * ngrid / vol

def _get_k_dm_wo_robust_fitting(mydf:isdf_fast.PBC_ISDF_Info, dm):

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
    naux = aoRg.shape[1]
    IP_ID = mydf.IP_ID

    buffer = mydf.jk_buffer
    buffer1 = np.ndarray((nao,naux), dtype=dm.dtype, buffer=buffer, offset=0)
    buffer2 = np.ndarray((naux,naux), dtype=dm.dtype, buffer=buffer, offset=nao * naux * dm.dtype.itemsize)
    buffer6 = np.ndarray((naux,nao), dtype=dm.dtype, buffer=buffer, offset=(nao * naux + naux * naux) * dm.dtype.itemsize)
    buffer4 = np.ndarray((nao,nao), dtype=dm.dtype, buffer=buffer, offset=0)
    
    density_RgRg  = np.asarray(lib.dot(dm, aoRg, c=buffer1), order='C')
    lib.ddot(aoRg.T, density_RgRg, c=buffer2)
    density_RgRg = buffer2
    
    lib.cwise_mul(W, density_RgRg, out=density_RgRg)
    tmp = density_RgRg
    tmp = np.asarray(lib.dot(tmp, aoRg.T, c=buffer6), order='C')

    K = buffer4
    lib.ddot_withbuffer(aoRg, tmp, c=K, beta=0, buf=mydf.ddot_buf)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_k_dm")

    return K * ngrid / vol

def get_jk_dm_outcore(mydf, dm, hermi=1, kpt=np.zeros(3),
                 kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
    '''JK for given k-point'''

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    assert with_j is True and with_k is True

    #### explore the linearity of J K with respect to dm ####

    #### perform the calculation ####

    if mydf.jk_buffer is None:  # allocate the buffer for get jk
        # mydf._allocate_jk_buffer(mydf, datatype=dm.dtype)
        mydf._allocate_jk_buffer(datatype=dm.dtype)

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

    if mydf.outcore and mydf.with_robust_fitting == True:
        # raise NotImplementedError("outcore robust fitting has bugs and is extremely slow.")
        vj, vk = _get_jk_dm_outcore(mydf, dm)
    else:
        if mydf.outcore:
            vj = _get_j_dm_wo_robust_fitting(mydf, dm)
            vk = _get_k_dm_wo_robust_fitting(mydf, dm)
        else:
            vj = isdf_jk._contract_j_dm(mydf, dm, mydf.with_robust_fitting)
            vk = isdf_jk._contract_k_dm(mydf, dm, mydf.with_robust_fitting)

    t1 = log.timer('sr jk', *t1)

    return vj, vk

class PBC_ISDF_Info_outcore(isdf_fast.PBC_ISDF_Info):
    ''' Interpolative separable density fitting (ISDF) for periodic systems.
    Not recommended as the locality is not explored! 
    V, W, aux_basis is written on disk 
    
    Examples:

    >>> pbc_isdf = PBC_ISDF_Info_outcore(cell, max_buf_memory=max_memory)
    >>> pbc_isdf.build_IP_Sandeep_outcore(c=C)
    >>> pbc_isdf.build_auxiliary_Coulomb_outcore()
    >>> from pyscf.pbc import scf
    >>> mf = scf.RHF(cell)
    >>> pbc_isdf.direct_scf = mf.direct_scf
    >>> mf.with_df = pbc_isdf
    >>> mf.verbose = 0
    >>> mf.kernel()
    
    '''
    
    def __init__(self, mol:Cell, max_buf_memory:int, outcore=True, with_robust_fitting=True, aoR=None, kmesh=None):

        self.max_buf_memory = max_buf_memory
        self.IO_buf         = np.zeros((max_buf_memory//8), dtype=np.float64)

        super().__init__(mol=mol,aoR=aoR,with_robust_fitting=with_robust_fitting, kmesh=kmesh)

        self.IO_FILE        = None

        self._cached_bunchsize_ReadV = None
        self._cached_IP_partition    = None

        self.mesh = mol.mesh

        #print("self.mesh = ", self.mesh)

        self.saveAoR = True
        assert self.saveAoR  # we do not support saveAoR = False

        self.outcore = outcore
        
        # if outcore:
        #     self._allocate_jk_buffer = _allocate_jk_buffer_outcore

    def _allocate_jk_buffer(self, datatype):
        if self.outcore:
            _allocate_jk_buffer_outcore(self, datatype)
        else:
            super()._allocate_jk_buffer(datatype)

    def __del__(self):

        if self.IO_FILE is not None:
            try:
                os.system("rm %s" % (self.IO_FILE))
            except:
                pass

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

    select_IP = isdf_fast._select_IP_direct  # you can change it!

    def build_IP_Sandeep_outcore(self, IO_File:str = None, c:int = 5, m:int = 5, IP_ID = None):

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if IP_ID is None:
            self.IP_ID = self.select_IP(c, m)
            self.IP_ID = np.asarray(self.IP_ID, dtype=np.int32)
        else:
            self.IP_ID = IP_ID
        #print("IP_ID = ", self.IP_ID)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "select IP", self)

        if IO_File is None:
            # generate a random file name start with tmp_
            import random
            import string
            IO_File = "tmp_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)) + ".hdf5"

        #print("IO_File = ", IO_File)

        # construct aoR

        if self.coords is None:
            from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
            df_tmp = MultiGridFFTDF2(self.cell)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3)

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        coords_IP = self.coords[self.IP_ID]
        weight    = np.sqrt(self.cell.vol / self.ngrids)
        self.aoRg = self._numint.eval_ao(self.cell, coords_IP)[0].T * weight  # the T is important
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aoR", self)

        self.naux = self.aoRg.shape[1]
        self.c    = c

        #print("naux = ", self.naux)

        self.chunk_size, self.nRow_IO_V, self.blksize_aux, self.bunchsize_readV, self.grid_bunchsize, self.blksize_W, self.use_large_chunk_W  = _determine_bunchsize(
            self.nao, self.naux, self.mesh, self.IO_buf.size, self.saveAoR)

        # construct aux basis

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.outcore:
            #print("construct aux basis in outcore mode")
            _construct_aux_basis_IO(self, IO_File, self.IO_buf)
        else:
            #print("construct aux basis in incore mode")
            _construct_aux_basis(self)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aux basis", self)

        self.IO_FILE = IO_File

    def build_auxiliary_Coulomb_outcore(self, mesh=None):

        if mesh is None:
            mesh = self.cell.mesh

        mesh = np.asarray(mesh, dtype=np.int32)
        self.mesh = np.asarray(self.mesh, dtype=np.int32)

        if mesh[0] != self.mesh[0] or mesh[1] != self.mesh[1] or mesh[2] != self.mesh[2]:
            print("warning: mesh is not consistent with the previous one")

        self.mesh = mesh
        self.W    = np.zeros((self.naux, self.naux), dtype=np.float64)

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _construct_V_W_IO2(self, mesh, self.IO_FILE, self.IO_buf)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct V and W", self)

    get_jk = get_jk_dm_outcore

if __name__ == '__main__':

    C = 5
    from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info

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
    cell.verbose = 4

    # cell.ke_cutoff  = 70   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 32
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 1])

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx = grids.mesh[0]

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR)
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=True)
    pbc_isdf_info.build_auxiliary_Coulomb(cell, mesh)

    ### perform scf ###

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7
    
    pbc_isdf_info.with_robust_fitting = False

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7
    
    ### outcore ###

    max_memory = 100*1000*1000 # 10M
        
    pbc_isdf_info1 = PBC_ISDF_Info_outcore(cell, max_buf_memory=max_memory, outcore=False, with_robust_fitting=True, aoR=aoR)
    pbc_isdf_info1.build_IP_Sandeep_outcore(c=C)
    pbc_isdf_info1.build_auxiliary_Coulomb(mesh=mesh)
    
    V_R = pbc_isdf_info1.V_R
    aux_bas = pbc_isdf_info1.aux_basis
    aoR = pbc_isdf_info1.aoR
    W = pbc_isdf_info1.W
    
    IP_ID = pbc_isdf_info1.IP_ID
    
    pbc_isdf_info = PBC_ISDF_Info_outcore(cell, max_buf_memory=max_memory, outcore=True, with_robust_fitting=True)
    pbc_isdf_info.build_IP_Sandeep_outcore(c=C, IP_ID=IP_ID)
    pbc_isdf_info.build_auxiliary_Coulomb_outcore()
    
    f_aux_basis = h5py.File(pbc_isdf_info.IO_FILE, 'r')
    V_R_IO      = f_aux_basis[V_DATASET]
    
    diff = np.linalg.norm(V_R - V_R_IO)
    print("diff = ", diff)
    
    aux_basis_IO = f_aux_basis[AUX_BASIS_DATASET]
    diff = np.linalg.norm(aux_basis_IO - aux_bas)
    print("diff = ", diff)
    
    aoR_IO = f_aux_basis[AOR_DATASET]
    diff = np.linalg.norm(aoR_IO - aoR)
    print("diff = ", diff)
    
    W2 = pbc_isdf_info.W
    
    diff = np.linalg.norm(W - W2)
    print("diff = ", diff)
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df   = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol  = 1e-7
    mf.kernel()
    
    mf = scf.RHF(cell)
    pbc_isdf_info1.direct_scf = mf.direct_scf
    mf.with_df   = pbc_isdf_info1
    mf.max_cycle = 100
    mf.conv_tol  = 1e-7
    mf.kernel()
    
