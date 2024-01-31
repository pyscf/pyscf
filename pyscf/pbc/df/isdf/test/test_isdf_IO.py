
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

AUX_BASIS_DATASET = 'aux_basis'
V_DATASET         = 'V'

def _construct_aux_basis_IO(mydf:isdf_fast.PBC_ISDF_Info, IO_File:str, IO_buf:np.ndarray):
    '''
    IO_buf_memory: seems to be redundant
    '''

    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_aux_basis = h5py.File(IO_File, 'a')
            if AUX_BASIS_DATASET in f_aux_basis:
                del (f_aux_basis[AUX_BASIS_DATASET])
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

    blksize = IO_buf_memory//2// (8 * naux)
    chunks  = (IO_buf_memory//(8*2)//ngrids, ngrids)

    if chunks[0] > naux:
        chunks = (naux, ngrids)

    assert(blksize > 0)
    assert(chunks[0] > 0)

    print("IO_buf_memory = ", IO_buf_memory)
    print("blksize       = ", blksize)
    print("chunks        = ", chunks)

    A = np.asarray(lib.dot(aoRg.T, aoRg), order='C')
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

    h5d_aux_basis = f_aux_basis.create_dataset(AUX_BASIS_DATASET, (naux, ngrids), 'f8', chunks=chunks)

    def save(col0, col1, buf:np.ndarray):
        assert(buf.shape == (naux, col1-col0))
        h5d_aux_basis[:, col0:col1] = buf

    buf_calculate = np.ndarray((naux, blksize), dtype=IO_buf.dtype, buffer=IO_buf)
    buf_write     = np.ndarray((naux, blksize), dtype=IO_buf.dtype, buffer=IO_buf,
                               offset=naux*blksize*IO_buf.dtype.itemsize)

    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    ## get the coord of grids

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
    df_tmp = MultiGridFFTDF2(mydf.cell)
    NumInts = df_tmp._numint
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    # coords  = mydf.coords
    assert coords is not None

    ## get the coor

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    nThread = lib.num_threads()
    weight  = np.sqrt(mydf.cell.vol / ngrids)

    with lib.call_in_background(save) as async_write:
        for p0, p1 in lib.prange(0, ngrids, blksize):

            # build aux basis

            aoR = NumInts.eval_ao(mydf.cell, coords[p0:p1])[0].T * weight

            if p1!=p0 + blksize:
                buf_calculate = np.ndarray((naux, p1-p0), dtype=IO_buf.dtype, buffer=buf_calculate)  # the last chunk

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

            buf_write, buf_calculate = buf_calculate, buf_write

    ### close ###

    aoR = None

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

    IO_buf_memory = IO_buf.size * IO_buf.dtype.itemsize

    nao    = mydf.nao
    naux   = mydf.naux
    aoRg   = mydf.aoRg
    ngrids = mydf.ngrids

    mesh = np.asarray(mesh, dtype=np.int32)
    mesh_complex = np.asarray([mesh[0], mesh[1], mesh[2]//2+1], dtype=np.int32)

    ### the Coulomb kernel ###

    coulG      = tools.get_coulG(cell, mesh=mesh)
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

    THREAD_BUF_PERCENTAGE = 0.05

    print("IO_buf.size = ", IO_buf.size)

    memory_thread      = int(0.05 * IO_buf.size)

    bunchsize          = memory_thread // nThread // (mesh_complex[0] * mesh_complex[1] * mesh_complex[2] * 2)
    if bunchsize > 16:
        bunchsize = 16
    if bunchsize == 0:
        bunchsize = 1

    print("bunchsize = ", bunchsize)

    bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
    bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
    memory_thread      = bufsize_per_thread * nThread

    if memory_thread > IO_buf.size:
        raise ValueError("IO_buf is not large enough, memory_thread = %d, IO_buf.size = %d" %
                         (memory_thread, IO_buf.size))

    print("memory_thread = ", memory_thread)

    bunch_size_IO = ((IO_buf.size - memory_thread) // 5) // (ngrids)

    if bunch_size_IO > naux:
        bunch_size_IO = naux
    if bunch_size_IO == 0:
        raise ValueError("IO_buf is not large enough, bunch_size_IO = %d, IO_buf.size = %d" %
                         (bunch_size_IO, IO_buf.size))
    if bunch_size_IO < nThread:
        print("WARNING: bunch_size_IO = %d < nThread = %d" % (bunch_size_IO, nThread))

    print("bunch_size_IO = ", bunch_size_IO)

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
        if mydf.ddot_buf.size < bunch_size_IO * bunch_size_IO * nThread:
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

    with lib.call_in_background(load_aux_basis_async, sync=True) as prefetch:
        with lib.call_in_background(save_V, sync=True) as async_write:

            load_aux_basis(0, min(bunch_size_IO, naux), buf_aux_basis_bra) # force to load first bunch

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

                async_write(p0, p1, buf_V1)
                
                buf_V1, buf_V2 = buf_V2, buf_V1

                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                print("construct W[%5d:%5d] wall time: %12.6f CPU time: %12.6f" %
                      (p0, p1, t2[1] - t1[1], t2[0] - t1[0]))

    # print(buf_aux_basis[:, 0])

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

def _copy_b_to_a(a:np.ndarray, b:np.ndarray, size:int):
    assert(a.shape == b.shape)
    a.ravel()[:size] = b.ravel()[:size]

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
    buf_size            = buf_size_per_thread * nthread

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

    df_tmp = MultiGridFFTDF2(mydf.cell)
    NumInts = df_tmp._numint
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    # coords  = mydf.coords
    assert coords is not None

    results = []

    for p0, p1 in lib.prange(0, natm, nthread):
        taskinfo = []

        # clear buffer

        buf_tmp[:] = 0.0

        for atm_id in range(p0, p1):
            grid_ID = np.where(mydf.partition == atm_id)[0]
            aoR_atm_tmp = NumInts.eval_ao(mydf.cell, coords[grid_ID])[0]
            # print("aoR_atm.shape = ", aoR_atm.shape)

            # create buffer for this atm

            dtypesize = buf.dtype.itemsize

            offset = buf_size_per_thread * (atm_id - p0) * dtypesize

            aoR_atm = np.ndarray((grid_ID.shape[0], nao), dtype=np.float64, buffer=buf_tmp, offset=offset)
            _copy_b_to_a(aoR_atm, aoR_atm_tmp, aoR_atm_tmp.size)
            # print(aoR_atm[0])
            offset += nao*grid_ID.shape[0] * dtypesize

            nao_atm  = nao_per_atm[atm_id]
            naux_now = int(np.sqrt(c*nao_atm)) + m
            naux2_now = naux_now * naux_now

            Q = np.ndarray((naux2_now, naux2_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux2_now*naux2_now * dtypesize

            R = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux2_now*grid_ID.shape[0] * dtypesize

            aoR_atm1 = np.ndarray((grid_ID.shape[0], naux_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*grid_ID.shape[0] * dtypesize

            aoR_atm2 = np.ndarray((grid_ID.shape[0], naux_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*grid_ID.shape[0] * dtypesize

            aoPairBuffer = np.ndarray(
                (grid_ID.shape[0], naux_now*naux_now), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset += naux_now*naux_now*grid_ID.shape[0] * dtypesize

            task_now = (grid_ID, aoR_atm, nao, nao_atm, c, m)
            task_buffer = (Q, R, aoR_atm1, aoR_atm2, aoPairBuffer)

            taskinfo.append(_atm_IP_task(task_now, task_buffer))

        res_tmp = da.compute(*taskinfo, scheduler='threads')
        for res in res_tmp:
            # print("res = ", res)
            results.extend(res)

    # print("results = ", results)
    results.sort()

    return results


def _get_jk_dm_IO():
    pass

class PBC_ISDF_Info_IO(isdf_fast.PBC_ISDF_Info):
    def __init__(self, mol:Cell, max_buf_memory:int):
        super().__init__(mol=mol)

        self.max_buf_memory = max_buf_memory
        self.IO_buf         = np.zeros((max_buf_memory//8), dtype=np.float64)
        self.IO_FILE        = None

    def __del__(self):

        if self.IO_FILE is not None:
            os.system("rm %s" % (self.IO_FILE))

    def build_IP_Sandeep_IO(self, IO_File:str, c:int, m:int = 5):

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.IP_ID = _select_IP_direct(self, c, m)
        self.IP_ID = np.asarray(self.IP_ID, dtype=np.int32)
        print("IP_ID = ", self.IP_ID)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "select IP")

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

        buf_size  = self.nao*ngrid_on_atm                      # aoR_atm
        buf_size += naux_max2*naux_max2                        # Q
        buf_size += naux_max2*ngrid_on_atm                     # R
        buf_size += naux_max*ngrid_on_atm*2                    # aoR_atm1, aoR_atm2
        buf_size += naux_max*naux_max*ngrid_on_atm             # aoPairBuffer


        return buf_size

    def build_auxiliary_Coulomb_IO(self, mesh):

        self.mesh = mesh
        self.W = np.zeros((self.naux, self.naux), dtype=np.float64)

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _construct_V_W_IO(self, mesh, self.IO_FILE, self.IO_buf)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct V and W")


C = 12

if __name__ == "__main__":

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

    cell.ke_cutoff  = 256   # kinetic energy cutoff in a.u.
    # cell.ke_cutoff = 128
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

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = isdf_fast.PBC_ISDF_Info(cell, aoR, cutoff_aoValue=1e-6, cutoff_QR=1e-3)
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=False)

    IO_buf_memory = int(3e7)  # 30M
    IO_buf = np.zeros((IO_buf_memory//8), dtype=np.float64)
    IO_File = "test.h5"

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

        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])

    # IO_File = None

    os.system("rm %s" % (IO_File))

    pbc_isdf_info_IO = PBC_ISDF_Info_IO(cell, max_buf_memory=IO_buf_memory)

    pbc_isdf_info_IO.build_IP_Sandeep_IO(c=C, IO_File=IO_File)
    pbc_isdf_info_IO.build_auxiliary_Coulomb_IO(mesh)

    f_aux_basis = h5py.File(IO_File, 'r')
    aux_basis   = f_aux_basis[AUX_BASIS_DATASET][:]
    print("aux_basis.shape = ", aux_basis.shape)
    pbc_isdf_info_IO.aux_basis = aux_basis
    pbc_isdf_info_IO.aoR       = aoR
    # pbc_isdf_info_IO.check_AOPairError()
    
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

        idx = np.where(np.abs(diff) > 1e-8)

        ## print

        print("idx = ", idx)

        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])

    V = f_aux_basis[V_DATASET][:]
    V_bench = pbc_isdf_info.V_R

    if np.allclose(V, V_bench):
        print("PASS")
    else:
        ## print the difference

        diff = V - V_bench

        ## find where is the large difference

        idx = np.where(np.abs(diff) > 1e-8)

        ## print

        print("idx = ", idx)

        for i in range(idx[0].shape[0]):
            print("idx = ", idx[0][i], idx[1][i], diff[idx[0][i], idx[1][i]])