
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

def _construct_aux_basis_IO(mydf:isdf_fast.PBC_ISDF_Info, IO_File:str, IO_buf_memory:int, IO_buf:np.ndarray):

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

    # npt_find = c * nao_atm + 10
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
        # print("G1.shape = ", G1.shape)
        # print("G2.shape = ", G2.shape)
        # print("aoR_atm.shape = ", aoR_atm.shape)
        # print("aoR_atm1.shape = ", aoR_atm1.shape)
        lib.dot(aoR_atm, G1, c=aoR_atm1)
        lib.dot(aoR_atm, G2, c=aoR_atm2)
    aoPair = np.einsum('ki,kj->kij', aoR_atm1, aoR_atm2, out=aoPairBuffer.reshape(aoR_atm.shape[0],naux_tmp,naux_tmp)).reshape(grid_ID.shape[0], -1)
    pivot = _colpivot_qr(aoPair, Q, R, max_rank=npt_find)
    pivot_ID = grid_ID[pivot]  # the global ID
    # print("pivot_ID = ", pivot_ID)
    return pivot_ID

def _copy_b_to_a(a:np.ndarray, b:np.ndarray, size:int):
    assert(a.shape == b.shape)
    a.ravel()[:size] = b.ravel()[:size]

# @profile
def _select_IP_direct(mydf:isdf_fast.PBC_ISDF_Info, buf:np.ndarray, c:int, m:int):

    bunchsize = lib.num_threads()

    ### determine the largest grids point of one atm ###

    # ngrid_on_atm = np.zeros((mydf.cell.natm), dtype=np.int32)

    # for atm_id in mydf.partition:
    #     ngrid_on_atm[atm_id] += 1

    # ngrid_on_atm = np.max(ngrid_on_atm)

    natm         = mydf.cell.natm
    nao          = mydf.nao
    # naux         = mydf.naux
    # aoRg         = mydf.aoRg
    # ngrids       = mydf.ngrids

    naux_max = 0

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

    if buf.size < buf_size:
        # reallocate
        buf = np.zeros((buf_size), dtype=np.float64)
        print("reallocate buf of size = ", buf_size * 8)
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

    def build_IP_Sandeep_IO(self, IO_File:str, c:int, m:int=5):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.IP_ID = _select_IP_direct(self, self.IO_buf, c, m)
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
        _construct_aux_basis_IO(self, IO_File, self.max_buf_memory, self.IO_buf)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aux basis")


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

C = 10

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
    # cell.ke_cutoff = 32
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
    pbc_isdf_info.check_AOPairError()

    IO_buf_memory = int(3e7)  # 30M
    IO_buf = np.zeros((IO_buf_memory), dtype=np.float64)
    IO_File = "test.h5"

    _construct_aux_basis_IO(pbc_isdf_info, IO_File, IO_buf_memory, IO_buf)

    f_aux_basis = h5py.File(IO_File, 'r')
    aux_basis   = f_aux_basis[AUX_BASIS_DATASET]

    print("aux_basis.shape = ", aux_basis.shape)
    # print(np.allclose(aux_basis, pbc_isdf_info.aux_basis))

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

    f_aux_basis = h5py.File(IO_File, 'r')
    aux_basis   = f_aux_basis[AUX_BASIS_DATASET]    
    pbc_isdf_info_IO.aux_basis = aux_basis
    pbc_isdf_info_IO.aoR       = aoR
    pbc_isdf_info_IO.check_AOPairError()

    os.system("rm %s" % (IO_File))