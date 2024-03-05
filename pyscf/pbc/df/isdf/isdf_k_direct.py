
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

import pyscf.pbc.df.isdf.isdf_fast as isdf_fast
import pyscf.pbc.df.isdf.isdf_k as isdf_k
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

import ctypes
from multiprocessing import Pool

import memory_profiler

from memory_profiler import profile

libfft = lib.load_library('libfft')
libpbc = lib.load_library('libpbc')

from pyscf.lib import misc
_np_helper = misc.load_library('libnp_helper')

####### construct aux basis and W ####### 

@profile
def _construct_aux_basis_W_kSym_Direct(mydf:isdf_fast.PBC_ISDF_Info, IO_buf:np.ndarray):
    
    #### preprocess ####
    
    nGrids   = mydf.ngrids
    nGridPrim = mydf.nGridPrim
    nIP_Prim = mydf.nIP_Prim
    
    Mesh = mydf.mesh
    Mesh = np.array(Mesh, dtype=np.int32)
    Ls   = mydf.Ls
    Ls   = np.array(Ls, dtype=np.int32)
    MeshPrim = np.array(Mesh) // np.array(Ls)
    MeshPrim = np.array(MeshPrim, dtype=np.int32)
    meshPrim = MeshPrim
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    naux = mydf.naux
    naoPrim = mydf.nao // np.prod(Ls)
    nao  = mydf.nao
    
    ####### 2. construct A ########
    
    offset = 0
    buf_A = np.ndarray((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    buf_A_real = np.ndarray((nIP_Prim, nIP_Prim * ncell), dtype=np.double, buffer=IO_buf, offset=offset)
    
    offset += nIP_Prim * nIP_Prim * ncell_complex * buf_A.itemsize
    buf_A_fft = np.ndarray((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    
    ### NOTE: here we consume 2 * nIP_Prim * nIP_Prim * ncell_complex * 16 bytes
    
    aoRg = mydf.aoRg[:, :nIP_Prim] # only the first box is used
    A    = np.asarray(lib.ddot(aoRg.T, mydf.aoRg, c=buf_A_real), order='C')
    lib.square_inPlace(A)
    
    fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    fn(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_Prim),
        ctypes.c_int(nIP_Prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_A_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    A = None    
    A = buf_A
    
    buf_A_real = None
    buf_A_fft  = None
    
    ####### 3. diag A ########
     
    fn_zcopy_col = getattr(_np_helper, "NPzcopy_col", None)
    assert fn_zcopy_col is not None
     
    block_A_e = np.zeros((ncell_complex, nIP_Prim), dtype=np.double)
    
    iloc = 0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                
                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                
                #### diag A ####
                        
                buf_A_diag = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset) 
                # buf_A_diag.ravel()[:] = A[:, iloc*nIP_Prim:(iloc+1)*nIP_Prim].ravel()[:]
                fn_zcopy_col(
                    ctypes.c_int(nIP_Prim),
                    buf_A_diag.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(0),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(buf_A_diag.shape[1]),
                    A.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(iloc*nIP_Prim),
                    ctypes.c_int((iloc+1)*nIP_Prim),
                    ctypes.c_int(A.shape[1])
                )
                
                with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
                    e, h = scipy.linalg.eigh(buf_A_diag)
                block_A_e[iloc] = e
                h = h.copy()
                # A[:, iloc*nIP_Prim:(iloc+1)*nIP_Prim] = h
                fn_zcopy_col(
                    ctypes.c_int(nIP_Prim),
                    A.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(iloc*nIP_Prim),
                    ctypes.c_int((iloc+1)*nIP_Prim),
                    ctypes.c_int(A.shape[1]),
                    h.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(0),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(h.shape[1])
                )
                h = None
                
                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                
                print("diagA %5d wall time: %12.6f CPU time: %12.6f" % (iloc, t2[1] - t1[1], t2[0] - t1[0]))
                iloc += 1                
    
    ########
    
    itask_2_xyz = np.zeros((ncell_complex, 3), dtype=np.int32)
    loc = 0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                itask_2_xyz[loc, 0] = ix
                itask_2_xyz[loc, 1] = iy
                itask_2_xyz[loc, 2] = iz
                loc += 1
    
    # get freq # 
    
    fn_FREQ = getattr(libpbc, "_FREQ", None)
    assert fn_FREQ is not None
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    FREQ = np.zeros((ncell_complex, nGridPrim), dtype=np.complex128)
    
    fn_FREQ(
        FREQ.ctypes.data_as(ctypes.c_void_p),
        meshPrim.ctypes.data_as(ctypes.c_void_p),
        Ls.ctypes.data_as(ctypes.c_void_p)
    )
    
    # freq1 = np.array(range(meshPrim[0]), dtype=np.float64)
    # freq2 = np.array(range(meshPrim[1]), dtype=np.float64)
    # freq3 = np.array(range(meshPrim[2]), dtype=np.float64)
    # freq_q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    # freq1 = np.array(range(Ls[0]), dtype=np.float64)
    # freq2 = np.array(range(Ls[1]), dtype=np.float64)
    # freq3 = np.array(range(Ls[2]//2+1), dtype=np.float64)
    # freq_Q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    # FREQ = np.einsum("ijkl,ipqs->ijklpqs", freq_Q, freq_q)
    # FREQ[0] /= (Ls[0] * meshPrim[0])
    # FREQ[1] /= (Ls[1] * meshPrim[1])
    # FREQ[2] /= (Ls[2] * meshPrim[2])
    # FREQ = np.einsum("ijklpqs->jklpqs", FREQ)
    # FREQ  = FREQ.reshape(-1, np.prod(meshPrim)).copy()
    # FREQ  = np.exp(-2.0j * np.pi * FREQ)
    # freq_Q = None
    # freq_q = None
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    _benchmark_time(t1, t2, "FREQ")
    
    t1 = t2
    
    fn_permutation = getattr(libpbc, "_get_permutation", None)
    assert fn_permutation is not None
    
    # def _permutation(nx, ny, nz, shift_x, shift_y, shift_z):
    #     res = np.zeros((nx*ny*nz), dtype=numpy.int32)
    #     loc_now = 0
    #     for ix in range(nx):
    #         for iy in range(ny):
    #             for iz in range(nz):
    #                 ix2 = (nx - ix - shift_x) % nx
    #                 iy2 = (ny - iy - shift_y) % ny
    #                 iz2 = (nz - iz - shift_z) % nz
    #                 
    #                 loc = ix2 * ny * nz + iy2 * nz + iz2
    #                 # res[loc_now] = loc
    #                 res[loc] = loc_now
    #                 loc_now += 1
    #     return res

    permutation = np.zeros((8, nGridPrim), dtype=np.int32)
    # permutation[0] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 0, 0)
    # permutation[1] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 0, 1)
    # permutation[2] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 1, 0)
    # permutation[3] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 1, 1)
    # permutation[4] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 0, 0)
    # permutation[5] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 0, 1)
    # permutation[6] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 1, 0)
    # permutation[7] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 1, 1)
    
    fn_permutation(
        meshPrim.ctypes.data_as(ctypes.c_void_p),
        permutation.ctypes.data_as(ctypes.c_void_p)
    )
    
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    _benchmark_time(t1, t2, "permutation")
    
    fac = np.sqrt(np.prod(Ls) / np.prod(Mesh))
    
    coords_prim = mydf.ordered_grid_coords[:nGridPrim]
    
    ######## 4. construct B and W ########
    
    # W = np.zeros((nIP_Prim, nIP_Prim*ncell), dtype=np.float64)
    # W = np.ndarray((nIP_Prim, nIP_Prim*ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    W_buf2 = np.zeros((nIP_Prim, nIP_Prim*ncell_complex), dtype=np.complex128)
    W_buf3 = np.ndarray((nIP_Prim, nIP_Prim*ncell), dtype=np.float64, buffer=W_buf2, offset=0)

    offset_B_block = offset
    B_block = np.ndarray((nIP_Prim, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_B_block)
    offset += nIP_Prim * nGridPrim * B_block.itemsize
    
    #### NOTE: here we consume nIP_Prim * nIP_Prim * ncell+complex  * 16 +  nIP_Prim * nGridPrim * 16 bytes
    
    offset_backup = offset
    
    offset_B_fft_buf = offset_backup
    B_BufFFT = np.ndarray((nIP_Prim, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_B_fft_buf)
    offset += nIP_Prim * nGridPrim * B_BufFFT.itemsize
    
    #### NOTE: here we consume nIP_Prim * nIP_Prim * ncell+complex  * 16 +  2 * nIP_Prim * nGridPrim * 16 bytes
    
    fn_final_fft = getattr(libpbc, "_FinalFFT", None)
    assert fn_final_fft is not None
    
    #### construct coulG ####
    
    coulG = tools.get_coulG(mydf.cell, mesh=Mesh)
    coulG = coulG.reshape(meshPrim[0], Ls[0], meshPrim[1], Ls[1], meshPrim[2], Ls[2])
    coulG = coulG.transpose(1, 3, 5, 0, 2, 4).reshape(-1, np.prod(meshPrim)).copy()
    
    fn_coulG = getattr(libpbc, "_construct_W_multiG", None)
    assert(fn_coulG is not None)
    
    #### determine other bufsize ####
    
    size_now = IO_buf.size - offset_backup // 8
    
    bunchsize = size_now // (nao * 2 + nao + nIP_Prim * ncell_complex * 2 * 2 + nIP_Prim + nIP_Prim * 2)
    bunchsize = min(bunchsize, nGridPrim)
    
    if bunchsize < 1e4 and bunchsize < nGridPrim//32:
        print("bunchsize %d is too small, please check the buffer size" % (bunchsize))
        
    # bunchsize = None # to determine
    
    offset_now = offset_backup
    
    offset_aoR_buf1 = offset_now
    AoR_Buf1 = np.ndarray((nao, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_aoR_buf1)
    offset_now += nao * bunchsize * AoR_Buf1.itemsize
    
    offset_aoR_bufpack = offset_now
    AoR_BufPack = np.ndarray((nao, bunchsize), dtype=np.float64, buffer=IO_buf, offset=offset_aoR_bufpack)
    offset_now += nao * bunchsize * AoR_BufPack.itemsize
    
    offset_B_buf1 = offset_now
    B_Buf1 = np.ndarray((nIP_Prim, ncell_complex, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1)
    offset_now += nIP_Prim * ncell_complex * bunchsize * B_Buf1.itemsize
    
    offset_B_buf1_FFT = offset_now
    B_Buf1_FFT_buf = np.ndarray((nIP_Prim, ncell_complex, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1_FFT)
    offset_now += nIP_Prim * ncell_complex * bunchsize * B_Buf1_FFT_buf.itemsize
    
    offset_B_buf2 = offset_now
    B_Buf2 = np.ndarray((nIP_Prim, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf2)
    offset_now += nIP_Prim * bunchsize * B_Buf2.itemsize
    
    offset_ddot_res = offset_now
    buf_ddot_res = np.ndarray((nIP_Prim, bunchsize), dtype=np.float64, buffer=IO_buf, offset=offset_ddot_res)
    offset_now += nIP_Prim * bunchsize * buf_ddot_res.itemsize
    
    ###### ######
    
    offset_now = offset_backup
    
    offset_buf_A = offset_now
    buf_A = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_A)
    offset_now += nIP_Prim * nIP_Prim * buf_A.itemsize
    
    offset_A2 = offset_now
    buf_A2 = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset_A2)
    offset_now += nIP_Prim * nIP_Prim * buf_A2.itemsize
    
    offset_buf_W = offset_now
    W_buf = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W)
    offset_now += nIP_Prim * nIP_Prim * W_buf.itemsize
    
    nThread = lib.num_threads()
    offset_buf_dot = offset_now
    ddot_buf = np.ndarray((nIP_Prim*nIP_Prim+2, nThread), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_dot)
    offset_now += nIP_Prim * nIP_Prim * nThread * ddot_buf.itemsize
    
    size_now = IO_buf.size - (offset_now // 8)
    
    ###### ######
    
    B_bunchsize = size_now // (nIP_Prim * 2 * 2)
    B_bunchsize = min(B_bunchsize, nGridPrim)
    
    if B_bunchsize < 1e4 and B_bunchsize < nGridPrim//32:
        print("B_bunchsize %d is too small, please check the buffer size" % (B_bunchsize))
    
    print("bunchsize   = ", bunchsize)
    print("B_bunchsize = ", B_bunchsize)
    
    offset_buf_W2 = offset_now
    W_buf_2 = np.ndarray((nIP_Prim, B_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W2)
    offset_now += nIP_Prim * B_bunchsize * W_buf_2.itemsize
    
    offset_buf_W3 = offset_now
    W_buf_3 = np.ndarray((nIP_Prim, B_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W3)
    offset_now += nIP_Prim * B_bunchsize * W_buf_3.itemsize
    
    ###### ######
    
    offset_now = offset_backup
    offset_W_fft = offset_now
    W_buf_fft = np.ndarray((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset_W_fft)
    
    
    #### NOTE: here we consume nIP_Prim * nIP_Prim * ncell+complex  * 16 +  2 * nIP_Prim * nGridPrim * 16 bytes, this is the smallest buffer size
    
    weight = np.sqrt(mydf.cell.vol / mydf.ngrids)

    fn_dcopy_row = getattr(_np_helper, "NPdcopy_row", None)
    assert fn_dcopy_row is not None
    
    fn_zslice = getattr(_np_helper, "NPzslice32", None)
    assert fn_zslice is not None

    fn_z_i_ij_ij = getattr(_np_helper, "NPz_i_ij_ij", None)
    assert fn_z_i_ij_ij is not None

    fn_dpack_tensor_3d_midloc = getattr(_np_helper, "NPdpack_tensor_3d_midloc", None)
    assert fn_dpack_tensor_3d_midloc is not None

    fn_zextract_tensor_3d_midloc = getattr(_np_helper, "NPzextract_tensor_3d_midloc", None)
    assert fn_zextract_tensor_3d_midloc is not None

    iloc = 0
    for ix_B in range(Ls[0]):
        for iy_B in range(Ls[1]):
            for iz_B in range(Ls[2]//2+1):
                
                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                
                for bunch_begin, bunch_end in lib.prange(0, nGridPrim, bunchsize):
                
                    # bunch_begin = iloc * bunchsize
                    # bunch_end   = (iloc + 1) * bunchsize
                    # bunch_begin = min(bunch_begin, nGridPrim)
                    # bunch_end   = min(bunch_end, nGridPrim)
    
                    AoR_Buf1 = np.ndarray((nao, bunch_end-bunch_begin), dtype=np.complex128, buffer=IO_buf, offset=offset_aoR_buf1)
                    AoR_Buf1 = ISDF_eval_gto(mydf.cell, coords=coords_prim[bunch_begin:bunch_end], out=AoR_Buf1) * weight
    
                    # print("AoR_Buf1.shape = ", AoR_Buf1.shape)  
    
                    ####### 4.1 construct B #######
                
                    # for p0, p1 in lib.prange(0, bunch_end-bunch_begin, sub_bunchsize):
                    
                    p1 = bunch_end
                    p0 = bunch_begin
                    
                    # print("p0 = %5d, p1 = %5d" % (p0, p1))
                    
                    AoR_BufPack = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_aoR_bufpack)
                    B_Buf1 = np.ndarray((nIP_Prim, ncell, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_B_buf1)
                    
                    for ix2 in range(Ls[0]):
                        for iy2 in range(Ls[1]):
                            for iz2 in range(Ls[2]):                                
                                
                                # pack aoR #

                                for ix_row in range(Ls[0]):
                                    for iy_row in range(Ls[1]):
                                        for iz_row in range(Ls[2]):

                                            loc_row1 = ix_row * Ls[1] * Ls[2] + iy_row * Ls[2] + iz_row

                                            row_begin1 = loc_row1 * naoPrim
                                            row_end1   = (loc_row1 + 1) * naoPrim

                                            ix3 = (ix_row - ix2 + Ls[0]) % Ls[0]
                                            iy3 = (iy_row - iy2 + Ls[1]) % Ls[1]
                                            iz3 = (iz_row - iz2 + Ls[2]) % Ls[2]

                                            loc_row2 = ix3 * Ls[1] * Ls[2] + iy3 * Ls[2] + iz3

                                            row_begin2 = loc_row2 * naoPrim
                                            row_end2   = (loc_row2 + 1) * naoPrim

                                            # print("p0, p1 = ", p0, p1)
                                            # print("AoR_BufPack.shape = ", AoR_BufPack.shape)
                                            # print("AoR_Buf1.shape    = ", AoR_Buf1.shape)

                                            # AoR_BufPack[row_begin1:row_end1, :] = AoR_Buf1[row_begin2:row_end2, :]
                                            
                                            fn_dcopy_row(
                                                ctypes.c_int(AoR_BufPack.shape[1]),
                                                AoR_BufPack.ctypes.data_as(ctypes.c_void_p),
                                                ctypes.c_int(row_begin1),
                                                ctypes.c_int(row_end1),
                                                ctypes.c_int(AoR_BufPack.shape[1]),
                                                AoR_Buf1.ctypes.data_as(ctypes.c_void_p),
                                                ctypes.c_int(row_begin2),
                                                ctypes.c_int(row_end2),
                                                ctypes.c_int(AoR_Buf1.shape[1])
                                            )

                                # perform one dot #
                                    
                                loc = ix2 * Ls[1] * Ls[2] + iy2 * Ls[2] + iz2
                                        
                                # k_begin = loc * bunchsize
                                # k_end   = (loc + 1) * bunchsize
                                        
                                buf_ddot_res = np.ndarray((nIP_Prim, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_ddot_res)
                                lib.ddot_withbuffer(aoRg.T, AoR_BufPack, c=buf_ddot_res, buf=ddot_buf)
                                # B_Buf1[:, loc, :] = buf_ddot_res
                                
                                fn_dpack_tensor_3d_midloc(
                                    ctypes.c_int(B_Buf1.shape[0]),
                                    ctypes.c_int(B_Buf1.shape[1]),
                                    ctypes.c_int(B_Buf1.shape[2]),
                                    B_Buf1.ctypes.data_as(ctypes.c_void_p),
                                    buf_ddot_res.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(buf_ddot_res.shape[1]),
                                    ctypes.c_int(loc)
                                )
                                
                                buf_ddot_res = None

                    lib.square_inPlace(B_Buf1)
                    
                    B_Buf1_FFT_buf = np.ndarray((nIP_Prim, ncell_complex, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1_FFT)
                    
                    fn(
                        B_Buf1.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nIP_Prim),
                        ctypes.c_int(p1-p0),
                        Ls.ctypes.data_as(ctypes.c_void_p),
                        B_Buf1_FFT_buf.ctypes.data_as(ctypes.c_void_p)
                    )
                    
                    B_Buf1_FFT_buf = None

                    B_Buf1_complex = np.ndarray((nIP_Prim, ncell_complex, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1)
                    B_Buf2 = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf2)
                    # B_Buf2.ravel()[:] = B_Buf1_complex[:, iloc, :].ravel()[:]
                    
                    fn_zextract_tensor_3d_midloc(
                        ctypes.c_int(B_Buf1_complex.shape[0]),
                        ctypes.c_int(B_Buf1_complex.shape[1]),
                        ctypes.c_int(B_Buf1_complex.shape[2]),
                        B_Buf2.ctypes.data_as(ctypes.c_void_p),
                        B_Buf1_complex.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(B_Buf2.shape[1]),
                        ctypes.c_int(iloc)
                    )
                    
                    # print("p0 = %5d, p1 = %5d" % (p0, p1))                       
                    # print("B_buf2 = ", B_Buf2)
                    
                    # B_block[:, bunch_begin:bunch_end] = B_Buf2
                    
                    fn_zcopy_col(
                        ctypes.c_int(nIP_Prim),
                        B_block.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(bunch_begin),
                        ctypes.c_int(bunch_end),
                        ctypes.c_int(B_block.shape[1]),
                        B_Buf2.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(0),
                        ctypes.c_int(p1-p0),
                        ctypes.c_int(B_Buf2.shape[1])
                    )
                    
                    AoR_Buf1       = None
                    AoR_BufPack    = None
                    B_Buf1         = None
                    B_Buf1_complex = None
                    B_Buf2         = None
                
                # print("B_block = ", B_block)
                    
                ## solve AX = B ##
                
                buf_A = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_A)
                begin = iloc       * nIP_Prim
                end   = (iloc + 1) * nIP_Prim
                
                # buf_A.ravel()[:] = A[:, begin:end].ravel()[:]
                
                fn_zcopy_col(
                    ctypes.c_int(nIP_Prim),
                    buf_A.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(0),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(buf_A.shape[1]),
                    A.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(begin),
                    ctypes.c_int(end),
                    ctypes.c_int(A.shape[1])
                )
                
                e = block_A_e[iloc]
                
                # print("e = ", e)
                
                e_max = np.max(e)
                e_min_cutoff = e_max * isdf_k.COND_CUTOFF
                where1 = np.where(e < e_min_cutoff)[0]
                e1 = e[where1]
                if e1.shape[0] > 0:
                    print("eigenvalues indicate numerical instability")
                    for loc_e, x in enumerate(e1):
                        print("e1[%3d] = %15.6e" % (loc_e, x))
                where = np.where(e > e_min_cutoff)[0]
                where = np.array(where, dtype=np.int32)
                e = e[where]
                
                buf_A2 = np.ndarray((nIP_Prim, e.shape[0]), dtype=np.complex128, buffer=IO_buf, offset=offset_A2)
                # buf_A2.ravel()[:] = buf_A[:, where].ravel()[:]
                
                fn_zslice(
                    buf_A2.ctypes.data_as(ctypes.c_void_p),
                    buf_A.ctypes.data_as(ctypes.c_void_p),
                    where.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(e.shape[0]),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(buf_A.shape[1]),
                )
            
                # buf_A2 = buf_A[:, where]
                
                # print("we get here!")
                
                e = 1.0 / e
                
                for p0, p1 in lib.prange(0, nGridPrim, B_bunchsize):
                    B_Buf_in_W = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W2)
                    # B_Buf_in_W.ravel()[:] = B_block[:, p0:p1].ravel()[:]
                    fn_zcopy_col(
                        ctypes.c_int(nIP_Prim),
                        B_Buf_in_W.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(0),
                        ctypes.c_int(p1-p0),
                        ctypes.c_int(B_Buf_in_W.shape[1]),
                        B_block.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(p0),
                        ctypes.c_int(p1),
                        ctypes.c_int(B_block.shape[1])
                    )
                    B_ddot = np.ndarray((e.shape[0], p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W3)
                    lib.dot(buf_A2.T.conj(), B_Buf_in_W, c=B_ddot)
                    # B_ddot = (1.0/e).reshape(-1,1) * B_ddot
                    fn_z_i_ij_ij(
                        B_ddot.ctypes.data_as(ctypes.c_void_p),
                        e.ctypes.data_as(ctypes.c_void_p),
                        B_ddot.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(e.shape[0]),
                        ctypes.c_int(p1-p0),
                    )
                    lib.dot(buf_A2, B_ddot, c=B_Buf_in_W)
                    # B_block[:, p0:p1] = B_Buf_in_W
                    fn_zcopy_col(
                        ctypes.c_int(nIP_Prim),
                        B_block.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(p0),
                        ctypes.c_int(p1),
                        ctypes.c_int(B_block.shape[1]),
                        B_Buf_in_W.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(0),
                        ctypes.c_int(p1-p0),
                        ctypes.c_int(B_Buf_in_W.shape[1])
                    )
                    ##### clear #####
                    B_Buf_in_W = None
                    B_ddot = None
                
                e = None    
                
                # B1 = np.dot(buf_A2.T.conj(), B_block)
                # B1 = (1.0/e).reshape(-1,1) * B1
                # B_block = np.dot(buf_A2, B1)
                
                # print(B_block)
                
                # print("we get here!")
                                
                ########### 5. construct W ###########
                
                fn_zset0 = getattr(_np_helper, "NPzset0", None)
                assert fn_zset0 is not None
                
                # W_buf.ravel()[:] = 0.0 # reset W_buf
                
                W_buf = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W)
                
                fn_zset0(
                    W_buf.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int64(W_buf.size)
                )
                
                B_fft_buf = np.ndarray((nIP_Prim, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_B_fft_buf)
                                
                fn_final_fft(
                    B_block.ctypes.data_as(ctypes.c_void_p),
                    FREQ[iloc].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(nGridPrim),
                    meshPrim.ctypes.data_as(ctypes.c_void_p),
                    B_fft_buf.ctypes.data_as(ctypes.c_void_p)
                )
                
                B_fft_buf = None
                
                
                # print("aux_basis = ", B_block)
                
                B_block *= fac
                                
                for p0, p1 in lib.prange(0, nGridPrim, B_bunchsize):
                    W_buf_2 = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W2)
                    # W_buf_2.ravel()[:] = B_block[:, p0:p1].ravel()[:]
                    fn_zcopy_col(
                        ctypes.c_int(nIP_Prim),
                        W_buf_2.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(0),
                        ctypes.c_int(p1-p0),
                        ctypes.c_int(W_buf_2.shape[1]),
                        B_block.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(p0),
                        ctypes.c_int(p1),
                        ctypes.c_int(B_block.shape[1])
                    )
                    W_buf_3 = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_buf_W3)
                    # W_buf_3.ravel()[:] = W_buf_2.ravel()[:]
                    fn_zcopy_col(
                        ctypes.c_int(nIP_Prim),
                        W_buf_3.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(0),
                        ctypes.c_int(p1-p0),
                        ctypes.c_int(W_buf_3.shape[1]),
                        B_block.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(p0),
                        ctypes.c_int(p1),
                        ctypes.c_int(B_block.shape[1])
                    )
                        
                    iloc2 = ix_B * Ls[1] * (Ls[2]) + iy_B * (Ls[2]) + iz_B
                    fn_coulG(
                        ctypes.c_int(nIP_Prim),
                        ctypes.c_int(p0),
                        ctypes.c_int(p1),
                        W_buf_3.ctypes.data_as(ctypes.c_void_p),
                        coulG[iloc2].ctypes.data_as(ctypes.c_void_p)
                    )                    
                    lib.dot(W_buf_2, W_buf_3.T.conj(), c=W_buf, beta=1)
                    
                    W_buf_2 = None
                    W_buf_3 = None  
                    
                # print("W_buf = ", W_buf)
                # print("we get here!")
                
                k_begin = iloc * nIP_Prim
                k_end   = (iloc + 1) * nIP_Prim
                
                # W_buf2[:, k_begin:k_end] = W_buf
                
                fn_zcopy_col(
                    ctypes.c_int(nIP_Prim),
                    W_buf2.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(k_begin),
                    ctypes.c_int(k_end),
                    ctypes.c_int(W_buf2.shape[1]),
                    W_buf.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(0),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(W_buf.shape[1])
                )
                
                iloc += 1

                buf_A  = None
                buf_A2 = None
                W_buf  = None
                
                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                
                print("iloc %5d wall time: %12.6f CPU time: %12.6f" % (iloc, t2[1] - t1[1], t2[0] - t1[0]))
    
    ### final free ??? ### 
    
    fn = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert(fn is not None)
    
    W_buf_fft = np.ndarray((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset_W_fft)
    # W_buf_fft = np.zeros((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128)
    
    fn(
        W_buf2.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_Prim),
        ctypes.c_int(nIP_Prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        W_buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    W = np.zeros((nIP_Prim, nIP_Prim*ncell), dtype=np.float64)
    # W.ravel()[:] = W_buf3.ravel()[:]
    
    fn_dcopy = getattr(_np_helper, "NPdcopy", None)
    assert fn_dcopy is not None
    
    fn_dcopy(
        W.ctypes.data_as(ctypes.c_void_p),
        W_buf3.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(W.size)
    )
    
    mydf.W = W
    
    A = None
    B_block = None    
    B_BufFFT = None
    
    W0 = np.zeros((nIP_Prim, nIP_Prim), dtype=np.float64)
    
    ### used in get J ### 
    
    for i in range(ncell):
            
        k_begin = i * nIP_Prim
        k_end   = (i + 1) * nIP_Prim
            
        W0 += W[:, k_begin:k_end]

    mydf.W0 = W0
    
    W_buf2 = None
    W_buf3 = None
    coulG  = None
    permutation = None
    
    return W

class PBC_ISDF_Info_kSym_Direct(isdf_k.PBC_ISDF_Info_kSym):
    
    # @profile 
    def __init__(self, mol:Cell, max_buf_memory:int, Ls=[1,1,1]):
        super().__init__(mol=mol, max_buf_memory=max_buf_memory, outcore=True, with_robust_fitting=False, aoR=None, Ls=Ls) 
        self.outcore = False
    
    def __del__(self):
        pass
    
    # @profile
    def _allocate_jk_buffer(self, dtype=np.float64):
        
        if self.jk_buffer is not None:
            return
            
        num_threads   = lib.num_threads()
        nIP_Prim      = self.nIP_Prim
        nGridPrim     = self.nGridPrim
        ncell_complex = self.Ls[0] * self.Ls[1] * (self.Ls[2]//2+1)
        nao_prim      = self.nao // np.prod(self.Ls)
        naux          = self.naux
        nao           = self.nao
        ngrids        = nGridPrim * self.Ls[0] * self.Ls[1] * self.Ls[2]
        ncell         = np.prod(self.Ls)
        
        buf_J = self._get_bufsize_get_j()
        buf_K = self._get_bufsize_get_k()
        
        size_W = nIP_Prim * nIP_Prim * ncell * 2 + 2 * nIP_Prim * nGridPrim * 3 + num_threads * (nIP_Prim * nIP_Prim + 2) * 2
        
        size_buf = max(buf_J, buf_K, size_W)
        
        if hasattr(self, "IO_buf"):
            if self.IO_buf.size < size_buf:
                print("reallocate of size = ", size_buf)
                self.IO_buf = np.zeros((size_buf), dtype=np.float64)
        else:
            self.IO_buf = np.zeros((size_buf), dtype=np.float64)

        self.jk_buffer = np.ndarray((size_buf), dtype=np.float64, buffer=self.IO_buf, offset=0)
        size_ddot_buf = ((nIP_Prim*nIP_Prim)+2)*num_threads
        self.ddot_buf  = np.ndarray((size_ddot_buf), dtype=np.float64)

    # @profile
    def build_kISDF_obj(self, c:int = 5, m:int = 5, global_selection=False):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.IP_ID = self.select_IP(c, m, global_selection=global_selection)  # prim_gridID
        self.IP_ID = np.asarray(self.IP_ID, dtype=np.int32)
        # print("IP_ID = ", self.IP_ID)
        # print("len(IP_ID) = ", len(self.IP_ID))
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "select IP")
        
        if self.coords is None:
            from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
            df_tmp = MultiGridFFTDF2(self.cell)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3).copy()
        
        self.naux = self.aoRg.shape[1]
        # print("naux = ", self.naux)
        self.c    = c
        
        self._allocate_jk_buffer()
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _construct_aux_basis_W_kSym_Direct(self, self.IO_buf)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aux basis and W matrix")

    def build_IP_auxbasis(self, IO_File:str = None, c:int = 5, m:int = 5):
        raise NotImplementedError("build_IP_auxbasis is not implemented in PBC_ISDF_Info_kSym_Direct")

    def build_auxiliary_Coulomb(self):
        raise NotImplementedError("build_auxiliary_Coulomb is not implemented in PBC_ISDF_Info_kSym_Direct")

C = 7
M = 5

from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info

if __name__ == '__main__':
    
    # boxlen = 3.5668
    # prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
        
    # atm = [
    #     ['C', (0.     , 0.     , 0.    )],
    #     ['C', (0.8917 , 0.8917 , 0.8917)],
    #     ['C', (1.7834 , 1.7834 , 0.    )],
    #     ['C', (2.6751 , 2.6751 , 0.8917)],
    #     ['C', (1.7834 , 0.     , 1.7834)],
    #     ['C', (2.6751 , 0.8917 , 2.6751)],
    #     ['C', (0.     , 1.7834 , 1.7834)],
    #     ['C', (0.8917 , 2.6751 , 2.6751)],
    # ]
    
    KE_CUTOFF = 256
    boxlen = 4.2
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['Li', (0.0 , 0.0 , 0.0)],
        ['Li', (2.1 , 2.1 , 0.0)],
        ['Li', (0.0 , 2.1 , 2.1)],
        ['Li', (2.1 , 0.0 , 2.1)],
        ['H',  (0.0 , 0.0 , 2.1)],
        ['H',  (0.0 , 2.1 , 0.0)],
        ['H',  (2.1 , 0.0 , 0.0)],
        ['H',  (2.1 , 2.1 , 2.1)],
    ]
    
    prim_cell = isdf_k.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis='gth-dzvp')
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    # Ls = [2, 2, 2]
    # Ls = [2, 1, 3]
    # Ls = [3, 3, 3]
    # Ls = [2, 2, 3]
    Ls = [1, 1, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = isdf_k.build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=KE_CUTOFF, mesh=mesh, basis='gth-dzvp')
    
    pbc_isdf_info = PBC_ISDF_Info_kSym_Direct(cell, 800 * 1000 * 1000, Ls=Ls)
    pbc_isdf_info.build_kISDF_obj(c=C, m=M)
    
    # pbc_isdf_info2 = isdf_k.PBC_ISDF_Info_kSym(cell, 80 * 1000 * 1000, Ls=Ls, outcore=False, with_robust_fitting=False, aoR=None)
    # pbc_isdf_info2.build_IP_auxbasis(c=C, m=M, IP_ID=pbc_isdf_info.IP_ID)
    # pbc_isdf_info2.build_auxiliary_Coulomb()
    
    
    from pyscf.pbc import scf
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()
    
    exit(1)
    
    #### another test ####
    
    pbc_isdf_info = PBC_ISDF_Info_kSym_Direct(cell, 800 * 1000 * 1000, Ls=Ls)
    pbc_isdf_info.build_kISDF_obj(c=C, m=M, global_selection=True)
    
    from pyscf.pbc import scf
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()
    