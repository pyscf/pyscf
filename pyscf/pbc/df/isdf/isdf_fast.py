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
from pyscf.lib import logger
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
from pyscf.pbc.dft import multigrid
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from isdf_jk import _benchmark_time
import isdf_ao2mo as isdf_ao2mo
import isdf_jk as isdf_jk
from isdf_eval_gto import ISDF_eval_gto
from isdf_tools_kSampling import _kmesh_to_Kpoints

############ global variables ############

BASIS_CUTOFF               = 1e-18  # too small may lead to numerical instability
CRITERION_CALL_PARALLEL_QR = 256

############ subroutines --- select IP and build aux basis ############

def _select_IP_direct(mydf, c:int, m:int, first_natm=None, global_IP_selection=True, 
                      aoR_cutoff = None,
                      rela_cutoff = 0.0, 
                      no_retriction_on_nIP = False,
                      use_mpi=False):
    r''' Select the interpolation points (IP) based on the given criteria.
    
    Args:
        mydf : object
            The interpolative separable density fitting (ISDF) object.
            
        c : int
            if rela_cutoff is None or 0.0, control the number of IPs selected with c * nao at most.
            
            
        rela_cutoff : float
            The relative cutoff value for IP selection.
            IPs with values smaller than rela_cutoff * max_QR_value will not be selected.
            Default is 0.0. (no control via QR values)


    Kwargs:
        first_natm : int
            The number of atoms to be considered for IP selection.
            If not given, all atoms will be considered.
            If set,       it *should* be used in ISDF with k-sampling class, first_natm is the number of atoms in the first cell.
            
        global_IP_selection : bool
            Whether to perform global IP selection.
            If True, IPs will be re-selected after the individual selection of each atom.
            Default is True.
            
        aoR_cutoff : float
            The cutoff value for AO values.
            Points with max AO values smaller than this cutoff will not be considered for IP selection.
            Default is None.
            
        no_retriction_on_nIP : bool
            Whether to remove the restriction on the number of IPs.
            If True, there will be no limit on the number of selected IPs.
            Default is False.
            
        use_mpi : bool
            Whether to use MPI for parallel computation.
            Default is False.
            
        m : int 
            Control the number of 

        Returns:
            selected_IP : list
                The list of selected interpolation points.
        
        Ref:
        
        (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720
        
    '''
    
    if use_mpi:
        from isdf_tools_mpi import rank, comm_size, comm, allgather, bcast
        if rank == 0:
            logger.debug4(mydf, "_select_IP_direct: num_threads = %d", lib.num_threads())
    else:
        rank = 0
        logger.debug4(mydf, "_select_IP_direct: num_threads = %d", lib.num_threads())

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
    
    if hasattr(mydf, "IO_buf"):
        buf = mydf.IO_buf
    else:
        buf = np.zeros((buf_size), dtype=np.float64)
        mydf.IO_buf = buf
        
    if buf.size < buf_size:
        mydf.IO_buf = np.zeros((buf_size), dtype=np.float64)
        buf = mydf.IO_buf
    buf_tmp = np.ndarray((buf_size), dtype=np.float64, buffer=buf)

    ### loop over atm ###

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(mydf.cell)
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    assert coords is not None

    results = []

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
                logger.debug4(mydf, "_select_IP_direct: aoR_cutoff = %12.6e", aoR_cutoff)
                max_row = np.max(np.abs(aoR_atm), axis=1)
                where = np.where(max_row > mydf.aoR_cutoff)[0]
                aoR_atm = aoR_atm[where]
                nao_tmp = aoR_atm.shape[0]

            # create buffer for this atm

            dtypesize = buf.dtype.itemsize

            offset += nao_tmp*grid_ID.shape[0] * dtypesize

            nao_atm  = nao_per_atm[atm_id]
            naux_now = int(np.sqrt(c*nao_atm)) + m
            naux2_now = naux_now * naux_now

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
            offset       += (nthread+1)*(grid_ID.shape[0]+1) * dtypesize
            global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64, buffer=buf_tmp, offset=offset)
            offset       += grid_ID.shape[0] * dtypesize

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
                        
            cutoff   = abs(R[npt_find-1, npt_find-1])
            pivot = pivot[:npt_find]
            pivot.sort()
            results.extend(list(grid_ID[pivot]))
            
            logger.debug4(mydf, "_select_IP_direct: ngrid = %d, npt_find = %d, cutoff = %12.6e", grid_ID.shape[0], npt_find, cutoff)

    if use_mpi:
        comm.Barrier()
        results = allgather(results)
    results.sort()
    
    ### global IP selection, we can use this step to avoid numerical issue ###
    
    ### but this step is not necessary if locality is explored ###

    if global_IP_selection and rank == 0:

        #if mydf.verbose:
        #    print("global IP selection")

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

        R       = np.ndarray((naux2_now, len(results)), dtype=np.float64)
        offset += naux2_now*len(results) * dtypesize

        aoRg1   = np.ndarray((naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
        offset += naux_now*len(results) * dtypesize

        aoRg2   = np.ndarray((naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
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

        npt_find      = ctypes.c_int(0)
        pivot         = np.arange(len(results), dtype=np.int32)
        thread_buffer = np.ndarray((nthread+1, len(results)+1), dtype=np.float64, buffer=buf_tmp, offset=offset)
        offset       += (nthread+1)*(len(results)+1) * dtypesize
        global_buffer = np.ndarray((1, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
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
                
        cutoff   = abs(R[npt_find-1, npt_find-1])
        pivot = pivot[:npt_find]

        pivot.sort()

        results = np.array(results, dtype=np.int32)
        results = list(results[pivot])
    
        logger.debug4(mydf, "_select_IP_direct: ngrid = %d, npt_find = %d, cutoff = %12.6e", len(results), npt_find, cutoff)

    if global_IP_selection and use_mpi:
        results = bcast(results)

    return results

def build_aux_basis(mydf, debug=True, use_mpi=False):
    '''build the auxiliary basis for ISDF given IP_ID and aoR.
    '''
    
    if use_mpi:
        from isdf_tools_mpi import rank, bcast, comm
    
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
        
    e = None
    h = None
    
    if not use_mpi or (use_mpi and rank == 0):
        A = np.asarray(lib.ddot(aoRg.T, aoRg, c=buffer1), order='C')  # buffer 1 size = naux * naux
        lib.square_inPlace(A)
        
        t11 = (lib.logger.process_clock(), lib.logger.perf_counter())
        e, h = scipy.linalg.eigh(A)
        t12 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        _benchmark_time(t11, t12, "diag_A", mydf)
        
        logger.debug4(mydf, "build_aux_basis: condition number = %12.6e", e[-1]/e[0])
        
        where = np.where(e > e[-1]*1e-16)[0]
        e = e[where]
        h = h[:, where]
        
    if use_mpi:
        e = bcast(e)
        h = bcast(h)
    
    mydf.aux_basis = np.asarray(lib.ddot(aoRg.T, aoR), order='C')   # buffer 2 size = naux * ngrids
    lib.square_inPlace(mydf.aux_basis)
    
    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)

    nThread = lib.num_threads()
    nGrids  = aoR.shape[1]
    Bunchsize = nGrids // nThread
    
    buffer2 = np.ndarray((e.shape[0] , mydf.aux_basis.shape[1]), dtype=np.double, buffer=mydf.jk_buffer,
             offset=mydf.naux * mydf.naux * mydf.jk_buffer.dtype.itemsize)
    B = np.asarray(lib.ddot(h.T, mydf.aux_basis, c=buffer2), order='C')
    lib.d_i_ij_ij(1.0/e, B, out=B)
    np.asarray(lib.ddot(h, B, c=mydf.aux_basis), order='C')

    if use_mpi:
        comm.Barrier()

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    _benchmark_time(t1, t2, "build_auxiliary_basis", mydf)

    mydf.naux = naux
    mydf.aoRg = aoRg

from pyscf.pbc import df

class PBC_ISDF_Info(df.fft.FFTDF):
    ''' Interpolative separable density fitting (ISDF) for periodic systems.
    Not recommended as the locality is not explored! 
    
    Examples:

    >>> #### code to construct aoR ommited ###
    >>> aoR  *= np.sqrt(cell.vol / ngrids)
    >>> pbc_isdf = PBC_ISDF_Info(cell, aoR=aoR)
    >>> pbc_isdf.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=False)
    >>> pbc_isdf.build_auxiliary_Coulomb()
    >>> from pyscf.pbc import scf
    >>> mf = scf.RHF(cell)
    >>> pbc_isdf.direct_scf = mf.direct_scf
    >>> mf.with_df = pbc_isdf
    >>> mf.verbose = 0
    >>> mf.kernel()
    
    '''

    def __init__(self, mol:Cell, 
                 aoR: np.ndarray = None,  ## convention: aoR is scaled by np.sqrt(mol.vol / ngrids)
                 with_robust_fitting=True,
                 kmesh=None,
                 get_partition=True,
                 verbose = None
                 ):

        if kmesh == None:
            kmesh = numpy.asarray([1,1,1], dtype=numpy.int32)
        KPoints = _kmesh_to_Kpoints(mol, kmesh)   ### WARNING: this subroutine is not correct ! 
        
        super().__init__(cell=mol, kpts=KPoints)

        if verbose is not None:
            self.verbose = verbose

        ## the following variables are used in build_sandeep

        self.with_robust_fitting = with_robust_fitting

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
        self.nao  = mol.nao_nr()

        from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

        logger.info(self, "PBC_ISDF_Info: mol.ke_cutoff = %f", mol.ke_cutoff)

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

        self.ao2atomID = ao2atomID

        # given aoG, determine at given grid point, which ao has the maximal abs value

        if aoR is not None:
            self.partition = np.argmax(np.abs(aoR), axis=0)
            # map aoID to atomID
            self.partition = np.asarray([ao2atomID[x] for x in self.partition])
            grids   = df_tmp.grids
            self.coords  = np.asarray(grids.coords).reshape(-1,3)
            self._numints = df_tmp._numint
        else:
            grids   = df_tmp.grids
            coords  = np.asarray(grids.coords).reshape(-1,3)
            NumInts = df_tmp._numint

            coords_now = coords
            
            if kmesh is not None:
                
                mesh       = mol.mesh
                meshPrim   = np.array(mesh, dtype=np.int32) // kmesh
                coords_now = coords_now.reshape(kmesh[0], meshPrim[0], kmesh[1], meshPrim[1], kmesh[2], meshPrim[2], 3)
                coords_now = coords_now.transpose(0, 2, 4, 1, 3, 5, 6).reshape(-1, 3)
                coords_now = coords_now[:np.prod(meshPrim), :]

            self.partition = np.zeros(coords_now.shape[0], dtype=np.int32)

            from isdf_eval_gto import ISDF_eval_gto

            if hasattr(self, "IO_buf"):
                logger.debug4(self, "PBC_ISDF_Info: IO_buf is already allocated")
            else:
                logger.debug4(self, "PBC_ISDF_Info: IO_buf is not allocated")
                max_memory = max(2000, self.max_memory-lib.current_memory()[0])
                self.IO_buf = np.zeros((int(max_memory*1e6//8),), dtype=np.double)

            logger.debug4(self, "PBC_ISDF_Info: IO_buf.size     = %d", self.IO_buf.size)
            logger.debug4(self, "PBC_ISDF_Info: coords.shape[0] = %d", coords_now.shape[0])
            logger.debug4(self, "PBC_ISDF_Info: self.nao        = %d", self.nao)

            bufsize = min(self.IO_buf.size, 4*1e9/8) // 2
            bunchsize = int(bufsize / (self.nao))

            assert bunchsize > 0
            
            if get_partition and aoR is None:
                for p0, p1 in lib.prange(0, coords_now.shape[0], bunchsize):
                    AoR_Buf = np.ndarray((self.nao, p1-p0), dtype=np.complex128, buffer=self.IO_buf, offset=0)
                    AoR_Buf = ISDF_eval_gto(self.cell, coords=coords_now[p0:p1], out=AoR_Buf)
                    res     = np.argmax(np.abs(AoR_Buf), axis=0)
                    self.partition[p0:p1] = np.asarray([ao2atomID[x] for x in res])
                    AoR_Buf = None
            else:
                self.partition = None
                
            res = None
            
            self.coords = coords
            self._numints = NumInts

    def _allocate_jk_buffer(self, datatype):

        if self.jk_buffer is None:

            nao    = self.nao
            ngrids = self.ngrids
            naux   = self.naux

            logger.debug4(self, "_allocate_jk_buffer: nao = %d, ngrids = %d, naux = %d", nao, ngrids, naux) 
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
                self.ddot_buf  = np.zeros((nThreadsOMP, max((nao*nao)+2, ngrids)), dtype=datatype)

        else:
            assert self.jk_buffer.dtype == datatype
            assert self.ddot_buf.dtype == datatype

    def build(self):
        raise NotImplementedError

    def build_only_partition(self):
        raise NotImplementedError

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
    
    def get_A_B(self):
        '''aux basis is contructed via solving AX=B
        '''

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
        ''' select the interpolation points (IP) based on the given criteria using Sandeep's method.
        Ref:
        (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720
        '''

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

        # for each atm

        if not hasattr(self, "use_mpi"):
            self.use_mpi = False
            rank = 0

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        if IP_ID is None:
            IP_ID  = _select_IP_direct(self, c, m, global_IP_selection=global_IP_selection, use_mpi=self.use_mpi)
            IP_ID.sort()
            IP_ID  = np.array(IP_ID, dtype=np.int32)
        self.IP_ID = np.array(IP_ID, dtype=np.int32)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if rank == 0:
            _benchmark_time(t1, t2, "build_IP", self)
        t1 = t2

        # build the auxiliary basis

        self.c = c
        build_aux_basis(self)

    def build_auxiliary_Coulomb(self, cell:Cell = None, mesh=None, debug=True):
        ''' build V and W matrix see eq(13) of Sandeep2022.
        
        Ref:
        (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720
        '''

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
            #print("cell.__class__ = ", cell.__class__)

        coulG = tools.get_coulG(cell, mesh=mesh)

        V_R = constrcuct_V_CCode(self.aux_basis, mesh, coulG)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_V_R", self)
        t1 = t2

        W = lib.ddot(a=self.aux_basis, b=V_R.T)

        self.coulG = coulG.copy()

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug:
            _benchmark_time(t1, t2, "build_auxiliary_Coulomb_W", self)

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

        logger.debug4(self, "check_AOPairError")

        for i in range(nao):
            
            coeff = numpy.einsum('k,jk->jk', aoRg[i, :], aoRg).reshape(-1, self.IP_ID.shape[0])
            aoPair = numpy.einsum('k,jk->jk', aoR[i, :], aoR).reshape(-1, aoR.shape[1])
            aoPair_approx = coeff @ self.aux_basis

            diff = aoPair - aoPair_approx
            diff_pair_abs_max = np.max(np.abs(diff), axis=1)

            for j in range(diff_pair_abs_max.shape[0]):
                logger.debug4(self, "(%5d, %5d, %15.8e)", i, j, diff_pair_abs_max[j])

    def __del__(self):
        return

    @property
    def kpt(self):
        return np.zeros(3)

    def get_pp(self, kpts=None):
        if hasattr(self, "PP") and self.PP is not None:
            return self.PP
        else:
            
            t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
            cell = self.cell.copy()
            cell.omega = 0.0
            if hasattr(self, "ke_cutoff_pp"):
                cell.ke_cutoff = self.ke_cutoff_pp
            cell.build()
            df_tmp = multigrid.MultiGridFFTDF2(cell)
            v_pp_loc2_nl = df_tmp.get_pp()
            v_pp_loc1_G = df_tmp.vpplocG_part1
            v_pp_loc1 = multigrid.multigrid_pair._get_j_pass2(df_tmp, v_pp_loc1_G)
            self.PP = (v_pp_loc1 + v_pp_loc2_nl)[0]
            t1 = (lib.logger.process_clock(), lib.logger.perf_counter()) 
            if self.verbose:
                _benchmark_time(t0, t1, "get_pp", self)
                
            #### kpts #### 
            
            if kpts is not None:
                
                nkpts = kpts.shape[0]
                
                if hasattr(self, "kmesh") and self.kmesh is not None:
                    pass
                else:
                    self.kmesh = np.asarray([1,1,1], dtype=np.int32)
                kmesh = np.asarray(self.kmesh, dtype=np.int32)
                #print("kmesh = ", kmesh)
                #print("kpts.shape = ", kpts.shape)
                assert kpts.shape[0] == np.prod(self.kmesh, dtype=np.int32) or kpts.shape[0] == 1 or kpts.ndim == 1
                is_single_kpt = kpts.shape[0] == 1 or kpts.ndim == 1
                
                if is_single_kpt:
                    #### xxx use the original version, Xing's code is not the same as FFTDF's result xxx ####
                    if hasattr(self, "_use_super_pp"):
                        if self._use_super_pp:
                            print("super pp is used!")
                            self.PP = super().get_pp(kpts=np.zeros(3))
                    #### use the calculated one by default ####
                    print("single kpt case!")
                    return self.PP
                
                #### the following is used to test KRHF #### 
                
                ### info used in super().get_pp() ###
                
                assert hasattr(self, "prim_cell")
                
                # super_cell = self.cell
                # super_mesh = self.cell.mesh
                # self.cell = self.prim_cell
                # self.mesh = self.cell.mesh
                # from pyscf.pbc.dft import gen_grid
                # self.grids = gen_grid.UniformGrids(self.cell)
                # self.PP = super().get_pp(kpts=kpts)
                # self.cell = super_cell
                # self.mesh = super_mesh
                # return self.PP
                
                nao_prim = self.cell.nao_nr() // nkpts 
                assert self.cell.nao_nr() % nkpts == 0
                self.PP = self.PP[:nao_prim, :].copy()
                
                n_complex = self.kmesh[0] * self.kmesh[1] * (self.kmesh[2]//2+1)
                n_cell    = np.prod(self.kmesh)
                
                PP_complex = np.zeros((nao_prim, n_complex * nao_prim), dtype=np.complex128)
                PP_real    = np.ndarray((nao_prim, n_cell * nao_prim), dtype=np.double, buffer=PP_complex)
                PP_real.ravel()[:] = self.PP.ravel()
                buf_fft    = np.zeros((nao_prim, n_complex, nao_prim), dtype=np.complex128)
                
                fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
                assert fn1 is not None 
                
                fn1(
                    PP_real.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_prim),
                    ctypes.c_int(nao_prim),
                    kmesh.ctypes.data_as(ctypes.c_void_p),
                    buf_fft.ctypes.data_as(ctypes.c_void_p)
                )
                del buf_fft 
                
                from  isdf_tools_densitymatrix import pack_JK_in_FFT_space
                
                PP_complex = PP_complex.conj().copy()
                self.PP = pack_JK_in_FFT_space(PP_complex, kmesh, nao_prim)
                
            return self.PP
        
    def LS_THC_recompression(self, X:np.ndarray, force_LS_THC=True):
        
        from isdf_ao2mo import LS_THC 
        
        if force_LS_THC:
            self.with_robust_fitting = False
            self.force_LS_THC        = True
            self.W    = LS_THC(self, X) / (self.ngrids/self.cell.vol)
            self.aoRg = X
            self.V_R  = None
        else:
            self.force_LS_THC        = False
            self.W2    = LS_THC(self, X) / (self.ngrids/self.cell.vol)
            self.aoRg2 = X
    
    def aoRg_full(self):
        return self.aoRg, None
        
    ##### functions defined in isdf_ao2mo.py #####

    get_eri = get_ao_eri = isdf_ao2mo.get_eri
    ao2mo = get_mo_eri = isdf_ao2mo.general
    ao2mo_7d = isdf_ao2mo.ao2mo_7d  # seems to be only called in kadc and kccsd, NOT implemented!

    ##### functions defined in isdf_jk.py #####

    get_jk = isdf_jk.get_jk_dm



if __name__ == '__main__':

    C = 15
    
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
    cell.verbose = 10

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
    nx     = grids.mesh[0]

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR=aoR)
    pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=C, global_IP_selection=False)
    pbc_isdf_info.build_auxiliary_Coulomb()

    ### perform scf ###

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()

    print("mf.with_df.IP_ID = ", mf.with_df.IP_ID)
    print("mf.with_df.partition = ", mf.with_df.partition)
    
    for i in range(cell.natm):
        print("i = ", i, "partition = ", mf.with_df.partition[mf.with_df.partition == i].shape[0])

    #exit(1)

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
    #mf.kernel()
    pbc_isdf_info.with_robust_fitting = True

    ##### test the LS_THC_recompression ##### 
    
    _pbc_isdf_info = PBC_ISDF_Info(cell, aoR)
    _pbc_isdf_info.build_IP_Sandeep(build_global_basis=True, c=12, global_IP_selection=False)

    pbc_isdf_info.LS_THC_recompression(_pbc_isdf_info.aoRg, force_LS_THC=False)
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 10
    mf.conv_tol = 1e-7
    mf.kernel()
    
    pbc_isdf_info.LS_THC_recompression(_pbc_isdf_info.aoRg)
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 10
    mf.conv_tol = 1e-7
    mf.kernel()