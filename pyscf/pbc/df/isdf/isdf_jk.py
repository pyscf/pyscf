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
import numpy as np
import numpy
import ctypes

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
libpbc = lib.load_library('libpbc')

##################################################
#
# only Gamma Point
#
##################################################

######### tools #########

def _benchmark_time(t1, t2, label, rec):
    lib.logger.debug4(rec, "%20s wall time: %12.6f CPU time: %12.6f" % (label, t2[1] - t1[1], t2[0] - t1[0]))

def _contract_j_dm(mydf, dm, with_robust_fitting=True, use_mpi=False):
    '''

    Args:
        mydf  : density fitting object
        dm    : the density matrix

    '''
    
    assert use_mpi == False
    
    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao  = dm.shape[0]

    cell = mydf.cell
    assert cell.nao == nao
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

    ## constract dm and aoR

    # need allocate memory, size = nao  * ngrid, (buffer 1)

    lib.ddot(dm, aoR, c=buffer1)  
    tmp1 = buffer1

    # need allocate memory, size = ngrid, (buffer 2)

    density_R = np.asarray(lib.multiply_sum_isdf(aoR, tmp1, out=buffer2), order='C')

    # need allocate memory, size = nao  * naux, (buffer 3)

    # lib.dslice(tmp1, IP_ID, buffer3)
    # tmp1 = buffer3
    tmp1 = lib.ddot(dm, aoRg)  

    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg, tmp1, out=buffer4),
                            order='C')  # need allocate memory, size = naux, (buffer 4)

    # This should be the leading term of the computation cost in a single-thread mode.

    # need allocate memory, size = naux, (buffer 5)

    J = None

    if with_robust_fitting:
        J = np.asarray(lib.ddot_withbuffer(V_R, density_R.reshape(-1,1), c=buffer5.reshape(-1,1), buf=mydf.ddot_buf), order='C').reshape(-1)   # with buffer, size 
        
        # do not need allocate memory, use buffer 3

        J = np.asarray(lib.d_ij_j_ij(aoRg, J, out=buffer3), order='C')

        # need allocate memory, size = nao  * nao, (buffer 6)

        J = np.asarray(lib.ddot_withbuffer(aoRg, J.T, c=buffer6, buf=mydf.ddot_buf), order='C')
            
        # do not need allocate memory, use buffer 2

        J2 = np.asarray(lib.dot(V_R.T, density_Rg.reshape(-1,1), c=buffer2.reshape(-1,1)), order='C').reshape(-1)

        # do not need allocate memory, use buffer 1

        # J2 = np.einsum('ij,j->ij', aoR, J2)
        J2 = np.asarray(lib.d_ij_j_ij(aoR, J2, out=buffer1), order='C')

        # do not need allocate memory, use buffer 6

        # J += np.asarray(lib.dot(aoR, J2.T), order='C')
        lib.ddot_withbuffer(aoR, J2.T, c=J, beta=1, buf=mydf.ddot_buf)

    #### step 3. get J term3

    # do not need allocate memory, use buffer 2

    tmp = np.asarray(lib.dot(W, density_Rg.reshape(-1,1), c=buffer8.reshape(-1,1)), order='C').reshape(-1)
    
    # do not need allocate memory, use buffer 1 but viewed as buffer 7
    
    tmp = np.asarray(lib.d_ij_j_ij(aoRg, tmp, out=buffer7), order='C')
    
    # do not need allocate memory, use buffer 6
    
    if with_robust_fitting:
        lib.ddot_withbuffer(aoRg, -tmp.T, c=J, beta=1, buf=mydf.ddot_buf)
    else:
        J = buffer6
        lib.ddot_withbuffer(aoRg, tmp.T, c=J, beta=0, buf=mydf.ddot_buf)

    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm", mydf)

    return J * ngrid / vol

def _contract_j_dm_fast(mydf, dm, with_robust_fitting=True, use_mpi=False):

    assert use_mpi == False
    
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
             
    #### step 3. get J 
    
    J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_fast", mydf)
    
    return J * ngrid / vol

def _contract_j_dm_wo_robust_fitting(mydf, dm, with_robust_fitting=False, use_mpi=False):
    
    assert with_robust_fitting == False
    assert use_mpi == False
    
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
    
    tmp1 = lib.ddot(dm, aoRg)  
    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg, tmp1),
                            order='C') 
    tmp = np.asarray(lib.dot(W, density_Rg.reshape(-1,1)), order='C').reshape(-1)
    tmp = np.asarray(lib.d_ij_j_ij(aoRg, tmp), order='C')

    J = lib.ddot(aoRg, tmp.T)

    del tmp1 
    tmp1 = None
    del tmp 
    tmp = None
    del density_Rg
    density_Rg = None

    t2 = (logger.process_clock(), logger.perf_counter())

    _benchmark_time(t1, t2, "_contract_j_dm_wo_robust_fitting", mydf)
    
    return J * ngrid / vol

def _contract_k_dm(mydf, dm, with_robust_fitting=True, use_mpi=False):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    assert use_mpi == False
    
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

    #### step 1. get density value on real space grid and IPs

    # need allocate memory, size = nao  * ngrid, this buffer does not need anymore  (buffer 1)

    density_RgR  = np.asarray(lib.dot(dm, aoR, c=buffer1), order='C')
    
    # need allocate memory, size = naux * ngrid                                     (buffer 2)

    # density_RgR  = np.asarray(lib.dot(aoRg.T, density_RgR, c=buffer2), order='C')
    lib.ddot(aoRg.T, density_RgR, c=buffer2)
    density_RgR = buffer2

    # need allocate memory, size = naux * naux                                      (buffer 3)

    density_RgRg = lib.ddot(dm, aoRg)
    density_RgRg = lib.ddot(aoRg.T, density_RgRg)

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

        # do not need allocate memory, size = naux * nao,   (buffer 1, but viewed as buffer5)
    
        K = np.asarray(lib.ddot_withbuffer(tmp, aoR.T, c=buffer5, buf=mydf.ddot_buf), order='C')

        ### the order due to the fact that naux << ngrid  # need allocate memory, size = nao * nao,           (buffer 4)

        K  = np.asarray(lib.ddot_withbuffer(aoRg, K, c=buffer4, buf=mydf.ddot_buf), order='C')

        K += K.T

    #### step 3. get K term3

    ### todo: optimize the following 4 lines, it seems that they may not parallize!
    # pointwise multiplication, do not need allocate memory, size = naux * naux, use buffer for (buffer 3)
    # tmp = W * density_RgRg

    lib.cwise_mul(W, density_RgRg, out=density_RgRg)
    tmp = density_RgRg

    # do not need allocate memory, size = naux * nao, use buffer 2 but viewed as buffer 6
    
    tmp = np.asarray(lib.dot(tmp, aoRg.T, c=buffer6), order='C')

    # K  -= np.asarray(lib.dot(aoRg, tmp, c=K, beta=1), order='C')     # do not need allocate memory, size = nao * nao, (buffer 4)
    
    if with_robust_fitting:
        lib.ddot_withbuffer(aoRg, -tmp, c=K, beta=1, buf=mydf.ddot_buf)
    else:
        K = buffer4
        lib.ddot_withbuffer(aoRg, tmp, c=K, beta=0, buf=mydf.ddot_buf)

    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm", mydf)

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
    if hasattr(mydf, "ddot_buf") and mydf.ddot_buf is not None:
        K = lib.ddot_withbuffer(aoRg, tmp, buf=mydf.ddot_buf)
    else:
        K = lib.ddot(aoRg, tmp)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    # if mydf.verbose:
    _benchmark_time(t1, t2, "_contract_k_dm_wo_robust_fitting", mydf)
    
    del tmp
    tmp = None
    del density_RgRg
    density_RgRg = None
    
    return K * ngrid / vol # take care this factor 

def get_jk_dm(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, 
           use_mpi = False, **kwargs):
    
    '''JK for given k-point'''
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1 or dm.shape[0] == 2
        #dm = dm[0]
    else:
        assert dm.ndim == 2
        dm = dm.reshape(1, dm.shape[0], dm.shape[1])
        
    nset = dm.shape[0]

    if hasattr(mydf, 'Ls'):
        from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
        dm = symmetrize_dm(dm, mydf.Ls)
    else:
        if hasattr(mydf, 'kmesh'):
            from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
            dm = symmetrize_dm(dm, mydf.kmesh)

    #### perform the calculation ####

    if mydf.jk_buffer is None:  # allocate the buffer for get jk
        mydf._allocate_jk_buffer(dm.dtype)

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
    else:
        exxdiv = None

    #vj = vk = None
    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)

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

    for iset in range(nset):

        if with_j:
            if mydf.with_robust_fitting:
                vj[iset] = _contract_j_dm_fast(mydf, dm[iset], mydf.with_robust_fitting, use_mpi)
            else:
                vj[iset] = _contract_j_dm_wo_robust_fitting(mydf, dm[iset], mydf.with_robust_fitting, use_mpi)   
        if with_k:
            if mydf.with_robust_fitting:
                vk[iset] = _contract_k_dm(mydf, dm[iset], mydf.with_robust_fitting, use_mpi)
            else:
                vk[iset] = _contract_k_dm_wo_robust_fitting(mydf, dm[iset], mydf.with_robust_fitting, use_mpi)
            if exxdiv == 'ewald':
                print("WARNING: ISDF does not support ewald")

    ##### the following code is added to deal with _ewald_exxdiv_for_G0 #####
    
    from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks, _ewald_exxdiv_for_G0
    
    kpts = kpt.reshape(1,3)
    kpts = np.asarray(kpts)
    dm_kpts = dm.copy()
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if nset > 2:
        logger.warn(mydf, 'nset > 2, please confirm what you are doing, for RHF nset == 1, for UHF nset == 2')
    assert nkpts == 1
    
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == 1

    if is_zero(kpts_band) and is_zero(kpts):
        vk = vk.reshape(nset,nband,nao,nao)
    else:
        raise NotImplementedError("ISDF does not support kpts_band != 0")

    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(mydf.cell, kpts, dms, vk, kpts_band=kpts_band)
    
    vk = vk.reshape(nset,nao,nao)

    t1 = log.timer('sr jk', *t1)

    return vj, vk