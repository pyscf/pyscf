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
from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.lib import logger, zdotNN, zdotCN, zdotNC
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member


from memory_profiler import profile

##################################################
#
# only Gamma Point
#
##################################################

def _benchmark_time(t1, t2, label):
    print("%20s wall time: %12.6f CPU time: %12.6f" % (label, t2[1] - t1[1], t2[0] - t1[0]))

def _contract_j_mo(mydf, mo_coeffs):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    nao  = mo_coeffs.shape[0]
    # nocc = mo_coeffs.shape[1]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    V_R  = mydf.V_R
    # naux = aoRg.shape[1]

    #### step 1. get density value on real space grid and IPs

    #### TODO: make the following transformation linear-scaling

    moRg = np.asarray(lib.dot(mo_coeffs.T, aoRg), order='C')
    moR  = np.asarray(lib.dot(mo_coeffs.T, aoR), order='C')

    #### step 2. get J term1 and term2

    density_R  = np.sum(moR*moR, axis=0)
    density_Rg = np.sum(moRg*moRg, axis=0)

    ## TODO: remove the redundancy due to the symmetry

    rho_mu_nu_Rg = np.einsum('ij,kj->ikj', aoRg, aoRg)

    # J = np.asarray(lib.dot(V_R, density_R), order='C')
    J = np.dot(V_R, density_R)
    # J = np.asarray(lib.dot(rho_mu_nu_Rg, J), order='C')
    J = np.dot(rho_mu_nu_Rg, J)

    J2 = np.dot(V_R.T, density_Rg)
    J2 = np.einsum('ij,j->ij', aoR, J2)
    J += np.asarray(lib.dot(aoR, J2.T), order='C')

    #### step 3. get J term3

    # tmp = np.asarray(lib.dot(W, density_Rg), order='C')
    tmp = np.dot(W, density_Rg)
    # J -= np.asarray(lib.dot(rho_mu_nu_Rg, tmp), order='C')
    J -= np.dot(rho_mu_nu_Rg, tmp)
    # J = np.dot(rho_mu_nu_Rg, tmp)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_j_mo")

    return J * 2.0 * ngrid / vol  # 2.0 due to RHF

def _contract_k_mo(mydf, mo_coeffs):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    nao  = mo_coeffs.shape[0]
    nocc = mo_coeffs.shape[1]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    V_R  = mydf.V_R
    naux = aoRg.shape[1]

    #### step 1. get density value on real space grid and IPs

    moRg = np.asarray(lib.dot(mo_coeffs.T, aoRg), order='C')
    moR  = np.asarray(lib.dot(mo_coeffs.T, aoR), order='C')

    #### step 2. get K term1 and term2

    density_RgR = np.asarray(lib.dot(moRg.T, moR), order='C')
    tmp = V_R * density_RgR  # pointwise multiplication

    K = np.asarray(lib.dot(tmp, aoR.T), order='C')
    K = np.asarray(lib.dot(aoRg, K), order='C')  ### the order due to the fact that naux << ngrid
    K += K.T

    #### step 3. get K term3

    density_RgRg = np.asarray(lib.dot(moRg.T, moRg), order='C')
    tmp = W * density_RgRg  # pointwise multiplication
    tmp = np.asarray(lib.dot(tmp, aoRg.T), order='C')
    K -= np.asarray(lib.dot(aoRg, tmp), order='C')
    # K = np.asarray(lib.dot(aoRg, tmp), order='C')

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_k_mo")

    return K * 2.0 * ngrid / vol  # 2.0 due to RHF

def get_jk_mo(mydf, occ_mo_coeff, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''

    vj = vk = None

    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        raise NotImplementedError("ISDF does not support kpts_band != kpt")

    # cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not np.iscomplexobj(occ_mo_coeff)

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)

    if with_j:
        vj = _contract_j_mo(mydf, occ_mo_coeff)
    if with_k:
        vk = _contract_k_mo(mydf, occ_mo_coeff)
        if exxdiv == 'ewald':
            raise NotImplementedError("ISDF does not support ewald")


    t1 = log.timer('sr jk', *t1)
    return vj, vk

# @profile
def _contract_j_dm(mydf, dm):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao  = dm.shape[0]

    cell = mydf.cell
    # print("cell.nao", cell.nao)
    # print("nao     ", nao)
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    V_R  = mydf.V_R
    naux = aoRg.shape[1]
    IP_ID = mydf.IP_ID

    # print("address of mydf.aoR = ", mydf.aoR.__array_interface__['data'][0])
    # print("address of aoR      = ", aoR.__array_interface__['data'][0])

    #### step 2. get J term1 and term2

    # buffersize = nao * ngrid + ngrid + nao * naux + naux + naux + nao * nao
    # buffer = np.empty(buffersize, dtype=dm.dtype)
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

    # print("address of mydf.jk_buffer = ", id(mydf.jk_buffer))
    # print("address of buffer         = ", id(buffer))
    # print("address of buffer1        = ", id(buffer1))
    # import sys
    # print("size    of buffer1        = ", sys.getsizeof(buffer1))
    # print(buffer.__array_interface__['data'][0])
    # print(buffer1.__array_interface__['data'][0])

    # print(buffer2.__array_interface__['data'][0])
    # print(buffer3.__array_interface__['data'][0])
    # print(buffer4.__array_interface__['data'][0])
    # print(buffer5.__array_interface__['data'][0])
    # print(buffer6.__array_interface__['data'][0])
    # print(buffer7.__array_interface__['data'][0])
    # print(buffer8.__array_interface__['data'][0])
    # print("begin_work")

    # ptr1 = buffer1.__array_interface__['data'][0]
    # ptr2 = buffer2.__array_interface__['data'][0]
    # ptr3 = buffer3.__array_interface__['data'][0]
    # ptr4 = buffer4.__array_interface__['data'][0]
    # ptr5 = buffer5.__array_interface__['data'][0]
    # ptr6 = buffer6.__array_interface__['data'][0]
    # ptr7 = buffer7.__array_interface__['data'][0]
    # ptr8 = buffer8.__array_interface__['data'][0]

    ## constract dm and aoR

    # need allocate memory, size = nao  * ngrid, (buffer 1)

    # tmp1 = np.asarray(lib.dot(dm, aoR, c=buffer1), order='C')
    # print('dm.flags: %s' % str(dm.flags))
    # print('aoR.flags: %s' % str(aoR.flags))
    # print("before calling ddot in _contract_j_dm")
    # print("dm.shape", dm.shape)
    # print("aoR.shape", aoR.shape)
    lib.ddot(dm, aoR, c=buffer1)  
    # print("after calling ddot in _contract_j_dm")
    tmp1 = buffer1
    # print(buffer1.__array_interface__['data'][0])
    # print(tmp1.__array_interface__['data'][0])
    # print("address of aoR      = ", aoR.__array_interface__['data'][0])

    # assert tmp1.__array_interface__['data'][0] == ptr1

    # need allocate memory, size = ngrid, (buffer 2)
    density_R = np.asarray(lib.multiply_sum_isdf(aoR, tmp1, out=buffer2), order='C')

    # assert density_R.__array_interface__['data'][0] == ptr2

    # need allocate memory, size = nao  * naux, (buffer 3)

    lib.dslice(tmp1, IP_ID, buffer3)
    tmp1 = buffer3

    # assert tmp1.__array_interface__['data'][0] == ptr3

    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg, tmp1, out=buffer4),
                            order='C')  # need allocate memory, size = naux, (buffer 4)

    # assert density_Rg.__array_interface__['data'][0] == ptr4

    # This should be the leading term of the computation cost in a single-thread mode.

    # need allocate memory, size = naux, (buffer 5)

    J = np.asarray(lib.ddot_withbuffer(V_R, density_R.reshape(-1,1), c=buffer5.reshape(-1,1), buf=mydf.ddot_buf), order='C').reshape(-1)   # with buffer, size 

    # assert J.__array_interface__['data'][0] == ptr5

    # do not need allocate memory, use buffer 3

    # J = np.einsum('ij,j->ij', aoRg, J)
    J = np.asarray(lib.d_ij_j_ij(aoRg, J, out=buffer3), order='C')

    # assert J.__array_interface__['data'][0] == ptr3

    # need allocate memory, size = nao  * nao, (buffer 6)

    J = np.asarray(lib.ddot_withbuffer(aoRg, J.T, c=buffer6, buf=mydf.ddot_buf), order='C')
    # assert J.__array_interface__['data'][0] == ptr6

    # do not need allocate memory, use buffer 2

    J2 = np.asarray(lib.dot(V_R.T, density_Rg.reshape(-1,1), c=buffer2.reshape(-1,1)), order='C').reshape(-1)
    # assert J2.__array_interface__['data'][0] == ptr2

    # do not need allocate memory, use buffer 1

    # J2 = np.einsum('ij,j->ij', aoR, J2)
    J2 = np.asarray(lib.d_ij_j_ij(aoR, J2, out=buffer1), order='C')
    # assert J2.__array_interface__['data'][0] == ptr1

    # do not need allocate memory, use buffer 6

    # J += np.asarray(lib.dot(aoR, J2.T), order='C')
    lib.ddot_withbuffer(aoR, J2.T, c=J, beta=1, buf=mydf.ddot_buf)
    # assert J.__array_interface__['data'][0] == ptr6

    #### step 3. get J term3

    # do not need allocate memory, use buffer 2

    tmp = np.asarray(lib.dot(W, density_Rg.reshape(-1,1), c=buffer8.reshape(-1,1)), order='C').reshape(-1)

    # assert tmp.__array_interface__['data'][0] == ptr8

    # do not need allocate memory, use buffer 1 but viewed as buffer 7

    # tmp = np.einsum('ij,j->ij', aoRg, tmp)
    tmp = np.asarray(lib.d_ij_j_ij(aoRg, tmp, out=buffer7), order='C')

    # assert tmp.__array_interface__['data'][0] == ptr7

    # do not need allocate memory, use buffer 6

    # J -= np.asarray(lib.dot(aoRg, tmp.T), order='C')
    lib.ddot_withbuffer(aoRg, -tmp.T, c=J, beta=1, buf=mydf.ddot_buf)

    # assert J.__array_interface__['data'][0] == ptr6

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_j_dm")

    return J * ngrid / vol

# @profile
def _contract_k_dm(mydf, dm):
    '''

    Args:
        mydf       :
        mo_coeffs  : the occupied MO coefficients

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    # for i in range(dm.shape[0]):
    #     for j in range(dm.shape[1]):
    #         if abs(dm[i,j]) > 1e-6:
    #             print("dm[%d,%d] = %12.6f" % (i,j,dm[i,j]))

    nao  = dm.shape[0]

    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    assert ngrid == mydf.ngrids
    vol = cell.vol

    W    = mydf.W
    aoRg = mydf.aoRg
    aoR  = mydf.aoR
    V_R  = mydf.V_R
    naux = aoRg.shape[1]
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

    # print("address of mydf.jk_buffer = ", id(mydf.jk_buffer))
    # print("address of buffer         = ", id(buffer))
    # print("address of buffer1        = ", id(buffer1))

    # ptr1 = buffer1.__array_interface__['data'][0]

    #### step 1. get density value on real space grid and IPs

    # need allocate memory, size = nao  * ngrid, this buffer does not need anymore  (buffer 1)

    density_RgR  = np.asarray(lib.dot(dm, aoR, c=buffer1), order='C')
    # print("buffer1.size", buffer1.size)
    # assert density_RgR.__array_interface__['data'][0] == ptr1

    # need allocate memory, size = naux * ngrid                                     (buffer 2)

    # density_RgR  = np.asarray(lib.dot(aoRg.T, density_RgR, c=buffer2), order='C')
    lib.ddot(aoRg.T, density_RgR, c=buffer2)
    density_RgR = buffer2
    # assert density_RgR.__array_interface__['data'] == buffer2.__array_interface__['data']

    # need allocate memory, size = naux * naux                                      (buffer 3)

    # density_RgRg = density_RgR[:, IP_ID]
    lib.dslice(density_RgR, IP_ID, buffer3)
    density_RgRg = buffer3

    # assert density_RgRg.__array_interface__['data'] == buffer3.__array_interface__['data']

    #### step 2. get K term1 and term2

    ### todo: optimize the following 4 lines, it seems that they may not parallize!

    # tmp = V_R * density_RgR  # pointwise multiplication, TODO: this term should be parallized
    # do not need allocate memory, size = naux * ngrid, (buffer 2)

    # tmp = np.asarray(lib.cwise_mul(V_R, density_RgR, out=buffer2), order='C')

    lib.cwise_mul(V_R, density_RgR, out=buffer2)
    tmp = buffer2

    # assert tmp.__array_interface__['data'] == buffer2.__array_interface__['data']

    # do not need allocate memory, size = naux * nao,   (buffer 1, but viewed as buffer5)

    K  = np.asarray(lib.ddot_withbuffer(tmp, aoR.T, c=buffer5, buf=mydf.ddot_buf), order='C')

    # assert K.__array_interface__['data'] == buffer5.__array_interface__['data']

    ### the order due to the fact that naux << ngrid  # need allocate memory, size = nao * nao,           (buffer 4)

    K  = np.asarray(lib.ddot_withbuffer(aoRg, K, c=buffer4, buf=mydf.ddot_buf), order='C')
    K += K.T

    # assert K.__array_interface__['data'] == buffer4.__array_interface__['data']

    #### step 3. get K term3

    ### todo: optimize the following 4 lines, it seems that they may not parallize!
    # pointwise multiplication, do not need allocate memory, size = naux * naux, use buffer for (buffer 3)
    # tmp = W * density_RgRg

    lib.cwise_mul(W, density_RgRg, out=density_RgRg)
    tmp = density_RgRg

    # assert tmp.__array_interface__['data'] == buffer3.__array_interface__['data']

    # do not need allocate memory, size = naux * nao, use buffer 2 but viewed as buffer 6
    tmp = np.asarray(lib.dot(tmp, aoRg.T, c=buffer6), order='C')

    # assert tmp.__array_interface__['data'] == buffer6.__array_interface__['data']

    # K  -= np.asarray(lib.dot(aoRg, tmp, c=K, beta=1), order='C')     # do not need allocate memory, size = nao * nao, (buffer 4)
    lib.ddot_withbuffer(aoRg, -tmp, c=K, beta=1, buf=mydf.ddot_buf)

    # assert K.__array_interface__['data'] == buffer4.__array_interface__['data']

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_k_dm")

    return K * ngrid / vol

def get_jk_dm(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
    '''JK for given k-point'''
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    #### explore the linearity of J K with respect to dm #### 

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

    if with_j:
        vj = _contract_j_dm(mydf, dm)
    if with_k:
        vk = _contract_k_dm(mydf, dm)
        if exxdiv == 'ewald':
            # raise NotImplemented("ISDF does not support ewald")
            print("WARNING: ISDF does not support ewald")


    t1 = log.timer('sr jk', *t1)

    #### explore the linearity of J K with respect to dm #### 

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


### TODO use sparse matrix to store the data

def _contract_j_mo_sparse():
    pass

def _contract_k_mo_sparse():
    pass

def get_jk_mo_sparse():
    pass
