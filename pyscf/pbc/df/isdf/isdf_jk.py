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

    #### step 1. get density value on real space grid and IPs

    #### TODO: make the following transformation linear-scaling

    ao_pair_R  = np.einsum('ij,kj->ikj', aoR, aoR)
    ao_pair_Rg = np.einsum('ij,kj->ikj', aoRg, aoRg)

    #### step 2. get J term1 and term2

    density_R  = np.einsum('ij,ijk->k', dm, ao_pair_R)
    density_Rg = np.einsum('ij,ijk->k', dm, ao_pair_Rg)

    ## TODO: remove the redundancy due to the symmetry

    rho_mu_nu_Rg = np.einsum('ij,kj->ikj', aoRg, aoRg)


    # J = np.asarray(lib.dot(V_R, density_R), order='C')
    J = np.dot(V_R, density_R)
    # J = np.asarray(lib.dot(rho_mu_nu_Rg, J), order='C')
    J = np.dot(rho_mu_nu_Rg, J)
    # J += J.T

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
    _benchmark_time(t1, t2, "_contract_j_dm")

    return J * ngrid / vol

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

    nao  = dm.shape[0]

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

    density_RgR = np.asarray(lib.dot(dm, aoR), order='C')
    density_RgR = np.asarray(lib.dot(aoRg.T, density_RgR), order='C')
    density_RgRg = np.asarray(lib.dot(dm, aoRg), order='C')
    density_RgRg = np.asarray(lib.dot(aoRg.T, density_RgRg), order='C')

    #### step 2. get K term1 and term2

    tmp = V_R * density_RgR  # pointwise multiplication

    K = np.asarray(lib.dot(tmp, aoR.T), order='C')
    K = np.asarray(lib.dot(aoRg, K), order='C')  ### the order due to the fact that naux << ngrid
    K += K.T

    #### step 3. get K term3

    tmp = W * density_RgRg  # pointwise multiplication
    tmp = np.asarray(lib.dot(tmp, aoRg.T), order='C')
    K -= np.asarray(lib.dot(aoRg, tmp), order='C')
    # K = np.asarray(lib.dot(aoRg, tmp), order='C')

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "_contract_k_dm")

    return K * ngrid / vol

def get_jk_dm(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
    '''JK for given k-point'''

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
    return vj, vk


### TODO use sparse matrix to store the data

def _contract_j_mo_sparse():
    pass

def _contract_k_mo_sparse():
    pass

def get_jk_mo_sparse():
    pass
