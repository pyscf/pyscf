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

import numpy, scipy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from pyscf.pbc.df.isdf.isdf_tools_local import aoR_Holder
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

############ subroutines ---- AO2MO ############

def isdf_eri_robust_fit(mydf, W, aoRg, aoR, V_r, verbose=None):
    r'''
    
    Get (AO) electron repulsion integrals (ERI) from ISDF with robust fitting. 
    Illurstrate the idea of iSDF with robust fitting in a human-readable way.
    
    Args:
        mydf : ISDF objects 
        W    : W matrix in Sandeep2022 eq 13
        aoR  : AO values on grids (typically uniform mesh)
        aoRg : Atomic orbitals' values on interpolation ponts. 
        V_r  : V matrix in Sandeep2022 eq 13

    Return: ERI with s1 symmetry
    
    NOTE: it is an abandoned func
    
    Ref:
    
    (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720

    '''

    cell = mydf.cell
    nao  = cell.nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    eri = numpy.zeros((nao,nao,nao,nao))

    pair_Rg = np.einsum('ix,jx->ijx', aoRg, aoRg)
    pair_R  = np.einsum('ix,jx->ijx', aoR, aoR)

    ### step 1, term1

    path    = np.einsum_path('ijx,xy,kly->ijkl', pair_Rg, V_r, pair_R, optimize='optimal')[0]
    eri_tmp = np.einsum('ijx,xy,kly->ijkl', pair_Rg, V_r, pair_R, optimize=path)

    ### step 2, term2

    eri = eri_tmp + eri_tmp.transpose(2,3,0,1)

    ### step 3, term3

    path = np.einsum_path('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize='optimal')[0]
    eri -= np.einsum('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize=path)

    return eri * ngrid / vol


def isdf_eri(mydf, mo_coeff = None, verbose=None):
    
    """
    Perform AO2MO transformation from ISDF with robust fitting with s4 symmetry
    Locality is supported if explored!
    
    Args:
        mydf      :
        mo_coeff  : Molecular orbital coefficients.
    
    Returns:
        eri       : MO-ERI with s4 symmetry.
    
    TODO:
    when eri is very small, use DGEMM!
    
    """
    
    #### basic info #### 
    
    direct = mydf.direct
    if direct is True:
        raise NotImplementedError("direct is not supported in isdf_eri_robust")
    with_robust_fitting = mydf.with_robust_fitting
    
    nao   = mydf.cell.nao
    naux  = mydf.naux
    vol   = mydf.cell.vol
    ngrid = np.prod(mydf.cell.mesh)
    natm  = mydf.cell.natm
    
    if mo_coeff is not None:
        assert mo_coeff.shape[0] == nao
        nmo = mo_coeff.shape[1]
    else:
        nmo = nao
    
    size  = nmo * (nmo + 1) // 2
    eri   = numpy.zeros((size, size))
    
    aoR  = mydf.aoR
    aoRg = mydf.aoRg
    assert isinstance(aoR, list)
    assert isinstance(aoRg, list)
    
    if mo_coeff is not None:
        
        moR  = []
        moRg = []
        
        for i in range(natm):
            
            if with_robust_fitting:
                ao_involved     = aoR[i].ao_involved
                mo_coeff_packed = mo_coeff[ao_involved,:].copy()
                _moR            = lib.ddot(mo_coeff_packed.T, aoR[i].aoR)
                mo_involved     = np.arange(nmo)
                moR.append(
                    aoR_Holder(
                        aoR = _moR,
                        ao_involved = mo_involved,
                        local_gridID_begin  = aoR[i].local_gridID_begin,
                        local_gridID_end    = aoR[i].local_gridID_end,
                        global_gridID_begin = aoR[i].global_gridID_begin,
                        global_gridID_end   = aoR[i].global_gridID_end)
                )
            else:
                moR.append(None)
            
            ao_involved     = aoRg[i].ao_involved
            mo_coeff_packed = mo_coeff[ao_involved,:].copy()
            _moRg           = lib.ddot(mo_coeff_packed.T, aoRg[i].aoR)
            mo_involved     = np.arange(nmo)
            moRg.append(
                aoR_Holder(
                    aoR = _moRg,
                    ao_involved = mo_involved,
                    local_gridID_begin  = aoRg[i].local_gridID_begin,
                    local_gridID_end    = aoRg[i].local_gridID_end,
                    global_gridID_begin = aoRg[i].global_gridID_begin,
                    global_gridID_end   = aoRg[i].global_gridID_end)
            )
    else:
        moR = aoR
        moRg = aoRg
    
    max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in moR  if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in moR  if aoR_holder is not None])
    max_nIP_involved   = np.max([aoR_holder.aoR.shape[1] for aoR_holder in moRg if aoR_holder is not None])
    
    ###### loop over basic info to allocate the buf ######
    
    aoPairRg_buf  = np.zeros((max_nao_involved, max_nao_involved, max_nIP_involved))
    aoPairRg_buf2 = np.zeros((max_nao_involved, max_nao_involved, max_nIP_involved))
    if with_robust_fitting:
        aoPairR_buf = np.zeros((max_nao_involved, max_nao_involved, max_ngrid_involved))
    else:
        aoPairR_buf = None
    
    if with_robust_fitting:
        V_W_pack_buf = np.zeros((max_nIP_involved, max_ngrid_involved)) 
    else:
        V_W_pack_buf = np.zeros((max_nIP_involved, max_nIP_involved)) 
    
    max_npair    = (max_nao_involved * (max_nao_involved + 1)) // 2
    suberi_buf   = np.zeros((max_npair, max_npair))
    ddot_res_buf = np.zeros((max_nIP_involved, max_npair)) 
    
    #### involved function #### 
    
    fn_packcol = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol is not None
    
    fn_unpack_suberi_to_eri = getattr(libpbc, "_unpack_suberi_to_eri", None)
    assert fn_unpack_suberi_to_eri is not None
    
    fn_pack_aoR_to_aoPairR = getattr(libpbc, "_pack_aoR_to_aoPairR_same", None)
    assert fn_pack_aoR_to_aoPairR is not None
    
    ### V_R term ###

    V_R = mydf.V_R
    
    if with_robust_fitting:
        
        for partition_i in range(natm):
            
            aoRg_i            = moRg[partition_i]
            ao_involved_i     = aoRg_i.ao_involved
            nao_i             = aoRg_i.aoR.shape[0]
            global_IP_begin_i = aoRg_i.global_gridID_begin
            nIP_i             = aoRg_i.aoR.shape[1]
            nPair_i           = (nao_i * (nao_i + 1)) // 2
            aoPair_i          = np.ndarray((nPair_i, nIP_i), dtype=np.float64, buffer=aoPairRg_buf)

            fn_pack_aoR_to_aoPairR(
                aoRg_i.aoR.ctypes.data_as(ctypes.c_void_p),
                aoPair_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_i),
                ctypes.c_int(nIP_i)
            )
            
            for partition_j in range(natm):
                
                aoR_j             = moR[partition_j]
                ao_involved_j     = aoR_j.ao_involved
                nao_j             = aoR_j.aoR.shape[0]
                global_IP_begin_j = aoR_j.global_gridID_begin
                ngrid_j           = aoR_j.aoR.shape[1]
                nPair_j           = (nao_j * (nao_j + 1)) // 2
                aoPair_j          = np.ndarray((nPair_j, ngrid_j), dtype=np.float64, buffer=aoPairR_buf)
                
                fn_pack_aoR_to_aoPairR(
                    aoR_j.aoR.ctypes.data_as(ctypes.c_void_p),
                    aoPair_j.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_j),
                    ctypes.c_int(ngrid_j)
                )
                
                V_packed = np.ndarray((nIP_i, ngrid_j), dtype=np.float64, buffer=V_W_pack_buf)
                
                fn_packcol(
                    V_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_i),
                    ctypes.c_int(ngrid_j),
                    V_R[global_IP_begin_i:global_IP_begin_i+nIP_i, :].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_i),
                    ctypes.c_int(V_R.shape[1]),
                    ctypes.c_int(global_IP_begin_j),
                    ctypes.c_int(global_IP_begin_j+ngrid_j)
                )
                
                ddot_res = np.ndarray((nIP_i, nPair_j), dtype=np.float64, buffer=ddot_res_buf)
                lib.ddot(V_packed, aoPair_j.T, c=ddot_res)
                sub_eri  = np.ndarray((nPair_i, nPair_j), dtype=np.float64, buffer=suberi_buf)
                lib.ddot(aoPair_i, ddot_res, c=sub_eri)
                
                transpose = 1
                fn_unpack_suberi_to_eri(
                    eri.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao),
                    sub_eri.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_i),
                    ao_involved_i.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_j),
                    ao_involved_j.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(transpose)
                )
    
    ### W   term ### 
    
    W = mydf.W
    
    for partition_i in range(natm):
        
        aoRg_i            = moRg[partition_i]
        ao_involved_i     = aoRg_i.ao_involved
        nao_i             = aoRg_i.aoR.shape[0]
        global_IP_begin_i = aoRg_i.global_gridID_begin
        nIP_i             = aoRg_i.aoR.shape[1]
        nPair_i           = (nao_i * (nao_i + 1)) // 2
        aoPair_i          = np.ndarray((nPair_i, nIP_i), dtype=np.float64, buffer=aoPairRg_buf)
        
        fn_pack_aoR_to_aoPairR(
            aoRg_i.aoR.ctypes.data_as(ctypes.c_void_p),
            aoPair_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_i),
            ctypes.c_int(nIP_i)
        )
        
        for partition_j in range(partition_i+1):
            
            aoRg_j            = moRg[partition_j]
            ao_involved_j     = aoRg_j.ao_involved
            nao_j             = aoRg_j.aoR.shape[0]
            global_IP_begin_j = aoRg_j.global_gridID_begin
            nIP_j             = aoRg_j.aoR.shape[1]
            nPair_j           = (nao_j * (nao_j + 1)) // 2
            aoPair_j          = np.ndarray((nPair_j, nIP_j), dtype=np.float64, buffer=aoPairRg_buf2)
            
            fn_pack_aoR_to_aoPairR(
                aoRg_j.aoR.ctypes.data_as(ctypes.c_void_p),
                aoPair_j.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_j),
                ctypes.c_int(nIP_j)
            )
            
            ## pack_W ##
            
            W_packed = np.ndarray((nIP_i, nIP_j), dtype=np.float64, buffer=V_W_pack_buf)
            
            fn_packcol(
                W_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_i),
                ctypes.c_int(nIP_j),
                W[global_IP_begin_i:global_IP_begin_i+nIP_i, :].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_i),
                ctypes.c_int(W.shape[1]),
                ctypes.c_int(global_IP_begin_j),
                ctypes.c_int(global_IP_begin_j+nIP_j)
            )
            
            ddot_res = np.ndarray((nIP_i, nPair_j), dtype=np.float64, buffer=ddot_res_buf)
            lib.ddot(W_packed, aoPair_j.T, c=ddot_res)
            sub_eri  = np.ndarray((nPair_i, nPair_j), dtype=np.float64, buffer=suberi_buf)
            
            alpha = 1
            if with_robust_fitting:
                alpha = -1
            lib.ddot(aoPair_i, ddot_res, c=sub_eri, alpha=alpha)

            transpose = 1
            if partition_i == partition_j:
                transpose = 0

            fn_unpack_suberi_to_eri(
                eri.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                sub_eri.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_i),
                ao_involved_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_j),
                ao_involved_j.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(transpose)
            )
    
    ### del buf ###
    
    # assert np.allclose(eri, eri.T)
    
    del aoPairRg_buf
    del aoPairRg_buf2
    del aoPairR_buf
        
    return eri * ngrid / vol
    
def isdf_eri_2(mydf, mo_coeff = None, verbose=None):
    
    """
    Perform AO2MO transformation from ISDF with robust fitting with s4 symmetry
    Locality is supported if explored!
    
    Args:
        mydf      :
        mo_coeff  : Molecular orbital coefficients.
    
    Returns:
        eri       : MO-ERI with s4 symmetry.
    
    NOTE: 
    
    For small eri case 
    
    """
    
    #### basic info #### 
    
    assert mo_coeff is not None
    
    direct = mydf.direct
    if direct is True:
        raise NotImplementedError("direct is not supported in isdf_eri_robust")
    with_robust_fitting = mydf.with_robust_fitting
    
    nao   = mydf.cell.nao
    naux  = mydf.naux
    vol   = mydf.cell.vol
    ngrid = np.prod(mydf.cell.mesh)
    natm  = mydf.cell.natm
    
    if mo_coeff is not None:
        assert mo_coeff.shape[0] == nao
        nmo = mo_coeff.shape[1]
    else:
        nmo = nao
    
    size  = nmo * (nmo + 1) // 2
    eri   = numpy.zeros((size, size))
    
    aoR  = mydf.aoR
    aoRg = mydf.aoRg
    assert isinstance(aoR, list)
    assert isinstance(aoRg, list)
    
    if mo_coeff is not None:
        
        moR  = []
        moRg = []
        
        for i in range(natm):
            
            if with_robust_fitting:
                ao_involved     = aoR[i].ao_involved
                mo_coeff_packed = mo_coeff[ao_involved,:].copy()
                _moR            = lib.ddot(mo_coeff_packed.T, aoR[i].aoR)
                mo_involved     = np.arange(nmo)
                moR.append(
                    aoR_Holder(
                        aoR = _moR,
                        ao_involved = mo_involved,
                        local_gridID_begin  = aoR[i].local_gridID_begin,
                        local_gridID_end    = aoR[i].local_gridID_end,
                        global_gridID_begin = aoR[i].global_gridID_begin,
                        global_gridID_end   = aoR[i].global_gridID_end)
                )
            else:
                moR.append(None)
            
            ao_involved     = aoRg[i].ao_involved
            mo_coeff_packed = mo_coeff[ao_involved,:].copy()
            _moRg           = lib.ddot(mo_coeff_packed.T, aoRg[i].aoR)
            mo_involved     = np.arange(nmo)
            moRg.append(
                aoR_Holder(
                    aoR = _moRg,
                    ao_involved = mo_involved,
                    local_gridID_begin  = aoRg[i].local_gridID_begin,
                    local_gridID_end    = aoRg[i].local_gridID_end,
                    global_gridID_begin = aoRg[i].global_gridID_begin,
                    global_gridID_end   = aoRg[i].global_gridID_end)
            )
    else:
        moR = aoR
        moRg = aoRg
    
    max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in moR  if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in moR  if aoR_holder is not None])
    max_nIP_involved   = np.max([aoR_holder.aoR.shape[1] for aoR_holder in moRg if aoR_holder is not None])
    
    ###### loop over basic info to allocate the buf ######
    
    #max_npair    = (max_nao_involved * (max_nao_involved + 1)) // 2
    #ddot_res_buf = np.zeros((max_nIP_involved, max_npair)) 
    max_npair = nmo * (nmo + 1) // 2
    npair = max_npair
    suberi       = np.zeros((npair, npair))
    ddot_res_buf = np.zeros((naux, npair)) 

    aoPairRg_buf  = np.zeros((nmo, nmo, max_nIP_involved))
    #aoPairRg_buf2 = np.zeros((max_nao_involved, max_nao_involved, max_nIP_involved))
    aoPairRg = np.zeros((npair, naux))
    
    if with_robust_fitting:
        aoPairR_buf = np.zeros((nmo, nmo, max_ngrid_involved))
        aoPairR = np.zeros((npair, ngrid))
    else:
        aoPairR_buf = None
    
    #### involved function #### 
    
    fn_packcol = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol is not None
    
    fn_unpack_suberi_to_eri = getattr(libpbc, "_unpack_suberi_to_eri", None)
    assert fn_unpack_suberi_to_eri is not None
    
    fn_pack_aoR_to_aoPairR = getattr(libpbc, "_pack_aoR_to_aoPairR_same", None)
    assert fn_pack_aoR_to_aoPairR is not None
    
    ### construct aoPairRg, aoPairR ###
    
    for partition_i in range(natm):
            
        aoRg_i            = moRg[partition_i]
        ao_involved_i     = aoRg_i.ao_involved
        nao_i             = aoRg_i.aoR.shape[0]
        global_IP_begin_i = aoRg_i.global_gridID_begin
        nIP_i             = aoRg_i.aoR.shape[1]
        nPair_i           = (nao_i * (nao_i + 1)) // 2
        assert nPair_i   == npair
        aoPair_i          = np.ndarray((nPair_i, nIP_i), dtype=np.float64, buffer=aoPairRg_buf)

        fn_pack_aoR_to_aoPairR(
            aoRg_i.aoR.ctypes.data_as(ctypes.c_void_p),
            aoPair_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_i),
            ctypes.c_int(nIP_i)
        )
        
        aoPairRg[:, global_IP_begin_i:global_IP_begin_i+nIP_i] = aoPair_i
        
        if with_robust_fitting:
            
            aoR_i             = moR[partition_i]
            ao_involved_i     = aoR_i.ao_involved
            nao_i             = aoR_i.aoR.shape[0]
            global_IP_begin_i = aoR_i.global_gridID_begin
            ngrid_i           = aoR_i.aoR.shape[1]
            nPair_i           = (nao_i * (nao_i + 1)) // 2
            assert nPair_i   == npair
            aoPair_i          = np.ndarray((nPair_i, ngrid_i), dtype=np.float64, buffer=aoPairR_buf)
            
            fn_pack_aoR_to_aoPairR(
                aoR_i.aoR.ctypes.data_as(ctypes.c_void_p),
                aoPair_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_i),
                ctypes.c_int(ngrid_i)
            )
            
            aoPairR[:, global_IP_begin_i:global_IP_begin_i+ngrid_i] = aoPair_i
    
    ### V_R term ###

    V_R = mydf.V_R
    
    if with_robust_fitting:
        
        lib.ddot(V_R, aoPairR.T, c=ddot_res_buf)
        lib.ddot(aoPairRg, ddot_res_buf, c=suberi)
        eri += suberi
        eri += suberi.T
            
    ### W   term ### 
    
    W = mydf.W
    
    lib.ddot(W, aoPairRg.T, c=ddot_res_buf)
    lib.ddot(aoPairRg, ddot_res_buf, c=suberi)
    if with_robust_fitting:
        eri -= suberi
    else:
        eri += suberi
        
    ### del buf ###
    
    # assert np.allclose(eri, eri.T)
    
    del aoPairRg_buf
    #del aoPairRg_buf2
    del aoPairR_buf
    del aoPairRg
    del aoPairR
        
    return eri * ngrid / vol

def isdf_eri_ovov(mydf, mo_coeff_o: np.ndarray = None, mo_coeff_v: np.ndarray = None, verbose=None):
    
    """
    Perform AO2MO transformation from ISDF for specific orbital types (ovov), for MP2 calculation
    Locality is supported if explored!
    
    Args:
        mydf       : ISDF objects.
        mo_coeff_o : Molecular orbital coefficients for occupied orbitals
        mo_coeff_v : Molecular orbital coefficients for virtual  orbitals
        
    Return:
        eri : ovov part of MO-ERI
        
    """
    
    #### basic info #### 
    
    direct = mydf.direct
    if direct is True:
        raise NotImplementedError("direct is not supported in isdf_eri_robust")
    with_robust_fitting = mydf.with_robust_fitting
    
    nao   = mydf.cell.nao
    naux  = mydf.naux
    vol   = mydf.cell.vol
    ngrid = np.prod(mydf.cell.mesh)
    natm  = mydf.cell.natm
    
    nao_o = mo_coeff_o.shape[1]
    nao_v = mo_coeff_v.shape[1]
    
    size  = nao_o * nao_v
    eri   = numpy.zeros((size, size))
    
    aoR  = mydf.aoR
    aoRg = mydf.aoRg
    assert isinstance(aoR, list)
    assert isinstance(aoRg, list)
    
    ############ transformation of moRg/moR ############
        
    moR_o  = []
    moRg_o = []
        
    moR_v  = []
    moRg_v = []
        
    for i in range(natm):
            
        if with_robust_fitting:
            ao_involved     = aoR[i].ao_involved
            mo_coeff_packed = mo_coeff_o[ao_involved,:].copy()
            _moR            = lib.ddot(mo_coeff_packed.T, aoR[i].aoR)
            mo_involved     = np.arange(nao_o)
            moR_o.append(
                aoR_Holder(
                    aoR = _moR,
                    ao_involved = mo_involved,
                    local_gridID_begin  = aoR[i].local_gridID_begin,
                    local_gridID_end    = aoR[i].local_gridID_end,
                    global_gridID_begin = aoR[i].global_gridID_begin,
                    global_gridID_end   = aoR[i].global_gridID_end)
            )
                
            mo_coeff_packed = mo_coeff_v[ao_involved,:].copy()
            _moR            = lib.ddot(mo_coeff_packed.T, aoR[i].aoR)
            mo_involved     = np.arange(nao_v)
            moR_v.append(
                aoR_Holder(
                    aoR = _moR,
                    ao_involved = mo_involved,
                    local_gridID_begin  = aoR[i].local_gridID_begin,
                    local_gridID_end    = aoR[i].local_gridID_end,
                    global_gridID_begin = aoR[i].global_gridID_begin,
                    global_gridID_end   = aoR[i].global_gridID_end)
            )
                
        else:
            moR_o.append(None)
            moR_v.append(None)
            
        ao_involved     = aoRg[i].ao_involved
        mo_coeff_packed = mo_coeff_o[ao_involved,:].copy()
        _moRg           = lib.ddot(mo_coeff_packed.T, aoRg[i].aoR)
        mo_involved     = np.arange(nao_o)
        moRg_o.append(
            aoR_Holder(
                aoR = _moRg,
                ao_involved = mo_involved,
                local_gridID_begin  = aoRg[i].local_gridID_begin,
                local_gridID_end    = aoRg[i].local_gridID_end,
                global_gridID_begin = aoRg[i].global_gridID_begin,
                global_gridID_end   = aoRg[i].global_gridID_end)
        )
        
        mo_coeff_packed = mo_coeff_v[ao_involved,:].copy()
        _moRg           = lib.ddot(mo_coeff_packed.T, aoRg[i].aoR)
        mo_involved     = np.arange(nao_v)
        moRg_v.append(
            aoR_Holder(
                aoR = _moRg,
                ao_involved = mo_involved,
                local_gridID_begin  = aoRg[i].local_gridID_begin,
                local_gridID_end    = aoRg[i].local_gridID_end,
                global_gridID_begin = aoRg[i].global_gridID_begin,
                global_gridID_end   = aoRg[i].global_gridID_end)
        )
    
    ########################################################
    
    max_nao_involved   = max(nao_o, nao_v)
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in moR_o  if aoR_holder is not None])
    max_nIP_involved   = np.max([aoR_holder.aoR.shape[1] for aoR_holder in moRg_o if aoR_holder is not None])
    
    ###### loop over basic info to allocate the buf ######
    
    aoPairRg_buf  = np.zeros((nao_o, nao_v, max_nIP_involved))
    aoPairRg_buf2 = np.zeros((nao_o, nao_v, max_nIP_involved))
    if with_robust_fitting:
        aoPairR_buf = np.zeros((nao_o, nao_v, max_ngrid_involved))
    else:
        aoPairR_buf = None
        
    if with_robust_fitting:
        V_W_pack_buf = np.zeros((max_nIP_involved, max_ngrid_involved)) 
    else:
        V_W_pack_buf = np.zeros((max_nIP_involved, max_nIP_involved)) 
    
    max_npair    = nao_o * nao_v
    suberi_buf   = np.zeros((max_npair, max_npair))
    ddot_res_buf = np.zeros((max_nIP_involved, max_npair)) 
    
    #### involved function #### 
    
    fn_packcol = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol is not None
    
    fn_unpack_suberi_to_eri = getattr(libpbc, "_unpack_suberi_to_eri_ovov", None)
    assert fn_unpack_suberi_to_eri is not None
    
    fn_pack_aoR_to_aoPairR = getattr(libpbc, "_pack_aoR_to_aoPairR_diff", None)
    assert fn_pack_aoR_to_aoPairR is not None
    
    ### V_R term ###

    V_R = mydf.V_R
    
    if with_robust_fitting:
        
        for partition_i in range(natm):
            
            aoRg_i_o          = moRg_o[partition_i]
            nocc_i            = aoRg_i_o.aoR.shape[0]
            
            aoRg_i_v          = moRg_v[partition_i]
            nvir_i            = aoRg_i_v.aoR.shape[0]
            
            global_IP_begin_i = aoRg_i_o.global_gridID_begin
            nIP_i             = aoRg_i_o.aoR.shape[1]
            
            nPair_i           = nocc_i * nvir_i
            aoPair_i          = np.ndarray((nPair_i, nIP_i), dtype=np.float64, buffer=aoPairRg_buf)
            
            fn_pack_aoR_to_aoPairR(
                aoRg_i_o.aoR.ctypes.data_as(ctypes.c_void_p),
                aoRg_i_v.aoR.ctypes.data_as(ctypes.c_void_p),
                aoPair_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc_i),
                ctypes.c_int(nvir_i),
                ctypes.c_int(nIP_i)
            )
            
            for partition_j in range(natm):
                
                aoR_j_o           = moR_o[partition_j]
                nocc_j            = aoR_j_o.aoR.shape[0]
                
                aoR_j_v           = moR_v[partition_j]
                nvir_j            = aoR_j_v.aoR.shape[0]
                
                global_IP_begin_j = aoR_j_o.global_gridID_begin
                ngrid_j           = aoR_j_o.aoR.shape[1]
                
                nPair_j           = nocc_j * nvir_j
                aoPair_j          = np.ndarray((nPair_j, ngrid_j), dtype=np.float64, buffer=aoPairR_buf)
                
                fn_pack_aoR_to_aoPairR(
                    aoR_j_o.aoR.ctypes.data_as(ctypes.c_void_p),
                    aoR_j_v.aoR.ctypes.data_as(ctypes.c_void_p),
                    aoPair_j.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nocc_j),
                    ctypes.c_int(nvir_j),
                    ctypes.c_int(ngrid_j)
                )
                
                V_packed = np.ndarray((nIP_i, ngrid_j), dtype=np.float64, buffer=V_W_pack_buf)
                
                fn_packcol(
                    V_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_i),
                    ctypes.c_int(ngrid_j),
                    V_R[global_IP_begin_i:global_IP_begin_i+nIP_i, :].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_i),
                    ctypes.c_int(V_R.shape[1]),
                    ctypes.c_int(global_IP_begin_j),
                    ctypes.c_int(global_IP_begin_j+ngrid_j)
                )
                
                ddot_res = np.ndarray((nIP_i, nPair_j), dtype=np.float64, buffer=ddot_res_buf)
                lib.ddot(V_packed, aoPair_j.T, c=ddot_res)
                sub_eri  = np.ndarray((nPair_i, nPair_j), dtype=np.float64, buffer=suberi_buf)
                lib.ddot(aoPair_i, ddot_res, c=sub_eri)
                
                assert nPair_i == nPair_j == (nao_o * nao_v)
                
                transpose = 1
                fn_unpack_suberi_to_eri(
                    eri.ctypes.data_as(ctypes.c_void_p),
                    sub_eri.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nPair_i),
                    ctypes.c_int(transpose)
                )
    
    ### W   term ### 
    
    W = mydf.W
    
    for partition_i in range(natm):
        
        aoRg_i_o          = moRg_o[partition_i]
        nocc_i            = aoRg_i_o.aoR.shape[0]
        
        aoRg_i_v          = moRg_v[partition_i]
        nvir_i            = aoRg_i_v.aoR.shape[0]
        
        global_IP_begin_i = aoRg_i_o.global_gridID_begin
        nIP_i             = aoRg_i_o.aoR.shape[1]
        
        nPair_i           = nocc_i * nvir_i
        aoPair_i          = np.ndarray((nPair_i, nIP_i), dtype=np.float64, buffer=aoPairRg_buf)
        
        fn_pack_aoR_to_aoPairR(
            aoRg_i_o.aoR.ctypes.data_as(ctypes.c_void_p),
            aoRg_i_v.aoR.ctypes.data_as(ctypes.c_void_p),
            aoPair_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc_i),
            ctypes.c_int(nvir_i),
            ctypes.c_int(nIP_i)
        )
        
        for partition_j in range(partition_i+1):
            
            aoRg_j_o          = moRg_o[partition_j]
            nocc_j            = aoRg_j_o.aoR.shape[0]
            
            aoRg_j_v          = moRg_v[partition_j]
            nvir_j            = aoRg_j_v.aoR.shape[0]
            
            global_IP_begin_j = aoRg_j_o.global_gridID_begin
            nIP_j             = aoRg_j_o.aoR.shape[1]
            
            nPair_j           = nocc_j * nvir_j
            aoPair_j          = np.ndarray((nPair_j, nIP_j), dtype=np.float64, buffer=aoPairRg_buf2)
            
            fn_pack_aoR_to_aoPairR(
                aoRg_j_o.aoR.ctypes.data_as(ctypes.c_void_p),
                aoRg_j_v.aoR.ctypes.data_as(ctypes.c_void_p),
                aoPair_j.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc_j),
                ctypes.c_int(nvir_j),
                ctypes.c_int(nIP_j)
            )
            
            ## pack_W ##
            
            W_packed = np.ndarray((nIP_i, nIP_j), dtype=np.float64, buffer=V_W_pack_buf)
            
            fn_packcol(
                W_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_i),
                ctypes.c_int(nIP_j),
                W[global_IP_begin_i:global_IP_begin_i+nIP_i, :].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_i),
                ctypes.c_int(W.shape[1]),
                ctypes.c_int(global_IP_begin_j),
                ctypes.c_int(global_IP_begin_j+nIP_j)
            )
            
            ddot_res = np.ndarray((nIP_i, nPair_j), dtype=np.float64, buffer=ddot_res_buf)
            lib.ddot(W_packed, aoPair_j.T, c=ddot_res)
            sub_eri  = np.ndarray((nPair_i, nPair_j), dtype=np.float64, buffer=suberi_buf)
            
            assert nPair_i == nPair_j == (nao_o * nao_v)
            
            alpha = 1
            if with_robust_fitting:
                alpha = -1
            lib.ddot(aoPair_i, ddot_res, c=sub_eri, alpha=alpha)
    
            transpose = 1
            if partition_i == partition_j:
                transpose = 0
            
            fn_unpack_suberi_to_eri(
                eri.ctypes.data_as(ctypes.c_void_p),
                sub_eri.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nPair_i),
                ctypes.c_int(transpose)
            )
    
    ### del buf ###
    
    assert np.allclose(eri, eri.T)
    
    del aoPairRg_buf
    del aoPairRg_buf2
    del aoPairR_buf
        
    return eri.reshape(nao_o, nao_v, nao_o, nao_v) * ngrid / vol
  
def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):

    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'isdf_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros((nao,nao,nao,nao))

    # kpti, kptj, kptk, kptl = kptijkl
    # q = kptj - kpti
    # coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    # coords = cell.gen_uniform_grids(mydf.mesh)
    # max_memory = mydf.max_memory - lib.current_memory()[0]

####################

# gamma point, the integral is real and with s4 symmetry

    if gamma_point(kptijkl):

        eri = isdf_eri(mydf, verbose=mydf.cell.verbose)
        
        if compact:
            return eri
        else:
            return ao2mo.restore(1, eri, nao)

####################
# aosym = s1, complex integrals

    else:
        raise NotImplementedError


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    '''General MO integral transformation'''

    from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
    warn_pbc2d_eri(mydf)
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    mo_coeffs = [numpy.asarray(mo, order='F') for mo in mo_coeffs]
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'fft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros([mo.shape[1] for mo in mo_coeffs])

    allreal = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    q = kptj - kpti
    # coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    # coords = cell.gen_uniform_grids(mydf.mesh)
    max_memory = mydf.max_memory - lib.current_memory()[0]

    if hasattr(mydf, "W2") or (hasattr(mydf, "force_LS_THC") and mydf.force_LS_THC == True):          # NOTE: this means that LS_THC_recompression is called, we do not perform ao2mo with robust fitting, as it is very expensive!
        #print("use_LS_THC_anyway")
        use_LS_THC_anyway = True
    else:
        #print("no_use_LS_THC_anyway")
        use_LS_THC_anyway = False

    IsMOERI = (iden_coeffs(mo_coeffs[0], mo_coeffs[1]) and
               iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[0], mo_coeffs[3]))
    if not IsMOERI:
        IsOVOV = False
        IsGeneral = False
    else:
        IsOVOV = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
                 iden_coeffs(mo_coeffs[1], mo_coeffs[3]))
        if IsOVOV:
            IsGeneral = False
        else:
            IsGeneral = True

    if gamma_point(kptijkl) and allreal:

        ##### check whether LS-THC anyway #####
        
        if use_LS_THC_anyway:
            
            vol   = mydf.cell.vol
            ngrid = np.prod(mydf.cell.mesh)
            
            if hasattr(mydf, "W2"):
                eri = LS_THC_moeri(mydf, mydf.W2, mydf.aoRg2, mo_coeffs) * ngrid / vol
            else:
                eri = LS_THC_moeri(mydf, mydf.W, mydf.aoRg, mo_coeffs) * ngrid / vol
            if compact:
                if IsMOERI:
                    return ao2mo.restore(4, eri, nao)
                else:
                    return eri
            else:
                return eri

        if ((iden_coeffs(mo_coeffs[0], mo_coeffs[1]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[3]))):
            
            #### Full MO-ERI ####
            
            t1  = (lib.logger.process_clock(), lib.logger.perf_counter())
            eri = isdf_eri(mydf, mo_coeffs[0].copy(), verbose=mydf.cell.verbose)
            # eri = isdf_eri_2(mydf, mo_coeffs[0].copy(), verbose=mydf.cell.verbose) # requires aoPairR, which is very expensive
            t2  = (lib.logger.process_clock(), lib.logger.perf_counter())
            _benchmark_time(t1, t2, 'isdf_eri', mydf)
        
            if compact:
                return eri
            else:
                return ao2mo.restore(1, eri, nao)
        else:
            
            #### ovov MO-ERI ####
            
            if ((iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
                 iden_coeffs(mo_coeffs[1], mo_coeffs[3]))):
                
                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                eri = isdf_eri_ovov(mydf, mo_coeffs[0].copy(), mo_coeffs[1].copy(), verbose=mydf.cell.verbose)
                t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                _benchmark_time(t1, t2, 'isdf_eri_ovov', mydf)
            
                if compact:
                    print("compact is not supported in general with ov ov mode")
                    return eri
                else:
                    return eri
            
            else:
                    raise NotImplementedError

    else:
        raise NotImplementedError
    
    return

def ao2mo_7d(mydf, mo_coeff_kpts, kpts=None, factor=1, out=None):
    raise NotImplementedError

############ subroutines ---- LS-THC ############

def LS_THC(mydf, R:np.ndarray):
    '''
    Least-Square Tensorhypercontraction decomposition of ERI.
    Given an R matrix, compute the Z matrix such that the electron repulsion integral (ERI) can be expressed as eri ~ R R Z R R.
    Supports both ISDF w./w.o. robust fitting.
    
    Args:
        mydf : ISDF objects. 
        R    : A matrix used in the computation of the ERI.

    Returns:
        Z    :  eri = R R Z R R.

    Ref:
        (1) Martinez2012: Parrish, Hohenstein, Martinez and Sherill. J. Chem. Phys. 137, 224106 (2012), DOI: https://doi.org/10.1063/1.4768233

    '''
    
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    nGrid_R = R.shape[1]
    nao     = R.shape[0]
    
    assert nao == mydf.cell.nao
    
    ngrid = np.prod(mydf.cell.mesh)
    nIP   = mydf.naux
    naux  = mydf.naux
    vol   = mydf.cell.vol
    natm  = mydf.cell.natm
    
    Z = np.zeros((nGrid_R, nGrid_R))
    
    #### step 1 construct ####
    
    RR = lib.ddot(R.T, R)
    lib.square_inPlace(RR)
    
    # diag RR #
    
    D_RR, U_RR = scipy.linalg.eigh(RR)    
    D_RR_inv   = (1.0/D_RR).copy()
    
    ## for debug ##
    
    log.debug4("***** LS_THC ***** ")
    log.debug4("max D_RR         = %f", np.max(D_RR))
    log.debug4("min D_RR         = %f", np.min(D_RR))
    log.debug4("condition number = %f", np.max(D_RR)/np.min(D_RR))
    
    #### step 2 construct R R ERI R R with O(N^3) cost #### 
    
    # build (RX)^{PA} = \sum_mu R_mu^P X_\mu^A with X = aoRg # 
    
    RX = np.zeros((nGrid_R, nIP))
    
    aoRg = mydf.aoRg
    
    if isinstance(aoRg, np.ndarray):
        
        RX = lib.ddot(R.T, aoRg)

    else:
        
        assert isinstance(aoRg, list)
        
        for partition_i in range(natm):
        
            aoRg_i            = aoRg[partition_i]
            ao_involved_i     = aoRg_i.ao_involved
            nao_i             = aoRg_i.aoR.shape[0]
            global_IP_begin_i = aoRg_i.global_gridID_begin
            nIP_i             = aoRg_i.aoR.shape[1]
        
            R_packed = R[ao_involved_i,:].copy() 
            RX_tmp   = lib.ddot(R_packed.T, aoRg_i.aoR)
        
            RX[:,global_IP_begin_i:global_IP_begin_i+nIP_i] = RX_tmp 
    
    RX = lib.square_inPlace(RX)
        
    # build (RY)^{PB} = \sum_mu R_mu^P Y_\mu^B with Y = aoR # 
    
    if mydf.with_robust_fitting:
        
        if isinstance(mydf.aoR, np.ndarray):
            
            RY = lib.ddot(R.T, mydf.aoR)
            
        else:
            
            assert isinstance(mydf.aoR, list)
            
            aoR = mydf.aoR
            RY = np.zeros((nGrid_R, ngrid))
            for partition_i in range(natm):
            
                aoR_i            = aoR[partition_i]
                ao_involved_i    = aoR_i.ao_involved
                nao_i            = aoR_i.aoR.shape[0]
                global_gridID_i  = aoR_i.global_gridID_begin
                ngrid_i          = aoR_i.aoR.shape[1]
            
                R_packed = R[ao_involved_i,:].copy()
                RY_tmp   = lib.ddot(R_packed.T, aoR_i.aoR)
            
                RY[:,global_gridID_i:global_gridID_i+ngrid_i] = RY_tmp
    
        RY = lib.square_inPlace(RY)
    else:
        RY = None
    
    #### V term ####
    
    with_robust_fitting = mydf.with_robust_fitting
    
    if with_robust_fitting:
        V_R = mydf.V_R
        Z_tmp1 = lib.ddot(V_R, RY.T)
        lib.ddot(RX, Z_tmp1, c=Z)
        Z += Z.T
        del Z_tmp1
        
    #### W term #### 
    
    W = mydf.W
    Z_tmp2 = lib.ddot(W, RX.T)
    if with_robust_fitting:
        lib.ddot(RX, Z_tmp2, c=Z, alpha=-1, beta=1)
    else:
        lib.ddot(RX, Z_tmp2, c=Z)
    del Z_tmp2
    
    Z1 = lib.ddot(U_RR.T, Z)
    Z2 = lib.ddot(Z1, U_RR, c=Z)
    Z  = Z2 
    
    lib.d_i_ij_ij(D_RR_inv, Z, out=Z)
    lib.d_ij_j_ij(Z, D_RR_inv, out=Z)
    lib.ddot(U_RR, Z, c=Z1)
    lib.ddot(Z1, U_RR.T, c=Z)
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    log.timer('LS_THC fitting', *t1)
    
    return Z * ngrid / vol

def LS_THC_eri(Z:np.ndarray, R:np.ndarray):
    
    einsum_str = "iP,jP,PQ,kQ,lQ->ijkl"
    
    path_info = np.einsum_path(einsum_str, R,R,Z,R,R, optimize='optimal')
    
    return np.einsum(einsum_str,R,R,Z,R,R,optimize=path_info[0])

def LS_THC_moeri(mydf, Z:np.ndarray, R:np.ndarray, mo_coeff:np.ndarray):
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    assert len(mo_coeff) == 4
    moRg       = [lib.ddot(x.T, R) for x in mo_coeff]
    einsum_str = "iP,jP,PQ,kQ,lQ->ijkl"
    path_info  = np.einsum_path(einsum_str, moRg[0], moRg[1], Z, moRg[2], moRg[3], optimize='optimal')
    res = np.einsum(einsum_str, moRg[0], moRg[1], Z, moRg[2], moRg[3], optimize=path_info[0])
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    log.timer('LS_THC MOERI', *t1)
    return res
    
############ Laplace transformation ############ 

import bisect

def _find_laplace(laplace_holder:dict, R, error):
    '''
    find the laplace holder with the smallest error that is larger than the given error
    
    Args:
        laplace_holder: dict
            a dictionary that contains all the laplace holder
        R: float
            1/x is fitted as summation of exponential functions on interval [1,R]
        error: float
            the relative error threshold
    
    Return:

    '''
    
    ### find key via binary search ###
        
    keys = list(laplace_holder.keys())
    keys.sort()
        
    index = bisect.bisect_left(keys, R)
    if index == len(keys):
        return None
    else:
        key = keys[index]
        items = laplace_holder[key]
        items.sort(key=lambda x: x['error'], reverse=True)
        for item in items:
            if item['error'] <= error:
                return item
        return None

def _build_laplace_holder(r_min, r_max, rel_error):
    '''
    '''
    
    import os, pickle
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path  = os.path.join(script_dir, 'laplace.pkl')
    with open(file_path, 'rb') as f:
        laplace_holder = pickle.load(f) 
    
    item_found = _find_laplace(laplace_holder, r_max/r_min, rel_error)
    
    if item_found is None:
        raise NotImplementedError("No laplace holder found")
    
    return {
        "a_values":np.array(item_found['a_values'])/r_min,
        "b_values":np.array(item_found['b_values'])/r_min,
        "degree"  :item_found['degree'],
        "error"   :item_found['error']
    }

class laplace_holder:
    r''' laplace transformation of energy denominator 
    For order 2, 1/(ea+eb-ei-ej) ~ \sum_T (tao_v)_a^T (tao_v)_b^T (tao_o)_i^T (tao_o)_j^T 
    
    Ref:
        (1) Almlof1992   : J. Chem. Phys. 96, 489-494 (1992) https://doi.org/10.1063/1.462485
        (2) Hackbusch2008: J. Chem. Phys. 129, 044112 (2008) https://doi.org/10.1063/1.2958921
        (3) https://gitlab.mis.mpg.de/scicomp/EXP_SUM
    
    '''
    
    _keys = {
        'mo_ene', 'nocc', 'order', 'holder', 'a_values', 'b_values',
        '_degree', '_error', 'laplace_occ', 'laplace_vir'
    }
    
    def __init__(self, 
                 mo_ene, nocc, 
                 order=2, rel_error=1e-6, verbose=True):
        
        occ_ene = mo_ene[:nocc]
        vir_ene = mo_ene[nocc:]
        
        max_occ = np.max(occ_ene)
        min_occ = np.min(occ_ene)
        min_vir = np.min(vir_ene)
        max_vir = np.max(vir_ene)
        
        if max_occ > min_vir+1e-8:
            print("Warning: max_occ > min_vir")
            raise NotImplementedError

        r_min = (min_vir - max_occ) * order
        r_max = (max_vir - min_occ) * order
        
        self.mo_ene = mo_ene
        self.nocc   = nocc
        self.order  = order
        self.holder = _build_laplace_holder(r_min, r_max, rel_error)
        self.a_values  = self.holder['a_values']
        self.b_values  = self.holder['b_values']
        self._degree   = self.holder['degree']
        self._error    = self.holder['error']
        
        self.laplace_occ = self._build_laplace_occ(occ_ene, order=order)
        self.laplace_vir = self._build_laplace_vir(vir_ene, order=order)
    
    @property
    def degree(self):
        return self._degree
    
    @property
    def error(self):
        return self._error

    @property
    def delta_full(self):
        if self.order != 2:
            raise NotImplementedError
        else:
            return np.einsum("iP,jP,aP,bP->ijab", self.laplace_occ, self.laplace_occ, self.laplace_vir, self.laplace_vir)

    def _build_laplace_occ(self, occ_ene, order=2):
        
        nocc   = len(occ_ene)
        degree = self.degree
        res    = np.zeros((nocc, degree))
        order2 = 2 * order
        
        for icol, (a, b) in enumerate(zip(self.a_values, self.b_values)):
            res[:, icol] = (a**((1.0/(float(order2)))))*np.exp(b*occ_ene)

        return res
    
    def _build_laplace_vir(self, vir_ene, order=2):
        
        nvir   = len(vir_ene)
        degree = self.degree
        res    = np.zeros((nvir, degree))
        order2 = 2 * order
        
        print("vir_ene = ", vir_ene)
        
        for icol, (a, b) in enumerate(zip(self.a_values, self.b_values)):
            res[:, icol] = (a**((1.0/(float(order2)))))*np.exp(-b*vir_ene)
                
        return res