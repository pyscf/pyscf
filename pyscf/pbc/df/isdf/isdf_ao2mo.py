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

import numpy, scipy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact

import numpy as np
import ctypes

from pyscf.pbc.df.isdf.isdf_tools_local import aoR_Holder

libpbc = lib.load_library('libpbc')

def isdf_eri_robust_fit(mydf, W, aoRg, aoR, V_r, verbose=None):
    '''
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

    path = np.einsum_path('ijx,xy,kly->ijkl', pair_Rg, V_r, pair_R, optimize='optimal')

    if verbose is not None and verbose > 0:
        # print("aoRg.shape     = ", aoRg.shape)
        # print("aoR.shape      = ", aoR.shape)
        # print("V_r.shape      = ", V_r.shape)
        print("path for term1 is ", path[0])
        print("opt            is ", path[1])

    # exit(1)

    path    = path[0]
    eri_tmp = np.einsum('ijx,xy,kly->ijkl', pair_Rg, V_r, pair_R, optimize=path)

    ### step 2, term2

    eri = eri_tmp + eri_tmp.transpose(2,3,0,1)

    ### step 3, term3

    path = np.einsum_path('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize='optimal')

    if verbose is not None and verbose > 0:
        print("path for term3 is ", path[0])
        print("opt            is ", path[1])

    path    = path[0]
    eri    -= np.einsum('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize=path)
    # eri     = np.einsum('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize=path)

    # print("ngrids = ", np.prod(cell.mesh))

    return eri * ngrid / vol


def isdf_eri(mydf, 
             mo_coeff = None,
             verbose=None):
    
    """
    locality if explored! 
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
    

def isdf_eri_ovov(mydf, 
                  mo_coeff_o: np.ndarray = None,
                  mo_coeff_v: np.ndarray = None,
                  verbose=None):
    
    """
    locality if explored! 
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
    
    # max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in moR  if aoR_holder is not None])
    max_nao_involved = max(nao_o, nao_v)
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

        #:ao_pairs_G = get_ao_pairs_G(mydf, kptijkl[:2], q, compact=compact)
        #:ao_pairs_G *= numpy.sqrt(coulG).reshape(-1,1)
        #:eri = lib.dot(ao_pairs_G.T, ao_pairs_G, cell.vol/ngrids**2)
        # ao = mydf._numint.eval_ao(cell, coords, kpti)[0]
        # ao = numpy.asarray(ao.T, order='C')
        # eri = _contract_compact(mydf, (ao,ao), coulG, max_memory=max_memory)
        
        #eri = isdf_eri_robust_fit(mydf, mydf.W, mydf.aoRg, mydf.aoR, mydf.V_R, verbose=mydf.cell.verbose)
        
        eri = isdf_eri(mydf, verbose=mydf.cell.verbose)
        
        if compact:
            # return ao2mo.restore(4, eri, nao)
            return eri
        else:
            # return eri.reshape(nao**2,nao**2)
            return ao2mo.restore(1, eri, nao)
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

    if gamma_point(kptijkl) and allreal:

        if ((iden_coeffs(mo_coeffs[0], mo_coeffs[1]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[3]))):
            
            eri = isdf_eri(mydf, mo_coeffs[0].copy(), verbose=mydf.cell.verbose)
        
            if compact:
                return eri
            else:
                return ao2mo.restore(1, eri, nao)
        else:
            
            if ((iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
                 iden_coeffs(mo_coeffs[1], mo_coeffs[3]))):
                
                eri = isdf_eri_ovov(mydf, mo_coeffs[0].copy(), mo_coeffs[1].copy(), verbose=mydf.cell.verbose)
            
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

def LS_THC(mydf, R:np.ndarray):
    '''
    given R matrix, get Z matrix such that eri = R R Z R R
    '''
    
    nGrid_R = R.shape[1]
    nao     = R.shape[0]
    
    assert nao == mydf.cell.nao
    
    ngrid   = np.prod(mydf.cell.mesh)
    nIP     = mydf.naux
    naux    = mydf.naux
    vol     = mydf.cell.vol
    natm    = mydf.cell.natm
    
    Z = np.zeros((nGrid_R, nGrid_R))
    
    #### step 1 construct ####
    
    RR = lib.ddot(R.T, R)
    lib.square_inPlace(RR)
    
    # diag RR #
    
    D_RR, U_RR = scipy.linalg.eigh(RR)
    
    print('dimension = ', D_RR.shape[0])
    
    D_RR_inv = (1.0/D_RR).copy()
    
    ## for debug ##
    
    print("max D_RR", np.max(D_RR))
    print("min D_RR", np.min(D_RR))
    print("condition number = ", np.max(D_RR)/np.min(D_RR))
    
    #### step 2 construct R R ERI R R with O(N^3) cost #### 
    
    # build (RX)^{PA} = \sum_mu R_mu^P X_\mu^A with X = aoRg # 
    
    RX = np.zeros((nGrid_R, nIP))
    
    aoRg = mydf.aoRg
    
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
    
    # V term #
    
    with_robust_fitting = mydf.with_robust_fitting
    
    if with_robust_fitting:
        V_R = mydf.V_R
        Z_tmp1 = lib.ddot(V_R, RY.T)
        lib.ddot(RX, Z_tmp1, c=Z)
        Z += Z.T
        del Z_tmp1
        
    # W term # 
    
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
    
    return Z * ngrid / vol

def LS_THC_eri(Z:np.ndarray, R:np.ndarray):
    
    einsum_str = "iP,jP,PQ,kQ,lQ->ijkl"
    
    path_info = np.einsum_path(einsum_str, R,R,Z,R,R, optimize='optimal')
    
    return np.einsum(einsum_str, R,R,Z,R,R, optimize=path_info[0])