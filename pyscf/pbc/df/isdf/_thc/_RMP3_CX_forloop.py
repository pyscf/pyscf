import numpy
import numpy as np
import ctypes
import copy
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, allgather_pickle
def RMP3_CX_1_forloop_Q_R_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        Q_bunchsize = 8,
                                                        R_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    # assign the size of each tensor
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M13_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _M1_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NOCC)))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _M4_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NVIR)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _INPUT_0_sliced_size = (NTHC_INT * NTHC_INT)
    _M7_perm_size    = (Q_bunchsize * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _INPUT_3_sliced_size = (NOCC * NTHC_INT)
    _INPUT_16_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_5_sliced_size = (NTHC_INT * NTHC_INT)
    _M14_sliced_size = (NTHC_INT * (Q_bunchsize * V_bunchsize))
    _M15_perm_size   = (R_bunchsize * (NTHC_INT * (V_bunchsize * Q_bunchsize)))
    _INPUT_22_sliced_size = (NVIR * N_LAPLACE)
    _M20_size        = (NTHC_INT * (W_bunchsize * (R_bunchsize * (V_bunchsize * Q_bunchsize))))
    _M18_sliced_size = (NTHC_INT * (R_bunchsize * W_bunchsize))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M2_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M0_size         = (V_bunchsize * (W_bunchsize * NOCC))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M5_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M7_size         = (Q_bunchsize * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M8_size         = (NTHC_INT * (Q_bunchsize * (V_bunchsize * W_bunchsize)))
    _M3_size         = (V_bunchsize * (W_bunchsize * NVIR))
    _INPUT_6_sliced_size = (NOCC * NTHC_INT)
    _M10_size        = (V_bunchsize * (Q_bunchsize * R_bunchsize))
    _M15_size        = (R_bunchsize * (NTHC_INT * (V_bunchsize * Q_bunchsize)))
    _M16_size        = (NVIR * (R_bunchsize * (V_bunchsize * Q_bunchsize)))
    _M20_perm_size   = (NTHC_INT * (W_bunchsize * (R_bunchsize * (V_bunchsize * Q_bunchsize))))
    _M9_size         = (Q_bunchsize * (R_bunchsize * NOCC))
    _M12_size        = (NTHC_INT * (V_bunchsize * (Q_bunchsize * R_bunchsize)))
    _M17_size        = (W_bunchsize * (R_bunchsize * (V_bunchsize * (Q_bunchsize * NVIR))))
    _M19_size        = (Q_bunchsize * (V_bunchsize * (R_bunchsize * (NTHC_INT * W_bunchsize))))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M11_size)
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_15_sliced_size)
    bucked_0_size    = max(bucked_0_size, _M1_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_17_sliced_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_0_sliced_size)
    bucked_0_size    = max(bucked_0_size, _M7_perm_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_3_sliced_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_16_sliced_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_5_sliced_size)
    bucked_0_size    = max(bucked_0_size, _M14_sliced_size)
    bucked_0_size    = max(bucked_0_size, _M15_perm_size)
    bucked_0_size    = max(bucked_0_size, _INPUT_22_sliced_size)
    bucked_0_size    = max(bucked_0_size, _M20_size)
    bucked_0_size    = max(bucked_0_size, _M18_sliced_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M14_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M18_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _INPUT_19_sliced_size)
    bucked_3_size    = max(bucked_3_size, _M2_size)
    bucked_3_size    = max(bucked_3_size, _M6_perm_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _M0_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M5_size)
    bucked_4_size    = max(bucked_4_size, _M7_size)
    bucked_4_size    = max(bucked_4_size, _M8_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M3_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_6_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M10_size)
    bucked_5_size    = max(bucked_5_size, _M15_size)
    bucked_5_size    = max(bucked_5_size, _M16_size)
    bucked_5_size    = max(bucked_5_size, _M20_perm_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M9_size)
    bucked_6_size    = max(bucked_6_size, _M12_size)
    bucked_6_size    = max(bucked_6_size, _M17_size)
    bucked_6_size    = max(bucked_6_size, _M19_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    return output

def RMP3_CX_1_forloop_Q_R_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("bS,QVb->SQV"   , _INPUT_8        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("kR,kW->RWk"    , _INPUT_7        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("kU,RWk->URW"   , _INPUT_13       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("TU,QVWT->UQVW" , _INPUT_10       , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jQ,jR->QRj"    , _INPUT_3        , _INPUT_6        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("jV,QRj->VQR"   , _INPUT_16       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("RS,VQR->SVQR"  , _INPUT_5        , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("SVQR,SQV->RSVQ", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 2, 3, 1)    )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("cS,RVQS->cRVQ" , _INPUT_9        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("cW,cRVQ->WRVQc", _INPUT_22       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("cU,WRVQc->UWRVQ", _INPUT_14       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (4, 3, 2, 0, 1) )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("UQVW,URW->QVRUW", _M8             , _M18            )
    del _M8         
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("QVRUW,QVRUW->" , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_CX_1_forloop_Q_R(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bS,QVb->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M11_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 kR,kW->RWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 kU,RWk->URW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M13_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M0.ctypes.data),
                                   ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M0.shape[0]),
                                   ctypes.c_int(_M0.shape[1]))
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _size_dim_1      = _size_dim_1 * _M1.shape[2]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M2.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
    _M2          = _M2_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M3              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M3.shape[0]),
                                   ctypes.c_int(_M3.shape[1]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _size_dim_1      = _size_dim_1 * _M4.shape[2]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TPVW,TPVW->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                      ctypes.c_void_p(_M5.ctypes.data),
                                      ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_int(_M2.shape[0]),
                                      ctypes.c_int(_M2.shape[1]),
                                      ctypes.c_int(_M2.shape[2]),
                                      ctypes.c_int(_M2.shape[3]))
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TPVW->TVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M6_perm.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 QTVW->QVWT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 TU,QVWT->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M7_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 jQ,jR->QRj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_6.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 jV,QRj->VQR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_16.shape[0]
    _INPUT_16_reshaped = _INPUT_16.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_16_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 RS,VQR->SVQR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_int(_INPUT_5.shape[0]),
                                   ctypes.c_int(_INPUT_5.shape[1]),
                                   ctypes.c_int(_M10.shape[0]),
                                   ctypes.c_int(_M10.shape[1]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 SVQR,SQV->RSVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M12.shape[0]),
                                     ctypes.c_int(_M12.shape[1]),
                                     ctypes.c_int(_M12.shape[2]),
                                     ctypes.c_int(_M12.shape[3]))
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 RSVQ->RVQS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                 ctypes.c_int(_M15.shape[0]),
                                 ctypes.c_int(_M15.shape[1]),
                                 ctypes.c_int(_M15.shape[2]),
                                 ctypes.c_int(_M15.shape[3]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 cS,RVQS->cRVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 cW,cRVQ->WRVQc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0234_12340_wob = getattr(libpbc, "fn_contraction_01_0234_12340_wob", None)
    assert fn_contraction_01_0234_12340_wob is not None
    _M17             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_0234_12340_wob(ctypes.c_void_p(_INPUT_22.ctypes.data),
                                     ctypes.c_void_p(_M16.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_INPUT_22.shape[0]),
                                     ctypes.c_int(_INPUT_22.shape[1]),
                                     ctypes.c_int(_M16.shape[1]),
                                     ctypes.c_int(_M16.shape[2]),
                                     ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 cU,WRVQc->UWRVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _size_dim_1      = _size_dim_1 * _M17.shape[3]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M17_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M17        
    del _M17_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 UWRVQ->QVRUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_43201_wob = getattr(libpbc, "fn_permutation_01234_43201_wob", None)
    assert fn_permutation_01234_43201_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_43201_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 UQVW,URW->QVRUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_043_12403_wob = getattr(libpbc, "fn_contraction_0123_043_12403_wob", None)
    assert fn_contraction_0123_043_12403_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_043_12403_wob(ctypes.c_void_p(_M8.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_void_p(_M19.ctypes.data),
                                      ctypes.c_int(_M8.shape[0]),
                                      ctypes.c_int(_M8.shape[1]),
                                      ctypes.c_int(_M8.shape[2]),
                                      ctypes.c_int(_M8.shape[3]),
                                      ctypes.c_int(_M18.shape[1]))
    del _M8         
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 QVRUW,QVRUW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_CX_1_forloop_Q_R_forloop_Q_R(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      Q_bunchsize = 8,
                                      R_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    fn_contraction_01_0234_12340_wob = getattr(libpbc, "fn_contraction_01_0234_12340_wob", None)
    assert fn_contraction_01_0234_12340_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    fn_permutation_01234_43201_wob = getattr(libpbc, "fn_permutation_01234_43201_wob", None)
    assert fn_permutation_01234_43201_wob is not None
    fn_contraction_0123_043_12403_wob = getattr(libpbc, "fn_contraction_0123_043_12403_wob", None)
    assert fn_contraction_0123_043_12403_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        Q_begin = rank*bunchsize
        Q_end = (rank+1)*bunchsize
        Q_begin          = min(Q_begin, NTHC_INT)
        Q_end            = min(Q_end, NTHC_INT)
    else:
        Q_begin          = 0               
        Q_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_1_forloop_Q_R_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           Q_bunchsize = Q_bunchsize,
                                                                           R_bunchsize = R_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 bQ,bV->QVb
    offset_now       = offset_0        
    _M11_offset      = offset_now      
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   3 bS,QVb->SQV
    offset_now       = offset_1        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M11_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step   4 kR,kW->RWk
    offset_now       = offset_0        
    _M13_offset      = offset_now      
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   5 kU,RWk->URW
    offset_now       = offset_2        
    _M18_offset      = offset_now      
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M13_reshaped.T, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
    # step   6 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step   7 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step   8 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_0        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step   9 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_3        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  10 iV,iW->VWi
            offset_now       = offset_4        
            _M0_offset       = offset_now      
            _M0              = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M0_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M0.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  11 iP,VWi->PVWi
            offset_now       = offset_0        
            _M1_offset       = offset_now      
            _M1              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M1_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                           ctypes.c_void_p(_M0.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_int(_INPUT_1.shape[0]),
                                           ctypes.c_int(_INPUT_1.shape[1]),
                                           ctypes.c_int(_M0.shape[0]),
                                           ctypes.c_int(_M0.shape[1]))
            # step  12 iT,PVWi->TPVW
            offset_now       = offset_3        
            _M2_offset       = offset_now      
            _M2              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M2_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
            _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _size_dim_1      = _size_dim_1 * _M1.shape[2]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M2.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M2.shape[0]
            _M2_reshaped = _M2.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_11_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
            _M2              = _M2_reshaped.reshape(*shape_backup)
            # step  13 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_0        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  14 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  15 aV,aW->VWa
            offset_now       = offset_5        
            _M3_offset       = offset_now      
            _M3              = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M3_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M3.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  16 aP,VWa->PVWa
            offset_now       = offset_0        
            _M4_offset       = offset_now      
            _M4              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M4_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_int(_INPUT_2.shape[0]),
                                           ctypes.c_int(_INPUT_2.shape[1]),
                                           ctypes.c_int(_M3.shape[0]),
                                           ctypes.c_int(_M3.shape[1]))
            # step  17 aT,PVWa->TPVW
            offset_now       = offset_4        
            _M5_offset       = offset_now      
            _M5              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M5_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
            _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M4.shape[0]
            _size_dim_1      = _size_dim_1 * _M4.shape[1]
            _size_dim_1      = _size_dim_1 * _M4.shape[2]
            _M4_reshaped = _M4.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M5.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M5.shape[0]
            _M5_reshaped = _M5.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_12_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
            _M5              = _M5_reshaped.reshape(*shape_backup)
            # step  18 TPVW,TPVW->TPVW
            offset_now       = offset_0        
            _M6_offset       = offset_now      
            _M6              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M6_offset)
            fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                              ctypes.c_void_p(_M5.ctypes.data),
                                              ctypes.c_void_p(_M6.ctypes.data),
                                              ctypes.c_int(_M2.shape[0]),
                                              ctypes.c_int(_M2.shape[1]),
                                              ctypes.c_int(_M2.shape[2]),
                                              ctypes.c_int(_M2.shape[3]))
            # step  19 TPVW->TVWP
            _M6_perm_offset  = offset_3        
            _M6_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  20 start for loop with indices ('V', 'W', 'Q')
            for Q_0, Q_1 in lib.prange(Q_begin,Q_end,Q_bunchsize):
                # step  21 slice _INPUT_0 with indices ['Q']
                _INPUT_0_sliced_offset = offset_0        
                _INPUT_0_sliced  = np.ndarray((NTHC_INT, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_0_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(Q_0),
                             ctypes.c_int(Q_1))
                # step  22 PQ,TVWP->QTVW
                offset_now       = offset_4        
                _M7_offset       = offset_now      
                _M7              = np.ndarray(((Q_1-Q_0), NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M7_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_0_sliced.shape[0]
                _INPUT_0_sliced_reshaped = _INPUT_0_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M7.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7.shape[0]
                _M7_reshaped = _M7.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_0_sliced_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
                _M7              = _M7_reshaped.reshape(*shape_backup)
                # step  23 QTVW->QVWT
                _M7_perm_offset  = offset_0        
                _M7_perm         = np.ndarray(((Q_1-Q_0), (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
                fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                             ctypes.c_void_p(_M7_perm.ctypes.data),
                                             ctypes.c_int((Q_1-Q_0)),
                                             ctypes.c_int(NTHC_INT),
                                             ctypes.c_int((V_1-V_0)),
                                             ctypes.c_int((W_1-W_0)))
                # step  24 TU,QVWT->UQVW
                offset_now       = offset_4        
                _M8_offset       = offset_now      
                _M8              = np.ndarray((NTHC_INT, (Q_1-Q_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M8_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
                _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
                _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M8.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8.shape[0]
                _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_10_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
                _M8              = _M8_reshaped.reshape(*shape_backup)
                # step  25 start for loop with indices ('V', 'W', 'Q', 'R')
                for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
                    # step  26 slice _INPUT_3 with indices ['Q']
                    _INPUT_3_sliced_offset = offset_0        
                    _INPUT_3_sliced  = np.ndarray((NOCC, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_3_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
                    # step  27 slice _INPUT_6 with indices ['R']
                    _INPUT_6_sliced_offset = offset_5        
                    _INPUT_6_sliced  = np.ndarray((NOCC, (R_1-R_0)), buffer = buffer, offset = _INPUT_6_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(R_0),
                                 ctypes.c_int(R_1))
                    # step  28 jQ,jR->QRj
                    offset_now       = offset_6        
                    _M9_offset       = offset_now      
                    _M9              = np.ndarray(((Q_1-Q_0), (R_1-R_0), NOCC), buffer = buffer, offset = _M9_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M9.ctypes.data),
                                                 ctypes.c_int(_INPUT_3_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_3_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_6_sliced.shape[1]))
                    # step  29 slice _INPUT_16 with indices ['V']
                    _INPUT_16_sliced_offset = offset_0        
                    _INPUT_16_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_16_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_16.shape[0]),
                                 ctypes.c_int(_INPUT_16.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  30 jV,QRj->VQR
                    offset_now       = offset_5        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray(((V_1-V_0), (Q_1-Q_0), (R_1-R_0)), buffer = buffer, offset = _M10_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_16_sliced.shape[0]
                    _INPUT_16_sliced_reshaped = _INPUT_16_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M9.shape[0]
                    _size_dim_1      = _size_dim_1 * _M9.shape[1]
                    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M10.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_16_sliced_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
                    _M10             = _M10_reshaped.reshape(*shape_backup)
                    # step  31 slice _INPUT_5 with indices ['R']
                    _INPUT_5_sliced_offset = offset_0        
                    _INPUT_5_sliced  = np.ndarray(((R_1-R_0), NTHC_INT), buffer = buffer, offset = _INPUT_5_sliced_offset)
                    fn_slice_2_0(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                 ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_5.shape[0]),
                                 ctypes.c_int(_INPUT_5.shape[1]),
                                 ctypes.c_int(R_0),
                                 ctypes.c_int(R_1))
                    # step  32 RS,VQR->SVQR
                    offset_now       = offset_6        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (V_1-V_0), (Q_1-Q_0), (R_1-R_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M10.ctypes.data),
                                                   ctypes.c_void_p(_M12.ctypes.data),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                                   ctypes.c_int(_M10.shape[0]),
                                                   ctypes.c_int(_M10.shape[1]))
                    # step  33 slice _M14 with indices ['Q', 'V']
                    _M14_sliced_offset = offset_0        
                    _M14_sliced      = np.ndarray((NTHC_INT, (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(Q_0),
                                   ctypes.c_int(Q_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  34 SVQR,SQV->RSVQ
                    offset_now       = offset_5        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((R_1-R_0), NTHC_INT, (V_1-V_0), (Q_1-Q_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M15.ctypes.data),
                                                     ctypes.c_int(_M12.shape[0]),
                                                     ctypes.c_int(_M12.shape[1]),
                                                     ctypes.c_int(_M12.shape[2]),
                                                     ctypes.c_int(_M12.shape[3]))
                    # step  35 RSVQ->RVQS
                    _M15_perm_offset = offset_0        
                    _M15_perm        = np.ndarray(((R_1-R_0), (V_1-V_0), (Q_1-Q_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                                 ctypes.c_int((R_1-R_0)),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((V_1-V_0)),
                                                 ctypes.c_int((Q_1-Q_0)))
                    # step  36 cS,RVQS->cRVQ
                    offset_now       = offset_5        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NVIR, (R_1-R_0), (V_1-V_0), (Q_1-Q_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
                    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_9_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  37 slice _INPUT_22 with indices ['W']
                    _INPUT_22_sliced_offset = offset_0        
                    _INPUT_22_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_22_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_22.shape[0]),
                                 ctypes.c_int(_INPUT_22.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  38 cW,cRVQ->WRVQc
                    offset_now       = offset_6        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((W_1-W_0), (R_1-R_0), (V_1-V_0), (Q_1-Q_0), NVIR), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_0234_12340_wob(ctypes.c_void_p(_INPUT_22_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M16.ctypes.data),
                                                     ctypes.c_void_p(_M17.ctypes.data),
                                                     ctypes.c_int(_INPUT_22_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_22_sliced.shape[1]),
                                                     ctypes.c_int(_M16.shape[1]),
                                                     ctypes.c_int(_M16.shape[2]),
                                                     ctypes.c_int(_M16.shape[3]))
                    # step  39 cU,WRVQc->UWRVQ
                    offset_now       = offset_0        
                    _M20_offset      = offset_now      
                    _M20             = np.ndarray((NTHC_INT, (W_1-W_0), (R_1-R_0), (V_1-V_0), (Q_1-Q_0)), buffer = buffer, offset = _M20_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
                    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17.shape[3]
                    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M20.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M20.shape[0]
                    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_14_reshaped.T, _M17_reshaped.T, c=_M20_reshaped)
                    _M20             = _M20_reshaped.reshape(*shape_backup)
                    # step  40 UWRVQ->QVRUW
                    _M20_perm_offset = offset_5        
                    _M20_perm        = np.ndarray(((Q_1-Q_0), (V_1-V_0), (R_1-R_0), NTHC_INT, (W_1-W_0)), buffer = buffer, offset = _M20_perm_offset)
                    fn_permutation_01234_43201_wob(ctypes.c_void_p(_M20.ctypes.data),
                                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((R_1-R_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((Q_1-Q_0)))
                    # step  41 slice _M18 with indices ['R', 'W']
                    _M18_sliced_offset = offset_0        
                    _M18_sliced      = np.ndarray((NTHC_INT, (R_1-R_0), (W_1-W_0)), buffer = buffer, offset = _M18_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_sliced.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(R_0),
                                   ctypes.c_int(R_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  42 UQVW,URW->QVRUW
                    offset_now       = offset_6        
                    _M19_offset      = offset_now      
                    _M19             = np.ndarray(((Q_1-Q_0), (V_1-V_0), (R_1-R_0), NTHC_INT, (W_1-W_0)), buffer = buffer, offset = _M19_offset)
                    fn_contraction_0123_043_12403_wob(ctypes.c_void_p(_M8.ctypes.data),
                                                      ctypes.c_void_p(_M18_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M19.ctypes.data),
                                                      ctypes.c_int(_M8.shape[0]),
                                                      ctypes.c_int(_M8.shape[1]),
                                                      ctypes.c_int(_M8.shape[2]),
                                                      ctypes.c_int(_M8.shape[3]),
                                                      ctypes.c_int(_M18_sliced.shape[1]))
                    # step  43 QVRUW,QVRUW->
                    output_tmp       = ctypes.c_double(0.0)
                    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                           ctypes.c_void_p(_M20_perm.ctypes.data),
                           ctypes.c_int(_M19.size),
                           ctypes.pointer(output_tmp))
                    output_tmp = output_tmp.value
                    _M21 += output_tmp
                # step  44 end   for loop with indices ('V', 'W', 'Q', 'R')
            # step  45 end   for loop with indices ('V', 'W', 'Q')
        # step  46 end   for loop with indices ('V', 'W')
    # step  47 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_1_forloop_Q_S_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        Q_bunchsize = 8,
                                                        S_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    # assign the size of each tensor
    _M13_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _M1_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NVIR)))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _M4_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NOCC)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _INPUT_0_sliced_size = (NTHC_INT * NTHC_INT)
    _M7_perm_size    = (Q_bunchsize * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M8_perm_size    = (NTHC_INT * (Q_bunchsize * (V_bunchsize * W_bunchsize)))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M2_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M0_size         = (V_bunchsize * (W_bunchsize * NVIR))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M5_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M7_size         = (Q_bunchsize * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M8_size         = (NTHC_INT * (Q_bunchsize * (V_bunchsize * W_bunchsize)))
    _INPUT_4_sliced_size = (NVIR * NTHC_INT)
    _INPUT_18_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_5_sliced_size = (NTHC_INT * NTHC_INT)
    _M14_sliced_size = (NTHC_INT * (Q_bunchsize * V_bunchsize))
    _M15_perm_size   = (S_bunchsize * (NTHC_INT * (V_bunchsize * Q_bunchsize)))
    _INPUT_20_sliced_size = (NOCC * N_LAPLACE)
    _M18_size        = (NTHC_INT * (W_bunchsize * (S_bunchsize * (V_bunchsize * Q_bunchsize))))
    _M3_size         = (V_bunchsize * (W_bunchsize * NOCC))
    _INPUT_8_sliced_size = (NVIR * NTHC_INT)
    _M11_size        = (V_bunchsize * (Q_bunchsize * S_bunchsize))
    _M15_size        = (S_bunchsize * (NTHC_INT * (V_bunchsize * Q_bunchsize)))
    _M16_size        = (NOCC * (S_bunchsize * (V_bunchsize * Q_bunchsize)))
    _M18_perm_size   = (NTHC_INT * (W_bunchsize * (S_bunchsize * (V_bunchsize * Q_bunchsize))))
    _M10_size        = (Q_bunchsize * (S_bunchsize * NVIR))
    _M12_size        = (NTHC_INT * (V_bunchsize * (Q_bunchsize * S_bunchsize)))
    _M17_size        = (W_bunchsize * (S_bunchsize * (V_bunchsize * (Q_bunchsize * NOCC))))
    _M19_packed_size = (NTHC_INT * (W_bunchsize * S_bunchsize))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M20_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_17_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_15_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_0_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M7_perm_size)
    bucked_1_size    = max(bucked_1_size, _M8_perm_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M20_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M14_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M2_size)
    bucked_4_size    = max(bucked_4_size, _M6_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M0_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_19_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M5_size)
    bucked_5_size    = max(bucked_5_size, _M7_size)
    bucked_5_size    = max(bucked_5_size, _M8_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_4_sliced_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_18_sliced_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_5_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M14_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M15_perm_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_20_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M18_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M3_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_8_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M11_size)
    bucked_6_size    = max(bucked_6_size, _M15_size)
    bucked_6_size    = max(bucked_6_size, _M16_size)
    bucked_6_size    = max(bucked_6_size, _M18_perm_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M10_size)
    bucked_7_size    = max(bucked_7_size, _M12_size)
    bucked_7_size    = max(bucked_7_size, _M17_size)
    bucked_7_size    = max(bucked_7_size, _M19_packed_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    return output

def RMP3_CX_1_forloop_Q_S_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("TU,QVWT->UQVW" , _INPUT_10       , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 3, 1, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("bQ,bS->QSb"    , _INPUT_4        , _INPUT_8        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bV,QSb->VQS"   , _INPUT_18       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("RS,VQS->RVQS"  , _INPUT_5        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("RVQS,RQV->SRVQ", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 2, 3, 1)    )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("kR,SVQR->kSVQ" , _INPUT_7        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("kW,kSVQ->WSVQk", _INPUT_20       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("kU,WSVQk->UWSVQ", _INPUT_13       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 1, 2, 4, 3) )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("UWQV,UWSQV->UWS", _M8_perm        , _M18_perm       )
    del _M8_perm    
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("UWS,UWS->"     , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_CX_1_forloop_Q_S(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 USW->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M0.ctypes.data),
                                   ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M0.shape[0]),
                                   ctypes.c_int(_M0.shape[1]))
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _size_dim_1      = _size_dim_1 * _M1.shape[2]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M2.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
    _M2          = _M2_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M3              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M3.shape[0]),
                                   ctypes.c_int(_M3.shape[1]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _size_dim_1      = _size_dim_1 * _M4.shape[2]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TPVW,TPVW->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                      ctypes.c_void_p(_M5.ctypes.data),
                                      ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_int(_M2.shape[0]),
                                      ctypes.c_int(_M2.shape[1]),
                                      ctypes.c_int(_M2.shape[2]),
                                      ctypes.c_int(_M2.shape[3]))
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 TPVW->TVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M6_perm.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 QTVW->QVWT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 TU,QVWT->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M7_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 UQVW->UWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M8_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 bQ,bS->QSb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_8.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 bV,QSb->VQS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_18.shape[0]
    _INPUT_18_reshaped = _INPUT_18.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_18_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 RS,VQS->RVQS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_231_0231_wob = getattr(libpbc, "fn_contraction_01_231_0231_wob", None)
    assert fn_contraction_01_231_0231_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_231_0231_wob(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_int(_INPUT_5.shape[0]),
                                   ctypes.c_int(_INPUT_5.shape[1]),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 RVQS,RQV->SRVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M12.shape[0]),
                                     ctypes.c_int(_M12.shape[1]),
                                     ctypes.c_int(_M12.shape[2]),
                                     ctypes.c_int(_M12.shape[3]))
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SRVQ->SVQR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                 ctypes.c_int(_M15.shape[0]),
                                 ctypes.c_int(_M15.shape[1]),
                                 ctypes.c_int(_M15.shape[2]),
                                 ctypes.c_int(_M15.shape[3]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 kR,SVQR->kSVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 kW,kSVQ->WSVQk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0234_12340_wob = getattr(libpbc, "fn_contraction_01_0234_12340_wob", None)
    assert fn_contraction_01_0234_12340_wob is not None
    _M17             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_0234_12340_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                     ctypes.c_void_p(_M16.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_INPUT_20.shape[0]),
                                     ctypes.c_int(_INPUT_20.shape[1]),
                                     ctypes.c_int(_M16.shape[1]),
                                     ctypes.c_int(_M16.shape[2]),
                                     ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 kU,WSVQk->UWSVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _size_dim_1      = _size_dim_1 * _M17.shape[3]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M17_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17        
    del _M17_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 UWSVQ->UWSQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(_M18.shape[3]),
                                   ctypes.c_int(_M18.shape[4]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 UWQV,UWSQV->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01423_014_wob = getattr(libpbc, "fn_contraction_0123_01423_014_wob", None)
    assert fn_contraction_0123_01423_014_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_01423_014_wob(ctypes.c_void_p(_M8_perm.ctypes.data),
                                      ctypes.c_void_p(_M18_perm.ctypes.data),
                                      ctypes.c_void_p(_M19.ctypes.data),
                                      ctypes.c_int(_M8_perm.shape[0]),
                                      ctypes.c_int(_M8_perm.shape[1]),
                                      ctypes.c_int(_M8_perm.shape[2]),
                                      ctypes.c_int(_M8_perm.shape[3]),
                                      ctypes.c_int(_M18_perm.shape[2]))
    del _M8_perm    
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 UWS,UWS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_CX_1_forloop_Q_S_forloop_Q_S(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      Q_bunchsize = 8,
                                      S_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    fn_contraction_01_0234_12340_wob = getattr(libpbc, "fn_contraction_01_0234_12340_wob", None)
    assert fn_contraction_01_0234_12340_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_contraction_0123_01423_014_wob = getattr(libpbc, "fn_contraction_0123_01423_014_wob", None)
    assert fn_contraction_0123_01423_014_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_contraction_01_231_0231_wob = getattr(libpbc, "fn_contraction_01_231_0231_wob", None)
    assert fn_contraction_01_231_0231_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        Q_begin = rank*bunchsize
        Q_end = (rank+1)*bunchsize
        Q_begin          = min(Q_begin, NTHC_INT)
        Q_end            = min(Q_end, NTHC_INT)
    else:
        Q_begin          = 0               
        Q_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_1_forloop_Q_S_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           Q_bunchsize = Q_bunchsize,
                                                                           S_bunchsize = S_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 cS,cW->SWc
    offset_now       = offset_0        
    _M13_offset      = offset_now      
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   2 cU,SWc->USW
    offset_now       = offset_1        
    _M20_offset      = offset_now      
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    # step   3 allocate   _M19
    offset_now       = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    _M19_offset      = offset_now      
    _M19.ravel()[:] = 0.0
    # step   4 USW->UWS
    _M20_perm_offset = offset_2        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   5 jQ,jV->QVj
    offset_now       = offset_1        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   6 jR,QVj->RQV
    offset_now       = offset_3        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step   7 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step   8 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step   9 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_1        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  10 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  11 aV,aW->VWa
            offset_now       = offset_5        
            _M0_offset       = offset_now      
            _M0              = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M0_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M0.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  12 aP,VWa->PVWa
            offset_now       = offset_1        
            _M1_offset       = offset_now      
            _M1              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M1_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                           ctypes.c_void_p(_M0.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_int(_INPUT_2.shape[0]),
                                           ctypes.c_int(_INPUT_2.shape[1]),
                                           ctypes.c_int(_M0.shape[0]),
                                           ctypes.c_int(_M0.shape[1]))
            # step  13 aT,PVWa->TPVW
            offset_now       = offset_4        
            _M2_offset       = offset_now      
            _M2              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M2_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
            _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _size_dim_1      = _size_dim_1 * _M1.shape[2]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M2.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M2.shape[0]
            _M2_reshaped = _M2.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_12_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
            _M2              = _M2_reshaped.reshape(*shape_backup)
            # step  14 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_1        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  15 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_5        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  16 iV,iW->VWi
            offset_now       = offset_6        
            _M3_offset       = offset_now      
            _M3              = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M3_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M3.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  17 iP,VWi->PVWi
            offset_now       = offset_1        
            _M4_offset       = offset_now      
            _M4              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M4_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_int(_INPUT_1.shape[0]),
                                           ctypes.c_int(_INPUT_1.shape[1]),
                                           ctypes.c_int(_M3.shape[0]),
                                           ctypes.c_int(_M3.shape[1]))
            # step  18 iT,PVWi->TPVW
            offset_now       = offset_5        
            _M5_offset       = offset_now      
            _M5              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M5_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
            _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M4.shape[0]
            _size_dim_1      = _size_dim_1 * _M4.shape[1]
            _size_dim_1      = _size_dim_1 * _M4.shape[2]
            _M4_reshaped = _M4.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M5.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M5.shape[0]
            _M5_reshaped = _M5.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_11_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
            _M5              = _M5_reshaped.reshape(*shape_backup)
            # step  19 TPVW,TPVW->TPVW
            offset_now       = offset_1        
            _M6_offset       = offset_now      
            _M6              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M6_offset)
            fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                              ctypes.c_void_p(_M5.ctypes.data),
                                              ctypes.c_void_p(_M6.ctypes.data),
                                              ctypes.c_int(_M2.shape[0]),
                                              ctypes.c_int(_M2.shape[1]),
                                              ctypes.c_int(_M2.shape[2]),
                                              ctypes.c_int(_M2.shape[3]))
            # step  20 TPVW->TVWP
            _M6_perm_offset  = offset_4        
            _M6_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  21 start for loop with indices ('V', 'W', 'Q')
            for Q_0, Q_1 in lib.prange(Q_begin,Q_end,Q_bunchsize):
                # step  22 slice _INPUT_0 with indices ['Q']
                _INPUT_0_sliced_offset = offset_1        
                _INPUT_0_sliced  = np.ndarray((NTHC_INT, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_0_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(Q_0),
                             ctypes.c_int(Q_1))
                # step  23 PQ,TVWP->QTVW
                offset_now       = offset_5        
                _M7_offset       = offset_now      
                _M7              = np.ndarray(((Q_1-Q_0), NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M7_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_0_sliced.shape[0]
                _INPUT_0_sliced_reshaped = _INPUT_0_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M7.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7.shape[0]
                _M7_reshaped = _M7.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_0_sliced_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
                _M7              = _M7_reshaped.reshape(*shape_backup)
                # step  24 QTVW->QVWT
                _M7_perm_offset  = offset_1        
                _M7_perm         = np.ndarray(((Q_1-Q_0), (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
                fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                             ctypes.c_void_p(_M7_perm.ctypes.data),
                                             ctypes.c_int((Q_1-Q_0)),
                                             ctypes.c_int(NTHC_INT),
                                             ctypes.c_int((V_1-V_0)),
                                             ctypes.c_int((W_1-W_0)))
                # step  25 TU,QVWT->UQVW
                offset_now       = offset_5        
                _M8_offset       = offset_now      
                _M8              = np.ndarray((NTHC_INT, (Q_1-Q_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M8_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
                _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
                _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M8.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8.shape[0]
                _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_10_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
                _M8              = _M8_reshaped.reshape(*shape_backup)
                # step  26 UQVW->UWQV
                _M8_perm_offset  = offset_1        
                _M8_perm         = np.ndarray((NTHC_INT, (W_1-W_0), (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0312_wob(ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_void_p(_M8_perm.ctypes.data),
                                             ctypes.c_int(NTHC_INT),
                                             ctypes.c_int((Q_1-Q_0)),
                                             ctypes.c_int((V_1-V_0)),
                                             ctypes.c_int((W_1-W_0)))
                # step  27 start for loop with indices ('V', 'W', 'Q', 'S')
                for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
                    # step  28 slice _INPUT_4 with indices ['Q']
                    _INPUT_4_sliced_offset = offset_5        
                    _INPUT_4_sliced  = np.ndarray((NVIR, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_4_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
                    # step  29 slice _INPUT_8 with indices ['S']
                    _INPUT_8_sliced_offset = offset_6        
                    _INPUT_8_sliced  = np.ndarray((NVIR, (S_1-S_0)), buffer = buffer, offset = _INPUT_8_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(S_0),
                                 ctypes.c_int(S_1))
                    # step  30 bQ,bS->QSb
                    offset_now       = offset_7        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray(((Q_1-Q_0), (S_1-S_0), NVIR), buffer = buffer, offset = _M10_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M10.ctypes.data),
                                                 ctypes.c_int(_INPUT_4_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_4_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_8_sliced.shape[1]))
                    # step  31 slice _INPUT_18 with indices ['V']
                    _INPUT_18_sliced_offset = offset_5        
                    _INPUT_18_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_18_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  32 bV,QSb->VQS
                    offset_now       = offset_6        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((V_1-V_0), (Q_1-Q_0), (S_1-S_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_18_sliced.shape[0]
                    _INPUT_18_sliced_reshaped = _INPUT_18_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_18_sliced_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  33 slice _INPUT_5 with indices ['S']
                    _INPUT_5_sliced_offset = offset_5        
                    _INPUT_5_sliced  = np.ndarray((NTHC_INT, (S_1-S_0)), buffer = buffer, offset = _INPUT_5_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                 ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_5.shape[0]),
                                 ctypes.c_int(_INPUT_5.shape[1]),
                                 ctypes.c_int(S_0),
                                 ctypes.c_int(S_1))
                    # step  34 RS,VQS->RVQS
                    offset_now       = offset_7        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (V_1-V_0), (Q_1-Q_0), (S_1-S_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_231_0231_wob(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M11.ctypes.data),
                                                   ctypes.c_void_p(_M12.ctypes.data),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                                   ctypes.c_int(_M11.shape[0]),
                                                   ctypes.c_int(_M11.shape[1]))
                    # step  35 slice _M14 with indices ['Q', 'V']
                    _M14_sliced_offset = offset_5        
                    _M14_sliced      = np.ndarray((NTHC_INT, (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(Q_0),
                                   ctypes.c_int(Q_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  36 RVQS,RQV->SRVQ
                    offset_now       = offset_6        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((S_1-S_0), NTHC_INT, (V_1-V_0), (Q_1-Q_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M15.ctypes.data),
                                                     ctypes.c_int(_M12.shape[0]),
                                                     ctypes.c_int(_M12.shape[1]),
                                                     ctypes.c_int(_M12.shape[2]),
                                                     ctypes.c_int(_M12.shape[3]))
                    # step  37 SRVQ->SVQR
                    _M15_perm_offset = offset_5        
                    _M15_perm        = np.ndarray(((S_1-S_0), (V_1-V_0), (Q_1-Q_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                                 ctypes.c_int((S_1-S_0)),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((V_1-V_0)),
                                                 ctypes.c_int((Q_1-Q_0)))
                    # step  38 kR,SVQR->kSVQ
                    offset_now       = offset_6        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NOCC, (S_1-S_0), (V_1-V_0), (Q_1-Q_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
                    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_7_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  39 slice _INPUT_20 with indices ['W']
                    _INPUT_20_sliced_offset = offset_5        
                    _INPUT_20_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_20_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_20.shape[0]),
                                 ctypes.c_int(_INPUT_20.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  40 kW,kSVQ->WSVQk
                    offset_now       = offset_7        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((W_1-W_0), (S_1-S_0), (V_1-V_0), (Q_1-Q_0), NOCC), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_0234_12340_wob(ctypes.c_void_p(_INPUT_20_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M16.ctypes.data),
                                                     ctypes.c_void_p(_M17.ctypes.data),
                                                     ctypes.c_int(_INPUT_20_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_20_sliced.shape[1]),
                                                     ctypes.c_int(_M16.shape[1]),
                                                     ctypes.c_int(_M16.shape[2]),
                                                     ctypes.c_int(_M16.shape[3]))
                    # step  41 kU,WSVQk->UWSVQ
                    offset_now       = offset_5        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0), (V_1-V_0), (Q_1-Q_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
                    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17.shape[3]
                    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_13_reshaped.T, _M17_reshaped.T, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  42 UWSVQ->UWSQV
                    _M18_perm_offset = offset_6        
                    _M18_perm        = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0), (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = _M18_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M18.ctypes.data),
                                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((S_1-S_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((Q_1-Q_0)))
                    # step  43 UWQV,UWSQV->UWS
                    offset_now       = offset_7        
                    _M19_packed_offset = offset_now      
                    _M19_packed      = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0)), buffer = buffer, offset = _M19_packed_offset)
                    fn_contraction_0123_01423_014_wob(ctypes.c_void_p(_M8_perm.ctypes.data),
                                                      ctypes.c_void_p(_M18_perm.ctypes.data),
                                                      ctypes.c_void_p(_M19_packed.ctypes.data),
                                                      ctypes.c_int(_M8_perm.shape[0]),
                                                      ctypes.c_int(_M8_perm.shape[1]),
                                                      ctypes.c_int(_M8_perm.shape[2]),
                                                      ctypes.c_int(_M8_perm.shape[3]),
                                                      ctypes.c_int(_M18_perm.shape[2]))
                    # step  44 pack  _M19 with indices ['W', 'S']
                    fn_packadd_3_1_2(ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                     ctypes.c_int(_M19.shape[0]),
                                     ctypes.c_int(_M19.shape[1]),
                                     ctypes.c_int(_M19.shape[2]),
                                     ctypes.c_int(W_0),
                                     ctypes.c_int(W_1),
                                     ctypes.c_int(S_0),
                                     ctypes.c_int(S_1))
                # step  45 end   for loop with indices ('V', 'W', 'Q', 'S')
                # step  46 deallocate ['_M8']
            # step  47 end   for loop with indices ('V', 'W', 'Q')
            # step  48 deallocate ['_M6']
        # step  49 end   for loop with indices ('V', 'W')
    # step  50 end   for loop with indices ('V',)
    # step  51 deallocate ['_M14']
    # step  52 UWS,UWS->
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(output_tmp))
    _M21 = output_tmp.value
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_1_forloop_R_U_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        U_bunchsize = 8,
                                                        R_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M13_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _M1_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NVIR)))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _M4_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NOCC)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M7_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _INPUT_10_sliced_size = (NTHC_INT * NTHC_INT)
    _M14_sliced_size = (NTHC_INT * (R_bunchsize * V_bunchsize))
    _M15_perm_size   = (U_bunchsize * (W_bunchsize * (R_bunchsize * (NTHC_INT * V_bunchsize))))
    _INPUT_18_sliced_size = (NVIR * N_LAPLACE)
    _M18_size        = (NTHC_INT * (U_bunchsize * (W_bunchsize * R_bunchsize)))
    _INPUT_7_sliced_size = (NOCC * NTHC_INT)
    _INPUT_20_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_5_sliced_size = (NTHC_INT * NTHC_INT)
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M2_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M7_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M0_size         = (V_bunchsize * (W_bunchsize * NVIR))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M5_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M8_size         = (U_bunchsize * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M3_size         = (V_bunchsize * (W_bunchsize * NOCC))
    _M15_size        = (U_bunchsize * (W_bunchsize * (R_bunchsize * (NTHC_INT * V_bunchsize))))
    _M16_size        = (NVIR * (U_bunchsize * (W_bunchsize * (R_bunchsize * V_bunchsize))))
    _M18_perm_size   = (NTHC_INT * (U_bunchsize * (W_bunchsize * R_bunchsize)))
    _M17_size        = (NVIR * (U_bunchsize * (W_bunchsize * R_bunchsize)))
    _INPUT_13_sliced_size = (NOCC * NTHC_INT)
    _M11_size        = (W_bunchsize * (R_bunchsize * U_bunchsize))
    _M19_packed_size = (NTHC_INT * (W_bunchsize * U_bunchsize))
    _M10_size        = (R_bunchsize * (U_bunchsize * NOCC))
    _M12_size        = (NTHC_INT * (W_bunchsize * (U_bunchsize * R_bunchsize)))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M20_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_17_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_15_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M7_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_10_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M14_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M15_perm_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_18_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M18_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_7_sliced_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_20_sliced_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_5_sliced_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M20_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M14_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M2_size)
    bucked_4_size    = max(bucked_4_size, _M6_perm_size)
    bucked_4_size    = max(bucked_4_size, _M7_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M0_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_19_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M5_size)
    bucked_5_size    = max(bucked_5_size, _M8_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M3_size)
    bucked_6_size    = max(bucked_6_size, _M15_size)
    bucked_6_size    = max(bucked_6_size, _M16_size)
    bucked_6_size    = max(bucked_6_size, _M18_perm_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M17_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_13_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M11_size)
    bucked_7_size    = max(bucked_7_size, _M19_packed_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M10_size)
    bucked_8_size    = max(bucked_8_size, _M12_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_1_forloop_R_U_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("cU,cW->UWc"    , _INPUT_14       , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("cS,UWc->SUW"   , _INPUT_9        , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jR,jV->RVj"    , _INPUT_6        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("jQ,RVj->QRV"   , _INPUT_3        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("TU,QVWT->UQVW" , _INPUT_10       , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("UQVW,QRV->UWRQV", _M8             , _M14            )
    del _M8         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 1, 2, 4, 3) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("bQ,UWRVQ->bUWRV", _INPUT_4        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("bV,bUWRV->bUWR", _INPUT_18       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("bS,bUWR->SUWR" , _INPUT_8        , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 2, 1, 3)    )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("kR,kU->RUk"    , _INPUT_7        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("kW,RUk->WRU"   , _INPUT_20       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("RS,WRU->SWUR"  , _INPUT_5        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("SWUR,SWUR->SWU", _M12            , _M18_perm       )
    del _M12        
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("SWU,SWU->"     , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_1_forloop_R_U(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 cU,cW->UWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 cS,UWc->SUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 SUW->SWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jR,jV->RVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 jQ,RVj->QRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M0.ctypes.data),
                                   ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M0.shape[0]),
                                   ctypes.c_int(_M0.shape[1]))
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _size_dim_1      = _size_dim_1 * _M1.shape[2]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M2.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
    _M2          = _M2_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M3              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M3.shape[0]),
                                   ctypes.c_int(_M3.shape[1]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _size_dim_1      = _size_dim_1 * _M4.shape[2]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TPVW,TPVW->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                      ctypes.c_void_p(_M5.ctypes.data),
                                      ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_int(_M2.shape[0]),
                                      ctypes.c_int(_M2.shape[1]),
                                      ctypes.c_int(_M2.shape[2]),
                                      ctypes.c_int(_M2.shape[3]))
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 TPVW->TVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M6_perm.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 QTVW->QVWT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 TU,QVWT->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M7_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 UQVW,QRV->UWRQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_142_03412_wob = getattr(libpbc, "fn_contraction_0123_142_03412_wob", None)
    assert fn_contraction_0123_142_03412_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_142_03412_wob(ctypes.c_void_p(_M8.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_M8.shape[0]),
                                      ctypes.c_int(_M8.shape[1]),
                                      ctypes.c_int(_M8.shape[2]),
                                      ctypes.c_int(_M8.shape[3]),
                                      ctypes.c_int(_M14.shape[1]))
    del _M8         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 UWRQV->UWRVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 bQ,UWRVQ->bUWRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 bV,bUWRV->bUWR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02341_0234_wob = getattr(libpbc, "fn_contraction_01_02341_0234_wob", None)
    assert fn_contraction_01_02341_0234_wob is not None
    _M17             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_02341_0234_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                     ctypes.c_void_p(_M16.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_INPUT_18.shape[0]),
                                     ctypes.c_int(_INPUT_18.shape[1]),
                                     ctypes.c_int(_M16.shape[1]),
                                     ctypes.c_int(_M16.shape[2]),
                                     ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 bS,bUWR->SUWR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M17_reshaped, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SUWR->SWUR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0213_wob = getattr(libpbc, "fn_permutation_0123_0213_wob", None)
    assert fn_permutation_0123_0213_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0213_wob(ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_void_p(_M18_perm.ctypes.data),
                                 ctypes.c_int(_M18.shape[0]),
                                 ctypes.c_int(_M18.shape[1]),
                                 ctypes.c_int(_M18.shape[2]),
                                 ctypes.c_int(_M18.shape[3]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 kR,kU->RUk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_13.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 kW,RUk->WRU 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_20.shape[0]
    _INPUT_20_reshaped = _INPUT_20.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_20_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 RS,WRU->SWUR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_203_1230_wob = getattr(libpbc, "fn_contraction_01_203_1230_wob", None)
    assert fn_contraction_01_203_1230_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_203_1230_wob(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_int(_INPUT_5.shape[0]),
                                   ctypes.c_int(_INPUT_5.shape[1]),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[2]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 SWUR,SWUR->SWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_012_wob = getattr(libpbc, "fn_contraction_0123_0123_012_wob", None)
    assert fn_contraction_0123_0123_012_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_0123_012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_void_p(_M18_perm.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M12.shape[0]),
                                     ctypes.c_int(_M12.shape[1]),
                                     ctypes.c_int(_M12.shape[2]),
                                     ctypes.c_int(_M12.shape[3]))
    del _M12        
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 SWU,SWU-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_1_forloop_R_U_forloop_U_R(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      U_bunchsize = 8,
                                      R_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_permutation_0123_0213_wob = getattr(libpbc, "fn_permutation_0123_0213_wob", None)
    assert fn_permutation_0123_0213_wob is not None
    fn_contraction_0123_142_03412_wob = getattr(libpbc, "fn_contraction_0123_142_03412_wob", None)
    assert fn_contraction_0123_142_03412_wob is not None
    fn_contraction_01_02341_0234_wob = getattr(libpbc, "fn_contraction_01_02341_0234_wob", None)
    assert fn_contraction_01_02341_0234_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_contraction_01_203_1230_wob = getattr(libpbc, "fn_contraction_01_203_1230_wob", None)
    assert fn_contraction_01_203_1230_wob is not None
    fn_contraction_0123_0123_012_wob = getattr(libpbc, "fn_contraction_0123_0123_012_wob", None)
    assert fn_contraction_0123_0123_012_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        U_begin = rank*bunchsize
        U_end = (rank+1)*bunchsize
        U_begin          = min(U_begin, NTHC_INT)
        U_end            = min(U_end, NTHC_INT)
    else:
        U_begin          = 0               
        U_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_1_forloop_R_U_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           U_bunchsize = U_bunchsize,
                                                                           R_bunchsize = R_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 cU,cW->UWc
    offset_now       = offset_0        
    _M13_offset      = offset_now      
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   2 cS,UWc->SUW
    offset_now       = offset_1        
    _M20_offset      = offset_now      
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    # step   3 allocate   _M19
    offset_now       = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    _M19_offset      = offset_now      
    _M19.ravel()[:] = 0.0
    # step   4 SUW->SWU
    _M20_perm_offset = offset_2        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   5 jR,jV->RVj
    offset_now       = offset_1        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   6 jQ,RVj->QRV
    offset_now       = offset_3        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step   7 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step   8 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step   9 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_1        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  10 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  11 aV,aW->VWa
            offset_now       = offset_5        
            _M0_offset       = offset_now      
            _M0              = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M0_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M0.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  12 aP,VWa->PVWa
            offset_now       = offset_1        
            _M1_offset       = offset_now      
            _M1              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M1_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                           ctypes.c_void_p(_M0.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_int(_INPUT_2.shape[0]),
                                           ctypes.c_int(_INPUT_2.shape[1]),
                                           ctypes.c_int(_M0.shape[0]),
                                           ctypes.c_int(_M0.shape[1]))
            # step  13 aT,PVWa->TPVW
            offset_now       = offset_4        
            _M2_offset       = offset_now      
            _M2              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M2_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
            _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _size_dim_1      = _size_dim_1 * _M1.shape[2]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M2.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M2.shape[0]
            _M2_reshaped = _M2.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_12_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
            _M2              = _M2_reshaped.reshape(*shape_backup)
            # step  14 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_1        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  15 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_5        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  16 iV,iW->VWi
            offset_now       = offset_6        
            _M3_offset       = offset_now      
            _M3              = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M3_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M3.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  17 iP,VWi->PVWi
            offset_now       = offset_1        
            _M4_offset       = offset_now      
            _M4              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M4_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_int(_INPUT_1.shape[0]),
                                           ctypes.c_int(_INPUT_1.shape[1]),
                                           ctypes.c_int(_M3.shape[0]),
                                           ctypes.c_int(_M3.shape[1]))
            # step  18 iT,PVWi->TPVW
            offset_now       = offset_5        
            _M5_offset       = offset_now      
            _M5              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M5_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
            _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M4.shape[0]
            _size_dim_1      = _size_dim_1 * _M4.shape[1]
            _size_dim_1      = _size_dim_1 * _M4.shape[2]
            _M4_reshaped = _M4.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M5.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M5.shape[0]
            _M5_reshaped = _M5.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_11_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
            _M5              = _M5_reshaped.reshape(*shape_backup)
            # step  19 TPVW,TPVW->TPVW
            offset_now       = offset_1        
            _M6_offset       = offset_now      
            _M6              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M6_offset)
            fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                              ctypes.c_void_p(_M5.ctypes.data),
                                              ctypes.c_void_p(_M6.ctypes.data),
                                              ctypes.c_int(_M2.shape[0]),
                                              ctypes.c_int(_M2.shape[1]),
                                              ctypes.c_int(_M2.shape[2]),
                                              ctypes.c_int(_M2.shape[3]))
            # step  20 TPVW->TVWP
            _M6_perm_offset  = offset_4        
            _M6_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  21 PQ,TVWP->QTVW
            offset_now       = offset_1        
            _M7_offset       = offset_now      
            _M7              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M7_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
            _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
            _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M7.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M7.shape[0]
            _M7_reshaped = _M7.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
            _M7              = _M7_reshaped.reshape(*shape_backup)
            # step  22 QTVW->QVWT
            _M7_perm_offset  = offset_4        
            _M7_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                         ctypes.c_void_p(_M7_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  23 start for loop with indices ('V', 'W', 'U')
            for U_0, U_1 in lib.prange(U_begin,U_end,U_bunchsize):
                # step  24 slice _INPUT_10 with indices ['U']
                _INPUT_10_sliced_offset = offset_1        
                _INPUT_10_sliced = np.ndarray((NTHC_INT, (U_1-U_0)), buffer = buffer, offset = _INPUT_10_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_10.shape[0]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_int(U_0),
                             ctypes.c_int(U_1))
                # step  25 TU,QVWT->UQVW
                offset_now       = offset_5        
                _M8_offset       = offset_now      
                _M8              = np.ndarray(((U_1-U_0), NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M8_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_10_sliced.shape[0]
                _INPUT_10_sliced_reshaped = _INPUT_10_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
                _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M8.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8.shape[0]
                _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_10_sliced_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
                _M8              = _M8_reshaped.reshape(*shape_backup)
                # step  26 start for loop with indices ('V', 'W', 'U', 'R')
                for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
                    # step  27 slice _M14 with indices ['R', 'V']
                    _M14_sliced_offset = offset_1        
                    _M14_sliced      = np.ndarray((NTHC_INT, (R_1-R_0), (V_1-V_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(R_0),
                                   ctypes.c_int(R_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  28 UQVW,QRV->UWRQV
                    offset_now       = offset_6        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((U_1-U_0), (W_1-W_0), (R_1-R_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_0123_142_03412_wob(ctypes.c_void_p(_M8.ctypes.data),
                                                      ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M15.ctypes.data),
                                                      ctypes.c_int(_M8.shape[0]),
                                                      ctypes.c_int(_M8.shape[1]),
                                                      ctypes.c_int(_M8.shape[2]),
                                                      ctypes.c_int(_M8.shape[3]),
                                                      ctypes.c_int(_M14_sliced.shape[1]))
                    # step  29 UWRQV->UWRVQ
                    _M15_perm_offset = offset_1        
                    _M15_perm        = np.ndarray(((U_1-U_0), (W_1-W_0), (R_1-R_0), (V_1-V_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                                   ctypes.c_int((U_1-U_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((R_1-R_0)),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  30 bQ,UWRVQ->bUWRV
                    offset_now       = offset_6        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NVIR, (U_1-U_0), (W_1-W_0), (R_1-R_0), (V_1-V_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
                    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_4_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  31 slice _INPUT_18 with indices ['V']
                    _INPUT_18_sliced_offset = offset_1        
                    _INPUT_18_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_18_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  32 bV,bUWRV->bUWR
                    offset_now       = offset_7        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray((NVIR, (U_1-U_0), (W_1-W_0), (R_1-R_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_02341_0234_wob(ctypes.c_void_p(_INPUT_18_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M16.ctypes.data),
                                                     ctypes.c_void_p(_M17.ctypes.data),
                                                     ctypes.c_int(_INPUT_18_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_18_sliced.shape[1]),
                                                     ctypes.c_int(_M16.shape[1]),
                                                     ctypes.c_int(_M16.shape[2]),
                                                     ctypes.c_int(_M16.shape[3]))
                    # step  33 bS,bUWR->SUWR
                    offset_now       = offset_1        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (U_1-U_0), (W_1-W_0), (R_1-R_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
                    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17.shape[0]
                    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_8_reshaped.T, _M17_reshaped, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  34 SUWR->SWUR
                    _M18_perm_offset = offset_6        
                    _M18_perm        = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0), (R_1-R_0)), buffer = buffer, offset = _M18_perm_offset)
                    fn_permutation_0123_0213_wob(ctypes.c_void_p(_M18.ctypes.data),
                                                 ctypes.c_void_p(_M18_perm.ctypes.data),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((U_1-U_0)),
                                                 ctypes.c_int((W_1-W_0)),
                                                 ctypes.c_int((R_1-R_0)))
                    # step  35 slice _INPUT_7 with indices ['R']
                    _INPUT_7_sliced_offset = offset_1        
                    _INPUT_7_sliced  = np.ndarray((NOCC, (R_1-R_0)), buffer = buffer, offset = _INPUT_7_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(R_0),
                                 ctypes.c_int(R_1))
                    # step  36 slice _INPUT_13 with indices ['U']
                    _INPUT_13_sliced_offset = offset_7        
                    _INPUT_13_sliced = np.ndarray((NOCC, (U_1-U_0)), buffer = buffer, offset = _INPUT_13_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(U_0),
                                 ctypes.c_int(U_1))
                    # step  37 kR,kU->RUk
                    offset_now       = offset_8        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray(((R_1-R_0), (U_1-U_0), NOCC), buffer = buffer, offset = _M10_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M10.ctypes.data),
                                                 ctypes.c_int(_INPUT_7_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_7_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_13_sliced.shape[1]))
                    # step  38 slice _INPUT_20 with indices ['W']
                    _INPUT_20_sliced_offset = offset_1        
                    _INPUT_20_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_20_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_20.shape[0]),
                                 ctypes.c_int(_INPUT_20.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  39 kW,RUk->WRU
                    offset_now       = offset_7        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((W_1-W_0), (R_1-R_0), (U_1-U_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_20_sliced.shape[0]
                    _INPUT_20_sliced_reshaped = _INPUT_20_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_20_sliced_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  40 slice _INPUT_5 with indices ['R']
                    _INPUT_5_sliced_offset = offset_1        
                    _INPUT_5_sliced  = np.ndarray(((R_1-R_0), NTHC_INT), buffer = buffer, offset = _INPUT_5_sliced_offset)
                    fn_slice_2_0(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                 ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_5.shape[0]),
                                 ctypes.c_int(_INPUT_5.shape[1]),
                                 ctypes.c_int(R_0),
                                 ctypes.c_int(R_1))
                    # step  41 RS,WRU->SWUR
                    offset_now       = offset_8        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0), (R_1-R_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_203_1230_wob(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M11.ctypes.data),
                                                   ctypes.c_void_p(_M12.ctypes.data),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                                   ctypes.c_int(_M11.shape[0]),
                                                   ctypes.c_int(_M11.shape[2]))
                    # step  42 SWUR,SWUR->SWU
                    offset_now       = offset_7        
                    _M19_packed_offset = offset_now      
                    _M19_packed      = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0)), buffer = buffer, offset = _M19_packed_offset)
                    fn_contraction_0123_0123_012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_void_p(_M18_perm.ctypes.data),
                                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                                     ctypes.c_int(_M12.shape[0]),
                                                     ctypes.c_int(_M12.shape[1]),
                                                     ctypes.c_int(_M12.shape[2]),
                                                     ctypes.c_int(_M12.shape[3]))
                    # step  43 pack  _M19 with indices ['W', 'U']
                    fn_packadd_3_1_2(ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                     ctypes.c_int(_M19.shape[0]),
                                     ctypes.c_int(_M19.shape[1]),
                                     ctypes.c_int(_M19.shape[2]),
                                     ctypes.c_int(W_0),
                                     ctypes.c_int(W_1),
                                     ctypes.c_int(U_0),
                                     ctypes.c_int(U_1))
                # step  44 end   for loop with indices ('V', 'W', 'U', 'R')
                # step  45 deallocate ['_M8']
            # step  46 end   for loop with indices ('V', 'W', 'U')
            # step  47 deallocate ['_M7']
        # step  48 end   for loop with indices ('V', 'W')
    # step  49 end   for loop with indices ('V',)
    # step  50 deallocate ['_M14']
    # step  51 SWU,SWU->
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(output_tmp))
    _M21 = output_tmp.value
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_1_forloop_S_U_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        U_bunchsize = 8,
                                                        S_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M13_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _M1_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NOCC)))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _M4_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NVIR)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M7_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _INPUT_10_sliced_size = (NTHC_INT * NTHC_INT)
    _M14_sliced_size = (NTHC_INT * (S_bunchsize * V_bunchsize))
    _M15_perm_size   = (U_bunchsize * (W_bunchsize * (S_bunchsize * (NTHC_INT * V_bunchsize))))
    _INPUT_16_sliced_size = (NOCC * N_LAPLACE)
    _M18_size        = (NTHC_INT * (U_bunchsize * (W_bunchsize * S_bunchsize)))
    _INPUT_9_sliced_size = (NVIR * NTHC_INT)
    _INPUT_22_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_5_sliced_size = (NTHC_INT * NTHC_INT)
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M2_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M7_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M0_size         = (V_bunchsize * (W_bunchsize * NOCC))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M5_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M8_size         = (U_bunchsize * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M3_size         = (V_bunchsize * (W_bunchsize * NVIR))
    _M15_size        = (U_bunchsize * (W_bunchsize * (S_bunchsize * (NTHC_INT * V_bunchsize))))
    _M16_size        = (NOCC * (U_bunchsize * (W_bunchsize * (S_bunchsize * V_bunchsize))))
    _M18_perm_size   = (NTHC_INT * (U_bunchsize * (W_bunchsize * S_bunchsize)))
    _M17_size        = (NOCC * (U_bunchsize * (W_bunchsize * S_bunchsize)))
    _INPUT_14_sliced_size = (NVIR * NTHC_INT)
    _M11_size        = (W_bunchsize * (S_bunchsize * U_bunchsize))
    _M19_packed_size = (NTHC_INT * (W_bunchsize * U_bunchsize))
    _M10_size        = (S_bunchsize * (U_bunchsize * NVIR))
    _M12_size        = (NTHC_INT * (W_bunchsize * (U_bunchsize * S_bunchsize)))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M20_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_15_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_17_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M7_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_10_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M14_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M15_perm_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_16_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M18_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_9_sliced_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_22_sliced_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_5_sliced_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M20_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M14_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_19_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M2_size)
    bucked_4_size    = max(bucked_4_size, _M6_perm_size)
    bucked_4_size    = max(bucked_4_size, _M7_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M0_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_21_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M5_size)
    bucked_5_size    = max(bucked_5_size, _M8_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M3_size)
    bucked_6_size    = max(bucked_6_size, _M15_size)
    bucked_6_size    = max(bucked_6_size, _M16_size)
    bucked_6_size    = max(bucked_6_size, _M18_perm_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M17_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_14_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M11_size)
    bucked_7_size    = max(bucked_7_size, _M19_packed_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M10_size)
    bucked_8_size    = max(bucked_8_size, _M12_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_1_forloop_S_U_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("kU,kW->UWk"    , _INPUT_13       , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("kR,UWk->RUW"   , _INPUT_7        , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("bS,bV->SVb"    , _INPUT_8        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("bQ,SVb->QSV"   , _INPUT_4        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("TU,QVWT->UQVW" , _INPUT_10       , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("UQVW,QSV->UWSQV", _M8             , _M14            )
    del _M8         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 1, 2, 4, 3) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("jQ,UWSVQ->jUWSV", _INPUT_3        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("jV,jUWSV->jUWS", _INPUT_16       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("jR,jUWS->RUWS" , _INPUT_6        , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 2, 1, 3)    )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("cS,cU->SUc"    , _INPUT_9        , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("cW,SUc->WSU"   , _INPUT_22       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("RS,WSU->RWUS"  , _INPUT_5        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("RWUS,RWUS->RWU", _M12            , _M18_perm       )
    del _M12        
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("RWU,RWU->"     , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_1_forloop_S_U(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 kU,kW->UWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 kR,UWk->RUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 RUW->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bS,bV->SVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bQ,SVb->QSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M0.ctypes.data),
                                   ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M0.shape[0]),
                                   ctypes.c_int(_M0.shape[1]))
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _size_dim_1      = _size_dim_1 * _M1.shape[2]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M2.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
    _M2          = _M2_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M3              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M3.shape[0]),
                                   ctypes.c_int(_M3.shape[1]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _size_dim_1      = _size_dim_1 * _M4.shape[2]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TPVW,TPVW->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                      ctypes.c_void_p(_M5.ctypes.data),
                                      ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_int(_M2.shape[0]),
                                      ctypes.c_int(_M2.shape[1]),
                                      ctypes.c_int(_M2.shape[2]),
                                      ctypes.c_int(_M2.shape[3]))
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 TPVW->TVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M6_perm.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 QTVW->QVWT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 TU,QVWT->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M7_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 UQVW,QSV->UWSQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_142_03412_wob = getattr(libpbc, "fn_contraction_0123_142_03412_wob", None)
    assert fn_contraction_0123_142_03412_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_142_03412_wob(ctypes.c_void_p(_M8.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_M8.shape[0]),
                                      ctypes.c_int(_M8.shape[1]),
                                      ctypes.c_int(_M8.shape[2]),
                                      ctypes.c_int(_M8.shape[3]),
                                      ctypes.c_int(_M14.shape[1]))
    del _M8         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 UWSQV->UWSVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 jQ,UWSVQ->jUWSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 jV,jUWSV->jUWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02341_0234_wob = getattr(libpbc, "fn_contraction_01_02341_0234_wob", None)
    assert fn_contraction_01_02341_0234_wob is not None
    _M17             = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_02341_0234_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                     ctypes.c_void_p(_M16.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_INPUT_16.shape[0]),
                                     ctypes.c_int(_INPUT_16.shape[1]),
                                     ctypes.c_int(_M16.shape[1]),
                                     ctypes.c_int(_M16.shape[2]),
                                     ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 jR,jUWS->RUWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M17_reshaped, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 RUWS->RWUS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0213_wob = getattr(libpbc, "fn_permutation_0123_0213_wob", None)
    assert fn_permutation_0123_0213_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0213_wob(ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_void_p(_M18_perm.ctypes.data),
                                 ctypes.c_int(_M18.shape[0]),
                                 ctypes.c_int(_M18.shape[1]),
                                 ctypes.c_int(_M18.shape[2]),
                                 ctypes.c_int(_M18.shape[3]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 cS,cU->SUc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 cW,SUc->WSU 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_22.shape[0]
    _INPUT_22_reshaped = _INPUT_22.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_22_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 RS,WSU->RWUS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_213_0231_wob = getattr(libpbc, "fn_contraction_01_213_0231_wob", None)
    assert fn_contraction_01_213_0231_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_213_0231_wob(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_int(_INPUT_5.shape[0]),
                                   ctypes.c_int(_INPUT_5.shape[1]),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[2]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 RWUS,RWUS->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_012_wob = getattr(libpbc, "fn_contraction_0123_0123_012_wob", None)
    assert fn_contraction_0123_0123_012_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_0123_012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_void_p(_M18_perm.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M12.shape[0]),
                                     ctypes.c_int(_M12.shape[1]),
                                     ctypes.c_int(_M12.shape[2]),
                                     ctypes.c_int(_M12.shape[3]))
    del _M12        
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 RWU,RWU-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_1_forloop_S_U_forloop_U_S(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      U_bunchsize = 8,
                                      S_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_o             
    _INPUT_8         = X_v             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_permutation_0123_0213_wob = getattr(libpbc, "fn_permutation_0123_0213_wob", None)
    assert fn_permutation_0123_0213_wob is not None
    fn_contraction_0123_142_03412_wob = getattr(libpbc, "fn_contraction_0123_142_03412_wob", None)
    assert fn_contraction_0123_142_03412_wob is not None
    fn_contraction_01_02341_0234_wob = getattr(libpbc, "fn_contraction_01_02341_0234_wob", None)
    assert fn_contraction_01_02341_0234_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_213_0231_wob = getattr(libpbc, "fn_contraction_01_213_0231_wob", None)
    assert fn_contraction_01_213_0231_wob is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_contraction_0123_0123_012_wob = getattr(libpbc, "fn_contraction_0123_0123_012_wob", None)
    assert fn_contraction_0123_0123_012_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        U_begin = rank*bunchsize
        U_end = (rank+1)*bunchsize
        U_begin          = min(U_begin, NTHC_INT)
        U_end            = min(U_end, NTHC_INT)
    else:
        U_begin          = 0               
        U_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_1_forloop_S_U_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           U_bunchsize = U_bunchsize,
                                                                           S_bunchsize = S_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 kU,kW->UWk
    offset_now       = offset_0        
    _M13_offset      = offset_now      
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   2 kR,UWk->RUW
    offset_now       = offset_1        
    _M20_offset      = offset_now      
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    # step   3 allocate   _M19
    offset_now       = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    _M19_offset      = offset_now      
    _M19.ravel()[:] = 0.0
    # step   4 RUW->RWU
    _M20_perm_offset = offset_2        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   5 bS,bV->SVb
    offset_now       = offset_1        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   6 bQ,SVb->QSV
    offset_now       = offset_3        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step   7 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step   8 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step   9 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_1        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  10 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_4        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  11 iV,iW->VWi
            offset_now       = offset_5        
            _M0_offset       = offset_now      
            _M0              = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M0_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M0.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  12 iP,VWi->PVWi
            offset_now       = offset_1        
            _M1_offset       = offset_now      
            _M1              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M1_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                           ctypes.c_void_p(_M0.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_int(_INPUT_1.shape[0]),
                                           ctypes.c_int(_INPUT_1.shape[1]),
                                           ctypes.c_int(_M0.shape[0]),
                                           ctypes.c_int(_M0.shape[1]))
            # step  13 iT,PVWi->TPVW
            offset_now       = offset_4        
            _M2_offset       = offset_now      
            _M2              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M2_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
            _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _size_dim_1      = _size_dim_1 * _M1.shape[2]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M2.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M2.shape[0]
            _M2_reshaped = _M2.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_11_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
            _M2              = _M2_reshaped.reshape(*shape_backup)
            # step  14 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_1        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  15 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_5        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  16 aV,aW->VWa
            offset_now       = offset_6        
            _M3_offset       = offset_now      
            _M3              = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M3_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M3.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  17 aP,VWa->PVWa
            offset_now       = offset_1        
            _M4_offset       = offset_now      
            _M4              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M4_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_int(_INPUT_2.shape[0]),
                                           ctypes.c_int(_INPUT_2.shape[1]),
                                           ctypes.c_int(_M3.shape[0]),
                                           ctypes.c_int(_M3.shape[1]))
            # step  18 aT,PVWa->TPVW
            offset_now       = offset_5        
            _M5_offset       = offset_now      
            _M5              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M5_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
            _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M4.shape[0]
            _size_dim_1      = _size_dim_1 * _M4.shape[1]
            _size_dim_1      = _size_dim_1 * _M4.shape[2]
            _M4_reshaped = _M4.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M5.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M5.shape[0]
            _M5_reshaped = _M5.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_12_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
            _M5              = _M5_reshaped.reshape(*shape_backup)
            # step  19 TPVW,TPVW->TPVW
            offset_now       = offset_1        
            _M6_offset       = offset_now      
            _M6              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M6_offset)
            fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                              ctypes.c_void_p(_M5.ctypes.data),
                                              ctypes.c_void_p(_M6.ctypes.data),
                                              ctypes.c_int(_M2.shape[0]),
                                              ctypes.c_int(_M2.shape[1]),
                                              ctypes.c_int(_M2.shape[2]),
                                              ctypes.c_int(_M2.shape[3]))
            # step  20 TPVW->TVWP
            _M6_perm_offset  = offset_4        
            _M6_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  21 PQ,TVWP->QTVW
            offset_now       = offset_1        
            _M7_offset       = offset_now      
            _M7              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M7_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
            _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
            _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M7.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M7.shape[0]
            _M7_reshaped = _M7.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M7_reshaped)
            _M7              = _M7_reshaped.reshape(*shape_backup)
            # step  22 QTVW->QVWT
            _M7_perm_offset  = offset_4        
            _M7_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                         ctypes.c_void_p(_M7_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  23 start for loop with indices ('V', 'W', 'U')
            for U_0, U_1 in lib.prange(U_begin,U_end,U_bunchsize):
                # step  24 slice _INPUT_10 with indices ['U']
                _INPUT_10_sliced_offset = offset_1        
                _INPUT_10_sliced = np.ndarray((NTHC_INT, (U_1-U_0)), buffer = buffer, offset = _INPUT_10_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_10.shape[0]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_int(U_0),
                             ctypes.c_int(U_1))
                # step  25 TU,QVWT->UQVW
                offset_now       = offset_5        
                _M8_offset       = offset_now      
                _M8              = np.ndarray(((U_1-U_0), NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M8_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_10_sliced.shape[0]
                _INPUT_10_sliced_reshaped = _INPUT_10_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
                _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M8.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8.shape[0]
                _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_10_sliced_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
                _M8              = _M8_reshaped.reshape(*shape_backup)
                # step  26 start for loop with indices ('V', 'W', 'U', 'S')
                for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
                    # step  27 slice _M14 with indices ['S', 'V']
                    _M14_sliced_offset = offset_1        
                    _M14_sliced      = np.ndarray((NTHC_INT, (S_1-S_0), (V_1-V_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(S_0),
                                   ctypes.c_int(S_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  28 UQVW,QSV->UWSQV
                    offset_now       = offset_6        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((U_1-U_0), (W_1-W_0), (S_1-S_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_0123_142_03412_wob(ctypes.c_void_p(_M8.ctypes.data),
                                                      ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M15.ctypes.data),
                                                      ctypes.c_int(_M8.shape[0]),
                                                      ctypes.c_int(_M8.shape[1]),
                                                      ctypes.c_int(_M8.shape[2]),
                                                      ctypes.c_int(_M8.shape[3]),
                                                      ctypes.c_int(_M14_sliced.shape[1]))
                    # step  29 UWSQV->UWSVQ
                    _M15_perm_offset = offset_1        
                    _M15_perm        = np.ndarray(((U_1-U_0), (W_1-W_0), (S_1-S_0), (V_1-V_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                                   ctypes.c_int((U_1-U_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((S_1-S_0)),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  30 jQ,UWSVQ->jUWSV
                    offset_now       = offset_6        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NOCC, (U_1-U_0), (W_1-W_0), (S_1-S_0), (V_1-V_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
                    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_3_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  31 slice _INPUT_16 with indices ['V']
                    _INPUT_16_sliced_offset = offset_1        
                    _INPUT_16_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_16_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_16.shape[0]),
                                 ctypes.c_int(_INPUT_16.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  32 jV,jUWSV->jUWS
                    offset_now       = offset_7        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray((NOCC, (U_1-U_0), (W_1-W_0), (S_1-S_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_02341_0234_wob(ctypes.c_void_p(_INPUT_16_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M16.ctypes.data),
                                                     ctypes.c_void_p(_M17.ctypes.data),
                                                     ctypes.c_int(_INPUT_16_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_16_sliced.shape[1]),
                                                     ctypes.c_int(_M16.shape[1]),
                                                     ctypes.c_int(_M16.shape[2]),
                                                     ctypes.c_int(_M16.shape[3]))
                    # step  33 jR,jUWS->RUWS
                    offset_now       = offset_1        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (U_1-U_0), (W_1-W_0), (S_1-S_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
                    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17.shape[0]
                    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_6_reshaped.T, _M17_reshaped, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  34 RUWS->RWUS
                    _M18_perm_offset = offset_6        
                    _M18_perm        = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0), (S_1-S_0)), buffer = buffer, offset = _M18_perm_offset)
                    fn_permutation_0123_0213_wob(ctypes.c_void_p(_M18.ctypes.data),
                                                 ctypes.c_void_p(_M18_perm.ctypes.data),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((U_1-U_0)),
                                                 ctypes.c_int((W_1-W_0)),
                                                 ctypes.c_int((S_1-S_0)))
                    # step  35 slice _INPUT_9 with indices ['S']
                    _INPUT_9_sliced_offset = offset_1        
                    _INPUT_9_sliced  = np.ndarray((NVIR, (S_1-S_0)), buffer = buffer, offset = _INPUT_9_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(S_0),
                                 ctypes.c_int(S_1))
                    # step  36 slice _INPUT_14 with indices ['U']
                    _INPUT_14_sliced_offset = offset_7        
                    _INPUT_14_sliced = np.ndarray((NVIR, (U_1-U_0)), buffer = buffer, offset = _INPUT_14_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(U_0),
                                 ctypes.c_int(U_1))
                    # step  37 cS,cU->SUc
                    offset_now       = offset_8        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray(((S_1-S_0), (U_1-U_0), NVIR), buffer = buffer, offset = _M10_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_14_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M10.ctypes.data),
                                                 ctypes.c_int(_INPUT_9_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_9_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_14_sliced.shape[1]))
                    # step  38 slice _INPUT_22 with indices ['W']
                    _INPUT_22_sliced_offset = offset_1        
                    _INPUT_22_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_22_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_22.shape[0]),
                                 ctypes.c_int(_INPUT_22.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  39 cW,SUc->WSU
                    offset_now       = offset_7        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((W_1-W_0), (S_1-S_0), (U_1-U_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_22_sliced.shape[0]
                    _INPUT_22_sliced_reshaped = _INPUT_22_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_22_sliced_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  40 slice _INPUT_5 with indices ['S']
                    _INPUT_5_sliced_offset = offset_1        
                    _INPUT_5_sliced  = np.ndarray((NTHC_INT, (S_1-S_0)), buffer = buffer, offset = _INPUT_5_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_5.ctypes.data),
                                 ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_5.shape[0]),
                                 ctypes.c_int(_INPUT_5.shape[1]),
                                 ctypes.c_int(S_0),
                                 ctypes.c_int(S_1))
                    # step  41 RS,WSU->RWUS
                    offset_now       = offset_8        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0), (S_1-S_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_213_0231_wob(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M11.ctypes.data),
                                                   ctypes.c_void_p(_M12.ctypes.data),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                                   ctypes.c_int(_M11.shape[0]),
                                                   ctypes.c_int(_M11.shape[2]))
                    # step  42 RWUS,RWUS->RWU
                    offset_now       = offset_7        
                    _M19_packed_offset = offset_now      
                    _M19_packed      = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0)), buffer = buffer, offset = _M19_packed_offset)
                    fn_contraction_0123_0123_012_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_void_p(_M18_perm.ctypes.data),
                                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                                     ctypes.c_int(_M12.shape[0]),
                                                     ctypes.c_int(_M12.shape[1]),
                                                     ctypes.c_int(_M12.shape[2]),
                                                     ctypes.c_int(_M12.shape[3]))
                    # step  43 pack  _M19 with indices ['W', 'U']
                    fn_packadd_3_1_2(ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                     ctypes.c_int(_M19.shape[0]),
                                     ctypes.c_int(_M19.shape[1]),
                                     ctypes.c_int(_M19.shape[2]),
                                     ctypes.c_int(W_0),
                                     ctypes.c_int(W_1),
                                     ctypes.c_int(U_0),
                                     ctypes.c_int(U_1))
                # step  44 end   for loop with indices ('V', 'W', 'U', 'S')
                # step  45 deallocate ['_M8']
            # step  46 end   for loop with indices ('V', 'W', 'U')
            # step  47 deallocate ['_M7']
        # step  48 end   for loop with indices ('V', 'W')
    # step  49 end   for loop with indices ('V',)
    # step  50 deallocate ['_M14']
    # step  51 RWU,RWU->
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(output_tmp))
    _M21 = output_tmp.value
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_2_forloop_P_R_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        P_bunchsize = 8,
                                                        R_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    # assign the size of each tensor
    _M11_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M18_size        = (NTHC_INT * (W_bunchsize * (P_bunchsize * V_bunchsize)))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M7_sliced_size  = (P_bunchsize * (V_bunchsize * NOCC))
    _INPUT_2_sliced_size = (NVIR * NTHC_INT)
    _INPUT_18_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_0_sliced_size = (NTHC_INT * NTHC_INT)
    _M14_sliced_size = (NTHC_INT * (R_bunchsize * V_bunchsize))
    _M15_perm_size   = (P_bunchsize * (NTHC_INT * (V_bunchsize * R_bunchsize)))
    _M17_size        = (W_bunchsize * (P_bunchsize * (R_bunchsize * (V_bunchsize * NVIR))))
    _M20_perm_size   = (NTHC_INT * (W_bunchsize * (P_bunchsize * (R_bunchsize * V_bunchsize))))
    _M12_size        = (V_bunchsize * (W_bunchsize * NVIR))
    _M13_size        = (W_bunchsize * (P_bunchsize * (V_bunchsize * NOCC)))
    _INPUT_7_sliced_size = (NVIR * NTHC_INT)
    _M9_size         = (V_bunchsize * (P_bunchsize * R_bunchsize))
    _M15_size        = (P_bunchsize * (NTHC_INT * (V_bunchsize * R_bunchsize)))
    _M16_size        = (NVIR * (P_bunchsize * (V_bunchsize * R_bunchsize)))
    _M20_size        = (NTHC_INT * (W_bunchsize * (P_bunchsize * (R_bunchsize * V_bunchsize))))
    _M6_sliced_size  = (NTHC_INT * (R_bunchsize * W_bunchsize))
    _M8_size         = (P_bunchsize * (R_bunchsize * NVIR))
    _M10_size        = (NTHC_INT * (V_bunchsize * (R_bunchsize * P_bunchsize)))
    _M19_size        = (R_bunchsize * (P_bunchsize * (V_bunchsize * (NTHC_INT * W_bunchsize))))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M11_size)
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M5_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M14_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M7_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M3_size)
    bucked_3_size    = max(bucked_3_size, _INPUT_17_sliced_size)
    bucked_3_size    = max(bucked_3_size, _INPUT_19_sliced_size)
    bucked_3_size    = max(bucked_3_size, _M18_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M7_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_2_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_18_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_0_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M14_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M15_perm_size)
    bucked_4_size    = max(bucked_4_size, _M17_size)
    bucked_4_size    = max(bucked_4_size, _M20_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M12_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M13_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_7_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M9_size)
    bucked_6_size    = max(bucked_6_size, _M15_size)
    bucked_6_size    = max(bucked_6_size, _M16_size)
    bucked_6_size    = max(bucked_6_size, _M20_size)
    bucked_6_size    = max(bucked_6_size, _M6_sliced_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M8_size)
    bucked_7_size    = max(bucked_7_size, _M10_size)
    bucked_7_size    = max(bucked_7_size, _M19_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    return output

def RMP3_CX_2_forloop_P_R_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("jR,jV->RVj"    , _INPUT_6        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("jQ,RVj->QRV"   , _INPUT_3        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("USW,USW->USW"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("RS,UWS->RUW"   , _INPUT_5        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TU,RWU->TRW"   , _INPUT_10       , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("iP,iV->PVi"    , _INPUT_1        , _INPUT_15       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iW,PVi->WPVi"  , _INPUT_19       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("iT,WPVi->TWPV" , _INPUT_11       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("bP,bR->PRb"    , _INPUT_2        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("bV,PRb->VPR"   , _INPUT_18       , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("PQ,VPR->QVRP"  , _INPUT_0        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("QVRP,QRV->PQVR", _M10            , _M14            )
    del _M10        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 2, 3, 1)    )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("aQ,PVRQ->aPVR" , _INPUT_4        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("VWa,aPVR->WPRVa", _M12            , _M16            )
    del _M12        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("aT,WPRVa->TWPRV", _INPUT_12       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (3, 2, 4, 0, 1) )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TRW,TWPV->RPVTW", _M6             , _M18            )
    del _M6         
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("RPVTW,RPVTW->" , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_CX_2_forloop_P_R(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 jR,jV->RVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jQ,RVj->QRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M11_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 USW,USW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 USW->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RS,UWS->RUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RUW->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TU,RWU->TRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 iP,iV->PVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iW,PVi->WPVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                   ctypes.c_void_p(_M7.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_19.shape[0]),
                                   ctypes.c_int(_INPUT_19.shape[1]),
                                   ctypes.c_int(_M7.shape[0]),
                                   ctypes.c_int(_M7.shape[1]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 iT,WPVi->TWPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M13_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 bP,bR->PRb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 bV,PRb->VPR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_18.shape[0]
    _INPUT_18_reshaped = _INPUT_18.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_18_reshaped.T, _M8_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8         
    del _M8_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 PQ,VPR->QVRP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_203_1230_wob = getattr(libpbc, "fn_contraction_01_203_1230_wob", None)
    assert fn_contraction_01_203_1230_wob is not None
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_203_1230_wob(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_0.shape[0]),
                                   ctypes.c_int(_INPUT_0.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[2]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 QVRP,QRV->PQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M10.shape[0]),
                                     ctypes.c_int(_M10.shape[1]),
                                     ctypes.c_int(_M10.shape[2]),
                                     ctypes.c_int(_M10.shape[3]))
    del _M10        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 PQVR->PVRQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                 ctypes.c_int(_M15.shape[0]),
                                 ctypes.c_int(_M15.shape[1]),
                                 ctypes.c_int(_M15.shape[2]),
                                 ctypes.c_int(_M15.shape[3]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 aQ,PVRQ->aPVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 VWa,aPVR->WPRVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_2304_13402_wob = getattr(libpbc, "fn_contraction_012_2304_13402_wob", None)
    assert fn_contraction_012_2304_13402_wob is not None
    _M17             = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_012_2304_13402_wob(ctypes.c_void_p(_M12.ctypes.data),
                                      ctypes.c_void_p(_M16.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_M12.shape[0]),
                                      ctypes.c_int(_M12.shape[1]),
                                      ctypes.c_int(_M12.shape[2]),
                                      ctypes.c_int(_M16.shape[1]),
                                      ctypes.c_int(_M16.shape[3]))
    del _M12        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 aT,WPRVa->TWPRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _size_dim_1      = _size_dim_1 * _M17.shape[3]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M17_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M17        
    del _M17_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 TWPRV->RPVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_32401_wob = getattr(libpbc, "fn_permutation_01234_32401_wob", None)
    assert fn_permutation_01234_32401_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_32401_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 TRW,TWPV->RPVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0234_13402_wob = getattr(libpbc, "fn_contraction_012_0234_13402_wob", None)
    assert fn_contraction_012_0234_13402_wob is not None
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0234_13402_wob(ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_void_p(_M19.ctypes.data),
                                      ctypes.c_int(_M6.shape[0]),
                                      ctypes.c_int(_M6.shape[1]),
                                      ctypes.c_int(_M6.shape[2]),
                                      ctypes.c_int(_M18.shape[2]),
                                      ctypes.c_int(_M18.shape[3]))
    del _M6         
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 RPVTW,RPVTW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_CX_2_forloop_P_R_forloop_P_R(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      P_bunchsize = 8,
                                      R_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_contraction_01_203_1230_wob = getattr(libpbc, "fn_contraction_01_203_1230_wob", None)
    assert fn_contraction_01_203_1230_wob is not None
    fn_contraction_012_2304_13402_wob = getattr(libpbc, "fn_contraction_012_2304_13402_wob", None)
    assert fn_contraction_012_2304_13402_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_contraction_012_0234_13402_wob = getattr(libpbc, "fn_contraction_012_0234_13402_wob", None)
    assert fn_contraction_012_0234_13402_wob is not None
    fn_permutation_01234_32401_wob = getattr(libpbc, "fn_permutation_01234_32401_wob", None)
    assert fn_permutation_01234_32401_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        P_begin = rank*bunchsize
        P_end = (rank+1)*bunchsize
        P_begin          = min(P_begin, NTHC_INT)
        P_end            = min(P_end, NTHC_INT)
    else:
        P_begin          = 0               
        P_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_2_forloop_P_R_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           P_bunchsize = P_bunchsize,
                                                                           R_bunchsize = R_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 jR,jV->RVj
    offset_now       = offset_0        
    _M11_offset      = offset_now      
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   3 jQ,RVj->QRV
    offset_now       = offset_1        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M11_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step   4 cS,cW->SWc
    offset_now       = offset_0        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   5 cU,SWc->USW
    offset_now       = offset_2        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   6 kS,kW->SWk
    offset_now       = offset_0        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   7 kU,SWk->USW
    offset_now       = offset_3        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   8 USW,USW->USW
    offset_now       = offset_0        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   9 USW->UWS
    _M4_perm_offset  = offset_2        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  10 RS,UWS->RUW
    offset_now       = offset_0        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  11 RUW->RWU
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  12 TU,RWU->TRW
    offset_now       = offset_0        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  13 iP,iV->PVi
    offset_now       = offset_2        
    _M7_offset       = offset_now      
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    # step  14 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  15 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  16 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_3        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  17 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  18 aV,aW->VWa
            offset_now       = offset_5        
            _M12_offset      = offset_now      
            _M12             = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M12_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M12.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  19 start for loop with indices ('V', 'W', 'P')
            for P_0, P_1 in lib.prange(P_begin,P_end,P_bunchsize):
                # step  20 slice _INPUT_19 with indices ['W']
                _INPUT_19_sliced_offset = offset_3        
                _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                             ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_19.shape[0]),
                             ctypes.c_int(_INPUT_19.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  21 slice _M7 with indices ['P', 'V']
                _M7_sliced_offset = offset_4        
                _M7_sliced       = np.ndarray(((P_1-P_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M7_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M7.ctypes.data),
                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                               ctypes.c_int(_M7.shape[0]),
                               ctypes.c_int(_M7.shape[1]),
                               ctypes.c_int(_M7.shape[2]),
                               ctypes.c_int(P_0),
                               ctypes.c_int(P_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  22 iW,PVi->WPVi
                offset_now       = offset_6        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (P_1-P_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_19_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_19_sliced.shape[1]),
                                               ctypes.c_int(_M7_sliced.shape[0]),
                                               ctypes.c_int(_M7_sliced.shape[1]))
                # step  23 iT,WPVi->TWPV
                offset_now       = offset_3        
                _M18_offset      = offset_now      
                _M18             = np.ndarray((NTHC_INT, (W_1-W_0), (P_1-P_0), (V_1-V_0)), buffer = buffer, offset = _M18_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
                _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M18.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M18.shape[0]
                _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_11_reshaped.T, _M13_reshaped.T, c=_M18_reshaped)
                _M18             = _M18_reshaped.reshape(*shape_backup)
                # step  24 start for loop with indices ('V', 'W', 'P', 'R')
                for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
                    # step  25 slice _INPUT_2 with indices ['P']
                    _INPUT_2_sliced_offset = offset_4        
                    _INPUT_2_sliced  = np.ndarray((NVIR, (P_1-P_0)), buffer = buffer, offset = _INPUT_2_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
                    # step  26 slice _INPUT_7 with indices ['R']
                    _INPUT_7_sliced_offset = offset_6        
                    _INPUT_7_sliced  = np.ndarray((NVIR, (R_1-R_0)), buffer = buffer, offset = _INPUT_7_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(R_0),
                                 ctypes.c_int(R_1))
                    # step  27 bP,bR->PRb
                    offset_now       = offset_7        
                    _M8_offset       = offset_now      
                    _M8              = np.ndarray(((P_1-P_0), (R_1-R_0), NVIR), buffer = buffer, offset = _M8_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M8.ctypes.data),
                                                 ctypes.c_int(_INPUT_2_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_2_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_7_sliced.shape[1]))
                    # step  28 slice _INPUT_18 with indices ['V']
                    _INPUT_18_sliced_offset = offset_4        
                    _INPUT_18_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_18_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  29 bV,PRb->VPR
                    offset_now       = offset_6        
                    _M9_offset       = offset_now      
                    _M9              = np.ndarray(((V_1-V_0), (P_1-P_0), (R_1-R_0)), buffer = buffer, offset = _M9_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_18_sliced.shape[0]
                    _INPUT_18_sliced_reshaped = _INPUT_18_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M8.shape[0]
                    _size_dim_1      = _size_dim_1 * _M8.shape[1]
                    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M9.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M9.shape[0]
                    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_18_sliced_reshaped.T, _M8_reshaped.T, c=_M9_reshaped)
                    _M9              = _M9_reshaped.reshape(*shape_backup)
                    # step  30 slice _INPUT_0 with indices ['P']
                    _INPUT_0_sliced_offset = offset_4        
                    _INPUT_0_sliced  = np.ndarray(((P_1-P_0), NTHC_INT), buffer = buffer, offset = _INPUT_0_sliced_offset)
                    fn_slice_2_0(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                 ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_0.shape[0]),
                                 ctypes.c_int(_INPUT_0.shape[1]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
                    # step  31 PQ,VPR->QVRP
                    offset_now       = offset_7        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray((NTHC_INT, (V_1-V_0), (R_1-R_0), (P_1-P_0)), buffer = buffer, offset = _M10_offset)
                    fn_contraction_01_203_1230_wob(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M9.ctypes.data),
                                                   ctypes.c_void_p(_M10.ctypes.data),
                                                   ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                                   ctypes.c_int(_M9.shape[0]),
                                                   ctypes.c_int(_M9.shape[2]))
                    # step  32 slice _M14 with indices ['R', 'V']
                    _M14_sliced_offset = offset_4        
                    _M14_sliced      = np.ndarray((NTHC_INT, (R_1-R_0), (V_1-V_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(R_0),
                                   ctypes.c_int(R_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  33 QVRP,QRV->PQVR
                    offset_now       = offset_6        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((P_1-P_0), NTHC_INT, (V_1-V_0), (R_1-R_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                                     ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M15.ctypes.data),
                                                     ctypes.c_int(_M10.shape[0]),
                                                     ctypes.c_int(_M10.shape[1]),
                                                     ctypes.c_int(_M10.shape[2]),
                                                     ctypes.c_int(_M10.shape[3]))
                    # step  34 PQVR->PVRQ
                    _M15_perm_offset = offset_4        
                    _M15_perm        = np.ndarray(((P_1-P_0), (V_1-V_0), (R_1-R_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                                 ctypes.c_int((P_1-P_0)),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((V_1-V_0)),
                                                 ctypes.c_int((R_1-R_0)))
                    # step  35 aQ,PVRQ->aPVR
                    offset_now       = offset_6        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NVIR, (P_1-P_0), (V_1-V_0), (R_1-R_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
                    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_4_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  36 VWa,aPVR->WPRVa
                    offset_now       = offset_4        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((W_1-W_0), (P_1-P_0), (R_1-R_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M17_offset)
                    fn_contraction_012_2304_13402_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                      ctypes.c_void_p(_M16.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_M12.shape[0]),
                                                      ctypes.c_int(_M12.shape[1]),
                                                      ctypes.c_int(_M12.shape[2]),
                                                      ctypes.c_int(_M16.shape[1]),
                                                      ctypes.c_int(_M16.shape[3]))
                    # step  37 aT,WPRVa->TWPRV
                    offset_now       = offset_6        
                    _M20_offset      = offset_now      
                    _M20             = np.ndarray((NTHC_INT, (W_1-W_0), (P_1-P_0), (R_1-R_0), (V_1-V_0)), buffer = buffer, offset = _M20_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
                    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17.shape[3]
                    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M20.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M20.shape[0]
                    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_12_reshaped.T, _M17_reshaped.T, c=_M20_reshaped)
                    _M20             = _M20_reshaped.reshape(*shape_backup)
                    # step  38 TWPRV->RPVTW
                    _M20_perm_offset = offset_4        
                    _M20_perm        = np.ndarray(((R_1-R_0), (P_1-P_0), (V_1-V_0), NTHC_INT, (W_1-W_0)), buffer = buffer, offset = _M20_perm_offset)
                    fn_permutation_01234_32401_wob(ctypes.c_void_p(_M20.ctypes.data),
                                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((R_1-R_0)),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  39 slice _M6 with indices ['R', 'W']
                    _M6_sliced_offset = offset_6        
                    _M6_sliced       = np.ndarray((NTHC_INT, (R_1-R_0), (W_1-W_0)), buffer = buffer, offset = _M6_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M6_sliced.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]),
                                   ctypes.c_int(R_0),
                                   ctypes.c_int(R_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  40 TRW,TWPV->RPVTW
                    offset_now       = offset_7        
                    _M19_offset      = offset_now      
                    _M19             = np.ndarray(((R_1-R_0), (P_1-P_0), (V_1-V_0), NTHC_INT, (W_1-W_0)), buffer = buffer, offset = _M19_offset)
                    fn_contraction_012_0234_13402_wob(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M18.ctypes.data),
                                                      ctypes.c_void_p(_M19.ctypes.data),
                                                      ctypes.c_int(_M6_sliced.shape[0]),
                                                      ctypes.c_int(_M6_sliced.shape[1]),
                                                      ctypes.c_int(_M6_sliced.shape[2]),
                                                      ctypes.c_int(_M18.shape[2]),
                                                      ctypes.c_int(_M18.shape[3]))
                    # step  41 RPVTW,RPVTW->
                    output_tmp       = ctypes.c_double(0.0)
                    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                           ctypes.c_void_p(_M20_perm.ctypes.data),
                           ctypes.c_int(_M19.size),
                           ctypes.pointer(output_tmp))
                    output_tmp = output_tmp.value
                    _M21 += output_tmp
                # step  42 end   for loop with indices ('V', 'W', 'P', 'R')
            # step  43 end   for loop with indices ('V', 'W', 'P')
        # step  44 end   for loop with indices ('V', 'W')
    # step  45 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_2_forloop_P_T_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        T_bunchsize = 8,
                                                        P_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M8_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M19_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M14_size        = (NTHC_INT * (W_bunchsize * (T_bunchsize * V_bunchsize)))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M9_sliced_size  = (T_bunchsize * (V_bunchsize * NVIR))
    _INPUT_1_sliced_size = (NOCC * NTHC_INT)
    _M11_size        = (P_bunchsize * (T_bunchsize * (V_bunchsize * W_bunchsize)))
    _M15_size        = (P_bunchsize * (NTHC_INT * (T_bunchsize * (V_bunchsize * W_bunchsize))))
    _M16_size        = (NOCC * (P_bunchsize * (T_bunchsize * (V_bunchsize * W_bunchsize))))
    _M17_perm_size   = (P_bunchsize * (T_bunchsize * (W_bunchsize * (NOCC * V_bunchsize))))
    _M18_perm_size   = (NTHC_INT * (P_bunchsize * (T_bunchsize * (W_bunchsize * V_bunchsize))))
    _M10_size        = (V_bunchsize * (W_bunchsize * NOCC))
    _M13_size        = (W_bunchsize * (T_bunchsize * (V_bunchsize * NVIR)))
    _INPUT_11_sliced_size = (NOCC * NTHC_INT)
    _INPUT_0_sliced_size = (NTHC_INT * NTHC_INT)
    _M15_perm_size   = (P_bunchsize * (NTHC_INT * (T_bunchsize * (V_bunchsize * W_bunchsize))))
    _INPUT_16_sliced_size = (NOCC * N_LAPLACE)
    _M18_size        = (NTHC_INT * (P_bunchsize * (T_bunchsize * (W_bunchsize * V_bunchsize))))
    _M6_perm_sliced_size = (T_bunchsize * (NTHC_INT * W_bunchsize))
    _M7_size         = (P_bunchsize * (T_bunchsize * NOCC))
    _M12_size        = (NTHC_INT * (T_bunchsize * (V_bunchsize * (W_bunchsize * P_bunchsize))))
    _M17_size        = (P_bunchsize * (T_bunchsize * (W_bunchsize * (NOCC * V_bunchsize))))
    _M19_packed_size = (NTHC_INT * (P_bunchsize * V_bunchsize))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M20_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M0_size)
    bucked_2_size    = max(bucked_2_size, _M2_size)
    bucked_2_size    = max(bucked_2_size, _M4_size)
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M6_size)
    bucked_2_size    = max(bucked_2_size, _M9_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M1_size)
    bucked_3_size    = max(bucked_3_size, _M4_perm_size)
    bucked_3_size    = max(bucked_3_size, _M5_perm_size)
    bucked_3_size    = max(bucked_3_size, _M6_perm_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _M3_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_15_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M14_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _INPUT_19_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M9_sliced_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_1_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M11_size)
    bucked_5_size    = max(bucked_5_size, _M15_size)
    bucked_5_size    = max(bucked_5_size, _M16_size)
    bucked_5_size    = max(bucked_5_size, _M17_perm_size)
    bucked_5_size    = max(bucked_5_size, _M18_perm_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M10_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M13_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_11_sliced_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_0_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M15_perm_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_16_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M18_size)
    bucked_7_size    = max(bucked_7_size, _M6_perm_sliced_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M7_size)
    bucked_8_size    = max(bucked_8_size, _M12_size)
    bucked_8_size    = max(bucked_8_size, _M17_size)
    bucked_8_size    = max(bucked_8_size, _M19_packed_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_2_forloop_P_T_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("bP,bV->PVb"    , _INPUT_2        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("bR,PVb->RPV"   , _INPUT_7        , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("USW,USW->USW"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("RS,UWS->RUW"   , _INPUT_5        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TU,RWU->TRW"   , _INPUT_10       , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (1, 0, 2)       )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("aT,aV->TVa"    , _INPUT_12       , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("aW,TVa->WTVa"  , _INPUT_21       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("aQ,WTVa->QWTV" , _INPUT_4        , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("iP,iT->PTi"    , _INPUT_1        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("PTi,VWi->PTVW" , _M7             , _M10            )
    del _M7         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("PQ,PTVW->QTVWP", _INPUT_0        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("QTVWP,QWTV->PQTVW", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 2, 3, 4, 1) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("jQ,PTVWQ->jPTVW", _INPUT_3        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("jV,jPTVW->PTWjV", _INPUT_16       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (0, 1, 2, 4, 3) )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("jR,PTWVj->RPTWV", _INPUT_6        , _M17_perm       )
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 1, 4, 2, 3) )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("RTW,RPVTW->RPV", _M6_perm        , _M18_perm       )
    del _M6_perm    
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("RPV,RPV->"     , _M19            , _M20            )
    del _M19        
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_CX_2_forloop_P_T(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 bP,bV->PVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bR,PVb->RPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M8_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M8         
    del _M8_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 USW,USW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 USW->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RS,UWS->RUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RUW->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TU,RWU->TRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TRW->RTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_102_wob = getattr(libpbc, "fn_permutation_012_102_wob", None)
    assert fn_permutation_012_102_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_102_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aT,aV->TVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 aW,TVa->WTVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_21.shape[0]),
                                   ctypes.c_int(_INPUT_21.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 aQ,WTVa->QWTV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_11.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 PTi,VWi->PTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_M7_reshaped, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M7         
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 PQ,PTVW->QTVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0234_12340_wob = getattr(libpbc, "fn_contraction_01_0234_12340_wob", None)
    assert fn_contraction_01_0234_12340_wob is not None
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_0234_12340_wob(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                     ctypes.c_void_p(_M11.ctypes.data),
                                     ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_int(_INPUT_0.shape[0]),
                                     ctypes.c_int(_INPUT_0.shape[1]),
                                     ctypes.c_int(_M11.shape[1]),
                                     ctypes.c_int(_M11.shape[2]),
                                     ctypes.c_int(_M11.shape[3]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 QTVWP,QWTV->PQTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_0312_40123_wob = getattr(libpbc, "fn_contraction_01234_0312_40123_wob", None)
    assert fn_contraction_01234_0312_40123_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_0312_40123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                        ctypes.c_void_p(_M14.ctypes.data),
                                        ctypes.c_void_p(_M15.ctypes.data),
                                        ctypes.c_int(_M12.shape[0]),
                                        ctypes.c_int(_M12.shape[1]),
                                        ctypes.c_int(_M12.shape[2]),
                                        ctypes.c_int(_M12.shape[3]),
                                        ctypes.c_int(_M12.shape[4]))
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 PQTVW->PTVWQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_02341_wob = getattr(libpbc, "fn_permutation_01234_02341_wob", None)
    assert fn_permutation_01234_02341_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_02341_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 jQ,PTVWQ->jPTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 jV,jPTVW->PTWjV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02314_23401_wob = getattr(libpbc, "fn_contraction_01_02314_23401_wob", None)
    assert fn_contraction_01_02314_23401_wob is not None
    _M17             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_02314_23401_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                      ctypes.c_void_p(_M16.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_INPUT_16.shape[0]),
                                      ctypes.c_int(_INPUT_16.shape[1]),
                                      ctypes.c_int(_M16.shape[1]),
                                      ctypes.c_int(_M16.shape[2]),
                                      ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 PTWjV->PTWVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 jR,PTWVj->RPTWV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M17_perm_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M17_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 RPTWV->RPVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01423_wob = getattr(libpbc, "fn_permutation_01234_01423_wob", None)
    assert fn_permutation_01234_01423_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_01423_wob(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(_M18.shape[3]),
                                   ctypes.c_int(_M18.shape[4]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 RTW,RPVTW->RPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_03412_034_wob = getattr(libpbc, "fn_contraction_012_03412_034_wob", None)
    assert fn_contraction_012_03412_034_wob is not None
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_03412_034_wob(ctypes.c_void_p(_M6_perm.ctypes.data),
                                     ctypes.c_void_p(_M18_perm.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M6_perm.shape[0]),
                                     ctypes.c_int(_M6_perm.shape[1]),
                                     ctypes.c_int(_M6_perm.shape[2]),
                                     ctypes.c_int(_M18_perm.shape[1]),
                                     ctypes.c_int(_M18_perm.shape[2]))
    del _M6_perm    
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 RPV,RPV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_CX_2_forloop_P_T_forloop_T_P(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      T_bunchsize = 8,
                                      P_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_permutation_01234_01423_wob = getattr(libpbc, "fn_permutation_01234_01423_wob", None)
    assert fn_permutation_01234_01423_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_contraction_012_03412_034_wob = getattr(libpbc, "fn_contraction_012_03412_034_wob", None)
    assert fn_contraction_012_03412_034_wob is not None
    fn_contraction_01_0234_12340_wob = getattr(libpbc, "fn_contraction_01_0234_12340_wob", None)
    assert fn_contraction_01_0234_12340_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_permutation_012_102_wob = getattr(libpbc, "fn_permutation_012_102_wob", None)
    assert fn_permutation_012_102_wob is not None
    fn_contraction_01234_0312_40123_wob = getattr(libpbc, "fn_contraction_01234_0312_40123_wob", None)
    assert fn_contraction_01234_0312_40123_wob is not None
    fn_contraction_01_02314_23401_wob = getattr(libpbc, "fn_contraction_01_02314_23401_wob", None)
    assert fn_contraction_01_02314_23401_wob is not None
    fn_permutation_01234_02341_wob = getattr(libpbc, "fn_permutation_01234_02341_wob", None)
    assert fn_permutation_01234_02341_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        T_begin = rank*bunchsize
        T_end = (rank+1)*bunchsize
        T_begin          = min(T_begin, NTHC_INT)
        T_end            = min(T_end, NTHC_INT)
    else:
        T_begin          = 0               
        T_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_2_forloop_P_T_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           T_bunchsize = T_bunchsize,
                                                                           P_bunchsize = P_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 bP,bV->PVb
    offset_now       = offset_0        
    _M8_offset       = offset_now      
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   2 bR,PVb->RPV
    offset_now       = offset_1        
    _M20_offset      = offset_now      
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M8_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    # step   3 allocate   _M19
    offset_now       = offset_0        
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = offset_now)
    _M19_offset      = offset_now      
    _M19.ravel()[:] = 0.0
    # step   4 kS,kW->SWk
    offset_now       = offset_2        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   5 kU,SWk->USW
    offset_now       = offset_3        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   6 cS,cW->SWc
    offset_now       = offset_2        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   7 cU,SWc->USW
    offset_now       = offset_4        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   8 USW,USW->USW
    offset_now       = offset_2        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   9 USW->UWS
    _M4_perm_offset  = offset_3        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  10 RS,UWS->RUW
    offset_now       = offset_2        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  11 RUW->RWU
    _M5_perm_offset  = offset_3        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  12 TU,RWU->TRW
    offset_now       = offset_2        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  13 TRW->RTW
    _M6_perm_offset  = offset_3        
    _M6_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_perm_offset)
    fn_permutation_012_102_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  14 aT,aV->TVa
    offset_now       = offset_2        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    # step  15 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  16 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  17 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_4        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  18 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_5        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  19 iV,iW->VWi
            offset_now       = offset_6        
            _M10_offset      = offset_now      
            _M10             = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M10_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M10.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  20 start for loop with indices ('V', 'W', 'T')
            for T_0, T_1 in lib.prange(T_begin,T_end,T_bunchsize):
                # step  21 slice _INPUT_21 with indices ['W']
                _INPUT_21_sliced_offset = offset_4        
                _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                             ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_21.shape[0]),
                             ctypes.c_int(_INPUT_21.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  22 slice _M9 with indices ['T', 'V']
                _M9_sliced_offset = offset_5        
                _M9_sliced       = np.ndarray(((T_1-T_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M9_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M9_sliced.ctypes.data),
                               ctypes.c_int(_M9.shape[0]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  23 aW,TVa->WTVa
                offset_now       = offset_7        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (T_1-T_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                               ctypes.c_void_p(_M9_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_21_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_21_sliced.shape[1]),
                                               ctypes.c_int(_M9_sliced.shape[0]),
                                               ctypes.c_int(_M9_sliced.shape[1]))
                # step  24 aQ,WTVa->QWTV
                offset_now       = offset_4        
                _M14_offset      = offset_now      
                _M14             = np.ndarray((NTHC_INT, (W_1-W_0), (T_1-T_0), (V_1-V_0)), buffer = buffer, offset = _M14_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
                _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M14.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M14.shape[0]
                _M14_reshaped = _M14.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_4_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
                _M14             = _M14_reshaped.reshape(*shape_backup)
                # step  25 start for loop with indices ('V', 'W', 'T', 'P')
                for P_0, P_1 in lib.prange(0,NTHC_INT,P_bunchsize):
                    # step  26 slice _INPUT_1 with indices ['P']
                    _INPUT_1_sliced_offset = offset_5        
                    _INPUT_1_sliced  = np.ndarray((NOCC, (P_1-P_0)), buffer = buffer, offset = _INPUT_1_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
                    # step  27 slice _INPUT_11 with indices ['T']
                    _INPUT_11_sliced_offset = offset_7        
                    _INPUT_11_sliced = np.ndarray((NOCC, (T_1-T_0)), buffer = buffer, offset = _INPUT_11_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
                    # step  28 iP,iT->PTi
                    offset_now       = offset_8        
                    _M7_offset       = offset_now      
                    _M7              = np.ndarray(((P_1-P_0), (T_1-T_0), NOCC), buffer = buffer, offset = _M7_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M7.ctypes.data),
                                                 ctypes.c_int(_INPUT_1_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_1_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_11_sliced.shape[1]))
                    # step  29 PTi,VWi->PTVW
                    offset_now       = offset_5        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((P_1-P_0), (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M7.shape[0]
                    _size_dim_1      = _size_dim_1 * _M7.shape[1]
                    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _size_dim_1      = _size_dim_1 * _M11.shape[1]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_M7_reshaped, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  30 slice _INPUT_0 with indices ['P']
                    _INPUT_0_sliced_offset = offset_7        
                    _INPUT_0_sliced  = np.ndarray(((P_1-P_0), NTHC_INT), buffer = buffer, offset = _INPUT_0_sliced_offset)
                    fn_slice_2_0(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                 ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_0.shape[0]),
                                 ctypes.c_int(_INPUT_0.shape[1]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
                    # step  31 PQ,PTVW->QTVWP
                    offset_now       = offset_8        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (T_1-T_0), (V_1-V_0), (W_1-W_0), (P_1-P_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_0234_12340_wob(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M11.ctypes.data),
                                                     ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                                     ctypes.c_int(_M11.shape[1]),
                                                     ctypes.c_int(_M11.shape[2]),
                                                     ctypes.c_int(_M11.shape[3]))
                    # step  32 QTVWP,QWTV->PQTVW
                    offset_now       = offset_5        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((P_1-P_0), NTHC_INT, (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_01234_0312_40123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                        ctypes.c_void_p(_M14.ctypes.data),
                                                        ctypes.c_void_p(_M15.ctypes.data),
                                                        ctypes.c_int(_M12.shape[0]),
                                                        ctypes.c_int(_M12.shape[1]),
                                                        ctypes.c_int(_M12.shape[2]),
                                                        ctypes.c_int(_M12.shape[3]),
                                                        ctypes.c_int(_M12.shape[4]))
                    # step  33 PQTVW->PTVWQ
                    _M15_perm_offset = offset_7        
                    _M15_perm        = np.ndarray(((P_1-P_0), (T_1-T_0), (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_01234_02341_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((W_1-W_0)))
                    # step  34 jQ,PTVWQ->jPTVW
                    offset_now       = offset_5        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NOCC, (P_1-P_0), (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
                    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_3_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  35 slice _INPUT_16 with indices ['V']
                    _INPUT_16_sliced_offset = offset_7        
                    _INPUT_16_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_16_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_16.shape[0]),
                                 ctypes.c_int(_INPUT_16.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  36 jV,jPTVW->PTWjV
                    offset_now       = offset_8        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((P_1-P_0), (T_1-T_0), (W_1-W_0), NOCC, (V_1-V_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_02314_23401_wob(ctypes.c_void_p(_INPUT_16_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M16.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_INPUT_16_sliced.shape[0]),
                                                      ctypes.c_int(_INPUT_16_sliced.shape[1]),
                                                      ctypes.c_int(_M16.shape[1]),
                                                      ctypes.c_int(_M16.shape[2]),
                                                      ctypes.c_int(_M16.shape[4]))
                    # step  37 PTWjV->PTWVj
                    _M17_perm_offset = offset_5        
                    _M17_perm        = np.ndarray(((P_1-P_0), (T_1-T_0), (W_1-W_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M17_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int(NOCC),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  38 jR,PTWVj->RPTWV
                    offset_now       = offset_7        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0), (W_1-W_0), (V_1-V_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
                    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
                    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_6_reshaped.T, _M17_perm_reshaped.T, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  39 RPTWV->RPVTW
                    _M18_perm_offset = offset_5        
                    _M18_perm        = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (T_1-T_0), (W_1-W_0)), buffer = buffer, offset = _M18_perm_offset)
                    fn_permutation_01234_01423_wob(ctypes.c_void_p(_M18.ctypes.data),
                                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  40 slice _M6 with indices ['T', 'W']
                    _M6_perm_sliced_offset = offset_7        
                    _M6_perm_sliced  = np.ndarray((NTHC_INT, (T_1-T_0), (W_1-W_0)), buffer = buffer, offset = _M6_perm_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M6_perm.ctypes.data),
                                   ctypes.c_void_p(_M6_perm_sliced.ctypes.data),
                                   ctypes.c_int(_M6_perm.shape[0]),
                                   ctypes.c_int(_M6_perm.shape[1]),
                                   ctypes.c_int(_M6_perm.shape[2]),
                                   ctypes.c_int(T_0),
                                   ctypes.c_int(T_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  41 RTW,RPVTW->RPV
                    offset_now       = offset_8        
                    _M19_packed_offset = offset_now      
                    _M19_packed      = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0)), buffer = buffer, offset = _M19_packed_offset)
                    fn_contraction_012_03412_034_wob(ctypes.c_void_p(_M6_perm_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M18_perm.ctypes.data),
                                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                                     ctypes.c_int(_M6_perm_sliced.shape[0]),
                                                     ctypes.c_int(_M6_perm_sliced.shape[1]),
                                                     ctypes.c_int(_M6_perm_sliced.shape[2]),
                                                     ctypes.c_int(_M18_perm.shape[1]),
                                                     ctypes.c_int(_M18_perm.shape[2]))
                    # step  42 pack  _M19 with indices ['V', 'P']
                    fn_packadd_3_1_2(ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                     ctypes.c_int(_M19.shape[0]),
                                     ctypes.c_int(_M19.shape[1]),
                                     ctypes.c_int(_M19.shape[2]),
                                     ctypes.c_int(P_0),
                                     ctypes.c_int(P_1),
                                     ctypes.c_int(V_0),
                                     ctypes.c_int(V_1))
                # step  43 end   for loop with indices ('V', 'W', 'T', 'P')
                # step  44 deallocate ['_M14']
            # step  45 end   for loop with indices ('V', 'W', 'T')
            # step  46 deallocate ['_M10']
        # step  47 end   for loop with indices ('V', 'W')
    # step  48 end   for loop with indices ('V',)
    # step  49 deallocate ['_M9', '_M6']
    # step  50 RPV,RPV->
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(output_tmp))
    _M21 = output_tmp.value
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_2_forloop_Q_R_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        Q_bunchsize = 8,
                                                        R_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M10_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M20_size        = (NTHC_INT * (W_bunchsize * (Q_bunchsize * V_bunchsize)))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M10_sliced_size = (Q_bunchsize * (V_bunchsize * NVIR))
    _M19_size        = (NTHC_INT * (W_bunchsize * (Q_bunchsize * V_bunchsize)))
    _M12_size        = (V_bunchsize * (W_bunchsize * NOCC))
    _M13_size        = (W_bunchsize * (Q_bunchsize * (V_bunchsize * NVIR)))
    _INPUT_3_sliced_size = (NOCC * NTHC_INT)
    _INPUT_16_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_0_sliced_size = (NTHC_INT * NTHC_INT)
    _M14_sliced_size = (NTHC_INT * (R_bunchsize * V_bunchsize))
    _M15_perm_size   = (Q_bunchsize * (NTHC_INT * (V_bunchsize * R_bunchsize)))
    _M17_size        = (W_bunchsize * (Q_bunchsize * (R_bunchsize * (V_bunchsize * NOCC))))
    _M18_perm_size   = (NTHC_INT * (W_bunchsize * (Q_bunchsize * (R_bunchsize * V_bunchsize))))
    _INPUT_6_sliced_size = (NOCC * NTHC_INT)
    _M8_size         = (V_bunchsize * (Q_bunchsize * R_bunchsize))
    _M15_size        = (Q_bunchsize * (NTHC_INT * (V_bunchsize * R_bunchsize)))
    _M16_size        = (NOCC * (Q_bunchsize * (V_bunchsize * R_bunchsize)))
    _M18_size        = (NTHC_INT * (W_bunchsize * (Q_bunchsize * (R_bunchsize * V_bunchsize))))
    _M6_perm_sliced_size = (NTHC_INT * (R_bunchsize * W_bunchsize))
    _M7_size         = (Q_bunchsize * (R_bunchsize * NOCC))
    _M9_size         = (NTHC_INT * (V_bunchsize * (R_bunchsize * Q_bunchsize)))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M10_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_15_sliced_size)
    bucked_1_size    = max(bucked_1_size, _INPUT_21_sliced_size)
    bucked_1_size    = max(bucked_1_size, _M20_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M6_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M3_size)
    bucked_3_size    = max(bucked_3_size, _M14_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_19_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M10_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M19_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M12_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M13_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_3_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_16_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_0_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M14_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M15_perm_size)
    bucked_6_size    = max(bucked_6_size, _M17_size)
    bucked_6_size    = max(bucked_6_size, _M18_perm_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _INPUT_6_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M8_size)
    bucked_7_size    = max(bucked_7_size, _M15_size)
    bucked_7_size    = max(bucked_7_size, _M16_size)
    bucked_7_size    = max(bucked_7_size, _M18_size)
    bucked_7_size    = max(bucked_7_size, _M6_perm_sliced_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M7_size)
    bucked_8_size    = max(bucked_8_size, _M9_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_2_forloop_Q_R_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("aQ,aV->QVa"    , _INPUT_4        , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("USW,USW->USW"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("RS,UWS->RUW"   , _INPUT_5        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TU,RWU->TRW"   , _INPUT_10       , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 1)       )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bR,bV->RVb"    , _INPUT_7        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("bP,RVb->PRV"   , _INPUT_2        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("aW,QVa->WQVa"  , _INPUT_21       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("aT,WQVa->TWQV" , _INPUT_12       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("jQ,jR->QRj"    , _INPUT_3        , _INPUT_6        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("jV,QRj->VQR"   , _INPUT_16       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("PQ,VQR->PVRQ"  , _INPUT_0        , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("PVRQ,PRV->QPVR", _M9             , _M14            )
    del _M9         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 2, 3, 1)    )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("iP,QVRP->iQVR" , _INPUT_1        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("VWi,iQVR->WQRVi", _M12            , _M16            )
    del _M12        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("iT,WQRVi->TWQRV", _INPUT_11       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 1, 2, 4, 3) )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TWR,TWQVR->TWQV", _M6_perm        , _M18_perm       )
    del _M6_perm    
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TWQV,TWQV->"   , _M19            , _M20            )
    del _M19        
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2_forloop_Q_R(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 aQ,aV->QVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 USW,USW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 USW->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 RS,UWS->RUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RUW->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 TU,RWU->TRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TRW->TWR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 bR,bV->RVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bP,RVb->PRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M11_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 aW,QVa->WQVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_21.shape[0]),
                                   ctypes.c_int(_INPUT_21.shape[1]),
                                   ctypes.c_int(_M10.shape[0]),
                                   ctypes.c_int(_M10.shape[1]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 aT,WQVa->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 jQ,jR->QRj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_6.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 jV,QRj->VQR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_16.shape[0]
    _INPUT_16_reshaped = _INPUT_16.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_16_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 PQ,VQR->PVRQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_213_0231_wob = getattr(libpbc, "fn_contraction_01_213_0231_wob", None)
    assert fn_contraction_01_213_0231_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_213_0231_wob(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_0.shape[0]),
                                   ctypes.c_int(_INPUT_0.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[2]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 PVRQ,PRV->QPVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M9.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M9.shape[0]),
                                     ctypes.c_int(_M9.shape[1]),
                                     ctypes.c_int(_M9.shape[2]),
                                     ctypes.c_int(_M9.shape[3]))
    del _M9         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 QPVR->QVRP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                 ctypes.c_int(_M15.shape[0]),
                                 ctypes.c_int(_M15.shape[1]),
                                 ctypes.c_int(_M15.shape[2]),
                                 ctypes.c_int(_M15.shape[3]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 iP,QVRP->iQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 VWi,iQVR->WQRVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_2304_13402_wob = getattr(libpbc, "fn_contraction_012_2304_13402_wob", None)
    assert fn_contraction_012_2304_13402_wob is not None
    _M17             = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_012_2304_13402_wob(ctypes.c_void_p(_M12.ctypes.data),
                                      ctypes.c_void_p(_M16.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_M12.shape[0]),
                                      ctypes.c_int(_M12.shape[1]),
                                      ctypes.c_int(_M12.shape[2]),
                                      ctypes.c_int(_M16.shape[1]),
                                      ctypes.c_int(_M16.shape[3]))
    del _M12        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 iT,WQRVi->TWQRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _size_dim_1      = _size_dim_1 * _M17.shape[3]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M17_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17        
    del _M17_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 TWQRV->TWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(_M18.shape[3]),
                                   ctypes.c_int(_M18.shape[4]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TWR,TWQVR->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M6_perm.ctypes.data),
                                      ctypes.c_void_p(_M18_perm.ctypes.data),
                                      ctypes.c_void_p(_M19.ctypes.data),
                                      ctypes.c_int(_M6_perm.shape[0]),
                                      ctypes.c_int(_M6_perm.shape[1]),
                                      ctypes.c_int(_M6_perm.shape[2]),
                                      ctypes.c_int(_M18_perm.shape[2]),
                                      ctypes.c_int(_M18_perm.shape[3]))
    del _M6_perm    
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TWQV,TWQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2_forloop_Q_R_forloop_Q_R(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      Q_bunchsize = 8,
                                      R_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_012_01342_0134_plus_wob = getattr(libpbc, "fn_contraction_012_01342_0134_plus_wob", None)
    assert fn_contraction_012_01342_0134_plus_wob is not None
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_213_0231_wob = getattr(libpbc, "fn_contraction_01_213_0231_wob", None)
    assert fn_contraction_01_213_0231_wob is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_012_2304_13402_wob = getattr(libpbc, "fn_contraction_012_2304_13402_wob", None)
    assert fn_contraction_012_2304_13402_wob is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        Q_begin = rank*bunchsize
        Q_end = (rank+1)*bunchsize
        Q_begin          = min(Q_begin, NTHC_INT)
        Q_end            = min(Q_end, NTHC_INT)
    else:
        Q_begin          = 0               
        Q_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_2_forloop_Q_R_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           Q_bunchsize = Q_bunchsize,
                                                                           R_bunchsize = R_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 aQ,aV->QVa
    offset_now       = offset_0        
    _M10_offset      = offset_now      
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M10_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    # step   3 cS,cW->SWc
    offset_now       = offset_1        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   4 cU,SWc->USW
    offset_now       = offset_2        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   5 kS,kW->SWk
    offset_now       = offset_1        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   6 kU,SWk->USW
    offset_now       = offset_3        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   7 USW,USW->USW
    offset_now       = offset_1        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   8 USW->UWS
    _M4_perm_offset  = offset_2        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   9 RS,UWS->RUW
    offset_now       = offset_1        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  10 RUW->RWU
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  11 TU,RWU->TRW
    offset_now       = offset_1        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  12 TRW->TWR
    _M6_perm_offset  = offset_2        
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  13 bR,bV->RVb
    offset_now       = offset_1        
    _M11_offset      = offset_now      
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step  14 bP,RVb->PRV
    offset_now       = offset_3        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M11_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step  15 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  16 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  17 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_1        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  18 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_4        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  19 iV,iW->VWi
            offset_now       = offset_5        
            _M12_offset      = offset_now      
            _M12             = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M12_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M12.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  20 start for loop with indices ('V', 'W', 'Q')
            for Q_0, Q_1 in lib.prange(Q_begin,Q_end,Q_bunchsize):
                # step  21 slice _INPUT_21 with indices ['W']
                _INPUT_21_sliced_offset = offset_1        
                _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                             ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_21.shape[0]),
                             ctypes.c_int(_INPUT_21.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  22 slice _M10 with indices ['Q', 'V']
                _M10_sliced_offset = offset_4        
                _M10_sliced      = np.ndarray(((Q_1-Q_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M10_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_void_p(_M10_sliced.ctypes.data),
                               ctypes.c_int(_M10.shape[0]),
                               ctypes.c_int(_M10.shape[1]),
                               ctypes.c_int(_M10.shape[2]),
                               ctypes.c_int(Q_0),
                               ctypes.c_int(Q_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  23 aW,QVa->WQVa
                offset_now       = offset_6        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (Q_1-Q_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                               ctypes.c_void_p(_M10_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_21_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_21_sliced.shape[1]),
                                               ctypes.c_int(_M10_sliced.shape[0]),
                                               ctypes.c_int(_M10_sliced.shape[1]))
                # step  24 aT,WQVa->TWQV
                offset_now       = offset_1        
                _M20_offset      = offset_now      
                _M20             = np.ndarray((NTHC_INT, (W_1-W_0), (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = _M20_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
                _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M20.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M20.shape[0]
                _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_12_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
                _M20             = _M20_reshaped.reshape(*shape_backup)
                # step  25 allocate   _M19
                offset_now       = offset_4        
                _M19             = np.ndarray((NTHC_INT, (W_1-W_0), (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = offset_now)
                _M19_offset      = offset_now      
                _M19.ravel()[:] = 0.0
                # step  26 start for loop with indices ('V', 'W', 'Q', 'R')
                for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
                    # step  27 slice _INPUT_3 with indices ['Q']
                    _INPUT_3_sliced_offset = offset_6        
                    _INPUT_3_sliced  = np.ndarray((NOCC, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_3_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
                    # step  28 slice _INPUT_6 with indices ['R']
                    _INPUT_6_sliced_offset = offset_7        
                    _INPUT_6_sliced  = np.ndarray((NOCC, (R_1-R_0)), buffer = buffer, offset = _INPUT_6_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(R_0),
                                 ctypes.c_int(R_1))
                    # step  29 jQ,jR->QRj
                    offset_now       = offset_8        
                    _M7_offset       = offset_now      
                    _M7              = np.ndarray(((Q_1-Q_0), (R_1-R_0), NOCC), buffer = buffer, offset = _M7_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M7.ctypes.data),
                                                 ctypes.c_int(_INPUT_3_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_3_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_6_sliced.shape[1]))
                    # step  30 slice _INPUT_16 with indices ['V']
                    _INPUT_16_sliced_offset = offset_6        
                    _INPUT_16_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_16_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_16.shape[0]),
                                 ctypes.c_int(_INPUT_16.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  31 jV,QRj->VQR
                    offset_now       = offset_7        
                    _M8_offset       = offset_now      
                    _M8              = np.ndarray(((V_1-V_0), (Q_1-Q_0), (R_1-R_0)), buffer = buffer, offset = _M8_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_16_sliced.shape[0]
                    _INPUT_16_sliced_reshaped = _INPUT_16_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M7.shape[0]
                    _size_dim_1      = _size_dim_1 * _M7.shape[1]
                    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M8.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M8.shape[0]
                    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_16_sliced_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
                    _M8              = _M8_reshaped.reshape(*shape_backup)
                    # step  32 slice _INPUT_0 with indices ['Q']
                    _INPUT_0_sliced_offset = offset_6        
                    _INPUT_0_sliced  = np.ndarray((NTHC_INT, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_0_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                 ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_0.shape[0]),
                                 ctypes.c_int(_INPUT_0.shape[1]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
                    # step  33 PQ,VQR->PVRQ
                    offset_now       = offset_8        
                    _M9_offset       = offset_now      
                    _M9              = np.ndarray((NTHC_INT, (V_1-V_0), (R_1-R_0), (Q_1-Q_0)), buffer = buffer, offset = _M9_offset)
                    fn_contraction_01_213_0231_wob(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M8.ctypes.data),
                                                   ctypes.c_void_p(_M9.ctypes.data),
                                                   ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                                   ctypes.c_int(_M8.shape[0]),
                                                   ctypes.c_int(_M8.shape[2]))
                    # step  34 slice _M14 with indices ['R', 'V']
                    _M14_sliced_offset = offset_6        
                    _M14_sliced      = np.ndarray((NTHC_INT, (R_1-R_0), (V_1-V_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(R_0),
                                   ctypes.c_int(R_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  35 PVRQ,PRV->QPVR
                    offset_now       = offset_7        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((Q_1-Q_0), NTHC_INT, (V_1-V_0), (R_1-R_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M9.ctypes.data),
                                                     ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M15.ctypes.data),
                                                     ctypes.c_int(_M9.shape[0]),
                                                     ctypes.c_int(_M9.shape[1]),
                                                     ctypes.c_int(_M9.shape[2]),
                                                     ctypes.c_int(_M9.shape[3]))
                    # step  36 QPVR->QVRP
                    _M15_perm_offset = offset_6        
                    _M15_perm        = np.ndarray(((Q_1-Q_0), (V_1-V_0), (R_1-R_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                 ctypes.c_void_p(_M15_perm.ctypes.data),
                                                 ctypes.c_int((Q_1-Q_0)),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((V_1-V_0)),
                                                 ctypes.c_int((R_1-R_0)))
                    # step  37 iP,QVRP->iQVR
                    offset_now       = offset_7        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NOCC, (Q_1-Q_0), (V_1-V_0), (R_1-R_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
                    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_1_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  38 VWi,iQVR->WQRVi
                    offset_now       = offset_6        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((W_1-W_0), (Q_1-Q_0), (R_1-R_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M17_offset)
                    fn_contraction_012_2304_13402_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                      ctypes.c_void_p(_M16.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_M12.shape[0]),
                                                      ctypes.c_int(_M12.shape[1]),
                                                      ctypes.c_int(_M12.shape[2]),
                                                      ctypes.c_int(_M16.shape[1]),
                                                      ctypes.c_int(_M16.shape[3]))
                    # step  39 iT,WQRVi->TWQRV
                    offset_now       = offset_7        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (W_1-W_0), (Q_1-Q_0), (R_1-R_0), (V_1-V_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
                    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17.shape[3]
                    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_11_reshaped.T, _M17_reshaped.T, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  40 TWQRV->TWQVR
                    _M18_perm_offset = offset_6        
                    _M18_perm        = np.ndarray((NTHC_INT, (W_1-W_0), (Q_1-Q_0), (V_1-V_0), (R_1-R_0)), buffer = buffer, offset = _M18_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M18.ctypes.data),
                                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((Q_1-Q_0)),
                                                   ctypes.c_int((R_1-R_0)),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  41 slice _M6 with indices ['R', 'W']
                    _M6_perm_sliced_offset = offset_7        
                    _M6_perm_sliced  = np.ndarray((NTHC_INT, (W_1-W_0), (R_1-R_0)), buffer = buffer, offset = _M6_perm_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M6_perm.ctypes.data),
                                   ctypes.c_void_p(_M6_perm_sliced.ctypes.data),
                                   ctypes.c_int(_M6_perm.shape[0]),
                                   ctypes.c_int(_M6_perm.shape[1]),
                                   ctypes.c_int(_M6_perm.shape[2]),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1),
                                   ctypes.c_int(R_0),
                                   ctypes.c_int(R_1))
                    # step  42 TWR,TWQVR->TWQV
                    offset_now       = offset_4        
                    fn_contraction_012_01342_0134_plus_wob(ctypes.c_void_p(_M6_perm_sliced.ctypes.data),
                                                           ctypes.c_void_p(_M18_perm.ctypes.data),
                                                           ctypes.c_void_p(_M19.ctypes.data),
                                                           ctypes.c_int(_M6_perm_sliced.shape[0]),
                                                           ctypes.c_int(_M6_perm_sliced.shape[1]),
                                                           ctypes.c_int(_M6_perm_sliced.shape[2]),
                                                           ctypes.c_int(_M18_perm.shape[2]),
                                                           ctypes.c_int(_M18_perm.shape[3]))
                # step  43 end   for loop with indices ('V', 'W', 'Q', 'R')
                # step  44 TWQV,TWQV->
                output_tmp       = ctypes.c_double(0.0)
                fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                       ctypes.c_void_p(_M20.ctypes.data),
                       ctypes.c_int(_M19.size),
                       ctypes.pointer(output_tmp))
                output_tmp = output_tmp.value
                _M21 += output_tmp
            # step  45 end   for loop with indices ('V', 'W', 'Q')
        # step  46 end   for loop with indices ('V', 'W')
    # step  47 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_2_forloop_Q_T_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        T_bunchsize = 8,
                                                        Q_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M14_size        = (NTHC_INT * (W_bunchsize * (T_bunchsize * V_bunchsize)))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M9_sliced_size  = (T_bunchsize * (V_bunchsize * NOCC))
    _INPUT_4_sliced_size = (NVIR * NTHC_INT)
    _M11_size        = (Q_bunchsize * (T_bunchsize * (V_bunchsize * W_bunchsize)))
    _M15_size        = (Q_bunchsize * (NTHC_INT * (T_bunchsize * (V_bunchsize * W_bunchsize))))
    _M16_size        = (NVIR * (Q_bunchsize * (T_bunchsize * (V_bunchsize * W_bunchsize))))
    _M17_perm_size   = (Q_bunchsize * (T_bunchsize * (W_bunchsize * (NVIR * V_bunchsize))))
    _M20_perm_size   = (NTHC_INT * (Q_bunchsize * (T_bunchsize * (W_bunchsize * V_bunchsize))))
    _M10_size        = (V_bunchsize * (W_bunchsize * NVIR))
    _M13_size        = (W_bunchsize * (T_bunchsize * (V_bunchsize * NOCC)))
    _INPUT_12_sliced_size = (NVIR * NTHC_INT)
    _INPUT_0_sliced_size = (NTHC_INT * NTHC_INT)
    _M15_perm_size   = (Q_bunchsize * (NTHC_INT * (T_bunchsize * (V_bunchsize * W_bunchsize))))
    _INPUT_18_sliced_size = (NVIR * N_LAPLACE)
    _M20_size        = (NTHC_INT * (Q_bunchsize * (T_bunchsize * (W_bunchsize * V_bunchsize))))
    _M6_sliced_size  = (T_bunchsize * (NTHC_INT * W_bunchsize))
    _M8_size         = (Q_bunchsize * (T_bunchsize * NVIR))
    _M12_size        = (NTHC_INT * (T_bunchsize * (V_bunchsize * (W_bunchsize * Q_bunchsize))))
    _M17_size        = (Q_bunchsize * (T_bunchsize * (W_bunchsize * (NVIR * V_bunchsize))))
    _M18_sliced_size = (NTHC_INT * (Q_bunchsize * V_bunchsize))
    _M19_size        = (T_bunchsize * (W_bunchsize * (Q_bunchsize * (V_bunchsize * NTHC_INT))))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M9_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M7_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_17_sliced_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_19_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M14_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M3_size)
    bucked_3_size    = max(bucked_3_size, _M18_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M9_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_4_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M11_size)
    bucked_4_size    = max(bucked_4_size, _M15_size)
    bucked_4_size    = max(bucked_4_size, _M16_size)
    bucked_4_size    = max(bucked_4_size, _M17_perm_size)
    bucked_4_size    = max(bucked_4_size, _M20_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M10_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M13_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_12_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_0_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M15_perm_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_18_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M20_size)
    bucked_6_size    = max(bucked_6_size, _M6_sliced_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M8_size)
    bucked_7_size    = max(bucked_7_size, _M12_size)
    bucked_7_size    = max(bucked_7_size, _M17_size)
    bucked_7_size    = max(bucked_7_size, _M18_sliced_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M19_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_2_forloop_Q_T_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("iT,iV->TVi"    , _INPUT_11       , _INPUT_15       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("USW,USW->USW"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("RS,UWS->RUW"   , _INPUT_5        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TU,RWU->TRW"   , _INPUT_10       , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iW,TVi->WTVi"  , _INPUT_19       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("iP,WTVi->PWTV" , _INPUT_1        , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("aQ,aT->QTa"    , _INPUT_4        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("QTa,VWa->QTVW" , _M8             , _M10            )
    del _M8         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("PQ,QTVW->PTVWQ", _INPUT_0        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("PTVWQ,PWTV->QPTVW", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 2, 3, 4, 1) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("bP,QTVWP->bQTVW", _INPUT_2        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("bV,bQTVW->QTWbV", _INPUT_18       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (0, 1, 2, 4, 3) )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("bR,QTWVb->RQTWV", _INPUT_7        , _M17_perm       )
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (2, 3, 1, 4, 0) )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TRW,RQV->TWQVR", _M6             , _M18            )
    del _M6         
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TWQVR,TWQVR->" , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2_forloop_Q_T(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 iT,iV->TVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 USW,USW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 USW->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 RS,UWS->RUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RUW->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 TU,RWU->TRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M7_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iW,TVi->WTVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_19.shape[0]),
                                   ctypes.c_int(_INPUT_19.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 iP,WTVi->PWTV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 aQ,aT->QTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 QTa,VWa->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_M8_reshaped, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M8         
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 PQ,QTVW->PTVWQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_1234_02341_wob = getattr(libpbc, "fn_contraction_01_1234_02341_wob", None)
    assert fn_contraction_01_1234_02341_wob is not None
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_1234_02341_wob(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                     ctypes.c_void_p(_M11.ctypes.data),
                                     ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_int(_INPUT_0.shape[0]),
                                     ctypes.c_int(_INPUT_0.shape[1]),
                                     ctypes.c_int(_M11.shape[1]),
                                     ctypes.c_int(_M11.shape[2]),
                                     ctypes.c_int(_M11.shape[3]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 PTVWQ,PWTV->QPTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_0312_40123_wob = getattr(libpbc, "fn_contraction_01234_0312_40123_wob", None)
    assert fn_contraction_01234_0312_40123_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_0312_40123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                        ctypes.c_void_p(_M14.ctypes.data),
                                        ctypes.c_void_p(_M15.ctypes.data),
                                        ctypes.c_int(_M12.shape[0]),
                                        ctypes.c_int(_M12.shape[1]),
                                        ctypes.c_int(_M12.shape[2]),
                                        ctypes.c_int(_M12.shape[3]),
                                        ctypes.c_int(_M12.shape[4]))
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 QPTVW->QTVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_02341_wob = getattr(libpbc, "fn_permutation_01234_02341_wob", None)
    assert fn_permutation_01234_02341_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_02341_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 bP,QTVWP->bQTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 bV,bQTVW->QTWbV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02314_23401_wob = getattr(libpbc, "fn_contraction_01_02314_23401_wob", None)
    assert fn_contraction_01_02314_23401_wob is not None
    _M17             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_02314_23401_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                      ctypes.c_void_p(_M16.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_INPUT_18.shape[0]),
                                      ctypes.c_int(_INPUT_18.shape[1]),
                                      ctypes.c_int(_M16.shape[1]),
                                      ctypes.c_int(_M16.shape[2]),
                                      ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QTWbV->QTWVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 bR,QTWVb->RQTWV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M17_perm_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M17_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 RQTWV->TWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_23140_wob = getattr(libpbc, "fn_permutation_01234_23140_wob", None)
    assert fn_permutation_01234_23140_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_23140_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TRW,RQV->TWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_134_02341_wob = getattr(libpbc, "fn_contraction_012_134_02341_wob", None)
    assert fn_contraction_012_134_02341_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_012_134_02341_wob(ctypes.c_void_p(_M6.ctypes.data),
                                     ctypes.c_void_p(_M18.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M6.shape[0]),
                                     ctypes.c_int(_M6.shape[1]),
                                     ctypes.c_int(_M6.shape[2]),
                                     ctypes.c_int(_M18.shape[1]),
                                     ctypes.c_int(_M18.shape[2]))
    del _M6         
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TWQVR,TWQVR-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2_forloop_Q_T_forloop_T_Q(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      T_bunchsize = 8,
                                      Q_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_012_134_02341_wob = getattr(libpbc, "fn_contraction_012_134_02341_wob", None)
    assert fn_contraction_012_134_02341_wob is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_1234_02341_wob = getattr(libpbc, "fn_contraction_01_1234_02341_wob", None)
    assert fn_contraction_01_1234_02341_wob is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_slice_3_0_2 = getattr(libpbc, "fn_slice_3_0_2", None)
    assert fn_slice_3_0_2 is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_permutation_01234_02341_wob = getattr(libpbc, "fn_permutation_01234_02341_wob", None)
    assert fn_permutation_01234_02341_wob is not None
    fn_contraction_01234_0312_40123_wob = getattr(libpbc, "fn_contraction_01234_0312_40123_wob", None)
    assert fn_contraction_01234_0312_40123_wob is not None
    fn_contraction_01_02314_23401_wob = getattr(libpbc, "fn_contraction_01_02314_23401_wob", None)
    assert fn_contraction_01_02314_23401_wob is not None
    fn_permutation_01234_23140_wob = getattr(libpbc, "fn_permutation_01234_23140_wob", None)
    assert fn_permutation_01234_23140_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        T_begin = rank*bunchsize
        T_end = (rank+1)*bunchsize
        T_begin          = min(T_begin, NTHC_INT)
        T_end            = min(T_end, NTHC_INT)
    else:
        T_begin          = 0               
        T_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_2_forloop_Q_T_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           T_bunchsize = T_bunchsize,
                                                                           Q_bunchsize = Q_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 iT,iV->TVi
    offset_now       = offset_0        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    # step   3 cS,cW->SWc
    offset_now       = offset_1        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   4 cU,SWc->USW
    offset_now       = offset_2        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   5 kS,kW->SWk
    offset_now       = offset_1        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   6 kU,SWk->USW
    offset_now       = offset_3        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   7 USW,USW->USW
    offset_now       = offset_1        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   8 USW->UWS
    _M4_perm_offset  = offset_2        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   9 RS,UWS->RUW
    offset_now       = offset_1        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  10 RUW->RWU
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  11 TU,RWU->TRW
    offset_now       = offset_1        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  12 jQ,jV->QVj
    offset_now       = offset_2        
    _M7_offset       = offset_now      
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step  13 jR,QVj->RQV
    offset_now       = offset_3        
    _M18_offset      = offset_now      
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M7_reshaped.T, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
    # step  14 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  15 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  16 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_2        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  17 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  18 aV,aW->VWa
            offset_now       = offset_5        
            _M10_offset      = offset_now      
            _M10             = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M10_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M10.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  19 start for loop with indices ('V', 'W', 'T')
            for T_0, T_1 in lib.prange(T_begin,T_end,T_bunchsize):
                # step  20 slice _INPUT_19 with indices ['W']
                _INPUT_19_sliced_offset = offset_2        
                _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                             ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_19.shape[0]),
                             ctypes.c_int(_INPUT_19.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  21 slice _M9 with indices ['T', 'V']
                _M9_sliced_offset = offset_4        
                _M9_sliced       = np.ndarray(((T_1-T_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M9_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M9_sliced.ctypes.data),
                               ctypes.c_int(_M9.shape[0]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  22 iW,TVi->WTVi
                offset_now       = offset_6        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (T_1-T_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                               ctypes.c_void_p(_M9_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_19_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_19_sliced.shape[1]),
                                               ctypes.c_int(_M9_sliced.shape[0]),
                                               ctypes.c_int(_M9_sliced.shape[1]))
                # step  23 iP,WTVi->PWTV
                offset_now       = offset_2        
                _M14_offset      = offset_now      
                _M14             = np.ndarray((NTHC_INT, (W_1-W_0), (T_1-T_0), (V_1-V_0)), buffer = buffer, offset = _M14_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
                _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M14.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M14.shape[0]
                _M14_reshaped = _M14.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_1_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
                _M14             = _M14_reshaped.reshape(*shape_backup)
                # step  24 start for loop with indices ('V', 'W', 'T', 'Q')
                for Q_0, Q_1 in lib.prange(0,NTHC_INT,Q_bunchsize):
                    # step  25 slice _INPUT_4 with indices ['Q']
                    _INPUT_4_sliced_offset = offset_4        
                    _INPUT_4_sliced  = np.ndarray((NVIR, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_4_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
                    # step  26 slice _INPUT_12 with indices ['T']
                    _INPUT_12_sliced_offset = offset_6        
                    _INPUT_12_sliced = np.ndarray((NVIR, (T_1-T_0)), buffer = buffer, offset = _INPUT_12_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
                    # step  27 aQ,aT->QTa
                    offset_now       = offset_7        
                    _M8_offset       = offset_now      
                    _M8              = np.ndarray(((Q_1-Q_0), (T_1-T_0), NVIR), buffer = buffer, offset = _M8_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M8.ctypes.data),
                                                 ctypes.c_int(_INPUT_4_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_4_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_12_sliced.shape[1]))
                    # step  28 QTa,VWa->QTVW
                    offset_now       = offset_4        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((Q_1-Q_0), (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M8.shape[0]
                    _size_dim_1      = _size_dim_1 * _M8.shape[1]
                    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _size_dim_1      = _size_dim_1 * _M11.shape[1]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_M8_reshaped, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  29 slice _INPUT_0 with indices ['Q']
                    _INPUT_0_sliced_offset = offset_6        
                    _INPUT_0_sliced  = np.ndarray((NTHC_INT, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_0_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_0.ctypes.data),
                                 ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_0.shape[0]),
                                 ctypes.c_int(_INPUT_0.shape[1]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
                    # step  30 PQ,QTVW->PTVWQ
                    offset_now       = offset_7        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (T_1-T_0), (V_1-V_0), (W_1-W_0), (Q_1-Q_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_1234_02341_wob(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M11.ctypes.data),
                                                     ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                                     ctypes.c_int(_M11.shape[1]),
                                                     ctypes.c_int(_M11.shape[2]),
                                                     ctypes.c_int(_M11.shape[3]))
                    # step  31 PTVWQ,PWTV->QPTVW
                    offset_now       = offset_4        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((Q_1-Q_0), NTHC_INT, (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_01234_0312_40123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                        ctypes.c_void_p(_M14.ctypes.data),
                                                        ctypes.c_void_p(_M15.ctypes.data),
                                                        ctypes.c_int(_M12.shape[0]),
                                                        ctypes.c_int(_M12.shape[1]),
                                                        ctypes.c_int(_M12.shape[2]),
                                                        ctypes.c_int(_M12.shape[3]),
                                                        ctypes.c_int(_M12.shape[4]))
                    # step  32 QPTVW->QTVWP
                    _M15_perm_offset = offset_6        
                    _M15_perm        = np.ndarray(((Q_1-Q_0), (T_1-T_0), (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_01234_02341_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                                   ctypes.c_int((Q_1-Q_0)),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((W_1-W_0)))
                    # step  33 bP,QTVWP->bQTVW
                    offset_now       = offset_4        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NVIR, (Q_1-Q_0), (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
                    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_2_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  34 slice _INPUT_18 with indices ['V']
                    _INPUT_18_sliced_offset = offset_6        
                    _INPUT_18_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_18_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(V_0),
                                 ctypes.c_int(V_1))
                    # step  35 bV,bQTVW->QTWbV
                    offset_now       = offset_7        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((Q_1-Q_0), (T_1-T_0), (W_1-W_0), NVIR, (V_1-V_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_02314_23401_wob(ctypes.c_void_p(_INPUT_18_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M16.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_INPUT_18_sliced.shape[0]),
                                                      ctypes.c_int(_INPUT_18_sliced.shape[1]),
                                                      ctypes.c_int(_M16.shape[1]),
                                                      ctypes.c_int(_M16.shape[2]),
                                                      ctypes.c_int(_M16.shape[4]))
                    # step  36 QTWbV->QTWVb
                    _M17_perm_offset = offset_4        
                    _M17_perm        = np.ndarray(((Q_1-Q_0), (T_1-T_0), (W_1-W_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M17_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                                   ctypes.c_int((Q_1-Q_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int(NVIR),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  37 bR,QTWVb->RQTWV
                    offset_now       = offset_6        
                    _M20_offset      = offset_now      
                    _M20             = np.ndarray((NTHC_INT, (Q_1-Q_0), (T_1-T_0), (W_1-W_0), (V_1-V_0)), buffer = buffer, offset = _M20_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
                    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
                    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M20.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M20.shape[0]
                    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_7_reshaped.T, _M17_perm_reshaped.T, c=_M20_reshaped)
                    _M20             = _M20_reshaped.reshape(*shape_backup)
                    # step  38 RQTWV->TWQVR
                    _M20_perm_offset = offset_4        
                    _M20_perm        = np.ndarray(((T_1-T_0), (W_1-W_0), (Q_1-Q_0), (V_1-V_0), NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
                    fn_permutation_01234_23140_wob(ctypes.c_void_p(_M20.ctypes.data),
                                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((Q_1-Q_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  39 slice _M6 with indices ['T', 'W']
                    _M6_sliced_offset = offset_6        
                    _M6_sliced       = np.ndarray(((T_1-T_0), NTHC_INT, (W_1-W_0)), buffer = buffer, offset = _M6_sliced_offset)
                    fn_slice_3_0_2(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M6_sliced.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]),
                                   ctypes.c_int(T_0),
                                   ctypes.c_int(T_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  40 slice _M18 with indices ['Q', 'V']
                    _M18_sliced_offset = offset_7        
                    _M18_sliced      = np.ndarray((NTHC_INT, (Q_1-Q_0), (V_1-V_0)), buffer = buffer, offset = _M18_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_sliced.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(Q_0),
                                   ctypes.c_int(Q_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  41 TRW,RQV->TWQVR
                    offset_now       = offset_8        
                    _M19_offset      = offset_now      
                    _M19             = np.ndarray(((T_1-T_0), (W_1-W_0), (Q_1-Q_0), (V_1-V_0), NTHC_INT), buffer = buffer, offset = _M19_offset)
                    fn_contraction_012_134_02341_wob(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M18_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M19.ctypes.data),
                                                     ctypes.c_int(_M6_sliced.shape[0]),
                                                     ctypes.c_int(_M6_sliced.shape[1]),
                                                     ctypes.c_int(_M6_sliced.shape[2]),
                                                     ctypes.c_int(_M18_sliced.shape[1]),
                                                     ctypes.c_int(_M18_sliced.shape[2]))
                    # step  42 TWQVR,TWQVR->
                    output_tmp       = ctypes.c_double(0.0)
                    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                           ctypes.c_void_p(_M20_perm.ctypes.data),
                           ctypes.c_int(_M19.size),
                           ctypes.pointer(output_tmp))
                    output_tmp = output_tmp.value
                    _M21 += output_tmp
                # step  43 end   for loop with indices ('V', 'W', 'T', 'Q')
            # step  44 end   for loop with indices ('V', 'W', 'T')
        # step  45 end   for loop with indices ('V', 'W')
    # step  46 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_3_forloop_P_T_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        P_bunchsize = 8,
                                                        T_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M18_size        = (NTHC_INT * (W_bunchsize * (P_bunchsize * V_bunchsize)))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M7_sliced_size  = (P_bunchsize * (V_bunchsize * NOCC))
    _INPUT_2_sliced_size = (NVIR * NTHC_INT)
    _M11_size        = (P_bunchsize * (T_bunchsize * (V_bunchsize * W_bunchsize)))
    _M19_size        = (T_bunchsize * (NTHC_INT * (P_bunchsize * (V_bunchsize * W_bunchsize))))
    _M10_size        = (V_bunchsize * (W_bunchsize * NVIR))
    _M13_size        = (W_bunchsize * (P_bunchsize * (V_bunchsize * NOCC)))
    _INPUT_12_sliced_size = (NVIR * NTHC_INT)
    _INPUT_10_sliced_size = (NTHC_INT * NTHC_INT)
    _M6_sliced_size  = (NTHC_INT * (P_bunchsize * V_bunchsize))
    _M16_size        = (NVIR * (P_bunchsize * (V_bunchsize * (T_bunchsize * W_bunchsize))))
    _M17_perm_size   = (P_bunchsize * (V_bunchsize * (T_bunchsize * (NVIR * W_bunchsize))))
    _M20_perm_size   = (NTHC_INT * (P_bunchsize * (V_bunchsize * (T_bunchsize * W_bunchsize))))
    _M8_size         = (P_bunchsize * (T_bunchsize * NVIR))
    _M12_size        = (NTHC_INT * (P_bunchsize * (V_bunchsize * (W_bunchsize * T_bunchsize))))
    _M14_sliced_size = (NTHC_INT * (T_bunchsize * W_bunchsize))
    _INPUT_22_sliced_size = (NVIR * N_LAPLACE)
    _M20_size        = (NTHC_INT * (P_bunchsize * (V_bunchsize * (T_bunchsize * W_bunchsize))))
    _M15_size        = (P_bunchsize * (V_bunchsize * (T_bunchsize * (W_bunchsize * NTHC_INT))))
    _M17_size        = (P_bunchsize * (V_bunchsize * (T_bunchsize * (NVIR * W_bunchsize))))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M7_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M9_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_17_sliced_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_19_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M18_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M3_size)
    bucked_3_size    = max(bucked_3_size, _M14_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M7_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_2_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M11_size)
    bucked_4_size    = max(bucked_4_size, _M19_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M10_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M13_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_12_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_10_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M6_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M16_size)
    bucked_6_size    = max(bucked_6_size, _M17_perm_size)
    bucked_6_size    = max(bucked_6_size, _M20_perm_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M8_size)
    bucked_7_size    = max(bucked_7_size, _M12_size)
    bucked_7_size    = max(bucked_7_size, _M14_sliced_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_22_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M20_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M15_size)
    bucked_8_size    = max(bucked_8_size, _M17_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_3_forloop_P_T_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("iP,iV->PVi"    , _INPUT_1        , _INPUT_15       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("RQV,RQV->RQV"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PQ,RVQ->PRV"   , _INPUT_0        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("RS,PVR->SPV"   , _INPUT_5        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("kT,kW->TWk"    , _INPUT_11       , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("kS,TWk->STW"   , _INPUT_8        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iW,PVi->WPVi"  , _INPUT_19       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("iU,WPVi->UWPV" , _INPUT_13       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("aP,aT->PTa"    , _INPUT_2        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("PTa,VWa->PTVW" , _M8             , _M10            )
    del _M8         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("TU,PTVW->UPVWT", _INPUT_10       , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("UPVWT,UWPV->TUPVW", _M12            , _M18            )
    del _M12        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("SPV,STW->PVTWS", _M6             , _M14            )
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("cS,PVTWS->cPVTW", _INPUT_9        , _M15            )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("cW,cPVTW->PVTcW", _INPUT_22       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (0, 1, 2, 4, 3) )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("cU,PVTWc->UPVTW", _INPUT_14       , _M17_perm       )
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (3, 0, 1, 2, 4) )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TUPVW,TUPVW->" , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_CX_3_forloop_P_T(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 iP,iV->PVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RQV,RQV->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 RQV->RVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 PQ,RVQ->PRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PRV->PVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,PVR->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 kT,kW->TWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 kS,TWk->STW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iW,PVi->WPVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                   ctypes.c_void_p(_M7.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_19.shape[0]),
                                   ctypes.c_int(_INPUT_19.shape[1]),
                                   ctypes.c_int(_M7.shape[0]),
                                   ctypes.c_int(_M7.shape[1]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 iU,WPVi->UWPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M13_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 PTa,VWa->PTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_M8_reshaped, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M8         
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 TU,PTVW->UPVWT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2034_12340_wob = getattr(libpbc, "fn_contraction_01_2034_12340_wob", None)
    assert fn_contraction_01_2034_12340_wob is not None
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_2034_12340_wob(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                     ctypes.c_void_p(_M11.ctypes.data),
                                     ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_int(_INPUT_10.shape[0]),
                                     ctypes.c_int(_INPUT_10.shape[1]),
                                     ctypes.c_int(_M11.shape[0]),
                                     ctypes.c_int(_M11.shape[2]),
                                     ctypes.c_int(_M11.shape[3]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 UPVWT,UWPV->TUPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_0312_40123_wob = getattr(libpbc, "fn_contraction_01234_0312_40123_wob", None)
    assert fn_contraction_01234_0312_40123_wob is not None
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_0312_40123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                        ctypes.c_void_p(_M18.ctypes.data),
                                        ctypes.c_void_p(_M19.ctypes.data),
                                        ctypes.c_int(_M12.shape[0]),
                                        ctypes.c_int(_M12.shape[1]),
                                        ctypes.c_int(_M12.shape[2]),
                                        ctypes.c_int(_M12.shape[3]),
                                        ctypes.c_int(_M12.shape[4]))
    del _M12        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 SPV,STW->PVTWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_034_12340_wob = getattr(libpbc, "fn_contraction_012_034_12340_wob", None)
    assert fn_contraction_012_034_12340_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_012_034_12340_wob(ctypes.c_void_p(_M6.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M6.shape[0]),
                                     ctypes.c_int(_M6.shape[1]),
                                     ctypes.c_int(_M6.shape[2]),
                                     ctypes.c_int(_M14.shape[1]),
                                     ctypes.c_int(_M14.shape[2]))
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 cS,PVTWS->cPVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _size_dim_1      = _size_dim_1 * _M15.shape[2]
    _size_dim_1      = _size_dim_1 * _M15.shape[3]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped, _M15_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15        
    del _M15_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 cW,cPVTW->PVTcW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02341_23401_wob = getattr(libpbc, "fn_contraction_01_02341_23401_wob", None)
    assert fn_contraction_01_02341_23401_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_02341_23401_wob(ctypes.c_void_p(_INPUT_22.ctypes.data),
                                      ctypes.c_void_p(_M16.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_INPUT_22.shape[0]),
                                      ctypes.c_int(_INPUT_22.shape[1]),
                                      ctypes.c_int(_M16.shape[1]),
                                      ctypes.c_int(_M16.shape[2]),
                                      ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 PVTcW->PVTWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 cU,PVTWc->UPVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M17_perm_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M17_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 UPVTW->TUPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_30124_wob = getattr(libpbc, "fn_permutation_01234_30124_wob", None)
    assert fn_permutation_01234_30124_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_30124_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TUPVW,TUPVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_CX_3_forloop_P_T_forloop_P_T(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      P_bunchsize = 8,
                                      T_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_permutation_01234_30124_wob = getattr(libpbc, "fn_permutation_01234_30124_wob", None)
    assert fn_permutation_01234_30124_wob is not None
    fn_contraction_01_2034_12340_wob = getattr(libpbc, "fn_contraction_01_2034_12340_wob", None)
    assert fn_contraction_01_2034_12340_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_contraction_01234_0312_40123_wob = getattr(libpbc, "fn_contraction_01234_0312_40123_wob", None)
    assert fn_contraction_01234_0312_40123_wob is not None
    fn_contraction_01_02341_23401_wob = getattr(libpbc, "fn_contraction_01_02341_23401_wob", None)
    assert fn_contraction_01_02341_23401_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_contraction_012_034_12340_wob = getattr(libpbc, "fn_contraction_012_034_12340_wob", None)
    assert fn_contraction_012_034_12340_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        P_begin = rank*bunchsize
        P_end = (rank+1)*bunchsize
        P_begin          = min(P_begin, NTHC_INT)
        P_end            = min(P_end, NTHC_INT)
    else:
        P_begin          = 0               
        P_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_3_forloop_P_T_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           P_bunchsize = P_bunchsize,
                                                                           T_bunchsize = T_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 iP,iV->PVi
    offset_now       = offset_0        
    _M7_offset       = offset_now      
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    # step   3 bQ,bV->QVb
    offset_now       = offset_1        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   4 bR,QVb->RQV
    offset_now       = offset_2        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   5 jQ,jV->QVj
    offset_now       = offset_1        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   6 jR,QVj->RQV
    offset_now       = offset_3        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   7 RQV,RQV->RQV
    offset_now       = offset_1        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   8 RQV->RVQ
    _M4_perm_offset  = offset_2        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   9 PQ,RVQ->PRV
    offset_now       = offset_1        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  10 PRV->PVR
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  11 RS,PVR->SPV
    offset_now       = offset_1        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  12 kT,kW->TWk
    offset_now       = offset_2        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step  13 kS,TWk->STW
    offset_now       = offset_3        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step  14 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  15 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  16 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_2        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  17 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  18 aV,aW->VWa
            offset_now       = offset_5        
            _M10_offset      = offset_now      
            _M10             = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M10_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M10.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  19 start for loop with indices ('V', 'W', 'P')
            for P_0, P_1 in lib.prange(P_begin,P_end,P_bunchsize):
                # step  20 slice _INPUT_19 with indices ['W']
                _INPUT_19_sliced_offset = offset_2        
                _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                             ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_19.shape[0]),
                             ctypes.c_int(_INPUT_19.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  21 slice _M7 with indices ['P', 'V']
                _M7_sliced_offset = offset_4        
                _M7_sliced       = np.ndarray(((P_1-P_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M7_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M7.ctypes.data),
                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                               ctypes.c_int(_M7.shape[0]),
                               ctypes.c_int(_M7.shape[1]),
                               ctypes.c_int(_M7.shape[2]),
                               ctypes.c_int(P_0),
                               ctypes.c_int(P_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  22 iW,PVi->WPVi
                offset_now       = offset_6        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (P_1-P_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_19_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_19_sliced.shape[1]),
                                               ctypes.c_int(_M7_sliced.shape[0]),
                                               ctypes.c_int(_M7_sliced.shape[1]))
                # step  23 iU,WPVi->UWPV
                offset_now       = offset_2        
                _M18_offset      = offset_now      
                _M18             = np.ndarray((NTHC_INT, (W_1-W_0), (P_1-P_0), (V_1-V_0)), buffer = buffer, offset = _M18_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
                _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M18.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M18.shape[0]
                _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_13_reshaped.T, _M13_reshaped.T, c=_M18_reshaped)
                _M18             = _M18_reshaped.reshape(*shape_backup)
                # step  24 start for loop with indices ('V', 'W', 'P', 'T')
                for T_0, T_1 in lib.prange(0,NTHC_INT,T_bunchsize):
                    # step  25 slice _INPUT_2 with indices ['P']
                    _INPUT_2_sliced_offset = offset_4        
                    _INPUT_2_sliced  = np.ndarray((NVIR, (P_1-P_0)), buffer = buffer, offset = _INPUT_2_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
                    # step  26 slice _INPUT_12 with indices ['T']
                    _INPUT_12_sliced_offset = offset_6        
                    _INPUT_12_sliced = np.ndarray((NVIR, (T_1-T_0)), buffer = buffer, offset = _INPUT_12_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
                    # step  27 aP,aT->PTa
                    offset_now       = offset_7        
                    _M8_offset       = offset_now      
                    _M8              = np.ndarray(((P_1-P_0), (T_1-T_0), NVIR), buffer = buffer, offset = _M8_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M8.ctypes.data),
                                                 ctypes.c_int(_INPUT_2_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_2_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_12_sliced.shape[1]))
                    # step  28 PTa,VWa->PTVW
                    offset_now       = offset_4        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((P_1-P_0), (T_1-T_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M8.shape[0]
                    _size_dim_1      = _size_dim_1 * _M8.shape[1]
                    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _size_dim_1      = _size_dim_1 * _M11.shape[1]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_M8_reshaped, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  29 slice _INPUT_10 with indices ['T']
                    _INPUT_10_sliced_offset = offset_6        
                    _INPUT_10_sliced = np.ndarray(((T_1-T_0), NTHC_INT), buffer = buffer, offset = _INPUT_10_sliced_offset)
                    fn_slice_2_0(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                 ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_10.shape[0]),
                                 ctypes.c_int(_INPUT_10.shape[1]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
                    # step  30 TU,PTVW->UPVWT
                    offset_now       = offset_7        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0), (T_1-T_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_2034_12340_wob(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M11.ctypes.data),
                                                     ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                                     ctypes.c_int(_M11.shape[0]),
                                                     ctypes.c_int(_M11.shape[2]),
                                                     ctypes.c_int(_M11.shape[3]))
                    # step  31 UPVWT,UWPV->TUPVW
                    offset_now       = offset_4        
                    _M19_offset      = offset_now      
                    _M19             = np.ndarray(((T_1-T_0), NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M19_offset)
                    fn_contraction_01234_0312_40123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                        ctypes.c_void_p(_M18.ctypes.data),
                                                        ctypes.c_void_p(_M19.ctypes.data),
                                                        ctypes.c_int(_M12.shape[0]),
                                                        ctypes.c_int(_M12.shape[1]),
                                                        ctypes.c_int(_M12.shape[2]),
                                                        ctypes.c_int(_M12.shape[3]),
                                                        ctypes.c_int(_M12.shape[4]))
                    # step  32 slice _M6 with indices ['P', 'V']
                    _M6_sliced_offset = offset_6        
                    _M6_sliced       = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0)), buffer = buffer, offset = _M6_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M6_sliced.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]),
                                   ctypes.c_int(P_0),
                                   ctypes.c_int(P_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  33 slice _M14 with indices ['T', 'W']
                    _M14_sliced_offset = offset_7        
                    _M14_sliced      = np.ndarray((NTHC_INT, (T_1-T_0), (W_1-W_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(T_0),
                                   ctypes.c_int(T_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  34 SPV,STW->PVTWS
                    offset_now       = offset_8        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((P_1-P_0), (V_1-V_0), (T_1-T_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M15_offset)
                    fn_contraction_012_034_12340_wob(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M15.ctypes.data),
                                                     ctypes.c_int(_M6_sliced.shape[0]),
                                                     ctypes.c_int(_M6_sliced.shape[1]),
                                                     ctypes.c_int(_M6_sliced.shape[2]),
                                                     ctypes.c_int(_M14_sliced.shape[1]),
                                                     ctypes.c_int(_M14_sliced.shape[2]))
                    # step  35 cS,PVTWS->cPVTW
                    offset_now       = offset_6        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NVIR, (P_1-P_0), (V_1-V_0), (T_1-T_0), (W_1-W_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
                    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15.shape[3]
                    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_9_reshaped, _M15_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  36 slice _INPUT_22 with indices ['W']
                    _INPUT_22_sliced_offset = offset_7        
                    _INPUT_22_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_22_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_22.shape[0]),
                                 ctypes.c_int(_INPUT_22.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  37 cW,cPVTW->PVTcW
                    offset_now       = offset_8        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((P_1-P_0), (V_1-V_0), (T_1-T_0), NVIR, (W_1-W_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_02341_23401_wob(ctypes.c_void_p(_INPUT_22_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M16.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_INPUT_22_sliced.shape[0]),
                                                      ctypes.c_int(_INPUT_22_sliced.shape[1]),
                                                      ctypes.c_int(_M16.shape[1]),
                                                      ctypes.c_int(_M16.shape[2]),
                                                      ctypes.c_int(_M16.shape[3]))
                    # step  38 PVTcW->PVTWc
                    _M17_perm_offset = offset_6        
                    _M17_perm        = np.ndarray(((P_1-P_0), (V_1-V_0), (T_1-T_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M17_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int(NVIR),
                                                   ctypes.c_int((W_1-W_0)))
                    # step  39 cU,PVTWc->UPVTW
                    offset_now       = offset_7        
                    _M20_offset      = offset_now      
                    _M20             = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (T_1-T_0), (W_1-W_0)), buffer = buffer, offset = _M20_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
                    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
                    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M20.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M20.shape[0]
                    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_14_reshaped.T, _M17_perm_reshaped.T, c=_M20_reshaped)
                    _M20             = _M20_reshaped.reshape(*shape_backup)
                    # step  40 UPVTW->TUPVW
                    _M20_perm_offset = offset_6        
                    _M20_perm        = np.ndarray(((T_1-T_0), NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M20_perm_offset)
                    fn_permutation_01234_30124_wob(ctypes.c_void_p(_M20.ctypes.data),
                                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((W_1-W_0)))
                    # step  41 TUPVW,TUPVW->
                    output_tmp       = ctypes.c_double(0.0)
                    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                           ctypes.c_void_p(_M20_perm.ctypes.data),
                           ctypes.c_int(_M19.size),
                           ctypes.pointer(output_tmp))
                    output_tmp = output_tmp.value
                    _M21 += output_tmp
                # step  42 end   for loop with indices ('V', 'W', 'P', 'T')
            # step  43 end   for loop with indices ('V', 'W', 'P')
        # step  44 end   for loop with indices ('V', 'W')
    # step  45 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_3_forloop_P_U_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        P_bunchsize = 8,
                                                        U_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    bucked_9_size    = 0               
    # assign the size of each tensor
    _M8_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M20_size        = (NTHC_INT * (W_bunchsize * (P_bunchsize * V_bunchsize)))
    _M6_sliced_size  = (NTHC_INT * (P_bunchsize * V_bunchsize))
    _M16_size        = (NOCC * (P_bunchsize * (V_bunchsize * (U_bunchsize * W_bunchsize))))
    _M17_perm_size   = (P_bunchsize * (V_bunchsize * (U_bunchsize * (NOCC * W_bunchsize))))
    _M18_perm_size   = (NTHC_INT * (P_bunchsize * (V_bunchsize * (U_bunchsize * W_bunchsize))))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M8_sliced_size  = (P_bunchsize * (V_bunchsize * NVIR))
    _M19_size        = (NTHC_INT * (P_bunchsize * (V_bunchsize * W_bunchsize)))
    _M10_size        = (V_bunchsize * (W_bunchsize * NOCC))
    _M13_size        = (W_bunchsize * (P_bunchsize * (V_bunchsize * NVIR)))
    _M20_perm_size   = (NTHC_INT * (W_bunchsize * (P_bunchsize * V_bunchsize)))
    _M14_sliced_size = (NTHC_INT * (U_bunchsize * W_bunchsize))
    _INPUT_20_sliced_size = (NOCC * N_LAPLACE)
    _M18_size        = (NTHC_INT * (P_bunchsize * (V_bunchsize * (U_bunchsize * W_bunchsize))))
    _INPUT_1_sliced_size = (NOCC * NTHC_INT)
    _M11_size        = (P_bunchsize * (U_bunchsize * (V_bunchsize * W_bunchsize)))
    _M15_size        = (P_bunchsize * (V_bunchsize * (U_bunchsize * (W_bunchsize * NTHC_INT))))
    _M17_size        = (P_bunchsize * (V_bunchsize * (U_bunchsize * (NOCC * W_bunchsize))))
    _INPUT_13_sliced_size = (NOCC * NTHC_INT)
    _INPUT_10_sliced_size = (NTHC_INT * NTHC_INT)
    _M7_size         = (P_bunchsize * (U_bunchsize * NOCC))
    _M12_size        = (NTHC_INT * (P_bunchsize * (V_bunchsize * (W_bunchsize * U_bunchsize))))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M8_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M9_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_15_sliced_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_21_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M20_size)
    bucked_2_size    = max(bucked_2_size, _M6_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M16_size)
    bucked_2_size    = max(bucked_2_size, _M17_perm_size)
    bucked_2_size    = max(bucked_2_size, _M18_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M3_size)
    bucked_3_size    = max(bucked_3_size, _M14_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_19_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M8_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M19_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M10_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M13_size)
    bucked_6_size    = max(bucked_6_size, _M20_perm_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M14_sliced_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_20_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M18_size)
    bucked_7_size    = max(bucked_7_size, _INPUT_1_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M11_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M15_size)
    bucked_8_size    = max(bucked_8_size, _M17_size)
    bucked_8_size    = max(bucked_8_size, _INPUT_13_sliced_size)
    bucked_8_size    = max(bucked_8_size, _INPUT_10_sliced_size)
    # bucket 9
    bucked_9_size    = max(bucked_9_size, _M7_size)
    bucked_9_size    = max(bucked_9_size, _M12_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    output.append(bucked_9_size)
    return output

def RMP3_CX_3_forloop_P_U_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("aP,aV->PVa"    , _INPUT_2        , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("RQV,RQV->RQV"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PQ,RVQ->PRV"   , _INPUT_0        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("RS,PVR->SPV"   , _INPUT_5        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("cU,cW->UWc"    , _INPUT_14       , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("cS,UWc->SUW"   , _INPUT_9        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("aW,PVa->WPVa"  , _INPUT_21       , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("aT,WPVa->TWPV" , _INPUT_12       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 3, 1)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("SPV,SUW->PVUWS", _M6             , _M14            )
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("kS,PVUWS->kPVUW", _INPUT_8        , _M15            )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("kW,kPVUW->PVUkW", _INPUT_20       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (0, 1, 2, 4, 3) )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("kT,PVUWk->TPVUW", _INPUT_11       , _M17_perm       )
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 1, 2, 4, 3) )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("iP,iU->PUi"    , _INPUT_1        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("PUi,VWi->PUVW" , _M7             , _M10            )
    del _M7         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("TU,PUVW->TPVWU", _INPUT_10       , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TPVWU,TPVWU->TPVW", _M12            , _M18_perm       )
    del _M12        
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TPVW,TPVW->"   , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_3_forloop_P_U(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 aP,aV->PVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RQV,RQV->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 RQV->RVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 PQ,RVQ->PRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PRV->PVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,PVR->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 cU,cW->UWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 cS,UWc->SUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M10             = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 aW,PVa->WPVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_21.shape[0]),
                                   ctypes.c_int(_INPUT_21.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 aT,WPVa->TWPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 TWPV->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 SPV,SUW->PVUWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_034_12340_wob = getattr(libpbc, "fn_contraction_012_034_12340_wob", None)
    assert fn_contraction_012_034_12340_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_012_034_12340_wob(ctypes.c_void_p(_M6.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M6.shape[0]),
                                     ctypes.c_int(_M6.shape[1]),
                                     ctypes.c_int(_M6.shape[2]),
                                     ctypes.c_int(_M14.shape[1]),
                                     ctypes.c_int(_M14.shape[2]))
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 kS,PVUWS->kPVUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _size_dim_1      = _size_dim_1 * _M15.shape[2]
    _size_dim_1      = _size_dim_1 * _M15.shape[3]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped, _M15_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15        
    del _M15_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 kW,kPVUW->PVUkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02341_23401_wob = getattr(libpbc, "fn_contraction_01_02341_23401_wob", None)
    assert fn_contraction_01_02341_23401_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_02341_23401_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                      ctypes.c_void_p(_M16.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_INPUT_20.shape[0]),
                                      ctypes.c_int(_INPUT_20.shape[1]),
                                      ctypes.c_int(_M16.shape[1]),
                                      ctypes.c_int(_M16.shape[2]),
                                      ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 PVUkW->PVUWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 kT,PVUWk->TPVUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M17_perm_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M17_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 TPVUW->TPVWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(_M18.shape[3]),
                                   ctypes.c_int(_M18.shape[4]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 iP,iU->PUi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_13.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 PUi,VWi->PUVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_M7_reshaped, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M7         
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 TU,PUVW->TPVWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2134_02341_wob = getattr(libpbc, "fn_contraction_01_2134_02341_wob", None)
    assert fn_contraction_01_2134_02341_wob is not None
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_2134_02341_wob(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                     ctypes.c_void_p(_M11.ctypes.data),
                                     ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_int(_INPUT_10.shape[0]),
                                     ctypes.c_int(_INPUT_10.shape[1]),
                                     ctypes.c_int(_M11.shape[0]),
                                     ctypes.c_int(_M11.shape[2]),
                                     ctypes.c_int(_M11.shape[3]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TPVWU,TPVWU->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_0123_wob = getattr(libpbc, "fn_contraction_01234_01234_0123_wob", None)
    assert fn_contraction_01234_01234_0123_wob is not None
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_01234_0123_wob(ctypes.c_void_p(_M12.ctypes.data),
                                        ctypes.c_void_p(_M18_perm.ctypes.data),
                                        ctypes.c_void_p(_M19.ctypes.data),
                                        ctypes.c_int(_M12.shape[0]),
                                        ctypes.c_int(_M12.shape[1]),
                                        ctypes.c_int(_M12.shape[2]),
                                        ctypes.c_int(_M12.shape[3]),
                                        ctypes.c_int(_M12.shape[4]))
    del _M12        
    del _M18_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TPVW,TPVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_3_forloop_P_U_forloop_P_U(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      P_bunchsize = 8,
                                      U_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_contraction_01_2134_02341_wob = getattr(libpbc, "fn_contraction_01_2134_02341_wob", None)
    assert fn_contraction_01_2134_02341_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_contraction_01234_01234_0123_plus_wob = getattr(libpbc, "fn_contraction_01234_01234_0123_plus_wob", None)
    assert fn_contraction_01234_01234_0123_plus_wob is not None
    fn_contraction_01_02341_23401_wob = getattr(libpbc, "fn_contraction_01_02341_23401_wob", None)
    assert fn_contraction_01_02341_23401_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_contraction_012_034_12340_wob = getattr(libpbc, "fn_contraction_012_034_12340_wob", None)
    assert fn_contraction_012_034_12340_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        P_begin = rank*bunchsize
        P_end = (rank+1)*bunchsize
        P_begin          = min(P_begin, NTHC_INT)
        P_end            = min(P_end, NTHC_INT)
    else:
        P_begin          = 0               
        P_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_3_forloop_P_U_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           P_bunchsize = P_bunchsize,
                                                                           U_bunchsize = U_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    offset_9         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[9])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 aP,aV->PVa
    offset_now       = offset_0        
    _M8_offset       = offset_now      
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    # step   3 bQ,bV->QVb
    offset_now       = offset_1        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   4 bR,QVb->RQV
    offset_now       = offset_2        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   5 jQ,jV->QVj
    offset_now       = offset_1        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   6 jR,QVj->RQV
    offset_now       = offset_3        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   7 RQV,RQV->RQV
    offset_now       = offset_1        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   8 RQV->RVQ
    _M4_perm_offset  = offset_2        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   9 PQ,RVQ->PRV
    offset_now       = offset_1        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  10 PRV->PVR
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  11 RS,PVR->SPV
    offset_now       = offset_1        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  12 cU,cW->UWc
    offset_now       = offset_2        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step  13 cS,UWc->SUW
    offset_now       = offset_3        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M9_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    # step  14 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  15 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  16 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_2        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  17 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_4        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  18 iV,iW->VWi
            offset_now       = offset_5        
            _M10_offset      = offset_now      
            _M10             = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M10_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M10.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  19 start for loop with indices ('V', 'W', 'P')
            for P_0, P_1 in lib.prange(P_begin,P_end,P_bunchsize):
                # step  20 slice _INPUT_21 with indices ['W']
                _INPUT_21_sliced_offset = offset_2        
                _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                             ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_21.shape[0]),
                             ctypes.c_int(_INPUT_21.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  21 slice _M8 with indices ['P', 'V']
                _M8_sliced_offset = offset_4        
                _M8_sliced       = np.ndarray(((P_1-P_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M8_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M8.ctypes.data),
                               ctypes.c_void_p(_M8_sliced.ctypes.data),
                               ctypes.c_int(_M8.shape[0]),
                               ctypes.c_int(_M8.shape[1]),
                               ctypes.c_int(_M8.shape[2]),
                               ctypes.c_int(P_0),
                               ctypes.c_int(P_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  22 aW,PVa->WPVa
                offset_now       = offset_6        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (P_1-P_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                               ctypes.c_void_p(_M8_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_21_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_21_sliced.shape[1]),
                                               ctypes.c_int(_M8_sliced.shape[0]),
                                               ctypes.c_int(_M8_sliced.shape[1]))
                # step  23 aT,WPVa->TWPV
                offset_now       = offset_2        
                _M20_offset      = offset_now      
                _M20             = np.ndarray((NTHC_INT, (W_1-W_0), (P_1-P_0), (V_1-V_0)), buffer = buffer, offset = _M20_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
                _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M20.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M20.shape[0]
                _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_12_reshaped.T, _M13_reshaped.T, c=_M20_reshaped)
                _M20             = _M20_reshaped.reshape(*shape_backup)
                # step  24 allocate   _M19
                offset_now       = offset_4        
                _M19             = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = offset_now)
                _M19_offset      = offset_now      
                _M19.ravel()[:] = 0.0
                # step  25 TWPV->TPVW
                _M20_perm_offset = offset_6        
                _M20_perm        = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M20_perm_offset)
                fn_permutation_0123_0231_wob(ctypes.c_void_p(_M20.ctypes.data),
                                             ctypes.c_void_p(_M20_perm.ctypes.data),
                                             ctypes.c_int(NTHC_INT),
                                             ctypes.c_int((W_1-W_0)),
                                             ctypes.c_int((P_1-P_0)),
                                             ctypes.c_int((V_1-V_0)))
                # step  26 start for loop with indices ('V', 'W', 'P', 'U')
                for U_0, U_1 in lib.prange(0,NTHC_INT,U_bunchsize):
                    # step  27 slice _M6 with indices ['P', 'V']
                    _M6_sliced_offset = offset_2        
                    _M6_sliced       = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0)), buffer = buffer, offset = _M6_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M6_sliced.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]),
                                   ctypes.c_int(P_0),
                                   ctypes.c_int(P_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  28 slice _M14 with indices ['U', 'W']
                    _M14_sliced_offset = offset_7        
                    _M14_sliced      = np.ndarray((NTHC_INT, (U_1-U_0), (W_1-W_0)), buffer = buffer, offset = _M14_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_sliced.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(U_0),
                                   ctypes.c_int(U_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  29 SPV,SUW->PVUWS
                    offset_now       = offset_8        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((P_1-P_0), (V_1-V_0), (U_1-U_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M15_offset)
                    fn_contraction_012_034_12340_wob(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M14_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M15.ctypes.data),
                                                     ctypes.c_int(_M6_sliced.shape[0]),
                                                     ctypes.c_int(_M6_sliced.shape[1]),
                                                     ctypes.c_int(_M6_sliced.shape[2]),
                                                     ctypes.c_int(_M14_sliced.shape[1]),
                                                     ctypes.c_int(_M14_sliced.shape[2]))
                    # step  30 kS,PVUWS->kPVUW
                    offset_now       = offset_2        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NOCC, (P_1-P_0), (V_1-V_0), (U_1-U_0), (W_1-W_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
                    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15.shape[3]
                    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_8_reshaped, _M15_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  31 slice _INPUT_20 with indices ['W']
                    _INPUT_20_sliced_offset = offset_7        
                    _INPUT_20_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_20_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_20.shape[0]),
                                 ctypes.c_int(_INPUT_20.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  32 kW,kPVUW->PVUkW
                    offset_now       = offset_8        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((P_1-P_0), (V_1-V_0), (U_1-U_0), NOCC, (W_1-W_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_01_02341_23401_wob(ctypes.c_void_p(_INPUT_20_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M16.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_INPUT_20_sliced.shape[0]),
                                                      ctypes.c_int(_INPUT_20_sliced.shape[1]),
                                                      ctypes.c_int(_M16.shape[1]),
                                                      ctypes.c_int(_M16.shape[2]),
                                                      ctypes.c_int(_M16.shape[3]))
                    # step  33 PVUkW->PVUWk
                    _M17_perm_offset = offset_2        
                    _M17_perm        = np.ndarray(((P_1-P_0), (V_1-V_0), (U_1-U_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M17_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M17.ctypes.data),
                                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((U_1-U_0)),
                                                   ctypes.c_int(NOCC),
                                                   ctypes.c_int((W_1-W_0)))
                    # step  34 kT,PVUWk->TPVUW
                    offset_now       = offset_7        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (U_1-U_0), (W_1-W_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
                    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[3]
                    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_11_reshaped.T, _M17_perm_reshaped.T, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  35 TPVUW->TPVWU
                    _M18_perm_offset = offset_2        
                    _M18_perm        = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0), (U_1-U_0)), buffer = buffer, offset = _M18_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M18.ctypes.data),
                                                   ctypes.c_void_p(_M18_perm.ctypes.data),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((P_1-P_0)),
                                                   ctypes.c_int((V_1-V_0)),
                                                   ctypes.c_int((U_1-U_0)),
                                                   ctypes.c_int((W_1-W_0)))
                    # step  36 slice _INPUT_1 with indices ['P']
                    _INPUT_1_sliced_offset = offset_7        
                    _INPUT_1_sliced  = np.ndarray((NOCC, (P_1-P_0)), buffer = buffer, offset = _INPUT_1_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
                    # step  37 slice _INPUT_13 with indices ['U']
                    _INPUT_13_sliced_offset = offset_8        
                    _INPUT_13_sliced = np.ndarray((NOCC, (U_1-U_0)), buffer = buffer, offset = _INPUT_13_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(U_0),
                                 ctypes.c_int(U_1))
                    # step  38 iP,iU->PUi
                    offset_now       = offset_9        
                    _M7_offset       = offset_now      
                    _M7              = np.ndarray(((P_1-P_0), (U_1-U_0), NOCC), buffer = buffer, offset = _M7_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M7.ctypes.data),
                                                 ctypes.c_int(_INPUT_1_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_1_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_13_sliced.shape[1]))
                    # step  39 PUi,VWi->PUVW
                    offset_now       = offset_7        
                    _M11_offset      = offset_now      
                    _M11             = np.ndarray(((P_1-P_0), (U_1-U_0), (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M11_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M7.shape[0]
                    _size_dim_1      = _size_dim_1 * _M7.shape[1]
                    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M10.shape[0]
                    _size_dim_1      = _size_dim_1 * _M10.shape[1]
                    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M11.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M11.shape[0]
                    _size_dim_1      = _size_dim_1 * _M11.shape[1]
                    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
                    lib.ddot(_M7_reshaped, _M10_reshaped.T, c=_M11_reshaped)
                    _M11             = _M11_reshaped.reshape(*shape_backup)
                    # step  40 slice _INPUT_10 with indices ['U']
                    _INPUT_10_sliced_offset = offset_8        
                    _INPUT_10_sliced = np.ndarray((NTHC_INT, (U_1-U_0)), buffer = buffer, offset = _INPUT_10_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                 ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_10.shape[0]),
                                 ctypes.c_int(_INPUT_10.shape[1]),
                                 ctypes.c_int(U_0),
                                 ctypes.c_int(U_1))
                    # step  41 TU,PUVW->TPVWU
                    offset_now       = offset_9        
                    _M12_offset      = offset_now      
                    _M12             = np.ndarray((NTHC_INT, (P_1-P_0), (V_1-V_0), (W_1-W_0), (U_1-U_0)), buffer = buffer, offset = _M12_offset)
                    fn_contraction_01_2134_02341_wob(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M11.ctypes.data),
                                                     ctypes.c_void_p(_M12.ctypes.data),
                                                     ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                                     ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                                     ctypes.c_int(_M11.shape[0]),
                                                     ctypes.c_int(_M11.shape[2]),
                                                     ctypes.c_int(_M11.shape[3]))
                    # step  42 TPVWU,TPVWU->TPVW
                    offset_now       = offset_4        
                    fn_contraction_01234_01234_0123_plus_wob(ctypes.c_void_p(_M12.ctypes.data),
                                                             ctypes.c_void_p(_M18_perm.ctypes.data),
                                                             ctypes.c_void_p(_M19.ctypes.data),
                                                             ctypes.c_int(_M12.shape[0]),
                                                             ctypes.c_int(_M12.shape[1]),
                                                             ctypes.c_int(_M12.shape[2]),
                                                             ctypes.c_int(_M12.shape[3]),
                                                             ctypes.c_int(_M12.shape[4]))
                # step  43 end   for loop with indices ('V', 'W', 'P', 'U')
                # step  44 TPVW,TPVW->
                output_tmp       = ctypes.c_double(0.0)
                fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                       ctypes.c_void_p(_M20_perm.ctypes.data),
                       ctypes.c_int(_M19.size),
                       ctypes.pointer(output_tmp))
                output_tmp = output_tmp.value
                _M21 += output_tmp
            # step  45 end   for loop with indices ('V', 'W', 'P')
        # step  46 end   for loop with indices ('V', 'W')
    # step  47 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_3_forloop_S_T_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        T_bunchsize = 8,
                                                        S_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    bucked_9_size    = 0               
    # assign the size of each tensor
    _M9_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _M12_perm_size   = (V_bunchsize * (W_bunchsize * NOCC))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M14_size        = (NTHC_INT * (W_bunchsize * (T_bunchsize * V_bunchsize)))
    _M12_size        = (V_bunchsize * (W_bunchsize * NOCC))
    _M11_sliced_size = (T_bunchsize * (V_bunchsize * NVIR))
    _M6_sliced_size  = (S_bunchsize * (NTHC_INT * V_bunchsize))
    _M15_perm_size   = (S_bunchsize * (W_bunchsize * (T_bunchsize * (NTHC_INT * V_bunchsize))))
    _M16_perm_size   = (NOCC * (S_bunchsize * (W_bunchsize * (T_bunchsize * V_bunchsize))))
    _M17_perm_size   = (W_bunchsize * (NOCC * (S_bunchsize * T_bunchsize)))
    _INPUT_8_sliced_size = (NOCC * NTHC_INT)
    _INPUT_20_sliced_size = (NOCC * N_LAPLACE)
    _INPUT_10_sliced_size = (NTHC_INT * NTHC_INT)
    _M19_packed_size = (NTHC_INT * (W_bunchsize * S_bunchsize))
    _M13_size        = (W_bunchsize * (T_bunchsize * (V_bunchsize * NVIR)))
    _M15_size        = (S_bunchsize * (W_bunchsize * (T_bunchsize * (NTHC_INT * V_bunchsize))))
    _M16_size        = (NOCC * (S_bunchsize * (W_bunchsize * (T_bunchsize * V_bunchsize))))
    _M17_size        = (W_bunchsize * (NOCC * (S_bunchsize * T_bunchsize)))
    _M18_size        = (NTHC_INT * (W_bunchsize * (S_bunchsize * T_bunchsize)))
    _INPUT_11_sliced_size = (NOCC * NTHC_INT)
    _M8_size         = (W_bunchsize * (S_bunchsize * T_bunchsize))
    _M7_size         = (S_bunchsize * (T_bunchsize * NOCC))
    _M10_size        = (NTHC_INT * (W_bunchsize * (S_bunchsize * T_bunchsize)))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M9_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M20_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M20_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M1_size)
    bucked_3_size    = max(bucked_3_size, _M4_perm_size)
    bucked_3_size    = max(bucked_3_size, _M5_perm_size)
    bucked_3_size    = max(bucked_3_size, _M11_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _M3_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_15_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M12_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _INPUT_19_sliced_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_21_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M14_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M12_size)
    bucked_6_size    = max(bucked_6_size, _M11_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M6_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M15_perm_size)
    bucked_6_size    = max(bucked_6_size, _M16_perm_size)
    bucked_6_size    = max(bucked_6_size, _M17_perm_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_8_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_20_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_10_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M19_packed_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _M13_size)
    bucked_7_size    = max(bucked_7_size, _M15_size)
    bucked_7_size    = max(bucked_7_size, _M16_size)
    bucked_7_size    = max(bucked_7_size, _M17_size)
    bucked_7_size    = max(bucked_7_size, _M18_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _INPUT_11_sliced_size)
    bucked_8_size    = max(bucked_8_size, _M8_size)
    # bucket 9
    bucked_9_size    = max(bucked_9_size, _M7_size)
    bucked_9_size    = max(bucked_9_size, _M10_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    output.append(bucked_9_size)
    return output

def RMP3_CX_3_forloop_S_T_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("RQV,RQV->RQV"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PQ,RVQ->PRV"   , _INPUT_0        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("RS,PVR->SPV"   , _INPUT_5        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aT,aV->TVa"    , _INPUT_12       , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12_perm        = np.transpose(_M12            , (1, 2, 0)       )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("aW,TVa->WTVa"  , _INPUT_21       , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("aP,WTVa->PWTV" , _INPUT_2        , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("SPV,PWTV->SWTPV", _M6             , _M14            )
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 1, 2, 4, 3) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("iP,SWTVP->iSWTV", _INPUT_1        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (2, 0, 1, 3, 4) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("WiV,WiSTV->WiST", _M12_perm       , _M16_perm       )
    del _M12_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (0, 2, 3, 1)    )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("iU,WSTi->UWST" , _INPUT_13       , _M17_perm       )
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("kS,kT->STk"    , _INPUT_8        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("kW,STk->WST"   , _INPUT_20       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("TU,WST->UWST"  , _INPUT_10       , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("UWST,UWST->UWS", _M10            , _M18            )
    del _M10        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("UWS,UWS->"     , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_CX_3_forloop_S_T(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M9_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 USW->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 RQV,RQV->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RQV->RVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 PQ,RVQ->PRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 PRV->PVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 RS,PVR->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aT,aV->TVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 VWi->WiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M12_perm        = np.ndarray((N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M12.ctypes.data),
                               ctypes.c_void_p(_M12_perm.ctypes.data),
                               ctypes.c_int(_M12.shape[0]),
                               ctypes.c_int(_M12.shape[1]),
                               ctypes.c_int(_M12.shape[2]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 aW,TVa->WTVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_21.shape[0]),
                                   ctypes.c_int(_INPUT_21.shape[1]),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 aP,WTVa->PWTV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 SPV,PWTV->SWTPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_1342_03412_wob = getattr(libpbc, "fn_contraction_012_1342_03412_wob", None)
    assert fn_contraction_012_1342_03412_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_1342_03412_wob(ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_M6.shape[0]),
                                      ctypes.c_int(_M6.shape[1]),
                                      ctypes.c_int(_M6.shape[2]),
                                      ctypes.c_int(_M14.shape[1]),
                                      ctypes.c_int(_M14.shape[2]))
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 SWTPV->SWTVP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 iP,SWTVP->iSWTV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 iSWTV->WiSTV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_20134_wob = getattr(libpbc, "fn_permutation_01234_20134_wob", None)
    assert fn_permutation_01234_20134_wob is not None
    _M16_perm        = np.ndarray((N_LAPLACE, NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_20134_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 WiV,WiSTV->WiST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M17             = np.ndarray((N_LAPLACE, NOCC, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M12_perm.ctypes.data),
                                      ctypes.c_void_p(_M16_perm.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_M12_perm.shape[0]),
                                      ctypes.c_int(_M12_perm.shape[1]),
                                      ctypes.c_int(_M12_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[3]))
    del _M12_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 WiST->WSTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M17_perm        = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M17.ctypes.data),
                                 ctypes.c_void_p(_M17_perm.ctypes.data),
                                 ctypes.c_int(_M17.shape[0]),
                                 ctypes.c_int(_M17.shape[1]),
                                 ctypes.c_int(_M17.shape[2]),
                                 ctypes.c_int(_M17.shape[3]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 iU,WSTi->UWST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M17_perm_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M17_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 kS,kT->STk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_11.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 kW,STk->WST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_20.shape[0]
    _INPUT_20_reshaped = _INPUT_20.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_20_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TU,WST->UWST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_10.shape[0]),
                                   ctypes.c_int(_INPUT_10.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 UWST,UWST->UWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_012_wob = getattr(libpbc, "fn_contraction_0123_0123_012_wob", None)
    assert fn_contraction_0123_0123_012_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_0123_012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                     ctypes.c_void_p(_M18.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M10.shape[0]),
                                     ctypes.c_int(_M10.shape[1]),
                                     ctypes.c_int(_M10.shape[2]),
                                     ctypes.c_int(_M10.shape[3]))
    del _M10        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    # step 28 UWS,UWS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_CX_3_forloop_S_T_forloop_T_S(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      T_bunchsize = 8,
                                      S_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_012_1342_03412_wob = getattr(libpbc, "fn_contraction_012_1342_03412_wob", None)
    assert fn_contraction_012_1342_03412_wob is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_slice_3_0_2 = getattr(libpbc, "fn_slice_3_0_2", None)
    assert fn_slice_3_0_2 is not None
    fn_permutation_01234_20134_wob = getattr(libpbc, "fn_permutation_01234_20134_wob", None)
    assert fn_permutation_01234_20134_wob is not None
    fn_contraction_0123_0123_012_wob = getattr(libpbc, "fn_contraction_0123_0123_012_wob", None)
    assert fn_contraction_0123_0123_012_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        T_begin = rank*bunchsize
        T_end = (rank+1)*bunchsize
        T_begin          = min(T_begin, NTHC_INT)
        T_end            = min(T_end, NTHC_INT)
    else:
        T_begin          = 0               
        T_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_3_forloop_S_T_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           T_bunchsize = T_bunchsize,
                                                                           S_bunchsize = S_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    offset_9         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[9])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 cS,cW->SWc
    offset_now       = offset_0        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   2 cU,SWc->USW
    offset_now       = offset_1        
    _M20_offset      = offset_now      
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M9_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    # step   3 allocate   _M19
    offset_now       = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    _M19_offset      = offset_now      
    _M19.ravel()[:] = 0.0
    # step   4 USW->UWS
    _M20_perm_offset = offset_2        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   5 bQ,bV->QVb
    offset_now       = offset_1        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   6 bR,QVb->RQV
    offset_now       = offset_3        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   7 jQ,jV->QVj
    offset_now       = offset_1        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   8 jR,QVj->RQV
    offset_now       = offset_4        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   9 RQV,RQV->RQV
    offset_now       = offset_1        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step  10 RQV->RVQ
    _M4_perm_offset  = offset_3        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  11 PQ,RVQ->PRV
    offset_now       = offset_1        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step  12 PRV->PVR
    _M5_perm_offset  = offset_3        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  13 RS,PVR->SPV
    offset_now       = offset_1        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  14 aT,aV->TVa
    offset_now       = offset_3        
    _M11_offset      = offset_now      
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    # step  15 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  16 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  17 slice _INPUT_15 with indices ['V']
            _INPUT_15_sliced_offset = offset_4        
            _INPUT_15_sliced = np.ndarray((NOCC, (V_1-V_0)), buffer = buffer, offset = _INPUT_15_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_15.ctypes.data),
                         ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_15.shape[0]),
                         ctypes.c_int(_INPUT_15.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  18 slice _INPUT_19 with indices ['W']
            _INPUT_19_sliced_offset = offset_5        
            _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_19.shape[0]),
                         ctypes.c_int(_INPUT_19.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  19 iV,iW->VWi
            offset_now       = offset_6        
            _M12_offset      = offset_now      
            _M12             = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M12_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M12.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  20 VWi->WiV
            _M12_perm_offset = offset_4        
            _M12_perm        = np.ndarray(((W_1-W_0), NOCC, (V_1-V_0)), buffer = buffer, offset = _M12_perm_offset)
            fn_permutation_012_120_wob(ctypes.c_void_p(_M12.ctypes.data),
                                       ctypes.c_void_p(_M12_perm.ctypes.data),
                                       ctypes.c_int((V_1-V_0)),
                                       ctypes.c_int((W_1-W_0)),
                                       ctypes.c_int(NOCC))
            # step  21 start for loop with indices ('V', 'W', 'T')
            for T_0, T_1 in lib.prange(T_begin,T_end,T_bunchsize):
                # step  22 slice _INPUT_21 with indices ['W']
                _INPUT_21_sliced_offset = offset_5        
                _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                             ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_21.shape[0]),
                             ctypes.c_int(_INPUT_21.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  23 slice _M11 with indices ['T', 'V']
                _M11_sliced_offset = offset_6        
                _M11_sliced      = np.ndarray(((T_1-T_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M11_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M11.ctypes.data),
                               ctypes.c_void_p(_M11_sliced.ctypes.data),
                               ctypes.c_int(_M11.shape[0]),
                               ctypes.c_int(_M11.shape[1]),
                               ctypes.c_int(_M11.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  24 aW,TVa->WTVa
                offset_now       = offset_7        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (T_1-T_0), (V_1-V_0), NVIR), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                               ctypes.c_void_p(_M11_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_21_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_21_sliced.shape[1]),
                                               ctypes.c_int(_M11_sliced.shape[0]),
                                               ctypes.c_int(_M11_sliced.shape[1]))
                # step  25 aP,WTVa->PWTV
                offset_now       = offset_5        
                _M14_offset      = offset_now      
                _M14             = np.ndarray((NTHC_INT, (W_1-W_0), (T_1-T_0), (V_1-V_0)), buffer = buffer, offset = _M14_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
                _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M14.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M14.shape[0]
                _M14_reshaped = _M14.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_2_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
                _M14             = _M14_reshaped.reshape(*shape_backup)
                # step  26 start for loop with indices ('V', 'W', 'T', 'S')
                for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
                    # step  27 slice _M6 with indices ['S', 'V']
                    _M6_sliced_offset = offset_6        
                    _M6_sliced       = np.ndarray(((S_1-S_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M6_sliced_offset)
                    fn_slice_3_0_2(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M6_sliced.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]),
                                   ctypes.c_int(S_0),
                                   ctypes.c_int(S_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  28 SPV,PWTV->SWTPV
                    offset_now       = offset_7        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((S_1-S_0), (W_1-W_0), (T_1-T_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_012_1342_03412_wob(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M14.ctypes.data),
                                                      ctypes.c_void_p(_M15.ctypes.data),
                                                      ctypes.c_int(_M6_sliced.shape[0]),
                                                      ctypes.c_int(_M6_sliced.shape[1]),
                                                      ctypes.c_int(_M6_sliced.shape[2]),
                                                      ctypes.c_int(_M14.shape[1]),
                                                      ctypes.c_int(_M14.shape[2]))
                    # step  29 SWTPV->SWTVP
                    _M15_perm_offset = offset_6        
                    _M15_perm        = np.ndarray(((S_1-S_0), (W_1-W_0), (T_1-T_0), (V_1-V_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                                   ctypes.c_int((S_1-S_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  30 iP,SWTVP->iSWTV
                    offset_now       = offset_7        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NOCC, (S_1-S_0), (W_1-W_0), (T_1-T_0), (V_1-V_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
                    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_1_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  31 iSWTV->WiSTV
                    _M16_perm_offset = offset_6        
                    _M16_perm        = np.ndarray(((W_1-W_0), NOCC, (S_1-S_0), (T_1-T_0), (V_1-V_0)), buffer = buffer, offset = _M16_perm_offset)
                    fn_permutation_01234_20134_wob(ctypes.c_void_p(_M16.ctypes.data),
                                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                                   ctypes.c_int(NOCC),
                                                   ctypes.c_int((S_1-S_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((T_1-T_0)),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  32 WiV,WiSTV->WiST
                    offset_now       = offset_7        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((W_1-W_0), NOCC, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M12_perm.ctypes.data),
                                                      ctypes.c_void_p(_M16_perm.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_M12_perm.shape[0]),
                                                      ctypes.c_int(_M12_perm.shape[1]),
                                                      ctypes.c_int(_M12_perm.shape[2]),
                                                      ctypes.c_int(_M16_perm.shape[2]),
                                                      ctypes.c_int(_M16_perm.shape[3]))
                    # step  33 WiST->WSTi
                    _M17_perm_offset = offset_6        
                    _M17_perm        = np.ndarray(((W_1-W_0), (S_1-S_0), (T_1-T_0), NOCC), buffer = buffer, offset = _M17_perm_offset)
                    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M17.ctypes.data),
                                                 ctypes.c_void_p(_M17_perm.ctypes.data),
                                                 ctypes.c_int((W_1-W_0)),
                                                 ctypes.c_int(NOCC),
                                                 ctypes.c_int((S_1-S_0)),
                                                 ctypes.c_int((T_1-T_0)))
                    # step  34 iU,WSTi->UWST
                    offset_now       = offset_7        
                    _M18_offset      = offset_now      
                    _M18             = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M18_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
                    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
                    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M18.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M18.shape[0]
                    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_13_reshaped.T, _M17_perm_reshaped.T, c=_M18_reshaped)
                    _M18             = _M18_reshaped.reshape(*shape_backup)
                    # step  35 slice _INPUT_8 with indices ['S']
                    _INPUT_8_sliced_offset = offset_6        
                    _INPUT_8_sliced  = np.ndarray((NOCC, (S_1-S_0)), buffer = buffer, offset = _INPUT_8_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(S_0),
                                 ctypes.c_int(S_1))
                    # step  36 slice _INPUT_11 with indices ['T']
                    _INPUT_11_sliced_offset = offset_8        
                    _INPUT_11_sliced = np.ndarray((NOCC, (T_1-T_0)), buffer = buffer, offset = _INPUT_11_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
                    # step  37 kS,kT->STk
                    offset_now       = offset_9        
                    _M7_offset       = offset_now      
                    _M7              = np.ndarray(((S_1-S_0), (T_1-T_0), NOCC), buffer = buffer, offset = _M7_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M7.ctypes.data),
                                                 ctypes.c_int(_INPUT_8_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_8_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_11_sliced.shape[1]))
                    # step  38 slice _INPUT_20 with indices ['W']
                    _INPUT_20_sliced_offset = offset_6        
                    _INPUT_20_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_20_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_20.shape[0]),
                                 ctypes.c_int(_INPUT_20.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  39 kW,STk->WST
                    offset_now       = offset_8        
                    _M8_offset       = offset_now      
                    _M8              = np.ndarray(((W_1-W_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_20_sliced.shape[0]
                    _INPUT_20_sliced_reshaped = _INPUT_20_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M7.shape[0]
                    _size_dim_1      = _size_dim_1 * _M7.shape[1]
                    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M8.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M8.shape[0]
                    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_20_sliced_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
                    _M8              = _M8_reshaped.reshape(*shape_backup)
                    # step  40 slice _INPUT_10 with indices ['T']
                    _INPUT_10_sliced_offset = offset_6        
                    _INPUT_10_sliced = np.ndarray(((T_1-T_0), NTHC_INT), buffer = buffer, offset = _INPUT_10_sliced_offset)
                    fn_slice_2_0(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                 ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_10.shape[0]),
                                 ctypes.c_int(_INPUT_10.shape[1]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
                    # step  41 TU,WST->UWST
                    offset_now       = offset_9        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M10_offset)
                    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M8.ctypes.data),
                                                   ctypes.c_void_p(_M10.ctypes.data),
                                                   ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                                   ctypes.c_int(_M8.shape[0]),
                                                   ctypes.c_int(_M8.shape[1]))
                    # step  42 UWST,UWST->UWS
                    offset_now       = offset_6        
                    _M19_packed_offset = offset_now      
                    _M19_packed      = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0)), buffer = buffer, offset = _M19_packed_offset)
                    fn_contraction_0123_0123_012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                                     ctypes.c_void_p(_M18.ctypes.data),
                                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                                     ctypes.c_int(_M10.shape[0]),
                                                     ctypes.c_int(_M10.shape[1]),
                                                     ctypes.c_int(_M10.shape[2]),
                                                     ctypes.c_int(_M10.shape[3]))
                    # step  43 pack  _M19 with indices ['W', 'S']
                    fn_packadd_3_1_2(ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_void_p(_M19_packed.ctypes.data),
                                     ctypes.c_int(_M19.shape[0]),
                                     ctypes.c_int(_M19.shape[1]),
                                     ctypes.c_int(_M19.shape[2]),
                                     ctypes.c_int(W_0),
                                     ctypes.c_int(W_1),
                                     ctypes.c_int(S_0),
                                     ctypes.c_int(S_1))
                # step  44 end   for loop with indices ('V', 'W', 'T', 'S')
                # step  45 deallocate ['_M14']
            # step  46 end   for loop with indices ('V', 'W', 'T')
            # step  47 deallocate ['_M12']
        # step  48 end   for loop with indices ('V', 'W')
    # step  49 end   for loop with indices ('V',)
    # step  50 deallocate ['_M6', '_M11']
    # step  51 UWS,UWS->
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(output_tmp))
    _M21 = output_tmp.value
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

def RMP3_CX_3_forloop_S_U_determine_bucket_size_forloop(NVIR        : int,
                                                        NOCC        : int,
                                                        N_LAPLACE   : int,
                                                        NTHC_INT    : int,
                                                        U_bunchsize = 8,
                                                        S_bunchsize = 8,
                                                        V_bunchsize = 1,
                                                        W_bunchsize = 1):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    bucked_4_size    = 0               
    bucked_5_size    = 0               
    bucked_6_size    = 0               
    bucked_7_size    = 0               
    bucked_8_size    = 0               
    # assign the size of each tensor
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _M12_perm_size   = (V_bunchsize * (W_bunchsize * NVIR))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M14_size        = (NTHC_INT * (W_bunchsize * (U_bunchsize * V_bunchsize)))
    _M12_size        = (V_bunchsize * (W_bunchsize * NVIR))
    _M11_sliced_size = (U_bunchsize * (V_bunchsize * NOCC))
    _M6_sliced_size  = (S_bunchsize * (NTHC_INT * V_bunchsize))
    _M15_perm_size   = (S_bunchsize * (W_bunchsize * (U_bunchsize * (NTHC_INT * V_bunchsize))))
    _M16_perm_size   = (NVIR * (S_bunchsize * (W_bunchsize * (U_bunchsize * V_bunchsize))))
    _M17_perm_size   = (W_bunchsize * (NVIR * (S_bunchsize * U_bunchsize)))
    _M20_perm_size   = (NTHC_INT * (W_bunchsize * (S_bunchsize * U_bunchsize)))
    _M13_size        = (W_bunchsize * (U_bunchsize * (V_bunchsize * NOCC)))
    _M15_size        = (S_bunchsize * (W_bunchsize * (U_bunchsize * (NTHC_INT * V_bunchsize))))
    _M16_size        = (NVIR * (S_bunchsize * (W_bunchsize * (U_bunchsize * V_bunchsize))))
    _M17_size        = (W_bunchsize * (NVIR * (S_bunchsize * U_bunchsize)))
    _M20_size        = (NTHC_INT * (W_bunchsize * (S_bunchsize * U_bunchsize)))
    _INPUT_9_sliced_size = (NVIR * NTHC_INT)
    _INPUT_22_sliced_size = (NVIR * N_LAPLACE)
    _INPUT_10_sliced_size = (NTHC_INT * NTHC_INT)
    _M18_sliced_size = (NTHC_INT * (S_bunchsize * W_bunchsize))
    _INPUT_14_sliced_size = (NVIR * NTHC_INT)
    _M9_size         = (W_bunchsize * (S_bunchsize * U_bunchsize))
    _M19_size        = (U_bunchsize * (NTHC_INT * (W_bunchsize * S_bunchsize)))
    _M8_size         = (S_bunchsize * (U_bunchsize * NVIR))
    _M10_size        = (NTHC_INT * (W_bunchsize * (S_bunchsize * U_bunchsize)))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M5_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _M4_perm_size)
    bucked_1_size    = max(bucked_1_size, _M5_perm_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M7_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_17_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M12_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M18_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _INPUT_21_sliced_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_19_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M14_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M12_size)
    bucked_5_size    = max(bucked_5_size, _M11_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M6_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M15_perm_size)
    bucked_5_size    = max(bucked_5_size, _M16_perm_size)
    bucked_5_size    = max(bucked_5_size, _M17_perm_size)
    bucked_5_size    = max(bucked_5_size, _M20_perm_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M13_size)
    bucked_6_size    = max(bucked_6_size, _M15_size)
    bucked_6_size    = max(bucked_6_size, _M16_size)
    bucked_6_size    = max(bucked_6_size, _M17_size)
    bucked_6_size    = max(bucked_6_size, _M20_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_9_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_22_sliced_size)
    bucked_6_size    = max(bucked_6_size, _INPUT_10_sliced_size)
    bucked_6_size    = max(bucked_6_size, _M18_sliced_size)
    # bucket 7
    bucked_7_size    = max(bucked_7_size, _INPUT_14_sliced_size)
    bucked_7_size    = max(bucked_7_size, _M9_size)
    bucked_7_size    = max(bucked_7_size, _M19_size)
    # bucket 8
    bucked_8_size    = max(bucked_8_size, _M8_size)
    bucked_8_size    = max(bucked_8_size, _M10_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    output.append(bucked_7_size)
    output.append(bucked_8_size)
    return output

def RMP3_CX_3_forloop_S_U_naive(Z           : np.ndarray,
                                X_o         : np.ndarray,
                                X_v         : np.ndarray,
                                tau_o       : np.ndarray,
                                tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("RQV,RQV->RQV"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PQ,RVQ->PRV"   , _INPUT_0        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("RS,PVR->SPV"   , _INPUT_5        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("iU,iV->UVi"    , _INPUT_13       , _INPUT_15       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("kT,SWk->TSW"   , _INPUT_11       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12_perm        = np.transpose(_M12            , (1, 2, 0)       )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iW,UVi->WUVi"  , _INPUT_19       , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("iP,WUVi->PWUV" , _INPUT_1        , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("SPV,PWUV->SWUPV", _M6             , _M14            )
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 1, 2, 4, 3) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("aP,SWUVP->aSWUV", _INPUT_2        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (2, 0, 1, 3, 4) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("WaV,WaSUV->WaSU", _M12_perm       , _M16_perm       )
    del _M12_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (0, 2, 3, 1)    )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("aT,WSUa->TWSU" , _INPUT_12       , _M17_perm       )
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (3, 0, 1, 2)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("cS,cU->SUc"    , _INPUT_9        , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("cW,SUc->WSU"   , _INPUT_22       , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("TU,WSU->TWSU"  , _INPUT_10       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TWSU,TSW->UTWS", _M10            , _M18            )
    del _M10        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("UTWS,UTWS->"   , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_CX_3_forloop_S_U(Z           : np.ndarray,
                          X_o         : np.ndarray,
                          X_v         : np.ndarray,
                          tau_o       : np.ndarray,
                          tau_v       : np.ndarray):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 RQV,RQV->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RQV->RVQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 PQ,RVQ->PRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 PRV->PVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RS,PVR->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iU,iV->UVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 kT,SWk->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M7_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 VWa->WaV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M12_perm        = np.ndarray((N_LAPLACE, NVIR, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M12.ctypes.data),
                               ctypes.c_void_p(_M12_perm.ctypes.data),
                               ctypes.c_int(_M12.shape[0]),
                               ctypes.c_int(_M12.shape[1]),
                               ctypes.c_int(_M12.shape[2]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 iW,UVi->WUVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_19.shape[0]),
                                   ctypes.c_int(_INPUT_19.shape[1]),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 iP,WUVi->PWUV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 SPV,PWUV->SWUPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_1342_03412_wob = getattr(libpbc, "fn_contraction_012_1342_03412_wob", None)
    assert fn_contraction_012_1342_03412_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_1342_03412_wob(ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_M6.shape[0]),
                                      ctypes.c_int(_M6.shape[1]),
                                      ctypes.c_int(_M6.shape[2]),
                                      ctypes.c_int(_M14.shape[1]),
                                      ctypes.c_int(_M14.shape[2]))
    del _M6         
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 SWUPV->SWUVP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 aP,SWUVP->aSWUV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 aSWUV->WaSUV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_20134_wob = getattr(libpbc, "fn_permutation_01234_20134_wob", None)
    assert fn_permutation_01234_20134_wob is not None
    _M16_perm        = np.ndarray((N_LAPLACE, NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_20134_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 WaV,WaSUV->WaSU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M17             = np.ndarray((N_LAPLACE, NVIR, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M12_perm.ctypes.data),
                                      ctypes.c_void_p(_M16_perm.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_M12_perm.shape[0]),
                                      ctypes.c_int(_M12_perm.shape[1]),
                                      ctypes.c_int(_M12_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[3]))
    del _M12_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 WaSU->WSUa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M17_perm        = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M17.ctypes.data),
                                 ctypes.c_void_p(_M17_perm.ctypes.data),
                                 ctypes.c_int(_M17.shape[0]),
                                 ctypes.c_int(_M17.shape[1]),
                                 ctypes.c_int(_M17.shape[2]),
                                 ctypes.c_int(_M17.shape[3]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 aT,WSUa->TWSU 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M17_perm_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M17_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 TWSU->UTWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_3012_wob = getattr(libpbc, "fn_permutation_0123_3012_wob", None)
    assert fn_permutation_0123_3012_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_3012_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 cS,cU->SUc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 cW,SUc->WSU 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_22.shape[0]
    _INPUT_22_reshaped = _INPUT_22.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_22_reshaped.T, _M8_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8         
    del _M8_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TU,WSU->TWSU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_231_0231_wob = getattr(libpbc, "fn_contraction_01_231_0231_wob", None)
    assert fn_contraction_01_231_0231_wob is not None
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_231_0231_wob(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_10.shape[0]),
                                   ctypes.c_int(_INPUT_10.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 TWSU,TSW->UTWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                     ctypes.c_void_p(_M18.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M10.shape[0]),
                                     ctypes.c_int(_M10.shape[1]),
                                     ctypes.c_int(_M10.shape[2]),
                                     ctypes.c_int(_M10.shape[3]))
    del _M10        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    # step 28 UTWS,UTWS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_CX_3_forloop_S_U_forloop_U_S(Z           : np.ndarray,
                                      X_o         : np.ndarray,
                                      X_v         : np.ndarray,
                                      tau_o       : np.ndarray,
                                      tau_v       : np.ndarray,
                                      buffer      : np.ndarray,
                                      U_bunchsize = 8,
                                      S_bunchsize = 8,
                                      V_bunchsize = 1,
                                      W_bunchsize = 1,
                                      use_mpi = False):
    # assign the size of the indices
    NVIR             = X_v.shape[0]    
    NOCC             = X_o.shape[0]    
    N_LAPLACE        = tau_o.shape[1]  
    NTHC_INT         = Z.shape[0]      
    # check the input parameters
    assert NTHC_INT == Z.shape[0]
    assert NTHC_INT == Z.shape[1]
    assert NOCC == X_o.shape[0]
    assert NTHC_INT == X_o.shape[1]
    assert NVIR == X_v.shape[0]
    assert NTHC_INT == X_v.shape[1]
    assert NOCC == tau_o.shape[0]
    assert N_LAPLACE == tau_o.shape[1]
    assert NVIR == tau_v.shape[0]
    assert N_LAPLACE == tau_v.shape[1]
    # assign the input/output parameters
    _INPUT_0         = Z               
    _INPUT_1         = X_o             
    _INPUT_2         = X_v             
    _INPUT_3         = X_o             
    _INPUT_4         = X_v             
    _INPUT_5         = Z               
    _INPUT_6         = X_o             
    _INPUT_7         = X_v             
    _INPUT_8         = X_o             
    _INPUT_9         = X_v             
    _INPUT_10        = Z               
    _INPUT_11        = X_o             
    _INPUT_12        = X_v             
    _INPUT_13        = X_o             
    _INPUT_14        = X_v             
    _INPUT_15        = tau_o           
    _INPUT_16        = tau_o           
    _INPUT_17        = tau_v           
    _INPUT_18        = tau_v           
    _INPUT_19        = tau_o           
    _INPUT_20        = tau_o           
    _INPUT_21        = tau_v           
    _INPUT_22        = tau_v           
    nthreads         = lib.num_threads()
    _M21             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_0123_021_3012_wob = getattr(libpbc, "fn_contraction_0123_021_3012_wob", None)
    assert fn_contraction_0123_021_3012_wob is not None
    fn_contraction_012_1342_03412_wob = getattr(libpbc, "fn_contraction_012_1342_03412_wob", None)
    assert fn_contraction_012_1342_03412_wob is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_permutation_0123_3012_wob = getattr(libpbc, "fn_permutation_0123_3012_wob", None)
    assert fn_permutation_0123_3012_wob is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_slice_3_0_1 = getattr(libpbc, "fn_slice_3_0_1", None)
    assert fn_slice_3_0_1 is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_slice_3_0_2 = getattr(libpbc, "fn_slice_3_0_2", None)
    assert fn_slice_3_0_2 is not None
    fn_permutation_01234_20134_wob = getattr(libpbc, "fn_permutation_01234_20134_wob", None)
    assert fn_permutation_01234_20134_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    fn_contraction_01_231_0231_wob = getattr(libpbc, "fn_contraction_01_231_0231_wob", None)
    assert fn_contraction_01_231_0231_wob is not None
    if use_mpi:
        bunchsize = NTHC_INT//comm_size + 1
        U_begin = rank*bunchsize
        U_end = (rank+1)*bunchsize
        U_begin          = min(U_begin, NTHC_INT)
        U_end            = min(U_end, NTHC_INT)
    else:
        U_begin          = 0               
        U_end            = NTHC_INT        
    # preallocate buffer
    bucket_size      = RMP3_CX_3_forloop_S_U_determine_bucket_size_forloop(NVIR = NVIR,
                                                                           NOCC = NOCC,
                                                                           N_LAPLACE = N_LAPLACE,
                                                                           NTHC_INT = NTHC_INT,
                                                                           V_bunchsize = V_bunchsize,
                                                                           W_bunchsize = W_bunchsize,
                                                                           U_bunchsize = U_bunchsize,
                                                                           S_bunchsize = S_bunchsize)
    bufsize_now      = buffer.size     
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    offset_3         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[3])
    offset_4         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[4])
    offset_5         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[5])
    offset_6         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[6])
    offset_7         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[7])
    offset_8         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[8])
    bufsize          = offset_now      
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step   0 start for loop with indices ()
    # step   1 allocate   _M21
    _M21             = 0.0             
    # step   2 bQ,bV->QVb
    offset_now       = offset_0        
    _M0_offset       = offset_now      
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   3 bR,QVb->RQV
    offset_now       = offset_1        
    _M1_offset       = offset_now      
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    # step   4 jQ,jV->QVj
    offset_now       = offset_0        
    _M2_offset       = offset_now      
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step   5 jR,QVj->RQV
    offset_now       = offset_2        
    _M3_offset       = offset_now      
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    # step   6 RQV,RQV->RQV
    offset_now       = offset_0        
    _M4_offset       = offset_now      
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    # step   7 RQV->RVQ
    _M4_perm_offset  = offset_1        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step   8 PQ,RVQ->PRV
    offset_now       = offset_0        
    _M5_offset       = offset_now      
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped, _M4_perm_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    # step   9 PRV->PVR
    _M5_perm_offset  = offset_1        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE))
    # step  10 RS,PVR->SPV
    offset_now       = offset_0        
    _M6_offset       = offset_now      
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6              = _M6_reshaped.reshape(*shape_backup)
    # step  11 iU,iV->UVi
    offset_now       = offset_1        
    _M11_offset      = offset_now      
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    # step  12 kS,kW->SWk
    offset_now       = offset_2        
    _M7_offset       = offset_now      
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step  13 kT,SWk->TSW
    offset_now       = offset_3        
    _M18_offset      = offset_now      
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M7_reshaped.T, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
    # step  14 start for loop with indices ('V',)
    for V_0, V_1 in lib.prange(0,N_LAPLACE,V_bunchsize):
        # step  15 start for loop with indices ('V', 'W')
        for W_0, W_1 in lib.prange(0,N_LAPLACE,W_bunchsize):
            # step  16 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_2        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  17 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_4        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  18 aV,aW->VWa
            offset_now       = offset_5        
            _M12_offset      = offset_now      
            _M12             = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M12_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M12.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  19 VWa->WaV
            _M12_perm_offset = offset_2        
            _M12_perm        = np.ndarray(((W_1-W_0), NVIR, (V_1-V_0)), buffer = buffer, offset = _M12_perm_offset)
            fn_permutation_012_120_wob(ctypes.c_void_p(_M12.ctypes.data),
                                       ctypes.c_void_p(_M12_perm.ctypes.data),
                                       ctypes.c_int((V_1-V_0)),
                                       ctypes.c_int((W_1-W_0)),
                                       ctypes.c_int(NVIR))
            # step  20 start for loop with indices ('V', 'W', 'U')
            for U_0, U_1 in lib.prange(U_begin,U_end,U_bunchsize):
                # step  21 slice _INPUT_19 with indices ['W']
                _INPUT_19_sliced_offset = offset_4        
                _INPUT_19_sliced = np.ndarray((NOCC, (W_1-W_0)), buffer = buffer, offset = _INPUT_19_sliced_offset)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_19.ctypes.data),
                             ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_19.shape[0]),
                             ctypes.c_int(_INPUT_19.shape[1]),
                             ctypes.c_int(W_0),
                             ctypes.c_int(W_1))
                # step  22 slice _M11 with indices ['U', 'V']
                _M11_sliced_offset = offset_5        
                _M11_sliced      = np.ndarray(((U_1-U_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M11_sliced_offset)
                fn_slice_3_0_1(ctypes.c_void_p(_M11.ctypes.data),
                               ctypes.c_void_p(_M11_sliced.ctypes.data),
                               ctypes.c_int(_M11.shape[0]),
                               ctypes.c_int(_M11.shape[1]),
                               ctypes.c_int(_M11.shape[2]),
                               ctypes.c_int(U_0),
                               ctypes.c_int(U_1),
                               ctypes.c_int(V_0),
                               ctypes.c_int(V_1))
                # step  23 iW,UVi->WUVi
                offset_now       = offset_6        
                _M13_offset      = offset_now      
                _M13             = np.ndarray(((W_1-W_0), (U_1-U_0), (V_1-V_0), NOCC), buffer = buffer, offset = _M13_offset)
                fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                               ctypes.c_void_p(_M11_sliced.ctypes.data),
                                               ctypes.c_void_p(_M13.ctypes.data),
                                               ctypes.c_int(_INPUT_19_sliced.shape[0]),
                                               ctypes.c_int(_INPUT_19_sliced.shape[1]),
                                               ctypes.c_int(_M11_sliced.shape[0]),
                                               ctypes.c_int(_M11_sliced.shape[1]))
                # step  24 iP,WUVi->PWUV
                offset_now       = offset_4        
                _M14_offset      = offset_now      
                _M14             = np.ndarray((NTHC_INT, (W_1-W_0), (U_1-U_0), (V_1-V_0)), buffer = buffer, offset = _M14_offset)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
                _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M13.shape[0]
                _size_dim_1      = _size_dim_1 * _M13.shape[1]
                _size_dim_1      = _size_dim_1 * _M13.shape[2]
                _M13_reshaped = _M13.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(_M14.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M14.shape[0]
                _M14_reshaped = _M14.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_1_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
                _M14             = _M14_reshaped.reshape(*shape_backup)
                # step  25 start for loop with indices ('V', 'W', 'U', 'S')
                for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
                    # step  26 slice _M6 with indices ['S', 'V']
                    _M6_sliced_offset = offset_5        
                    _M6_sliced       = np.ndarray(((S_1-S_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M6_sliced_offset)
                    fn_slice_3_0_2(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M6_sliced.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]),
                                   ctypes.c_int(S_0),
                                   ctypes.c_int(S_1),
                                   ctypes.c_int(V_0),
                                   ctypes.c_int(V_1))
                    # step  27 SPV,PWUV->SWUPV
                    offset_now       = offset_6        
                    _M15_offset      = offset_now      
                    _M15             = np.ndarray(((S_1-S_0), (W_1-W_0), (U_1-U_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M15_offset)
                    fn_contraction_012_1342_03412_wob(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                                      ctypes.c_void_p(_M14.ctypes.data),
                                                      ctypes.c_void_p(_M15.ctypes.data),
                                                      ctypes.c_int(_M6_sliced.shape[0]),
                                                      ctypes.c_int(_M6_sliced.shape[1]),
                                                      ctypes.c_int(_M6_sliced.shape[2]),
                                                      ctypes.c_int(_M14.shape[1]),
                                                      ctypes.c_int(_M14.shape[2]))
                    # step  28 SWUPV->SWUVP
                    _M15_perm_offset = offset_5        
                    _M15_perm        = np.ndarray(((S_1-S_0), (W_1-W_0), (U_1-U_0), (V_1-V_0), NTHC_INT), buffer = buffer, offset = _M15_perm_offset)
                    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                                   ctypes.c_int((S_1-S_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((U_1-U_0)),
                                                   ctypes.c_int(NTHC_INT),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  29 aP,SWUVP->aSWUV
                    offset_now       = offset_6        
                    _M16_offset      = offset_now      
                    _M16             = np.ndarray((NVIR, (S_1-S_0), (W_1-W_0), (U_1-U_0), (V_1-V_0)), buffer = buffer, offset = _M16_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
                    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[2]
                    _size_dim_1      = _size_dim_1 * _M15_perm.shape[3]
                    _M15_perm_reshaped = _M15_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M16.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M16.shape[0]
                    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_2_reshaped, _M15_perm_reshaped.T, c=_M16_reshaped)
                    _M16             = _M16_reshaped.reshape(*shape_backup)
                    # step  30 aSWUV->WaSUV
                    _M16_perm_offset = offset_5        
                    _M16_perm        = np.ndarray(((W_1-W_0), NVIR, (S_1-S_0), (U_1-U_0), (V_1-V_0)), buffer = buffer, offset = _M16_perm_offset)
                    fn_permutation_01234_20134_wob(ctypes.c_void_p(_M16.ctypes.data),
                                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                                   ctypes.c_int(NVIR),
                                                   ctypes.c_int((S_1-S_0)),
                                                   ctypes.c_int((W_1-W_0)),
                                                   ctypes.c_int((U_1-U_0)),
                                                   ctypes.c_int((V_1-V_0)))
                    # step  31 WaV,WaSUV->WaSU
                    offset_now       = offset_6        
                    _M17_offset      = offset_now      
                    _M17             = np.ndarray(((W_1-W_0), NVIR, (S_1-S_0), (U_1-U_0)), buffer = buffer, offset = _M17_offset)
                    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M12_perm.ctypes.data),
                                                      ctypes.c_void_p(_M16_perm.ctypes.data),
                                                      ctypes.c_void_p(_M17.ctypes.data),
                                                      ctypes.c_int(_M12_perm.shape[0]),
                                                      ctypes.c_int(_M12_perm.shape[1]),
                                                      ctypes.c_int(_M12_perm.shape[2]),
                                                      ctypes.c_int(_M16_perm.shape[2]),
                                                      ctypes.c_int(_M16_perm.shape[3]))
                    # step  32 WaSU->WSUa
                    _M17_perm_offset = offset_5        
                    _M17_perm        = np.ndarray(((W_1-W_0), (S_1-S_0), (U_1-U_0), NVIR), buffer = buffer, offset = _M17_perm_offset)
                    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M17.ctypes.data),
                                                 ctypes.c_void_p(_M17_perm.ctypes.data),
                                                 ctypes.c_int((W_1-W_0)),
                                                 ctypes.c_int(NVIR),
                                                 ctypes.c_int((S_1-S_0)),
                                                 ctypes.c_int((U_1-U_0)))
                    # step  33 aT,WSUa->TWSU
                    offset_now       = offset_6        
                    _M20_offset      = offset_now      
                    _M20             = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0), (U_1-U_0)), buffer = buffer, offset = _M20_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
                    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
                    _size_dim_1      = _size_dim_1 * _M17_perm.shape[2]
                    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M20.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M20.shape[0]
                    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_12_reshaped.T, _M17_perm_reshaped.T, c=_M20_reshaped)
                    _M20             = _M20_reshaped.reshape(*shape_backup)
                    # step  34 TWSU->UTWS
                    _M20_perm_offset = offset_5        
                    _M20_perm        = np.ndarray(((U_1-U_0), NTHC_INT, (W_1-W_0), (S_1-S_0)), buffer = buffer, offset = _M20_perm_offset)
                    fn_permutation_0123_3012_wob(ctypes.c_void_p(_M20.ctypes.data),
                                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                                 ctypes.c_int(NTHC_INT),
                                                 ctypes.c_int((W_1-W_0)),
                                                 ctypes.c_int((S_1-S_0)),
                                                 ctypes.c_int((U_1-U_0)))
                    # step  35 slice _INPUT_9 with indices ['S']
                    _INPUT_9_sliced_offset = offset_6        
                    _INPUT_9_sliced  = np.ndarray((NVIR, (S_1-S_0)), buffer = buffer, offset = _INPUT_9_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(S_0),
                                 ctypes.c_int(S_1))
                    # step  36 slice _INPUT_14 with indices ['U']
                    _INPUT_14_sliced_offset = offset_7        
                    _INPUT_14_sliced = np.ndarray((NVIR, (U_1-U_0)), buffer = buffer, offset = _INPUT_14_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(U_0),
                                 ctypes.c_int(U_1))
                    # step  37 cS,cU->SUc
                    offset_now       = offset_8        
                    _M8_offset       = offset_now      
                    _M8              = np.ndarray(((S_1-S_0), (U_1-U_0), NVIR), buffer = buffer, offset = _M8_offset)
                    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                                                 ctypes.c_void_p(_INPUT_14_sliced.ctypes.data),
                                                 ctypes.c_void_p(_M8.ctypes.data),
                                                 ctypes.c_int(_INPUT_9_sliced.shape[0]),
                                                 ctypes.c_int(_INPUT_9_sliced.shape[1]),
                                                 ctypes.c_int(_INPUT_14_sliced.shape[1]))
                    # step  38 slice _INPUT_22 with indices ['W']
                    _INPUT_22_sliced_offset = offset_6        
                    _INPUT_22_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_22_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_22.shape[0]),
                                 ctypes.c_int(_INPUT_22.shape[1]),
                                 ctypes.c_int(W_0),
                                 ctypes.c_int(W_1))
                    # step  39 cW,SUc->WSU
                    offset_now       = offset_7        
                    _M9_offset       = offset_now      
                    _M9              = np.ndarray(((W_1-W_0), (S_1-S_0), (U_1-U_0)), buffer = buffer, offset = _M9_offset)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _INPUT_22_sliced.shape[0]
                    _INPUT_22_sliced_reshaped = _INPUT_22_sliced.reshape(_size_dim_1,-1)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M8.shape[0]
                    _size_dim_1      = _size_dim_1 * _M8.shape[1]
                    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
                    shape_backup = copy.deepcopy(_M9.shape)
                    _size_dim_1      = 1               
                    _size_dim_1      = _size_dim_1 * _M9.shape[0]
                    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
                    lib.ddot(_INPUT_22_sliced_reshaped.T, _M8_reshaped.T, c=_M9_reshaped)
                    _M9              = _M9_reshaped.reshape(*shape_backup)
                    # step  40 slice _INPUT_10 with indices ['U']
                    _INPUT_10_sliced_offset = offset_6        
                    _INPUT_10_sliced = np.ndarray((NTHC_INT, (U_1-U_0)), buffer = buffer, offset = _INPUT_10_sliced_offset)
                    fn_slice_2_1(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                 ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                 ctypes.c_int(_INPUT_10.shape[0]),
                                 ctypes.c_int(_INPUT_10.shape[1]),
                                 ctypes.c_int(U_0),
                                 ctypes.c_int(U_1))
                    # step  41 TU,WSU->TWSU
                    offset_now       = offset_8        
                    _M10_offset      = offset_now      
                    _M10             = np.ndarray((NTHC_INT, (W_1-W_0), (S_1-S_0), (U_1-U_0)), buffer = buffer, offset = _M10_offset)
                    fn_contraction_01_231_0231_wob(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                                   ctypes.c_void_p(_M9.ctypes.data),
                                                   ctypes.c_void_p(_M10.ctypes.data),
                                                   ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                                   ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                                   ctypes.c_int(_M9.shape[0]),
                                                   ctypes.c_int(_M9.shape[1]))
                    # step  42 slice _M18 with indices ['S', 'W']
                    _M18_sliced_offset = offset_6        
                    _M18_sliced      = np.ndarray((NTHC_INT, (S_1-S_0), (W_1-W_0)), buffer = buffer, offset = _M18_sliced_offset)
                    fn_slice_3_1_2(ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M18_sliced.ctypes.data),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]),
                                   ctypes.c_int(_M18.shape[2]),
                                   ctypes.c_int(S_0),
                                   ctypes.c_int(S_1),
                                   ctypes.c_int(W_0),
                                   ctypes.c_int(W_1))
                    # step  43 TWSU,TSW->UTWS
                    offset_now       = offset_7        
                    _M19_offset      = offset_now      
                    _M19             = np.ndarray(((U_1-U_0), NTHC_INT, (W_1-W_0), (S_1-S_0)), buffer = buffer, offset = _M19_offset)
                    fn_contraction_0123_021_3012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                                     ctypes.c_void_p(_M18_sliced.ctypes.data),
                                                     ctypes.c_void_p(_M19.ctypes.data),
                                                     ctypes.c_int(_M10.shape[0]),
                                                     ctypes.c_int(_M10.shape[1]),
                                                     ctypes.c_int(_M10.shape[2]),
                                                     ctypes.c_int(_M10.shape[3]))
                    # step  44 UTWS,UTWS->
                    output_tmp       = ctypes.c_double(0.0)
                    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
                           ctypes.c_void_p(_M20_perm.ctypes.data),
                           ctypes.c_int(_M19.size),
                           ctypes.pointer(output_tmp))
                    output_tmp = output_tmp.value
                    _M21 += output_tmp
                # step  45 end   for loop with indices ('V', 'W', 'U', 'S')
            # step  46 end   for loop with indices ('V', 'W', 'U')
        # step  47 end   for loop with indices ('V', 'W')
    # step  48 end   for loop with indices ('V',)
    # clean the final forloop
    # MPI finalize
    if use_mpi:
        _M21 = reduce(_M21, root=0)
        _M21 = bcast(_M21, root=0)
    return _M21

if __name__ == "__main__":
    NVIR = np.random.randint(8, 16)
    NOCC = np.random.randint(8, 16)
    N_LAPLACE = np.random.randint(8, 16)
    NTHC_INT = np.random.randint(8, 16)
    NTHC_AMPLITUDE = np.random.randint(8, 16)
    Z                = np.random.random((NTHC_INT, NTHC_INT))
    X_o              = np.random.random((NOCC, NTHC_INT))
    X_v              = np.random.random((NVIR, NTHC_INT))
    T                = np.random.random((NTHC_AMPLITUDE, NTHC_AMPLITUDE))
    TAU_o            = np.random.random((NOCC, NTHC_AMPLITUDE))
    TAU_v            = np.random.random((NVIR, NTHC_AMPLITUDE))
    tau_o            = np.random.random((NOCC, N_LAPLACE))
    tau_v            = np.random.random((NVIR, N_LAPLACE))
    T1               = np.random.random((NOCC, NVIR))
    buffer           = np.random.random((16))
    Z = (Z + Z.T)/2 
    # test for RMP3_CX_1_forloop_Q_R and RMP3_CX_1_forloop_Q_R_naive
    benchmark        = RMP3_CX_1_forloop_Q_R_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_1_forloop_Q_R(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_1_forloop_Q_R_forloop_Q_R
    output3          = RMP3_CX_1_forloop_Q_R_forloop_Q_R(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_1_forloop_Q_S and RMP3_CX_1_forloop_Q_S_naive
    benchmark        = RMP3_CX_1_forloop_Q_S_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_1_forloop_Q_S(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_1_forloop_Q_S_forloop_Q_S
    output3          = RMP3_CX_1_forloop_Q_S_forloop_Q_S(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_1_forloop_R_U and RMP3_CX_1_forloop_R_U_naive
    benchmark        = RMP3_CX_1_forloop_R_U_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_1_forloop_R_U(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_1_forloop_R_U_forloop_U_R
    output3          = RMP3_CX_1_forloop_R_U_forloop_U_R(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_1_forloop_S_U and RMP3_CX_1_forloop_S_U_naive
    benchmark        = RMP3_CX_1_forloop_S_U_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_1_forloop_S_U(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_1_forloop_S_U_forloop_U_S
    output3          = RMP3_CX_1_forloop_S_U_forloop_U_S(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_2_forloop_P_R and RMP3_CX_2_forloop_P_R_naive
    benchmark        = RMP3_CX_2_forloop_P_R_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_2_forloop_P_R(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_2_forloop_P_R_forloop_P_R
    output3          = RMP3_CX_2_forloop_P_R_forloop_P_R(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_2_forloop_P_T and RMP3_CX_2_forloop_P_T_naive
    benchmark        = RMP3_CX_2_forloop_P_T_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_2_forloop_P_T(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_2_forloop_P_T_forloop_T_P
    output3          = RMP3_CX_2_forloop_P_T_forloop_T_P(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_2_forloop_Q_R and RMP3_CX_2_forloop_Q_R_naive
    benchmark        = RMP3_CX_2_forloop_Q_R_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_2_forloop_Q_R(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_2_forloop_Q_R_forloop_Q_R
    output3          = RMP3_CX_2_forloop_Q_R_forloop_Q_R(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_2_forloop_Q_T and RMP3_CX_2_forloop_Q_T_naive
    benchmark        = RMP3_CX_2_forloop_Q_T_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_2_forloop_Q_T(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_2_forloop_Q_T_forloop_T_Q
    output3          = RMP3_CX_2_forloop_Q_T_forloop_T_Q(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_3_forloop_P_T and RMP3_CX_3_forloop_P_T_naive
    benchmark        = RMP3_CX_3_forloop_P_T_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_3_forloop_P_T(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_3_forloop_P_T_forloop_P_T
    output3          = RMP3_CX_3_forloop_P_T_forloop_P_T(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_3_forloop_P_U and RMP3_CX_3_forloop_P_U_naive
    benchmark        = RMP3_CX_3_forloop_P_U_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_3_forloop_P_U(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_3_forloop_P_U_forloop_P_U
    output3          = RMP3_CX_3_forloop_P_U_forloop_P_U(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_3_forloop_S_T and RMP3_CX_3_forloop_S_T_naive
    benchmark        = RMP3_CX_3_forloop_S_T_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_3_forloop_S_T(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_3_forloop_S_T_forloop_T_S
    output3          = RMP3_CX_3_forloop_S_T_forloop_T_S(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP3_CX_3_forloop_S_U and RMP3_CX_3_forloop_S_U_naive
    benchmark        = RMP3_CX_3_forloop_S_U_naive(Z               ,
                                                   X_o             ,
                                                   X_v             ,
                                                   tau_o           ,
                                                   tau_v           )
    output           = RMP3_CX_3_forloop_S_U(Z               ,
                                             X_o             ,
                                             X_v             ,
                                             tau_o           ,
                                             tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CX_3_forloop_S_U_forloop_U_S
    output3          = RMP3_CX_3_forloop_S_U_forloop_U_S(Z               ,
                                                         X_o             ,
                                                         X_v             ,
                                                         tau_o           ,
                                                         tau_v           ,
                                                         buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
