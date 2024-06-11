import numpy
import numpy as np
import ctypes
import copy
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

def RMP3_CC_determine_bucket_size_forloop(NVIR        : int,
                                          NOCC        : int,
                                          N_LAPLACE   : int,
                                          NTHC_INT    : int,
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
    _M12_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M14_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M16_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M13_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M19_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M15_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_15_sliced_size = (NOCC * N_LAPLACE)
    _M1_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NOCC)))
    _INPUT_17_sliced_size = (NVIR * N_LAPLACE)
    _M4_size         = (NTHC_INT * (V_bunchsize * (W_bunchsize * NVIR)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M17_size        = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M18_sliced_size = (NTHC_INT * (NTHC_INT * V_bunchsize))
    _M20_perm_size   = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M8_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_19_sliced_size = (NOCC * N_LAPLACE)
    _M2_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M17_perm_size   = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M0_size         = (V_bunchsize * (W_bunchsize * NOCC))
    _INPUT_21_sliced_size = (NVIR * N_LAPLACE)
    _M5_size         = (NTHC_INT * (NTHC_INT * (V_bunchsize * W_bunchsize)))
    _M20_packed_size = (NTHC_INT * (W_bunchsize * NTHC_INT))
    _M3_size         = (V_bunchsize * (W_bunchsize * NVIR))
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M12_size)
    bucked_0_size    = max(bucked_0_size, _M14_size)
    bucked_0_size    = max(bucked_0_size, _M16_size)
    bucked_0_size    = max(bucked_0_size, _M20_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M13_size)
    bucked_1_size    = max(bucked_1_size, _M19_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M15_size)
    bucked_2_size    = max(bucked_2_size, _M7_size)
    bucked_2_size    = max(bucked_2_size, _M9_size)
    bucked_2_size    = max(bucked_2_size, _M11_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_15_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _INPUT_17_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M4_size)
    bucked_2_size    = max(bucked_2_size, _M6_size)
    bucked_2_size    = max(bucked_2_size, _M17_size)
    bucked_2_size    = max(bucked_2_size, _M18_sliced_size)
    bucked_2_size    = max(bucked_2_size, _M20_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M8_size)
    bucked_3_size    = max(bucked_3_size, _M18_size)
    # bucket 4
    bucked_4_size    = max(bucked_4_size, _M10_size)
    bucked_4_size    = max(bucked_4_size, _INPUT_19_sliced_size)
    bucked_4_size    = max(bucked_4_size, _M2_size)
    bucked_4_size    = max(bucked_4_size, _M6_perm_size)
    bucked_4_size    = max(bucked_4_size, _M17_perm_size)
    # bucket 5
    bucked_5_size    = max(bucked_5_size, _M0_size)
    bucked_5_size    = max(bucked_5_size, _INPUT_21_sliced_size)
    bucked_5_size    = max(bucked_5_size, _M5_size)
    bucked_5_size    = max(bucked_5_size, _M20_packed_size)
    # bucket 6
    bucked_6_size    = max(bucked_6_size, _M3_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    output.append(bucked_4_size)
    output.append(bucked_5_size)
    output.append(bucked_6_size)
    return output

def RMP3_CC_naive(Z           : np.ndarray,
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
    _M12             = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M14            )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("USW,USW->USW"  , _M13            , _M15            )
    del _M13        
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TU,USW->TSW"   , _INPUT_10       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("RQV,RQV->RQV"  , _M8             , _M10            )
    del _M8         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("RS,RQV->SQV"   , _INPUT_5        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (1, 3, 0, 2)    )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("TWQV,SQV->TWS" , _M17_perm       , _M18            )
    del _M17_perm   
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TSW,TSW->"     , _M19            , _M20_perm       )
    del _M19        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_CC(Z           : np.ndarray,
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
    # step 0 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M13.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M12_reshaped.T, c=_M13_reshaped)
    _M13         = _M13_reshaped.reshape(*shape_backup)
    del _M12        
    del _M12_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M14.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M15.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M14_reshaped.T, c=_M15_reshaped)
    _M15         = _M15_reshaped.reshape(*shape_backup)
    del _M14        
    del _M14_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 USW,USW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_int(_M13.shape[0]),
                                   ctypes.c_int(_M13.shape[1]),
                                   ctypes.c_int(_M13.shape[2]))
    del _M13        
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 TU,USW->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M19.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M16_reshaped, c=_M19_reshaped)
    _M19         = _M19_reshaped.reshape(*shape_backup)
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 jQ,jV->QVj 
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RQV,RQV->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]),
                                   ctypes.c_int(_M8.shape[2]))
    del _M8         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 RS,RQV->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M11_reshaped, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 iV,iW->VWi 
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
    _benchmark_time(t1, t2, "step 13")
    # step 13 iP,VWi->PVWi 
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
    _benchmark_time(t1, t2, "step 14")
    # step 14 iT,PVWi->TPVW 
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
    _benchmark_time(t1, t2, "step 15")
    # step 15 aV,aW->VWa 
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 aP,VWa->PVWa 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 aT,PVWa->TPVW 
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
    _benchmark_time(t1, t2, "step 18")
    # step 18 TPVW,TPVW->TPVW 
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 TPVW->TVWP 
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
    _benchmark_time(t1, t2, "step 20")
    # step 20 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M17_reshaped)
    _M17         = _M17_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 QTVW->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1302_wob = getattr(libpbc, "fn_permutation_0123_1302_wob", None)
    assert fn_permutation_0123_1302_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_1302_wob(ctypes.c_void_p(_M17.ctypes.data),
                                 ctypes.c_void_p(_M17_perm.ctypes.data),
                                 ctypes.c_int(_M17.shape[0]),
                                 ctypes.c_int(_M17.shape[1]),
                                 ctypes.c_int(_M17.shape[2]),
                                 ctypes.c_int(_M17.shape[3]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 TWQV,SQV->TWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
    _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _size_dim_1      = _size_dim_1 * _M20.shape[1]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M17_perm_reshaped, _M18_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M17_perm   
    del _M18        
    del _M18_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 TWS->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 TSW,TSW-> 
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
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_CC_forloop_(Z           : np.ndarray,
                     X_o         : np.ndarray,
                     X_v         : np.ndarray,
                     tau_o       : np.ndarray,
                     tau_v       : np.ndarray,
                     buffer      : np.ndarray,
                     V_bunchsize = 1,
                     W_bunchsize = 1):
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
    fn_permutation_0123_1302_wob = getattr(libpbc, "fn_permutation_0123_1302_wob", None)
    assert fn_permutation_0123_1302_wob is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    fn_slice_3_2 = getattr(libpbc, "fn_slice_3_2", None)
    assert fn_slice_3_2 is not None
    fn_packadd_3_1 = getattr(libpbc, "fn_packadd_3_1", None)
    assert fn_packadd_3_1 is not None
    # preallocate buffer
    bucket_size      = RMP3_CC_determine_bucket_size_forloop(NVIR = NVIR,
                                                             NOCC = NOCC,
                                                             N_LAPLACE = N_LAPLACE,
                                                             NTHC_INT = NTHC_INT,
                                                             V_bunchsize = V_bunchsize,
                                                             W_bunchsize = W_bunchsize)
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
    # step   1 kS,kW->SWk
    offset_now       = offset_0        
    _M12_offset      = offset_now      
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M12_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    # step   2 kU,SWk->USW
    offset_now       = offset_1        
    _M13_offset      = offset_now      
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M13.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M12_reshaped.T, c=_M13_reshaped)
    _M13             = _M13_reshaped.reshape(*shape_backup)
    # step   3 cS,cW->SWc
    offset_now       = offset_0        
    _M14_offset      = offset_now      
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M14_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M14.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    # step   4 cU,SWc->USW
    offset_now       = offset_2        
    _M15_offset      = offset_now      
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M15.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M14_reshaped.T, c=_M15_reshaped)
    _M15             = _M15_reshaped.reshape(*shape_backup)
    # step   5 USW,USW->USW
    offset_now       = offset_0        
    _M16_offset      = offset_now      
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_int(_M13.shape[0]),
                                   ctypes.c_int(_M13.shape[1]),
                                   ctypes.c_int(_M13.shape[2]))
    # step   6 TU,USW->TSW
    offset_now       = offset_1        
    _M19_offset      = offset_now      
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M19_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M19.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M16_reshaped, c=_M19_reshaped)
    _M19             = _M19_reshaped.reshape(*shape_backup)
    # step   7 allocate   _M20
    offset_now       = offset_0        
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    _M20_offset      = offset_now      
    _M20.ravel()[:] = 0.0
    # step   8 bQ,bV->QVb
    offset_now       = offset_2        
    _M7_offset       = offset_now      
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    # step   9 bR,QVb->RQV
    offset_now       = offset_3        
    _M8_offset       = offset_now      
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M8_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8              = _M8_reshaped.reshape(*shape_backup)
    # step  10 jQ,jV->QVj
    offset_now       = offset_2        
    _M9_offset       = offset_now      
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    # step  11 jR,QVj->RQV
    offset_now       = offset_4        
    _M10_offset      = offset_now      
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    # step  12 RQV,RQV->RQV
    offset_now       = offset_2        
    _M11_offset      = offset_now      
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]),
                                   ctypes.c_int(_M8.shape[2]))
    # step  13 RS,RQV->SQV
    offset_now       = offset_3        
    _M18_offset      = offset_now      
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M11_reshaped, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
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
            _M0_offset       = offset_now      
            _M0              = np.ndarray(((V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M0_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_19_sliced.ctypes.data),
                                         ctypes.c_void_p(_M0.ctypes.data),
                                         ctypes.c_int(_INPUT_15_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_15_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_19_sliced.shape[1]))
            # step  19 iP,VWi->PVWi
            offset_now       = offset_2        
            _M1_offset       = offset_now      
            _M1              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NOCC), buffer = buffer, offset = _M1_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                           ctypes.c_void_p(_M0.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_int(_INPUT_1.shape[0]),
                                           ctypes.c_int(_INPUT_1.shape[1]),
                                           ctypes.c_int(_M0.shape[0]),
                                           ctypes.c_int(_M0.shape[1]))
            # step  20 iT,PVWi->TPVW
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
            # step  21 slice _INPUT_17 with indices ['V']
            _INPUT_17_sliced_offset = offset_2        
            _INPUT_17_sliced = np.ndarray((NVIR, (V_1-V_0)), buffer = buffer, offset = _INPUT_17_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_17.ctypes.data),
                         ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_17.shape[0]),
                         ctypes.c_int(_INPUT_17.shape[1]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  22 slice _INPUT_21 with indices ['W']
            _INPUT_21_sliced_offset = offset_5        
            _INPUT_21_sliced = np.ndarray((NVIR, (W_1-W_0)), buffer = buffer, offset = _INPUT_21_sliced_offset)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_21.ctypes.data),
                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_21.shape[0]),
                         ctypes.c_int(_INPUT_21.shape[1]),
                         ctypes.c_int(W_0),
                         ctypes.c_int(W_1))
            # step  23 aV,aW->VWa
            offset_now       = offset_6        
            _M3_offset       = offset_now      
            _M3              = np.ndarray(((V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M3_offset)
            fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17_sliced.ctypes.data),
                                         ctypes.c_void_p(_INPUT_21_sliced.ctypes.data),
                                         ctypes.c_void_p(_M3.ctypes.data),
                                         ctypes.c_int(_INPUT_17_sliced.shape[0]),
                                         ctypes.c_int(_INPUT_17_sliced.shape[1]),
                                         ctypes.c_int(_INPUT_21_sliced.shape[1]))
            # step  24 aP,VWa->PVWa
            offset_now       = offset_2        
            _M4_offset       = offset_now      
            _M4              = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NVIR), buffer = buffer, offset = _M4_offset)
            fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_int(_INPUT_2.shape[0]),
                                           ctypes.c_int(_INPUT_2.shape[1]),
                                           ctypes.c_int(_M3.shape[0]),
                                           ctypes.c_int(_M3.shape[1]))
            # step  25 aT,PVWa->TPVW
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
            # step  26 TPVW,TPVW->TPVW
            offset_now       = offset_2        
            _M6_offset       = offset_now      
            _M6              = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M6_offset)
            fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                              ctypes.c_void_p(_M5.ctypes.data),
                                              ctypes.c_void_p(_M6.ctypes.data),
                                              ctypes.c_int(_M2.shape[0]),
                                              ctypes.c_int(_M2.shape[1]),
                                              ctypes.c_int(_M2.shape[2]),
                                              ctypes.c_int(_M2.shape[3]))
            # step  27 TPVW->TVWP
            _M6_perm_offset  = offset_4        
            _M6_perm         = np.ndarray((NTHC_INT, (V_1-V_0), (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
            fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  28 PQ,TVWP->QTVW
            offset_now       = offset_2        
            _M17_offset      = offset_now      
            _M17             = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0), (W_1-W_0)), buffer = buffer, offset = _M17_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
            _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
            _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
            _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M17.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M17.shape[0]
            _M17_reshaped = _M17.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_0_reshaped.T, _M6_perm_reshaped.T, c=_M17_reshaped)
            _M17             = _M17_reshaped.reshape(*shape_backup)
            # step  29 QTVW->TWQV
            _M17_perm_offset = offset_4        
            _M17_perm        = np.ndarray((NTHC_INT, (W_1-W_0), NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M17_perm_offset)
            fn_permutation_0123_1302_wob(ctypes.c_void_p(_M17.ctypes.data),
                                         ctypes.c_void_p(_M17_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((V_1-V_0)),
                                         ctypes.c_int((W_1-W_0)))
            # step  30 slice _M18 with indices ['V']
            _M18_sliced_offset = offset_2        
            _M18_sliced      = np.ndarray((NTHC_INT, NTHC_INT, (V_1-V_0)), buffer = buffer, offset = _M18_sliced_offset)
            fn_slice_3_2(ctypes.c_void_p(_M18.ctypes.data),
                         ctypes.c_void_p(_M18_sliced.ctypes.data),
                         ctypes.c_int(_M18.shape[0]),
                         ctypes.c_int(_M18.shape[1]),
                         ctypes.c_int(_M18.shape[2]),
                         ctypes.c_int(V_0),
                         ctypes.c_int(V_1))
            # step  31 TWQV,SQV->TWS
            offset_now       = offset_5        
            _M20_packed_offset = offset_now      
            _M20_packed      = np.ndarray((NTHC_INT, (W_1-W_0), NTHC_INT), buffer = buffer, offset = _M20_packed_offset)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M17_perm.shape[0]
            _size_dim_1      = _size_dim_1 * _M17_perm.shape[1]
            _M17_perm_reshaped = _M17_perm.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M18_sliced.shape[0]
            _M18_sliced_reshaped = _M18_sliced.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(_M20_packed.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M20_packed.shape[0]
            _size_dim_1      = _size_dim_1 * _M20_packed.shape[1]
            _M20_packed_reshaped = _M20_packed.reshape(_size_dim_1,-1)
            lib.ddot(_M17_perm_reshaped, _M18_sliced_reshaped.T, c=_M20_packed_reshaped)
            _M20_packed      = _M20_packed_reshaped.reshape(*shape_backup)
            # step  32 pack  _M20 with indices ['W']
            fn_packadd_3_1(ctypes.c_void_p(_M20.ctypes.data),
                           ctypes.c_void_p(_M20_packed.ctypes.data),
                           ctypes.c_int(_M20.shape[0]),
                           ctypes.c_int(_M20.shape[1]),
                           ctypes.c_int(_M20.shape[2]),
                           ctypes.c_int(W_0),
                           ctypes.c_int(W_1))
        # step  33 end   for loop with indices ('V', 'W')
    # step  34 end   for loop with indices ('V',)
    # step  35 deallocate ['_M18']
    # step  36 TWS->TSW
    _M20_perm_offset = offset_2        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(NTHC_INT),
                               ctypes.c_int(N_LAPLACE),
                               ctypes.c_int(NTHC_INT))
    # step  37 TSW,TSW->
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M19.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M19.size),
           ctypes.pointer(output_tmp))
    _M21 = output_tmp.value
    # clean the final forloop
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
    # test for RMP3_CC and RMP3_CC_naive
    benchmark        = RMP3_CC_naive(Z               ,
                                     X_o             ,
                                     X_v             ,
                                     tau_o           ,
                                     tau_v           )
    output           = RMP3_CC(Z               ,
                               X_o             ,
                               X_v             ,
                               tau_o           ,
                               tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP3_CC_forloop_
    output3          = RMP3_CC_forloop_(Z               ,
                                        X_o             ,
                                        X_v             ,
                                        tau_o           ,
                                        tau_v           ,
                                        buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
