import numpy
import numpy as np
import ctypes
import copy
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

def RMP2_K_forloop_P_R_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       P_bunchsize = 8,
                                                       R_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _INPUT_5
    tmp              = (NTHC_INT * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _INPUT_1_sliced
    tmp              = (NOCC * P_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_6_sliced
    tmp              = (NOCC * R_bunchsize)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (P_bunchsize * (R_bunchsize * NOCC))
    output           = max(output, tmp)
    # cmpr _INPUT_10_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (T_bunchsize * (P_bunchsize * R_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (P_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (T_bunchsize * (R_bunchsize * P_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M5_sliced
    tmp              = (NTHC_INT * (R_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (P_bunchsize * (NTHC_INT * (T_bunchsize * R_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (P_bunchsize * (NTHC_INT * (T_bunchsize * R_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NOCC * (P_bunchsize * (T_bunchsize * R_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_11_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (P_bunchsize * (R_bunchsize * (NOCC * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (P_bunchsize * (R_bunchsize * (NOCC * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (P_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (P_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (R_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M10_packed
    tmp              = (NTHC_INT * (P_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_P_R_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                P_bunchsize = 8,
                                                                R_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M7_size         = (NOCC * (P_bunchsize * (T_bunchsize * R_bunchsize)))
    _M0_size         = (P_bunchsize * (R_bunchsize * NOCC))
    _INPUT_1_sliced_size = (NOCC * P_bunchsize)
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M8_size         = (P_bunchsize * (R_bunchsize * (NOCC * T_bunchsize)))
    _INPUT_5_size    = (NTHC_INT * NTHC_INT)
    _INPUT_6_sliced_size = (NOCC * R_bunchsize)
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_11_sliced_size = (NOCC * T_bunchsize)
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M6_size         = (P_bunchsize * (NTHC_INT * (T_bunchsize * R_bunchsize)))
    _M1_size         = (T_bunchsize * (P_bunchsize * R_bunchsize))
    _M10_packed_size = (NTHC_INT * (P_bunchsize * T_bunchsize))
    _M5_sliced_size  = (NTHC_INT * (R_bunchsize * T_bunchsize))
    _INPUT_10_sliced_size = (NOCC * T_bunchsize)
    _INPUT_5_sliced_size = (R_bunchsize * NTHC_INT)
    _INPUT_0_sliced_size = (P_bunchsize * NTHC_INT)
    _M2_size         = (NTHC_INT * (T_bunchsize * (R_bunchsize * P_bunchsize)))
    _M9_size         = (NTHC_INT * (P_bunchsize * (R_bunchsize * T_bunchsize)))
    _M4_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    # cmpr _INPUT_5_size + _M3_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M4_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M4_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _INPUT_1_sliced_size + _INPUT_6_sliced_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _INPUT_1_sliced_size)
    size_now         = (size_now + _INPUT_6_sliced_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _INPUT_10_sliced_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _INPUT_10_sliced_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M1_size + _INPUT_0_sliced_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M2_size + _M5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M2_size)
    size_now         = (size_now + _M5_sliced_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M6_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M7_size + _INPUT_11_sliced_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _INPUT_11_sliced_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M9_size + _INPUT_5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    output           = max(output, size_now)
    # cmpr _INPUT_5_size + _M11_size + _M10_size + _M5_size + _M0_size + _M10_packed_size
    size_now         = 0               
    size_now         = (size_now + _INPUT_5_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M10_packed_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_P_R_naive(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _INPUT_5_perm    = np.transpose(_INPUT_5        , (1, 0)          )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("aP,aT->PTa"    , _INPUT_2        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,PTa->SPT"   , _INPUT_9        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("bR,bT->RTb"    , _INPUT_7        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("bQ,RTb->QRT"   , _INPUT_4        , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,iR->PRi"    , _INPUT_1        , _INPUT_6        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iT,PRi->TPR"   , _INPUT_10       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("PQ,TPR->QTRP"  , _INPUT_0        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("QTRP,QRT->PQTR", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("jQ,PTRQ->jPTR" , _INPUT_3        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("jT,jPTR->PRjT" , _INPUT_11       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 1, 3, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jS,PRTj->SPRT" , _INPUT_8        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9_perm         = np.transpose(_M9             , (0, 1, 3, 2)    )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("SR,SPTR->SPT"  , _INPUT_5_perm   , _M9_perm        )
    del _INPUT_5_perm
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("SPT,SPT->"     , _M10            , _M11            )
    del _M10        
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    return _M12

def RMP2_K_forloop_P_R(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 RS->SR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01_10 = getattr(libpbc, "fn_permutation_01_10", None)
    assert fn_permutation_01_10 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT), dtype=np.float64)
    _INPUT_5_perm    = np.ndarray((NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_01_10(ctypes.c_void_p(_INPUT_5.ctypes.data),
                         ctypes.c_void_p(_INPUT_5_perm.ctypes.data),
                         ctypes.c_int(_INPUT_5.shape[0]),
                         ctypes.c_int(_INPUT_5.shape[1]),
                         ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 aS,PTa->SPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M3_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M3         
    del _M3_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bR,bT->RTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bQ,RTb->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iP,iR->PRi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iT,PRi->TPR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 PQ,TPR->QTRP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_203_1230 = getattr(libpbc, "fn_contraction_01_203_1230", None)
    assert fn_contraction_01_203_1230 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_203_1230(ctypes.c_void_p(_INPUT_0.ctypes.data),
                               ctypes.c_void_p(_M1.ctypes.data),
                               ctypes.c_void_p(_M2.ctypes.data),
                               ctypes.c_int(_INPUT_0.shape[0]),
                               ctypes.c_int(_INPUT_0.shape[1]),
                               ctypes.c_int(_M1.shape[0]),
                               ctypes.c_int(_M1.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 QTRP,QRT->PQTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012(ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_void_p(_M5.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_M2.shape[0]),
                                 ctypes.c_int(_M2.shape[1]),
                                 ctypes.c_int(_M2.shape[2]),
                                 ctypes.c_int(_M2.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 PQTR->PTRQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                             ctypes.c_void_p(_M6_perm.ctypes.data),
                             ctypes.c_int(_M6.shape[0]),
                             ctypes.c_int(_M6.shape[1]),
                             ctypes.c_int(_M6.shape[2]),
                             ctypes.c_int(_M6.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 jQ,PTRQ->jPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jT,jPTR->PRjT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                ctypes.c_void_p(_M7.ctypes.data),
                                ctypes.c_void_p(_M8.ctypes.data),
                                ctypes.c_int(_INPUT_11.shape[0]),
                                ctypes.c_int(_INPUT_11.shape[1]),
                                ctypes.c_int(_M7.shape[1]),
                                ctypes.c_int(_M7.shape[3]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 PRjT->PRTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NOCC, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 jS,PRTj->SPRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 SPRT->SPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M9_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 SR,SPTR->SPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_5_perm.ctypes.data),
                               ctypes.c_void_p(_M9_perm.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_5_perm.shape[0]),
                               ctypes.c_int(_INPUT_5_perm.shape[1]),
                               ctypes.c_int(_M9_perm.shape[1]),
                               ctypes.c_int(_M9_perm.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 SPT,SPT-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    del _M10        
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    return _M12

def RMP2_K_forloop_P_R_forloop_P_R(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   P_bunchsize = 8,
                                   R_bunchsize = 8,
                                   T_bunchsize = 1):
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_permutation_01_10 = getattr(libpbc, "fn_permutation_01_10", None)
    assert fn_permutation_01_10 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_203_1230 = getattr(libpbc, "fn_contraction_01_203_1230", None)
    assert fn_contraction_01_203_1230 is not None
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_P_R_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          P_bunchsize = P_bunchsize,
                                                                          R_bunchsize = R_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_P_R_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   P_bunchsize = P_bunchsize,
                                                                                   R_bunchsize = R_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_P         = None            
    offset_P_R       = None            
    offset_P_R_T     = None            
    # step   0 start for loop with indices ()
    # step   1 RS->SR
    _INPUT_5_perm_offset = offset_now      
    offset_now       = (offset_now + (_INPUT_5.size * _itemsize))
    _INPUT_5_perm    = np.ndarray((NTHC_INT, NTHC_INT), buffer = buffer, offset = _INPUT_5_perm_offset)
    fn_permutation_01_10(ctypes.c_void_p(_INPUT_5.ctypes.data),
                         ctypes.c_void_p(_INPUT_5_perm.ctypes.data),
                         ctypes.c_int(NTHC_INT),
                         ctypes.c_int(NTHC_INT),
                         ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   2 aP,aT->PTa
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 aS,PTa->SPT
    _M11_offset      = offset_now      
    _M11_offset      = min(_M11_offset, _M3_offset)
    offset_now       = _M11_offset     
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    offset_now       = (_M11_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M3_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M11.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   4 allocate   _M10
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = offset_now)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    fn_clean(ctypes.c_void_p(_M10.ctypes.data),
             ctypes.c_int(_M10.size))
    # step   5 bR,bT->RTb
    _M4_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   6 bQ,RTb->QRT
    _M5_offset       = offset_now      
    _M5_offset       = min(_M5_offset, _M4_offset)
    offset_now       = _M5_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M4_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M5.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   7 start for loop with indices ('P',)
    for P_0, P_1 in lib.prange(0,NTHC_INT,P_bunchsize):
        if offset_P == None:
            offset_P         = offset_now      
        else:
            offset_now       = offset_P        
        # step   8 start for loop with indices ('P', 'R')
        for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
            if offset_P_R == None:
                offset_P_R       = offset_now      
            else:
                offset_now       = offset_P_R      
            # step   9 slice _INPUT_1 with indices ['P']
            _INPUT_1_sliced_offset = offset_now      
            _INPUT_1_sliced  = np.ndarray((NOCC, (P_1-P_0)), buffer = buffer, offset = _INPUT_1_sliced_offset)
            size_item        = (NOCC * ((P_1-P_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_1.ctypes.data),
                         ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_1.shape[0]),
                         ctypes.c_int(_INPUT_1.shape[1]),
                         ctypes.c_int(P_0),
                         ctypes.c_int(P_1))
            # step  10 slice _INPUT_6 with indices ['R']
            _INPUT_6_sliced_offset = offset_now      
            _INPUT_6_sliced  = np.ndarray((NOCC, (R_1-R_0)), buffer = buffer, offset = _INPUT_6_sliced_offset)
            size_item        = (NOCC * ((R_1-R_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_6.ctypes.data),
                         ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_6.shape[0]),
                         ctypes.c_int(_INPUT_6.shape[1]),
                         ctypes.c_int(R_0),
                         ctypes.c_int(R_1))
            # step  11 iP,iR->PRi
            _M0_offset       = offset_now      
            _M0_offset       = min(_M0_offset, _INPUT_1_sliced_offset)
            _M0_offset       = min(_M0_offset, _INPUT_6_sliced_offset)
            offset_now       = _M0_offset      
            tmp_itemsize     = ((P_1-P_0) * ((R_1-R_0) * (NOCC * _itemsize)))
            _M0              = np.ndarray(((P_1-P_0), (R_1-R_0), NOCC), buffer = buffer, offset = _M0_offset)
            offset_now       = (_M0_offset + tmp_itemsize)
            fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                                     ctypes.c_void_p(_M0.ctypes.data),
                                     ctypes.c_int(_INPUT_1_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_1_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_6_sliced.shape[1]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 start for loop with indices ('P', 'R', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_P_R_T == None:
                    offset_P_R_T     = offset_now      
                else:
                    offset_now       = offset_P_R_T    
                # step  13 slice _INPUT_10 with indices ['T']
                _INPUT_10_sliced_offset = offset_now      
                _INPUT_10_sliced = np.ndarray((NOCC, (T_1-T_0)), buffer = buffer, offset = _INPUT_10_sliced_offset)
                size_item        = (NOCC * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_10.shape[0]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  14 iT,PRi->TPR
                _M1_offset       = offset_now      
                _M1_offset       = min(_M1_offset, _INPUT_10_sliced_offset)
                offset_now       = _M1_offset      
                ddot_buffer      = np.ndarray(((T_1-T_0), (P_1-P_0), (R_1-R_0)), buffer = linearop_buf)
                tmp_itemsize     = ((T_1-T_0) * ((P_1-P_0) * ((R_1-R_0) * _itemsize)))
                _M1              = np.ndarray(((T_1-T_0), (P_1-P_0), (R_1-R_0)), buffer = buffer, offset = _M1_offset)
                offset_now       = (_M1_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_10_sliced.shape[0]
                _INPUT_10_sliced_reshaped = _INPUT_10_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M0.shape[0]
                _size_dim_1      = _size_dim_1 * _M0.shape[1]
                _M0_reshaped = _M0.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_10_sliced_reshaped.T, _M0_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M1.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  15 slice _INPUT_0 with indices ['P']
                _INPUT_0_sliced_offset = offset_now      
                _INPUT_0_sliced  = np.ndarray(((P_1-P_0), NTHC_INT), buffer = buffer, offset = _INPUT_0_sliced_offset)
                size_item        = ((P_1-P_0) * (NTHC_INT * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_0(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(P_0),
                             ctypes.c_int(P_1))
                # step  16 PQ,TPR->QTRP
                _M2_offset       = offset_now      
                _M2_offset       = min(_M2_offset, _INPUT_0_sliced_offset)
                _M2_offset       = min(_M2_offset, _M1_offset)
                offset_now       = _M2_offset      
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((R_1-R_0) * ((P_1-P_0) * _itemsize))))
                _M2              = np.ndarray((NTHC_INT, (T_1-T_0), (R_1-R_0), (P_1-P_0)), buffer = buffer, offset = _M2_offset)
                offset_now       = (_M2_offset + tmp_itemsize)
                fn_contraction_01_203_1230(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_void_p(_M2.ctypes.data),
                                           ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                           ctypes.c_int(_M1.shape[0]),
                                           ctypes.c_int(_M1.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  17 slice _M5 with indices ['R', 'T']
                _M5_sliced_offset = offset_now      
                _M5_sliced       = np.ndarray((NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M5_sliced_offset)
                size_item        = (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_sliced.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]),
                               ctypes.c_int(R_0),
                               ctypes.c_int(R_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  18 QTRP,QRT->PQTR
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M2_offset)
                _M6_offset       = min(_M6_offset, _M5_sliced_offset)
                offset_now       = _M6_offset      
                tmp_itemsize     = ((P_1-P_0) * (NTHC_INT * ((T_1-T_0) * ((R_1-R_0) * _itemsize))))
                _M6              = np.ndarray(((P_1-P_0), NTHC_INT, (T_1-T_0), (R_1-R_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                fn_contraction_0123_021_3012(ctypes.c_void_p(_M2.ctypes.data),
                                             ctypes.c_void_p(_M5_sliced.ctypes.data),
                                             ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_int(_M2.shape[0]),
                                             ctypes.c_int(_M2.shape[1]),
                                             ctypes.c_int(_M2.shape[2]),
                                             ctypes.c_int(_M2.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 PQTR->PTRQ
                _M6_perm_offset  = _M6_offset      
                _M6_perm         = np.ndarray(((P_1-P_0), (T_1-T_0), (R_1-R_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 jQ,PTRQ->jPTR
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M6_perm_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NOCC, (P_1-P_0), (T_1-T_0), (R_1-R_0)), buffer = linearop_buf)
                tmp_itemsize     = (NOCC * ((P_1-P_0) * ((T_1-T_0) * ((R_1-R_0) * _itemsize))))
                _M7              = np.ndarray((NOCC, (P_1-P_0), (T_1-T_0), (R_1-R_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
                _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_3_reshaped, _M6_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M7.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  21 slice _INPUT_11 with indices ['T']
                _INPUT_11_sliced_offset = offset_now      
                _INPUT_11_sliced = np.ndarray((NOCC, (T_1-T_0)), buffer = buffer, offset = _INPUT_11_sliced_offset)
                size_item        = (NOCC * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_11.shape[0]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  22 jT,jPTR->PRjT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _INPUT_11_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((P_1-P_0) * ((R_1-R_0) * (NOCC * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((P_1-P_0), (R_1-R_0), NOCC, (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                            ctypes.c_void_p(_M7.ctypes.data),
                                            ctypes.c_void_p(_M8.ctypes.data),
                                            ctypes.c_int(_INPUT_11_sliced.shape[0]),
                                            ctypes.c_int(_INPUT_11_sliced.shape[1]),
                                            ctypes.c_int(_M7.shape[1]),
                                            ctypes.c_int(_M7.shape[3]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 PRjT->PRTj
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((P_1-P_0), (R_1-R_0), (T_1-T_0), NOCC), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int(NOCC),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  24 jS,PRTj->SPRT
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (P_1-P_0), (R_1-R_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((P_1-P_0) * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NTHC_INT, (P_1-P_0), (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
                _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_8_reshaped.T, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M9.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  25 SPRT->SPTR
                _M9_perm_offset  = _M9_offset      
                _M9_perm         = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0), (R_1-R_0)), buffer = buffer, offset = _M9_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M9.ctypes.data),
                                         ctypes.c_void_p(_M9_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  26 slice _INPUT_5 with indices ['R']
                _INPUT_5_perm_sliced_offset = offset_now      
                _INPUT_5_perm_sliced = np.ndarray((NTHC_INT, (R_1-R_0)), buffer = buffer, offset = _INPUT_5_perm_sliced_offset)
                size_item        = (NTHC_INT * ((R_1-R_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_5_perm.ctypes.data),
                             ctypes.c_void_p(_INPUT_5_perm_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_5_perm.shape[0]),
                             ctypes.c_int(_INPUT_5_perm.shape[1]),
                             ctypes.c_int(R_0),
                             ctypes.c_int(R_1))
                # step  27 SR,SPTR->SPT
                _M10_packed_offset = offset_now      
                _M10_packed_offset = min(_M10_packed_offset, _INPUT_5_perm_sliced_offset)
                _M10_packed_offset = min(_M10_packed_offset, _M9_perm_offset)
                offset_now       = _M10_packed_offset
                tmp_itemsize     = (NTHC_INT * ((P_1-P_0) * ((T_1-T_0) * _itemsize)))
                _M10_packed      = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M10_packed_offset)
                offset_now       = (_M10_packed_offset + tmp_itemsize)
                fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_5_perm_sliced.ctypes.data),
                                           ctypes.c_void_p(_M9_perm.ctypes.data),
                                           ctypes.c_void_p(_M10_packed.ctypes.data),
                                           ctypes.c_int(_INPUT_5_perm_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_5_perm_sliced.shape[1]),
                                           ctypes.c_int(_M9_perm.shape[1]),
                                           ctypes.c_int(_M9_perm.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  28 pack  _M10 with indices ['P', 'T']
                fn_packadd_3_1_2(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_packed.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
            # step  29 end   for loop with indices ('P', 'R', 'T')
            # step  30 deallocate ['_M0']
        # step  31 end   for loop with indices ('P', 'R')
    # step  32 end   for loop with indices ('P',)
    # step  33 deallocate ['_M5']
    # step  34 SPT,SPT->
    _M12_offset      = offset_now      
    _M12_offset      = min(_M12_offset, _M10_offset)
    _M12_offset      = min(_M12_offset, _M11_offset)
    offset_now       = _M12_offset     
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(output_tmp))
    _M12 = output_tmp.value
    # clean the final forloop
    return _M12

def RMP2_K_forloop_P_S_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       P_bunchsize = 8,
                                                       S_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M4
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _INPUT_2_sliced
    tmp              = (NVIR * P_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_9_sliced
    tmp              = (NVIR * S_bunchsize)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (P_bunchsize * (S_bunchsize * NVIR))
    output           = max(output, tmp)
    # cmpr _INPUT_12_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (T_bunchsize * (P_bunchsize * S_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (P_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (T_bunchsize * (S_bunchsize * P_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M5_sliced
    tmp              = (NTHC_INT * (S_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (P_bunchsize * (NTHC_INT * (T_bunchsize * S_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (P_bunchsize * (NTHC_INT * (T_bunchsize * S_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NVIR * (P_bunchsize * (T_bunchsize * S_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_13_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (P_bunchsize * (S_bunchsize * (NVIR * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (P_bunchsize * (S_bunchsize * (NVIR * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (P_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (P_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (NTHC_INT * S_bunchsize)
    output           = max(output, tmp)
    # cmpr _M9_sliced
    tmp              = (NTHC_INT * (P_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (S_bunchsize * (P_bunchsize * (T_bunchsize * NTHC_INT)))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_P_S_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                P_bunchsize = 8,
                                                                S_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M7_size         = (NVIR * (P_bunchsize * (T_bunchsize * S_bunchsize)))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_9_sliced_size = (NVIR * S_bunchsize)
    _INPUT_13_sliced_size = (NVIR * T_bunchsize)
    _M9_sliced_size  = (NTHC_INT * (P_bunchsize * T_bunchsize))
    _M10_size        = (S_bunchsize * (P_bunchsize * (T_bunchsize * NTHC_INT)))
    _M8_size         = (P_bunchsize * (S_bunchsize * (NVIR * T_bunchsize)))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (P_bunchsize * (S_bunchsize * T_bunchsize)))
    _M3_size         = (NTHC_INT * (T_bunchsize * (S_bunchsize * P_bunchsize)))
    _M6_size         = (P_bunchsize * (NTHC_INT * (T_bunchsize * S_bunchsize)))
    _INPUT_2_sliced_size = (NVIR * P_bunchsize)
    _M1_size         = (P_bunchsize * (S_bunchsize * NVIR))
    _INPUT_12_sliced_size = (NVIR * T_bunchsize)
    _M5_sliced_size  = (NTHC_INT * (S_bunchsize * T_bunchsize))
    _INPUT_5_sliced_size = (NTHC_INT * S_bunchsize)
    _M2_size         = (T_bunchsize * (P_bunchsize * S_bunchsize))
    _INPUT_0_sliced_size = (P_bunchsize * NTHC_INT)
    _M9_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    # cmpr _M4_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M0_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _INPUT_2_sliced_size + _INPUT_9_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_2_sliced_size)
    size_now         = (size_now + _INPUT_9_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _INPUT_12_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _INPUT_12_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M2_size + _INPUT_0_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M2_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M3_size + _M5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M5_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M6_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M7_size + _INPUT_13_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _INPUT_13_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M11_size + _INPUT_5_sliced_size + _M9_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    size_now         = (size_now + _M9_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M11_size + _M10_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_P_S_naive(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("jS,jT->STj"    , _INPUT_8        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("jQ,STj->QST"   , _INPUT_3        , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,iT->PTi"    , _INPUT_1        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("iR,PTi->RPT"   , _INPUT_6        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("aP,aS->PSa"    , _INPUT_2        , _INPUT_9        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("aT,PSa->TPS"   , _INPUT_12       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("PQ,TPS->QTSP"  , _INPUT_0        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("QTSP,QST->PQTS", _M3             , _M5             )
    del _M3         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("bQ,PTSQ->bPTS" , _INPUT_4        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("bT,bPTS->PSbT" , _INPUT_13       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 1, 3, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bR,PSTb->RPST" , _INPUT_7        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (2, 1, 3, 0)    )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("RS,RPT->SPTR"  , _INPUT_5        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("SPTR,SPTR->"   , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_P_S(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 jS,jT->STj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jQ,STj->QST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 iR,PTi->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M0_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 aP,aS->PSa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 aT,PSa->TPS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
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
    _benchmark_time(t1, t2, "step 6")
    # step 6 PQ,TPS->QTSP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_203_1230 = getattr(libpbc, "fn_contraction_01_203_1230", None)
    assert fn_contraction_01_203_1230 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_203_1230(ctypes.c_void_p(_INPUT_0.ctypes.data),
                               ctypes.c_void_p(_M2.ctypes.data),
                               ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_int(_INPUT_0.shape[0]),
                               ctypes.c_int(_INPUT_0.shape[1]),
                               ctypes.c_int(_M2.shape[0]),
                               ctypes.c_int(_M2.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QTSP,QST->PQTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012(ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_void_p(_M5.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_M3.shape[0]),
                                 ctypes.c_int(_M3.shape[1]),
                                 ctypes.c_int(_M3.shape[2]),
                                 ctypes.c_int(_M3.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M3         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQTS->PTSQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                             ctypes.c_void_p(_M6_perm.ctypes.data),
                             ctypes.c_int(_M6.shape[0]),
                             ctypes.c_int(_M6.shape[1]),
                             ctypes.c_int(_M6.shape[2]),
                             ctypes.c_int(_M6.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 bQ,PTSQ->bPTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 bT,bPTS->PSbT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NVIR, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                ctypes.c_void_p(_M7.ctypes.data),
                                ctypes.c_void_p(_M8.ctypes.data),
                                ctypes.c_int(_INPUT_13.shape[0]),
                                ctypes.c_int(_INPUT_13.shape[1]),
                                ctypes.c_int(_M7.shape[1]),
                                ctypes.c_int(_M7.shape[3]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 PSbT->PSTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NVIR, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bR,PSTb->RPST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M8_perm_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 RPST->SPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_2130 = getattr(libpbc, "fn_permutation_0123_2130", None)
    assert fn_permutation_0123_2130 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_2130(ctypes.c_void_p(_M11.ctypes.data),
                             ctypes.c_void_p(_M11_perm.ctypes.data),
                             ctypes.c_int(_M11.shape[0]),
                             ctypes.c_int(_M11.shape[1]),
                             ctypes.c_int(_M11.shape[2]),
                             ctypes.c_int(_M11.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 RS,RPT->SPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_5.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_5.shape[0]),
                               ctypes.c_int(_INPUT_5.shape[1]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 SPTR,SPTR-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11_perm.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_P_S_forloop_P_S(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   P_bunchsize = 8,
                                   S_bunchsize = 8,
                                   T_bunchsize = 1):
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_203_1230 = getattr(libpbc, "fn_contraction_01_203_1230", None)
    assert fn_contraction_01_203_1230 is not None
    fn_permutation_0123_2130 = getattr(libpbc, "fn_permutation_0123_2130", None)
    assert fn_permutation_0123_2130 is not None
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_P_S_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          P_bunchsize = P_bunchsize,
                                                                          S_bunchsize = S_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_P_S_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   P_bunchsize = P_bunchsize,
                                                                                   S_bunchsize = S_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_P         = None            
    offset_P_S       = None            
    offset_P_S_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 jS,jT->STj
    _M4_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 jQ,STj->QST
    _M5_offset       = offset_now      
    _M5_offset       = min(_M5_offset, _M4_offset)
    offset_now       = _M5_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M4_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M5.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   4 iP,iT->PTi
    _M0_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M0_offset)
    offset_now       = (_M0_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 iR,PTi->RPT
    _M9_offset       = offset_now      
    _M9_offset       = min(_M9_offset, _M0_offset)
    offset_now       = _M9_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    offset_now       = (_M9_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M0_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M9.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   6 start for loop with indices ('P',)
    for P_0, P_1 in lib.prange(0,NTHC_INT,P_bunchsize):
        if offset_P == None:
            offset_P         = offset_now      
        else:
            offset_now       = offset_P        
        # step   7 start for loop with indices ('P', 'S')
        for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
            if offset_P_S == None:
                offset_P_S       = offset_now      
            else:
                offset_now       = offset_P_S      
            # step   8 slice _INPUT_2 with indices ['P']
            _INPUT_2_sliced_offset = offset_now      
            _INPUT_2_sliced  = np.ndarray((NVIR, (P_1-P_0)), buffer = buffer, offset = _INPUT_2_sliced_offset)
            size_item        = (NVIR * ((P_1-P_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_2.ctypes.data),
                         ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_2.shape[0]),
                         ctypes.c_int(_INPUT_2.shape[1]),
                         ctypes.c_int(P_0),
                         ctypes.c_int(P_1))
            # step   9 slice _INPUT_9 with indices ['S']
            _INPUT_9_sliced_offset = offset_now      
            _INPUT_9_sliced  = np.ndarray((NVIR, (S_1-S_0)), buffer = buffer, offset = _INPUT_9_sliced_offset)
            size_item        = (NVIR * ((S_1-S_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_9.ctypes.data),
                         ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_9.shape[0]),
                         ctypes.c_int(_INPUT_9.shape[1]),
                         ctypes.c_int(S_0),
                         ctypes.c_int(S_1))
            # step  10 aP,aS->PSa
            _M1_offset       = offset_now      
            _M1_offset       = min(_M1_offset, _INPUT_2_sliced_offset)
            _M1_offset       = min(_M1_offset, _INPUT_9_sliced_offset)
            offset_now       = _M1_offset      
            tmp_itemsize     = ((P_1-P_0) * ((S_1-S_0) * (NVIR * _itemsize)))
            _M1              = np.ndarray(((P_1-P_0), (S_1-S_0), NVIR), buffer = buffer, offset = _M1_offset)
            offset_now       = (_M1_offset + tmp_itemsize)
            fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                                     ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_int(_INPUT_2_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_2_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_9_sliced.shape[1]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  11 start for loop with indices ('P', 'S', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_P_S_T == None:
                    offset_P_S_T     = offset_now      
                else:
                    offset_now       = offset_P_S_T    
                # step  12 slice _INPUT_12 with indices ['T']
                _INPUT_12_sliced_offset = offset_now      
                _INPUT_12_sliced = np.ndarray((NVIR, (T_1-T_0)), buffer = buffer, offset = _INPUT_12_sliced_offset)
                size_item        = (NVIR * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_12.shape[0]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  13 aT,PSa->TPS
                _M2_offset       = offset_now      
                _M2_offset       = min(_M2_offset, _INPUT_12_sliced_offset)
                offset_now       = _M2_offset      
                ddot_buffer      = np.ndarray(((T_1-T_0), (P_1-P_0), (S_1-S_0)), buffer = linearop_buf)
                tmp_itemsize     = ((T_1-T_0) * ((P_1-P_0) * ((S_1-S_0) * _itemsize)))
                _M2              = np.ndarray(((T_1-T_0), (P_1-P_0), (S_1-S_0)), buffer = buffer, offset = _M2_offset)
                offset_now       = (_M2_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_12_sliced.shape[0]
                _INPUT_12_sliced_reshaped = _INPUT_12_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M1.shape[0]
                _size_dim_1      = _size_dim_1 * _M1.shape[1]
                _M1_reshaped = _M1.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_12_sliced_reshaped.T, _M1_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M2.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  14 slice _INPUT_0 with indices ['P']
                _INPUT_0_sliced_offset = offset_now      
                _INPUT_0_sliced  = np.ndarray(((P_1-P_0), NTHC_INT), buffer = buffer, offset = _INPUT_0_sliced_offset)
                size_item        = ((P_1-P_0) * (NTHC_INT * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_0(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(P_0),
                             ctypes.c_int(P_1))
                # step  15 PQ,TPS->QTSP
                _M3_offset       = offset_now      
                _M3_offset       = min(_M3_offset, _INPUT_0_sliced_offset)
                _M3_offset       = min(_M3_offset, _M2_offset)
                offset_now       = _M3_offset      
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((S_1-S_0) * ((P_1-P_0) * _itemsize))))
                _M3              = np.ndarray((NTHC_INT, (T_1-T_0), (S_1-S_0), (P_1-P_0)), buffer = buffer, offset = _M3_offset)
                offset_now       = (_M3_offset + tmp_itemsize)
                fn_contraction_01_203_1230(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                           ctypes.c_void_p(_M2.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                           ctypes.c_int(_M2.shape[0]),
                                           ctypes.c_int(_M2.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  16 slice _M5 with indices ['S', 'T']
                _M5_sliced_offset = offset_now      
                _M5_sliced       = np.ndarray((NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M5_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_sliced.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]),
                               ctypes.c_int(S_0),
                               ctypes.c_int(S_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  17 QTSP,QST->PQTS
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M3_offset)
                _M6_offset       = min(_M6_offset, _M5_sliced_offset)
                offset_now       = _M6_offset      
                tmp_itemsize     = ((P_1-P_0) * (NTHC_INT * ((T_1-T_0) * ((S_1-S_0) * _itemsize))))
                _M6              = np.ndarray(((P_1-P_0), NTHC_INT, (T_1-T_0), (S_1-S_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                fn_contraction_0123_021_3012(ctypes.c_void_p(_M3.ctypes.data),
                                             ctypes.c_void_p(_M5_sliced.ctypes.data),
                                             ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_int(_M3.shape[0]),
                                             ctypes.c_int(_M3.shape[1]),
                                             ctypes.c_int(_M3.shape[2]),
                                             ctypes.c_int(_M3.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  18 PQTS->PTSQ
                _M6_perm_offset  = _M6_offset      
                _M6_perm         = np.ndarray(((P_1-P_0), (T_1-T_0), (S_1-S_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 bQ,PTSQ->bPTS
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M6_perm_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NVIR, (P_1-P_0), (T_1-T_0), (S_1-S_0)), buffer = linearop_buf)
                tmp_itemsize     = (NVIR * ((P_1-P_0) * ((T_1-T_0) * ((S_1-S_0) * _itemsize))))
                _M7              = np.ndarray((NVIR, (P_1-P_0), (T_1-T_0), (S_1-S_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
                _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_4_reshaped, _M6_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M7.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  20 slice _INPUT_13 with indices ['T']
                _INPUT_13_sliced_offset = offset_now      
                _INPUT_13_sliced = np.ndarray((NVIR, (T_1-T_0)), buffer = buffer, offset = _INPUT_13_sliced_offset)
                size_item        = (NVIR * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_13.shape[0]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  21 bT,bPTS->PSbT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _INPUT_13_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((P_1-P_0) * ((S_1-S_0) * (NVIR * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((P_1-P_0), (S_1-S_0), NVIR, (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                            ctypes.c_void_p(_M7.ctypes.data),
                                            ctypes.c_void_p(_M8.ctypes.data),
                                            ctypes.c_int(_INPUT_13_sliced.shape[0]),
                                            ctypes.c_int(_INPUT_13_sliced.shape[1]),
                                            ctypes.c_int(_M7.shape[1]),
                                            ctypes.c_int(_M7.shape[3]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  22 PSbT->PSTb
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((P_1-P_0), (S_1-S_0), (T_1-T_0), NVIR), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int(NVIR),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 bR,PSTb->RPST
                _M11_offset      = offset_now      
                _M11_offset      = min(_M11_offset, _M8_perm_offset)
                offset_now       = _M11_offset     
                ddot_buffer      = np.ndarray((NTHC_INT, (P_1-P_0), (S_1-S_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((P_1-P_0) * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M11             = np.ndarray((NTHC_INT, (P_1-P_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M11_offset)
                offset_now       = (_M11_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
                _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_7_reshaped.T, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M11.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  24 RPST->SPTR
                _M11_perm_offset = _M11_offset     
                _M11_perm        = np.ndarray(((S_1-S_0), (P_1-P_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M11_perm_offset)
                fn_permutation_0123_2130(ctypes.c_void_p(_M11.ctypes.data),
                                         ctypes.c_void_p(_M11_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 slice _INPUT_5 with indices ['S']
                _INPUT_5_sliced_offset = offset_now      
                _INPUT_5_sliced  = np.ndarray((NTHC_INT, (S_1-S_0)), buffer = buffer, offset = _INPUT_5_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(S_0),
                             ctypes.c_int(S_1))
                # step  26 slice _M9 with indices ['P', 'T']
                _M9_sliced_offset = offset_now      
                _M9_sliced       = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M9_sliced_offset)
                size_item        = (NTHC_INT * ((P_1-P_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M9_sliced.ctypes.data),
                               ctypes.c_int(_M9.shape[0]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_int(P_0),
                               ctypes.c_int(P_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  27 RS,RPT->SPTR
                _M10_offset      = offset_now      
                _M10_offset      = min(_M10_offset, _INPUT_5_sliced_offset)
                _M10_offset      = min(_M10_offset, _M9_sliced_offset)
                offset_now       = _M10_offset     
                tmp_itemsize     = ((S_1-S_0) * ((P_1-P_0) * ((T_1-T_0) * (NTHC_INT * _itemsize))))
                _M10             = np.ndarray(((S_1-S_0), (P_1-P_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M10_offset)
                offset_now       = (_M10_offset + tmp_itemsize)
                fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                           ctypes.c_void_p(_M9_sliced.ctypes.data),
                                           ctypes.c_void_p(_M10.ctypes.data),
                                           ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                           ctypes.c_int(_M9_sliced.shape[1]),
                                           ctypes.c_int(_M9_sliced.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  28 SPTR,SPTR->
                output_tmp       = ctypes.c_double(0.0)
                fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                       ctypes.c_void_p(_M11_perm.ctypes.data),
                       ctypes.c_int(_M10.size),
                       ctypes.pointer(output_tmp))
                output_tmp = output_tmp.value
                _M12 += output_tmp
            # step  29 end   for loop with indices ('P', 'S', 'T')
        # step  30 end   for loop with indices ('P', 'S')
    # step  31 end   for loop with indices ('P',)
    # clean the final forloop
    return _M12

def RMP2_K_forloop_Q_R_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       Q_bunchsize = 8,
                                                       R_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M4
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _INPUT_4_sliced
    tmp              = (NVIR * Q_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_7_sliced
    tmp              = (NVIR * R_bunchsize)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (Q_bunchsize * (R_bunchsize * NVIR))
    output           = max(output, tmp)
    # cmpr _INPUT_13_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (T_bunchsize * (Q_bunchsize * R_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (NTHC_INT * Q_bunchsize)
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (T_bunchsize * (R_bunchsize * Q_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M5_sliced
    tmp              = (NTHC_INT * (R_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (Q_bunchsize * (NTHC_INT * (T_bunchsize * R_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (Q_bunchsize * (NTHC_INT * (T_bunchsize * R_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NVIR * (Q_bunchsize * (T_bunchsize * R_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_12_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (Q_bunchsize * (R_bunchsize * (NVIR * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (Q_bunchsize * (R_bunchsize * (NVIR * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (Q_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (Q_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (R_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M9_sliced
    tmp              = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (R_bunchsize * (Q_bunchsize * (T_bunchsize * NTHC_INT)))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_Q_R_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                Q_bunchsize = 8,
                                                                R_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M7_size         = (NVIR * (Q_bunchsize * (T_bunchsize * R_bunchsize)))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_13_sliced_size = (NVIR * T_bunchsize)
    _M9_sliced_size  = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    _M10_size        = (R_bunchsize * (Q_bunchsize * (T_bunchsize * NTHC_INT)))
    _M8_size         = (Q_bunchsize * (R_bunchsize * (NVIR * T_bunchsize)))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_7_sliced_size = (NVIR * R_bunchsize)
    _M11_size        = (NTHC_INT * (Q_bunchsize * (R_bunchsize * T_bunchsize)))
    _M3_size         = (NTHC_INT * (T_bunchsize * (R_bunchsize * Q_bunchsize)))
    _M6_size         = (Q_bunchsize * (NTHC_INT * (T_bunchsize * R_bunchsize)))
    _M1_size         = (Q_bunchsize * (R_bunchsize * NVIR))
    _INPUT_12_sliced_size = (NVIR * T_bunchsize)
    _M5_sliced_size  = (NTHC_INT * (R_bunchsize * T_bunchsize))
    _INPUT_5_sliced_size = (R_bunchsize * NTHC_INT)
    _INPUT_4_sliced_size = (NVIR * Q_bunchsize)
    _M2_size         = (T_bunchsize * (Q_bunchsize * R_bunchsize))
    _INPUT_0_sliced_size = (NTHC_INT * Q_bunchsize)
    _M9_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    # cmpr _M4_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M0_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _INPUT_4_sliced_size + _INPUT_7_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_4_sliced_size)
    size_now         = (size_now + _INPUT_7_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _INPUT_13_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _INPUT_13_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M2_size + _INPUT_0_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M2_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M3_size + _M5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M5_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M6_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M7_size + _INPUT_12_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _INPUT_12_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M11_size + _INPUT_5_sliced_size + _M9_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    size_now         = (size_now + _M9_sliced_size)
    output           = max(output, size_now)
    # cmpr _M5_size + _M9_size + _M1_size + _M11_size + _M10_size
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_Q_R_naive(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iR,iT->RTi"    , _INPUT_6        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("iP,RTi->PRT"   , _INPUT_1        , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("jQ,jT->QTj"    , _INPUT_3        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jS,QTj->SQT"   , _INPUT_8        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("bQ,bR->QRb"    , _INPUT_4        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("bT,QRb->TQR"   , _INPUT_13       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("PQ,TQR->PTRQ"  , _INPUT_0        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("PTRQ,PRT->QPTR", _M3             , _M5             )
    del _M3         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("aP,QTRP->aQTR" , _INPUT_2        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("aT,aQTR->QRaT" , _INPUT_12       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 1, 3, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,QRTa->SQRT" , _INPUT_9        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (2, 1, 3, 0)    )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("RS,SQT->RQTS"  , _INPUT_5        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("RQTS,RQTS->"   , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_Q_R(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 iR,iT->RTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 iP,RTi->PRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M0_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bQ,bR->QRb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 bT,QRb->TQR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M2.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M1_reshaped.T, c=_M2_reshaped)
    _M2          = _M2_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 PQ,TQR->PTRQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_213_0231 = getattr(libpbc, "fn_contraction_01_213_0231", None)
    assert fn_contraction_01_213_0231 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_213_0231(ctypes.c_void_p(_INPUT_0.ctypes.data),
                               ctypes.c_void_p(_M2.ctypes.data),
                               ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_int(_INPUT_0.shape[0]),
                               ctypes.c_int(_INPUT_0.shape[1]),
                               ctypes.c_int(_M2.shape[0]),
                               ctypes.c_int(_M2.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 PTRQ,PRT->QPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012(ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_void_p(_M5.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_M3.shape[0]),
                                 ctypes.c_int(_M3.shape[1]),
                                 ctypes.c_int(_M3.shape[2]),
                                 ctypes.c_int(_M3.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M3         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 QPTR->QTRP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                             ctypes.c_void_p(_M6_perm.ctypes.data),
                             ctypes.c_int(_M6.shape[0]),
                             ctypes.c_int(_M6.shape[1]),
                             ctypes.c_int(_M6.shape[2]),
                             ctypes.c_int(_M6.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aP,QTRP->aQTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 aT,aQTR->QRaT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NVIR, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                ctypes.c_void_p(_M7.ctypes.data),
                                ctypes.c_void_p(_M8.ctypes.data),
                                ctypes.c_int(_INPUT_12.shape[0]),
                                ctypes.c_int(_INPUT_12.shape[1]),
                                ctypes.c_int(_M7.shape[1]),
                                ctypes.c_int(_M7.shape[3]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 QRaT->QRTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NVIR, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aS,QRTa->SQRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M8_perm_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 SQRT->RQTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_2130 = getattr(libpbc, "fn_permutation_0123_2130", None)
    assert fn_permutation_0123_2130 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_2130(ctypes.c_void_p(_M11.ctypes.data),
                             ctypes.c_void_p(_M11_perm.ctypes.data),
                             ctypes.c_int(_M11.shape[0]),
                             ctypes.c_int(_M11.shape[1]),
                             ctypes.c_int(_M11.shape[2]),
                             ctypes.c_int(_M11.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 RS,SQT->RQTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_123_0231 = getattr(libpbc, "fn_contraction_01_123_0231", None)
    assert fn_contraction_01_123_0231 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_01_123_0231(ctypes.c_void_p(_INPUT_5.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_5.shape[0]),
                               ctypes.c_int(_INPUT_5.shape[1]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RQTS,RQTS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11_perm.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_Q_R_forloop_Q_R(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   Q_bunchsize = 8,
                                   R_bunchsize = 8,
                                   T_bunchsize = 1):
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_01_123_0231 = getattr(libpbc, "fn_contraction_01_123_0231", None)
    assert fn_contraction_01_123_0231 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_contraction_01_213_0231 = getattr(libpbc, "fn_contraction_01_213_0231", None)
    assert fn_contraction_01_213_0231 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_permutation_0123_2130 = getattr(libpbc, "fn_permutation_0123_2130", None)
    assert fn_permutation_0123_2130 is not None
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_Q_R_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          Q_bunchsize = Q_bunchsize,
                                                                          R_bunchsize = R_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_Q_R_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   Q_bunchsize = Q_bunchsize,
                                                                                   R_bunchsize = R_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_Q         = None            
    offset_Q_R       = None            
    offset_Q_R_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 iR,iT->RTi
    _M4_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 iP,RTi->PRT
    _M5_offset       = offset_now      
    _M5_offset       = min(_M5_offset, _M4_offset)
    offset_now       = _M5_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M4_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M5.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   4 jQ,jT->QTj
    _M0_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M0_offset)
    offset_now       = (_M0_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 jS,QTj->SQT
    _M9_offset       = offset_now      
    _M9_offset       = min(_M9_offset, _M0_offset)
    offset_now       = _M9_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    offset_now       = (_M9_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M0_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M9.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   6 start for loop with indices ('Q',)
    for Q_0, Q_1 in lib.prange(0,NTHC_INT,Q_bunchsize):
        if offset_Q == None:
            offset_Q         = offset_now      
        else:
            offset_now       = offset_Q        
        # step   7 start for loop with indices ('Q', 'R')
        for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
            if offset_Q_R == None:
                offset_Q_R       = offset_now      
            else:
                offset_now       = offset_Q_R      
            # step   8 slice _INPUT_4 with indices ['Q']
            _INPUT_4_sliced_offset = offset_now      
            _INPUT_4_sliced  = np.ndarray((NVIR, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_4_sliced_offset)
            size_item        = (NVIR * ((Q_1-Q_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_4.ctypes.data),
                         ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_4.shape[0]),
                         ctypes.c_int(_INPUT_4.shape[1]),
                         ctypes.c_int(Q_0),
                         ctypes.c_int(Q_1))
            # step   9 slice _INPUT_7 with indices ['R']
            _INPUT_7_sliced_offset = offset_now      
            _INPUT_7_sliced  = np.ndarray((NVIR, (R_1-R_0)), buffer = buffer, offset = _INPUT_7_sliced_offset)
            size_item        = (NVIR * ((R_1-R_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_7.ctypes.data),
                         ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_7.shape[0]),
                         ctypes.c_int(_INPUT_7.shape[1]),
                         ctypes.c_int(R_0),
                         ctypes.c_int(R_1))
            # step  10 bQ,bR->QRb
            _M1_offset       = offset_now      
            _M1_offset       = min(_M1_offset, _INPUT_4_sliced_offset)
            _M1_offset       = min(_M1_offset, _INPUT_7_sliced_offset)
            offset_now       = _M1_offset      
            tmp_itemsize     = ((Q_1-Q_0) * ((R_1-R_0) * (NVIR * _itemsize)))
            _M1              = np.ndarray(((Q_1-Q_0), (R_1-R_0), NVIR), buffer = buffer, offset = _M1_offset)
            offset_now       = (_M1_offset + tmp_itemsize)
            fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                                     ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_int(_INPUT_4_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_4_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_7_sliced.shape[1]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  11 start for loop with indices ('Q', 'R', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_Q_R_T == None:
                    offset_Q_R_T     = offset_now      
                else:
                    offset_now       = offset_Q_R_T    
                # step  12 slice _INPUT_13 with indices ['T']
                _INPUT_13_sliced_offset = offset_now      
                _INPUT_13_sliced = np.ndarray((NVIR, (T_1-T_0)), buffer = buffer, offset = _INPUT_13_sliced_offset)
                size_item        = (NVIR * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_13.shape[0]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  13 bT,QRb->TQR
                _M2_offset       = offset_now      
                _M2_offset       = min(_M2_offset, _INPUT_13_sliced_offset)
                offset_now       = _M2_offset      
                ddot_buffer      = np.ndarray(((T_1-T_0), (Q_1-Q_0), (R_1-R_0)), buffer = linearop_buf)
                tmp_itemsize     = ((T_1-T_0) * ((Q_1-Q_0) * ((R_1-R_0) * _itemsize)))
                _M2              = np.ndarray(((T_1-T_0), (Q_1-Q_0), (R_1-R_0)), buffer = buffer, offset = _M2_offset)
                offset_now       = (_M2_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_13_sliced.shape[0]
                _INPUT_13_sliced_reshaped = _INPUT_13_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M1.shape[0]
                _size_dim_1      = _size_dim_1 * _M1.shape[1]
                _M1_reshaped = _M1.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_13_sliced_reshaped.T, _M1_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M2.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  14 slice _INPUT_0 with indices ['Q']
                _INPUT_0_sliced_offset = offset_now      
                _INPUT_0_sliced  = np.ndarray((NTHC_INT, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_0_sliced_offset)
                size_item        = (NTHC_INT * ((Q_1-Q_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(Q_0),
                             ctypes.c_int(Q_1))
                # step  15 PQ,TQR->PTRQ
                _M3_offset       = offset_now      
                _M3_offset       = min(_M3_offset, _INPUT_0_sliced_offset)
                _M3_offset       = min(_M3_offset, _M2_offset)
                offset_now       = _M3_offset      
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((R_1-R_0) * ((Q_1-Q_0) * _itemsize))))
                _M3              = np.ndarray((NTHC_INT, (T_1-T_0), (R_1-R_0), (Q_1-Q_0)), buffer = buffer, offset = _M3_offset)
                offset_now       = (_M3_offset + tmp_itemsize)
                fn_contraction_01_213_0231(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                           ctypes.c_void_p(_M2.ctypes.data),
                                           ctypes.c_void_p(_M3.ctypes.data),
                                           ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                           ctypes.c_int(_M2.shape[0]),
                                           ctypes.c_int(_M2.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  16 slice _M5 with indices ['R', 'T']
                _M5_sliced_offset = offset_now      
                _M5_sliced       = np.ndarray((NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M5_sliced_offset)
                size_item        = (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_sliced.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]),
                               ctypes.c_int(R_0),
                               ctypes.c_int(R_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  17 PTRQ,PRT->QPTR
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M3_offset)
                _M6_offset       = min(_M6_offset, _M5_sliced_offset)
                offset_now       = _M6_offset      
                tmp_itemsize     = ((Q_1-Q_0) * (NTHC_INT * ((T_1-T_0) * ((R_1-R_0) * _itemsize))))
                _M6              = np.ndarray(((Q_1-Q_0), NTHC_INT, (T_1-T_0), (R_1-R_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                fn_contraction_0123_021_3012(ctypes.c_void_p(_M3.ctypes.data),
                                             ctypes.c_void_p(_M5_sliced.ctypes.data),
                                             ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_int(_M3.shape[0]),
                                             ctypes.c_int(_M3.shape[1]),
                                             ctypes.c_int(_M3.shape[2]),
                                             ctypes.c_int(_M3.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  18 QPTR->QTRP
                _M6_perm_offset  = _M6_offset      
                _M6_perm         = np.ndarray(((Q_1-Q_0), (T_1-T_0), (R_1-R_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 aP,QTRP->aQTR
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M6_perm_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NVIR, (Q_1-Q_0), (T_1-T_0), (R_1-R_0)), buffer = linearop_buf)
                tmp_itemsize     = (NVIR * ((Q_1-Q_0) * ((T_1-T_0) * ((R_1-R_0) * _itemsize))))
                _M7              = np.ndarray((NVIR, (Q_1-Q_0), (T_1-T_0), (R_1-R_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
                _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_2_reshaped, _M6_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M7.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  20 slice _INPUT_12 with indices ['T']
                _INPUT_12_sliced_offset = offset_now      
                _INPUT_12_sliced = np.ndarray((NVIR, (T_1-T_0)), buffer = buffer, offset = _INPUT_12_sliced_offset)
                size_item        = (NVIR * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_12.shape[0]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  21 aT,aQTR->QRaT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _INPUT_12_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((Q_1-Q_0) * ((R_1-R_0) * (NVIR * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((Q_1-Q_0), (R_1-R_0), NVIR, (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                            ctypes.c_void_p(_M7.ctypes.data),
                                            ctypes.c_void_p(_M8.ctypes.data),
                                            ctypes.c_int(_INPUT_12_sliced.shape[0]),
                                            ctypes.c_int(_INPUT_12_sliced.shape[1]),
                                            ctypes.c_int(_M7.shape[1]),
                                            ctypes.c_int(_M7.shape[3]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  22 QRaT->QRTa
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((Q_1-Q_0), (R_1-R_0), (T_1-T_0), NVIR), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int(NVIR),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 aS,QRTa->SQRT
                _M11_offset      = offset_now      
                _M11_offset      = min(_M11_offset, _M8_perm_offset)
                offset_now       = _M11_offset     
                ddot_buffer      = np.ndarray((NTHC_INT, (Q_1-Q_0), (R_1-R_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((Q_1-Q_0) * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M11             = np.ndarray((NTHC_INT, (Q_1-Q_0), (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M11_offset)
                offset_now       = (_M11_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
                _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_9_reshaped.T, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M11.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  24 SQRT->RQTS
                _M11_perm_offset = _M11_offset     
                _M11_perm        = np.ndarray(((R_1-R_0), (Q_1-Q_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M11_perm_offset)
                fn_permutation_0123_2130(ctypes.c_void_p(_M11.ctypes.data),
                                         ctypes.c_void_p(_M11_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 slice _INPUT_5 with indices ['R']
                _INPUT_5_sliced_offset = offset_now      
                _INPUT_5_sliced  = np.ndarray(((R_1-R_0), NTHC_INT), buffer = buffer, offset = _INPUT_5_sliced_offset)
                size_item        = ((R_1-R_0) * (NTHC_INT * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_0(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(R_0),
                             ctypes.c_int(R_1))
                # step  26 slice _M9 with indices ['Q', 'T']
                _M9_sliced_offset = offset_now      
                _M9_sliced       = np.ndarray((NTHC_INT, (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M9_sliced_offset)
                size_item        = (NTHC_INT * ((Q_1-Q_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M9_sliced.ctypes.data),
                               ctypes.c_int(_M9.shape[0]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_int(Q_0),
                               ctypes.c_int(Q_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  27 RS,SQT->RQTS
                _M10_offset      = offset_now      
                _M10_offset      = min(_M10_offset, _INPUT_5_sliced_offset)
                _M10_offset      = min(_M10_offset, _M9_sliced_offset)
                offset_now       = _M10_offset     
                tmp_itemsize     = ((R_1-R_0) * ((Q_1-Q_0) * ((T_1-T_0) * (NTHC_INT * _itemsize))))
                _M10             = np.ndarray(((R_1-R_0), (Q_1-Q_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M10_offset)
                offset_now       = (_M10_offset + tmp_itemsize)
                fn_contraction_01_123_0231(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                           ctypes.c_void_p(_M9_sliced.ctypes.data),
                                           ctypes.c_void_p(_M10.ctypes.data),
                                           ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                           ctypes.c_int(_M9_sliced.shape[1]),
                                           ctypes.c_int(_M9_sliced.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  28 RQTS,RQTS->
                output_tmp       = ctypes.c_double(0.0)
                fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                       ctypes.c_void_p(_M11_perm.ctypes.data),
                       ctypes.c_int(_M10.size),
                       ctypes.pointer(output_tmp))
                output_tmp = output_tmp.value
                _M12 += output_tmp
            # step  29 end   for loop with indices ('Q', 'R', 'T')
        # step  30 end   for loop with indices ('Q', 'R')
    # step  31 end   for loop with indices ('Q',)
    # clean the final forloop
    return _M12

def RMP2_K_forloop_Q_S_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       Q_bunchsize = 8,
                                                       S_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _INPUT_3_sliced
    tmp              = (NOCC * Q_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_8_sliced
    tmp              = (NOCC * S_bunchsize)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (Q_bunchsize * (S_bunchsize * NOCC))
    output           = max(output, tmp)
    # cmpr _INPUT_11_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (T_bunchsize * (Q_bunchsize * S_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (NTHC_INT * Q_bunchsize)
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (T_bunchsize * (S_bunchsize * Q_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M5_sliced
    tmp              = (NTHC_INT * (S_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (Q_bunchsize * (NTHC_INT * (T_bunchsize * S_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (Q_bunchsize * (NTHC_INT * (T_bunchsize * S_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NOCC * (Q_bunchsize * (T_bunchsize * S_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_10_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (Q_bunchsize * (S_bunchsize * (NOCC * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (Q_bunchsize * (S_bunchsize * (NOCC * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (Q_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (Q_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (NTHC_INT * S_bunchsize)
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M10_packed
    tmp              = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_Q_S_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                Q_bunchsize = 8,
                                                                S_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M7_size         = (NOCC * (Q_bunchsize * (T_bunchsize * S_bunchsize)))
    _M0_size         = (Q_bunchsize * (S_bunchsize * NOCC))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M8_size         = (Q_bunchsize * (S_bunchsize * (NOCC * T_bunchsize)))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_11_sliced_size = (NOCC * T_bunchsize)
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M6_size         = (Q_bunchsize * (NTHC_INT * (T_bunchsize * S_bunchsize)))
    _INPUT_3_sliced_size = (NOCC * Q_bunchsize)
    _M1_size         = (T_bunchsize * (Q_bunchsize * S_bunchsize))
    _M10_packed_size = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    _M5_sliced_size  = (NTHC_INT * (S_bunchsize * T_bunchsize))
    _INPUT_10_sliced_size = (NOCC * T_bunchsize)
    _INPUT_5_sliced_size = (NTHC_INT * S_bunchsize)
    _INPUT_8_sliced_size = (NOCC * S_bunchsize)
    _INPUT_0_sliced_size = (NTHC_INT * Q_bunchsize)
    _M2_size         = (NTHC_INT * (T_bunchsize * (S_bunchsize * Q_bunchsize)))
    _M9_size         = (NTHC_INT * (Q_bunchsize * (S_bunchsize * T_bunchsize)))
    _M4_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    # cmpr _M3_size
    size_now         = 0               
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M4_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M4_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _INPUT_3_sliced_size + _INPUT_8_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _INPUT_3_sliced_size)
    size_now         = (size_now + _INPUT_8_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _INPUT_11_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _INPUT_11_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M1_size + _INPUT_0_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M2_size + _M5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M2_size)
    size_now         = (size_now + _M5_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M6_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M7_size + _INPUT_10_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _INPUT_10_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M9_size + _INPUT_5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M5_size + _M0_size + _M10_packed_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M10_packed_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_Q_S_naive(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("bQ,bT->QTb"    , _INPUT_4        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bR,QTb->RQT"   , _INPUT_7        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aS,aT->STa"    , _INPUT_9        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("aP,STa->PST"   , _INPUT_2        , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("jQ,jS->QSj"    , _INPUT_3        , _INPUT_8        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("jT,QSj->TQS"   , _INPUT_11       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("PQ,TQS->PTSQ"  , _INPUT_0        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("PTSQ,PST->QPTS", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("iP,QTSP->iQTS" , _INPUT_1        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("iT,iQTS->QSiT" , _INPUT_10       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 1, 3, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("iR,QSTi->RQST" , _INPUT_6        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9_perm         = np.transpose(_M9             , (0, 1, 3, 2)    )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("RS,RQTS->RQT"  , _INPUT_5        , _M9_perm        )
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("RQT,RQT->"     , _M10            , _M11            )
    del _M10        
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_Q_S(Z           : np.ndarray,
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bR,QTb->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M3_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M3         
    del _M3_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 aS,aT->STa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 aP,STa->PST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 jQ,jS->QSj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, NTHC_INT, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jT,QSj->TQS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 PQ,TQS->PTSQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_213_0231 = getattr(libpbc, "fn_contraction_01_213_0231", None)
    assert fn_contraction_01_213_0231 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_contraction_01_213_0231(ctypes.c_void_p(_INPUT_0.ctypes.data),
                               ctypes.c_void_p(_M1.ctypes.data),
                               ctypes.c_void_p(_M2.ctypes.data),
                               ctypes.c_int(_INPUT_0.shape[0]),
                               ctypes.c_int(_INPUT_0.shape[1]),
                               ctypes.c_int(_M1.shape[0]),
                               ctypes.c_int(_M1.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 PTSQ,PST->QPTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012(ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_void_p(_M5.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_M2.shape[0]),
                                 ctypes.c_int(_M2.shape[1]),
                                 ctypes.c_int(_M2.shape[2]),
                                 ctypes.c_int(_M2.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 QPTS->QTSP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                             ctypes.c_void_p(_M6_perm.ctypes.data),
                             ctypes.c_int(_M6.shape[0]),
                             ctypes.c_int(_M6.shape[1]),
                             ctypes.c_int(_M6.shape[2]),
                             ctypes.c_int(_M6.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iP,QTSP->iQTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
    _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iT,iQTS->QSiT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                ctypes.c_void_p(_M7.ctypes.data),
                                ctypes.c_void_p(_M8.ctypes.data),
                                ctypes.c_int(_INPUT_10.shape[0]),
                                ctypes.c_int(_INPUT_10.shape[1]),
                                ctypes.c_int(_M7.shape[1]),
                                ctypes.c_int(_M7.shape[3]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 QSiT->QSTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NOCC, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 iR,QSTi->RQST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 RQST->RQTS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M9_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 RS,RQTS->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_5.ctypes.data),
                               ctypes.c_void_p(_M9_perm.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_5.shape[0]),
                               ctypes.c_int(_INPUT_5.shape[1]),
                               ctypes.c_int(_M9_perm.shape[1]),
                               ctypes.c_int(_M9_perm.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RQT,RQT-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    del _M10        
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_Q_S_forloop_Q_S(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   Q_bunchsize = 8,
                                   S_bunchsize = 8,
                                   T_bunchsize = 1):
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
    _INPUT_10        = tau_o           
    _INPUT_11        = tau_o           
    _INPUT_12        = tau_v           
    _INPUT_13        = tau_v           
    nthreads         = lib.num_threads()
    _M12             = 0.0             
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # fetch function pointers
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_contraction_01_213_0231 = getattr(libpbc, "fn_contraction_01_213_0231", None)
    assert fn_contraction_01_213_0231 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_0213_2301 = getattr(libpbc, "fn_contraction_01_0213_2301", None)
    assert fn_contraction_01_0213_2301 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_Q_S_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          Q_bunchsize = Q_bunchsize,
                                                                          S_bunchsize = S_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_Q_S_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   Q_bunchsize = Q_bunchsize,
                                                                                   S_bunchsize = S_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_Q         = None            
    offset_Q_S       = None            
    offset_Q_S_T     = None            
    # step   0 start for loop with indices ()
    # step   1 bQ,bT->QTb
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   2 bR,QTb->RQT
    _M11_offset      = offset_now      
    _M11_offset      = min(_M11_offset, _M3_offset)
    offset_now       = _M11_offset     
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    offset_now       = (_M11_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M3_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M11.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   3 allocate   _M10
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = offset_now)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    fn_clean(ctypes.c_void_p(_M10.ctypes.data),
             ctypes.c_int(_M10.size))
    # step   4 aS,aT->STa
    _M4_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 aP,STa->PST
    _M5_offset       = offset_now      
    _M5_offset       = min(_M5_offset, _M4_offset)
    offset_now       = _M5_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M4_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
            ctypes.c_void_p(_M5.ctypes.data),
            ctypes.c_int(ddot_buffer.size))
    # step   6 start for loop with indices ('Q',)
    for Q_0, Q_1 in lib.prange(0,NTHC_INT,Q_bunchsize):
        if offset_Q == None:
            offset_Q         = offset_now      
        else:
            offset_now       = offset_Q        
        # step   7 start for loop with indices ('Q', 'S')
        for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
            if offset_Q_S == None:
                offset_Q_S       = offset_now      
            else:
                offset_now       = offset_Q_S      
            # step   8 slice _INPUT_3 with indices ['Q']
            _INPUT_3_sliced_offset = offset_now      
            _INPUT_3_sliced  = np.ndarray((NOCC, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_3_sliced_offset)
            size_item        = (NOCC * ((Q_1-Q_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_3.ctypes.data),
                         ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_3.shape[0]),
                         ctypes.c_int(_INPUT_3.shape[1]),
                         ctypes.c_int(Q_0),
                         ctypes.c_int(Q_1))
            # step   9 slice _INPUT_8 with indices ['S']
            _INPUT_8_sliced_offset = offset_now      
            _INPUT_8_sliced  = np.ndarray((NOCC, (S_1-S_0)), buffer = buffer, offset = _INPUT_8_sliced_offset)
            size_item        = (NOCC * ((S_1-S_0) * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_1(ctypes.c_void_p(_INPUT_8.ctypes.data),
                         ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_8.shape[0]),
                         ctypes.c_int(_INPUT_8.shape[1]),
                         ctypes.c_int(S_0),
                         ctypes.c_int(S_1))
            # step  10 jQ,jS->QSj
            _M0_offset       = offset_now      
            _M0_offset       = min(_M0_offset, _INPUT_3_sliced_offset)
            _M0_offset       = min(_M0_offset, _INPUT_8_sliced_offset)
            offset_now       = _M0_offset      
            tmp_itemsize     = ((Q_1-Q_0) * ((S_1-S_0) * (NOCC * _itemsize)))
            _M0              = np.ndarray(((Q_1-Q_0), (S_1-S_0), NOCC), buffer = buffer, offset = _M0_offset)
            offset_now       = (_M0_offset + tmp_itemsize)
            fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                                     ctypes.c_void_p(_M0.ctypes.data),
                                     ctypes.c_int(_INPUT_3_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_3_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_8_sliced.shape[1]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  11 start for loop with indices ('Q', 'S', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_Q_S_T == None:
                    offset_Q_S_T     = offset_now      
                else:
                    offset_now       = offset_Q_S_T    
                # step  12 slice _INPUT_11 with indices ['T']
                _INPUT_11_sliced_offset = offset_now      
                _INPUT_11_sliced = np.ndarray((NOCC, (T_1-T_0)), buffer = buffer, offset = _INPUT_11_sliced_offset)
                size_item        = (NOCC * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_11.shape[0]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  13 jT,QSj->TQS
                _M1_offset       = offset_now      
                _M1_offset       = min(_M1_offset, _INPUT_11_sliced_offset)
                offset_now       = _M1_offset      
                ddot_buffer      = np.ndarray(((T_1-T_0), (Q_1-Q_0), (S_1-S_0)), buffer = linearop_buf)
                tmp_itemsize     = ((T_1-T_0) * ((Q_1-Q_0) * ((S_1-S_0) * _itemsize)))
                _M1              = np.ndarray(((T_1-T_0), (Q_1-Q_0), (S_1-S_0)), buffer = buffer, offset = _M1_offset)
                offset_now       = (_M1_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_11_sliced.shape[0]
                _INPUT_11_sliced_reshaped = _INPUT_11_sliced.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M0.shape[0]
                _size_dim_1      = _size_dim_1 * _M0.shape[1]
                _M0_reshaped = _M0.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_11_sliced_reshaped.T, _M0_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M1.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  14 slice _INPUT_0 with indices ['Q']
                _INPUT_0_sliced_offset = offset_now      
                _INPUT_0_sliced  = np.ndarray((NTHC_INT, (Q_1-Q_0)), buffer = buffer, offset = _INPUT_0_sliced_offset)
                size_item        = (NTHC_INT * ((Q_1-Q_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(Q_0),
                             ctypes.c_int(Q_1))
                # step  15 PQ,TQS->PTSQ
                _M2_offset       = offset_now      
                _M2_offset       = min(_M2_offset, _INPUT_0_sliced_offset)
                _M2_offset       = min(_M2_offset, _M1_offset)
                offset_now       = _M2_offset      
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((S_1-S_0) * ((Q_1-Q_0) * _itemsize))))
                _M2              = np.ndarray((NTHC_INT, (T_1-T_0), (S_1-S_0), (Q_1-Q_0)), buffer = buffer, offset = _M2_offset)
                offset_now       = (_M2_offset + tmp_itemsize)
                fn_contraction_01_213_0231(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                           ctypes.c_void_p(_M1.ctypes.data),
                                           ctypes.c_void_p(_M2.ctypes.data),
                                           ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                           ctypes.c_int(_M1.shape[0]),
                                           ctypes.c_int(_M1.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  16 slice _M5 with indices ['S', 'T']
                _M5_sliced_offset = offset_now      
                _M5_sliced       = np.ndarray((NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M5_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_sliced.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]),
                               ctypes.c_int(S_0),
                               ctypes.c_int(S_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  17 PTSQ,PST->QPTS
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M2_offset)
                _M6_offset       = min(_M6_offset, _M5_sliced_offset)
                offset_now       = _M6_offset      
                tmp_itemsize     = ((Q_1-Q_0) * (NTHC_INT * ((T_1-T_0) * ((S_1-S_0) * _itemsize))))
                _M6              = np.ndarray(((Q_1-Q_0), NTHC_INT, (T_1-T_0), (S_1-S_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                fn_contraction_0123_021_3012(ctypes.c_void_p(_M2.ctypes.data),
                                             ctypes.c_void_p(_M5_sliced.ctypes.data),
                                             ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_int(_M2.shape[0]),
                                             ctypes.c_int(_M2.shape[1]),
                                             ctypes.c_int(_M2.shape[2]),
                                             ctypes.c_int(_M2.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  18 QPTS->QTSP
                _M6_perm_offset  = _M6_offset      
                _M6_perm         = np.ndarray(((Q_1-Q_0), (T_1-T_0), (S_1-S_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 iP,QTSP->iQTS
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M6_perm_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NOCC, (Q_1-Q_0), (T_1-T_0), (S_1-S_0)), buffer = linearop_buf)
                tmp_itemsize     = (NOCC * ((Q_1-Q_0) * ((T_1-T_0) * ((S_1-S_0) * _itemsize))))
                _M7              = np.ndarray((NOCC, (Q_1-Q_0), (T_1-T_0), (S_1-S_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
                _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_1_reshaped, _M6_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M7.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  20 slice _INPUT_10 with indices ['T']
                _INPUT_10_sliced_offset = offset_now      
                _INPUT_10_sliced = np.ndarray((NOCC, (T_1-T_0)), buffer = buffer, offset = _INPUT_10_sliced_offset)
                size_item        = (NOCC * ((T_1-T_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_10.shape[0]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_int(T_0),
                             ctypes.c_int(T_1))
                # step  21 iT,iQTS->QSiT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _INPUT_10_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((Q_1-Q_0) * ((S_1-S_0) * (NOCC * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((Q_1-Q_0), (S_1-S_0), NOCC, (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_01_0213_2301(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                            ctypes.c_void_p(_M7.ctypes.data),
                                            ctypes.c_void_p(_M8.ctypes.data),
                                            ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                            ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                            ctypes.c_int(_M7.shape[1]),
                                            ctypes.c_int(_M7.shape[3]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  22 QSiT->QSTi
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((Q_1-Q_0), (S_1-S_0), (T_1-T_0), NOCC), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int(NOCC),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 iR,QSTi->RQST
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (Q_1-Q_0), (S_1-S_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((Q_1-Q_0) * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NTHC_INT, (Q_1-Q_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
                _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_6_reshaped.T, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                fn_copy(ctypes.c_void_p(ddot_buffer.ctypes.data),
                        ctypes.c_void_p(_M9.ctypes.data),
                        ctypes.c_int(ddot_buffer.size))
                # step  24 RQST->RQTS
                _M9_perm_offset  = _M9_offset      
                _M9_perm         = np.ndarray((NTHC_INT, (Q_1-Q_0), (T_1-T_0), (S_1-S_0)), buffer = buffer, offset = _M9_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M9.ctypes.data),
                                         ctypes.c_void_p(_M9_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 slice _INPUT_5 with indices ['S']
                _INPUT_5_sliced_offset = offset_now      
                _INPUT_5_sliced  = np.ndarray((NTHC_INT, (S_1-S_0)), buffer = buffer, offset = _INPUT_5_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * _itemsize))
                offset_now       = (offset_now + size_item)
                fn_slice_2_1(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(S_0),
                             ctypes.c_int(S_1))
                # step  26 RS,RQTS->RQT
                _M10_packed_offset = offset_now      
                _M10_packed_offset = min(_M10_packed_offset, _INPUT_5_sliced_offset)
                _M10_packed_offset = min(_M10_packed_offset, _M9_perm_offset)
                offset_now       = _M10_packed_offset
                tmp_itemsize     = (NTHC_INT * ((Q_1-Q_0) * ((T_1-T_0) * _itemsize)))
                _M10_packed      = np.ndarray((NTHC_INT, (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M10_packed_offset)
                offset_now       = (_M10_packed_offset + tmp_itemsize)
                fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                           ctypes.c_void_p(_M9_perm.ctypes.data),
                                           ctypes.c_void_p(_M10_packed.ctypes.data),
                                           ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                           ctypes.c_int(_M9_perm.shape[1]),
                                           ctypes.c_int(_M9_perm.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  27 pack  _M10 with indices ['Q', 'T']
                fn_packadd_3_1_2(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_packed.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
            # step  28 end   for loop with indices ('Q', 'S', 'T')
            # step  29 deallocate ['_M0']
        # step  30 end   for loop with indices ('Q', 'S')
    # step  31 end   for loop with indices ('Q',)
    # step  32 deallocate ['_M5']
    # step  33 RQT,RQT->
    _M12_offset      = offset_now      
    _M12_offset      = min(_M12_offset, _M10_offset)
    _M12_offset      = min(_M12_offset, _M11_offset)
    offset_now       = _M12_offset     
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(output_tmp))
    _M12 = output_tmp.value
    # clean the final forloop
    return _M12

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
    # test for RMP2_K_forloop_P_R and RMP2_K_forloop_P_R_naive
    benchmark        = RMP2_K_forloop_P_R_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_P_R(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_P_R_forloop_P_R
    output3          = RMP2_K_forloop_P_R_forloop_P_R(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_P_S and RMP2_K_forloop_P_S_naive
    benchmark        = RMP2_K_forloop_P_S_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_P_S(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_P_S_forloop_P_S
    output3          = RMP2_K_forloop_P_S_forloop_P_S(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_Q_R and RMP2_K_forloop_Q_R_naive
    benchmark        = RMP2_K_forloop_Q_R_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_Q_R(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_Q_R_forloop_Q_R
    output3          = RMP2_K_forloop_Q_R_forloop_Q_R(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_Q_S and RMP2_K_forloop_Q_S_naive
    benchmark        = RMP2_K_forloop_Q_S_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_Q_S(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_Q_S_forloop_Q_S
    output3          = RMP2_K_forloop_Q_S_forloop_Q_S(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
