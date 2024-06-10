import numpy
import numpy as np
import ctypes
import copy
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

def RMP2_K_forloop_P_b_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       P_bunchsize = 8,
                                                       b_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (P_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_4_sliced
    tmp              = (b_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (P_bunchsize * (b_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NOCC * (P_bunchsize * b_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_11_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (T_bunchsize * (P_bunchsize * (b_bunchsize * NOCC)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (T_bunchsize * (P_bunchsize * b_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (T_bunchsize * (P_bunchsize * b_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M3_sliced
    tmp              = (NTHC_INT * (T_bunchsize * b_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6_sliced
    tmp              = (NTHC_INT * (P_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (b_bunchsize * (P_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (b_bunchsize * (P_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (NTHC_INT * (b_bunchsize * (P_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (NTHC_INT * (b_bunchsize * (P_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M10_packed
    tmp              = (NTHC_INT * (P_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_P_b_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                P_bunchsize = 8,
                                                                b_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_11_sliced_size = (NOCC * T_bunchsize)
    _M1_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M3_sliced_size  = (NTHC_INT * (T_bunchsize * b_bunchsize))
    _M9_size         = (NTHC_INT * (T_bunchsize * (P_bunchsize * b_bunchsize)))
    _M8_size         = (NTHC_INT * (b_bunchsize * (P_bunchsize * T_bunchsize)))
    _INPUT_4_sliced_size = (b_bunchsize * NTHC_INT)
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M10_packed_size = (NTHC_INT * (P_bunchsize * T_bunchsize))
    _M0_size         = (P_bunchsize * (b_bunchsize * NTHC_INT))
    _M7_size         = (b_bunchsize * (P_bunchsize * (NTHC_INT * T_bunchsize)))
    _M5_size         = (T_bunchsize * (P_bunchsize * (b_bunchsize * NOCC)))
    _M6_sliced_size  = (NTHC_INT * (P_bunchsize * T_bunchsize))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NOCC * (P_bunchsize * b_bunchsize))
    _INPUT_0_sliced_size = (P_bunchsize * NTHC_INT)
    # cmpr _M2_size
    size_now         = 0               
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M1_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _INPUT_0_sliced_size + _INPUT_4_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    size_now         = (size_now + _INPUT_4_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M0_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M4_size + _INPUT_11_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _INPUT_11_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M4_size + _M5_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M4_size + _M9_size + _M3_sliced_size + _M6_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M3_sliced_size)
    size_now         = (size_now + _M6_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M4_size + _M9_size + _M7_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M7_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M4_size + _M9_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M6_size + _M3_size + _M4_size + _M10_packed_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M10_packed_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_P_b_naive(Z           : np.ndarray,
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
    _M2              = np.einsum("aP,aT->PTa"    , _INPUT_2        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,PTa->SPT"   , _INPUT_9        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iP,iT->PTi"    , _INPUT_1        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("iR,PTi->RPT"   , _INPUT_6        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("bR,bT->RTb"    , _INPUT_7        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("PQ,bQ->PbQ"    , _INPUT_0        , _INPUT_4        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("jQ,PbQ->jPb"   , _INPUT_3        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("jT,jPb->TPbj"  , _INPUT_11       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jS,TPbj->STPb" , _INPUT_8        , _M5             )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9_perm         = np.transpose(_M9             , (0, 2, 1, 3)    )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("RTb,RPT->bPRT" , _M3             , _M6             )
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 1, 3, 2)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("RS,bPTR->SbPT" , _INPUT_5        , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("SPTb,SPTb->SPT", _M8_perm        , _M9_perm        )
    del _M8_perm    
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("SPT,SPT->"     , _M10            , _M11            )
    del _M10        
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_P_b(Z           : np.ndarray,
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
    # step 0 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aS,PTa->SPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M2_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 iR,PTi->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M1_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bR,bT->RTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 PQ,bQ->PbQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    _buffer          = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jQ,PbQ->jPb 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NOCC, NTHC_INT, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped, _M0_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 jT,jPb->TPbj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    _buffer          = np.ndarray((N_LAPLACE, NTHC_INT, NVIR, NOCC), dtype=np.float64)
    _M5              = np.ndarray((N_LAPLACE, NTHC_INT, NVIR, NOCC), dtype=np.float64)
    fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_11.ctypes.data),
                               ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_int(_INPUT_11.shape[0]),
                               ctypes.c_int(_INPUT_11.shape[1]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 jS,TPbj->STPb 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _size_dim_1      = _size_dim_1 * _M5.shape[2]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M5_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M5         
    del _M5_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 STPb->SPTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0213 = getattr(libpbc, "fn_permutation_0123_0213", None)
    assert fn_permutation_0123_0213 is not None
    _buffer          = np.ndarray((nthreads, N_LAPLACE, NTHC_INT, NVIR), dtype=np.float64)
    _M9_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_0213(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RTb,RPT->bPRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    _buffer          = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M7              = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_031_2301(ctypes.c_void_p(_M3.ctypes.data),
                                ctypes.c_void_p(_M6.ctypes.data),
                                ctypes.c_void_p(_M7.ctypes.data),
                                ctypes.c_int(_M3.shape[0]),
                                ctypes.c_int(_M3.shape[1]),
                                ctypes.c_int(_M3.shape[2]),
                                ctypes.c_int(_M6.shape[1]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 bPRT->bPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M7_perm         = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M7.ctypes.data),
                             ctypes.c_void_p(_M7_perm.ctypes.data),
                             ctypes.c_int(_M7.shape[0]),
                             ctypes.c_int(_M7.shape[1]),
                             ctypes.c_int(_M7.shape[2]),
                             ctypes.c_int(_M7.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RS,bPTR->SbPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M7_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 SbPT->SPTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 SPTb,SPTb->SPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_012 = getattr(libpbc, "fn_contraction_0123_0123_012", None)
    assert fn_contraction_0123_0123_012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0123_012(ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_void_p(_M9_perm.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_M8_perm.shape[0]),
                                 ctypes.c_int(_M8_perm.shape[1]),
                                 ctypes.c_int(_M8_perm.shape[2]),
                                 ctypes.c_int(_M8_perm.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M8_perm    
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 SPT,SPT-> 
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

def RMP2_K_forloop_P_b_forloop_P_b(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   P_bunchsize = 8,
                                   b_bunchsize = 8,
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
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_0123_0123_012 = getattr(libpbc, "fn_contraction_0123_0123_012", None)
    assert fn_contraction_0123_0123_012 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_permutation_0123_0213 = getattr(libpbc, "fn_permutation_0123_0213", None)
    assert fn_permutation_0123_0213 is not None
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_P_b_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          P_bunchsize = P_bunchsize,
                                                                          b_bunchsize = b_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_P_b_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   P_bunchsize = P_bunchsize,
                                                                                   b_bunchsize = b_bunchsize,
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
    offset_P_b       = None            
    offset_P_b_T     = None            
    # step   0 start for loop with indices ()
    # step   1 aP,aT->PTa
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   2 aS,PTa->SPT
    _M11_offset      = offset_now      
    _M11_offset      = min(_M11_offset, _M2_offset)
    offset_now       = _M11_offset     
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    offset_now       = (_M11_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M11.ravel()[:] = ddot_buffer.ravel()[:]
    # step   3 allocate   _M10
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = offset_now)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M10.ravel()[:] = 0.0
    # step   4 iP,iT->PTi
    _M1_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M1_offset)
    offset_now       = (_M1_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 iR,PTi->RPT
    _M6_offset       = offset_now      
    _M6_offset       = min(_M6_offset, _M1_offset)
    offset_now       = _M6_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    offset_now       = (_M6_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M1_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M6.ravel()[:] = ddot_buffer.ravel()[:]
    # step   6 bR,bT->RTb
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   7 start for loop with indices ('P',)
    for P_0, P_1 in lib.prange(0,NTHC_INT,P_bunchsize):
        if offset_P == None:
            offset_P         = offset_now      
        else:
            offset_now       = offset_P        
        # step   8 start for loop with indices ('P', 'b')
        for b_0, b_1 in lib.prange(0,NVIR,b_bunchsize):
            if offset_P_b == None:
                offset_P_b       = offset_now      
            else:
                offset_now       = offset_P_b      
            # step   9 slice _INPUT_0 with indices ['P']
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
            # step  10 slice _INPUT_4 with indices ['b']
            _INPUT_4_sliced_offset = offset_now      
            _INPUT_4_sliced  = np.ndarray(((b_1-b_0), NTHC_INT), buffer = buffer, offset = _INPUT_4_sliced_offset)
            size_item        = ((b_1-b_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_4.ctypes.data),
                         ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_4.shape[0]),
                         ctypes.c_int(_INPUT_4.shape[1]),
                         ctypes.c_int(b_0),
                         ctypes.c_int(b_1))
            # step  11 PQ,bQ->PbQ
            _M0_offset       = offset_now      
            _M0_offset       = min(_M0_offset, _INPUT_0_sliced_offset)
            _M0_offset       = min(_M0_offset, _INPUT_4_sliced_offset)
            offset_now       = _M0_offset      
            tmp_itemsize     = ((P_1-P_0) * ((b_1-b_0) * (NTHC_INT * _itemsize)))
            _M0              = np.ndarray(((P_1-P_0), (b_1-b_0), NTHC_INT), buffer = buffer, offset = _M0_offset)
            offset_now       = (_M0_offset + tmp_itemsize)
            fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_4_sliced.ctypes.data),
                                     ctypes.c_void_p(_M0.ctypes.data),
                                     ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_4_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 jQ,PbQ->jPb
            _M4_offset       = offset_now      
            _M4_offset       = min(_M4_offset, _M0_offset)
            offset_now       = _M4_offset      
            ddot_buffer      = np.ndarray((NOCC, (P_1-P_0), (b_1-b_0)), buffer = linearop_buf)
            tmp_itemsize     = (NOCC * ((P_1-P_0) * ((b_1-b_0) * _itemsize)))
            _M4              = np.ndarray((NOCC, (P_1-P_0), (b_1-b_0)), buffer = buffer, offset = _M4_offset)
            offset_now       = (_M4_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
            _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M0.shape[0]
            _size_dim_1      = _size_dim_1 * _M0.shape[1]
            _M0_reshaped = _M0.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_3_reshaped, _M0_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M4.ravel()[:] = ddot_buffer.ravel()[:]
            # step  13 start for loop with indices ('P', 'b', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_P_b_T == None:
                    offset_P_b_T     = offset_now      
                else:
                    offset_now       = offset_P_b_T    
                # step  14 slice _INPUT_11 with indices ['T']
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
                # step  15 jT,jPb->TPbj
                _M5_offset       = offset_now      
                _M5_offset       = min(_M5_offset, _INPUT_11_sliced_offset)
                offset_now       = _M5_offset      
                tmp_itemsize     = ((T_1-T_0) * ((P_1-P_0) * ((b_1-b_0) * (NOCC * _itemsize))))
                _M5              = np.ndarray(((T_1-T_0), (P_1-P_0), (b_1-b_0), NOCC), buffer = buffer, offset = _M5_offset)
                offset_now       = (_M5_offset + tmp_itemsize)
                fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_void_p(_M5.ctypes.data),
                                           ctypes.c_int(_INPUT_11_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_11_sliced.shape[1]),
                                           ctypes.c_int(_M4.shape[1]),
                                           ctypes.c_int(_M4.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  16 jS,TPbj->STPb
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M5_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (T_1-T_0), (P_1-P_0), (b_1-b_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((P_1-P_0) * ((b_1-b_0) * _itemsize))))
                _M9              = np.ndarray((NTHC_INT, (T_1-T_0), (P_1-P_0), (b_1-b_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
                _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M5.shape[0]
                _size_dim_1      = _size_dim_1 * _M5.shape[1]
                _size_dim_1      = _size_dim_1 * _M5.shape[2]
                _M5_reshaped = _M5.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_8_reshaped.T, _M5_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  17 STPb->SPTb
                _M9_perm_offset  = _M9_offset      
                _M9_perm         = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0), (b_1-b_0)), buffer = buffer, offset = _M9_perm_offset)
                fn_permutation_0123_0213(ctypes.c_void_p(_M9.ctypes.data),
                                         ctypes.c_void_p(_M9_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((b_1-b_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  18 slice _M3 with indices ['T', 'b']
                _M3_sliced_offset = offset_now      
                _M3_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (b_1-b_0)), buffer = buffer, offset = _M3_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((b_1-b_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_sliced.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(b_0),
                               ctypes.c_int(b_1))
                # step  19 slice _M6 with indices ['P', 'T']
                _M6_sliced_offset = offset_now      
                _M6_sliced       = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M6_sliced_offset)
                size_item        = (NTHC_INT * ((P_1-P_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_sliced.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]),
                               ctypes.c_int(P_0),
                               ctypes.c_int(P_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  20 RTb,RPT->bPRT
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M3_sliced_offset)
                _M7_offset       = min(_M7_offset, _M6_sliced_offset)
                offset_now       = _M7_offset      
                tmp_itemsize     = ((b_1-b_0) * ((P_1-P_0) * (NTHC_INT * ((T_1-T_0) * _itemsize))))
                _M7              = np.ndarray(((b_1-b_0), (P_1-P_0), NTHC_INT, (T_1-T_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                fn_contraction_012_031_2301(ctypes.c_void_p(_M3_sliced.ctypes.data),
                                            ctypes.c_void_p(_M6_sliced.ctypes.data),
                                            ctypes.c_void_p(_M7.ctypes.data),
                                            ctypes.c_int(_M3_sliced.shape[0]),
                                            ctypes.c_int(_M3_sliced.shape[1]),
                                            ctypes.c_int(_M3_sliced.shape[2]),
                                            ctypes.c_int(_M6_sliced.shape[1]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  21 bPRT->bPTR
                _M7_perm_offset  = _M7_offset      
                _M7_perm         = np.ndarray(((b_1-b_0), (P_1-P_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M7.ctypes.data),
                                         ctypes.c_void_p(_M7_perm.ctypes.data),
                                         ctypes.c_int((b_1-b_0)),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  22 RS,bPTR->SbPT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M7_perm_offset)
                offset_now       = _M8_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (b_1-b_0), (P_1-P_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((b_1-b_0) * ((P_1-P_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray((NTHC_INT, (b_1-b_0), (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
                _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
                _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_5_reshaped.T, _M7_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M8.ravel()[:] = ddot_buffer.ravel()[:]
                # step  23 SbPT->SPTb
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0), (b_1-b_0)), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((b_1-b_0)),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  24 SPTb,SPTb->SPT
                _M10_packed_offset = offset_now      
                _M10_packed_offset = min(_M10_packed_offset, _M8_perm_offset)
                _M10_packed_offset = min(_M10_packed_offset, _M9_perm_offset)
                offset_now       = _M10_packed_offset
                tmp_itemsize     = (NTHC_INT * ((P_1-P_0) * ((T_1-T_0) * _itemsize)))
                _M10_packed      = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M10_packed_offset)
                offset_now       = (_M10_packed_offset + tmp_itemsize)
                fn_contraction_0123_0123_012(ctypes.c_void_p(_M8_perm.ctypes.data),
                                             ctypes.c_void_p(_M9_perm.ctypes.data),
                                             ctypes.c_void_p(_M10_packed.ctypes.data),
                                             ctypes.c_int(_M8_perm.shape[0]),
                                             ctypes.c_int(_M8_perm.shape[1]),
                                             ctypes.c_int(_M8_perm.shape[2]),
                                             ctypes.c_int(_M8_perm.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 pack  _M10 with indices ['P', 'T']
                fn_packadd_3_1_2(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_packed.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1))
            # step  26 end   for loop with indices ('P', 'b', 'T')
            # step  27 deallocate ['_M4']
        # step  28 end   for loop with indices ('P', 'b')
    # step  29 end   for loop with indices ('P',)
    # step  30 deallocate ['_M6', '_M3']
    # step  31 SPT,SPT->
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

def RMP2_K_forloop_P_j_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       P_bunchsize = 8,
                                                       j_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (P_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_3_sliced
    tmp              = (j_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (P_bunchsize * (j_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NVIR * (P_bunchsize * j_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_13_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (T_bunchsize * (P_bunchsize * (j_bunchsize * NVIR)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (T_bunchsize * (P_bunchsize * j_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6_sliced
    tmp              = (NTHC_INT * (P_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (j_bunchsize * (NTHC_INT * (P_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (j_bunchsize * (NTHC_INT * (P_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (j_bunchsize * (P_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (j_bunchsize * (P_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M3_sliced
    tmp              = (NTHC_INT * (T_bunchsize * j_bunchsize))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M10_packed
    tmp              = (NTHC_INT * (T_bunchsize * P_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_P_j_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                P_bunchsize = 8,
                                                                j_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_3_sliced_size = (j_bunchsize * NTHC_INT)
    _M1_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (j_bunchsize * (NTHC_INT * (P_bunchsize * T_bunchsize)))
    _M9_size         = (NTHC_INT * (j_bunchsize * (P_bunchsize * T_bunchsize)))
    _M3_sliced_size  = (NTHC_INT * (T_bunchsize * j_bunchsize))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_13_sliced_size = (NVIR * T_bunchsize)
    _M0_size         = (P_bunchsize * (j_bunchsize * NTHC_INT))
    _M10_packed_size = (NTHC_INT * (T_bunchsize * P_bunchsize))
    _M7_size         = (NTHC_INT * (T_bunchsize * (P_bunchsize * j_bunchsize)))
    _M5_size         = (T_bunchsize * (P_bunchsize * (j_bunchsize * NVIR)))
    _M6_sliced_size  = (NTHC_INT * (P_bunchsize * T_bunchsize))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M10_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M4_size         = (NVIR * (P_bunchsize * j_bunchsize))
    _INPUT_0_sliced_size = (P_bunchsize * NTHC_INT)
    # cmpr _M2_size
    size_now         = 0               
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M1_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _INPUT_0_sliced_size + _INPUT_3_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    size_now         = (size_now + _INPUT_3_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M0_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M4_size + _INPUT_13_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _INPUT_13_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M4_size + _M5_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M4_size + _M7_size + _M6_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M6_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M4_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M4_size + _M9_size + _M3_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M3_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M6_size + _M4_size + _M10_packed_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M10_packed_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_P_j_naive(Z           : np.ndarray,
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
    _M2              = np.einsum("aP,aT->PTa"    , _INPUT_2        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,PTa->SPT"   , _INPUT_9        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 2, 1)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("jS,jT->STj"    , _INPUT_8        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iP,iT->PTi"    , _INPUT_1        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("iR,PTi->RPT"   , _INPUT_6        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("PQ,jQ->PjQ"    , _INPUT_0        , _INPUT_3        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("bQ,PjQ->bPj"   , _INPUT_4        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("bT,bPj->TPjb"  , _INPUT_13       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("bR,TPjb->RTPj" , _INPUT_7        , _M5             )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("RPT,RTPj->jRPT", _M6             , _M7             )
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("RS,jPTR->SjPT" , _INPUT_5        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9_perm         = np.transpose(_M9             , (0, 3, 2, 1)    )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("STj,STPj->STP" , _M3             , _M9_perm        )
    del _M3         
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("STP,STP->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_P_j(Z           : np.ndarray,
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
    # step 0 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aS,PTa->SPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M2_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 SPT->STP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M11_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jS,jT->STj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iR,PTi->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M1_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 PQ,jQ->PjQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    _buffer          = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 bQ,PjQ->bPj 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NVIR, NTHC_INT, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped, _M0_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 bT,bPj->TPjb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    _buffer          = np.ndarray((N_LAPLACE, NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _M5              = np.ndarray((N_LAPLACE, NTHC_INT, NOCC, NVIR), dtype=np.float64)
    fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_13.ctypes.data),
                               ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_int(_INPUT_13.shape[0]),
                               ctypes.c_int(_INPUT_13.shape[1]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 bR,TPjb->RTPj 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _size_dim_1      = _size_dim_1 * _M5.shape[2]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M5_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M5         
    del _M5_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RPT,RTPj->jRPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0213_3012 = getattr(libpbc, "fn_contraction_012_0213_3012", None)
    assert fn_contraction_012_0213_3012 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0213_3012(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M7.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jRPT->jPTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RS,jPTR->SjPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 SjPT->STPj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0321 = getattr(libpbc, "fn_permutation_0123_0321", None)
    assert fn_permutation_0123_0321 is not None
    _buffer          = np.ndarray((nthreads, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M9_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_0321(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 STj,STPj->STP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0132_013 = getattr(libpbc, "fn_contraction_012_0132_013", None)
    assert fn_contraction_012_0132_013 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_012_0132_013(ctypes.c_void_p(_M3.ctypes.data),
                                ctypes.c_void_p(_M9_perm.ctypes.data),
                                ctypes.c_void_p(_M10.ctypes.data),
                                ctypes.c_int(_M3.shape[0]),
                                ctypes.c_int(_M3.shape[1]),
                                ctypes.c_int(_M3.shape[2]),
                                ctypes.c_int(_M9_perm.shape[2]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M3         
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 STP,STP-> 
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

def RMP2_K_forloop_P_j_forloop_P_j(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   P_bunchsize = 8,
                                   j_bunchsize = 8,
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
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_permutation_0123_0321 = getattr(libpbc, "fn_permutation_0123_0321", None)
    assert fn_permutation_0123_0321 is not None
    fn_contraction_012_0132_013 = getattr(libpbc, "fn_contraction_012_0132_013", None)
    assert fn_contraction_012_0132_013 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_012_0213_3012 = getattr(libpbc, "fn_contraction_012_0213_3012", None)
    assert fn_contraction_012_0213_3012 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_P_j_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          P_bunchsize = P_bunchsize,
                                                                          j_bunchsize = j_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_P_j_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   P_bunchsize = P_bunchsize,
                                                                                   j_bunchsize = j_bunchsize,
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
    offset_P_j       = None            
    offset_P_j_T     = None            
    # step   0 start for loop with indices ()
    # step   1 aP,aT->PTa
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   2 aS,PTa->SPT
    _M11_offset      = offset_now      
    _M11_offset      = min(_M11_offset, _M2_offset)
    offset_now       = _M11_offset     
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    offset_now       = (_M11_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M11.ravel()[:] = ddot_buffer.ravel()[:]
    # step   3 allocate   _M10
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NTHC_INT * _itemsize)))
    _M10_offset      = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M10.ravel()[:] = 0.0
    # step   4 SPT->STP
    _M11_perm_offset = _M11_offset     
    _M11_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M11_perm_offset)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(NTHC_INT),
                           ctypes.c_int(NTHC_INT),
                           ctypes.c_int(N_LAPLACE),
                           ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 jS,jT->STj
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   6 iP,iT->PTi
    _M1_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M1_offset)
    offset_now       = (_M1_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   7 iR,PTi->RPT
    _M6_offset       = offset_now      
    _M6_offset       = min(_M6_offset, _M1_offset)
    offset_now       = _M6_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    offset_now       = (_M6_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M1_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M6.ravel()[:] = ddot_buffer.ravel()[:]
    # step   8 start for loop with indices ('P',)
    for P_0, P_1 in lib.prange(0,NTHC_INT,P_bunchsize):
        if offset_P == None:
            offset_P         = offset_now      
        else:
            offset_now       = offset_P        
        # step   9 start for loop with indices ('P', 'j')
        for j_0, j_1 in lib.prange(0,NOCC,j_bunchsize):
            if offset_P_j == None:
                offset_P_j       = offset_now      
            else:
                offset_now       = offset_P_j      
            # step  10 slice _INPUT_0 with indices ['P']
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
            # step  11 slice _INPUT_3 with indices ['j']
            _INPUT_3_sliced_offset = offset_now      
            _INPUT_3_sliced  = np.ndarray(((j_1-j_0), NTHC_INT), buffer = buffer, offset = _INPUT_3_sliced_offset)
            size_item        = ((j_1-j_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_3.ctypes.data),
                         ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_3.shape[0]),
                         ctypes.c_int(_INPUT_3.shape[1]),
                         ctypes.c_int(j_0),
                         ctypes.c_int(j_1))
            # step  12 PQ,jQ->PjQ
            _M0_offset       = offset_now      
            _M0_offset       = min(_M0_offset, _INPUT_0_sliced_offset)
            _M0_offset       = min(_M0_offset, _INPUT_3_sliced_offset)
            offset_now       = _M0_offset      
            tmp_itemsize     = ((P_1-P_0) * ((j_1-j_0) * (NTHC_INT * _itemsize)))
            _M0              = np.ndarray(((P_1-P_0), (j_1-j_0), NTHC_INT), buffer = buffer, offset = _M0_offset)
            offset_now       = (_M0_offset + tmp_itemsize)
            fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_3_sliced.ctypes.data),
                                     ctypes.c_void_p(_M0.ctypes.data),
                                     ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_3_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  13 bQ,PjQ->bPj
            _M4_offset       = offset_now      
            _M4_offset       = min(_M4_offset, _M0_offset)
            offset_now       = _M4_offset      
            ddot_buffer      = np.ndarray((NVIR, (P_1-P_0), (j_1-j_0)), buffer = linearop_buf)
            tmp_itemsize     = (NVIR * ((P_1-P_0) * ((j_1-j_0) * _itemsize)))
            _M4              = np.ndarray((NVIR, (P_1-P_0), (j_1-j_0)), buffer = buffer, offset = _M4_offset)
            offset_now       = (_M4_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
            _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M0.shape[0]
            _size_dim_1      = _size_dim_1 * _M0.shape[1]
            _M0_reshaped = _M0.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_4_reshaped, _M0_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M4.ravel()[:] = ddot_buffer.ravel()[:]
            # step  14 start for loop with indices ('P', 'j', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_P_j_T == None:
                    offset_P_j_T     = offset_now      
                else:
                    offset_now       = offset_P_j_T    
                # step  15 slice _INPUT_13 with indices ['T']
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
                # step  16 bT,bPj->TPjb
                _M5_offset       = offset_now      
                _M5_offset       = min(_M5_offset, _INPUT_13_sliced_offset)
                offset_now       = _M5_offset      
                tmp_itemsize     = ((T_1-T_0) * ((P_1-P_0) * ((j_1-j_0) * (NVIR * _itemsize))))
                _M5              = np.ndarray(((T_1-T_0), (P_1-P_0), (j_1-j_0), NVIR), buffer = buffer, offset = _M5_offset)
                offset_now       = (_M5_offset + tmp_itemsize)
                fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_void_p(_M5.ctypes.data),
                                           ctypes.c_int(_INPUT_13_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_13_sliced.shape[1]),
                                           ctypes.c_int(_M4.shape[1]),
                                           ctypes.c_int(_M4.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  17 bR,TPjb->RTPj
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M5_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (T_1-T_0), (P_1-P_0), (j_1-j_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((P_1-P_0) * ((j_1-j_0) * _itemsize))))
                _M7              = np.ndarray((NTHC_INT, (T_1-T_0), (P_1-P_0), (j_1-j_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
                _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M5.shape[0]
                _size_dim_1      = _size_dim_1 * _M5.shape[1]
                _size_dim_1      = _size_dim_1 * _M5.shape[2]
                _M5_reshaped = _M5.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_7_reshaped.T, _M5_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M7.ravel()[:] = ddot_buffer.ravel()[:]
                # step  18 slice _M6 with indices ['P', 'T']
                _M6_sliced_offset = offset_now      
                _M6_sliced       = np.ndarray((NTHC_INT, (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M6_sliced_offset)
                size_item        = (NTHC_INT * ((P_1-P_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_sliced.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]),
                               ctypes.c_int(P_0),
                               ctypes.c_int(P_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  19 RPT,RTPj->jRPT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M6_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((j_1-j_0) * (NTHC_INT * ((P_1-P_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((j_1-j_0), NTHC_INT, (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_012_0213_3012(ctypes.c_void_p(_M6_sliced.ctypes.data),
                                             ctypes.c_void_p(_M7.ctypes.data),
                                             ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_int(_M6_sliced.shape[0]),
                                             ctypes.c_int(_M6_sliced.shape[1]),
                                             ctypes.c_int(_M6_sliced.shape[2]),
                                             ctypes.c_int(_M7.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 jRPT->jPTR
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((j_1-j_0), (P_1-P_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((j_1-j_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  21 RS,jPTR->SjPT
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (j_1-j_0), (P_1-P_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((j_1-j_0) * ((P_1-P_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NTHC_INT, (j_1-j_0), (P_1-P_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
                _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_5_reshaped.T, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  22 SjPT->STPj
                _M9_perm_offset  = _M9_offset      
                _M9_perm         = np.ndarray((NTHC_INT, (T_1-T_0), (P_1-P_0), (j_1-j_0)), buffer = buffer, offset = _M9_perm_offset)
                fn_permutation_0123_0321(ctypes.c_void_p(_M9.ctypes.data),
                                         ctypes.c_void_p(_M9_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((j_1-j_0)),
                                         ctypes.c_int((P_1-P_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 slice _M3 with indices ['T', 'j']
                _M3_sliced_offset = offset_now      
                _M3_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (j_1-j_0)), buffer = buffer, offset = _M3_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((j_1-j_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_sliced.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(j_0),
                               ctypes.c_int(j_1))
                # step  24 STj,STPj->STP
                _M10_packed_offset = offset_now      
                _M10_packed_offset = min(_M10_packed_offset, _M3_sliced_offset)
                _M10_packed_offset = min(_M10_packed_offset, _M9_perm_offset)
                offset_now       = _M10_packed_offset
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((P_1-P_0) * _itemsize)))
                _M10_packed      = np.ndarray((NTHC_INT, (T_1-T_0), (P_1-P_0)), buffer = buffer, offset = _M10_packed_offset)
                offset_now       = (_M10_packed_offset + tmp_itemsize)
                fn_contraction_012_0132_013(ctypes.c_void_p(_M3_sliced.ctypes.data),
                                            ctypes.c_void_p(_M9_perm.ctypes.data),
                                            ctypes.c_void_p(_M10_packed.ctypes.data),
                                            ctypes.c_int(_M3_sliced.shape[0]),
                                            ctypes.c_int(_M3_sliced.shape[1]),
                                            ctypes.c_int(_M3_sliced.shape[2]),
                                            ctypes.c_int(_M9_perm.shape[2]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 pack  _M10 with indices ['P', 'T']
                fn_packadd_3_1_2(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_packed.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1),
                                 ctypes.c_int(P_0),
                                 ctypes.c_int(P_1))
            # step  26 end   for loop with indices ('P', 'j', 'T')
            # step  27 deallocate ['_M4']
        # step  28 end   for loop with indices ('P', 'j')
    # step  29 end   for loop with indices ('P',)
    # step  30 deallocate ['_M6', '_M3']
    # step  31 STP,STP->
    _M12_offset      = offset_now      
    _M12_offset      = min(_M12_offset, _M10_offset)
    _M12_offset      = min(_M12_offset, _M11_perm_offset)
    offset_now       = _M12_offset     
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11_perm.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(output_tmp))
    _M12 = output_tmp.value
    # clean the final forloop
    return _M12

def RMP2_K_forloop_Q_a_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       Q_bunchsize = 8,
                                                       a_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M1
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (NTHC_INT * Q_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_2_sliced
    tmp              = (a_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (Q_bunchsize * (a_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NOCC * (Q_bunchsize * a_bunchsize))
    output           = max(output, tmp)
    # cmpr _INPUT_10_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (T_bunchsize * (Q_bunchsize * (a_bunchsize * NOCC)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (T_bunchsize * (Q_bunchsize * a_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7_sliced
    tmp              = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (a_bunchsize * (NTHC_INT * (T_bunchsize * Q_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (a_bunchsize * (NTHC_INT * (T_bunchsize * Q_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (a_bunchsize * (T_bunchsize * Q_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (a_bunchsize * (T_bunchsize * Q_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M3_sliced
    tmp              = (NTHC_INT * (T_bunchsize * a_bunchsize))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M10_packed
    tmp              = (NTHC_INT * (T_bunchsize * Q_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_Q_a_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                Q_bunchsize = 8,
                                                                a_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (NTHC_INT * (T_bunchsize * (Q_bunchsize * a_bunchsize)))
    _M3_sliced_size  = (NTHC_INT * (T_bunchsize * a_bunchsize))
    _M1_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (a_bunchsize * (NTHC_INT * (T_bunchsize * Q_bunchsize)))
    _M7_sliced_size  = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    _M9_size         = (NTHC_INT * (a_bunchsize * (T_bunchsize * Q_bunchsize)))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_10_sliced_size = (NOCC * T_bunchsize)
    _INPUT_0_sliced_size = (NTHC_INT * Q_bunchsize)
    _INPUT_2_sliced_size = (a_bunchsize * NTHC_INT)
    _M0_size         = (Q_bunchsize * (a_bunchsize * NTHC_INT))
    _M10_packed_size = (NTHC_INT * (T_bunchsize * Q_bunchsize))
    _M5_size         = (T_bunchsize * (Q_bunchsize * (a_bunchsize * NOCC)))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M10_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M4_size         = (NOCC * (Q_bunchsize * a_bunchsize))
    _M7_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    # cmpr _M1_size
    size_now         = 0               
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M2_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _INPUT_0_sliced_size + _INPUT_2_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    size_now         = (size_now + _INPUT_2_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M0_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M4_size + _INPUT_10_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _INPUT_10_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M4_size + _M5_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M4_size + _M6_size + _M7_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M7_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M4_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M4_size + _M9_size + _M3_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M3_sliced_size)
    output           = max(output, size_now)
    # cmpr _M11_size + _M10_size + _M3_size + _M7_size + _M4_size + _M10_packed_size
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M10_packed_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_Q_a_naive(Z           : np.ndarray,
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
    _M1              = np.einsum("jQ,jT->QTj"    , _INPUT_3        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("jS,QTj->SQT"   , _INPUT_8        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 2, 1)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("aS,aT->STa"    , _INPUT_9        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("bQ,bT->QTb"    , _INPUT_4        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("bR,QTb->RQT"   , _INPUT_7        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("PQ,aP->QaP"    , _INPUT_0        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iP,QaP->iQa"   , _INPUT_1        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("iT,iQa->TQai"  , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("iR,TQai->RTQa" , _INPUT_6        , _M5             )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("RTQa,RQT->aRTQ", _M6             , _M7             )
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("RS,aTQR->SaTQ" , _INPUT_5        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9_perm         = np.transpose(_M9             , (0, 2, 3, 1)    )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("STa,STQa->STQ" , _M3             , _M9_perm        )
    del _M3         
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("STQ,STQ->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_Q_a(Z           : np.ndarray,
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
    # step 0 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M1_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 SQT->STQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M11_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 aS,aT->STa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 bR,QTb->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M2_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 PQ,aP->QaP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,QaP->iQa 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NOCC, NTHC_INT, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped, _M0_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iT,iQa->TQai 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    _buffer          = np.ndarray((N_LAPLACE, NTHC_INT, NVIR, NOCC), dtype=np.float64)
    _M5              = np.ndarray((N_LAPLACE, NTHC_INT, NVIR, NOCC), dtype=np.float64)
    fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_10.ctypes.data),
                               ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_int(_INPUT_10.shape[0]),
                               ctypes.c_int(_INPUT_10.shape[1]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iR,TQai->RTQa 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _size_dim_1      = _size_dim_1 * _M5.shape[2]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M5_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5         
    del _M5_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RTQa,RQT->aRTQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    _buffer          = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M8              = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_021_3012(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 aRTQ->aTQR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M8_perm         = np.ndarray((NVIR, N_LAPLACE, NTHC_INT, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RS,aTQR->SaTQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NVIR, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 SaTQ->STQa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NVIR, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M9_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NVIR), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 STa,STQa->STQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0132_013 = getattr(libpbc, "fn_contraction_012_0132_013", None)
    assert fn_contraction_012_0132_013 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_012_0132_013(ctypes.c_void_p(_M3.ctypes.data),
                                ctypes.c_void_p(_M9_perm.ctypes.data),
                                ctypes.c_void_p(_M10.ctypes.data),
                                ctypes.c_int(_M3.shape[0]),
                                ctypes.c_int(_M3.shape[1]),
                                ctypes.c_int(_M3.shape[2]),
                                ctypes.c_int(_M9_perm.shape[2]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M3         
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 STQ,STQ-> 
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

def RMP2_K_forloop_Q_a_forloop_Q_a(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   Q_bunchsize = 8,
                                   a_bunchsize = 8,
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
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    fn_contraction_012_0132_013 = getattr(libpbc, "fn_contraction_012_0132_013", None)
    assert fn_contraction_012_0132_013 is not None
    fn_contraction_0123_021_3012 = getattr(libpbc, "fn_contraction_0123_021_3012", None)
    assert fn_contraction_0123_021_3012 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_packadd_3_1_2 = getattr(libpbc, "fn_packadd_3_1_2", None)
    assert fn_packadd_3_1_2 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_Q_a_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          Q_bunchsize = Q_bunchsize,
                                                                          a_bunchsize = a_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_Q_a_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   Q_bunchsize = Q_bunchsize,
                                                                                   a_bunchsize = a_bunchsize,
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
    offset_Q_a       = None            
    offset_Q_a_T     = None            
    # step   0 start for loop with indices ()
    # step   1 jQ,jT->QTj
    _M1_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M1_offset)
    offset_now       = (_M1_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   2 jS,QTj->SQT
    _M11_offset      = offset_now      
    _M11_offset      = min(_M11_offset, _M1_offset)
    offset_now       = _M11_offset     
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    offset_now       = (_M11_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M1_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M11.ravel()[:] = ddot_buffer.ravel()[:]
    # step   3 allocate   _M10
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = offset_now)
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NTHC_INT * _itemsize)))
    _M10_offset      = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M10.ravel()[:] = 0.0
    # step   4 SQT->STQ
    _M11_perm_offset = _M11_offset     
    _M11_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M11_perm_offset)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(NTHC_INT),
                           ctypes.c_int(NTHC_INT),
                           ctypes.c_int(N_LAPLACE),
                           ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 aS,aT->STa
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   6 bQ,bT->QTb
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   7 bR,QTb->RQT
    _M7_offset       = offset_now      
    _M7_offset       = min(_M7_offset, _M2_offset)
    offset_now       = _M7_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    offset_now       = (_M7_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M7.ravel()[:] = ddot_buffer.ravel()[:]
    # step   8 start for loop with indices ('Q',)
    for Q_0, Q_1 in lib.prange(0,NTHC_INT,Q_bunchsize):
        if offset_Q == None:
            offset_Q         = offset_now      
        else:
            offset_now       = offset_Q        
        # step   9 start for loop with indices ('Q', 'a')
        for a_0, a_1 in lib.prange(0,NVIR,a_bunchsize):
            if offset_Q_a == None:
                offset_Q_a       = offset_now      
            else:
                offset_now       = offset_Q_a      
            # step  10 slice _INPUT_0 with indices ['Q']
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
            # step  11 slice _INPUT_2 with indices ['a']
            _INPUT_2_sliced_offset = offset_now      
            _INPUT_2_sliced  = np.ndarray(((a_1-a_0), NTHC_INT), buffer = buffer, offset = _INPUT_2_sliced_offset)
            size_item        = ((a_1-a_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_2.ctypes.data),
                         ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_2.shape[0]),
                         ctypes.c_int(_INPUT_2.shape[1]),
                         ctypes.c_int(a_0),
                         ctypes.c_int(a_1))
            # step  12 PQ,aP->QaP
            _M0_offset       = offset_now      
            _M0_offset       = min(_M0_offset, _INPUT_0_sliced_offset)
            _M0_offset       = min(_M0_offset, _INPUT_2_sliced_offset)
            offset_now       = _M0_offset      
            tmp_itemsize     = ((Q_1-Q_0) * ((a_1-a_0) * (NTHC_INT * _itemsize)))
            _M0              = np.ndarray(((Q_1-Q_0), (a_1-a_0), NTHC_INT), buffer = buffer, offset = _M0_offset)
            offset_now       = (_M0_offset + tmp_itemsize)
            fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_2_sliced.ctypes.data),
                                     ctypes.c_void_p(_M0.ctypes.data),
                                     ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_2_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  13 iP,QaP->iQa
            _M4_offset       = offset_now      
            _M4_offset       = min(_M4_offset, _M0_offset)
            offset_now       = _M4_offset      
            ddot_buffer      = np.ndarray((NOCC, (Q_1-Q_0), (a_1-a_0)), buffer = linearop_buf)
            tmp_itemsize     = (NOCC * ((Q_1-Q_0) * ((a_1-a_0) * _itemsize)))
            _M4              = np.ndarray((NOCC, (Q_1-Q_0), (a_1-a_0)), buffer = buffer, offset = _M4_offset)
            offset_now       = (_M4_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
            _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M0.shape[0]
            _size_dim_1      = _size_dim_1 * _M0.shape[1]
            _M0_reshaped = _M0.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_1_reshaped, _M0_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M4.ravel()[:] = ddot_buffer.ravel()[:]
            # step  14 start for loop with indices ('Q', 'a', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_Q_a_T == None:
                    offset_Q_a_T     = offset_now      
                else:
                    offset_now       = offset_Q_a_T    
                # step  15 slice _INPUT_10 with indices ['T']
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
                # step  16 iT,iQa->TQai
                _M5_offset       = offset_now      
                _M5_offset       = min(_M5_offset, _INPUT_10_sliced_offset)
                offset_now       = _M5_offset      
                tmp_itemsize     = ((T_1-T_0) * ((Q_1-Q_0) * ((a_1-a_0) * (NOCC * _itemsize))))
                _M5              = np.ndarray(((T_1-T_0), (Q_1-Q_0), (a_1-a_0), NOCC), buffer = buffer, offset = _M5_offset)
                offset_now       = (_M5_offset + tmp_itemsize)
                fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_void_p(_M5.ctypes.data),
                                           ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                           ctypes.c_int(_M4.shape[1]),
                                           ctypes.c_int(_M4.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  17 iR,TQai->RTQa
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M5_offset)
                offset_now       = _M6_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (T_1-T_0), (Q_1-Q_0), (a_1-a_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((Q_1-Q_0) * ((a_1-a_0) * _itemsize))))
                _M6              = np.ndarray((NTHC_INT, (T_1-T_0), (Q_1-Q_0), (a_1-a_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
                _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M5.shape[0]
                _size_dim_1      = _size_dim_1 * _M5.shape[1]
                _size_dim_1      = _size_dim_1 * _M5.shape[2]
                _M5_reshaped = _M5.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_6_reshaped.T, _M5_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M6.ravel()[:] = ddot_buffer.ravel()[:]
                # step  18 slice _M7 with indices ['Q', 'T']
                _M7_sliced_offset = offset_now      
                _M7_sliced       = np.ndarray((NTHC_INT, (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M7_sliced_offset)
                size_item        = (NTHC_INT * ((Q_1-Q_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M7.ctypes.data),
                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                               ctypes.c_int(_M7.shape[0]),
                               ctypes.c_int(_M7.shape[1]),
                               ctypes.c_int(_M7.shape[2]),
                               ctypes.c_int(Q_0),
                               ctypes.c_int(Q_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  19 RTQa,RQT->aRTQ
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M6_offset)
                _M8_offset       = min(_M8_offset, _M7_sliced_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((a_1-a_0) * (NTHC_INT * ((T_1-T_0) * ((Q_1-Q_0) * _itemsize))))
                _M8              = np.ndarray(((a_1-a_0), NTHC_INT, (T_1-T_0), (Q_1-Q_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_0123_021_3012(ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_void_p(_M7_sliced.ctypes.data),
                                             ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_int(_M6.shape[0]),
                                             ctypes.c_int(_M6.shape[1]),
                                             ctypes.c_int(_M6.shape[2]),
                                             ctypes.c_int(_M6.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 aRTQ->aTQR
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((a_1-a_0), (T_1-T_0), (Q_1-Q_0), NTHC_INT), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((a_1-a_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  21 RS,aTQR->SaTQ
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (a_1-a_0), (T_1-T_0), (Q_1-Q_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((a_1-a_0) * ((T_1-T_0) * ((Q_1-Q_0) * _itemsize))))
                _M9              = np.ndarray((NTHC_INT, (a_1-a_0), (T_1-T_0), (Q_1-Q_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
                _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_5_reshaped.T, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  22 SaTQ->STQa
                _M9_perm_offset  = _M9_offset      
                _M9_perm         = np.ndarray((NTHC_INT, (T_1-T_0), (Q_1-Q_0), (a_1-a_0)), buffer = buffer, offset = _M9_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M9.ctypes.data),
                                         ctypes.c_void_p(_M9_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((a_1-a_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 slice _M3 with indices ['T', 'a']
                _M3_sliced_offset = offset_now      
                _M3_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (a_1-a_0)), buffer = buffer, offset = _M3_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((a_1-a_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_sliced.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(a_0),
                               ctypes.c_int(a_1))
                # step  24 STa,STQa->STQ
                _M10_packed_offset = offset_now      
                _M10_packed_offset = min(_M10_packed_offset, _M3_sliced_offset)
                _M10_packed_offset = min(_M10_packed_offset, _M9_perm_offset)
                offset_now       = _M10_packed_offset
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((Q_1-Q_0) * _itemsize)))
                _M10_packed      = np.ndarray((NTHC_INT, (T_1-T_0), (Q_1-Q_0)), buffer = buffer, offset = _M10_packed_offset)
                offset_now       = (_M10_packed_offset + tmp_itemsize)
                fn_contraction_012_0132_013(ctypes.c_void_p(_M3_sliced.ctypes.data),
                                            ctypes.c_void_p(_M9_perm.ctypes.data),
                                            ctypes.c_void_p(_M10_packed.ctypes.data),
                                            ctypes.c_int(_M3_sliced.shape[0]),
                                            ctypes.c_int(_M3_sliced.shape[1]),
                                            ctypes.c_int(_M3_sliced.shape[2]),
                                            ctypes.c_int(_M9_perm.shape[2]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 pack  _M10 with indices ['Q', 'T']
                fn_packadd_3_1_2(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_packed.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(T_0),
                                 ctypes.c_int(T_1),
                                 ctypes.c_int(Q_0),
                                 ctypes.c_int(Q_1))
            # step  26 end   for loop with indices ('Q', 'a', 'T')
            # step  27 deallocate ['_M4']
        # step  28 end   for loop with indices ('Q', 'a')
    # step  29 end   for loop with indices ('Q',)
    # step  30 deallocate ['_M7', '_M3']
    # step  31 STQ,STQ->
    _M12_offset      = offset_now      
    _M12_offset      = min(_M12_offset, _M10_offset)
    _M12_offset      = min(_M12_offset, _M11_perm_offset)
    offset_now       = _M12_offset     
    output_tmp       = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11_perm.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(output_tmp))
    _M12 = output_tmp.value
    # clean the final forloop
    return _M12

def RMP2_K_forloop_Q_i_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       Q_bunchsize = 8,
                                                       i_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M1
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _INPUT_0_sliced
    tmp              = (NTHC_INT * Q_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_1_sliced
    tmp              = (i_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (Q_bunchsize * (i_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NVIR * (Q_bunchsize * i_bunchsize))
    output           = max(output, tmp)
    # cmpr _M3_sliced
    tmp              = (NTHC_INT * (T_bunchsize * i_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6_sliced
    tmp              = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (i_bunchsize * (Q_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (i_bunchsize * (Q_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (NTHC_INT * (i_bunchsize * (Q_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9_sliced
    tmp              = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (i_bunchsize * (NTHC_INT * (Q_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_12_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (T_bunchsize * (Q_bunchsize * (i_bunchsize * NVIR)))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (T_bunchsize * (Q_bunchsize * i_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (T_bunchsize * (Q_bunchsize * i_bunchsize)))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_Q_i_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                Q_bunchsize = 8,
                                                                i_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M3_sliced_size  = (NTHC_INT * (T_bunchsize * i_bunchsize))
    _M1_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (NTHC_INT * (i_bunchsize * (Q_bunchsize * T_bunchsize)))
    _M9_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (T_bunchsize * (Q_bunchsize * i_bunchsize)))
    _INPUT_1_sliced_size = (i_bunchsize * NTHC_INT)
    _M0_size         = (Q_bunchsize * (i_bunchsize * NTHC_INT))
    _M7_size         = (i_bunchsize * (Q_bunchsize * (NTHC_INT * T_bunchsize)))
    _M5_size         = (T_bunchsize * (Q_bunchsize * (i_bunchsize * NVIR)))
    _M6_sliced_size  = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    _INPUT_12_sliced_size = (NVIR * T_bunchsize)
    _M3_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M9_sliced_size  = (NTHC_INT * (Q_bunchsize * T_bunchsize))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M10_size        = (i_bunchsize * (NTHC_INT * (Q_bunchsize * T_bunchsize)))
    _M4_size         = (NVIR * (Q_bunchsize * i_bunchsize))
    _INPUT_0_sliced_size = (NTHC_INT * Q_bunchsize)
    # cmpr _M1_size
    size_now         = 0               
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M2_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _INPUT_0_sliced_size + _INPUT_1_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _INPUT_0_sliced_size)
    size_now         = (size_now + _INPUT_1_sliced_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M0_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M4_size + _M3_sliced_size + _M6_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M3_sliced_size)
    size_now         = (size_now + _M6_sliced_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M4_size + _M7_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M7_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M4_size + _M8_size + _M9_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M8_size)
    size_now         = (size_now + _M9_sliced_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M4_size + _M10_size + _INPUT_12_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _INPUT_12_sliced_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M4_size + _M10_size + _M5_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M9_size + _M6_size + _M3_size + _M4_size + _M10_size + _M11_size
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M11_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_Q_i_naive(Z           : np.ndarray,
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
    _M1              = np.einsum("jQ,jT->QTj"    , _INPUT_3        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jS,QTj->SQT"   , _INPUT_8        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("bQ,bT->QTb"    , _INPUT_4        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("bR,QTb->RQT"   , _INPUT_7        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("iR,iT->RTi"    , _INPUT_6        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("PQ,iP->QiP"    , _INPUT_0        , _INPUT_1        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,QiP->aQi"   , _INPUT_2        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("RTi,RQT->iQRT" , _M3             , _M6             )
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 1, 3, 2)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("RS,iQTR->SiQT" , _INPUT_5        , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("SiQT,SQT->iSQT", _M8             , _M9             )
    del _M8         
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("aT,aQi->TQia"  , _INPUT_12       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,TQia->STQi" , _INPUT_9        , _M5             )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (3, 0, 2, 1)    )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("iSQT,iSQT->"   , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    return _M12

def RMP2_K_forloop_Q_i(Z           : np.ndarray,
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
    # step 0 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M1_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bR,QTb->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M2_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iR,iT->RTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 PQ,iP->QiP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_0.ctypes.data),
                             ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_0.shape[0]),
                             ctypes.c_int(_INPUT_0.shape[1]),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 aP,QiP->aQi 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NVIR, NTHC_INT, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped, _M0_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 RTi,RQT->iQRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M7              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_031_2301(ctypes.c_void_p(_M3.ctypes.data),
                                ctypes.c_void_p(_M6.ctypes.data),
                                ctypes.c_void_p(_M7.ctypes.data),
                                ctypes.c_int(_M3.shape[0]),
                                ctypes.c_int(_M3.shape[1]),
                                ctypes.c_int(_M3.shape[2]),
                                ctypes.c_int(_M6.shape[1]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iQRT->iQTR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M7_perm         = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M7.ctypes.data),
                             ctypes.c_void_p(_M7_perm.ctypes.data),
                             ctypes.c_int(_M7.shape[0]),
                             ctypes.c_int(_M7.shape[1]),
                             ctypes.c_int(_M7.shape[2]),
                             ctypes.c_int(_M7.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,iQTR->SiQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M7_perm_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M7_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 SiQT,SQT->iSQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_023_1023 = getattr(libpbc, "fn_contraction_0123_023_1023", None)
    assert fn_contraction_0123_023_1023 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M10             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_023_1023(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 aT,aQi->TQia 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    _buffer          = np.ndarray((N_LAPLACE, NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _M5              = np.ndarray((N_LAPLACE, NTHC_INT, NOCC, NVIR), dtype=np.float64)
    fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_12.ctypes.data),
                               ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_int(_INPUT_12.shape[0]),
                               ctypes.c_int(_INPUT_12.shape[1]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aS,TQia->STQi 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _size_dim_1      = _size_dim_1 * _M5.shape[2]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M5_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M5         
    del _M5_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 STQi->iSQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_3021 = getattr(libpbc, "fn_permutation_0123_3021", None)
    assert fn_permutation_0123_3021 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    _M11_perm        = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_3021(ctypes.c_void_p(_M11.ctypes.data),
                             ctypes.c_void_p(_M11_perm.ctypes.data),
                             ctypes.c_int(_M11.shape[0]),
                             ctypes.c_int(_M11.shape[1]),
                             ctypes.c_int(_M11.shape[2]),
                             ctypes.c_int(_M11.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 iSQT,iSQT-> 
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
    _benchmark_time(t1, t2, "step 15")
    return _M12

def RMP2_K_forloop_Q_i_forloop_Q_i(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   Q_bunchsize = 8,
                                   i_bunchsize = 8,
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
    fn_contraction_01_023_1230 = getattr(libpbc, "fn_contraction_01_023_1230", None)
    assert fn_contraction_01_023_1230 is not None
    fn_permutation_0123_3021 = getattr(libpbc, "fn_permutation_0123_3021", None)
    assert fn_permutation_0123_3021 is not None
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_0123_023_1023 = getattr(libpbc, "fn_contraction_0123_023_1023", None)
    assert fn_contraction_0123_023_1023 is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_Q_i_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          Q_bunchsize = Q_bunchsize,
                                                                          i_bunchsize = i_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_Q_i_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   Q_bunchsize = Q_bunchsize,
                                                                                   i_bunchsize = i_bunchsize,
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
    offset_Q_i       = None            
    offset_Q_i_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 jQ,jT->QTj
    _M1_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M1_offset)
    offset_now       = (_M1_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 jS,QTj->SQT
    _M9_offset       = offset_now      
    _M9_offset       = min(_M9_offset, _M1_offset)
    offset_now       = _M9_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    offset_now       = (_M9_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M1_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M9.ravel()[:] = ddot_buffer.ravel()[:]
    # step   4 bQ,bT->QTb
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 bR,QTb->RQT
    _M6_offset       = offset_now      
    _M6_offset       = min(_M6_offset, _M2_offset)
    offset_now       = _M6_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    offset_now       = (_M6_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M6.ravel()[:] = ddot_buffer.ravel()[:]
    # step   6 iR,iT->RTi
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   7 start for loop with indices ('Q',)
    for Q_0, Q_1 in lib.prange(0,NTHC_INT,Q_bunchsize):
        if offset_Q == None:
            offset_Q         = offset_now      
        else:
            offset_now       = offset_Q        
        # step   8 start for loop with indices ('Q', 'i')
        for i_0, i_1 in lib.prange(0,NOCC,i_bunchsize):
            if offset_Q_i == None:
                offset_Q_i       = offset_now      
            else:
                offset_now       = offset_Q_i      
            # step   9 slice _INPUT_0 with indices ['Q']
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
            # step  10 slice _INPUT_1 with indices ['i']
            _INPUT_1_sliced_offset = offset_now      
            _INPUT_1_sliced  = np.ndarray(((i_1-i_0), NTHC_INT), buffer = buffer, offset = _INPUT_1_sliced_offset)
            size_item        = ((i_1-i_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_1.ctypes.data),
                         ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_1.shape[0]),
                         ctypes.c_int(_INPUT_1.shape[1]),
                         ctypes.c_int(i_0),
                         ctypes.c_int(i_1))
            # step  11 PQ,iP->QiP
            _M0_offset       = offset_now      
            _M0_offset       = min(_M0_offset, _INPUT_0_sliced_offset)
            _M0_offset       = min(_M0_offset, _INPUT_1_sliced_offset)
            offset_now       = _M0_offset      
            tmp_itemsize     = ((Q_1-Q_0) * ((i_1-i_0) * (NTHC_INT * _itemsize)))
            _M0              = np.ndarray(((Q_1-Q_0), (i_1-i_0), NTHC_INT), buffer = buffer, offset = _M0_offset)
            offset_now       = (_M0_offset + tmp_itemsize)
            fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_0_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_1_sliced.ctypes.data),
                                     ctypes.c_void_p(_M0.ctypes.data),
                                     ctypes.c_int(_INPUT_0_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_0_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_1_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 aP,QiP->aQi
            _M4_offset       = offset_now      
            _M4_offset       = min(_M4_offset, _M0_offset)
            offset_now       = _M4_offset      
            ddot_buffer      = np.ndarray((NVIR, (Q_1-Q_0), (i_1-i_0)), buffer = linearop_buf)
            tmp_itemsize     = (NVIR * ((Q_1-Q_0) * ((i_1-i_0) * _itemsize)))
            _M4              = np.ndarray((NVIR, (Q_1-Q_0), (i_1-i_0)), buffer = buffer, offset = _M4_offset)
            offset_now       = (_M4_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
            _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M0.shape[0]
            _size_dim_1      = _size_dim_1 * _M0.shape[1]
            _M0_reshaped = _M0.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_2_reshaped, _M0_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M4.ravel()[:] = ddot_buffer.ravel()[:]
            # step  13 start for loop with indices ('Q', 'i', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_Q_i_T == None:
                    offset_Q_i_T     = offset_now      
                else:
                    offset_now       = offset_Q_i_T    
                # step  14 slice _M3 with indices ['T', 'i']
                _M3_sliced_offset = offset_now      
                _M3_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (i_1-i_0)), buffer = buffer, offset = _M3_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((i_1-i_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_sliced.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(i_0),
                               ctypes.c_int(i_1))
                # step  15 slice _M6 with indices ['Q', 'T']
                _M6_sliced_offset = offset_now      
                _M6_sliced       = np.ndarray((NTHC_INT, (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M6_sliced_offset)
                size_item        = (NTHC_INT * ((Q_1-Q_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_sliced.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]),
                               ctypes.c_int(Q_0),
                               ctypes.c_int(Q_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  16 RTi,RQT->iQRT
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M3_sliced_offset)
                _M7_offset       = min(_M7_offset, _M6_sliced_offset)
                offset_now       = _M7_offset      
                tmp_itemsize     = ((i_1-i_0) * ((Q_1-Q_0) * (NTHC_INT * ((T_1-T_0) * _itemsize))))
                _M7              = np.ndarray(((i_1-i_0), (Q_1-Q_0), NTHC_INT, (T_1-T_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                fn_contraction_012_031_2301(ctypes.c_void_p(_M3_sliced.ctypes.data),
                                            ctypes.c_void_p(_M6_sliced.ctypes.data),
                                            ctypes.c_void_p(_M7.ctypes.data),
                                            ctypes.c_int(_M3_sliced.shape[0]),
                                            ctypes.c_int(_M3_sliced.shape[1]),
                                            ctypes.c_int(_M3_sliced.shape[2]),
                                            ctypes.c_int(_M6_sliced.shape[1]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  17 iQRT->iQTR
                _M7_perm_offset  = _M7_offset      
                _M7_perm         = np.ndarray(((i_1-i_0), (Q_1-Q_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M7.ctypes.data),
                                         ctypes.c_void_p(_M7_perm.ctypes.data),
                                         ctypes.c_int((i_1-i_0)),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  18 RS,iQTR->SiQT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M7_perm_offset)
                offset_now       = _M8_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (i_1-i_0), (Q_1-Q_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((i_1-i_0) * ((Q_1-Q_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray((NTHC_INT, (i_1-i_0), (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
                _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
                _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_5_reshaped.T, _M7_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M8.ravel()[:] = ddot_buffer.ravel()[:]
                # step  19 slice _M9 with indices ['Q', 'T']
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
                # step  20 SiQT,SQT->iSQT
                _M10_offset      = offset_now      
                _M10_offset      = min(_M10_offset, _M8_offset)
                _M10_offset      = min(_M10_offset, _M9_sliced_offset)
                offset_now       = _M10_offset     
                tmp_itemsize     = ((i_1-i_0) * (NTHC_INT * ((Q_1-Q_0) * ((T_1-T_0) * _itemsize))))
                _M10             = np.ndarray(((i_1-i_0), NTHC_INT, (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M10_offset)
                offset_now       = (_M10_offset + tmp_itemsize)
                fn_contraction_0123_023_1023(ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_void_p(_M9_sliced.ctypes.data),
                                             ctypes.c_void_p(_M10.ctypes.data),
                                             ctypes.c_int(_M8.shape[0]),
                                             ctypes.c_int(_M8.shape[1]),
                                             ctypes.c_int(_M8.shape[2]),
                                             ctypes.c_int(_M8.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  21 slice _INPUT_12 with indices ['T']
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
                # step  22 aT,aQi->TQia
                _M5_offset       = offset_now      
                _M5_offset       = min(_M5_offset, _INPUT_12_sliced_offset)
                offset_now       = _M5_offset      
                tmp_itemsize     = ((T_1-T_0) * ((Q_1-Q_0) * ((i_1-i_0) * (NVIR * _itemsize))))
                _M5              = np.ndarray(((T_1-T_0), (Q_1-Q_0), (i_1-i_0), NVIR), buffer = buffer, offset = _M5_offset)
                offset_now       = (_M5_offset + tmp_itemsize)
                fn_contraction_01_023_1230(ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                           ctypes.c_void_p(_M4.ctypes.data),
                                           ctypes.c_void_p(_M5.ctypes.data),
                                           ctypes.c_int(_INPUT_12_sliced.shape[0]),
                                           ctypes.c_int(_INPUT_12_sliced.shape[1]),
                                           ctypes.c_int(_M4.shape[1]),
                                           ctypes.c_int(_M4.shape[2]),
                                           ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 aS,TQia->STQi
                _M11_offset      = offset_now      
                _M11_offset      = min(_M11_offset, _M5_offset)
                offset_now       = _M11_offset     
                ddot_buffer      = np.ndarray((NTHC_INT, (T_1-T_0), (Q_1-Q_0), (i_1-i_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((T_1-T_0) * ((Q_1-Q_0) * ((i_1-i_0) * _itemsize))))
                _M11             = np.ndarray((NTHC_INT, (T_1-T_0), (Q_1-Q_0), (i_1-i_0)), buffer = buffer, offset = _M11_offset)
                offset_now       = (_M11_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
                _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M5.shape[0]
                _size_dim_1      = _size_dim_1 * _M5.shape[1]
                _size_dim_1      = _size_dim_1 * _M5.shape[2]
                _M5_reshaped = _M5.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_9_reshaped.T, _M5_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M11.ravel()[:] = ddot_buffer.ravel()[:]
                # step  24 STQi->iSQT
                _M11_perm_offset = _M11_offset     
                _M11_perm        = np.ndarray(((i_1-i_0), NTHC_INT, (Q_1-Q_0), (T_1-T_0)), buffer = buffer, offset = _M11_perm_offset)
                fn_permutation_0123_3021(ctypes.c_void_p(_M11.ctypes.data),
                                         ctypes.c_void_p(_M11_perm.ctypes.data),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_int((Q_1-Q_0)),
                                         ctypes.c_int((i_1-i_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  25 iSQT,iSQT->
                output_tmp       = ctypes.c_double(0.0)
                fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                       ctypes.c_void_p(_M11_perm.ctypes.data),
                       ctypes.c_int(_M10.size),
                       ctypes.pointer(output_tmp))
                output_tmp = output_tmp.value
                _M12 += output_tmp
            # step  26 end   for loop with indices ('Q', 'i', 'T')
        # step  27 end   for loop with indices ('Q', 'i')
    # step  28 end   for loop with indices ('Q',)
    # clean the final forloop
    return _M12

def RMP2_K_forloop_R_a_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       R_bunchsize = 8,
                                                       a_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (R_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_9_sliced
    tmp              = (a_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (R_bunchsize * (a_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NOCC * (R_bunchsize * a_bunchsize))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NOCC * (R_bunchsize * a_bunchsize))
    output           = max(output, tmp)
    # cmpr _M0_sliced
    tmp              = (NTHC_INT * (T_bunchsize * a_bunchsize))
    output           = max(output, tmp)
    # cmpr _M4_sliced
    tmp              = (NTHC_INT * (R_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (a_bunchsize * (R_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (a_bunchsize * (R_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (a_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7_sliced
    tmp              = (NTHC_INT * (R_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (a_bunchsize * (NTHC_INT * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (a_bunchsize * (NTHC_INT * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NOCC * (a_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_11_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NOCC * (a_bunchsize * R_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_R_a_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                R_bunchsize = 8,
                                                                a_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (NTHC_INT * (a_bunchsize * (R_bunchsize * T_bunchsize)))
    _INPUT_11_sliced_size = (NOCC * T_bunchsize)
    _M1_size         = (R_bunchsize * (a_bunchsize * NTHC_INT))
    _M8_size         = (a_bunchsize * (NTHC_INT * (R_bunchsize * T_bunchsize)))
    _M7_sliced_size  = (NTHC_INT * (R_bunchsize * T_bunchsize))
    _M9_size         = (NOCC * (a_bunchsize * (R_bunchsize * T_bunchsize)))
    _M11_size        = (NOCC * (R_bunchsize * a_bunchsize))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M4_sliced_size  = (NTHC_INT * (R_bunchsize * T_bunchsize))
    _M5_size         = (a_bunchsize * (R_bunchsize * (NTHC_INT * T_bunchsize)))
    _M0_sliced_size  = (NTHC_INT * (T_bunchsize * a_bunchsize))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _INPUT_9_sliced_size = (a_bunchsize * NTHC_INT)
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NOCC * (a_bunchsize * R_bunchsize))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_5_sliced_size = (R_bunchsize * NTHC_INT)
    _M7_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    # cmpr _M3_size
    size_now         = 0               
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M2_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _INPUT_5_sliced_size + _INPUT_9_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    size_now         = (size_now + _INPUT_9_sliced_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M1_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M0_sliced_size + _M4_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M0_sliced_size)
    size_now         = (size_now + _M4_sliced_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M5_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M6_size + _M7_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M7_sliced_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M9_size + _INPUT_11_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_11_sliced_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_R_a_naive(Z           : np.ndarray,
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
    _M3              = np.einsum("bR,bT->RTb"    , _INPUT_7        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("bQ,RTb->QRT"   , _INPUT_4        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iR,iT->RTi"    , _INPUT_6        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iP,RTi->PRT"   , _INPUT_1        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("aP,aT->PTa"    , _INPUT_2        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("RS,aS->RaS"    , _INPUT_5        , _INPUT_9        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("jS,RaS->jRa"   , _INPUT_8        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 2, 1)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PTa,PRT->aRPT" , _M0             , _M4             )
    del _M0         
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 1, 3, 2)    )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("PQ,aRTP->QaRT" , _INPUT_0        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("QaRT,QRT->aQRT", _M6             , _M7             )
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jQ,aRTQ->jaRT" , _INPUT_3        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("jT,jaRT->jaR"  , _INPUT_11       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("jaR,jaR->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_R_a(Z           : np.ndarray,
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
    # step 0 bR,bT->RTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bQ,RTb->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M3_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M3         
    del _M3_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 iR,iT->RTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 iP,RTi->PRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M2_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,aS->RaS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    _buffer          = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jS,RaS->jRa 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NOCC, NTHC_INT, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped, _M1_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 jRa->jaR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NVIR), dtype=np.float64)
    _M11_perm        = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PTa,PRT->aRPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    _buffer          = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M5              = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_031_2301(ctypes.c_void_p(_M0.ctypes.data),
                                ctypes.c_void_p(_M4.ctypes.data),
                                ctypes.c_void_p(_M5.ctypes.data),
                                ctypes.c_int(_M0.shape[0]),
                                ctypes.c_int(_M0.shape[1]),
                                ctypes.c_int(_M0.shape[2]),
                                ctypes.c_int(_M4.shape[1]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M0         
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aRPT->aRTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M5_perm         = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M5.ctypes.data),
                             ctypes.c_void_p(_M5_perm.ctypes.data),
                             ctypes.c_int(_M5.shape[0]),
                             ctypes.c_int(_M5.shape[1]),
                             ctypes.c_int(_M5.shape[2]),
                             ctypes.c_int(_M5.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 PQ,aRTP->QaRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[2]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 QaRT,QRT->aQRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_023_1023 = getattr(libpbc, "fn_contraction_0123_023_1023", None)
    assert fn_contraction_0123_023_1023 is not None
    _buffer          = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_023_1023(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 aQRT->aRTQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 jQ,aRTQ->jaRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NOCC, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jT,jaRT->jaR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    _buffer          = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_11.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_11.shape[0]),
                               ctypes.c_int(_INPUT_11.shape[1]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 jaR,jaR-> 
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

def RMP2_K_forloop_R_a_forloop_R_a(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   R_bunchsize = 8,
                                   a_bunchsize = 8,
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
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_0123_023_1023 = getattr(libpbc, "fn_contraction_0123_023_1023", None)
    assert fn_contraction_0123_023_1023 is not None
    fn_contraction_01_0231_023_plus = getattr(libpbc, "fn_contraction_01_0231_023_plus", None)
    assert fn_contraction_01_0231_023_plus is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_R_a_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          R_bunchsize = R_bunchsize,
                                                                          a_bunchsize = a_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_R_a_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   R_bunchsize = R_bunchsize,
                                                                                   a_bunchsize = a_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_R         = None            
    offset_R_a       = None            
    offset_R_a_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 bR,bT->RTb
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 bQ,RTb->QRT
    _M7_offset       = offset_now      
    _M7_offset       = min(_M7_offset, _M3_offset)
    offset_now       = _M7_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    offset_now       = (_M7_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M3_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M7.ravel()[:] = ddot_buffer.ravel()[:]
    # step   4 iR,iT->RTi
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 iP,RTi->PRT
    _M4_offset       = offset_now      
    _M4_offset       = min(_M4_offset, _M2_offset)
    offset_now       = _M4_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M4.ravel()[:] = ddot_buffer.ravel()[:]
    # step   6 aP,aT->PTa
    _M0_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    offset_now       = (_M0_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   7 start for loop with indices ('R',)
    for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
        if offset_R == None:
            offset_R         = offset_now      
        else:
            offset_now       = offset_R        
        # step   8 start for loop with indices ('R', 'a')
        for a_0, a_1 in lib.prange(0,NVIR,a_bunchsize):
            if offset_R_a == None:
                offset_R_a       = offset_now      
            else:
                offset_now       = offset_R_a      
            # step   9 slice _INPUT_5 with indices ['R']
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
            # step  10 slice _INPUT_9 with indices ['a']
            _INPUT_9_sliced_offset = offset_now      
            _INPUT_9_sliced  = np.ndarray(((a_1-a_0), NTHC_INT), buffer = buffer, offset = _INPUT_9_sliced_offset)
            size_item        = ((a_1-a_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_9.ctypes.data),
                         ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_9.shape[0]),
                         ctypes.c_int(_INPUT_9.shape[1]),
                         ctypes.c_int(a_0),
                         ctypes.c_int(a_1))
            # step  11 RS,aS->RaS
            _M1_offset       = offset_now      
            _M1_offset       = min(_M1_offset, _INPUT_5_sliced_offset)
            _M1_offset       = min(_M1_offset, _INPUT_9_sliced_offset)
            offset_now       = _M1_offset      
            tmp_itemsize     = ((R_1-R_0) * ((a_1-a_0) * (NTHC_INT * _itemsize)))
            _M1              = np.ndarray(((R_1-R_0), (a_1-a_0), NTHC_INT), buffer = buffer, offset = _M1_offset)
            offset_now       = (_M1_offset + tmp_itemsize)
            fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_9_sliced.ctypes.data),
                                     ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_9_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 jS,RaS->jRa
            _M11_offset      = offset_now      
            _M11_offset      = min(_M11_offset, _M1_offset)
            offset_now       = _M11_offset     
            ddot_buffer      = np.ndarray((NOCC, (R_1-R_0), (a_1-a_0)), buffer = linearop_buf)
            tmp_itemsize     = (NOCC * ((R_1-R_0) * ((a_1-a_0) * _itemsize)))
            _M11             = np.ndarray((NOCC, (R_1-R_0), (a_1-a_0)), buffer = buffer, offset = _M11_offset)
            offset_now       = (_M11_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
            _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_8_reshaped, _M1_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M11.ravel()[:] = ddot_buffer.ravel()[:]
            # step  13 allocate   _M10
            _M10             = np.ndarray((NOCC, (a_1-a_0), (R_1-R_0)), buffer = buffer, offset = offset_now)
            tmp_itemsize     = (NOCC * ((a_1-a_0) * ((R_1-R_0) * _itemsize)))
            _M10_offset      = offset_now      
            offset_now       = (offset_now + tmp_itemsize)
            _M10.ravel()[:] = 0.0
            # step  14 jRa->jaR
            _M11_perm_offset = _M11_offset     
            _M11_perm        = np.ndarray((NOCC, (a_1-a_0), (R_1-R_0)), buffer = buffer, offset = _M11_perm_offset)
            fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(NOCC),
                                   ctypes.c_int((R_1-R_0)),
                                   ctypes.c_int((a_1-a_0)),
                                   ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  15 start for loop with indices ('R', 'a', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_R_a_T == None:
                    offset_R_a_T     = offset_now      
                else:
                    offset_now       = offset_R_a_T    
                # step  16 slice _M0 with indices ['T', 'a']
                _M0_sliced_offset = offset_now      
                _M0_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (a_1-a_0)), buffer = buffer, offset = _M0_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((a_1-a_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M0.ctypes.data),
                               ctypes.c_void_p(_M0_sliced.ctypes.data),
                               ctypes.c_int(_M0.shape[0]),
                               ctypes.c_int(_M0.shape[1]),
                               ctypes.c_int(_M0.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(a_0),
                               ctypes.c_int(a_1))
                # step  17 slice _M4 with indices ['R', 'T']
                _M4_sliced_offset = offset_now      
                _M4_sliced       = np.ndarray((NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M4_sliced_offset)
                size_item        = (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_sliced.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_int(R_0),
                               ctypes.c_int(R_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  18 PTa,PRT->aRPT
                _M5_offset       = offset_now      
                _M5_offset       = min(_M5_offset, _M0_sliced_offset)
                _M5_offset       = min(_M5_offset, _M4_sliced_offset)
                offset_now       = _M5_offset      
                tmp_itemsize     = ((a_1-a_0) * ((R_1-R_0) * (NTHC_INT * ((T_1-T_0) * _itemsize))))
                _M5              = np.ndarray(((a_1-a_0), (R_1-R_0), NTHC_INT, (T_1-T_0)), buffer = buffer, offset = _M5_offset)
                offset_now       = (_M5_offset + tmp_itemsize)
                fn_contraction_012_031_2301(ctypes.c_void_p(_M0_sliced.ctypes.data),
                                            ctypes.c_void_p(_M4_sliced.ctypes.data),
                                            ctypes.c_void_p(_M5.ctypes.data),
                                            ctypes.c_int(_M0_sliced.shape[0]),
                                            ctypes.c_int(_M0_sliced.shape[1]),
                                            ctypes.c_int(_M0_sliced.shape[2]),
                                            ctypes.c_int(_M4_sliced.shape[1]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 aRPT->aRTP
                _M5_perm_offset  = _M5_offset      
                _M5_perm         = np.ndarray(((a_1-a_0), (R_1-R_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M5.ctypes.data),
                                         ctypes.c_void_p(_M5_perm.ctypes.data),
                                         ctypes.c_int((a_1-a_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 PQ,aRTP->QaRT
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M5_perm_offset)
                offset_now       = _M6_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (a_1-a_0), (R_1-R_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((a_1-a_0) * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M6              = np.ndarray((NTHC_INT, (a_1-a_0), (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
                _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M5_perm.shape[2]
                _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_0_reshaped.T, _M5_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M6.ravel()[:] = ddot_buffer.ravel()[:]
                # step  21 slice _M7 with indices ['R', 'T']
                _M7_sliced_offset = offset_now      
                _M7_sliced       = np.ndarray((NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M7_sliced_offset)
                size_item        = (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M7.ctypes.data),
                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                               ctypes.c_int(_M7.shape[0]),
                               ctypes.c_int(_M7.shape[1]),
                               ctypes.c_int(_M7.shape[2]),
                               ctypes.c_int(R_0),
                               ctypes.c_int(R_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  22 QaRT,QRT->aQRT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M6_offset)
                _M8_offset       = min(_M8_offset, _M7_sliced_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((a_1-a_0) * (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((a_1-a_0), NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_0123_023_1023(ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_void_p(_M7_sliced.ctypes.data),
                                             ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_int(_M6.shape[0]),
                                             ctypes.c_int(_M6.shape[1]),
                                             ctypes.c_int(_M6.shape[2]),
                                             ctypes.c_int(_M6.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 aQRT->aRTQ
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((a_1-a_0), (R_1-R_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((a_1-a_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  24 jQ,aRTQ->jaRT
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NOCC, (a_1-a_0), (R_1-R_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NOCC * ((a_1-a_0) * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NOCC, (a_1-a_0), (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
                _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_3_reshaped, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  25 slice _INPUT_11 with indices ['T']
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
                # step  26 jT,jaRT->jaR
                fn_contraction_01_0231_023_plus(ctypes.c_void_p(_INPUT_11_sliced.ctypes.data),
                                                ctypes.c_void_p(_M9.ctypes.data),
                                                ctypes.c_void_p(_M10.ctypes.data),
                                                ctypes.c_int(_INPUT_11_sliced.shape[0]),
                                                ctypes.c_int(_INPUT_11_sliced.shape[1]),
                                                ctypes.c_int(_M9.shape[1]),
                                                ctypes.c_int(_M9.shape[2]),
                                                ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  27 end   for loop with indices ('R', 'a', 'T')
            # step  28 jaR,jaR->
            output_tmp       = ctypes.c_double(0.0)
            fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                   ctypes.c_void_p(_M11_perm.ctypes.data),
                   ctypes.c_int(_M10.size),
                   ctypes.pointer(output_tmp))
            output_tmp = output_tmp.value
            _M12 += output_tmp
        # step  29 end   for loop with indices ('R', 'a')
    # step  30 end   for loop with indices ('R',)
    # clean the final forloop
    return _M12

def RMP2_K_forloop_R_j_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       R_bunchsize = 8,
                                                       j_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (R_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_8_sliced
    tmp              = (j_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (R_bunchsize * (j_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NVIR * (R_bunchsize * j_bunchsize))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NVIR * (R_bunchsize * j_bunchsize))
    output           = max(output, tmp)
    # cmpr _M0_sliced
    tmp              = (NTHC_INT * (T_bunchsize * j_bunchsize))
    output           = max(output, tmp)
    # cmpr _M5_sliced
    tmp              = (NTHC_INT * (R_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (j_bunchsize * (R_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (j_bunchsize * (R_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (j_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M4_sliced
    tmp              = (NTHC_INT * (R_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (j_bunchsize * (NTHC_INT * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (j_bunchsize * (NTHC_INT * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NVIR * (j_bunchsize * (R_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_12_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NVIR * (j_bunchsize * R_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_R_j_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                R_bunchsize = 8,
                                                                j_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (j_bunchsize * (R_bunchsize * (NTHC_INT * T_bunchsize)))
    _M1_size         = (R_bunchsize * (j_bunchsize * NTHC_INT))
    _M8_size         = (j_bunchsize * (NTHC_INT * (R_bunchsize * T_bunchsize)))
    _M9_size         = (NVIR * (j_bunchsize * (R_bunchsize * T_bunchsize)))
    _M11_size        = (NVIR * (R_bunchsize * j_bunchsize))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M4_sliced_size  = (NTHC_INT * (R_bunchsize * T_bunchsize))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_8_sliced_size = (j_bunchsize * NTHC_INT)
    _M0_sliced_size  = (NTHC_INT * (T_bunchsize * j_bunchsize))
    _INPUT_12_sliced_size = (NVIR * T_bunchsize)
    _M5_sliced_size  = (NTHC_INT * (R_bunchsize * T_bunchsize))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NVIR * (j_bunchsize * R_bunchsize))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_5_sliced_size = (R_bunchsize * NTHC_INT)
    _M7_size         = (NTHC_INT * (j_bunchsize * (R_bunchsize * T_bunchsize)))
    # cmpr _M2_size
    size_now         = 0               
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M3_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _INPUT_5_sliced_size + _INPUT_8_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    size_now         = (size_now + _INPUT_8_sliced_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M1_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M0_sliced_size + _M5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M0_sliced_size)
    size_now         = (size_now + _M5_sliced_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M6_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M7_size + _M4_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_sliced_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M9_size + _INPUT_12_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_12_sliced_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_R_j_naive(Z           : np.ndarray,
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
    _M2              = np.einsum("iR,iT->RTi"    , _INPUT_6        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iP,RTi->PRT"   , _INPUT_1        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("bR,bT->RTb"    , _INPUT_7        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("bQ,RTb->QRT"   , _INPUT_4        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("jQ,jT->QTj"    , _INPUT_3        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("RS,jS->RjS"    , _INPUT_5        , _INPUT_8        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,RjS->aRj"   , _INPUT_9        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 2, 1)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("QTj,QRT->jRQT" , _M0             , _M5             )
    del _M0         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 1, 3, 2)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,jRTQ->PjRT" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("PRT,PjRT->jPRT", _M4             , _M7             )
    del _M4         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("aP,jRTP->ajRT" , _INPUT_2        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("aT,ajRT->ajR"  , _INPUT_12       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("ajR,ajR->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_R_j(Z           : np.ndarray,
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
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 iP,RTi->PRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M2_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bR,bT->RTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bQ,RTb->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M3_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M3         
    del _M3_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 jQ,jT->QTj 
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
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,jS->RjS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    _buffer          = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 aS,RjS->aRj 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NVIR, NTHC_INT, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped, _M1_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 aRj->ajR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NOCC), dtype=np.float64)
    _M11_perm        = np.ndarray((NVIR, NOCC, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 QTj,QRT->jRQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M6              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_031_2301(ctypes.c_void_p(_M0.ctypes.data),
                                ctypes.c_void_p(_M5.ctypes.data),
                                ctypes.c_void_p(_M6.ctypes.data),
                                ctypes.c_int(_M0.shape[0]),
                                ctypes.c_int(_M0.shape[1]),
                                ctypes.c_int(_M0.shape[2]),
                                ctypes.c_int(_M5.shape[1]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M0         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 jRQT->jRTQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M6_perm         = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M6.ctypes.data),
                             ctypes.c_void_p(_M6_perm.ctypes.data),
                             ctypes.c_int(_M6.shape[0]),
                             ctypes.c_int(_M6.shape[1]),
                             ctypes.c_int(_M6.shape[2]),
                             ctypes.c_int(_M6.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 PQ,jRTQ->PjRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
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
    lib.ddot(_INPUT_0_reshaped, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 PRT,PjRT->jPRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0312_3012 = getattr(libpbc, "fn_contraction_012_0312_3012", None)
    assert fn_contraction_012_0312_3012 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0312_3012(ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_M4.shape[0]),
                                 ctypes.c_int(_M4.shape[1]),
                                 ctypes.c_int(_M4.shape[2]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 jPRT->jRTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 aP,jRTP->ajRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NVIR, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 aT,ajRT->ajR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    _buffer          = np.ndarray((NVIR, NOCC, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NVIR, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_12.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_12.shape[0]),
                               ctypes.c_int(_INPUT_12.shape[1]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 ajR,ajR-> 
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

def RMP2_K_forloop_R_j_forloop_R_j(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   R_bunchsize = 8,
                                   j_bunchsize = 8,
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
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_contraction_012_0312_3012 = getattr(libpbc, "fn_contraction_012_0312_3012", None)
    assert fn_contraction_012_0312_3012 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_01_0231_023_plus = getattr(libpbc, "fn_contraction_01_0231_023_plus", None)
    assert fn_contraction_01_0231_023_plus is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_R_j_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          R_bunchsize = R_bunchsize,
                                                                          j_bunchsize = j_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_R_j_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   R_bunchsize = R_bunchsize,
                                                                                   j_bunchsize = j_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_R         = None            
    offset_R_j       = None            
    offset_R_j_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 iR,iT->RTi
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 iP,RTi->PRT
    _M4_offset       = offset_now      
    _M4_offset       = min(_M4_offset, _M2_offset)
    offset_now       = _M4_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M4.ravel()[:] = ddot_buffer.ravel()[:]
    # step   4 bR,bT->RTb
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_int(_INPUT_7.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 bQ,RTb->QRT
    _M5_offset       = offset_now      
    _M5_offset       = min(_M5_offset, _M3_offset)
    offset_now       = _M5_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped.T, _M3_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M5.ravel()[:] = ddot_buffer.ravel()[:]
    # step   6 jQ,jT->QTj
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
    # step   7 start for loop with indices ('R',)
    for R_0, R_1 in lib.prange(0,NTHC_INT,R_bunchsize):
        if offset_R == None:
            offset_R         = offset_now      
        else:
            offset_now       = offset_R        
        # step   8 start for loop with indices ('R', 'j')
        for j_0, j_1 in lib.prange(0,NOCC,j_bunchsize):
            if offset_R_j == None:
                offset_R_j       = offset_now      
            else:
                offset_now       = offset_R_j      
            # step   9 slice _INPUT_5 with indices ['R']
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
            # step  10 slice _INPUT_8 with indices ['j']
            _INPUT_8_sliced_offset = offset_now      
            _INPUT_8_sliced  = np.ndarray(((j_1-j_0), NTHC_INT), buffer = buffer, offset = _INPUT_8_sliced_offset)
            size_item        = ((j_1-j_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_8.ctypes.data),
                         ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_8.shape[0]),
                         ctypes.c_int(_INPUT_8.shape[1]),
                         ctypes.c_int(j_0),
                         ctypes.c_int(j_1))
            # step  11 RS,jS->RjS
            _M1_offset       = offset_now      
            _M1_offset       = min(_M1_offset, _INPUT_5_sliced_offset)
            _M1_offset       = min(_M1_offset, _INPUT_8_sliced_offset)
            offset_now       = _M1_offset      
            tmp_itemsize     = ((R_1-R_0) * ((j_1-j_0) * (NTHC_INT * _itemsize)))
            _M1              = np.ndarray(((R_1-R_0), (j_1-j_0), NTHC_INT), buffer = buffer, offset = _M1_offset)
            offset_now       = (_M1_offset + tmp_itemsize)
            fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_8_sliced.ctypes.data),
                                     ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_8_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 aS,RjS->aRj
            _M11_offset      = offset_now      
            _M11_offset      = min(_M11_offset, _M1_offset)
            offset_now       = _M11_offset     
            ddot_buffer      = np.ndarray((NVIR, (R_1-R_0), (j_1-j_0)), buffer = linearop_buf)
            tmp_itemsize     = (NVIR * ((R_1-R_0) * ((j_1-j_0) * _itemsize)))
            _M11             = np.ndarray((NVIR, (R_1-R_0), (j_1-j_0)), buffer = buffer, offset = _M11_offset)
            offset_now       = (_M11_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
            _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_9_reshaped, _M1_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M11.ravel()[:] = ddot_buffer.ravel()[:]
            # step  13 allocate   _M10
            _M10             = np.ndarray((NVIR, (j_1-j_0), (R_1-R_0)), buffer = buffer, offset = offset_now)
            tmp_itemsize     = (NVIR * ((j_1-j_0) * ((R_1-R_0) * _itemsize)))
            _M10_offset      = offset_now      
            offset_now       = (offset_now + tmp_itemsize)
            _M10.ravel()[:] = 0.0
            # step  14 aRj->ajR
            _M11_perm_offset = _M11_offset     
            _M11_perm        = np.ndarray((NVIR, (j_1-j_0), (R_1-R_0)), buffer = buffer, offset = _M11_perm_offset)
            fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(NVIR),
                                   ctypes.c_int((R_1-R_0)),
                                   ctypes.c_int((j_1-j_0)),
                                   ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  15 start for loop with indices ('R', 'j', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_R_j_T == None:
                    offset_R_j_T     = offset_now      
                else:
                    offset_now       = offset_R_j_T    
                # step  16 slice _M0 with indices ['T', 'j']
                _M0_sliced_offset = offset_now      
                _M0_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (j_1-j_0)), buffer = buffer, offset = _M0_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((j_1-j_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M0.ctypes.data),
                               ctypes.c_void_p(_M0_sliced.ctypes.data),
                               ctypes.c_int(_M0.shape[0]),
                               ctypes.c_int(_M0.shape[1]),
                               ctypes.c_int(_M0.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(j_0),
                               ctypes.c_int(j_1))
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
                # step  18 QTj,QRT->jRQT
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M0_sliced_offset)
                _M6_offset       = min(_M6_offset, _M5_sliced_offset)
                offset_now       = _M6_offset      
                tmp_itemsize     = ((j_1-j_0) * ((R_1-R_0) * (NTHC_INT * ((T_1-T_0) * _itemsize))))
                _M6              = np.ndarray(((j_1-j_0), (R_1-R_0), NTHC_INT, (T_1-T_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                fn_contraction_012_031_2301(ctypes.c_void_p(_M0_sliced.ctypes.data),
                                            ctypes.c_void_p(_M5_sliced.ctypes.data),
                                            ctypes.c_void_p(_M6.ctypes.data),
                                            ctypes.c_int(_M0_sliced.shape[0]),
                                            ctypes.c_int(_M0_sliced.shape[1]),
                                            ctypes.c_int(_M0_sliced.shape[2]),
                                            ctypes.c_int(_M5_sliced.shape[1]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 jRQT->jRTQ
                _M6_perm_offset  = _M6_offset      
                _M6_perm         = np.ndarray(((j_1-j_0), (R_1-R_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int((j_1-j_0)),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 PQ,jRTQ->PjRT
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M6_perm_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (j_1-j_0), (R_1-R_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((j_1-j_0) * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M7              = np.ndarray((NTHC_INT, (j_1-j_0), (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
                _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_0_reshaped, _M6_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M7.ravel()[:] = ddot_buffer.ravel()[:]
                # step  21 slice _M4 with indices ['R', 'T']
                _M4_sliced_offset = offset_now      
                _M4_sliced       = np.ndarray((NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M4_sliced_offset)
                size_item        = (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_sliced.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_int(R_0),
                               ctypes.c_int(R_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  22 PRT,PjRT->jPRT
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M4_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((j_1-j_0) * (NTHC_INT * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((j_1-j_0), NTHC_INT, (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_012_0312_3012(ctypes.c_void_p(_M4_sliced.ctypes.data),
                                             ctypes.c_void_p(_M7.ctypes.data),
                                             ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_int(_M4_sliced.shape[0]),
                                             ctypes.c_int(_M4_sliced.shape[1]),
                                             ctypes.c_int(_M4_sliced.shape[2]),
                                             ctypes.c_int(_M7.shape[1]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 jPRT->jRTP
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((j_1-j_0), (R_1-R_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((j_1-j_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((R_1-R_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  24 aP,jRTP->ajRT
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NVIR, (j_1-j_0), (R_1-R_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NVIR * ((j_1-j_0) * ((R_1-R_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NVIR, (j_1-j_0), (R_1-R_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
                _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_2_reshaped, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  25 slice _INPUT_12 with indices ['T']
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
                # step  26 aT,ajRT->ajR
                fn_contraction_01_0231_023_plus(ctypes.c_void_p(_INPUT_12_sliced.ctypes.data),
                                                ctypes.c_void_p(_M9.ctypes.data),
                                                ctypes.c_void_p(_M10.ctypes.data),
                                                ctypes.c_int(_INPUT_12_sliced.shape[0]),
                                                ctypes.c_int(_INPUT_12_sliced.shape[1]),
                                                ctypes.c_int(_M9.shape[1]),
                                                ctypes.c_int(_M9.shape[2]),
                                                ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  27 end   for loop with indices ('R', 'j', 'T')
            # step  28 ajR,ajR->
            output_tmp       = ctypes.c_double(0.0)
            fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                   ctypes.c_void_p(_M11_perm.ctypes.data),
                   ctypes.c_int(_M10.size),
                   ctypes.pointer(output_tmp))
            output_tmp = output_tmp.value
            _M12 += output_tmp
        # step  29 end   for loop with indices ('R', 'j')
    # step  30 end   for loop with indices ('R',)
    # clean the final forloop
    return _M12

def RMP2_K_forloop_S_b_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       S_bunchsize = 8,
                                                       b_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (NTHC_INT * S_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_7_sliced
    tmp              = (b_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (S_bunchsize * (b_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NOCC * (S_bunchsize * b_bunchsize))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NOCC * (S_bunchsize * b_bunchsize))
    output           = max(output, tmp)
    # cmpr _M0_sliced
    tmp              = (NTHC_INT * (T_bunchsize * b_bunchsize))
    output           = max(output, tmp)
    # cmpr _M5_sliced
    tmp              = (NTHC_INT * (S_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (b_bunchsize * (S_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (b_bunchsize * (S_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (b_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M4_sliced
    tmp              = (NTHC_INT * (S_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (b_bunchsize * (NTHC_INT * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (b_bunchsize * (NTHC_INT * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NOCC * (b_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_10_sliced
    tmp              = (NOCC * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NOCC * (b_bunchsize * S_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_S_b_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                S_bunchsize = 8,
                                                                b_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (b_bunchsize * (S_bunchsize * (NTHC_INT * T_bunchsize)))
    _M1_size         = (S_bunchsize * (b_bunchsize * NTHC_INT))
    _M8_size         = (b_bunchsize * (NTHC_INT * (S_bunchsize * T_bunchsize)))
    _M9_size         = (NOCC * (b_bunchsize * (S_bunchsize * T_bunchsize)))
    _M11_size        = (NOCC * (S_bunchsize * b_bunchsize))
    _INPUT_10_sliced_size = (NOCC * T_bunchsize)
    _INPUT_7_sliced_size = (b_bunchsize * NTHC_INT)
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M4_sliced_size  = (NTHC_INT * (S_bunchsize * T_bunchsize))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M0_sliced_size  = (NTHC_INT * (T_bunchsize * b_bunchsize))
    _M5_sliced_size  = (NTHC_INT * (S_bunchsize * T_bunchsize))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NOCC * (b_bunchsize * S_bunchsize))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_5_sliced_size = (NTHC_INT * S_bunchsize)
    _M7_size         = (NTHC_INT * (b_bunchsize * (S_bunchsize * T_bunchsize)))
    # cmpr _M3_size
    size_now         = 0               
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M2_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _INPUT_5_sliced_size + _INPUT_7_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    size_now         = (size_now + _INPUT_7_sliced_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M1_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M0_sliced_size + _M5_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M0_sliced_size)
    size_now         = (size_now + _M5_sliced_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M6_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M7_size + _M4_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_sliced_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M4_size + _M5_size + _M0_size + _M11_size + _M10_size + _M9_size + _INPUT_10_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M5_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_10_sliced_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_S_b_naive(Z           : np.ndarray,
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
    _M3              = np.einsum("aS,aT->STa"    , _INPUT_9        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,STa->PST"   , _INPUT_2        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jS,jT->STj"    , _INPUT_8        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("jQ,STj->QST"   , _INPUT_3        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("bQ,bT->QTb"    , _INPUT_4        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("RS,bR->SbR"    , _INPUT_5        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("iR,SbR->iSb"   , _INPUT_6        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 2, 1)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("QTb,QST->bSQT" , _M0             , _M5             )
    del _M0         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 1, 3, 2)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,bSTQ->PbST" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("PST,PbST->bPST", _M4             , _M7             )
    del _M4         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("iP,bSTP->ibST" , _INPUT_1        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iT,ibST->ibS"  , _INPUT_10       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("ibS,ibS->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_S_b(Z           : np.ndarray,
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
    # step 0 aS,aT->STa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aP,STa->PST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M3_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M3         
    del _M3_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 jS,jT->STj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jQ,STj->QST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M2_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,bR->SbR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iR,SbR->iSb 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NOCC, NTHC_INT, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped, _M1_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iSb->ibS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NVIR), dtype=np.float64)
    _M11_perm        = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 QTb,QST->bSQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    _buffer          = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M6              = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_031_2301(ctypes.c_void_p(_M0.ctypes.data),
                                ctypes.c_void_p(_M5.ctypes.data),
                                ctypes.c_void_p(_M6.ctypes.data),
                                ctypes.c_int(_M0.shape[0]),
                                ctypes.c_int(_M0.shape[1]),
                                ctypes.c_int(_M0.shape[2]),
                                ctypes.c_int(_M5.shape[1]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M0         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 bSQT->bSTQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M6_perm         = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M6.ctypes.data),
                             ctypes.c_void_p(_M6_perm.ctypes.data),
                             ctypes.c_int(_M6.shape[0]),
                             ctypes.c_int(_M6.shape[1]),
                             ctypes.c_int(_M6.shape[2]),
                             ctypes.c_int(_M6.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 PQ,bSTQ->PbST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
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
    lib.ddot(_INPUT_0_reshaped, _M6_perm_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6_perm    
    del _M6_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 PST,PbST->bPST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0312_3012 = getattr(libpbc, "fn_contraction_012_0312_3012", None)
    assert fn_contraction_012_0312_3012 is not None
    _buffer          = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NVIR, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0312_3012(ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_M4.shape[0]),
                                 ctypes.c_int(_M4.shape[1]),
                                 ctypes.c_int(_M4.shape[2]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bPST->bSTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NVIR, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 iP,bSTP->ibST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NOCC, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
    _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_1_reshaped, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 iT,ibST->ibS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    _buffer          = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_10.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_10.shape[0]),
                               ctypes.c_int(_INPUT_10.shape[1]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 ibS,ibS-> 
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

def RMP2_K_forloop_S_b_forloop_S_b(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   S_bunchsize = 8,
                                   b_bunchsize = 8,
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
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_contraction_012_0312_3012 = getattr(libpbc, "fn_contraction_012_0312_3012", None)
    assert fn_contraction_012_0312_3012 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_01_0231_023_plus = getattr(libpbc, "fn_contraction_01_0231_023_plus", None)
    assert fn_contraction_01_0231_023_plus is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_S_b_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          S_bunchsize = S_bunchsize,
                                                                          b_bunchsize = b_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_S_b_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   S_bunchsize = S_bunchsize,
                                                                                   b_bunchsize = b_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_S         = None            
    offset_S_b       = None            
    offset_S_b_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 aS,aT->STa
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 aP,STa->PST
    _M4_offset       = offset_now      
    _M4_offset       = min(_M4_offset, _M3_offset)
    offset_now       = _M4_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M3_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M4.ravel()[:] = ddot_buffer.ravel()[:]
    # step   4 jS,jT->STj
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 jQ,STj->QST
    _M5_offset       = offset_now      
    _M5_offset       = min(_M5_offset, _M2_offset)
    offset_now       = _M5_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M5.ravel()[:] = ddot_buffer.ravel()[:]
    # step   6 bQ,bT->QTb
    _M0_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    offset_now       = (_M0_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   7 start for loop with indices ('S',)
    for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
        if offset_S == None:
            offset_S         = offset_now      
        else:
            offset_now       = offset_S        
        # step   8 start for loop with indices ('S', 'b')
        for b_0, b_1 in lib.prange(0,NVIR,b_bunchsize):
            if offset_S_b == None:
                offset_S_b       = offset_now      
            else:
                offset_now       = offset_S_b      
            # step   9 slice _INPUT_5 with indices ['S']
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
            # step  10 slice _INPUT_7 with indices ['b']
            _INPUT_7_sliced_offset = offset_now      
            _INPUT_7_sliced  = np.ndarray(((b_1-b_0), NTHC_INT), buffer = buffer, offset = _INPUT_7_sliced_offset)
            size_item        = ((b_1-b_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_7.ctypes.data),
                         ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_7.shape[0]),
                         ctypes.c_int(_INPUT_7.shape[1]),
                         ctypes.c_int(b_0),
                         ctypes.c_int(b_1))
            # step  11 RS,bR->SbR
            _M1_offset       = offset_now      
            _M1_offset       = min(_M1_offset, _INPUT_5_sliced_offset)
            _M1_offset       = min(_M1_offset, _INPUT_7_sliced_offset)
            offset_now       = _M1_offset      
            tmp_itemsize     = ((S_1-S_0) * ((b_1-b_0) * (NTHC_INT * _itemsize)))
            _M1              = np.ndarray(((S_1-S_0), (b_1-b_0), NTHC_INT), buffer = buffer, offset = _M1_offset)
            offset_now       = (_M1_offset + tmp_itemsize)
            fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_7_sliced.ctypes.data),
                                     ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_7_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 iR,SbR->iSb
            _M11_offset      = offset_now      
            _M11_offset      = min(_M11_offset, _M1_offset)
            offset_now       = _M11_offset     
            ddot_buffer      = np.ndarray((NOCC, (S_1-S_0), (b_1-b_0)), buffer = linearop_buf)
            tmp_itemsize     = (NOCC * ((S_1-S_0) * ((b_1-b_0) * _itemsize)))
            _M11             = np.ndarray((NOCC, (S_1-S_0), (b_1-b_0)), buffer = buffer, offset = _M11_offset)
            offset_now       = (_M11_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
            _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_6_reshaped, _M1_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M11.ravel()[:] = ddot_buffer.ravel()[:]
            # step  13 allocate   _M10
            _M10             = np.ndarray((NOCC, (b_1-b_0), (S_1-S_0)), buffer = buffer, offset = offset_now)
            tmp_itemsize     = (NOCC * ((b_1-b_0) * ((S_1-S_0) * _itemsize)))
            _M10_offset      = offset_now      
            offset_now       = (offset_now + tmp_itemsize)
            _M10.ravel()[:] = 0.0
            # step  14 iSb->ibS
            _M11_perm_offset = _M11_offset     
            _M11_perm        = np.ndarray((NOCC, (b_1-b_0), (S_1-S_0)), buffer = buffer, offset = _M11_perm_offset)
            fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(NOCC),
                                   ctypes.c_int((S_1-S_0)),
                                   ctypes.c_int((b_1-b_0)),
                                   ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  15 start for loop with indices ('S', 'b', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_S_b_T == None:
                    offset_S_b_T     = offset_now      
                else:
                    offset_now       = offset_S_b_T    
                # step  16 slice _M0 with indices ['T', 'b']
                _M0_sliced_offset = offset_now      
                _M0_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (b_1-b_0)), buffer = buffer, offset = _M0_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((b_1-b_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M0.ctypes.data),
                               ctypes.c_void_p(_M0_sliced.ctypes.data),
                               ctypes.c_int(_M0.shape[0]),
                               ctypes.c_int(_M0.shape[1]),
                               ctypes.c_int(_M0.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(b_0),
                               ctypes.c_int(b_1))
                # step  17 slice _M5 with indices ['S', 'T']
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
                # step  18 QTb,QST->bSQT
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M0_sliced_offset)
                _M6_offset       = min(_M6_offset, _M5_sliced_offset)
                offset_now       = _M6_offset      
                tmp_itemsize     = ((b_1-b_0) * ((S_1-S_0) * (NTHC_INT * ((T_1-T_0) * _itemsize))))
                _M6              = np.ndarray(((b_1-b_0), (S_1-S_0), NTHC_INT, (T_1-T_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                fn_contraction_012_031_2301(ctypes.c_void_p(_M0_sliced.ctypes.data),
                                            ctypes.c_void_p(_M5_sliced.ctypes.data),
                                            ctypes.c_void_p(_M6.ctypes.data),
                                            ctypes.c_int(_M0_sliced.shape[0]),
                                            ctypes.c_int(_M0_sliced.shape[1]),
                                            ctypes.c_int(_M0_sliced.shape[2]),
                                            ctypes.c_int(_M5_sliced.shape[1]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 bSQT->bSTQ
                _M6_perm_offset  = _M6_offset      
                _M6_perm         = np.ndarray(((b_1-b_0), (S_1-S_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M6.ctypes.data),
                                         ctypes.c_void_p(_M6_perm.ctypes.data),
                                         ctypes.c_int((b_1-b_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 PQ,bSTQ->PbST
                _M7_offset       = offset_now      
                _M7_offset       = min(_M7_offset, _M6_perm_offset)
                offset_now       = _M7_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (b_1-b_0), (S_1-S_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((b_1-b_0) * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M7              = np.ndarray((NTHC_INT, (b_1-b_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M7_offset)
                offset_now       = (_M7_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
                _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M6_perm.shape[2]
                _M6_perm_reshaped = _M6_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_0_reshaped, _M6_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M7.ravel()[:] = ddot_buffer.ravel()[:]
                # step  21 slice _M4 with indices ['S', 'T']
                _M4_sliced_offset = offset_now      
                _M4_sliced       = np.ndarray((NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M4_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_sliced.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_int(S_0),
                               ctypes.c_int(S_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  22 PST,PbST->bPST
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M4_sliced_offset)
                _M8_offset       = min(_M8_offset, _M7_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((b_1-b_0) * (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((b_1-b_0), NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_012_0312_3012(ctypes.c_void_p(_M4_sliced.ctypes.data),
                                             ctypes.c_void_p(_M7.ctypes.data),
                                             ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_int(_M4_sliced.shape[0]),
                                             ctypes.c_int(_M4_sliced.shape[1]),
                                             ctypes.c_int(_M4_sliced.shape[2]),
                                             ctypes.c_int(_M7.shape[1]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 bPST->bSTP
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((b_1-b_0), (S_1-S_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((b_1-b_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  24 iP,bSTP->ibST
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NOCC, (b_1-b_0), (S_1-S_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NOCC * ((b_1-b_0) * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NOCC, (b_1-b_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_1.shape[0]
                _INPUT_1_reshaped = _INPUT_1.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_1_reshaped, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  25 slice _INPUT_10 with indices ['T']
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
                # step  26 iT,ibST->ibS
                fn_contraction_01_0231_023_plus(ctypes.c_void_p(_INPUT_10_sliced.ctypes.data),
                                                ctypes.c_void_p(_M9.ctypes.data),
                                                ctypes.c_void_p(_M10.ctypes.data),
                                                ctypes.c_int(_INPUT_10_sliced.shape[0]),
                                                ctypes.c_int(_INPUT_10_sliced.shape[1]),
                                                ctypes.c_int(_M9.shape[1]),
                                                ctypes.c_int(_M9.shape[2]),
                                                ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  27 end   for loop with indices ('S', 'b', 'T')
            # step  28 ibS,ibS->
            output_tmp       = ctypes.c_double(0.0)
            fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                   ctypes.c_void_p(_M11_perm.ctypes.data),
                   ctypes.c_int(_M10.size),
                   ctypes.pointer(output_tmp))
            output_tmp = output_tmp.value
            _M12 += output_tmp
        # step  29 end   for loop with indices ('S', 'b')
    # step  30 end   for loop with indices ('S',)
    # clean the final forloop
    return _M12

def RMP2_K_forloop_S_i_determine_buf_head_size_forloop(NVIR        : int,
                                                       NOCC        : int,
                                                       N_LAPLACE   : int,
                                                       NTHC_INT    : int,
                                                       S_bunchsize = 8,
                                                       i_bunchsize = 8,
                                                       T_bunchsize = 1):
    # init
    output           = 0               
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _INPUT_5_sliced
    tmp              = (NTHC_INT * S_bunchsize)
    output           = max(output, tmp)
    # cmpr _INPUT_6_sliced
    tmp              = (i_bunchsize * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (S_bunchsize * (i_bunchsize * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NVIR * (S_bunchsize * i_bunchsize))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NVIR * (S_bunchsize * i_bunchsize))
    output           = max(output, tmp)
    # cmpr _M0_sliced
    tmp              = (NTHC_INT * (T_bunchsize * i_bunchsize))
    output           = max(output, tmp)
    # cmpr _M4_sliced
    tmp              = (NTHC_INT * (S_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (i_bunchsize * (S_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (i_bunchsize * (S_bunchsize * (NTHC_INT * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (i_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M7_sliced
    tmp              = (NTHC_INT * (S_bunchsize * T_bunchsize))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (i_bunchsize * (NTHC_INT * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (i_bunchsize * (NTHC_INT * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NVIR * (i_bunchsize * (S_bunchsize * T_bunchsize)))
    output           = max(output, tmp)
    # cmpr _INPUT_13_sliced
    tmp              = (NVIR * T_bunchsize)
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NVIR * (i_bunchsize * S_bunchsize))
    output           = max(output, tmp)
    return output

def RMP2_K_forloop_S_i_determine_buf_size_intermediates_forloop(NVIR        : int,
                                                                NOCC        : int,
                                                                N_LAPLACE   : int,
                                                                NTHC_INT    : int,
                                                                S_bunchsize = 8,
                                                                i_bunchsize = 8,
                                                                T_bunchsize = 1):
    # init
    output           = 0               
    _M6_size         = (NTHC_INT * (i_bunchsize * (S_bunchsize * T_bunchsize)))
    _M1_size         = (S_bunchsize * (i_bunchsize * NTHC_INT))
    _M8_size         = (i_bunchsize * (NTHC_INT * (S_bunchsize * T_bunchsize)))
    _M7_sliced_size  = (NTHC_INT * (S_bunchsize * T_bunchsize))
    _M9_size         = (NVIR * (i_bunchsize * (S_bunchsize * T_bunchsize)))
    _M11_size        = (NVIR * (S_bunchsize * i_bunchsize))
    _INPUT_13_sliced_size = (NVIR * T_bunchsize)
    _M0_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _INPUT_6_sliced_size = (i_bunchsize * NTHC_INT)
    _M4_sliced_size  = (NTHC_INT * (S_bunchsize * T_bunchsize))
    _M5_size         = (i_bunchsize * (S_bunchsize * (NTHC_INT * T_bunchsize)))
    _M0_sliced_size  = (NTHC_INT * (T_bunchsize * i_bunchsize))
    _M3_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NVIR * (i_bunchsize * S_bunchsize))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _INPUT_5_sliced_size = (NTHC_INT * S_bunchsize)
    _M7_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    # cmpr _M2_size
    size_now         = 0               
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M3_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _INPUT_5_sliced_size + _INPUT_6_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _INPUT_5_sliced_size)
    size_now         = (size_now + _INPUT_6_sliced_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M1_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M0_sliced_size + _M4_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M0_sliced_size)
    size_now         = (size_now + _M4_sliced_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M5_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M6_size + _M7_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M7_sliced_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M8_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M7_size + _M4_size + _M0_size + _M11_size + _M10_size + _M9_size + _INPUT_13_sliced_size
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    size_now         = (size_now + _M4_size)
    size_now         = (size_now + _M0_size)
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _INPUT_13_sliced_size)
    output           = max(output, size_now)
    return output

def RMP2_K_forloop_S_i_naive(Z           : np.ndarray,
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
    _M2              = np.einsum("jS,jT->STj"    , _INPUT_8        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("jQ,STj->QST"   , _INPUT_3        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("aS,aT->STa"    , _INPUT_9        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,STa->PST"   , _INPUT_2        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,iT->PTi"    , _INPUT_1        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("RS,iR->SiR"    , _INPUT_5        , _INPUT_6        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bR,SiR->bSi"   , _INPUT_7        , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 2, 1)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PTi,PST->iSPT" , _M0             , _M4             )
    del _M0         
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 1, 3, 2)    )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("PQ,iSTP->QiST" , _INPUT_0        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("QiST,QST->iQST", _M6             , _M7             )
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (0, 2, 3, 1)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("bQ,iSTQ->biST" , _INPUT_4        , _M8_perm        )
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("bT,biST->biS"  , _INPUT_13       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("biS,biS->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    return _M12

def RMP2_K_forloop_S_i(Z           : np.ndarray,
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
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jQ,STj->QST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M2_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 aS,aT->STa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 aP,STa->PST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M4.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M3_reshaped.T, c=_M4_reshaped)
    _M4          = _M4_reshaped.reshape(*shape_backup)
    del _M3         
    del _M3_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iP,iT->PTi 
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
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,iR->SiR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    _buffer          = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    _M1              = np.ndarray((NTHC_INT, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_5.ctypes.data),
                             ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_M1.ctypes.data),
                             ctypes.c_int(_INPUT_5.shape[0]),
                             ctypes.c_int(_INPUT_5.shape[1]),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 bR,SiR->bSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NVIR, NTHC_INT, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped, _M1_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M1         
    del _M1_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 bSi->biS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NOCC), dtype=np.float64)
    _M11_perm        = np.ndarray((NVIR, NOCC, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PTi,PST->iSPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M5              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_031_2301(ctypes.c_void_p(_M0.ctypes.data),
                                ctypes.c_void_p(_M4.ctypes.data),
                                ctypes.c_void_p(_M5.ctypes.data),
                                ctypes.c_int(_M0.shape[0]),
                                ctypes.c_int(_M0.shape[1]),
                                ctypes.c_int(_M0.shape[2]),
                                ctypes.c_int(_M4.shape[1]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M0         
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iSPT->iSTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M5_perm         = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0132(ctypes.c_void_p(_M5.ctypes.data),
                             ctypes.c_void_p(_M5_perm.ctypes.data),
                             ctypes.c_int(_M5.shape[0]),
                             ctypes.c_int(_M5.shape[1]),
                             ctypes.c_int(_M5.shape[2]),
                             ctypes.c_int(_M5.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 PQ,iSTP->QiST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M5_perm.shape[2]
    _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M5_perm_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5_perm    
    del _M5_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 QiST,QST->iQST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_023_1023 = getattr(libpbc, "fn_contraction_0123_023_1023", None)
    assert fn_contraction_0123_023_1023 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_023_1023(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 iQST->iSTQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M8_perm         = np.ndarray((NOCC, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                             ctypes.c_void_p(_M8_perm.ctypes.data),
                             ctypes.c_int(_M8.shape[0]),
                             ctypes.c_int(_M8.shape[1]),
                             ctypes.c_int(_M8.shape[2]),
                             ctypes.c_int(_M8.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 bQ,iSTQ->biST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NVIR, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
    _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
    _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_4_reshaped, _M8_perm_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M8_perm    
    del _M8_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 bT,biST->biS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_0231_023 = getattr(libpbc, "fn_contraction_01_0231_023", None)
    assert fn_contraction_01_0231_023 is not None
    _buffer          = np.ndarray((NVIR, NOCC, NTHC_INT), dtype=np.float64)
    _M10             = np.ndarray((NVIR, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_0231_023(ctypes.c_void_p(_INPUT_13.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_void_p(_M10.ctypes.data),
                               ctypes.c_int(_INPUT_13.shape[0]),
                               ctypes.c_int(_INPUT_13.shape[1]),
                               ctypes.c_int(_M9.shape[1]),
                               ctypes.c_int(_M9.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 biS,biS-> 
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

def RMP2_K_forloop_S_i_forloop_S_i(Z           : np.ndarray,
                                   X_o         : np.ndarray,
                                   X_v         : np.ndarray,
                                   tau_o       : np.ndarray,
                                   tau_v       : np.ndarray,
                                   buffer      : np.ndarray,
                                   S_bunchsize = 8,
                                   i_bunchsize = 8,
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
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    fn_slice_2_0 = getattr(libpbc, "fn_slice_2_0", None)
    assert fn_slice_2_0 is not None
    fn_contraction_01_20_120 = getattr(libpbc, "fn_contraction_01_20_120", None)
    assert fn_contraction_01_20_120 is not None
    fn_slice_2_1 = getattr(libpbc, "fn_slice_2_1", None)
    assert fn_slice_2_1 is not None
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    fn_slice_3_1_2 = getattr(libpbc, "fn_slice_3_1_2", None)
    assert fn_slice_3_1_2 is not None
    fn_contraction_012_031_2301 = getattr(libpbc, "fn_contraction_012_031_2301", None)
    assert fn_contraction_012_031_2301 is not None
    fn_permutation_0123_0132 = getattr(libpbc, "fn_permutation_0123_0132", None)
    assert fn_permutation_0123_0132 is not None
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    fn_permutation_0123_0231 = getattr(libpbc, "fn_permutation_0123_0231", None)
    assert fn_permutation_0123_0231 is not None
    fn_contraction_0123_023_1023 = getattr(libpbc, "fn_contraction_0123_023_1023", None)
    assert fn_contraction_0123_023_1023 is not None
    fn_contraction_01_0231_023_plus = getattr(libpbc, "fn_contraction_01_0231_023_plus", None)
    assert fn_contraction_01_0231_023_plus is not None
    # preallocate buffer
    bufsize1         = RMP2_K_forloop_S_i_determine_buf_head_size_forloop(NVIR = NVIR,
                                                                          NOCC = NOCC,
                                                                          N_LAPLACE = N_LAPLACE,
                                                                          NTHC_INT = NTHC_INT,
                                                                          S_bunchsize = S_bunchsize,
                                                                          i_bunchsize = i_bunchsize,
                                                                          T_bunchsize = T_bunchsize)
    bufsize2         = RMP2_K_forloop_S_i_determine_buf_size_intermediates_forloop(NVIR = NVIR,
                                                                                   NOCC = NOCC,
                                                                                   N_LAPLACE = N_LAPLACE,
                                                                                   NTHC_INT = NTHC_INT,
                                                                                   S_bunchsize = S_bunchsize,
                                                                                   i_bunchsize = i_bunchsize,
                                                                                   T_bunchsize = T_bunchsize)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    offset_now       = (bufsize1 * _itemsize)
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # declare some useful variables to trace offset for each loop
    offset_S         = None            
    offset_S_i       = None            
    offset_S_i_T     = None            
    # step   0 start for loop with indices ()
    # step   1 allocate   _M12
    _M12             = 0.0             
    # step   2 jS,jT->STj
    _M2_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    offset_now       = (_M2_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_8.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_8.shape[0]),
                             ctypes.c_int(_INPUT_8.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   3 jQ,STj->QST
    _M7_offset       = offset_now      
    _M7_offset       = min(_M7_offset, _M2_offset)
    offset_now       = _M7_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    offset_now       = (_M7_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_3.shape[0]
    _INPUT_3_reshaped = _INPUT_3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_3_reshaped.T, _M2_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M7.ravel()[:] = ddot_buffer.ravel()[:]
    # step   4 aS,aT->STa
    _M3_offset       = offset_now      
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M3              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    offset_now       = (_M3_offset + tmp_itemsize)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M3.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    # step   5 aP,STa->PST
    _M4_offset       = offset_now      
    _M4_offset       = min(_M4_offset, _M3_offset)
    offset_now       = _M4_offset      
    ddot_buffer      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_2.shape[0]
    _INPUT_2_reshaped = _INPUT_2.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(ddot_buffer.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
    ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_2_reshaped.T, _M3_reshaped.T, c=ddot_buffer_reshaped)
    ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
    _M4.ravel()[:] = ddot_buffer.ravel()[:]
    # step   6 iP,iT->PTi
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
    # step   7 start for loop with indices ('S',)
    for S_0, S_1 in lib.prange(0,NTHC_INT,S_bunchsize):
        if offset_S == None:
            offset_S         = offset_now      
        else:
            offset_now       = offset_S        
        # step   8 start for loop with indices ('S', 'i')
        for i_0, i_1 in lib.prange(0,NOCC,i_bunchsize):
            if offset_S_i == None:
                offset_S_i       = offset_now      
            else:
                offset_now       = offset_S_i      
            # step   9 slice _INPUT_5 with indices ['S']
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
            # step  10 slice _INPUT_6 with indices ['i']
            _INPUT_6_sliced_offset = offset_now      
            _INPUT_6_sliced  = np.ndarray(((i_1-i_0), NTHC_INT), buffer = buffer, offset = _INPUT_6_sliced_offset)
            size_item        = ((i_1-i_0) * (NTHC_INT * _itemsize))
            offset_now       = (offset_now + size_item)
            fn_slice_2_0(ctypes.c_void_p(_INPUT_6.ctypes.data),
                         ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                         ctypes.c_int(_INPUT_6.shape[0]),
                         ctypes.c_int(_INPUT_6.shape[1]),
                         ctypes.c_int(i_0),
                         ctypes.c_int(i_1))
            # step  11 RS,iR->SiR
            _M1_offset       = offset_now      
            _M1_offset       = min(_M1_offset, _INPUT_5_sliced_offset)
            _M1_offset       = min(_M1_offset, _INPUT_6_sliced_offset)
            offset_now       = _M1_offset      
            tmp_itemsize     = ((S_1-S_0) * ((i_1-i_0) * (NTHC_INT * _itemsize)))
            _M1              = np.ndarray(((S_1-S_0), (i_1-i_0), NTHC_INT), buffer = buffer, offset = _M1_offset)
            offset_now       = (_M1_offset + tmp_itemsize)
            fn_contraction_01_20_120(ctypes.c_void_p(_INPUT_5_sliced.ctypes.data),
                                     ctypes.c_void_p(_INPUT_6_sliced.ctypes.data),
                                     ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_int(_INPUT_5_sliced.shape[0]),
                                     ctypes.c_int(_INPUT_5_sliced.shape[1]),
                                     ctypes.c_int(_INPUT_6_sliced.shape[0]),
                                     ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  12 bR,SiR->bSi
            _M11_offset      = offset_now      
            _M11_offset      = min(_M11_offset, _M1_offset)
            offset_now       = _M11_offset     
            ddot_buffer      = np.ndarray((NVIR, (S_1-S_0), (i_1-i_0)), buffer = linearop_buf)
            tmp_itemsize     = (NVIR * ((S_1-S_0) * ((i_1-i_0) * _itemsize)))
            _M11             = np.ndarray((NVIR, (S_1-S_0), (i_1-i_0)), buffer = buffer, offset = _M11_offset)
            offset_now       = (_M11_offset + tmp_itemsize)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
            _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * _M1.shape[0]
            _size_dim_1      = _size_dim_1 * _M1.shape[1]
            _M1_reshaped = _M1.reshape(_size_dim_1,-1)
            shape_backup = copy.deepcopy(ddot_buffer.shape)
            _size_dim_1      = 1               
            _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
            ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
            lib.ddot(_INPUT_7_reshaped, _M1_reshaped.T, c=ddot_buffer_reshaped)
            ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
            _M11.ravel()[:] = ddot_buffer.ravel()[:]
            # step  13 allocate   _M10
            _M10             = np.ndarray((NVIR, (i_1-i_0), (S_1-S_0)), buffer = buffer, offset = offset_now)
            tmp_itemsize     = (NVIR * ((i_1-i_0) * ((S_1-S_0) * _itemsize)))
            _M10_offset      = offset_now      
            offset_now       = (offset_now + tmp_itemsize)
            _M10.ravel()[:] = 0.0
            # step  14 bSi->biS
            _M11_perm_offset = _M11_offset     
            _M11_perm        = np.ndarray((NVIR, (i_1-i_0), (S_1-S_0)), buffer = buffer, offset = _M11_perm_offset)
            fn_permutation_012_021(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(NVIR),
                                   ctypes.c_int((S_1-S_0)),
                                   ctypes.c_int((i_1-i_0)),
                                   ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  15 start for loop with indices ('S', 'i', 'T')
            for T_0, T_1 in lib.prange(0,N_LAPLACE,T_bunchsize):
                if offset_S_i_T == None:
                    offset_S_i_T     = offset_now      
                else:
                    offset_now       = offset_S_i_T    
                # step  16 slice _M0 with indices ['T', 'i']
                _M0_sliced_offset = offset_now      
                _M0_sliced       = np.ndarray((NTHC_INT, (T_1-T_0), (i_1-i_0)), buffer = buffer, offset = _M0_sliced_offset)
                size_item        = (NTHC_INT * ((T_1-T_0) * ((i_1-i_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M0.ctypes.data),
                               ctypes.c_void_p(_M0_sliced.ctypes.data),
                               ctypes.c_int(_M0.shape[0]),
                               ctypes.c_int(_M0.shape[1]),
                               ctypes.c_int(_M0.shape[2]),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1),
                               ctypes.c_int(i_0),
                               ctypes.c_int(i_1))
                # step  17 slice _M4 with indices ['S', 'T']
                _M4_sliced_offset = offset_now      
                _M4_sliced       = np.ndarray((NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M4_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_sliced.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]),
                               ctypes.c_int(S_0),
                               ctypes.c_int(S_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  18 PTi,PST->iSPT
                _M5_offset       = offset_now      
                _M5_offset       = min(_M5_offset, _M0_sliced_offset)
                _M5_offset       = min(_M5_offset, _M4_sliced_offset)
                offset_now       = _M5_offset      
                tmp_itemsize     = ((i_1-i_0) * ((S_1-S_0) * (NTHC_INT * ((T_1-T_0) * _itemsize))))
                _M5              = np.ndarray(((i_1-i_0), (S_1-S_0), NTHC_INT, (T_1-T_0)), buffer = buffer, offset = _M5_offset)
                offset_now       = (_M5_offset + tmp_itemsize)
                fn_contraction_012_031_2301(ctypes.c_void_p(_M0_sliced.ctypes.data),
                                            ctypes.c_void_p(_M4_sliced.ctypes.data),
                                            ctypes.c_void_p(_M5.ctypes.data),
                                            ctypes.c_int(_M0_sliced.shape[0]),
                                            ctypes.c_int(_M0_sliced.shape[1]),
                                            ctypes.c_int(_M0_sliced.shape[2]),
                                            ctypes.c_int(_M4_sliced.shape[1]),
                                            ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  19 iSPT->iSTP
                _M5_perm_offset  = _M5_offset      
                _M5_perm         = np.ndarray(((i_1-i_0), (S_1-S_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
                fn_permutation_0123_0132(ctypes.c_void_p(_M5.ctypes.data),
                                         ctypes.c_void_p(_M5_perm.ctypes.data),
                                         ctypes.c_int((i_1-i_0)),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  20 PQ,iSTP->QiST
                _M6_offset       = offset_now      
                _M6_offset       = min(_M6_offset, _M5_perm_offset)
                offset_now       = _M6_offset      
                ddot_buffer      = np.ndarray((NTHC_INT, (i_1-i_0), (S_1-S_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NTHC_INT * ((i_1-i_0) * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M6              = np.ndarray((NTHC_INT, (i_1-i_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M6_offset)
                offset_now       = (_M6_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
                _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M5_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M5_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M5_perm.shape[2]
                _M5_perm_reshaped = _M5_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_0_reshaped.T, _M5_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M6.ravel()[:] = ddot_buffer.ravel()[:]
                # step  21 slice _M7 with indices ['S', 'T']
                _M7_sliced_offset = offset_now      
                _M7_sliced       = np.ndarray((NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M7_sliced_offset)
                size_item        = (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize)))
                offset_now       = (offset_now + size_item)
                fn_slice_3_1_2(ctypes.c_void_p(_M7.ctypes.data),
                               ctypes.c_void_p(_M7_sliced.ctypes.data),
                               ctypes.c_int(_M7.shape[0]),
                               ctypes.c_int(_M7.shape[1]),
                               ctypes.c_int(_M7.shape[2]),
                               ctypes.c_int(S_0),
                               ctypes.c_int(S_1),
                               ctypes.c_int(T_0),
                               ctypes.c_int(T_1))
                # step  22 QiST,QST->iQST
                _M8_offset       = offset_now      
                _M8_offset       = min(_M8_offset, _M6_offset)
                _M8_offset       = min(_M8_offset, _M7_sliced_offset)
                offset_now       = _M8_offset      
                tmp_itemsize     = ((i_1-i_0) * (NTHC_INT * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M8              = np.ndarray(((i_1-i_0), NTHC_INT, (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M8_offset)
                offset_now       = (_M8_offset + tmp_itemsize)
                fn_contraction_0123_023_1023(ctypes.c_void_p(_M6.ctypes.data),
                                             ctypes.c_void_p(_M7_sliced.ctypes.data),
                                             ctypes.c_void_p(_M8.ctypes.data),
                                             ctypes.c_int(_M6.shape[0]),
                                             ctypes.c_int(_M6.shape[1]),
                                             ctypes.c_int(_M6.shape[2]),
                                             ctypes.c_int(_M6.shape[3]),
                                             ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  23 iQST->iSTQ
                _M8_perm_offset  = _M8_offset      
                _M8_perm         = np.ndarray(((i_1-i_0), (S_1-S_0), (T_1-T_0), NTHC_INT), buffer = buffer, offset = _M8_perm_offset)
                fn_permutation_0123_0231(ctypes.c_void_p(_M8.ctypes.data),
                                         ctypes.c_void_p(_M8_perm.ctypes.data),
                                         ctypes.c_int((i_1-i_0)),
                                         ctypes.c_int(NTHC_INT),
                                         ctypes.c_int((S_1-S_0)),
                                         ctypes.c_int((T_1-T_0)),
                                         ctypes.c_void_p(linearop_buf.ctypes.data))
                # step  24 bQ,iSTQ->biST
                _M9_offset       = offset_now      
                _M9_offset       = min(_M9_offset, _M8_perm_offset)
                offset_now       = _M9_offset      
                ddot_buffer      = np.ndarray((NVIR, (i_1-i_0), (S_1-S_0), (T_1-T_0)), buffer = linearop_buf)
                tmp_itemsize     = (NVIR * ((i_1-i_0) * ((S_1-S_0) * ((T_1-T_0) * _itemsize))))
                _M9              = np.ndarray((NVIR, (i_1-i_0), (S_1-S_0), (T_1-T_0)), buffer = buffer, offset = _M9_offset)
                offset_now       = (_M9_offset + tmp_itemsize)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _INPUT_4.shape[0]
                _INPUT_4_reshaped = _INPUT_4.reshape(_size_dim_1,-1)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[0]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[1]
                _size_dim_1      = _size_dim_1 * _M8_perm.shape[2]
                _M8_perm_reshaped = _M8_perm.reshape(_size_dim_1,-1)
                shape_backup = copy.deepcopy(ddot_buffer.shape)
                _size_dim_1      = 1               
                _size_dim_1      = _size_dim_1 * ddot_buffer.shape[0]
                ddot_buffer_reshaped = ddot_buffer.reshape(_size_dim_1,-1)
                lib.ddot(_INPUT_4_reshaped, _M8_perm_reshaped.T, c=ddot_buffer_reshaped)
                ddot_buffer      = ddot_buffer_reshaped.reshape(*shape_backup)
                _M9.ravel()[:] = ddot_buffer.ravel()[:]
                # step  25 slice _INPUT_13 with indices ['T']
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
                # step  26 bT,biST->biS
                fn_contraction_01_0231_023_plus(ctypes.c_void_p(_INPUT_13_sliced.ctypes.data),
                                                ctypes.c_void_p(_M9.ctypes.data),
                                                ctypes.c_void_p(_M10.ctypes.data),
                                                ctypes.c_int(_INPUT_13_sliced.shape[0]),
                                                ctypes.c_int(_INPUT_13_sliced.shape[1]),
                                                ctypes.c_int(_M9.shape[1]),
                                                ctypes.c_int(_M9.shape[2]),
                                                ctypes.c_void_p(linearop_buf.ctypes.data))
            # step  27 end   for loop with indices ('S', 'i', 'T')
            # step  28 biS,biS->
            output_tmp       = ctypes.c_double(0.0)
            fn_dot(ctypes.c_void_p(_M10.ctypes.data),
                   ctypes.c_void_p(_M11_perm.ctypes.data),
                   ctypes.c_int(_M10.size),
                   ctypes.pointer(output_tmp))
            output_tmp = output_tmp.value
            _M12 += output_tmp
        # step  29 end   for loop with indices ('S', 'i')
    # step  30 end   for loop with indices ('S',)
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
    # test for RMP2_K_forloop_P_b and RMP2_K_forloop_P_b_naive
    benchmark        = RMP2_K_forloop_P_b_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_P_b(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_P_b_forloop_P_b
    output3          = RMP2_K_forloop_P_b_forloop_P_b(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_P_j and RMP2_K_forloop_P_j_naive
    benchmark        = RMP2_K_forloop_P_j_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_P_j(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_P_j_forloop_P_j
    output3          = RMP2_K_forloop_P_j_forloop_P_j(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_Q_a and RMP2_K_forloop_Q_a_naive
    benchmark        = RMP2_K_forloop_Q_a_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_Q_a(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_Q_a_forloop_Q_a
    output3          = RMP2_K_forloop_Q_a_forloop_Q_a(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_Q_i and RMP2_K_forloop_Q_i_naive
    benchmark        = RMP2_K_forloop_Q_i_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_Q_i(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_Q_i_forloop_Q_i
    output3          = RMP2_K_forloop_Q_i_forloop_Q_i(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_R_a and RMP2_K_forloop_R_a_naive
    benchmark        = RMP2_K_forloop_R_a_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_R_a(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_R_a_forloop_R_a
    output3          = RMP2_K_forloop_R_a_forloop_R_a(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_R_j and RMP2_K_forloop_R_j_naive
    benchmark        = RMP2_K_forloop_R_j_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_R_j(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_R_j_forloop_R_j
    output3          = RMP2_K_forloop_R_j_forloop_R_j(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_S_b and RMP2_K_forloop_S_b_naive
    benchmark        = RMP2_K_forloop_S_b_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_S_b(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_S_b_forloop_S_b
    output3          = RMP2_K_forloop_S_b_forloop_S_b(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
    # test for RMP2_K_forloop_S_i and RMP2_K_forloop_S_i_naive
    benchmark        = RMP2_K_forloop_S_i_naive(Z               ,
                                                X_o             ,
                                                X_v             ,
                                                tau_o           ,
                                                tau_v           )
    output           = RMP2_K_forloop_S_i(Z               ,
                                          X_o             ,
                                          X_v             ,
                                          tau_o           ,
                                          tau_v           )
    assert np.allclose(output, benchmark)
    print(output)   
    print(benchmark)
    # test for RMP2_K_forloop_S_i_forloop_S_i
    output3          = RMP2_K_forloop_S_i_forloop_S_i(Z               ,
                                                      X_o             ,
                                                      X_v             ,
                                                      tau_o           ,
                                                      tau_v           ,
                                                      buffer          )
    assert np.allclose(output3, benchmark)
    print(output3)  
