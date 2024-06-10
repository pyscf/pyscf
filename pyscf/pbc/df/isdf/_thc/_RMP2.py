import numpy
import numpy as np
import ctypes
import copy
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

def RMP2_J_naive(Z           : np.ndarray,
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
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("jQ,jT->QTj"    , _INPUT_3        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("jS,QTj->SQT"   , _INPUT_8        , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("bQ,bT->QTb"    , _INPUT_4        , _INPUT_13       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("bS,QTb->SQT"   , _INPUT_9        , _M5             )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("SQT,SQT->SQT"  , _M6             , _M8             )
    del _M6         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("RS,SQT->RQT"   , _INPUT_5        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (1, 0, 2)       )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iP,iT->PTi"    , _INPUT_1        , _INPUT_10       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("iR,PTi->RPT"   , _INPUT_6        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("aP,aT->PTa"    , _INPUT_2        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("aR,PTa->RPT"   , _INPUT_7        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("RPT,RPT->RPT"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("PQ,RTP->QRT"   , _INPUT_0        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("QRT,QRT->"     , _M10            , _M11_perm       )
    del _M10        
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    return _M12

def RMP2_J(Z           : np.ndarray,
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
    # step 0 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M7.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    _M5              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M5.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bS,QTb->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M6.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M5_reshaped.T, c=_M6_reshaped)
    _M6          = _M6_reshaped.reshape(*shape_backup)
    del _M5         
    del _M5_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 SQT,SQT->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012 = getattr(libpbc, "fn_contraction_012_012_012", None)
    assert fn_contraction_012_012_012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M8.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,SQT->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M9_reshaped, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 RQT->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_102 = getattr(libpbc, "fn_permutation_012_102", None)
    assert fn_permutation_012_102 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_102(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iR,PTi->RPT 
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 aP,aT->PTa 
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
    _benchmark_time(t1, t2, "step 10")
    # step 10 aR,PTa->RPT 
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
    _benchmark_time(t1, t2, "step 11")
    # step 11 RPT,RPT->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012 = getattr(libpbc, "fn_contraction_012_012_012", None)
    assert fn_contraction_012_012_012 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012(ctypes.c_void_p(_M1.ctypes.data),
                               ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_int(_M1.shape[0]),
                               ctypes.c_int(_M1.shape[1]),
                               ctypes.c_int(_M1.shape[2]),
                               ctypes.c_void_p(_buffer.ctypes.data))
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RPT->RTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _buffer          = np.ndarray((nthreads, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_021(ctypes.c_void_p(_M4.ctypes.data),
                           ctypes.c_void_p(_M4_perm.ctypes.data),
                           ctypes.c_int(_M4.shape[0]),
                           ctypes.c_int(_M4.shape[1]),
                           ctypes.c_int(_M4.shape[2]),
                           ctypes.c_void_p(_buffer.ctypes.data))
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 PQ,RTP->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M4_perm_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M4_perm    
    del _M4_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 QRT,QRT-> 
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

def RMP2_J_determine_buf_head_size(NVIR        : int,
                                   NOCC        : int,
                                   N_LAPLACE   : int,
                                   NTHC_INT    : int):
    # init
    output           = 0               
    # cmpr _INPUT_0
    tmp              = (NTHC_INT * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_1
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_2
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_3
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_4
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_5
    tmp              = (NTHC_INT * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_6
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_7
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_8
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_9
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_10
    tmp              = (NOCC * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _INPUT_11
    tmp              = (NOCC * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _INPUT_12
    tmp              = (NVIR * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _INPUT_13
    tmp              = (NVIR * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M11_perm
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M4_perm
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    return output

def RMP2_J_determine_buf_size_intermediates(NVIR        : int,
                                            NOCC        : int,
                                            N_LAPLACE   : int,
                                            NTHC_INT    : int):
    # init
    output           = 0               
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M12_size        = 1               
    # cmpr _M7 
    size_now         = 0               
    size_now         = (size_now + _M7_size)
    output           = max(output, size_now)
    # cmpr _M8 
    size_now         = 0               
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M8 _M5 
    size_now         = 0               
    size_now         = (size_now + _M8_size)
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M8 _M6 
    size_now         = 0               
    size_now         = (size_now + _M8_size)
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M9 
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    output           = max(output, size_now)
    # cmpr _M11 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    output           = max(output, size_now)
    # cmpr _M11 _M2 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M11 _M3 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M11 _M3 _M0 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M11 _M3 _M1 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M3_size)
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M11 _M4 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M4_size)
    output           = max(output, size_now)
    # cmpr _M11 _M10 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    size_now         = (size_now + _M10_size)
    output           = max(output, size_now)
    # cmpr _M12 
    size_now         = 0               
    size_now         = (size_now + _M12_size)
    output           = max(output, size_now)
    return output

def RMP2_J_opt_mem(Z           : np.ndarray,
                   X_o         : np.ndarray,
                   X_v         : np.ndarray,
                   tau_o       : np.ndarray,
                   tau_v       : np.ndarray,
                   buffer      : np.ndarray):
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
    # deal with buffer
    bufsize1         = RMP2_J_determine_buf_head_size(NVIR = NVIR,
                                                      NOCC = NOCC,
                                                      N_LAPLACE = N_LAPLACE,
                                                      NTHC_INT = NTHC_INT)
    bufsize2         = RMP2_J_determine_buf_size_intermediates(NVIR = NVIR,
                                                               NOCC = NOCC,
                                                               N_LAPLACE = N_LAPLACE,
                                                               NTHC_INT = NTHC_INT)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    bufsize1_item    = (bufsize1 * _itemsize)
    offset_now       = bufsize1_item   
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # step 0 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M7_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M7.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M8_offset       = _M7_offset      
    offset_now       = (_M7_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M8_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M7_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M8.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M5_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M5              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M5_offset)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_INPUT_13.ctypes.data),
                             ctypes.c_void_p(_M5.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_INPUT_13.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bS,QTb->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6_offset       = _M5_offset      
    offset_now       = (_M5_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_9_reshaped.T, _M5_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M6.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 SQT,SQT->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012 = getattr(libpbc, "fn_contraction_012_012_012", None)
    assert fn_contraction_012_012_012 is not None
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M9_offset       = min(_M6_offset, _M8_offset)
    offset_now       = (_M9_offset + tmp_itemsize)
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    fn_contraction_012_012_012(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M8.ctypes.data),
                               ctypes.c_void_p(_M9.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]),
                               ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,SQT->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11_offset      = _M9_offset      
    offset_now       = (_M9_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped, _M9_reshaped, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M11.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 RQT->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_102 = getattr(libpbc, "fn_permutation_012_102", None)
    assert fn_permutation_012_102 is not None
    _M11_perm_offset = _M11_offset     
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_perm_offset)
    fn_permutation_012_102(ctypes.c_void_p(_M11.ctypes.data),
                           ctypes.c_void_p(_M11_perm.ctypes.data),
                           ctypes.c_int(_M11.shape[0]),
                           ctypes.c_int(_M11.shape[1]),
                           ctypes.c_int(_M11.shape[2]),
                           ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M2_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_10.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_10.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iR,PTi->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M3_offset       = _M2_offset      
    offset_now       = (_M2_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M2_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M3.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M0_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_int(_INPUT_2.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 aR,PTa->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M1_offset       = _M0_offset      
    offset_now       = (_M0_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M0_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M1.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 RPT,RPT->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012 = getattr(libpbc, "fn_contraction_012_012_012", None)
    assert fn_contraction_012_012_012 is not None
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M4_offset       = min(_M1_offset, _M3_offset)
    offset_now       = (_M4_offset + tmp_itemsize)
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012(ctypes.c_void_p(_M1.ctypes.data),
                               ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_int(_M1.shape[0]),
                               ctypes.c_int(_M1.shape[1]),
                               ctypes.c_int(_M1.shape[2]),
                               ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RPT->RTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021 = getattr(libpbc, "fn_permutation_012_021", None)
    assert fn_permutation_012_021 is not None
    _M4_perm_offset  = _M4_offset      
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021(ctypes.c_void_p(_M4.ctypes.data),
                           ctypes.c_void_p(_M4_perm.ctypes.data),
                           ctypes.c_int(_M4.shape[0]),
                           ctypes.c_int(_M4.shape[1]),
                           ctypes.c_int(_M4.shape[2]),
                           ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 PQ,RTP->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = _M4_perm_offset 
    offset_now       = (_M4_perm_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M4_perm.shape[1]
    _M4_perm_reshaped = _M4_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M4_perm_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M10.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 QRT,QRT-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M10.ctypes.data),
           ctypes.c_void_p(_M11_perm.ctypes.data),
           ctypes.c_int(_M10.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    return _M12

def RMP2_K_naive(Z           : np.ndarray,
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
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,aP->iaP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,iaP->Qia"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aS,aT->STa"    , _INPUT_9        , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("Qia,STa->QiST" , _M1             , _M4             )
    del _M1         
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("iT,QiST->QSiT" , _INPUT_10       , _M5             )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("jQ,jT->QTj"    , _INPUT_3        , _INPUT_11       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("jS,QTj->SQT"   , _INPUT_8        , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("QSiT,SQT->iQST", _M6             , _M8             )
    del _M6         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9_perm         = np.transpose(_M9             , (1, 3, 2, 0)    )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iR,bR->ibR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,ibR->Sib"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("Sib,QTSi->bQT" , _M3             , _M9_perm        )
    del _M3         
    del _M9_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("bQ,bQT->bT"    , _INPUT_4        , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("bT,bT->"       , _INPUT_13       , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    return _M12

def RMP2_K(Z           : np.ndarray,
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
    # step 0 iP,aP->iaP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    _buffer          = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 PQ,iaP->Qia 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
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
    # step 3 Qia,STa->QiST 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M1         
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iT,QiST->QSiT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301 = getattr(libpbc, "fn_contraction_01_2031_2301", None)
    assert fn_contraction_01_2031_2301 is not None
    _buffer          = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                ctypes.c_void_p(_M5.ctypes.data),
                                ctypes.c_void_p(_M6.ctypes.data),
                                ctypes.c_int(_INPUT_10.shape[0]),
                                ctypes.c_int(_INPUT_10.shape[1]),
                                ctypes.c_int(_M5.shape[0]),
                                ctypes.c_int(_M5.shape[2]),
                                ctypes.c_void_p(_buffer.ctypes.data))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    _buffer          = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M7.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QSiT,SQT->iQST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_103_2013 = getattr(libpbc, "fn_contraction_0123_103_2013", None)
    assert fn_contraction_0123_103_2013 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M9              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_103_2013(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]),
                                 ctypes.c_void_p(_buffer.ctypes.data))
    del _M6         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iQST->QTSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1320 = getattr(libpbc, "fn_permutation_0123_1320", None)
    assert fn_permutation_0123_1320 is not None
    _buffer          = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _M9_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_1320(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iR,bR->ibR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    _buffer          = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    _M2              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RS,ibR->Sib 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 Sib,QTSi->bQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M9_perm.shape[1]
    _M9_perm_reshaped = _M9_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped.T, _M9_perm_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M3         
    del _M9_perm    
    del _M3_reshaped
    del _M9_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bQ,bQT->bT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_012_02 = getattr(libpbc, "fn_contraction_01_012_02", None)
    assert fn_contraction_01_012_02 is not None
    _buffer          = np.ndarray((NVIR, N_LAPLACE), dtype=np.float64)
    _M11             = np.ndarray((NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_012_02(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_M10.ctypes.data),
                             ctypes.c_void_p(_M11.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_M10.shape[2]),
                             ctypes.c_void_p(_buffer.ctypes.data))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 bT,bT-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_INPUT_13.ctypes.data),
           ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_int(_INPUT_13.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    return _M12

def RMP2_K_determine_buf_head_size(NVIR        : int,
                                   NOCC        : int,
                                   N_LAPLACE   : int,
                                   NTHC_INT    : int):
    # init
    output           = 0               
    # cmpr _INPUT_0
    tmp              = (NTHC_INT * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_1
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_2
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_3
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_4
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_5
    tmp              = (NTHC_INT * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_6
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_7
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_8
    tmp              = (NOCC * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_9
    tmp              = (NVIR * NTHC_INT)
    output           = max(output, tmp)
    # cmpr _INPUT_10
    tmp              = (NOCC * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _INPUT_11
    tmp              = (NOCC * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _INPUT_12
    tmp              = (NVIR * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _INPUT_13
    tmp              = (NVIR * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _M0
    tmp              = (NOCC * (NVIR * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M1
    tmp              = (NTHC_INT * (NOCC * NVIR))
    output           = max(output, tmp)
    # cmpr _M2
    tmp              = (NOCC * (NVIR * NTHC_INT))
    output           = max(output, tmp)
    # cmpr _M3
    tmp              = (NTHC_INT * (NOCC * NVIR))
    output           = max(output, tmp)
    # cmpr _M4
    tmp              = (NTHC_INT * (N_LAPLACE * NVIR))
    output           = max(output, tmp)
    # cmpr _M5
    tmp              = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    output           = max(output, tmp)
    # cmpr _M6
    tmp              = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    output           = max(output, tmp)
    # cmpr _M7
    tmp              = (NTHC_INT * (N_LAPLACE * NOCC))
    output           = max(output, tmp)
    # cmpr _M8
    tmp              = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M9
    tmp              = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    output           = max(output, tmp)
    # cmpr _M10
    tmp              = (NVIR * (NTHC_INT * N_LAPLACE))
    output           = max(output, tmp)
    # cmpr _M11
    tmp              = (NVIR * N_LAPLACE)
    output           = max(output, tmp)
    # cmpr _M9_perm
    tmp              = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    output           = max(output, tmp)
    return output

def RMP2_K_determine_buf_size_intermediates(NVIR        : int,
                                            NOCC        : int,
                                            N_LAPLACE   : int,
                                            NTHC_INT    : int):
    # init
    output           = 0               
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M4_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M5_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M2_size         = (NOCC * (NVIR * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NVIR))
    _M10_size        = (NVIR * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NVIR * N_LAPLACE)
    _M12_size        = 1               
    # cmpr _M0 
    size_now         = 0               
    size_now         = (size_now + _M0_size)
    output           = max(output, size_now)
    # cmpr _M1 
    size_now         = 0               
    size_now         = (size_now + _M1_size)
    output           = max(output, size_now)
    # cmpr _M1 _M4 
    size_now         = 0               
    size_now         = (size_now + _M1_size)
    size_now         = (size_now + _M4_size)
    output           = max(output, size_now)
    # cmpr _M5 
    size_now         = 0               
    size_now         = (size_now + _M5_size)
    output           = max(output, size_now)
    # cmpr _M6 
    size_now         = 0               
    size_now         = (size_now + _M6_size)
    output           = max(output, size_now)
    # cmpr _M6 _M7 
    size_now         = 0               
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M7_size)
    output           = max(output, size_now)
    # cmpr _M6 _M8 
    size_now         = 0               
    size_now         = (size_now + _M6_size)
    size_now         = (size_now + _M8_size)
    output           = max(output, size_now)
    # cmpr _M9 
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    output           = max(output, size_now)
    # cmpr _M9 _M2 
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M2_size)
    output           = max(output, size_now)
    # cmpr _M9 _M3 
    size_now         = 0               
    size_now         = (size_now + _M9_size)
    size_now         = (size_now + _M3_size)
    output           = max(output, size_now)
    # cmpr _M10 
    size_now         = 0               
    size_now         = (size_now + _M10_size)
    output           = max(output, size_now)
    # cmpr _M11 
    size_now         = 0               
    size_now         = (size_now + _M11_size)
    output           = max(output, size_now)
    # cmpr _M12 
    size_now         = 0               
    size_now         = (size_now + _M12_size)
    output           = max(output, size_now)
    return output

def RMP2_K_opt_mem(Z           : np.ndarray,
                   X_o         : np.ndarray,
                   X_v         : np.ndarray,
                   tau_o       : np.ndarray,
                   tau_v       : np.ndarray,
                   buffer      : np.ndarray):
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
    # deal with buffer
    bufsize1         = RMP2_K_determine_buf_head_size(NVIR = NVIR,
                                                      NOCC = NOCC,
                                                      N_LAPLACE = N_LAPLACE,
                                                      NTHC_INT = NTHC_INT)
    bufsize2         = RMP2_K_determine_buf_size_intermediates(NVIR = NVIR,
                                                               NOCC = NOCC,
                                                               N_LAPLACE = N_LAPLACE,
                                                               NTHC_INT = NTHC_INT)
    bufsize          = (bufsize1 + bufsize2)
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    _itemsize        = buffer.itemsize 
    bufsize1_item    = (bufsize1 * _itemsize)
    offset_now       = bufsize1_item   
    linearop_buf     = np.ndarray((bufsize1), buffer = buffer)
    # step 0 iP,aP->iaP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    tmp_itemsize     = (NOCC * (NVIR * (NTHC_INT * _itemsize)))
    _M0_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_1.ctypes.data),
                             ctypes.c_void_p(_INPUT_2.ctypes.data),
                             ctypes.c_void_p(_M0.ctypes.data),
                             ctypes.c_int(_INPUT_1.shape[0]),
                             ctypes.c_int(_INPUT_1.shape[1]),
                             ctypes.c_int(_INPUT_2.shape[0]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 PQ,iaP->Qia 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = _M0_offset      
    offset_now       = (_M0_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = linearop_buf)
    _M1              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M0_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M1.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 aS,aT->STa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NVIR * _itemsize)))
    _M4_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_9.ctypes.data),
                             ctypes.c_void_p(_INPUT_12.ctypes.data),
                             ctypes.c_void_p(_M4.ctypes.data),
                             ctypes.c_int(_INPUT_9.shape[0]),
                             ctypes.c_int(_INPUT_9.shape[1]),
                             ctypes.c_int(_INPUT_12.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qia,STa->QiST 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M5_offset       = min(_M1_offset, _M4_offset)
    offset_now       = (_M5_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M5              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[1]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M4_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M5.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iT,QiST->QSiT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301 = getattr(libpbc, "fn_contraction_01_2031_2301", None)
    assert fn_contraction_01_2031_2301 is not None
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (NOCC * (N_LAPLACE * _itemsize))))
    _M6_offset       = _M5_offset      
    offset_now       = (_M5_offset + tmp_itemsize)
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_2031_2301(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                ctypes.c_void_p(_M5.ctypes.data),
                                ctypes.c_void_p(_M6.ctypes.data),
                                ctypes.c_int(_INPUT_10.shape[0]),
                                ctypes.c_int(_INPUT_10.shape[1]),
                                ctypes.c_int(_M5.shape[0]),
                                ctypes.c_int(_M5.shape[2]),
                                ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120 = getattr(libpbc, "fn_contraction_01_02_120", None)
    assert fn_contraction_01_02_120 is not None
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NOCC * _itemsize)))
    _M7_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120(ctypes.c_void_p(_INPUT_3.ctypes.data),
                             ctypes.c_void_p(_INPUT_11.ctypes.data),
                             ctypes.c_void_p(_M7.ctypes.data),
                             ctypes.c_int(_INPUT_3.shape[0]),
                             ctypes.c_int(_INPUT_3.shape[1]),
                             ctypes.c_int(_INPUT_11.shape[1]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M8_offset       = _M7_offset      
    offset_now       = (_M7_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M8_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M7_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M8.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QSiT,SQT->iQST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_103_2013 = getattr(libpbc, "fn_contraction_0123_103_2013", None)
    assert fn_contraction_0123_103_2013 is not None
    tmp_itemsize     = (NOCC * (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M9_offset       = min(_M6_offset, _M8_offset)
    offset_now       = (_M9_offset + tmp_itemsize)
    _M9              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    fn_contraction_0123_103_2013(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]),
                                 ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iQST->QTSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1320 = getattr(libpbc, "fn_permutation_0123_1320", None)
    assert fn_permutation_0123_1320 is not None
    _M9_perm_offset  = _M9_offset      
    _M9_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), buffer = buffer, offset = _M9_perm_offset)
    fn_permutation_0123_1320(ctypes.c_void_p(_M9.ctypes.data),
                             ctypes.c_void_p(_M9_perm.ctypes.data),
                             ctypes.c_int(_M9.shape[0]),
                             ctypes.c_int(_M9.shape[1]),
                             ctypes.c_int(_M9.shape[2]),
                             ctypes.c_int(_M9.shape[3]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iR,bR->ibR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021 = getattr(libpbc, "fn_contraction_01_21_021", None)
    assert fn_contraction_01_21_021 is not None
    tmp_itemsize     = (NOCC * (NVIR * (NTHC_INT * _itemsize)))
    _M2_offset       = offset_now      
    offset_now       = (offset_now + tmp_itemsize)
    _M2              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021(ctypes.c_void_p(_INPUT_6.ctypes.data),
                             ctypes.c_void_p(_INPUT_7.ctypes.data),
                             ctypes.c_void_p(_M2.ctypes.data),
                             ctypes.c_int(_INPUT_6.shape[0]),
                             ctypes.c_int(_INPUT_6.shape[1]),
                             ctypes.c_int(_INPUT_7.shape[0]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RS,ibR->Sib 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M3_offset       = _M2_offset      
    offset_now       = (_M2_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = linearop_buf)
    _M3              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M2_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M3.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 Sib,QTSi->bQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NVIR * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = min(_M3_offset, _M9_perm_offset)
    offset_now       = (_M10_offset + tmp_itemsize)
    buffer_ddot      = np.ndarray((NVIR, NTHC_INT, N_LAPLACE), buffer = linearop_buf)
    _M10             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M9_perm.shape[1]
    _M9_perm_reshaped = _M9_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(buffer_ddot.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * buffer_ddot.shape[0]
    buffer_ddot_reshaped = buffer_ddot.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped.T, _M9_perm_reshaped.T, c=buffer_ddot_reshaped)
    buffer_ddot      = buffer_ddot_reshaped.reshape(*shape_backup)
    _M10.ravel()[:] = buffer_ddot.ravel()[:]
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bQ,bQT->bT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_012_02 = getattr(libpbc, "fn_contraction_01_012_02", None)
    assert fn_contraction_01_012_02 is not None
    tmp_itemsize     = (NVIR * (N_LAPLACE * _itemsize))
    _M11_offset      = _M10_offset     
    offset_now       = (_M10_offset + tmp_itemsize)
    _M11             = np.ndarray((NVIR, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_012_02(ctypes.c_void_p(_INPUT_4.ctypes.data),
                             ctypes.c_void_p(_M10.ctypes.data),
                             ctypes.c_void_p(_M11.ctypes.data),
                             ctypes.c_int(_INPUT_4.shape[0]),
                             ctypes.c_int(_INPUT_4.shape[1]),
                             ctypes.c_int(_M10.shape[2]),
                             ctypes.c_void_p(linearop_buf.ctypes.data))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 bT,bT-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M12             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_INPUT_13.ctypes.data),
           ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_int(_INPUT_13.size),
           ctypes.pointer(_M12))
    _M12 = _M12.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
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
    # test for RMP2_J and RMP2_J_naive
    benchmark        = RMP2_J_naive(Z               ,
                                    X_o             ,
                                    X_v             ,
                                    tau_o           ,
                                    tau_v           )
    output           = RMP2_J(Z               ,
                              X_o             ,
                              X_v             ,
                              tau_o           ,
                              tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP2_J_opt_mem(Z               ,
                                      X_o             ,
                                      X_v             ,
                                      tau_o           ,
                                      tau_v           ,
                                      buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP2_K and RMP2_K_naive
    benchmark        = RMP2_K_naive(Z               ,
                                    X_o             ,
                                    X_v             ,
                                    tau_o           ,
                                    tau_v           )
    output           = RMP2_K(Z               ,
                              X_o             ,
                              X_v             ,
                              tau_o           ,
                              tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP2_K_opt_mem(Z               ,
                                      X_o             ,
                                      X_v             ,
                                      tau_o           ,
                                      tau_v           ,
                                      buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
