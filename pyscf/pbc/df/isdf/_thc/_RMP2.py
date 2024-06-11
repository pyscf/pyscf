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
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
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
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_11.shape[1]))
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
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M5              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_M5.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_13.shape[1]))
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
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]))
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
    fn_permutation_012_102_wob = getattr(libpbc, "fn_permutation_012_102_wob", None)
    assert fn_permutation_012_102_wob is not None
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_102_wob(ctypes.c_void_p(_M11.ctypes.data),
                               ctypes.c_void_p(_M11_perm.ctypes.data),
                               ctypes.c_int(_M11.shape[0]),
                               ctypes.c_int(_M11.shape[1]),
                               ctypes.c_int(_M11.shape[2]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_10.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_10.shape[1]))
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
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[1]))
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
    _benchmark_time(t1, t2, "step 12")
    # step 12 RPT->RTP 
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

def RMP2_J_determine_bucket_size(NVIR        : int,
                                 NOCC        : int,
                                 N_LAPLACE   : int,
                                 NTHC_INT    : int):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    bucked_3_size    = 0               
    # assign the size of each tensor
    _INPUT_0_size    = (NTHC_INT * NTHC_INT)
    _INPUT_1_size    = (NOCC * NTHC_INT)
    _INPUT_2_size    = (NVIR * NTHC_INT)
    _INPUT_3_size    = (NOCC * NTHC_INT)
    _INPUT_4_size    = (NVIR * NTHC_INT)
    _INPUT_5_size    = (NTHC_INT * NTHC_INT)
    _INPUT_6_size    = (NOCC * NTHC_INT)
    _INPUT_7_size    = (NVIR * NTHC_INT)
    _INPUT_8_size    = (NOCC * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NOCC * N_LAPLACE)
    _INPUT_11_size   = (NOCC * N_LAPLACE)
    _INPUT_12_size   = (NVIR * N_LAPLACE)
    _INPUT_13_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_perm_size   = _M11_size       
    _M4_perm_size    = _M4_size        
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M7_size)
    bucked_0_size    = max(bucked_0_size, _M5_size)
    bucked_0_size    = max(bucked_0_size, _M9_size)
    bucked_0_size    = max(bucked_0_size, _M11_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M8_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M10_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M6_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M1_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
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
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # deal with buffer
    bucket_size      = RMP2_J_determine_bucket_size(NVIR = NVIR,
                                                    NOCC = NOCC,
                                                    N_LAPLACE = N_LAPLACE,
                                                    NTHC_INT = NTHC_INT)
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
    bufsize          = offset_now      
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step 0 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7_offset       = offset_0        
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_11.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M8_offset       = offset_1        
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M8_offset)
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
    _M8              = _M8_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bQ,bT->QTb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M5_offset       = offset_0        
    _M5              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M5_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_M5.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_13.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 bS,QTb->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6_offset       = offset_2        
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M6_offset)
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
    _M6              = _M6_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 SQT,SQT->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M9_offset       = offset_0        
    _M9              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]),
                                   ctypes.c_int(_M6.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 RS,SQT->RQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
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
    _M11             = _M11_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 RQT->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_102_wob = getattr(libpbc, "fn_permutation_012_102_wob", None)
    assert fn_permutation_012_102_wob is not None
    _M11_perm_offset = offset_0        
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_perm_offset)
    fn_permutation_012_102_wob(ctypes.c_void_p(_M11.ctypes.data),
                               ctypes.c_void_p(_M11_perm.ctypes.data),
                               ctypes.c_int(_M11.shape[0]),
                               ctypes.c_int(_M11.shape[1]),
                               ctypes.c_int(_M11.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,iT->PTi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_10.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_10.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iR,PTi->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M3_offset       = offset_2        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aP,aT->PTa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 aR,PTa->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M1_offset       = offset_3        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 RPT,RPT->RPT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M4_offset       = offset_1        
    _M4              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M4_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_M1.shape[0]),
                                   ctypes.c_int(_M1.shape[1]),
                                   ctypes.c_int(_M1.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RPT->RTP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M4_perm_offset  = offset_2        
    _M4_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M4_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M4.ctypes.data),
                               ctypes.c_void_p(_M4_perm.ctypes.data),
                               ctypes.c_int(_M4.shape[0]),
                               ctypes.c_int(_M4.shape[1]),
                               ctypes.c_int(_M4.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 PQ,RTP->QRT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_1        
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
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
    _M10             = _M10_reshaped.reshape(*shape_backup)
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
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
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
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # step 0 iP,aP->iaP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_2.shape[0]))
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
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[1]))
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
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                    ctypes.c_void_p(_M5.ctypes.data),
                                    ctypes.c_void_p(_M6.ctypes.data),
                                    ctypes.c_int(_INPUT_10.shape[0]),
                                    ctypes.c_int(_INPUT_10.shape[1]),
                                    ctypes.c_int(_M5.shape[0]),
                                    ctypes.c_int(_M5.shape[2]))
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_11.shape[1]))
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
    fn_contraction_0123_103_2013_wob = getattr(libpbc, "fn_contraction_0123_103_2013_wob", None)
    assert fn_contraction_0123_103_2013_wob is not None
    _M9              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_103_2013_wob(ctypes.c_void_p(_M6.ctypes.data),
                                     ctypes.c_void_p(_M8.ctypes.data),
                                     ctypes.c_void_p(_M9.ctypes.data),
                                     ctypes.c_int(_M6.shape[0]),
                                     ctypes.c_int(_M6.shape[1]),
                                     ctypes.c_int(_M6.shape[2]),
                                     ctypes.c_int(_M6.shape[3]))
    del _M6         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iQST->QTSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1320_wob = getattr(libpbc, "fn_permutation_0123_1320_wob", None)
    assert fn_permutation_0123_1320_wob is not None
    _M9_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_1320_wob(ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_void_p(_M9_perm.ctypes.data),
                                 ctypes.c_int(_M9.shape[0]),
                                 ctypes.c_int(_M9.shape[1]),
                                 ctypes.c_int(_M9.shape[2]),
                                 ctypes.c_int(_M9.shape[3]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iR,bR->ibR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
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
    fn_contraction_01_012_02_wob = getattr(libpbc, "fn_contraction_01_012_02_wob", None)
    assert fn_contraction_01_012_02_wob is not None
    _M11             = np.ndarray((NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_012_02_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_M10.shape[2]))
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

def RMP2_K_determine_bucket_size(NVIR        : int,
                                 NOCC        : int,
                                 N_LAPLACE   : int,
                                 NTHC_INT    : int):
    # init
    output = []     
    bucked_0_size    = 0               
    bucked_1_size    = 0               
    bucked_2_size    = 0               
    # assign the size of each tensor
    _INPUT_0_size    = (NTHC_INT * NTHC_INT)
    _INPUT_1_size    = (NOCC * NTHC_INT)
    _INPUT_2_size    = (NVIR * NTHC_INT)
    _INPUT_3_size    = (NOCC * NTHC_INT)
    _INPUT_4_size    = (NVIR * NTHC_INT)
    _INPUT_5_size    = (NTHC_INT * NTHC_INT)
    _INPUT_6_size    = (NOCC * NTHC_INT)
    _INPUT_7_size    = (NVIR * NTHC_INT)
    _INPUT_8_size    = (NOCC * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NOCC * N_LAPLACE)
    _INPUT_11_size   = (NOCC * N_LAPLACE)
    _INPUT_12_size   = (NVIR * N_LAPLACE)
    _INPUT_13_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NVIR * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NVIR))
    _M4_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M5_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M8_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M10_size        = (NVIR * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NVIR * N_LAPLACE)
    _M9_perm_size    = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M9_perm_size    = _M9_size        
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    bucked_0_size    = max(bucked_0_size, _M9_perm_size)
    bucked_0_size    = max(bucked_0_size, _M11_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _M7_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M10_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M8_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
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
    fn_copy      = getattr(libpbc, "fn_copy", None)
    assert fn_copy is not None
    fn_add       = getattr(libpbc, "fn_add", None)
    assert fn_add is not None
    fn_clean     = getattr(libpbc, "fn_clean", None)
    assert fn_clean is not None
    # deal with buffer
    bucket_size      = RMP2_K_determine_bucket_size(NVIR = NVIR,
                                                    NOCC = NOCC,
                                                    N_LAPLACE = N_LAPLACE,
                                                    NTHC_INT = NTHC_INT)
    _itemsize        = buffer.itemsize 
    offset_now       = 0               
    offset_0         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[0])
    offset_1         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[1])
    offset_2         = (offset_now * _itemsize)
    offset_now       = (offset_now + bucket_size[2])
    bufsize          = offset_now      
    bufsize_now      = buffer.size     
    if (bufsize > bufsize_now):
        buffer           = np.ndarray((bufsize), dtype=np.float64)
    # step 0 iP,aP->iaP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0_offset       = offset_0        
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_2.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 PQ,iaP->Qia 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = offset_1        
    _M1              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M1_offset)
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
    _M1              = _M1_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 aS,aT->STa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M4_offset       = offset_0        
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qia,STa->QiST 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M5_offset)
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
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iT,QiST->QSiT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M6_offset       = offset_0        
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_10.ctypes.data),
                                    ctypes.c_void_p(_M5.ctypes.data),
                                    ctypes.c_void_p(_M6.ctypes.data),
                                    ctypes.c_int(_INPUT_10.shape[0]),
                                    ctypes.c_int(_INPUT_10.shape[1]),
                                    ctypes.c_int(_M5.shape[0]),
                                    ctypes.c_int(_M5.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jT->QTj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7_offset       = offset_1        
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_11.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jS,QTj->SQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M8_offset       = offset_2        
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M8_offset)
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
    _M8              = _M8_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QSiT,SQT->iQST 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_103_2013_wob = getattr(libpbc, "fn_contraction_0123_103_2013_wob", None)
    assert fn_contraction_0123_103_2013_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    fn_contraction_0123_103_2013_wob(ctypes.c_void_p(_M6.ctypes.data),
                                     ctypes.c_void_p(_M8.ctypes.data),
                                     ctypes.c_void_p(_M9.ctypes.data),
                                     ctypes.c_int(_M6.shape[0]),
                                     ctypes.c_int(_M6.shape[1]),
                                     ctypes.c_int(_M6.shape[2]),
                                     ctypes.c_int(_M6.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iQST->QTSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1320_wob = getattr(libpbc, "fn_permutation_0123_1320_wob", None)
    assert fn_permutation_0123_1320_wob is not None
    _M9_perm_offset  = offset_0        
    _M9_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), buffer = buffer, offset = _M9_perm_offset)
    fn_permutation_0123_1320_wob(ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_void_p(_M9_perm.ctypes.data),
                                 ctypes.c_int(_M9.shape[0]),
                                 ctypes.c_int(_M9.shape[1]),
                                 ctypes.c_int(_M9.shape[2]),
                                 ctypes.c_int(_M9.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iR,bR->ibR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 RS,ibR->Sib 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M3_offset)
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
    _M3              = _M3_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 Sib,QTSi->bQT 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NVIR * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_1        
    _M10             = np.ndarray((NVIR, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
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
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bQ,bQT->bT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_012_02_wob = getattr(libpbc, "fn_contraction_01_012_02_wob", None)
    assert fn_contraction_01_012_02_wob is not None
    _M11_offset      = offset_0        
    _M11             = np.ndarray((NVIR, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_012_02_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_M10.shape[2]))
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
