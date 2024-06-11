import numpy
import numpy as np
import ctypes
import copy
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

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
    _M3              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (1, 3, 0, 2)    )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("RQV,RQV->RQV"  , _M8             , _M10            )
    del _M8         
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("RS,RQV->SQV"   , _INPUT_5        , _M11            )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("TWQV,SQV->TWS" , _M17_perm       , _M18            )
    del _M17_perm   
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M14            )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("USW,USW->USW"  , _M13            , _M15            )
    del _M13        
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TU,USW->TSW"   , _INPUT_10       , _M16            )
    del _M16        
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
    # step 0 aV,aW->VWa 
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
    _benchmark_time(t1, t2, "step 1")
    # step 1 aP,VWa->PVWa 
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
    _benchmark_time(t1, t2, "step 2")
    # step 2 aT,PVWa->TPVW 
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
    _benchmark_time(t1, t2, "step 3")
    # step 3 iV,iW->VWi 
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
    _benchmark_time(t1, t2, "step 4")
    # step 4 iP,VWi->PVWi 
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
    _benchmark_time(t1, t2, "step 5")
    # step 5 iT,PVWi->TPVW 
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
    _benchmark_time(t1, t2, "step 6")
    # step 6 TPVW,TPVW->TPVW 
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
    _benchmark_time(t1, t2, "step 7")
    # step 7 TPVW->TVWP 
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
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQ,TVWP->QTVW 
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 QTVW->TWQV 
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
    _benchmark_time(t1, t2, "step 10")
    # step 10 jQ,jV->QVj 
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
    _benchmark_time(t1, t2, "step 11")
    # step 11 jR,QVj->RQV 
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
    _benchmark_time(t1, t2, "step 12")
    # step 12 bQ,bV->QVb 
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
    _benchmark_time(t1, t2, "step 13")
    # step 13 bR,QVb->RQV 
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
    _benchmark_time(t1, t2, "step 14")
    # step 14 RQV,RQV->RQV 
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
    _benchmark_time(t1, t2, "step 15")
    # step 15 RS,RQV->SQV 
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 TWQV,SQV->TWS 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 TWS->TSW 
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
    _benchmark_time(t1, t2, "step 18")
    # step 18 cS,cW->SWc 
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 cU,SWc->USW 
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
    _benchmark_time(t1, t2, "step 20")
    # step 20 kS,kW->SWk 
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
    _benchmark_time(t1, t2, "step 21")
    # step 21 kU,SWk->USW 
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
    _benchmark_time(t1, t2, "step 22")
    # step 22 USW,USW->USW 
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
    _benchmark_time(t1, t2, "step 23")
    # step 23 TU,USW->TSW 
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

def RMP3_CC_determine_bucket_size(NVIR        : int,
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
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (N_LAPLACE * (N_LAPLACE * NOCC))
    _M1_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NOCC)))
    _M2_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M3_size         = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M4_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M5_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M7_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M8_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M12_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M13_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M14_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M15_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M16_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M17_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M19_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M17_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M20_perm_size   = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M6_perm_size    = _M6_size        
    _M17_perm_size   = _M17_size       
    _M20_perm_size   = _M20_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M3_size)
    bucked_0_size    = max(bucked_0_size, _M5_size)
    bucked_0_size    = max(bucked_0_size, _M6_perm_size)
    bucked_0_size    = max(bucked_0_size, _M17_perm_size)
    bucked_0_size    = max(bucked_0_size, _M20_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M17_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M7_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M20_size)
    bucked_1_size    = max(bucked_1_size, _M14_size)
    bucked_1_size    = max(bucked_1_size, _M12_size)
    bucked_1_size    = max(bucked_1_size, _M16_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M6_size)
    bucked_2_size    = max(bucked_2_size, _M10_size)
    bucked_2_size    = max(bucked_2_size, _M18_size)
    bucked_2_size    = max(bucked_2_size, _M15_size)
    bucked_2_size    = max(bucked_2_size, _M19_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M8_size)
    bucked_3_size    = max(bucked_3_size, _M13_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_CC_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_CC_determine_bucket_size(NVIR = NVIR,
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
    # step 0 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M3_offset       = offset_0        
    _M3              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M3_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M4_offset       = offset_1        
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M3.shape[0]),
                                   ctypes.c_int(_M3.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M5_offset       = offset_0        
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M5_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M1_offset       = offset_2        
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M1_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M0.ctypes.data),
                                   ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M0.shape[0]),
                                   ctypes.c_int(_M0.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M2_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 TPVW,TPVW->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    _M6_offset       = offset_2        
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                      ctypes.c_void_p(_M5.ctypes.data),
                                      ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_int(_M2.shape[0]),
                                      ctypes.c_int(_M2.shape[1]),
                                      ctypes.c_int(_M2.shape[2]),
                                      ctypes.c_int(_M2.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TPVW->TVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M6_perm_offset  = offset_0        
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M6_perm.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M17_offset      = offset_1        
    _M17             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M17_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 QTVW->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1302_wob = getattr(libpbc, "fn_permutation_0123_1302_wob", None)
    assert fn_permutation_0123_1302_wob is not None
    _M17_perm_offset = offset_0        
    _M17_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_perm_offset)
    fn_permutation_0123_1302_wob(ctypes.c_void_p(_M17.ctypes.data),
                                 ctypes.c_void_p(_M17_perm.ctypes.data),
                                 ctypes.c_int(_M17.shape[0]),
                                 ctypes.c_int(_M17.shape[1]),
                                 ctypes.c_int(_M17.shape[2]),
                                 ctypes.c_int(_M17.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_2        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M7_offset       = offset_1        
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M8_offset       = offset_3        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 RQV,RQV->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]),
                                   ctypes.c_int(_M8.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RS,RQV->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M18_offset      = offset_2        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 TWQV,SQV->TWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NTHC_INT * _itemsize)))
    _M20_offset      = offset_1        
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_offset)
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
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 TWS->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M20_perm_offset = offset_0        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M14_offset      = offset_1        
    _M14             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M14_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M14.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M15_offset      = offset_2        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12_offset      = offset_1        
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M12_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M13_offset      = offset_3        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 USW,USW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_012_012_wob = getattr(libpbc, "fn_contraction_012_012_012_wob", None)
    assert fn_contraction_012_012_012_wob is not None
    _M16_offset      = offset_1        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    fn_contraction_012_012_012_wob(ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_int(_M13.shape[0]),
                                   ctypes.c_int(_M13.shape[1]),
                                   ctypes.c_int(_M13.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 TU,USW->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M19_offset      = offset_2        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_CX_1_naive(Z           : np.ndarray,
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
    _M3              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M3             )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M1             )
    del _M1         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TPVW,TPVW->TPVW", _M2             , _M5             )
    del _M2         
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 3, 1)    )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("PQ,TVWP->QTVW" , _INPUT_0        , _M6_perm        )
    del _M6_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("TU,QVWT->UQVW" , _INPUT_10       , _M7_perm        )
    del _M7_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (1, 2, 0, 3)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jR,kR->jkR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("RS,jkR->Sjk"   , _INPUT_5        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("kU,kW->UWk"    , _INPUT_13       , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("Sjk,UWk->SjUW" , _M10            , _M11            )
    del _M10        
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("SjUW,USW->jSUW", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("jSUW,QVj->SUWQV", _M15            , _M16            )
    del _M15        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (3, 4, 0, 1, 2) )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("QVUW,QVSUW->QVS", _M8_perm        , _M17_perm       )
    del _M8_perm    
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("bS,QVb->SQV"   , _INPUT_8        , _M19            )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (1, 2, 0)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("QVS,QVS->"     , _M18            , _M20_perm       )
    del _M18        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_1(Z           : np.ndarray,
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
    # step 0 iV,iW->VWi 
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
    _benchmark_time(t1, t2, "step 1")
    # step 1 iP,VWi->PVWi 
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
    _benchmark_time(t1, t2, "step 2")
    # step 2 iT,PVWi->TPVW 
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
    _benchmark_time(t1, t2, "step 3")
    # step 3 aV,aW->VWa 
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
    _benchmark_time(t1, t2, "step 4")
    # step 4 aP,VWa->PVWa 
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
    _benchmark_time(t1, t2, "step 5")
    # step 5 aT,PVWa->TPVW 
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
    _benchmark_time(t1, t2, "step 6")
    # step 6 TPVW,TPVW->TPVW 
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
    _benchmark_time(t1, t2, "step 7")
    # step 7 TPVW->TVWP 
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
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQ,TVWP->QTVW 
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 QTVW->QVWT 
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
    _benchmark_time(t1, t2, "step 10")
    # step 10 TU,QVWT->UQVW 
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
    _benchmark_time(t1, t2, "step 11")
    # step 11 UQVW->QVUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1203_wob = getattr(libpbc, "fn_permutation_0123_1203_wob", None)
    assert fn_permutation_0123_1203_wob is not None
    _M8_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_1203_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M9              = np.ndarray((NOCC, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 kU,kW->UWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 Sjk,UWk->SjUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M10_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M10        
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 cS,cW->SWc 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 SjUW,USW->jSUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_203_1023_wob = getattr(libpbc, "fn_contraction_0123_203_1023_wob", None)
    assert fn_contraction_0123_203_1023_wob is not None
    _M15             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_203_1023_wob(ctypes.c_void_p(_M12.ctypes.data),
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
    # step 19 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M16             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 jSUW,QVj->SUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_M15_reshaped.T, _M16_reshaped.T, c=_M17_reshaped)
    _M17         = _M17_reshaped.reshape(*shape_backup)
    del _M15        
    del _M16        
    del _M15_reshaped
    del _M16_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SUWQV->QVSUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34012_wob = getattr(libpbc, "fn_permutation_01234_34012_wob", None)
    assert fn_permutation_01234_34012_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_34012_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QVUW,QVSUW->QVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01423_014_wob = getattr(libpbc, "fn_contraction_0123_01423_014_wob", None)
    assert fn_contraction_0123_01423_014_wob is not None
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_contraction_0123_01423_014_wob(ctypes.c_void_p(_M8_perm.ctypes.data),
                                      ctypes.c_void_p(_M17_perm.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_int(_M8_perm.shape[0]),
                                      ctypes.c_int(_M8_perm.shape[1]),
                                      ctypes.c_int(_M8_perm.shape[2]),
                                      ctypes.c_int(_M8_perm.shape[3]),
                                      ctypes.c_int(_M17_perm.shape[2]))
    del _M8_perm    
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 bS,QVb->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 SQV->QVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 QVS,QVS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M18.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M18.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M18        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_1_determine_bucket_size(NVIR        : int,
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
    _INPUT_7_size    = (NOCC * NTHC_INT)
    _INPUT_8_size    = (NVIR * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M1_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M2_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M3_size         = (N_LAPLACE * (N_LAPLACE * NOCC))
    _M4_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NOCC)))
    _M5_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M6_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M7_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M8_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M9_size         = (NOCC * (NOCC * NTHC_INT))
    _M10_size        = (NTHC_INT * (NOCC * NOCC))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M12_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M13_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M14_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M15_size        = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M16_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M17_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M18_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M7_perm_size    = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M8_perm_size    = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M17_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_perm_size    = _M6_size        
    _M7_perm_size    = _M7_size        
    _M8_perm_size    = _M8_size        
    _M20_perm_size   = _M20_size       
    _M17_perm_size   = _M17_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M3_size)
    bucked_0_size    = max(bucked_0_size, _M5_size)
    bucked_0_size    = max(bucked_0_size, _M6_perm_size)
    bucked_0_size    = max(bucked_0_size, _M7_perm_size)
    bucked_0_size    = max(bucked_0_size, _M8_perm_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    bucked_0_size    = max(bucked_0_size, _M20_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M7_size)
    bucked_1_size    = max(bucked_1_size, _M8_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M13_size)
    bucked_1_size    = max(bucked_1_size, _M15_size)
    bucked_1_size    = max(bucked_1_size, _M17_perm_size)
    bucked_1_size    = max(bucked_1_size, _M20_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M6_size)
    bucked_2_size    = max(bucked_2_size, _M10_size)
    bucked_2_size    = max(bucked_2_size, _M14_size)
    bucked_2_size    = max(bucked_2_size, _M16_size)
    bucked_2_size    = max(bucked_2_size, _M18_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M12_size)
    bucked_3_size    = max(bucked_3_size, _M17_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_CX_1_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_CX_1_determine_bucket_size(NVIR = NVIR,
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
    # step 0 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M3_offset       = offset_0        
    _M3              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M3_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M3.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M4_offset       = offset_1        
    _M4              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M3.ctypes.data),
                                   ctypes.c_void_p(_M4.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M3.shape[0]),
                                   ctypes.c_int(_M3.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M5_offset       = offset_0        
    _M5              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M5_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M1_offset       = offset_2        
    _M1              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M1_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M0.ctypes.data),
                                   ctypes.c_void_p(_M1.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M0.shape[0]),
                                   ctypes.c_int(_M0.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M2_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 TPVW,TPVW->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0123_0123_wob = getattr(libpbc, "fn_contraction_0123_0123_0123_wob", None)
    assert fn_contraction_0123_0123_0123_wob is not None
    _M6_offset       = offset_2        
    _M6              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M6_offset)
    fn_contraction_0123_0123_0123_wob(ctypes.c_void_p(_M2.ctypes.data),
                                      ctypes.c_void_p(_M5.ctypes.data),
                                      ctypes.c_void_p(_M6.ctypes.data),
                                      ctypes.c_int(_M2.shape[0]),
                                      ctypes.c_int(_M2.shape[1]),
                                      ctypes.c_int(_M2.shape[2]),
                                      ctypes.c_int(_M2.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TPVW->TVWP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M6_perm_offset  = offset_0        
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_void_p(_M6_perm.ctypes.data),
                                 ctypes.c_int(_M6.shape[0]),
                                 ctypes.c_int(_M6.shape[1]),
                                 ctypes.c_int(_M6.shape[2]),
                                 ctypes.c_int(_M6.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQ,TVWP->QTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_1        
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M7_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 QTVW->QVWT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm_offset  = offset_0        
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M7_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TU,QVWT->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M8_offset       = offset_1        
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M8_offset)
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 UQVW->QVUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1203_wob = getattr(libpbc, "fn_permutation_0123_1203_wob", None)
    assert fn_permutation_0123_1203_wob is not None
    _M8_perm_offset  = offset_0        
    _M8_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M8_perm_offset)
    fn_permutation_0123_1203_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NOCC, NOCC, NTHC_INT), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NOCC * _itemsize)))
    _M10_offset      = offset_2        
    _M10             = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_5.shape[0]
    _INPUT_5_reshaped = _INPUT_5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_5_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 kU,kW->UWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 Sjk,UWk->SjUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M12_offset      = offset_3        
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M10_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13_offset      = offset_1        
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M14_offset      = offset_2        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M13_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 SjUW,USW->jSUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_203_1023_wob = getattr(libpbc, "fn_contraction_0123_203_1023_wob", None)
    assert fn_contraction_0123_203_1023_wob is not None
    _M15_offset      = offset_1        
    _M15             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    fn_contraction_0123_203_1023_wob(ctypes.c_void_p(_M12.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M12.shape[0]),
                                     ctypes.c_int(_M12.shape[1]),
                                     ctypes.c_int(_M12.shape[2]),
                                     ctypes.c_int(_M12.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M16_offset      = offset_2        
    _M16             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M16_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 jSUW,QVj->SUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M17_offset      = offset_3        
    _M17             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_M15_reshaped.T, _M16_reshaped.T, c=_M17_reshaped)
    _M17             = _M17_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SUWQV->QVSUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34012_wob = getattr(libpbc, "fn_permutation_01234_34012_wob", None)
    assert fn_permutation_01234_34012_wob is not None
    _M17_perm_offset = offset_1        
    _M17_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_perm_offset)
    fn_permutation_01234_34012_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QVUW,QVSUW->QVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01423_014_wob = getattr(libpbc, "fn_contraction_0123_01423_014_wob", None)
    assert fn_contraction_0123_01423_014_wob is not None
    _M18_offset      = offset_2        
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M18_offset)
    fn_contraction_0123_01423_014_wob(ctypes.c_void_p(_M8_perm.ctypes.data),
                                      ctypes.c_void_p(_M17_perm.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_int(_M8_perm.shape[0]),
                                      ctypes.c_int(_M8_perm.shape[1]),
                                      ctypes.c_int(_M8_perm.shape[2]),
                                      ctypes.c_int(_M8_perm.shape[3]),
                                      ctypes.c_int(_M17_perm.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19_offset      = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 bS,QVb->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M20_offset      = offset_1        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 SQV->QVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M20_perm_offset = offset_0        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 QVS,QVS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M18.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M18.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2_naive(Z           : np.ndarray,
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
    _M7              = np.einsum("iP,bP->ibP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("PQ,ibP->Qib"   , _INPUT_0        , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("bR,bV->RVb"    , _INPUT_7        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("Qib,RVb->QiRV" , _M8             , _M12            )
    del _M8         
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("iV,QiRV->QRiV" , _INPUT_15       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M15            )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("QRiV,RQV->iQRV", _M14            , _M16            )
    del _M14        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("iT,iW->TWi"    , _INPUT_11       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("iQRV,TWi->QRVTW", _M17            , _M18            )
    del _M17        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19_perm        = np.transpose(_M19            , (3, 4, 0, 2, 1) )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("kU,SWk->USW"   , _INPUT_13       , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("USW,USW->USW"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("RS,UWS->RUW"   , _INPUT_5        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("TU,RWU->TRW"   , _INPUT_10       , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (0, 2, 1)       )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("TWR,TWQVR->TWQV", _M6_perm        , _M19_perm       )
    del _M6_perm    
    del _M19_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 3, 1)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("aQ,VWa->QVWa"  , _INPUT_4        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aT,QVWa->TQVW" , _INPUT_12       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TQVW,TQVW->"   , _M11            , _M20_perm       )
    del _M11        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2(Z           : np.ndarray,
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
    # step 0 iP,bP->ibP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M7              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_2.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 PQ,ibP->Qib 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bR,bV->RVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qib,RVb->QiRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M13.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    lib.ddot(_M8_reshaped, _M12_reshaped.T, c=_M13_reshaped)
    _M13         = _M13_reshaped.reshape(*shape_backup)
    del _M8         
    del _M12        
    del _M12_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiRV->QRiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_void_p(_M14.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M13.shape[0]),
                                    ctypes.c_int(_M13.shape[2]))
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M15_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15        
    del _M15_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QRiV,RQV->iQRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_103_2013_wob = getattr(libpbc, "fn_contraction_0123_103_2013_wob", None)
    assert fn_contraction_0123_103_2013_wob is not None
    _M17             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_103_2013_wob(ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M16.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_M14.shape[0]),
                                     ctypes.c_int(_M14.shape[1]),
                                     ctypes.c_int(_M14.shape[2]),
                                     ctypes.c_int(_M14.shape[3]))
    del _M14        
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iT,iW->TWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iQRV,TWi->QRVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _size_dim_1      = _size_dim_1 * _M18.shape[1]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M19.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    lib.ddot(_M17_reshaped.T, _M18_reshaped.T, c=_M19_reshaped)
    _M19         = _M19_reshaped.reshape(*shape_backup)
    del _M17        
    del _M18        
    del _M17_reshaped
    del _M18_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 QRVTW->TWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34021_wob = getattr(libpbc, "fn_permutation_01234_34021_wob", None)
    assert fn_permutation_01234_34021_wob is not None
    _M19_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_34021_wob(ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_void_p(_M19_perm.ctypes.data),
                                   ctypes.c_int(_M19.shape[0]),
                                   ctypes.c_int(_M19.shape[1]),
                                   ctypes.c_int(_M19.shape[2]),
                                   ctypes.c_int(_M19.shape[3]),
                                   ctypes.c_int(_M19.shape[4]))
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 kS,kW->SWk 
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
    _benchmark_time(t1, t2, "step 12")
    # step 12 kU,SWk->USW 
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
    _benchmark_time(t1, t2, "step 13")
    # step 13 cS,cW->SWc 
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
    _benchmark_time(t1, t2, "step 14")
    # step 14 cU,SWc->USW 
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
    _benchmark_time(t1, t2, "step 15")
    # step 15 USW,USW->USW 
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 USW->UWS 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 RS,UWS->RUW 
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
    _benchmark_time(t1, t2, "step 18")
    # step 18 RUW->RWU 
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 TU,RWU->TRW 
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
    _benchmark_time(t1, t2, "step 20")
    # step 20 TRW->TWR 
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
    _benchmark_time(t1, t2, "step 21")
    # step 21 TWR,TWQVR->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M6_perm.ctypes.data),
                                      ctypes.c_void_p(_M19_perm.ctypes.data),
                                      ctypes.c_void_p(_M20.ctypes.data),
                                      ctypes.c_int(_M6_perm.shape[0]),
                                      ctypes.c_int(_M6_perm.shape[1]),
                                      ctypes.c_int(_M6_perm.shape[2]),
                                      ctypes.c_int(_M19_perm.shape[2]),
                                      ctypes.c_int(_M19_perm.shape[3]))
    del _M6_perm    
    del _M19_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 TWQV->TQVW 
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
    _benchmark_time(t1, t2, "step 23")
    # step 23 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 aQ,VWa->QVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 aT,QVWa->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _size_dim_1      = _size_dim_1 * _M10.shape[2]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TQVW,TQVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M11.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M11        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_2_determine_bucket_size(NVIR        : int,
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
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NOCC * (NVIR * NTHC_INT))
    _M8_size         = (NTHC_INT * (NOCC * NVIR))
    _M9_size         = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M10_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M11_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M12_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M13_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M14_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M15_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M16_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M17_size        = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M18_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M19_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M20_size        = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M19_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_perm_size   = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M19_perm_size   = _M19_size       
    _M20_perm_size   = _M20_size       
    _M4_perm_size    = _M4_size        
    _M5_perm_size    = _M5_size        
    _M6_perm_size    = _M6_size        
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M7_size)
    bucked_0_size    = max(bucked_0_size, _M12_size)
    bucked_0_size    = max(bucked_0_size, _M14_size)
    bucked_0_size    = max(bucked_0_size, _M18_size)
    bucked_0_size    = max(bucked_0_size, _M19_perm_size)
    bucked_0_size    = max(bucked_0_size, _M20_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M8_size)
    bucked_1_size    = max(bucked_1_size, _M15_size)
    bucked_1_size    = max(bucked_1_size, _M17_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M20_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M13_size)
    bucked_2_size    = max(bucked_2_size, _M16_size)
    bucked_2_size    = max(bucked_2_size, _M19_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M6_perm_size)
    bucked_2_size    = max(bucked_2_size, _M10_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M1_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_CX_2_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_CX_2_determine_bucket_size(NVIR = NVIR,
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
    # step 0 iP,bP->ibP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M7_offset       = offset_0        
    _M7              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_2.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 PQ,ibP->Qib 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M8_offset       = offset_1        
    _M8              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M8_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_0.shape[0]
    _INPUT_0_reshaped = _INPUT_0.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_0_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8              = _M8_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 bR,bV->RVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12_offset      = offset_0        
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M12_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qib,RVb->QiRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M13_offset      = offset_2        
    _M13             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M13.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    lib.ddot(_M8_reshaped, _M12_reshaped.T, c=_M13_reshaped)
    _M13             = _M13_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiRV->QRiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M14_offset      = offset_0        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_void_p(_M14.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M13.shape[0]),
                                    ctypes.c_int(_M13.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15_offset      = offset_1        
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M16_offset      = offset_2        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M15_reshaped.T, c=_M16_reshaped)
    _M16             = _M16_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QRiV,RQV->iQRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_103_2013_wob = getattr(libpbc, "fn_contraction_0123_103_2013_wob", None)
    assert fn_contraction_0123_103_2013_wob is not None
    _M17_offset      = offset_1        
    _M17             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    fn_contraction_0123_103_2013_wob(ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_void_p(_M16.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_M14.shape[0]),
                                     ctypes.c_int(_M14.shape[1]),
                                     ctypes.c_int(_M14.shape[2]),
                                     ctypes.c_int(_M14.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iT,iW->TWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18_offset      = offset_0        
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M18_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iQRV,TWi->QRVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M19_offset      = offset_2        
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M19_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _size_dim_1      = _size_dim_1 * _M18.shape[1]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M19.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    lib.ddot(_M17_reshaped.T, _M18_reshaped.T, c=_M19_reshaped)
    _M19             = _M19_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 QRVTW->TWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34021_wob = getattr(libpbc, "fn_permutation_01234_34021_wob", None)
    assert fn_permutation_01234_34021_wob is not None
    _M19_perm_offset = offset_0        
    _M19_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M19_perm_offset)
    fn_permutation_01234_34021_wob(ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_void_p(_M19_perm.ctypes.data),
                                   ctypes.c_int(_M19.shape[0]),
                                   ctypes.c_int(_M19.shape[1]),
                                   ctypes.c_int(_M19.shape[2]),
                                   ctypes.c_int(_M19.shape[3]),
                                   ctypes.c_int(_M19.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 kU,SWk->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M3_offset       = offset_2        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M1_offset       = offset_3        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 USW,USW->USW 
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 USW->UWS 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 RS,UWS->RUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5_offset       = offset_1        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 RUW->RWU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 TU,RWU->TRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6_offset       = offset_1        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 TRW->TWR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M6_perm_offset  = offset_2        
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 TWR,TWQVR->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M20_offset      = offset_1        
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M6_perm.ctypes.data),
                                      ctypes.c_void_p(_M19_perm.ctypes.data),
                                      ctypes.c_void_p(_M20.ctypes.data),
                                      ctypes.c_int(_M6_perm.shape[0]),
                                      ctypes.c_int(_M6_perm.shape[1]),
                                      ctypes.c_int(_M6_perm.shape[2]),
                                      ctypes.c_int(_M19_perm.shape[2]),
                                      ctypes.c_int(_M19_perm.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 TWQV->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M20_perm_offset = offset_0        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 aQ,VWa->QVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M10_offset      = offset_2        
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M10_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 aT,QVWa->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _size_dim_1      = _size_dim_1 * _M10.shape[2]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11             = _M11_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TQVW,TQVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M11.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M11.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_3_naive(Z           : np.ndarray,
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
    _M7              = np.einsum("iU,cU->icU"    , _INPUT_13       , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("TU,icU->Tic"   , _INPUT_10       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("Tic,SWc->TiSW" , _M8             , _M11            )
    del _M8         
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iW,TiSW->TSiW" , _INPUT_19       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("kT,SWk->TSW"   , _INPUT_11       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("TSW,TSiW->iTSW", _M10            , _M13            )
    del _M10        
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("iP,iV->PVi"    , _INPUT_1        , _INPUT_15       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("iTSW,PVi->TSWPV", _M14            , _M15            )
    del _M14        
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (3, 4, 0, 2, 1) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("bR,QVb->RQV"   , _INPUT_7        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("jR,QVj->RQV"   , _INPUT_6        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("RQV,RQV->RQV"  , _M1             , _M3             )
    del _M1         
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4_perm         = np.transpose(_M4             , (0, 2, 1)       )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("PQ,RVQ->PRV"   , _INPUT_0        , _M4_perm        )
    del _M4_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5_perm         = np.transpose(_M5             , (0, 2, 1)       )
    del _M5         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("RS,PVR->SPV"   , _INPUT_5        , _M5_perm        )
    del _M5_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6_perm         = np.transpose(_M6             , (1, 2, 0)       )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("PVS,PVTWS->PVTW", _M6_perm        , _M16_perm       )
    del _M6_perm    
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("aP,VWa->PVWa"  , _INPUT_2        , _M18            )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("aT,PVWa->TPVW" , _INPUT_12       , _M19            )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (1, 2, 0, 3)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("PVTW,PVTW->"   , _M17            , _M20_perm       )
    del _M17        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_3(Z           : np.ndarray,
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
    # step 0 iU,cU->icU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M7              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 TU,icU->Tic 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Tic,SWc->TiSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M8_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M8         
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iW,TiSW->TSiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 kT,SWk->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TSW,TSiW->iTSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0132_3012_wob = getattr(libpbc, "fn_contraction_012_0132_3012_wob", None)
    assert fn_contraction_012_0132_3012_wob is not None
    _M14             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0132_3012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                     ctypes.c_void_p(_M13.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_int(_M10.shape[0]),
                                     ctypes.c_int(_M10.shape[1]),
                                     ctypes.c_int(_M10.shape[2]),
                                     ctypes.c_int(_M13.shape[2]))
    del _M10        
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iP,iV->PVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iTSW,PVi->TSWPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_M14_reshaped.T, _M15_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M14        
    del _M15        
    del _M14_reshaped
    del _M15_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TSWPV->PVTWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34021_wob = getattr(libpbc, "fn_permutation_01234_34021_wob", None)
    assert fn_permutation_01234_34021_wob is not None
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_34021_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RQV,RQV->RQV 
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 RQV->RVQ 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 PQ,RVQ->PRV 
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
    _benchmark_time(t1, t2, "step 18")
    # step 18 PRV->PVR 
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 RS,PVR->SPV 
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
    _benchmark_time(t1, t2, "step 20")
    # step 20 SPV->PVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 PVS,PVTWS->PVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M6_perm.ctypes.data),
                                      ctypes.c_void_p(_M16_perm.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_M6_perm.shape[0]),
                                      ctypes.c_int(_M6_perm.shape[1]),
                                      ctypes.c_int(_M6_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[3]))
    del _M6_perm    
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TPVW->PVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1203_wob = getattr(libpbc, "fn_permutation_0123_1203_wob", None)
    assert fn_permutation_0123_1203_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_1203_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 PVTW,PVTW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M17.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M17.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M17        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_CX_3_determine_bucket_size(NVIR        : int,
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
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M1_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M2_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M3_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M4_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M7_size         = (NOCC * (NVIR * NTHC_INT))
    _M8_size         = (NTHC_INT * (NOCC * NVIR))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M12_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M13_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M14_size        = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M15_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M16_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M17_size        = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M18_size        = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M19_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M20_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M16_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M4_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M5_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M6_perm_size    = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M16_perm_size   = _M16_size       
    _M4_perm_size    = _M4_size        
    _M5_perm_size    = _M5_size        
    _M6_perm_size    = _M6_size        
    _M20_perm_size   = _M20_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M7_size)
    bucked_0_size    = max(bucked_0_size, _M11_size)
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _M15_size)
    bucked_0_size    = max(bucked_0_size, _M16_perm_size)
    bucked_0_size    = max(bucked_0_size, _M18_size)
    bucked_0_size    = max(bucked_0_size, _M20_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M8_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M14_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M17_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M12_size)
    bucked_2_size    = max(bucked_2_size, _M10_size)
    bucked_2_size    = max(bucked_2_size, _M16_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M4_perm_size)
    bucked_2_size    = max(bucked_2_size, _M5_perm_size)
    bucked_2_size    = max(bucked_2_size, _M6_perm_size)
    bucked_2_size    = max(bucked_2_size, _M19_size)
    bucked_2_size    = max(bucked_2_size, _M20_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M1_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_CX_3_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_CX_3_determine_bucket_size(NVIR = NVIR,
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
    # step 0 iU,cU->icU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M7_offset       = offset_0        
    _M7              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 TU,icU->Tic 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M8_offset       = offset_1        
    _M8              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M8_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M7_reshaped.T, c=_M8_reshaped)
    _M8              = _M8_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11_offset      = offset_0        
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Tic,SWc->TiSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M12_offset      = offset_2        
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M8_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iW,TiSW->TSiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13_offset      = offset_0        
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 kT,SWk->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M10_offset      = offset_2        
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TSW,TSiW->iTSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0132_3012_wob = getattr(libpbc, "fn_contraction_012_0132_3012_wob", None)
    assert fn_contraction_012_0132_3012_wob is not None
    _M14_offset      = offset_1        
    _M14             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    fn_contraction_012_0132_3012_wob(ctypes.c_void_p(_M10.ctypes.data),
                                     ctypes.c_void_p(_M13.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_int(_M10.shape[0]),
                                     ctypes.c_int(_M10.shape[1]),
                                     ctypes.c_int(_M10.shape[2]),
                                     ctypes.c_int(_M13.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iP,iV->PVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15_offset      = offset_0        
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 iTSW,PVi->TSWPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M16_offset      = offset_2        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_M14_reshaped.T, _M15_reshaped.T, c=_M16_reshaped)
    _M16             = _M16_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 TSWPV->PVTWS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34021_wob = getattr(libpbc, "fn_permutation_01234_34021_wob", None)
    assert fn_permutation_01234_34021_wob is not None
    _M16_perm_offset = offset_0        
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M16_perm_offset)
    fn_permutation_01234_34021_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 bR,QVb->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M3_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M2.shape[0]
    _size_dim_1      = _size_dim_1 * _M2.shape[1]
    _M2_reshaped = _M2.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M3.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jR,QVj->RQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M1_offset       = offset_3        
    _M1              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M1_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M0.shape[0]
    _size_dim_1      = _size_dim_1 * _M0.shape[1]
    _M0_reshaped = _M0.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M1.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RQV,RQV->RQV 
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 RQV->RVQ 
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 PQ,RVQ->PRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M5_offset       = offset_1        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 PRV->PVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M5_perm_offset  = offset_2        
    _M5_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M5_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M5.ctypes.data),
                               ctypes.c_void_p(_M5_perm.ctypes.data),
                               ctypes.c_int(_M5.shape[0]),
                               ctypes.c_int(_M5.shape[1]),
                               ctypes.c_int(_M5.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 RS,PVR->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M6_offset       = offset_1        
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
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 SPV->PVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M6_perm_offset  = offset_2        
    _M6_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M6_perm_offset)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M6.ctypes.data),
                               ctypes.c_void_p(_M6_perm.ctypes.data),
                               ctypes.c_int(_M6.shape[0]),
                               ctypes.c_int(_M6.shape[1]),
                               ctypes.c_int(_M6.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 PVS,PVTWS->PVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_01342_0134_wob = getattr(libpbc, "fn_contraction_012_01342_0134_wob", None)
    assert fn_contraction_012_01342_0134_wob is not None
    _M17_offset      = offset_1        
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    fn_contraction_012_01342_0134_wob(ctypes.c_void_p(_M6_perm.ctypes.data),
                                      ctypes.c_void_p(_M16_perm.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_int(_M6_perm.shape[0]),
                                      ctypes.c_int(_M6_perm.shape[1]),
                                      ctypes.c_int(_M6_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[2]),
                                      ctypes.c_int(_M16_perm.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18_offset      = offset_0        
    _M18             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M18_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 aP,VWa->PVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M19_offset      = offset_2        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 aT,PVWa->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M20_offset      = offset_0        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TPVW->PVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1203_wob = getattr(libpbc, "fn_permutation_0123_1203_wob", None)
    assert fn_permutation_0123_1203_wob is not None
    _M20_perm_offset = offset_2        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_0123_1203_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 PVTW,PVTW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M17.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M17.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_1_naive(Z           : np.ndarray,
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
    _M4              = np.einsum("kU,cU->kcU"    , _INPUT_13       , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,kcU->Tkc"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("Tkc,SWc->TkSW" , _M5             , _M6             )
    del _M5         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("kW,TkSW->TSkW" , _INPUT_20       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (1, 0, 3, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jR,kR->jkR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,jkR->Sjk"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("Sjk,STWk->SjTW", _M3             , _M8_perm        )
    del _M3         
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("aT,VWa->TVWa"  , _INPUT_12       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("jQ,aQ->jaQ"    , _INPUT_3        , _INPUT_4        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,jaQ->Pja"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("Pja,TVWa->PjTVW", _M1             , _M13            )
    del _M1         
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("jV,PjTVW->PTWjV", _INPUT_16       , _M14            )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("iT,PVWi->TPVW" , _INPUT_11       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("TPVW,PTWjV->jTPVW", _M11            , _M15            )
    del _M11        
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (2, 3, 0, 1, 4) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("PVjTW,SjTW->PVS", _M16_perm       , _M17            )
    del _M16_perm   
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("bP,bV->PVb"    , _INPUT_2        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("bS,PVb->SPV"   , _INPUT_8        , _M19            )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (1, 2, 0)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("PVS,PVS->"     , _M18            , _M20_perm       )
    del _M18        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_XX_1(Z           : np.ndarray,
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
    # step 0 kU,cU->kcU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 TU,kcU->Tkc 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Tkc,SWc->TkSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M5         
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 kW,TkSW->TSkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                    ctypes.c_void_p(_M7.ctypes.data),
                                    ctypes.c_void_p(_M8.ctypes.data),
                                    ctypes.c_int(_INPUT_20.shape[0]),
                                    ctypes.c_int(_INPUT_20.shape[1]),
                                    ctypes.c_int(_M7.shape[0]),
                                    ctypes.c_int(_M7.shape[2]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 TSkW->STWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
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
    _benchmark_time(t1, t2, "step 8")
    # step 8 Sjk,STWk->SjTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0342_0134_wob = getattr(libpbc, "fn_contraction_012_0342_0134_wob", None)
    assert fn_contraction_012_0342_0134_wob is not None
    _M17             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0342_0134_wob(ctypes.c_void_p(_M3.ctypes.data),
                                     ctypes.c_void_p(_M8_perm.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_M3.shape[0]),
                                     ctypes.c_int(_M3.shape[1]),
                                     ctypes.c_int(_M3.shape[2]),
                                     ctypes.c_int(_M8_perm.shape[1]),
                                     ctypes.c_int(_M8_perm.shape[2]))
    del _M3         
    del _M8_perm    
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aV,aW->VWa 
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
    _benchmark_time(t1, t2, "step 10")
    # step 10 aT,VWa->TVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_12.shape[0]),
                                   ctypes.c_int(_INPUT_12.shape[1]),
                                   ctypes.c_int(_M12.shape[0]),
                                   ctypes.c_int(_M12.shape[1]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jQ,aQ->jaQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_4.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 PQ,jaQ->Pja 
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
    lib.ddot(_INPUT_0_reshaped, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 Pja,TVWa->PjTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M1         
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jV,PjTVW->PTWjV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20314_23401_wob = getattr(libpbc, "fn_contraction_01_20314_23401_wob", None)
    assert fn_contraction_01_20314_23401_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_20314_23401_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_INPUT_16.shape[0]),
                                      ctypes.c_int(_INPUT_16.shape[1]),
                                      ctypes.c_int(_M14.shape[0]),
                                      ctypes.c_int(_M14.shape[2]),
                                      ctypes.c_int(_M14.shape[4]))
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _size_dim_1      = _size_dim_1 * _M10.shape[2]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11         = _M11_reshaped.reshape(*shape_backup)
    del _M10        
    del _M10_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 TPVW,PTWjV->jTPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_10342_40123_wob = getattr(libpbc, "fn_contraction_0123_10342_40123_wob", None)
    assert fn_contraction_0123_10342_40123_wob is not None
    _M16             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_10342_40123_wob(ctypes.c_void_p(_M11.ctypes.data),
                                        ctypes.c_void_p(_M15.ctypes.data),
                                        ctypes.c_void_p(_M16.ctypes.data),
                                        ctypes.c_int(_M11.shape[0]),
                                        ctypes.c_int(_M11.shape[1]),
                                        ctypes.c_int(_M11.shape[2]),
                                        ctypes.c_int(_M11.shape[3]),
                                        ctypes.c_int(_M15.shape[3]))
    del _M11        
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 jTPVW->PVjTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_23014_wob = getattr(libpbc, "fn_permutation_01234_23014_wob", None)
    assert fn_permutation_01234_23014_wob is not None
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_23014_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 PVjTW,SjTW->PVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[1]
    _M16_perm_reshaped = _M16_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _size_dim_1      = _size_dim_1 * _M18.shape[1]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_M16_perm_reshaped, _M17_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M16_perm   
    del _M17        
    del _M17_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 bP,bV->PVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 bS,PVb->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 SPV->PVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 PVS,PVS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M18.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M18.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M18        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_XX_1_determine_bucket_size(NVIR        : int,
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
    _INPUT_7_size    = (NOCC * NTHC_INT)
    _INPUT_8_size    = (NVIR * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NOCC * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NOCC))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M7_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M8_size         = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M9_size         = (N_LAPLACE * (N_LAPLACE * NOCC))
    _M10_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NOCC)))
    _M11_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M12_size        = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M13_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M14_size        = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M15_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M16_size        = (NOCC * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M17_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M18_size        = (NTHC_INT * (N_LAPLACE * NTHC_INT))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M8_perm_size    = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M16_perm_size   = (NOCC * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M20_perm_size   = _M20_size       
    _M8_perm_size    = _M8_size        
    _M16_perm_size   = _M16_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M17_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    bucked_0_size    = max(bucked_0_size, _M20_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M5_size)
    bucked_1_size    = max(bucked_1_size, _M8_perm_size)
    bucked_1_size    = max(bucked_1_size, _M12_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M14_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M16_perm_size)
    bucked_1_size    = max(bucked_1_size, _M20_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M7_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M13_size)
    bucked_2_size    = max(bucked_2_size, _M15_size)
    bucked_2_size    = max(bucked_2_size, _M18_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M1_size)
    bucked_3_size    = max(bucked_3_size, _M10_size)
    bucked_3_size    = max(bucked_3_size, _M16_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_1_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_XX_1_determine_bucket_size(NVIR = NVIR,
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
    # step 0 kU,cU->kcU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_0        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 TU,kcU->Tkc 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_1        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_0        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Tkc,SWc->TkSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_2        
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 kW,TkSW->TSkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                    ctypes.c_void_p(_M7.ctypes.data),
                                    ctypes.c_void_p(_M8.ctypes.data),
                                    ctypes.c_int(_INPUT_20.shape[0]),
                                    ctypes.c_int(_INPUT_20.shape[1]),
                                    ctypes.c_int(_M7.shape[0]),
                                    ctypes.c_int(_M7.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 TSkW->STWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M8_perm_offset  = offset_1        
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M8_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_0        
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NOCC * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M3_offset)
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
    _benchmark_time(t1, t2, "step 8")
    # step 8 Sjk,STWk->SjTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0342_0134_wob = getattr(libpbc, "fn_contraction_012_0342_0134_wob", None)
    assert fn_contraction_012_0342_0134_wob is not None
    _M17_offset      = offset_0        
    _M17             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    fn_contraction_012_0342_0134_wob(ctypes.c_void_p(_M3.ctypes.data),
                                     ctypes.c_void_p(_M8_perm.ctypes.data),
                                     ctypes.c_void_p(_M17.ctypes.data),
                                     ctypes.c_int(_M3.shape[0]),
                                     ctypes.c_int(_M3.shape[1]),
                                     ctypes.c_int(_M3.shape[2]),
                                     ctypes.c_int(_M8_perm.shape[1]),
                                     ctypes.c_int(_M8_perm.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12_offset      = offset_1        
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M12_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 aT,VWa->TVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13_offset      = offset_2        
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_12.shape[0]),
                                   ctypes.c_int(_INPUT_12.shape[1]),
                                   ctypes.c_int(_M12.shape[0]),
                                   ctypes.c_int(_M12.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jQ,aQ->jaQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_4.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 PQ,jaQ->Pja 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = offset_3        
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
    lib.ddot(_INPUT_0_reshaped, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 Pja,TVWa->PjTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M14_offset      = offset_1        
    _M14             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jV,PjTVW->PTWjV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20314_23401_wob = getattr(libpbc, "fn_contraction_01_20314_23401_wob", None)
    assert fn_contraction_01_20314_23401_wob is not None
    _M15_offset      = offset_2        
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_20314_23401_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_INPUT_16.shape[0]),
                                      ctypes.c_int(_INPUT_16.shape[1]),
                                      ctypes.c_int(_M14.shape[0]),
                                      ctypes.c_int(_M14.shape[2]),
                                      ctypes.c_int(_M14.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M10_offset      = offset_3        
    _M10             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M10_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_void_p(_M10.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M9.shape[0]),
                                   ctypes.c_int(_M9.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 iT,PVWi->TPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _size_dim_1      = _size_dim_1 * _M10.shape[2]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M11.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M10_reshaped.T, c=_M11_reshaped)
    _M11             = _M11_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 TPVW,PTWjV->jTPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_10342_40123_wob = getattr(libpbc, "fn_contraction_0123_10342_40123_wob", None)
    assert fn_contraction_0123_10342_40123_wob is not None
    _M16_offset      = offset_3        
    _M16             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    fn_contraction_0123_10342_40123_wob(ctypes.c_void_p(_M11.ctypes.data),
                                        ctypes.c_void_p(_M15.ctypes.data),
                                        ctypes.c_void_p(_M16.ctypes.data),
                                        ctypes.c_int(_M11.shape[0]),
                                        ctypes.c_int(_M11.shape[1]),
                                        ctypes.c_int(_M11.shape[2]),
                                        ctypes.c_int(_M11.shape[3]),
                                        ctypes.c_int(_M15.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 jTPVW->PVjTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_23014_wob = getattr(libpbc, "fn_permutation_01234_23014_wob", None)
    assert fn_permutation_01234_23014_wob is not None
    _M16_perm_offset = offset_1        
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_perm_offset)
    fn_permutation_01234_23014_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 PVjTW,SjTW->PVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NTHC_INT * _itemsize)))
    _M18_offset      = offset_2        
    _M18             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[1]
    _M16_perm_reshaped = _M16_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _size_dim_1      = _size_dim_1 * _M18.shape[1]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_M16_perm_reshaped, _M17_reshaped.T, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 bP,bV->PVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19_offset      = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_2.shape[0]),
                                 ctypes.c_int(_INPUT_2.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 bS,PVb->SPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M20_offset      = offset_1        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 SPV->PVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_120_wob = getattr(libpbc, "fn_permutation_012_120_wob", None)
    assert fn_permutation_012_120_wob is not None
    _M20_perm_offset = offset_0        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_120_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 PVS,PVS-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M18.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M18.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_XX_2_naive(Z           : np.ndarray,
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
    _M2              = np.einsum("jR,bR->jbR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,jbR->Sjb"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3_perm         = np.transpose(_M3             , (0, 2, 1)       )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("Sbj,QVj->SbQV" , _M3_perm        , _M12            )
    del _M3_perm    
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("bV,SbQV->SQbV" , _INPUT_18       , _M13            )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14_perm        = np.transpose(_M14            , (1, 0, 3, 2)    )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,bP->ibP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,ibP->Qib"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("Qib,QSVb->QiSV", _M1             , _M14_perm       )
    del _M1         
    del _M14_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("iV,QiSV->QSiV" , _INPUT_15       , _M15            )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (0, 3, 1, 2)    )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iU,cU->icU"    , _INPUT_13       , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,icU->Tic"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("Tic,SWc->TiSW" , _M5             , _M9             )
    del _M5         
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("iW,TiSW->TSiW" , _INPUT_19       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("kS,kW->SWk"    , _INPUT_8        , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("kT,SWk->TSW"   , _INPUT_11       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("TSiW,TSW->iTSW", _M11            , _M18            )
    del _M11        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19_perm        = np.transpose(_M19            , (1, 3, 2, 0)    )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("QVSi,TWSi->QVTW", _M16_perm       , _M19_perm       )
    del _M16_perm   
    del _M19_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (2, 0, 1, 3)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("aQ,VWa->QVWa"  , _INPUT_4        , _M6             )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("aT,QVWa->TQVW" , _INPUT_12       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TQVW,TQVW->"   , _M8             , _M20_perm       )
    del _M8         
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_2(Z           : np.ndarray,
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
    # step 0 jR,bR->jbR 
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
    _benchmark_time(t1, t2, "step 1")
    # step 1 RS,jbR->Sjb 
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
    _benchmark_time(t1, t2, "step 2")
    # step 2 Sjb->Sbj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M3_perm         = np.ndarray((NTHC_INT, NVIR, NOCC), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_perm.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 Sbj,QVj->SbQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[1]
    _M3_perm_reshaped = _M3_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M13.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    lib.ddot(_M3_perm_reshaped, _M12_reshaped.T, c=_M13_reshaped)
    _M13         = _M13_reshaped.reshape(*shape_backup)
    del _M3_perm    
    del _M12        
    del _M12_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 bV,SbQV->SQbV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, NVIR, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_void_p(_M14.ctypes.data),
                                    ctypes.c_int(_INPUT_18.shape[0]),
                                    ctypes.c_int(_INPUT_18.shape[1]),
                                    ctypes.c_int(_M13.shape[0]),
                                    ctypes.c_int(_M13.shape[2]))
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 SQbV->QSVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M14_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M14.ctypes.data),
                                 ctypes.c_void_p(_M14_perm.ctypes.data),
                                 ctypes.c_int(_M14.shape[0]),
                                 ctypes.c_int(_M14.shape[1]),
                                 ctypes.c_int(_M14.shape[2]),
                                 ctypes.c_int(_M14.shape[3]))
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,bP->ibP 
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
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQ,ibP->Qib 
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 Qib,QSVb->QiSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0342_0134_wob = getattr(libpbc, "fn_contraction_012_0342_0134_wob", None)
    assert fn_contraction_012_0342_0134_wob is not None
    _M15             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0342_0134_wob(ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_void_p(_M14_perm.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M1.shape[0]),
                                     ctypes.c_int(_M1.shape[1]),
                                     ctypes.c_int(_M1.shape[2]),
                                     ctypes.c_int(_M14_perm.shape[1]),
                                     ctypes.c_int(_M14_perm.shape[2]))
    del _M1         
    del _M14_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iV,QiSV->QSiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M15.ctypes.data),
                                    ctypes.c_void_p(_M16.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M15.shape[0]),
                                    ctypes.c_int(_M15.shape[2]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 QSiV->QVSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_void_p(_M16_perm.ctypes.data),
                                 ctypes.c_int(_M16.shape[0]),
                                 ctypes.c_int(_M16.shape[1]),
                                 ctypes.c_int(_M16.shape[2]),
                                 ctypes.c_int(_M16.shape[3]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 iU,cU->icU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 TU,icU->Tic 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 cS,cW->SWc 
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
    _benchmark_time(t1, t2, "step 15")
    # step 15 Tic,SWc->TiSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M5         
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 iW,TiSW->TSiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M10.ctypes.data),
                                    ctypes.c_void_p(_M11.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M10.shape[0]),
                                    ctypes.c_int(_M10.shape[2]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M17.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 kT,SWk->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 TSiW,TSW->iTSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_013_2013_wob = getattr(libpbc, "fn_contraction_0123_013_2013_wob", None)
    assert fn_contraction_0123_013_2013_wob is not None
    _M19             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_013_2013_wob(ctypes.c_void_p(_M11.ctypes.data),
                                     ctypes.c_void_p(_M18.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M11.shape[0]),
                                     ctypes.c_int(_M11.shape[1]),
                                     ctypes.c_int(_M11.shape[2]),
                                     ctypes.c_int(_M11.shape[3]))
    del _M11        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 iTSW->TWSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1320_wob = getattr(libpbc, "fn_permutation_0123_1320_wob", None)
    assert fn_permutation_0123_1320_wob is not None
    _M19_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_1320_wob(ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_void_p(_M19_perm.ctypes.data),
                                 ctypes.c_int(_M19.shape[0]),
                                 ctypes.c_int(_M19.shape[1]),
                                 ctypes.c_int(_M19.shape[2]),
                                 ctypes.c_int(_M19.shape[3]))
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 QVSi,TWSi->QVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[1]
    _M16_perm_reshaped = _M16_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[1]
    _M19_perm_reshaped = _M19_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _size_dim_1      = _size_dim_1 * _M20.shape[1]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M16_perm_reshaped, _M19_perm_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M16_perm   
    del _M19_perm   
    del _M19_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QVTW->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_2013_wob = getattr(libpbc, "fn_permutation_0123_2013_wob", None)
    assert fn_permutation_0123_2013_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_2013_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 aQ,VWa->QVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M7.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]))
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 aT,QVWa->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _size_dim_1      = _size_dim_1 * _M7.shape[2]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8          = _M8_reshaped.reshape(*shape_backup)
    del _M7         
    del _M7_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TQVW,TQVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M8.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M8.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M8         
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_2_determine_bucket_size(NVIR        : int,
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
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NVIR * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NVIR))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M7_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M8_size         = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M10_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M11_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M12_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M13_size        = (NTHC_INT * (NVIR * (NTHC_INT * N_LAPLACE)))
    _M14_size        = (NTHC_INT * (NTHC_INT * (NVIR * N_LAPLACE)))
    _M15_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M16_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M17_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M19_size        = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M20_size        = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M3_perm_size    = (NTHC_INT * (NOCC * NVIR))
    _M14_perm_size   = (NTHC_INT * (NTHC_INT * (NVIR * N_LAPLACE)))
    _M16_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M19_perm_size   = (NOCC * (NTHC_INT * (NTHC_INT * N_LAPLACE)))
    _M20_perm_size   = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M3_perm_size    = _M3_size        
    _M16_perm_size   = _M16_size       
    _M20_perm_size   = _M20_size       
    _M14_perm_size   = _M14_size       
    _M19_perm_size   = _M19_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M3_perm_size)
    bucked_0_size    = max(bucked_0_size, _M14_size)
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M15_size)
    bucked_0_size    = max(bucked_0_size, _M16_perm_size)
    bucked_0_size    = max(bucked_0_size, _M20_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M3_size)
    bucked_1_size    = max(bucked_1_size, _M12_size)
    bucked_1_size    = max(bucked_1_size, _M14_perm_size)
    bucked_1_size    = max(bucked_1_size, _M16_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M19_perm_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M8_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M13_size)
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M17_size)
    bucked_2_size    = max(bucked_2_size, _M19_size)
    bucked_2_size    = max(bucked_2_size, _M20_size)
    bucked_2_size    = max(bucked_2_size, _M7_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M10_size)
    bucked_3_size    = max(bucked_3_size, _M18_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_2_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_XX_2_determine_bucket_size(NVIR = NVIR,
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
    # step 0 jR,bR->jbR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_0        
    _M2              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 RS,jbR->Sjb 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M3_offset       = offset_1        
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
    _benchmark_time(t1, t2, "step 2")
    # step 2 Sjb->Sbj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M3_perm_offset  = offset_0        
    _M3_perm         = np.ndarray((NTHC_INT, NVIR, NOCC), buffer = buffer, offset = _M3_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_perm.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12_offset      = offset_1        
    _M12             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M12_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 Sbj,QVj->SbQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NVIR * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M13_offset      = offset_2        
    _M13             = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[1]
    _M3_perm_reshaped = _M3_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M13.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    lib.ddot(_M3_perm_reshaped, _M12_reshaped.T, c=_M13_reshaped)
    _M13             = _M13_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 bV,SbQV->SQbV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M14_offset      = offset_0        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, NVIR, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_void_p(_M14.ctypes.data),
                                    ctypes.c_int(_INPUT_18.shape[0]),
                                    ctypes.c_int(_INPUT_18.shape[1]),
                                    ctypes.c_int(_M13.shape[0]),
                                    ctypes.c_int(_M13.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 SQbV->QSVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M14_perm_offset = offset_1        
    _M14_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M14_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M14.ctypes.data),
                                 ctypes.c_void_p(_M14_perm.ctypes.data),
                                 ctypes.c_int(_M14.shape[0]),
                                 ctypes.c_int(_M14.shape[1]),
                                 ctypes.c_int(_M14.shape[2]),
                                 ctypes.c_int(_M14.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,bP->ibP 
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
    _benchmark_time(t1, t2, "step 8")
    # step 8 PQ,ibP->Qib 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = offset_2        
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 Qib,QSVb->QiSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0342_0134_wob = getattr(libpbc, "fn_contraction_012_0342_0134_wob", None)
    assert fn_contraction_012_0342_0134_wob is not None
    _M15_offset      = offset_0        
    _M15             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    fn_contraction_012_0342_0134_wob(ctypes.c_void_p(_M1.ctypes.data),
                                     ctypes.c_void_p(_M14_perm.ctypes.data),
                                     ctypes.c_void_p(_M15.ctypes.data),
                                     ctypes.c_int(_M1.shape[0]),
                                     ctypes.c_int(_M1.shape[1]),
                                     ctypes.c_int(_M1.shape[2]),
                                     ctypes.c_int(_M14_perm.shape[1]),
                                     ctypes.c_int(_M14_perm.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iV,QiSV->QSiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M16_offset      = offset_1        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M15.ctypes.data),
                                    ctypes.c_void_p(_M16.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M15.shape[0]),
                                    ctypes.c_int(_M15.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 QSiV->QVSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M16_perm_offset = offset_0        
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), buffer = buffer, offset = _M16_perm_offset)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_void_p(_M16_perm.ctypes.data),
                                 ctypes.c_int(_M16.shape[0]),
                                 ctypes.c_int(_M16.shape[1]),
                                 ctypes.c_int(_M16.shape[2]),
                                 ctypes.c_int(_M16.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 iU,cU->icU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_1        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 TU,icU->Tic 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 Tic,SWc->TiSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M10_offset      = offset_3        
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 iW,TiSW->TSiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M10.ctypes.data),
                                    ctypes.c_void_p(_M11.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M10.shape[0]),
                                    ctypes.c_int(_M10.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 kS,kW->SWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M17_offset      = offset_2        
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M17_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M17.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 kT,SWk->TSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M18_offset      = offset_3        
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_11.shape[0]
    _INPUT_11_reshaped = _INPUT_11.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_11_reshaped.T, _M17_reshaped.T, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 TSiW,TSW->iTSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_013_2013_wob = getattr(libpbc, "fn_contraction_0123_013_2013_wob", None)
    assert fn_contraction_0123_013_2013_wob is not None
    _M19_offset      = offset_2        
    _M19             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M19_offset)
    fn_contraction_0123_013_2013_wob(ctypes.c_void_p(_M11.ctypes.data),
                                     ctypes.c_void_p(_M18.ctypes.data),
                                     ctypes.c_void_p(_M19.ctypes.data),
                                     ctypes.c_int(_M11.shape[0]),
                                     ctypes.c_int(_M11.shape[1]),
                                     ctypes.c_int(_M11.shape[2]),
                                     ctypes.c_int(_M11.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 iTSW->TWSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1320_wob = getattr(libpbc, "fn_permutation_0123_1320_wob", None)
    assert fn_permutation_0123_1320_wob is not None
    _M19_perm_offset = offset_1        
    _M19_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), buffer = buffer, offset = _M19_perm_offset)
    fn_permutation_0123_1320_wob(ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_void_p(_M19_perm.ctypes.data),
                                 ctypes.c_int(_M19.shape[0]),
                                 ctypes.c_int(_M19.shape[1]),
                                 ctypes.c_int(_M19.shape[2]),
                                 ctypes.c_int(_M19.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 QVSi,TWSi->QVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M20_offset      = offset_2        
    _M20             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M16_perm.shape[1]
    _M16_perm_reshaped = _M16_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[1]
    _M19_perm_reshaped = _M19_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _size_dim_1      = _size_dim_1 * _M20.shape[1]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M16_perm_reshaped, _M19_perm_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QVTW->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_2013_wob = getattr(libpbc, "fn_permutation_0123_2013_wob", None)
    assert fn_permutation_0123_2013_wob is not None
    _M20_perm_offset = offset_0        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_0123_2013_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_1        
    _M6              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 aQ,VWa->QVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M7_offset       = offset_2        
    _M7              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M7_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M6.ctypes.data),
                                   ctypes.c_void_p(_M7.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M6.shape[0]),
                                   ctypes.c_int(_M6.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 aT,QVWa->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M8_offset       = offset_1        
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M8_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _size_dim_1      = _size_dim_1 * _M7.shape[2]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M8.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M7_reshaped.T, c=_M8_reshaped)
    _M8              = _M8_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 TQVW,TQVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M8.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M8.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_3_naive(Z           : np.ndarray,
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
    _INPUT_6         = X_v             
    _INPUT_7         = X_v             
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
    _M0              = np.einsum("iP,aP->iaP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,iaP->Qia"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aR,aV->RVa"    , _INPUT_6        , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("Qia,RVa->QiRV" , _M1             , _M11            )
    del _M1         
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iV,QiRV->QRiV" , _INPUT_15       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13_perm        = np.transpose(_M13            , (1, 0, 3, 2)    )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iT,cT->icT"    , _INPUT_11       , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,icT->Uic"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("cR,cW->RWc"    , _INPUT_7        , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("Uic,RWc->UiRW" , _M5             , _M8             )
    del _M5         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iW,UiRW->URiW" , _INPUT_19       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10_perm        = np.transpose(_M10            , (1, 0, 3, 2)    )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("RUWi,RQVi->RUWQV", _M10_perm       , _M13_perm       )
    del _M10_perm   
    del _M13_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14_perm        = np.transpose(_M14            , (1, 2, 3, 4, 0) )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("bS,dS->bdS"    , _INPUT_8        , _INPUT_9        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,bdS->Rbd"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("dU,dW->UWd"    , _INPUT_14       , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("Rbd,UWd->RbUW" , _M3             , _M6             )
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("bQ,bV->QVb"    , _INPUT_4        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("RUWb,QVb->RUWQV", _M7_perm        , _M15            )
    del _M7_perm    
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (1, 2, 3, 4, 0) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("UWQVR,UWQVR->UWQV", _M14_perm       , _M16_perm       )
    del _M14_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("jV,jW->VWj"    , _INPUT_16       , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("jQ,VWj->QVWj"  , _INPUT_3        , _M18            )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("jU,QVWj->UQVW" , _INPUT_13       , _M19            )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 3, 1, 2)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("UWQV,UWQV->"   , _M17            , _M20_perm       )
    del _M17        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_XX_3(Z           : np.ndarray,
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
    _INPUT_6         = X_v             
    _INPUT_7         = X_v             
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
    # step 2 aR,aV->RVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qia,RVa->QiRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M1         
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiRV->QRiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 QRiV->RQVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M13_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_void_p(_M13_perm.ctypes.data),
                                 ctypes.c_int(_M13.shape[0]),
                                 ctypes.c_int(_M13.shape[1]),
                                 ctypes.c_int(_M13.shape[2]),
                                 ctypes.c_int(_M13.shape[3]))
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iT,cT->icT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TU,icT->Uic 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 cR,cW->RWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 Uic,RWc->UiRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M8_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M5         
    del _M8         
    del _M8_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iW,UiRW->URiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M9.ctypes.data),
                                    ctypes.c_void_p(_M10.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M9.shape[0]),
                                    ctypes.c_int(_M9.shape[2]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 URiW->RUWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M10_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_perm.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(_M10.shape[3]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RUWi,RQVi->RUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0453_01245_wob = getattr(libpbc, "fn_contraction_0123_0453_01245_wob", None)
    assert fn_contraction_0123_0453_01245_wob is not None
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0453_01245_wob(ctypes.c_void_p(_M10_perm.ctypes.data),
                                       ctypes.c_void_p(_M13_perm.ctypes.data),
                                       ctypes.c_void_p(_M14.ctypes.data),
                                       ctypes.c_int(_M10_perm.shape[0]),
                                       ctypes.c_int(_M10_perm.shape[1]),
                                       ctypes.c_int(_M10_perm.shape[2]),
                                       ctypes.c_int(_M10_perm.shape[3]),
                                       ctypes.c_int(_M13_perm.shape[1]),
                                       ctypes.c_int(_M13_perm.shape[2]))
    del _M10_perm   
    del _M13_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 RUWQV->UWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12340_wob = getattr(libpbc, "fn_permutation_01234_12340_wob", None)
    assert fn_permutation_01234_12340_wob is not None
    _M14_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_12340_wob(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_perm.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(_M14.shape[3]),
                                   ctypes.c_int(_M14.shape[4]))
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 bS,dS->bdS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NVIR, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_9.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RS,bdS->Rbd 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NVIR, NVIR), dtype=np.float64)
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
    lib.ddot(_INPUT_5_reshaped, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 dU,dW->UWd 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 Rbd,UWd->RbUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M3         
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 RbUW->RUWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 RUWb,QVb->RUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_M7_perm_reshaped, _M15_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M15        
    del _M15_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 RUWQV->UWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12340_wob = getattr(libpbc, "fn_permutation_01234_12340_wob", None)
    assert fn_permutation_01234_12340_wob is not None
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_12340_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 UWQVR,UWQVR->UWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_0123_wob = getattr(libpbc, "fn_contraction_01234_01234_0123_wob", None)
    assert fn_contraction_01234_01234_0123_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_01234_0123_wob(ctypes.c_void_p(_M14_perm.ctypes.data),
                                        ctypes.c_void_p(_M16_perm.ctypes.data),
                                        ctypes.c_void_p(_M17.ctypes.data),
                                        ctypes.c_int(_M14_perm.shape[0]),
                                        ctypes.c_int(_M14_perm.shape[1]),
                                        ctypes.c_int(_M14_perm.shape[2]),
                                        ctypes.c_int(_M14_perm.shape[3]),
                                        ctypes.c_int(_M14_perm.shape[4]))
    del _M14_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 jV,jW->VWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18             = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_16.shape[0]),
                                 ctypes.c_int(_INPUT_16.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 jQ,VWj->QVWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                   ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_int(_INPUT_3.shape[0]),
                                   ctypes.c_int(_INPUT_3.shape[1]),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 jU,QVWj->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 UQVW->UWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 UWQV,UWQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M17.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M17.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M17        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_XX_3_determine_bucket_size(NVIR        : int,
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
    _INPUT_6_size    = (NVIR * NTHC_INT)
    _INPUT_7_size    = (NVIR * NTHC_INT)
    _INPUT_8_size    = (NVIR * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NVIR * (NVIR * NTHC_INT))
    _M3_size         = (NTHC_INT * (NVIR * NVIR))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M7_size         = (NTHC_INT * (NVIR * (NTHC_INT * N_LAPLACE)))
    _M8_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M9_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M10_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M12_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M13_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M14_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M15_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M16_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M17_size        = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M18_size        = (N_LAPLACE * (N_LAPLACE * NOCC))
    _M19_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NOCC)))
    _M20_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M13_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M10_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M14_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M7_perm_size    = (NTHC_INT * (NVIR * (NTHC_INT * N_LAPLACE)))
    _M16_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M7_perm_size    = _M7_size        
    _M16_perm_size   = _M16_size       
    _M13_perm_size   = _M13_size       
    _M14_perm_size   = _M14_size       
    _M20_perm_size   = _M20_size       
    _M10_perm_size   = _M10_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M11_size)
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M10_size)
    bucked_0_size    = max(bucked_0_size, _M14_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    bucked_0_size    = max(bucked_0_size, _M7_perm_size)
    bucked_0_size    = max(bucked_0_size, _M16_perm_size)
    bucked_0_size    = max(bucked_0_size, _M18_size)
    bucked_0_size    = max(bucked_0_size, _M20_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _M13_perm_size)
    bucked_1_size    = max(bucked_1_size, _M14_perm_size)
    bucked_1_size    = max(bucked_1_size, _M19_size)
    bucked_1_size    = max(bucked_1_size, _M20_perm_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M12_size)
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M10_perm_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M15_size)
    bucked_2_size    = max(bucked_2_size, _M17_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M9_size)
    bucked_3_size    = max(bucked_3_size, _M7_size)
    bucked_3_size    = max(bucked_3_size, _M16_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_3_opt_mem(Z           : np.ndarray,
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
    _INPUT_6         = X_v             
    _INPUT_7         = X_v             
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
    # deal with buffer
    bucket_size      = RMP3_XX_3_determine_bucket_size(NVIR = NVIR,
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
    # step 2 aR,aV->RVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11_offset      = offset_0        
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qia,RVa->QiRV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M12_offset      = offset_2        
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiRV->QRiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13_offset      = offset_0        
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 QRiV->RQVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M13_perm_offset = offset_1        
    _M13_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M13_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_void_p(_M13_perm.ctypes.data),
                                 ctypes.c_int(_M13.shape[0]),
                                 ctypes.c_int(_M13.shape[1]),
                                 ctypes.c_int(_M13.shape[2]),
                                 ctypes.c_int(_M13.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iT,cT->icT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_0        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TU,icT->Uic 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 cR,cW->RWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_7.shape[0]),
                                 ctypes.c_int(_INPUT_7.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 Uic,RWc->UiRW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M9_offset       = offset_3        
    _M9              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M8_reshaped.T, c=_M9_reshaped)
    _M9              = _M9_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iW,UiRW->URiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M10_offset      = offset_0        
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M9.ctypes.data),
                                    ctypes.c_void_p(_M10.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M9.shape[0]),
                                    ctypes.c_int(_M9.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 URiW->RUWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M10_perm_offset = offset_2        
    _M10_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M10_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_perm.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(_M10.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RUWi,RQVi->RUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0453_01245_wob = getattr(libpbc, "fn_contraction_0123_0453_01245_wob", None)
    assert fn_contraction_0123_0453_01245_wob is not None
    _M14_offset      = offset_0        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    fn_contraction_0123_0453_01245_wob(ctypes.c_void_p(_M10_perm.ctypes.data),
                                       ctypes.c_void_p(_M13_perm.ctypes.data),
                                       ctypes.c_void_p(_M14.ctypes.data),
                                       ctypes.c_int(_M10_perm.shape[0]),
                                       ctypes.c_int(_M10_perm.shape[1]),
                                       ctypes.c_int(_M10_perm.shape[2]),
                                       ctypes.c_int(_M10_perm.shape[3]),
                                       ctypes.c_int(_M13_perm.shape[1]),
                                       ctypes.c_int(_M13_perm.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 RUWQV->UWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12340_wob = getattr(libpbc, "fn_permutation_01234_12340_wob", None)
    assert fn_permutation_01234_12340_wob is not None
    _M14_perm_offset = offset_1        
    _M14_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M14_perm_offset)
    fn_permutation_01234_12340_wob(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_perm.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(_M14.shape[3]),
                                   ctypes.c_int(_M14.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 bS,dS->bdS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_0        
    _M2              = np.ndarray((NVIR, NVIR, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_9.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RS,bdS->Rbd 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NVIR * (NVIR * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NVIR, NVIR), buffer = buffer, offset = _M3_offset)
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
    lib.ddot(_INPUT_5_reshaped, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 dU,dW->UWd 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_0        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_14.shape[0]),
                                 ctypes.c_int(_INPUT_14.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 Rbd,UWd->RbUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NVIR * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_3        
    _M7              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 RbUW->RUWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm_offset  = offset_0        
    _M7_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M7_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 bQ,bV->QVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15_offset      = offset_2        
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 RUWb,QVb->RUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M16_offset      = offset_3        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_M7_perm_reshaped, _M15_reshaped.T, c=_M16_reshaped)
    _M16             = _M16_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 RUWQV->UWQVR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12340_wob = getattr(libpbc, "fn_permutation_01234_12340_wob", None)
    assert fn_permutation_01234_12340_wob is not None
    _M16_perm_offset = offset_0        
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M16_perm_offset)
    fn_permutation_01234_12340_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 UWQVR,UWQVR->UWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_0123_wob = getattr(libpbc, "fn_contraction_01234_01234_0123_wob", None)
    assert fn_contraction_01234_01234_0123_wob is not None
    _M17_offset      = offset_2        
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    fn_contraction_01234_01234_0123_wob(ctypes.c_void_p(_M14_perm.ctypes.data),
                                        ctypes.c_void_p(_M16_perm.ctypes.data),
                                        ctypes.c_void_p(_M17.ctypes.data),
                                        ctypes.c_int(_M14_perm.shape[0]),
                                        ctypes.c_int(_M14_perm.shape[1]),
                                        ctypes.c_int(_M14_perm.shape[2]),
                                        ctypes.c_int(_M14_perm.shape[3]),
                                        ctypes.c_int(_M14_perm.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 jV,jW->VWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18_offset      = offset_0        
    _M18             = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M18_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_16.shape[0]),
                                 ctypes.c_int(_INPUT_16.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 jQ,VWj->QVWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M19_offset      = offset_1        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                   ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_int(_INPUT_3.shape[0]),
                                   ctypes.c_int(_INPUT_3.shape[1]),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 jU,QVWj->UQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M20_offset      = offset_0        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 UQVW->UWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M20_perm_offset = offset_1        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 UWQV,UWQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M17.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M17.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    return _M21

def RMP3_XX_4_naive(Z           : np.ndarray,
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
    _INPUT_6         = X_v             
    _INPUT_7         = X_v             
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
    _M0              = np.einsum("iP,bP->ibP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,ibP->Qib"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("bS,bV->SVb"    , _INPUT_8        , _INPUT_18       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("Qib,SVb->QiSV" , _M1             , _M8             )
    del _M1         
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iV,QiSV->QSiV" , _INPUT_15       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10_perm        = np.transpose(_M10            , (0, 1, 3, 2)    )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("iT,iW->TWi"    , _INPUT_11       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("QSVi,TWi->QSVTW", _M10_perm       , _M11            )
    del _M10_perm   
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("aR,cR->acR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,acR->Sac"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3_perm         = np.transpose(_M3             , (0, 2, 1)       )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("aQ,aV->QVa"    , _INPUT_4        , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("Sca,QVa->ScQV" , _M3_perm        , _M6             )
    del _M3_perm    
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("cT,cW->TWc"    , _INPUT_12       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("SQVc,TWc->SQVTW", _M7_perm        , _M13            )
    del _M7_perm    
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("QSVTW,SQVTW->QSVTW", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("jU,dU->jdU"    , _INPUT_13       , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,jdU->Tjd"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("dS,dW->SWd"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("Tjd,SWd->TjSW" , _M5             , _M16            )
    del _M5         
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("jW,TjSW->TSjW" , _INPUT_20       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18_perm        = np.transpose(_M18            , (0, 1, 3, 2)    )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("TSWj,QVj->TSWQV", _M18_perm       , _M19            )
    del _M18_perm   
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (3, 1, 4, 0, 2) )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("QSVTW,QSVTW->" , _M15            , _M20_perm       )
    del _M15        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_4(Z           : np.ndarray,
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
    _INPUT_6         = X_v             
    _INPUT_7         = X_v             
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
    # step 0 iP,bP->ibP 
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
    # step 1 PQ,ibP->Qib 
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
    # step 2 bS,bV->SVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qib,SVb->QiSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M8_reshaped.T, c=_M9_reshaped)
    _M9          = _M9_reshaped.reshape(*shape_backup)
    del _M1         
    del _M8         
    del _M8_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiSV->QSiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M9.ctypes.data),
                                    ctypes.c_void_p(_M10.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M9.shape[0]),
                                    ctypes.c_int(_M9.shape[2]))
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 QSiV->QSVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132_wob = getattr(libpbc, "fn_permutation_0123_0132_wob", None)
    assert fn_permutation_0123_0132_wob is not None
    _M10_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_0132_wob(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_perm.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(_M10.shape[3]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iT,iW->TWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QSVi,TWi->QSVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[2]
    _M10_perm_reshaped = _M10_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _size_dim_1      = _size_dim_1 * _M12.shape[2]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M10_perm_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M10_perm   
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 aR,cR->acR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NVIR, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,acR->Sac 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NVIR, NVIR), dtype=np.float64)
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
    _benchmark_time(t1, t2, "step 10")
    # step 10 Sac->Sca 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M3_perm         = np.ndarray((NTHC_INT, NVIR, NVIR), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_perm.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 aQ,aV->QVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 Sca,QVa->ScQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[1]
    _M3_perm_reshaped = _M3_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_perm_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M3_perm    
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 ScQV->SQVc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 cT,cW->TWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 SQVc,TWc->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _size_dim_1      = _size_dim_1 * _M14.shape[2]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M7_perm_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 QSVTW,SQVTW->QSVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_10234_01234_wob = getattr(libpbc, "fn_contraction_01234_10234_01234_wob", None)
    assert fn_contraction_01234_10234_01234_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_10234_01234_wob(ctypes.c_void_p(_M12.ctypes.data),
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
    _benchmark_time(t1, t2, "step 17")
    # step 17 jU,dU->jdU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 TU,jdU->Tjd 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 dS,dW->SWd 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M16             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 Tjd,SWd->TjSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M16_reshaped.T, c=_M17_reshaped)
    _M17         = _M17_reshaped.reshape(*shape_backup)
    del _M5         
    del _M16        
    del _M16_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 jW,TjSW->TSjW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                    ctypes.c_void_p(_M17.ctypes.data),
                                    ctypes.c_void_p(_M18.ctypes.data),
                                    ctypes.c_int(_INPUT_20.shape[0]),
                                    ctypes.c_int(_INPUT_20.shape[1]),
                                    ctypes.c_int(_M17.shape[0]),
                                    ctypes.c_int(_M17.shape[2]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 TSjW->TSWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132_wob = getattr(libpbc, "fn_permutation_0123_0132_wob", None)
    assert fn_permutation_0123_0132_wob is not None
    _M18_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_0132_wob(ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_void_p(_M18_perm.ctypes.data),
                                 ctypes.c_int(_M18.shape[0]),
                                 ctypes.c_int(_M18.shape[1]),
                                 ctypes.c_int(_M18.shape[2]),
                                 ctypes.c_int(_M18.shape[3]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 TSWj,QVj->TSWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M18_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M18_perm.shape[2]
    _M18_perm_reshaped = _M18_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _size_dim_1      = _size_dim_1 * _M20.shape[1]
    _size_dim_1      = _size_dim_1 * _M20.shape[2]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M18_perm_reshaped, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M18_perm   
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TSWQV->QSVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_31402_wob = getattr(libpbc, "fn_permutation_01234_31402_wob", None)
    assert fn_permutation_01234_31402_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_31402_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 QSVTW,QSVTW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M15.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M15.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M15        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_4_determine_bucket_size(NVIR        : int,
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
    _INPUT_6_size    = (NVIR * NTHC_INT)
    _INPUT_7_size    = (NVIR * NTHC_INT)
    _INPUT_8_size    = (NVIR * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NVIR * (NVIR * NTHC_INT))
    _M3_size         = (NTHC_INT * (NVIR * NVIR))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M7_size         = (NTHC_INT * (NVIR * (NTHC_INT * N_LAPLACE)))
    _M8_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M9_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M10_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M12_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M13_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M14_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M15_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M16_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M17_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M18_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M20_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M10_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M3_perm_size    = (NTHC_INT * (NVIR * NVIR))
    _M7_perm_size    = (NTHC_INT * (NVIR * (NTHC_INT * N_LAPLACE)))
    _M18_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M3_perm_size    = _M3_size        
    _M7_perm_size    = _M7_size        
    _M10_perm_size   = _M10_size       
    _M20_perm_size   = _M20_size       
    _M18_perm_size   = _M18_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M10_size)
    bucked_0_size    = max(bucked_0_size, _M11_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M3_perm_size)
    bucked_0_size    = max(bucked_0_size, _M7_perm_size)
    bucked_0_size    = max(bucked_0_size, _M15_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _M10_perm_size)
    bucked_1_size    = max(bucked_1_size, _M3_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M13_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M16_size)
    bucked_1_size    = max(bucked_1_size, _M18_size)
    bucked_1_size    = max(bucked_1_size, _M19_size)
    bucked_1_size    = max(bucked_1_size, _M20_perm_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M9_size)
    bucked_2_size    = max(bucked_2_size, _M12_size)
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M18_perm_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M7_size)
    bucked_3_size    = max(bucked_3_size, _M14_size)
    bucked_3_size    = max(bucked_3_size, _M17_size)
    bucked_3_size    = max(bucked_3_size, _M20_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_4_opt_mem(Z           : np.ndarray,
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
    _INPUT_6         = X_v             
    _INPUT_7         = X_v             
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
    # deal with buffer
    bucket_size      = RMP3_XX_4_determine_bucket_size(NVIR = NVIR,
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
    # step 0 iP,bP->ibP 
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
    # step 1 PQ,ibP->Qib 
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
    # step 2 bS,bV->SVb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qib,SVb->QiSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M9_offset       = offset_2        
    _M9              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M9_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M8.shape[0]
    _size_dim_1      = _size_dim_1 * _M8.shape[1]
    _M8_reshaped = _M8.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M9.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M8_reshaped.T, c=_M9_reshaped)
    _M9              = _M9_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiSV->QSiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M10_offset      = offset_0        
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M9.ctypes.data),
                                    ctypes.c_void_p(_M10.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M9.shape[0]),
                                    ctypes.c_int(_M9.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 QSiV->QSVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132_wob = getattr(libpbc, "fn_permutation_0123_0132_wob", None)
    assert fn_permutation_0123_0132_wob is not None
    _M10_perm_offset = offset_1        
    _M10_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M10_perm_offset)
    fn_permutation_0123_0132_wob(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_perm.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(_M10.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iT,iW->TWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11_offset      = offset_0        
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 QSVi,TWi->QSVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M12_offset      = offset_2        
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[2]
    _M10_perm_reshaped = _M10_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _size_dim_1      = _size_dim_1 * _M12.shape[2]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M10_perm_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 aR,cR->acR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_0        
    _M2              = np.ndarray((NVIR, NVIR, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,acR->Sac 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NVIR * (NVIR * _itemsize)))
    _M3_offset       = offset_1        
    _M3              = np.ndarray((NTHC_INT, NVIR, NVIR), buffer = buffer, offset = _M3_offset)
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
    _benchmark_time(t1, t2, "step 10")
    # step 10 Sac->Sca 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M3_perm_offset  = offset_0        
    _M3_perm         = np.ndarray((NTHC_INT, NVIR, NVIR), buffer = buffer, offset = _M3_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_perm.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 aQ,aV->QVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_1        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_4.shape[0]),
                                 ctypes.c_int(_INPUT_4.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 Sca,QVa->ScQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NVIR * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_3        
    _M7              = np.ndarray((NTHC_INT, NVIR, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[1]
    _M3_perm_reshaped = _M3_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_perm_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 ScQV->SQVc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm_offset  = offset_0        
    _M7_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M7_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 cT,cW->TWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13_offset      = offset_1        
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_12.shape[0]),
                                 ctypes.c_int(_INPUT_12.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 SQVc,TWc->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M14_offset      = offset_3        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _size_dim_1      = _size_dim_1 * _M14.shape[2]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M7_perm_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 QSVTW,SQVTW->QSVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_10234_01234_wob = getattr(libpbc, "fn_contraction_01234_10234_01234_wob", None)
    assert fn_contraction_01234_10234_01234_wob is not None
    _M15_offset      = offset_0        
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    fn_contraction_01234_10234_01234_wob(ctypes.c_void_p(_M12.ctypes.data),
                                         ctypes.c_void_p(_M14.ctypes.data),
                                         ctypes.c_void_p(_M15.ctypes.data),
                                         ctypes.c_int(_M12.shape[0]),
                                         ctypes.c_int(_M12.shape[1]),
                                         ctypes.c_int(_M12.shape[2]),
                                         ctypes.c_int(_M12.shape[3]),
                                         ctypes.c_int(_M12.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 jU,dU->jdU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_1        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 TU,jdU->Tjd 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 dS,dW->SWd 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M16_offset      = offset_1        
    _M16             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M16_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 Tjd,SWd->TjSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M17_offset      = offset_3        
    _M17             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M16_reshaped.T, c=_M17_reshaped)
    _M17             = _M17_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 jW,TjSW->TSjW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M18_offset      = offset_1        
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                    ctypes.c_void_p(_M17.ctypes.data),
                                    ctypes.c_void_p(_M18.ctypes.data),
                                    ctypes.c_int(_INPUT_20.shape[0]),
                                    ctypes.c_int(_INPUT_20.shape[1]),
                                    ctypes.c_int(_M17.shape[0]),
                                    ctypes.c_int(_M17.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 TSjW->TSWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0132_wob = getattr(libpbc, "fn_permutation_0123_0132_wob", None)
    assert fn_permutation_0123_0132_wob is not None
    _M18_perm_offset = offset_2        
    _M18_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M18_perm_offset)
    fn_permutation_0123_0132_wob(ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_void_p(_M18_perm.ctypes.data),
                                 ctypes.c_int(_M18.shape[0]),
                                 ctypes.c_int(_M18.shape[1]),
                                 ctypes.c_int(_M18.shape[2]),
                                 ctypes.c_int(_M18.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19_offset      = offset_1        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 TSWj,QVj->TSWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M20_offset      = offset_3        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M18_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M18_perm.shape[2]
    _M18_perm_reshaped = _M18_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _size_dim_1      = _size_dim_1 * _M20.shape[1]
    _size_dim_1      = _size_dim_1 * _M20.shape[2]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M18_perm_reshaped, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 TSWQV->QSVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_31402_wob = getattr(libpbc, "fn_permutation_01234_31402_wob", None)
    assert fn_permutation_01234_31402_wob is not None
    _M20_perm_offset = offset_1        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_01234_31402_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 QSVTW,QSVTW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M15.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M15.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_5_naive(Z           : np.ndarray,
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
    _INPUT_8         = X_o             
    _INPUT_9         = X_o             
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
    _M8              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("aT,VWa->TVWa"  , _INPUT_12       , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,aP->iaP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,iaP->Qia"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("Qia,TVWa->QiTVW", _M1             , _M9             )
    del _M1         
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("iV,QiTVW->QTWiV", _INPUT_15       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (1, 2, 0, 4, 3) )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("iR,kR->ikR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,ikR->Sik"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("kT,kW->TWk"    , _INPUT_11       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("Sik,TWk->SiTW" , _M3             , _M6             )
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (2, 3, 0, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("TWSi,TWQVi->TWSQV", _M7_perm        , _M11_perm       )
    del _M7_perm    
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17_perm        = np.transpose(_M17            , (2, 3, 4, 0, 1) )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("bV,bW->VWb"    , _INPUT_18       , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("bQ,VWb->QVWb"  , _INPUT_4        , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("lU,bU->lbU"    , _INPUT_13       , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,lbU->Tlb"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("Tlb,QVWb->TlQVW", _M5             , _M13            )
    del _M5         
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("lW,TlQVW->TQVlW", _INPUT_20       , _M14            )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15_perm        = np.transpose(_M15            , (0, 1, 2, 4, 3) )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("lS,TQVWl->STQVW", _INPUT_9        , _M15_perm       )
    del _M15_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (0, 2, 3, 1, 4) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("SQVTW,SQVTW->SQV", _M16_perm       , _M17_perm       )
    del _M16_perm   
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("jS,QVj->SQV"   , _INPUT_8        , _M19            )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("SQV,SQV->"     , _M18            , _M20            )
    del _M18        
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_5(Z           : np.ndarray,
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
    _INPUT_8         = X_o             
    _INPUT_9         = X_o             
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
    # step 0 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aT,VWa->TVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_12.shape[0]),
                                   ctypes.c_int(_INPUT_12.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 iP,aP->iaP 
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
    _benchmark_time(t1, t2, "step 3")
    # step 3 PQ,iaP->Qia 
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
    _benchmark_time(t1, t2, "step 4")
    # step 4 Qia,TVWa->QiTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _size_dim_1      = _size_dim_1 * _M9.shape[2]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M1         
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iV,QiTVW->QTWiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20314_23401_wob = getattr(libpbc, "fn_contraction_01_20314_23401_wob", None)
    assert fn_contraction_01_20314_23401_wob is not None
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_20314_23401_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                      ctypes.c_void_p(_M10.ctypes.data),
                                      ctypes.c_void_p(_M11.ctypes.data),
                                      ctypes.c_int(_INPUT_15.shape[0]),
                                      ctypes.c_int(_INPUT_15.shape[1]),
                                      ctypes.c_int(_M10.shape[0]),
                                      ctypes.c_int(_M10.shape[2]),
                                      ctypes.c_int(_M10.shape[4]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 QTWiV->TWQVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12043_wob = getattr(libpbc, "fn_permutation_01234_12043_wob", None)
    assert fn_permutation_01234_12043_wob is not None
    _M11_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_01234_12043_wob(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]),
                                   ctypes.c_int(_M11.shape[2]),
                                   ctypes.c_int(_M11.shape[3]),
                                   ctypes.c_int(_M11.shape[4]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iR,kR->ikR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RS,ikR->Sik 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 kT,kW->TWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 Sik,TWk->SiTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M3         
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 SiTW->TWSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_2301_wob = getattr(libpbc, "fn_permutation_0123_2301_wob", None)
    assert fn_permutation_0123_2301_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), dtype=np.float64)
    fn_permutation_0123_2301_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 TWSi,TWQVi->TWSQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01453_01245_wob = getattr(libpbc, "fn_contraction_0123_01453_01245_wob", None)
    assert fn_contraction_0123_01453_01245_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_01453_01245_wob(ctypes.c_void_p(_M7_perm.ctypes.data),
                                        ctypes.c_void_p(_M11_perm.ctypes.data),
                                        ctypes.c_void_p(_M17.ctypes.data),
                                        ctypes.c_int(_M7_perm.shape[0]),
                                        ctypes.c_int(_M7_perm.shape[1]),
                                        ctypes.c_int(_M7_perm.shape[2]),
                                        ctypes.c_int(_M7_perm.shape[3]),
                                        ctypes.c_int(_M11_perm.shape[2]),
                                        ctypes.c_int(_M11_perm.shape[3]))
    del _M7_perm    
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 TWSQV->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_23401_wob = getattr(libpbc, "fn_permutation_01234_23401_wob", None)
    assert fn_permutation_01234_23401_wob is not None
    _M17_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_23401_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 bQ,VWb->QVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M12.shape[0]),
                                   ctypes.c_int(_M12.shape[1]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 lU,bU->lbU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 TU,lbU->Tlb 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 Tlb,QVWb->TlQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M5         
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 lW,TlQVW->TQVlW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20341_23401_wob = getattr(libpbc, "fn_contraction_01_20341_23401_wob", None)
    assert fn_contraction_01_20341_23401_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_20341_23401_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_INPUT_20.shape[0]),
                                      ctypes.c_int(_INPUT_20.shape[1]),
                                      ctypes.c_int(_M14.shape[0]),
                                      ctypes.c_int(_M14.shape[2]),
                                      ctypes.c_int(_M14.shape[3]))
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 TQVlW->TQVWl 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M15_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 lS,TQVWl->STQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
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
    lib.ddot(_INPUT_9_reshaped.T, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M15_perm   
    del _M15_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 STQVW->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_02314_wob = getattr(libpbc, "fn_permutation_01234_02314_wob", None)
    assert fn_permutation_01234_02314_wob is not None
    _M16_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_02314_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 SQVTW,SQVTW->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_012_wob = getattr(libpbc, "fn_contraction_01234_01234_012_wob", None)
    assert fn_contraction_01234_01234_012_wob is not None
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_01234_012_wob(ctypes.c_void_p(_M16_perm.ctypes.data),
                                       ctypes.c_void_p(_M17_perm.ctypes.data),
                                       ctypes.c_void_p(_M18.ctypes.data),
                                       ctypes.c_int(_M16_perm.shape[0]),
                                       ctypes.c_int(_M16_perm.shape[1]),
                                       ctypes.c_int(_M16_perm.shape[2]),
                                       ctypes.c_int(_M16_perm.shape[3]),
                                       ctypes.c_int(_M16_perm.shape[4]))
    del _M16_perm   
    del _M17_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 jS,QVj->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 SQV,SQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M18.ctypes.data),
           ctypes.c_void_p(_M20.ctypes.data),
           ctypes.c_int(_M18.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M18        
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_5_determine_bucket_size(NVIR        : int,
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
    _INPUT_7_size    = (NOCC * NTHC_INT)
    _INPUT_8_size    = (NOCC * NTHC_INT)
    _INPUT_9_size    = (NOCC * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NOCC * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NOCC))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M7_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M8_size         = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M9_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M10_size        = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M11_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M12_size        = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M13_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M14_size        = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M15_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M16_size        = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M17_size        = (NTHC_INT * (N_LAPLACE * (NTHC_INT * (NTHC_INT * N_LAPLACE))))
    _M18_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M19_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M11_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M7_perm_size    = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M17_perm_size   = (NTHC_INT * (N_LAPLACE * (NTHC_INT * (NTHC_INT * N_LAPLACE))))
    _M15_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M16_perm_size   = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M11_perm_size   = _M11_size       
    _M17_perm_size   = _M17_size       
    _M7_perm_size    = _M7_size        
    _M15_perm_size   = _M15_size       
    _M16_perm_size   = _M16_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M10_size)
    bucked_0_size    = max(bucked_0_size, _M11_perm_size)
    bucked_0_size    = max(bucked_0_size, _M17_perm_size)
    bucked_0_size    = max(bucked_0_size, _M19_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    bucked_1_size    = max(bucked_1_size, _M7_perm_size)
    bucked_1_size    = max(bucked_1_size, _M12_size)
    bucked_1_size    = max(bucked_1_size, _M4_size)
    bucked_1_size    = max(bucked_1_size, _M14_size)
    bucked_1_size    = max(bucked_1_size, _M15_perm_size)
    bucked_1_size    = max(bucked_1_size, _M16_perm_size)
    bucked_1_size    = max(bucked_1_size, _M20_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M17_size)
    bucked_2_size    = max(bucked_2_size, _M13_size)
    bucked_2_size    = max(bucked_2_size, _M15_size)
    bucked_2_size    = max(bucked_2_size, _M16_size)
    bucked_2_size    = max(bucked_2_size, _M18_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M7_size)
    bucked_3_size    = max(bucked_3_size, _M5_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_5_opt_mem(Z           : np.ndarray,
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
    _INPUT_7         = X_o             
    _INPUT_8         = X_o             
    _INPUT_9         = X_o             
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
    # deal with buffer
    bucket_size      = RMP3_XX_5_determine_bucket_size(NVIR = NVIR,
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
    # step 0 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aT,VWa->TVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_12.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_12.shape[0]),
                                   ctypes.c_int(_INPUT_12.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 iP,aP->iaP 
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
    _benchmark_time(t1, t2, "step 3")
    # step 3 PQ,iaP->Qia 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = offset_2        
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
    _benchmark_time(t1, t2, "step 4")
    # step 4 Qia,TVWa->QiTVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M10_offset      = offset_0        
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _size_dim_1      = _size_dim_1 * _M9.shape[2]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 iV,QiTVW->QTWiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20314_23401_wob = getattr(libpbc, "fn_contraction_01_20314_23401_wob", None)
    assert fn_contraction_01_20314_23401_wob is not None
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_20314_23401_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                      ctypes.c_void_p(_M10.ctypes.data),
                                      ctypes.c_void_p(_M11.ctypes.data),
                                      ctypes.c_int(_INPUT_15.shape[0]),
                                      ctypes.c_int(_INPUT_15.shape[1]),
                                      ctypes.c_int(_M10.shape[0]),
                                      ctypes.c_int(_M10.shape[2]),
                                      ctypes.c_int(_M10.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 QTWiV->TWQVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12043_wob = getattr(libpbc, "fn_permutation_01234_12043_wob", None)
    assert fn_permutation_01234_12043_wob is not None
    _M11_perm_offset = offset_0        
    _M11_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M11_perm_offset)
    fn_permutation_01234_12043_wob(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]),
                                   ctypes.c_int(_M11.shape[2]),
                                   ctypes.c_int(_M11.shape[3]),
                                   ctypes.c_int(_M11.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iR,kR->ikR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 RS,ikR->Sik 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NOCC * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M3_offset)
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
    _benchmark_time(t1, t2, "step 9")
    # step 9 kT,kW->TWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_1        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 Sik,TWk->SiTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_3        
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 SiTW->TWSi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_2301_wob = getattr(libpbc, "fn_permutation_0123_2301_wob", None)
    assert fn_permutation_0123_2301_wob is not None
    _M7_perm_offset  = offset_1        
    _M7_perm         = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NOCC), buffer = buffer, offset = _M7_perm_offset)
    fn_permutation_0123_2301_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 TWSi,TWQVi->TWSQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01453_01245_wob = getattr(libpbc, "fn_contraction_0123_01453_01245_wob", None)
    assert fn_contraction_0123_01453_01245_wob is not None
    _M17_offset      = offset_2        
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    fn_contraction_0123_01453_01245_wob(ctypes.c_void_p(_M7_perm.ctypes.data),
                                        ctypes.c_void_p(_M11_perm.ctypes.data),
                                        ctypes.c_void_p(_M17.ctypes.data),
                                        ctypes.c_int(_M7_perm.shape[0]),
                                        ctypes.c_int(_M7_perm.shape[1]),
                                        ctypes.c_int(_M7_perm.shape[2]),
                                        ctypes.c_int(_M7_perm.shape[3]),
                                        ctypes.c_int(_M11_perm.shape[2]),
                                        ctypes.c_int(_M11_perm.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 TWSQV->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_23401_wob = getattr(libpbc, "fn_permutation_01234_23401_wob", None)
    assert fn_permutation_01234_23401_wob is not None
    _M17_perm_offset = offset_0        
    _M17_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_perm_offset)
    fn_permutation_01234_23401_wob(ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_void_p(_M17_perm.ctypes.data),
                                   ctypes.c_int(_M17.shape[0]),
                                   ctypes.c_int(_M17.shape[1]),
                                   ctypes.c_int(_M17.shape[2]),
                                   ctypes.c_int(_M17.shape[3]),
                                   ctypes.c_int(_M17.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M12_offset      = offset_1        
    _M12             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M12_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M12.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 bQ,VWb->QVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M13_offset      = offset_2        
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M12.ctypes.data),
                                   ctypes.c_void_p(_M13.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M12.shape[0]),
                                   ctypes.c_int(_M12.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 lU,bU->lbU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_1        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 TU,lbU->Tlb 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_3        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 Tlb,QVWb->TlQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M14_offset      = offset_1        
    _M14             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _size_dim_1      = _size_dim_1 * _M13.shape[2]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 lW,TlQVW->TQVlW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20341_23401_wob = getattr(libpbc, "fn_contraction_01_20341_23401_wob", None)
    assert fn_contraction_01_20341_23401_wob is not None
    _M15_offset      = offset_2        
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_20341_23401_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                      ctypes.c_void_p(_M14.ctypes.data),
                                      ctypes.c_void_p(_M15.ctypes.data),
                                      ctypes.c_int(_INPUT_20.shape[0]),
                                      ctypes.c_int(_INPUT_20.shape[1]),
                                      ctypes.c_int(_M14.shape[0]),
                                      ctypes.c_int(_M14.shape[2]),
                                      ctypes.c_int(_M14.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 TQVlW->TQVWl 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M15_perm_offset = offset_1        
    _M15_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M15_perm_offset)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M15_perm.ctypes.data),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]),
                                   ctypes.c_int(_M15.shape[2]),
                                   ctypes.c_int(_M15.shape[3]),
                                   ctypes.c_int(_M15.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 lS,TQVWl->STQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M16_offset      = offset_2        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_9.shape[0]
    _INPUT_9_reshaped = _INPUT_9.reshape(_size_dim_1,-1)
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
    lib.ddot(_INPUT_9_reshaped.T, _M15_perm_reshaped.T, c=_M16_reshaped)
    _M16             = _M16_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 STQVW->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_02314_wob = getattr(libpbc, "fn_permutation_01234_02314_wob", None)
    assert fn_permutation_01234_02314_wob is not None
    _M16_perm_offset = offset_1        
    _M16_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_perm_offset)
    fn_permutation_01234_02314_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 SQVTW,SQVTW->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_012_wob = getattr(libpbc, "fn_contraction_01234_01234_012_wob", None)
    assert fn_contraction_01234_01234_012_wob is not None
    _M18_offset      = offset_2        
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    fn_contraction_01234_01234_012_wob(ctypes.c_void_p(_M16_perm.ctypes.data),
                                       ctypes.c_void_p(_M17_perm.ctypes.data),
                                       ctypes.c_void_p(_M18.ctypes.data),
                                       ctypes.c_int(_M16_perm.shape[0]),
                                       ctypes.c_int(_M16_perm.shape[1]),
                                       ctypes.c_int(_M16_perm.shape[2]),
                                       ctypes.c_int(_M16_perm.shape[3]),
                                       ctypes.c_int(_M16_perm.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M19_offset      = offset_0        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M19.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 jS,QVj->SQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M20_offset      = offset_1        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_8.shape[0]
    _INPUT_8_reshaped = _INPUT_8.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_8_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 SQV,SQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M18.ctypes.data),
           ctypes.c_void_p(_M20.ctypes.data),
           ctypes.c_int(_M18.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    return _M21

def RMP3_XX_6_naive(Z           : np.ndarray,
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
    _INPUT_8         = X_o             
    _INPUT_9         = X_o             
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
    _M8              = np.einsum("aV,aW->VWa"    , _INPUT_17       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("aQ,VWa->QVWa"  , _INPUT_4        , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("kT,aT->kaT"    , _INPUT_11       , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,kaT->Uka"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("Uka,QVWa->UkQVW", _M5             , _M9             )
    del _M5         
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("kW,UkQVW->UQVkW", _INPUT_19       , _M10            )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11_perm        = np.transpose(_M11            , (0, 1, 2, 4, 3) )
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("kR,UQVWk->RUQVW", _INPUT_7        , _M11_perm       )
    del _M11_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jS,lS->jlS"    , _INPUT_8        , _INPUT_9        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,jlS->Rjl"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("lU,lW->UWl"    , _INPUT_13       , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("Rjl,UWl->RjUW" , _M3             , _M6             )
    del _M3         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7_perm         = np.transpose(_M7             , (0, 2, 3, 1)    )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("RUWj,QVj->RUWQV", _M7_perm        , _M13            )
    del _M7_perm    
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("RUQVW,RUWQV->RUQVW", _M12            , _M14            )
    del _M12        
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("bV,bW->VWb"    , _INPUT_18       , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("bU,VWb->UVWb"  , _INPUT_14       , _M16            )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("iP,bP->ibP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,ibP->Qib"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("Qib,UVWb->QiUVW", _M1             , _M17            )
    del _M1         
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("iV,QiUVW->QUWiV", _INPUT_15       , _M18            )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19_perm        = np.transpose(_M19            , (0, 1, 2, 4, 3) )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("iR,QUWVi->RQUWV", _INPUT_6        , _M19_perm       )
    del _M19_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 2, 1, 4, 3) )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("RUQVW,RUQVW->" , _M15            , _M20_perm       )
    del _M15        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_XX_6(Z           : np.ndarray,
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
    _INPUT_8         = X_o             
    _INPUT_9         = X_o             
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
    # step 0 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aQ,VWa->QVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 kT,aT->kaT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 TU,kaT->Uka 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 Uka,QVWa->UkQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _size_dim_1      = _size_dim_1 * _M9.shape[2]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M5         
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kW,UkQVW->UQVkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20341_23401_wob = getattr(libpbc, "fn_contraction_01_20341_23401_wob", None)
    assert fn_contraction_01_20341_23401_wob is not None
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_20341_23401_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                      ctypes.c_void_p(_M10.ctypes.data),
                                      ctypes.c_void_p(_M11.ctypes.data),
                                      ctypes.c_int(_INPUT_19.shape[0]),
                                      ctypes.c_int(_INPUT_19.shape[1]),
                                      ctypes.c_int(_M10.shape[0]),
                                      ctypes.c_int(_M10.shape[2]),
                                      ctypes.c_int(_M10.shape[3]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 UQVkW->UQVWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]),
                                   ctypes.c_int(_M11.shape[2]),
                                   ctypes.c_int(_M11.shape[3]),
                                   ctypes.c_int(_M11.shape[4]))
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 kR,UQVWk->RUQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[3]
    _M11_perm_reshaped = _M11_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M11_perm_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M11_perm   
    del _M11_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 jS,lS->jlS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_9.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,jlS->Rjl 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
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
    lib.ddot(_INPUT_5_reshaped, _M2_reshaped.T, c=_M3_reshaped)
    _M3          = _M3_reshaped.reshape(*shape_backup)
    del _M2         
    del _M2_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 lU,lW->UWl 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 Rjl,UWl->RjUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M3         
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RjUW->RUWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 RUWj,QVj->RUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _size_dim_1      = _size_dim_1 * _M14.shape[2]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M7_perm_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14         = _M14_reshaped.reshape(*shape_backup)
    del _M7_perm    
    del _M13        
    del _M13_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RUQVW,RUWQV->RUQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01423_01234_wob = getattr(libpbc, "fn_contraction_01234_01423_01234_wob", None)
    assert fn_contraction_01234_01423_01234_wob is not None
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_01423_01234_wob(ctypes.c_void_p(_M12.ctypes.data),
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M16             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 bU,VWb->UVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_int(_INPUT_14.shape[0]),
                                   ctypes.c_int(_INPUT_14.shape[1]),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 iP,bP->ibP 
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 PQ,ibP->Qib 
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
    _benchmark_time(t1, t2, "step 20")
    # step 20 Qib,UVWb->QiUVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _size_dim_1      = _size_dim_1 * _M18.shape[1]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M17_reshaped.T, c=_M18_reshaped)
    _M18         = _M18_reshaped.reshape(*shape_backup)
    del _M1         
    del _M17        
    del _M17_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 iV,QiUVW->QUWiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20314_23401_wob = getattr(libpbc, "fn_contraction_01_20314_23401_wob", None)
    assert fn_contraction_01_20314_23401_wob is not None
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_20314_23401_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_void_p(_M19.ctypes.data),
                                      ctypes.c_int(_INPUT_15.shape[0]),
                                      ctypes.c_int(_INPUT_15.shape[1]),
                                      ctypes.c_int(_M18.shape[0]),
                                      ctypes.c_int(_M18.shape[2]),
                                      ctypes.c_int(_M18.shape[4]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QUWiV->QUWVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M19_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_void_p(_M19_perm.ctypes.data),
                                   ctypes.c_int(_M19.shape[0]),
                                   ctypes.c_int(_M19.shape[1]),
                                   ctypes.c_int(_M19.shape[2]),
                                   ctypes.c_int(_M19.shape[3]),
                                   ctypes.c_int(_M19.shape[4]))
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 iR,QUWVi->RQUWV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[3]
    _M19_perm_reshaped = _M19_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M19_perm_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19_perm   
    del _M19_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 RQUWV->RUQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_02143_wob = getattr(libpbc, "fn_permutation_01234_02143_wob", None)
    assert fn_permutation_01234_02143_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_02143_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 RUQVW,RUQVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M15.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M15.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M15        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_XX_6_determine_bucket_size(NVIR        : int,
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
    _INPUT_7_size    = (NOCC * NTHC_INT)
    _INPUT_8_size    = (NOCC * NTHC_INT)
    _INPUT_9_size    = (NOCC * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NOCC * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NOCC))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M7_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M8_size         = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M9_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M10_size        = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M11_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M12_size        = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M13_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M14_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M15_size        = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M16_size        = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M17_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M18_size        = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M19_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M20_size        = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M11_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M7_perm_size    = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M19_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M11_perm_size   = _M11_size       
    _M7_perm_size    = _M7_size        
    _M19_perm_size   = _M19_size       
    _M20_perm_size   = _M20_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M10_size)
    bucked_0_size    = max(bucked_0_size, _M11_perm_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    bucked_0_size    = max(bucked_0_size, _M7_perm_size)
    bucked_0_size    = max(bucked_0_size, _M15_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M9_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M12_size)
    bucked_1_size    = max(bucked_1_size, _M16_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M18_size)
    bucked_1_size    = max(bucked_1_size, _M19_perm_size)
    bucked_1_size    = max(bucked_1_size, _M20_perm_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M13_size)
    bucked_2_size    = max(bucked_2_size, _M17_size)
    bucked_2_size    = max(bucked_2_size, _M19_size)
    bucked_2_size    = max(bucked_2_size, _M20_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M7_size)
    bucked_3_size    = max(bucked_3_size, _M14_size)
    bucked_3_size    = max(bucked_3_size, _M1_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_6_opt_mem(Z           : np.ndarray,
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
    _INPUT_7         = X_o             
    _INPUT_8         = X_o             
    _INPUT_9         = X_o             
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
    # deal with buffer
    bucket_size      = RMP3_XX_6_determine_bucket_size(NVIR = NVIR,
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
    # step 0 aV,aW->VWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_17.shape[0]),
                                 ctypes.c_int(_INPUT_17.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 aQ,VWa->QVWa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M9_offset       = offset_1        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 kT,aT->kaT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_0        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 TU,kaT->Uka 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 Uka,QVWa->UkQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M10_offset      = offset_0        
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _size_dim_1      = _size_dim_1 * _M9.shape[2]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kW,UkQVW->UQVkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20341_23401_wob = getattr(libpbc, "fn_contraction_01_20341_23401_wob", None)
    assert fn_contraction_01_20341_23401_wob is not None
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_20341_23401_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                      ctypes.c_void_p(_M10.ctypes.data),
                                      ctypes.c_void_p(_M11.ctypes.data),
                                      ctypes.c_int(_INPUT_19.shape[0]),
                                      ctypes.c_int(_INPUT_19.shape[1]),
                                      ctypes.c_int(_M10.shape[0]),
                                      ctypes.c_int(_M10.shape[2]),
                                      ctypes.c_int(_M10.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 UQVkW->UQVWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M11_perm_offset = offset_0        
    _M11_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M11_perm_offset)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M11.ctypes.data),
                                   ctypes.c_void_p(_M11_perm.ctypes.data),
                                   ctypes.c_int(_M11.shape[0]),
                                   ctypes.c_int(_M11.shape[1]),
                                   ctypes.c_int(_M11.shape[2]),
                                   ctypes.c_int(_M11.shape[3]),
                                   ctypes.c_int(_M11.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 kR,UQVWk->RUQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M12_offset      = offset_1        
    _M12             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_7.shape[0]
    _INPUT_7_reshaped = _INPUT_7.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M11_perm.shape[3]
    _M11_perm_reshaped = _M11_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_7_reshaped.T, _M11_perm_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 jS,lS->jlS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_0        
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_9.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 RS,jlS->Rjl 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NOCC * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M3_offset)
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
    lib.ddot(_INPUT_5_reshaped, _M2_reshaped.T, c=_M3_reshaped)
    _M3              = _M3_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 lU,lW->UWl 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_0        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 Rjl,UWl->RjUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_3        
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3.shape[0]
    _size_dim_1      = _size_dim_1 * _M3.shape[1]
    _M3_reshaped = _M3.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M3_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 RjUW->RUWj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M7_perm_offset  = offset_0        
    _M7_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M7_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M7.ctypes.data),
                                 ctypes.c_void_p(_M7_perm.ctypes.data),
                                 ctypes.c_int(_M7.shape[0]),
                                 ctypes.c_int(_M7.shape[1]),
                                 ctypes.c_int(_M7.shape[2]),
                                 ctypes.c_int(_M7.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M13_offset      = offset_2        
    _M13             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 RUWj,QVj->RUWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M14_offset      = offset_3        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M7_perm.shape[2]
    _M7_perm_reshaped = _M7_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M13.shape[0]
    _size_dim_1      = _size_dim_1 * _M13.shape[1]
    _M13_reshaped = _M13.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M14.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _size_dim_1      = _size_dim_1 * _M14.shape[1]
    _size_dim_1      = _size_dim_1 * _M14.shape[2]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    lib.ddot(_M7_perm_reshaped, _M13_reshaped.T, c=_M14_reshaped)
    _M14             = _M14_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RUQVW,RUWQV->RUQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01423_01234_wob = getattr(libpbc, "fn_contraction_01234_01423_01234_wob", None)
    assert fn_contraction_01234_01423_01234_wob is not None
    _M15_offset      = offset_0        
    _M15             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M15_offset)
    fn_contraction_01234_01423_01234_wob(ctypes.c_void_p(_M12.ctypes.data),
                                         ctypes.c_void_p(_M14.ctypes.data),
                                         ctypes.c_void_p(_M15.ctypes.data),
                                         ctypes.c_int(_M12.shape[0]),
                                         ctypes.c_int(_M12.shape[1]),
                                         ctypes.c_int(_M12.shape[2]),
                                         ctypes.c_int(_M12.shape[3]),
                                         ctypes.c_int(_M12.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M16_offset      = offset_1        
    _M16             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M16_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M16.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 bU,VWb->UVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M17_offset      = offset_2        
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M17_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_14.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M17.ctypes.data),
                                   ctypes.c_int(_INPUT_14.shape[0]),
                                   ctypes.c_int(_INPUT_14.shape[1]),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 iP,bP->ibP 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                 ctypes.c_void_p(_INPUT_2.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_1.shape[0]),
                                 ctypes.c_int(_INPUT_1.shape[1]),
                                 ctypes.c_int(_INPUT_2.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 PQ,ibP->Qib 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = offset_3        
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
    _benchmark_time(t1, t2, "step 20")
    # step 20 Qib,UVWb->QiUVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M18_offset      = offset_1        
    _M18             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _size_dim_1      = _size_dim_1 * _M17.shape[2]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M18.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M18.shape[0]
    _size_dim_1      = _size_dim_1 * _M18.shape[1]
    _M18_reshaped = _M18.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M17_reshaped.T, c=_M18_reshaped)
    _M18             = _M18_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 iV,QiUVW->QUWiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20314_23401_wob = getattr(libpbc, "fn_contraction_01_20314_23401_wob", None)
    assert fn_contraction_01_20314_23401_wob is not None
    _M19_offset      = offset_2        
    _M19             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_20314_23401_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_void_p(_M19.ctypes.data),
                                      ctypes.c_int(_INPUT_15.shape[0]),
                                      ctypes.c_int(_INPUT_15.shape[1]),
                                      ctypes.c_int(_M18.shape[0]),
                                      ctypes.c_int(_M18.shape[2]),
                                      ctypes.c_int(_M18.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 QUWiV->QUWVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_01243_wob = getattr(libpbc, "fn_permutation_01234_01243_wob", None)
    assert fn_permutation_01234_01243_wob is not None
    _M19_perm_offset = offset_1        
    _M19_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M19_perm_offset)
    fn_permutation_01234_01243_wob(ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_void_p(_M19_perm.ctypes.data),
                                   ctypes.c_int(_M19.shape[0]),
                                   ctypes.c_int(_M19.shape[1]),
                                   ctypes.c_int(_M19.shape[2]),
                                   ctypes.c_int(_M19.shape[3]),
                                   ctypes.c_int(_M19.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 iR,QUWVi->RQUWV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M20_offset      = offset_2        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_6.shape[0]
    _INPUT_6_reshaped = _INPUT_6.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[2]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[3]
    _M19_perm_reshaped = _M19_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_6_reshaped.T, _M19_perm_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 RQUWV->RUQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_02143_wob = getattr(libpbc, "fn_permutation_01234_02143_wob", None)
    assert fn_permutation_01234_02143_wob is not None
    _M20_perm_offset = offset_1        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_01234_02143_wob(ctypes.c_void_p(_M20.ctypes.data),
                                   ctypes.c_void_p(_M20_perm.ctypes.data),
                                   ctypes.c_int(_M20.shape[0]),
                                   ctypes.c_int(_M20.shape[1]),
                                   ctypes.c_int(_M20.shape[2]),
                                   ctypes.c_int(_M20.shape[3]),
                                   ctypes.c_int(_M20.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 RUQVW,RUQVW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M15.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M15.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    return _M21

def RMP3_XX_7_naive(Z           : np.ndarray,
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
    _M0              = np.einsum("iP,aP->iaP"    , _INPUT_1        , _INPUT_2        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,iaP->Qia"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,aV->SVa"    , _INPUT_8        , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("Qia,SVa->QiSV" , _M1             , _M11            )
    del _M1         
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("iV,QiSV->QSiV" , _INPUT_15       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13_perm        = np.transpose(_M13            , (1, 0, 3, 2)    )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("iU,cU->icU"    , _INPUT_13       , _INPUT_14       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,icU->Tic"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("Tic,SWc->TiSW" , _M5             , _M6             )
    del _M5         
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("iW,TiSW->TSiW" , _INPUT_19       , _M7             )
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8_perm         = np.transpose(_M8             , (1, 0, 3, 2)    )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("STWi,SQVi->STWQV", _M8_perm        , _M13_perm       )
    del _M8_perm    
    del _M13_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14_perm        = np.transpose(_M14            , (1, 2, 3, 4, 0) )
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jR,kR->jkR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,jkR->Sjk"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3_perm         = np.transpose(_M3             , (0, 2, 1)       )
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("jQ,jV->QVj"    , _INPUT_3        , _INPUT_16       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("Skj,QVj->SkQV" , _M3_perm        , _M9             )
    del _M3_perm    
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10_perm        = np.transpose(_M10            , (0, 2, 3, 1)    )
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M15             = np.einsum("kT,kW->TWk"    , _INPUT_11       , _INPUT_20       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("SQVk,TWk->SQVTW", _M10_perm       , _M15            )
    del _M10_perm   
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16_perm        = np.transpose(_M16            , (3, 4, 1, 2, 0) )
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("TWQVS,TWQVS->TWQV", _M14_perm       , _M16_perm       )
    del _M14_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("bV,bW->VWb"    , _INPUT_18       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("bQ,VWb->QVWb"  , _INPUT_4        , _M18            )
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("bT,QVWb->TQVW" , _INPUT_12       , _M19            )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (0, 3, 1, 2)    )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("TWQV,TWQV->"   , _M17            , _M20_perm       )
    del _M17        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_XX_7(Z           : np.ndarray,
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
    # step 2 aS,aV->SVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qia,SVa->QiSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M1         
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiSV->QSiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 QSiV->SQVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M13_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_void_p(_M13_perm.ctypes.data),
                                 ctypes.c_int(_M13.shape[0]),
                                 ctypes.c_int(_M13.shape[1]),
                                 ctypes.c_int(_M13.shape[2]),
                                 ctypes.c_int(_M13.shape[3]))
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iU,cU->icU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TU,icU->Tic 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 Tic,SWc->TiSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M5         
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iW,TiSW->TSiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M7.ctypes.data),
                                    ctypes.c_void_p(_M8.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M7.shape[0]),
                                    ctypes.c_int(_M7.shape[2]))
    del _M7         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TSiW->STWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 STWi,SQVi->STWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0453_01245_wob = getattr(libpbc, "fn_contraction_0123_0453_01245_wob", None)
    assert fn_contraction_0123_0453_01245_wob is not None
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_0453_01245_wob(ctypes.c_void_p(_M8_perm.ctypes.data),
                                       ctypes.c_void_p(_M13_perm.ctypes.data),
                                       ctypes.c_void_p(_M14.ctypes.data),
                                       ctypes.c_int(_M8_perm.shape[0]),
                                       ctypes.c_int(_M8_perm.shape[1]),
                                       ctypes.c_int(_M8_perm.shape[2]),
                                       ctypes.c_int(_M8_perm.shape[3]),
                                       ctypes.c_int(_M13_perm.shape[1]),
                                       ctypes.c_int(_M13_perm.shape[2]))
    del _M8_perm    
    del _M13_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 STWQV->TWQVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12340_wob = getattr(libpbc, "fn_permutation_01234_12340_wob", None)
    assert fn_permutation_01234_12340_wob is not None
    _M14_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_12340_wob(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_perm.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(_M14.shape[3]),
                                   ctypes.c_int(_M14.shape[4]))
    del _M14        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 Sjk->Skj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M3_perm         = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_perm.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]))
    del _M3         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 jQ,jV->QVj 
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
    _benchmark_time(t1, t2, "step 18")
    # step 18 Skj,QVj->SkQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[1]
    _M3_perm_reshaped = _M3_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M3_perm_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M3_perm    
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 SkQV->SQVk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M10_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_perm.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(_M10.shape[3]))
    del _M10        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 kT,kW->TWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SQVk,TWk->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[2]
    _M10_perm_reshaped = _M10_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_M10_perm_reshaped, _M15_reshaped.T, c=_M16_reshaped)
    _M16         = _M16_reshaped.reshape(*shape_backup)
    del _M10_perm   
    del _M15        
    del _M15_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 SQVTW->TWQVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34120_wob = getattr(libpbc, "fn_permutation_01234_34120_wob", None)
    assert fn_permutation_01234_34120_wob is not None
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), dtype=np.float64)
    fn_permutation_01234_34120_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 TWQVS,TWQVS->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_0123_wob = getattr(libpbc, "fn_contraction_01234_01234_0123_wob", None)
    assert fn_contraction_01234_01234_0123_wob is not None
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_01234_01234_0123_wob(ctypes.c_void_p(_M14_perm.ctypes.data),
                                        ctypes.c_void_p(_M16_perm.ctypes.data),
                                        ctypes.c_void_p(_M17.ctypes.data),
                                        ctypes.c_int(_M14_perm.shape[0]),
                                        ctypes.c_int(_M14_perm.shape[1]),
                                        ctypes.c_int(_M14_perm.shape[2]),
                                        ctypes.c_int(_M14_perm.shape[3]),
                                        ctypes.c_int(_M14_perm.shape[4]))
    del _M14_perm   
    del _M16_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 bQ,VWb->QVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]))
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 bT,QVWb->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M19        
    del _M19_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 TQVW->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    # step 28 TWQV,TWQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M17.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M17.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M17        
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_XX_7_determine_bucket_size(NVIR        : int,
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
    _INPUT_7_size    = (NOCC * NTHC_INT)
    _INPUT_8_size    = (NVIR * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NOCC * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NOCC))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M7_size         = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M8_size         = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M9_size         = (NTHC_INT * (N_LAPLACE * NOCC))
    _M10_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M12_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M13_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M14_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M15_size        = (NTHC_INT * (N_LAPLACE * NOCC))
    _M16_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M17_size        = (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE)))
    _M18_size        = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M19_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M20_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M13_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M8_perm_size    = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M14_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M3_perm_size    = (NTHC_INT * (NOCC * NOCC))
    _M10_perm_size   = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M16_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * N_LAPLACE))))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M3_perm_size    = _M3_size        
    _M10_perm_size   = _M10_size       
    _M16_perm_size   = _M16_size       
    _M13_perm_size   = _M13_size       
    _M14_perm_size   = _M14_size       
    _M20_perm_size   = _M20_size       
    _M8_perm_size    = _M8_size        
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M0_size)
    bucked_0_size    = max(bucked_0_size, _M11_size)
    bucked_0_size    = max(bucked_0_size, _M13_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M6_size)
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M14_size)
    bucked_0_size    = max(bucked_0_size, _M2_size)
    bucked_0_size    = max(bucked_0_size, _M3_perm_size)
    bucked_0_size    = max(bucked_0_size, _M10_perm_size)
    bucked_0_size    = max(bucked_0_size, _M16_perm_size)
    bucked_0_size    = max(bucked_0_size, _M18_size)
    bucked_0_size    = max(bucked_0_size, _M20_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M1_size)
    bucked_1_size    = max(bucked_1_size, _M13_perm_size)
    bucked_1_size    = max(bucked_1_size, _M14_perm_size)
    bucked_1_size    = max(bucked_1_size, _M19_size)
    bucked_1_size    = max(bucked_1_size, _M20_perm_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M12_size)
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M8_perm_size)
    bucked_2_size    = max(bucked_2_size, _M3_size)
    bucked_2_size    = max(bucked_2_size, _M9_size)
    bucked_2_size    = max(bucked_2_size, _M15_size)
    bucked_2_size    = max(bucked_2_size, _M17_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M7_size)
    bucked_3_size    = max(bucked_3_size, _M10_size)
    bucked_3_size    = max(bucked_3_size, _M16_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_7_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_XX_7_determine_bucket_size(NVIR = NVIR,
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
    # step 2 aS,aV->SVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11_offset      = offset_0        
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 Qia,SVa->QiSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M12_offset      = offset_2        
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 iV,QiSV->QSiV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13_offset      = offset_0        
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_15.shape[0]),
                                    ctypes.c_int(_INPUT_15.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 QSiV->SQVi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M13_perm_offset = offset_1        
    _M13_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M13_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_void_p(_M13_perm.ctypes.data),
                                 ctypes.c_int(_M13.shape[0]),
                                 ctypes.c_int(_M13.shape[1]),
                                 ctypes.c_int(_M13.shape[2]),
                                 ctypes.c_int(_M13.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iU,cU->icU 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_0        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_13.ctypes.data),
                                 ctypes.c_void_p(_INPUT_14.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_13.shape[0]),
                                 ctypes.c_int(_INPUT_13.shape[1]),
                                 ctypes.c_int(_INPUT_14.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 TU,icU->Tic 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_0        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 Tic,SWc->TiSW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M7_offset       = offset_3        
    _M7              = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _size_dim_1      = _size_dim_1 * _M7.shape[1]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 iW,TiSW->TSiW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_19.ctypes.data),
                                    ctypes.c_void_p(_M7.ctypes.data),
                                    ctypes.c_void_p(_M8.ctypes.data),
                                    ctypes.c_int(_INPUT_19.shape[0]),
                                    ctypes.c_int(_INPUT_19.shape[1]),
                                    ctypes.c_int(_M7.shape[0]),
                                    ctypes.c_int(_M7.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 TSiW->STWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M8_perm_offset  = offset_2        
    _M8_perm         = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M8_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_void_p(_M8_perm.ctypes.data),
                                 ctypes.c_int(_M8.shape[0]),
                                 ctypes.c_int(_M8.shape[1]),
                                 ctypes.c_int(_M8.shape[2]),
                                 ctypes.c_int(_M8.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 STWi,SQVi->STWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_0453_01245_wob = getattr(libpbc, "fn_contraction_0123_0453_01245_wob", None)
    assert fn_contraction_0123_0453_01245_wob is not None
    _M14_offset      = offset_0        
    _M14             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    fn_contraction_0123_0453_01245_wob(ctypes.c_void_p(_M8_perm.ctypes.data),
                                       ctypes.c_void_p(_M13_perm.ctypes.data),
                                       ctypes.c_void_p(_M14.ctypes.data),
                                       ctypes.c_int(_M8_perm.shape[0]),
                                       ctypes.c_int(_M8_perm.shape[1]),
                                       ctypes.c_int(_M8_perm.shape[2]),
                                       ctypes.c_int(_M8_perm.shape[3]),
                                       ctypes.c_int(_M13_perm.shape[1]),
                                       ctypes.c_int(_M13_perm.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 STWQV->TWQVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_12340_wob = getattr(libpbc, "fn_permutation_01234_12340_wob", None)
    assert fn_permutation_01234_12340_wob is not None
    _M14_perm_offset = offset_1        
    _M14_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M14_perm_offset)
    fn_permutation_01234_12340_wob(ctypes.c_void_p(_M14.ctypes.data),
                                   ctypes.c_void_p(_M14_perm.ctypes.data),
                                   ctypes.c_int(_M14.shape[0]),
                                   ctypes.c_int(_M14.shape[1]),
                                   ctypes.c_int(_M14.shape[2]),
                                   ctypes.c_int(_M14.shape[3]),
                                   ctypes.c_int(_M14.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_0        
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NOCC * _itemsize)))
    _M3_offset       = offset_2        
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M3_offset)
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
    _benchmark_time(t1, t2, "step 16")
    # step 16 Sjk->Skj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_021_wob = getattr(libpbc, "fn_permutation_012_021_wob", None)
    assert fn_permutation_012_021_wob is not None
    _M3_perm_offset  = offset_0        
    _M3_perm         = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M3_perm_offset)
    fn_permutation_012_021_wob(ctypes.c_void_p(_M3.ctypes.data),
                               ctypes.c_void_p(_M3_perm.ctypes.data),
                               ctypes.c_int(_M3.shape[0]),
                               ctypes.c_int(_M3.shape[1]),
                               ctypes.c_int(_M3.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 jQ,jV->QVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M9_offset       = offset_2        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_16.ctypes.data),
                                 ctypes.c_void_p(_M9.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_16.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 Skj,QVj->SkQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M10_offset      = offset_3        
    _M10             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M3_perm.shape[1]
    _M3_perm_reshaped = _M3_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _size_dim_1      = _size_dim_1 * _M10.shape[1]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_M3_perm_reshaped, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    # step 19 SkQV->SQVk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0231_wob = getattr(libpbc, "fn_permutation_0123_0231_wob", None)
    assert fn_permutation_0123_0231_wob is not None
    _M10_perm_offset = offset_0        
    _M10_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M10_perm_offset)
    fn_permutation_0123_0231_wob(ctypes.c_void_p(_M10.ctypes.data),
                                 ctypes.c_void_p(_M10_perm.ctypes.data),
                                 ctypes.c_int(_M10.shape[0]),
                                 ctypes.c_int(_M10.shape[1]),
                                 ctypes.c_int(_M10.shape[2]),
                                 ctypes.c_int(_M10.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 kT,kW->TWk 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15_offset      = offset_2        
    _M15             = np.ndarray((NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_20.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_20.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SQVk,TWk->SQVTW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NTHC_INT * (N_LAPLACE * _itemsize)))))
    _M16_offset      = offset_3        
    _M16             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M16_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[1]
    _size_dim_1      = _size_dim_1 * _M10_perm.shape[2]
    _M10_perm_reshaped = _M10_perm.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M15.shape[0]
    _size_dim_1      = _size_dim_1 * _M15.shape[1]
    _M15_reshaped = _M15.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M16.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    lib.ddot(_M10_perm_reshaped, _M15_reshaped.T, c=_M16_reshaped)
    _M16             = _M16_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 SQVTW->TWQVS 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_34120_wob = getattr(libpbc, "fn_permutation_01234_34120_wob", None)
    assert fn_permutation_01234_34120_wob is not None
    _M16_perm_offset = offset_0        
    _M16_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE, NTHC_INT), buffer = buffer, offset = _M16_perm_offset)
    fn_permutation_01234_34120_wob(ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_void_p(_M16_perm.ctypes.data),
                                   ctypes.c_int(_M16.shape[0]),
                                   ctypes.c_int(_M16.shape[1]),
                                   ctypes.c_int(_M16.shape[2]),
                                   ctypes.c_int(_M16.shape[3]),
                                   ctypes.c_int(_M16.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 TWQVS,TWQVS->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01234_01234_0123_wob = getattr(libpbc, "fn_contraction_01234_01234_0123_wob", None)
    assert fn_contraction_01234_01234_0123_wob is not None
    _M17_offset      = offset_2        
    _M17             = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    fn_contraction_01234_01234_0123_wob(ctypes.c_void_p(_M14_perm.ctypes.data),
                                        ctypes.c_void_p(_M16_perm.ctypes.data),
                                        ctypes.c_void_p(_M17.ctypes.data),
                                        ctypes.c_int(_M14_perm.shape[0]),
                                        ctypes.c_int(_M14_perm.shape[1]),
                                        ctypes.c_int(_M14_perm.shape[2]),
                                        ctypes.c_int(_M14_perm.shape[3]),
                                        ctypes.c_int(_M14_perm.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M18_offset      = offset_0        
    _M18             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M18_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M18.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    # step 25 bQ,VWb->QVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M19_offset      = offset_1        
    _M19             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M19_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_4.ctypes.data),
                                   ctypes.c_void_p(_M18.ctypes.data),
                                   ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_int(_INPUT_4.shape[0]),
                                   ctypes.c_int(_INPUT_4.shape[1]),
                                   ctypes.c_int(_M18.shape[0]),
                                   ctypes.c_int(_M18.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 26")
    # step 26 bT,QVWb->TQVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M20_offset      = offset_0        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_12.shape[0]
    _INPUT_12_reshaped = _INPUT_12.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19.shape[0]
    _size_dim_1      = _size_dim_1 * _M19.shape[1]
    _size_dim_1      = _size_dim_1 * _M19.shape[2]
    _M19_reshaped = _M19.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_12_reshaped.T, _M19_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 27")
    # step 27 TQVW->TWQV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_0312_wob = getattr(libpbc, "fn_permutation_0123_0312_wob", None)
    assert fn_permutation_0123_0312_wob is not None
    _M20_perm_offset = offset_1        
    _M20_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_0123_0312_wob(ctypes.c_void_p(_M20.ctypes.data),
                                 ctypes.c_void_p(_M20_perm.ctypes.data),
                                 ctypes.c_int(_M20.shape[0]),
                                 ctypes.c_int(_M20.shape[1]),
                                 ctypes.c_int(_M20.shape[2]),
                                 ctypes.c_int(_M20.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 28")
    # step 28 TWQV,TWQV-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M17.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M17.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 29")
    return _M21

def RMP3_XX_8_naive(Z           : np.ndarray,
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
    _M15             = np.einsum("bV,bW->VWb"    , _INPUT_18       , _INPUT_21       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M16             = np.einsum("bP,VWb->PVWb"  , _INPUT_2        , _M15            )
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M4              = np.einsum("kT,bT->kbT"    , _INPUT_11       , _INPUT_12       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.einsum("TU,kbT->Ukb"   , _INPUT_10       , _M4             )
    del _M4         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.einsum("Ukb,PVWb->UkPVW", _M5             , _M16            )
    del _M5         
    del _M16        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M18             = np.einsum("kW,UkPVW->UPVkW", _INPUT_20       , _M17            )
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M8              = np.einsum("iV,iW->VWi"    , _INPUT_15       , _INPUT_19       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M9              = np.einsum("iP,VWi->PVWi"  , _INPUT_1        , _M8             )
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.einsum("iU,PVWi->UPVW" , _INPUT_13       , _M9             )
    del _M9         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19             = np.einsum("UPVW,UPVkW->kUPVW", _M10            , _M18            )
    del _M10        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M19_perm        = np.transpose(_M19            , (1, 4, 0, 2, 3) )
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M0              = np.einsum("jQ,aQ->jaQ"    , _INPUT_3        , _INPUT_4        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M1              = np.einsum("PQ,jaQ->Pja"   , _INPUT_0        , _M0             )
    del _M0         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M11             = np.einsum("aS,aV->SVa"    , _INPUT_8        , _INPUT_17       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.einsum("Pja,SVa->PjSV" , _M1             , _M11            )
    del _M1         
    del _M11        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13             = np.einsum("jV,PjSV->PSjV" , _INPUT_16       , _M12            )
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M13_perm        = np.transpose(_M13            , (1, 0, 3, 2)    )
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M2              = np.einsum("jR,kR->jkR"    , _INPUT_6        , _INPUT_7        )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.einsum("RS,jkR->Sjk"   , _INPUT_5        , _M2             )
    del _M2         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 19")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M14             = np.einsum("Sjk,SPVj->SkPV", _M3             , _M13_perm       )
    del _M3         
    del _M13_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.einsum("SkPV,UWkPV->SUW", _M14            , _M19_perm       )
    del _M14        
    del _M19_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20_perm        = np.transpose(_M20            , (1, 0, 2)       )
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M6              = np.einsum("cS,cW->SWc"    , _INPUT_9        , _INPUT_22       )
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.einsum("cU,SWc->USW"   , _INPUT_14       , _M6             )
    del _M6         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    t1 = (logger.process_clock(), logger.perf_counter())
    _M21             = np.einsum("USW,USW->"     , _M7             , _M20_perm       )
    del _M7         
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_XX_8(Z           : np.ndarray,
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
    # step 0 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bP,VWb->PVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M16             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]))
    del _M15        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 kT,bT->kbT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 TU,kbT->Ukb 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5          = _M5_reshaped.reshape(*shape_backup)
    del _M4         
    del _M4_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 Ukb,PVWb->UkPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M17             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M16_reshaped.T, c=_M17_reshaped)
    _M17         = _M17_reshaped.reshape(*shape_backup)
    del _M5         
    del _M16        
    del _M16_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kW,UkPVW->UPVkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20341_23401_wob = getattr(libpbc, "fn_contraction_01_20341_23401_wob", None)
    assert fn_contraction_01_20341_23401_wob is not None
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_20341_23401_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_int(_INPUT_20.shape[0]),
                                      ctypes.c_int(_INPUT_20.shape[1]),
                                      ctypes.c_int(_M17.shape[0]),
                                      ctypes.c_int(_M17.shape[2]),
                                      ctypes.c_int(_M17.shape[3]))
    del _M17        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), dtype=np.float64)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    del _M8         
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iU,PVWi->UPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _size_dim_1      = _size_dim_1 * _M9.shape[2]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10         = _M10_reshaped.reshape(*shape_backup)
    del _M9         
    del _M9_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 UPVW,UPVkW->kUPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01243_40123_wob = getattr(libpbc, "fn_contraction_0123_01243_40123_wob", None)
    assert fn_contraction_0123_01243_40123_wob is not None
    _M19             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), dtype=np.float64)
    fn_contraction_0123_01243_40123_wob(ctypes.c_void_p(_M10.ctypes.data),
                                        ctypes.c_void_p(_M18.ctypes.data),
                                        ctypes.c_void_p(_M19.ctypes.data),
                                        ctypes.c_int(_M10.shape[0]),
                                        ctypes.c_int(_M10.shape[1]),
                                        ctypes.c_int(_M10.shape[2]),
                                        ctypes.c_int(_M10.shape[3]),
                                        ctypes.c_int(_M18.shape[3]))
    del _M10        
    del _M18        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 kUPVW->UWkPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_14023_wob = getattr(libpbc, "fn_permutation_01234_14023_wob", None)
    assert fn_permutation_01234_14023_wob is not None
    _M19_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_01234_14023_wob(ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_void_p(_M19_perm.ctypes.data),
                                   ctypes.c_int(_M19.shape[0]),
                                   ctypes.c_int(_M19.shape[1]),
                                   ctypes.c_int(_M19.shape[2]),
                                   ctypes.c_int(_M19.shape[3]),
                                   ctypes.c_int(_M19.shape[4]))
    del _M19        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jQ,aQ->jaQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_4.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 PQ,jaQ->Pja 
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
    lib.ddot(_INPUT_0_reshaped, _M0_reshaped.T, c=_M1_reshaped)
    _M1          = _M1_reshaped.reshape(*shape_backup)
    del _M0         
    del _M0_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 aS,aV->SVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 Pja,SVa->PjSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12         = _M12_reshaped.reshape(*shape_backup)
    del _M1         
    del _M11        
    del _M11_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 jV,PjSV->PSjV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), dtype=np.float64)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_16.shape[0]),
                                    ctypes.c_int(_INPUT_16.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    del _M12        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 PSjV->SPVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M13_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), dtype=np.float64)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_void_p(_M13_perm.ctypes.data),
                                 ctypes.c_int(_M13.shape[0]),
                                 ctypes.c_int(_M13.shape[1]),
                                 ctypes.c_int(_M13.shape[2]),
                                 ctypes.c_int(_M13.shape[3]))
    del _M13        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), dtype=np.float64)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), dtype=np.float64)
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 Sjk,SPVj->SkPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0341_0234_wob = getattr(libpbc, "fn_contraction_012_0341_0234_wob", None)
    assert fn_contraction_012_0341_0234_wob is not None
    _M14             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_contraction_012_0341_0234_wob(ctypes.c_void_p(_M3.ctypes.data),
                                     ctypes.c_void_p(_M13_perm.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_int(_M3.shape[0]),
                                     ctypes.c_int(_M3.shape[1]),
                                     ctypes.c_int(_M3.shape[2]),
                                     ctypes.c_int(_M13_perm.shape[1]),
                                     ctypes.c_int(_M13_perm.shape[2]))
    del _M3         
    del _M13_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 SkPV,UWkPV->SUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[1]
    _M19_perm_reshaped = _M19_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M14_reshaped, _M19_perm_reshaped.T, c=_M20_reshaped)
    _M20         = _M20_reshaped.reshape(*shape_backup)
    del _M14        
    del _M19_perm   
    del _M19_perm_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SUW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_102_wob = getattr(libpbc, "fn_permutation_012_102_wob", None)
    assert fn_permutation_012_102_wob is not None
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    fn_permutation_012_102_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    del _M20        
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), dtype=np.float64)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), dtype=np.float64)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M6_reshaped.T, c=_M7_reshaped)
    _M7          = _M7_reshaped.reshape(*shape_backup)
    del _M6         
    del _M6_reshaped
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 USW,USW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M7.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M7.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    del _M7         
    del _M20_perm   
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
    return _M21

def RMP3_XX_8_determine_bucket_size(NVIR        : int,
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
    _INPUT_7_size    = (NOCC * NTHC_INT)
    _INPUT_8_size    = (NVIR * NTHC_INT)
    _INPUT_9_size    = (NVIR * NTHC_INT)
    _INPUT_10_size   = (NTHC_INT * NTHC_INT)
    _INPUT_11_size   = (NOCC * NTHC_INT)
    _INPUT_12_size   = (NVIR * NTHC_INT)
    _INPUT_13_size   = (NOCC * NTHC_INT)
    _INPUT_14_size   = (NVIR * NTHC_INT)
    _INPUT_15_size   = (NOCC * N_LAPLACE)
    _INPUT_16_size   = (NOCC * N_LAPLACE)
    _INPUT_17_size   = (NVIR * N_LAPLACE)
    _INPUT_18_size   = (NVIR * N_LAPLACE)
    _INPUT_19_size   = (NOCC * N_LAPLACE)
    _INPUT_20_size   = (NOCC * N_LAPLACE)
    _INPUT_21_size   = (NVIR * N_LAPLACE)
    _INPUT_22_size   = (NVIR * N_LAPLACE)
    _M0_size         = (NOCC * (NVIR * NTHC_INT))
    _M1_size         = (NTHC_INT * (NOCC * NVIR))
    _M2_size         = (NOCC * (NOCC * NTHC_INT))
    _M3_size         = (NTHC_INT * (NOCC * NOCC))
    _M4_size         = (NOCC * (NVIR * NTHC_INT))
    _M5_size         = (NTHC_INT * (NOCC * NVIR))
    _M6_size         = (NTHC_INT * (N_LAPLACE * NVIR))
    _M7_size         = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M8_size         = (N_LAPLACE * (N_LAPLACE * NOCC))
    _M9_size         = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NOCC)))
    _M10_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE)))
    _M11_size        = (NTHC_INT * (N_LAPLACE * NVIR))
    _M12_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M13_size        = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M14_size        = (NTHC_INT * (NOCC * (NTHC_INT * N_LAPLACE)))
    _M15_size        = (N_LAPLACE * (N_LAPLACE * NVIR))
    _M16_size        = (NTHC_INT * (N_LAPLACE * (N_LAPLACE * NVIR)))
    _M17_size        = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M18_size        = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (NOCC * N_LAPLACE))))
    _M19_size        = (NOCC * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M20_size        = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M19_perm_size   = (NOCC * (NTHC_INT * (NTHC_INT * (N_LAPLACE * N_LAPLACE))))
    _M13_perm_size   = (NTHC_INT * (NTHC_INT * (NOCC * N_LAPLACE)))
    _M20_perm_size   = (NTHC_INT * (NTHC_INT * N_LAPLACE))
    _M19_perm_size   = _M19_size       
    _M20_perm_size   = _M20_size       
    _M13_perm_size   = _M13_size       
    # determine the size of each bucket
    # bucket 0
    bucked_0_size    = max(bucked_0_size, _M15_size)
    bucked_0_size    = max(bucked_0_size, _M4_size)
    bucked_0_size    = max(bucked_0_size, _M17_size)
    bucked_0_size    = max(bucked_0_size, _M8_size)
    bucked_0_size    = max(bucked_0_size, _M10_size)
    bucked_0_size    = max(bucked_0_size, _M19_perm_size)
    bucked_0_size    = max(bucked_0_size, _M20_perm_size)
    # bucket 1
    bucked_1_size    = max(bucked_1_size, _M16_size)
    bucked_1_size    = max(bucked_1_size, _M18_size)
    bucked_1_size    = max(bucked_1_size, _M0_size)
    bucked_1_size    = max(bucked_1_size, _M11_size)
    bucked_1_size    = max(bucked_1_size, _M13_size)
    bucked_1_size    = max(bucked_1_size, _M2_size)
    bucked_1_size    = max(bucked_1_size, _M14_size)
    bucked_1_size    = max(bucked_1_size, _M6_size)
    # bucket 2
    bucked_2_size    = max(bucked_2_size, _M5_size)
    bucked_2_size    = max(bucked_2_size, _M9_size)
    bucked_2_size    = max(bucked_2_size, _M19_size)
    bucked_2_size    = max(bucked_2_size, _M1_size)
    bucked_2_size    = max(bucked_2_size, _M13_perm_size)
    bucked_2_size    = max(bucked_2_size, _M20_size)
    bucked_2_size    = max(bucked_2_size, _M7_size)
    # bucket 3
    bucked_3_size    = max(bucked_3_size, _M12_size)
    bucked_3_size    = max(bucked_3_size, _M3_size)
    # append each bucket size to the output
    output.append(bucked_0_size)
    output.append(bucked_1_size)
    output.append(bucked_2_size)
    output.append(bucked_3_size)
    return output

def RMP3_XX_8_opt_mem(Z           : np.ndarray,
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
    # deal with buffer
    bucket_size      = RMP3_XX_8_determine_bucket_size(NVIR = NVIR,
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
    # step 0 bV,bW->VWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M15_offset      = offset_0        
    _M15             = np.ndarray((N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M15_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_18.ctypes.data),
                                 ctypes.c_void_p(_INPUT_21.ctypes.data),
                                 ctypes.c_void_p(_M15.ctypes.data),
                                 ctypes.c_int(_INPUT_18.shape[0]),
                                 ctypes.c_int(_INPUT_18.shape[1]),
                                 ctypes.c_int(_INPUT_21.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 1")
    # step 1 bP,VWb->PVWb 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M16_offset      = offset_1        
    _M16             = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NVIR), buffer = buffer, offset = _M16_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_2.ctypes.data),
                                   ctypes.c_void_p(_M15.ctypes.data),
                                   ctypes.c_void_p(_M16.ctypes.data),
                                   ctypes.c_int(_INPUT_2.shape[0]),
                                   ctypes.c_int(_INPUT_2.shape[1]),
                                   ctypes.c_int(_M15.shape[0]),
                                   ctypes.c_int(_M15.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 2")
    # step 2 kT,bT->kbT 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M4_offset       = offset_0        
    _M4              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M4_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_11.ctypes.data),
                                 ctypes.c_void_p(_INPUT_12.ctypes.data),
                                 ctypes.c_void_p(_M4.ctypes.data),
                                 ctypes.c_int(_INPUT_11.shape[0]),
                                 ctypes.c_int(_INPUT_11.shape[1]),
                                 ctypes.c_int(_INPUT_12.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 3")
    # step 3 TU,kbT->Ukb 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M5_offset       = offset_2        
    _M5              = np.ndarray((NTHC_INT, NOCC, NVIR), buffer = buffer, offset = _M5_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_10.shape[0]
    _INPUT_10_reshaped = _INPUT_10.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M4.shape[0]
    _size_dim_1      = _size_dim_1 * _M4.shape[1]
    _M4_reshaped = _M4.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M5.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_10_reshaped.T, _M4_reshaped.T, c=_M5_reshaped)
    _M5              = _M5_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 4")
    # step 4 Ukb,PVWb->UkPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize)))))
    _M17_offset      = offset_0        
    _M17             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M17_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M5.shape[0]
    _size_dim_1      = _size_dim_1 * _M5.shape[1]
    _M5_reshaped = _M5.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M16.shape[0]
    _size_dim_1      = _size_dim_1 * _M16.shape[1]
    _size_dim_1      = _size_dim_1 * _M16.shape[2]
    _M16_reshaped = _M16.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M17.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M17.shape[0]
    _size_dim_1      = _size_dim_1 * _M17.shape[1]
    _M17_reshaped = _M17.reshape(_size_dim_1,-1)
    lib.ddot(_M5_reshaped, _M16_reshaped.T, c=_M17_reshaped)
    _M17             = _M17_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 5")
    # step 5 kW,UkPVW->UPVkW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_20341_23401_wob = getattr(libpbc, "fn_contraction_01_20341_23401_wob", None)
    assert fn_contraction_01_20341_23401_wob is not None
    _M18_offset      = offset_1        
    _M18             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC, N_LAPLACE), buffer = buffer, offset = _M18_offset)
    fn_contraction_01_20341_23401_wob(ctypes.c_void_p(_INPUT_20.ctypes.data),
                                      ctypes.c_void_p(_M17.ctypes.data),
                                      ctypes.c_void_p(_M18.ctypes.data),
                                      ctypes.c_int(_INPUT_20.shape[0]),
                                      ctypes.c_int(_INPUT_20.shape[1]),
                                      ctypes.c_int(_M17.shape[0]),
                                      ctypes.c_int(_M17.shape[2]),
                                      ctypes.c_int(_M17.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 6")
    # step 6 iV,iW->VWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M8_offset       = offset_0        
    _M8              = np.ndarray((N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M8_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_15.ctypes.data),
                                 ctypes.c_void_p(_INPUT_19.ctypes.data),
                                 ctypes.c_void_p(_M8.ctypes.data),
                                 ctypes.c_int(_INPUT_15.shape[0]),
                                 ctypes.c_int(_INPUT_15.shape[1]),
                                 ctypes.c_int(_INPUT_19.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 7")
    # step 7 iP,VWi->PVWi 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_230_1230_wob = getattr(libpbc, "fn_contraction_01_230_1230_wob", None)
    assert fn_contraction_01_230_1230_wob is not None
    _M9_offset       = offset_2        
    _M9              = np.ndarray((NTHC_INT, N_LAPLACE, N_LAPLACE, NOCC), buffer = buffer, offset = _M9_offset)
    fn_contraction_01_230_1230_wob(ctypes.c_void_p(_INPUT_1.ctypes.data),
                                   ctypes.c_void_p(_M8.ctypes.data),
                                   ctypes.c_void_p(_M9.ctypes.data),
                                   ctypes.c_int(_INPUT_1.shape[0]),
                                   ctypes.c_int(_INPUT_1.shape[1]),
                                   ctypes.c_int(_M8.shape[0]),
                                   ctypes.c_int(_M8.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 8")
    # step 8 iU,PVWi->UPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * (N_LAPLACE * _itemsize))))
    _M10_offset      = offset_0        
    _M10             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M10_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_13.shape[0]
    _INPUT_13_reshaped = _INPUT_13.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M9.shape[0]
    _size_dim_1      = _size_dim_1 * _M9.shape[1]
    _size_dim_1      = _size_dim_1 * _M9.shape[2]
    _M9_reshaped = _M9.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M10.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M10.shape[0]
    _M10_reshaped = _M10.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_13_reshaped.T, _M9_reshaped.T, c=_M10_reshaped)
    _M10             = _M10_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 9")
    # step 9 UPVW,UPVkW->kUPVW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_0123_01243_40123_wob = getattr(libpbc, "fn_contraction_0123_01243_40123_wob", None)
    assert fn_contraction_0123_01243_40123_wob is not None
    _M19_offset      = offset_2        
    _M19             = np.ndarray((NOCC, NTHC_INT, NTHC_INT, N_LAPLACE, N_LAPLACE), buffer = buffer, offset = _M19_offset)
    fn_contraction_0123_01243_40123_wob(ctypes.c_void_p(_M10.ctypes.data),
                                        ctypes.c_void_p(_M18.ctypes.data),
                                        ctypes.c_void_p(_M19.ctypes.data),
                                        ctypes.c_int(_M10.shape[0]),
                                        ctypes.c_int(_M10.shape[1]),
                                        ctypes.c_int(_M10.shape[2]),
                                        ctypes.c_int(_M10.shape[3]),
                                        ctypes.c_int(_M18.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 10")
    # step 10 kUPVW->UWkPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_01234_14023_wob = getattr(libpbc, "fn_permutation_01234_14023_wob", None)
    assert fn_permutation_01234_14023_wob is not None
    _M19_perm_offset = offset_0        
    _M19_perm        = np.ndarray((NTHC_INT, N_LAPLACE, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M19_perm_offset)
    fn_permutation_01234_14023_wob(ctypes.c_void_p(_M19.ctypes.data),
                                   ctypes.c_void_p(_M19_perm.ctypes.data),
                                   ctypes.c_int(_M19.shape[0]),
                                   ctypes.c_int(_M19.shape[1]),
                                   ctypes.c_int(_M19.shape[2]),
                                   ctypes.c_int(_M19.shape[3]),
                                   ctypes.c_int(_M19.shape[4]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 11")
    # step 11 jQ,aQ->jaQ 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M0_offset       = offset_1        
    _M0              = np.ndarray((NOCC, NVIR, NTHC_INT), buffer = buffer, offset = _M0_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_3.ctypes.data),
                                 ctypes.c_void_p(_INPUT_4.ctypes.data),
                                 ctypes.c_void_p(_M0.ctypes.data),
                                 ctypes.c_int(_INPUT_3.shape[0]),
                                 ctypes.c_int(_INPUT_3.shape[1]),
                                 ctypes.c_int(_INPUT_4.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 12")
    # step 12 PQ,jaQ->Pja 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NVIR * _itemsize)))
    _M1_offset       = offset_2        
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
    lib.ddot(_INPUT_0_reshaped, _M0_reshaped.T, c=_M1_reshaped)
    _M1              = _M1_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 13")
    # step 13 aS,aV->SVa 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M11_offset      = offset_1        
    _M11             = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M11_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_8.ctypes.data),
                                 ctypes.c_void_p(_INPUT_17.ctypes.data),
                                 ctypes.c_void_p(_M11.ctypes.data),
                                 ctypes.c_int(_INPUT_8.shape[0]),
                                 ctypes.c_int(_INPUT_8.shape[1]),
                                 ctypes.c_int(_INPUT_17.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 14")
    # step 14 Pja,SVa->PjSV 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NTHC_INT * (N_LAPLACE * _itemsize))))
    _M12_offset      = offset_3        
    _M12             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M12_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M1.shape[0]
    _size_dim_1      = _size_dim_1 * _M1.shape[1]
    _M1_reshaped = _M1.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M11.shape[0]
    _size_dim_1      = _size_dim_1 * _M11.shape[1]
    _M11_reshaped = _M11.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M12.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M12.shape[0]
    _size_dim_1      = _size_dim_1 * _M12.shape[1]
    _M12_reshaped = _M12.reshape(_size_dim_1,-1)
    lib.ddot(_M1_reshaped, _M11_reshaped.T, c=_M12_reshaped)
    _M12             = _M12_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 15")
    # step 15 jV,PjSV->PSjV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_2031_2301_wob = getattr(libpbc, "fn_contraction_01_2031_2301_wob", None)
    assert fn_contraction_01_2031_2301_wob is not None
    _M13_offset      = offset_1        
    _M13             = np.ndarray((NTHC_INT, NTHC_INT, NOCC, N_LAPLACE), buffer = buffer, offset = _M13_offset)
    fn_contraction_01_2031_2301_wob(ctypes.c_void_p(_INPUT_16.ctypes.data),
                                    ctypes.c_void_p(_M12.ctypes.data),
                                    ctypes.c_void_p(_M13.ctypes.data),
                                    ctypes.c_int(_INPUT_16.shape[0]),
                                    ctypes.c_int(_INPUT_16.shape[1]),
                                    ctypes.c_int(_M12.shape[0]),
                                    ctypes.c_int(_M12.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 16")
    # step 16 PSjV->SPVj 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_0123_1032_wob = getattr(libpbc, "fn_permutation_0123_1032_wob", None)
    assert fn_permutation_0123_1032_wob is not None
    _M13_perm_offset = offset_2        
    _M13_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE, NOCC), buffer = buffer, offset = _M13_perm_offset)
    fn_permutation_0123_1032_wob(ctypes.c_void_p(_M13.ctypes.data),
                                 ctypes.c_void_p(_M13_perm.ctypes.data),
                                 ctypes.c_int(_M13.shape[0]),
                                 ctypes.c_int(_M13.shape[1]),
                                 ctypes.c_int(_M13.shape[2]),
                                 ctypes.c_int(_M13.shape[3]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 17")
    # step 17 jR,kR->jkR 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_21_021_wob = getattr(libpbc, "fn_contraction_01_21_021_wob", None)
    assert fn_contraction_01_21_021_wob is not None
    _M2_offset       = offset_1        
    _M2              = np.ndarray((NOCC, NOCC, NTHC_INT), buffer = buffer, offset = _M2_offset)
    fn_contraction_01_21_021_wob(ctypes.c_void_p(_INPUT_6.ctypes.data),
                                 ctypes.c_void_p(_INPUT_7.ctypes.data),
                                 ctypes.c_void_p(_M2.ctypes.data),
                                 ctypes.c_int(_INPUT_6.shape[0]),
                                 ctypes.c_int(_INPUT_6.shape[1]),
                                 ctypes.c_int(_INPUT_7.shape[0]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 18")
    # step 18 RS,jkR->Sjk 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NOCC * (NOCC * _itemsize)))
    _M3_offset       = offset_3        
    _M3              = np.ndarray((NTHC_INT, NOCC, NOCC), buffer = buffer, offset = _M3_offset)
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
    _benchmark_time(t1, t2, "step 19")
    # step 19 Sjk,SPVj->SkPV 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_012_0341_0234_wob = getattr(libpbc, "fn_contraction_012_0341_0234_wob", None)
    assert fn_contraction_012_0341_0234_wob is not None
    _M14_offset      = offset_1        
    _M14             = np.ndarray((NTHC_INT, NOCC, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M14_offset)
    fn_contraction_012_0341_0234_wob(ctypes.c_void_p(_M3.ctypes.data),
                                     ctypes.c_void_p(_M13_perm.ctypes.data),
                                     ctypes.c_void_p(_M14.ctypes.data),
                                     ctypes.c_int(_M3.shape[0]),
                                     ctypes.c_int(_M3.shape[1]),
                                     ctypes.c_int(_M3.shape[2]),
                                     ctypes.c_int(_M13_perm.shape[1]),
                                     ctypes.c_int(_M13_perm.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 20")
    # step 20 SkPV,UWkPV->SUW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M20_offset      = offset_2        
    _M20             = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M14.shape[0]
    _M14_reshaped = _M14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[0]
    _size_dim_1      = _size_dim_1 * _M19_perm.shape[1]
    _M19_perm_reshaped = _M19_perm.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M20.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M20.shape[0]
    _M20_reshaped = _M20.reshape(_size_dim_1,-1)
    lib.ddot(_M14_reshaped, _M19_perm_reshaped.T, c=_M20_reshaped)
    _M20             = _M20_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 21")
    # step 21 SUW->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_permutation_012_102_wob = getattr(libpbc, "fn_permutation_012_102_wob", None)
    assert fn_permutation_012_102_wob is not None
    _M20_perm_offset = offset_0        
    _M20_perm        = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M20_perm_offset)
    fn_permutation_012_102_wob(ctypes.c_void_p(_M20.ctypes.data),
                               ctypes.c_void_p(_M20_perm.ctypes.data),
                               ctypes.c_int(_M20.shape[0]),
                               ctypes.c_int(_M20.shape[1]),
                               ctypes.c_int(_M20.shape[2]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 22")
    # step 22 cS,cW->SWc 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_contraction_01_02_120_wob = getattr(libpbc, "fn_contraction_01_02_120_wob", None)
    assert fn_contraction_01_02_120_wob is not None
    _M6_offset       = offset_1        
    _M6              = np.ndarray((NTHC_INT, N_LAPLACE, NVIR), buffer = buffer, offset = _M6_offset)
    fn_contraction_01_02_120_wob(ctypes.c_void_p(_INPUT_9.ctypes.data),
                                 ctypes.c_void_p(_INPUT_22.ctypes.data),
                                 ctypes.c_void_p(_M6.ctypes.data),
                                 ctypes.c_int(_INPUT_9.shape[0]),
                                 ctypes.c_int(_INPUT_9.shape[1]),
                                 ctypes.c_int(_INPUT_22.shape[1]))
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 23")
    # step 23 cU,SWc->USW 
    t1 = (logger.process_clock(), logger.perf_counter())
    tmp_itemsize     = (NTHC_INT * (NTHC_INT * (N_LAPLACE * _itemsize)))
    _M7_offset       = offset_2        
    _M7              = np.ndarray((NTHC_INT, NTHC_INT, N_LAPLACE), buffer = buffer, offset = _M7_offset)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _INPUT_14.shape[0]
    _INPUT_14_reshaped = _INPUT_14.reshape(_size_dim_1,-1)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M6.shape[0]
    _size_dim_1      = _size_dim_1 * _M6.shape[1]
    _M6_reshaped = _M6.reshape(_size_dim_1,-1)
    shape_backup = copy.deepcopy(_M7.shape)
    _size_dim_1      = 1               
    _size_dim_1      = _size_dim_1 * _M7.shape[0]
    _M7_reshaped = _M7.reshape(_size_dim_1,-1)
    lib.ddot(_INPUT_14_reshaped.T, _M6_reshaped.T, c=_M7_reshaped)
    _M7              = _M7_reshaped.reshape(*shape_backup)
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 24")
    # step 24 USW,USW-> 
    t1 = (logger.process_clock(), logger.perf_counter())
    fn_dot       = getattr(libpbc, "fn_dot", None)
    assert fn_dot is not None
    _M21             = ctypes.c_double(0.0)
    fn_dot(ctypes.c_void_p(_M7.ctypes.data),
           ctypes.c_void_p(_M20_perm.ctypes.data),
           ctypes.c_int(_M7.size),
           ctypes.pointer(_M21))
    _M21 = _M21.value
    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "step 25")
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
    output2          = RMP3_CC_opt_mem(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           ,
                                       buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_CX_1 and RMP3_CX_1_naive
    benchmark        = RMP3_CX_1_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_CX_1(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_CX_1_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_CX_2 and RMP3_CX_2_naive
    benchmark        = RMP3_CX_2_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_CX_2(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_CX_2_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_CX_3 and RMP3_CX_3_naive
    benchmark        = RMP3_CX_3_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_CX_3(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_CX_3_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_1 and RMP3_XX_1_naive
    benchmark        = RMP3_XX_1_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_1(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_1_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_2 and RMP3_XX_2_naive
    benchmark        = RMP3_XX_2_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_2(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_2_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_3 and RMP3_XX_3_naive
    benchmark        = RMP3_XX_3_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_3(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_3_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_4 and RMP3_XX_4_naive
    benchmark        = RMP3_XX_4_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_4(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_4_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_5 and RMP3_XX_5_naive
    benchmark        = RMP3_XX_5_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_5(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_5_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_6 and RMP3_XX_6_naive
    benchmark        = RMP3_XX_6_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_6(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_6_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_7 and RMP3_XX_7_naive
    benchmark        = RMP3_XX_7_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_7(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_7_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
    # test for RMP3_XX_8 and RMP3_XX_8_naive
    benchmark        = RMP3_XX_8_naive(Z               ,
                                       X_o             ,
                                       X_v             ,
                                       tau_o           ,
                                       tau_v           )
    output           = RMP3_XX_8(Z               ,
                                 X_o             ,
                                 X_v             ,
                                 tau_o           ,
                                 tau_v           )
    assert np.allclose(output, benchmark)
    output2          = RMP3_XX_8_opt_mem(Z               ,
                                         X_o             ,
                                         X_v             ,
                                         tau_o           ,
                                         tau_v           ,
                                         buffer          )
    assert np.allclose(output2, benchmark)
    print(output)   
    print(benchmark)
    print(output2)  
