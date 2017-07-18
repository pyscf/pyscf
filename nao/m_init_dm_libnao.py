from __future__ import print_function, division
import numpy as np


def init_dm_libnao(dm):
  from pyscf.nao.m_libnao import libnao
  from ctypes import POINTER, c_double, c_int64, byref
  
  d = np.require(dm, dtype=c_double, requirements='C')

  libnao.init_dm_libnao.argtypes = (POINTER(c_double), 
    POINTER(c_int64), # nreim 
    POINTER(c_int64), # norbs
    POINTER(c_int64), # nspin
    POINTER(c_int64), # nkpoints
    POINTER(c_int64)) # alloc_stat
  
  alloc_stat = c_int64(-999)

  libnao.init_dm_libnao(d.ctypes.data_as(POINTER(c_double)),
    c_int64(d.shape[-1]), 
    c_int64(d.shape[-2]),
    c_int64(d.shape[-4]),
    c_int64(d.shape[-5]),
    byref(alloc_stat))

  if alloc_stat.value!=0 : 
    raise RuntimeError('could not allocate?') 
    return None

  return dm
