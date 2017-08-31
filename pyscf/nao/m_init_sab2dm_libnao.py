from __future__ import print_function, division
import numpy as np


def init_dm_libnao(sab2dm):
  from pyscf.nao.m_libnao import libnao
  from pyscf.nao.m_sv_chain_data import sv_chain_data
  from ctypes import POINTER, c_double, c_int64
  d = np.require(sab2dm, dtype=c_double, requirements='C')
  libnao.init_sab2dm_libnao.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int64))
  libnao.init_sab2dm_libnao(d.ctypes.data_as(POINTER(c_double)), c_int64(d.shape[1]), c_int64(d.shape[0]) )
  return True
