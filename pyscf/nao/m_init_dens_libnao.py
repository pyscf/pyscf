from __future__ import print_function, division
from pyscf.nao.m_libnao import libnao
from ctypes import POINTER, c_int64, byref

libnao.init_dens_libnao.argtypes = ( POINTER(c_int64), ) # info

def init_dens_libnao():
  """ Initilize the auxiliary for computing the density on libnao site """

  info = c_int64(-999)
  libnao.init_dens_libnao( byref(info))

  return info.value
