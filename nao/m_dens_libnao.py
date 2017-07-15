from __future__ import print_function, division
from pyscf.nao.m_libnao import libnao
from ctypes import POINTER, c_double, c_int64
from numpy import require, zeros

libnao.dens_libnao.argtypes = (
  POINTER(c_double),# crds[i,0:3] 
  POINTER(c_int64), # ncrds
  POINTER(c_double),# dens[i,0:]  
  POINTER(c_int64)) # ndens


def dens_libnao(crds, nspin):
  """  Compute the electronic density using library call """
  assert crds.ndim==2  
  assert crds.shape[-1]==3
  
  nc = crds.shape[0]
  crds_cp = require(crds, dtype=c_double, requirements='C')
  dens = require( zeros((nc, nspin)), dtype=c_double, requirements='CW')
  
  libnao.dens_libnao(
    crds_cp.ctypes.data_as(POINTER(c_double)),
    c_int64(nc),
    dens.ctypes.data_as(POINTER(c_double)),
    c_int64(nspin))

  return dens
