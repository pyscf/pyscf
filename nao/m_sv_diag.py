import numpy
from scipy.linalg import eigh 
from pyscf.nao.m_sv_get_denmat import sv_get_denmat

#
#
#
def sv_diag(sv, kvec=[0.0,0.0,0.0], spin=0, prec='64'):
  h = sv_get_denmat(sv, mattype='hamiltonian', kvec=kvec, spin=spin, prec=prec)
  s = sv_get_denmat(sv, mattype='overlap', kvec=kvec, spin=spin, prec=prec)
  
  print(h.shape, type(h[0,0]))
  
  return( eigh(h, s) )
