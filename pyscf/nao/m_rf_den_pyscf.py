from __future__ import print_function, division
from numpy import dot, zeros, einsum, array, sqrt, where

def rf_den_pyscf(self, ww):
  """ Full matrix interacting response from tdscf class"""
  assert hasattr(self, 'tdscf') 
  pov = self.get_vertex_pov()[0]
  nprd = pov.shape[0]
  nov = pov.shape[1]*pov.shape[2]
  pov = pov.reshape([nprd,nov])

  rf = zeros([len(ww),nprd,nprd], dtype=self.dtypeComplex)

  for e,(x,y) in zip(self.tdscf.e, self.tdscf.xy):
    xpy = (2*(x+y)).reshape(nov)
    pI = dot(pov,xpy)
    ppqI = einsum('p,q->pq', pI, pI)

    for iw,w in enumerate(ww):
      rf[iw] = rf[iw] + ppqI * ( 1.0/(w-e)-1.0/(w+e) )

  return rf


