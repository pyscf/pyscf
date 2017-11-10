from __future__ import print_function, division
import sys, numpy as np
from numpy import dot, zeros, einsum
from pyscf.nao import tddft_iter
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l

class gw(tddft_iter):

  def __init__(self, **kw):
    """ Constructor G0W0 class """
    # how to exclude from the input the dtype and xc_code ?
    tddft_iter.__init__(self, dtype=np.float64, xc_code='RPA', **kw)
    self.xc_code = 'G0W0'
    assert self.cc_da.shape[1]==self.nprod
    self.kernel_sq = pack2den_l(self.kernel)
    self.v_dab_ds = self.pb.get_dp_vertex_doubly_sparse(axis=2)
  
  def rf0_cmplx_ref(self, ww):
    """ Full matrix response in the basis of atom-centered product functions """
    rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
    v_arr = self.pb.get_dp_vertex_array()    
    
    zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
    for n,(en,fn) in enumerate(zip(self.ksn2e[0,0,0:self.nfermi], self.ksn2f[0, 0, 0:self.nfermi])):
      vx = dot(v_arr, self.xocc[n,:])
      for m,(em,fm) in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        if (fn - fm)<0 : break
        vxx_a = dot(vx, self.xvrt[m,:]) * self.cc_da
        for iw,comega in enumerate(ww):
          zvxx_a[iw,:] = vxx_a * (fn - fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
        rf0 = rf0 + einsum('wa,b->wab', zvxx_a, vxx_a)

    return rf0
  
  rf0 = rf0_cmplx_ref
  
  def si_c(self, ww):
    from numpy.linalg import solve
    """ 
    This computes the correlation part of the screened interaction W_c
    by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K 
    """
    rf0 = si0 = self.rf0(ww)
    for iw,w in enumerate(ww):
      k_c = dot(self.kernel_sq, rf0[iw,:,:])
      b = dot(k_c, self.kernel_sq)
      k_c = np.eye(self.nprod)-k_c
      si0[iw,:,:] = solve(k_c, b)

    return si0

