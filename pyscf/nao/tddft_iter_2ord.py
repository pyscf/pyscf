from __future__ import print_function, division
import numpy as np
from numpy import transpose, zeros, array, argmax, require
from pyscf.nao import tddft_iter


class tddft_iter_2ord(tddft_iter):
  """ Iterative TDDFT with a high-energy part of the KS eigenvectors compressed """

  def __init__(self, **kw):
    tddft_iter.__init__(self, **kw)

  def kchi0_mv(self, v, comega=None):
    """ Operator O = K chi0 """
    self.matvec_ncalls+=1
    w = self.comega_current if comega is None else comega
    chi0 = self.apply_rf0(v, w)
    
    chi0_reim = require(chi0.real, dtype=self.dtype, requirements=["A", "O"])
    matvec_real = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim)
    
    chi0_reim = require(chi0.imag, dtype=self.dtype, requirements=["A", "O"])
    matvec_imag = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim)

    return (matvec_real + 1.0j*matvec_imag)

  def umkckc_mv(self, v):
    """ Operator O = [1- K chi0 K chi0] """
    return v - self.kchi0_mv(self.kchi0_mv(v))

  def upkc_mv(self, v):
    """ Operator O = [1 + K chi0] """
    return v + self.kchi0_mv(v)

  def solve_umkckc(self, vext, comega=1j*0.0, x0=None):
    """ This solves a system of linear equations 
           (1 - K chi0 K chi0 ) X = vext 
     or computes 
           X = (1 - K chi0 K chi0 )^{-1} vext 
    """
    from scipy.sparse.linalg import LinearOperator, lgmres
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff2_op = LinearOperator((self.nprod,self.nprod), matvec=self.umkckc_mv, dtype=self.dtypeComplex)

    if self.res_method == "absolute":
        tol = 0.0
        atol = self.tddft_iter_tol
    elif self.res_method == "relative":
        tol = self.tddft_iter_tol
        atol = 0.0
    elif self.res_method == "both":
        tol = self.tddft_iter_tol
        atol = self.tddft_iter_tol
    else:
        raise ValueError("Unknow res_method")

    resgm,info = lgmres(veff2_op, np.require(vext, dtype=self.dtypeComplex,
                                             requirements='C'), x0=x0, 
                        tol=tol, atol=atol, maxiter=self.maxiter)
    
    if info != 0:  print("LGMRES Warning: info = {0}".format(info))

    return resgm

  def polariz_upkc(self, comegas):
    """ Compute interacting polarizability along the xx direction using an alternative algorighm with chi = chi0 (1+K chi0) [1-K chi0 K chi0]^(-1) """
    pxx = zeros((len(comegas),3), dtype=self.dtypeComplex)
    vext = transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      v1 = self.solve_umkckc(vext[0], comega)
      dn1 = self.apply_rf0(v1, comega)
      dn2 = self.apply_rf0(self.kchi0_mv(v1), comega)
      pxx[iw,1] = np.dot(vext[0], dn1)
      pxx[iw,2] = np.dot(vext[0], dn2)
      pxx[iw,0] = np.dot(vext[0], dn1+dn2)
    return pxx

  def polariz_dckcd(self, comegas):
    """ Compute a term <d chi0 K chi0 d> """
    pxx = zeros(len(comegas), dtype=self.dtypeComplex)
    vext = transpose(self.moms1)
    for iw, w in enumerate(comegas):
      v1 = self.kchi0_mv(vext[0], comega=w)
      dn1 = self.apply_rf0(v1, w)
      pxx[iw] = np.dot(vext[0], dn1)
    return pxx
