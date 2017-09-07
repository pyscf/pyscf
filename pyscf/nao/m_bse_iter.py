from __future__ import print_function, division
import numpy as np
from timeit import default_timer as timer
from pyscf.nao.m_tddft_iter import tddft_iter_c
from scipy.linalg import blas
from pyscf.nao.m_tddft_iter import use_numba
if use_numba: from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba

class bse_iter_c(tddft_iter_c):

  def __init__(self, sv, pb, iter_tol=1e-2, iter_broadening=0.00367493,
          nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None, xc_code='RPA',
          GPU=False, precision="single", **kvargs):
    """ Iterative BSE a la PK, DF, OC JCTC """
    
    tddft_iter_c.__init__(self, sv, pb, tddft_iter_tol=iter_tol, tddft_iter_broadening=iter_broadening,
          nfermi_tol=nfermi_tol, telec=telec, nelec=nelec, fermi_energy=fermi_energy, xc_code=xc_code,
          GPU=GPU, precision=precision, **kvargs)

  def apply_l0(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    self.l0_ncalls+=1
    nb2v = np.dot(self.xocc, sab)
    nm2v = blas.cgemm(1.0, nb2v, np.transpose(self.xvrt))
    if use_numba:
      div_eigenenergy_numba(self.ksn2e, self.ksn2f, self.nfermi,
        self.vstart, comega, nm2v, self.ksn2e.shape[2])
    else:
      for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
        for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
          m = j+n+1-self.vstart
          nm2v[n,m] = nm2v[n,m] * (fn-fm) *\
          ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )

    nb2v = blas.cgemm(1.0, nm2v, self.xvrt)
    ab2v = blas.cgemm(1.0, np.transpose(self.xocc), nb2v)
    return ab2v
