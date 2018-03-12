# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import numpy as np
from timeit import default_timer as timer
from pyscf.nao.m_tddft_iter import tddft_iter_c
from scipy.linalg import blas
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l
from pyscf.nao.m_tddft_iter import use_numba
if use_numba: from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba

class bse_iter_c(tddft_iter_c):

  def __init__(self, sv, pb, iter_tol=1e-2, iter_broadening=0.00367493,
          nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None, xc_code='RPA',
          GPU=False, precision="single", **kvargs):
    """ Iterative BSE a la PK, DF, OC JCTC 
      additionally to the fields from tddft_iter_c, we add the dipole matrix elements dab[ixyz][a,b]
      which is constructed as list of numpy arrays 
       $ d_i = \int f^a(r) r_i f^b(r) dr $
    """
    
    tddft_iter_c.__init__(self, sv, pb, tddft_iter_tol=iter_tol, tddft_iter_broadening=iter_broadening,
          nfermi_tol=nfermi_tol, telec=telec, nelec=nelec, fermi_energy=fermi_energy, xc_code=xc_code,
          GPU=GPU, precision=precision, **kvargs)

    self.dab = [d.toarray() for d in sv.dipole_coo()]
    self.norbs2 = self.norbs**2
    kernel_den = pack2den_l(self.kernel)
    n = self.norbs
    self.kernel_4p = (((self.v_dab.T*(self.cc_da*kernel_den))*self.cc_da.T)*self.v_dab).reshape([n*n,n*n])
    #print(type(self.kernel_4p), self.kernel_4p.shape, 'this is just a reference kernel, must be removed later for sure')

    if xc_code=='CIS' or xc_code=='HF':
      self.kernel_4p = self.kernel_4p - 0.5*np.einsum('(abcd->acbd)', self.kernel_4p.reshape([n,n,n,n])).reshape([n*n,n*n])


  def apply_l0(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.norbs2), "%r,%r"%(sab.size,self.norbs2)

    sab = sab.reshape([self.norbs,self.norbs])
    self.l0_ncalls+=1
    nb2v = np.dot(self.xocc, sab)
    nm2v = blas.cgemm(1.0, nb2v, np.transpose(self.xvrt))
    if use_numba:
      div_eigenenergy_numba(self.ksn2e, self.ksn2f, self.nfermi,
        self.vstart, comega, nm2v.real, nm2v.imag, self.ksn2e.shape[2])
    else:
      for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
        for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
          m = j+n+1-self.vstart
          nm2v[n,m] = nm2v[n,m] * (fn-fm) *\
          ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )

    nb2v = blas.cgemm(1.0, nm2v, self.xvrt)
    ab2v = blas.cgemm(1.0, np.transpose(self.xocc), nb2v)
    return ab2v

  def seff(self, sext, comega=1j*0.0):
    """ This computes an effective two point field (scalar non-local potential) given an external two point field.
        L = L0 (1 - K L0)^-1
        We want therefore an effective X_eff for a given X_ext
        X_eff = (1 - K L0)^-1 X_ext   or   we need to solve linear equation
        (1 - K L0) X_eff = X_ext  

        The operator (1 - K L0) is named self.sext2seff_matvec """
    
    from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    assert sext.size==(self.norbs2), "%r,%r"%(sext.size,self.norbs2)

    self.comega_current = comega
    op = LinearOperator((self.norbs2,self.norbs2), matvec=self.sext2seff_matvec, dtype=self.dtypeComplex)
    sext_shape = np.require(sext.reshape(self.norbs2), dtype=self.dtypeComplex, requirements='C')
    resgm,info = gmres_alias(op, sext_shape, tol=self.tddft_iter_tol)
    return (resgm.reshape([self.norbs,self.norbs]),info)

  def sext2seff_matvec(self, sab):
    """ This is operator which we effectively want to have inverted (1 - K L0) and find the action of it's 
    inverse by solving a linear equation with a GMRES method. See the method seff(...)"""
    self.matvec_ncalls+=1 
    
    l0 = self.apply_l0(sab, self.comega_current).reshape(self.norbs2)
    
    l0_reim = np.require(l0.real, dtype=self.dtype, requirements=["A", "O"])     # real part
    mv_real = np.dot(self.kernel_4p, l0_reim)
    
    l0_reim = np.require(l0.imag, dtype=self.dtype, requirements=["A", "O"])     # imaginary part
    mv_imag = np.dot(self.kernel_4p, l0_reim)

    return sab - (mv_real + 1.0j*mv_imag)

  def apply_l(self, sab, comega=1j*0.0):
    """ This applies the interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    seff,info = self.seff(sab, comega)
    return self.apply_l0( seff, comega )
