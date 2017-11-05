from __future__ import print_function, division
import numpy as np
from timeit import default_timer as timer
from pyscf.nao.tddft_iter import use_numba, tddft_iter
from scipy.linalg import blas
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l
if use_numba: from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba

class bse_iter(tddft_iter):

  def __init__(self, **kw):
    """ Iterative BSE a la PK, DF, OC JCTC 
      additionally to the fields from tddft_iter_c, we add the dipole matrix elements dab[ixyz][a,b]
      which is constructed as list of numpy arrays 
       $ d_i = \int f^a(r) r_i f^b(r) dr $
    """
    tddft_iter.__init__(self, **kw)
    self.l0_ncalls = 0
    self.dip_ab = [d.toarray() for d in self.dipole_coo()]
    self.norbs2 = self.norbs**2
    kernel_den = pack2den_l(self.kernel)
    n = self.norbs
    v_dab = self.v_dab
    cc_da = self.cc_da
    self.kernel_4p = (((v_dab.T*(cc_da*kernel_den))*cc_da.T)*v_dab).reshape([n*n,n*n])
    #print(type(self.kernel_4p), self.kernel_4p.shape, 'this is just a reference kernel, must be removed later for sure')

    xc = self.xc_code.split(',')[0]
    if xc=='CIS' or xc=='HF' or xc=='GW':
      pass
      self.kernel_4p -= 0.5*np.einsum('abcd->bcad', self.kernel_4p.reshape([n,n,n,n])).reshape([n*n,n*n])
    elif xc=='RPA' or xc=='LDA' or xc=='GGA':
      pass
    else :
      print(' ?? xc_code ', self.xc_code, xc)
      raise RuntimeError('??? xc_code ???')


  def apply_l0(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.norbs2), "%r,%r"%(sab.size,self.norbs2)

    sab = sab.reshape([self.norbs,self.norbs])
    self.l0_ncalls+=1
    nb2v = np.dot(self.xocc, sab)
    nm2v = blas.cgemm(1.0, nb2v, np.transpose(self.xvrt))
    for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
      for m,[em,fm] in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        #print(n,m+self.vstart,fn-fm)
        nm2v[n,m] = nm2v[n,m] * (fn-fm) * \
          ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )

    #print('padding m<n, which can be also detected as negative occupation difference ')
    for n,fn in enumerate(self.ksn2f[0, 0, 0:self.nfermi]):
      for m,fm in enumerate(self.ksn2f[0,0,self.vstart:n]):
        #print(n,m+self.vstart,fn-fm)
        nm2v[n, m] = 0.0

    #raise RuntimeError('debug')
    nb2v = blas.cgemm(1.0, nm2v, self.xvrt)
    ab2v = blas.cgemm(1.0, np.transpose(self.xocc), nb2v)
    #ab2v = (ab2v + ab2v.T)/2.0
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

  def comp_polariz_nonin_ave(self, comegas):
    """ Non-interacting average polarizability """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for ixyz in range(3):
      for iw,omega in enumerate(comegas):
        vab = self.apply_l0(self.dip_ab[ixyz], omega)
        p[iw] += (vab*self.dip_ab[ixyz]).sum()/3.0
    return p

  def comp_polariz_inter_ave(self, comegas):
    """ Compute a direction-averaged interacting polarizability  """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for ixyz in range(3):
      for iw,omega in enumerate(comegas):
        vab = self.apply_l(self.dip_ab[ixyz], omega)
        p[iw] += (vab*self.dip_ab[ixyz]).sum()/3.0
    return p
