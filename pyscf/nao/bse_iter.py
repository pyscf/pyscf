from __future__ import print_function, division
import numpy as np
from numpy import array, argmax, einsum, require, zeros, dot
from timeit import default_timer as timer
from pyscf.nao import gw
from scipy.linalg import blas
from pyscf.nao.m_pack2den import pack2den_u

class bse_iter(gw):

  def __init__(self, **kw):
    """ 
      Iterative BSE a la PK, DF, OC JCTC 
      additionally to the fields from tddft_iter_c, we add the dipole matrix elements dab[ixyz][a,b]
      which is constructed as list of numpy arrays 
       $ d_i = \int f^a(r) r_i f^b(r) dr $
    """
    xc_code_kw = kw['xc_code'] if 'xc_code' in kw else None
    gw.__init__(self, **kw)
    
    self.l0_ncalls = 0
    self.dip_ab = [d.toarray() for d in self.dipole_coo()]
    self.norbs2 = self.norbs**2

    self.xc_code = self.xc_code if xc_code_kw is None else xc_code_kw 
    xc = self.xc_code.split(',')[0].upper()
    if self.verbosity>0: 
      print(__name__, '     xc_code_mf ', self.xc_code_mf)
      print(__name__, ' xc_code_kernel ', self.xc_code_kernel)
      print(__name__, '    xc_code_scf ', self.xc_code_scf)
      print(__name__, '        xc_code ', self.xc_code)

    if (xc=='LDA' or xc=='GGA') and self.xc_code_kernel.upper()=='RPA':
      """ Need to add LDA or GGA xc kernel to Hartree kernel..."""
      self.comp_fxc_pack(kernel=self.kernel, **kw)

    kernel_den = pack2den_u(self.kernel)
    n = self.norbs
    v_dab = self.v_dab
    cc_da = self.cc_da
    # self.x = self.mo_coeff[0,:,:,:,0] 
    
    """ Start with the Hartree kernel """
    self.kernel_4p = (((v_dab.T*(cc_da*kernel_den))*cc_da.T)*v_dab).reshape([n*n,n*n])
      
    if xc=='CIS' or xc=='HF' or xc=='GW':
      """ Add exchange operator """
      self.kernel_4p -= 0.5*einsum('abcd->bcad', self.kernel_4p.reshape([n,n,n,n])).reshape([n*n,n*n])
      self.kernel_4p_w = self.kernel_4p # .kernel_4p_w -- this will be used in the iterative procedure
    elif xc=='GWw':
      """ Add exchange operator """
      self.kernel_4p -= 0.5*einsum('abcd->bcad', self.kernel_4p.reshape([n,n,n,n])).reshape([n*n,n*n])
      self.si_aa_comega = None
    elif xc=='RPA' or xc=='LDA' or xc=='GGA':
      self.kernel_4p_w = self.kernel_4p
    else :
      print(' ?? xc_code ', self.xc_code, xc)
      raise RuntimeError('??? xc_code ???')

    if xc=='GW':
      """ Add correlation operator """
      w_c_4p = (((v_dab.T*(cc_da*  self.si_c([0.0])[0].real ))*cc_da.T)*v_dab).reshape([n*n,n*n])
      self.kernel_4p -= 0.5*einsum('abcd->bcad', w_c_4p.reshape([n,n,n,n])).reshape([n*n,n*n])
      self.kernel_4p_w = self.kernel_4p # .kernel_4p_w -- this will be used in the iterative procedure
      #print(self.kernel_4p_w.dtype, __name__, self.kernel_4p_w.sum())

    if xc=='GW' or xc=='GWW':
      self.define_e_x_l0(self.mo_energy_gw, self.mo_coeff_gw)
    else:
      self.define_e_x_l0(self.mo_energy, self.mo_coeff)

  def define_e_x_l0(self, mo_energy, mo_coeff):
    """
      Define eigenvalues and eigenvectors which will be used in the two-particle Green's function L0(1,2,3,4)
    """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

    self.ksn2e_l0 = np.copy(mo_energy).reshape((self.nspin,self.norbs))

    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e_l0, self.fermi_energy)
    if np.all(ksn2fd>self.nfermi_tol):
      print(__name__, self.telec, nfermi_tol, ksn2fd)
      raise RuntimeError('telec is too high?')
    self.ksn2f_l0 = (3-self.nspin)*ksn2fd

    self.x_l0 = np.copy(mo_coeff).reshape((self.nspin,self.norbs,self.norbs)) 

    tol = self.nfermi_tol
    self.nfermi_l0 = array([argmax(ksn2fd[s,:]<tol) for s in range(self.nspin)], dtype=int)
    self.vstart_l0 = array([argmax(1-ksn2fd[s,:]>=tol) for s in range(self.nspin)], dtype=int)
    self.xocc_l0 = [mo_coeff[0,s,:nfermi,:,0] for s,nfermi in enumerate(self.nfermi_l0)]
    self.xvrt_l0 = [mo_coeff[0,s,vstart:,:,0] for s,vstart in enumerate(self.vstart_l0)]

  def apply_l0(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.nspin*self.norbs2)
    self.l0_ncalls+=1
    
    sab = sab.reshape((self.nspin, self.norbs,self.norbs))
    ab2v = np.zeros_like(sab, dtype=self.dtypeComplex)
    
    for s,(ab,xv,xo,x,n2e,n2f) in enumerate(zip(
      sab,self.xvrt_l0,self.xocc_l0,self.x_l0,self.ksn2e_l0,self.ksn2f_l0)):
        
      nm2v = zeros((self.norbs,self.norbs), self.dtypeComplex)
      nm2v[self.vstart_l0[s]:, :self.nfermi_l0[s]] = blas.cgemm(1.0, dot(xv,ab),xo.T)
      nm2v[:self.nfermi_l0[s], self.vstart_l0[s]:] = blas.cgemm(1.0, dot(xo,ab),xv.T)
    
      for n,(en,fn) in enumerate(zip(n2e,n2f)):
        for m,(em,fm) in enumerate(zip(n2e,n2f)):
          nm2v[n,m] = nm2v[n,m] * (fn-fm) / (comega - (em - en))

      nb2v = blas.cgemm(1.0, nm2v, x)
      ab2v[s] += blas.cgemm(1.0, x.T, nb2v)
      
    return ab2v.reshape(-1)

  def apply_l0_ref(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.nspin*self.norbs2)
    self.l0_ncalls+=1
        
    sab = sab.reshape((self.nspin,self.norbs,self.norbs))
    ab2v = np.zeros_like(sab, dtype=self.dtypeComplex)
    for s in range(self.nspin):
      nb2v = dot(self.x_l0[s], sab[s])
      nm2v = blas.cgemm(1.0, nb2v, self.x_l0[s].T)
    
      for n,[en,fn] in enumerate(zip(self.ksn2e_l0[s,:],self.ksn2f_l0[s,:])):
        for m,[em,fm] in enumerate(zip(self.ksn2e_l0[s,:],self.ksn2f_l0[s,:])):
          nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)))

      nb2v = blas.cgemm(1.0, nm2v, self.x_l0[s])
      ab2v[s] += blas.cgemm(1.0, self.x_l0[s].T, nb2v)
    return ab2v.reshape(-1)

  def seff(self, sext, comega=1j*0.0):
    """ This computes an effective two point field (scalar non-local potential) given an external two point field.
        L = L0 (1 - K L0)^-1
        We want therefore an effective X_eff for a given X_ext
        X_eff = (1 - K L0)^-1 X_ext   or   we need to solve linear equation
        (1 - K L0) X_eff = X_ext  

        The operator (1 - K L0) is named self.sext2seff_matvec """
    
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import lgmres as gmres_alias
    #from spipy.sparse.linalg import gmres as gmres_alias
    nsnn = self.nspin*self.norbs2
    assert sext.size==nsnn
    
    self.comega_current = comega
    xc = self.xc_code.split(',')[0].upper()
    if xc=='GWW': # to be removed
      """ 
        Add a correlation operator: frequency-dependent screened interaction 
        K_c(12,34,omega) = - 0.5 W_c(1,2,omega) delta(13)delta(24) 
        REMARK: this is only to see the effect of using single frequency in L, L0 and K
      """
      if self.si_aa_comega != comega:
        self.si_aa_comega,n,v_dab,cc_da = comega,self.norbs,self.v_dab,self.cc_da
        si_aa = self.si_c([comega])[0]
        w_c_4p = (((v_dab.T*(cc_da*  si_aa ))*cc_da.T)*v_dab).reshape([n*n,n*n])
        self.kernel_4p_w = self.kernel_4p - 0.5*einsum('abcd->bcad', w_c_4p.reshape([n,n,n,n])).reshape([n*n,n*n])
        #print(self.kernel_4p_w.dtype, comega, self.kernel_4p_w.sum())
        
    op = LinearOperator((nsnn,nsnn), matvec=self.sext2seff_matvec, dtype=self.dtypeComplex)
    sext_shape = np.require(sext.reshape(nsnn), dtype=self.dtypeComplex, requirements='C')
    resgm,info = gmres_alias(op, sext_shape, tol=self.tddft_iter_tol)
    return (resgm.reshape([self.norbs,self.norbs]),info)

  def sext2seff_matvec(self, sab):
    """ This is operator which we effectively want to have inverted (1 - K L0) and find the action of it's 
    inverse by solving a linear equation with a GMRES method. See the method seff(...)"""
    self.matvec_ncalls+=1 
    l0 = self.apply_l0(sab, self.comega_current)
    
    l0_reim = require(l0.real, dtype=self.dtype, requirements=["A","O"])  # real part
    mv_real = dot(self.kernel_4p_w, l0_reim)
    
    l0_reim = require(l0.imag, dtype=self.dtype, requirements=["A", "O"]) # imaginary
    mv_imag = dot(self.kernel_4p_w, l0_reim)

    return sab - (mv_real + 1.0j*mv_imag)

  def apply_l(self, sab, comega=1j*0.0):
    """ This applies the interacting, two-point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    seff,info = self.seff(sab, comega)
    return self.apply_l0( seff, comega )

  def comp_polariz_nonin_ave(self, comegas):
    """ Non-interacting average polarizability """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for iw,omega in enumerate(comegas):
      for dip in self.dip_ab:
        if self.verbosity>1: print(__name__, ixyz, iw)
        d = np.concatenate([dip.reshape(-1) for s in range(self.nspin)])
        vab = self.apply_l0(d, omega)
        p[iw] += (vab*d).sum()/3.0
    return p

  def comp_polariz_inter_ave(self, comegas):
    """ Compute a direction-averaged interacting polarizability  """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for iw,omega in enumerate(comegas):
      for dip in self.dip_ab:
        if self.verbosity>1: print(__name__, ixyz, iw)
        d = np.concatenate([dip.reshape(-1) for s in range(self.nspin)])
        vab = self.apply_l(d, omega)
        p[iw] += (vab*d).sum()/3.0
    return p
