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

    assert self.xc_code_kernel.upper()=='RPA', '{} ?'.format(self.xc_code_kernel)

    # Allocation of kernels in the product basis to be used in Hartree and exchange-type kernels
    dtype, ns, p = self.dtype, self.nspin, self.nprod
    self.q2hk = zeros((2*ns-1,p,p),dtype=dtype) if xc in ['LDA','GGA'] else zeros((1,p,p),dtype=dtype)

    if xc in ['CIS','HF']: self.q2xk = self.q2hk
    elif xc in ['GW']: self.q2xk = zeros((1,p,p),dtype=dtype)
    else: self.q2xk = None
      
    # Computation of kernels in the product basis...
    for fh in self.q2hk: fh[:,:]=pack2den_u(self.kernel)
    if xc in ['LDA', 'GGA']:
      for q,m in enumerate(self.comp_fxc_pack(**kw)): self.q2hk[q] += pack2den_u(m)
    elif xc in ['GW']:
      self.q2xk[0]=pack2den_u(self.kernel)+self.si_c([0.0])[0].real

    # Allocation of the four-point kernel(s) which is to be used when RAM available
    n, v_dab, cc_da = self.norbs, self.v_dab, self.cc_da
    if xc in ['LDA','GGA']:
      self.q2k_4p = zeros((2*ns-1,n*n,n*n),dtype=dtype)
    elif xc in ['HF', 'CIS', 'GW']: # 1 or 2? Examples of UHF+TD of H2O and H2B(spin=3): 158, 159
      self.q2k_4p = zeros((1, n*n, n*n),dtype=dtype) 
    elif xc in ['RPA']:
      self.q2k_4p = zeros((1, n*n, n*n),dtype=dtype)
    else:
      raise RuntimeError('Unknown funct: '+xc)
      
    q2k_4p = self.q2k_4p # q2k_4p is now alias to the  self.q2k_4p
    
    # Computation of the four-point interaction kernel...
    for q,fhxc in enumerate(self.q2hk): # Start with Hartree interaction kernel 
      q2k_4p[q] = (((v_dab.T*(cc_da*fhxc))*cc_da.T)*v_dab).reshape([n*n,n*n])
      
    if xc in ['CIS', 'HF', 'GW']: # Add Fock-like (exchange) interaction kernel 
      w_xc_4p = (((v_dab.T*(cc_da* self.q2xk[0] ))*cc_da.T)*v_dab).reshape([n*n,n*n])
      q2k_4p[0] -= 0.5*einsum('abcd->bcad', w_xc_4p.reshape([n,n,n,n])).reshape([n*n,n*n])

    if self.nspin==1:
      self.ss2kernel_4p = [[q2k_4p[0]]]
    elif self.nspin==2:
      if len(q2k_4p)==1 :
        self.ss2kernel_4p = [[q2k_4p[0], q2k_4p[0]], [q2k_4p[0],q2k_4p[0]]]
      elif len(q2k_4p)==2 : # # 1 or 2? Examples of UHF+TD of H2O and H2B(spin=3): 158, 159
        self.ss2kernel_4p = [[q2k_4p[0], q2k_4p[1]], [q2k_4p[1],q2k_4p[0]]]
      else:
        self.ss2kernel_4p = [[q2k_4p[0], q2k_4p[1]], [q2k_4p[1],q2k_4p[2]]]

    if xc=='GW':
      self.define_e_x_l0(self.mo_energy_gw, self.mo_coeff_gw, **kw)
    else:
      self.define_e_x_l0(self.mo_energy, self.mo_coeff, **kw)

    #print(__name__, 'Fermi energy', self.fermi_energy)
    #np.set_printoptions(linewidth=1000)
    #for s in range(self.nspin):
    #  print(self.ksn2f_l0[s,:10])
    #  print(self.ksn2e_l0[s,:10])
    #for s in range(self.nspin):
    #  print(self.mo_energy[0,s,:10])
    #  print(self.mo_occ[0,s,:10])


  def define_e_x_l0(self, mo_energy, mo_coeff, **kw):
    """  Define eigenvalues and eigenvecs for the two-particle Green's function L0(1,2,3,4) """
    self.ksn2e_l0 = np.copy(mo_energy).reshape((self.nspin,self.norbs))

    if 'fermi_energy' in kw:
      from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
      self.fermi_energy = kw['fermi_energy']
      print(__name__, 'Fermi energy is specified => recompute occupations')
      ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
      for s,n2fd in enumerate(ksn2fd[0]):
        if not all(n2fd>self.nfermi_tol): continue
        print(self.telec, s, self.nfermi_tol, n2fd)
        raise RuntimeError(__name__, 'telec is too high?')
      ksn2f = self.ksn2f_l0 = (3-self.nspin)*ksn2fd
    else:
      ksn2f = self.ksn2f_l0 = self.mo_occ.reshape((self.nspin,self.norbs))

    self.x_l0 = np.copy(mo_coeff).reshape((self.nspin,self.norbs,self.norbs)) 

    tol = self.nfermi_tol
    self.nfermi_l0 = array([argmax(ksn2f[s]<tol) for s in range(self.nspin)], dtype=int)
    self.vstart_l0 = array([argmax(1-ksn2f[s]>=tol) for s in range(self.nspin)], dtype=int)
    self.xocc_l0 = [mo_coeff[0,s,:nf,:,0] for s,nf in enumerate(self.nfermi_l0)]
    self.xvrt_l0 = [mo_coeff[0,s,vs:,:,0] for s,vs in enumerate(self.vstart_l0)]

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

  def apply_l0_spin_non_diag(self, sab, comega=1j*0.0):
    """ The definition of L0 which is not spin-diagonal even for parallel-spin references"""
    assert sab.size==(self.nspin*self.norbs2)
    self.l0_ncalls+=1
        
    sab = sab.reshape((self.nspin,self.norbs,self.norbs))
    ab2v = np.zeros_like(sab, dtype=self.dtypeComplex)
    for ns in range(self.nspin):
      for ms in range(self.nspin):
        nb2v = dot(self.x_l0[ns], sab[ns])
        nm2v = blas.cgemm(1.0, nb2v, self.x_l0[ms].T)
    
        for n,[en,fn] in enumerate(zip(self.ksn2e_l0[ns,:],self.ksn2f_l0[ns,:])):
          for m,[em,fm] in enumerate(zip(self.ksn2e_l0[ms,:],self.ksn2f_l0[ms,:])):
            nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)))

        nb2v = blas.cgemm(1.0, nm2v, self.x_l0[ms])
        ab2v[ns] += blas.cgemm(0.5, self.x_l0[ns].T, nb2v)
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
    op = LinearOperator((nsnn,nsnn), matvec=self.sext2seff_matvec, dtype=self.dtypeComplex)
    sext_shape = np.require(sext.reshape(nsnn), dtype=self.dtypeComplex, requirements='C')
    resgm,info = gmres_alias(op, sext_shape, tol=self.tddft_iter_tol)
    return (resgm.reshape(-1),info)

  def sext2seff_matvec(self, sab):
    """ This is operator which we effectively want to have inverted (1 - K L0) and find the action of it's 
    inverse by solving a linear equation with a GMRES method. See the method seff(...)"""
    self.matvec_ncalls+=1 
    kl0v = self.apply_kernel4p( self.apply_l0(sab, self.comega_current) )
    return sab - kl0v

  def apply_kernel4p(self, ddm):
    """ This applies the 4-point interaction kernel. This operator can be arbitrarily complex,
    therefore we formulate it as a procedure. """
    if self.nspin==1:
      return self.apply_kernel4p_nspin1(ddm)
    elif self.nspin==2:
      return self.apply_kernel4p_nspin2(ddm)

  def apply_kernel4p_nspin1(self, ddm):
    reim = require(ddm.real, dtype=self.dtype, requirements=["A","O"]) # real part
    mv_real = dot(self.ss2kernel_4p[0][0], reim)
    
    reim = require(ddm.imag, dtype=self.dtype, requirements=["A","O"]) # imaginary
    mv_imag = dot(self.ss2kernel_4p[0][0], reim)
    return mv_real+1j*mv_imag

  def apply_kernel4p_nspin2(self, ddm):
    res = np.zeros((2,self.nspin,self.norbs2), dtype=self.dtype)
    aux = np.zeros((self.norbs2), dtype=self.dtype)
    s2ddm = ddm.reshape((self.nspin,self.norbs2))

    for s in range(self.nspin):
      for t in range(self.nspin):
        for ireim,sreim in enumerate(('real', 'imag')):
          aux[:] = require(getattr(s2ddm[t], sreim), dtype=self.dtype, requirements=["A","O"])
          res[ireim,s] += np.dot(self.ss2kernel_4p[s][t], aux)

    return res[0].reshape(-1)+1j*res[1].reshape(-1)

  def apply_l(self, sab, comega=1j*0.0):
    """ This applies the interacting, two-point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    seff,info = self.seff(sab, comega)
    return self.apply_l0( seff, comega )

  def comp_polariz_nonin_ave(self, comegas):
    """ Non-interacting average polarizability, i.e. <d |L0| d>"""
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for iw,omega in enumerate(comegas):
      for ixyz,dip in enumerate(self.dip_ab):
        if self.verbosity>1: print(__name__, ixyz, iw)
        d = np.concatenate([dip.reshape(-1) for s in range(self.nspin)])
        vab = self.apply_l0(d, omega)
        p[iw] += (vab*d).sum()/3.0
    return p

  def comp_polariz_inter_ave(self, comegas):
    """ Compute a direction-averaged interacting polarizability, i.e. <d |L| d>  """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for iw,omega in enumerate(comegas):
      for ixyz,dip in enumerate(self.dip_ab):
        if self.verbosity>1: print(__name__, ixyz, iw)
        d = np.concatenate([dip.reshape(-1) for s in range(self.nspin)])
        vab = self.apply_l(d, omega)
        p[iw] += (vab*d).sum()/3.0
    return p
