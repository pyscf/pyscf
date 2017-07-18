from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator

class tddft_iter_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2, tddft_iter_broadening=0.00367493, nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None):
    """ Iterative TDDFT a la PK, DF, OC JCTC """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    assert tddft_iter_tol>1e-6
    assert type(tddft_iter_broadening)==float
    assert sv.wfsx.x.shape[-1]==1 # i.e. real eigenvectors we accept here

    self.rf0_ncalls = 0
    self.matvec_ncalls = 0
    self.tddft_iter_tol = tddft_iter_tol
    self.eps = tddft_iter_broadening
    self.sv, self.pb, self.norbs, self.nspin = sv, pb, sv.norbs, sv.nspin
    #print('before vertex_coo')
    self.v_dab = pb.get_dp_vertex_coo(dtype=np.float32).tocsr()
    #print('before conversion coefficients coo')
    self.cc_da = pb.get_da2cc_coo(dtype=np.float32).tocsr()
    #print('before moments')
    self.moms0,self.moms1 = pb.comp_moments(dtype=np.float32)
    self.nprod = self.moms0.size
    #print('before kernel')
    self.kernel = pb.comp_coulomb_den(dtype=np.float32)
    self.telec = sv.hsx.telec if telec is None else telec
    self.nelec = sv.hsx.nelec if nelec is None else nelec
    self.fermi_energy = sv.fermi_energy if fermi_energy is None else fermi_energy
    self.x  = np.require(sv.wfsx.x, dtype=np.float32, requirements='CW')
    self.ksn2e = np.require(sv.wfsx.ksn2e, dtype=np.float32, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)
    #print('before xocc, xvrt')
    self.xocc = self.x[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = self.x[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?
    
  def apply_rf0(self, v, comega=1j*0.0):
    """ This applies the non-interacting response function to a vector (a set of vectors?) """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    self.rf0_ncalls+=1
    vdp = self.cc_da * np.require(v, dtype=np.complex64)
    no = self.norbs
    sab = csr_matrix((np.transpose(vdp)*self.v_dab).reshape([no,no]))
    nb2v = self.xocc*sab
    #nm2v = np.zeros([self.nfermi,len(self.xvrt)], dtype=np.complex64)
    nm2v = np.dot(nb2v, np.transpose(self.xvrt))
    
    for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
      for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
        m = j+n+1-self.vstart
        nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
        
    nb2v = np.dot(nm2v,self.xvrt)
    ab2v = np.dot(np.transpose(self.xocc),nb2v).reshape(no*no)
    vdp = self.v_dab*ab2v
    res = vdp*self.cc_da
    return res

  def comp_veff(self, vext, comega=1j*0.0):
    """ This computes an effective field (scalar potential) given the external scalar potential """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=np.complex64)
    resgm = gmres_alias(veff_op, np.require(vext, dtype=np.complex64, requirements='C'), tol=self.tddft_iter_tol)
    return resgm
  
  def vext2veff_matvec(self, v):
    self.matvec_ncalls+=1 
    return v-np.dot(self.kernel, self.apply_rf0(v, self.comega_current))

  def comp_polariz_xx(self, comegas):
    """ Polarizability """
    polariz = np.zeros_like(comegas, dtype=np.complex64)
    for iw,comega in enumerate(comegas):
      veff,info = self.comp_veff(self.moms1[:,0], comega)
      #print(iw, info, veff.sum())
      polariz[iw] = np.dot(self.moms1[:,0], self.apply_rf0( veff, comega ))
    return polariz
    
