from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix

class tddft_iter_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2, tddft_iter_broadening=0.00367493, nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None):
    """ Iterative TDDFT a la PK, DF, OC JCTC """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    assert tddft_iter_tol>1e-12
    assert type(tddft_iter_broadening)==float
    assert sv.wfsx.x.shape[-1]==1 # i.e. real eigenvectors we accept here

    self.tddft_iter_tol,self.eps = tddft_iter_tol,tddft_iter_broadening
    self.sv, self.pb, self.norbs, self.nspin = sv, pb, sv.norbs, sv.nspin
    self.v_dab = pb.get_dp_vertex_coo(dtype=np.float32).tocsr()
    self.cc_da = pb.get_da2cc_coo(dtype=np.float32).tocsr()
    self.moms0,self.moms1 = pb.comp_moments(dtype=np.float32)
    self.kernel = pb.comp_coulomb_pack(dtype=np.float32)
    self.telec = sv.hsx.telec if telec is None else telec
    self.nelec = sv.hsx.nelec if nelec is None else nelec
    self.fermi_energy = sv.fermi_energy if fermi_energy is None else fermi_energy
    self.x  = np.require(sv.wfsx.x, dtype=np.float32, requirements='CW')
    self.ksn2e = np.require(sv.wfsx.ksn2e, dtype=np.float32, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)
    self.xocc = self.x[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = self.x[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?
    
  def apply_rf0(self, v, omega=0.0, eps_in=None):
    """ This applies the non-interacting response function to a vector or a set of vectors """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    eps = self.eps if eps_in is None else eps_in
    vdp = self.cc_da * np.require(v, dtype=np.complex64)
    n = self.norbs
    sab = csr_matrix((np.transpose(vdp)*self.v_dab).reshape([n,n]))
    nb2v = self.xocc*sab
    #nm2v = np.zeros([self.nfermi,len(self.xvrt)], dtype=np.complex64)
    nm2v = np.dot(nb2v, np.transpose(self.xvrt))
    
    for i,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
      for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        nm2v[i,j] = nm2v[i,j] * (fn-fm) * ( 1.0 / (omega - (em - en) + 1j*eps) - 1.0 / (omega + (em - en) + 1j*eps) )
        
    nb2v = np.dot(nm2v,self.xvrt)
    ab2v = np.dot(np.transpose(self.xocc),nb2v).reshape(n*n)
    vdp = self.v_dab*ab2v
    res = vdp*self.cc_da
    return res
