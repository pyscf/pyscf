from __future__ import print_function, division
import sys, numpy as np
from numpy import concatenate, ravel, diagflat, zeros, einsum, stack
from pyscf.nao import scf
from copy import copy

class qchem_inter_rf(scf):
  """ Quantum-chemical interacting response function """
  
  def __init__(self, **kw):
    """ Constructor... """
    scf.__init__(self, **kw)

    nf = self.nfermi[0]
    nv = self.norbs-self.vstart[0]
    self.FmE = np.add.outer(self.ksn2e[0,0,self.vstart[0]:],-self.ksn2e[0,0,:self.nfermi[0]]).T
    self.sqrt_FmE = np.sqrt(self.FmE).reshape([nv*nf])
    self.kernel_qchem_inter_rf()

  def kernel_qchem_inter_rf_pos_neg(self, **kw):
    """ This is constructing the E_m-E_n and E_n-E_m matrices """
    h_rpa = diagflat(concatenate((ravel(self.FmE),-ravel(self.FmE))))
    print(h_rpa.shape)

    nf = self.nfermi[0]
    nv = self.norbs-self.vstart[0]
    vs = self.vstart[0]
    neh = nf*nv
    x = self.mo_coeff[0,0,:,:,0]
    pab2v = self.pb.get_ac_vertex_array()
    self.pmn2v = pmn2v = einsum('nb,pmb->pmn', x[:nf,:], einsum('ma,pab->pmb', x[vs:,:], pab2v))
    pmn2c = einsum('qp,pmn->qmn', self.hkernel_den, pmn2v)
    meri = einsum('pmn,pik->mnik', pmn2c, pmn2v).reshape((nf*nv,nf*nv))
    #print(meri.shape)
    #meri.fill(0.0)
    h_rpa[:neh, :neh] = h_rpa[:neh, :neh]+meri
    h_rpa[:neh, neh:] = h_rpa[:neh, neh:]+meri
    h_rpa[neh:, :neh] = h_rpa[neh:, :neh]-meri
    h_rpa[neh:, neh:] = h_rpa[neh:, neh:]-meri
    edif, s2z = np.linalg.eig(h_rpa)
    print(abs(h_rpa-h_rpa.transpose()).sum())
    print('edif', edif.real*27.2114)
    
    return 
  
  def inter_rf(self, ww):
    """ This delivers the interacting response function in the product basis"""
    rf = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
    p,m,n = self.pmn2v.shape
    sp2v = np.dot(self.s2xpy, self.pmn2v.reshape(p,m*n).T)
    for iw,w in enumerate(ww):
      for iOmega,(Omega,p2v) in enumerate(zip(self.s2omega, sp2v)):
        p2z = p2v*(2.0/(w-Omega)-2.0/(w+Omega))
        rf[iw] += np.outer(p2z,p2v)
    return rf
  
  def kernel_qchem_inter_rf(self, **kw):
    from pyscf.gw.gw import rpa_AB_matrices
    """ This is constructing A B matrices and diagonalizes the problem """

    nf = self.nfermi[0]
    nv = self.norbs-self.vstart[0]
    vs = self.vstart[0]
    
    x = self.mo_coeff[0,0,:,:,0]
    pab2v = self.pb.get_ac_vertex_array()
    self.pmn2v = pmn2v = einsum('nb,pmb->pmn', x[:nf,:], einsum('ma,pab->pmb', x[vs:,:], pab2v))
    pmn2c = einsum('qp,pmn->qmn', self.hkernel_den, pmn2v)
    meri = einsum('pmn,pik->mnik', pmn2c, pmn2v)
    #meri.fill(0.0)

    A = (diagflat( self.FmE ).reshape([nv,nf,nv,nf]) + meri).reshape([nv*nf,nv*nf])
    B = meri.reshape([nv*nf,nv*nf])

    assert np.allclose(A, A.transpose())
    assert np.allclose(B, B.transpose())

    AmB = A-B
    n = len(AmB)
    print(__name__)
    for i in range(n): print(i, AmB[i,i])
    
    ham_rpa = np.multiply(self.sqrt_FmE[:,None], np.multiply(A+B, self.sqrt_FmE))
    esq, self.s2z = np.linalg.eigh(ham_rpa)
    self.s2omega = np.sqrt(esq)
    print(self.s2omega*27.2114)

    self.s2z = self.s2z.T
    self.s2xpy = np.zeros_like(self.s2z)
    for s,(e2,z) in enumerate(zip(esq, self.s2z)):
      w = np.sqrt(e2)
      self.s2xpy[s] = np.multiply(self.sqrt_FmE, self.s2z[s])/np.sqrt(w)
      #print(e2, abs(np.dot(ham_rpa,z)-e2*z).sum())
    return self.s2omega,self.s2z
