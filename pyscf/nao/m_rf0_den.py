from __future__ import print_function, division
import sys, numpy as np
from numpy import stack, dot, zeros, einsum, array
from timeit import default_timer as timer

def rf0_den(self, ww):
  """ Full matrix response in the basis of atom-centered product functions, and spin-resolved, or better said spin-summed """
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  if hasattr(self, 'pab2v_den'):
    v = self.pab2v_den
  else:
    self.pab2v_den = v = einsum('pab->apb', self.pb.get_ac_vertex_array())
  
  zxvx = zeros((len(ww),self.nprod,self.bsize,self.bsize), dtype=self.dtypeComplex)

  for sn in range(self.nspin):
    nn = list(range(0,self.nfermi[sn],self.bsize))+[self.nfermi[sn]]
    for sm in range(self.nspin):
      mm = list(range(self.vstart[sm],self.norbs,self.bsize))+[self.norbs]
    
      for nbs,nbf in zip(nn,nn[1:]):
        vx = dot(v, self.x[sn,nbs:nbf,:].T)
        for mbs,mbf in zip(mm,mm[1:]):
          xvx = einsum('mb,bpn->pmn', self.x[sm,mbs:mbf,:],vx)
          fmn = np.add.outer(-self.ksn2f[0,sm,mbs:mbf], self.ksn2f[0,sn,nbs:nbf])
          emn = np.add.outer( self.ksn2e[0,sm,mbs:mbf],-self.ksn2e[0,sn,nbs:nbf])
          zxvx.fill(0.0)
          for iw,comega in enumerate(ww):
            zxvx[iw,:,0:mbf-mbs,0:nbf-nbs] = (xvx * fmn)* (1.0/ (comega - emn) - 1.0 / (comega + emn))
      
          rf0 += einsum('wpmn,qmn->wpq', zxvx[...,0:mbf-mbs,0:nbf-nbs], xvx)

  return rf0


def rf0_cmplx_ref_blk(self, ww):
  """ Full matrix response in the basis of atom-centered product functions, and spin-resolved,
  or better said spin-summed """
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  v = einsum('pab->apb', self.pb.get_ac_vertex_array())
  #print('v.shape', v.shape)
  
  t1 = timer()
  if self.verbosity>1: print(__name__, 'self.ksn2e', self.ksn2e, len(ww))
      
  bsize = 40
  zxvx = zeros((len(ww), self.nprod,bsize,bsize), dtype=self.dtypeComplex)
  sn2e = self.ksn2e.reshape((self.nspin*self.norbs))
  sn2f = self.ksn2f.reshape((self.nspin*self.norbs))
  sn2x = self.x.reshape((self.nspin*self.norbs,self.norbs))

  nn = range(0,len(sn2x),bsize)+[len(sn2x)]
  #print('nn = ', nn, sn2x.shape, len(sn2x))

  for nbs,nbf in zip(nn,nn[1:]):
    vx = dot(v, sn2x[nbs:nbf,:].T)      
    for mbs,mbf in zip(nn,nn[1:]):
      xvx = einsum('mb,bpn->pmn', sn2x[mbs:mbf,:],vx)
      fmn = np.add.outer(-sn2f[mbs:mbf], sn2f[nbs:nbf])
      emn = np.add.outer( sn2e[mbs:mbf],-sn2e[nbs:nbf])
      zxvx.fill(0.0)
      for iw,comega in enumerate(ww):
        zxvx[iw,:,0:mbf-mbs,0:nbf-nbs] = (xvx * fmn) / (comega - emn)
      
      rf0 += einsum('wpmn,qmn->wpq', zxvx[...,0:mbf-mbs,0:nbf-nbs], xvx)

  t2 = timer()
  if self.verbosity>0: print(__name__, 'rf0_ref_blk', t2-t1)
  return rf0

def rf0_cmplx_ref(self, ww):
  """ Full matrix response in the basis of atom-centered product functions, and spin-resolved,
  or better said spin-summed """
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  v = self.pb.get_ac_vertex_array()
  
  t1 = timer()
  if self.verbosity>1: print(__name__, 'self.ksn2e', self.ksn2e, len(ww))
      
  zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
  sn2e = self.ksn2e.reshape((self.nspin*self.norbs))
  sn2f = self.ksn2f.reshape((self.nspin*self.norbs))
  sn2x = self.x.reshape((self.nspin*self.norbs,self.norbs))
  for en,fn,xn in zip(sn2e, sn2f, sn2x):
    vx = dot(v, xn)
    for em,fm,xm in zip(sn2e,sn2f,sn2x):
      vxx_a = dot(vx, xm.T)
      for iw,comega in enumerate(ww):
        zvxx_a[iw,:] = vxx_a * (fn - fm)/ (comega - (em - en))
      rf0 += einsum('wa,b->wab', zvxx_a, vxx_a)

  t2 = timer()
  if self.verbosity>0: print(__name__, 'rf0_ref_loop', t2-t1)
  return rf0

def rf0_cmplx_vertex_dp(self, ww):
  """ Full matrix response in the basis of atom-centered product functions """
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  v_arr = self.pb.get_dp_vertex_array()
  
  zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
  for n,(en,fn) in enumerate(zip(self.ksn2e[0,0,0:self.nfermi], self.ksn2f[0, 0, 0:self.nfermi])):
    vx = dot(v_arr, self.xocc[0][n,:])
    for m,(em,fm) in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
      if (fn - fm)<0 : break
      vxx_a = dot(vx, self.xvrt[0][m,:]) * self.cc_da
      for iw,comega in enumerate(ww):
        zvxx_a[iw,:] = vxx_a * (fn - fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
      rf0 = rf0 + einsum('wa,b->wab', zvxx_a, vxx_a)
  return rf0

def rf0_cmplx_vertex_ac(self, ww):
  """ Full matrix response in the basis of atom-centered product functions """
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  v = self.pb.get_ac_vertex_array()
  
  t1 = timer()
  if self.verbosity>1: print(__name__, 'self.ksn2e', self.ksn2e, len(ww))
  #print(self.ksn2e[0,0,0]-self.ksn2e)
  #print(self.ksn2f)
  #print(' abs(v).sum(), ww.sum(), self.nfermi, self.vstart ')
  #print(abs(v).sum(), ww.sum(), self.nfermi, self.vstart)
  
  zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
  for n,(en,fn) in enumerate(zip(self.ksn2e[0,0,0:self.nfermi[0]], self.ksn2f[0, 0, 0:self.nfermi[0]])):
    #if self.verbosity>1: print(__name__, 'n =', n)
    vx = dot(v, self.xocc[0][n,:])
    for m,(em,fm) in enumerate(zip(self.ksn2e[0,0,self.vstart[0]:],self.ksn2f[0,0,self.vstart[0]:])):
      if (fn - fm)<0 : break
      vxx_a = dot(vx, self.xvrt[0][m,:].T)
      for iw,comega in enumerate(ww):
        zvxx_a[iw,:] = vxx_a * (fn - fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
      rf0 += einsum('wa,b->wab', zvxx_a, vxx_a)

  t2 = timer()
  if self.verbosity>1: print(__name__, 'finished rf0', t2-t1)
  return rf0

