from __future__ import print_function, division
import sys
import numpy as np
from numpy import stack, dot, zeros, einsum, array
from timeit import default_timer as timer
import scipy.linalg.blas as blas

from pyscf.nao.m_libnao import libnao
from ctypes import c_double, c_int64, c_int, POINTER

try:
  import numba as nb
  use_numba = True
except:
  use_numba = False

if use_numba:

    @nb.jit(nopython=True, parallel=False)
    def calc_part_rf0_numba(xvx, zxvx_re, zxvx_im):

        rf0_re = np.zeros((zxvx_re.shape[0], zxvx_re.shape[0]), dtype=np.float64)
        rf0_im = np.zeros((zxvx_re.shape[0], zxvx_re.shape[0]), dtype=np.float64)

        for ib in range(zxvx_im.shape[2]):
            rf0_tmp = zxvx_re[:, :, ib].dot(xvx[:, :, ib].T)
            rf0_re += rf0_tmp

            rf0_tmp = zxvx_im[:, :, ib].dot(xvx[:, :, ib].T)
            rf0_im += rf0_tmp

        return rf0_re + complex(0.0, 1.0)*rf0_im

    @nb.jit(nopython=True, parallel=False)
    def add_outer(a, b):
        c = np.zeros((a.size, b.size))

        for i in range(a.size):
            for j in range(b.size):
                c[i, j] = a[i] + b[j]
        return c

    @nb.jit(nopython=True, parallel=False)
    def rf0_den_numba(comega, X, ksn2f, ksn2e, pab2v_den, nprod, norbs, bsize, nspin,
                      nfermi, vstart):
      """
      Full matrix response in the basis of atom-centered product functions for
      parallel spins.

      Blas version to speed up matrix matrix multiplication
      spped up of 7.237 compared to einsum version for C20 system
      """
      
      rf0 = np.zeros((nprod, nprod), dtype=np.complex128)
      zxvx = np.zeros((nprod, bsize, bsize), dtype=np.complex128)

      for s in range(nspin):
        nn = list(range(0, nfermi[s], bsize)) + [nfermi[s]]
        mm = list(range(vstart[s], norbs, bsize)) + [norbs]
        
        for nbs, nbf in zip(nn, nn[1:]):

          vx = np.zeros((pab2v_den.shape[0], pab2v_den.shape[1], nbf-nbs))
          #vx_ref = dot(pab2v_den, X[s,nbs:nbf,:].T)
          for iv in range(pab2v_den.shape[0]):
              vx[iv, :, :] = np.dot(pab2v_den[iv, :, :], X[s,nbs:nbf,:].T)

          for mbs, mbf in zip(mm,mm[1:]):
            xvx = calc_XVX_numba(X[s,mbs:mbf,:], vx)

            fmn = add_outer(-ksn2f[0,s,mbs:mbf],  ksn2f[0,s,nbs:nbf])
            emn = add_outer( ksn2e[0,s,mbs:mbf], -ksn2e[0,s,nbs:nbf])
            zxvx.fill(0.0)
            zxvx[:,0:mbf-mbs,0:nbf-nbs] = (xvx * fmn)* (1.0/ (comega - emn) - 1.0 / (comega + emn))

            rf0_nb = calc_part_rf0_numba(xvx, zxvx[:, 0:mbf-mbs, 0:nbf-nbs].real,
                                         zxvx[:, 0:mbf-mbs, 0:nbf-nbs].imag)

            rf0 += rf0_nb
      return rf0

    @nb.jit(nopython=True, parallel=False)
    def calc_XVX_numba(X, VX):

        XVX = np.zeros((VX.shape[1], X.shape[0], VX.shape[2]))

        for i in range(XVX.shape[2]):

            XVX[:, :, i] = np.dot(VX[:, :, i].T, X.T)

        return XVX


    @nb.jit(nopython=True, parallel=True)
    def si_correlation(ww, X, kernel_sq, ksn2f, ksn2e, pab2v_den, nprod,
                       norbs, bsize, nspin, nfermi, vstart):
        """
        This computes the correlation part of the screened interaction W_c
        by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K
        or v_{ind}\sim W_{c} = (1-v\chi_{0})^{-1}v\chi_{0}v
        scr_inter[w,p,q], where w in ww, p and q in 0..self.nprod
        """

        si0 = np.zeros((ww.size, nprod, nprod), dtype=np.complex128)
        
        for iw in nb.prange(ww.size):
        #for iw in range(ww.size):
            
            rf0 = rf0_den_numba(ww[iw], X, ksn2f, ksn2e, pab2v_den,
                                nprod, norbs, bsize, nspin,
                                nfermi, vstart)

            # divide ww into complex(w) which is along imaginary
            # axis (real=0) and grid index(iw)

            #  kernel_sq or hkernel_den is bare coloumb or hartree, rf0
            # is \chi_{0}, so here k_c=v*chi_{0}
            k_c = np.dot(kernel_sq, rf0.real) + \
                complex(0.0, 1.0)*np.dot(kernel_sq, rf0.imag)

            # here v\chi_{0}v or k_c*v
            b = np.dot(k_c.real, kernel_sq) + complex(0.0, 1.0)*\
                np.dot(k_c.imag, kernel_sq)

            # here (1-v\chi_{0}) or 1-k_c. 1=eye(nprod)
            k_c = np.eye(nprod) - k_c

            # k_c * W = v\chi_{0}v = b --> W = np.linalg.solve(K_c,b)
            si0[iw,:,:] = np.linalg.solve(k_c, b)

        return si0

else:

    def si_correlation(rf0, si0, ww, kernel_sq, nprod):
        """
        This computes the correlation part of the screened interaction W_c
        by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K
        or v_{ind}\sim W_{c} = (1-v\chi_{0})^{-1}v\chi_{0}v
        scr_inter[w,p,q], where w in ww, p and q in 0..self.nprod
        """
        
        for iw, w in enumerate(ww):
            
            # divide ww into complex(w) which is along imaginary
            # axis (real=0) and grid index(iw)

            #  kernel_sq or hkernel_den is bare coloumb or hartree, rf0
            # is \chi_{0}, so here k_c=v*chi_{0}
            k_c = np.dot(kernel_sq, rf0[iw,:,:])

            # here v\chi_{0}v or k_c*v
            b = np.dot(k_c, kernel_sq)

            # here (1-v\chi_{0}) or 1-k_c. 1=eye(nprod)
            k_c = np.eye(nprod) - k_c

            # k_c * W = v\chi_{0}v = b --> W = np.linalg.solve(K_c,b)
            si0[iw,:,:] = np.linalg.solve(k_c, b)

        return si0

def calc_part_rf0(xvx, zxvx_re, zxvx_im):

    rf0_re = np.zeros((zxvx_re.shape[0], zxvx_re.shape[1], zxvx_re.shape[1]), dtype=np.float64)
    rf0_im = np.zeros((zxvx_re.shape[0], zxvx_re.shape[1], zxvx_re.shape[1]), dtype=np.float64)

    for iw in range(zxvx_re.shape[0]):
        for ib in range(zxvx_re.shape[3]):
            rf0_tmp = run_blasDGEMM(1.0, zxvx_re[iw, :, :, ib],
                                    xvx[:, :, ib], trans_a = 0, trans_b = 1)
            rf0_re[iw, :, :] += rf0_tmp

            rf0_tmp = run_blasDGEMM(1.0, zxvx_im[iw, :, :, ib],
                                    xvx[:, :, ib], trans_a = 0, trans_b = 1)
            rf0_im[iw, :, :] += rf0_tmp

    return rf0_re + complex(0.0, 1.0)*rf0_im


def rf0_den(self, ww):
  """
  Full matrix response in the basis of atom-centered product functions for
  parallel spins.

  Blas version to speed up matrix matrix multiplication
  spped up of 7.237 compared to einsum version for C20 system
  """
  
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  if hasattr(self, 'pab2v_den'):
    v = self.pab2v_den
  else:
    self.pab2v_den = v = einsum('pab->apb', self.pb.get_ac_vertex_array())
  
  zxvx = zeros((len(ww),self.nprod,self.bsize,self.bsize), dtype=self.dtypeComplex)

  print("enter rf0_den")
  t1 = timer()
  for s in range(self.nspin):
    nn = list(range(0,self.nfermi[s],self.bsize))+[self.nfermi[s]]
    mm = list(range(self.vstart[s],self.norbs,self.bsize))+[self.norbs]
    
    for nbs,nbf in zip(nn,nn[1:]):
      # replace this dot with a blas call to dgemv or zgemv
      vx = dot(v, self.x[s,nbs:nbf,:].T)
      #vx = blas.dgemv(v, self.x[s,nbs:nbf,:].T)
      for mbs,mbf in zip(mm,mm[1:]):
        xvx = calc_XVX(self.x[s,mbs:mbf,:], vx)

        fmn = np.add.outer(-self.ksn2f[0,s,mbs:mbf], self.ksn2f[0,s,nbs:nbf])
        emn = np.add.outer( self.ksn2e[0,s,mbs:mbf],-self.ksn2e[0,s,nbs:nbf])
        zxvx.fill(0.0)
        for iw, comega in enumerate(ww):
          zxvx[iw,:,0:mbf-mbs,0:nbf-nbs] = (xvx * fmn)* (1.0/ (comega - emn) - 1.0 / (comega + emn))

        t3 = timer()
        rf0_nb = calc_part_rf0(xvx, zxvx[:, :, 0:mbf-mbs, 0:nbf-nbs].real,
                               zxvx[:, :, 0:mbf-mbs, 0:nbf-nbs].imag)
        t4 = timer()
        print("end calc_part_rf0_numba: ", t4-t3)

        rf0 += rf0_nb
  t2 = timer()
  print("spend time rf0_den: ", t2-t1)

  return rf0


def run_blasZGEMM(alpha, A, B, **kwargs):

  return blas.zgemm(alpha, A, B, **kwargs)

def run_blasDGEMM(alpha, A, B, **kwargs):

  return blas.dgemm(alpha, A, B, **kwargs)


def rf0_den_einsum(self, ww):
  """
  Full matrix response in the basis of atom-centered product functions for 
  parallel spins
  
  einsum version, slow
  """
  
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  if hasattr(self, 'pab2v_den'):
    v = self.pab2v_den
  else:
    self.pab2v_den = v = einsum('pab->apb', self.pb.get_ac_vertex_array())
  
  zxvx = zeros((len(ww),self.nprod,self.bsize,self.bsize), dtype=self.dtypeComplex)

  for s in range(self.nspin):
    nn = list(range(0,self.nfermi[s],self.bsize))+[self.nfermi[s]]
    mm = list(range(self.vstart[s],self.norbs,self.bsize))+[self.norbs]
    
    for nbs,nbf in zip(nn,nn[1:]):
      vx = dot(v, self.x[s,nbs:nbf,:].T)
      for mbs,mbf in zip(mm,mm[1:]):
        xvx = einsum('mb,bpn->pmn', self.x[s,mbs:mbf,:],vx)

        fmn = np.add.outer(-self.ksn2f[0,s,mbs:mbf], self.ksn2f[0,s,nbs:nbf])
        emn = np.add.outer( self.ksn2e[0,s,mbs:mbf],-self.ksn2e[0,s,nbs:nbf])
        zxvx.fill(0.0)
        for iw,comega in enumerate(ww):
          zxvx[iw,:,0:mbf-mbs,0:nbf-nbs] = (xvx * fmn)* (1.0/ (comega - emn) - 1.0 / (comega + emn))
      
        rf0 += einsum('wpmn,qmn->wpq', zxvx[...,0:mbf-mbs,0:nbf-nbs], xvx)
  
  return rf0

def calc_XVX(X, VX):

    XVX = np.zeros((VX.shape[1], X.shape[0], VX.shape[2]))

    for i in range(XVX.shape[2]):

        XVX[:, :, i] = blas.dgemm(1.0, VX[:, :, i], X, trans_a=1, trans_b=1)

    return XVX


def rf0_cmplx_ref_blk(self, ww):
  """ Full matrix response in the basis of atom-centered product functions """
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
  """ Full matrix response in the basis of atom-centered product functions """
  rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
  v = self.pb.get_ac_vertex_array()
  
  t1 = timer()
  if self.verbosity>1: print(__name__, 'self.ksn2e', self.ksn2e, len(ww))
      
  zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
  for s in range(self.nspin):
    n2e = self.ksn2e[0,s,:]
    n2f = self.ksn2f[0,s,:]
    n2x = self.x[s,:,:]
    for en,fn,xn in zip(n2e,n2f,n2x):
      vx = dot(v, xn)
      for em,fm,xm in zip(n2e,n2f,n2x):
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

