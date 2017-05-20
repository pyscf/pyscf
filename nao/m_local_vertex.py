from __future__ import print_function, division
import numpy as np
import sys
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_pack2den import ij2pack
from scipy.linalg import eigh
from timeit import default_timer as timer

#
#
#
class local_vertex_c(ao_matelem_c):
  '''
    Constructor of the local product functions and the product vertex coefficients.
  '''
  def __init__(self, ao_log):
    ao_matelem_c.__init__(self, ao_log)
    nr = self.nr
    kk = self.kk
    self.dkappa_pp = 4*np.pi*np.log( kk[1]/kk[0])*kk
    self.c2r_c = c2r_c(2*self.jmx) # local vertex c2r[:,:] coefficients
    
  #
  #
  #
  def get_local_vertex(self, sp):
    """
      Constructor of vertex for a given specie
      Args:
        sp : specie number
      Result:
        Dictionary with the product functions, vertex coefficients and eigenvalues
        in each angular-momentum "sektor"
          dominant products functions: j2xff
          dominant vertex coefficients (angular part of): j2xww
          eigenvalues of Coulomb metric : j2eva
      
    """
    assert(sp>-1)

    mu2s = self.ao1.sp_mu2s[sp]
    mu2j = self.ao1.sp_mu2j[sp]
    info = self.ao1.sp2info[sp]
    mu2ff = self.ao1.psi_log[sp]
    no = self.ao1.sp2norbs[sp]
    nmu = self.ao1.sp2nmult[sp]
    
    jmx_sp = np.amax(mu2j)
    j2nf=np.zeros((2*jmx_sp+1), dtype='int64')
    for mu1,j1,s1,f1 in info:
      for mu2,j2,s2,f2 in info:
        if mu2<mu1: continue
        for j in range(abs(j1-j2),j1+j2+1,2): 
          j2nf[j] = j2nf[j] + 1
    
    j_p2mus = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j_p2js  = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j2p = np.zeros((2*jmx_sp+1), dtype='int64')
    for mu1,j1,s1,f1 in info:
      for mu2,j2,s2,f2 in info:
        if mu2<mu1: continue
        for j in range(abs(j1-j2),j1+j2+1,2):
          j_p2mus[j][j2p[j]] = [mu1,mu2]
          j_p2js[j][j2p[j]] = [j1,j2]
          j2p[j]+=1

    pack2ff = np.zeros((nmu*(nmu+1)//2,self.nr), dtype='float64') # storage for original products
    for mu2 in range(nmu):
      for mu1 in range(mu2+1): pack2ff[ij2pack(mu1,mu2),:] = mu2ff[mu1,:]*mu2ff[mu2,:]
    
    j2xff = [] # Storage for dominant product's functions (list of numpy arrays: x*f(r)*f(r))
    j2xww = [] # Storage for dominant product's vertex (angular part of: x*wigner*wigner)
    j2eva = [] # Storage for eigenvalues in each angular momentum "sektor"
    hc_c2r = np.conj(self.c2r_c._c2r).transpose()
    c2r_jm = self.c2r_c._j
    jc = self._j
    t1 = 0
    tstart = timer()
    xww2 = np.zeros((no,no), dtype=np.complex128)
    xww3 = np.zeros((no,no), dtype=np.complex128)
    for j,dim in enumerate(j2nf): # Metrik ist dim * dim in diesem Sektor
      lev2ff = np.zeros((dim,self.nr))
      for lev in range(dim): lev2ff[lev,:] = self.sbt(pack2ff[ ij2pack( *j_p2mus[j][lev] ),:], j, 1)
      metric = np.zeros((dim,dim), dtype='float64')
      for level_1 in range(dim):
        for level_2 in range(level_1+1):
          metric[level_2,level_1]=metric[level_1,level_2]=(lev2ff[level_1,:]*lev2ff[level_2,:]*self.dkappa_pp).sum()  # Coulomb Metrik enthaelt Faktor 1/p**2

      eva,x=eigh(metric)
      j2eva.append(eva)

      xff = np.zeros((dim,self.nr))   #!!!! Jetzt dominante Orbitale bilden
      for domi in range(dim):
        for n in range(dim):
          xff[domi,:] = xff[domi,:] + x[n,domi]*pack2ff[ij2pack(*j_p2mus[j][n]),:]
      j2xff.append(xff)
      
      #xe = np.zeros((dim,dim))
      #for i,ev in enumerate(eva): xe[:,i] = x[:,i]*ev
      #metric1 = np.matmul(x,xe.transpose())
      #print(sum(sum(abs(metric1-metric1.transpose()))), sum(sum(abs(metric1-metric))))

      kinematical_vertex = np.zeros((dim, 2*j+1, no, no), dtype='float64')
      for num,[[mu1,mu2], [j1,j2]] in enumerate(zip(j_p2mus[j],j_p2js[j])):
        if j<abs(j1-j2) or j>j1+j2 : continue
        for m1,o1 in zip(range(-j1,j1+1), range(mu2s[mu1],mu2s[mu1]+2*j1+1)):
          for m2,o2 in zip(range(-j2,j2+1), range(mu2s[mu2],mu2s[mu2]+2*j2+1)):
            m=m1+m2
            if abs(m)>j: continue
            i3y=self.get_gaunt(j1,m1,j2,m2)*(-1.0)**m
            kinematical_vertex[num,j+m,o2,o1] = kinematical_vertex[num,j+m,o1,o2] = i3y[j-abs(j1-j2)]
      
      xww = np.zeros((dim, 2*j+1, no, no), dtype='float64')
      
      for domi in range(dim):
        xww0 = np.zeros((2*j+1, no, no), dtype='float64')
        for num in range(dim): 
          xww0[:,:,:] = xww0[:,:,:] + x[num,domi]*kinematical_vertex[num,:,:,:]
                
        xww1 = np.zeros((2*j+1, no, no), dtype='complex128')
        for m in range(-j,j+1):
          for m1 in range(-abs(m),abs(m)+1,2*abs(m) if m!=0 else 1):
            xww1[j+m,:,:]=xww1[j+m,:,:]+hc_c2r[c2r_jm+m1,c2r_jm+m]*xww0[j+m1,:,:]

        t1s = timer()
        for m in range(-j,j+1):
          
          xww2.fill(0.0)
          for mu1,j1,s1,f1 in info:
            for m1 in range(-j1,j1+1):
              for n1 in range(-abs(m1),abs(m1)+1,2*abs(m1) if m1!=0 else 1):
                xww2[s1+m1+j1,:]=xww2[s1+m1+j1,:]+self._c2r[m1+jc,n1+jc] * xww1[j+m,s1+n1+j1,:]

          xww3.fill(0.0)
          for mu2,j2,s2,f2 in info:
            for m2 in range(-j2,j2+1):
              for n2 in range(-abs(m2),abs(m2)+1,2*abs(m2) if m2!=0 else 1):
                xww3[:,s2+m2+j2]=xww3[:,s2+m2+j2]+self._c2r[m2+jc,n2+jc] * xww2[:,s2+n2+j2]
                            
          xww[domi,j+m,:,:] = xww3[:,:].real
          
        t1 = t1 + (timer()-t1s)
      j2xww.append(xww)
    tfinish = timer()
    #print(tfinish-tstart, t1)
    
    return {"j2xww": j2xww, "j2xff": j2xff, "j2eva": j2eva }
