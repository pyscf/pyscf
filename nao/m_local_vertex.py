from __future__ import print_function
from __future__ import division
import numpy as np
import sys
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_gaunt import gaunt_c
from pyscf.nao.m_csphar import csphar
from pyscf.nao.m_log_interp import log_interp
from pyscf.nao.m_ao_matelem import ao_matelem_c
from scipy.linalg import eigh

#
#
#
def ij2pack(i,j):
  ma = max(i,j)
  return ma*(ma+1)//2+min(i,j)

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
    self.dkappa_pp = 4*np.pi*np.log( kk[nr-1]/kk[0])/(nr-1)*kk
    self.c2r_c = c2r_c(2*self.jmx)
    
  #
  #
  #
  def get_local_vertex(self, sp):
    """
      Constructor of vertex for a given specie
    """
    assert(sp>-1)

    jmx_sp = max(self.sp_mu2j[sp,:])
    j2nf=np.zeros((2*jmx_sp+1), dtype='int64')
    for mu1,j1,s1,f1 in self.sp2info[sp]:
      for mu2,j2,s2,f2 in self.sp2info[sp]:
        for j in range(abs(j1-j2),j1+j2+1,2): j2nf[j] = j2nf[j] + 1

    j_p2mus = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j_p2js  = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j2p = np.zeros((2*jmx_sp+1), dtype='int64')
    for mu1,j1,s1,f1 in self.sp2info[sp]:
      for mu2,j2,s2,f2 in self.sp2info[sp]:
        for j in range(abs(j1-j2),j1+j2+1,2):
          j_p2mus[j][j2p[j]] = [mu1,mu2]
          j_p2js[j][j2p[j]] = [j1,j2]
          j2p[j] = j2p[j] + 1

    # 
    nmu = len(self.sp2mults[sp])
    pack2ff = np.zeros((nmu*(nmu+1)//2,self.nr), dtype='float64') # storage for original products
    for mu2 in self.sp2mults[sp]:
      for mu1 in range(mu2+1):
        pack2ff[ij2pack(mu1,mu2),:] = self.psi_log[sp,mu1,:]*self.psi_log[sp,mu2,:]
    
    j2xff = [] # Storage for dominant product's functions (list of numpy arrays: x*f(r)*f(r))
    j2xww = [] # Storage for dominant product's vertex (angular part of: x*wigner*wigner)
    j2eva = [] # Storage for eigenvalues in each angular momentum "sektor"
    hc_c2r = np.conj(self.c2r_c._c2r)
    c2r_jm = self.c2r_c._j
    for j,dim in enumerate(j2nf): # Metrik ist dim * dim in diesem Sektor 

      metric = np.zeros((dim,dim), dtype='float64')
      for level_1 in range(dim):
        ff12_p = self.sbt(pack2ff[ ij2pack( *j_p2mus[j][level_1] ),:], j, 1)
        for level_2 in range(level_1+1):
          ff34_p = self.sbt(pack2ff[ ij2pack( *j_p2mus[j][level_2] ),:], j, 1)
          metric[level_2,level_1]=metric[level_1,level_2]=sum(ff12_p*ff34_p*self.dkappa_pp)  # Coulomb Metrik enthaelt Faktor 1/p**2

      eva,x=eigh(metric)
      j2eva.append(eva)
        
      xff = np.zeros((dim,self.nr))   #!!!! Jetzt dominante Orbitale bilden
      for domi in range(dim):  
        for n in range(dim):
          xff[domi,:] = xff[domi,:] + x[domi,n]*pack2ff[ij2pack(*j_p2mus[j][n]),:]
      j2xff.append(xff)

      xww = np.zeros((dim, (jmx_sp+1)**2, (jmx_sp+1)**2, 2*(jmx_sp*2)+1), dtype='float64')
      for domi in range(dim):
        xg0 = np.zeros(((jmx_sp+1)**2, (jmx_sp+1)**2, 2*(jmx_sp*2)+1), dtype='float64')
        for n in range(dim):
          j1,j2 = j_p2js[j][n]
          if j<abs(j1-j2) or j>j1+j2 : continue
          for m1 in range(-j1,j1+1) : 
            jm1 = j1*(j1+1)+m1
            for m2 in range(-j2,j2+1): 
              m,jm2 = m1+m2,j2*(j2+1)+m2
              i3y=self.get_gaunt(j1,m1,j2,m2)*(-1.0)**m
              xg0[jm1,jm2,2*jmx_sp+m] = xg0[jm1,jm2,2*jmx_sp+m] + x[domi,n]*i3y[j-abs(j1-j2)]

        xg1 = np.zeros((2*(jmx_sp*2)+1,(jmx_sp+1)**2,(jmx_sp+1)**2), dtype='complex128')
        for m in range(-j,j+1):
          for m1 in range(-j,j+1):
            xg1[2*jmx_sp+m,:,:]=xg1[2*jmx_sp+m,:,:] + hc_c2r[c2r_jm+m1,c2r_jm+m]*xg0[:,:,2*jmx_sp+m1]

        for m in range(-j,j+1):
          xg2 = np.zeros(((jmx_sp+1)**2,(jmx_sp+1)**2), dtype='complex128')
          for j1 in range(jmx_sp+1):
            for m1 in range(-j1,j1+1):
              jm1 = j1*(j1+1)+m1
              for j2 in range(jmx_sp+1):
                for m2 in range(-j2,j2+1):
                  jm2 = j2*(j2+1)+m2
                  for n1 in range(-j1,j1+1):
                    jn1 = j1*(j1+1)+n1
                    for n2 in range(-j2,j2+1):
                      jn2 = j2*(j2+1)+n2
                      xg2[jm1,jm2]=xg2[jm1,jm2]+self._c2r[m1,n1]*self._c2r[m2,n2] * xg1[2*jmx_sp+m,jn1,jn2]
          xww[domi,:,:,2*jmx_sp+m] = xg2[:,:].real
      
      j2xww.append(xww)
    return {"j2xww": j2xww, "j2xff": j2xff, "j2eva": j2eva }
