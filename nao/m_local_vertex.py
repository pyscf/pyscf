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
    pack2orig_prd = np.zeros((nmu*(nmu+1)//2,self.nr), dtype='float64') # storage for original products
    for mu2 in self.sp2mults[sp]:
      for mu1 in range(mu2+1):
        pack2orig_prd[ij2pack(mu1,mu2),:] = self.psi_log[sp,mu1,:]*self.psi_log[sp,mu2,:]
    
    for j,dim in enumerate(j2nf): # Metrik ist dim * dim in diesem Sektor 
      
      kinematical_vertex = np.zeros((self.sp2norbs[sp],self.sp2norbs[sp],dim,2*(2*jmx_sp)+1), dtype='float64')
      for nummer in range(dim):
        mu1,mu2,j1,j2 = *j_p2mus[j][nummer],*j_p2js[j][nummer]
        s1,s2 = self.sp_mu2s[sp,mu1],self.sp_mu2s[sp,mu2]
        #print(nummer,'|',mu1,j1,s1+j1-j1,s1+j1+j1,'|', mu2,j2,s2+j2-j2,s2+j2+j2)
        if  abs(j1-j2)<=j and j<=j1+j2:
          for m1 in range(-j1,j1+1):
            for m2 in range(-j2,j2+1):
              m=m1+m2
              i3y = self.get_gaunt(j1,m1,j2,m2)*(-1.0)**m
              kinematical_vertex[s1+j1+m1,s2+j2+m2,nummer,m]=i3y[j-abs(j1-j2)]
              kinematical_vertex[s2+j2+m2,s1+j1+m1,nummer,m]=kinematical_vertex[s1+j1+m1,s2+j2+m2,nummer,m]

      metric = np.zeros((dim,dim), dtype='float64')
      for level_1 in range(dim):
        psi12_p = self.sbt(pack2orig_prd[ ij2pack( *j_p2mus[j][level_1] ),:], j, 1)
        for level_2 in range(level_1+1):
          psi34_p = self.sbt(pack2orig_prd[ ij2pack( *j_p2mus[j][level_2] ),:], j, 1)
          metric[level_2,level_1]=metric[level_1,level_2]=sum(psi12_p*psi34_p*self.dkappa_pp)  # Coulomb Metrik enthaelt Faktor 1/p**2

      E,X=eigh(metric)
      print(j, dim, E)
      
    return 0
