# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division
import numpy as np
import sys
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_pack2den import ij2pack_u
from scipy.linalg import eigh
from timeit import default_timer as timer

#
#
#
class local_vertex_c(ao_matelem_c):
  ''' Constructor of the local product functions and the product vertex coefficients. '''
  def __init__(self, ao_log):
    ao_matelem_c.__init__(self, ao_log.rr, ao_log.pp)
    self.init_one_set(ao_log) # @classmethod ???
    self.dkappa_pp = 4*np.pi*np.log( self.kk[1]/self.kk[0])*self.kk
    self.c2r_c = c2r_c(2*self.jmx) # local vertex c2r[:,:] coefficients
    
  #
  def get_local_vertex(self, sp):
    from pyscf.nao.m_thrj import thrj
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
    j2nf=np.zeros((2*jmx_sp+1), dtype=int) # count number of radial functions products per angular momentum
    for mu1,j1,s1,f1 in info:
      for mu2,j2,s2,f2 in info:
        if mu2<mu1: continue
        for j in range(abs(j1-j2),j1+j2+1,2): 
          j2nf[j] = j2nf[j] + 1
    
    j_p2mus = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j_p2js  = [ [p for p in range(j2nf[j]) ] for j in range(2*jmx_sp+1)]
    j2p = np.zeros((2*jmx_sp+1), dtype=int)
    for mu1,j1,s1,f1 in info:
      for mu2,j2,s2,f2 in info:
        if mu2<mu1: continue
        for j in range(abs(j1-j2),j1+j2+1,2):
          j_p2mus[j][j2p[j]] = [mu1,mu2]
          j_p2js[j][j2p[j]] = [j1,j2]
          j2p[j]+=1
    
    pack2ff = np.zeros((nmu*(nmu+1)//2,self.nr)) # storage for original products
    for mu2 in range(nmu):
      for mu1 in range(mu2+1): pack2ff[ij2pack_u(mu1,mu2),:] = mu2ff[mu1,:]*mu2ff[mu2,:]
    
    j2xff     = [] # Storage for dominant product's functions (list of numpy arrays: x*f(r)*f(r))
    j2xww     = [] # Storage for dominant product's vertex (angular part of: x*wigner*wigner)
    j2eva     = [] # Storage for eigenvalues in each angular momentum "sector"
    t1 = 0
    tstart = timer()
    for j,dim in enumerate(j2nf): # Metrik ist dim * dim in diesem Sektor
      lev2ff = np.zeros((dim,self.nr))
      for lev in range(dim): lev2ff[lev,:] = self.sbt(pack2ff[ ij2pack_u( *j_p2mus[j][lev] ),:], j, 1)
      metric = np.zeros((dim,dim))
      for lev_1 in range(dim):
        for lev_2 in range(lev_1+1):
          metric[lev_2,lev_1]=metric[lev_1,lev_2]=(lev2ff[lev_1,:]*lev2ff[lev_2,:]*self.dkappa_pp).sum()  # Coulomb Metrik enthaelt Faktor 1/p**2

      eva,x=eigh(metric)
      j2eva.append(eva)

      xff = np.zeros((dim,self.nr))   #!!!! Jetzt dominante Orbitale bilden
      for domi in range(dim):
        for n in range(dim):
          xff[domi,:] = xff[domi,:] + x[n,domi]*pack2ff[ij2pack_u(*j_p2mus[j][n]),:]
      j2xff.append(xff)
      
      kinematical_vertex = np.zeros((dim, 2*j+1, no, no)) # Build expansion coefficients V^ab_mu defined by f^a(r) f^b(r) = V^ab_mu F^mu(r) 
      for num,[[mu1,mu2], [j1,j2]] in enumerate(zip(j_p2mus[j],j_p2js[j])):
        if j<abs(j1-j2) or j>j1+j2 : continue
        for m1,o1 in zip(range(-j1,j1+1), range(mu2s[mu1],mu2s[mu1+1])):
          for m2,o2 in zip(range(-j2,j2+1), range(mu2s[mu2],mu2s[mu2+1])):
            m=m1+m2
            if abs(m)>j: continue
            i3y=self.get_gaunt(j1,m1,j2,m2)*(-1.0)**m
            kinematical_vertex[num,j+m,o2,o1] = kinematical_vertex[num,j+m,o1,o2] = i3y[j-abs(j1-j2)]
      
      xww = np.zeros((dim, 2*j+1, no, no))
      for domi in range(dim):
        xww0 = np.einsum('n,nmab->mab', x[:,domi], kinematical_vertex[:,:,:,:])
        xww[domi,:,:,:] = self.c2r_c.c2r_moo(j, xww0, info)
      j2xww.append(xww)

    #tfinish = timer()
    #print(tfinish-tstart, t1)
    
    return {"j2xww": j2xww, "j2xff": j2xff, "j2eva": j2eva}
