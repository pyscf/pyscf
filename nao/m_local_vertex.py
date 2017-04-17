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

#
#
#
class local_vertex_c(ao_matelem_c):
  '''
    Constructor of the local product functions and the product vertex coefficients.
  '''
  def __init__(self, ao_log):
    ao_matelem_c.__init__(self, ao_log)
    
    self.sp_j2nf = []
    for sp in self.species:
      self.sp_j2nf.append(np.zeros((2*max(self.sp_mu2j[sp,:])+1), dtype='int64'))
      for mu1,j1,s1,f1 in self.sp2info[sp]:
        for mu2,j2,s2,f2 in self.sp2info[sp]:
          for j in range(abs(j1-j2),j1+j2+1,2):
            self.sp_j2nf[sp][j] = self.sp_j2nf[sp][j] + 1

    self.sp_j_p2mus = []
    for sp in self.species:
      jmx_sp = max(self.sp_mu2j[sp,:])
      self.sp_j_p2mus.append( [ [p for p in range(self.sp_j2nf[sp][j]) ] for j in range(2*jmx_sp+1)])
      j2p = np.zeros((2*max(self.sp_mu2j[sp,:])+1), dtype='int64')
      for mu1,j1,s1,f1 in self.sp2info[sp]:
        for mu2,j2,s2,f2 in self.sp2info[sp]:
          for j in range(abs(j1-j2),j1+j2+1,2):
            self.sp_j_p2mus[sp][j][j2p[j]] = [mu1,mu2]
            j2p[j] = j2p[j] + 1
        
  #
  #
  #
  def get_local_vertex(self, sp):
    """
      Constructor of vertex for a given specie
    """
    assert(sp>-1)
    
    return 0
