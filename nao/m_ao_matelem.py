from __future__ import division
import numpy as np

from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_sbt import sbt_c

#
#
#
class ao_matelem_c(sbt_c):
  '''
  Evaluator of matrix elements
  '''
  def __init__(self, ao_log):

    self.psi_log = ao_log.psi_log
    self.sp_mu2j = ao_log.sp_mu2j
    
    self.jmx  = np.amax(self.sp_mu2j)
    self.species  = range(len(ao_log.sp2nmult))
    self.sp2mults = [ range(ao_log.sp2nmult[sp]) for sp in self.species ]
    
    self.sp2norbs = [ sum(self.sp_mu2j[sp,0:ao_log.sp2nmult[sp]]*2+1) for sp in self.species ]

    sbt_c.__init__(self, ao_log.rr, ao_log.pp)
    self.psi_log_mom = np.zeros(self.psi_log.shape)

    for sp in self.species:
      for mu,am in zip(self.sp2mults[sp], self.sp_mu2j[sp]):
        self.psi_log_mom[sp,mu,:] = self.exe( self.psi_log[sp,mu,:], am, 1)
    
    #print(self.psi_log_mom)

  #
  #
  #
  def get_overlap(self, sp1, sp2, R1, R2):
    assert(sp1>-1)
    assert(sp2>-1)

    shape = [self.sp2norbs[sp] for sp in (sp1,sp2)]
    oo2over = np.zeros(shape)
    
    R2mR1 = R2-R1
    ylm = np.csphar( R2mR1, 2*self.jmx+1 )
    dist = np.sqrt(sum(R2mR1*R2mR1))
        
    
