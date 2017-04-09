from __future__ import division
import numpy as np
import sys
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_gaunt import gaunt_c
from pyscf.nao.m_csphar import csphar
from pyscf.nao.m_log_interp import log_interp
#
#
#
class ao_matelem_c(sbt_c, c2r_c, gaunt_c):
  '''
  Evaluator of matrix elements
  '''
  def __init__(self, ao_log):

    self.jmx  = np.amax(ao_log.sp_mu2j)

    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao_log.rr, ao_log.pp)
    gaunt_c.__init__(self, self.jmx)
    
    self.psi_log = ao_log.psi_log
    self.sp_mu2j = ao_log.sp_mu2j
    self.sp_mu2rcut = ao_log.sp_mu2rcut
    
    self.species  = range(len(ao_log.sp2nmult))
    self.sp2mults = [ range(ao_log.sp2nmult[sp]) for sp in self.species ]

    self.sp2norbs = [ sum(self.sp_mu2j[sp,0:ao_log.sp2nmult[sp]]*2+1) for sp in self.species ]
    shape = (self.sp_mu2j.shape[0],self.sp_mu2j.shape[1]+1)
    self.sp_mu2s = np.zeros(shape, dtype='int32')
    for sp in self.species:
      for mu in self.sp2mults[sp]:
        self.sp_mu2s[sp,mu+1] = sum(2*self.sp_mu2j[sp,0:mu+1]+1)

    self.sp2info = []
    for sp in self.species:
      self.sp2info.append([
        [mu, self.sp_mu2j[sp][mu], self.sp_mu2s[sp][mu], self.sp_mu2s[sp,mu+1]] for mu in self.sp2mults[sp] ])

#    for sp in self.species:
#      for mu in self.sp2info[sp]:
#        print(sp, mu)

    self.psi_log_mom = np.zeros(self.psi_log.shape)

    for sp in self.species:
      for mu,am in zip(self.sp2mults[sp], self.sp_mu2j[sp]):
        self.psi_log_mom[sp,mu,:] = self.sbt( self.psi_log[sp,mu,:], am, 1)
    
    dr = np.log(ao_log.rr[1]/ao_log.rr[0])
    self.rr3_dr = ao_log.rr**3 * dr
    self.four_pi = 4*np.pi
    self.const = np.sqrt(np.pi/2.0)

    #print(self.psi_log_mom)

  #
  #
  #
  def get_overlap(self, sp1, sp2, R1, R2):
    assert(sp1>-1)
    assert(sp2>-1)

    shape = [self.sp2norbs[sp] for sp in (sp1,sp2)]
    overlaps = np.zeros(shape)
    
    R2mR1 = R2-R1
    ylm = csphar( R2mR1, 2*self.jmx+1 )
    dist = np.sqrt(sum(R2mR1*R2mR1))

    cS = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype='complex128')
    cmat = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype='complex128')
    rS = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype='float64')

    if(dist<1.0e-5):

      for mu1,l1 in zip(self.sp2mults[sp1], self.sp_mu2j[sp1,:]):
        s1,f1=self.sp_mu2s[sp1,mu1],self.sp_mu2s[sp1,mu1+1]
        for mu2,l2 in zip(self.sp2mults[sp2], self.sp_mu2j[sp2,:]):
          s2,f2=self.sp_mu2s[sp2,mu2],self.sp_mu2s[sp2,mu2+1]
          cS.fill(0.0); rS.fill(0.0);
          if l1==l2 : 
            sum1 = sum(self.psi_log[sp1,mu1,:]*self.psi_log[sp2,mu2,:] * self.rr3_dr)
            for m1 in range(-l1,l1+1): cS[m1+self.jmx,m1+self.jmx]=sum1
            self.c2r_( l1,l2, self.jmx,cS,rS,cmat)

          overlaps[s1:f1,s2:f2] = rS[-l1+self.jmx:l1+1+self.jmx,-l2+self.jmx:l2+1+self.jmx]

    else:

      f1f2_mom = np.zeros((self.nr), dtype='float64')
      l2S = np.zeros((2*self.jmx+1), dtype='float64')
      for mu1,l1 in zip(self.sp2mults[sp1], self.sp_mu2j[sp1,:]):
        s1,f1=self.sp_mu2s[sp1,mu1],self.sp_mu2s[sp1,mu1+1]
        for mu2,l2 in zip(self.sp2mults[sp2], self.sp_mu2j[sp2,:]):
          if self.sp_mu2rcut[sp1,mu1]+self.sp_mu2rcut[sp2,mu2]<dist: continue
          s2,f2=self.sp_mu2s[sp2,mu2],self.sp_mu2s[sp2,mu2+1]
          rS.fill(0.0); cS.fill(0.0);
          f1f2_mom = self.psi_log_mom[sp2,mu2,:] * self.psi_log_mom[sp1,mu1,:]
          l2S.fill(0.0)
          for l in range( abs(l1-l2), l1+l2+1):
            f1f2_rea = self.sbt(f1f2_mom, l,-1)
            l2S[l] = log_interp(f1f2_rea, dist, self.rhomin, self.dr_jt)*self.const*self.four_pi
          
          cS.fill(0.0)
          rS.fill(0.0)
          for m1 in range(-l1,l1+1):
            for m2 in range(-l2,l2+1):
              gc = self.gaunt(l1,-m1,l2,m2)
              m3 = m2-m1
              for l3ind,l3 in enumerate(range(abs(l1-l2),l1+l2+1)):
                if abs(m3) > l3 : continue
                cS[m1+self.jmx,m2+self.jmx] = cS[m1+self.jmx,m2+self.jmx] + l2S[l3]*ylm[ l3*(l3+1)+m3] * \
                  gc[l3ind] * (-1.0)**((3*l1+l2+l)/2+m2)
          
          self.c2r_( l1,l2, self.jmx,cS,rS,cmat)
          overlaps[s1:f1,s2:f2] = rS[-l1+self.jmx:l1+1+self.jmx,-l2+self.jmx:l2+1+self.jmx]

    return overlaps
