from __future__ import division, print_function
import numpy as np
from pyscf.nao.m_csphar import csphar
from pyscf.nao.m_log_interp import comp_coeffs

#
#
#
def coulomb_am(self, sp1, sp2, R1, R2):
  """
    Computes Coulomb overlap for an atom pair. The atom pair is given by a pair of species indices and the coordinates of the atoms.
    <a|r^-1|b> = \iint a(r)|r-r'|b(r')  dr dr'
    Args: 
      self: class instance of ao_matelem_c
      sp1,sp2 : specie indices, and
      R1,R2 :   respective coordinates
    Result:
      matrix of orbital overlaps
    The procedure uses the angular momentum algebra and spherical Bessel transform
    to compute the bilocal overlaps.
  """
  assert(sp1>-1)
  assert(sp2>-1)

  shape = [self.sp2norbs[sp] for sp in (sp1,sp2)]
  overlaps = np.zeros(shape)
  
  R2mR1 = np.array(R2)-np.array(R1)
  
  ylm = csphar( R2mR1, 2*self.jmx+1 )
  dist = np.sqrt(sum(R2mR1*R2mR1))
  cS = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype='complex128')
  cmat = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype='complex128')
  rS = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype='float64')

  f1f2_mom = np.zeros((self.nr), dtype='float64')
  l2S = np.zeros((2*self.jmx+1), dtype='float64')
  ir,coeffs = comp_coeffs(self.interp_rr, dist) 
  for mu2,l2,s2,f2 in self.sp2info[sp2]:
    for mu1,l1,s1,f1 in self.sp2info[sp1]:
      if self.sp_mu2rcut[sp1][mu1]+self.sp_mu2rcut[sp2][mu2]<dist: continue
      f1f2_mom = self.psi_log_mom[sp2][mu2,:] * self.psi_log_mom[sp1][mu1,:]/pp
      l2S.fill(0.0)
      for l3 in range( abs(l1-l2), l1+l2+1):
        f1f2_rea = self.sbt(f1f2_mom, l3, -1)
        l2S[l3] = (f1f2_rea[ir:ir+6]*coeffs).sum()*self.const
          
      cS.fill(0.0) 
      for m1 in range(-l1,l1+1):
        for m2 in range(-l2,l2+1):
          gc = self.get_gaunt(l1,-m1,l2,m2)
          m3 = m2-m1
          for l3ind,l3 in enumerate(range(abs(l1-l2),l1+l2+1)):
            if abs(m3) > l3 : continue
            cS[m1+self.jmx,m2+self.jmx] = cS[m1+self.jmx,m2+self.jmx] + l2S[l3]*ylm[ l3*(l3+1)+m3] * \
              gc[l3ind] * (-1.0)**((3*l1+l2+l3)//2+m2)
                  
      rS.fill(0.0)
      self.c2r_( l1,l2, self.jmx,cS,rS,cmat)
      overlaps[s1:f1,s2:f2] = 4*np.pi*rS[-l1+self.jmx:l1+1+self.jmx,-l2+self.jmx:l2+1+self.jmx]

  return overlaps
