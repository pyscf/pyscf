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

from __future__ import division, print_function
import numpy as np
from pyscf.nao.m_csphar import csphar
from pyscf.nao.m_log_interp import comp_coeffs

#
#
#
def laplace_am(self, sp1, R1, sp2, R2):
  """
    Computes brakets of Laplace operator for an atom pair. 
    The atom pair is given by a pair of species indices
    and the coordinates of the atoms.
    Args: 
      self: class instance of ao_matelem_c
      sp1,sp2 : specie indices, and
      R1,R2 :   respective coordinates
    Result:
      matrix of Laplace operator brakets
  """
  shape = [self.ao1.sp2norbs[sp] for sp in (sp1,sp2)]
  overlaps = np.zeros(shape)
  
  R2mR1 = np.array(R2)-np.array(R1)
  
  psi_log = self.ao1.psi_log
  psi_log_mom = self.ao1.psi_log_mom
  sp_mu2rcut = self.ao1.sp_mu2rcut
  sp2info = self.ao1.sp2info
  pp = self.ao1.pp
  rr = self.ao1.rr
  ylm = csphar( R2mR1, 2*self.jmx+1 )
  dist = np.sqrt(sum(R2mR1*R2mR1))
  cS = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype=np.complex128)
  cmat = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype=np.complex128)
  rS = np.zeros((self.jmx*2+1,self.jmx*2+1))
  j = self.jmx
  
  if(dist<1.0e-5): 
    
    for [mu1,l1,s1,f1],ff1 in zip(sp2info[sp1],psi_log[sp1]):
      ff1_diff = self.interp_rr.diff(ff1)
      for [mu2,l2,s2,f2],ff2 in zip(sp2info[sp2],psi_log[sp2]):
        ff2_diff = self.interp_rr.diff(ff2)
        cS.fill(0.0); rS.fill(0.0);
        if l1==l2 : 
          sum2 = sum(ff1_diff*ff2_diff*self.rr3_dr)+l1*(l1+1)*sum(ff1*ff2*rr)*self.dr_jt
          for m1 in range(-l1,l1+1): cS[m1+self.jmx,m1+self.jmx]=sum2
          self.c2r_( l1,l2, self.jmx,cS,rS,cmat)

        overlaps[s1:f1,s2:f2] = rS[-l1+j:l1+1+j,-l2+j:l2+1+j]

  else:

    f1f2_mom = np.zeros((self.nr))
    l2S = np.zeros((2*self.jmx+1))
    ir,coeffs = comp_coeffs(self.interp_rr, dist)
    for [mu2,l2,s2,f2],rcut2,ff2 in zip(sp2info[sp2],sp_mu2rcut[sp2],psi_log_mom[sp2]):
      for [mu1,l1,s1,f1],rcut1,ff1 in zip(sp2info[sp1],sp_mu2rcut[sp1],psi_log_mom[sp1]):
        if rcut1+rcut2<dist: continue
        f1f2_mom = ff2 * ff1 * self.pp2
        l2S.fill(0.0)
        for l3 in range( abs(l1-l2), l1+l2+1):
          f1f2_rea = self.sbt(f1f2_mom, l3, -1)
          l2S[l3] = (f1f2_rea[ir:ir+6]*coeffs).sum()
        
        l2S = l2S*self.const*4*np.pi
          
        cS.fill(0.0) 
        for m1 in range(-l1,l1+1):
          for m2 in range(-l2,l2+1):
            gc, m3 = self.get_gaunt(l1,-m1,l2,m2), m2-m1
            for l3ind,l3 in enumerate(range(abs(l1-l2),l1+l2+1)):
              if abs(m3) > l3 : continue
              cS[m1+j,m2+j] = cS[m1+j,m2+j] + l2S[l3]*ylm[ l3*(l3+1)+m3] * gc[l3ind] * (-1.0)**((3*l1+l2+l3)//2+m2)

        self.c2r_( l1,l2, self.jmx,cS,rS,cmat)
        overlaps[s1:f1,s2:f2] = rS[-l1+j:l1+j+1,-l2+j:l2+j+1]

  return overlaps
