from __future__ import division, print_function
import numpy as np
import sys
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_gaunt import gaunt_c
from pyscf.nao.m_log_interp import log_interp_c

#
#
#
def build_3dgrid(me, sp1, R1, sp2, R2, level=3):
  from pyscf import dft
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_gauleg import leggauss_ab

  if ( (R1-R2)**2 ).sum()<1e-7 :
    mol = system_vars_c(atom=[ [int(me.sp2charge[sp1]), R1] ])
  else :
    mol = system_vars_c(atom=[ [int(me.sp2charge[sp1]), R1], [int(me.sp2charge[sp2]), R2] ])

  atom2rcut=np.array([me.sp_mu2rcut[sp].max() for sp in (sp1,sp2)])
  grids = dft.gen_grid.Grids(mol)
  grids.level = level # precision as implemented in pyscf
  grids.radi_method=leggauss_ab
  grids.build(atom2rcut=atom2rcut)
  return grids


#
#
#
class ao_matelem_c(sbt_c, c2r_c, gaunt_c):
  '''
  Evaluator of matrix elements given by the numerical atomic orbitals.
  The class will contain 
    the Gaunt coefficients, 
    the complex -> real transform (for spherical harmonics) and 
    the spherical Bessel transform.
  '''
  def __init__(self, sv):
    ao_log = sv.ao_log
    self.jmx  = -1
    for mu2j in ao_log.sp_mu2j: self.jmx = max(self.jmx, max(mu2j))

    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao_log.rr, ao_log.pp, lmax=2*self.jmx+1)
    gaunt_c.__init__(self, self.jmx)

    self.interp_rr = log_interp_c(self.rr)
    
    self.psi_log = ao_log.psi_log
    self.psi_log_rl = ao_log.psi_log_rl
    self.sp_mu2j = ao_log.sp_mu2j
    self.sp_mu2rcut = ao_log.sp_mu2rcut
    self.sp2nmult = ao_log.sp2nmult
    self.sp_mu2s  = ao_log.sp_mu2s
    self.sp2norbs  = ao_log.sp2norbs
    self.species  = range(len(ao_log.sp2nmult))
    self.sp2mults = [ range(ao_log.sp2nmult[sp]) for sp in self.species ]
    self.sp2charge = sv.sp2charge
    
    self.sp2info = []
    for sp in self.species:
      self.sp2info.append([
        [mu, self.sp_mu2j[sp][mu], self.sp_mu2s[sp][mu], self.sp_mu2s[sp][mu+1]] for mu in self.sp2mults[sp] ])

    self.psi_log_mom = []
    for sp,nmu in zip(self.species,self.sp2nmult):
      mu2ao = np.zeros((nmu,self.nr), dtype='float64')
      for mu,am in zip(self.sp2mults[sp], self.sp_mu2j[sp]): mu2ao[mu,:] = self.sbt( self.psi_log[sp][mu,:], am, 1)
      self.psi_log_mom.append(mu2ao)
    
    dr = np.log(ao_log.rr[1]/ao_log.rr[0])
    self.rr3_dr = ao_log.rr**3 * dr
    self.four_pi = 4*np.pi
    self.const = np.sqrt(np.pi/2.0)

  #
  def overlap_am(self, sp1, sp2, R1, R2):
    from pyscf.nao.m_overlap_am import overlap_am as overlap 
    return overlap(self, sp1, sp2, R1, R2)

  #
  def overlap_ni(self, sp1, sp2, R1, R2, **kvargs):
    from pyscf.nao.m_overlap_ni import overlap_ni
    return overlap_ni(self, sp1, sp2, R1, R2, **kvargs)

#
#
#
if __name__ == '__main__':
  from pyscf.nao.m_system_vars import system_vars_c
  sv  = system_vars_c()
  me = ao_matelem_c(sv)
  R1 = sv.atom2coord[0]
  R2 = sv.atom2coord[1]

  overlap_am = me.overlap_am(0, 0, R1, R2)
  for lev in range(9):
    overlap_ni = me.overlap_ni(0, 0, R1, R2, level=lev)
    print(lev)
    print(abs(overlap_ni-overlap_am).sum()/overlap_am.size)
    print(abs(overlap_ni-overlap_am).max())
    print(overlap_ni[-1,-1], overlap_am[-1,-1])
