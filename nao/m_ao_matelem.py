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
    mol = system_vars_c(atom=[ [int(me.ao1.sp2charge[sp1]), R1] ])
  else :
    mol = system_vars_c(atom=[ [int(me.ao1.sp2charge[sp1]), R1], [int(me.ao1.sp2charge[sp2]), R2] ])

  atom2rcut=np.array([me.ao1.sp_mu2rcut[sp].max() for sp in (sp1,sp2)])
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
  def __init__(self, ao1, ao2=None):
    
    if ao2==None: # Usual matrix elements, compatibility constructor which will become obsolete...
      self.init1(ao1)
      return
    
    self.init2(ao1, ao2)
  
  def init2(self, ao1, ao2):
    """ Constructor for matrix elements between molecular species <a_mol1|O|b_mol2> 
        or three-center matrix elements <a,b |C| p> """
    self.jmx = max(ao1.jmx, ao2.jmx)
    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao1.rr, ao1.pp, lmax=2*self.jmx+1)
    gaunt_c.__init__(self, self.jmx)
    assert(ao1.rr==ao2.rr)
    
    self.interp_rr = log_interp_c(self.rr)
    self.interp_pp = log_interp_c(self.kk)
    self.ao1 = ao1
    self.ao2 = ao2
  
  #
  #
  #
  def init1(self, ao_log):
    """ Constructor for common matrix elements  <a|O|b> """
    self.jmx = ao_log.jmx

    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao_log.rr, ao_log.pp, lmax=2*self.jmx+1)
    gaunt_c.__init__(self, self.jmx)

    self.interp_rr = log_interp_c(self.rr)
    self.interp_pp = log_interp_c(self.kk)

    self.ao1 = ao_log

    self.ao1._add_sp2info()
    self.ao1._add_psi_log_mom()

    self.psi_log_mom = []
    for sp,[nmu,mu2ff,mu2j] in enumerate(zip(self.ao1.sp2nmult,self.ao1.psi_log,self.ao1.sp_mu2j)):
      mu2ao = np.zeros((nmu,self.nr), dtype='float64')
      for mu,[am,ff] in enumerate(zip(mu2j,mu2ff)): mu2ao[mu,:] = self.sbt( ff, am, 1 )
      self.psi_log_mom.append(mu2ao)
    
    self.rr3_dr = ao_log.rr**3 * np.log(ao_log.rr[1]/ao_log.rr[0])
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
