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
    mol = system_vars_c(atom=[ [int(me.aos[0].sp2charge[sp1]), R1] ])
  else :
    mol = system_vars_c(atom=[ [int(me.aos[0].sp2charge[sp1]), R1], [int(me.aos[1].sp2charge[sp2]), R2] ])

  atom2rcut=np.array([me.aos[isp].sp_mu2rcut[sp].max() for isp,sp in enumerate([sp1,sp2])])
  grids = dft.gen_grid.Grids(mol)
  grids.level = level # precision as implemented in pyscf
  grids.radi_method=leggauss_ab
  grids.build(atom2rcut=atom2rcut)
  return grids

#
#
#
def build_3dgrid3c(me, sp1, sp2, R1, R2, sp3, R3, level=3):
  from pyscf import dft
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_gauleg import leggauss_ab

  d12 = ((R1-R2)**2).sum()
  d13 = ((R1-R3)**2).sum()
  d23 = ((R2-R3)**2).sum()
  z1 = int(me.aos[0].sp2charge[sp1])
  z2 = int(me.aos[0].sp2charge[sp2])
  z3 = int(me.aos[1].sp2charge[sp3])
  rc1 = me.aos[0].sp2rcut[sp1]
  rc2 = me.aos[0].sp2rcut[sp2]
  rc3 = me.aos[1].sp2rcut[sp3]
  
  if d12<1e-7 and d23<1e-7 :
    mol = system_vars_c(atom=[ [z1, R1] ])
  elif d12<1e-7 and d23>1e-7 and d13>1e-7:
    mol = system_vars_c(atom=[ [z1, R1], [z3, R3] ])
  elif d23<1e-7 and d12>1e-7 and d13>1e-7:
    mol = system_vars_c(atom=[ [z1, R1], [z2, R2] ])
  elif d13<1e-7 and d12>1e-7 and d23>1e-7:
    mol = system_vars_c(atom=[ [z1, R1], [z2, R2] ])
  else :
    mol = system_vars_c(atom=[ [z1, R1], [z2, R2], [z3, R3] ])

  atom2rcut=np.array([rc1, rc2, rc3])
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
    """ Constructor for general matrix elements  <a|O|b> between the same molecule, different molecules, between atomic orbitals and product orbitals"""
    self.interp_rr = log_interp_c(ao1.rr)
    self.interp_pp = log_interp_c(ao1.pp)
    self.rr3_dr = ao1.rr**3 * np.log(ao1.rr[1]/ao1.rr[0])
    self.four_pi = 4*np.pi
    self.const = np.sqrt(np.pi/2.0)

    self.jmx = ao1.jmx
    if ao2 is not None: self.jmx = max(self.jmx, ao2.jmx)

    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao1.rr, ao1.pp, lmax=2*self.jmx+1)
    gaunt_c.__init__(self, self.jmx)

    self.ao1 = ao1
    self.ao1._add_sp2info()
    self.ao1._add_psi_log_mom()

    if ao2 is not None:
      self.ao2 = ao2
      self.ao2._add_sp2info()
      self.ao2._add_psi_log_mom()
    else : 
      self.ao2 = self.ao1
    
    self.aos = [self.ao1, self.ao2]

  #
  def overlap_am(self, sp1,R1, sp2,R2):
    from pyscf.nao.m_overlap_am import overlap_am as overlap 
    return overlap(self, sp1,R1, sp2,R2)

  #
  def overlap_ni(self, sp1,R1, sp2,R2, **kvargs):
    from pyscf.nao.m_overlap_ni import overlap_ni
    return overlap_ni(self, sp1,R1, sp2,R2, **kvargs)
  
  def coulomb_am(self, sp1,R1, sp2,R2):
    from pyscf.nao.m_coulomb_am import coulomb_am as ext
    return ext(self, sp1,R1, sp2,R2)


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
