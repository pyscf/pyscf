from __future__ import division, print_function
import numpy as np
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_gaunt import gaunt_c
from pyscf.nao.m_log_interp import log_interp_c

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
  def overlap_ni(self, sp1, sp2, R1, R2, level=None):
    """
      Computes overlap for an atom pair. The atom pair is given by a pair of species indices
      and the coordinates of the atoms.
      Args: 
        sp1,sp2 : specie indices, and
        R1,R2 :   respective coordinates in Bohr, atomic units
      Result:
        matrix of orbital overlaps
      The procedure uses the numerical integration in coordinate space.
    """
    
    from pyscf import gto
    from pyscf import dft
    from pyscf.nao.m_gauleg import leggauss_ab
    from pyscf.nao.m_ao_eval import ao_eval
    #from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    from timeit import default_timer as timer
    
    assert(sp1>-1)
    assert(sp2>-1)

    shape = [self.sp2norbs[sp] for sp in (sp1,sp2)]
    
    start1 = timer()
    if ((R1-R2)**2).sum()<1e-7 :
      mol = gto.M( atom=[ [int(self.sp2charge[sp1]), R1]],)
    else :
      mol = gto.M( atom=[ [int(self.sp2charge[sp1]), R1], [int(self.sp2charge[sp2]), R2] ],)
    end1 = timer()
    
    start2 = timer()
    atom2rcut=np.array([self.sp_mu2rcut[sp].max() for sp in (sp1,sp2)])
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3 if level is None else level # precision as implemented in pyscf
    grids.radi_method=leggauss_ab
    grids.build(atom2rcut=atom2rcut)
    end2 = timer()
    
    start3 = timer()
    ao1 = ao_eval(self, R1, sp1, grids.coords)
    ao2 = ao_eval(self, R2, sp2, grids.coords)
    end3 = timer()
    
    start4 = timer()
    ao1 = ao1 * grids.weights
    overlaps = np.einsum("ij,kj->ik", ao1, ao2) #      overlaps = np.matmul(ao1, ao2.T)
    end4 = timer()
    
    print(end1-start1, end2-start2, end3-start3, end4-start4)
    
    return overlaps

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
