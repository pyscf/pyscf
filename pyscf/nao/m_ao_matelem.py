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
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_c2r import c2r_c
from pyscf.nao.m_log_interp import log_interp_c
from pyscf.nao.m_ao_log_hartree import ao_log_hartree
from timeit import default_timer as timer
from pyscf.nao.m_gaunt import gaunt_c

#
#
#
def build_3dgrid(me, sp1, R1, sp2, R2, level=3):
  from pyscf import dft
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_gauleg import gauss_legendre

  assert sp1>=0
  assert sp2>=0

  if ( (R1-R2)**2 ).sum()<1e-7 :
    mol=system_vars_c().init_xyzlike([ [int(me.aos[0].sp2charge[sp1]), R1] ])
  else :
    mol=system_vars_c().init_xyzlike([ [int(me.aos[0].sp2charge[sp1]), R1], [int(me.aos[1].sp2charge[sp2]), R2] ])

  atom2rcut=np.array([me.aos[isp].sp_mu2rcut[sp].max() for isp,sp in enumerate([sp1,sp2])])
  grids = dft.gen_grid.Grids(mol)
  grids.level = level # precision as implemented in pyscf
  grids.radi_method = gauss_legendre
  grids.build(atom2rcut=atom2rcut)
  #grids.build()
  return grids

#
#
#
def build_3dgrid3c(me, sp1, sp2, R1, R2, sp3, R3, level=3):
  from pyscf import dft
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_gauleg import gauss_legendre

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
  grids.radi_method = gauss_legendre
  grids.build(atom2rcut=atom2rcut)
  return grids

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
  def __init__(self, rr, pp, sv=None, dm=None):
    """ Basic """
    from pyscf.nao.m_init_dm_libnao import init_dm_libnao
    from pyscf.nao.m_init_dens_libnao import init_dens_libnao
    
    self.interp_rr = log_interp_c(rr)
    self.interp_pp = log_interp_c(pp)
    self.rr3_dr = rr**3 * np.log(rr[1]/rr[0])
    self.dr_jt  = np.log(rr[1]/rr[0])
    self.four_pi = 4*np.pi
    self.const = np.sqrt(np.pi/2.0)
    self.pp2 = pp**2
    self.sv = None if sv is None else sv.init_libnao()
    self.dm = None if dm is None else init_dm_libnao(dm)
    if dm is not None and sv is not None : init_dens_libnao()

  # @classmethod # I don't understand something about classmethod
  def init_one_set(self, ao, **kvargs):
    """ Constructor for two-center matrix elements, i.e. one set of radial orbitals per specie is provided """
    self.jmx = ao.jmx
    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao.rr, ao.pp, lmax=2*self.jmx+1)
    gaunt_c.__init__(self, self.jmx)
    self.ao1 = ao
    self.ao1._add_sp2info()
    self.ao1._add_psi_log_mom()
    
    self.ao2 = self.ao1
    self.ao2_hartree = ao_log_hartree(self.ao1, **kvargs)
    self.aos = [self.ao1, self.ao2]
    return self

  # @classmethod # I don't understand something about classmethod
  def init_two_sets(self, ao1, ao2, **kvargs):
    """ Constructor for matrix elements between product functions and orbital's products: two sets of radial orbitals must be provided. """
    self.jmx = max(ao1.jmx, ao2.jmx)
    c2r_c.__init__(self, self.jmx)
    sbt_c.__init__(self, ao1.rr, ao1.pp, lmax=2*self.jmx+1)
    gaunt_c.__init__(self, self.jmx)
    self.ao1 = ao1
    self.ao1._add_sp2info()
    self.ao1._add_psi_log_mom()
    self.pp2 = self.ao1.pp**2

    self.ao2 = ao2
    self.ao2._add_sp2info()
    self.ao2_hartree = ao_log_hartree(self.ao2, **kvargs)
    self.ao2._add_psi_log_mom()
    self.aos = [self.ao1, self.ao2]
    return self

  #
  def overlap_am(self, sp1,R1, sp2,R2):
    from pyscf.nao.m_overlap_am import overlap_am as overlap 
    return overlap(self, sp1,R1, sp2,R2)

  def overlap_ni(self, sp1,R1, sp2,R2, **kvargs):
    from pyscf.nao.m_overlap_ni import overlap_ni
    return overlap_ni(self, sp1,R1, sp2,R2, **kvargs)
  
  def coulomb_am(self, sp1,R1, sp2,R2):
    from pyscf.nao.m_coulomb_am import coulomb_am as ext
    return ext(self, sp1,R1, sp2,R2)

  def coulomb_ni(self, sp1,R1, sp2,R2,**kvargs):
    from pyscf.nao.m_eri2c import eri2c as ext
    return ext(self, sp1,R1, sp2,R2,**kvargs)

  def xc_scalar(self, sp1,R1, sp2,R2,**kvargs):
    from pyscf.nao.m_xc_scalar_ni import xc_scalar_ni as ext
    return ext(self, sp1,R1, sp2,R2,**kvargs)
