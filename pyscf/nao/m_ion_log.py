from __future__ import print_function, division
from pyscf.nao.m_log_interp import log_interp_c
import numpy as np

#
#
#
class ion_log_c():
  '''
  holder of radial orbitals on logarithmic grid and bookkeeping info
  Args:
    ion : ion structure (read from ion files from siesta)

  Returns:
    ao_log:
      ion (ion structure from m_siesta_ion or m_siesta_ion_xml).
      nr (int): number of radial point
      rr
      pp
      mu2ff
      mu2ff_rl
      rcut (float): array containing the rcutoff of each specie
      mu2rcut (array, float)
      mu2s array containing pointers to the start indices for each radial multiplett
      norbs number of atomic orbitals 
      interp_rr instance of log_interp_c to interpolate along real-space axis
      interp_pp instance of log_interp_c to interpolate along momentum-space axis

  Examples:

  >>> sv = system_vars()
  >>> ao = ao_log_c(sv.sp2ion)
  >>> print(ao.psi_log.shape)
  '''
  def __init__(self, ao_log, sp):

    self.ion = ao_log.sp2ion[sp]
    self.rr,self.pp,self.nr = ao_log.rr,ao_log.pp,ao_log.nr
    self.interp_rr = log_interp_c(self.rr)
    self.interp_pp = log_interp_c(self.pp)
    
    self.mu2j = ao_log.sp_mu2j[sp]
    self.nmult= len(self.mu2j)
    self.mu2s = ao_log.sp_mu2s[sp]
    self.norbs= self.mu2s[-1]

    self.mu2ff = ao_log.psi_log[sp]
    self.mu2ff_rl = ao_log.psi_log_rl[sp]    
    self.mu2rcut = ao_log.sp_mu2rcut[sp]
    self.rcut = np.amax(self.mu2rcut)
    self.charge = ao_log.sp2charge[sp]
