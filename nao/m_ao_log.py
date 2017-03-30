from __future__ import division
import numpy
from pyscf.nao.m_siesta_ion_add_sp2 import _siesta_ion_add_sp2
from pyscf.nao.m_next235 import next235
from pyscf.nao.m_log_mesh import log_mesh
from pyscf.nao.m_siesta_ion_interp import siesta_ion_interp
#
#
#
class ao_log_c():
  '''
  holder of radial orbitals on logarithmic grid.
  Args:
    ions : list of ion structures (read from ion files from siesta)

  Returns:
    ao_log:
      sp2ion (ion structure from m_siesta_ion or m_siesta_ion_xml):
        List of structure composed of several field read from the ions file.
      nr (int): number of radial point
      rmin (float)
      kmax (float)
      rmax (float)
      rr
      pp
      psi_log
      psi_log_rl
      sp2rcut (array, float): array containing the rcutoff of each specie
      sp_mu2rcut (array, float)

  Examples:

  >>> sv = system_vars()
  >>> ao = ao_log_c(sv.sp2ion)
  >>> print(ao.psi_log.shape)
  '''
  def __init__(self, sp2ion, nr=None, rmin=None, rmax=None, kmax=None):
    
    self.sp2ion = sp2ion
    _siesta_ion_add_sp2(self, sp2ion)
    
    npts = max(max(sp2ion[sp]["paos"]["npts"]) for sp in range(self.nspecies))
    self.nr = next235( max(2.0*npts, 1024.0) ) if nr==None else nr
    assert(self.nr>2)

    #min_delta = []

    dmin = min(min(sp2ion[sp]["paos"]["delta"]) for sp in range(self.nspecies))

    if dmin == 0.0:
      raise ValueError('Error in ion file, dmin = 0.0!!!')
    
    self.rmin = dmin if rmin is None else rmin
    self.kmax = 1.0/dmin/numpy.pi if kmax==None else kmax
    
    dmax = 2.3*max(max(sp2ion[sp]["paos"]["cutoff"]) for sp in range(self.nspecies))
    if dmax == 0.0:
      raise ValueError('Error in ion file, dmax = 0.0!!!')
    
    self.rmax = dmax if rmax is None else rmax
    
    self.rr, self.pp = log_mesh(self.nr, self.rmin, self.rmax, self.kmax)
    
    self.psi_log = siesta_ion_interp(self.rr, sp2ion, 1)
    self.psi_log_rl = siesta_ion_interp(self.rr, sp2ion, 0)
    
    self.sp2rcut = numpy.array([max(sp2ion[sp]["paos"]["cutoff"]) for sp in range(self.nspecies)], dtype='float64')
    
    self.sp_mu2rcut = numpy.empty((self.nspecies,self.psi_log.shape[1]), dtype='float64', order='F')
    self.sp_mu2rcut.fill(-999.0)
    
    for sp in range(self.nspecies):
      self.sp_mu2rcut[sp,0:self.sp2nmult[sp]] = sp2ion[sp]["paos"]["cutoff"][0:self.sp2nmult[sp]]

    #call sp2ion_to_psi_log(sv%sp2ion, sv%rr, sv%psi_log)
    #call init_psi_log_rl(sv%psi_log, sv%rr, sv%uc%mu_sp2j, sv%uc%sp2nmult, sv%psi_log_rl)
    #call sp2ion_to_core(sv%sp2ion, sv%rr, sv%core_log, sv%sp2has_core, sv%sp2rcut_core)

