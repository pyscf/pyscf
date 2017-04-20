from __future__ import print_function
from __future__ import division
import numpy as np
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
    
    _siesta_ion_add_sp2(self, sp2ion) # adds the fields for counting, .nspecies etc.
    self.sp2ion = sp2ion
    
    npts = max(max(ion["paos"]["npts"]) for ion in sp2ion)
    self.nr = next235( max(2.0*npts, 1024.0) ) if nr==None else nr
    assert(self.nr>2)

    dmin = min(min(ion["paos"]["delta"]) for ion in sp2ion)
    assert(dmin>0.0)
    
    self.rmin = dmin if rmin is None else rmin
    assert(self.rmin>0.0)

    self.kmax = 1.0/dmin/np.pi if kmax==None else kmax
    assert(self.kmax>0.0)
    
    dmax = 2.3*max(max(ion["paos"]["cutoff"]) for ion in sp2ion)
    assert(dmax>0.0)
    
    self.rmax = dmax if rmax is None else rmax
    assert(self.rmax>0.0)
    
    self.rr,self.pp = log_mesh(self.nr, self.rmin, self.rmax, self.kmax)
    
    self.psi_log = siesta_ion_interp(self.rr, sp2ion, 1)
    self.psi_log_rl = siesta_ion_interp(self.rr, sp2ion, 0)
    
    self.sp_mu2rcut = [ np.array(ion["paos"]["cutoff"], dtype='float64') for ion in sp2ion]
    self.sp2rcut = np.array([np.amax(rcuts) for rcuts in self.sp_mu2rcut], dtype='float64')
    
    #call sp2ion_to_psi_log(sv%sp2ion, sv%rr, sv%psi_log)
    #call init_psi_log_rl(sv%psi_log, sv%rr, sv%uc%mu_sp2j, sv%uc%sp2nmult, sv%psi_log_rl)
    #call sp2ion_to_core(sv%sp2ion, sv%rr, sv%core_log, sv%sp2has_core, sv%sp2rcut_core)

