from __future__ import print_function, division
import numpy as np
from pyscf.nao.m_siesta_ion_add_sp2 import _siesta_ion_add_sp2
from pyscf.nao.m_next235 import next235
from pyscf.nao.m_log_mesh import log_mesh
from pyscf.nao.m_log_interp import log_interp_c
from pyscf.nao.m_siesta_ion_interp import siesta_ion_interp


#
#
#
def get_default_log_mesh_gto(gto, tol=1e-7):
  rmin_gcs = 10.0
  rmax_gcs = -1.0
  akmx_gcs = -1.0

  seen_species = [] # this is auxiliary to organize the loop over species 
  for ia in range(gto.natm):
    if gto.atom_symbol(ia) in seen_species: continue
    seen_species.append(gto.atom_symbol(ia))
    for sid in gto.atom_shell_ids(ia):
      for power,coeffs in zip(gto.bas_exp(sid), gto.bas_ctr_coeff(sid)):
        for coeff in coeffs:
          rmin_gcs = min(rmin_gcs, np.sqrt( abs(np.log(1.0-tol)/power )))
          rmax_gcs = max(rmax_gcs, np.sqrt( abs(np.log(abs(coeff))-np.log(tol))/power ))
          akmx_gcs = max(akmx_gcs, np.sqrt( abs(np.log(abs(coeff))-np.log(tol))*4*power ))

  if rmin_gcs<1e-9 : print('rmin_gcs<1D-9')     # Last check 
  if rmax_gcs>1e+2 : print('rmax_gcs>1D+2')
  if akmx_gcs>1e+4 : print('akmx_gcs>1D+4', __name__)
  return rmin_gcs,rmax_gcs,akmx_gcs

#
#
#
class ao_log_c():
  '''
  holder of radial orbitals on logarithmic grid.
  Args:
    ions : list of ion structures (read from ion files from siesta)
      or 
    gto  : gaussian type of orbitals object from pySCF

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
      interp_rr instance of log_interp_c to interpolate along real-space axis
      interp_pp instance of log_interp_c to interpolate along momentum-space axis

  Examples:

  >>> sv = system_vars()
  >>> ao = ao_log_c(sv.sp2ion)
  >>> print(ao.psi_log.shape)
  '''
  def __init__(self, sp2ion=None, gto=None, sv=None, nr=None, rmin=None, rmax=None, kmax=None):
    """
      Initializes numerical orbitals from a previous pySCF calculation or from SIESTA calculation (really numerical orbitals)
    """
    if sp2ion is None:
      self.init_gto(gto, sv, nr, rmin, rmax, kmax)
    else :
      self.init_ion(sp2ion, nr, rmin, rmax, kmax)
  
  #
  #
  def init_gto(self, gto, sv, nr=None, rmin=None, rmax=None, kmax=None):
    """
      Get's data from a previous pySCF calculation, intializes numerical orbitals from the Gaussian type of orbitals etc.
    """
    self.gto = gto
    self.nr = 1024
    print(get_default_log_mesh_gto(gto, tol=1e-7))
    
#    print(dir(gto))
#    print(dir(sv))
    
    self.sp_mu2j = []
    self.sp_mu2s = []

    seen_species = [] # this is auxiliary to organize the loop over species 
    for ia,sp in enumerate(sv.atom2sp):
      if sp in seen_species: continue
      seen_species.append(sp)
      self.sp_mu2j.append(np.array([gto.bas_angular(shell_id) for shell_id in gto.atom_shell_ids(ia)], dtype='int64'))
#      for sid in gto.atom_shell_ids(ia):
#        print(ia, sid, gto.bas_angular(sid), gto.bas_exp(sid), gto.bas_kappa(sid), gto.bas_ctr_coeff(sid))
        
#    print(self.sp_mu2j)

  #
  #  
  def init_ion(self, sp2ion, nr=None, rmin=None, rmax=None, kmax=None):
    """
      Reads data from a previous SIESTA calculation, interpolates the orbitals on a single log mesh.
    """
    _siesta_ion_add_sp2(self, sp2ion) # adds the fields for counting, .nspecies etc.
    self.sp2norbs = np.array([mu2s[self.sp2nmult[sp]] for sp,mu2s in enumerate(self.sp_mu2s)], dtype='int64')
    
    self.sp2ion = sp2ion
    
    npts = max(max(ion["paos"]["npts"]) for ion in sp2ion)
    self.nr = next235( max(2.0*npts, 1024.0) ) if nr is None else nr
    assert(self.nr>2)

    dmin = min(min(ion["paos"]["delta"]) for ion in sp2ion)
    assert(dmin>0.0)
    
    self.rmin = dmin if rmin is None else rmin
    assert(self.rmin>0.0)

    self.kmax = 1.0/dmin/np.pi if kmax is None else kmax
    assert(self.kmax>0.0)
    
    dmax = 2.3*max(max(ion["paos"]["cutoff"]) for ion in sp2ion)
    assert(dmax>0.0)
    
    self.rmax = dmax if rmax is None else rmax
    assert(self.rmax>0.0)
    
    self.rr,self.pp = log_mesh(self.nr, self.rmin, self.rmax, self.kmax)
    self.interp_rr = log_interp_c(self.rr)
    self.interp_pp = log_interp_c(self.pp)
    
    self.psi_log = siesta_ion_interp(self.rr, sp2ion, 1)
    self.psi_log_rl = siesta_ion_interp(self.rr, sp2ion, 0)
    
    self.sp_mu2rcut = [ np.array(ion["paos"]["cutoff"], dtype='float64') for ion in sp2ion]
    self.sp2rcut = np.array([np.amax(rcuts) for rcuts in self.sp_mu2rcut], dtype='float64')
    self.sp2charge = [int(ion['z']) for ion in self.sp2ion]
    
    #call sp2ion_to_psi_log(sv%sp2ion, sv%rr, sv%psi_log)
    #call init_psi_log_rl(sv%psi_log, sv%rr, sv%uc%mu_sp2j, sv%uc%sp2nmult, sv%psi_log_rl)
    #call sp2ion_to_core(sv%sp2ion, sv%rr, sv%core_log, sv%sp2has_core, sv%sp2rcut_core)


if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao.m_ao_log import ao_log_c
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
  ao_log = ao_log_c(gto=mol)
  
  print(ao_log.sp2norbs)
  
  
  
