from __future__ import print_function, division
import sys, numpy as np

from pyscf.nao.m_color import color as bc
from pyscf.nao.m_system_vars_dos import system_vars_dos, system_vars_pdos
from pyscf.nao.m_siesta2blanko_csr import _siesta2blanko_csr
from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
from pyscf.nao.m_siesta_ion_add_sp2 import _siesta_ion_add_sp2
from pyscf.nao.m_ao_log import ao_log_c

#
#
#
def get_orb2m(sv):
  orb2m = np.empty(sv.norbs, dtype='int64')
  orb = 0
  for atom,sp in enumerate(sv.atom2sp):
    for mu,j in enumerate(sv.sp_mu2j[sp]):
      for m in range(-j,j+1): orb2m[orb],orb = m,orb+1
  return orb2m

#
#
#
def get_orb2j(sv):
  orb2j = np.empty(sv.norbs, dtype='int64')
  orb = 0
  for atom,sp in enumerate(sv.atom2sp):
    for mu,j in enumerate(sv.sp_mu2j[sp]):
      for m in range(-j,j+1): orb2j[orb],orb = j,orb+1
  return orb2j

#
#
#
def overlap_check(sv, tol=1e-5, **kvargs):
  over = sv.overlap_coo(**kvargs).tocsr()
  diff = (sv.hsx.s4_csr-over).sum()
  summ = (sv.hsx.s4_csr+over).sum()
  ac = diff/summ<tol
  if not ac: print(diff, summ)
  return ac

#
#
#
class nao():

  def __init__(self, **kw):
    """ 
      Constructor of NAO class
    """
    if 'gto' in kw:
      self.init_pyscf_gto(**kw)
    else:
      raise RuntimeError('unknown init method')
    #print(kw)
    #print(dir(kw))

  #
  #
  #
  def init_pyscf_gto(self, **kw):
    """Interpret previous pySCF calculation"""
    from pyscf.lib import logger

    gto = kw['gto']
    self.verbose = kw['verbose'] if 'verbose' in kw else 0
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self.cart = False
    self._nelectron = gto.nelectron
    self._built = True
    self.max_memory = 20000

    self.spin = gto.spin
    self.nspin = gto.spin+1
    self.label = kw['label'] if 'label' in kw else 'pyscf'
    self.mol=gto # Only some data must be copied, not the whole object. Otherwise, an eventual deepcopy(...) may fail.
    self.natm=self.natoms = gto.natm
    a2s = [gto.atom_symbol(ia) for ia in range(gto.natm) ]
    self.sp2symbol = sorted(list(set(a2s)))
    self.nspecies = len(self.sp2symbol)
    self.atom2sp = np.empty((gto.natm), dtype=np.int64)
    for ia,sym in enumerate(a2s): self.atom2sp[ia] = self.sp2symbol.index(sym)

    self.sp2charge = [-999]*self.nspecies
    for ia,sp in enumerate(self.atom2sp): self.sp2charge[sp]=gto.atom_charge(ia)
    self.ao_log = ao_log_c().init_ao_log_gto_suggest_mesh(nao=self, **kw)
    self.atom2coord = np.zeros((self.natm, 3))
    for ia,coord in enumerate(gto.atom_coords()): self.atom2coord[ia,:]=coord # must be in Bohr already?
    self.atom2s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2s[atom+1]=self.atom2s[atom]+self.ao_log.sp2norbs[sp]
    self.norbs = self.norbs_sc = self.atom2s[-1]
    self.ucell = 30.0*np.eye(3)
    self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]
    self._atom = gto._atom
    self.basis = gto.basis
    ### implement when needed  self.init_libnao()
    self.nbas = self.atom2mu_s[-1] # total number of radial orbitals
    self.mu2orb_s = np.zeros((self.nbas+1), dtype=np.int64)
    for sp,mu_s in zip(self.atom2sp,self.atom2mu_s):
      for mu,j in enumerate(self.ao_log.sp_mu2j[sp]): self.mu2orb_s[mu_s+mu+1] = self.mu2orb_s[mu_s+mu] + 2*j+1
    return self

  # More functions for similarity with Mole
  def atom_symbol(self, ia): return self.sp2symbol[self.atom2sp[ia]]
  def atom_charge(self, ia): return self.sp2charge[self.atom2sp[ia]]
  def atom_charges(self): return np.array([self.sp2charge[sp] for sp in self.atom2sp], dtype='int64')
  def atom_coord(self, ia): return self.atom2coord[ia,:]
  def atom_coords(self): return self.atom2coord
  def nao_nr(self): return self.norbs
  def atom_nelec_core(self, ia): return self.sp2charge[self.atom2sp[ia]]-self.ao_log.sp2valence[self.atom2sp[ia]]
  def ao_loc_nr(self): return self.mu2orb_s[0:self.natm]

  # More functions for convenience (see PDoS)
  def get_orb2j(self): return get_orb2j(self)
  def get_orb2m(self): return get_orb2m(self)

  def overlap_coo(self, **kvargs):   # Compute overlap matrix for the molecule
    from pyscf.nao.m_overlap_coo import overlap_coo
    return overlap_coo(self, **kvargs)

  def overlap_lil(self, **kvargs):   # Compute overlap matrix in list of lists format
    from pyscf.nao.m_overlap_lil import overlap_lil
    return overlap_lil(self, **kvargs)

  def laplace_coo(self):   # Compute matrix of Laplace brakets for the whole molecule
    from pyscf.nao.m_overlap_coo import overlap_coo
    from pyscf.nao.m_laplace_am import laplace_am
    return overlap_coo(self, funct=laplace_am)
  
  def vnucele_coo_coulomb(self, **kvargs): # Compute matrix elements of attraction by Coulomb forces from point nuclei
    from pyscf.nao.m_vnucele_coo_coulomb import vnucele_coo_coulomb
    return vnucele_coo_coulomb(self, **kvargs)

  def dipole_coo(self, **kvargs):   # Compute dipole matrix elements for the given system
    from pyscf.nao.m_dipole_coo import dipole_coo
    return dipole_coo(self, **kvargs)
  
  def overlap_check(self, tol=1e-5, **kvargs): # Works only after init_siesta_xml(), extend ?
    return overlap_check(self, tol=1e-5, **kvargs)

  def energy_nuc(self, charges=None, coords=None):
    """ Potential energy of electrostatic repulsion of point nuclei """
    from scipy.spatial.distance import cdist
    chrg = self.atom_charges() if charges is None else charges
    crds = self.atom_coords() if coords is None else coords
    identity = np.identity(len(chrg))
    return ((chrg[:,None]*chrg[None,:])*(1.0/(cdist(crds, crds)+identity)-identity)).sum()*0.5

  def build_3dgrid_pp(self, level=3):
    """ Build a global grid and weights for a molecular integration (integration in 3-dimensional coordinate space) """
    from pyscf import dft
    from pyscf.nao.m_gauleg import leggauss_ab
    grid = dft.gen_grid.Grids(self)
    grid.level = level # precision as implemented in pyscf
    grid.radi_method=leggauss_ab
    atom2rcut=np.zeros(self.natoms)
    for ia,sp in enumerate(self.atom2sp): atom2rcut[ia] = self.ao_log.sp2rcut[sp]
    grid.build(atom2rcut=atom2rcut)
    return grid

  def build_3dgrid_ae(self, level=3):
    """ Build a global grid and weights for a molecular integration (integration in 3-dimensional coordinate space) """
    from pyscf import dft
    grid = dft.gen_grid.Grids(self)
    grid.level = level # precision as implemented in pyscf
    grid.build()
    return grid

  def comp_aos_den(self, coords):
    """ Compute the atomic orbitals for a given set of (Cartesian) coordinates. """
    from pyscf.nao.m_aos_libnao import aos_libnao
    if not self.init_sv_libnao : raise RuntimeError('not self.init_sv_libnao')
    return aos_libnao(coords, self.norbs)

  def comp_vnuc_coulomb(self, coords):
    from scipy.spatial.distance import cdist
    ncoo = coords.shape[0]
    vnuc = np.zeros(ncoo)
    for R,sp in zip(self.atom2coord, self.atom2sp):
      dd, Z = cdist(R.reshape((1,3)), coords).reshape(ncoo), self.sp2charge[sp]
      vnuc = vnuc - Z / dd 
    return vnuc
    
  def init_libnao(self):
    """ Initialization of data on libnao site """
    from pyscf.nao.m_libnao import libnao
    from pyscf.nao.m_sv_chain_data import sv_chain_data
    from ctypes import POINTER, c_double, c_int64, c_int32, byref

    #libnao.init_aos_libnao.argtypes = (POINTER(c_int64), POINTER(c_int64))
    #info = c_int64(-999)
    #libnao.init_aos_libnao(c_int64(self.norbs), byref(info))
    #if info.value!=0: raise RuntimeError("info!=0")
    raise RuntimeError('not implemented!')
    return self
  
  @property
  def nelectron(self):
    if self._nelectron is None:
      return tot_electrons(self)
    else:
      return self._nelectron

#
# Example of reading pySCF orbitals.
#
if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao import nao
  import matplotlib.pyplot as plt
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvtz') # coordinates in Angstrom!
  sv = nao(gto=mol, rcut_tol=1e-8, nr=512, rmin=1e-5)
  
  print(sv.ao_log.sp2norbs)
  print(sv.ao_log.sp2nmult)
  print(sv.ao_log.sp2rcut)
  print(sv.ao_log.sp_mu2rcut)
  print(sv.ao_log.nr)
  print(sv.ao_log.rr[0:4], sv.ao_log.rr[-1:-5:-1])
  print(sv.ao_log.psi_log[0].shape, sv.ao_log.psi_log_rl[0].shape)

  sp = 0
  for mu,[ff,j] in enumerate(zip(sv.ao_log.psi_log[sp], sv.ao_log.sp_mu2j[sp])):
    nc = abs(ff).max()
    if j==0 : plt.plot(sv.ao_log.rr, ff/nc, '--', label=str(mu)+' j='+str(j))
    if j>0 : plt.plot(sv.ao_log.rr, ff/nc, label=str(mu)+' j='+str(j))

  plt.legend()
  #plt.xlim(0.0, 10.0)
  #plt.show()
