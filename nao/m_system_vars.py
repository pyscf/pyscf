from __future__ import print_function, division
import numpy as np
import sys

from pyscf.nao.m_color import color as bc
from pyscf.nao.m_siesta_xml import siesta_xml
from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
from pyscf.nao.m_siesta_hsx import siesta_hsx_c
from pyscf.nao.m_siesta2blanko_csr import _siesta2blanko_csr
from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
from pyscf.nao.m_sv_diag import sv_diag 
from pyscf.nao.m_siesta_ion_add_sp2 import _siesta_ion_add_sp2
from pyscf.nao.m_ao_log import ao_log_c
from pyscf.lib import logger
from pyscf.data import chemical_symbols

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
def diag_check(self, atol=1e-5, rtol=1e-4):
  ksn2e = self.xml_dict['ksn2e']
  ac = True
  for k,kvec in enumerate(self.xml_dict["k2xyzw"]):
    for spin in range(self.nspin):
      e,x = sv_diag(self, kvec=kvec, spin=spin)
      eref = ksn2e[k,spin,:]
      acks = np.allclose(eref,e,atol=atol,rtol=rtol)
      ac = ac and acks
      if(not acks):
        aerr = sum(abs(eref-e))/len(e)
        print("diag_check: "+bc.RED+str(k)+' '+str(spin)+' '+str(aerr)+bc.ENDC)
  return ac

#
#
#
def overlap_check(sv, tol=1e-5, **kvargs):
  from pyscf.nao.m_comp_overlap_coo import comp_overlap_coo
  over = comp_overlap_coo(sv, **kvargs).tocsr()
  diff = (sv.hsx.s4_csr-over).sum()
  summ = (sv.hsx.s4_csr+over).sum()
  ac = diff/summ<tol
  if not ac: print(diff, summ)
  return ac

#
#
#
class system_vars_c():

  def __init__(self, label=None, Atoms=None, atom=None, gto=None, **kvargs):
    """ 
      Constructor of system_vars class: so far can be initialized 
      with SIESTA orbitals and Hamiltonian and wavefunctions
    """
    self.verbose = logger.NOTE  # To be similar to Mole object...
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None

    if atom is not None: # Simple xyz-like constructor of (molecular) geometry from a list [n1, [x1,y1,z1], n2, [x2,y2,z2]], where n* are nuclear charges, and [xyz] are coordinates in Bohr
      atom2charge = [atm[0] for atm in atom]
      self.atom2coord = np.array([atm[1] for atm in atom], dtype='float64')
      self.sp2charge = list(set(atom2charge))
      self.sp2symbol = [chemical_symbols[z] for z in self.sp2charge]
      self.atom2sp = [self.sp2charge.index(charge) for charge in atom2charge]
      self.natm = len(self.atom2sp)
      return

    if label is not None and Atoms is None: # Read from SIESTA without ASE input
      self.label = label
      self.xml_dict = siesta_xml(self.label)
      self.wfsx = siesta_wfsx_c(self.label)
      self.hsx = siesta_hsx_c(self.label, **kvargs)
      self.norbs_sc = self.wfsx.norbs if self.hsx.orb_sc2orb_uc is None else len(self.hsx.orb_sc2orb_uc)
      self.init_siesta_xml()
      self.sp2symbol = [str(ion['symbol'].replace(' ', '')) for ion in self.sp2ion]
      self.sp2charge = self.ao_log.sp2charge
      return
      
    if Atoms is not None:  # Read from SIESTA with ASE input
      self.label = 'ase' if label is None else label
      self.xml_dict = siesta_xml(self.label)
      self.wfsx = siesta_wfsx_c(self.label)
      self.hsx = siesta_hsx_c(self.label, **kvargs)
      self.norbs_sc = self.wfsx.norbs if self.hsx.orb_sc2orb_uc is None else len(self.hsx.orb_sc2orb_uc)
      self.init_ase_atoms(Atoms)
      self.sp2symbol = [str(ion['symbol'].replace(' ', '')) for ion in self.sp2ion]
      self.sp2charge = self.ao_log.sp2charge
      return
    
    if gto is not None: # Interpret previous pySCF calculation
      self.label = 'pyscf' if label is None else label
      self.init_pyscf_gto(gto, **kvargs)
      
    
  #
  #
  #
  def init_pyscf_gto(self, gto, **kvargs):
    """Interpret previous pySCF calculation"""
    self.natm = gto.natm
    a2s = [gto.atom_symbol(ia) for ia in range(gto.natm) ]
    self.sp2symbol = sorted(list(set(a2s)))
    self.nspecies = len(self.sp2symbol)
    self.atom2sp = np.empty((self.natm), dtype='int64')
    for ia,sym in enumerate(a2s): self.atom2sp[ia] = self.sp2symbol.index(sym)

    self.sp2charge = [-999]*self.nspecies
    for ia,sp in enumerate(self.atom2sp): self.sp2charge[sp]=gto.atom_charge(ia)
    self.ao_log = ao_log_c(gto=gto, sv=self, **kvargs)
    self.atom2coord = np.zeros((self.natm, 3))
    for ia,coord in enumerate(gto.atom_coords()): self.atom2coord[ia,:]=coord # must be in Bohr already...

    print('finished with GTO.')
      
    
  #
  #
  #
  def init_ase_atoms(self, Atoms):
    """ Initialise system vars using siesta file and Atom object from ASE."""
    try:
      import ase
    except:
      warn('no ASE installed: try via siesta.xml')
      self.init_siesta_xml()

    self.Atoms = Atoms
   
    ##### The parameters as fields     
    self.sp2ion = []
    species = []
    for sp in Atoms.get_chemical_symbols():
      if sp not in species:
        species.append(sp)
        self.sp2ion.append(siesta_ion_xml(sp+'.ion.xml'))
    
    _add_mu_sp2(self, self.sp2ion)
    self.sp2ao_log = ao_log_c(self.sp2ion)
  
    self.natm   = Atoms.get_positions().shape[0]
    self.norbs  = self.wfsx.norbs
    self.nspin  = self.wfsx.nspin
    self.nkpoints  = self.wfsx.nkpoints

    strspecie2sp = {}
    for sp in range(len(self.wfsx.sp2strspecie)): strspecie2sp[self.wfsx.sp2strspecie[sp]] = sp
    
    self.atom2sp = np.empty((self.natoms), dtype='int64')
    for i, sp in enumerate(Atoms.get_chemical_symbols()):
      self.atom2sp[i] = strspecie2sp[sp]
    
    orb2m = get_orb2m(self)
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)
    
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.X[:,:,n,s,k])

  #
  #
  #
  def init_siesta_xml(self):
    """
    Initialise system var using only the siesta files (siesta.xml in particular is needed).
    """
    self.Atoms = None
    
    ##### The parameters as fields     
    self.sp2ion = []
    for sp in self.wfsx.sp2strspecie:
      self.sp2ion.append(siesta_ion_xml(sp+'.ion.xml'))
    
    _siesta_ion_add_sp2(self, self.sp2ion)
    self.ao_log = ao_log_c(self.sp2ion)
    
    self.atom2coord = self.xml_dict['atom2coord']
    self.natm = len(self.xml_dict['atom2sp'])
    self.norbs  = self.wfsx.norbs 
    self.nspin  = self.wfsx.nspin
    self.nkpoints  = self.wfsx.nkpoints

    strspecie2sp = {}
    for sp,strsp in enumerate(self.wfsx.sp2strspecie): strspecie2sp[strsp] = sp
    
    self.atom2sp = np.empty((self.natm), dtype='int64')
    for o,atom in enumerate(self.wfsx.orb2atm):
      self.atom2sp[atom-1] = strspecie2sp[self.wfsx.orb2strspecie[o]]

    orb2m = get_orb2m(self)
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)
    
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.X[:,:,n,s,k])

  #
  # More functions for similarity with Mole
  #
  def atom_symbol(self, ia): return self.sp2symbol[self.atom2sp[ia]]
  def atom_charge(self, ia): return self.sp2charge[self.atom2sp[ia]]
  def atom_charges(self): return np.array([self.sp2charge[sp] for sp in self.atom2sp], dtype='int64')
  def atom_coord(self, ia): return self.atom2coord[ia,:]
  def atom_coords(self): return self.atom2coord

#
# Example of reading pySCF orbitals.
#
if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao.m_system_vars import system_vars_c
  import matplotlib.pyplot as plt
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvtz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol)
  print(sv.ao_log.sp2norbs)
  print(sv.ao_log.sp2nmult)

  sp = 0
  for mu,[ff,j] in enumerate(zip(sv.ao_log.psi_log[sp], sv.ao_log.sp_mu2j[sp])):
    nc = abs(ff).max()
    if j==0 : plt.plot(sv.ao_log.rr, ff/nc, '--', label=str(mu)+' j='+str(j))
    if j>0 : plt.plot(sv.ao_log.rr, ff/nc, label=str(mu)+' j='+str(j))
  
  plt.legend()
  #plt.xlim(0.0, 10.0)
  plt.show()
  
