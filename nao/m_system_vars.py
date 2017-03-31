import numpy
import sys
import re
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
#
#
#
def get_orb2m(sv):
  orb2m = numpy.empty(sv.norbs, dtype='int64')
  orb = 0
  for atom in range(sv.natoms):
    sp = sv.atom2sp[atom]
    nmult = sv.sp2nmult[sp]
    for mu in range(nmult):
      j = sv.sp_mu2j[sp,mu]
      for m in range(-j,j+1):
        orb2m[orb] = m
        orb = orb + 1
  return(orb2m)

#
#
#
def diag_check(self):
  ksn2e = self.xml_dict['ksn2e']
  ac = True
  for k in range(self.nkpoints):
    kvec = self.xml_dict["k2xyzw"][k,0:3]
    for spin in range(self.nspin):
      e,x = sv_diag(self, kvec=kvec, spin=spin)
      eref = ksn2e[k,spin,:]
      acks = numpy.allclose(eref,e,atol=1e-5,rtol=1e-4)
      ac = ac and acks
      if(not acks):
        aerr = sum(abs(eref-e))/len(e)
        print("diag_check: "+bc.RED+str(k)+' '+str(spin)+' '+str(aerr)+bc.ENDC)
  return(ac)

#
#
#
class system_vars_c():
  def __init__(self, label='siesta', Atoms=None, forcetype=-1):

    self.label = label
    self.xml_dict = siesta_xml(self.label)
    self.wfsx = siesta_wfsx_c(self.label)
    self.hsx = siesta_hsx_c(self.label, forcetype)
  
    if Atoms is None:
      self.init_siesta_xml()
    else:
      self.init_ase_atoms(Atoms)

  def init_ase_atoms(self, Atoms):
    """
      Initialise system vars using siesta file and Atom object from ASE.
    """

    try:
      import ase
    except:
      self.init_pure_siesta()

    self.Atoms = Atoms
   
    ##### The parameters as fields     
    self.sp2ion = []
    for sp in Atoms.get_chemical_symbols(): 
      self.sp2ion.append(siesta_ion_xml(sp+self.wfsx.ion_suffix[sp]+'.ion.xml'))
    
    _siesta_ion_add_sp2(self, self.sp2ion)
    self.sp2ao_log = ao_log_c(self.sp2ion)
  
    self.natoms = Atoms.get_positions().shape[0]
    self.norbs  = self.wfsx.norbs 
    self.nspin  = self.wfsx.nspin
    self.nkpoints  = self.wfsx.nkpoints

    strspecie2sp = {}
    for sp in range(len(self.wfsx.sp2strspecie)): strspecie2sp[self.wfsx.sp2strspecie[sp]] = sp
    
    self.atom2sp = numpy.empty((self.natoms), dtype='int64')
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

 
  def init_siesta_xml(self):
    """
    Initialise system var using only the siesta files.
    """
    self.Atoms = None
    
    ##### The parameters as fields     
    self.sp2ion = []
    for sp in self.wfsx.sp2strspecie:
      self.sp2ion.append(siesta_ion_xml(sp+self.wfsx.ion_suffix[sp]+'.ion.xml'))
    
    _siesta_ion_add_sp2(self, self.sp2ion)
    self.ao_log = ao_log_c(self.sp2ion)
  
    self.natoms = len(self.xml_dict['atom2sp'])
    self.norbs  = self.wfsx.norbs 
    self.nspin  = self.wfsx.nspin
    self.nkpoints  = self.wfsx.nkpoints

    strspecie2sp = {}
    for sp in range(len(self.wfsx.sp2strspecie)): strspecie2sp[self.wfsx.sp2strspecie[sp]] = sp
    
    self.atom2sp = numpy.empty((self.natoms), dtype='int64')
    for o in range(self.wfsx.norbs):
      atom = self.wfsx.orb2atm[o]
      strspecie = self.wfsx.orb2strspecie[o]
      self.atom2sp[atom-1] = strspecie2sp[strspecie]

    orb2m = get_orb2m(self)
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)
    
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.X[:,:,n,s,k])
