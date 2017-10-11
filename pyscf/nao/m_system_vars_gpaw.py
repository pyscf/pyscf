from __future__ import print_function, division
import numpy as np
import sys

def system_vars_gpaw(self, calc, label="gpaw", chdir='.', **kvargs):
  """
    Initialise system variables with gpaw inputs:
    Input parameters:
    -----------------
      calc: GPAW object
      kvargs: optional arguments, need a list of precise arguments somewhere
  """
  from pyscf.lib import logger
  from pyscf.lib.parameters import ELEMENTS as chemical_symbols
  from pyscf.nao.m_gpaw_wfsx import gpaw_wfsx_c
  from pyscf.nao.m_gpaw_hsx import gpaw_hsx_c
  from pyscf.nao.m_ao_log import ao_log_c
  import ase.units as units

  self.label = label
  self.chdir = '.'
  self.verbose = logger.NOTE
  
  self.ao_log = ao_log_c().init_ao_log_gpaw(calc.setups)
  self.atom2coord = calc.get_atoms().get_positions()/units.Bohr
  self.natm = self.natoms = len(self.atom2coord)
  
  self.atom2sp = np.array([self.ao_log.sp2key.index(key) for key in calc.setups.id_a], dtype=np.int64)
  self.ucell = calc.atoms.get_cell()/units.Bohr
  self.norbs = calc.setups.nao
  self.norbs_sc = self.norbs
  self.nspin = calc.get_number_of_spins()
  self.nkpoints  = 1
  self.fermi_energy = float(calc.get_fermi_level()/units.Ha) # ensure that fermi_energy is float type
  self.atom2s = np.zeros((self.natm+1), dtype=np.int64)

  for atom, sp in enumerate(self.atom2sp):
    self.atom2s[atom+1] = self.atom2s[atom] + self.ao_log.sp2norbs[sp]

  self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
  for atom, sp in enumerate(self.atom2sp):
    self.atom2mu_s[atom+1] = self.atom2mu_s[atom] + self.ao_log.sp2nmult[sp]

  self.sp2symbol = [chemical_symbols[Z] for Z in self.ao_log.sp2charge]
  self.sp2charge = self.ao_log.sp2charge
  self.wfsx = gpaw_wfsx_c(calc)

  self.hsx = gpaw_hsx_c(self, calc)

  #print(self.atom2coord)
  #print(self.natoms)
  #print(self.atom2sp)
  #print(self.norbs)
  #print(self.nspin)
  #print(self.nkpoints)
  #print(self.fermi_energy)
  #print(self.atom2s)
  #print(self.atom2mu_s)
  #print(self.sp2symbol)
  #print(self.sp2charge)

  self.state = 'should be useful for something'

  return self
  
  
