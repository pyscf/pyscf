from __future__ import print_function, division
import os
import sys
import numpy as np
from numpy import zeros, empty 

class gpaw_wfsx_c():
  
  def __init__(self, calc):
    """ Gathers the information on the available wavefunctions (Kohn-Sham or Hartree-Fock orbitals) """
    assert calc.wfs.mode.lower()=='lcao'

    print(dir(calc))
    #print(dir(calc.get_atoms()))
    #print(dir(calc.scf))
    #print(dir(calc.setups))
    #print(calc.setups.nao)
    print(dir(calc.get_pseudo_wave_function()))
    print(calc.get_pseudo_wave_function().shape)

    self.nreim = 1 # calc.wfs.pbc
    self.nspin = calc.get_number_of_spins()
    self.norbs = calc.setups.nao
    self.nbands= calc.parameters['nbands']
    self.k2xyz = calc.parameters['kpts']
    self.nkpoints = len(self.k2xyz)

    self.ksn2e = np.zeros((self.nkpoints, self.nspin, self.nbands))
    for ik in range(self.nkpoints):
      for spin in range(self.nspin):
        self.ksn2e[ik, spin, :] = calc.wfs.collect_eigenvalues(spin,ik)

    #self.x = np.zeros((self.nkpoints, self.nspin, self.nbands, self.norbs, self.nreim))
    #for ik in range(self.nkpoints):
      #for spin in range(self.nspin):
        #for band in range(self.nbands):
          #self.x[ik, spin, band, :, 0] = calc.wfs.get_wave_function_array(spin,ik,band)

    #print(' dir(calc.wfs) ', dir(calc.wfs) )
    #print(' dir(calc.wfs) collect_eigenvalues ', calc.wfs.collect_eigenvalues(0,0))
    #print(' calc.parameters ', calc.parameters)

    #print(calc.wfs.nspins)
    #print(calc.wfs.nvalence)
    #print(dir(calc.wfs.kpt_u))
    #print(calc.wfs.ksl)
    #print(calc.wfs.mode)
    #print(dir(calc.wfs.basis_functions))
    #print(dir(calc.wfs.atom_partition))
    #print(' x.shape ', x.shape)

