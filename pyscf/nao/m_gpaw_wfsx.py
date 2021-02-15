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

from __future__ import print_function, division
import os
import sys
import numpy as np
from numpy import zeros, empty 
import warnings

class gpaw_wfsx_c():
  
  def __init__(self, calc):
    """
        Gathers the information on the available wavefunctions
        (Kohn-Sham or Hartree-Fock orbitals)
    """
    assert calc.wfs.mode.lower()=='lcao'

    self.nreim = 1 # Only real part? because wavefunctions from gpaw are complex
    self.nspin = calc.get_number_of_spins()
    self.norbs = calc.setups.nao
    self.nbands= calc.parameters['nbands']
    self.k2xyz = calc.parameters['kpts']
    self.nkpoints = len(self.k2xyz)

    self.ksn2e = np.zeros((self.nkpoints, self.nspin, self.nbands))
    for ik in range(self.nkpoints):
      for spin in range(self.nspin):
        self.ksn2e[ik, spin, :] = calc.wfs.collect_eigenvalues(spin,ik)

    # Import wavefunctions from GPAW calculator
    self.x = np.zeros((self.nkpoints, self.nspin, self.nbands, self.norbs, self.nreim))
    for k in range(calc.wfs.kd.nibzkpts):
        for s in range(calc.wfs.nspins):
            C_nM = calc.wfs.collect_array('C_nM', k, s)
            self.x[k, s, :, :, 0] = C_nM.real
