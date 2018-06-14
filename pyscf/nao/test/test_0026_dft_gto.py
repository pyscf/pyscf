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
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_dft_gto(self):
    #!/usr/bin/env python
    #
    # Author: Qiming Sun <osirpt.sun@gmail.com>
    #
    from pyscf import gto, dft
    '''
    A simple example to run DFT calculation.
    See pyscf/dft/vxc.py for the complete list of available XC functional
    '''
    
    mol = gto.Mole().build( atom = 'H 0 0 0; F 0 0 1.1', basis = '631g', verbose = 0)    
    #print(mol._atom)
    #print(mol.nao_nr())
    mydft = dft.RKS(mol)
    mydft.xc = 'lda,vwn'  #; mydft.xc = 'b3lyp'
    mydft.kernel()

    # Orbital energies, Mulliken population etc.
    #mydft.analyze()
  
if __name__ == "__main__": unittest.main()
