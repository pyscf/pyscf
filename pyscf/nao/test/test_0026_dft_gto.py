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
