from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_dft_sv(self):
    """ Try to run pySCF's DFT with system_vars_c """
    from pyscf import dft
    from pyscf.nao import system_vars_c

    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', chdir=dname)
    self.assertEqual(sv.atom_nelec_core(0), 2)
    for ia in range(1,3): self.assertEqual(sv.atom_nelec_core(ia), 0)

    mydft = dft.RKS(sv)
    #mydft.xc = 'lda,vwn'  #; mydft.xc = 'b3lyp'
    #mydft.kernel()

    # Orbital energies, Mulliken population etc.
    #mydft.analyze()
  
if __name__ == "__main__": unittest.main()
