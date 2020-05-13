from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao.scf import scf

class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    
    dname = os.path.dirname(os.path.abspath(__file__))
    myhf = scf(label='water', cd=dname, verbosity=0, kmat_timing=0.0, kmat_algo='ac_vertex_fm')
    myhf.kernel_scf()
    #print('myhf.kmat_timing ', myhf.kmat_timing)
    #self.assertAlmostEqual(myhf.mo_energy[0,0,0], -1.3274696934511327)
    #self.assertAlmostEqual(myhf.mo_energy[0,0,22], 3.9299990077335423)

    evv = """-1.327562175613441253e+00
-6.905262939172691627e-01
-5.589475688235930884e-01
-4.922020424846831887e-01
2.290155206378258479e-01
3.174893463193735887e-01
5.540345137648966523e-01
6.130321041903931123e-01
6.472060299563430208e-01
6.860482088525959865e-01
7.660860668588928002e-01
1.047708035123237069e+00
1.150818715246961110e+00
1.325653294531685900e+00
1.339787239142159470e+00
1.462064190170056399e+00
1.849090417072683845e+00
1.919615876645932140e+00
1.994329957715350155e+00
2.136971257834380022e+00
2.167793445723626089e+00
2.916247161227477847e+00
3.929996366084439696e+00"""

    np.savetxt('test_0041_rescf_scf_mo_energy.txt', myhf.mo_energy[0,0,:].T)
    
    for eref, e in zip(evv.splitlines(), myhf.mo_energy[0,0,:]):
      self.assertAlmostEqual(float(eref), e)

if __name__ == "__main__": unittest.main()
