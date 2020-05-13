from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import gw
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0167_cn_ugw_spin_contam(self):
    """ Interacting case """
    spin = 1
    mol=gto.M(verbose=0,atom='C 0 0 -0.6;N 0 0 0.52',basis='cc-pvdz',spin=spin)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()

    ss_2sp1 = gto_mf.spin_square()
    self.assertTrue(np.allclose(ss_2sp1, (0.9768175623447295, 2.2152359353754889)))
    
    nao_gw = gw(gto=mol, mf=gto_mf, verbosity=0)
    nao_gw.kernel_gw()
    #nao_gw.report()
    
    ss_2sp1 = nao_gw.spin_square()
    #print(' nao_gw.spin_square() ', ss_2sp1)
    self.assertTrue(np.allclose(ss_2sp1, (0.9767930087836918, 2.2152137673675574)))
    


if __name__ == "__main__": unittest.main()
