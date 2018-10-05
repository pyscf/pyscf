from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import mf as mf_c
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0166_cn_uhf_spin_contamination(self):
    """ Interacting case """
    spin = 1
    mol=gto.M(verbose=0,atom='C 0 0 -0.6;N 0 0 0.52',basis='cc-pvdz',spin=spin)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()

    #print( ' gto_mf.spin_square() ', gto_mf.spin_square(), spin*0.5*(spin*0.5+1), spin+1 )

    ss_2sp1 = gto_mf.spin_square()
    self.assertTrue(np.allclose(ss_2sp1, (0.9768175623447295, 2.2152359353754889)))
    
    nao_mf = mf_c(gto=mol, mf=gto_mf, verbosity=0, gen_pb=False)
    
    ss_2sp1 = nao_mf.spin_square()
    self.assertTrue(np.allclose(ss_2sp1, (0.9767930087836918, 2.2152137673675574)))
    
    #print(' nao_mf.spin_square() ', nao_mf.spin_square())

if __name__ == "__main__": unittest.main()
