from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import gw
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0168_cn_rohf(self):
    """ Interacting case """
    spin = 1
    mol=gto.M(verbose=0,atom='C 0 0 -0.6;N 0 0 0.52',basis='cc-pvdz',spin=spin)
    gto_mf = scf.ROHF(mol)
    gto_mf.kernel()

    ss_2sp1 = gto_mf.spin_square()
    
    nao_gw = gw(gto=mol, mf=gto_mf, verbosity=0, nocc=4)
    nao_gw.kernel_gw()
    #nao_gw.report()
    #print(nao_gw.spin_square())


if __name__ == "__main__": unittest.main()
