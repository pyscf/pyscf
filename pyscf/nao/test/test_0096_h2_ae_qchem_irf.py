from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf, gw as gto_gw
from pyscf.nao.qchem_inter_rf import qchem_inter_rf

mol = gto.M( verbose = 1,
    atom = '''
        H    0.0   2.0      0.0
        H   -1.0   0.0      0.0
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pVDZ',)

class KnowValues(unittest.TestCase):
    
  def test_qchem_irf(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    gto_hf = scf.RHF(mol)
    gto_hf.kernel()
    print(gto_hf.mo_energy)
    
    qrf = qchem_inter_rf(mf=gto_hf, gto=mol, pb_algorithm='fp', verbosity=1)
    print(qrf.s2omega, qrf.s2z)
    
if __name__ == "__main__": unittest.main()
