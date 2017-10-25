from __future__ import print_function, division
import unittest
from pyscf import gto

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_dipole_coo(self):
    """ Test computation of dipole matrix elements """
    from pyscf.nao import nao
    sv = nao(gto=mol)
    dipme = sv.dipole_coo()
    
    self.assertAlmostEqual(dipme[0].sum(), 23.8167121803)
    self.assertAlmostEqual(dipme[1].sum(), 18.9577251654)
    self.assertAlmostEqual(dipme[2].sum(), 48.1243277097)

#    self.assertAlmostEqual(dipme[0].sum(), 23.816263714841725)
#    self.assertAlmostEqual(dipme[1].sum(), 18.958562546276568)    
#    self.assertAlmostEqual(dipme[2].sum(), 48.124023241543377)

if __name__ == "__main__": unittest.main()
