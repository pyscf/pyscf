#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import lib
import pyscf.lib.parameters as param

mol0 = gto.Mole()
mol0.atom = [
    [1  , (0.,1.,1.)],  # D
    ["O1", (0.,0.,0.)],
    [1  , (1.,1.,0.)], ] # H
mol0.nucmod = { "O":param.MI_NUC_GAUSS, 3:param.MI_NUC_GAUSS }
mol0.mass = { "O":18, 1:3 }
mol0.basis = {
    "O": [(0, 0, (15, 1)), ],
    "H": [(0, 0, (1, 1, 0), (3, 3, 1), (5, 1, 0)),
          (1, 0, (1, 1)), ]}
mol0.symmetry = 1
mol0.charge = 1
mol0.spin = 1
mol0.basis['O'].extend(gto.mole.expand_etbs(((0, 4, 1, 1.8),
                                            (1, 3, 2, 1.8),
                                            (2, 2, 1, 1.8),)))
mol0.verbose = 4
mol0.output = None
mol0.build()

class KnowValues(unittest.TestCase):
    def test_intor_cross(self):
        mol1 = mol0.unpack(mol0.pack())
        mol1.atom = '''
                1    0  1  1
                O    0  0  0
                h    1  1  0'''
        mol1.basis = {'O': gto.basis.parse('''
C    S
   3047.5249000              0.0018347        
    457.3695100              0.0140373        
    103.9486900              0.0688426        
     29.2101550              0.2321844        
      9.2866630              0.4679413        
      3.1639270              0.3623120        
C    SP
      7.8682724             -0.1193324              0.0689991        
      1.8812885             -0.1608542              0.3164240        
      0.5442493              1.1434564              0.7443083        
C    SP
      0.1687144              1.0000000              1.0000000'''),
                      'H': '6-31g'}
        mol1.build()
        v = gto.mole.intor_cross('cint1e_ovlp_sph', mol0, mol1)
        self.assertAlmostEqual(numpy.linalg.norm(v), 7.3215230702679968, 1)

    def test_num_basis(self):
        self.assertEqual(mol0.nao_nr(), 34)
        self.assertEqual(mol0.nao_2c(), 68)

    def test_time_reversal_map(self):
        tao = [-2, 1, -4, 3, 6, -5, 10, -9, 8, -7, -12, 11, -14, 13, -16, 15, -18, 17,
               -20, 19, 22, -21, 26, -25, 24, -23, 28, -27, 32, -31, 30, -29, 34, -33,
               38, -37, 36, -35, -42, 41, -40, 39, -48, 47, -46, 45, -44, 43, -52, 51,
               -50, 49, -58, 57, -56, 55, -54, 53, -60, 59, -62, 61, 64, -63, 68, -67,
               66, -65]
        self.assertEqual(mol0.time_reversal_map(), tao)

    def test_check_sanity(self):
        mol1 = mol0.copy()
        mol1.x = None
        mol1.copy = None
        mol1.check_sanity(mol1)

    def test_nao_range(self):
        self.assertEqual(mol0.nao_nr_range(1,4), (2, 7))
        self.assertEqual(mol0.nao_2c_range(1,4), (4, 14))
        self.assertEqual(numpy.dot(range(mol0.nbas+1), mol0.ao_loc_nr()), 2151)
        self.assertEqual(numpy.dot(range(mol0.nbas+1), mol0.ao_loc_2c()), 4302)

    def test_search_bas(self):
        self.assertEqual(mol0.search_shell_id(1, 1), 7)
        self.assertEqual(mol0.search_ao_nr(1, 1, -1, 5), None)
        self.assertEqual(mol0.search_ao_nr(1, 1, -1, 4), 16)


if __name__ == "__main__":
    print("test mole.py")
    unittest.main()
