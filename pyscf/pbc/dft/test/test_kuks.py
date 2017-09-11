#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

class KnowValues(unittest.TestCase):
    def test_klda8_cubic_kpt_222(self):
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.a = '''3.5668  0.      0.
                    0.      3.5668  0.
                    0.      0.      3.5668'''
        cell.gs = np.array([8]*3)
        cell.atom ='''
C, 0.,  0.,  0.
C, 0.8917,  0.8917,  0.8917
C, 1.7834,  1.7834,  0.
C, 2.6751,  2.6751,  0.8917
C, 1.7834,  0.    ,  1.7834
C, 2.6751,  0.8917,  2.6751
C, 0.    ,  1.7834,  1.7834
C, 0.8917,  2.6751,  2.6751'''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        kpts = cell.make_kpts((2,2,2), with_gamma_point=False)
        mf = pbcdft.KUKS(cell, kpts)
        mf.xc = 'lda,vwn'
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.42583489512954, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.kuks")
    unittest.main()

