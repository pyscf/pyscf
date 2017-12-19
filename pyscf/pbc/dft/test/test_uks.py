#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc
pyscf.pbc.DEBUG = False


class KnowValues(unittest.TestCase):
    def test_pp_UKS(self):
        cell = pbcgto.Cell()

        cell.unit = 'A'
        cell.atom = '''
            Si    2.715348700    2.715348700    0.000000000;
            Si    2.715348700    0.000000000    2.715348700;
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'

        Lx = Ly = Lz = 5.430697500
        cell.a = np.diag([Lx,Ly,Lz])
        cell.mesh = np.array([17]*3)

        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        mf = pbcdft.UKS(cell)
        mf.xc = 'blyp'
        self.assertAlmostEqual(mf.scf(), -7.6058004283213396, 8)

        mf.xc = 'lda,vwn'
        self.assertAlmostEqual(mf.scf(), -7.6162130840535092, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.uks")
    unittest.main()
