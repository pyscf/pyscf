import scipy
import numpy
#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import pyscf.pbc.gto as pbcgto
from pyscf.pbc import dft as pdft
from pyscf.pbc import tools as ptools


cell = pbcgto.Cell()
cell.atom = '''
C 0              0              0
C 1.685068785275 1.685068785275 1.685068785275'''
cell.a = '''
0.000000000, 3.370137571, 3.370137571
3.370137571, 0.000000000, 3.370137571
3.370137571, 3.370137571, 0.000000000
'''
cell.basis = 'gth-szv'
cell.unit = 'B'
cell.pseudo = 'gth-pade'
cell.gs = [12]*3
cell.verbose = 0
cell.build()


class KnowValues(unittest.TestCase):
    def test_gamma_vs_ks(self):
        mf = pdft.KRKS(cell)
        mf.kpts = cell.make_kpts([1,1,3])
        ek = mf.kernel()

        scell = ptools.super_cell(cell, [1,1,3])
        scell.gs = [12,12,36]
        mf = pdft.RKS(scell)
        eg = mf.kernel()
        self.assertAlmostEqual(ek, eg/3, 5)


if __name__ == '__main__':
    print("Full Tests for gamma point vs k-points")
    unittest.main()

