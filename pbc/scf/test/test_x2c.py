#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import df

lib.param.LIGHT_SPEED = 2
cell = gto.Cell()
cell.build(unit = 'B',
           a = numpy.eye(3)*4,
           gs = [5]*3,
           atom = 'H 0 0 0; H 0 0 1.8',
           verbose = 0,
           basis='sto3g')

class KnowValues(unittest.TestCase):
    def test_hf(self):
        mf = scf.sfx2c1e(scf.RHF(cell))
        mf.with_df = df.PWDF(cell)
        dm = mf.get_init_guess()
        h1 = mf.get_hcore()
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.48262776581, 9)
        kpts = cell.make_kpts([3,1,1])
        h1 = mf.get_hcore(kpt=kpts[1])
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.07153704875+0j, 9)

    def test_khf(self):
        mf = scf.sfx2c1e(scf.KRHF(cell))
        mf.with_df = df.PWDF(cell)
        mf.kpts = cell.make_kpts([3,1,1])
        dm = mf.get_init_guess()
        h1 = mf.get_hcore()
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]), -0.48262776581+0j, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]), -0.07153704875+0j, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]), -0.07153704875+0j, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.x2c")
    unittest.main()
