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

cell = gto.Cell()
cell.build(unit = 'B',
           a = numpy.eye(3)*4,
           mesh = [11]*3,
           atom = 'H 0 0 0; H 0 0 1.8',
           verbose = 0,
           basis='sto3g')

class KnownValues(unittest.TestCase):
    def test_hf(self):
        lib.param.LIGHT_SPEED, c = 2, lib.param.LIGHT_SPEED
        mf = scf.RHF(cell).sfx2c1e()
        mf.with_df = df.PWDF(cell)
        dm = mf.get_init_guess()
        h1 = mf.get_hcore()
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.47578184212352159+0j, 9)
        kpts = cell.make_kpts([3,1,1])
        h1 = mf.get_hcore(kpt=kpts[1])
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm, h1), -0.09637799091491725+0j, 9)
        lib.param.LIGHT_SPEED = c

    def test_khf(self):
        lib.param.LIGHT_SPEED, c = 2, lib.param.LIGHT_SPEED
        mf = scf.KRHF(cell).sfx2c1e()
        mf.with_df = df.PWDF(cell)
        mf.kpts = cell.make_kpts([3,1,1])
        dm = mf.get_init_guess()
        h1 = mf.get_hcore()
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm[0], h1[0]),-0.47578184212352159+0j, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm[1], h1[1]),-0.09637799091491725+0j, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji', dm[2], h1[2]),-0.09637799091491725+0j, 9)
        lib.param.LIGHT_SPEED = c


if __name__ == '__main__':
    print("Full Tests for pbc.scf.x2c")
    unittest.main()
