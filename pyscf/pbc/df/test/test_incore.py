# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import incore
import pyscf.pbc
pyscf.pbc.DEBUG = False


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_aux_e2(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        cell.a = numpy.eye(3) * 3.
        cell.mesh = numpy.array([41]*3)
        cell.atom = 'He 0 1 1; He 1 1 0'
        cell.basis = { 'He': [[0, (0.8, 1.0)],
                              [0, (1.2, 1.0)]] }
        cell.verbose = 0
        cell.build(0, 0)
        auxcell = incore.format_aux_basis(cell)
        a1 = incore.aux_e2(cell, auxcell, 'int3c1e_sph')
        self.assertAlmostEqual(finger(a1), 0.1208944790152819, 9)
        a2 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', aosym='s2ij')
        self.assertTrue(numpy.allclose(a1, lib.unpack_tril(a2, axis=0).reshape(a1.shape)))

        numpy.random.seed(3)
        kpt = numpy.random.random(3)
        kptij_lst = numpy.array([[kpt,kpt]])
        a1 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', kptij_lst=kptij_lst)
        self.assertAlmostEqual(finger(a1), -0.073719031689332651-0.054002639392614758j, 9)
        a2 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', aosym='s2', kptij_lst=kptij_lst)
        self.assertTrue(numpy.allclose(a1, lib.unpack_tril(a2, 1, axis=0).reshape(a1.shape)))

        numpy.random.seed(1)
        kptij_lst = numpy.random.random((1,2,3))
        a1 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', kptij_lst=kptij_lst)
        self.assertAlmostEqual(finger(a1), 0.039329191948685879-0.039836453846241987j, 9)

if __name__ == '__main__':
    print("Full Tests for pbc.df.incore")
    unittest.main()

