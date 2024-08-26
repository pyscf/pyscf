#!/usr/bin/env python
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
import scipy.linalg
from pyscf import gto
from pyscf.tools import mo_mapping

class KnowValues(unittest.TestCase):
    def test_mo_map(self):
        mol1 = gto.M(atom = '''O 0 0 0; O 0 0 1''', basis='6-31g')
        mol2 = gto.M(atom = '''O 0 0 0; O 0 0 1''', basis='ccpvdz')
        numpy.random.seed(1)
        mo1 = numpy.random.random((mol1.nao_nr(), 10))*.2
        mo2 = numpy.random.random((mol2.nao_nr(), 15))*.2
        idx,s = mo_mapping.mo_map(mol1, mo1, mol2, mo2)
        self.assertTrue(numpy.allclose(idx,
            [[1,11], [2,11], [7,11], [8,11], [9,11],]))
        self.assertAlmostEqual(abs(s).sum(), 54.5514791873481, 9)

    def test_mo_1to1map(self):
        mol1 = gto.M(atom = '''O 0 0 0; O 0 0 1''', basis='6-31g')
        mol2 = gto.M(atom = '''O 0 0 0; O 0 0 1''', basis='ccpvdz')
        s = gto.intor_cross('int1e_ovlp_sph', mol1, mol2)
        idx = mo_mapping.mo_1to1map(s)
        self.assertTrue(numpy.allclose(idx,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22]))

    def test_mo_comps(self):
        mol = gto.M(atom = '''O 0 0 0; O 0 0 1''', basis='ccpvdz')
        numpy.random.seed(1)

        mo = numpy.random.random((mol.nao_nr(), 15))
        idx4d = [i for i,s in enumerate(mol.ao_labels()) if '4d' in s]
        c = mo_mapping.mo_comps(idx4d, mol, mo, cart=False)
        self.assertAlmostEqual(abs(c).sum(), 0, 12)

        # FIXME:
        # mo = numpy.random.random((mol.nao_cart(), 15)) * .2
        # c = mo_mapping.mo_comps(lambda x: '3d' in x, mol, mo, cart=True)
        # self.assertAlmostEqual(abs(c).sum(), 1.0643140119943388, 9)

if __name__ == "__main__":
    print("Full Tests for mo_mapping")
    unittest.main()
