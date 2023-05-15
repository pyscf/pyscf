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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import df


class KnownValues(unittest.TestCase):
    def test_aug_etb(self):
        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = 'cc-pvdz',
        )
        df.addons.aug_etb(mol)

        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = ('cc-pvdz', [[4, (1., 1.)]])
        )
        auxbasis = df.addons.aug_etb(mol)
        self.assertEqual(len(auxbasis['O']), 36)
        self.assertEqual(len(auxbasis['H']), 12)

    def test_make_auxbasis(self):
        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = {'default': 'cc-pvdz', 'O':'631g'}
        )
        auxmol = df.addons.make_auxmol(mol, auxbasis={'default':'ccpvdz-fit'})
        self.assertEqual(auxmol.nao_nr(), 146)
        auxbasis = df.addons.make_auxbasis(mol, mp2fit=True)
        self.assertEqual(auxbasis['O'], 'cc-pvdz-ri')
        self.assertEqual(auxbasis['H'], 'cc-pvdz-ri')

        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = {'H': [[0,(1.,1.)], [1, (.5, 1.)]],
                     'O': ('631g', [[0, 0, (1., 1.)]])}
        )
        auxbasis = df.addons.make_auxbasis(mol)
        self.assertEqual(len(auxbasis['O']), 32)
        self.assertEqual(len(auxbasis['H']), 3)

    def test_default_auxbasis(self):
        mol = gto.M(atom='He 0 0 0; O 0 0 1', basis='ccpvdz')
        auxbasis = df.addons.make_auxbasis(mol)
        self.assertTrue(auxbasis['O'] == 'cc-pvdz-jkfit')
        self.assertTrue(isinstance(auxbasis['He'], list))


if __name__ == "__main__":
    print("Full Tests for df.addons")
    unittest.main()
