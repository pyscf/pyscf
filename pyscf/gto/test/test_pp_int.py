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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
from pyscf import gto
from pyscf import lib
from pyscf.gto import pp_int

class KnownValues(unittest.TestCase):
    def test_pp_int(self):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        mol.pseudo = 'gth-hf-rev'
        mol.basis = 'cc-pvdz'
        mol.precision = 1e-10
        mol.build()
        # nimgs is not needed for pyscf-2.4. Set nimgs for pp_int in old version
        mol.nimgs = 0
        h = pp_int.get_gth_pp(mol)
        self.assertAlmostEqual(lib.fp(h), -26.02782083310315, 9)

if __name__ == "__main__":
    print("Full Tests for PP int")
    unittest.main()
