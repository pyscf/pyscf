#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto, scf
from pyscf.solvent import ddpcm


class KnownValues(unittest.TestCase):
    def test_ddpcm_scf(self):
        mol = gto.M(atom='''
               6        0.000000    0.000000   -0.542500
               8        0.000000    0.000000    0.677500
               1        0.000000    0.935307   -1.082500
               1        0.000000   -0.935307   -1.082500
                    ''', basis='sto3g', verbose=7,
                    output='/dev/null')
        pcm = ddpcm.DDPCM(mol)
        pcm.lmax = 6
        pcm.lebedev_order = 17
        mf = scf.RHF(mol).ddPCM(pcm).run()
        self.assertAlmostEqual(mf.e_tot, -112.3544929827, 8)


if __name__ == "__main__":
    print("Full Tests for ddpcm")
    unittest.main()

