# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#
import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import scf,adc,mp
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell

def setUpModule():
    global cell, kadc
    cell = gto.M(
        unit='B',
        a=[[0.,          3.37013733,  3.37013733],
             [3.37013733,  0.,          3.37013733],
             [3.37013733,  3.37013733,  0.        ]],
        mesh=[13]*3,
        atom='''He 0 0 0
                  He 1.68506866 1.68506866 1.68506866''',
        basis='gth-dzv',
        pseudo='gth-pade',
        verbose=0,
    )

    nmp = [1,1,2]

    # periodic calculation at gamma point
    kpts = cell.make_kpts((nmp))
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
    kadc  = adc.KRADC(kmf)
    kadc.method_type = 'ea'

def tearDownModule():
    global cell, kadc
    del cell, kadc

class KnownValues(unittest.TestCase):

    def test_ea_adc2_k(self):

        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.83425538, 4)
        self.assertAlmostEqual(e[0][1], 1.29595017, 4)
        self.assertAlmostEqual(e[0][2], 1.68125009, 4)

        self.assertAlmostEqual(p[0][0], 1.96030215, 4)
        self.assertAlmostEqual(p[0][1], 0.00398257, 4)
        self.assertAlmostEqual(p[0][2], 0.00000299, 4)

    def test_ea_adc2x_k_high_cost(self):

        nmp = [2,2,2]
        kpts = cell.make_kpts((nmp))
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
        kadc  = adc.KRADC(kmf)
        kadc.method = 'adc(2)-x'
        kadc.method_type = 'ea'
        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.82483536, 4)
        self.assertAlmostEqual(e[0][1], 1.38987893, 4)
        self.assertAlmostEqual(e[0][2], 1.38987895, 4)

        self.assertAlmostEqual(p[0][0], 1.95209795, 4)
        self.assertAlmostEqual(p[0][1], 0.00000001, 4)
        self.assertAlmostEqual(p[0][2], 0.00000002, 4)

    def test_ea_adc3_k_skip(self):

        kadc.method = 'adc(3)'
        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.83386812, 4)
        self.assertAlmostEqual(e[0][1], 1.26993734, 4)
        self.assertAlmostEqual(e[0][2], 1.56058118, 4)

        self.assertAlmostEqual(p[0][0], 1.95985989, 4)
        self.assertAlmostEqual(p[0][1], 0.00111690, 4)
        self.assertAlmostEqual(p[0][2], 0.00385444, 4)

if __name__ == "__main__":
    print("k-point calculations for EA-ADC methods")
    unittest.main()
