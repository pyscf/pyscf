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
    global cell, kpts, kadc
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
    kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
    kadc  = adc.KRADC(kmf)

def tearDownModule():
    global cell, kadc
    del cell, kadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2_k(self):

        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.03211224, 4)
        self.assertAlmostEqual(e[0][1], 0.52088413, 4)
        self.assertAlmostEqual(e[0][2], 0.92916398, 4)

        self.assertAlmostEqual(p[0][0], 1.89132918, 4)
        self.assertAlmostEqual(p[0][1], 1.80157487, 4)
        self.assertAlmostEqual(p[0][2], 0.00004972, 4)

    def test_ip_adc2x_k_high_cost(self):

        nmp = [2,2,2]
        kpts = cell.make_kpts((nmp))
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
        kadc  = adc.KRADC(kmf)
        kadc.method = 'adc(2)-x'
        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.13741437, 4)
        self.assertAlmostEqual(e[0][1], 0.65279043, 4)
        self.assertAlmostEqual(e[0][2], 1.08236251, 4)

        self.assertAlmostEqual(p[0][0], 1.93375487, 4)
        self.assertAlmostEqual(p[0][1], 1.90733872, 4)
        self.assertAlmostEqual(p[0][2], 0.00357151, 4)

    def test_ip_adc3_k(self):

        kmf = scf.KRHF(cell, kpts,exxdiv=None).run()
        kadc  = adc.KRADC(kmf)
        kadc.method = 'adc(3)'
        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.03297725, 4)
        self.assertAlmostEqual(e[0][1], 0.52724032, 4)
        self.assertAlmostEqual(e[0][2], 0.85718537, 4)

        self.assertAlmostEqual(p[0][0], 1.90849228, 4)
        self.assertAlmostEqual(p[0][1], 1.80306363, 4)
        self.assertAlmostEqual(p[0][2], 0.00578080, 4)

if __name__ == "__main__":
    print("k-point calculations for IP-ADC methods")
    unittest.main()
