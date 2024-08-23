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
    global cell, myadc, kadc
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

    # treating supercell at gamma point
    supcell = super_cell(cell,nmp)
    mf  = scf.RHF(supcell,exxdiv=None).density_fit()
    ehf  = mf.kernel()
    myadc = mol_adc.RADC(mf)
    myadc.method_type = 'ea'
    myadc.approx_trans_moments = True

    # periodic calculation at gamma point
    kpts = cell.make_kpts((nmp))
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
    kadc  = adc.KRADC(kmf)
    kadc.method_type = 'ea'

def tearDownModule():
    global cell, myadc, kadc
    del cell, myadc, kadc

class KnownValues(unittest.TestCase):
    def test_ea_adc2_supercell(self):
        e1,v1,p1,x1 = myadc.kernel(nroots=4)
        self.assertAlmostEqual(e1[1], 0.83425538, 6)
        self.assertAlmostEqual(e1[3], 1.29595017, 6)

        self.assertAlmostEqual(p1[0], 1.98185661, 6)
        self.assertAlmostEqual(p1[1], 1.96030165, 6)
        self.assertAlmostEqual(p1[3], 0.00398214, 6)

    def test_ea_adc2_supercell_vs_k(self):
        e1,v1,p1,x1 = myadc.kernel(nroots=5)
        e2, v2, p2, x2 = kadc.kernel(nroots=5,kptlist=[0])

        ediff = e1[1] - e2[0][0]
        pdiff = p1[1] - p2[0][0]

        self.assertAlmostEqual(ediff, 0.00000000, 4)
        self.assertAlmostEqual(pdiff, 0.00000000, 4)

    def test_ea_adc2x_supercell_vs_k(self):

        myadc.method = 'adc(2)-x'
        e1,v1,p1,x1 = myadc.kernel(nroots=3)

        kadc.method = 'adc(2)-x'
        e2, v2, p2, x2 = kadc.kernel(nroots=3,kptlist=[0])

        ediff = e1[1] - e2[0][0]
        pdiff = p1[1] - p2[0][0]

        self.assertAlmostEqual(ediff, 0.00000000, 4)
        self.assertAlmostEqual(pdiff, 0.00000000, 4)

    def test_ea_adc3_supercell_vs_k(self):

        myadc.method = 'adc(3)'
        e1,v1,p1,x1 = myadc.kernel(nroots=3)

        kadc.method = 'adc(3)'
        e2, v2, p2, x2 = kadc.kernel(nroots=3,kptlist=[0])

        ediff = e1[1] - e2[0][0]
        pdiff = p1[1] - p2[0][0]

        self.assertAlmostEqual(ediff, 0.00000000, 3)
        self.assertAlmostEqual(pdiff, 0.00000000, 3)

if __name__ == "__main__":
    print("Supercell vs k-point calculations for EA-ADC methods")
    unittest.main()
