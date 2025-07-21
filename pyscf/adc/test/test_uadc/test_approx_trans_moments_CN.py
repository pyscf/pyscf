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
# Author: Terrence Stahl <terrencestahl1@@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc
    basis = 'cc-pVDZ'
    mol = gto.Mole()
    mol.atom = '''
        C 0.00000000 0.00000000 -1.18953886
        N 0.00000000 0.00000000 1.01938091
         '''
    mol.basis = {'C': basis,
                 'N': basis,}
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc


class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.approx_trans_moments = True
        myadc.compute_spin_square = True

        e,v,p,x = myadc.kernel(nroots=5)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789393239, 6)
        self.assertAlmostEqual(e[1],0.0789393239, 6)
        self.assertAlmostEqual(e[2],0.1397085217, 6)
        self.assertAlmostEqual(e[3],0.2552893678, 6)

        self.assertAlmostEqual(p[0],0.00338362, 6)
        self.assertAlmostEqual(p[1],0.00338362, 6)
        self.assertAlmostEqual(p[2],0.01188668, 6)
        self.assertAlmostEqual(p[3],0.00598859, 6)

        self.assertAlmostEqual(spin[0],0.88753961, 5)
        self.assertAlmostEqual(spin[1],0.88753961, 5)
        self.assertAlmostEqual(spin[2],1.10172022, 5)
        self.assertAlmostEqual(spin[3],2.66684999, 5)

    def test_ea_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ea"
        myadc.approx_trans_moments = True

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], -0.1123860177, 6)
        self.assertAlmostEqual(e[1],  0.1309866865, 6)
        self.assertAlmostEqual(e[2],  0.1309866865, 6)
        self.assertAlmostEqual(e[3],  0.1652852298, 6)

        self.assertAlmostEqual(p[0], 0.91914237, 6)
        self.assertAlmostEqual(p[1], 0.92740352, 6)
        self.assertAlmostEqual(p[2], 0.92740352, 6)
        self.assertAlmostEqual(p[3], 0.93979624, 6)


    def test_ip_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        myadc.approx_trans_moments = True

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4963840022, 6)
        self.assertAlmostEqual(e[1], 0.4963840022, 6)
        self.assertAlmostEqual(e[2], 0.5237162997, 6)
        self.assertAlmostEqual(e[3], 0.5237162997, 6)

        self.assertAlmostEqual(p[0], 0.89871663, 6)
        self.assertAlmostEqual(p[1], 0.89871663, 6)
        self.assertAlmostEqual(p[2], 0.93748642, 6)
        self.assertAlmostEqual(p[3], 0.93748642, 6)


if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
