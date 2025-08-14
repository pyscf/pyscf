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
import numpy as np
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
    mol.verbose = 0
    mol.symmetry = "c2v"
    mol.spin = 1
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
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789393239, 6)
        self.assertAlmostEqual(e[1],0.0789393239, 6)
        self.assertAlmostEqual(e[2],0.1397085217, 6)
        self.assertAlmostEqual(e[3],0.2552893678, 6)

        self.assertAlmostEqual(p[0],0.00403599, 6)
        self.assertAlmostEqual(p[1],0.00403599, 6)
        self.assertAlmostEqual(p[2],0.02229693, 6)
        self.assertAlmostEqual(p[3],0.00597127, 6)

        self.assertAlmostEqual(spin[0],0.81903415 , 5)
        self.assertAlmostEqual(spin[1],0.81903415 , 5)
        self.assertAlmostEqual(spin[2],0.97833065 , 5)
        self.assertAlmostEqual(spin[3],2.70435538 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0066160563, 6)
        self.assertAlmostEqual(e[1],0.0066160563, 6)
        self.assertAlmostEqual(e[2],0.0674217414, 6)
        self.assertAlmostEqual(e[3],0.1755586417, 6)

        self.assertAlmostEqual(p[0],0.00027751, 6)
        self.assertAlmostEqual(p[1],0.00027751, 6)
        self.assertAlmostEqual(p[2],0.01004905, 6)
        self.assertAlmostEqual(p[3],0.00001243, 6)

        self.assertAlmostEqual(spin[0],0.76591155 , 5)
        self.assertAlmostEqual(spin[1],0.76591155 , 5)
        self.assertAlmostEqual(spin[2],0.77023553 , 5)
        self.assertAlmostEqual(spin[3],4.03885373 , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0432157465, 6)
        self.assertAlmostEqual(e[1],0.0432157465, 6)
        self.assertAlmostEqual(e[2],0.1276752421, 6)
        self.assertAlmostEqual(e[3],0.1848576902, 6)

        self.assertAlmostEqual(p[0],0.00192927, 6)
        self.assertAlmostEqual(p[1],0.00192927, 6)
        self.assertAlmostEqual(p[2],0.01278698, 6)
        self.assertAlmostEqual(p[3],0.00014258, 6)

        self.assertAlmostEqual(spin[0],0.78024113, 5)
        self.assertAlmostEqual(spin[1],0.78024113, 5)
        self.assertAlmostEqual(spin[2],0.79826249, 5)
        self.assertAlmostEqual(spin[3],4.10509903, 5)
if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
