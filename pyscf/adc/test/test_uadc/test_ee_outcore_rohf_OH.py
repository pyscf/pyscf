from pyscf.adc.uadc_ee import get_spin_square
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
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
    basis = 'cc-pVDZ'

    mol = gto.Mole()
    mol.atom = '''
        O 0.00000000 0.00000000 -0.10864763
        H 0.00000000 0.00000000 1.72431679
         '''
    mol.basis = {'H': basis,
                 'O': basis,}
    mol.verbose = 0
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.build()

    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc


class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0017134635, 6)
        self.assertAlmostEqual(e[1],0.1629470674, 6)
        self.assertAlmostEqual(e[2],0.2991743493, 6)
        self.assertAlmostEqual(e[3],0.3369031510, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00255436, 6)
        self.assertAlmostEqual(p[2],0.00356489, 6)
        self.assertAlmostEqual(p[3],0.01811875, 6)

        self.assertAlmostEqual(spin[0],0.75053521 , 5)
        self.assertAlmostEqual(spin[1],0.75064928 , 5)
        self.assertAlmostEqual(spin[2],2.43570834 , 5)
        self.assertAlmostEqual(spin[3],1.14753801 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0126491070, 6)
        self.assertAlmostEqual(e[1], 0.1437881080, 6)
        self.assertAlmostEqual(e[2], 0.2676394327, 6)
        self.assertAlmostEqual(e[3], 0.2991598135, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00219744 , 6)
        self.assertAlmostEqual(p[2],0.00012567 , 6)
        self.assertAlmostEqual(p[3],0.01591094 , 6)

        self.assertAlmostEqual(spin[0], 0.75043297 , 5)
        self.assertAlmostEqual(spin[1],0.75037407  , 5)
        self.assertAlmostEqual(spin[2],3.65784145  , 5)
        self.assertAlmostEqual(spin[3],0.80459280  , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0019456025, 6)
        self.assertAlmostEqual(e[1], 0.1567874644, 6)
        self.assertAlmostEqual(e[2], 0.2837638864, 6)
        self.assertAlmostEqual(e[3], 0.3187665826, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00239679 , 6)
        self.assertAlmostEqual(p[2],0.00000001 , 6)
        self.assertAlmostEqual(p[3],0.01492111 , 6)

        self.assertAlmostEqual(spin[0], 0.75027556 , 5)
        self.assertAlmostEqual(spin[1],0.75009053  , 5)
        self.assertAlmostEqual(spin[2],3.72613641  , 5)
        self.assertAlmostEqual(spin[3],0.75901326  , 5)
if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
