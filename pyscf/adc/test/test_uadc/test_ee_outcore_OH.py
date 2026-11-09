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
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ee import get_spin_square

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

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    myadc.max_cycle = 200

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

        self.assertAlmostEqual(e[0],0.0023522150, 6)
        self.assertAlmostEqual(e[1],0.1647973308, 6)
        self.assertAlmostEqual(e[2],0.2986841630, 6)
        self.assertAlmostEqual(e[3],0.3371941604, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.002642369044511607, 6)
        self.assertAlmostEqual(p[2],0.003610649647366494, 6)
        self.assertAlmostEqual(p[3],0.017775953540500516, 6)

        self.assertAlmostEqual(spin[0],0.7522904063542506 , 5)
        self.assertAlmostEqual(spin[1],0.7522576307332169 , 5)
        self.assertAlmostEqual(spin[2],2.4209847454057316 , 5)
        self.assertAlmostEqual(spin[3],1.1624176868637628 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0120336045, 6)
        self.assertAlmostEqual(e[1], 0.1451768357, 6)
        self.assertAlmostEqual(e[2], 0.2705711303, 6)
        self.assertAlmostEqual(e[3], 0.3014583658, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0022511415415091174 , 6)
        self.assertAlmostEqual(p[2],0.00027815045632255885 , 6)
        self.assertAlmostEqual(p[3],0.016528446067415534 , 6)

        self.assertAlmostEqual(spin[0], 0.7505358324502609 , 5)
        self.assertAlmostEqual(spin[1],0.7504465152068183  , 5)
        self.assertAlmostEqual(spin[2],3.5572050239861834  , 5)
        self.assertAlmostEqual(spin[3],0.8620825777207242  , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0018738819, 6)
        self.assertAlmostEqual(e[1], 0.1573286345, 6)
        self.assertAlmostEqual(e[2], 0.2886390881, 6)
        self.assertAlmostEqual(e[3], 0.3214724068, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0024042299624731397 , 6)
        self.assertAlmostEqual(p[2],9.016719283603531e-05 , 6)
        self.assertAlmostEqual(p[3],0.01624110858261477 , 6)

        self.assertAlmostEqual(spin[0], 0.75005719 , 5)
        self.assertAlmostEqual(spin[1],0.75008973  , 5)
        self.assertAlmostEqual(spin[2],3.67821801  , 5)
        self.assertAlmostEqual(spin[3],0.79204834  , 5)
if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
