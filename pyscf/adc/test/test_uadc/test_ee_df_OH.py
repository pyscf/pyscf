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
    np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

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

    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc.max_cycle = 200

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0023319460, 6)
        self.assertAlmostEqual(e[1],0.1647722041, 6)
        self.assertAlmostEqual(e[2],0.2984586991, 6)
        self.assertAlmostEqual(e[3],0.3367095228, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00260353, 6)
        self.assertAlmostEqual(p[2],0.00375463, 6)
        self.assertAlmostEqual(p[3],0.01792791, 6)

        self.assertAlmostEqual(spin[0],0.75100157 , 5)
        self.assertAlmostEqual(spin[1],0.75099304 , 5)
        self.assertAlmostEqual(spin[2],2.41368812 , 5)
        self.assertAlmostEqual(spin[3],1.16590822 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0121277301, 6)
        self.assertAlmostEqual(e[1], 0.1450929061, 6)
        self.assertAlmostEqual(e[2], 0.2704365078, 6)
        self.assertAlmostEqual(e[3], 0.3008905139, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00221907  , 6)
        self.assertAlmostEqual(p[2],0.00029972  , 6)
        self.assertAlmostEqual(p[3],0.01678338  , 6)

        self.assertAlmostEqual(spin[0], 0.74929626 , 5)
        self.assertAlmostEqual(spin[1],0.74927343  , 5)
        self.assertAlmostEqual(spin[2],3.55446067  , 5)
        self.assertAlmostEqual(spin[3],0.86133577  , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0019124461, 6)
        self.assertAlmostEqual(e[1], 0.1572857305, 6)
        self.assertAlmostEqual(e[2], 0.2885897977, 6)
        self.assertAlmostEqual(e[3], 0.3209639554, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00240146  , 6)
        self.assertAlmostEqual(p[2],0.00008816  , 6)
        self.assertAlmostEqual(p[3],0.01624065  , 6)

        self.assertAlmostEqual(spin[0], 0.74912395 , 5)
        self.assertAlmostEqual(spin[1],0.74918016  , 5)
        self.assertAlmostEqual(spin[2],3.68420585  , 5)
        self.assertAlmostEqual(spin[3],0.79024543  , 5)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
