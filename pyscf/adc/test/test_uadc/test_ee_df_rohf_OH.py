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

    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0016936152, 6)
        self.assertAlmostEqual(e[1],0.1629225558, 6)
        self.assertAlmostEqual(e[2],0.2989590171, 6)
        self.assertAlmostEqual(e[3],0.3364100601, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00255275, 6)
        self.assertAlmostEqual(p[2],0.00360777, 6)
        self.assertAlmostEqual(p[3],0.01805734, 6)

        self.assertAlmostEqual(spin[0],0.75053470 , 5)
        self.assertAlmostEqual(spin[1],0.75064937 , 5)
        self.assertAlmostEqual(spin[2],2.43044095 , 5)
        self.assertAlmostEqual(spin[3],1.15231877 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0127433223, 6)
        self.assertAlmostEqual(e[1], 0.1437039204, 6)
        self.assertAlmostEqual(e[2], 0.2675152327, 6)
        self.assertAlmostEqual(e[3], 0.2986007925, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00219515  , 6)
        self.assertAlmostEqual(p[2],0.00012411  , 6)
        self.assertAlmostEqual(p[3],0.01590708  , 6)

        self.assertAlmostEqual(spin[0], 0.75043116 , 5)
        self.assertAlmostEqual(spin[1],0.75037374  , 5)
        self.assertAlmostEqual(spin[2],3.65768347  , 5)
        self.assertAlmostEqual(spin[3],0.80434287  , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0019861472, 6)
        self.assertAlmostEqual(e[1], 0.1567423657, 6)
        self.assertAlmostEqual(e[2], 0.2837047778, 6)
        self.assertAlmostEqual(e[3], 0.3182824901, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00239405  , 6)
        self.assertAlmostEqual(p[2],0.00000015  , 6)
        self.assertAlmostEqual(p[3],0.01492925  , 6)

        self.assertAlmostEqual(spin[0], 0.75027498 , 5)
        self.assertAlmostEqual(spin[1],0.75009197  , 5)
        self.assertAlmostEqual(spin[2],3.72597142  , 5)
        self.assertAlmostEqual(spin[3],0.75913888  , 5)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
