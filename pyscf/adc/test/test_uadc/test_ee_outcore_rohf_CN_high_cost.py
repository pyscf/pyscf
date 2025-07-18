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
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc
    basis = 'cc-pVDZ'
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
        C 0.00000000 0.00000000 -1.18953886
        N 0.00000000 0.00000000 1.01938091
         '''
    mol.basis = {'C': basis,
                 'N': basis,}
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.build()

    mf = scf.ROHF(mol)
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
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0540102949, 6)
        self.assertAlmostEqual(e[1],0.0540102949, 6)
        self.assertAlmostEqual(e[2],0.0955821322, 6)
        self.assertAlmostEqual(e[3],0.2557350668, 6)

        self.assertAlmostEqual(p[0],0.00330109, 6)
        self.assertAlmostEqual(p[1],0.00330109, 6)
        self.assertAlmostEqual(p[2],0.03417973, 6)
        self.assertAlmostEqual(p[3],0.00247641, 6)

        self.assertAlmostEqual(spin[0],0.75455998 , 5)
        self.assertAlmostEqual(spin[1],0.75455998 , 5)
        self.assertAlmostEqual(spin[2],0.76652057 , 5)
        self.assertAlmostEqual(spin[3],2.83811703 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0164020499, 6)
        self.assertAlmostEqual(e[1],0.0164020499, 6)
        self.assertAlmostEqual(e[2],0.0530368555, 6)
        self.assertAlmostEqual(e[3],0.1763402895, 6)

        self.assertAlmostEqual(p[0],0.00089364, 6)
        self.assertAlmostEqual(p[1],0.00089364, 6)
        self.assertAlmostEqual(p[2],0.01674309, 6)
        self.assertAlmostEqual(p[3],0.00080394, 6)

        self.assertAlmostEqual(spin[0],0.75474307 , 5)
        self.assertAlmostEqual(spin[1],0.75474307 , 5)
        self.assertAlmostEqual(spin[2],0.76071932 , 5)
        self.assertAlmostEqual(spin[3],3.30672580 , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362916619, 6)
        self.assertAlmostEqual(e[1],0.0362916619, 6)
        self.assertAlmostEqual(e[2],0.1200404011, 6)
        self.assertAlmostEqual(e[3],0.1747467421, 6)

        self.assertAlmostEqual(p[0],0.00215562, 6)
        self.assertAlmostEqual(p[1],0.00215562, 6)
        self.assertAlmostEqual(p[2],0.02070979, 6)
        self.assertAlmostEqual(p[3],0.00142281, 6)

        self.assertAlmostEqual(spin[0],0.75778870 , 5)
        self.assertAlmostEqual(spin[1],0.75778870 , 5)
        self.assertAlmostEqual(spin[2],0.79970595 , 5)
        self.assertAlmostEqual(spin[3],3.45250153 , 5)
if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
