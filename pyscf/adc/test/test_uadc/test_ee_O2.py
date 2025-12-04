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
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc

    basis = 'cc-pVDZ'
    r = 1.215774

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['O', ( 0., 0.    , -r/2)],
        ['O', ( 0., 0., r/2)],]
    mol.basis = {'O': basis,}

    mol.spin = 2
    mol.symmetry = True
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

        self.assertAlmostEqual(e[0],0.2442040058, 6)
        self.assertAlmostEqual(e[1],0.2442040058, 6)
        self.assertAlmostEqual(e[2],0.2500895487, 6)
        self.assertAlmostEqual(e[3],0.3518735948, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.18215230, 6)

        self.assertAlmostEqual(spin[0],1.99267715 , 5)
        self.assertAlmostEqual(spin[1],1.99267715 , 5)
        self.assertAlmostEqual(spin[2],1.99361354 , 5)
        self.assertAlmostEqual(spin[3],2.00275098 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2287763003, 6)
        self.assertAlmostEqual(e[1],0.2287763003, 6)
        self.assertAlmostEqual(e[2],0.2335850406, 6)
        self.assertAlmostEqual(e[3],0.3372202072, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.16786657, 6)

        self.assertAlmostEqual(spin[0],1.98845818 , 5)
        self.assertAlmostEqual(spin[1],1.98845818 , 5)
        self.assertAlmostEqual(spin[2],1.98889755 , 5)
        self.assertAlmostEqual(spin[3],1.99565572 , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2109693053, 6)
        self.assertAlmostEqual(e[1],0.2109693053, 6)
        self.assertAlmostEqual(e[2],0.2160659670, 6)
        self.assertAlmostEqual(e[3],0.3206475370, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.16879764, 6)

        self.assertAlmostEqual(spin[0],1.99154478 , 5)
        self.assertAlmostEqual(spin[1],1.99154478 , 5)
        self.assertAlmostEqual(spin[2],1.99181334 , 5)
        self.assertAlmostEqual(spin[3],1.99737928 , 5)
if __name__ == "__main__":
    print("EE calculations for different ADC methods for O2 molecule")
    unittest.main()
