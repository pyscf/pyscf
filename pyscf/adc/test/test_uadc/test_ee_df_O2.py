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

        self.assertAlmostEqual(e[0],0.2443043831, 6)
        self.assertAlmostEqual(e[1],0.2443043831, 6)
        self.assertAlmostEqual(e[2],0.2503033466, 6)
        self.assertAlmostEqual(e[3],0.3518290857, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.18285732263373, 6)

        self.assertAlmostEqual(spin[0],2.0069058940056594 , 5)
        self.assertAlmostEqual(spin[1],2.0069058920633562 , 5)
        self.assertAlmostEqual(spin[2],2.0071744818590807 , 5)
        self.assertAlmostEqual(spin[3],2.0149233315499453 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2287827140, 6)
        self.assertAlmostEqual(e[1],0.2287827140, 6)
        self.assertAlmostEqual(e[2],0.2336997839, 6)
        self.assertAlmostEqual(e[3],0.3371119024, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000 , 6)
        self.assertAlmostEqual(p[2],0.00000000 , 6)
        self.assertAlmostEqual(p[3],0.1682086735092642 , 6)

        self.assertAlmostEqual(spin[0],2.002479452493251 , 5)
        self.assertAlmostEqual(spin[1],2.0024794545476725 , 5)
        self.assertAlmostEqual(spin[2],2.002244325022952 , 5)
        self.assertAlmostEqual(spin[3],2.007659488143016 , 5)

    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4, guess = "cis")
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2287827140, 6)
        self.assertAlmostEqual(e[1],0.2287827140, 6)
        self.assertAlmostEqual(e[2],0.2336997839, 6)
        self.assertAlmostEqual(e[3],0.3371119023, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000 , 6)
        self.assertAlmostEqual(p[2],0.00000000 , 6)
        self.assertAlmostEqual(p[3],0.16820800678287948 , 6)

        self.assertAlmostEqual(spin[0],2.002479444545658 , 5)
        self.assertAlmostEqual(spin[1],2.002479452648217 , 5)
        self.assertAlmostEqual(spin[2],2.0022443206530305 , 5)
        self.assertAlmostEqual(spin[3],2.0076593393088524 , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2110128931, 6)
        self.assertAlmostEqual(e[1],0.2110128931, 6)
        self.assertAlmostEqual(e[2],0.2162214015, 6)
        self.assertAlmostEqual(e[3],0.3205866681, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000 , 6)
        self.assertAlmostEqual(p[2],0.00000000 , 6)
        self.assertAlmostEqual(p[3],0.16860063021225857 , 6)

        self.assertAlmostEqual(spin[0],1.99868774 , 5)
        self.assertAlmostEqual(spin[1],1.99868774 , 5)
        self.assertAlmostEqual(spin[2],1.99894334 , 5)
        self.assertAlmostEqual(spin[3],2.00232015 , 5)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for O2 molecule")
    unittest.main()
