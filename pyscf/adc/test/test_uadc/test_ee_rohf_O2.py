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
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2536868317, 6)
        self.assertAlmostEqual(e[1],0.2536868317, 6)
        self.assertAlmostEqual(e[2],0.2594502231, 6)
        self.assertAlmostEqual(e[3],0.3621178740, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.18014923047124629, 6)

        self.assertAlmostEqual(spin[0],2.0005196840671475 , 5)
        self.assertAlmostEqual(spin[1],2.0005196828075142 , 5)
        self.assertAlmostEqual(spin[2],2.0006607644293712 , 5)
        self.assertAlmostEqual(spin[3],2.006848302414686 , 5)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2388257320, 6)
        self.assertAlmostEqual(e[1],0.2388257320, 6)
        self.assertAlmostEqual(e[2],0.2434547828, 6)
        self.assertAlmostEqual(e[3],0.3475847540, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.16609689429286933, 6)

        self.assertAlmostEqual(spin[0],2.0005030460576725 , 5)
        self.assertAlmostEqual(spin[1],2.0005030431813227 , 5)
        self.assertAlmostEqual(spin[2],2.0006908765245255 , 5)
        self.assertAlmostEqual(spin[3],2.00393731 , 5)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.2073282139, 6)
        self.assertAlmostEqual(e[1],0.2073282139, 6)
        self.assertAlmostEqual(e[2],0.2123139019, 6)
        self.assertAlmostEqual(e[3],0.3155432845, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.16727580653586102, 6)

        self.assertAlmostEqual(spin[0],1.99956853 , 5)
        self.assertAlmostEqual(spin[1],1.99956853 , 5)
        self.assertAlmostEqual(spin[2],1.99960415 , 5)
        self.assertAlmostEqual(spin[3],2.00292510 , 5)
if __name__ == "__main__":
    print("EE calculations for different ADC methods for O2 molecule")
    unittest.main()
