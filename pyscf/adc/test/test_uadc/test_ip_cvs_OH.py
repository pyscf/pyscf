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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
    r = 0.969286393
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0.    , -r/2   )],
        ['H', (0., 0.    ,  r/2)],]
    mol.basis = {'O':'aug-cc-pvdz',
                 'H':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.symmetry = False
    mol.spin  = 1
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.ncvs = 1
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 19.94886897889140, 6)
        self.assertAlmostEqual(e[1], 20.00065270524671, 6)
        self.assertAlmostEqual(e[2], 21.15964585685805, 6)

        self.assertAlmostEqual(p[0], 0.77673935197461, 6)
        self.assertAlmostEqual(p[1], 0.77815919145219, 6)
        self.assertAlmostEqual(p[2], 0.00000004371982, 6)

    def test_ip_adc2x(self):

        myadc.ncvs = 1
        myadc.method = "adc(2)-x"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 19.97858926911032, 6)
        self.assertAlmostEqual(e[1], 20.02485253967649, 6)
        self.assertAlmostEqual(e[2], 20.51382057069969, 6)

        self.assertAlmostEqual(p[0], 0.79694705687688, 6)
        self.assertAlmostEqual(p[1], 0.78942190965820, 6)
        self.assertAlmostEqual(p[2], 0.00825541597069, 6)

    def test_ip_adc3(self):

        myadc.ncvs = 1
        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072194, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 20.19831674615190, 6)
        self.assertAlmostEqual(e[1], 20.23682912597306, 6)
        self.assertAlmostEqual(e[2], 20.51828465422586, 6)

        self.assertAlmostEqual(p[0], 0.83426641906575, 6)
        self.assertAlmostEqual(p[1], 0.80657545189575, 6)
        self.assertAlmostEqual(p[2], 0.02836825280158, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
