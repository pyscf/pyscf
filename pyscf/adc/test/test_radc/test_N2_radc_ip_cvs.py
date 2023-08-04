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
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0.    , -r/2   )],
        ['N', (0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.ncvs = 2
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        e,v,p,x = myadc.kernel(nroots=2)

        self.assertAlmostEqual(e[0], 15.12281031796864, 6)
        self.assertAlmostEqual(e[1], 15.12611217935994, 6)


        self.assertAlmostEqual(p[0], 1.54262807973040, 6)
        self.assertAlmostEqual(p[1], 1.54152768244107, 6)


    def test_ip_adc2x(self):

        myadc.ncvs = 2
        myadc.method = "adc(2)-x"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 15.10850837488190, 6)
        self.assertAlmostEqual(e[1], 15.11180785851825, 6)
        self.assertAlmostEqual(e[2], 15.74525610353200, 6)

        self.assertAlmostEqual(p[0], 1.51596080565362, 6)
        self.assertAlmostEqual(p[1], 1.51447639333099, 6)
        self.assertAlmostEqual(p[2], 0.00000030441510, 6)

    def test_ip_adc3(self):

        myadc.ncvs = 2
        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31694173142858517 , 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 15.28014264694833, 6)
        self.assertAlmostEqual(e[1], 15.28358689153576, 6)
        self.assertAlmostEqual(e[2], 15.74525610336261, 6)

        self.assertAlmostEqual(p[0], 1.64260781902935, 6)
        self.assertAlmostEqual(p[1], 1.64123055314380, 6)
        self.assertAlmostEqual(p[2], 0.00000030441505, 6)

if __name__ == "__main__":
    print("IP calculations for different RADC methods for nitrogen molecule")
    unittest.main()
