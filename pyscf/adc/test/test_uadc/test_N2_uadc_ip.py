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
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.5434389897908212, 6)
        self.assertAlmostEqual(e[1], 0.5434389942222756, 6)
        self.assertAlmostEqual(e[2], 0.6240296265084732, 6)

        self.assertAlmostEqual(p[0], 0.884404855445607, 6)
        self.assertAlmostEqual(p[1], 0.8844048539643351, 6)
        self.assertAlmostEqual(p[2], 0.9096460559671828, 6)

    def test_ip_adc2_oneroot(self):

        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.5434389897908212, 6)

        self.assertAlmostEqual(p[0], 0.884404855445607, 6)

    def test_ip_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.5405255355249104, 6)
        self.assertAlmostEqual(e[1], 0.5405255399061982, 6)
        self.assertAlmostEqual(e[2], 0.62080267098272, 6)

        self.assertAlmostEqual(p[0], 0.875664254715628 , 6)
        self.assertAlmostEqual(p[1], 0.8756642844804134 , 6)
        self.assertAlmostEqual(p[2], 0.9076434703549277, 6)

    def test_ip_adc3_high_cost(self):

        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.31694173142858517 , 6)

        self.assertAlmostEqual(e[0], 0.5667526838174817, 6)
        self.assertAlmostEqual(e[1], 0.5667526888293601, 6)
        self.assertAlmostEqual(e[2], 0.6099995181296374, 6)

        self.assertAlmostEqual(p[0], 0.9086596203469742, 6)
        self.assertAlmostEqual(p[1], 0.9086596190173993, 6)
        self.assertAlmostEqual(p[2], 0.9214613318791076, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods")
    unittest.main()
