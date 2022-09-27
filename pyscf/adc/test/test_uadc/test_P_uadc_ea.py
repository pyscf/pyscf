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
    mol = gto.Mole()
    mol.atom = [
        ['P', (0., 0.    , 0.)],]
    mol.basis = {'P':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.spin = 3
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.irrep_nelec = {'A1g':(3,3),'E1ux':(2,1),'E1uy':(2,1),'A1u':(2,1)}
    mf.kernel()
    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.07351147825008204, 6)

        myadcea = adc.uadc_ea.UADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.00570584313941, 6)
        self.assertAlmostEqual(e[1], -0.00570584313941, 6)
        self.assertAlmostEqual(e[2], -0.00570584313941, 6)

        self.assertAlmostEqual(p[0], 0.95200973206193, 6)
        self.assertAlmostEqual(p[1], 0.95200973206193, 6)
        self.assertAlmostEqual(p[2], 0.95200973206193, 6)

    def test_ea_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.07351147825008204, 6)

        myadcea = adc.uadc_ea.UADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.02422205199715, 6)
        self.assertAlmostEqual(e[1], -0.02422205199715, 6)
        self.assertAlmostEqual(e[2], -0.02422205199715, 6)

        self.assertAlmostEqual(p[0], 0.86563811906540, 6)
        self.assertAlmostEqual(p[1], 0.86563811906540, 6)
        self.assertAlmostEqual(p[2], 0.86563811906541, 6)

    def test_ea_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.08920409983024569, 6)

        myadcea = adc.uadc_ea.UADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.01331220104400, 6)
        self.assertAlmostEqual(e[1], -0.01331220104400, 6)
        self.assertAlmostEqual(e[2], -0.01331220104400, 6)

        self.assertAlmostEqual(p[0], 0.87295109889981, 6)
        self.assertAlmostEqual(p[1], 0.87295109889981, 6)
        self.assertAlmostEqual(p[2], 0.87295109889981, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods for open-shell atom")
    unittest.main()
