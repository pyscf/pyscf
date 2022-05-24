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
        ['P', ( 0., 0.    , 0.)],]
    mol.basis = {'P':'aug-cc-pvqz'}
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
        self.assertAlmostEqual(e, -0.14132152692445013, 6)

        myadcea = adc.uadc.UADCEA(myadc) 
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.018742831458017323, 6)
        self.assertAlmostEqual(e[1], -0.018742831458015467, 6)
        self.assertAlmostEqual(e[2], -0.018742831458015204, 6)

        self.assertAlmostEqual(p[0], 0.9443611827330631, 6)
        self.assertAlmostEqual(p[1], 0.9443611827330618, 6)
        self.assertAlmostEqual(p[2], 0.9443611827330606, 6)

    def test_ea_adc2x_high_cost(self):
        myadc.method = "adc(2)-x"
        myadc.kernel_gs()
        myadcea = adc.uadc.UADCEA(myadc) 
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.03555509265158591, 6)
        self.assertAlmostEqual(e[1], -0.035555092651584234, 6)
        self.assertAlmostEqual(e[2], -0.035555092651584234, 6)

        self.assertAlmostEqual(p[0], 0.8654797798217256, 6)
        self.assertAlmostEqual(p[1], 0.8654797798217235, 6)
        self.assertAlmostEqual(p[2], 0.8654797798217245, 6)

    def test_ea_adc3_high_cost(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.15600026295977562, 6)

        myadcea = adc.uadc.UADCEA(myadc) 
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0],-0.023863148551389757, 6)
        self.assertAlmostEqual(e[1],-0.023863148551388515, 6)
        self.assertAlmostEqual(e[2],-0.023863148551387207, 6)

        self.assertAlmostEqual(p[0], 0.8718954487972692, 6)
        self.assertAlmostEqual(p[1], 0.8718954487972691, 6)
        self.assertAlmostEqual(p[2], 0.8718954487972662, 6)
      
if __name__ == "__main__":
    print("EA calculations for different ADC methods for open-shell atom")
    unittest.main()
