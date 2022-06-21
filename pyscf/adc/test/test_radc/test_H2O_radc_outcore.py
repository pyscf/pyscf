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
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
    mol = gto.Mole()
    r = 0.957492
    x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
    y = r * math.cos(104.468205* math.pi/(2 * 180.0))
    mol.atom = [
        ['O', ( 0., 0.    , 0)],
        ['H', ( 0., -x, y)],
        ['H', ( 0., x , y)],]
    mol.basis = {'H': 'aug-cc-pVDZ',
                 'O': 'aug-cc-pVDZ',}
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


    def test_ea_adc2(self):

        myadc.max_memory = 20
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2218560609876961, 6)

        
        myadcea = adc.radc.RADCEA(myadc) 
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0287675413010661, 6)
        self.assertAlmostEqual(e[1], 0.0553475511361251, 6)
        self.assertAlmostEqual(e[2], 0.1643553780332306, 6)

        self.assertAlmostEqual(p[0], 1.9868196915945326, 6)
        self.assertAlmostEqual(p[1], 1.9941128865405613, 6)
        self.assertAlmostEqual(p[2], 1.9760420333383126, 6)


    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2263968409281272, 6)

        myadcea = adc.radc.RADCEA(myadc) 
        e,v,p,x = myadcea.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.0277406670820452, 6)
        self.assertAlmostEqual(e[1], 0.0551456657778995, 6)
        self.assertAlmostEqual(e[2], 0.1620710279026066, 6)
        self.assertAlmostEqual(e[3], 0.1882010099486046, 6)

        self.assertAlmostEqual(p[0], 1.9814233118436899, 6)
        self.assertAlmostEqual(p[1], 1.9920778842193207, 6)
        self.assertAlmostEqual(p[2], 1.9676462978544356, 6)
        self.assertAlmostEqual(p[3], 1.9743650630026532, 6)


    def test_ip_adc3(self):
  
        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        myadc.incore_complete = False
        myadc.max_memory = 10

        e,v,p,x = myadc.kernel(nroots=3)
        myadc.analyze()
      
        self.assertAlmostEqual(e[0], 0.4777266577555, 6)
        self.assertAlmostEqual(e[1], 0.5619000705967, 6)
        self.assertAlmostEqual(e[2], 0.7119986982840, 6)

        self.assertAlmostEqual(p[0], 1.8482158265750, 6)
        self.assertAlmostEqual(p[1], 1.8499506292187, 6)
        self.assertAlmostEqual(p[2], 1.8653509527805, 6)

if __name__ == "__main__":
    print("Out-of-core EA and IP  calculations for different RADC methods for water molecule")
    unittest.main()
