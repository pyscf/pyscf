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

    def test_ip_adc2(self):
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.1413215269244437, 6)

        myadcip = adc.uadc.UADCIP(myadc) 
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.38713687643570405, 6)
        self.assertAlmostEqual(e[1], 0.3871368764357055, 6)
        self.assertAlmostEqual(e[2], 0.3871368764357069, 6)

        self.assertAlmostEqual(p[0], 0.9442888630617206, 6)
        self.assertAlmostEqual(p[1], 0.9442888630617206, 6)
        self.assertAlmostEqual(p[2], 0.9442888630617201, 6)

    def test_ip_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.1413215269244437, 6)

        myadcip = adc.uadc.UADCIP(myadc) 
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.3751918934459225, 6)
        self.assertAlmostEqual(e[1], 0.375191893445926,  6)
        self.assertAlmostEqual(e[2], 0.37519189344592646, 6)

        self.assertAlmostEqual(p[0], 0.9197702110961208,  6)
        self.assertAlmostEqual(p[1], 0.9197702110961219,  6)
        self.assertAlmostEqual(p[2], 0.9197702110961211,  6)

    def test_ip_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.15600026295970712, 6)

        myadcip = adc.uadc.UADCIP(myadc) 
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.3839911922956251, 6)
        self.assertAlmostEqual(e[1], 0.3839911922956276, 6)
        self.assertAlmostEqual(e[2], 0.3839911922956283, 6)

        self.assertAlmostEqual(p[0], 0.9232360195076491, 6)
        self.assertAlmostEqual(p[1], 0.9232360195076498, 6)
        self.assertAlmostEqual(p[2], 0.9232360195076497, 6)
      
if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell atom")
    unittest.main()
