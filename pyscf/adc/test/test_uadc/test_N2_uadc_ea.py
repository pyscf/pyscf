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

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.32201692499346535, 6)

        self.assertAlmostEqual(e[0], 0.09617819142992463, 6)
        self.assertAlmostEqual(e[1], 0.09617819161216855, 6)
        self.assertAlmostEqual(e[2], 0.1258326904883586, 6)

        self.assertAlmostEqual(p[0], 0.9916427196092643, 6)
        self.assertAlmostEqual(p[1], 0.9916427196903126, 6)
        self.assertAlmostEqual(p[2], 0.9817184085436222, 6)

    def test_ea_adc2_oneroot(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.09617819142992463, 6)

        self.assertAlmostEqual(p[0], 0.9916427196092643, 6)

    def test_ea_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.0953065329249756, 6)
        self.assertAlmostEqual(e[1], 0.09530653311160658, 6)
        self.assertAlmostEqual(e[2], 0.12388330778444741, 6)
        self.assertAlmostEqual(e[3], 0.1238833087377404, 6)

        self.assertAlmostEqual(p[0], 0.9890885390419444 , 6)
        self.assertAlmostEqual(p[1],0.9890885391436558 , 6)
        self.assertAlmostEqual(p[2],0.9757598335805556 , 6)
        self.assertAlmostEqual(p[3],0.9757598335315953 , 6)

    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31694173142858517 , 6)

        myadc.method_type = "ea"
        e,v,p,x,adc_es = myadc.ea_adc(nroots=3)
        myadc.analyze()

        self.assertAlmostEqual(e[0], 0.09836545519294707, 6)
        self.assertAlmostEqual(e[1], 0.09836545535648182, 6)
        self.assertAlmostEqual(e[2], 0.12957093060937017, 6)

        self.assertAlmostEqual(p[0], 0.9920495595411523, 6)
        self.assertAlmostEqual(p[1], 0.9920495596160825, 6)
        self.assertAlmostEqual(p[2], 0.9819275025204279, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods")
    unittest.main()
