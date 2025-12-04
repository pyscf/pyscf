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
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc, myadc_fr
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
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc_fr = adc.ADC(mf,frozen=(1,1))
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):

        myadc.max_memory = 30
        myadc.incore_complete = False
        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387806, 6)

        self.assertAlmostEqual(e[0], -0.048666915263496924, 6)
        self.assertAlmostEqual(e[1], 0.030845983085818485, 6)
        self.assertAlmostEqual(e[2], 0.03253522816723711, 6)

        self.assertAlmostEqual(p[0], 0.9228959646746451, 6)
        self.assertAlmostEqual(p[1], 0.9953781149964537, 6)
        self.assertAlmostEqual(p[2], 0.9956169835481459, 6)

    def test_ea_adc2x(self):

        myadc.max_memory = 300
        myadc.incore_complete = False
        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.07750642898162931, 6)
        self.assertAlmostEqual(e[1], 0.029292010466571882, 6)
        self.assertAlmostEqual(e[2], 0.030814773752482663, 6)

        self.assertAlmostEqual(p[0], 0.8323987058794676, 6)
        self.assertAlmostEqual(p[1], 0.9918705979602267, 6)
        self.assertAlmostEqual(p[2], 0.9772855298541363, 6)

    def test_ea_adc3_high_cost(self):

        myadc.max_memory = 300
        myadc.incore_complete = False
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072136, 6)

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        myadc.analyze()

        self.assertAlmostEqual(e[0], -0.045097652872531736, 6)
        self.assertAlmostEqual(e[1], 0.03004291636971322, 6)
        self.assertAlmostEqual(e[2], 0.03153897437644345, 6)

        self.assertAlmostEqual(p[0], 0.8722483551941809, 6)
        self.assertAlmostEqual(p[1], 0.9927117650068699, 6)
        self.assertAlmostEqual(p[2], 0.9766456031927034, 6)

    def test_ea_adc3_high_cost_frozen(self):

        myadc_fr.max_memory = 300
        myadc_fr.incore_complete = False
        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.17416890784191413, 6)

        myadc_fr.method_type = "ea"
        e,v,p,x = myadc_fr.kernel(nroots=3)
        myadc_fr.analyze()

        self.assertAlmostEqual(e[0], -0.0449624364309033, 6)
        self.assertAlmostEqual(e[1],  0.0300414467534955, 6)
        self.assertAlmostEqual(e[2],  0.03153991727098654, 6)

        self.assertAlmostEqual(p[0], 0.8722200055138165, 6)
        self.assertAlmostEqual(p[1], 0.9927113317116674, 6)
        self.assertAlmostEqual(p[2], 0.9767596218115033, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods for open-shell molecule")
    unittest.main()
