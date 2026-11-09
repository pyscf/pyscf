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
import numpy as np
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc, myadc_fr
    mol = gto.Mole()
    r = 0.957492
    x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
    y = r * math.cos(104.468205* math.pi/(2 * 180.0))
    mol.atom = [
        ['O', (0., 0.    , 0)],
        ['H', (0., -x, y)],
        ['H', (0., x , y)],]
    mol.basis = {'H': 'cc-pVDZ',
                 'O': 'cc-pVDZ',}
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc_fr = adc.ADC(mf,frozen=1)
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.max_memory = 20
        myadc.incore_complete = False
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4034634878946100, 6)
        self.assertAlmostEqual(e[1], 0.4908881395275673, 6)
        self.assertAlmostEqual(e[2], 0.6573303400764507, 6)

        self.assertAlmostEqual(p[0], 1.81594619519855, 6)
        self.assertAlmostEqual(p[1], 1.8271278901200159, 6)
        self.assertAlmostEqual(p[2], 1.8580497651729588, 6)

    def test_ip_adc2x(self):

        myadc.max_memory = 20
        myadc.incore_complete = False
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2039852016968376, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4085610789192171, 6)
        self.assertAlmostEqual(e[1], 0.4949784593692911, 6)
        self.assertAlmostEqual(e[2], 0.6602619900185128, 6)

        self.assertAlmostEqual(p[0], 1.829425572392697, 6)
        self.assertAlmostEqual(p[1], 1.8379855493632293, 6)
        self.assertAlmostEqual(p[2], 1.86681228777205, 6)

    def test_ip_adc3(self):

        myadc.max_memory = 20
        myadc.incore_complete = False
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.2107769014592799, 6)

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4481211042230935, 6)
        self.assertAlmostEqual(e[1], 0.5316292617891758, 6)
        self.assertAlmostEqual(e[2], 0.6850054080600295, 6)
        self.assertAlmostEqual(e[3], 1.1090318744878, 6)

        self.assertAlmostEqual(p[0], 1.8683046103300471, 6)
        self.assertAlmostEqual(p[1], 1.8720323166638342, 6)
        self.assertAlmostEqual(p[2], 1.8882246920860226, 6)
        self.assertAlmostEqual(p[3], 0.16511568362076579, 6)

    def test_ip_adc3_frozen(self):

        myadc_fr.max_memory = 20
        myadc_fr.incore_complete = False
        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.20864693991051744, 6)

        e,v,p,x = myadc_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4480066842416781, 6)
        self.assertAlmostEqual(e[1], 0.5315377164038694, 6)
        self.assertAlmostEqual(e[2], 0.6849075186702855, 6)
        self.assertAlmostEqual(e[3], 1.1090980093898242, 6)

        self.assertAlmostEqual(p[0], 1.868284007940769, 6)
        self.assertAlmostEqual(p[1], 1.8720001297929698, 6)
        self.assertAlmostEqual(p[2], 1.8881913246364712, 6)
        self.assertAlmostEqual(p[3], 0.16536157361871964, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for water molecule")
    unittest.main()
