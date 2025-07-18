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
        ['O', (0., 0.    , 0)],
        ['H', (0., -x, y)],
        ['H', (0., x , y)],]
    mol.basis = {'H': 'aug-cc-pVDZ',
                 'O': 'aug-cc-pVDZ',}
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):
        myadc.method = "adc(2)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.22185606131601238, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        myadcea.approx_trans_moments = True
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0287675413010661, 6)
        self.assertAlmostEqual(e[1], 0.0553475511361251, 6)
        self.assertAlmostEqual(e[2], 0.1643553780332306, 6)

        self.assertAlmostEqual(p[0],1.9868096728772893, 6)
        self.assertAlmostEqual(p[1],1.994118278569895 , 6)
        self.assertAlmostEqual(p[2],1.975969169959369 , 6)


    def test_ip_adc2(self):
        myadc.method = "adc(2)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.22185606131601238, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        myadcip.approx_trans_moments = True
        e,v,p,x = myadcip.kernel(nroots=4)
        myadcip.analyze()

        self.assertAlmostEqual(e[0], 0.4133257511, 6)
        self.assertAlmostEqual(e[1], 0.4978545288, 6)
        self.assertAlmostEqual(e[2], 0.6607769739, 6)
        self.assertAlmostEqual(e[3], 1.0540027121, 6)

        self.assertAlmostEqual(p[0], 1.76836736, 6)
        self.assertAlmostEqual(p[1], 1.77319945, 6)
        self.assertAlmostEqual(p[2], 1.80354103, 6)
        self.assertAlmostEqual(p[3], 0.00396362, 6)


if __name__ == "__main__":
    print("Approximate transition moments calculations for different RADC methods for water molecule")
    unittest.main()
