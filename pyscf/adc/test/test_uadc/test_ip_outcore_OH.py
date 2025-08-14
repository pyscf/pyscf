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
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
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

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.max_memory = 30
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387906, 6)

        self.assertAlmostEqual(e[0], 0.4342864327917968, 6)
        self.assertAlmostEqual(e[1], 0.47343844767816784, 6)
        self.assertAlmostEqual(e[2], 0.5805631452815511, 6)

        self.assertAlmostEqual(p[0], 0.9066975034860368, 6)
        self.assertAlmostEqual(p[1], 0.8987660491377468, 6)
        self.assertAlmostEqual(p[2], 0.9119655964285802, 6)

    def test_ip_adc2x(self):

        myadc.max_memory = 300
        myadc.incore_complete = False
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4389083582117278, 6)
        self.assertAlmostEqual(e[1], 0.45720829251439343, 6)
        self.assertAlmostEqual(e[2], 0.5588942056812034, 6)

        self.assertAlmostEqual(p[0], 0.9169548953028459, 6)
        self.assertAlmostEqual(p[1], 0.6997121885268642, 6)
        self.assertAlmostEqual(p[2], 0.212879313736106, 6)

    def test_ip_adc3_high_cost(self):

        myadc.max_memory = 300
        myadc.incore_complete = False
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072194, 6)

        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=3)
        myadc.analyze()

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)
        self.assertAlmostEqual(e[1], 0.4872370596653387, 6)
        self.assertAlmostEqual(e[2], 0.5726961805214643, 6)

        self.assertAlmostEqual(p[0], 0.9282869467221032, 6)
        self.assertAlmostEqual(p[1], 0.5188529241094367, 6)
        self.assertAlmostEqual(p[2], 0.40655844616580944, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
