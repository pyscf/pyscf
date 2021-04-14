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
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf import mp

r_CO = 1.21
r_CH = 1.12
theta = 116.5
x = r_CH * math.cos(theta * math.pi/(2 * 180.0))
y = r_CH * math.sin(theta * math.pi/(2 * 180.0))

class KnownValues(unittest.TestCase):
    def test_check_mp2(self):
        mol = gto.Mole()
        mol.atom = [
            ['C', ( 0.0, 0.0, 0.0)],
            ['O', ( 0.0, r_CO , 0.0)],
            ['H', ( 0.0, -x, y)],
            ['H', ( 0.0, -x , -y)],]
        mol.basis = '3-21g'
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        # Ensure phase
        mf.mo_coeff[:,mf.mo_coeff.sum(axis=0) < 0] *= -1

        mp2 = mp.MP2(mf)
        e_mp2 = mp2.kernel()[0]

        myadc = adc.ADC(mf)
        myadc.max_memory = 1
        e_adc_mp2, t_amp1, t_amp2 = myadc.kernel_gs()

        diff_mp2 = e_adc_mp2 - e_mp2

        self.assertAlmostEqual(diff_mp2, 0.0000000000000, 6)

        t_amp1_n = numpy.linalg.norm(t_amp1[0])
        t_amp2_n = numpy.linalg.norm(t_amp2[0])

        self.assertAlmostEqual(t_amp1_n, 0.0430170343900, 6)
        self.assertAlmostEqual(t_amp2_n, 0.2427924473992, 6)

        self.assertAlmostEqual(lib.fp(t_amp1[0]), 0.0196348032865, 6)
        self.assertAlmostEqual(lib.fp(t_amp2[0]), 0.1119036046756, 6)

    def test_check_mp2_high_cost(self):
        mol = gto.Mole()
        mol.atom = [
            ['C', ( 0.0, 0.0, 0.0)],
            ['O', ( 0.0, r_CO , 0.0)],
            ['H', ( 0.0, -x, y)],
            ['H', ( 0.0, -x , -y)],]
        mol.basis = {'H': 'aug-cc-pVQZ',
                     'C': 'aug-cc-pVQZ', 
                     'O': 'aug-cc-pVQZ',}
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        # Ensure phase
        mf.mo_coeff[:,mf.mo_coeff.sum(axis=0) < 0] *= -1

        mp2 = mp.MP2(mf)
        e_mp2 = mp2.kernel()[0]

        myadc = adc.ADC(mf)
        myadc.max_memory = 20
        e_adc_mp2, t_amp1, t_amp2 = myadc.kernel_gs()

        diff_mp2 = e_adc_mp2 - e_mp2

        self.assertAlmostEqual(diff_mp2, 0.0000000000000, 6)

        t_amp1_n = numpy.linalg.norm(t_amp1[0])
        t_amp2_n = numpy.linalg.norm(t_amp2[0])

        self.assertAlmostEqual(t_amp1_n, 0.0456504320024, 6)
        self.assertAlmostEqual(t_amp2_n, 0.2977897530749, 6)

        self.assertAlmostEqual(lib.fp(t_amp1[0]), -0.008983054536, 6)
        self.assertAlmostEqual(lib.fp(t_amp2[0]), -0.639727653133, 6)

if __name__ == "__main__":
    print("Ground state calculations for small memory RADC methods for H2CO molecule")
    unittest.main()
