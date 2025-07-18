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
from pyscf import df

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
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        myadcipcvs = adc.uadc_ip_cvs.UADCIPCVS(myadc)
        myadcipcvs.ncvs = 2
        e,v,p,x = myadcipcvs.kernel(nroots=2)

        self.assertAlmostEqual(e[0], 15.12281031547323, 6)
        self.assertAlmostEqual(e[1], 15.12281031548646, 6)

        self.assertAlmostEqual(p[0], 0.77131403962843, 6)
        self.assertAlmostEqual(p[1], 0.77131403962002, 6)

    def test_ip_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        myadcipcvs = adc.uadc_ip_cvs.UADCIPCVS(myadc)
        myadcipcvs.ncvs = 2
        e,v,p,x = myadcipcvs.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 15.10850837418996, 6)
        self.assertAlmostEqual(e[1], 15.10850837420428, 6)
        self.assertAlmostEqual(e[2], 15.11180785804999, 6)
        self.assertAlmostEqual(e[3], 15.11180785806416, 6)

        self.assertAlmostEqual(p[0], 0.75798034434940, 6)
        self.assertAlmostEqual(p[1], 0.75798040042682, 6)
        self.assertAlmostEqual(p[2], 0.75723819837873, 6)
        self.assertAlmostEqual(p[3], 0.75723819850822, 6)

    def test_ip_adc3_high_cost(self):

        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31694173142858517 , 6)

        myadcipcvs = adc.uadc_ip_cvs.UADCIPCVS(myadc)
        myadcipcvs.ncvs = 2
        e,v,p,x = myadcipcvs.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 15.28014264411697, 6)
        self.assertAlmostEqual(e[1], 15.28014264412627, 6)
        self.assertAlmostEqual(e[2], 15.28358688862723, 6)
        self.assertAlmostEqual(e[3], 15.28358688863590, 6)

        self.assertAlmostEqual(p[0], 0.82130398832535, 6)
        self.assertAlmostEqual(p[1], 0.82130396559585, 6)
        self.assertAlmostEqual(p[2], 0.82061528343281, 6)
        self.assertAlmostEqual(p[3], 0.82061528704539, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
