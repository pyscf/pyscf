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
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ip import get_spin_square

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

    def test_ip_adc2(self):

        myadc.max_memory = 30
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387906, 6)

        self.assertAlmostEqual(e[0], 0.4342864327917968, 6)
        self.assertAlmostEqual(e[1], 0.47343844767816784, 6)
        self.assertAlmostEqual(e[2], 0.5805631452815511, 6)

        self.assertAlmostEqual(p[0], 0.9058488149525856, 6)
        self.assertAlmostEqual(p[1], 0.8979216632613445, 6)
        self.assertAlmostEqual(p[2], 0.9115851423147069, 6)

        self.assertAlmostEqual(spin[0],2.001826471842403 , 5)
        self.assertAlmostEqual(spin[1],1.0850551931110326 , 5)
        self.assertAlmostEqual(spin[2],2.001630820669052 , 5)

    def test_ip_adc2x(self):

        myadc.max_memory = 300
        myadc.incore_complete = False
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0], 0.4389083582117278, 6)
        self.assertAlmostEqual(e[1], 0.45720829251439343, 6)
        self.assertAlmostEqual(e[2], 0.5588942056812034, 6)

        self.assertAlmostEqual(p[0], 0.9161237660123824, 6)
        self.assertAlmostEqual(p[1], 0.6988041757561676, 6)
        self.assertAlmostEqual(p[2], 0.2128612620680331, 6)

        self.assertAlmostEqual(spin[0],2.001538757123655 , 5)
        self.assertAlmostEqual(spin[1],1.8256843856500033 , 5)
        self.assertAlmostEqual(spin[2],0.19585453729020275 , 5)

    def test_ip_adc3_high_cost(self):

        myadc.max_memory = 300
        myadc.incore_complete = False
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072194, 6)

        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]
        myadc.analyze()

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)
        self.assertAlmostEqual(e[1], 0.4872370596653387, 6)
        self.assertAlmostEqual(e[2], 0.5726961805214643, 6)

        self.assertAlmostEqual(p[0], 0.9285086275683777, 6)
        self.assertAlmostEqual(p[1], 0.518972490173223, 6)
        self.assertAlmostEqual(p[2], 0.40659866949569623, 6)

        self.assertAlmostEqual(spin[0],2.00025760 , 5)
        self.assertAlmostEqual(spin[1],1.96962920 , 5)
        self.assertAlmostEqual(spin[2],0.05053658 , 5)
    def test_ip_adc3_high_cost_frozen(self):

        myadc_fr.max_memory = 300
        myadc_fr.incore_complete = False
        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.174168907841914, 6)

        myadc_fr.method_type = "ip"
        e,v,p,x = myadc_fr.kernel(nroots=3)
        myadc_fr.analyze()

        self.assertAlmostEqual(e[0], 0.4793358282930479, 6)
        self.assertAlmostEqual(e[1], 0.4872390877594536, 6)
        self.assertAlmostEqual(e[2], 0.5726733430259214, 6)

        self.assertAlmostEqual(p[0], 0.9285016651480348, 6)
        self.assertAlmostEqual(p[1], 0.5201406212527856, 6)
        self.assertAlmostEqual(p[2], 0.4054240789596952, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
