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
# Author: Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ee import get_spin_square as get_spin_square_ee

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0., 0.)],
        ['H', (0., 0., 0.9697)],]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.spin = 1
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_fno_gs(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0)
        ADCFG.kernel_gs(pct_occ=0.95)

        self.assertAlmostEqual(ADCFG.e_corr_can, -0.1509990493, 6)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0233741322, 6)
        self.assertEqual(len(ADCFG.frozen[0]), 6)
        self.assertEqual(len(ADCFG.frozen[1]), 7)

    def test_osfno_gs(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0)
        ADCFG.if_osfno = True
        ADCFG.kernel_gs(pct_occ=0.95)

        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0240946057, 6)
        self.assertEqual(len(ADCFG.frozen[0]), 6)
        self.assertEqual(len(ADCFG.frozen[1]), 6)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        self.assertEqual(myadc.nvir_b - myadc.nvir_a, 1)

    def test_osfno_ea(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ea')
        ADCFG.if_osfno = True
        ADCFG.kernel(nroots=3, pct_occ=0.90)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ea'
        e,v,p,x = myadc.kernel(nroots=3)

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.03549859, 6)
        self.assertAlmostEqual(e[1], 0.16628554, 6)
        self.assertAlmostEqual(e[2], 0.18655859, 6)


        self.assertEqual(len(ADCFG.frozen[0]), 7)
        self.assertEqual(len(ADCFG.frozen[1]), 7)
        self.assertEqual(myadc.nvir_b - myadc.nvir_a, 1)

    def test_ssfno_ee(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ee', ref_state=1)
        ADCFG.kernel(nroots=3, thresh=1e-3)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ee'
        myadc.conv_tol = 1e-8
        myadc.tol_residual = 1e-6
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square_ee(myadc._adc_es)[0]

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], -0.0017148008, 6)
        self.assertAlmostEqual(e[1],  0.1578325448, 6)
        self.assertAlmostEqual(e[2],  0.2538784631, 6)

        self.assertAlmostEqual(spin[0], 0.74945130 , 4)
        self.assertAlmostEqual(spin[1], 0.74941619 , 4)
        self.assertAlmostEqual(spin[2], 3.69806781 , 4)

        self.assertEqual(len(ADCFG.frozen[0]), 5)
        self.assertEqual(len(ADCFG.frozen[1]), 5)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0095566166, 6)

    def test_ssfno_ip_trans_guess(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ip', ref_state=1)
        ADCFG.trans_guess = True
        ADCFG.kernel(nroots=3, pct_occ=0.90)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ip'
        myadc.conv_tol = 1e-8
        myadc.tol_residual = 1e-6
        e,v,p,x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.4583022367, 6)
        self.assertAlmostEqual(e[1], 0.4758405834, 6)
        self.assertAlmostEqual(e[2], 0.5852997495, 6)

        self.assertAlmostEqual(p[0], 0.946861, 6)
        self.assertAlmostEqual(p[1], 0.727942, 6)
        self.assertAlmostEqual(p[2], 0.219144, 6)


        self.assertEqual(len(ADCFG.frozen[0]), 8)
        self.assertEqual(len(ADCFG.frozen[1]), 10)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0564949471, 6)

    def test_ssfno_ea(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ea', ref_state=1)
        ADCFG.kernel(nroots=3, thresh=1e-3)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ea'
        e,v,p,x = myadc.kernel(nroots=3)

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.0377227177, 6)
        self.assertAlmostEqual(e[1], 0.174180062, 6)
        self.assertAlmostEqual(e[2], 0.1767330484, 6)


        self.assertEqual(len(ADCFG.frozen[0]), 5)
        self.assertEqual(len(ADCFG.frozen[1]), 5)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0103985555, 6)

if __name__ == "__main__":
    print("FNO/OSFNO calculations for UADC for open-shell OH molecule")
    unittest.main()
