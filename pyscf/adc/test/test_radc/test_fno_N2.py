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

def setUpModule():
    global mol, mf
    r = 1.0977
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0., -r/2)],
        ['N', (0., 0.,  r/2)],]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_fno_gs(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0)
        ADCFG.kernel_gs(pct_occ=0.95)

        self.assertAlmostEqual(ADCFG.e_corr_can, -0.3105971132, 6)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0526483858, 6)
        self.assertEqual(len(ADCFG.frozen), 10)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff, mo_energy=ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        _,_,_ = myadc.kernel_gs()
        self.assertAlmostEqual(ADCFG.correct_corr(myadc.e_corr), -0.3004184571, 6)

    def test_ssfno_ip_trans_guess(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ip', ref_state=1)
        ADCFG.trans_guess = True
        ADCFG.kernel(nroots=3, pct_occ=0.95)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff, mo_energy=ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ip'
        myadc.conv_tol = 1e-8
        myadc.tol_residual = 1e-6
        e,v,p,x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.5583378221, 6)
        self.assertAlmostEqual(e[1], 0.6037315973, 6)
        self.assertAlmostEqual(e[2], 0.6039161737, 6)

        self.assertEqual(len(ADCFG.frozen), 10)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0555474689, 6)

    def test_ssfno_ee(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ee', ref_state=1)
        ADCFG.kernel(nroots=5, nvir_act=16)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff, mo_energy=ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ee'
        e,v,p,x = myadc.kernel(nroots=5)

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.34545342, 6)
        self.assertAlmostEqual(e[1], 0.3266533, 6)
        self.assertAlmostEqual(e[2], 0.3772226, 6)

        self.assertEqual(len(ADCFG.frozen), 5)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0198418154, 6)

if __name__ == "__main__":
    print("FNO/SS-FNO calculations for RADC for N2 molecule")
    unittest.main()
