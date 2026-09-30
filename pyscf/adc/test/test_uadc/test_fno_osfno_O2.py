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
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf
    r = 1.2074
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0., -r/2)],
        ['O', (0., 0.,  r/2)],]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.spin = 2
    mol.build()
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_osfno_gs(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0)
        ADCFG.if_osfno = True
        ADCFG.kernel_gs(pct_occ=0.90)

        self.assertAlmostEqual(ADCFG.e_corr_can, -0.3829098026, 6)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0837895438, 6)
        self.assertEqual(len(ADCFG.frozen[0]), 8)
        self.assertEqual(len(ADCFG.frozen[1]), 8)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy, ADCFG.f_ov)
        self.assertEqual(myadc.nvir_b - myadc.nvir_a, 2)

        S = mf.get_ovlp()
        ova = myadc.mo_coeff[0].T.dot(S).dot(myadc.mo_coeff[0])
        ovb = myadc.mo_coeff[1].T.dot(S).dot(myadc.mo_coeff[1])
        self.assertAlmostEqual(np.abs(ova - np.eye(ova.shape[0])).max(), 0, 9)
        self.assertAlmostEqual(np.abs(ovb - np.eye(ovb.shape[0])).max(), 0, 9)

    def test_osfno_ee(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ee')
        ADCFG.if_osfno = True
        ADCFG.kernel(nroots=4, pct_occ=0.90)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy, ADCFG.f_ov)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ee'
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.21659673, 6)
        self.assertAlmostEqual(e[1], 0.21626663, 6)
        self.assertAlmostEqual(e[2], 0.2218296, 6)
        self.assertAlmostEqual(e[3], 0.32334699, 6)

        self.assertAlmostEqual(spin[0], 2.00097047, 3)
        self.assertAlmostEqual(spin[1], 2.00099478, 3)
        self.assertAlmostEqual(spin[2], 2.00094714, 3)
        self.assertAlmostEqual(spin[3], 2.00243634, 3)

    def test_fno_vs_osfno_spin_contamination(self):
        adc_can = adc.UADC(mf)
        adc_can.verbose = 0
        adc_can.method = 'adc(3)'
        adc_can.method_type = 'ee'
        adc_can.kernel(nroots=4)
        s2_can = get_spin_square(adc_can._adc_es)[0]

        for if_osfno in (False, True):
            ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ee')
            ADCFG.if_osfno = if_osfno
            ADCFG.kernel(nroots=4, pct_occ=0.90)
            myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                             ADCFG.mo_energy, ADCFG.f_ov)
            myadc.verbose = 0
            myadc.method = 'adc(3)'
            myadc.method_type = 'ee'
            myadc.kernel(nroots=4)
            spin = get_spin_square(myadc._adc_es)[0]
            dS2 = np.abs(spin - s2_can).max()
            if if_osfno:
                self.assertLess(dS2, 2e-3)
            else:
                self.assertGreater(dS2, 5e-2)

if __name__ == "__main__":
    print("OSFNO calculations for UADC for open-shell O2 molecule")
    unittest.main()
