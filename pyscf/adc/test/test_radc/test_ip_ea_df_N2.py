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
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf import df

def setUpModule():
    global mol, mf, myadc, myadc_fr
    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0.    , -r/2   )],
        ['N', (0., 0.    ,  r/2)],]
    mol.basis = {'N':'cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.kernel()
    myadc = adc.ADC(mf).density_fit(auxbasis='cc-pvdz-ri')
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc_fr = adc.ADC(mf,frozen=1).density_fit(auxbasis='cc-pvdz-ri')
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

class KnownValues(unittest.TestCase):

    def test_df_gs(self):

        myadc.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31081009625, 6)

    def test_dfhf_dfadc_gs(self):

        myadc.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.3108102956, 6)

    def test_dfadc3_ip(self):

        myadc = adc.ADC(mf).density_fit(auxbasis='cc-pvdz-ri')
        myadc.conv_tol = 1e-12
        myadc.tol_residual = 1e-6
        myadc.max_memory = 1
        myadc.method = "adc(3)"
        myadc.method_type = "ip"

        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.3061165912 , 6)

        self.assertAlmostEqual(e[0], 0.55609388, 6)
        self.assertAlmostEqual(e[1], 0.60109239, 6)
        self.assertAlmostEqual(e[2], 0.60109239, 6)

        self.assertAlmostEqual(p[0], 1.83255357, 6)
        self.assertAlmostEqual(p[1], 1.86389642, 6)
        self.assertAlmostEqual(p[2], 1.86389642, 6)

    def test_dfhf_dfadc2_ea(self):

        myadc.max_memory = 1
        myadc.method = "adc(2)"
        myadc.method_type = "ea"

        e,v,p,x = myadc.kernel(nroots=4)
        myadc.analyze()

        self.assertAlmostEqual(e[0], 0.14260766, 6)
        self.assertAlmostEqual(e[1], 0.14260766, 6)
        self.assertAlmostEqual(e[2], 0.55083845, 6)
        self.assertAlmostEqual(e[3], 0.76736577, 6)

        self.assertAlmostEqual(p[0], 1.86603796, 6)
        self.assertAlmostEqual(p[1], 1.86603796, 6)
        self.assertAlmostEqual(p[2], 1.92699634, 6)
        self.assertAlmostEqual(p[3], 1.88366005, 6)

    def test_hf_dfadc2_ea(self):

        mf = scf.RHF(mol).run()
        myadc = adc.ADC(mf).density_fit(auxbasis='cc-pvdz-ri')
        myadc.conv_tol = 1e-12
        myadc.tol_residual = 1e-6
        myadc.max_memory = 1
        myadc.method = "adc(2)"
        myadc.method_type = "ea"

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.14265314, 6)
        self.assertAlmostEqual(e[1], 0.14265314, 6)
        self.assertAlmostEqual(e[2], 0.55092042, 6)
        self.assertAlmostEqual(e[3], 0.76714415, 6)

        self.assertAlmostEqual(p[0], 1.86604908, 6)
        self.assertAlmostEqual(p[1], 1.86604908, 6)
        self.assertAlmostEqual(p[2], 1.92697854, 6)
        self.assertAlmostEqual(p[3], 1.88386011, 6)

    def test_dfadc3_ip_frozen(self):

        myadc_fr.max_memory = 1
        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ip"

        e,v,p,x = myadc_fr.kernel(nroots=3)
        e_corr = myadc_fr.e_corr

        self.assertAlmostEqual(e_corr, -0.3041022652649203 , 6)

        self.assertAlmostEqual(e[0], 0.556071972811554, 6)
        self.assertAlmostEqual(e[1], 0.601035715506966, 6)
        self.assertAlmostEqual(e[2], 0.601035715506968, 6)

        self.assertAlmostEqual(p[0], 1.832409243905315, 6)
        self.assertAlmostEqual(p[1], 1.863816239512485, 6)
        self.assertAlmostEqual(p[2], 1.863816239512485, 6)

if __name__ == "__main__":
    print("DF-ADC calculations for different RADC methods for nitrogen molecule")
    unittest.main()
