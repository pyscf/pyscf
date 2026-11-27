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

def setUpModule():
    global mol, mf, myadc, myadc_fr
    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0.    , -r/2   )],
        ['N', (0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
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

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):

        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.365895203706856, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.0961781923822576, 6)
        self.assertAlmostEqual(e[1], 0.1258326916409743, 6)
        self.assertAlmostEqual(e[2], 0.1380779405750178, 6)

        self.assertAlmostEqual(p[0], 1.983311406981808, 6)
        self.assertAlmostEqual(p[1], 1.9634308545029766, 6)
        self.assertAlmostEqual(p[2], 1.9779449892968433, 6)

        dm1_exc = myadcea.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 83.7154167633201, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 67.125000849147, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 66.2496820621055, 6)

    def test_ea_adc2_oneroot(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.0961781923822576, 6)

        self.assertAlmostEqual(p[0], 1.9833114052504708, 6)

    def test_ea_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"

        myadc.kernel_gs()
        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.365895203706856, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=4)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.32201692499346535, 6)

        self.assertAlmostEqual(e[0], 0.0953065329895602, 6)
        self.assertAlmostEqual(e[1], 0.1238833071439568, 6)
        self.assertAlmostEqual(e[2], 0.1365693813556231, 6)
        self.assertAlmostEqual(e[3], 0.1365693813556253, 6)

        self.assertAlmostEqual(p[0],1.9782067810173705, 6)
        self.assertAlmostEqual(p[1],1.9515304367929138, 6)
        self.assertAlmostEqual(p[2],1.9685631791867026, 6)
        self.assertAlmostEqual(p[3],1.9685631791867015, 6)

        dm1_exc = myadcea.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 83.67374798143112, 5)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 67.06120340703302, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 66.38905843913793, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 66.38905843913791, 6)

    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        myadc.method_type = "ea"

        myadc.kernel_gs()
        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.26407941243784, 6)

        myadcea = adc.radc_ea.RADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.31694173142858517 , 6)

        self.assertAlmostEqual(e[0], 0.0936790850738445, 6)
        self.assertAlmostEqual(e[1], 0.09836545539216629, 6)
        self.assertAlmostEqual(e[2], 0.1295709313652367, 6)

        self.assertAlmostEqual(p[0], 1.8324175318668088, 6)
        self.assertAlmostEqual(p[1], 1.9841043692706433, 6)
        self.assertAlmostEqual(p[2], 1.9637729885476354, 6)

        dm1_exc = myadcea.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 54.926419205699936, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 84.01334959967382, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 67.15340996040175, 6)

    def test_ea_adc3_frozen(self):

        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ea"

        myadc_fr.kernel_gs()
        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.262265370699176, 6)

        myadcea_fr = adc.radc_ea.RADCEA(myadc_fr)
        e,v,p,x = myadcea_fr.kernel(nroots=3)
        e_corr = myadcea_fr.e_corr

        self.assertAlmostEqual(e_corr, -0.3146743531878695 , 6)

        self.assertAlmostEqual(e[0], 0.0937398652995504, 6)
        self.assertAlmostEqual(e[1], 0.0983621084096212, 6)
        self.assertAlmostEqual(e[2], 0.1295704476416120, 6)

        self.assertAlmostEqual(p[0], 1.8324350368047113, 6)
        self.assertAlmostEqual(p[1], 1.9840978131606113, 6)
        self.assertAlmostEqual(p[2], 1.963749577203073, 6)

        dm1_exc = myadcea_fr.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 54.932345509170624, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 84.01085421841836, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 67.15120483661511, 6)

if __name__ == "__main__":
    print("EA calculations for different RADC methods for nitrogen molecule")
    unittest.main()
