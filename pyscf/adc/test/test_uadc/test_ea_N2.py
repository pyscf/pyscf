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

def rdms_test(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', mf.mo_coeff[0], dm_a, mf.mo_coeff[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', mf.mo_coeff[1], dm_b, mf.mo_coeff[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.32201692499346535, 6)

        self.assertAlmostEqual(e[0], 0.09617819142992463, 6)
        self.assertAlmostEqual(e[1], 0.09617819161216855, 6)
        self.assertAlmostEqual(e[2], 0.1258326904883586, 6)

        self.assertAlmostEqual(p[0], 0.9916427196092643, 6)
        self.assertAlmostEqual(p[1], 0.9916427196903126, 6)
        self.assertAlmostEqual(p[2], 0.9817184085436222, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 83.57588399390453, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 83.57588399983187, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 66.92768336289278, 6)

    def test_ea_adc2_oneroot(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.09617819142992463, 6)

        self.assertAlmostEqual(p[0], 0.9916427196092643, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 83.57588399402539, 6)

    def test_ea_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.0953065329249756, 6)
        self.assertAlmostEqual(e[1], 0.09530653311160658, 6)
        self.assertAlmostEqual(e[2], 0.12388330778444741, 6)
        self.assertAlmostEqual(e[3], 0.1238833087377404, 6)

        self.assertAlmostEqual(p[0], 0.9890885390419444 , 6)
        self.assertAlmostEqual(p[1],0.9890885391436558 , 6)
        self.assertAlmostEqual(p[2],0.9757598335805556 , 6)
        self.assertAlmostEqual(p[3],0.9757598335315953 , 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 83.5345442972655, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 83.5345443033782, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 66.8646391400683, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 66.8646393848044, 6)

    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31694173142858517 , 6)

        myadcea = adc.uadc_ea.UADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.09836545519294707, 6)
        self.assertAlmostEqual(e[1], 0.09836545535648182, 6)
        self.assertAlmostEqual(e[2], 0.12957093060937017, 6)

        self.assertAlmostEqual(p[0], 0.9920495595411523, 6)
        self.assertAlmostEqual(p[1], 0.9920495596160825, 6)
        self.assertAlmostEqual(p[2], 0.9819275025204279, 6)

        dm1_exc = np.array(myadcea.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 84.21966265529424, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 84.21966266060332, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 67.34876064018454, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods")
    unittest.main()
