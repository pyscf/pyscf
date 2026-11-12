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

        self.assertAlmostEqual(p[0], 0.9916557024452711, 6)
        self.assertAlmostEqual(p[1], 0.991655702461092, 6)
        self.assertAlmostEqual(p[2], 0.9817154528005753, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 83.71541981880094, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 83.71541982467924, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 67.12500375696602, 6)

    def test_ea_adc2_oneroot(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.09617819142992463, 6)

        self.assertAlmostEqual(p[0], 0.9916557024452705, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 83.7154198188013, 6)

    def test_ea_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.0953065329249756, 6)
        self.assertAlmostEqual(e[1], 0.09530653311160658, 6)
        self.assertAlmostEqual(e[2], 0.12388330778444741, 6)
        self.assertAlmostEqual(e[3], 0.1238833087377404, 6)

        self.assertAlmostEqual(p[0], 0.9891033830084986 , 6)
        self.assertAlmostEqual(p[1],0.989103383039204 , 6)
        self.assertAlmostEqual(p[2],0.9757652245876124 , 6)
        self.assertAlmostEqual(p[3],0.9757652247916179 , 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 83.67374351632323, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 83.67374352238434, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 67.06120463945703, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 67.06120488411699, 6)

    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        myadc.compute_spin_square = True
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31694173142858517 , 6)
        self.assertAlmostEqual(myadc.gs_spin_square, 0.0, 6)

        myadcea = adc.uadc_ea.UADCEA(myadc)
        e,v,p,x = myadcea.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.09836545519294707, 6)
        self.assertAlmostEqual(e[1], 0.09836545535648182, 6)
        self.assertAlmostEqual(e[2], 0.12957093060937017, 6)

        self.assertAlmostEqual(p[0], 0.9920521843780394, 6)
        self.assertAlmostEqual(p[1], 0.9920521844063711, 6)
        self.assertAlmostEqual(p[2], 0.9818864846584316, 6)

        dm1_exc = np.array(myadcea.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 84.01334722048755, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 84.01334722564656, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 67.15340938773821, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods")
    unittest.main()
