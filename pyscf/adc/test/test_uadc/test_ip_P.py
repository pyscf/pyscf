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
    mol = gto.Mole()
    mol.atom = [
        ['P', (0., 0.    , 0.)],]
    mol.basis = {'P':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.spin = 3
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.irrep_nelec = {'A1g':(3,3),'E1ux':(2,1),'E1uy':(2,1),'A1u':(2,1)}
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

    def test_ip_adc2(self):
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.07351147825007748, 6)

        myadcip = adc.uadc_ip.UADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.38071502275761, 6)
        self.assertAlmostEqual(e[1], 0.38071502275761, 6)
        self.assertAlmostEqual(e[2], 0.38071502275761, 6)

        self.assertAlmostEqual(p[0], 0.94893331006538, 6)
        self.assertAlmostEqual(p[1], 0.94893331006538, 6)
        self.assertAlmostEqual(p[2], 0.94893331006538, 6)

        dm1_exc = np.array(myadcip.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 21.53337229793503, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 21.53337229793504, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 21.53337229793506, 6)

    def test_ip_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.07351147825007748, 6)

        myadcip = adc.uadc_ip.UADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.36951642121691, 6)
        self.assertAlmostEqual(e[1], 0.36951642121691,  6)
        self.assertAlmostEqual(e[2], 0.36951642121691, 6)

        self.assertAlmostEqual(p[0], 0.92695415467651,  6)
        self.assertAlmostEqual(p[1], 0.92695415467651,  6)
        self.assertAlmostEqual(p[2], 0.92695415467651,  6)

        dm1_exc = np.array(myadcip.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 21.18511985058944, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 21.18511985058942, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 21.18511985058942, 6)

    def test_ip_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.0892040998302457, 6)

        myadcip = adc.uadc_ip.UADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.37866365404487, 6)
        self.assertAlmostEqual(e[1], 0.37866365404487, 6)
        self.assertAlmostEqual(e[2], 0.37866365404487, 6)

        self.assertAlmostEqual(p[0], 0.92603990249875, 6)
        self.assertAlmostEqual(p[1], 0.92603990249875, 6)
        self.assertAlmostEqual(p[2], 0.92603990249875, 6)

        dm1_exc = np.array(myadcip.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 21.31223422821962, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 21.31223422821966, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 21.31223422821963, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell atom")
    unittest.main()
