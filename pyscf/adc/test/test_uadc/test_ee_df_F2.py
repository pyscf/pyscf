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
# Author: Terrence Stahl <terrencestahl1@@gmail.com>
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
        ['F', (0., 0.    , -r/2   )],
        ['F', (0., 0.    ,  r/2)],]
    mol.basis = {'F':'cc-pvdz'}

    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc.max_memory = 1

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

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3954389168, 6)
        self.assertAlmostEqual(e[1],0.3954389168, 6)
        self.assertAlmostEqual(e[2],0.4626206038, 6)
        self.assertAlmostEqual(e[3],0.4626206038, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00101509, 6)
        self.assertAlmostEqual(p[3],0.00101509, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 40.39255755554787, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 40.39255755554789, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.46594285522331, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.46594285522332, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3737913888, 6)
        self.assertAlmostEqual(e[1],0.3737913888, 6)
        self.assertAlmostEqual(e[2],0.4397242667, 6)
        self.assertAlmostEqual(e[3],0.4397242667, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00101221, 6)
        self.assertAlmostEqual(p[3],0.00101221, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 40.44719491030430, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 40.44719491030431, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.49491596837851, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.49491596837851, 6)

    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4, guess="cis")

        self.assertAlmostEqual(e[0],0.3737913887927986, 6)
        self.assertAlmostEqual(e[1],0.3737913887928052, 6)
        self.assertAlmostEqual(e[2],0.4397242666784026, 6)
        self.assertAlmostEqual(e[3],0.4397242666784047, 6)

        self.assertAlmostEqual(p[0],1.2824740125029445e-18, 6)
        self.assertAlmostEqual(p[1],1.2824741368625982e-18, 6)
        self.assertAlmostEqual(p[2],0.001012208420973050, 6)
        self.assertAlmostEqual(p[3],0.001012208420972989, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 40.44719515611976, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 40.44719515611981, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.49491598756553, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.49491598756554, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3883587379, 6)
        self.assertAlmostEqual(e[1],0.3883587379, 6)
        self.assertAlmostEqual(e[2],0.4564907388, 6)
        self.assertAlmostEqual(e[3],0.4564907388, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00110716, 6)
        self.assertAlmostEqual(p[3],0.00110716, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 40.27503915560749, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 40.27503915560749, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.32269927919967, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.32269927919967, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for water molecule")
    unittest.main()
