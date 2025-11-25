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
import math
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
    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc.max_memory = 1
    myadc_fr = adc.ADC(mf,frozen=1).density_fit('cc-pvdz-ri')
    myadc_fr.max_memory = 1

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3526863602, 6)
        self.assertAlmostEqual(e[1],0.3526863602, 6)
        self.assertAlmostEqual(e[2],0.3835475416, 6)
        self.assertAlmostEqual(e[3],0.4025467968, 6)

        self.assertAlmostEqual(p[0], 2.855544182310304e-28, 6)
        self.assertAlmostEqual(p[1], 2.802986870504723e-28, 6)
        self.assertAlmostEqual(p[2], 1.9458147244035698e-13, 6)
        self.assertAlmostEqual(p[3], 6.481812538202186e-30, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 39.97509426976306, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 39.97509426976296, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 40.69394840350379, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 40.99987050864409, 4)


    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3200048134, 6)
        self.assertAlmostEqual(e[1],0.3200048134, 6)
        self.assertAlmostEqual(e[2],0.3672515777, 6)
        self.assertAlmostEqual(e[3],0.3825657097, 6)

        self.assertAlmostEqual(p[0], 9.086432277311463e-28, 6)
        self.assertAlmostEqual(p[1], 8.554120249383773e-28, 6)
        self.assertAlmostEqual(p[2], 1.7961311647425857e-14, 6)
        self.assertAlmostEqual(p[3], 4.191312429576048e-29, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 39.89421703577403, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 39.89421703577423, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 40.70732064167630, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 40.91094017494895, 4)


    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4, guess="cis")

        self.assertAlmostEqual(e[0],0.320004813344214, 6)
        self.assertAlmostEqual(e[1],0.320004813344215, 6)
        self.assertAlmostEqual(e[2],0.367251590751456, 6)
        self.assertAlmostEqual(e[3],0.382565730257061, 6)

        self.assertAlmostEqual(p[0], 2.628300736639355e-29, 6)
        self.assertAlmostEqual(p[1], 1.9474314374445225e-29, 6)
        self.assertAlmostEqual(p[2], 2.5741804358254784e-13, 6)
        self.assertAlmostEqual(p[3], 3.348087946831387e-30, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 39.89421955845101, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 39.89421955845102, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 40.70732874342255, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 40.91091417592432, 4)


    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3424328757, 6)
        self.assertAlmostEqual(e[1],0.3424328757, 6)
        self.assertAlmostEqual(e[2],0.3536378790, 6)
        self.assertAlmostEqual(e[3],0.3673436817, 6)

        self.assertAlmostEqual(p[0], 0.00000000, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.00000000, 6)
        self.assertAlmostEqual(p[3], 0.00000000, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 40.15369498656837, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 40.15369498656815, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 41.04165107132388, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 41.22412259571227, 4)


    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ee"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs,39.47555886005903, 6)

        myadcee_fr = adc.radc_ee.RADCEE(myadc_fr)
        e,v,p,x = myadcee_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3424936361760660, 6)
        self.assertAlmostEqual(e[1],0.3424936361760684, 6)
        self.assertAlmostEqual(e[2],0.3536413935766013, 6)
        self.assertAlmostEqual(e[3],0.3673494903947555, 6)

        dm1_exc = np.array(myadcee_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 40.15438985177842, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 40.15438985177845, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 41.04092622313121, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 41.22378314449435, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for nitrogen molecule")
    unittest.main()
