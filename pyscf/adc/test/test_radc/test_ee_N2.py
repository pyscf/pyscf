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

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3526821493, 6)
        self.assertAlmostEqual(e[1],0.3526821493, 6)
        self.assertAlmostEqual(e[2],0.3834249651, 6)
        self.assertAlmostEqual(e[3],0.4023911887, 6)

        self.assertAlmostEqual(p[0], 4.011980472665149e-28, 6)
        self.assertAlmostEqual(p[1], 3.896100510696497e-28, 6)
        self.assertAlmostEqual(p[2], 9.756742872824022e-12, 6)
        self.assertAlmostEqual(p[3], 7.573795105323254e-31, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 39.97872965521998, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 39.97872965522006, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 40.69761601957895, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 41.00886197958478, 4)


    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"


        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3200995643, 6)
        self.assertAlmostEqual(e[1],0.3200995643, 6)
        self.assertAlmostEqual(e[2],0.3671739857, 6)
        self.assertAlmostEqual(e[3],0.3825795703, 6)

        self.assertAlmostEqual(p[0], 6.35494709215813e-28, 6)
        self.assertAlmostEqual(p[1], 6.647082355981768e-28, 6)
        self.assertAlmostEqual(p[2], 2.4002663647119427e-16, 6)
        self.assertAlmostEqual(p[3], 2.0855075929786046e-30, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 39.89352215230514, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 39.89352215230519, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 40.70813622842074, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 40.91702978478106, 4)


    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3424250025, 6)
        self.assertAlmostEqual(e[1],0.3424250025, 6)
        self.assertAlmostEqual(e[2],0.3534967080, 6)
        self.assertAlmostEqual(e[3],0.3673275757, 6)

        self.assertAlmostEqual(p[0], 0.00000000, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.00000000, 6)
        self.assertAlmostEqual(p[3], 0.00000000, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 40.1490886679037, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 40.1490886679038, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 41.0398051598256, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 41.2278387969904, 4)


    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ee"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.47517224542625, 6)

        myadcee_fr = adc.radc_ee.RADCEE(myadc_fr)
        e,v,p,x = myadcee_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3424857152380730, 6)
        self.assertAlmostEqual(e[1],0.3424857152380766, 6)
        self.assertAlmostEqual(e[2],0.3535001951670751, 6)
        self.assertAlmostEqual(e[3],0.3673334752099558, 6)

        self.assertAlmostEqual(p[0], 0.00000000, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.00000000, 6)
        self.assertAlmostEqual(p[3], 0.00000000, 6)

        dm1_exc = np.array(myadcee_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 40.14978032095179, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 40.14978032095169, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 41.03909647851530, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 41.22750275223949, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for nitrogen molecule")
    unittest.main()
