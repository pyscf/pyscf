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
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc, myadc_fr
    basis = 'cc-pVDZ'
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
        C 0.00000000 0.00000000 -1.18953886
        N 0.00000000 0.00000000 1.01938091
         '''
    mol.basis = {'C': basis,
                 'N': basis,}
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=[1,1])

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

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
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0789393239, 6)
        self.assertAlmostEqual(e[1],0.0789393239, 6)
        self.assertAlmostEqual(e[2],0.1397085217, 6)
        self.assertAlmostEqual(e[3],0.2552893678, 6)

        self.assertAlmostEqual(p[0],0.0033836167048380134, 6)
        self.assertAlmostEqual(p[1],0.0033836167048380645, 6)
        self.assertAlmostEqual(p[2],0.011886684786878812, 6)
        self.assertAlmostEqual(p[3],0.005988588887973458, 6)

        self.assertAlmostEqual(spin[0],0.8875396093786092 , 5)
        self.assertAlmostEqual(spin[1],0.887539609378611 , 5)
        self.assertAlmostEqual(spin[2],1.1017202202637613 , 5)
        self.assertAlmostEqual(spin[3],2.6668499895288056 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.88473480537786, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.884734805377924, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 41.08482335383598, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.75582143793107, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0066160563, 6)
        self.assertAlmostEqual(e[1],0.0066160563, 6)
        self.assertAlmostEqual(e[2],0.0674217414, 6)
        self.assertAlmostEqual(e[3],0.1755586417, 6)

        self.assertAlmostEqual(p[0],0.00023276121952182924, 6)
        self.assertAlmostEqual(p[1],0.00023276121952187244, 6)
        self.assertAlmostEqual(p[2],0.005510428569707843, 6)
        self.assertAlmostEqual(p[3],0.00012683778037866073, 6)

        self.assertAlmostEqual(spin[0],0.8237305796467256 , 5)
        self.assertAlmostEqual(spin[1],0.8237305796467282 , 5)
        self.assertAlmostEqual(spin[2],0.8778829114629021 , 5)
        self.assertAlmostEqual(spin[3],4.015400829318012 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.352335746683075, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.352335746683075, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.61156613734316, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.585596777443975, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        myadc.max_memory = 20
        myadc.incore_complete = False

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0432157465, 6)
        self.assertAlmostEqual(e[1],0.0432157465, 6)
        self.assertAlmostEqual(e[2],0.1276752421, 6)
        self.assertAlmostEqual(e[3],0.1848576902, 6)

        self.assertAlmostEqual(p[0],0.001929265751970036, 6)
        self.assertAlmostEqual(p[1],0.001929265751970083, 6)
        self.assertAlmostEqual(p[2],0.012786981363066056, 6)
        self.assertAlmostEqual(p[3],0.00014257794859828544, 6)

        self.assertAlmostEqual(spin[0],0.80195386 , 5)
        self.assertAlmostEqual(spin[1],0.80195386 , 5)
        self.assertAlmostEqual(spin[2],0.82700993 , 5)
        self.assertAlmostEqual(spin[3],4.08789116 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.3393264898107, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.3393264898107, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.60286826470329, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.62371512382642, 6)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.max_memory = 20
        myadc_fr.incore_complete = False

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.04319092877771163, 6)
        self.assertAlmostEqual(e[1],0.04319092877771207, 6)
        self.assertAlmostEqual(e[2],0.12770468024295392, 6)
        self.assertAlmostEqual(e[3],0.18489326630403885, 6)

        self.assertAlmostEqual(p[0],0.001928948590435294, 6)
        self.assertAlmostEqual(p[1],0.001928948590435386, 6)
        self.assertAlmostEqual(p[2],0.01279618890362342, 6)
        self.assertAlmostEqual(p[3],0.00014137665143170662, 6)

        self.assertAlmostEqual(spin[0],0.80197118, 5)
        self.assertAlmostEqual(spin[1],0.80197118, 5)
        self.assertAlmostEqual(spin[2],0.82709336, 5)
        self.assertAlmostEqual(spin[3],4.08781662, 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.338680505718365, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.33868050571839, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.602128264908764, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.62340041293398, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
