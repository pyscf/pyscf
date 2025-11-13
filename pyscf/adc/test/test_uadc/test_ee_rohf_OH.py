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
from pyscf.adc.uadc_ee import get_spin_square

def setUpModule():
    global mol, mf, myadc
    basis = 'cc-pVDZ'

    mol = gto.Mole()
    mol.atom = '''
        O 0.00000000 0.00000000 -0.10864763
        H 0.00000000 0.00000000 1.72431679
         '''
    mol.basis = {'H': basis,
                 'O': basis,}
    mol.verbose = 0
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.build()

    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

def rdms_test(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', myadc.mo_coeff[0], dm_a, myadc.mo_coeff[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', myadc.mo_coeff[1], dm_b, myadc.mo_coeff[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0017134635, 6)
        self.assertAlmostEqual(e[1],0.1629470674, 6)
        self.assertAlmostEqual(e[2],0.2991743493, 6)
        self.assertAlmostEqual(e[3],0.3369031510, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00255436, 6)
        self.assertAlmostEqual(p[2],0.00356489, 6)
        self.assertAlmostEqual(p[3],0.01811875, 6)

        self.assertAlmostEqual(spin[0],0.75053521 , 5)
        self.assertAlmostEqual(spin[1],0.75064928 , 5)
        self.assertAlmostEqual(spin[2],2.43570834 , 5)
        self.assertAlmostEqual(spin[3],1.14753801 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.82834303951204, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.58906976246459, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.55185011193399, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.69989460440471, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0126491070, 6)
        self.assertAlmostEqual(e[1], 0.1437881080, 6)
        self.assertAlmostEqual(e[2], 0.2676394327, 6)
        self.assertAlmostEqual(e[3], 0.2991598135, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00219744 , 6)
        self.assertAlmostEqual(p[2],0.00012567 , 6)
        self.assertAlmostEqual(p[3],0.01591094 , 6)

        self.assertAlmostEqual(spin[0], 0.75043297 , 5)
        self.assertAlmostEqual(spin[1],0.75037407  , 5)
        self.assertAlmostEqual(spin[2],3.65784145  , 5)
        self.assertAlmostEqual(spin[3],0.80459280  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.88282768620535, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.71720611496241, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.23562693053046, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.67130682497491, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0019456025, 6)
        self.assertAlmostEqual(e[1], 0.1567874644, 6)
        self.assertAlmostEqual(e[2], 0.2837638864, 6)
        self.assertAlmostEqual(e[3], 0.3187665826, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00239567 , 6)
        self.assertAlmostEqual(p[2],0.00000000 , 6)
        self.assertAlmostEqual(p[3],0.01485308 , 6)

        self.assertAlmostEqual(spin[0], 0.75027556 , 5)
        self.assertAlmostEqual(spin[1],0.75009053  , 5)
        self.assertAlmostEqual(spin[2],3.72613641  , 5)
        self.assertAlmostEqual(spin[3],0.75901326  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.853375423229501, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.651068073945451, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.190262331871086, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.561776913279605, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
