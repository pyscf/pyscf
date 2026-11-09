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
    np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

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

    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')

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

        self.assertAlmostEqual(e[0],0.0016936152, 6)
        self.assertAlmostEqual(e[1],0.1629225558, 6)
        self.assertAlmostEqual(e[2],0.2989590171, 6)
        self.assertAlmostEqual(e[3],0.3364100601, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0026014708573550726, 6)
        self.assertAlmostEqual(p[2],0.0035304398511850806, 6)
        self.assertAlmostEqual(p[3],0.017837496001897775, 6)

        self.assertAlmostEqual(spin[0],0.750732691596832 , 5)
        self.assertAlmostEqual(spin[1],0.7509024216572371 , 5)
        self.assertAlmostEqual(spin[2],2.43044095 , 5)
        self.assertAlmostEqual(spin[3],1.15231877 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.77871127299143, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.547688991700122, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.53358526906051, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.650221829059156, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0127433223, 6)
        self.assertAlmostEqual(e[1], 0.1437039204, 6)
        self.assertAlmostEqual(e[2], 0.2675152327, 6)
        self.assertAlmostEqual(e[3], 0.2986007925, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.002234828651233847  , 6)
        self.assertAlmostEqual(p[2],0.00011707523964357899  , 6)
        self.assertAlmostEqual(p[3],0.01565573834218515  , 6)

        self.assertAlmostEqual(spin[0], 0.7506069828778004 , 5)
        self.assertAlmostEqual(spin[1],0.7505832857305483  , 5)
        self.assertAlmostEqual(spin[2],3.65768347  , 5)
        self.assertAlmostEqual(spin[3],0.80434287  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.833635871602548, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.676346480804337, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.209324675571462, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.653754419094913, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0019861472, 6)
        self.assertAlmostEqual(e[1], 0.1567423657, 6)
        self.assertAlmostEqual(e[2], 0.2837047778, 6)
        self.assertAlmostEqual(e[3], 0.3182824901, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0023940524971269193  , 6)
        self.assertAlmostEqual(p[2],0.00000007  , 6)
        self.assertAlmostEqual(p[3],0.014929245491506338  , 6)

        self.assertAlmostEqual(spin[0], 0.75002133 , 5)
        self.assertAlmostEqual(spin[1],0.75000123  , 5)
        self.assertAlmostEqual(spin[2],3.71870676  , 5)
        self.assertAlmostEqual(spin[3],0.76034789  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.87875726357427, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.679804669599099, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.204358867491596, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.594032522214427, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
