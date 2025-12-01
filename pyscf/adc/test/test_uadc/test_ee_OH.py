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

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    myadc.max_cycle = 200

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
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0023522150, 6)
        self.assertAlmostEqual(e[1],0.1647973308, 6)
        self.assertAlmostEqual(e[2],0.2986841630, 6)
        self.assertAlmostEqual(e[3],0.3371941604, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00260529, 6)
        self.assertAlmostEqual(p[2],0.00370851, 6)
        self.assertAlmostEqual(p[3],0.01799256, 6)

        self.assertAlmostEqual(spin[0],0.75100183 , 5)
        self.assertAlmostEqual(spin[1],0.75099278 , 5)
        self.assertAlmostEqual(spin[2],2.41928532 , 5)
        self.assertAlmostEqual(spin[3],1.16078708 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.837720262024684, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.591445758752077, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.557484739864687, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.703749840718693, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0120336045, 6)
        self.assertAlmostEqual(e[1], 0.1451768357, 6)
        self.assertAlmostEqual(e[2], 0.2705711303, 6)
        self.assertAlmostEqual(e[3], 0.3014583658, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00222149 , 6)
        self.assertAlmostEqual(p[2],0.00029737 , 6)
        self.assertAlmostEqual(p[3],0.01679878 , 6)

        self.assertAlmostEqual(spin[0], 0.74929673 , 5)
        self.assertAlmostEqual(spin[1],0.74927348  , 5)
        self.assertAlmostEqual(spin[2],3.55591433  , 5)
        self.assertAlmostEqual(spin[3],0.86054541  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.89770575792029, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.72721959210633, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.25297378650939, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.66851273379779, 6)

    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4, guess="cis")
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0120336045, 6)
        self.assertAlmostEqual(e[1], 0.1451768357, 6)
        self.assertAlmostEqual(e[2], 0.2705711303, 6)
        self.assertAlmostEqual(e[3], 0.3014583658, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00222151 , 6)
        self.assertAlmostEqual(p[2],0.00029736 , 6)
        self.assertAlmostEqual(p[3],0.01679885 , 6)

        self.assertAlmostEqual(spin[0],0.74929673 , 5)
        self.assertAlmostEqual(spin[1],0.74927346  , 5)
        self.assertAlmostEqual(spin[2],3.55591440  , 5)
        self.assertAlmostEqual(spin[3],0.86054548  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.89770739393802, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.72722082654919, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.25296905236634, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.66852911063182, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0018738819, 6)
        self.assertAlmostEqual(e[1], 0.1573286345, 6)
        self.assertAlmostEqual(e[2], 0.2886390881, 6)
        self.assertAlmostEqual(e[3], 0.3214724068, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00240992 , 6)
        self.assertAlmostEqual(p[2],0.00009444 , 6)
        self.assertAlmostEqual(p[3],0.01617088 , 6)

        self.assertAlmostEqual(spin[0], 0.74912312 , 5)
        self.assertAlmostEqual(spin[1],0.74917845  , 5)
        self.assertAlmostEqual(spin[2],3.68386876  , 5)
        self.assertAlmostEqual(spin[3],0.79073584  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.85669167691214, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.65289124997399, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.17386872621784, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.57796047485072, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
