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
# Author: Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ip import get_spin_square

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
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6

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

    def test_ip_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.42781880033860314, 6)
        self.assertAlmostEqual(e[1],0.46907045337105735, 6)
        self.assertAlmostEqual(e[2],0.5431040830749944, 6)
        self.assertAlmostEqual(e[3],0.5762174936729705, 6)

        self.assertAlmostEqual(p[0],0.9265960031744254, 6)
        self.assertAlmostEqual(p[1],0.9254224779016145, 6)
        self.assertAlmostEqual(p[2],0.9201867229238191, 6)
        self.assertAlmostEqual(p[3],0.9358802440416575, 6)

        self.assertAlmostEqual(spin[0],2.000059858727739 , 5)
        self.assertAlmostEqual(spin[1],1.0885067780881297 , 5)
        self.assertAlmostEqual(spin[2],8.913210701289032e-05 , 5)
        self.assertAlmostEqual(spin[3],2.000069813885596 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 10.890842960233874, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 10.928090954613072, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 11.030219323119942, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 10.721041547076947, 6)

    def test_ip_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.4258371206016643, 6)
        self.assertAlmostEqual(e[1],0.4422922300127146, 6)
        self.assertAlmostEqual(e[2],0.5207968948853017, 6)
        self.assertAlmostEqual(e[3],0.5375928753636899, 6)

        self.assertAlmostEqual(p[0],0.9253686259688075, 6)
        self.assertAlmostEqual(p[1],0.6625677843241929, 6)
        self.assertAlmostEqual(p[2],0.708151653650922, 6)
        self.assertAlmostEqual(p[3],0.26386542338663466, 6)

        self.assertAlmostEqual(spin[0],2.000117412211435 , 5)
        self.assertAlmostEqual(spin[1],1.8801574259081417 , 5)
        self.assertAlmostEqual(spin[2],0.003875130547862593 , 5)
        self.assertAlmostEqual(spin[3],0.1292968937664285 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 10.909463657422874, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 11.285692585465599, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 11.284691672596363, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 11.863409488762452, 6)

    def test_ip_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.4573063162162001, 6)
        self.assertAlmostEqual(e[1],0.4625698379928462, 6)
        self.assertAlmostEqual(e[2],0.5447632725711331, 6)
        self.assertAlmostEqual(e[3],0.5486099900053525, 6)

        self.assertAlmostEqual(p[0],0.9382688531842012, 6)
        self.assertAlmostEqual(p[1],0.5221200060893977, 6)
        self.assertAlmostEqual(p[2],0.47831063378102934, 6)
        self.assertAlmostEqual(p[3],0.416377569294319, 6)

        self.assertAlmostEqual(spin[0],2.0001175984678476 , 5)
        self.assertAlmostEqual(spin[1],1.9700724330353152 , 5)
        self.assertAlmostEqual(spin[2],0.006487158982737107 , 5)
        self.assertAlmostEqual(spin[3],0.0392210403642963 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 11.14619245448431, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 11.60631691179165, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 11.692931634106852, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 11.782867290934647, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for OH molecule with ROHF reference")
    unittest.main()
