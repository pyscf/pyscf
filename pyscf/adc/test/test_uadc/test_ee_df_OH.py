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

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
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

        self.assertAlmostEqual(e[0],0.0023319460, 6)
        self.assertAlmostEqual(e[1],0.1647722041, 6)
        self.assertAlmostEqual(e[2],0.2984586991, 6)
        self.assertAlmostEqual(e[3],0.3367095228, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.002641029829841539, 6)
        self.assertAlmostEqual(p[2],0.003656517697292993, 6)
        self.assertAlmostEqual(p[3],0.017715659957811148, 6)

        self.assertAlmostEqual(spin[0],0.7522899407918877 , 5)
        self.assertAlmostEqual(spin[1],0.7522570475485444 , 5)
        self.assertAlmostEqual(spin[2],2.41538856489086 , 5)
        self.assertAlmostEqual(spin[3],1.1675374360426267 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.772768974119947, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.529051932389352, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.51072756765637, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.671981800044968, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0121277301, 6)
        self.assertAlmostEqual(e[1], 0.1450929061, 6)
        self.assertAlmostEqual(e[2], 0.2704365078, 6)
        self.assertAlmostEqual(e[3], 0.3008905139, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0022491067650228487  , 6)
        self.assertAlmostEqual(p[2],0.0002804455953225166  , 6)
        self.assertAlmostEqual(p[3],0.016515406273268366  , 6)

        self.assertAlmostEqual(spin[0], 0.7505350578382504 , 5)
        self.assertAlmostEqual(spin[1],0.7504455580264451  , 5)
        self.assertAlmostEqual(spin[2],3.5557522010898035  , 5)
        self.assertAlmostEqual(spin[3],0.8628727750600103  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.83333272495166, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.665626786429073, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.214774091842784, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.640329643592988, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0019124461, 6)
        self.assertAlmostEqual(e[1], 0.1572857305, 6)
        self.assertAlmostEqual(e[2], 0.2885897977, 6)
        self.assertAlmostEqual(e[3], 0.3209639554, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.002401458941339693  , 6)
        self.assertAlmostEqual(p[2],8.8161773092754e-05  , 6)
        self.assertAlmostEqual(p[3],0.016240651756407938  , 6)

        self.assertAlmostEqual(spin[0], 0.75005767 , 5)
        self.assertAlmostEqual(spin[1],0.75009003  , 5)
        self.assertAlmostEqual(spin[2],3.67854726  , 5)
        self.assertAlmostEqual(spin[3],0.79155905  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.876507097496752, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.671284214272507, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.187936711933272, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.602322381654172, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
