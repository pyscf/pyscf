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
        self.assertAlmostEqual(p[1],0.0026423690445117006, 6)
        self.assertAlmostEqual(p[2],0.0036106496473683517, 6)
        self.assertAlmostEqual(p[3],0.017775953540499233, 6)

        self.assertAlmostEqual(spin[0],0.7522904063542506 , 5)
        self.assertAlmostEqual(spin[1],0.7522576307332178 , 5)
        self.assertAlmostEqual(spin[2],2.4209847454055358 , 5)
        self.assertAlmostEqual(spin[3],1.1624176868639315 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.772824583466086, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.529160643074512, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.507145256344522, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.665490411634828, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0120336045, 6)
        self.assertAlmostEqual(e[1], 0.1451768357, 6)
        self.assertAlmostEqual(e[2], 0.2705711303, 6)
        self.assertAlmostEqual(e[3], 0.3014583658, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0022511415415090983 , 6)
        self.assertAlmostEqual(p[2],0.0002781504563224257 , 6)
        self.assertAlmostEqual(p[3],0.01652844606741567 , 6)

        self.assertAlmostEqual(spin[0], 0.7505358324502627 , 5)
        self.assertAlmostEqual(spin[1],0.7504465152068174  , 5)
        self.assertAlmostEqual(spin[2],3.5572050239862154  , 5)
        self.assertAlmostEqual(spin[3],0.8620825777206886  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.83336927897167, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.66583211526736, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.21434753894557, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.631038187408834, 6)

    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4, guess="cis")
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0120336045, 6)
        self.assertAlmostEqual(e[1], 0.1451768357, 6)
        self.assertAlmostEqual(e[2], 0.2705711303, 6)
        self.assertAlmostEqual(e[3], 0.3014583658, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0022511605960848606 , 6)
        self.assertAlmostEqual(p[2],0.00027814254106808736 , 6)
        self.assertAlmostEqual(p[3],0.01652851698228148 , 6)

        self.assertAlmostEqual(spin[0],0.7505358312776593 , 5)
        self.assertAlmostEqual(spin[1],0.7504464970346452  , 5)
        self.assertAlmostEqual(spin[2],3.5572050912724036  , 5)
        self.assertAlmostEqual(spin[3],0.8620826405381923  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.83337091438862, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.665833348936594, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.214342795652065, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.631054555757274, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.0018738819, 6)
        self.assertAlmostEqual(e[1], 0.1573286345, 6)
        self.assertAlmostEqual(e[2], 0.2886390881, 6)
        self.assertAlmostEqual(e[3], 0.3214724068, 6)

        self.assertAlmostEqual(p[0],-0.00000000, 6)
        self.assertAlmostEqual(p[1],0.0024042299624731298 , 6)
        self.assertAlmostEqual(p[2],9.016719283617845e-05 , 6)
        self.assertAlmostEqual(p[3],0.016241108582614686 , 6)

        self.assertAlmostEqual(spin[0], 0.75005719 , 5)
        self.assertAlmostEqual(spin[1],0.75008973  , 5)
        self.assertAlmostEqual(spin[2],3.67821801  , 5)
        self.assertAlmostEqual(spin[3],0.79204834  , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 14.87648189234282, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 14.671498868364228, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 22.18814573937499, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.592694202413846, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for OH molecule")
    unittest.main()
