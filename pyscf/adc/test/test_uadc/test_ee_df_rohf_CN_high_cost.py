from pyscf.adc.uadc_ee import get_spin_square
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

    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc_fr = adc.ADC(mf,frozen=[1,1]).density_fit('cc-pvdz-ri')

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', myadc.mo_coeff_hf[0], dm_a, myadc.mo_coeff_hf[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', myadc.mo_coeff_hf[1], dm_b, myadc.mo_coeff_hf[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

def rdms_test_fr(dm_a,dm_b):
    r2_int = mol.intor('int1e_r2')
    dm_ao_a = np.einsum('pi,ij,qj->pq', myadc_fr.mo_coeff_hf[0], dm_a, myadc_fr.mo_coeff_hf[0].conj())
    dm_ao_b = np.einsum('pi,ij,qj->pq', myadc_fr.mo_coeff_hf[1], dm_b, myadc_fr.mo_coeff_hf[1].conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao_a+dm_ao_b)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0540311799, 6)
        self.assertAlmostEqual(e[1],0.0540311799, 6)
        self.assertAlmostEqual(e[2],0.0955877367, 6)
        self.assertAlmostEqual(e[3],0.2557509285, 6)

        self.assertAlmostEqual(p[0],0.003522710991629426, 6)
        self.assertAlmostEqual(p[1],0.0035227109916294427, 6)
        self.assertAlmostEqual(p[2],0.017251139093043242, 6)
        self.assertAlmostEqual(p[3],0.0013922492604078337, 6)

        self.assertAlmostEqual(spin[0],0.7556373244766919 , 5)
        self.assertAlmostEqual(spin[1],0.7556373244766874 , 5)
        self.assertAlmostEqual(spin[2],0.7679860817693429 , 5)
        self.assertAlmostEqual(spin[3],2.8381658311615974 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.556138997958605, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.55613899795861, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.851410102506, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.68103064671977, 4)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0163448011, 6)
        self.assertAlmostEqual(e[1],0.0163448011, 6)
        self.assertAlmostEqual(e[2],0.0529820632, 6)
        self.assertAlmostEqual(e[3],0.1763225747, 6)

        self.assertAlmostEqual(p[0],0.0009647395863965671, 6)
        self.assertAlmostEqual(p[1],0.0009647395863966104 , 6)
        self.assertAlmostEqual(p[2],0.00884233523268509 , 6)
        self.assertAlmostEqual(p[3],0.0005053112529702919 , 6)

        self.assertAlmostEqual(spin[0],0.7556231067434602 , 5)
        self.assertAlmostEqual(spin[1],0.7556231067434611 , 5)
        self.assertAlmostEqual(spin[2],0.7621051029882153 , 5)
        self.assertAlmostEqual(spin[3],3.3047303922738145 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.264098785842314, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.2640987858423, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.560875044258545, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.69696029523196, 4)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362451866, 6)
        self.assertAlmostEqual(e[1],0.0362451866, 6)
        self.assertAlmostEqual(e[2],0.1200323452, 6)
        self.assertAlmostEqual(e[3],0.1747553376, 6)

        self.assertAlmostEqual(p[0],0.0021536892401642117, 6)
        self.assertAlmostEqual(p[1],0.002153689240164236, 6)
        self.assertAlmostEqual(p[2],0.020705580027506585, 6)
        self.assertAlmostEqual(p[3],0.001430808456162377, 6)

        self.assertAlmostEqual(spin[0],0.75368535 , 5)
        self.assertAlmostEqual(spin[1],0.75368535 , 5)
        self.assertAlmostEqual(spin[2],0.79361946 , 5)
        self.assertAlmostEqual(spin[3],3.44107298 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.266259831098125, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.266259831098125, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.5548352413061, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.63763558721577, 4)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362186495, 6)
        self.assertAlmostEqual(e[1],0.0362186495, 6)
        self.assertAlmostEqual(e[2],0.1200847879, 6)
        self.assertAlmostEqual(e[3],0.1747766797, 6)

        self.assertAlmostEqual(p[0],0.0021529768901591725, 6)
        self.assertAlmostEqual(p[1],0.0021529768901592077, 6)
        self.assertAlmostEqual(p[2],0.020713392796838745, 6)
        self.assertAlmostEqual(p[3],0.0014329585553095256, 6)

        self.assertAlmostEqual(spin[0],0.75368453 , 5)
        self.assertAlmostEqual(spin[1],0.75368453 , 5)
        self.assertAlmostEqual(spin[2],0.79369576 , 5)
        self.assertAlmostEqual(spin[3],3.44111977 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][0],dm1_exc[1][0]), 41.26553326509657, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][1],dm1_exc[1][1]), 41.265533265096586, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][2],dm1_exc[1][2]), 40.55424254618611, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][3],dm1_exc[1][3]), 40.63733369305047, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
