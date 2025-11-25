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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
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
    r = 0.969286393
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0.    , -r/2   )],
        ['H', (0., 0.    ,  r/2)],]
    mol.basis = {'O':'aug-cc-pvdz',
                 'H':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.symmetry = False
    mol.spin  = 1
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc_fr = adc.ADC(mf,frozen=(1,1))
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

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

    def test_ea_adc2(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387806, 6)

        self.assertAlmostEqual(e[0], -0.048666915263496924, 6)
        self.assertAlmostEqual(e[1], 0.030845983085818485, 6)
        self.assertAlmostEqual(e[2], 0.03253522816723711, 6)

        self.assertAlmostEqual(p[0], 0.9228959646746451, 6)
        self.assertAlmostEqual(p[1], 0.9953781149964537, 6)
        self.assertAlmostEqual(p[2], 0.9956169835481459, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 28.53839279735063, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 59.09140179648612, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.84138531264457, 6)

    def test_ea_adc2_oneroot(self):

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=1)

        self.assertAlmostEqual(e[0], 0.030845983085818485, 6)

        self.assertAlmostEqual(p[0], 0.9953781149964537, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 59.091402065010804, 6)

    def test_ea_adc2x(self):

        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], -0.07750642898162931, 6)
        self.assertAlmostEqual(e[1], 0.029292010466571882, 6)
        self.assertAlmostEqual(e[2], 0.030814773752482663, 6)

        self.assertAlmostEqual(p[0], 0.8323987058794676, 6)
        self.assertAlmostEqual(p[1], 0.9918705979602267, 6)
        self.assertAlmostEqual(p[2], 0.9772855298541363, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 30.61119659420692, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.70177395791268, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.30670214186572, 6)

    def test_ea_adc3(self):

        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072136, 6)

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=3)
        myadc.analyze()

        self.assertAlmostEqual(e[0], -0.045097652872531736, 6)
        self.assertAlmostEqual(e[1], 0.03004291636971322, 6)
        self.assertAlmostEqual(e[2], 0.03153897437644345, 6)

        self.assertAlmostEqual(p[0], 0.8722483551941809, 6)
        self.assertAlmostEqual(p[1], 0.9927117650068699, 6)
        self.assertAlmostEqual(p[2], 0.9766456031927034, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 30.6543358205383, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.7753340756715, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.3649515046225, 6)

    def test_ea_adc3_frozen(self):

        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.1741689078419142, 6)

        myadc_fr.method_type = "ea"
        e,v,p,x = myadc_fr.kernel(nroots=3)
        myadc_fr.analyze()

        self.assertAlmostEqual(e[0], -0.04496243643090245, 6)
        self.assertAlmostEqual(e[1],  0.0300414467534960, 6)
        self.assertAlmostEqual(e[2],  0.0315399172709854, 6)

        self.assertAlmostEqual(p[0], 0.8722200055138173, 6)
        self.assertAlmostEqual(p[1], 0.9927113317116677, 6)
        self.assertAlmostEqual(p[2], 0.9767596218115034, 6)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 30.65748194797707, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 58.77465475399459, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 59.36539662826338, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods for open-shell molecule")
    unittest.main()
