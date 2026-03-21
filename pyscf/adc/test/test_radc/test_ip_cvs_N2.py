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
    global mol, mf, myadc
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

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        myadc.ncvs = 2
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        e,v,p,x = myadc.kernel(nroots=2)

        self.assertAlmostEqual(e[0], 15.12281031796864, 6)
        self.assertAlmostEqual(e[1], 15.12611217935994, 6)

        self.assertAlmostEqual(p[0], 1.54262807973040, 6)
        self.assertAlmostEqual(p[1], 1.54152768244107, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 34.77893499453598, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 34.78433184466488, 6)

    def test_ip_adc2x(self):

        myadc.ncvs = 2
        myadc.method = "adc(2)-x"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 15.10850837488190, 6)
        self.assertAlmostEqual(e[1], 15.11180785851825, 6)
        self.assertAlmostEqual(e[2], 15.74525610353200, 6)

        self.assertAlmostEqual(p[0], 1.51596080565362, 6)
        self.assertAlmostEqual(p[1], 1.51447639333099, 6)
        self.assertAlmostEqual(p[2], 0.00000030441510, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 34.74538223874991, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 34.75067998057301, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 36.30183742247967, 6)

    def test_ip_adc3(self):

        myadc.ncvs = 2
        myadc.method = "adc(3)"
        myadc.method_type = "ip"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.31694173142858517 , 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 15.28014264694833, 6)
        self.assertAlmostEqual(e[1], 15.28358689153576, 6)
        self.assertAlmostEqual(e[2], 15.74525610336261, 6)

        self.assertAlmostEqual(p[0], 1.64260781902935, 6)
        self.assertAlmostEqual(p[1], 1.64123055314380, 6)
        self.assertAlmostEqual(p[2], 0.00000030441505, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 34.73228837943413, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 34.73668157654031, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 36.30183742761254, 6)

if __name__ == "__main__":
    print("IP calculations for different RADC methods for nitrogen molecule")
    unittest.main()
