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

    def test_ip_adc2(self):

        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387906, 6)

        self.assertAlmostEqual(e[0], 0.4342864327917968, 6)
        self.assertAlmostEqual(e[1], 0.47343844767816784, 6)
        self.assertAlmostEqual(e[2], 0.5805631452815511, 6)

        self.assertAlmostEqual(p[0], 0.9066975034860368, 6)
        self.assertAlmostEqual(p[1], 0.8987660491377468, 6)
        self.assertAlmostEqual(p[2], 0.9119655964285802, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.34333109687462, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.42353478294355, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 16.11844180343718, 6)

    def test_ip_adc2x(self):

        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.16402828164387906, 6)

        e,v,p,x = myadc.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.4389083582117278, 6)
        self.assertAlmostEqual(e[1], 0.45720829251439343, 6)
        self.assertAlmostEqual(e[2], 0.5588942056812034, 6)

        self.assertAlmostEqual(p[0], 0.9169548953028459, 6)
        self.assertAlmostEqual(p[1], 0.6997121885268642, 6)
        self.assertAlmostEqual(p[2], 0.212879313736106, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.44106224314214, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.70743635068417, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 17.23999948606712, 6)

    def test_ip_adc3(self):

        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.17616203329072194, 6)

        myadc.method_type = "ip"
        e,v,p,x = myadc.kernel(nroots=3)
        myadc.analyze()

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)
        self.assertAlmostEqual(e[1], 0.4872370596653387, 6)
        self.assertAlmostEqual(e[2], 0.5726961805214643, 6)

        self.assertAlmostEqual(p[0], 0.9282869467221032, 6)
        self.assertAlmostEqual(p[1], 0.5188529241094367, 6)
        self.assertAlmostEqual(p[2], 0.40655844616580944, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.404206243396008, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.848816049838934, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 17.063649873323694, 6)

    def test_ip_adc3_frozen(self):

        myadc_fr.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.17416890784191427, 6)

        myadc_fr.method_type = "ip"
        e,v,p,x = myadc_fr.kernel(nroots=3)
        myadc_fr.analyze()

        self.assertAlmostEqual(e[0], 0.479335828293046, 6)
        self.assertAlmostEqual(e[1], 0.487239087759452, 6)
        self.assertAlmostEqual(e[2], 0.572673343025919, 6)

        self.assertAlmostEqual(p[0], 0.928283370231907, 6)
        self.assertAlmostEqual(p[1], 0.520023087174258, 6)
        self.assertAlmostEqual(p[2], 0.405384799660555, 6)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.404088067453547, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 16.847684757612303, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 17.064770798700994, 6)

    def test_ip_adc3_oneroot(self):

        myadc.method = "adc(3)"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)

        self.assertAlmostEqual(p[0], 0.9282869467221032, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 16.404206243395997, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
