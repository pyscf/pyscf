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
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
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

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16402828164387906, 6)

        r2_int = mol.intor('int1e_r2')
        dm1_gs_a,dm1_gs_b = myadc.make_rdm1_ground()
        mo_coeff_a,mo_coeff_b = mf.mo_coeff
        dm1_gs_ao_a = np.einsum('pi,ij,qj->pq', mo_coeff_a, dm1_gs_a, mo_coeff_a.conj())
        dm1_gs_ao_b = np.einsum('pi,ij,qj->pq', mo_coeff_b, dm1_gs_b, mo_coeff_b.conj())
        r2_gs_a = np.einsum('pq,pq->',r2_int,dm1_gs_ao_a) 
        r2_gs_b = np.einsum('pq,pq->',r2_int,dm1_gs_ao_b) 
        self.assertAlmostEqual(r2_gs_a, 11.90325443138132, 6)
        self.assertAlmostEqual(r2_gs_b, 9.493284854541251, 6)

        dm1_ip_a,dm1_ip_b = myadc.make_rdm1_excited()
        dm1_ip_ao_a = np.einsum('pi,kij,qj->kpq', mo_coeff_a, dm1_ip_a, mo_coeff_a.conj())
        dm1_ip_ao_b = np.einsum('pi,kij,qj->kpq', mo_coeff_b, dm1_ip_b, mo_coeff_b.conj())
        r2_ip_a = np.einsum('pq,kpq->k',r2_int,dm1_ip_ao_a) 
        r2_ip_b = np.einsum('pq,kpq->k',r2_int,dm1_ip_ao_b) 

        self.assertAlmostEqual(e[0], 0.4342864327917968, 6)
        self.assertAlmostEqual(e[1], 0.47343844767816784, 6)
        self.assertAlmostEqual(e[2], 0.5805631452815511, 6)

        self.assertAlmostEqual(p[0], 0.9066975034860368, 6)
        self.assertAlmostEqual(p[1], 0.8987660491377468, 6)
        self.assertAlmostEqual(p[2], 0.9119655964285802, 6)

        self.assertAlmostEqual(r2_ip_a[0], 10.54712633, 6)
        self.assertAlmostEqual(r2_ip_a[1], 8.12486002, 6)
        self.assertAlmostEqual(r2_ip_a[2], 10.72181645, 6)
        self.assertAlmostEqual(r2_ip_b[0], 5.79620476, 6)
        self.assertAlmostEqual(r2_ip_b[1], 8.29867476, 6)
        self.assertAlmostEqual(r2_ip_b[2], 5.39662535, 6)

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

    def test_ip_adc3_oneroot(self):

        myadc.method = "adc(3)"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.4794423247368058, 6)

        self.assertAlmostEqual(p[0], 0.9282869467221032, 6)

if __name__ == "__main__":
    print("IP calculations for different ADC methods for open-shell molecule")
    unittest.main()
