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

        self.assertAlmostEqual(p[0],0.00330261, 6)
        self.assertAlmostEqual(p[1],0.00330261, 6)
        self.assertAlmostEqual(p[2],0.03418049, 6)
        self.assertAlmostEqual(p[3],0.00247057, 6)

        self.assertAlmostEqual(spin[0],0.75455411 , 5)
        self.assertAlmostEqual(spin[1],0.75455411 , 5)
        self.assertAlmostEqual(spin[2],0.76650339 , 5)
        self.assertAlmostEqual(spin[3],2.83809698 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.35392455121452, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.35392455121458, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.52866264346929, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.49520967898046, 4)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0163448011, 6)
        self.assertAlmostEqual(e[1],0.0163448011, 6)
        self.assertAlmostEqual(e[2],0.0529820632, 6)
        self.assertAlmostEqual(e[3],0.1763225747, 6)

        self.assertAlmostEqual(p[0],0.00089089, 6)
        self.assertAlmostEqual(p[1],0.00089089 , 6)
        self.assertAlmostEqual(p[2],0.01672559 , 6)
        self.assertAlmostEqual(p[3],0.00080659 , 6)

        self.assertAlmostEqual(spin[0],0.75473533 , 5)
        self.assertAlmostEqual(spin[1],0.75473533 , 5)
        self.assertAlmostEqual(spin[2],0.76073466 , 5)
        self.assertAlmostEqual(spin[3],3.30472186 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.07688739953042, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.07688739953043, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.26948404959528, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.64198712678937, 4)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362451866, 6)
        self.assertAlmostEqual(e[1],0.0362451866, 6)
        self.assertAlmostEqual(e[2],0.1200323452, 6)
        self.assertAlmostEqual(e[3],0.1747553376, 6)

        self.assertAlmostEqual(p[0],0.00198999, 6)
        self.assertAlmostEqual(p[1],0.00198999, 6)
        self.assertAlmostEqual(p[2],0.02121277, 6)
        self.assertAlmostEqual(p[3],0.00135393, 6)

        self.assertAlmostEqual(spin[0],0.75778253 , 5)
        self.assertAlmostEqual(spin[1],0.75778253 , 5)
        self.assertAlmostEqual(spin[2],0.79995569 , 5)
        self.assertAlmostEqual(spin[3],3.45054439 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.36461324627249, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.36461324627246, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.62484983454539, 4)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.68897500426092, 4)

    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"

        myadc_fr.method_type = "ee"
        e,v,p,x = myadc_fr.kernel(nroots=4)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.0362186495, 6)
        self.assertAlmostEqual(e[1],0.0362186495, 6)
        self.assertAlmostEqual(e[2],0.1200847879, 6)
        self.assertAlmostEqual(e[3],0.1747766797, 6)

        self.assertAlmostEqual(p[0],0.00198969, 6)
        self.assertAlmostEqual(p[1],0.00198969, 6)
        self.assertAlmostEqual(p[2],0.02122918, 6)
        self.assertAlmostEqual(p[3],0.00135661, 6)

        self.assertAlmostEqual(spin[0],0.75778015 , 5)
        self.assertAlmostEqual(spin[1],0.75778015 , 5)
        self.assertAlmostEqual(spin[2],0.80003695 , 5)
        self.assertAlmostEqual(spin[3],3.45059815 , 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][0],dm1_exc[1][0]), 41.36395058407432, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][1],dm1_exc[1][1]), 41.36395058407432, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][2],dm1_exc[1][2]), 40.62433483846739, 4)
        self.assertAlmostEqual(rdms_test_fr(dm1_exc[0][3],dm1_exc[1][3]), 40.68872279656992, 4)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
