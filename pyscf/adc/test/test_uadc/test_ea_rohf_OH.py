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
from pyscf.adc.uadc_ea import get_spin_square

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

    def test_ea_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.003780234810081532, 6)
        self.assertAlmostEqual(e[1],0.17058485234819104, 6)
        self.assertAlmostEqual(e[2],0.18154337701234663, 6)
        self.assertAlmostEqual(e[3],0.7427685230510422, 6)

        self.assertAlmostEqual(p[0],0.933265514768101, 6)
        self.assertAlmostEqual(p[1],0.9865431694081672, 6)
        self.assertAlmostEqual(p[2],0.986588415332792, 6)
        self.assertAlmostEqual(p[3],0.963244512572626, 6)

        self.assertAlmostEqual(spin[0],0.00021429815258944274 , 5)
        self.assertAlmostEqual(spin[1],2.0000680023894866 , 5)
        self.assertAlmostEqual(spin[2],1.0257568144531723 , 5)
        self.assertAlmostEqual(spin[3],2.0001093449259613 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 19.112209511951495, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 27.296581376973368, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 27.415593738209427, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 22.160745013225895, 6)

    def test_ea_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],-0.004634998573098005, 6)
        self.assertAlmostEqual(e[1],0.16767337995443382, 6)
        self.assertAlmostEqual(e[2],0.17082020119628544, 6)
        self.assertAlmostEqual(e[3],0.1910533292422809, 6)

        self.assertAlmostEqual(p[0],0.9137563082891093, 6)
        self.assertAlmostEqual(p[1],0.9829803957297469, 6)
        self.assertAlmostEqual(p[2],0.6008915792230451, 6)
        self.assertAlmostEqual(p[3],0.3825943669222823, 6)

        self.assertAlmostEqual(spin[0],0.0011415294261380993 , 5)
        self.assertAlmostEqual(spin[1],2.0000550955618155 , 5)
        self.assertAlmostEqual(spin[2],1.9726808482518479 , 5)
        self.assertAlmostEqual(spin[3],0.03360282614627241 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 19.525233551771827, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 27.292961648302146, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 27.299245046001737, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 27.507055979121375, 6)

    def test_ea_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.032785195446375655, 6)
        self.assertAlmostEqual(e[1],0.17154743244595053, 6)
        self.assertAlmostEqual(e[2],0.17266305852205646, 6)
        self.assertAlmostEqual(e[3],0.19275455987078255, 6)

        self.assertAlmostEqual(p[0],0.9355025187631677, 6)
        self.assertAlmostEqual(p[1],0.9838701145298617, 6)
        self.assertAlmostEqual(p[2],0.5167687292173684, 6)
        self.assertAlmostEqual(p[3],0.4675725281631329, 6)

        self.assertAlmostEqual(spin[0],0.0010460235403808582 , 5)
        self.assertAlmostEqual(spin[1],2.000018101185584 , 5)
        self.assertAlmostEqual(spin[2],1.9913741862394203 , 5)
        self.assertAlmostEqual(spin[3],0.015169524607833829 , 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 19.343396737305003, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 27.343369046061916, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 27.32167753850262, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 27.524547831815973, 6)

if __name__ == "__main__":
    print("EA calculations for different ADC methods for OH molecule with ROHF reference")
    unittest.main()
