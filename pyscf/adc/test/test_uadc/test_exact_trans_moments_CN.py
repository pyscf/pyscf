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
    global mol, mf, myadc, myadc_fr
    basis = 'cc-pVDZ'
    mol = gto.Mole()
    mol.atom = '''
        C 0.00000000 0.00000000 -1.18953886
        N 0.00000000 0.00000000 1.01938091
         '''
    mol.basis = {'C': basis,
                 'N': basis,}
    mol.unit = 'Bohr'
    mol.symmetry = "c2v"
    mol.spin = 1
    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=(1,1))

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

    def test_ee_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ee"
        myadc.approx_trans_moments = False
        myadc.compute_spin_square = True

        e,v,p,x = myadc.kernel(nroots=5)
        spin = get_spin_square(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0],0.07893932394039124, 6)
        self.assertAlmostEqual(e[1],0.07893932394039192, 6)
        self.assertAlmostEqual(e[2],0.13970852166254022, 6)
        self.assertAlmostEqual(e[3],0.255289367789753, 6)

        self.assertAlmostEqual(p[0],0.0040359940153666595, 6)
        self.assertAlmostEqual(p[1],0.0040359940153667, 6)
        self.assertAlmostEqual(p[2],0.022296930965621038, 6)
        self.assertAlmostEqual(p[3],0.005971269382491182, 6)

        self.assertAlmostEqual(spin[0],0.819034148183496, 5)
        self.assertAlmostEqual(spin[1],0.8190341481835013, 5)
        self.assertAlmostEqual(spin[2],0.9783306549346342, 5)
        self.assertAlmostEqual(spin[3],2.704355376298712, 5)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.645076886712786, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.64507688671283, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.82458614227809, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.62313965448439, 6)

    def test_ea_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ea"
        myadc.approx_trans_moments = False

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], -0.11238601773252498, 6)
        self.assertAlmostEqual(e[1],  0.1309866865161511, 6)
        self.assertAlmostEqual(e[2],  0.13098668651615408, 6)
        self.assertAlmostEqual(e[3],  0.16528522980819868, 6)

        self.assertAlmostEqual(p[0], 0.9223022854993761, 6)
        self.assertAlmostEqual(p[1], 0.9306703811763251, 6)
        self.assertAlmostEqual(p[2], 0.930670381176325, 6)
        self.assertAlmostEqual(p[3], 0.9418227151305963, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 48.15960412960568, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 47.7244308416808, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 47.72443084168075, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 48.21673619316982, 6)

    def test_ip_adc2(self):
        myadc.method = "adc(2)"
        myadc.method_type = "ip"
        myadc.approx_trans_moments = False

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4963840021918169, 6)
        self.assertAlmostEqual(e[1], 0.49638400219181966, 6)
        self.assertAlmostEqual(e[2], 0.5237162997024555, 6)
        self.assertAlmostEqual(e[3], 0.5237162997024575, 6)

        self.assertAlmostEqual(p[0], 0.902180909793373, 6)
        self.assertAlmostEqual(p[1], 0.9021809097933734, 6)
        self.assertAlmostEqual(p[2], 0.939513945794342, 6)
        self.assertAlmostEqual(p[3], 0.9395139457943412, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 33.21772069609978, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 33.21772069609979, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 33.17568036148569, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 33.17568036148567, 6)

    def test_ee_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.method_type = "ee"
        myadc_fr.approx_trans_moments = False
        myadc_fr.compute_spin_square = True

        e,v,p,x = myadc_fr.kernel(nroots=5)
        spin = get_spin_square(myadc_fr._adc_es)[0]

        self.assertAlmostEqual(e[0],0.07892652538653322, 6)
        self.assertAlmostEqual(e[1],0.07892652538653487, 6)
        self.assertAlmostEqual(e[2],0.13982485859038327, 6)
        self.assertAlmostEqual(e[3],0.25530335531355575, 6)

        self.assertAlmostEqual(p[0],0.004036190574155908, 6)
        self.assertAlmostEqual(p[1],0.004036190574155983, 6)
        self.assertAlmostEqual(p[2],0.022321952811774536, 6)
        self.assertAlmostEqual(p[3],0.005958063811198265, 6)

        self.assertAlmostEqual(spin[0],0.819078366446389, 5)
        self.assertAlmostEqual(spin[1],0.8190783664463872, 5)
        self.assertAlmostEqual(spin[2],0.9784713231854472, 5)
        self.assertAlmostEqual(spin[3],2.704260909888104, 5)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 41.644160765733915, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 41.6441607657339, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 40.82341770942643, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 40.622277882500505, 6)

    def test_ea_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.method_type = "ea"
        myadc_fr.approx_trans_moments = False

        e,v,p,x = myadc_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], -0.11237494239711686, 6)
        self.assertAlmostEqual(e[1],  0.13105888047862704, 6)
        self.assertAlmostEqual(e[2],  0.1310588804786292, 6)
        self.assertAlmostEqual(e[3],  0.16529157724802437, 6)

        self.assertAlmostEqual(p[0], 0.9223111842983532, 6)
        self.assertAlmostEqual(p[1], 0.9306796659315683, 6)
        self.assertAlmostEqual(p[2], 0.9306796659315696, 6)
        self.assertAlmostEqual(p[3], 0.9418329500617146, 6)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 48.15849215956127, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 47.723741869844844, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 47.72374186984487, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 48.21568184514025, 6)

    def test_ip_adc2_frozen(self):
        myadc_fr.method = "adc(2)"
        myadc_fr.method_type = "ip"
        myadc_fr.approx_trans_moments = False

        e,v,p,x = myadc_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0], 0.4963773005814777, 6)
        self.assertAlmostEqual(e[1], 0.49637730058148033, 6)
        self.assertAlmostEqual(e[2], 0.5237016998217314, 6)
        self.assertAlmostEqual(e[3], 0.5237016998217328, 6)

        self.assertAlmostEqual(p[0], 0.9022051068469982, 6)
        self.assertAlmostEqual(p[1], 0.9022051068469977, 6)
        self.assertAlmostEqual(p[2], 0.9395185980730837, 6)
        self.assertAlmostEqual(p[3], 0.9395185980730828, 6)

        dm1_exc = np.array(myadc_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0][0],dm1_exc[1][0]), 33.21712052349486, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][1],dm1_exc[1][1]), 33.217120523494856, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][2],dm1_exc[1][2]), 33.17497417216622, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[0][3],dm1_exc[1][3]), 33.17497417216621, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for CN molecule")
    unittest.main()
