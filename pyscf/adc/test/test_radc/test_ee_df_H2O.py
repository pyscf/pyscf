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
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc, myadc_fr
    mol = gto.Mole()
    r = 0.957492
    x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
    y = r * math.cos(104.468205* math.pi/(2 * 180.0))
    mol.atom = [
        ['O', (0., 0.    , 0)],
        ['H', (0., -x, y)],
        ['H', (0., x , y)],]
    mol.basis = {'H': 'cc-pVDZ',
                 'O': 'cc-pVDZ',}
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc_fr = adc.ADC(mf,frozen=1).density_fit('cc-pvdz-ri')

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e,-0.20397003815111225, 6)
        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 19.073709731899555, 6)
        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.2964652641, 6)
        self.assertAlmostEqual(e[1],0.3721878388, 6)
        self.assertAlmostEqual(e[2],0.3931892768, 6)
        self.assertAlmostEqual(e[3],0.4706609107, 6)

        self.assertAlmostEqual(p[0], 0.0277095639387507, 6)
        self.assertAlmostEqual(p[1], 6.622301107235945e-29, 6)
        self.assertAlmostEqual(p[2], 0.0975602291208062, 6)
        self.assertAlmostEqual(p[3], 0.0737384878233883, 6)

        dm1_exc = np.array(myadc.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 26.90341078572877, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 28.35739435679268, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 26.81821211383783, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 28.58444958142687, 6)


    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.20397003815111225, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 19.073709731899555, 6)

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.2787834258, 6)
        self.assertAlmostEqual(e[1],0.3560598951, 6)
        self.assertAlmostEqual(e[2],0.3753963772, 6)
        self.assertAlmostEqual(e[3],0.4548887980, 6)

        self.assertAlmostEqual(p[0], 0.0254233730253111, 6)
        self.assertAlmostEqual(p[1], 4.222607388922207e-30, 6)
        self.assertAlmostEqual(p[2], 0.0916515608768791, 6)
        self.assertAlmostEqual(p[3], 0.0673955703391483, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 26.65090742696760, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 28.08419012517438, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 26.56548294196503, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 28.32546992531526, 6)


    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.21087156082147254, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs,19.043616749195607, 6)

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3046898037, 6)
        self.assertAlmostEqual(e[1],0.3787870164, 6)
        self.assertAlmostEqual(e[2],0.4017031629, 6)
        self.assertAlmostEqual(e[3],0.4769618372, 6)

        self.assertAlmostEqual(p[0], 0.02699877, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.09683797, 6)
        self.assertAlmostEqual(p[3], 0.07672429, 6)

        dm1_exc = np.array(myadcee.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 26.593891990682202, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 27.794208811985772, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 26.528184463439413, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 28.073824289908526, 6)


    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ee"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()
        self.assertAlmostEqual(e, -0.20874148305225715, 6)

        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs,19.04396573133547, 6)

        myadcee_fr = adc.radc_ee.RADCEE(myadc_fr)
        e,v,p,x = myadcee_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3045996449682784, 6)
        self.assertAlmostEqual(e[1],0.3786943953781337, 6)
        self.assertAlmostEqual(e[2],0.4016489362006092, 6)
        self.assertAlmostEqual(e[3],0.4769191360661929, 6)

        self.assertAlmostEqual(p[0], 0.02699001, 6)
        self.assertAlmostEqual(p[1], 0.00000000, 6)
        self.assertAlmostEqual(p[2], 0.09687069, 6)
        self.assertAlmostEqual(p[3], 0.07683850, 6)

        dm1_exc = np.array(myadcee_fr.make_rdm1())
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 26.59468632321121, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 27.79476409013939, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 26.52885696127928, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[3]), 28.07495816702845, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for water molecule")
    unittest.main()
