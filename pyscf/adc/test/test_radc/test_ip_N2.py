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
    myadc_fr = adc.ADC(mf,frozen=1)
    myadc_fr.conv_tol = 1e-12
    myadc_fr.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ip_adc2(self):

        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.365895203706856, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)

        self.assertAlmostEqual(e[0], 0.5434389910483670, 6)
        self.assertAlmostEqual(e[1], 0.6240296243595950, 6)
        self.assertAlmostEqual(e[2], 0.6240296243595956, 6)

        self.assertAlmostEqual(p[0], 1.7677415104906158, 6)
        self.assertAlmostEqual(p[1], 1.8191644533114988, 6)
        self.assertAlmostEqual(p[2], 1.8191644533114986, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 32.35717283141812, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 33.24996284608739, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 33.249962846087385, 6)

    def test_ip_adc2_oneroot(self):

        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.5434389910483670, 6)

        self.assertAlmostEqual(p[0], 1.767741510490615, 6)

    def test_ip_adc2x(self):

        myadc.method = "adc(2)-x"
        e, t_amp1, t_amp2 = myadc.kernel_gs()

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.32201692499346535, 6)

        self.assertAlmostEqual(e[0], 0.5405255360673243, 6)
        self.assertAlmostEqual(e[1], 0.6208026698756092, 6)
        self.assertAlmostEqual(e[2], 0.6208026698756107, 6)

        self.assertAlmostEqual(p[0], 1.7504601057892397, 6)
        self.assertAlmostEqual(p[1], 1.815230494782789, 6)
        self.assertAlmostEqual(p[2], 1.81523049478279, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 32.30087343971567, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 33.23195050337803, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 33.231950503378016, 6)

    def test_ip_adc3(self):

        myadc.method = "adc(3)"
        myadc.method_type = "ip"

        myadc.kernel_gs()
        dm1_gs = myadc.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.26407941243784, 6)

        myadcip = adc.radc_ip.RADCIP(myadc)
        e,v,p,x = myadcip.kernel(nroots=3)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.31694173142858517 , 6)

        self.assertAlmostEqual(e[0], 0.5667526829981027, 6)
        self.assertAlmostEqual(e[1], 0.6099995170092525, 6)
        self.assertAlmostEqual(e[2], 0.6099995170092529, 6)

        self.assertAlmostEqual(p[0], 1.816868663938332, 6)
        self.assertAlmostEqual(p[1], 1.8431203210181513, 6)
        self.assertAlmostEqual(p[2], 1.843120321018146, 6)

        dm1_exc = myadcip.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 32.3299819839068, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 33.48084048266182, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 33.480840482661705, 6)

    def test_ip_adc3_frozen(self):

        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ip"

        myadc_fr.kernel_gs()
        dm1_gs = myadc_fr.make_ref_rdm1()
        r2_gs = rdms_test(dm1_gs)
        self.assertAlmostEqual(r2_gs, 39.262265370699176, 6)

        myadcip_fr = adc.radc_ip.RADCIP(myadc_fr)
        e,v,p,x = myadcip_fr.kernel(nroots=3)
        e_corr = myadc_fr.e_corr

        self.assertAlmostEqual(e_corr, -0.3146743531878704 , 6)

        self.assertAlmostEqual(e[0], 0.566731666572698, 6)
        self.assertAlmostEqual(e[1], 0.609944679862951, 6)
        self.assertAlmostEqual(e[2], 0.609944679862952, 6)

        self.assertAlmostEqual(p[0], 1.8167313442987183, 6)
        self.assertAlmostEqual(p[1], 1.8430428145467466, 6)
        self.assertAlmostEqual(p[2], 1.8430428145467455, 6)

        dm1_exc = myadcip_fr.make_rdm1()
        self.assertAlmostEqual(rdms_test(dm1_exc[0]), 32.328662217021424, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[1]), 33.479565624230894, 6)
        self.assertAlmostEqual(rdms_test(dm1_exc[2]), 33.47956562423086, 6)

if __name__ == "__main__":
    print("IP calculations for different RADC methods for nitrogen molecule")
    unittest.main()
