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
#         Ning-Yuan Chen <cny003@outlook.com>
#

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ip import get_spin_square as spin_square_ip
from pyscf.adc.uadc_ea import get_spin_square as spin_square_ea
from pyscf import df

def setUpModule():
    global mol, mf, myadc
    r = 0.969286393
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0.    , -r/2   )],
        ['H', (0., 0.    ,  r/2)],]
    mol.basis = {'O':'cc-pvdz',
                 'H':'cc-pvdz'}
    mol.verbose = 0
    mol.symmetry = False
    mol.spin  = 1
    mol.build()
    mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):


    def test_hf_dfgs(self):

        mf = scf.UHF(mol).run()
        myadc = adc.ADC(mf)
        myadc.conv_tol = 1e-12
        myadc.tol_residual = 1e-6
        myadc.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.150979874, 6)


    def test_dfhs_dfgs(self):

        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.15094533756, 6)


    def test_ea_dfadc3(self):

        mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.kernel()
        myadc.with_df = df.DF(mol, auxbasis='cc-pvdz-ri')
        myadc.max_memory = 1
        myadc.method = "adc(3)"
        myadc.method_type = "ea"

        e,v,p,x = myadc.kernel(nroots=4)
        spin = spin_square_ea(myadc._adc_es)[0]

        self.assertAlmostEqual(e[0], 0.03349588, 6)
        self.assertAlmostEqual(e[1], 0.17178726, 6)
        self.assertAlmostEqual(e[2], 0.17734579, 6)
        self.assertAlmostEqual(e[3], 0.20135255, 6)

        self.assertAlmostEqual(p[0], 0.9364865, 6)
        self.assertAlmostEqual(p[1], 0.9840609990091582, 6)
        self.assertAlmostEqual(p[2], 0.7760193597284754, 6)
        self.assertAlmostEqual(p[3], 0.20825099009739692, 6)

        self.assertAlmostEqual(spin[0],0.00154661 , 5)
        self.assertAlmostEqual(spin[1],1.99998143 , 5)
        self.assertAlmostEqual(spin[2],1.80976643 , 5)
        self.assertAlmostEqual(spin[3],0.20293818 , 5)


    def test_ip_dfadc3_dif_aux_basis(self):

        mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.kernel()
        myadc.with_df = df.DF(mol, auxbasis='aug-cc-pvdz-ri')
        myadc.max_memory = 1
        myadc.method = "adc(3)"
        myadc.method_type = "ip"

        e,v,p,x = myadc.kernel(nroots=3)
        spin = spin_square_ip(myadc._adc_es)[0]
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.16330973, 6)

        self.assertAlmostEqual(e[0], 0.45707428, 6)
        self.assertAlmostEqual(e[1], 0.46818375, 6)
        self.assertAlmostEqual(e[2], 0.55652918, 6)

        self.assertAlmostEqual(p[0], 0.9387552064029395, 6)
        self.assertAlmostEqual(p[1], 0.5869577478677291, 6)
        self.assertAlmostEqual(p[2], 0.3511163890273265, 6)

        self.assertAlmostEqual(spin[0],2.00017692 , 5)
        self.assertAlmostEqual(spin[1],1.94783684 , 5)
        self.assertAlmostEqual(spin[2],0.06857549 , 5)


    def test_hf_dfadc3_ip(self):

        mf = scf.UHF(mol).run()
        myadc = adc.ADC(mf)
        myadc.conv_tol = 1e-12
        myadc.tol_residual = 1e-6
        myadc.with_df = df.DF(mol, auxbasis='aug-cc-pvdz-ri')
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=3)
        spin = spin_square_ip(myadc._adc_es)[0]
        myadc.analyze()
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.1633223874, 6)

        self.assertAlmostEqual(e[0], 0.45707376, 6)
        self.assertAlmostEqual(e[1], 0.46818480, 6)
        self.assertAlmostEqual(e[2], 0.55652975, 6)

        self.assertAlmostEqual(p[0], 0.9387505395407538, 6)
        self.assertAlmostEqual(p[1], 0.5869560351750759, 6)
        self.assertAlmostEqual(p[2], 0.35111354373738396 ,6)

        self.assertAlmostEqual(spin[0],2.00017672 , 5)
        self.assertAlmostEqual(spin[1],1.94783373 , 5)
        self.assertAlmostEqual(spin[2],0.06857857 , 5)

if __name__ == "__main__":
    print("DF-ADC calculations for different UADC methods for OH")
    unittest.main()
