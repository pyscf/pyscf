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
    global mol, mf, myadc
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

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf).density_fit('cc-pvdz-ri')
    myadc.conv_tol = 1e-12
    myadc.tol_residual = 1e-6
    myadc.max_memory = 1

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

def rdms_test(dm):
    r2_int = mol.intor('int1e_r2')
    dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm, mf.mo_coeff.conj())
    r2 = np.einsum('pq,pq->',r2_int,dm_ao)
    return r2

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.2718672171, 6)
        self.assertAlmostEqual(e[1],0.2964651998, 6)
        self.assertAlmostEqual(e[2],0.3575504337, 6)
        self.assertAlmostEqual(e[3],0.3721878511, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.02770956, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.00000000, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.2558563365, 6)
        self.assertAlmostEqual(e[1],0.2787834140, 6)
        self.assertAlmostEqual(e[2],0.3428343617, 6)
        self.assertAlmostEqual(e[3],0.3560598866, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.02542338, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.00000000, 6)

    def test_ee_adc2x_cis(self):
        myadc.method = "adc(2)-x"

        e,v,p,x = myadc.kernel(nroots=4, guess="cis")

        self.assertAlmostEqual(e[0],0.2558563365, 6)
        self.assertAlmostEqual(e[1],0.2787834140, 6)
        self.assertAlmostEqual(e[2],0.3428343617, 6)
        self.assertAlmostEqual(e[3],0.3560598866, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.02542337, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.00000000, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.2800031574, 6)
        self.assertAlmostEqual(e[1],0.3046897911, 6)
        self.assertAlmostEqual(e[2],0.3634812968, 6)
        self.assertAlmostEqual(e[3],0.3787870030, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.02699877, 6)
        self.assertAlmostEqual(p[2],0.00000000, 6)
        self.assertAlmostEqual(p[3],0.00000000, 6)


if __name__ == "__main__":
    print("EE calculations for different ADC methods for water molecule")
    unittest.main()
