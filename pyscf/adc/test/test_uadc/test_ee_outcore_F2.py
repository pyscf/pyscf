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
    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['F', (0., 0.    , -r/2   )],
        ['F', (0., 0.    ,  r/2)],]
    mol.basis = {'F':'cc-pvdz'}

    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol)
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

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3956243465, 6)
        self.assertAlmostEqual(e[1],0.3956243465, 6)
        self.assertAlmostEqual(e[2],0.4629576767, 6)
        self.assertAlmostEqual(e[3],0.4629576767, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00101630, 6)
        self.assertAlmostEqual(p[3],0.00101630, 6)

    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"

        myadc.method_type = "ee"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3739861956, 6)
        self.assertAlmostEqual(e[1],0.3739861956, 6)
        self.assertAlmostEqual(e[2],0.4400732595, 6)
        self.assertAlmostEqual(e[3],0.4400732595, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00101329, 6)
        self.assertAlmostEqual(p[3],0.00101329, 6)

    def test_ee_adc3(self):
        myadc.method = "adc(3)"

        myadc.method_type = "ee"
        myadc.max_memory = 20
        myadc.incore_complete = False
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3884663487, 6)
        self.assertAlmostEqual(e[1],0.3884663487, 6)
        self.assertAlmostEqual(e[2],0.4567656497, 6)
        self.assertAlmostEqual(e[3],0.4567656497, 6)

        self.assertAlmostEqual(p[0],0.00000000, 6)
        self.assertAlmostEqual(p[1],0.00000000, 6)
        self.assertAlmostEqual(p[2],0.00110811, 6)
        self.assertAlmostEqual(p[3],0.00110811, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for water molecule")
    unittest.main()
