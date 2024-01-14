#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
from functools import reduce

from pyscf import scf
from pyscf import gto
from pyscf import cc
from pyscf.cc import ccsd
from pyscf.cc import addons

def setUpModule():
    global mol, mf1, gmf, myrcc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf1 = scf.RHF(mol).run(conv_tol=1e-12)
    gmf = scf.addons.convert_to_ghf(mf1)
    myrcc = ccsd.CCSD(mf1).run()

def tearDownModule():
    global mol, mf1, gmf, myrcc
    mol.stdout.close()
    del mol, mf1, gmf, myrcc

class KnownValues(unittest.TestCase):
    def test_spin2spatial(self):
        t1g = addons.spatial2spin(myrcc.t1)
        t2g = addons.spatial2spin(myrcc.t2)
        orbspin = gmf.mo_coeff.orbspin
        t1a, t1b = addons.spin2spatial(t1g, orbspin)
        t2aa, t2ab, t2bb = addons.spin2spatial(t2g, orbspin)
        self.assertAlmostEqual(abs(myrcc.t1 - t1a).max(), 0, 12)
        self.assertAlmostEqual(abs(myrcc.t2 - t2ab).max(), 0, 12)

        self.assertAlmostEqual(abs(t1g - addons.spatial2spin((t1a,t1b), orbspin)).max(), 0, 12)
        self.assertAlmostEqual(abs(t2g - addons.spatial2spin((t2aa,t2ab,t2bb), orbspin)).max(), 0, 12)

    def test_convert_to_uccsd(self):
        myucc = addons.convert_to_uccsd(myrcc)
        myucc = addons.convert_to_uccsd(myucc)
        self.assertTrue(myucc.t1[0].shape, (5,8))
        self.assertTrue(myucc.t1[1].shape, (5,8))
        self.assertTrue(myucc.t2[0].shape, (5,5,8,8))
        self.assertTrue(myucc.t2[1].shape, (5,5,8,8))
        self.assertTrue(myucc.t2[2].shape, (5,5,8,8))

    def test_convert_to_gccsd(self):
        mygcc = addons.convert_to_uccsd(myrcc)
        mygcc = addons.convert_to_gccsd(myrcc)
        self.assertTrue(mygcc.t1.shape, (10,16))
        self.assertTrue(mygcc.t2.shape, (10,10,16,16))

        myucc = addons.convert_to_uccsd(myrcc)
        mygcc = addons.convert_to_gccsd(myucc)
        self.assertTrue(mygcc.t1.shape, (10,16))
        self.assertTrue(mygcc.t2.shape, (10,10,16,16))

        mygcc = addons.convert_to_gccsd(cc.GCCSD(gmf))
        self.assertTrue(isinstance(mygcc, cc.gccsd.GCCSD))

    def test_bccd_kernel_(self):
        mybcc = addons.bccd_kernel_(myrcc)
        self.assertAlmostEqual(abs(mybcc.t1).max(), 0, 4)
        e_r = mybcc.e_tot

        myucc = addons.convert_to_uccsd(myrcc)
        mybcc = addons.bccd_kernel_(myucc)
        e_u = mybcc.e_tot
        self.assertAlmostEqual(abs(e_u - e_r), 0, 6)

        mygcc = addons.convert_to_gccsd(myrcc)
        mybcc = addons.bccd_kernel_(mygcc)
        e_g = mybcc.e_tot
        self.assertAlmostEqual(abs(e_g - e_r), 0, 6)

if __name__ == "__main__":
    print("Tests for addons")
    unittest.main()
