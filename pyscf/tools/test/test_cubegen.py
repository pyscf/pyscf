#!/usr/bin/env python
# Copyright 2019 The PySCF Developers. All Rights Reserved.
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
import copy
import tempfile
from pyscf import lib, gto, scf
from pyscf.tools import cubegen

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = '''
    O 0.00000000,  0.000000,  0.119748
    H 0.00000000,  0.761561, -0.478993
    H 0.00000000, -0.761561, -0.478993 '''
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol).run()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_mep(self):
        ftmp = tempfile.NamedTemporaryFile()
        mep = cubegen.mep(mol, ftmp.name, mf.make_rdm1(),
                          nx=10, ny=10, nz=10)
        self.assertEqual(mep.shape, (10,10,10))
        self.assertAlmostEqual(lib.finger(mep), -0.3198103636180436, 9)

        mep = cubegen.mep(mol, ftmp.name, mf.make_rdm1(),
                          nx=10, ny=10, nz=10, resolution=0.5)
        self.assertEqual(mep.shape, (12,18,15))
        self.assertAlmostEqual(lib.finger(mep), -4.653995909548524, 9)

    def test_orb(self):
        ftmp = tempfile.NamedTemporaryFile()
        orb = cubegen.orbital(mol, ftmp.name, mf.mo_coeff[:,0],
                              nx=10, ny=10, nz=10)
        self.assertEqual(orb.shape, (10,10,10))
        self.assertAlmostEqual(lib.finger(orb), -0.11804191128016768, 9)

        orb = cubegen.orbital(mol, ftmp.name, mf.mo_coeff[:,0],
                              nx=10, ny=10, nz=10, resolution=0.5)
        self.assertEqual(orb.shape, (12,18,15))
        self.assertAlmostEqual(lib.finger(orb), -0.8591778390706646, 9)

        orb = cubegen.orbital(mol, ftmp.name, mf.mo_coeff[:,0],
                              nx=10, ny=1, nz=1)
        self.assertEqual(orb.shape, (10,1,1))
        self.assertAlmostEqual(lib.finger(orb), 6.921008881822988e-09, 9)


    def test_rho(self):
        ftmp = tempfile.NamedTemporaryFile()
        rho = cubegen.density(mol, ftmp.name, mf.make_rdm1(),
                              nx=10, ny=10, nz=10)
        self.assertEqual(rho.shape, (10,10,10))
        self.assertAlmostEqual(lib.finger(rho), -0.3740462814001553, 9)

        rho = cubegen.density(mol, ftmp.name, mf.make_rdm1(),
                              nx=10, ny=10, nz=10, resolution=0.5)
        self.assertEqual(rho.shape, (12,18,15))
        self.assertAlmostEqual(lib.finger(rho), -1.007950007160415, 9)

if __name__ == "__main__":
    print("Full Tests for molden")
    unittest.main()



