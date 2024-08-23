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
    mf = scf.RHF(mol).run(conv_tol=1e-10)

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_mep(self):
        with tempfile.NamedTemporaryFile() as ftmp:
            mep = cubegen.mep(mol, ftmp.name, mf.make_rdm1(),
                              nx=10, ny=10, nz=10)
            self.assertEqual(mep.shape, (10,10,10))
            self.assertAlmostEqual(lib.fp(mep), -0.3198103636180436, 5)

            mep = cubegen.mep(mol, ftmp.name, mf.make_rdm1(),
                              nx=10, ny=10, nz=10, resolution=0.5)
            self.assertEqual(mep.shape, (12,18,15))
            self.assertAlmostEqual(lib.fp(mep), -4.653995909548524, 5)

    def test_orb(self):
        with tempfile.NamedTemporaryFile() as ftmp:
            orb = cubegen.orbital(mol, ftmp.name, mf.mo_coeff[:,0],
                                  nx=10, ny=10, nz=10)
            self.assertEqual(orb.shape, (10,10,10))
            self.assertAlmostEqual(lib.fp(orb), -0.11804191128016768, 5)

            orb = cubegen.orbital(mol, ftmp.name, mf.mo_coeff[:,0],
                                  nx=10, ny=10, nz=10, resolution=0.5)
            self.assertEqual(orb.shape, (12,18,15))
            self.assertAlmostEqual(lib.fp(orb), -0.8591778390706646, 5)

            orb = cubegen.orbital(mol, ftmp.name, mf.mo_coeff[:,0],
                                  nx=10, ny=1, nz=1)
            self.assertEqual(orb.shape, (10,1,1))
            self.assertAlmostEqual(lib.fp(orb), 6.921008881822988e-09, 5)


    def test_rho(self):
        with tempfile.NamedTemporaryFile() as ftmp:
            rho = cubegen.density(mol, ftmp.name, mf.make_rdm1(),
                                  nx=10, ny=10, nz=10)
            self.assertEqual(rho.shape, (10,10,10))
            self.assertAlmostEqual(lib.fp(rho), -0.3740462814001553, 5)

            rho = cubegen.density(mol, ftmp.name, mf.make_rdm1(),
                                  nx=10, ny=10, nz=10, resolution=0.5)
            self.assertEqual(rho.shape, (12,18,15))
            self.assertAlmostEqual(lib.fp(rho), -1.007950007160415, 5)

    def test_rho_with_pbc(self):
        from pyscf.pbc.gto import Cell
        cell = Cell()
        cell.unit = 'B'
        cell.atom = '''
        C  0.          0.          0.
        C  1.68506879  1.68506879  1.68506879
        '''
        cell.a = '''
        0.          3.37013758  3.37013758
        3.37013758  0.          3.37013758
        3.37013758  3.37013758  0.
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.mesh = [11]*3
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()
        mf = cell.RHF().run()
        with tempfile.NamedTemporaryFile() as ftmp:
            rho = cubegen.density(cell, ftmp.name, mf.make_rdm1(),
                                  nx=10, ny=10, nz=10)
            cc = cubegen.Cube(cell)
            self.assertEqual(rho.shape, (10,10,10))
            self.assertAlmostEqual(lib.fp(rho), -0.253781345652853, 5)

            rho1 = cc.read(ftmp.name)
            self.assertAlmostEqual(abs(rho1 - rho).max(), 0, 5)
            self.assertAlmostEqual(abs(cc.box - cell.lattice_vectors()).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for molden")
    unittest.main()
