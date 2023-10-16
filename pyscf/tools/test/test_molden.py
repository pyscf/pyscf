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
import tempfile
from pyscf import lib, gto, scf
from pyscf.tools import molden

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = '''
    H  0.0000000000   0.0000000000   0.0000000000
    F  0.0000000000   0.0000000000   0.9000000000
               '''
    mol.basis = 'ccpvdz'
    mol.symmetry = True
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol).run()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_dump_scf(self):
        ftmp = tempfile.NamedTemporaryFile()
        fname = ftmp.name
        molden.dump_scf(mf, fname)
        res = molden.read(fname)
        mo_coeff = res[2]
        self.assertAlmostEqual(abs(mf.mo_coeff-mo_coeff).max(), 0, 12)

    def test_dump_uhf(self):
        ftmp = tempfile.NamedTemporaryFile()
        fname = ftmp.name
        with lib.temporary_env(mol, spin=2, charge=2):
            mf = scf.UHF(mol).run()
            molden.dump_scf(mf, fname)
            res = molden.read(fname)
            mo_coeff = res[2]
            self.assertEqual(res[0].spin, 2)
            self.assertAlmostEqual(abs(mf.mo_coeff[0]-mo_coeff[0]).max(), 0, 12)
            self.assertAlmostEqual(abs(mf.mo_coeff[1]-mo_coeff[1]).max(), 0, 12)

    def test_dump_cartesian_gto_orbital(self):
        ftmp = tempfile.NamedTemporaryFile()
        fname = ftmp.name
        with lib.temporary_env(mol, cart=True, symmetry=False):
            mf = scf.UHF(mol).run()
            molden.dump_scf(mf, fname)

            res = molden.read(fname)
            mo_coeff = res[2]
            self.assertAlmostEqual(abs(mf.mo_coeff[0]-mo_coeff[0]).max(), 0, 12)
            self.assertAlmostEqual(abs(mf.mo_coeff[1]-mo_coeff[1]).max(), 0, 12)

    def test_dump_cartesian_gto_symm_orbital(self):
        ftmp = tempfile.NamedTemporaryFile()
        fname = ftmp.name

        pmol = mol.copy()
        pmol.cart = True
        pmol.build()
        mf = scf.RHF(pmol).run()
        molden.from_mo(pmol, fname, mf.mo_coeff)

        res = molden.read(fname)
        mo_coeff = res[2]
        self.assertAlmostEqual(abs(mf.mo_coeff-mo_coeff).max(), 0, 12)


if __name__ == "__main__":
    print("Full Tests for molden")
    unittest.main()
