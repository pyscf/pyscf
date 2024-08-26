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

    def test_basis_not_sorted(self):
        with tempfile.NamedTemporaryFile('w') as ftmp:
            ftmp.write('''\
[Molden Format]
made by pyscf v[2.4.0]
[Atoms] (AU)
Ne   1   10     0.00000000000000     0.00000000000000     0.00000000000000
[GTO]
1 0
 s    8 1.00
                 17880  0.00073769817327139
                  2683  0.0056746782244738
                 611.5   0.028871187450674
                 173.5    0.10849560938601
                 56.64    0.29078802505672
                 20.42    0.44814064476114
                  7.81    0.25792047270532
                 1.653   0.015056839544698
 s    8 1.00
                 17880  -0.00033214909943481
                  2683  -0.0026205019065875
                 611.5  -0.013009816761002
                 173.5  -0.053420003125961
                 56.64   -0.14716522424261
                 20.42   -0.33838075724805
                  7.81   -0.20670101921688
                 1.653     1.0950299234565
 s    1 1.00
                0.4869                   1
 p    3 1.00
                 28.39   0.066171986640049
                  6.27    0.34485329752845
                 1.695    0.73045763818875
 p    1 1.00
                0.4317                   1
 d    1 1.00
                 2.202                   1
 s    1 1.00
                     1                   1

[5d]
[7f]
[9g]
''')
            ftmp.flush()
            mol = molden.load(ftmp.name)[0]
        self.assertEqual(mol._bas[:,1].tolist(), [0, 0, 0, 1, 1, 2, 0])

if __name__ == "__main__":
    print("Full Tests for molden")
    unittest.main()
