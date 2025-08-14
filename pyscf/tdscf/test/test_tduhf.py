#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
from pyscf import lib, gto, scf, tdscf, symm

def setUpModule():
    global mol, mol1, mf, mf1
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.symmetry = True
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-10)

    mol1 = gto.Mole()
    mol1.verbose = 7
    mol1.output = '/dev/null'
    mol1.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol1.basis = '631g'
    mol1.spin = 2
    mol1.build()
    mf1 = scf.UHF(mol1).run(conv_tol=1e-10)

def tearDownModule():
    global mol, mol1, mf, mf1
    mol1.stdout.close()
    del mol, mol1, mf, mf1

class KnownValues(unittest.TestCase):
    def test_tda(self):
        td = mf.TDA()
        td.nstates = 5
        e = td.kernel()[0]
        ref = [11.01748568, 11.01748568, 11.90277134, 11.90277134, 13.16955369]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_tdhf(self):
        td = mf.TDHF()
        td.nstates = 5
        td.conv_tol = 1e-5
        e = td.kernel()[0]
        ref = [10.89192986, 10.89192986, 11.83487865, 11.83487865, 12.6344099]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_tda1(self):
        td = mf1.TDA()
        td.nstates = 5
        e = td.kernel()[0]
        ref = [3.32113736, 18.55977052, 21.01474222, 21.61501962, 25.0938973]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_tdhf1(self):
        td = mf1.TDHF()
        td.nstates = 4
        e = td.kernel()[0]
        ref = [3.31267103, 18.4954748, 20.84935404, 21.54808392]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_symmetry_init_guess(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True, verbose=0)
        mf = mol.UHF.run()
        td = mf.TDA().run(nstates=1)
        self.assertAlmostEqual(td.e[0], 0.14147328219131602, 7)
        mo_coeff = mf.mo_coeff
        mo_occa, mo_occb = mf.mo_occ
        orbsyma, orbsymb = scf.uhf_symm.get_orbsym(mol, mo_coeff)
        x_syma = symm.direct_prod(orbsyma[mo_occa==1], orbsyma[mo_occa==0], mol.groupname)
        x_symb = symm.direct_prod(orbsymb[mo_occb==1], orbsymb[mo_occb==0], mol.groupname)
        wfnsyma = tdscf.rhf._analyze_wfnsym(td, x_syma, td.xy[0][0][0])
        wfnsymb = tdscf.rhf._analyze_wfnsym(td, x_symb, td.xy[0][0][1])
        self.assertAlmostEqual(wfnsyma, 'A1u')
        self.assertAlmostEqual(wfnsymb, 'A1u')

    def test_set_frozen(self):
        td = mf.TDA()
        td.frozen = 1
        mask = td.get_frozen_mask()
        self.assertEqual(mask[0].sum(), 10)
        self.assertEqual(mask[1].sum(), 10)
        td.set_frozen()
        mask = td.get_frozen_mask()
        self.assertEqual(mask[0].sum(), 10)
        self.assertEqual(mask[1].sum(), 10)
        td.frozen = [0,1]
        mask = td.get_frozen_mask()
        self.assertEqual(mask[0].sum(), 9)
        self.assertEqual(mask[1].sum(), 9)
        td.frozen = [1,9]
        mask = td.get_frozen_mask()
        self.assertEqual(mask[0].sum(), 9)
        self.assertEqual(mask[1].sum(), 9)
        td.frozen = [[0],[1,2]]
        mask = td.get_frozen_mask()
        self.assertEqual(mask[0].sum(), 10)
        self.assertEqual(mask[1].sum(), 9)

if __name__ == "__main__":
    print("Full Tests for uhf-TDA and uhf-TDHF")
    unittest.main()
