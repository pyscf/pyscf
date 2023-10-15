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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.symm import addons

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = 'cc-pvdz',
        symmetry = 1,
    )

    mf = scf.RHF(mol)
    mf.scf()

def tearDownModule():
    global mol, mf
    del mol, mf


class KnowValues(unittest.TestCase):
    def test_label_orb_symm(self):
        l = addons.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
        lab0 = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2',
                'A1', 'A1', 'B1', 'B2', 'A1', 'A2', 'B1', 'A1',
                'B2', 'B2', 'A1', 'B1', 'A2', 'A1', 'A1', 'B2']
        self.assertEqual(list(l), lab0)

    def test_symmetrize_orb(self):
        c = addons.symmetrize_orb(mol, mf.mo_coeff)
        self.assertTrue(numpy.allclose(c, mf.mo_coeff))
        numpy.random.seed(1)
        c = addons.symmetrize_orb(mol, numpy.random.random((mf.mo_coeff.shape)))
        self.assertAlmostEqual(numpy.linalg.norm(c), 10.163677602612152)

    def test_symmetrize_space(self):
        from pyscf import gto, symm, scf
        mol = gto.M(atom = 'C  0  0  0; H  1  1  1; H -1 -1  1; H  1 -1 -1; H -1  1 -1',
                    basis = 'sto3g', verbose=0)
        mf = scf.RHF(mol).run()
        mol.build(0, 0, symmetry='D2')
        mo = symm.symmetrize_space(mol, mf.mo_coeff)
        irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        #self.assertEqual(list(irreps), ['A','A','B1','B3','B2','A','B1','B3','B2'])
        self.assertEqual([x[0] for x in irreps], ['A','A','B','B','B','A','B','B','B'])

    def test_route(self):
        orbsym = [0, 3, 0, 2, 5, 6]
        res = addons.route(7, 3, orbsym)
        self.assertEqual(res, [0, 3, 4])


if __name__ == "__main__":
    print("Full Tests for symm.addons")
    unittest.main()
