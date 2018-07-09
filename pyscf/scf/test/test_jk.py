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

import copy
import numpy
import unittest
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.scf import jk

mol = gto.M(
    verbose = 7,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = '631g',
    cart = True,
)
mf = scf.RHF(mol).run(conv_tol=1e-10)

def tearDownModule():
    global mol, mf
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_range_separated_Coulomb(self):
        '''test range-separated Coulomb'''
        with mol.with_range_coulomb(0.2):
            dm = mf.make_rdm1()
            vk0 = jk.get_jk(mol, dm, 'ijkl,jk->il', hermi=0)
            vk1 = jk.get_jk(mol, (dm,dm), ['ijkl,jk->il','ijkl,li->kj'], hermi=1)
            self.assertAlmostEqual(abs(vk1[0]-vk0).max(), 0, 9)
            self.assertAlmostEqual(abs(vk1[1]-vk0).max(), 0, 9)
            self.assertAlmostEqual(lib.finger(vk0), 0.87325708945599279, 9)

            vk = scf.hf.get_jk(mol, dm)[1]
            self.assertAlmostEqual(abs(vk-vk0).max(), 0, 12)
        vk = scf.hf.get_jk(mol, dm)[1]
        self.assertTrue(abs(vk-vk0).max() > 0.1)

    def test_shls_slice(self):
        dm = mf.make_rdm1()
        ao_loc = mol.ao_loc_nr()
        shls_slice = [0, 2, 1, 4, 2, 5, 0, 4]
        locs = [ao_loc[i] for i in shls_slice]
        i0, i1, j0, j1, k0, k1, l0, l1 = locs

        vs = jk.get_jk(mol, (dm[j0:j1,k0:k1], dm[l0:l1,k0:k1]),
                       ['ijkl,jk->il', 'ijkl,lk->ij'], hermi=0,
                       intor='int2e_ip1', shls_slice=shls_slice)
        self.assertEqual(vs[0].shape, (3,2,6))
        self.assertEqual(vs[1].shape, (3,2,5))

    def test_mols(self):
        pmol = copy.copy(mol)
        mols = (mol, pmol, pmol, mol)
        dm = mf.make_rdm1()
        vj0 = jk.get_jk(mols, dm, 'ijkl,lk->ij')
        vj1 = scf.hf.get_jk(mol, dm)[0]
        self.assertAlmostEqual(abs(vj1-vj0).max(), 0, 9)
        self.assertAlmostEqual(lib.finger(vj0), 28.36214139459754, 9)


if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()

