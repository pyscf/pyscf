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
from pyscf import symm

def get_so(atoms, basis, cart=False):
    atoms = gto.mole.format_atom(atoms)
    gpname, origin, axes = symm.detect_symm(atoms)
    gpname, axes = symm.subgroup(gpname, axes)
    atoms = gto.mole.format_atom(atoms, origin, axes)
    try:
        mol = gto.M(atom=atoms, basis=basis)
    except RuntimeError:
        mol = gto.M(atom=atoms, basis=basis, spin=1)
    mol.cart = cart
    so = symm.basis.symm_adapted_basis(mol, gpname)[0]
    n = 0
    for c in so:
        if c.size > 0:
            n += c.shape[1]
    assert(n == mol.nao_nr())
    return n, so


class KnowValues(unittest.TestCase):
    def test_symm_orb_h2o(self):
        atoms = [['O' , (1. , 0.    , 0.   ,)],
                 [1   , (0. , -.757 , 0.587,)],
                 [1   , (0. , 0.757 , 0.587,)] ]
        basis = {'H': gto.basis.load('cc_pvqz', 'C'),
                 'O': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis  )[0], 165)
        self.assertEqual(get_so(atoms,basis,1)[0], 210)

    def test_symm_orb_d2h(self):
        atoms = [[1, (0., 0., 0.)],
                 [1, (1., 0., 0.)],
                 [1, (0., 1., 0.)],
                 [1, (0., 0., 1.)],
                 [1, (-1, 0., 0.)],
                 [1, (0.,-1., 0.)],
                 [1, (0., 0.,-1.)]]
        basis = {'H': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis  )[0], 385)
        self.assertEqual(get_so(atoms,basis,1)[0], 490)

    def test_symm_orb_c2v(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (0., 1., 0.)],
                 [1, (-2.,0.,-1.)],
                 [2, (0.,-1., 0.)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis)[0], 220)

    def test_symm_orb_c2h(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (0., 1., 0.)],
                 [1, (-1.,0.,-2.)],
                 [2, (0.,-1., 0.)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis  )[0], 220)
        self.assertEqual(get_so(atoms,basis,1)[0], 280)

        atoms = [[1, (1., 0., 1.)],
                 [1, (1., 0.,-1.)],
                 [2, (0., 0., 2.)],
                 [2, (2., 0.,-2.)],
                 [3, (1., 1., 0.)],
                 [3, (1.,-1., 0.)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),
                 'Li': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis  )[0], 330)
        self.assertEqual(get_so(atoms,basis,1)[0], 420)

    def test_symm_orb_d2(self):
        atoms = [[1, (1., 0., 1.)],
                 [1, (1., 0.,-1.)],
                 [2, (0., 0., 2.)],
                 [2, (2., 0., 2.)],
                 [2, (1., 1.,-2.)],
                 [2, (1.,-1.,-2.)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis  )[0], 330)
        self.assertEqual(get_so(atoms,basis,1)[0], 420)

    def test_symm_orb_ci(self):
        atoms = [[1, ( 1., 0., 0.)],
                 [2, ( 0., 1., 0.)],
                 [3, ( 0., 0., 1.)],
                 [4, ( .5, .5, .5)],
                 [1, (-1., 0., 0.)],
                 [2, ( 0.,-1., 0.)],
                 [3, ( 0., 0.,-1.)],
                 [4, (-.5,-.5,-.5)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),
                 'Li': gto.basis.load('cc_pvqz', 'C'),
                 'Be': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis)[0], 440)

    def test_symm_orb_cs(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (1., 0., 0.)],
                 [3, (2., 0.,-1.)],
                 [4, (0., 0., 1.)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),
                 'Li': gto.basis.load('cc_pvqz', 'C'),
                 'Be': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis)[0], 220)

    def test_symm_orb_c1(self):
        atoms = [[1, ( 1., 0., 0.)],
                 [2, ( 0., 1., 0.)],
                 [3, ( 0., 0., 1.)],
                 [4, ( .5, .5, .5)]]
        basis = {'H' : gto.basis.load('cc_pvqz', 'C'),
                 'He': gto.basis.load('cc_pvqz', 'C'),
                 'Li': gto.basis.load('cc_pvqz', 'C'),
                 'Be': gto.basis.load('cc_pvqz', 'C'),}
        self.assertEqual(get_so(atoms,basis)[0], 220)


if __name__ == "__main__":
    print("Full Tests symm.basis")
    unittest.main()
