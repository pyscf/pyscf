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
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import grad
from pyscf.qmmm import itrf

def setUpModule():
    global mol
    mol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = ''' H                 -0.00000000   -0.000    0.
     H                 -0.00000000   -0.000    1.
     H                 -0.00000000   -0.82    0.
     H                 -0.91000000   -0.020    0.''',
        basis = 'cc-pvdz')

def tearDownModule():
    global mol

class KnowValues(unittest.TestCase):
    def test_energy(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges)
        self.assertAlmostEqual(mf.kernel(), 2.0042702433049024, 9)
        self.assertEqual(mf.undo_qmmm().__class__.__name__, 'RHF')

    def test_grad(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges).run(conv_tol=1e-10)
        hfg = itrf.mm_charge_grad(grad.RHF(mf), coords, charges).run()
        self.assertAlmostEqual(numpy.linalg.norm(hfg.de), 26.978089280783195, 6)

        mfs = mf.as_scanner()
        e1 = mfs('''
                 H              -0.00000000   -0.000    0.001
                 H                 -0.00000000   -0.000    1.
                 H                 -0.00000000   -0.82    0.
                 H                 -0.91000000   -0.020    0.
                 ''')
        e2 = mfs('''
                 H              -0.00000000   -0.000   -0.001
                 H                 -0.00000000   -0.000    1.
                 H                 -0.00000000   -0.82    0.
                 H                 -0.91000000   -0.020    0.
                 ''')
        self.assertAlmostEqual((e1 - e2)/0.002*lib.param.BOHR, hfg.de[0,2], 5)

        bak = pyscf.DEBUG
        pyscf.DEBUG = 1
        ref = hfg.get_hcore()
        pyscf.DEBUG = 0
        v = hfg.get_hcore()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 12)
        pyscf.DEBUG = bak

    def test_hcore_cart(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mol = gto.M(
            verbose = 0,
            atom = '''C    0.000  -0.300    0.2
                      Ne   0.310   0.820    0.1''',
            basis = 'cc-pvdz',
            cart = True)
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges)
        h = mf.get_hcore()
        self.assertAlmostEqual(lib.finger(h), -147.92831183612765, 9)

        h = mf.nuc_grad_method().get_hcore()
        self.assertEqual(h.shape, (3,30,30))
        self.assertAlmostEqual(lib.finger(h), -178.29768724184771, 9)

    def test_casci(self):
        mol = gto.Mole()
        mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                       H                 -0.00000000   -0.84695236    0.59109389
                       H                 -0.00000000    0.89830571    0.52404783 '''
        mol.verbose = 0
        mol.basis = '6-31g'
        mol.build()

        coords = [(0.5,0.6,0.1)]
        charges = [-0.1]
        mf = itrf.add_mm_charges(scf.RHF(mol), coords, charges).run(conv_tol=1e-10)
        mc = mcscf.CASCI(mf, 4, 4).run()
        self.assertAlmostEqual(mc.e_tot, -75.98156095286714, 8)

        mf = scf.RHF(mol).run(conv_tol=1e-10)
        mc = itrf.add_mm_charges(mcscf.CASCI(mf, 4, 4), coords, charges).run()
        self.assertAlmostEqual(mc.e_tot, -75.98156095286714, 7)

    def test_casscf(self):
        mol = gto.Mole()
        mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                       H                 -0.00000000   -0.84695236    0.59109389
                       H                 -0.00000000    0.89830571    0.52404783 '''
        mol.verbose = 0
        mol.basis = '6-31g'
        mol.build()

        coords = [(0.5,0.6,0.1)]
        charges = [-0.1]
        mf = itrf.add_mm_charges(scf.RHF(mol), coords, charges).run()
        mc = mcscf.CASSCF(mf, 4, 4).run()
        self.assertAlmostEqual(mc.e_tot, -76.0461574155984, 7)

        mf = scf.RHF(mol).run()
        mc = itrf.add_mm_charges(mcscf.CASSCF(mf, 4, 4), coords, charges).run()
        self.assertAlmostEqual(mc.e_tot, -76.0461574155984, 7)



if __name__ == "__main__":
    print("Full Tests for qmmm")
    unittest.main()
