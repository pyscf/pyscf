#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import scf
from pyscf.scf import hirshfeld

def setUpModule():
    global mol
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                basis='6-31g*', verbose=0)

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_hirshfeld(self):
        mf = scf.RHF(mol).run(conv_tol=1e-11)
        q = mf.hirshfeld_charges(verbose=0)
        self.assertAlmostEqual(abs(q - [-0.3559771, 0.1779885, 0.1779885]).max(), 0, 6)
        self.assertAlmostEqual(q.sum(), 0, 6)

        # A closed-shell density gives the same charges with UHF and GHF
        for mf1 in (scf.UHF(mol), scf.GHF(mol)):
            q1 = mf1.run(conv_tol=1e-11).hirshfeld_charges(verbose=0)
            self.assertAlmostEqual(abs(q1 - q).max(), 0, 6)

    def test_hirshfeld_open_shell(self):
        ion = mol.copy()
        ion.charge = 1
        ion.spin = 1
        ion.build()
        for mf in (scf.UHF(ion), scf.ROHF(ion), scf.GHF(ion)):
            q = mf.run().hirshfeld_charges(verbose=0)
            self.assertAlmostEqual(q.sum(), 1, 6)

    def test_hirshfeld_separated_fragments(self):
        # Far apart, each fragment keeps its own charge
        mol1 = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587; Li 0 0 40',
                     basis='6-31g*', charge=1, verbose=0)
        q = scf.RHF(mol1).run().hirshfeld_charges(verbose=0)
        self.assertAlmostEqual(q[:3].sum(), 0, 6)
        self.assertAlmostEqual(q[3], 1, 6)

    def test_cm5(self):
        # Reference from CM5PAC (version 2015) of Marenich, Jerome, Cramer and
        # Truhlar, given the same geometry and Hirshfeld charges. The atoms
        # cover the atomwise parameters and all the pairwise ones.
        mol1 = gto.M(atom='''
            C 0.000 0.000 0.000; N 1.330 0.120 0.050; O -0.650 1.050 0.100
            H -0.520 -0.950 -0.080; H 1.880 -0.700 0.020; S 2.250 1.600 0.300
            F 3.500 1.400 -0.600; Cl -1.800 -1.900 1.200; C 1.100 3.000 0.500
            H 1.600 3.900 0.800; O 0.200 3.200 -0.600; N -2.900 0.400 -1.100
            H -3.500 1.100 -1.500''', basis='sto-3g', verbose=0)
        numpy.random.seed(7)
        qh = numpy.random.uniform(-.6, .6, mol1.natm)
        ref = [-0.450004, -0.002435, -0.118675, 0.350151, 0.779710, 0.084359,
               -0.005952, -0.520257, -0.293181, 0.064496, 0.182410, 0.173721,
               0.043645]
        q = hirshfeld.cm5_charges(mol1, hirshfeld=qh, verbose=0)
        self.assertAlmostEqual(abs(q - ref).max(), 0, 6)

        mf = scf.RHF(mol).run(conv_tol=1e-11)
        q = mf.cm5_charges(verbose=0)
        self.assertAlmostEqual(abs(q - [-0.6918981, 0.3459490, 0.3459490]).max(), 0, 6)
        self.assertAlmostEqual(q.sum(), 0, 6)


if __name__ == "__main__":
    print("Tests for Hirshfeld and CM5 charges")
    unittest.main()
