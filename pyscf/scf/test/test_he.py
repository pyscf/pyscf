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
from pyscf import scf
from pyscf import gto
from pyscf import lib
from pyscf.scf import atom_hf, atom_ks

def setUpModule():
    global mol
    # for cgto
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom.extend([[2, (0.,0.,0.)], ])
    mol.basis = {"He": 'cc-pvdz'}
    mol.build()

def tearDownModule():
    global mol
    del mol

class KnownValues_NR(unittest.TestCase):
    """non-relativistic"""
    def test_fock_1e(self):
        rhf = scf.RHF(mol)
        h1e = rhf.get_hcore(mol)
        s1e = rhf.get_ovlp(mol)
        e, c = rhf.eig(h1e, s1e)
        self.assertAlmostEqual(e[0], -1.9936233377269388, 12)

    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-10
        self.assertAlmostEqual(rhf.scf(), -2.8551604772427379, 10)

    def test_nr_uhf(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-10
        self.assertAlmostEqual(uhf.scf(), -2.8551604772427379, 10)

#    def test_gaussian_nucmod(self):
#        gnuc = hf.gto.molinf.MoleInfo()
#        gnuc.verbose = 0
#        gnuc.output = "out_he"
#        gnuc.atom.extend([[2, (0.,0.,0.)], ])
#        gnuc.etb = {"He": { "max_l": 1, "s": (4, .4, 3.8), "p": (2, 1, 3.4)}}
#        gnuc.nucmod = {1:2}
#        gnuc.build()
#        rhf = scf.RHF(gnuc)
#        rhf.conv_tol = 1e-10
#        rhf.potential("coulomb")
#        self.assertAlmostEqual(rhf.scf(), -2.8447211759894566, 10)
#        # restore nucmod
#        mol.nucmod = {1:1}
#        mol.build()

    def test_atomic_ks(self):
        mol = gto.M(
            atom = [["N", (0. , 0., .5)],
                    ["N", (0. , 0.,-.5)] ],
            basis = {"N": '6-31g'}
        )
        result = atom_hf.get_atm_nrhf(mol)
        self.assertAlmostEqual(result['N'][0], -53.823206125468346, 9)
        result = atom_ks.get_atm_nrks(mol)
        self.assertAlmostEqual(result['N'][0], -53.53518426665269, 9)

if __name__ == "__main__":
    print("Full Tests for He")
    unittest.main()
