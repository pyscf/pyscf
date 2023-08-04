#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.lo.cholesky import cholesky_mos


def setUpModule():
    global mol, mf
    mol = Mole()
    mol.atom = '''
    C        0.681068338      0.605116159      0.307300799
    C       -0.733665805      0.654940451     -0.299036438
    C       -1.523996730     -0.592207689      0.138683275
    H        0.609941801      0.564304456      1.384183068
    H        1.228991034      1.489024155      0.015946420
    H       -1.242251083      1.542928348      0.046243898
    H       -0.662968178      0.676527364     -1.376503770
    H       -0.838473936     -1.344174292      0.500629028
    H       -2.075136399     -0.983173387     -0.703807608
    H       -2.212637905     -0.323898759      0.926200671
    O        1.368219958     -0.565620846     -0.173113101
    H        2.250134219     -0.596689848      0.204857736
    '''
    mol.basis = 'STO-3G'
    mol.verbose = 0
    mol.output = None
    mol.build()
    mf = RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def setUp(self):
        self.mol = mol.copy()
        self.mo_coeff = mf.mo_coeff.copy()
        self.nocc = numpy.count_nonzero(mf.mo_occ > 0)
        self.rdm1_rhf = mf.make_rdm1()
        self.sao = mf.get_ovlp()

    def test_density(self):
        # Test whether the localized orbitals preserve the density.
        mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])
        rdm_loc = 2 * mo_loc.dot(mo_loc.T)
        matching = numpy.allclose(rdm_loc, self.rdm1_rhf, atol=1.0e-12)
        self.assertTrue(matching)

    def test_orth(self):
        # Test whether the localized orbitals are orthonormal.
        mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])
        smo = numpy.linalg.multi_dot([mo_loc.T, self.sao, mo_loc])
        matching = numpy.allclose(smo, numpy.eye(self.nocc), 1.0e-12)
        self.assertTrue(matching)

    def test_localization(self):
        # Check a few selected values of the orbital coefficient matrix.
        mo_loc = cholesky_mos(self.mo_coeff[:, :self.nocc])
        delta = 1.0e-6
        self.assertAlmostEqual(abs(mo_loc[22, 0]), 1.02618438, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[6, 3]), 0.10412481, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[27, 5]), 0.17253633, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[6, 8]), 0.63599723, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[14, 11]), 0.5673705, delta=delta)
        self.assertAlmostEqual(abs(mo_loc[4, 15]), 0.51124407, delta=delta)


if __name__ == "__main__":
    print("Test Cholesky localization")
    unittest.main()
