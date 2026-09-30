#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''
Semi-canonical ROMP2 for ROHF references with 4-center integrals
'''

import unittest
from pyscf import gto, scf, mp, lib


def _make_mol(atom, spin, basis='def2-svp', charge=0):
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = atom
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.build()
    return mol


class KnownValues(unittest.TestCase):
    def test_closed_shell_matches_rmp2(self):
        mol = _make_mol('H 0 0 0; F 0 0 1.1', 0, basis='cc-pvdz')
        mf = scf.ROHF(mol).run(conv_tol=1e-12)
        pt = mf.MP2().run()
        self.assertTrue(isinstance(pt, mp.romp2.ROMP2))
        pt2 = scf.RHF(mol).run(conv_tol=1e-12).MP2().run()
        self.assertAlmostEqual(pt.e_tot, pt2.e_tot, 8)
        self.assertAlmostEqual(pt.e_corr_singles, 0, 9)
        mol.stdout.close()

    def test_o2_triplet_doubles_vs_ump2(self):
        # the doubles part of ROMP2 is the UMP2 doubles energy evaluated with
        # the semi-canonical orbitals; check against UMP2 with these orbitals
        mol = _make_mol('O 0 0 0; O 0 0 1.2222', 2)
        mf = scf.ROHF(mol).run(conv_tol=1e-12)
        pt = mf.MP2().run()
        self.assertAlmostEqual(pt.e_corr_singles, -0.01236519, 6)
        mfu = mf.to_uhf()
        mfu.mo_coeff = pt.mo_coeff
        mfu.mo_energy = pt.mo_energy
        mfu.mo_occ = pt.mo_occ
        mfu.chkfile = lib.NamedTemporaryFile().name
        pu = mp.UMP2(mfu).run()
        self.assertAlmostEqual(pu.e_corr, pt.e_corr_ss + pt.e_corr_os, 9)
        self.assertAlmostEqual(pt.e_tot, mf.e_tot + pt.e_corr, 9)
        mol.stdout.close()

    def test_o2_triplet_vs_df(self):
        # the singles correction is independent of the two-electron integrals;
        # the doubles part differs from DF-ROMP2 only by the fitting error
        mol = _make_mol('O 0 0 0; O 0 0 1.2222', 2)
        mf = scf.ROHF(mol).run(conv_tol=1e-12)
        pt = mf.MP2().run()
        pt_df = scf.ROHF(mol).density_fit(auxbasis='def2svpjkfit').run(
            conv_tol=1e-12).MP2().run()
        # singles: REST reference (xc="mp2", def2-SVP/def2-universal-JKFIT)
        # is -0.01236550; see pyscf/mp/test/test_dfromp2.py
        self.assertAlmostEqual(pt.e_corr_singles, pt_df.e_corr_singles, 6)
        self.assertAlmostEqual(pt.e_corr, pt_df.e_corr, 3)
        mol.stdout.close()

    def test_frozen(self):
        mol = _make_mol('O 0 0 0; O 0 0 1.2222', 2)
        mf = scf.ROHF(mol).run(conv_tol=1e-12)
        pt = mf.MP2().run()
        pt_frozen = mp.ROMP2(mf, frozen=1).run()
        self.assertAlmostEqual(pt_frozen.e_tot, pt_frozen.e_hf + pt_frozen.e_corr, 9)
        self.assertLess(pt.e_corr, pt_frozen.e_corr)
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for ROMP2")
    unittest.main()
