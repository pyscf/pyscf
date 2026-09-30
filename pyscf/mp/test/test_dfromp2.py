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
Semi-canonical DF-ROMP2 for ROHF references
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


def _make_mf(mol):
    mf = scf.ROHF(mol).density_fit(auxbasis='def2svpjkfit')
    mf.conv_tol = 1e-12
    mf.chkfile = lib.NamedTemporaryFile().name
    mf.kernel()
    return mf


class KnownValues(unittest.TestCase):
    # O2 triplet, def2-SVP, def2-universal-JKFIT auxiliary basis.
    # Reference values from the REST program (semi-canonical ROMP2,
    # xc = "mp2", commit ef82ff5caa145db85549c8d9ef7aa056cf4c3133).
    # Geometry from pyscf/scf/test/test_stability.py.
    def test_o2_triplet(self):
        mol = _make_mol('O 0 0 0; O 0 0 1.2222', 2)
        mf = _make_mf(mol)
        pt = mf.DFROMP2().run()
        self.assertAlmostEqual(pt.e_corr, -0.38682976, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -0.11760541, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.25685885, 7)
        self.assertAlmostEqual(pt.e_corr_singles, -0.01236550, 7)
        self.assertAlmostEqual(pt.e_tot, mf.e_tot + pt.e_corr, 9)
        # SCS: the singles term is added unscaled to the scaled doubles
        self.assertAlmostEqual(pt.emp2_scs, pt.e_corr_singles +
                               pt.e_corr_ss/3. + pt.e_corr_os*1.2, 9)
        mol.stdout.close()

    # NH3+ doublet, def2-SVP, def2-universal-JKFIT auxiliary basis.
    # Reference values from the REST program (semi-canonical ROMP2,
    # xc = "mp2", commit ef82ff5caa145db85549c8d9ef7aa056cf4c3133).
    # Geometry from the REST workshop examples (ROHF_NH3_cation).
    def test_nh3_cation_doublet(self):
        mol = _make_mol('''
            N -2.1988391019  1.8973746268  0.0000000000
            H -1.1788391019  1.8973746268  0.0000000000
            H -2.5388353987  1.0925460144 -0.5263586446
            H -2.5388400276  2.7556271745 -0.4338224694
        ''', 1, charge=1)
        mf = _make_mf(mol)
        pt = mp.MP2(mf).run()
        self.assertTrue(isinstance(pt, mp.dfromp2.DFROMP2))
        self.assertAlmostEqual(pt.e_corr, -0.14990839, 7)
        self.assertAlmostEqual(pt.e_corr_ss, -0.03111863, 7)
        self.assertAlmostEqual(pt.e_corr_os, -0.11656739, 7)
        self.assertAlmostEqual(pt.e_corr_singles, -0.00222237, 7)
        mol.stdout.close()

    def test_closed_shell_matches_dfmp2(self):
        mol = _make_mol('H 0 0 0; F 0 0 1.1', 0, basis='cc-pvdz')
        mf = _make_mf(mol)
        pt = mf.DFROMP2().run()
        pt2 = mf.DFMP2().run()
        self.assertAlmostEqual(pt.e_tot, pt2.e_tot, 9)
        self.assertAlmostEqual(pt.e_corr_singles, 0, 9)
        mol.stdout.close()

    def test_include_singles(self):
        mol = _make_mol('O 0 0 0; O 0 0 1.2222', 2)
        mf = _make_mf(mol)
        pt = mf.DFROMP2().run()
        pt.include_singles = False
        pt.kernel()
        self.assertAlmostEqual(pt.e_corr, pt.e_corr_ss + pt.e_corr_os, 9)
        mol.stdout.close()

    def test_frozen(self):
        mol = _make_mol('O 0 0 0; O 0 0 1.2222', 2)
        mf = _make_mf(mol)
        pt = mf.DFROMP2().run()
        pt_frozen = mf.DFROMP2(frozen=1).run()
        self.assertAlmostEqual(pt_frozen.e_tot, pt_frozen.e_hf + pt_frozen.e_corr, 9)
        # freezing the 1s core reduces the magnitude of the correlation energy
        self.assertLess(pt.e_corr, pt_frozen.e_corr)
        # the core singles contribution is small but non-zero
        self.assertAlmostEqual(pt_frozen.e_corr_singles - pt.e_corr_singles, 0, 4)
        mol.stdout.close()


if __name__ == "__main__":
    print("Full Tests for DF-ROMP2")
    unittest.main()
