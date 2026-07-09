#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf import gto, scf
from pyscf.scf import dispersion


class KnownKS(scf.hf.KohnShamDFT):
    def __init__(self, xc='b3lyp'):
        self.xc = xc
        self.disp = None


class KnownHF(scf.hf.SCF):
    def __init__(self):
        self.disp = None


class TestDispersionLogic(unittest.TestCase):
    def test_parse_disp_none(self):
        # Case 1: All None
        self.assertEqual(dispersion.parse_disp(None, None), (None, None, False))

    def test_parse_disp_explicit(self):
        # Case 2: Explicit disp takes precedence
        # b3lyp normally has no disp.
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj'), ('b3lyp', 'd3bj', False))

        # disp with colon override method
        self.assertEqual(dispersion.parse_disp(None, 'd4:wb97x-3c'), ('wb97x-3c', 'd4', True))
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj:pbe'), ('pbe', 'd3bj', False))

        # disp with suffix
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj2b'), ('b3lyp', 'd3bj', False))
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bjatm'), ('b3lyp', 'd3bj', True))

        # d4 always implies 3body
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd4'), ('b3lyp', 'd4', True))

    def test_parse_disp_from_method(self):
        # Case 3: Infer from method
        # b3lyp -> no disp
        self.assertEqual(dispersion.parse_disp('b3lyp'), (None, None, False))

        # wb97x-d3bj -> d3bj
        self.assertEqual(dispersion.parse_disp('wb97x-d3bj'), ('wb97x', 'd3bj', False))

        # wb97x-d4s -> d4s
        self.assertEqual(dispersion.parse_disp('wb97x-d4s'), ('wb97x', 'd4s', True))

        # wb97x-3c -> d4, 3body=True (from whitelist)
        self.assertEqual(dispersion.parse_disp('wb97x-3c'), ('wb97x-3c', 'd4', True))

    def test_parse_disp_errors(self):
        # Unknown disp version
        with self.assertRaises(ValueError):
            dispersion.parse_disp('b3lyp', 'unknown_ver')

        # Disp specified but method unknown/missing (if disp string doesn't contain colon)
        # Actually parse_disp(None, 'd3bj') -> raises ValueError "the method used in dispersion d3bj is not specified."
        with self.assertRaises(ValueError):
            dispersion.parse_disp(None, 'd3bj')

    def test_check_disp(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1')

        # 1. RHF object (no .xc)
        mf_hf = scf.RHF(mol)
        self.assertFalse(dispersion.check_disp(mf_hf))

        # If mf.disp = None
        mf_hf.disp = None
        # parse_disp('hf', None) -> ('hf', None, False) -> check_disp returns False
        self.assertFalse(dispersion.check_disp(mf_hf))

        # If we set mf.disp = 'd3bj'
        mf_hf.disp = 'd3bj'
        self.assertTrue(dispersion.check_disp(mf_hf))

        # 2. KohnShamDFT object (has .xc)
        mf_dft = KnownKS()
        mf_dft.xc = 'b3lyp'
        mf_dft.disp = None

        # b3lyp -> no disp -> False
        self.assertFalse(dispersion.check_disp(mf_dft))

        # Explicit disp
        self.assertTrue(dispersion.check_disp(mf_dft, disp='d3bj'))

        # Explicit disp=False
        self.assertFalse(dispersion.check_disp(mf_dft, disp=False))

        # Implicit disp from method
        mf_dft.xc = 'wb97x-d3bj'
        self.assertTrue(dispersion.check_disp(mf_dft))

        # Unsupported disp version
        with self.assertRaises(ValueError):
            dispersion.check_disp(mf_dft, disp='unsupported')

    @unittest.skipIf(dispersion.dftd4 is None, 'DFTD4 not installed')
    def test_wb97x_d4(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''
              O     0.   0.       0.
              H     0.   -0.757   0.587
              H     0.   0.757    0.587''',
            basis = 'def2-svp')
        mf = mol.RKS(xc='wb97x-d4').run()
        if int(pyscf.__version__.split('.')[1]) > 15:
            # corresponding to
            # mf.xc = 'wb97x-v'
            # mf.nlc = False
            # mf.disp = 'd4:wb97x'
            self.assertAlmostEqual(mf.e_tot, -76.37197333535842, 8)
        else:
            # legacy value, corresponding to
            # mf.xc = 'wb97x'
            # mf.nlc = False
            # mf.disp = 'd4:wb97x-2008'
            self.assertAlmostEqual(mf.e_tot, -76.3377143469286, 8)

if __name__ == "__main__":
    unittest.main()
