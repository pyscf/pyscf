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

import numpy as np

from pyscf import gto, scf
from pyscf.scf import gcp as gcp_mod

try:
    from pyscf.dispersion import gcp as disp_gcp
except ImportError:
    disp_gcp = None

_HAS_GCP = disp_gcp is not None


def _make_ks(xc='b97-3c'):
    # pyscf.dft is imported here (and not at module level) so that loading
    # this test module does not replace the mock scf.hf.KohnShamDFT class
    # before other tests in the directory have run.
    from pyscf import dft
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    return mf


class TestParseGCP(unittest.TestCase):
    def test_parse_gcp(self):
        self.assertEqual(gcp_mod.parse_gcp('b97-3c'), ('b973c', 'def2mtzvp'))
        self.assertEqual(gcp_mod.parse_gcp('b97_3c'), ('b973c', 'def2mtzvp'))
        self.assertEqual(gcp_mod.parse_gcp('gga_xc_b97_3c'), ('b973c', 'def2mtzvp'))
        self.assertEqual(gcp_mod.parse_gcp('r2scan-3c'), ('r2scan3c', 'def2mtzvpp'))
        self.assertEqual(gcp_mod.parse_gcp('r2scan_3c'), ('r2scan3c', 'def2mtzvpp'))
        # wB97X-3c has no gCP/SRB correction
        self.assertIsNone(gcp_mod.parse_gcp('wb97x-3c'))
        self.assertIsNone(gcp_mod.parse_gcp('b3lyp'))
        self.assertIsNone(gcp_mod.parse_gcp(None))


class TestCheckGCP(unittest.TestCase):
    def test_check_gcp(self):
        mf = _make_ks('b3lyp')
        self.assertFalse(gcp_mod.check_gcp(mf))

        mf.xc = 'b97-3c'
        self.assertTrue(gcp_mod.check_gcp(mf))

        # disabled via mf.gcp
        mf.gcp = False
        self.assertFalse(gcp_mod.check_gcp(mf))
        self.assertFalse(gcp_mod.check_gcp(mf, gcp=False))

        # explicit enable
        mf.gcp = True
        self.assertTrue(gcp_mod.check_gcp(mf))

        # HF has no gCP
        mf_hf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 1'))
        self.assertFalse(gcp_mod.check_gcp(mf_hf))


@unittest.skipIf(not _HAS_GCP, 'pyscf-dispersion not installed')
class TestGCPEnergy(unittest.TestCase):
    def test_get_gcp_b97_3c(self):
        mf = _make_ks('b97-3c')
        e = gcp_mod.get_gcp(mf)
        self.assertAlmostEqual(e, -0.000818489452022319, 10)

    def test_get_gcp_r2scan_3c(self):
        mf = _make_ks('r2scan-3c')
        e = gcp_mod.get_gcp(mf)
        self.assertAlmostEqual(e, 0.0002982079859909563, 10)

    def test_get_gcp_override(self):
        # an explicit gCP method string overrides the xc-derived method,
        # following the disp convention
        mf = _make_ks('b3lyp')  # no gCP derived from xc
        self.assertFalse(gcp_mod.check_gcp(mf))
        self.assertTrue(gcp_mod.check_gcp(mf, gcp='b973c'))
        e = gcp_mod.get_gcp(mf, gcp='b973c')
        self.assertAlmostEqual(e, -0.000818489452022319, 10)

        # 'method:basis' selects both
        e = gcp_mod.get_gcp(mf, gcp='b973c:def2mtzvp')
        self.assertAlmostEqual(e, -0.000818489452022319, 10)
        e = gcp_mod.get_gcp(mf, gcp='r2scan3c')
        self.assertAlmostEqual(e, 0.0002982079859909563, 10)

        # disabled explicitly
        self.assertEqual(gcp_mod.get_gcp(mf, gcp=False), 0.)

    def test_get_gcp_gradient(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
        model = disp_gcp.GCP(mol, method='b973c', basis='def2mtzvp')
        out = model.get_counterpoise(grad=True)
        self.assertAlmostEqual(np.linalg.norm(out['gradient']),
                               0.0028068213098591467, 8)
        self.assertAlmostEqual(np.linalg.norm(out['virial']),
                               0.0037505817348592453, 8)


if __name__ == "__main__":
    print("Full Tests")
    unittest.main()
