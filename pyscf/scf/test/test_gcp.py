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
    import basis_set_exchange
except ImportError:
    basis_set_exchange = None

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


@unittest.skipIf(not _HAS_GCP, 'pyscf-dispersion not installed')
@unittest.skipIf(basis_set_exchange is None, 'basis_set_exchange not installed')
class TestDFT3C(unittest.TestCase):
    def test_dft3c_b97_3c_energy(self):
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='def2mtzvp', verbose=0)
        mf = dft.RKS(mol).dft3c('b97-3c').density_fit()
        mf.conv_tol = 1e-10
        e = mf.kernel()
        self.assertAlmostEqual(e, -76.398062389459, 8)
        self.assertAlmostEqual(mf.scf_summary['dispersion'], -0.000847206575, 8)
        self.assertAlmostEqual(mf.scf_summary['gcp'], -0.005630371832, 8)

    def test_dft3c_r2scan_3c_energy(self):
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='def2mtzvpp', verbose=0)
        mf = dft.RKS(mol).dft3c('r2scan-3c').density_fit()
        mf.conv_tol = 1e-10
        e = mf.kernel()
        self.assertAlmostEqual(e, -76.418888685035, 8)
        self.assertAlmostEqual(mf.scf_summary['dispersion'], -8.574981232398e-05, 8)
        self.assertAlmostEqual(mf.scf_summary['gcp'], 0.001801492550, 8)

    def test_r2scan_3c_d4_custom_charge_model(self):
        # The D4 correction of r2SCAN-3c uses a custom EEQ charge model
        # (ga=2.0, gc=1.0).  This is a special case of the dftd4 program
        # (app/driver.f90) that is not encoded in the damping parameter
        # table.
        from pyscf.dispersion import dftd4

        from pyscf.scf import dispersion as disp_mod
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='def2mtzvpp', verbose=0)
        ref_model = dftd4.DFTD4Dispersion(mol, xc='r2scan-3c', atm=True,
                                          ga=2.0, gc=1.0)
        e_ref = ref_model.get_dispersion()['energy']
        model = disp_mod.make_dftd4_model(mol, 'r2scan-3c', True)
        e = model.get_dispersion()['energy']
        self.assertAlmostEqual(e, e_ref, 12)
        # the default charge model gives a different (incorrect) value
        default_model = dftd4.DFTD4Dispersion(mol, xc='r2scan-3c', atm=True)
        self.assertNotAlmostEqual(default_model.get_dispersion()['energy'],
                                  e_ref, 6)

    def test_dft3c_basis(self):
        # B97-3c -> def2-mTZVP, r2SCAN-3c -> def2-mTZVPP, shared RI-J auxbasis
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
        mf = dft.RKS(mol).dft3c('b97-3c')
        self.assertEqual(mf.method3c, 'b97-3c')
        self.assertEqual(mf.mol.basis, 'def2mtzvp')
        self.assertEqual(mf.xc, 'b97-3c')
        # density fitting after dft3c picks the RI-J auxiliary basis
        self.assertEqual(mf.density_fit().auxbasis, 'def2-mTZVPP-RIJ')

        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvpp', verbose=0)
        mf = dft.RKS(mol).dft3c('r2scan-3c')
        self.assertEqual(mf.method3c, 'r2scan-3c')
        self.assertEqual(mf.mol.basis, 'def2mtzvpp')
        self.assertEqual(mf.xc, 'r2scan-3c')
        self.assertEqual(mf.density_fit().auxbasis, 'def2-mTZVPP-RIJ')

        with self.assertRaises(NotImplementedError):
            dft.RKS(mol).dft3c('pbeh-3c')

    def test_dft3c_density_fit_order(self):
        # density fitting can be applied before or after dft3c
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='def2mtzvp', verbose=0)
        mf1 = dft.RKS(mol).dft3c('b97-3c').density_fit()
        mf1.conv_tol = 1e-10
        e1 = mf1.kernel()

        mol2 = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                     basis='def2mtzvp', verbose=0)
        mf2 = dft.RKS(mol2).density_fit().dft3c('b97-3c')
        mf2.conv_tol = 1e-10
        e2 = mf2.kernel()
        self.assertEqual(mf2.with_df.auxbasis, 'def2-mTZVPP-RIJ')
        self.assertAlmostEqual(e1, e2, 8)

    def test_dft3c_undo(self):
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
        mf = dft.RKS(mol).dft3c('b97-3c')
        obj = mf.undo_dft3c()
        self.assertFalse(hasattr(obj, 'method3c'))
        # applying dft3c twice raises
        with self.assertRaises(RuntimeError):
            dft.RKS(mol).dft3c('b97-3c').dft3c('r2scan-3c')

    def test_mol_rks3c(self):
        # mol.RKS3C() convenience, like mol.RKS()
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        mf = mol.RKS3C()
        self.assertEqual(mf.method, 'b97-3c')
        self.assertEqual(mf.mol.basis, 'def2mtzvp')
        self.assertEqual(mf.xc, 'b97-3c')
        # the method kwarg goes through the mol dispatch
        mf = mol.RKS3C(method='r2scan-3c')
        self.assertEqual(mf.method, 'r2scan-3c')
        self.assertEqual(mf.mol.basis, 'def2mtzvpp')
        self.assertEqual(mf.xc, 'r2scan-3c')

        # module factory with positional method
        self.assertEqual(dft.RKS3C(mol, 'r2scan-3c').method, 'r2scan-3c')

        # UKS3C on an open-shell molecule
        o = gto.M(atom='O 0 0 0', spin=2, verbose=0)
        mu = o.UKS3C()
        self.assertEqual(mu.method, 'b97-3c')
        self.assertEqual(mu.xc, 'b97-3c')

        # RKS3C on an open-shell molecule dispatches to ROKS internally
        self.assertIn('ROKS', type(o.RKS3C()).__name__)
        self.assertEqual(o.ROKS3C().method, 'b97-3c')
        self.assertEqual(o.GKS3C().method, 'b97-3c')

    def test_dft3c_wb97x_3c(self):
        # wB97X-3c: wB97X-V + D4 in the ECP-based Grimme vDZP basis, no gCP
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', verbose=0)
        mf = mol.RKS3C(method='wb97x-3c')
        self.assertEqual(mf.method, 'wb97x-3c')
        self.assertEqual(mf.mol.basis, 'Grimme vDZP')
        self.assertEqual(mf.xc, 'wb97x-3c')
        # ECPs are set per element; light elements without an ECP are skipped
        self.assertNotIn('H', mf.mol.ecp)
        self.assertTrue(mf.mol.ecp)
        # VV10 NLC is replaced by D4; no gCP/SRB correction
        self.assertFalse(mf.do_nlc())
        self.assertTrue(mf.do_disp())
        self.assertFalse(mf.do_gcp())

        mf.conv_tol = 1e-10
        e = mf.kernel()
        self.assertAlmostEqual(e, -17.270909745768, 8)
        self.assertAlmostEqual(mf.scf_summary['dispersion'], -0.000263630665, 8)
        self.assertNotIn('gcp', mf.scf_summary)

        # density fitting has no dedicated auxiliary basis and falls back
        # to even-tempered functions
        mol2 = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', verbose=0)
        self.assertIsNone(dft.RKS(mol2).dft3c('wb97x-3c').density_fit().auxbasis)

        # switching to an all-electron 3c method clears the ECPs
        mf.method = 'b97-3c'
        self.assertEqual(mf.mol.ecp, {})
        self.assertEqual(mf.mol.basis, 'def2mtzvp')

    def test_dft3c_auto_auxbasis(self):
        # density_fit() without explicit auxbasis picks def2-mTZVPP-RIJ
        # through the DEFAULT_AUXBASIS_JFIT_BSE record
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'b97-3c'
        mf = mf.density_fit()
        self.assertEqual(mf.auxbasis, 'def2-mTZVPP-RIJ')

        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvpp', verbose=0)
        self.assertEqual(dft.RKS(mol).dft3c('r2scan-3c').density_fit().auxbasis,
                         'def2-mTZVPP-RIJ')

    def test_dft3c_no_bse_fallback(self):
        # Without basis_set_exchange, the RI-J auxiliary basis falls back
        # to the even-tempered basis instead of failing.
        from pyscf.df.addons import predefined_auxbasis
        from pyscf.gto.basis import bse
        dft = __import__('pyscf.dft', fromlist=['RKS'])

        saved = bse.basis_set_exchange
        bse.basis_set_exchange = None
        try:
            # the manual density_fit path falls back to the even-tempered basis
            mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
            self.assertIsNone(predefined_auxbasis(mol, 'def2mtzvp', 'b97-3c'))
            # the hybrid path still uses the bundled JK-fit basis
            self.assertEqual(predefined_auxbasis(mol, 'def2mtzvp', 'b3lyp'),
                             'def2-tzvp-jkfit')
            # density_fit after dft3c builds with the etb fallback
            mf = dft.RKS(mol).dft3c('b97-3c').density_fit()
            mf.with_df.build()
            self.assertIsNotNone(mf.with_df.auxmol)
            # density_fit before dft3c rebuilds with_df with the etb fallback
            mol2 = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
            mf2 = dft.RKS(mol2).density_fit().dft3c('b97-3c')
            mf2.with_df.build()
            self.assertIsNotNone(mf2.with_df.auxmol)
        finally:
            bse.basis_set_exchange = saved

    def test_dft3c_switches(self):
        # disp=gcp=False gives the XC-only energy
        dft = __import__('pyscf.dft', fromlist=['RKS'])
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='def2mtzvp', verbose=0)
        mf = dft.RKS(mol).dft3c('b97-3c')
        mf.disp = False
        mf.gcp = False
        mf.conv_tol = 1e-10
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.150635605714, 6)


if __name__ == "__main__":
    print("Full Tests")
    unittest.main()
