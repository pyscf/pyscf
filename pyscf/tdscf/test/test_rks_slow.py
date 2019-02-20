from pyscf.gto import Mole
from pyscf.scf import RKS, RHF
from pyscf.tdscf import TDDFT, TDHF
from pyscf.tdscf.rks_slow import PhysERI, TDRKS, mk_make_canonic

import numpy
from numpy import testing
import unittest

from test_common import retrieve_m, assert_vectors_close, tdhf_frozen_mask


class H20Test(unittest.TestCase):
    """Compare this (rks_slow) vs reference."""
    @classmethod
    def setUpClass(cls):
        cls.mol = mol = Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = 'cc-pvdz'
        mol.verbose = 5
        mol.build()

        cls.model_rks = model_rks = RKS(mol)
        model_rks.kernel()

        cls.td_model_rks = td_model_rks = TDDFT(model_rks)
        td_model_rks.nroots = 4
        td_model_rks.kernel()

        e = model_rks.mo_energy
        nocc = int(sum(model_rks.mo_occ) // 2)
        cls.ref_m = mk_make_canonic(retrieve_m(td_model_rks), e[:nocc], e[nocc:])

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rks
        del cls.model_rks
        del cls.mol

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = PhysERI(self.model_rks)
        mk, _ = e.tdhf_mk_form()
        testing.assert_allclose(self.ref_m, mk, atol=1e-13)

        # Test frozen
        for frozen in (1, [0, -1]):
            try:
                e = PhysERI(self.model_rks, frozen=frozen)
                mk, _ = e.tdhf_mk_form()
                ov_mask = tdhf_frozen_mask(e, kind="1ov")
                ref_m = self.ref_m[numpy.ix_(ov_mask, ov_mask)]
                testing.assert_allclose(ref_m, mk, atol=1e-13)

            except Exception:
                print("When testing with frozen={} the following exception occurred:".format(repr(frozen)))
                raise

    def test_class(self):
        """Tests container behavior."""
        model = TDRKS(self.model_rks)
        model.nroots = self.td_model_rks.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rks.e, atol=1e-5)
        assert_vectors_close(model.xy, self.td_model_rks.xy, atol=1e-2)

    def test_class_frozen(self):
        """Tests container behavior."""
        for frozen in (1, [0, -1]):
            try:
                model = TDRKS(self.model_rks, frozen=frozen)
                model.nroots = self.td_model_rks.nroots
                model.kernel()
                mask_o, mask_v = tdhf_frozen_mask(model.eri, kind="o,v")
                testing.assert_allclose(model.e, self.td_model_rks.e, atol=1e-3)
                assert_vectors_close(model.xy, numpy.array(self.td_model_rks.xy)[..., mask_o, :][..., mask_v], atol=1e-3)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise


class H20HFTest(unittest.TestCase):
    """Compare this (rks_slow) vs reference in a Hartree-Fock setup."""
    @classmethod
    def setUpClass(cls):
        cls.mol = mol = Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = 'cc-pvdz'
        mol.verbose = 5
        mol.build()

        cls.model_rhf = model_rhf = RHF(mol)
        model_rhf.kernel()

        cls.td_model_rhf = td_model_rhf = TDHF(model_rhf)
        td_model_rhf.nroots = 4
        td_model_rhf.kernel()

        cls.ref_m = retrieve_m(td_model_rhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rhf
        del cls.model_rhf
        del cls.mol

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = PhysERI(self.model_rhf)
        m = e.tdhf_full_form()
        testing.assert_allclose(self.ref_m, m, atol=1e-13)

        # Test frozen
        for proxy in (None, self.td_model_rhf):
            for frozen in (1, [0, -1]):
                try:
                    e = PhysERI(self.model_rhf, frozen=frozen)
                    m = e.tdhf_full_form()
                    ov_mask = tdhf_frozen_mask(e, kind="ov")
                    ref_m = self.ref_m[numpy.ix_(ov_mask, ov_mask)]
                    testing.assert_allclose(ref_m, m, atol=1e-13)

                except Exception:
                    print("When testing with proxy={} and frozen={} the following exception occurred:".format(
                        repr(proxy),
                        repr(frozen)
                    ))
                    raise

    def test_class(self):
        """Tests container behavior."""
        model = TDRKS(self.model_rhf)
        model.nroots = self.td_model_rhf.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-5)
        assert_vectors_close(model.xy, self.td_model_rhf.xy, atol=1e-2)

    def test_class_frozen(self):
        """Tests container behavior."""
        for frozen in (1, [0, -1]):
            try:
                model = TDRKS(self.model_rhf, frozen=frozen)
                model.nroots = self.td_model_rhf.nroots
                model.kernel()
                mask_o, mask_v = tdhf_frozen_mask(model.eri, kind="o,v")
                testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-3)
                assert_vectors_close(model.xy, numpy.array(self.td_model_rhf.xy)[..., mask_o, :][..., mask_v], atol=1e-3)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise
