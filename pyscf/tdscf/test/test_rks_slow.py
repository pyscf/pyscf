from pyscf.gto import Mole
from pyscf.scf import RKS
from pyscf.tdscf import TDDFT
from pyscf.tdscf.rks_slow import PhysERI, TDRKS, canonic

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
        cls.ref_m = canonic(retrieve_m(td_model_rks), e[:nocc], e[nocc:])

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rks
        del cls.model_rks
        del cls.mol

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = PhysERI(self.model_rks)
        mk, _ = e.fast_tdhf_matrix_set()
        testing.assert_allclose(self.ref_m, mk, atol=1e-13)

        # Test frozen
        for frozen in (1, [0, -1]):
            try:
                e = PhysERI(self.model_rks, frozen=frozen)
                mk, _ = e.fast_tdhf_matrix_set()
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
