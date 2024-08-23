from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.tdscf import TDHF
from pyscf.tdscf.rhf_slow import PhysERI, PhysERI4, PhysERI8, TDRHF
from pyscf.tdscf.common_slow import eig, full2ab, ab2full, full2mkk, mkk2full, ab2mkk, mkk2ab

import numpy
from numpy import testing
import unittest

from test_common import retrieve_m, retrieve_m_hf, assert_vectors_close, tdhf_frozen_mask


class H20Test(unittest.TestCase):
    """Compare this (rhf_slow) vs reference (pyscf)."""
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

    def test_conversion(self):
        """Tests common conversions."""
        e = PhysERI(self.model_rhf)

        full = e.tdhf_full_form()
        a, b = e.tdhf_ab_form()
        mk, k = e.tdhf_mk_form()

        testing.assert_allclose(full, ab2full(a, b))
        testing.assert_allclose(full, mkk2full(mk, k), atol=1e-13)

        _a, _b = full2ab(full)
        testing.assert_allclose(a, _a)
        testing.assert_allclose(b, _b)
        _a, _b = mkk2ab(mk, k)
        testing.assert_allclose(a, _a)
        testing.assert_allclose(b, _b)

        _mk, _k = full2mkk(full)
        testing.assert_allclose(mk, _mk)
        testing.assert_allclose(k, _k)
        _mk, _k = ab2mkk(a, b)
        testing.assert_allclose(mk, _mk)
        testing.assert_allclose(k, _k)

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERI, PhysERI4, PhysERI8):

            # Test plain
            try:
                e = eri(self.model_rhf)
                m = e.tdhf_full_form()

                # Test matrix vs ref
                testing.assert_allclose(m, retrieve_m_hf(e), atol=1e-14)

                # Test matrix vs pyscf
                testing.assert_allclose(self.ref_m, m, atol=1e-14)
                vals, vecs = eig(m, nroots=self.td_model_rhf.nroots)
                testing.assert_allclose(vals, self.td_model_rhf.e, atol=1e-5)

            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

            # Test frozen
            for frozen in (1, [0, -1]):
                try:
                    e = eri(self.model_rhf, frozen=frozen)
                    m = e.tdhf_full_form()
                    ov_mask = tdhf_frozen_mask(e)
                    ref_m = self.ref_m[numpy.ix_(ov_mask, ov_mask)]
                    testing.assert_allclose(ref_m, m, atol=1e-14)

                except Exception:
                    print("When testing {} with frozen={} the following exception occurred:".format(eri, repr(frozen)))
                    raise

    def test_class(self):
        """Tests container behavior."""
        model = TDRHF(self.model_rhf)
        model.nroots = self.td_model_rhf.nroots
        assert model.fast
        e, xy = model.kernel()
        model.fast = False
        model.kernel()
        # Slow vs fast
        testing.assert_allclose(model.e, e)
        assert_vectors_close(model.xy, xy)
        # ... vs ref
        testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-5)
        assert_vectors_close(model.xy, self.td_model_rhf.xy, atol=1e-2)
        # Test real-valued
        testing.assert_allclose(model.e.imag, 0, atol=1e-8)

    def test_class_frozen(self):
        """Tests container behavior."""
        for frozen in (1, [0, -1]):
            try:
                model = TDRHF(self.model_rhf, frozen=frozen)
                model.nroots = self.td_model_rhf.nroots
                model.kernel()
                mask_o, mask_v = tdhf_frozen_mask(model.eri, kind="o,v")
                testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-3)
                assert_vectors_close(model.xy,
                    numpy.array(self.td_model_rhf.xy)[..., mask_o, :][..., mask_v], atol=1e-3)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise

    def test_symm(self):
        """Tests 8-fold symmetry."""
        eri = PhysERI8(self.model_rhf)
        vals = eri.ao2mo((self.model_rhf.mo_coeff,) * 4)
        for i, c in eri.symmetries:
            testing.assert_allclose(vals, vals.transpose(*i), atol=1e-14)
