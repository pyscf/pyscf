from pyscf.gto import Mole
from pyscf.scf import RKS, RHF
from pyscf.tdscf import TDDFT, TDHF
from pyscf.tdscf.proxy import PhysERI, TDProxy, mk_make_canonic, molecular_response, orb2ov
from pyscf.tdscf.common_slow import format_frozen_mol

import numpy
from numpy import testing
import unittest

from test_common import retrieve_m, assert_vectors_close, tdhf_frozen_mask


class H20Test(unittest.TestCase):
    """Compare this (molecular proxy) vs reference (pyscf)."""
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
        e = PhysERI(self.model_rks, "dft")
        mk, _ = e.tdhf_mk_form()
        testing.assert_allclose(self.ref_m, mk, atol=1e-13)

        # Test frozen
        for frozen in (1, [0, -1]):
            try:
                e = PhysERI(self.model_rks, "dft", frozen=frozen)
                mk, _ = e.tdhf_mk_form()
                ov_mask = tdhf_frozen_mask(e, kind="1ov")
                ref_m = self.ref_m[numpy.ix_(ov_mask, ov_mask)]
                testing.assert_allclose(ref_m, mk, atol=1e-13)

            except Exception:
                print("When testing with frozen={} the following exception occurred:".format(repr(frozen)))
                raise

    def test_class(self):
        """Tests container behavior."""
        model = TDProxy(self.model_rks, "dft")
        model.nroots = self.td_model_rks.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rks.e, atol=1e-5)
        assert_vectors_close(model.xy, self.td_model_rks.xy, atol=1e-2)

    def test_class_frozen(self):
        """Tests container behavior."""
        for frozen in (1, [0, -1]):
            try:
                model = TDProxy(self.model_rks, "dft", frozen=frozen)
                model.nroots = self.td_model_rks.nroots
                model.kernel()
                mask_o, mask_v = tdhf_frozen_mask(model.eri, kind="o,v")
                testing.assert_allclose(model.e, self.td_model_rks.e, atol=1e-3)
                assert_vectors_close(model.xy,
                    numpy.array(self.td_model_rks.xy)[..., mask_o, :][..., mask_v], atol=1e-3)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise

    def test_raw_response(self):
        """Tests the `molecular_response` and whether it slices output properly."""
        eri = PhysERI(self.model_rks, "dft")
        ref_m_full = eri.proxy_response()

        # Test single
        for frozen in (1, [0, -1]):
            space = format_frozen_mol(frozen, eri.nmo_full)
            space_ov = orb2ov(space, eri.nocc_full)

            m = molecular_response(
                eri.proxy_vind,
                space,
                eri.nocc_full,
                eri.nmo_full,
                False,
                eri.proxy_model,
            )
            ref_m = ref_m_full[space_ov, :][:, space_ov]
            testing.assert_allclose(ref_m, m, atol=1e-12)

        # Test pair
        for frozen in ((1, 3), ([1, -2], [0, -1])):
            space = tuple(format_frozen_mol(i, eri.nmo_full) for i in frozen)
            space_ov = tuple(orb2ov(i, eri.nocc_full) for i in space)

            m = molecular_response(
                eri.proxy_vind,
                space,
                eri.nocc_full,
                eri.nmo_full,
                False,
                eri.proxy_model,
            )
            ref_m = ref_m_full[space_ov[0], :][:, space_ov[1]]
            testing.assert_allclose(ref_m, m, atol=1e-12)


class H20HFTest(unittest.TestCase):
    """Compare this (molecular proxy) vs reference (pyscf), Hartree-Fock setup."""
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
        e = PhysERI(self.model_rhf, "hf")
        m = e.tdhf_full_form()
        testing.assert_allclose(self.ref_m, m, atol=1e-13)

        # Test frozen
        for proxy in (None, self.td_model_rhf):
            for frozen in (1, [0, -1]):
                try:
                    e = PhysERI(self.model_rhf, "dft", frozen=frozen)
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
        model = TDProxy(self.model_rhf, "hf")
        model.nroots = self.td_model_rhf.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-5)
        assert_vectors_close(model.xy, self.td_model_rhf.xy, atol=1e-2)

    def test_class_frozen(self):
        """Tests container behavior."""
        for frozen in (1, [0, -1]):
            try:
                model = TDProxy(self.model_rhf, "hf", frozen=frozen)
                model.nroots = self.td_model_rhf.nroots
                model.kernel()
                mask_o, mask_v = tdhf_frozen_mask(model.eri, kind="o,v")
                testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-3)
                assert_vectors_close(model.xy,
                    numpy.array(self.td_model_rhf.xy)[..., mask_o, :][..., mask_v], atol=1e-3)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise

    def test_raw_response(self):
        """Tests the `molecular_response` and whether it slices output properly."""
        eri = PhysERI(self.model_rhf, "hf")
        ref_m_full = eri.proxy_response()

        # Test single
        for frozen in (1, [0, -1]):
            space = format_frozen_mol(frozen, eri.nmo_full)
            space_ov = orb2ov(space, eri.nocc_full)

            m = molecular_response(
                eri.proxy_vind,
                space,
                eri.nocc_full,
                eri.nmo_full,
                True,
                eri.proxy_model,
            )
            ref_m = tuple(i[space_ov, :][:, space_ov] for i in ref_m_full)
            testing.assert_allclose(ref_m, m, atol=1e-12)

        # Test pair
        for frozen in ((1, 3), ([1, -2], [0, -1])):
            space = tuple(format_frozen_mol(i, eri.nmo_full) for i in frozen)
            space_ov = tuple(orb2ov(i, eri.nocc_full) for i in space)

            m = molecular_response(
                eri.proxy_vind,
                space,
                eri.nocc_full,
                eri.nmo_full,
                True,
                eri.proxy_model,
            )
            ref_m = tuple(i[space_ov[0], :][:, space_ov[1]] for i in ref_m_full)
            testing.assert_allclose(ref_m, m, atol=1e-12)
