from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRKS
from pyscf.pbc.tdscf import KTDDFT
from pyscf.pbc.tdscf import kproxy_supercell
from pyscf.tdscf.common_slow import eig, format_frozen_k, format_frozen_mol

from test_common import retrieve_m, ov_order, assert_vectors_close

import unittest
from numpy import testing
import numpy


def density_fitting_ks(x):
    """
    Constructs density-fitting (Gamma-point) Kohn-Sham objects.
    Args:
        x (Cell): the supercell;

    Returns:
        The DF-KS object.
    """
    return KRKS(x).density_fit()


class DiamondTestGamma(unittest.TestCase):
    """Compare this (supercell proxy) @Gamma vs reference (pyscf)."""
    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        cls.model_krks = model_krks = KRKS(cell)
        model_krks.kernel()

        cls.td_model_krks = td_model_krks = KTDDFT(model_krks)
        td_model_krks.nroots = 5
        td_model_krks.kernel()

        cls.ref_m_krhf = retrieve_m(td_model_krks)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krks
        del cls.model_krks
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = kproxy_supercell.PhysERI(self.model_krks, "dft", [1, 1, 1], KRKS)
        m = e.tdhf_full_form()
        testing.assert_allclose(self.ref_m_krhf, m, atol=1e-14)
        vals, vecs = eig(m, nroots=self.td_model_krks.nroots)
        testing.assert_allclose(vals, self.td_model_krks.e, atol=1e-5)

    def test_class(self):
        """Tests container behavior."""
        model = kproxy_supercell.TDProxy(self.model_krks, "dft", [1, 1, 1], KRKS)
        model.nroots = self.td_model_krks.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_krks.e, atol=1e-5)
        assert_vectors_close(model.xy, numpy.array(self.td_model_krks.xy), atol=1e-12)


class DiamondTestShiftedGamma(unittest.TestCase):
    """Test this (supercell proxy) @non-Gamma: exception."""
    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        k = cell.get_abs_kpts((.1, .2, .3))

        cls.model_krks = model_krks = KRKS(cell, kpts=k).density_fit()
        model_krks.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krks
        del cls.cell

    def test_class(self):
        """Tests container behavior."""
        model = kproxy_supercell.TDProxy(self.model_krks, "dft", [1, 1, 1], density_fitting_ks)
        # Shifted k-point grid is not TRS: an exception should be raised
        with self.assertRaises(RuntimeError):
            model.kernel()


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (supercell proxy) @2kp vs supercell reference (pyscf)."""
    k = 2

    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        k = cell.make_kpts([cls.k, 1, 1])

        # K-points
        cls.model_krks = model_krks = KRKS(cell, k)
        model_krks.conv_tol = 1e-14
        model_krks.kernel()

        # Supercell reference
        cls.model_rks = model_rks = kproxy_supercell.k2s(model_krks, [cls.k, 1, 1], KRKS)

        # Ensure orbitals are real
        testing.assert_allclose(model_rks.mo_coeff[0].imag, 0, atol=1e-8)

        cls.ov_order = ov_order(model_krks)

        # The Gamma-point TD
        cls.td_model_rks = td_model_rks = KTDDFT(model_rks)
        td_model_rks.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rks
        del cls.model_krks
        del cls.model_rks
        del cls.cell

    def test_class(self):
        """Tests container behavior."""
        model = kproxy_supercell.TDProxy(self.model_krks, "dft", [self.k, 1, 1], KRKS)
        model.nroots = self.td_model_rks.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rks.e, atol=1e-5)
        if self.k == 2:
            vecs = model.xy.reshape(len(model.xy), -1)[:, self.ov_order]
            # A loose tolerance here because of a low plane-wave cutoff
            assert_vectors_close(vecs, numpy.array(self.td_model_rks.xy).squeeze(), atol=1e-3)
        # Test real
        testing.assert_allclose(model.e.imag, 0, atol=1e-8)

    def test_raw_response(self):
        """Tests the `supercell_response` and whether it slices output properly."""
        eri = kproxy_supercell.PhysERI(self.model_krks, "dft", [self.k, 1, 1], KRKS)
        ref_m_full = eri.proxy_response()

        # Test single
        for frozen in (1, [0, -1]):
            space = format_frozen_k(frozen, eri.nmo_full[0], len(eri.nmo_full))
            space_o = numpy.concatenate(tuple(i[:j] for i, j in zip(space, eri.nocc_full)))
            space_v = numpy.concatenate(tuple(i[j:] for i, j in zip(space, eri.nocc_full)))
            space_ov = numpy.logical_and(space_o[:, numpy.newaxis], space_v[numpy.newaxis, :]).reshape(-1)

            m = kproxy_supercell.supercell_response(
                eri.proxy_vind,
                numpy.concatenate(space),
                eri.nocc_full,
                eri.nmo_full,
                True,
                eri.model_super.supercell_inv_rotation,
                eri.proxy_model,
            )
            ref_m = tuple(i[space_ov][:, space_ov] for i in ref_m_full)
            testing.assert_allclose(ref_m, m, atol=1e-12)

        # Test pair
        for frozen in ((1, 3), ([1, -2], [0, -1])):
            space = tuple(format_frozen_k(i, eri.nmo_full[0], len(eri.nmo_full)) for i in frozen)
            space_o = tuple(numpy.concatenate(tuple(i[:j] for i, j in zip(s, eri.nocc_full))) for s in space)
            space_v = tuple(numpy.concatenate(tuple(i[j:] for i, j in zip(s, eri.nocc_full))) for s in space)
            space_ov = tuple(
                numpy.logical_and(i[:, numpy.newaxis], j[numpy.newaxis, :]).reshape(-1)
                for i, j in zip(space_o, space_v)
            )

            m = kproxy_supercell.supercell_response(
                eri.proxy_vind,
                (numpy.concatenate(space[0]), numpy.concatenate(space[1])),
                eri.nocc_full,
                eri.nmo_full,
                True,
                eri.model_super.supercell_inv_rotation,
                eri.proxy_model,
            )
            ref_m = tuple(i[space_ov[0]][:, space_ov[1]] for i in ref_m_full)
            testing.assert_allclose(ref_m, m, atol=1e-12)

    def test_raw_response_ov(self):
        """Tests the `molecular_response` and whether it slices output properly."""
        eri = kproxy_supercell.PhysERI(self.model_krks, "dft", [self.k, 1, 1], KRKS)
        ref_m_full = eri.proxy_response()
        s = sum(eri.nocc_full) * (sum(eri.nmo_full) - sum(eri.nocc_full))
        ref_m_full = tuple(i.reshape((s, s)) for i in ref_m_full)

        # Test single
        for frozen in (1, [0, -1]):
            space_ov = format_frozen_mol(frozen, s)

            m = kproxy_supercell.supercell_response_ov(
                eri.proxy_vind,
                space_ov,
                eri.nocc_full,
                eri.nmo_full,
                True,
                eri.model_super.supercell_inv_rotation,
                eri.proxy_model,
            )
            ref_m = tuple(i[space_ov, :][:, space_ov] for i in ref_m_full)
            testing.assert_allclose(ref_m, m, atol=1e-12)

        # Test pair
        for frozen in ((1, 3), ([1, -2], [0, -1])):
            space_ov = tuple(format_frozen_mol(i, s) for i in frozen)

            m = kproxy_supercell.supercell_response_ov(
                eri.proxy_vind,
                space_ov,
                eri.nocc_full,
                eri.nmo_full,
                True,
                eri.model_super.supercell_inv_rotation,
                eri.proxy_model,
            )
            ref_m = tuple(i[space_ov[0], :][:, space_ov[1]] for i in ref_m_full)
            testing.assert_allclose(ref_m, m, atol=1e-12)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (supercell proxy) @3kp vs supercell reference (pyscf)."""
    k = 3
