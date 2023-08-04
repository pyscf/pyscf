from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf import krhf_slow_supercell, kproxy_supercell
from pyscf.tdscf.common_slow import eig

from test_common import retrieve_m, assert_vectors_close

import unittest
from numpy import testing
import numpy


def density_fitting_hf(x):
    """
    Constructs density-fitting (Gamma-point) Hartree-Fock objects.
    Args:
        x (Cell): the supercell;

    Returns:
        The DF-HF object.
    """
    return KRHF(x).density_fit()


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

        cls.model_krhf = model_krhf = KRHF(cell).density_fit()
        model_krhf.kernel()

        cls.td_model_krhf = td_model_krhf = KTDHF(model_krhf)
        td_model_krhf.nroots = 5
        td_model_krhf.kernel()

        cls.ref_m_krhf = retrieve_m(td_model_krhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = kproxy_supercell.PhysERI(self.model_krhf, "hf", [1, 1, 1], density_fitting_hf)
        m = e.tdhf_full_form()
        testing.assert_allclose(self.ref_m_krhf, m, atol=1e-14)
        vals, vecs = eig(m, nroots=self.td_model_krhf.nroots)
        testing.assert_allclose(vals, self.td_model_krhf.e, atol=1e-5)

    def test_class(self):
        """Tests container behavior."""
        model = kproxy_supercell.TDProxy(self.model_krhf, "hf", [1, 1, 1], density_fitting_hf)
        model.nroots = self.td_model_krhf.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_krhf.e, atol=1e-5)
        assert_vectors_close(model.xy, numpy.array(self.td_model_krhf.xy), atol=1e-12)


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (supercell proxy) @2kp vs reference (krhf_supercell_slow)."""
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
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        # Add random phases
        numpy.random.seed(0)
        for i in model_krhf.mo_coeff:
            i *= numpy.exp(2.j * numpy.pi * numpy.random.rand(i.shape[1]))[numpy.newaxis, :]

        # The slow supercell KTDHF
        cls.td_model_krhf = td_model_krhf = krhf_slow_supercell.TDRHF(model_krhf)
        td_model_krhf.kernel()
        cls.ref_m = td_model_krhf.eri.tdhf_full_form()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests ERI."""
        e = kproxy_supercell.PhysERI(self.model_krhf, "hf", [self.k, 1, 1], density_fitting_hf)
        m = e.tdhf_full_form()

        # Test matrix vs ref
        testing.assert_allclose(m, self.ref_m, atol=1e-11)

        # Test transformations
        testing.assert_allclose(
            e.model_super.supercell_rotation.dot(e.model_super.supercell_inv_rotation).toarray(),
            numpy.eye(e.model_super.supercell_rotation.shape[0]),
        )

    def test_class(self):
        """Tests container behavior."""
        model = kproxy_supercell.TDProxy(self.model_krhf, "hf", [self.k, 1, 1], density_fitting_hf)
        model.nroots = self.td_model_krhf.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_krhf.e, atol=1e-5)
        # Test real
        testing.assert_allclose(model.e.imag, 0, atol=1e-8)

        nocc = nvirt = 4
        testing.assert_equal(model.xy.shape, (len(model.e), 2, self.k, self.k, nocc, nvirt))

        # Test only non-degenerate roots
        d = abs(model.e[1:] - model.e[:-1]) < 1e-8
        d = numpy.logical_or(numpy.concatenate(([False], d)), numpy.concatenate((d, [False])))
        d = numpy.logical_not(d)
        assert_vectors_close(self.td_model_krhf.xy[d], model.xy[d], atol=1e-5)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (supercell proxy) @3kp vs reference (krhf_supercell_slow)."""
    k = 3


class FrozenTest(unittest.TestCase):
    """Tests frozen behavior."""
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
        # cell.basis = 'sto-3g'
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
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests ERI."""
        for frozen in (1, [0, 1], [0, -1]):
            try:
                e = kproxy_supercell.PhysERI(self.model_krhf, "hf", [self.k, 1, 1], density_fitting_hf, frozen=frozen)
                m = e.tdhf_full_form()

                ref_e = krhf_slow_supercell.PhysERI4(self.model_krhf, frozen=frozen)
                ref_m = ref_e.tdhf_full_form()

                # Test matrix vs ref
                testing.assert_allclose(m, ref_m, atol=1e-11)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise


class FrozenTest3(FrozenTest):
    """Tests frozen behavior K=3."""
    k = 3
