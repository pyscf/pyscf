from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf import krhf_slow, kproxy
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
    """Compare this (k-proxy) @Gamma vs reference (pyscf), Hartree-Fock setup."""
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
        e = kproxy.PhysERI(self.model_krhf, "hf", [1, 1, 1], density_fitting_hf)
        m = e.tdhf_full_form(0)
        testing.assert_allclose(self.ref_m_krhf, m, atol=1e-14)
        vals, vecs = eig(m, nroots=self.td_model_krhf.nroots)
        testing.assert_allclose(vals, self.td_model_krhf.e, atol=1e-5)

    def test_class(self):
        """Tests container behavior."""
        model = kproxy.TDProxy(self.model_krhf, "hf", [1, 1, 1], density_fitting_hf)
        model.nroots = self.td_model_krhf.nroots
        assert not model.fast
        model.kernel(0)
        testing.assert_allclose(model.e[0], self.td_model_krhf.e, atol=1e-5)
        assert_vectors_close(model.xy[0], numpy.array(self.td_model_krhf.xy), atol=1e-12)


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (k-proxy) @2kp vs reference (krhf_slow), Hartree-Fock setup."""
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

        # The slow supercell KTDHF
        cls.td_model_krhf = td_model_krhf = krhf_slow.TDRHF(model_krhf)
        td_model_krhf.kernel()
        cls.ref_m = tuple(td_model_krhf.eri.tdhf_full_form(i) for i in range(cls.k))

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests ERI."""
        e = kproxy.PhysERI(self.model_krhf, "hf", [self.k, 1, 1], density_fitting_hf)
        for i in range(self.k):
            try:
                m = e.tdhf_full_form(i)

                # Test matrix vs ref
                testing.assert_allclose(m, self.ref_m[i], atol=1e-11)
            except Exception:
                print("When testing eri @k={:d} the following exception occurred:".format(i))
                raise

    def test_class(self):
        """Tests container behavior."""
        model = kproxy.TDProxy(self.model_krhf, "hf", [self.k, 1, 1], density_fitting_hf)
        model.nroots = self.td_model_krhf.nroots
        assert not model.fast
        model.kernel()
        for i in range(self.k):
            try:
                testing.assert_allclose(model.e[i], self.td_model_krhf.e[i], atol=1e-5)
                # Test real
                testing.assert_allclose(model.e[i].imag, 0, atol=1e-8)

                nocc = nvirt = 4
                testing.assert_equal(model.xy[i].shape, (len(model.e[i]), 2, self.k, nocc, nvirt))

                assert_vectors_close(self.td_model_krhf.xy[i], model.xy[i], atol=1e-5)

            except Exception:
                print("When testing model @k={:d} the following exception occurred:".format(i))
                raise


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (k-proxy) @3kp vs reference (krhf_slow), Hartree-Fock setup."""
    k = 3
