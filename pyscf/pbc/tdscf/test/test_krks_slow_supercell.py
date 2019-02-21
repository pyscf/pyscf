from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RKS, KRKS
from pyscf.pbc.tdscf import TDDFT, KTDDFT
from pyscf.pbc.tdscf import krks_slow_supercell, rks_slow
from pyscf.pbc.tools.pbc import super_cell
from pyscf.tdscf.common_slow import eig

from test_common import retrieve_m, retrieve_m_hf, adjust_mf_phase, ov_order, assert_vectors_close, tdhf_frozen_mask

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
    """Compare this (supercell_slow) @Gamma vs reference."""
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
        e = krks_slow_supercell.PhysERI(self.model_krks, [1, 1, 1], KRKS)
        m = e.tdhf_full_form()
        testing.assert_allclose(self.ref_m_krhf, m, atol=1e-14)
        vals, vecs = eig(m, nroots=self.td_model_krks.nroots)
        testing.assert_allclose(vals, self.td_model_krks.e, atol=1e-5)

    def test_class(self):
        """Tests container behavior."""
        model = krks_slow_supercell.TDRKS(self.model_krks, [1, 1, 1], KRKS)
        model.nroots = self.td_model_krks.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_krks.e, atol=1e-5)
        assert_vectors_close(model.xy, numpy.array(self.td_model_krks.xy), atol=1e-12)


class DiamondTestShiftedGamma(unittest.TestCase):
    """Compare this (supercell_slow) @non-Gamma 1kp vs rhf_slow."""
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
        model = krks_slow_supercell.TDRKS(self.model_krks, [1, 1, 1], density_fitting_ks)
        # Shifted k-point grid is not TRS: an exception should be raised
        with self.assertRaises(RuntimeError):
            model.kernel()


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (supercell_slow) @2kp vs pyscf (supercell), roots only."""
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

        # The Gamma-point reference
        cls.model_rks = model_rks = KRKS(super_cell(cell, [cls.k, 1, 1]))
        model_rks.conv_tol = 1e-14
        model_rks.kernel()

        # Ensure orbitals are real
        testing.assert_allclose(model_rks.mo_coeff[0].imag, 0, atol=1e-8)
        model_rks.mo_coeff[0] = model_rks.mo_coeff[0].real

        # K-points
        cls.model_krks = model_krks = KRKS(cell, k)
        model_krks.conv_tol = 1e-14
        model_krks.kernel()

        ke = numpy.concatenate(model_krks.mo_energy)
        ke.sort()

        # Make sure mo energies are the same
        testing.assert_allclose(model_rks.mo_energy[0], ke, atol=1e-5)

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
        model = krks_slow_supercell.TDRKS(self.model_krks, [self.k, 1, 1], KRKS)
        model.nroots = self.td_model_rks.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e.squeeze(), self.td_model_rks.e, atol=1e-5)
        # Test real
        testing.assert_allclose(model.e.imag, 0, atol=1e-8)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (supercell_slow) @3kp vs pyscf (supercell), roots only."""
    k = 3
