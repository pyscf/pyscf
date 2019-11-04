from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRKS, KRHF
from pyscf.pbc.tdscf import KTDDFT, KTDHF
from pyscf.pbc.tdscf.proxy import PhysERI, TDProxy

from test_common import retrieve_m, assert_vectors_close

import unittest
from numpy import testing


class DiamondTestGamma(unittest.TestCase):
    """Compare this (Gamma proxy) vs reference (pyscf)."""
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

        cls.model_rks = model_rks = KRKS(cell)
        model_rks.kernel()

        cls.td_model_rks = td_model_rks = KTDDFT(model_rks)
        td_model_rks.nroots = 5
        td_model_rks.kernel()

        cls.ref_m = retrieve_m(td_model_rks)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rks
        del cls.model_rks
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = PhysERI(self.model_rks, "dft")
        m = e.tdhf_full_form()

        # Test matrix vs pyscf
        testing.assert_allclose(self.ref_m, m, atol=1e-14)

    def test_class(self):
        """Tests container behavior."""
        model = TDProxy(self.model_rks, "dft")
        model.nroots = self.td_model_rks.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rks.e, atol=1e-12)
        assert_vectors_close(model.xy, self.td_model_rks.xy, atol=1e-12)
        # Test real
        testing.assert_allclose(model.e.imag, 0, atol=1e-8)


class DiamondHFTestGamma(unittest.TestCase):
    """Compare this (Gamma proxy) vs reference (pyscf), Hartree-Fock setup."""
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

        cls.model_rhf = model_rhf = KRHF(cell).density_fit()
        model_rhf.kernel()

        cls.td_model_rhf = td_model_rhf = KTDHF(model_rhf)
        td_model_rhf.nroots = 5
        td_model_rhf.kernel()

        cls.ref_m = retrieve_m(td_model_rhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rhf
        del cls.model_rhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        e = PhysERI(self.model_rhf, "hf")
        m = e.tdhf_full_form()

        # Test matrix vs pyscf
        testing.assert_allclose(self.ref_m, m, atol=1e-14)

    def test_class(self):
        """Tests container behavior."""
        model = TDProxy(self.model_rhf, "hf")
        model.nroots = self.td_model_rhf.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_rhf.e, atol=1e-12)
        assert_vectors_close(model.xy, self.td_model_rhf.xy, atol=1e-12)
        # Test real
        testing.assert_allclose(model.e.imag, 0, atol=1e-8)
