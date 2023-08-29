import os
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf import krhf_slow_gamma as ktd

import unittest
from numpy import testing
import numpy

from test_common import retrieve_m, retrieve_m_hf, assert_vectors_close, tdhf_frozen_mask


class DiamondTest(unittest.TestCase):
    """Compare this (krhf_slow_gamma) @2kp@Gamma vs reference (pyscf)."""
    k = 2
    k_c = (0, 0, 0)

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

        k = cell.make_kpts([cls.k, 1, 1], scaled_center=cls.k_c)

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.kernel()

        cls.td_model_krhf = td_model_krhf = KTDHF(model_krhf)
        td_model_krhf.kernel()

        cls.ref_m = retrieve_m(td_model_krhf)
        cls.ref_e = td_model_krhf.e

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            # Note that specific combintation of k-points results in real orbitals and allows testing PhysERI8
            try:
                e = eri(self.model_krhf)
                m = e.tdhf_full_form()

                # Test matrix vs ref
                testing.assert_allclose(m, retrieve_m_hf(e), atol=1e-11)

                # Test matrix vs pyscf
                testing.assert_allclose(self.ref_m, m, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_class(self):
        """Tests container behavior."""
        model = ktd.TDRHF(self.model_krhf)
        model.nroots = self.td_model_krhf.nroots
        assert not model.fast
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_krhf.e, atol=1e-5)
        nocc = nvirt = 4
        testing.assert_equal(model.xy.shape, (len(model.e), 2, self.k, nocc, nvirt))
        assert_vectors_close(model.xy, numpy.array(self.td_model_krhf.xy), atol=1e-2)


class FrozenTest(unittest.TestCase):
    """Tests frozen behavior."""
    k = 2
    k_c = (0, 0, 0)
    df_file = os.path.realpath(os.path.join(__file__, "..", "frozen_test_cderi.h5"))

    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = 'sto-3g'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        k = cell.make_kpts([cls.k, 1, 1], scaled_center=cls.k_c)

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        # model_krhf.with_df._cderi_to_save = cls.df_file
        model_krhf.with_df._cderi = cls.df_file
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        cls.td_model_krhf = model_ktd = ktd.TDRHF(model_krhf)
        model_ktd.nroots = 5
        model_ktd.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_class(self):
        """Tests container behavior (frozen vs non-frozen)."""
        for frozen in (1, [0, 1]):
            try:
                model = ktd.TDRHF(self.model_krhf, frozen=frozen)
                model.nroots = self.td_model_krhf.nroots
                model.kernel()
                mask_o, mask_v = tdhf_frozen_mask(model.eri, kind="o,v")
                testing.assert_allclose(model.e, self.td_model_krhf.e, atol=1e-3)
                assert_vectors_close(model.xy,
                    numpy.array(self.td_model_krhf.xy)[..., mask_o, :][..., mask_v], atol=1e-2)

            except Exception:
                print("When testing class with frozen={} the following exception occurred:".format(repr(frozen)))
                raise
