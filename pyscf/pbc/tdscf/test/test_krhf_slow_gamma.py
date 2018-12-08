from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf import krhf_slow_gamma as ktd
from pyscf.pbc.tdscf import krhf_slow_supercell as std

import unittest
from numpy import testing
import numpy

from test_common import retrieve_m, unphase


class DiamondTest(unittest.TestCase):
    """Compare this (krhf_slow_gamma) @2kp@Gamma vs reference."""
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
                m = ktd.build_matrix(e)

                testing.assert_allclose(self.ref_m, m, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = ktd.kernel(self.model_krhf, driver='eig', nroots=self.td_model_krhf.nroots)
        testing.assert_allclose(vals, self.ref_e, atol=1e-7)
        nocc = nvirt = 4
        testing.assert_equal(vecs.shape, (len(vals), 2, self.k, nocc, nvirt))
        vecs_ref = numpy.asarray(self.td_model_krhf.xy)
        testing.assert_equal(vecs_ref.shape, (len(vals), 2, self.k, nocc, nvirt))
        try:
            testing.assert_allclose(*unphase(vecs, vecs_ref), atol=1e-2)
        except Exception:
            # TODO
            print("This is a known bug: vector #1 from davidson is way off.")
            raise
