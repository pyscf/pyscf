from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RKS, KRKS
from pyscf.pbc.tdscf import TDDFT, KTDDFT
from pyscf.pbc.tdscf import krks_slow, rks_slow, krhf_slow, krks_slow_supercell
from pyscf.pbc.tdscf.krks_slow import PhysERI
from pyscf.pbc.tools.pbc import super_cell
from pyscf.tdscf.common_slow import eig, ab2full, format_frozen_mol

from test_common import retrieve_m, retrieve_m_hf, adjust_mf_phase, ov_order, assert_vectors_close, tdhf_frozen_mask
from test_krhf_slow import k2k

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


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (krks_slow) @2kp vs reference (krks_supercell_slow)."""
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

        cls.model_krks = model_krks = KRKS(cell, k)
        model_krks.kernel()

        cls.td_model_rks_supercell = krks_slow_supercell.TDRKS(model_krks, [cls.k, 1, 1], KRKS)
        cls.td_model_rks_supercell.kernel()
        cls.ref_m_supercell = cls.td_model_rks_supercell.eri.tdhf_full_form()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rks_supercell
        del cls.model_krks
        del cls.cell

    def test_class(self):
        """Tests container behavior."""
        model = krks_slow.TDRKS(self.model_krks, [self.k, 1, 1], KRKS)
        model.nroots = self.td_model_rks_supercell.nroots
        assert not model.fast
        model.kernel()
        o = v = 4

        # Concatenate everything
        ks = numpy.array(sum(([i] * len(model.e[i]) for i in range(self.k)), []))
        vals = numpy.concatenate(tuple(model.e[i] for i in range(self.k))).real
        vecs = numpy.concatenate(tuple(model.xy[i] for i in range(self.k)), axis=0)

        # Obtain order
        order = numpy.argsort(vals)

        # Sort
        vals = vals[order]
        vecs = vecs[order]
        ks = ks[order]

        # Verify
        testing.assert_allclose(vals, self.td_model_rks_supercell.e, atol=1e-7)
        # Following makes sense only for non-degenerate roots
        if self.k < 3:
            for k in range(self.k):
                # Prepare indexes
                r1, r2, c1, c2 = krhf_slow.get_block_k_ix(model.eri, k)
                r = k2k(r1, r2)
                c = k2k(c1, c2)

                # Select roots
                selection = ks == k
                vecs_ref = self.td_model_rks_supercell.xy[selection]
                vecs_test = vecs[selection]

                vecs_test_padded = numpy.zeros((len(vecs_test), 2 * self.k * self.k, o, v), dtype=vecs_test.dtype)
                vecs_test_padded[:, c] = vecs_test.reshape((len(vecs_test), 2 * self.k, o, v))
                vecs_test_padded = vecs_test_padded.reshape(vecs_ref.shape)

                testing.assert_equal(vecs_test.shape, (self.k * o * v, 2, self.k, o, v))
                testing.assert_equal(vecs_test_padded.shape, (self.k * o * v, 2, self.k, self.k, o, v))
                assert_vectors_close(vecs_test_padded, vecs_ref, atol=1e-7)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (krks_supercell_slow) @3kp vs supercell reference (pyscf)."""
    k = 3
