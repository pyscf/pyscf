from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRKS
from pyscf.pbc.tdscf import kproxy, krhf_slow, kproxy_supercell

from test_common import assert_vectors_close
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
    """Compare this (k-proxy) @2kp vs reference (krks_supercell_slow)."""
    k = 2
    call_ratios = (1./4, 1./4)

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

        cls.td_model_rks_supercell = kproxy_supercell.TDProxy(model_krks, "dft", [cls.k, 1, 1], KRKS)
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
        model = kproxy.TDProxy(self.model_krks, "dft", [self.k, 1, 1], KRKS)
        model.nroots = self.td_model_rks_supercell.nroots
        assert not model.fast
        for k in range(self.k):
            model.kernel(k)
            try:
                testing.assert_allclose(model.eri.proxy_vind.ratio, self.call_ratios[k])
            except Exception:
                print("During efficiency check @k={:d} the following exception occurred:".format(k))
                raise
            model.eri.proxy_vind.reset()
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
                # r = k2k(r1, r2)
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
    """Compare this (k-proxy) @3kp vs supercell reference (pyscf)."""
    k = 3
    call_ratios = (5./18, 4./9, 4./9)
