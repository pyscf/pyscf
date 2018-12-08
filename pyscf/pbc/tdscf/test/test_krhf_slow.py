from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf import krhf_slow as ktd, krhf_slow_supercell as std, krhf_slow_gamma as gtd
from pyscf.pbc.tools.pbc import super_cell

import unittest
from numpy import testing
import numpy

from test_common import retrieve_m, make_mf_phase_well_defined, unphase, ov_order


def k2k(*indexes):
    result = []
    offset = 0
    for i in indexes:
        result.append(offset + (i + numpy.arange(len(i)) * len(i)))
        offset += len(i) * len(i)
    return numpy.concatenate(result)


class DiamondTest(unittest.TestCase):
    """Compare this (krhf_slow) @2kp@Gamma vs `krhf_slow_gamma` and `krhf_slow_supercell`."""
    k = 2
    k_c = (0, 0, 0)
    test8 = True

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

        # Gamma
        cls.ref_m_gamma = m = gtd.build_matrix(gtd.PhysERI4(model_krhf))
        cls.ref_e_gamma, v = gtd.eig(m)
        cls.ref_v_gamma = gtd.vector_to_amplitudes(v, gtd.k_nocc(model_krhf), model_krhf.mo_coeff[0].shape[0])

        # Supercell
        cls.ref_m = m = std.build_matrix(std.PhysERI4(model_krhf))
        cls.ref_e, v = std.eig(m)
        cls.ref_v = std.vector_to_amplitudes(v, std.k_nocc(model_krhf), model_krhf.mo_coeff[0].shape[0])

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krhf
        del cls.cell

    def test_eri_gamma(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            if eri is not ktd.PhysERI8 or self.test8:
                try:

                    e = eri(self.model_krhf)
                    m = ktd.build_matrix(e, 0)

                    testing.assert_allclose(self.ref_m_gamma, m, atol=1e-12)
                except Exception:
                    print("When testing {} the following exception occurred:".format(eri))
                    raise

    def test_eig_kernel_gamma(self):
        """Tests default eig kernel behavior."""
        vals, vecs = ktd.kernel(self.model_krhf, 0, driver='eig')
        testing.assert_allclose(vals, self.ref_e_gamma, atol=1e-7)
        nocc = nvirt = 4
        testing.assert_equal(vecs.shape, (len(vals), 2, self.k, nocc, nvirt))
        testing.assert_allclose(*unphase(vecs, self.ref_v_gamma), atol=1e-6)

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        o = v = 4
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            if eri is not ktd.PhysERI8 or self.test8:
                try:
                    e = eri(self.model_krhf)

                    s = (2 * self.k * self.k, 2 * self.k * self.k, o*v, o*v)
                    m = numpy.zeros(s, dtype=complex)

                    for k in range(self.k):
                        # Prepare indexes
                        r1, r2, c1, c2 = ktd.get_block_k_ix(e, k)

                        r = k2k(r1, r2)
                        c = k2k(c1, c2)

                        # Build matrix
                        _m = ktd.build_matrix(e, k)

                        # Assign the submatrix
                        m[numpy.ix_(r, c)] = _m.reshape((2*self.k, o*v, 2*self.k, o*v)).transpose(0, 2, 1, 3)

                    m = m.transpose(0, 2, 1, 3).reshape(self.ref_m.shape)
                    testing.assert_allclose(self.ref_m, m, atol=1e-12)
                except Exception:
                    print("When testing {} the following exception occurred:".format(eri))
                    raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals = []
        vecs = []
        o = v = 4
        eri = ktd.PhysERI4(self.model_krhf)
        for k in range(self.k):
            va, ve = ktd.kernel(self.model_krhf, k, driver='eig')
            vals.append(va)
            vecs.append(ve)

        # Concatenate everything
        ks = numpy.array(sum(([i] * len(v) for i, v in enumerate(vals)), []))
        vals = numpy.concatenate(vals).real
        vecs = numpy.concatenate(vecs, axis=0)

        # Obtain order
        order = numpy.argsort(vals)

        # Sort
        vals = vals[order]
        vecs = vecs[order]
        ks = ks[order]

        # Verify
        testing.assert_allclose(vals, self.ref_e, atol=1e-7)
        for k in range(self.k):
            # Prepare indexes
            r1, r2, c1, c2 = ktd.get_block_k_ix(eri, k)
            r = k2k(r1, r2)
            c = k2k(c1, c2)

            # Select roots
            selection = ks == k
            vecs_ref = self.ref_v[selection]
            vecs_test = vecs[selection]

            vecs_test_padded = numpy.zeros((len(vecs_test), 2 * self.k * self.k, o, v), dtype=vecs_test.dtype)
            vecs_test_padded[:, c] = vecs_test.reshape((len(vecs_test), 2 * self.k, o, v))
            vecs_test_padded = vecs_test_padded.reshape(vecs_ref.shape)

            testing.assert_equal(vecs_test.shape, (self.k * o * v, 2, self.k, o, v))
            testing.assert_equal(vecs_test_padded.shape, (self.k * o * v, 2, self.k, self.k, o, v))
            testing.assert_allclose(*unphase(vecs_test_padded, vecs_ref), atol=1e-7)


class DiamondTest3(DiamondTest):
    """Compare this (krhf_slow) @3kp@Gamma vs vs `krhf_slow_supercell`."""
    k = 3
    k_c = (0.1, 0, 0)
    test8 = False
