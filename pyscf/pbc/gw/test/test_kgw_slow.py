from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import krhf_slow_supercell as std, krhf_slow as ktd
from pyscf.pbc.gw import kgw_slow_supercell as sgw, kgw_slow as kgw
from pyscf.pbc.tdscf.krhf_slow import get_block_k_ix

from test_common import adjust_mf_phase, adjust_td_phase, ov_order

import unittest
from numpy import testing
import numpy


def tdm_k2s(imds):
    tdm = imds.tdm
    e = imds.td_e
    nk = len(tdm)

    orb_order = []
    offsets = numpy.cumsum((0,) + imds.eri.nmo[:-1])
    for k in range(nk):
        orb_order.append(numpy.arange(offsets[k], offsets[k] + imds.eri.nocc[k]))
    for k in range(nk):
        orb_order.append(numpy.arange(offsets[k] + imds.eri.nocc[k], offsets[k] + imds.eri.nmo[k]))
    orb_order = numpy.concatenate(orb_order)

    tdm = numpy.array(tdm)
    # Dims: k_transfer, (xy), ki, roots, i, a
    result = []
    for k_transfer, tdm_t in enumerate(tdm):
        fw, bw, _, _ = get_block_k_ix(imds.eri, k_transfer)
        x, y = tdm_t

        # X
        x_s = []
        for k1 in range(nk):
            x_s.append([])
            for k2 in range(nk):
                if k2 == bw[k1]:
                    x_s[-1].append(x[k1])
                else:
                    x_s[-1].append(numpy.zeros_like(x[k1]))
        x_s = numpy.array(x_s).transpose(2, 0, 3, 1, 4)
        s = x_s.shape
        x_s = x_s.reshape((s[0], s[1]*s[2], s[3]*s[4]))

        # Y
        y_s = []
        for k1 in range(nk):
            y_s.append([])
            for k2 in range(nk):
                if k2 == fw[k1]:
                    y_s[-1].append(y[k1])
                else:
                    y_s[-1].append(numpy.zeros_like(y[k1]))
        y_s = numpy.array(y_s).transpose(2, 0, 3, 1, 4)
        s = y_s.shape
        y_s = y_s.reshape((s[0], s[1]*s[2], s[3]*s[4]))

        result.append([x_s, y_s])

    result = numpy.array(result)
    result = result.transpose(0, 2, 1, 3, 4)
    s = result.shape
    result = result.reshape(s[0] * s[1], s[2], s[3], s[4])
    # k_transfer, (xy), o, v
    order = numpy.argsort(numpy.concatenate(tuple(e[i] for i in range(nk))))
    # return result[order, ...]
    return result[numpy.ix_(order, (0, 1), orb_order, orb_order)]


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (k-version) @2kp vs rhf_slow_supercell."""
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
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        ke = numpy.concatenate(model_krhf.mo_energy)
        ke.sort()

        # Make sure no degeneracies are present
        testing.assert_array_less(1e-4, ke[1:] - ke[:-1])

        # TD
        cls.td_model_srhf = td_model_srhf = std.TDRHF(model_krhf)
        td_model_srhf.kernel()

        cls.td_model_krhf = td_model_krhf = ktd.TDRHF(model_krhf)
        td_model_krhf.kernel()

        # adjust_td_phase(td_model_srhf, td_model_krhf)

        # GW
        cls.gw = sgw.GW(td_model_srhf)
        cls.kgw = kgw.GW(td_model_krhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.kgw
        del cls.gw
        del cls.td_model_krhf
        del cls.td_model_srhf
        del cls.model_krhf
        del cls.cell

    def test_imds(self):

        testing.assert_allclose(
            tuple(self.gw.imds.get_rhs(i, components=True) for i in zip(*self.gw.imds.entire_space)),
            tuple(self.kgw.imds.get_rhs(i, components=True) for i in zip(*self.kgw.imds.entire_space)),
            atol=1e-8
        )

        tdm = tdm_k2s(self.kgw.imds)

        testing.assert_allclose(
            self.gw.imds.tdm,
            tdm,
            atol=1e-10
        )

    def __get_sigma__(self, imds, omega=(0, 1), basis_k=None, basis_orb=None):
        default_bk, default_bo = imds.entire_space
        if basis_k is None:
            basis_k = default_bk
        if basis_orb is None:
            basis_orb = default_bo
        result = numpy.zeros((len(omega), len(basis_k), len(basis_orb)), dtype=complex)
        for i_o, o in enumerate(omega):
            for i_k, k in enumerate(basis_k):
                for i_orb, orb in enumerate(basis_orb):
                    result[i_o, i_k, i_orb] = imds.get_sigma_element(o, (k, orb), 1e-3)
        return result

    def test_sigma(self):
        s1 = self.__get_sigma__(self.gw.imds)
        s2 = self.__get_sigma__(self.kgw.imds)
        testing.assert_allclose(s1, s2)

    def test_class(self):
        """Tests default eig kernel behavior."""
        self.gw.eta = self.kgw.eta = 1e-6
        self.gw.kernel()
        self.kgw.kernel()

        testing.assert_allclose(self.gw.mo_energy, self.kgw.mo_energy)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (k-version) @3kp vs rhf_slow_supercell."""
    k = 3
    k_c = (.1, 0, 0)
