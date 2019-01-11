from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RHF, KRHF
from pyscf.pbc.tdscf import rhf_slow as td, krhf_slow_supercell as ktd
from pyscf.pbc.gw import gw_slow as gw, kgw_slow_supercell as kgw
from pyscf.pbc.tools.pbc import super_cell

from test_common import adjust_mf_phase, adjust_td_phase

import unittest
from numpy import testing
import numpy


def ov_order_supercell(imds):
    nocc = imds.eri.nocc[0]
    nvirt = imds.eri.nmo[0] - nocc
    o = numpy.argsort(imds.o)
    o_p = o % nocc
    o_k = o // nocc
    v = numpy.argsort(imds.v)
    v_p = v % nvirt
    v_k = v // nvirt
    return numpy.concatenate([o_k, v_k]), numpy.concatenate([o_p, v_p + nocc]), numpy.concatenate([o_k * nocc + o_p, v_k * nvirt + v_p + nocc * imds.nk])


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (supercell_slow) @2kp vs rhf_slow (2x1x1 supercell)."""
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

        # The Gamma-point reference
        cls.model_rhf = model_rhf = RHF(super_cell(cell, [cls.k, 1, 1]), kpt=k[0]).density_fit()
        model_rhf.conv_tol = 1e-14
        model_rhf.kernel()

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        adjust_mf_phase(model_rhf, model_krhf)

        ke = numpy.concatenate(model_krhf.mo_energy)
        ke.sort()

        # Make sure mo energies are the same
        testing.assert_allclose(model_rhf.mo_energy, ke)

        # Make sure no degeneracies are present
        testing.assert_array_less(1e-4, ke[1:] - ke[:-1])

        # TD
        cls.td_model_rhf = td_model_rhf = td.TDRHF(model_rhf)
        td_model_rhf.kernel()

        cls.td_model_krhf = td_model_krhf = ktd.TDRHF(model_krhf)
        td_model_krhf.kernel()

        adjust_td_phase(td_model_rhf, td_model_krhf)

        # GW
        cls.gw = gw.GW(td_model_rhf)
        cls.kgw = kgw.GW(td_model_krhf)

        cls.order_k, cls.order_p, cls.order = ov_order_supercell(cls.kgw.imds)

        orbs = []
        for k in range(cls.k):
            for o in numpy.arange(2, 6):
                orbs.append(numpy.where(numpy.logical_and(cls.order_k == k, cls.order_p == o))[0][0])
        cls.gw.orbs = numpy.array(orbs)
        cls.kgw.orbs = numpy.arange(2, 6)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.kgw
        del cls.gw
        del cls.td_model_krhf
        del cls.td_model_rhf
        del cls.model_krhf
        del cls.model_rhf
        del cls.cell

    def test_imds(self):

        testing.assert_allclose(
            tuple(self.gw.imds.get_rhs(i, components=True) for i in self.gw.imds.entire_space[0]),
            tuple(self.kgw.imds.get_rhs(i, components=True) for i in zip(self.order_k, self.order_p)),
            atol=1e-8
        )
        testing.assert_allclose(
            self.gw.imds.tdm,
            self.kgw.imds.tdm[numpy.ix_(range(len(self.kgw.imds.td_e)), range(2), self.order, self.order)],
            atol=1e-7
        )

    def test_class(self):
        """Tests default eig kernel behavior."""
        self.gw.eta = self.kgw.eta = 1e-6
        self.gw.kernel()
        self.kgw.kernel()
        testing.assert_allclose(self.gw.mo_energy, self.kgw.mo_energy.reshape(-1))


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (supercell_slow) @3kp vs rhf_slow (3x1x1 supercell)."""
    k = 3
    k_c = (.1, 0, 0)
