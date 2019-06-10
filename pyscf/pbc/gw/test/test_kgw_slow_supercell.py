import os
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RHF, KRHF, KRKS
from pyscf.pbc.tdscf import rhf_slow as td, krhf_slow_supercell as ktd, proxy as tdp, kproxy_supercell as ktdp
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
    """Compare this (kgw_slow_supercell) @2kp vs gw_slow (2x1x1 supercell), HF."""
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
            tuple(self.gw.imds.get_rhs(i) for i in self.gw.imds.entire_space[0]),
            tuple(self.kgw.imds.get_rhs(i) for i in zip(self.order_k, self.order_p)),
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
    """Compare this (kgw_slow_supercell) @3kp vs gw_slow (3x1x1 supercell), HF."""
    k = 3
    k_c = (.1, 0, 0)


class FrozenTest(unittest.TestCase):
    """Tests frozen behavior."""
    k = 2
    k_c = (0, 0, 0)
    df_file = os.path.join(__file__, "..", "frozen_test_cderi.h5")

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
        model_krhf.with_df._cderi = cls.df_file
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        cls.td_model_krhf = td_model_krhf = ktd.TDRHF(model_krhf)
        td_model_krhf.nroots = 10
        td_model_krhf.kernel()

        cls.gw_model_krhf = gw_model_krhf = kgw.GW(td_model_krhf)
        gw_model_krhf.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.gw_model_krhf
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_imds_frozen(self):
        """Tests intermediates: frozen vs non-frozen."""
        frozen = 2
        sample_ref = (0, 2)
        sample_frozen = (0, 0)

        td = ktd.TDRHF(self.model_krhf, frozen=frozen)
        td.nroots = self.td_model_krhf.nroots
        td.kernel()

        adjust_td_phase(self.td_model_krhf, td)

        testing.assert_allclose(self.td_model_krhf.e, td.e, atol=1e-4)
        testing.assert_allclose(self.td_model_krhf.xy[..., 2:, :], td.xy, atol=1e-3)

        gw_frozen = kgw.GW(td)
        gw_frozen.kernel()

        selection = gw_frozen.imds.eri.space
        selection_o = numpy.concatenate(tuple(i[:j] for i, j in zip(selection, self.gw_model_krhf.imds.eri.nocc)))
        selection_v = numpy.concatenate(tuple(i[j:] for i, j in zip(selection, self.gw_model_krhf.imds.eri.nocc)))
        selection = numpy.concatenate([selection_o, selection_v])

        imd_ref = self.gw_model_krhf.imds.tdm[..., selection, :][..., selection]
        testing.assert_allclose(gw_frozen.imds.tdm, imd_ref, atol=1e-4)

        test_energies = numpy.linspace(-2, 3, 300)
        ref_samples = numpy.array(tuple(
            self.gw_model_krhf.imds.get_sigma_element(i, sample_ref, 0.01)
            for i in test_energies
        ))
        frozen_samples = numpy.array(tuple(
            gw_frozen.imds.get_sigma_element(i, sample_frozen, 0.01)
            for i in test_energies
        ))

        testing.assert_allclose(ref_samples, frozen_samples, atol=1e-4)
        testing.assert_allclose(
            self.gw_model_krhf.imds.get_rhs(sample_ref),
            gw_frozen.imds.get_rhs(sample_frozen),
            atol=1e-14,
        )

    def test_class(self):
        """Tests container behavior (frozen vs non-frozen)."""
        model = ktd.TDRHF(self.model_krhf, frozen=2)
        model.nroots = self.td_model_krhf.nroots
        model.kernel()

        gw_model = kgw.GW(model)
        gw_model.kernel()

        testing.assert_allclose(gw_model.mo_energy, self.gw_model_krhf.mo_energy[:, 2:], atol=1e-4)


class DiamondKSTestSupercell2(unittest.TestCase):
    """Compare this (kgw_slow_supercell) @2kp vs gw_slow (2x1x1 supercell), DFT."""
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

        # K-points
        cls.model_krks = model_krks = KRKS(cell, k)
        model_krks.conv_tol = 1e-14
        model_krks.kernel()

        adjust_mf_phase(model_rks, model_krks)

        ke = numpy.concatenate(model_krks.mo_energy)
        ke.sort()

        # Make sure mo energies are the same
        testing.assert_allclose(model_rks.mo_energy[0], ke)

        # TD
        cls.td_model_rks = td_model_rks = tdp.TDProxy(model_rks, "dft")
        td_model_rks.kernel()

        cls.td_model_krks = td_model_krks = ktdp.TDProxy(model_krks, "dft", [cls.k, 1, 1], KRKS)
        td_model_krks.kernel()

        # GW
        cls.gw = gw.GW(td_model_rks, td.TDRHF(model_rks).ao2mo())
        cls.kgw = kgw.GW(td_model_krks, ktd.TDRHF(model_krks).ao2mo())

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
        del cls.td_model_krks
        del cls.td_model_rks
        del cls.model_krks
        del cls.model_rks
        del cls.cell

    def test_class(self):
        """Tests default eig kernel behavior."""
        self.gw.eta = self.kgw.eta = 1e-6
        self.gw.kernel()
        self.kgw.kernel()
        testing.assert_allclose(self.gw.mo_energy, self.kgw.mo_energy.reshape(-1))


class DiamondKSTestSupercell3(unittest.TestCase):
    """Compare this (kgw_slow_supercell) @2kp vs gw_slow (2x1x1 supercell), DFT."""
    k = 3
