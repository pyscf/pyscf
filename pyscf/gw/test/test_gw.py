from pyscf.gto import Mole
from pyscf.scf import RHF, RKS
from pyscf.gw import GW
from pyscf.gw.gw_slow import GW as GW_slow
from pyscf.tdscf import TDRHF, TDRKS
from pyscf.tdscf.rhf_slow import TDRHF as TDRHF_slow
from pyscf.tdscf.proxy import TDProxy

import numpy
from numpy import testing
import unittest

from test_common import assert_vectors_close, adjust_td_phase


class H20Test(unittest.TestCase):
    """Compare gw and gw_slow."""
    @classmethod
    def setUpClass(cls):
        cls.mol = mol = Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = 'cc-pvdz'
        mol.verbose = 5
        mol.build()

        cls.model_rhf = model_rhf = RHF(mol)
        model_rhf.kernel()

        cls.model_rks = model_rks = RKS(mol)
        model_rks.xc = 'hf'
        model_rks.kernel()

        testing.assert_allclose(model_rhf.mo_energy, model_rks.mo_energy)
        assert_vectors_close(model_rhf.mo_coeff.T, model_rks.mo_coeff.T)

        model_rks.mo_coeff = model_rhf.mo_coeff

        cls.td_model_rks = td_model_rks = TDRHF(model_rks)
        td_model_rks.nroots = 4
        td_model_rks.kernel()

        cls.td_model_rhf_slow = td_model_rhf_slow = TDRHF_slow(model_rhf)
        td_model_rhf_slow.nroots = td_model_rks.nroots
        td_model_rhf_slow.kernel()

        cls.td_proxy_model = td_proxy_model = TDProxy(model_rks, proxy=td_model_rks)
        td_proxy_model.nroots = td_model_rks.nroots
        td_proxy_model.kernel()

        cls.gw = gw = GW(model_rks, td_model_rks)
        gw.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.gw
        del cls.td_proxy_model
        del cls.td_model_rhf_slow
        del cls.td_model_rks
        del cls.model_rhf
        del cls.model_rks
        del cls.mol

    def test_gw(self):
        """Tests container behavior."""
        gw_slow = GW_slow(self.td_model_rhf_slow)
        gw_slow.kernel()

        testing.assert_allclose(gw_slow.mo_energy, self.gw.mo_energy, atol=1e-6)

    def test_gw_proxy(self):
        """Tests container behavior (proxy)."""
        eri = TDRHF_slow(self.model_rks).ao2mo()
        assert eri is not None
        gw_slow = GW_slow(self.td_proxy_model, eri=eri)
        gw_slow.kernel()

        testing.assert_allclose(gw_slow.mo_energy, self.gw.mo_energy, atol=1e-6)

    def test_imds_frozen(self):
        """Tests intermediates: frozen vs non-frozen."""
        td = TDRHF_slow(self.model_rhf, frozen=1)
        td.nroots = self.td_model_rhf_slow.nroots
        td.kernel()

        adjust_td_phase(self.td_model_rhf_slow, td)

        gw_slow_ref = GW_slow(self.td_model_rhf_slow)
        gw_slow_ref.kernel()

        gw_slow_frozen = GW_slow(td)
        gw_slow_frozen.kernel()

        testing.assert_allclose(gw_slow_ref.imds.tdm[..., 1:, 1:], gw_slow_frozen.imds.tdm, atol=1e-4)

        test_energies = numpy.linspace(-2, 3, 300)
        ref_samples = numpy.array(tuple(gw_slow_ref.imds.get_sigma_element(i, 1, 0.01) for i in test_energies))
        frozen_samples = numpy.array(tuple(gw_slow_frozen.imds.get_sigma_element(i, 0, 0.01) for i in test_energies))

        testing.assert_allclose(ref_samples, frozen_samples, atol=5e-4)
        testing.assert_allclose(
            gw_slow_ref.imds.get_rhs(1),
            gw_slow_frozen.imds.get_rhs(0),
            atol=1e-13,
        )

    def test_gw_frozen(self):
        """Tests container behavior (frozen TDHF)."""
        td = TDRHF_slow(self.model_rhf, frozen=1)
        td.nroots = self.td_model_rhf_slow.nroots
        td.kernel()
        adjust_td_phase(self.td_model_rhf_slow, td)

        gw_slow_ref = GW_slow(self.td_model_rhf_slow)
        gw_slow_ref.kernel()
        gw_slow_frozen = GW_slow(td)
        gw_slow_frozen.kernel()

        testing.assert_allclose(gw_slow_frozen.mo_energy, gw_slow_ref.mo_energy[1:], atol=1e-4)


class H20KSTest(unittest.TestCase):
    """Compare gw and gw_slow on KS."""
    @classmethod
    def setUpClass(cls):
        cls.mol = mol = Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = 'cc-pvdz'
        mol.verbose = 5
        mol.build()

        cls.model_rks = model_rks = RKS(mol)
        # model_rks.xc = 'hf'
        model_rks.kernel()

        cls.td_model_rks = td_model_rks = TDRKS(model_rks)
        td_model_rks.nroots = 4
        td_model_rks.kernel()

        cls.td_proxy_model = td_proxy_model = TDProxy(model_rks, proxy=td_model_rks)
        td_proxy_model.nroots = td_model_rks.nroots
        td_proxy_model.kernel()

        testing.assert_allclose(td_model_rks.e, td_proxy_model.e)

        cls.gw = gw = GW(model_rks, td_model_rks)
        gw.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.gw
        del cls.td_proxy_model
        del cls.td_model_rks
        del cls.model_rks
        del cls.mol

    def test_gw(self):
        """Tests container behavior (proxy)."""
        eri = TDRHF_slow(self.model_rks).ao2mo()
        assert eri is not None
        gw_slow = GW_slow(self.td_proxy_model, eri=eri)
        gw_slow.kernel()
        # print self.gw.mo_energy

        testing.assert_allclose(gw_slow.mo_energy, self.gw.mo_energy, atol=1e-6)
