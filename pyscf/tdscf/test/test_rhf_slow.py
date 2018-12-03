from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.tdscf import TDHF
from pyscf.tdscf.rhf_slow import PhysERI, PhysERI4, PhysERI8, build_matrix, eig, kernel

import numpy
from numpy import testing
import unittest


def retrieve_m(model, **kwargs):
    vind, hdiag = model.gen_vind(model._scf, **kwargs)
    size = model.init_guess(model._scf, 1).shape[1]
    return vind(numpy.eye(size)).T


class H20Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = mol = Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = {'H': 'cc-pvdz',
                     'O': 'cc-pvdz', }
        mol.verbose = 5
        mol.build()

        cls.model_rhf = model_rhf = RHF(mol)
        model_rhf.kernel()

        cls.td_model_rhf = td_model_rhf = TDHF(model_rhf)
        td_model_rhf.nroots = 5
        td_model_rhf.kernel()

        cls.ref_m = retrieve_m(td_model_rhf)

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERI, PhysERI4, PhysERI8):
            e = eri(self.model_rhf)
            m = build_matrix(e)
            try:
                testing.assert_allclose(self.ref_m, m, atol=1e-14)
                vals, vecs = eig(m, nroots=self.td_model_rhf.nroots)
                testing.assert_allclose(vals, self.td_model_rhf.e, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = kernel(self.model_rhf, driver='eig', nroots=self.td_model_rhf.nroots)
        testing.assert_allclose(vals, self.td_model_rhf.e, atol=1e-5)
