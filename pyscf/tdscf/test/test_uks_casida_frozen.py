"""Frozen excitation masks must not change the ground-state XC density."""

import unittest

import numpy
from numpy.testing import assert_allclose

from pyscf import dft, gto, lib, tdscf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="He 0 0 0; He 0 0 1.1", basis="6-31g",
                    charge=1, spin=1, verbose=0)
        cls.mf = dft.UKS(mol)
        cls.mf.xc = "LDA,VWN"
        cls.mf.grids.level = 2
        cls.mf.conv_tol = 1e-12
        with lib.with_omp_threads(1):
            cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError("Synthetic helium UKS reference did not converge")

    def check_full_rpa(self, frozen):
        # The full RPA formulation uses the same excitation space but
        # evaluates the XC response at the complete ground-state density.
        with lib.with_omp_threads(1):
            casida = tdscf.uks.CasidaTDDFT(self.mf, frozen=frozen)
            vind, hdiag = casida.gen_vind()
            omega = vind(numpy.eye(len(hdiag))).T
            actual = numpy.sqrt(numpy.linalg.eigvalsh(omega))

            rpa = tdscf.uhf.TDHF(self.mf, frozen=frozen)
            vind, hdiag = rpa.gen_vind()
            eigenvalues = numpy.linalg.eigvals(vind(numpy.eye(len(hdiag))).T)
            assert_allclose(eigenvalues.imag, 0., atol=1e-9)
            expected = numpy.sort(eigenvalues.real[eigenvalues.real > 1e-9])
        assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)

    def test_frozen_occupied(self):
        self.check_full_rpa(1)

    def test_frozen_alpha_occupied(self):
        self.check_full_rpa(([0], []))

    def test_frozen_virtual(self):
        last = self.mf.mo_coeff.shape[-1] - 1
        self.check_full_rpa(([last], [last]))

    def test_no_frozen_orbitals(self):
        self.check_full_rpa(None)


if __name__ == '__main__':
    unittest.main()
