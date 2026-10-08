"""CPU DFT Hessians must reject unimplemented grid response."""
import unittest
import numpy
from pyscf import dft, gto


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        cls.rmf = dft.RKS(cls.mol, xc='pbe').run(conv_tol=1e-11)
        cls.umf = dft.UKS(cls.mol, xc='pbe').run(conv_tol=1e-11)
        cls.df_rmf = cls.rmf.density_fit().run(conv_tol=1e-11)
        cls.df_umf = cls.umf.density_fit().run(conv_tol=1e-11)
        for mf in (cls.rmf, cls.umf, cls.df_rmf, cls.df_umf):
            if not mf.converged:
                raise RuntimeError('test SCF did not converge')

    def test_explicit_grid_response_is_rejected(self):
        self._check_rejection((self.rmf, self.umf))

    def test_density_fitted_grid_response_is_rejected(self):
        self._check_rejection((self.df_rmf, self.df_umf))

    def _check_rejection(self, mean_fields):
        for mf in mean_fields:
            with self.subTest(method=type(mf).__name__):
                hess = mf.Hessian().set(grid_response=True)
                for operation in (hess.kernel, hess.hess, hess.hess_elec,
                                  hess.partial_hess_elec):
                    with self.assertRaisesRegex(NotImplementedError, 'grid response'):
                        operation()
                with self.assertRaisesRegex(NotImplementedError, 'grid response'):
                    hess.make_h1(mf.mo_coeff, mf.mo_occ)

    def test_default_grid_response_remains_false(self):
        self._check_default((self.rmf, self.umf))

    def test_density_fitted_default_grid_response_remains_false(self):
        self._check_default((self.df_rmf, self.df_umf))

    def _check_default(self, mean_fields):
        for mf in mean_fields:
            with self.subTest(method=type(mf).__name__):
                hess = mf.Hessian()
                self.assertFalse(hess.grid_response)
                result = hess.kernel()
                self.assertEqual(result.shape, (2, 2, 3, 3))
                self.assertTrue(numpy.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
