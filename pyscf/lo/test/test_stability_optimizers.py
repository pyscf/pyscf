# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Localization optimizers must preserve occupied spaces and their objective."""
import unittest
import numpy as np
from pyscf import gto, lo, scf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 -.75 .58; H 0 .75 .58',
                        basis='sto-3g', verbose=0)
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)
        cls.occupied = cls.mf.mo_coeff[:, cls.mf.mo_occ > 0]

    def check_optimizer(self, method):
        reference = method(self.mol, self.occupied)
        reference.conv_tol = 1e-9
        reference.kernel()
        optimizer = method(self.mol, self.occupied)
        optimizer.algorithm = 'bfgs'
        optimizer.conv_tol = 1e-9
        optimizer.max_cycle = 200
        coeff = optimizer.kernel()
        # Different optimizers can escape different symmetry-related stationary
        # points; BFGS must attain an objective at least as good as this reference.
        sign = -1 if optimizer.maximize else 1
        self.assertLessEqual(sign * optimizer.cost_function(),
                             sign * reference.cost_function() + 1e-7)
        self.assertLess(np.linalg.norm(optimizer.get_grad()), 1e-4)
        np.testing.assert_allclose(coeff @ coeff.T,
                                   self.occupied @ self.occupied.T, atol=1e-10)
        np.testing.assert_allclose(coeff.T @ self.mf.get_ovlp() @ coeff, np.eye(5), atol=1e-10)

    def test_boys_bfgs(self):
        self.check_optimizer(lo.Boys)

    def test_pipek_bfgs(self):
        self.check_optimizer(lo.PM)

    def test_pipek_jacobi_improves_localization(self):
        pm = lo.PM(self.mol, self.occupied)
        initial_cost = pm.cost_function()
        coeff, stable = pm.stability_jacobi(return_status=True)
        self.assertFalse(stable)
        localized = lo.PM(self.mol, coeff)
        self.assertGreater(localized.cost_function(), initial_cost)
        np.testing.assert_allclose(coeff @ coeff.T,
                                   self.occupied @ self.occupied.T, atol=1e-10)
        again, stable = localized.stability_jacobi(return_status=True)
        self.assertTrue(stable)
        np.testing.assert_allclose(again, coeff, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
