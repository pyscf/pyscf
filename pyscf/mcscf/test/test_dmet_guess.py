# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""DMET active-space selection preserves the mean-field determinant."""
import unittest
import numpy as np
from pyscf import gto, scf, lo
from pyscf.mcscf import dmet_cas


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3',
                    basis='sto-3g', symmetry=True, verbose=0)
        cls.mf = scf.RHF(mol).run(conv_tol=1e-12)

    def check_guess(self, canonicalize=True, freeze_imp=False, unrestricted=False):
        mf = self.mf.to_uhf() if unrestricted else self.mf
        ncas, nelec, mo = dmet_cas.kernel(mf, mf.make_rdm1(), [0], base=0,
                                         canonicalize=canonicalize, freeze_imp=freeze_imp)
        self.assertEqual((ncas, nelec), (2, 2))
        s = mf.get_ovlp()
        np.testing.assert_allclose(mo.T @ s @ mo, np.eye(4), atol=1e-11)
        dm = self.mf.make_rdm1()
        transformed = mo.T @ s @ dm @ s @ mo
        self.assertAlmostEqual(transformed[0, 0], 2, places=10)
        self.assertAlmostEqual(transformed[-1, -1], 0, places=10)
        self.assertAlmostEqual(np.trace(transformed[1:3, 1:3]), nelec, places=10)
        if freeze_imp or not canonicalize:
            impurity = lo.orth.orth_ao(mf.mol)[:, 0]
            self.assertAlmostEqual(abs(impurity @ s @ mo[:, 1]), 1, places=10)

    def test_canonicalized_guess(self):
        self.check_guess()

    def test_local_guess(self):
        self.check_guess(canonicalize=False)

    def test_frozen_impurity_guess(self):
        self.check_guess(freeze_imp=True)

    def test_unrestricted_closed_shell_guess(self):
        self.check_guess(unrestricted=True)


if __name__ == '__main__':
    unittest.main()
