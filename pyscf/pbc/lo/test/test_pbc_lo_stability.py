# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Jacobi localization preserves the periodic occupied projector."""
import unittest
import numpy as np
from pyscf.lo import orth
from pyscf.pbc import gto
from pyscf.pbc.lo.kpipek import KPM


class KnownValues(unittest.TestCase):
    def test_delocalized_pair(self):
        cell = gto.M(atom='He 0 0 0; He 0 0 2', a=np.eye(3)*5,
                     basis='sto-3g', precision=1e-7, verbose=0)
        kpts = cell.make_kpts([2, 1, 1])
        overlap = cell.pbc_intor('int1e_ovlp', kpts=kpts)
        rotation = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2)
        coeff = np.array([orth.lowdin(s) @ rotation for s in overlap])
        pm = KPM(cell, coeff, kpts)
        pm.pop_method = 'lowdin'
        initial = pm.cost_function()
        result, stable = pm.stability_jacobi(return_status=True)
        self.assertFalse(stable)
        localized = KPM(cell, result, kpts)
        localized.pop_method = 'lowdin'
        self.assertGreater(localized.cost_function(), initial)
        for before, after, s in zip(coeff, result, overlap):
            np.testing.assert_allclose(after @ after.conj().T, before @ before.conj().T, atol=1e-9)
            np.testing.assert_allclose(after.conj().T @ s @ after, np.eye(2), atol=1e-9)
        _, stable = localized.stability_jacobi(return_status=True)
        self.assertTrue(stable)


if __name__ == '__main__':
    unittest.main()
