# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Low-memory AGF2 moment contractions agree with the Python reference."""
import unittest
import numpy as np
from pyscf.agf2 import _agf2


class KnownValues(unittest.TestCase):
    def test_unrestricted_moments(self):
        rng = np.random.default_rng(37)
        qxi = (rng.normal(size=(5, 12)), rng.normal(size=(5, 8)))
        qja = (rng.normal(size=(5, 6)), rng.normal(size=(5, 6)))
        occ = (np.array([-1.2, -.8, -.3]), np.array([-1.1, -.5]))
        vir = (np.array([.2, .7]), np.array([.3, .6, 1.]))
        for ss, os in ((1., 1.), (0., 1.2), (.4, 0.)):
            with self.subTest(same_spin=ss, opposite_spin=os):
                ref = _agf2.build_mats_dfuagf2_outcore(qxi, qja, occ, vir, os_factor=os, ss_factor=ss)
                for builder in (_agf2.build_mats_dfuagf2_incore, _agf2.build_mats_dfuagf2_lowmem):
                    actual = builder(qxi, qja, occ, vir, os_factor=os, ss_factor=ss)
                    np.testing.assert_allclose(actual, ref, atol=1e-10)
                self.assertGreater(np.linalg.eigvalsh(ref[0]).min(), -1e-10)
                np.testing.assert_allclose(ref[1], ref[1].T, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
