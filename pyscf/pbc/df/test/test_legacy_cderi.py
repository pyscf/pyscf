# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Legacy full, triangular, and Coulomb-only DF checkpoints remain readable."""
import unittest
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, df
from pyscf.pbc.df.df import CDERIArray


class KnownValues(unittest.TestCase):
    def test_legacy_layouts(self):
        # The current DF implementation supports checkpoints from PySCF <= 2.0.
        # Test that compatibility here; "legacy" refers to the file formats,
        # not to an obsolete DF module.
        cell = gto.M(atom='He 0 0 0; He 1 0 0', a=np.eye(3)*4,
                     basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                     precision=1e-7, verbose=0)
        kpts = cell.make_kpts([2, 1, 1])
        fitting = df.GDF(cell, kpts)
        fitting.auxbasis = 'weigend'
        fitting.build()
        with CDERIArray(fitting._cderi) as data:
            blocks = {(i, j): data.load(kpts[i], kpts[j]) for i in range(2) for j in range(2)}
        layouts = {'full': [(0, 0), (0, 1), (1, 0), (1, 1)],
                   'triangular': [(0, 0), (1, 0), (1, 1)],
                   'coulomb': [(0, 0), (1, 1)]}
        for name, pairs in layouts.items():
            with self.subTest(layout=name), lib.H5TmpFile() as archive:
                archive['j3c-kptij'] = np.array([[kpts[i], kpts[j]] for i, j in pairs])
                for index, pair in enumerate(pairs):
                    archive[f'j3c/{index}'] = blocks[pair]
                archive.flush()
                with CDERIArray(archive.filename) as restored:
                    queries = pairs if name == 'coulomb' else blocks
                    for i, j in queries:
                        np.testing.assert_allclose(restored[i, j], blocks[i, j], atol=1e-12)
                        np.testing.assert_allclose(restored.load(kpts[i], kpts[j]), blocks[i, j], atol=1e-12)
                old = df.GDF(cell, kpts)
                old._cderi = archive.filename
                self.assertEqual(old.get_naoaux(), fitting.get_naoaux())


if __name__ == '__main__':
    unittest.main()
