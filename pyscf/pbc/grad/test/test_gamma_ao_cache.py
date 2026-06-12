#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from pyscf.pbc import dft, gto
import pyscf.pbc.dft.numint as pbcni


def _make(dz, xc, kmesh=(1, 1, 1)):
    cell = gto.Cell()
    cell.atom = [['H', [0.0, 0.0, 0.0]], ['H', [0.0, 0.0, 0.74 + dz]]]
    cell.a = np.eye(3) * 3.0
    cell.unit = 'angstrom'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.precision = 1e-9
    cell.mesh = [21] * 3
    cell.verbose = 0
    cell.build()
    mf = dft.KRKS(cell, kpts=cell.make_kpts(kmesh), xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


class KnownValues(unittest.TestCase):
    def _cached_equals_uncached(self, xc, kmesh=(1, 1, 1)):
        mf = _make(0.0, xc, kmesh)
        g = mf.nuc_grad_method()
        de_cached = g.kernel()
        # disable the AO cache by forcing every block to fall back to eval_ao
        orig = pbcni.AOCache.usable
        try:
            pbcni.AOCache.usable = lambda *args, **kwargs: False
            de_plain = g.kernel()
        finally:
            pbcni.AOCache.usable = orig
        self.assertAlmostEqual(abs(de_cached - de_plain).max(), 0, 9)
        return de_cached

    def test_lda_cache_consistency(self):
        self._cached_equals_uncached('lda,vwn')

    def test_gga_cache_consistency(self):
        self._cached_equals_uncached('pbe')

    def test_hybrid_cache_consistency(self):
        self._cached_equals_uncached('pbe0')

    def test_multik_cache_consistency(self):
        # the cache is keyed by the k-point set; it must also work for k>1
        self._cached_equals_uncached('pbe', kmesh=(1, 1, 2))

    def test_gga_finite_difference(self):
        de = self._cached_equals_uncached('pbe')
        disp = 1e-4
        ep = _make(+disp, 'pbe').e_tot
        em = _make(-disp, 'pbe').e_tot
        # bohr per angstrom
        bohr = 0.52917721092
        fd = (ep - em) / (2 * disp) * bohr
        self.assertAlmostEqual(de[1, 2], fd, 6)

    def test_memory_fallback(self):
        # max_memory=0 disables the cache at construction; result must match
        from pyscf.pbc.dft.numint import AOCache
        mf = _make(0.0, 'pbe')
        cache = AOCache(mf._numint, mf.cell, mf.grids, 2, kpts=mf.kpts,
                        max_memory=0)
        self.assertIsNone(cache.ao)
        self.assertFalse(cache.usable(2, mf.kpts))


if __name__ == '__main__':
    print('Gamma-point AO-cache gradient tests')
    unittest.main()
