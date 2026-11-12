#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from unittest import mock

import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import fft
from pyscf.pbc.df import fft_jk


def _make_system():
    cell = pgto.M(atom='He 0 0 0', a=np.eye(3) * 3,
                  basis=[[0, [.8, 1]], [1, [.5, 1]]],
                  mesh=[4] * 3, verbose=0)
    kpts = cell.make_kpts([2, 1, 1])
    mydf = fft.FFTDF(cell, kpts=kpts)
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    ngrids = len(cell.gen_uniform_grids(cell.mesh))
    return mydf, dm, kpts, cell.nao_nr(), ngrids


class FFTJKMemoryEstimateTest(unittest.TestCase):
    '''The block size must use the memory left after the current allocation.'''

    def setUp(self):
        self.mydf, self.dm, self.kpts, self.nao, self.ngrids = _make_system()
        self.mem_now = 900.
        self.mydf.max_memory = 1000.
        self._prange = lib.prange

    def _run(self, kernel, memory_factor):
        blocks = []

        def prange(start, stop, step):
            blocks.append(step)
            yield from self._prange(start, stop, step)

        with mock.patch.object(fft_jk.lib, 'current_memory',
                               return_value=(self.mem_now, 0.)), \
                mock.patch.object(fft_jk.lib, 'prange', side_effect=prange):
            kernel(self.mydf, self.dm, kpts=self.kpts)

        available = (self.mydf.max_memory - self.mem_now) * 1e6
        expected = int(min(self.nao, max(
            1, available / 16 / 4 / memory_factor / self.ngrids / self.nao)))
        self.assertGreater(expected, 1)
        self.assertEqual(set(blocks), {expected})

    def test_get_k_kpts_block_size(self):
        self._run(fft_jk.get_k_kpts, memory_factor=1)

    def test_get_k_e1_kpts_block_size(self):
        self._run(fft_jk.get_k_e1_kpts, memory_factor=3)


if __name__ == '__main__':
    unittest.main()
