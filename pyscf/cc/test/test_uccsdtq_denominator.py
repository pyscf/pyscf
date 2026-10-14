# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

"""T4 denominators for packed/full spin blocks and strided residuals."""

from itertools import combinations
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf import lib
from pyscf.cc import uccsdtq, _uccsdtq_utils as utils


class KnownValues(unittest.TestCase):
    def test_invalid_residual_storage(self):
        model = SimpleNamespace(nocc=(4, 4), nmo=(9, 9), level_shift=.19,
                                do_tri_max_t=True)
        energies = (np.arange(9, dtype=np.float64),) * 2
        shape = utils.block_shape(model, 4, 0, True)
        readonly = np.zeros(shape)
        readonly.flags.writeable = False
        misaligned = np.ndarray(shape, dtype=np.float64,
                                buffer=np.zeros(readonly.nbytes+1, dtype=np.uint8), offset=1)
        for bad in (readonly, misaligned, np.zeros(shape[::-1])):
            with self.subTest(writeable=bad.flags.writeable, aligned=bad.flags.aligned,
                              shape=bad.shape):
                residuals = [bad] + [np.zeros(utils.block_shape(model, 4, b, True))
                                     for b in range(1, 5)]
                with self.assertRaisesRegex(ValueError, 'invalid T4 residual storage'):
                    uccsdtq.r4_divide_e_(model, residuals, energies)

    def test_denominators(self):
        rng = np.random.default_rng(72)
        old_threads = lib.num_threads()
        try:
            for threads in (1, 2):
                lib.num_threads(threads)
                for compact in (False, True):
                    for nocc, nmo in (((4, 3), (9, 9)), ((1, 0), (3, 2))):
                        model = SimpleNamespace(nocc=nocc, nmo=nmo, level_shift=.19,
                                                do_tri_max_t=compact)
                        energies = tuple(np.r_[rng.uniform(-2., -1., no),
                                               rng.uniform(.2, 1., nm-no)]
                                         for no, nm in zip(nocc, nmo))
                        for layout in ('C', 'F', 'strided'):
                            with self.subTest(threads=threads, compact=compact,
                                              nocc=nocc, layout=layout):
                                residuals, expected = [], []
                                for block in range(5):
                                    shape = utils.block_shape(model, 4, block, compact)
                                    if layout == 'strided':
                                        base = np.empty(shape[:-1] + (2*shape[-1],))
                                        residual = base[..., ::-2]
                                    else:
                                        residual = np.empty(shape, order=layout)
                                    residual[...] = rng.normal(size=shape)
                                    denominator = np.zeros(shape)
                                    first = int(block > 2)
                                    counts = (max(4-block, block), min(4-block, block))
                                    axis = 0
                                    for spin, count in zip((first, 1-first), counts):
                                        if not count:
                                            continue
                                        no = nocc[spin]
                                        for e in (energies[spin][:no], -energies[spin][no:]):
                                            groups = (list(combinations(range(len(e)), count))
                                                      if compact else [(p,) for p in range(len(e))])
                                            sums = np.array([sum(e[p] for p in group) for group in groups])
                                            repeats = 1 if compact else count
                                            for _ in range(repeats):
                                                view_shape = [1] * len(shape)
                                                view_shape[axis] = sums.size
                                                denominator += sums.reshape(view_shape)
                                                axis += 1
                                    denominator -= 4 * model.level_shift
                                    expected.append(residual / denominator)
                                    residuals.append(residual)
                                returned = uccsdtq.r4_divide_e_(model, residuals, energies)
                                self.assertIs(returned, residuals)
                                for actual, reference in zip(residuals, expected):
                                    np.testing.assert_allclose(actual, reference, rtol=2e-15, atol=1e-15)
        finally:
            lib.num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
