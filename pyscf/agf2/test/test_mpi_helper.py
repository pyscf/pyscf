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

"""Partition and serial-collective contracts, without an MPI installation."""
import unittest

import numpy as np

from pyscf import lib
from pyscf.agf2 import mpi_helper as mpi


class KnownValues(unittest.TestCase):
    def test_serial_collectives(self):
        with lib.temporary_env(mpi, size=1, rank=0, comm=None):
            for value in (2., np.arange(6).reshape(2, 3), {'key': np.arange(3)}):
                for operation in (mpi.bcast, mpi.bcast_dict, mpi.reduce,
                                  mpi.allreduce, mpi.allreduce_safe_inplace):
                    with self.subTest(operation=operation.__name__):
                        self.assertIs(operation(value), value)
            self.assertIsNone(mpi.barrier())
            self.assertEqual(list(mpi.prange(3, 14, 4)), [(3, 7), (7, 11), (11, 14)])

    def test_distributed_ranges(self):
        for size in (1, 2, 4):
            for args in ((11,), (3, 17, 2), (7, 7), (2, 5), (17, 3, -2)):
                with self.subTest(size=size, args=args):
                    assigned = []
                    for rank in range(size):
                        with lib.temporary_env(mpi, size=size, rank=rank):
                            assigned.extend(mpi.nrange(*args))
                    self.assertEqual(sorted(assigned), sorted(range(*args)))

    def test_distributed_blocks(self):
        for size in (2, 4):
            for start, stop in ((0, 19), (3, 19), (7, 7), (2, 4)):
                for scale in (False, True):
                    with self.subTest(size=size, start=start, stop=stop, scale=scale):
                        assigned = []
                        for rank in range(size):
                            with lib.temporary_env(mpi, size=size, rank=rank,
                                                   SCALE_PRANGE_STEP=scale):
                                blocks = list(mpi.prange(start, stop, 8))
                            for left, right in blocks:
                                self.assertLess(left, right)
                                self.assertLessEqual(right - left, 8 // size if scale else 8)
                                assigned.extend(range(left, right))
                        # Every element must be visited exactly once, even if
                        # the interval starts at a nonzero index or is empty.
                        self.assertEqual(sorted(assigned), list(range(start, stop)))


    def test_scaled_step_smaller_than_process_count(self):
        assigned = []
        for rank in range(4):
            with lib.temporary_env(mpi, size=4, rank=rank, SCALE_PRANGE_STEP=True):
                assigned.extend(mpi.prange(3, 10, 1))
        self.assertEqual(assigned, [(i, i + 1) for i in range(3, 10)])


if __name__ == '__main__':
    unittest.main()
