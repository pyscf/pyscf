# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Oliver Backhouse <olbackhouse@gmail.com>
#         George Booth <george.booth@kcl.ac.uk>
#

'''
MPI helper functions using mpi4py
'''

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

try:
    from mpi4py import MPI as mpi
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except: #NOTE should we check for ImportError? This is OSError in python2, check how pyscf handles this elsewhere
    mpi = None
    comm = None
    size = 1
    rank = 0

SCALE_PRANGE_STEP = False


def reduce(obj, root=0, op=getattr(mpi, 'SUM', None)):
    ''' Reduce a matrix or scalar onto the root process and then
        broadcast result to all processes.

    Args:
        obj : int, float or complex scalar or array
            Object to reduce

    Kwargs:
        root : int
            Rank of the root process
        op : :mod:`mpi4py` operation
            :mod:`mpi4py` operation to reduce with. Default :mod:`mpi4py.MPI.SUM`

    Returns:
        obj : int, float or complex scalar or array
            Reduced object
    '''

    is_array = isinstance(obj, np.ndarray)

    if size == 1:
        return m

    m_red = np.zeros_like(m)
    comm.Reduce(np.asarray(m), m_red, op=op, root=root)

    m = m_red
    comm.Bcast(m, root=0)

    if not is_array:
        m = m.ravel()[0]

    return m


def nrange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start

    for i in range(start+rank, stop, step*size):
        yield i


def prange(start, stop, step):
    ''' :func:`lib.prange` distributed over MPI processes. Returns
        the range for a single MPI rank.
    '''

    if size == 1:
        for p0, p1 in lib.prange(start, stop, step):
            yield p0, p1
    else:
        if SCALE_PRANGE_STEP:
            step //= size

        split = lambda x : x * (stop-start) // size

        start0 = split(rank)
        stop0 = stop if rank == (size-1) else split(rank+1)

        for p0, p1 in lib.prange(start0, stop0, step):
            yield p0, p1
