# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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


def reduce(obj, root=0, op=mpi.SUM):
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
