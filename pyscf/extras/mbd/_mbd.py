#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Jan Hermann <jan@hermann.in>
#

import pyscf.lib
from pyscf.lib import ndpointer
import numpy as np
from ctypes import c_double, c_int


libmbd = pyscf.lib.load_library('libmbd')

libmbd.add_dipole_matrix.restype = None
libmbd.add_dipole_matrix.argtypes = (
    c_int, c_int,
    ndpointer(dtype=np.float64, ndim=2, flags=('C', 'W')),
    ndpointer(dtype=np.float64, ndim=2, flags='C'),
    ndpointer(dtype=np.float64, shape=(3,)),
    c_double,
    ndpointer(dtype=np.float64, ndim=1),
    ndpointer(dtype=np.float64, ndim=1),
    c_double, c_double
)


versions = {
    'bare': 0,
    'fermi,dip,gg': 1,
    'fermi,dip': 2,
}


def get_dipole(version, coords, alpha=None, R_vdw=None, beta=np.nan, a=np.nan):
    n = len(coords)
    T = np.zeros((3*n, 3*n))
    libmbd.add_dipole_matrix(versions[version], n, T, coords, None, 0, alpha, R_vdw, beta, a)
    return T
