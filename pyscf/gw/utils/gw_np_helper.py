#!/usr/bin/env python
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
#
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
#

"""
Math functions for GW and RPA.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
"""

import scipy.linalg as sla
import numpy as np


def _is_contiguous(arr):
    return arr.flags['C_CONTIGUOUS'] or arr.flags['F_CONTIGUOUS']


def array_scale(arr, alpha):
    """Scale an array in place by a scalar with BLAS if possible.
    arr <- alpha * arr

    Parameters
    ----------
    arr : array_like
        Array to be scaled.
    alpha : scalar
        Scale factor.
    """
    if _is_contiguous(arr):
        # level 1 BLAS is usually multithreaded
        scal = sla.get_blas_funcs('scal', (arr,))
        scal(a=alpha, x=arr.reshape(-1))
    else:
        arr *= alpha


def addto_diagonal(arr, x):
    """Add a scalar or vector to the diagonal of a matrix.

    Parameters
    ----------
    arr : (M, M) array_like
        Square matrix (will be modified)
    x : (M,) array_like | scalar
        Vector or scalar.

    Returns
    -------
    arr : (M, M) ndarray
        arr[i, i] <- arr[i, i] + x[i].
    """
    diag = arr.diagonal()
    np.fill_diagonal(arr, diag + x)
    return arr


def get_id_minus_pi(Pi):
    """Calculate I - Pi in place.

    Parameters
    ----------
    Pi : (M, M) array_like
        Input matrix.

    Returns
    -------
    id_minus_pi : (M, M) ndarray
        (I - Pi)
    """
    array_scale(Pi, -1.0)
    addto_diagonal(Pi, 1.0)
    return Pi


def get_id_minus_pi_inv(Pi, overwrite_input=False):
    """Calculate (I - Pi)^-1, given Pi

    Parameters
    ----------
    Pi : (M, M) array_like
        Input matrix. Must be C-contiguous.

    Returns
    -------
    Pi_inv : (M, M) ndarray
        (I - Pi)^-1
    """

    assert Pi.flags.c_contiguous
    id_minus_pi = get_id_minus_pi(Pi)
    id_minus_pi_inv = sla.inv(id_minus_pi.T, overwrite_a=overwrite_input, check_finite=False).T
    return id_minus_pi_inv


def get_id_minus_pi_inv_minus_id(Pi, overwrite_input=False):
    """Calculate (I - Pi)^-1 - I.

    Parameters
    ----------
    Pi : (M, M) array_like
        Input matrix. Must be C-contiguous.

    Returns
    -------
    Pi_inv : (M, M) ndarray
        (I - Pi)^-1 - I.
    """
    id_minus_pi_inv = get_id_minus_pi_inv(Pi, overwrite_input=overwrite_input)
    return addto_diagonal(id_minus_pi_inv, -1.0)


def mkslice(l):
    """
    Try to make a slice from a list of integers.
    If this is not possible, return the input unchanged.

    Parameters
    ----------
    l : slice | list | range | ndarray
        Various ways of representing a list of integer indices

    Returns
    -------
    slice | list | ndarray
        slice if possible, otherwise returns the input unchanged
    """
    # If l is already a slice, return it
    if isinstance(l, slice):
        return l

    if l is None or l == ():
        return slice(None)

    # range to slice is easy
    if isinstance(l, range):
        return slice(l.start, l.stop, l.step)

    if len(l) < 2:
        return l

    strides = np.diff(l)
    if not np.all(strides == strides[0]):
        return l
    else:
        start = l[0]
        stop = l[-1] + strides[0]
        step = strides[0]
        return slice(start, stop, step)
