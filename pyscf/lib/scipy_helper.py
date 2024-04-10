#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

'''
Extensions to the scipy.linalg module
'''

import numpy


# Numpy/scipy does not seem to have a convenient interface for
# pivoted Cholesky factorization. Newer versions of scipy (>=1.4) provide
# access to the raw lapack function, which is wrapped around here.
# With older versions of scipy, we use our own implementation instead.
try:
    from scipy.linalg.lapack import dpstrf as _dpstrf
except ImportError:
    def _pivoted_cholesky_wrapper(A, tol, lower):
        return pivoted_cholesky_python(A, tol=tol, lower=lower)
else:
    def _pivoted_cholesky_wrapper(A, tol, lower):
        N = A.shape[0]
        assert (A.shape == (N, N))
        L, piv, rank, info = _dpstrf(A, tol=tol, lower=lower)
        if info < 0:
            raise RuntimeError('Pivoted Cholesky factorization failed.')
        if lower:
            L[numpy.triu_indices(N, k=1)] = 0
            L[:, rank:] = 0
        else:
            L[numpy.tril_indices(N, k=-1)] = 0
            L[rank:, :] = 0
        return L, piv-1, rank


def pivoted_cholesky(A, tol=-1.0, lower=False):
    '''
    Performs a Cholesky factorization of A with full pivoting.
    A can be a (singular) positive semidefinite matrix.

    P.T * A * P = L * L.T   if   lower is True
    P.T * A * P = U.T * U   if   lower if False

    Use regular Cholesky factorization for positive definite matrices instead.

    Args:
        A : the matrix to be factorized
        tol : the stopping tolerance (see LAPACK documentation for dpstrf)
        lower : return lower triangular matrix L if true
                return upper triangular matrix U if false

    Returns:
        the factor L or U, the pivot vector (starting with 0), the rank
    '''
    return _pivoted_cholesky_wrapper(A, tol=tol, lower=lower)


def pivoted_cholesky_python(A, tol=-1.0, lower=False):
    '''
    Pedestrian implementation of Cholesky factorization with full column pivoting.
    The LAPACK version should be used instead whenever possible!

    Args:
        A : the positive semidefinite matrix to be factorized
        tol : stopping tolerance
        lower : return the lower or upper diagonal factorization

    Returns:
        the factor, the permutation vector, the rank
    '''
    N = A.shape[0]
    assert (A.shape == (N, N))

    D = numpy.diag(A.real).copy()
    if tol < 0:
        machine_epsilon = numpy.finfo(numpy.double).eps
        tol = N * machine_epsilon * numpy.amax(numpy.diag(A))

    L = numpy.zeros_like(A)
    piv = numpy.arange(N)
    rank = 0
    for k in range(N):
        s = k + numpy.argmax(D[k:])
        piv[k], piv[s] = piv[s], piv[k]
        D[k], D[s] = D[s], D[k]
        L[[k, s], :] = L[[s, k], :]
        if D[k] <= tol:
            break
        rank += 1
        L[k, k] = numpy.sqrt(D[k])
        L[k+1:, k] = (A[piv[k+1:], piv[k]] - numpy.dot(L[k+1:, :k], L[k, :k].conj())) / L[k, k]
        D[k+1:] -= abs(L[k+1:, k]) ** 2

    if lower:
        return L, piv, rank
    else:
        return L.conj().T, piv, rank
