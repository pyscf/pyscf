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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.soscf.ciah import CIAHOptimizerMixin, expmat


class SubspaceCIAHOptimizerMixin(CIAHOptimizerMixin):
    ''' Base class for unitary rotations with direct sum structure:
            U = U1 \oplus U2 \oplus U3 ...
        where each Ui is of shape (norb, norb)

        Args
        ----------
        norb : int
            Dimension of each subspace
        rtypes : array-like
            A list of integers specifying the rotation type of each subspace.
                0: real rotation
                1: complex rotation with fixed gague (zero diagonals)
                2: general complex rotation
    '''
    def __init__(self, norb, rtypes):
        self.norb = norb
        self.rtypes = rtypes

    @property
    def pdim(self):
        n = self.norb
        pdim_map = {0: n*(n-1)//2, 1: n*(n-1), 2: n*n}
        return sum([pdim_map[x] for x in self.rtypes])

    @property
    def sdim(self):
        return len(self.rtypes)

    def pack_uniq_var(self, mat):
        mat = numpy.reshape(mat, (self.sdim, self.norb, self.norb))
        tril_idx = numpy.tril_indices(self.norb, k=-1)
        v = []
        for i in range(self.sdim):
            v.append( mat[i].real[tril_idx] )
            if self.rtypes[i] > 0:
                v.append( mat[i].imag[tril_idx] )
            if self.rtypes[i] > 1:
                v.append( numpy.diag(mat[i].imag) )
        return numpy.hstack(v)

    def unpack_uniq_var(self, v):
        v = numpy.asarray(v).reshape(-1)
        assert( v.size == self.pdim )
        n = self.norb
        n2 = n*(n-1)//2
        tril_idx = numpy.tril_indices(n, k=-1)
        mat = numpy.zeros((self.sdim, n,n), dtype=numpy.complex128) # real if all type 0?
        p1 = 0
        for i in range(self.sdim):
            M = mat[i]
            p0, p1 = p1, p1+n2
            M[tril_idx] = v[p0:p1]
            if self.rtypes[i] > 0:
                p0, p1 = p1, p1+n2
                M[tril_idx] += v[p0:p1] * 1j
            if self.rtypes[i] > 1:
                p0, p1 = p1, p1+n
                numpy.fill_diagonal(M, v[p0:p1] * 0.5j)
            M -= M.T.conj()
        return mat

    def extract_rotation(self, dr, u0=None):
        dr = self.unpack_uniq_var(dr)
        u1 = numpy.asarray([expmat(drk) for drk in dr])
        if u0 is None:
            return u1
        else:
            return self.update_rotation(u0, u1)

    def update_rotation(self, u0, u1):
        return numpy.asarray([numpy.dot(u0k, u1k) for u0k,u1k in zip(u0,u1)])

    def zero_uniq_var(self):
        return numpy.zeros(self.pdim)
