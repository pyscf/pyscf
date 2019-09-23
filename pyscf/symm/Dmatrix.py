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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''Wigner rotation D-matrix for real spherical harmonics'''

from math import sqrt, factorial
from functools import reduce
import numpy

raise RuntimeError('TODO: test Dmatrix')

def dmatrix(l, alpha, beta, gamma):
    if l == 0:
        return numpy.eye(1)
    else:
        c = small_dmatrix(l, beta)
        n = l * 2 + 1
        d = numpy.zeros((n,n), dtype=complex)
        for i,m1 in enumerate(range(-l, l+1)):
            for j,m2 in enumerate(range(-l, l+1)):
                d[i,j] = c[i,j] * numpy.exp(-1j*alpha*m1-1j*m2*gamma)
        return dmat_to_real(d).real

# u * d * u^T is real matrix
def dmat_to_real(d):
    l = (d.shape[0]-1)//2
    u = transmat_to_real(l)
    return reduce(numpy.dot, (u, d, u.T))

# Ym' = Rm * U_mm'
def transmat_to_real(l):
    n = 2 * l + 1
    u = numpy.zeros((n,n),dtype=complex)
    u[l,l] = 1
    s2 = sqrt(2.)
    for m in range(1, l+1, 2):
        u[l-m,l-m] =-s2 * 1j
        u[l+m,l-m] = s2
        u[l-m,l+m] =-s2 * 1j
        u[l+m,l+m] =-s2
    for m in range(2, l+1, 2):
        u[l-m,l-m] =-s2 * 1j
        u[l+m,l-m] = s2
        u[l-m,l+m] = s2 * 1j
        u[l+m,l+m] = s2
    return u

def small_dmatrix(l, beta):
    c = numpy.cos(beta/2)
    s = numpy.sin(beta/2)
    if l == 0:
        return numpy.eye(1)
    elif l == 1:
        return numpy.array(((c**2        , sqrt(2)*c*s , s**2       ), \
                            (-sqrt(2)*c*s, c**2-s**2   , sqrt(2)*c*s), \
                            (s**2        , -sqrt(2)*c*s, c**2       )))
    elif l == 2:
        s631 = sqrt(6)*c*s*(c**2-s**2)
        s622 = sqrt(6)*(c*s)**2
        c4s2 = c**4-3*(c*s)**2
        c2s4 = 3*(c*s)**2-s**4
        c4s4 = c**4-4*(c*s)**2+s**4
        return numpy.array((( c**4     , 2*c**3*s, s622, 2*c*s**3, s**4    ), \
                            (-2*c**3*s , c4s2    , s631, c2s4    , 2*c*s**3), \
                            ( s622     ,-s631    , c4s4, s631    , s622    ), \
                            (-2*c*s**3 , c2s4    ,-s631, c4s2    , 2*c**3*s), \
                            ( s**4     ,-2*c*s**3, s622,-2*c**3*s, c**4    )))
    else:
        mat = numpy.zeros((2*l+1,2*l+1))
        for i,m1 in enumerate(range(-l, l+1)):
            for j,m2 in enumerate(range(-l, l+1)):
                if j < i:
                    continue
                fac = sqrt( factorial(l+m1)*factorial(l-m1) \
                           *factorial(l+m2)*factorial(l-m2))
                for k in range(max(m2-m1,0), min(l+m2, l-m1)+1):
                    mat[i,j] += (-1)**(m1+m2+k) \
                            * c**(2*l+m2-m1-2*k) * s**(m1-m2+2*k) \
                            / (factorial(l+m2-k) * factorial(k) \
                               * factorial(m1-m2+k) * factorial(l-m1-k))
                mat[i,j] *= fac
                mat[j,i] = (-1)**(m1+m2)*mat[i,j]
    return mat

if __name__ == '__main__':
    #print(small_dmatrix(1, .4))
    #print(small_dmatrix(2, .4))
    print(dmatrix(1, .1, .1, .1))
