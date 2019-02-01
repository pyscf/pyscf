#!/usr/bin/env python
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''Wigner rotation D-matrix for real spherical harmonics'''

from math import sqrt
from functools import reduce
import numpy
from scipy.special import factorial
from pyscf.symm import sph


def Dmatrix(l, alpha, beta, gamma, reorder_p=False):
    '''Wigner rotation D-matrix

    D_{mm'} = <lm|R(alpha,beta,gamma)|lm'>
    alpha, beta, gamma are Eular angles (in z-y-z convention)
    '''
    if l == 0:
        return numpy.eye(1)
    else:
        d = dmatrix(l, beta, reorder_p=False)
        ms = numpy.arange(-l, l+1)
        D = numpy.einsum('i,ij,j->ij', numpy.exp(-1j*alpha*ms), d,
                         numpy.exp(-1j*gamma*ms))

        D = _dmat_to_real(l, D, reorder_p=False)
        if reorder_p and l == 1:
            D = D[[2,0,1]][:,[2,0,1]]
        return D

def _dmat_to_real(l, d, reorder_p=False):
    ''' Transform the input D-matrix to make it compatible with the real
    spherical harmonic functions.
    '''
    # The input D matrix works for pure spherical harmonics. The real
    # representation should be U^\dagger * D * U, where U is the unitary
    # matrix that transform the complex harmonics to the real harmonics.
    u = sph.sph_pure2real(l, reorder_p)
    return reduce(numpy.dot, (u.conj().T, d, u)).real

def dmatrix(l, beta, reorder_p=False):
    '''Wigner small-d matrix (in z-y-z convention)'''
    c = numpy.cos(beta/2)
    s = numpy.sin(beta/2)
    if l == 0:
        return numpy.eye(1)
    elif l == 1:
        mat = numpy.array(((c**2        , sqrt(2)*c*s , s**2       ), \
                           (-sqrt(2)*c*s, c**2-s**2   , sqrt(2)*c*s), \
                           (s**2        , -sqrt(2)*c*s, c**2       )))
        if reorder_p:
            mat = mat[[2,0,1]][:,[2,0,1]]
        return mat
    elif l == 2:
        c3s = c**3*s
        s3c = s**3*c
        c2s2 = (c*s)**2
        c4 = c**4
        s4 = s**4
        s631 = sqrt(6)*(c3s-s3c)
        s622 = sqrt(6)*c2s2
        c4s2 = c4-3*c2s2
        c2s4 = 3*c2s2-s4
        c4s4 = c4-4*c2s2+s4
        return numpy.array((( c4    , 2*c3s, s622, 2*s3c, s4   ),
                            (-2*c3s , c4s2 , s631, c2s4 , 2*s3c),
                            ( s622  ,-s631 , c4s4, s631 , s622 ),
                            (-2*s3c , c2s4 ,-s631, c4s2 , 2*c3s),
                            ( s4    ,-2*s3c, s622,-2*c3s, c4   )))
    else:
        facs = factorial(numpy.arange(2*l+1))
        cs = c**numpy.arange(2*l+1)
        ss = s**numpy.arange(2*l+1)

        mat = numpy.zeros((2*l+1,2*l+1))
        for i,m1 in enumerate(range(-l, l+1)):
            for j,m2 in enumerate(range(-l, l+1)):
                #:fac = sqrt( factorial(l+m1)*factorial(l-m1) \
                #:           *factorial(l+m2)*factorial(l-m2))
                #:for k in range(max(m2-m1,0), min(l+m2, l-m1)+1):
                #:    mat[i,j] += (-1)**(m1+m2+k) \
                #:            * c**(2*l+m2-m1-2*k) * s**(m1-m2+2*k) \
                #:            / (factorial(l+m2-k) * factorial(k) \
                #:               * factorial(m1-m2+k) * factorial(l-m1-k))
                #:mat[i,j] *= fac
                k = numpy.arange(max(m2-m1,0), min(l+m2, l-m1)+1)
                tmp = (cs[2*l+m2-m1-2*k] * ss[m1-m2+2*k] /
                       (facs[l+m2-k] * facs[k] * facs[m1-m2+k] * facs[l-m1-k]))

                mask = ((m1+m2+k) & 0b1).astype(bool)
                mat[i,j] -= tmp[ mask].sum()
                mat[i,j] += tmp[~mask].sum()

        ms = numpy.arange(-l, l+1)
        msfac = numpy.sqrt(facs[l+ms] * facs[l-ms])
        mat *= numpy.einsum('i,j->ij', msfac, msfac)
    return mat


def get_euler_angles(v1, v2):
    '''The three Eular angles (alpha, beta, gamma) rotates vector v1 to vector v2
    through the transformation:

    tmp = numpy.dot(geom.rotation_mat((0,0,1), alpha), v1)
    new_y = numpy.dot(geom.rotation_mat((0,0,1), alpha), (0,1,0))
    tmp = numpy.dot(geom.rotation_mat(new_y, beta), tmp)
    new_z = numpy.dot(geom.rotation_mat(new_y, beta), (0,0,1))
    v2 = numpy.dot(geom.rotation_mat(new_z, gamma), tmp)

    which is equivalent to apply the transformation to v1 in the old axes
    (without transforming the axes as above)
    tmp = numpy.dot(geom.rotation_mat((0,0,1), gamma), v1)
    tmp = numpy.dot(geom.rotation_mat((0,1,0), beta), tmp)
    v2  = numpy.dot(geom.rotation_mat((0,0,1), alpha), tmp)

    Based on the equation above, this function returns one possible solution
    of (alpha, beta, gamma)
    '''
    norm1 = numpy.linalg.norm(v1)
    norm2 = numpy.linalg.norm(v2)
    assert(abs(norm1 - norm2) < 1e-12)

    xy_norm = numpy.linalg.norm(v1[:2])
    if xy_norm > 1e-12:
        gamma = -numpy.arccos(v1[0] / xy_norm)
    else:
        gamma = 0

    xy_norm = numpy.linalg.norm(v2[:2])
    if xy_norm > 1e-12:
        alpha = numpy.arccos(v2[0] / xy_norm)
    else:
        alpha = 0

    beta = numpy.arccos(v2[2]/norm1) - numpy.arccos(v1[2]/norm2)
    return alpha, beta, gamma

