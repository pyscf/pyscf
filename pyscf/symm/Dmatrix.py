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
    alpha, beta, gamma are Euler angles (in z-y-z convention)

    Kwargs:
        reorder_p (bool): Whether to put the p functions in the (x,y,z) order.
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

    Kwargs:
        reorder_p (bool): Whether to put the p functions in the (x,y,z) order.
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
        mat = numpy.array(((c**2        , sqrt(2)*c*s , s**2       ),
                           (-sqrt(2)*c*s, c**2-s**2   , sqrt(2)*c*s),
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


def get_euler_angles(c1, c2):
    '''Find the three Euler angles (alpha, beta, gamma in z-y-z convention)
    that rotates coordinates c1 to coordinates c2.

    yp  = numpy.einsum('j,kj->k', c1[1], geom.rotation_mat(c1[2], beta))
    tmp = numpy.einsum('ij,kj->ik', c1 , geom.rotation_mat(c1[2], alpha))
    tmp = numpy.einsum('ij,kj->ik', tmp, geom.rotation_mat(yp   , beta ))
    c2  = numpy.einsum('ij,kj->ik', tmp, geom.rotation_mat(c2[2], gamma))

    (For backward compatibility) if c1 and c2 are two points in the real
    space, the Euler angles define the rotation transforms the old coordinates
    to the new coordinates (new_x, new_y, new_z) in which c1 is identical to c2.

    tmp = numpy.einsum('j,kj->k', c1 , geom.rotation_mat((0,0,1), gamma))
    tmp = numpy.einsum('j,kj->k', tmp, geom.rotation_mat((0,1,0), beta) )
    c2  = numpy.einsum('j,kj->k', tmp, geom.rotation_mat((0,0,1), alpha))
    '''
    c1 = numpy.asarray(c1)
    c2 = numpy.asarray(c2)
    if c1.ndim == 2 and c2.ndim == 2:
        zz = c1[2].dot(c2[2])
        if abs(zz - 1.0) < 1e-12:
            beta = numpy.arccos(1.0)
        elif abs(zz + 1.0) < 1e-12:
            beta = numpy.arccos(-1.0)
        else:
            beta = numpy.arccos(zz)
        if abs(zz) < 1 - 1e-12:
            yp = numpy.cross(c1[2], c2[2])
            yp /= numpy.linalg.norm(yp)
        else:
            yp = c1[1]

        yy = yp.dot(c1[1])
        alpha = numpy.arccos(yy)
        if numpy.cross(c1[1], yp).dot(c1[2]) < 0:
            alpha = -alpha

        tmp = yp.dot(c2[1])
        if abs(tmp - 1.0) < 1e-12:
            gamma = numpy.arccos(1.0)
        elif abs(tmp + 1.0) < 1e-12:
            gamma = numpy.arccos(-1.0)
        else:
            gamma = numpy.arccos(tmp)
        if numpy.cross(yp, c2[1]).dot(c2[2]) < 0:
            gamma = -gamma

    else: # For backward compatibility, c1 and c2 are two points
        norm1 = numpy.linalg.norm(c1)
        norm2 = numpy.linalg.norm(c2)
        assert (abs(norm1 - norm2) < 1e-12)

        xy_norm = numpy.linalg.norm(c1[:2])
        if xy_norm > 1e-12:
            gamma = -numpy.arccos(c1[0] / xy_norm)
        else:
            gamma = 0

        xy_norm = numpy.linalg.norm(c2[:2])
        if xy_norm > 1e-12:
            alpha = numpy.arccos(c2[0] / xy_norm)
        else:
            alpha = 0

        beta = numpy.arccos(c2[2]/norm1) - numpy.arccos(c1[2]/norm2)

    return alpha, beta, gamma

