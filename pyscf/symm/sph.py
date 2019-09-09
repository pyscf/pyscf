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

'''
Spherical harmonics
'''

import numpy
import scipy.linalg
from pyscf.symm.cg import cg_spin

def real_sph_vec(r, lmax, reorder_p=False):
    '''Computes (all) real spherical harmonics up to the angular momentum lmax'''
    #:import scipy.special
    #:ngrid = r.shape[0]
    #:cosphi = r[:,2]
    #:sinphi = (1-cosphi**2)**.5
    #:costheta = numpy.ones(ngrid)
    #:sintheta = numpy.zeros(ngrid)
    #:costheta[sinphi!=0] = r[sinphi!=0,0] / sinphi[sinphi!=0]
    #:sintheta[sinphi!=0] = r[sinphi!=0,1] / sinphi[sinphi!=0]
    #:costheta[costheta> 1] = 1
    #:costheta[costheta<-1] =-1
    #:sintheta[sintheta> 1] = 1
    #:sintheta[sintheta<-1] =-1
    #:varphi = numpy.arccos(cosphi)
    #:theta = numpy.arccos(costheta)
    #:theta[sintheta<0] = 2*numpy.pi - theta[sintheta<0]
    #:ylms = []
    #:for l in range(lmax+1):
    #:    ylm = numpy.empty((l*2+1,ngrid))
    #:    ylm[l] = scipy.special.sph_harm(0, l, theta, varphi).real
    #:    for m in range(1, l+1):
    #:        f1 = scipy.special.sph_harm(-m, l, theta, varphi)
    #:        f2 = scipy.special.sph_harm( m, l, theta, varphi)
    #:        # complex to real spherical functions
    #:        if m % 2 == 1:
    #:            ylm[l-m] = (-f1.imag - f2.imag) / numpy.sqrt(2)
    #:            ylm[l+m] = ( f1.real - f2.real) / numpy.sqrt(2)
    #:        else:
    #:            ylm[l-m] = (-f1.imag + f2.imag) / numpy.sqrt(2)
    #:            ylm[l+m] = ( f1.real + f2.real) / numpy.sqrt(2)
    #:    ylms.append(ylm)
    #:return ylms

    # When r is a normalized vector:
    norm = 1./numpy.linalg.norm(r, axis=1)
    return multipoles(r*norm.reshape(-1,1), lmax, reorder_p)


def multipoles(r, lmax, reorder_dipole=True):
    '''
    Compute all multipoles upto lmax

    rad = numpy.linalg.norm(r, axis=1)
    ylms = real_ylm(r/rad.reshape(-1,1), lmax)
    pol = [rad**l*y for l, y in enumerate(ylms)]

    Kwargs:
        reorder_p : bool
            sort dipole to the order (x,y,z)
    '''
    from pyscf import gto

# libcint cart2sph transformation provide the capability to compute
# multipole directly.  cart2sph function is fast for low angular moment.
    ngrid = r.shape[0]
    xs = numpy.ones((lmax+1,ngrid))
    ys = numpy.ones((lmax+1,ngrid))
    zs = numpy.ones((lmax+1,ngrid))
    for i in range(1,lmax+1):
        xs[i] = xs[i-1] * r[:,0]
        ys[i] = ys[i-1] * r[:,1]
        zs[i] = zs[i-1] * r[:,2]
    ylms = []
    for l in range(lmax+1):
        nd = (l+1)*(l+2)//2
        c = numpy.empty((nd,ngrid))
        k = 0
        for lx in reversed(range(0, l+1)):
            for ly in reversed(range(0, l-lx+1)):
                lz = l - lx - ly
                c[k] = xs[lx] * ys[ly] * zs[lz]
                k += 1
        ylm = gto.cart2sph(l, c.T).T
        ylms.append(ylm)

# when call libcint, p functions are ordered as px,py,pz
# reorder px,py,pz to p(-1),p(0),p(1)
    if (not reorder_dipole) and lmax >= 1:
        ylms[1] = ylms[1][[1,2,0]]
    return ylms

def sph_pure2real(l, reorder_p=True):
    r'''
    Transformation matrix: from the pure spherical harmonic functions Y_m to
    the real spherical harmonic functions O_m.
          O_m = \sum Y_m' * U(m',m)
    Y(-1) = 1/\sqrt(2){-iO(-1) + O(1)}; Y(1) = 1/\sqrt(2){-iO(-1) - O(1)}
    Y(-2) = 1/\sqrt(2){-iO(-2) + O(2)}; Y(2) = 1/\sqrt(2){iO(-2) + O(2)}
    O(-1) = i/\sqrt(2){Y(-1) + Y(1)};   O(1) = 1/\sqrt(2){Y(-1) - Y(1)}
    O(-2) = i/\sqrt(2){Y(-2) - Y(2)};   O(2) = 1/\sqrt(2){Y(-2) + Y(2)}

    Kwargs:
        reorder_p (bool): Whether the p functions are in the (x,y,z) order.

    Returns:
        2D array U_{complex,real}
    '''
    n = 2 * l + 1
    u = numpy.zeros((n,n), dtype=complex)
    sqrthfr = numpy.sqrt(.5)
    sqrthfi = numpy.sqrt(.5)*1j

    if reorder_p and l == 1:
        u[1,2] = 1
        u[0,1] =  sqrthfi
        u[2,1] =  sqrthfi
        u[0,0] =  sqrthfr
        u[2,0] = -sqrthfr
    else:
        u[l,l] = 1
        for m in range(1, l+1, 2):
            u[l-m,l-m] =  sqrthfi
            u[l+m,l-m] =  sqrthfi
            u[l-m,l+m] =  sqrthfr
            u[l+m,l+m] = -sqrthfr
        for m in range(2, l+1, 2):
            u[l-m,l-m] =  sqrthfi
            u[l+m,l-m] = -sqrthfi
            u[l-m,l+m] =  sqrthfr
            u[l+m,l+m] =  sqrthfr

    return u

def sph_real2pure(l, reorder_p=True):
    ''' 
    Transformation matrix: from real spherical harmonic functions to the pure
    spherical harmonic functions.

    Kwargs:
        reorder_p (bool): Whether the real p functions are in the (x,y,z) order.
    '''
    # numpy.linalg.inv(sph_pure2real(l))
    return sph_pure2real(l, reorder_p).conj().T

# |spinor> = (|real_sph>, |real_sph>) * / u_alpha \
#                                       \ u_beta  /
# Return 2D array U_{sph,spinor}
def sph2spinor(l, reorder_p=True):
    if l == 0:
        return numpy.array((0., 1.)).reshape(1,-1), \
               numpy.array((1., 0.)).reshape(1,-1)
    else:
        u1 = sph_real2pure(l, reorder_p)
        ua = numpy.zeros((2*l+1,4*l+2),dtype=complex)
        ub = numpy.zeros((2*l+1,4*l+2),dtype=complex)
        j = l * 2 - 1
        mla = l + (-j-1)//2
        mlb = l + (-j+1)//2
        for k,mj in enumerate(range(-j, j+1, 2)):
            ua[:,k] = u1[:,mla] * cg_spin(l, j, mj, 1)
            ub[:,k] = u1[:,mlb] * cg_spin(l, j, mj,-1)
            mla += 1
            mlb += 1
        j = l * 2 + 1
        mla = l + (-j-1)//2
        mlb = l + (-j+1)//2
        for k,mj in enumerate(range(-j, j+1, 2)):
            if mla < 0:
                ua[:,l*2+k] = 0
            else:
                ua[:,l*2+k] = u1[:,mla] * cg_spin(l, j, mj, 1)
            if mlb >= 2*l+1:
                ub[:,l*2+k] = 0
            else:
                ub[:,l*2+k] = u1[:,mlb] * cg_spin(l, j, mj,-1)
            mla += 1
            mlb += 1
    return ua, ub
real2spinor = sph2spinor

# Returns 2D array U_{sph,spinor}
def sph2spinor_coeff(mol):
    '''Transformation matrix that transforms real-spherical GTOs to spinor
    GTOs for all basis functions

    Examples::

    >>> from pyscf import gto
    >>> from pyscf.symm import sph
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvtz')
    >>> ca, cb = sph.sph2spinor_coeff(mol)
    >>> s0 = mol.intor('int1e_ovlp_spinor')
    >>> s1 = ca.conj().T.dot(mol.intor('int1e_ovlp_sph')).dot(ca)
    >>> s1+= cb.conj().T.dot(mol.intor('int1e_ovlp_sph')).dot(cb)
    >>> print(abs(s1-s0).max())
    >>> 6.66133814775e-16
    '''
    lmax = max([mol.bas_angular(i) for i in range(mol.nbas)])
    ualst = []
    ublst = []
    for l in range(lmax+1):
        u1, u2 = sph2spinor(l, reorder_p=True)
        ualst.append(u1)
        ublst.append(u2)

    ca = []
    cb = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        kappa = mol.bas_kappa(ib)
        if kappa == 0:
            ua = ualst[l]
            ub = ublst[l]
        elif kappa < 0:
            ua = ualst[l][:,l*2:]
            ub = ublst[l][:,l*2:]
        else:
            ua = ualst[l][:,:l*2]
            ub = ublst[l][:,:l*2]
        nctr = mol.bas_nctr(ib)
        ca.extend([ua]*nctr)
        cb.extend([ub]*nctr)
    return numpy.stack([scipy.linalg.block_diag(*ca),
                        scipy.linalg.block_diag(*cb)])
real2spinor_whole = sph2spinor_coeff

def cart2spinor(l):
    '''Cartesian to spinor for angular moment l'''
    from pyscf import gto
    return gto.cart2spinor_l(l)


if __name__ == '__main__':
    for l in range(3):
        print(sph_pure2real(l))
        print(sph_real2pure(l))

    for l in range(3):
        print(sph2spinor(l)[0])
        print(sph2spinor(l)[1])
