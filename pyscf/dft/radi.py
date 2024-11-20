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

'''radii grids'''

import numpy
from pyscf.data import radii
from pyscf.data.elements import charge as elements_proton
from pyscf import __config__

BRAGG_RADII = radii.BRAGG
COVALENT_RADII = radii.COVALENT

# The effective atomic radius in Treutler's original paper (JCP 102, 346) was
# omitted in PySCF versions 2.6 and earlier. This can cause discrepancies in the
# numerical grids when compared to other software (See issue #2178 for details).
# Note that using the atom-specific radius may slightly alter the results of
# numerical integration, potentially leading to differences of ~ 1e-6 per atom
# in total energy.
# Disable this flag to make DFT grids consistent with old PySCF versions.
ATOM_SPECIFIC_TREUTLER_GRIDS = getattr(__config__, 'ATOM_SPECIFIC_TREUTLER_GRIDS', True)

# P.M.W. Gill, B.G. Johnson, J.A. Pople, Chem. Phys. Letters 209 (1993) 506-512
SG1RADII = numpy.array((
    0,
    1.0000,                                                 0.5882,
    3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,
    4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333))


# Murray, N.C. Handy, G.J. Laming,  Mol. Phys. 78, 997(1993)
def murray(n, *args, **kwargs):
    raise RuntimeError('Not implemented')

def becke(n, charge, *args, **kwargs):
    '''Becke, JCP 88, 2547 (1988); DOI:10.1063/1.454033'''
    if charge == 1:
        rm = BRAGG_RADII[charge]
    else:
        rm = BRAGG_RADII[charge] * .5

    # Points and weights for Gauss-Chebyshev quadrature points of the second kind
    # The weights are adjusted to integrate a function on the interval [-1, 1] with uniform weighting
    # instead of weighted by sqrt(1 - t^2) = sin(i pi / (n+1)).
    i = numpy.arange(n) + 1
    t = numpy.cos(i * numpy.pi / (n + 1))
    w = numpy.pi / (n + 1) * numpy.sin(i * numpy.pi / (n + 1))

    # Change of variables to map the domain to [0, inf)
    r = (1+t)/(1-t) * rm
    w *= 2/(1-t)**2 * rm
    return r, w

# scale rad and rad_weight if necessary
# gauss-legendre
def delley(n, *args, **kwargs):
    '''B. Delley radial grids. Ref. JCP 104, 9848 (1996); DOI:10.1063/1.471749. log2 algorithm'''
    r = numpy.empty(n)
    dr = numpy.empty(n)
    r_outer = 12.
    step = 1. / (n+1)
    rfac = r_outer / numpy.log(1 - (n*step)**2)
    for i in range(1, n+1):
        xi = rfac * numpy.log(1-(i*step)**2)
        r[i-1] = xi
        dri = rfac * (-2.0*i*(step)**2) / ((1-(i*step)**2)) # d xi / dr
        dr[i-1] = dri
    return r, dr
gauss_legendre = delley

def mura_knowles(n, charge=None, *args, **kwargs):
    '''Mura-Knowles [JCP 104, 9848 (1996); DOI:10.1063/1.471749] log3 quadrature radial grids'''
    r = numpy.empty(n)
    dr = numpy.empty(n)
# 7 for Li, Be, Na, Mg, K, Ca, otherwise 5
    if charge in (3, 4, 11, 12, 19, 20):
        far = 7
    else:
        far = 5.2
    for i in range(n):
        x = (i+.5) / n
        r[i] = -far * numpy.log(1-x**3)
        dr[i] = far * 3*x*x/((1-x**3)*n)
    return r, dr

# Gauss-Chebyshev of the second kind,  and the transformed interval [0,\infty)
# Ref  Matthias Krack and Andreas M. Koster,  J. Chem. Phys. 108 (1998), 3226
def gauss_chebyshev(n, *args, **kwargs):
    '''Gauss-Chebyshev [JCP 108, 3226 (1998); DOI:10.1063/1.475719) radial grids'''
    ln2 = 1 / numpy.log(2)
    fac = 16./3 / (n+1)
    x1 = numpy.arange(1,n+1) * numpy.pi / (n+1)
    xi = ((n-1-numpy.arange(n)*2) / (n+1.) +
          (1+2./3*numpy.sin(x1)**2) * numpy.sin(2*x1) / numpy.pi)
    xi = (xi - xi[::-1])/2
    r = 1 - numpy.log(1+xi) * ln2
    dr = fac * numpy.sin(x1)**4 * ln2/(1+xi)
    return r, dr

# Individually optimized Treutler/Ahlrichs radius parameter.
# H - Kr are taken from the original paper JCP 102, 346 (1995)
# Other elements are copied from Psi4 source code
_treutler_ahlrichs_xi = [1.0, # Ghost
    0.8,                               0.9,           # 1s
    1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,           # 2s2p
    1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,           # 3s3p
    1.5, 1.4,                                         # 4s
    1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, # 3d
              1.1, 1.0, 0.9, 0.9, 0.9, 0.9,           # 4p
    2.000, 1.700,                                                         # 5s
    1.500, 1.500, 1.350, 1.350, 1.250, 1.200, 1.250, 1.300, 1.500, 1.500, # 4d
                  1.300, 1.200, 1.200, 1.150, 1.150, 1.150,               # 5p
    2.500, 2.200,                                                         # 6s
           2.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,               # La, Ce-Eu
           1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,        # Gd, Tb-Lu
           1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, # 5d
                  1.500, 1.500, 1.500, 1.500, 1.500, 1.500,               # 6p
    2.500, 2.100,                                                         # 7s
           3.685, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,
           1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,
] # noqa: E124

def treutler_ahlrichs(n, chg, *args, **kwargs):
    '''
    Treutler-Ahlrichs [JCP 102, 346 (1995); DOI:10.1063/1.469408] (M4) radial grids
    '''
    if ATOM_SPECIFIC_TREUTLER_GRIDS:
        xi = _treutler_ahlrichs_xi[chg]
    else:
        xi = 1.
    r = numpy.empty(n)
    dr = numpy.empty(n)
    step = numpy.pi / (n+1)
    ln2 = xi / numpy.log(2)
    for i in range(n):
        x = numpy.cos((i+1)*step)
        r [i] = -ln2*(1+x)**.6 * numpy.log((1-x)/2)
        dr[i] = step * numpy.sin((i+1)*step) \
                * ln2*(1+x)**.6 *(-.6/(1+x)*numpy.log((1-x)/2)+1/(1-x))
    return r[::-1], dr[::-1]
treutler = treutler_ahlrichs


def becke_atomic_radii_adjust(mol, atomic_radii):
    '''Becke atomic radii adjust function'''
# Becke atomic size adjustment.  J. Chem. Phys. 88, 2547
# i > j
# fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
# fac(j,i) = -fac(i,j)
    charges = [elements_proton(x) for x in mol.elements]
    rad = atomic_radii[charges] + 1e-200
    rr = rad.reshape(-1,1) * (1./rad)
    a = .25 * (rr.T - rr)
    a[a<-.5] = -.5
    a[a>0.5] = 0.5
    #:return lambda i,j,g: g + a[i,j]*(1-g**2)
    def fadjust(i, j, g):
        g1 = g**2
        g1 -= 1.
        g1 *= -a[i,j]
        g1 += g
        return g1
    return fadjust

def treutler_atomic_radii_adjust(mol, atomic_radii):
    '''Treutler atomic radii adjust function: [JCP 102, 346 (1995); DOI:10.1063/1.469408]'''
# JCP 102, 346 (1995)
# i > j
# fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
# fac(j,i) = -fac(i,j)
    charges = [elements_proton(x) for x in mol.elements]
    rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
    rr = rad.reshape(-1,1) * (1./rad)
    a = .25 * (rr.T - rr)
    a[a<-.5] = -.5
    a[a>0.5] = 0.5
    #:return lambda i,j,g: g + a[i,j]*(1-g**2)
    def fadjust(i, j, g):
        g1 = g**2
        g1 -= 1.
        g1 *= -a[i,j]
        g1 += g
        return g1
    return fadjust
