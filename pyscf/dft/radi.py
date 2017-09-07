#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''radii grids'''

import numpy
from pyscf.lib.parameters import BOHR

#########################
# JCP 41 3199 (1964).
BRAGG_RADII = 1/BOHR * numpy.array((0,  # Ghost atom
        0.35,                                     1.40,             # 1s
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             # 2s2p
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             # 3s3p
        2.20, 1.80,                                                 # 4s
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, # 3d
                    1.30, 1.25, 1.15, 1.15, 1.15, 1.90,             # 4p
        2.35, 2.00,                                                 # 5s
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, # 4d
                    1.55, 1.45, 1.45, 1.40, 1.40, 2.10,             # 5p
        2.60, 2.15,                                                 # 6s
        1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   # La, Ce-Eu
        1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             # Gd, Tb-Lu
              1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, # 5d
                    1.90, 1.80, 1.60, 1.90, 1.45, 2.10,             # 6p
        1.80, 2.15,                                                 # 7s
        1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                    1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75))

# from Gerald Knizia's CtDftGrid, which is based on
#       http://en.wikipedia.org/wiki/Covalent_radius
# and
#       Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves,
#       Jorge Echeverria, Eduard Cremades, Flavia Barragan and Santiago
#       Alvarez.  Covalent radii revisited. Dalton Trans., 2008, 2832-2838,
#       doi:10.1039/b801115j
COVALENT_RADII = 1/BOHR * numpy.array((0,  # Ghost atom
        0.31,                                     0.28,             # 1s
        1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,             # 2s2p
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,             # 3s3p
        2.03, 1.76,                                                 # 4s
        1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22, # 3d
                    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,             # 4p
        2.20, 1.95,                                                 # 5s
        1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, # 4d
                    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,             # 5p
        2.44, 2.15,                                                 # 6s
        2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,                   # La, Ce-Eu
        1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,             # Gd, Tb-Lu
              1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, # 5d
                    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,             # 6p
        2.60, 2.21,                                                 # 7s
        2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69))


# P.M.W. Gill, B.G. Johnson, J.A. Pople, Chem. Phys. Letters 209 (1993) 506-512
SG1RADII = numpy.array((
    0,
    1.0000,                                                 0.5882,
    3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,
    4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333))


# Murray, N.C. Handy, G.J. Laming,  Mol. Phys. 78, 997(1993)
def murray(n, *args, **kwargs):
    raise RuntimeError('Not implemented')

def becke(n, charge, **kwargs):
    '''Becke, JCP, 88, 2547 (1988)'''
    if charge == 1:
        rm = BRAGG_RADII[charge]
    else:
        rm = BRAGG_RADII[charge] * .5
    t, w = numpy.polynomial.chebyshev.chebgauss(n)
    r = (1+t)/(1-t) * rm
    w *= 2/(1-t)**2 * rm
    return r, w

# scale rad and rad_weight if necessary
# gauss-legendre
def delley(n, *args, **kwargs):
    '''B. Delley radial grids. Ref. JCP, 104, 9848 log2 algorithm'''
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

def mura_knowles(n, charge=None, **kwargs):
    '''Mura-Knowles (JCP, 104, 9848) log3 quadrature radial grids'''
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
    '''Gauss-Chebyshev (JCP, 108, 3226) radial grids'''
    ln2 = 1 / numpy.log(2)
    fac = 16./3 / (n+1)
    x1 = numpy.arange(1,n+1) * numpy.pi / (n+1)
    xi = ((n-1-numpy.arange(n)*2) / (n+1.) +
          (1+2./3*numpy.sin(x1)**2) * numpy.sin(2*x1) / numpy.pi)
    xi = (xi - xi[::-1])/2
    r = 1 - numpy.log(1+xi) * ln2
    dr = fac * numpy.sin(x1)**4 * ln2/(1+xi)
    return r, dr


def treutler_ahlrichs(n, *args, **kwargs):
    '''Treutler-Ahlrichs (JCP 102, 346 (M4)) radial grids'''
    r = numpy.empty(n)
    dr = numpy.empty(n)
    step = numpy.pi / (n+1)
    ln2 = 1 / numpy.log(2)
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

    charges = mol.atom_charges()
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
    '''Treutler atomic radii adjust function: JCP, 102, 346'''
# JCP, 102, 346
# i > j
# fac(i,j) = \frac{1}{4} ( \frac{ra(j)}{ra(i)} - \frac{ra(i)}{ra(j)}
# fac(j,i) = -fac(i,j)
    charges = mol.atom_charges()
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

def _inter_distance(mol):
# see gto.mole.energy_nuc
    coords = mol.atom_coords()
    rr = numpy.dot(coords, coords.T)
    rd = rr.diagonal()
    rr = rd[:,None] + rd - rr*2
    rr[numpy.diag_indices_from(rr)] = 0
    return numpy.sqrt(rr)

