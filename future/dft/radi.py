#!/usr/bin/env python
#
# File: radi.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''radii grids'''

import numpy

#########################
# JCP 41 3199 (1964). In Angstrom (of the time, strictly)
BRAGG_RADII = numpy.array((
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
COVALENT_RADII = numpy.sqrt((
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

# scale rad and rad_weight if necessary
def delley(n): #GetAtomRadialGrid
    r = numpy.empty(n)
    dr = numpy.empty(n)
    r_outer = 12.
    step = 1. / (n+1)
    RFac = r_outer / numpy.log(1 - (n*step)**2)
    for i in range(1, n+1):
        xi = RFac * numpy.log(1-(i*step)**2);
        r[i-1] = xi
        dri = RFac * (-2.0*i*(step)**2) / ((1-(i*step)**2)) # d xi / dr
        dr[i-1] = dri
    w = r*r * dr * 4 * numpy.pi
    return r, w

# Mura-Knowles log3 quadrature (JCP,104,9848)
def mura_knowles(n):
    r = numpy.empty(n)
    dr = numpy.empty(n)
    far = 5.2 # 7 for Li, Be, Na, Mg, K, Ca, otherwise 5
    step = 1. / n
    for i in range(n):
        x = (i+.5) / n
        r[i] = -far * numpy.log(1-x**3)
        dr[i] = far * 3*x*x/((1-x**3)*n);
    w = r*r * dr * 4 * numpy.pi
    return r, w

def gauss_chebeshev(n): #rgd_nkcheb
    r = numpy.empty(n)
    dr = numpy.empty(n)
    step = 1. / (n+1)
    ln2 = 1 / numpy.log(2)
    fac = 16*step / 3
    for i in range(1, n+1):
        x1 = i * numpy.pi * step
        xi = (n+1-2*i) * step \
                + 1/numpy.pi * (1+2./3*numpy.sin(x1)**2) * numpy.sin(2*x1)
        r[i-1] = numpy.log(2/(1-xi)) * ln2
        wi = fac * numpy.sin(x1)**4
        dr[i-1] = wi * ln2/(1-xi)
    w = r*r * dr * 4 * numpy.pi
    return r[::-1], w[::-1]


#!--> gauss-chebeshev quadrature of the second kind(used by becke)
#  call rgd_Chebeshev(n,ra(i),rr(1,i),wr(1,i) )
#!--> gauss-legendre quadrature(delley)
#  call rgd_legendre(n,xgl,wgl,ra(i),rr0,wr0,rr(1,i),wr(1,i))
#!--> sective gauss-legendre quadrature
#  !nrg(i)=nsb(1,i)+nsb(2,i)+nsb(3,i)   
#  !--> this should be done before this subroutine
#  if(ipr.gt.0)write(6,29) i,nsb(1,i),nsb(2,i),nsb(3,i),nrg(i)
#   format(5x,'ndat nsb1,nsb2,nsb3,nrg  ',5i5)
#   call rgd_seclegendre(nsb(1,i),ra(i),rcv1,xgl,wgl
#  ,rr0,wr0,rr(1,i),wr(1,i) )
#!--> generalized gauss-laguerre integrarion
#  call rgd_glaguerre(n,ssb(i),rr0,wr0,rr(1,i),wr(1,i))
#!--> equally spaced points: murray,handy,laming, mol. phys. 78,997(1993)
#  call rgd_mhl(n,ra(i),rr(1,i),wr(1,i) )



if __name__ == '__main__':
    #print delley(10)
    #print mura_knowles(10)
    #print gauss_chebeshev(10)
    for i in range(1,60,3):
        print i, (0.75 + (3-1)*0.2) * 14.  * (i+2.)**(1/3.)
