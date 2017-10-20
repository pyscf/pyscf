#!/usr/bin/env python

import numpy

def cg_spin(l, jdouble, mjdouble, spin):
    '''Clebsch Gordon coefficient of <l,m,1/2,spin|j,mj>'''
    ll1 = 2 * l + 1
    if jdouble == 2*l+1:
        if spin > 0:
            c = numpy.sqrt(.5*(ll1+mjdouble)/ll1)
        else:
            c = numpy.sqrt(.5*(ll1-mjdouble)/ll1)
    elif jdouble == 2*l-1:
        if spin > 0:
            c =-numpy.sqrt(.5*(ll1-mjdouble)/ll1)
        else:
            c = numpy.sqrt(.5*(ll1+mjdouble)/ll1)
    else:
        c = 0
    return c


if __name__ == '__main__':
    for kappa in list(range(-4,0)) + list(range(1,4)):
        if kappa < 0:
            l = -kappa - 1
            j = l * 2 + 1
        else:
            l = kappa
            j = l * 2 - 1
        print(kappa,l,j)
        for mj in range(-j, j+1, 2):
            print(cg_spin(l, j, mj, 1), cg_spin(l, j, mj, -1))

