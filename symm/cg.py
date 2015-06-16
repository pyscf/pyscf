#!/usr/bin/env python

import numpy

# Clebsch Gordon coefficient of <l,m,1/2,spin|j,mj>
def cg_spin(l, jdouble, mjdouble, spin):
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

# Transformation matrix: from pure spherical harmonic function Y_m
# to real spherical harmonic function O_m.
#       O_m = \sum Y_m' * U(m',m)
# Y(-1) = 1/sqrt(2){-iO(-1) + O(1)}; Y(1) = 1/sqrt(2){-iO(-1) - O(1)}
# Y(-2) = 1/sqrt(2){-iO(-2) + O(2)}; Y(2) = 1/sqrt(2){iO(-2) + O(2)}
# O(-1) = i/sqrt(2){Y(-1) + Y(1)};   O(1) = 1/sqrt(2){Y(-1) - Y(1)}
# O(-2) = i/sqrt(2){Y(-2) - Y(2)};   O(2) = 1/sqrt(2){Y(-2) + Y(2)}
# U_{complex,real}
def sph_pure2real(l):
    u = numpy.zeros((2*l+1,2*l+1),dtype=complex)
    sqrthfr = numpy.sqrt(.5)
    sqrthfi = numpy.sqrt(.5)*1j

    if l == 1:
        u[1,2] = 1
        u[0,1] =  sqrthfi
        u[2,1] =  sqrthfi
        u[0,0] =  sqrthfr
        u[2,0] = -sqrthfr
    else:
        u[l,l] = 1
        for m in range(1,l+1,2):
            u[l-m,l-m] =  sqrthfi
            u[l+m,l-m] =  sqrthfi
            u[l-m,l+m] =  sqrthfr
            u[l+m,l+m] = -sqrthfr
        for m in range(2,l+1,2):
            u[l-m,l-m] =  sqrthfi
            u[l+m,l-m] = -sqrthfi
            u[l-m,l+m] =  sqrthfr
            u[l+m,l+m] =  sqrthfr

    return u

def sph_real2pure(l):
    return sph_pure2real(l).T.conj()

# |spinor> = (|real_sph>, |real_sph>) * / u_alpha \
#                                       \ u_beta  /
# U_{sph,spinor}
def real2spinor(l):
    if l == 0:
        return numpy.array((0., 1.)).reshape(1,-1), \
               numpy.array((1., 0.)).reshape(1,-1)
    else:
        u1 = sph_real2pure(l)
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

# U_{sph,spinor}
def real2spinor_whole(mol):
    lmax = max([mol.bas_angular(i) for i in range(mol.nbas)])
    ualst = []
    ublst = []
    for l in range(lmax+1):
        u1, u2 = real2spinor(l)
        ualst.append(u1)
        ublst.append(u2)

    ua = numpy.zeros((mol.nao_nr(),mol.nao_2c()), dtype=complex)
    ub = numpy.zeros_like(ua)
    p0 = 0
    p1 = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nctr = mol.bas_nctr(ib)
        n, m = ualst[l].shape
        for ic in range(nctr):
            ua[p0:p0+n,p1:p1+m] = ualst[l]
            ub[p0:p0+n,p1:p1+m] = ublst[l]
            p0 += n
            p1 += m
    return ua, ub

def cart2spinor(l):
    raise RuntimeError('TODO')


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

    for l in range(3):
        print(sph_pure2real(l))
        print(sph_real2pure(l))

    for l in range(3):
        print(real2spinor(l)[0])
        print(real2spinor(l)[1])
