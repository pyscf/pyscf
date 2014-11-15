#!/usr/bin/env python

import os
import ctypes
import numpy
import pyscf.lib as lib

_loaderpath = os.path.dirname(lib.__file__)
libcc = numpy.ctypeslib.load_library('libcc', _loaderpath)


# NOTE requisite on data continuous

def madd_acmn_mnbd(a, b, c, fac=1):
    m0, m1, m2, m3 = a.shape
    n0, n1, n2, n3 = b.shape
    assert(c.shape == (m0,n2,m1,n3) and m2 == n0 and m3 == n1)
    tmp = lib.dot(a.reshape(m0*m1,m2*m3), b.reshape(n0*n1,n2*n3), fac)
    c += tmp.reshape(m0,m1,n2,n3).transpose(0,2,1,3)
    return c

def madd_admn_mnbc(a, b, c, fac=1):
    m0, m1, m2, m3 = a.shape
    n0, n1, n2, n3 = b.shape
    assert(c.shape == (m0,n2,n3,m1) and m2 == n0 and m3 == n1)
    tmp = lib.dot(a.reshape(m0*m1,m2*m3), b.reshape(n0*n1,n2*n3), fac)
    c += tmp.reshape(m0,m1,n2,n3).transpose(0,2,3,1)
    return c

def madd_admn_bcmn(a, b, c, fac=1):
    m0, m1, m2, m3 = a.shape
    n0, n1, n2, n3 = b.shape
    assert(c.shape == (m0,n0,n1,m1) and m2 == n2 and m3 == n3)
    tmp = lib.dot(a.reshape(m0*m1,m2*m3),
                  b.reshape(n0*n1,n2*n3).T, fac)
    c += tmp.reshape(m0,m1,n0,n1).transpose(0,2,3,1)
    return c

def unpack_tril(tril):
    assert(tril.flags.c_contiguous)
    count = tril.shape[0]
    nd = int(numpy.sqrt(tril.shape[1]*2))
    mat = numpy.empty((count,nd,nd))
    libcc.CCunpack_tril(ctypes.c_int(count), ctypes.c_int(nd),
                        tril.ctypes.data_as(ctypes.c_void_p),
                        mat.ctypes.data_as(ctypes.c_void_p))
    return mat

# g[p,q,r,s] = v1*alpha + v2.transpose(0,2,1,3)*beta
def make_g0213(v1, v2, alpha=1, beta=1, inplace=False):
    assert(v1.flags.c_contiguous)
    assert(v2.flags.c_contiguous)
    if inplace:
        g = v1
    else:
        g = numpy.empty_like(v1)
    libcc.CCmake_g0213(g.ctypes.data_as(ctypes.c_void_p),
                       v1.ctypes.data_as(ctypes.c_void_p),
                       v2.ctypes.data_as(ctypes.c_void_p),
                       (ctypes.c_int*4)(*v1.shape),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return g

def make_g0132(v1, v2, alpha=1, beta=1, inplace=False):
    assert(v1.flags.c_contiguous)
    assert(v2.flags.c_contiguous)
    if inplace:
        g = v1
    else:
        g = numpy.empty_like(v1)
    libcc.CCmake_g0132(g.ctypes.data_as(ctypes.c_void_p),
                       v1.ctypes.data_as(ctypes.c_void_p),
                       v2.ctypes.data_as(ctypes.c_void_p),
                       (ctypes.c_int*4)(*v1.shape),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return g

# tau = t2 + numpy.einsum('ia,jb->ijba', t1a, t1b)
def make_tau(t2, t1a, t1b, inplace=False):
    assert(t2.flags.c_contiguous)
    if inplace:
        tau = t2
    else:
        tau = t2.copy()
    libcc.CCset_tau(tau.ctypes.data_as(ctypes.c_void_p),
                    t1a.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int*2)(*t1a.shape),
                    t1b.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int*2)(*t1b.shape))
    return tau

