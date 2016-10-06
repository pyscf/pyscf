#!/usr/bin/env python

import os
import ctypes
import numpy
from pyscf import lib

libcc = lib.load_library('libcc')


# NOTE requisite on data continuous

def unpack_tril(tril, out=None):
    return lib.unpack_tril(tril, 1, out=out)

def pack_tril(mat, out=None):
    return lib.pack_tril(mat, out=out)

# v1*alpha + v2.transpose(0,2,1,3)*beta
def make_0213(v1, v2, alpha=1, beta=1, out=None):
    assert(v1.flags.c_contiguous)
    assert(v2.flags.c_contiguous)
    out = numpy.ndarray(v1.shape, buffer=out)
    count, m = v1.shape[:2]
    libcc.CCmake_0213(out.ctypes.data_as(ctypes.c_void_p),
                      v1.ctypes.data_as(ctypes.c_void_p),
                      v2.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(count), ctypes.c_int(m),
                      ctypes.c_double(alpha), ctypes.c_double(beta))
    return out

# v1*alpha + v2.transpose(0,1,3,2)*beta
def make_0132(v1, v2, alpha=1, beta=1, out=None):
    assert(v1.flags.c_contiguous)
    assert(v2.flags.c_contiguous)
    out = numpy.ndarray(v1.shape, buffer=out)
    count = v1.shape[0] * v1.shape[1]
    m = v1.shape[2]
    libcc.CCmake_021(out.ctypes.data_as(ctypes.c_void_p),
                     v1.ctypes.data_as(ctypes.c_void_p),
                     v2.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(count), ctypes.c_int(m),
                     ctypes.c_double(alpha), ctypes.c_double(beta))
    return out

def make_021(v1, v2, alpha=1, beta=1, out=None):
    assert(v1.flags.c_contiguous)
    assert(v2.flags.c_contiguous)
    out = numpy.ndarray(v1.shape, buffer=out)
    count, m = v1.shape[:2]
    libcc.CCmake_021(out.ctypes.data_as(ctypes.c_void_p),
                     v1.ctypes.data_as(ctypes.c_void_p),
                     v2.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(count), ctypes.c_int(m),
                     ctypes.c_double(alpha), ctypes.c_double(beta))
    return out
# a special case of make_0132, with v1 == v2, alpha == beta == 1
def sum021(v1, out=None):
    out = numpy.ndarray(v1.shape, buffer=out)
    count, m = v1.shape[:2]
    libcc.CCsum021(out.ctypes.data_as(ctypes.c_void_p),
                   v1.ctypes.data_as(ctypes.c_void_p),
                   v1.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(count), ctypes.c_int(m))
    return out

# t2 + numpy.einsum('ia,jb->ijab', t1a, t1b)
def make_tau(t2, t1a, t1b, fac=1, out=None):
    nocc = t1a.shape[0]
    out = numpy.ndarray(t2.shape, buffer=out)
    for i in range(nocc):
        out[i] = numpy.einsum('a,jb->jab', t1a[i]*fac, t1b)
        out[i] += t2[i]
    return out

def precontract(a, diag_fac=1, out=None):
    assert(a.flags.c_contiguous)
    assert(a.ndim == 3)
    count, m = a.shape[:2]
    out = numpy.ndarray((count,m*(m+1)//2), buffer=out)
    libcc.CCprecontract(out.ctypes.data_as(ctypes.c_void_p),
                        a.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(count), ctypes.c_int(m),
                        ctypes.c_double(diag_fac))
    return out

