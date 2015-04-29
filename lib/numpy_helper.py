#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
from pyscf.lib import misc

_np_helper = misc.load_library('libnp_helper')

BLOCK_DIM = 192
HERMITIAN = 1
ANTIHERMI = 2


# 2d -> 1d
def pack_tril(mat):
    mat = numpy.ascontiguousarray(mat)
    nd = mat.shape[0]
    tril = numpy.empty(nd*(nd+1)//2, mat.dtype)
    if numpy.iscomplexobj(mat):
        fn = _np_helper.NPzpack_tril
    else:
        fn = _np_helper.NPdpack_tril
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), tril.ctypes.data_as(ctypes.c_void_p), \
       mat.ctypes.data_as(ctypes.c_void_p))
    return tril

# 1d -> 2d, write hermitian lower triangle to upper triangle
def unpack_tril(tril, filltriu=HERMITIAN):
    tril = numpy.ascontiguousarray(tril)
    nd = int(numpy.sqrt(tril.size*2))
    mat = numpy.empty((nd,nd), tril.dtype)
    if numpy.iscomplexobj(tril):
        fn = _np_helper.NPzunpack_tril
    else:
        fn = _np_helper.NPdunpack_tril
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), tril.ctypes.data_as(ctypes.c_void_p),
       mat.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(filltriu))
    return mat

# extract a row from a tril-packed matrix
def unpack_row(tril, row_id):
    tril = numpy.ascontiguousarray(tril)
    nd = int(numpy.sqrt(tril.size*2))
    mat = numpy.empty(nd, tril.dtype)
    if numpy.iscomplexobj(tril):
        fn = _np_helper.NPzunpack_row
    else:
        fn = _np_helper.NPdunpack_row
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), ctypes.c_int(row_id),
       tril.ctypes.data_as(ctypes.c_void_p),
       mat.ctypes.data_as(ctypes.c_void_p))
    return mat

# for i > j of 2d mat, mat[j,i] = mat[i,j]
def hermi_triu(mat, hermi=HERMITIAN, inplace=True):
    assert(hermi == HERMITIAN or hermi == ANTIHERMI)
    if not mat.flags.c_contiguous or not inplace:
        mat = mat.copy(order='C')
    nd = mat.shape[0]
    if numpy.iscomplexobj(mat):
        fn = _np_helper.NPzhermi_triu
    else:
        fn = _np_helper.NPdsymm_triu
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), mat.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(hermi))
    return mat


LINEAR_DEP_THRESHOLD = 1e-10
def solve_lineq_by_SVD(a, b):
    ''' a * x = b '''
    t, w, vH = numpy.linalg.svd(a)
    idx = []
    for i,wi in enumerate(w):
        if wi > LINEAR_DEP_THRESHOLD:
            idx.append(i)
    if idx:
        idx = numpy.array(idx)
        tb = numpy.dot(numpy.array(t[:,idx]).T.conj(), numpy.array(b))
        x = numpy.dot(numpy.array(vH[idx,:]).T.conj(), tb / w[idx])
    else:
        x = numpy.zeros_like(b)
    return x


def transpose(a, inplace=False):
    arow, acol = a.shape
    if inplace:
        assert(arow == acol)
        #nrblk = (arow-1) // BLOCK_DIM + 1
        ncblk = (acol-1) // BLOCK_DIM + 1
        tmp = numpy.empty((BLOCK_DIM,BLOCK_DIM), a.dtype)
        for j in range(ncblk):
            c0 = j * BLOCK_DIM
            c1 = c0 + BLOCK_DIM
            if c1 > acol:
                c1 = acol
            for i in range(j):
                r0 = i * BLOCK_DIM
                r1 = r0 + BLOCK_DIM
                if r1 > arow:
                    r1 = arow
                tmp[:c1-c0,:r1-r0] = a[c0:c1,r0:r1]
                a[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                a[r0:r1,c0:c1] = tmp[:c1-c0,:r1-r0].T
            # diagonal blocks
            r0 = j * BLOCK_DIM
            r1 = r0 + BLOCK_DIM
            if r1 > arow:
                r1 = arow
            a[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
        return a
    else:
        anew = numpy.empty((acol,arow), a.dtype)
# C code is ~5% faster for acol=arow=10000
# Note: when the input a is a submatrix of another array, cannot call NPd(z)transpose
# since NPd(z)transpose assumes data continuity
        if a.flags.c_contiguous:
            if numpy.iscomplexobj(a):
                fn = _np_helper.NPztranspose
            else:
                fn = _np_helper.NPdtranspose
            fn.restype = ctypes.c_void_p
            fn(ctypes.c_int(arow), ctypes.c_int(acol),
               a.ctypes.data_as(ctypes.c_void_p),
               anew.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(BLOCK_DIM))
        else:
            r1 = c1 = 0
            for c0 in range(0, acol-BLOCK_DIM, BLOCK_DIM):
                c1 = c0 + BLOCK_DIM
                for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                    r1 = r0 + BLOCK_DIM
                    anew[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                anew[c0:c1,r1:arow] = a[r1:arow,c0:c1].T
            for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                r1 = r0 + BLOCK_DIM
                anew[c1:acol,r0:r1] = a[r0:r1,c1:acol].T
            anew[c1:acol,r1:arow] = a[r1:arow,c1:acol].T
        return anew

def transpose_sum(a, inplace=False):
    assert(a.shape[0] == a.shape[1])
    if inplace:
        anew = a
    else:
        anew = numpy.empty_like(a)
    na = a.shape[0]
    nblk = (na-1) // BLOCK_DIM + 1
    for i in range(nblk):
        i0 = i*BLOCK_DIM
        i1 = i0 + BLOCK_DIM
        if i1 > na:
            i1 = na
        for j in range(i):
            j0 = j*BLOCK_DIM
            j1 = j0 + BLOCK_DIM
            tmp = a[i0:i1,j0:j1] + a[j0:j1,i0:i1].T
            anew[i0:i1,j0:j1] = tmp
            anew[j0:j1,i0:i1] = tmp.T
        tmp = a[i0:i1,i0:i1] + a[i0:i1,i0:i1].T
        anew[i0:i1,i0:i1] = tmp
    return anew

# NOTE: NOT assume array a, b to be C-contiguous, since a and b are two
# pointers we want to pass in.
# numpy.dot might not call optimized blas
def dot(a, b, alpha=1, c=None, beta=0):
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = numpy.empty((m,n))
        beta = 0
    else:
        assert(c.flags.c_contiguous)

    return _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha, beta)

def zdot(a, b, alpha=1, c=None, beta=0):
    if numpy.iscomplexobj(a):
        if numpy.iscomplexobj(b):
            k1 = dot(a.real+a.imag, b.real.copy())
            k2 = dot(a.real.copy(), b.imag-b.real)
            k3 = dot(a.imag.copy(), b.real+b.imag)
            if c is None:
                return k1-k3 + (k1+k2)*1j
            else:
                c[:] = c * beta + alpha * (k1-k3 + (k1+k2)*1j)
                return c
        else:
            ar = a.real.copy()
            ai = a.imag.copy()
            cr = dot(ar, b, alpha)
            ci = dot(ai, b, alpha)
            if c is None:
                return cr + ci*1j
            else:
                c[:] = c*beta + (cr+ci*1j)
                return c
    elif numpy.iscomplexobj(b):
        br = b.real.copy()
        bi = b.imag.copy()
        cr = dot(a, br, alpha)
        ci = dot(a, bi, alpha)
        if c is None:
            return cr + ci*1j
        else:
            c[:] = c*beta + (cr+ci*1j)
            return c
    else:
        return dot(a, b, alpha, c, beta)

# a, b, c in C-order
def _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha=1, beta=0,
           offseta=0, offsetb=0, offsetc=0):
    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    assert(c.flags.c_contiguous)
    assert(0 < m < 2147483648)
    assert(0 < n < 2147483648)
    assert(0 < k < 2147483648)

    _np_helper.NPdgemm(ctypes.c_char(trans_b.encode('ascii')),
                       ctypes.c_char(trans_a.encode('ascii')),
                       ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
                       ctypes.c_int(b.shape[1]), ctypes.c_int(a.shape[1]),
                       ctypes.c_int(c.shape[1]),
                       ctypes.c_int(offsetb), ctypes.c_int(offseta),
                       ctypes.c_int(offsetc),
                       b.ctypes.data_as(ctypes.c_void_p),
                       a.ctypes.data_as(ctypes.c_void_p),
                       c.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return c




if __name__ == '__main__':
    a = numpy.random.random((400,900))
    print(abs(a.T - transpose(a)).sum())
    b = a[:400,:400]
    c = numpy.copy(b)
    print(abs(b.T - transpose(c,inplace=True)).sum())

    a = numpy.random.random((400,400))
    b = a + a.T.conj()
    c = transpose_sum(a)
    print(abs(b-c).sum())

    a = a+a*.5j
    for i in range(400):
        a[i,i] = a[i,i].real
    b = a-a.T.conj()
    b = numpy.array((b,b))
    x = hermi_triu(b[0], hermi=2, inplace=0)
    print(abs(b[0]-x).sum())
    x = hermi_triu(b[1], hermi=2, inplace=0)
    print(abs(b[1]-x).sum())

    a = numpy.random.random((400,400))
    b = numpy.random.random((400,400))
    print(abs(dot(a  ,b  )-numpy.dot(a  ,b  )).sum())
    print(abs(dot(a  ,b.T)-numpy.dot(a  ,b.T)).sum())
    print(abs(dot(a.T,b  )-numpy.dot(a.T,b  )).sum())
    print(abs(dot(a.T,b.T)-numpy.dot(a.T,b.T)).sum())

    a = numpy.random.random((400,40))
    b = numpy.random.random((40,400))
    print(abs(dot(a  ,b  )-numpy.dot(a  ,b  )).sum())
    print(abs(dot(b  ,a  )-numpy.dot(b  ,a  )).sum())
    print(abs(dot(a.T,b.T)-numpy.dot(a.T,b.T)).sum())
    print(abs(dot(b.T,a.T)-numpy.dot(b.T,a.T)).sum())
    a = numpy.random.random((400,40))
    b = numpy.random.random((400,40))
    print(abs(dot(a  ,b.T)-numpy.dot(a  ,b.T)).sum())
    print(abs(dot(b  ,a.T)-numpy.dot(b  ,a.T)).sum())
    print(abs(dot(a.T,b  )-numpy.dot(a.T,b  )).sum())
    print(abs(dot(b.T,a  )-numpy.dot(b.T,a  )).sum())

    a = numpy.random.random((400,400))
    b = numpy.random.random((400,400))
    c = numpy.random.random((400,400))
    d = numpy.random.random((400,400))
    print(numpy.allclose(numpy.dot(a+b*1j, c+d*1j), zdot(a+b*1j, c+d*1j)))
