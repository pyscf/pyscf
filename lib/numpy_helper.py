#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import ctypes
import numpy

_alib = os.path.join(os.path.dirname(__file__), 'libnp_helper.so')
_np_helper = ctypes.CDLL(_alib)

BLOCK_DIM = 120
HERMITIAN = 1
ANTIHERMI = 2

def trace_ab(a, b):
    return (numpy.array(a,copy=False).T*numpy.array(b,copy=False)).sum()

# 2d -> 1d
def pack_tril(mat):
    if not mat.flags.c_contiguous:
        mat = mat.copy(order='C')
    nd = mat.shape[0]
    tril = numpy.empty(nd*(nd+1)/2, mat.dtype)
    if numpy.iscomplexobj(mat):
        fn = _np_helper.NPzpack_tril
    else:
        fn = _np_helper.NPdpack_tril
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), tril.ctypes.data_as(ctypes.c_void_p), \
       mat.ctypes.data_as(ctypes.c_void_p))
    return tril

# 1d -> 2d, write hermitian lower triangle to upper triangle
def unpack_tril(tril):
    nd = int(numpy.sqrt(tril.size*2))
    mat = numpy.empty((nd,nd), tril.dtype)
    if numpy.iscomplexobj(tril):
        fn = _np_helper.NPzunpack_tril
    else:
        fn = _np_helper.NPdunpack_tril
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), tril.ctypes.data_as(ctypes.c_void_p),
       mat.ctypes.data_as(ctypes.c_void_p))
    return mat

# extract a row from a tril-packed matrix
def unpack_row(tril, row_id):
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
    nrblk = (arow-1) / BLOCK_DIM + 1
    ncblk = (acol-1) / BLOCK_DIM + 1
    if inplace:
        assert(arow == acol)
        tmp = numpy.empty((BLOCK_DIM,BLOCK_DIM))
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
        anew = numpy.empty((acol,arow))
        # asigning might be slower than accessing
        for j in range(ncblk):
            c0 = j * BLOCK_DIM
            c1 = c0 + BLOCK_DIM
            if c1 > acol:
                c1 = acol
            for i in range(nrblk):
                r0 = i * BLOCK_DIM
                r1 = r0 + BLOCK_DIM
                if r1 > arow:
                    r1 = arow
                anew[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
        return anew

def transpose_sum(a, inplace=False):
    assert(a.shape[0] == a.shape[1])
    if inplace:
        anew = a
    else:
        anew = numpy.empty_like(a)
    na = a.shape[0]
    nblk = (na-1) / BLOCK_DIM + 1
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

def _np_helper_dot(dotname, a, b, c, alpha=1, beta=0):
    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    fn = getattr(_np_helper, dotname)
    fn.restype = ctypes.c_void_p
    fn(a.ctypes.data_as(ctypes.c_void_p), (ctypes.c_int*a.ndim)(*a.shape), \
       b.ctypes.data_as(ctypes.c_void_p), (ctypes.c_int*b.ndim)(*b.shape), \
       ctypes.c_double(alpha), ctypes.c_double(beta), \
       c.ctypes.data_as(ctypes.c_void_p))
    return c

def dot_aibj_cidj(a, b):
    c = numpy.empty((a.shape[0],a.shape[2],b.shape[0],b.shape[2]))
    return _np_helper_dot('NPdot_aibj_cidj', a, b, c)

def dot_aijb_cijd(a, b):
    c = numpy.empty((a.shape[0],a.shape[3],b.shape[0],b.shape[3]))
    return _np_helper_dot('NPdot_aijb_cijd', a, b, c)

def dot_aibj_cijd(a, b):
    c = numpy.empty((a.shape[0],a.shape[2],b.shape[0],b.shape[3]))
    return _np_helper_dot('NPdot_aibj_cijd', a, b, c)

def dot_aibj_icjd(a, b):
    c = numpy.empty((a.shape[0],a.shape[2],b.shape[1],b.shape[3]))
    return _np_helper_dot('NPdot_aibj_icjd', a, b, c)

def dot_aijb_icjd(a, b):
    c = numpy.empty((a.shape[0],a.shape[3],b.shape[1],b.shape[3]))
    return _np_helper_dot('NPdot_aijb_icjd', a, b, c)


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

    a = numpy.random.random((20,20,20,20))
    b = numpy.random.random((20,20,20,20))
    c1 = numpy.einsum('aibj,cidj->abcd', a, b)
    c2 = dot_aibj_cidj(a,b)
    print(abs(c1 - c2).sum())

    c1 = numpy.einsum('aijb,cijd->abcd', a, b)
    c2 = dot_aijb_cijd(a,b)
    print(abs(c1 - c2).sum())

    c1 = numpy.einsum('aibj,cijd->abcd', a, b)
    c2 = dot_aibj_cijd(a,b)
    print(abs(c1 - c2).sum())

    c1 = numpy.einsum('aibj,icjd->abcd', a, b)
    c2 = dot_aibj_icjd(a,b)
    print(abs(c1 - c2).sum())

    c1 = numpy.einsum('aijb,icjd->abcd', a, b)
    c2 = dot_aijb_icjd(a,b)
    print(abs(c1 - c2).sum())
