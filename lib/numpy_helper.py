#!/usr/bin/env python

import numpy

def trace_ab(a, b):
    return (numpy.array(a,copy=False).T*numpy.array(b,copy=False)).sum()

def pack_lowtri(mat, nd):
    mat1d = numpy.empty(nd*(nd+1)/2)
    n = 0
    for i in range(nd):
        for j in range(i+1):
            mat1d[n] = mat[i,j]
            n += 1
    return mat1d
def unpack_lowtri(mat1d, nd):
    mat = numpy.empty((nd,nd))
    n = 0
    for i in range(nd):
        for j in range(i+1):
            mat[i,j] = mat1d[n]
            mat[j,i] = mat1d[n].conj()
            n += 1
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


def transpose(a, blockdim=208):
    arow, acol = a.shape
    nrblk = (arow-1) / blockdim + 1
    ncblk = (acol-1) / blockdim + 1
    at = numpy.empty((acol,arow))
    # asigning might be slower than accessing
    for j in range(ncblk):
        c0 = j * blockdim
        c1 = c0 + blockdim
        if c1 > acol:
            c1 = acol
        for i in range(nrblk):
            r0 = i * blockdim
            r1 = r0 + blockdim
            if r1 > arow:
                r1 = arow
            at[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
    return at

def transpose_sum(a, blockdim=208, inplace=False):
    if inplace:
        anew = a
    else:
        anew = numpy.empty_like(a)
    na = a.shape[0]
    nblk = (na-1) / blockdim + 1
    for i in range(nblk):
        i0 = i*blockdim
        i1 = i0 + blockdim
        if i1 > na:
            i1 = na
        for j in range(i):
            j0 = j*blockdim
            j1 = j0 + blockdim
            tmp = a[i0:i1,j0:j1] + a[j0:j1,i0:i1].T
            anew[i0:i1,j0:j1] = tmp
            anew[j0:j1,i0:i1] = tmp.T
        tmp = a[i0:i1,i0:i1] + a[i0:i1,i0:i1].T
        anew[i0:i1,i0:i1] = tmp
    return anew

if __name__ == '__main__':
    a = numpy.random.random((400,900))
    print abs(a.T - transpose(a)).sum()

    import time
    a = numpy.zeros((13000,10000))
    t0 = time.time()
    at = numpy.empty((10000,13000))
    at[:] = a.T
    t1 = time.time()
    print t1 - t0
    for b in (176, 192, 200, 208, 224, 232, 256):
        t0 = t1
        at = transpose(a, b)
        t1 = time.time()
        print b, t1 - t0

    a = numpy.random.random((400,400))
    b = a+a.T
    c = transpose_sum(a)
    print abs(b-c).sum()
