#
# File: moleintor.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


import os
import numpy
import ctypes

import lib
alib = '../lib/_vhf.so'
_cint = ctypes.cdll.LoadLibrary(alib)
_cint.CINTcgtos_cart.restype = ctypes.c_int
_cint.CINTcgtos_spheric.restype = ctypes.c_int
_cint.CINTcgtos_spinor.restype = ctypes.c_int

def mole_intor(mol, intor_name, dim3, symmetric):
    '''non-relativitic and relativitic integral generator.
    symmetric=1 : hermitian, symmetric=2 : anti-hermitian'''
    if symmetric != 1 and symmetric != 2:
        baslst = range(mol.nbas)
        return intor_cross(mol, intor_name, baslst, baslst, dim3)

    atm = numpy.array(mol._atm, dtype=numpy.int32)
    bas = numpy.array(mol._bas, dtype=numpy.int32)
    env = numpy.array(mol._env, numpy.double)
    c_atm = atm.ctypes.data_as(lib.c_int_p)
    c_bas = bas.ctypes.data_as(lib.c_int_p)
    c_env = env.ctypes.data_as(lib.c_double_p)
    c_natm = ctypes.c_int(atm.shape[0])
    c_nbas = ctypes.c_int(bas.shape[0])

    if '_cart' in intor_name:
        dtype = numpy.double
        num_cgtos_of = lambda i: _cint.CINTcgtos_cart(ctypes.c_int(i), c_bas)
    elif '_sph' in intor_name:
        dtype = numpy.double
        num_cgtos_of = lambda i: _cint.CINTcgtos_spheric(ctypes.c_int(i), c_bas)
    else:
        dtype = numpy.complex
        num_cgtos_of = lambda i: _cint.CINTcgtos_spinor(ctypes.c_int(i), c_bas)

    bas_dim = [num_cgtos_of(i) for i in range(mol.nbas)]
    nbf = sum(bas_dim)
    mat = numpy.empty((dim3,nbf,nbf), dtype)

    c_intor = getattr(_cint, intor_name)
    ip = 0
    for i in range(mol.nbas):
        di = bas_dim[i]
        jp = 0
        for j in range(i+1):
            dj = bas_dim[j]

            c_shls = (ctypes.c_int * 2)(i, j)
            buf = numpy.empty((dim3,dj,di), dtype)
            c_buf = buf.ctypes.data_as(lib.c_double_p)
            c_intor(c_buf, c_shls, c_atm, c_natm, c_bas, c_nbas, c_env)
            mat[:,jp:jp+dj,ip:ip+di] = buf

            jp += dj
        ip += di
    if symmetric == 1:
        for i in range(nbf):
            for j in range(i):
                mat[:,i,j] = mat[:,j,i].conj()
    else:
        for i in range(nbf):
            for j in range(i):
                mat[:,i,j] = -mat[:,j,i].conj()
    if dim3 == 1:
        return mat.reshape(nbf,nbf).transpose()
    else:
        return mat.transpose((0,2,1))

def intor_cross(mol, intor_name, bras, kets, dim3):
    assert(max(bras) < mol.nbas)
    assert(max(kets) < mol.nbas)

    atm = numpy.array(mol._atm, dtype=numpy.int32)
    bas = numpy.array(mol._bas, dtype=numpy.int32)
    env = numpy.array(mol._env, numpy.double)
    c_atm = atm.ctypes.data_as(lib.c_int_p)
    c_bas = bas.ctypes.data_as(lib.c_int_p)
    c_env = env.ctypes.data_as(lib.c_double_p)
    c_natm = ctypes.c_int(atm.shape[0])
    c_nbas = ctypes.c_int(bas.shape[0])

    if '_cart' in intor_name:
        dtype = numpy.double
        num_cgtos_of = lambda i: _cint.CINTcgtos_cart(ctypes.c_int(i), c_bas)
    elif '_sph' in intor_name:
        dtype = numpy.double
        num_cgtos_of = lambda i: _cint.CINTcgtos_spheric(ctypes.c_int(i), c_bas)
    else:
        dtype = numpy.complex
        num_cgtos_of = lambda i: _cint.CINTcgtos_spinor(ctypes.c_int(i), c_bas)

    nbra = reduce(lambda n, i: n + num_cgtos_of(i), bras, 0)
    nket = reduce(lambda n, i: n + num_cgtos_of(i), kets, 0)
    mat = numpy.empty((dim3,nket,nbra), dtype)

    c_intor = getattr(_cint, intor_name)
    ip = 0
    for i in bras:
        di = num_cgtos_of(i)
        jp = 0
        for j in kets:
            dj = num_cgtos_of(j)

            c_shls = (ctypes.c_int * 2)(i, j)
            buf = numpy.empty((dim3,dj,di), dtype)
            c_buf = buf.ctypes.data_as(lib.c_double_p)
            c_intor(c_buf, c_shls, c_atm, c_natm, c_bas, c_nbas, c_env)
            mat[:,jp:jp+dj,ip:ip+di] = buf

            jp += dj
        ip += di
    if dim3 == 1:
        return mat.reshape(nket,nbra).transpose()
    else:
        return mat.transpose((0,2,1))
