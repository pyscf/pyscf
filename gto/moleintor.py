#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


import os
import numpy
import ctypes
import _ctypes
from pyscf import lib

_alib = os.path.join(os.path.dirname(lib.__file__), 'libcvhf.so')
_cint = ctypes.CDLL(_alib)
_cint.CINTcgto_cart.restype = ctypes.c_int
_cint.CINTcgto_spheric.restype = ctypes.c_int
_cint.CINTcgto_spinor.restype = ctypes.c_int

def getints(intor_name, atm, bas, env, bras=None, kets=None, dim3=1, hermi=0):
    '''non-relativitic and relativitic integral generator.
    hermi=0 : plain
    hermi=1 : hermitian
    hermi=2 : anti-hermitian'''

    nbas = len(bas)
    if bras is None:
        bras = range(nbas)
    if kets is None:
        kets = range(nbas)

    atm = numpy.array(atm, dtype=numpy.int32)
    bas = numpy.array(bas, dtype=numpy.int32)
    env = numpy.array(env, numpy.double)
    c_atm = atm.ctypes.data_as(lib.c_int_p)
    c_bas = bas.ctypes.data_as(lib.c_int_p)
    c_env = env.ctypes.data_as(lib.c_double_p)
    c_natm = ctypes.c_int(atm.shape[0])
    c_nbas = ctypes.c_int(bas.shape[0])
    nbra = len(bras)
    nket = len(kets)

    if '_cart' in intor_name:
        dtype = numpy.double
        num_cgto_of = _cint.CINTcgto_cart
        c_intor = _cint.GTO1eintor_cart
    elif '_sph' in intor_name:
        dtype = numpy.double
        num_cgto_of = _cint.CINTcgto_spheric
        c_intor = _cint.GTO1eintor_sph
    else:
        dtype = numpy.complex
        num_cgto_of = _cint.CINTcgto_spinor
        c_intor = _cint.GTO1eintor_spinor
    naoi = sum([num_cgto_of(ctypes.c_int(i), c_bas) for i in bras])
    naoj = sum([num_cgto_of(ctypes.c_int(i), c_bas) for i in kets])

    bralst = numpy.array(bras, dtype=numpy.int32)
    ketlst = numpy.array(kets, dtype=numpy.int32)
    mat = numpy.empty((dim3,naoi,naoj), dtype)
    fnaddr = ctypes.c_void_p(_ctypes.dlsym(_cint._handle, intor_name))
    c_intor(fnaddr, mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(dim3), \
            ctypes.c_int(hermi), \
            bralst.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbra),
            ketlst.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nket),
            c_atm, c_natm, c_bas, c_nbas, c_env)

    if dim3 == 1:
        mat = mat.reshape(naoi,naoj)
    if hermi == 0:
        return mat
    else:
        if dim3 == 1:
            lib.hermi_triu(mat, hermi=hermi)
        else:
            for i in range(dim3):
                lib.hermi_triu(mat[i], hermi=hermi)
        return mat

