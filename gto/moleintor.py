#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import numpy
import ctypes
import _ctypes
import pyscf.lib

_cint = pyscf.lib.load_library('libcvhf')
_cint.CINTcgto_cart.restype = ctypes.c_int
_cint.CINTcgto_spheric.restype = ctypes.c_int
_cint.CINTcgto_spinor.restype = ctypes.c_int

def getints(intor_name, atm, bas, env, bras=None, kets=None, comp=1, hermi=0):
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
    c_atm = atm.ctypes.data_as(pyscf.lib.c_int_p)
    c_bas = bas.ctypes.data_as(pyscf.lib.c_int_p)
    c_env = env.ctypes.data_as(pyscf.lib.c_double_p)
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
    mat = numpy.empty((comp,naoi,naoj), dtype)
    fnaddr = ctypes.c_void_p(_ctypes.dlsym(_cint._handle, intor_name))
    c_intor(fnaddr, mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp), \
            ctypes.c_int(hermi), \
            bralst.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbra),
            ketlst.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nket),
            c_atm, c_natm, c_bas, c_nbas, c_env)

    if comp == 1:
        mat = mat.reshape(naoi,naoj)
    if hermi == 0:
        return mat
    else:
        if comp == 1:
            pyscf.lib.hermi_triu(mat, hermi=hermi)
        else:
            for i in range(comp):
                pyscf.lib.hermi_triu(mat[i], hermi=hermi)
        return mat

def getints_by_shell(intor_name, shls, atm, bas, env, comp=1):
    if not (isinstance(atm, numpy.ndarray) and atm.dtype == numpy.int32):
        atm = numpy.array(atm, dtype=numpy.int32)
    if not (isinstance(bas, numpy.ndarray) and bas.dtype == numpy.int32):
        bas = numpy.array(bas, dtype=numpy.int32)
    if not (isinstance(env, numpy.ndarray) and env.dtype == numpy.double):
        env = numpy.array(env, numpy.double)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    natm = ctypes.c_int(atm.shape[0])
    nbas = ctypes.c_int(bas.shape[0])
    if '_cart' in intor_name:
        dtype = numpy.double
        num_cgto_of = lambda basid: _cint.CINTcgto_cart(ctypes.c_int(basid),
                                                        c_bas)
    elif '_sph' in intor_name:
        dtype = numpy.double
        num_cgto_of = lambda basid: _cint.CINTcgto_spheric(ctypes.c_int(basid),
                                                           c_bas)
    else:
        dtype = numpy.complex
        num_cgto_of = lambda basid: _cint.CINTcgto_spinor(ctypes.c_int(basid),
                                                          c_bas)
    if '3c2e' in intor_name or '2e3c' in intor_name:
        assert(len(shls) == 3)
        di, dj, dk = map(num_cgto_of, shls)
        buf = numpy.empty((di,dj,dk,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        nullopt = ctypes.c_void_p()
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*3)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), nullopt)
        if comp == 1:
            return buf.reshape(di,dj,dk)
        else:
            return buf.transpose(3,0,1,2)
    elif '2c2e' in intor_name or '2e2c' in intor_name:
        assert(len(shls) == 2)
        di, dj = map(num_cgto_of, shls)
        buf = numpy.empty((di,dj,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        nullopt = ctypes.c_void_p()
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), nullopt)
        if comp == 1:
            return buf.reshape(di,dj)
        else:
            return buf.transpose(2,0,1)
    elif '2e' in intor_name:
        assert(len(shls) == 4)
        di, dj, dk, dl = map(num_cgto_of, shls)
        buf = numpy.empty((di,dj,dk,dl,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        nullopt = ctypes.c_void_p()
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*4)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p), nullopt)
        if comp == 1:
            return buf.reshape(di,dj,dk,dl)
        else:
            return buf.transpose(4,0,1,2,3)
    else:
        assert(len(shls) == 2)
        di, dj = map(num_cgto_of, shls)
        buf = numpy.empty((di,dj,comp), dtype, order='F')
        fintor = getattr(_cint, intor_name)
        fintor(buf.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*2)(*shls),
               atm.ctypes.data_as(ctypes.c_void_p), natm,
               bas.ctypes.data_as(ctypes.c_void_p), nbas,
               env.ctypes.data_as(ctypes.c_void_p))
        if comp == 1:
            return buf.reshape(di,dj)
        else:
            return buf.transpose(2,0,1)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        ["H", (0,  0, 0  )],
        ["H", (0,  0, 1  )],
    ])
    mol.basis = {"H": 'cc-pvdz'}
    mol.build()
    mol.set_rinv_orig_(mol.atom_coord(0))
    for i in range(mol.nbas):
        for j in range(mol.nbas):
            print(i, j, getints_by_shell('cint1e_prinvxp_sph', (i,j),
                                         mol._atm, mol._bas, mol._env, 3))
