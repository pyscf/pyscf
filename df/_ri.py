#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import _ctypes
import numpy
import pyscf.lib
from pyscf import gto
from pyscf.scf import _vhf

libri = pyscf.lib.load_library('libri')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libri._handle, name))

def nr_auxe2(intor, basrange, atm, bas, env,
             aosym='s1', comp=1, cintopt=None, out=None, ijkoff=0,
             naoi=None, naoj=None, naoaux=None,
             iloc=None, jloc=None, kloc=None):
    assert(aosym[:2] in ('s1', 's2'))
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(len(atm))
    nbas = ctypes.c_int(len(bas))
    i0, i1, j0, j1, k0, k1 = basrange
    if 'ssc' in intor:
        if iloc is None: iloc = make_loc(i0, i1, bas)
        if jloc is None: jloc = make_loc(j0, j1, bas)
        if kloc is None: kloc = make_loc(k0, k1, bas, True)
    elif 'cart' in intor:
        if iloc is None: iloc = make_loc(i0, i1, bas, True)
        if jloc is None: jloc = make_loc(j0, j1, bas, True)
        if kloc is None: kloc = make_loc(k0, k1, bas, True)
    else:
        if iloc is None: iloc = make_loc(i0, i1, bas)
        if jloc is None: jloc = make_loc(j0, j1, bas)
        if kloc is None: kloc = make_loc(k0, k1, bas)
    if naoi is None:
        naoi = iloc[-1] - iloc[0]
    if naoj is None:
        naoj = jloc[-1] - jloc[0]
    if naoaux is None:
        naoaux = kloc[-1] - kloc[0]

    if aosym in ('s1'):
        fill = _fpointer('RIfill_s1_auxe2')
        ij_count = naoi * naoj
    else:
        fill = _fpointer('RIfill_s2ij_auxe2')
        ij_count = iloc[-1]*(iloc[-1]+1)//2 - iloc[0]*(iloc[0]+1)//2
    if comp == 1:
        shape = (ij_count,naoaux)
    else:
        shape = (comp,ij_count,naoaux)
    if out is None:
        out = numpy.empty(shape)
    else:
        out = numpy.ndarray(shape, buffer=out)

    basrange = numpy.asarray(basrange, numpy.int32)
    fintor = _fpointer(intor)
    if cintopt is None:
        intopt = _vhf.make_cintopt(atm, bas, env, intor)
    else:
        intopt = cintopt
    libri.RInr_3c2e_auxe2_drv(fintor, fill,
                              out.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_size_t(ijkoff),
                              ctypes.c_int(naoj), ctypes.c_int(naoaux),
                              (ctypes.c_int*6)(i0,i1-i0,j0,j1-j0,k0,k1-k0),
                              iloc.ctypes.data_as(ctypes.c_void_p),
                              jloc.ctypes.data_as(ctypes.c_void_p),
                              kloc.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(comp), intopt,
                              atm.ctypes.data_as(ctypes.c_void_p), natm,
                              bas.ctypes.data_as(ctypes.c_void_p), nbas,
                              env.ctypes.data_as(ctypes.c_void_p))
    if cintopt is None:
        libri.CINTdel_optimizer(ctypes.byref(intopt))
    return out

def totcart(bas):
    return ((bas[:,gto.ANG_OF]+1) * (bas[:,gto.ANG_OF]+2)//2 *
            bas[:,gto.NCTR_OF]).sum()
def totspheric(bas):
    return ((bas[:,gto.ANG_OF]*2+1) * bas[:,gto.NCTR_OF]).sum()
def make_loc(shl0, shl1, bas, cart=False):
    l = bas[shl0:shl1,gto.ANG_OF]
    if cart:
        dims = (l+1)*(l+2)//2 * bas[shl0:shl1,gto.NCTR_OF]
    else:
        dims = (l*2+1) * bas[shl0:shl1,gto.NCTR_OF]
    loc = numpy.empty(shl1-shl0+1, dtype=numpy.int32)
    loc[0] = 0
    dims.cumsum(dtype=numpy.int32, out=loc[1:])
    return loc
