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
             aosym='s1', comp=1, cintopt=None, vout=None, ijkoff=0,
             naoi=None, naoj=None, naoaux=None,
             iloc=None, jloc=None, kloc=None):
    assert(aosym in ('s1', 's2ij'))
    c_atm = numpy.asarray(atm, dtype=numpy.int32)
    c_bas = numpy.asarray(bas, dtype=numpy.int32)
    c_env = numpy.asarray(env)
    natm = ctypes.c_int(len(c_atm))
    nbas = ctypes.c_int(len(c_bas))
    i0, ic, j0, jc, k0, kc = basrange
    if 'ssc' in intor:
        if iloc is None: iloc = make_loc(i0, ic, _cgto_spheric(bas))
        if jloc is None: jloc = make_loc(j0, jc, _cgto_spheric(bas))
        if kloc is None: kloc = make_loc(k0, kc, _cgto_cart(bas))
    elif 'cart' in intor:
        if iloc is None: iloc = make_loc(i0, ic, _cgto_cart(bas))
        if jloc is None: jloc = make_loc(j0, jc, _cgto_cart(bas))
        if kloc is None: kloc = make_loc(k0, kc, _cgto_cart(bas))
    else:
        if iloc is None: iloc = make_loc(i0, ic, _cgto_spheric(bas))
        if jloc is None: jloc = make_loc(j0, jc, _cgto_spheric(bas))
        if kloc is None: kloc = make_loc(k0, kc, _cgto_spheric(bas))
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
        ij_count = naoi * (naoi+1) // 2
    if vout is None:
        vout = numpy.empty((ij_count,naoaux))
        ijkoff = 0
    else:
        assert(vout.flags.c_contiguous)

    basrange = numpy.asarray(basrange, numpy.int32)
    fintor = _fpointer(intor)
    if cintopt is None:
        intopt = _vhf.make_cintopt(c_atm, c_bas, c_env, intor)
    else:
        intopt = cintopt
    libri.RInr_3c2e_auxe2_drv(fintor, fill,
                              vout.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_size_t(ijkoff),
                              ctypes.c_int(naoj), ctypes.c_int(naoaux),
                              basrange.ctypes.data_as(ctypes.c_void_p),
                              iloc.ctypes.data_as(ctypes.c_void_p),
                              jloc.ctypes.data_as(ctypes.c_void_p),
                              kloc.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(comp), intopt,
                              c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                              c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                              c_env.ctypes.data_as(ctypes.c_void_p))
    if cintopt is None:
        libri.CINTdel_optimizer(ctypes.byref(intopt))
    return vout

def totcart(bas):
    return ((bas[:,gto.ANG_OF]+1) * (bas[:,gto.ANG_OF]+2)//2 *
            bas[:,gto.NCTR_OF]).sum()
def totspheric(bas):
    return ((bas[:,gto.ANG_OF]*2+1) * bas[:,gto.NCTR_OF]).sum()
def make_loc(shl0, shlc, num_cgto):
    loc = numpy.empty(shlc+1, dtype=numpy.int32)
    off = 0
    for k, i in enumerate(range(shl0, shl0+shlc)):
        loc[k] = off
        off += num_cgto(i)
    loc[shlc] = off
    return loc
def _cgto_spheric(bas):
    return lambda i: (bas[i,gto.ANG_OF]*2+1) * bas[i,gto.NCTR_OF]
def _cgto_cart(bas):
    def fcart(i):
        l = bas[i,gto.ANG_OF]
        return (l+1)*(l+2)//2 * bas[i,gto.NCTR_OF]
    return fcart
