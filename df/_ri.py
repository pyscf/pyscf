#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import pyscf.lib
from pyscf import gto

libri = pyscf.lib.load_library('libri')
libcgto = gto.moleintor.libcgto

def nr_auxe2(intor, atm, bas, env, shls_slice, ao_loc,
             aosym='s1', comp=1, cintopt=None, out=None):
    if aosym == 's1':
        atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
        bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
        env = numpy.asarray(env, dtype=numpy.double, order='C')
        natm = atm.shape[0]
        nbas = bas.shape[0]
        i0, i1, j0, j1, k0, k1 = shls_slice
        naoi = ao_loc[i1] - ao_loc[i0];
        naoj = ao_loc[j1] - ao_loc[j0];
        naok = ao_loc[k1] - ao_loc[k0];
        mat = numpy.ndarray((naoi*naoj, naok, comp), numpy.double,
                            buffer=out, order='F')
        if cintopt is None:
            intopt = gto.moleintor.make_cintopt(atm, bas, env, intor)
        else:
            intopt = cintopt
        drv = libcgto.GTOnr3c_drv
        drv(getattr(libri, intor), getattr(libri, 'RInr3c_fill_s1'),
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*6)(*(shls_slice[:6])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), intopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))

        if comp == 1:
            return mat.reshape(-1,naok)
        else:
            return numpy.rollaxis(mat, -1, 0)

    else:
        return gto.moleintor.getints3c(intor, atm, bas, env, shls_slice, comp,
                                       aosym, ao_loc, cintopt, out)

