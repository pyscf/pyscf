#!/usr/bin/env python

import ctypes
import numpy
from pyscf import lib

libcc = lib.load_library('libcc')

# t2 + numpy.einsum('ia,jb->ijab', t1a, t1b)
def make_tau(t2, t1a, t1b, fac=1, out=None):
    if out is None:
        out = numpy.empty_like(t2)
    out = numpy.einsum('ia,jb->ijab', t1a*fac, t1b, out=out)
    out += t2
    return out

