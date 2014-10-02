#!/usr/bin/env python

import os
import ctypes
import numpy
from pyscf import lib
import cistring

_alib = os.path.join(os.path.dirname(lib.__file__), 'libmcscf.so')
librdm = ctypes.CDLL(_alib)

def reorder_rdm(rdm1, rdm2, inplace=False):
    nmo = rdm1.shape[0]
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    rdm2 = lib.transpose_sum(rdm2.reshape(nmo*nmo,-1)) * .5
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

# dm_pq = <|p^+ q|>
def make_rdm1(fname, cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na,nlink,_ = link_index.shape
    rdm1 = numpy.empty((norb,norb))
    fn = getattr(librdm, fname)
    fn(rdm1.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb), ctypes.c_int(na),
       ctypes.c_int(nlink),
       link_index.ctypes.data_as(ctypes.c_void_p))
    return rdm1

def make_rdm12(fname, cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na,nlink,_ = link_index.shape
    rdm1 = numpy.empty((norb,norb))
    rdm2 = numpy.empty((norb,)*4)
    fn = getattr(librdm, fname)
    fn(rdm1.ctypes.data_as(ctypes.c_void_p),
       rdm2.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb), ctypes.c_int(na),
       ctypes.c_int(nlink),
       link_index.ctypes.data_as(ctypes.c_void_p))
    rdm2 = rdm2.transpose(1,0,2,3).copy('C')
    return reorder_rdm(rdm1, rdm2)
