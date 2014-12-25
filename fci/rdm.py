#!/usr/bin/env python

import os
import ctypes
import _ctypes
import numpy
import pyscf.lib
from pyscf.fci import cistring

librdm = pyscf.lib.load_library('libmcscf')

def reorder_rdm(rdm1, rdm2, inplace=False):
    nmo = rdm1.shape[0]
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    rdm2 = pyscf.lib.transpose_sum(rdm2.reshape(nmo*nmo,-1), inplace=True) * .5
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

# dm_pq = <|p^+ q|>
def make_rdm1_ms0(fname, cibra, ciket, norb, nelec, link_index=None):
    if isinstance(nelec, int):
        neleca = nelec//2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na,nlink,_ = link_index.shape
    rdm1 = numpy.empty((norb,norb))
    fn = getattr(librdm, fname)
    fn(rdm1.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb),
       ctypes.c_int(na), ctypes.c_int(na),
       ctypes.c_int(nlink), ctypes.c_int(nlink),
       link_index.ctypes.data_as(ctypes.c_void_p),
       link_index.ctypes.data_as(ctypes.c_void_p))
    return rdm1

# fci_rdm.c call dsyrk_, which might have bug on Debian-6
def make_rdm12_ms0(fname, cibra, ciket, norb, nelec, link_index=None):
    if isinstance(nelec, int):
        neleca = nelec//2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
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
    return reorder_rdm(rdm1, rdm2, inplace=True)

def make_rdm1(fname, cibra, ciket, norb, nelec, link_index=None):
    return make_rdm1_ms0(fname, cibra, ciket, norb, nelec, link_index)

def make_rdm12(fname, cibra, ciket, norb, nelec, link_index=None):
    return make_rdm12_ms0(fname, cibra, ciket, norb, nelec, link_index)

###################################################
#
# nelec and link_index are tuples of (alpha,beta)
#
def make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    if isinstance(nelec, int):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    rdm1 = numpy.empty((norb,norb))
    fn = getattr(librdm, fname)
    fn(rdm1.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb),
       ctypes.c_int(na), ctypes.c_int(nb),
       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
       link_indexa.ctypes.data_as(ctypes.c_void_p),
       link_indexb.ctypes.data_as(ctypes.c_void_p))
    return rdm1

# NOTE the rdm2 is calculated as <p^+ q r^+ s>, call reorder_rdm to transform
# to the normal rdm2, which is defined as <p^+ r^+ q s>
def make_rdm12_spin1(fname, cibra, ciket, norb, nelec,
                     link_index=None, symm=0):
    if isinstance(nelec, int):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    rdm1 = numpy.empty((norb,norb))
    rdm2 = numpy.empty((norb,norb,norb,norb))
    fn = _ctypes.dlsym(librdm._handle, fname)
    librdm.FCIrdm12_drv(ctypes.c_void_p(fn),
                        rdm1.ctypes.data_as(ctypes.c_void_p),
                        rdm2.ctypes.data_as(ctypes.c_void_p),
                        cibra.ctypes.data_as(ctypes.c_void_p),
                        ciket.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norb),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(symm))
    rdm2 = rdm2.transpose(1,0,2,3).copy('C')
    return rdm1, rdm2

