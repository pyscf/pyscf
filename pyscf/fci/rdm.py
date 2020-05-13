#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''FCI 1, 2, 3, 4-particle density matrices.

Note the 1-particle density matrix has the same convention as the mean-field
1-particle density matrix (see McWeeney's book Eq 5.4.20), which is
        dm[p,q] = < q^+ p >
The contraction between 1-particle Hamiltonian and 1-pdm is
        E = einsum('pq,qp', h1, 1pdm)
Different conventions are used in the high order density matrices:
        dm[p,q,r,s,...] = < p^+ r^+ ... s q >
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

librdm = lib.load_library('libfci')

def reorder_rdm(rdm1, rdm2, inplace=False):
    nmo = rdm1.shape[0]
    if not inplace:
        rdm2 = rdm2.copy()
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1.T
    #return rdm1, rdm2
    rdm2 = lib.transpose_sum(rdm2.reshape(nmo*nmo,-1), inplace=True) * .5
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

# dm[p,q] = <|q^+ p|>
def make_rdm1_ms0(fname, cibra, ciket, norb, nelec, link_index=None):
    assert(cibra is not None and ciket is not None)
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na, nlink = link_index.shape[:2]
    assert(cibra.size == na**2)
    assert(ciket.size == na**2)
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
    return rdm1.T

# NOTE rdm1 in this function is calculated as rdm1[p,q] = <q^+ p>;
# rdm2 is calculated as <p^+ q r^+ s>. Call reorder_rdm to transform to the
# normal rdm2, which is  dm2[p,q,r,s] = <p^+ r^+ s q>.
# symm = 1: bra, ket symmetry
# symm = 2: particle permutation symmetry
def make_rdm12_ms0(fname, cibra, ciket, norb, nelec, link_index=None, symm=0):
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    link_index = (link_index, link_index)
    return make_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index, symm)

make_rdm1 = make_rdm1_ms0
make_rdm12 = make_rdm12_ms0

###################################################
#
# nelec and link_index are tuples of (alpha,beta)
#
def make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None):
    assert(cibra is not None and ciket is not None)
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert(cibra.size == na*nb)
    assert(ciket.size == na*nb)
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
    return rdm1.T

# NOTE rdm1 in this function is calculated as rdm1[p,q] = <q^+ p>;
# rdm2 is calculated as <p^+ q r^+ s>. Call reorder_rdm to transform to the
# normal rdm2, which is  dm2[p,q,r,s] = <p^+ r^+ s q>.
# symm = 1: bra, ket symmetry
# symm = 2: particle permutation symmetry
def make_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index=None, symm=0):
    assert(cibra is not None and ciket is not None)
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert(cibra.size == na*nb)
    assert(ciket.size == na*nb)
    rdm1 = numpy.empty((norb,norb))
    rdm2 = numpy.empty((norb,norb,norb,norb))
    librdm.FCIrdm12_drv(getattr(librdm, fname),
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
    return rdm1.T, rdm2


##############################
#
# 3-particle and 4-particle density matrix for RHF-FCI wfn
#
# NOTE the dm3[p,q,r,s,t,u] is calculated as <p^+ q r^+ s t^+ u>
# call reorder_dm123 to transform dm3 to regular 3-pdm
def make_dm123(fname, cibra, ciket, norb, nelec):
    r'''Spin traced 1, 2 and 3-particle density matrices.

    .. note::
        In this function, 2pdm[p,q,r,s] is :math:`\langle p^\dagger q r^\dagger s\rangle`;
        3pdm[p,q,r,s,t,u] is :math:`\langle p^\dagger q r^\dagger s t^\dagger u\rangle`.

        After calling reorder_dm123, the 2pdm and 3pdm are transformed to
        the normal density matrices:
        2pdm[p,r,q,s] = :math:`\langle p^\dagger q^\dagger s r\rangle`
        3pdm[p,s,q,t,r,u] = :math:`\langle p^\dagger q^\dagger r^\dagger u t s\rangle`.
    '''
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert(cibra.size == na*nb)
    assert(ciket.size == na*nb)
    rdm1 = numpy.empty((norb,)*2)
    rdm2 = numpy.empty((norb,)*4)
    rdm3 = numpy.empty((norb,)*6)
    librdm.FCIrdm3_drv(getattr(librdm, fname),
                       rdm1.ctypes.data_as(ctypes.c_void_p),
                       rdm2.ctypes.data_as(ctypes.c_void_p),
                       rdm3.ctypes.data_as(ctypes.c_void_p),
                       cibra.ctypes.data_as(ctypes.c_void_p),
                       ciket.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(norb),
                       ctypes.c_int(na), ctypes.c_int(nb),
                       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                       link_indexa.ctypes.data_as(ctypes.c_void_p),
                       link_indexb.ctypes.data_as(ctypes.c_void_p))
    rdm3 = _complete_dm3_(rdm2, rdm3)
    return rdm1.T, rdm2, rdm3
def _complete_dm3_(dm2, dm3):
# fci_4pdm.c assumed symmetry p >= r >= t for 3-pdm <p^+ q r^+ s t^+ u>
# Using E^r_sE^p_q = E^p_qE^r_s - \delta_{qr}E^p_s + \delta_{ps}E^r_q to
# complete the full 3-pdm
    def transpose01(ijk, i, j, k):
        jik = ijk.transpose(1,0,2)
        jik[:,j] -= dm2[i,:,k,:]
        jik[i,:] += dm2[j,:,k,:]
        dm3[j,:,i,:,k,:] = jik
        return jik
    def transpose12(ijk, i, j, k):
        ikj = ijk.transpose(0,2,1)
        ikj[:,:,k] -= dm2[i,:,j,:]
        ikj[:,j,:] += dm2[i,:,k,:]
        dm3[i,:,k,:,j,:] = ikj
        return ikj

# ijk -> jik -> jki -> kji -> kij -> ikj
    norb = dm2.shape[0]
    for i in range(norb):
        for j in range(i+1):
            for k in range(j+1):
                tmp = transpose01(dm3[i,:,j,:,k,:].copy(), i, j, k)
                tmp = transpose12(tmp, j, i, k)
                tmp = transpose01(tmp, j, k, i)
                tmp = transpose12(tmp, k, j, i)
                tmp = transpose01(tmp, k, i, j)
    return dm3

def make_dm1234(fname, cibra, ciket, norb, nelec):
    r'''Spin traced 1, 2, 3 and 4-particle density matrices.

    .. note::
        In this function, 2pdm[p,q,r,s] is :math:`\langle p^\dagger q r^\dagger s\rangle`;
        3pdm[p,q,r,s,t,u] is :math:`\langle p^\dagger q r^\dagger s t^\dagger u\rangle`;
        4pdm[p,q,r,s,t,u,v,w] is :math:`\langle p^\dagger q r^\dagger s t^\dagger u v^\dagger w\rangle`.

        After calling reorder_dm123, the 2pdm and 3pdm are transformed to
        the normal density matrices:
        2pdm[p,r,q,s] = :math:`\langle p^\dagger q^\dagger s r\rangle`
        3pdm[p,s,q,t,r,u] = :math:`\langle p^\dagger q^\dagger r^\dagger u t s\rangle`.
        4pdm[p,t,q,u,r,v,s,w] = :math:`\langle p^\dagger q^\dagger r^\dagger s^dagger w v u t\rangle`.
    '''
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert(cibra.size == na*nb)
    assert(ciket.size == na*nb)
    rdm1 = numpy.empty((norb,)*2)
    rdm2 = numpy.empty((norb,)*4)
    rdm3 = numpy.empty((norb,)*6)
    rdm4 = numpy.empty((norb,)*8)
    librdm.FCIrdm4_drv(getattr(librdm, fname),
                       rdm1.ctypes.data_as(ctypes.c_void_p),
                       rdm2.ctypes.data_as(ctypes.c_void_p),
                       rdm3.ctypes.data_as(ctypes.c_void_p),
                       rdm4.ctypes.data_as(ctypes.c_void_p),
                       cibra.ctypes.data_as(ctypes.c_void_p),
                       ciket.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(norb),
                       ctypes.c_int(na), ctypes.c_int(nb),
                       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                       link_indexa.ctypes.data_as(ctypes.c_void_p),
                       link_indexb.ctypes.data_as(ctypes.c_void_p))
    rdm3 = _complete_dm3_(rdm2, rdm3)
    rdm4 = _complete_dm4_(rdm3, rdm4)
    return rdm1.T, rdm2, rdm3, rdm4
def _complete_dm4_(dm3, dm4):
# fci_4pdm.c assumed symmetry p >= r >= t >= v for 4-pdm <p^+ q r^+ s t^+ u v^+ w>
# Using E^r_sE^p_q = E^p_qE^r_s - \delta_{qr}E^p_s + \delta_{ps}E^r_q to
# complete the full 4-pdm
    def transpose01(ijkl, i, j, k, l):
        jikl = ijkl.transpose(1,0,2,3)
        jikl[:,j] -= dm3[i,:,k,:,l,:]
        jikl[i,:] += dm3[j,:,k,:,l,:]
        dm4[j,:,i,:,k,:,l,:] = jikl
        return jikl
    def transpose12(ijkl, i, j, k, l):
        ikjl = ijkl.transpose(0,2,1,3)
        ikjl[:,:,k] -= dm3[i,:,j,:,l,:]
        ikjl[:,j,:] += dm3[i,:,k,:,l,:]
        dm4[i,:,k,:,j,:,l,:] = ikjl
        return ikjl
    def transpose23(ijkl, i, j, k, l):
        ijlk = ijkl.transpose(0,1,3,2)
        ijlk[:,:,:,l] -= dm3[i,:,j,:,k,:]
        ijlk[:,:,k,:] += dm3[i,:,j,:,l,:]
        dm4[i,:,j,:,l,:,k,:] = ijlk
        return ijlk
    def chain(ijkl, i, j, k, l):
        tmp = transpose23(ijkl, i, j, k, l)
        tmp = transpose12(tmp, i, j, l, k)
        tmp = transpose23(tmp, i, l, j, k)
        tmp = transpose12(tmp, i, l, k, j)
        tmp = transpose23(tmp, i, k, l, j)
        return tmp

# ijkl -> ijlk -> iljk -> ilkj -> iklj -> ikjl
#      -> jikl -> jilk -> jlik -> jlki -> jkli -> jkil
#(ikjl)-> kijl -> kilj -> klij -> klji -> kjli -> kjil
#(iljk)-> lijk -> likj -> lkij -> lkji -> ljki -> ljik
    norb = dm3.shape[0]
    for i in range(norb):
        for k in range(i+1):
            for j in range(k+1):
                for l in range(j+1):
                    tmp = chain(dm4[i,:,j,:,k,:,l,:].copy(), i, j, k, l)
                    tmp = transpose01(tmp, i, k, j, l)
                    tmp = chain(tmp, k, i, j, l)
                    tmp = transpose01(dm4[i,:,j,:,k,:,l,:].copy(), i, j, k, l)
                    tmp = chain(tmp, j, i, k, l)
                    tmp = transpose01(dm4[i,:,l,:,j,:,k,:].copy(), i, l, j, k)
                    tmp = chain(tmp, l, i, j, k)
    return dm4

def reorder_dm12(rdm1, rdm2, inplace=True):
    return reorder_rdm(rdm1, rdm2, inplace)

# <p^+ q r^+ s t^+ u> => <p^+ r^+ t^+ u s q>
# rdm2[p,q,r,s] is <p^+ q r^+ s>
def reorder_dm123(rdm1, rdm2, rdm3, inplace=True):
    rdm1, rdm2 = reorder_rdm(rdm1, rdm2, inplace)
    if not inplace:
        rdm3 = rdm3.copy()
    norb = rdm1.shape[0]
    for q in range(norb):
        rdm3[:,q,q,:,:,:] -= rdm2
        rdm3[:,:,:,q,q,:] -= rdm2
        rdm3[:,q,:,:,q,:] -= rdm2.transpose(0,2,3,1)
        for s in range(norb):
            rdm3[:,q,q,s,s,:] -= rdm1.T
    return rdm1, rdm2, rdm3


# <p^+ q r^+ s t^+ u w^+ v> => <p^+ r^+ t^+ w^+ v u s q>
# rdm2, rdm3 are the (reordered) standard 2-pdm and 3-pdm
def reorder_dm1234(rdm1, rdm2, rdm3, rdm4, inplace=True):
    rdm1, rdm2, rdm3 = reorder_dm123(rdm1, rdm2, rdm3, inplace)
    if not inplace:
        rdm4 = rdm4.copy()
    norb = rdm1.shape[0]
    for q in range(norb):
        rdm4[:,q,:,:,:,:,q,:] -= rdm3.transpose(0,2,3,4,5,1)
        rdm4[:,:,:,q,:,:,q,:] -= rdm3.transpose(0,1,2,4,5,3)
        rdm4[:,:,:,:,:,q,q,:] -= rdm3
        rdm4[:,q,:,:,q,:,:,:] -= rdm3.transpose(0,2,3,1,4,5)
        rdm4[:,:,:,q,q,:,:,:] -= rdm3
        rdm4[:,q,q,:,:,:,:,:] -= rdm3
        for s in range(norb):
            rdm4[:,q,q,s,:,:,s,:] -= rdm2.transpose(0,2,3,1)
            rdm4[:,q,q,:,:,s,s,:] -= rdm2
            rdm4[:,q,:,:,q,s,s,:] -= rdm2.transpose(0,2,3,1)
            rdm4[:,q,:,s,q,:,s,:] -= rdm2.transpose(0,2,1,3)
            rdm4[:,q,:,s,s,:,q,:] -= rdm2.transpose(0,2,3,1)
            rdm4[:,:,:,s,s,q,q,:] -= rdm2
            rdm4[:,q,q,s,s,:,:,:] -= rdm2
            for u in range(norb):
                rdm4[:,q,q,s,s,u,u,:] -= rdm1.T
    return rdm1, rdm2, rdm3, rdm4

