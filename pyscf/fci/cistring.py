#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import math
import numpy
from pyscf import lib

libfci = lib.load_library('libfci')

def make_strings(orb_list, nelec):
    '''Generate string from the given orbital list.

    Returns:
        list of int64.  One element represents one string in binary format.
        The binary format takes the convention that the one bit stands for one
        orbital, bit-1 means occupied and bit-0 means unoccupied.  The lowest
        (right-most) bit corresponds to the lowest orbital in the orb_list.

    Examples:

    >>> [bin(x) for x in make_strings((0,1,2,3),2)]
    [0b11, 0b101, 0b110, 0b1001, 0b1010, 0b1100]
    >>> [bin(x) for x in make_strings((3,1,0,2),2)]
    [0b1010, 0b1001, 0b11, 0b1100, 0b110, 0b101]
    '''
    orb_list = list(orb_list)
    if len(orb_list) >= 64:
        return gen_occslst(orb_list, nelec)

    assert (nelec >= 0)
    if nelec == 0:
        return numpy.asarray([0], dtype=numpy.int64)
    elif nelec > len(orb_list):
        return numpy.asarray([], dtype=numpy.int64)
    def gen_str_iter(orb_list, nelec):
        if nelec == 1:
            res = [(1 << i) for i in orb_list]
        elif nelec >= len(orb_list):
            n = 0
            for i in orb_list:
                n = n | (1 << i)
            res = [n]
        else:
            restorb = orb_list[:-1]
            thisorb = 1 << orb_list[-1]
            res = gen_str_iter(restorb, nelec)
            for n in gen_str_iter(restorb, nelec-1):
                res.append(n | thisorb)
        return res
    strings = gen_str_iter(orb_list, nelec)
    assert (strings.__len__() == num_strings(len(orb_list),nelec))
    return numpy.asarray(strings, dtype=numpy.int64)
gen_strings4orblist = make_strings

def gen_occslst(orb_list, nelec):
    '''Generate occupied orbital list for each string.

    Returns:
        List of lists of int32. Each inner list has length equal to the number of
        electrons, and contains the occupied orbitals in the corresponding string.

    Example:

        >>> [bin(x) for x in make_strings((0, 1, 2, 3), 2)]
        ['0b11', '0b101', '0b110', '0b1001', '0b1010', '0b1100']
        >>> gen_occslst((0, 1, 2, 3), 2)
        OIndexList([[0, 1],
                    [0, 2],
                    [1, 2],
                    [0, 3],
                    [1, 3],
                    [2, 3]], dtype=int32)
    '''
    orb_list = list(orb_list)
    assert (nelec >= 0)
    if nelec == 0:
        return numpy.zeros((1,nelec), dtype=numpy.int32)
    elif nelec > len(orb_list):
        return numpy.zeros((0,nelec), dtype=numpy.int32)
    def gen_occs_iter(orb_list, nelec):
        if nelec == 1:
            res = [[i] for i in orb_list]
        elif nelec >= len(orb_list):
            res = [orb_list]
        else:
            restorb = orb_list[:-1]
            thisorb = orb_list[-1]
            res = gen_occs_iter(restorb, nelec)
            for n in gen_occs_iter(restorb, nelec-1):
                res.append(n + [thisorb])
        return res
    occslst = gen_occs_iter(orb_list, nelec)
    return numpy.asarray(occslst, dtype=numpy.int32).view(OIndexList)
# Add this symbol for backward compatibility. Should remove in the future.
_gen_occslst = gen_occslst

def _strs2occslst(strs, norb):
    na = len(strs)
    one_particle_strs = numpy.asarray([1 << i for i in range(norb)])
    occ_masks = (strs.reshape(-1,1) & one_particle_strs) != 0
    occslst = numpy.where(occ_masks)[1].reshape(na,-1)
    return numpy.asarray(occslst, dtype=numpy.int32).view(OIndexList)

def _occslst2strs(occslst):
    assert isinstance(occslst[0], OIndexList)
    occslst = numpy.asarray(occslst)
    na, nelec = occslst.shape
    strs = numpy.zeros(na, dtype=numpy.int64)
    for i in range(nelec):
        strs ^= 1 << occslst[:,i]
    return strs

class OIndexList(numpy.ndarray):
    pass

num_strings = lib.comb

def gen_linkstr_index_o1(orb_list, nelec, strs=None, tril=False):
    '''Look up table, for the strings relationship in terms of a
    creation-annihilating operator pair.

    Similar to gen_linkstr_index. The only difference is the input argument
    strs, which needs to be a list of OIndexList in this function.
    '''
    if nelec == 0:
        return numpy.zeros((0,0,4), dtype=numpy.int32)

    if strs is None:
        strs = gen_occslst(orb_list, nelec)
    occslst = strs

    orb_list = numpy.asarray(orb_list)
    norb = len(orb_list)
    assert (numpy.all(numpy.arange(norb) == orb_list))

    strdic = {tuple(s): i for i,s in enumerate(occslst)}
    nvir = norb - nelec
    def propgate1e(str0):
        addr0 = strdic[tuple(str0)]
        tab = numpy.empty((nelec,4), dtype=numpy.int32)
        tab[:,0] = tab[:,1] = str0
        tab[:,2] = addr0
        tab[:,3] = 1
        linktab = [tab]

        virmask = numpy.ones(norb, dtype=bool)
        virmask[str0] = False
        vir = orb_list[virmask]
        str0 = numpy.asarray(str0)
        # where to put vir-orb, ie how many occ-orb in the left
        where_vir = numpy.sum(str0.reshape(-1,1) < vir, axis=0)
        parity_occ_orb = 1  # parity for annihilating occupied orbital
        for n,i in enumerate(str0):  # loop over all occupied orbitals
            # o,v which index is bigger, to determine whether to annihilate occ-orb first
            reorder_to_ov = vir > i
            str1s = numpy.empty((nvir,nelec), dtype=int)
            str1s[:] = str0
            str1s[:,n] = vir
            str1s.sort(axis=1)
            addr = [strdic[tuple(s)] for s in str1s]
            parity = (where_vir + reorder_to_ov + 1) % 2 #? +1 so that even parity has +1, odd parity = 0
            parity[parity == 0] = -1
            parity *= parity_occ_orb
            tab = numpy.empty((nvir,4), dtype=numpy.int32)
            tab[:,0] = vir
            tab[:,1] = i
            tab[:,2] = addr
            tab[:,3] = parity
            linktab.append(tab)
            parity_occ_orb *= -1
        return numpy.vstack(linktab)

    lidx = [propgate1e(s) for s in occslst]
    lidx = numpy.asarray(lidx, dtype=numpy.int32)
    if tril:
        lidx = reform_linkstr_index(lidx)
    return lidx

# return [cre, des, target_address, parity]
def gen_linkstr_index(orb_list, nocc, strs=None, tril=False):
    '''Look up table, for the strings relationship in terms of a
    creation-annihilating operator pair.

    For given string str0, index[str0] is (nocc+nocc*nvir) x 4 array.
    The first nocc rows [i(:occ),i(:occ),str0,sign] are occupied-occupied
    excitations, which do not change the string. The next nocc*nvir rows
    [a(:vir),i(:occ),str1,sign] are occupied-virtual excitations, starting from
    str0, annihilating i, creating a, to get str1.
    '''
    if strs is None:
        strs = make_strings(orb_list, nocc)

    if isinstance(strs, OIndexList):
        return gen_linkstr_index_o1(orb_list, nocc, strs, tril)

    strs = numpy.array(strs, dtype=numpy.int64)
    assert (all(strs[:-1] < strs[1:]))
    norb = len(orb_list)
    nvir = norb - nocc
    na = strs.shape[0]
    link_index = numpy.empty((na,nocc*nvir+nocc,4), dtype=numpy.int32)
    libfci.FCIlinkstr_index(link_index.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(na),
                            ctypes.c_int(nocc),
                            strs.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(tril))
    return link_index

def reform_linkstr_index(link_index):
    '''Compress the (a, i) pair index in linkstr_index to a lower triangular
    index. The compressed indices can match the 4-fold symmetry of integrals.
    '''
    #for k, tab in enumerate(link_index):
    #    for j, (a, i, str1, sign) in enumerate(tab):
    #        if a > i:
    #            ai = a*(a+1)//2+i
    #        else:
    #            ai = i*(i+1)//2+a
    #        link_new[k,j] = (ai,0,str1,sign)
    link_new = link_index.copy()
    a = link_index[:,:,0]
    i = link_index[:,:,1]
    link_new[:,:,0] = numpy.maximum(a*(a+1)//2+i, i*(i+1)//2+a)
    link_new[:,:,1] = 0
    return link_new

def gen_linkstr_index_trilidx(orb_list, nocc, strs=None):
    r'''Generate linkstr_index with the assumption that :math:`p^+ q|0\rangle`
    where :math:`p > q`.
    So the resultant link_index has the structure ``[pq, *, str1, sign]``.
    It is identical to a call to ``reform_linkstr_index(gen_linkstr_index(...))``.
    '''
    return gen_linkstr_index(orb_list, nocc, strs, True)

def gen_cre_str_index(orb_list, nelec):
    '''linkstr_index to map between N electron string to N+1 electron string.
    It maps the given string to the address of the string which is generated by
    the creation operator.

    For given string str0, index[str0] is nvir x 4 array.  Each entry
    [i(cre),--,str1,sign] means starting from str0, creating i, to get str1.

    Returns [[cre, -, target_address, parity], ...]
    '''
    norb = len(orb_list)
    assert nelec < norb
    strs = make_strings(orb_list, nelec)
    if isinstance(strs, OIndexList):
        raise NotImplementedError('System with 64 orbitals or more')

    strs = numpy.array(strs, dtype=numpy.int64)
    na = strs.shape[0]
    link_index = numpy.empty((len(strs),norb-nelec,4), dtype=numpy.int32)
    libfci.FCIcre_str_index(link_index.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(na),
                            ctypes.c_int(nelec),
                            strs.ctypes.data_as(ctypes.c_void_p))
    return link_index

def gen_des_str_index(orb_list, nelec):
    '''linkstr_index to map between N electron string to N-1 electron string.
    It maps the given string to the address of the string which is generated by
    the annihilation operator.

    For given string str0, index[str0] is nvir x 4 array.  Each entry
    [--,i(des),str1,sign] means starting from str0, annihilating i, to get str1.

    Returns [[-, des, target_address, parity], ...]
    '''
    assert nelec > 0
    strs = make_strings(orb_list, nelec)
    if isinstance(strs, OIndexList):
        raise NotImplementedError('System with 64 orbitals or more')

    strs = numpy.array(strs, dtype=numpy.int64)
    norb = len(orb_list)
    na = strs.shape[0]
    link_index = numpy.empty((len(strs),nelec,4), dtype=numpy.int32)
    libfci.FCIdes_str_index(link_index.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(na),
                            ctypes.c_int(nelec),
                            strs.ctypes.data_as(ctypes.c_void_p))
    return link_index


# Determine the sign of  p^+ q |string0>
def cre_des_sign(p, q, string0):
    if p == q:
        return 1
    else:
        if (string0 & (1 << p)) or (not (string0 & (1 << q))):
            return 0
        elif p > q:
            mask = (1 << p) - (1 << (q+1))
        else:
            mask = (1 << q) - (1 << (p+1))
        return (-1) ** bin(string0 & mask).count('1')

# Determine the sign of  p^+ |string0>
def cre_sign(p, string0):
    if (string0 & (1 << p)):
        return 0
    else:
        return (-1) ** bin(string0 >> (p+1)).count('1')

# Determine the sign of  p |string0>
def des_sign(p, string0):
    if (not (string0 & (1 << p))):
        return 0
    else:
        return (-1) ** bin(string0 >> (p+1)).count('1')

# Determine the sign of  string1 = p^+ q |string0>
def parity(string0, string1):
    #sys.stderr.write('Function cistring.parity is deprecated\n')
    ss = string1 - string0
    def count_bit1(n):
        # see Hamming weight problem and K&R C program
        return bin(n).count('1')
    if ss > 0:
        # string1&ss gives the number of 1s between two strings
        return (-1) ** (count_bit1(string1 & ss))
    elif ss == 0:
        return 1
    else:
        return (-1) ** (count_bit1(string0 & (-ss)))

def addr2str(norb, nelec, addr):
    '''Convert CI determinant address to string'''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')
    max_addr = num_strings(norb, nelec)
    assert max_addr > addr

    if max_addr < 2**31:
        return addrs2str(norb, nelec, [addr])[0]
    else:
        return _addr2str(norb, nelec, addr)

def _addr2str(norb, nelec, addr):
    if addr == 0 or nelec == norb or nelec == 0:
        return (1 << nelec) - 1   # ..0011..11

    for i in reversed(range(norb)):
        addrcum = num_strings(i, nelec)
        if addrcum <= addr:
            return (1 << i) | _addr2str(i, nelec-1, addr-addrcum)

def addrs2str(norb, nelec, addrs):
    '''Convert a list of CI determinant address to string'''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')
    if num_strings(norb, nelec) >= 2**31:
        raise NotImplementedError('Large address')

    addrs = numpy.asarray(addrs, dtype=numpy.int32)
    assert (all(num_strings(norb, nelec) > addrs))
    count = addrs.size
    strs = numpy.empty(count, dtype=numpy.int64)
    libfci.FCIaddrs2str(strs.ctypes.data_as(ctypes.c_void_p),
                        addrs.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(count),
                        ctypes.c_int(norb), ctypes.c_int(nelec))
    return strs

def str2addr(norb, nelec, string):
    '''Convert string to CI determinant address'''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')

    if isinstance(string, str):
        assert (string.count('1') == nelec)
        string = int(string, 2)
    else:
        assert (bin(string).count('1') == nelec)

    if num_strings(norb, nelec) < 2**31:
        libfci.FCIstr2addr.restype = ctypes.c_int
        return libfci.FCIstr2addr(ctypes.c_int(norb), ctypes.c_int(nelec),
                                  ctypes.c_ulonglong(string))
    return _str2addr(norb, nelec, string)

def _str2addr(norb, nelec, string):
    if norb <= nelec or nelec == 0:
        return 0
    addr = 0
    for orbital_id in reversed(range(norb)):
        if (1 << orbital_id) & string:
            addr += num_strings(orbital_id, nelec)
            nelec -= 1
    return addr

def strs2addr(norb, nelec, strings):
    '''Convert a list of string to CI determinant address'''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')
    if num_strings(norb, nelec) >= 2**31:
        raise NotImplementedError('Large address')

    strings = numpy.asarray(strings, dtype=numpy.int64)
    count = strings.size
    addrs = numpy.empty(count, dtype=numpy.int32)
    libfci.FCIstrs2addr(addrs.ctypes.data_as(ctypes.c_void_p),
                        strings.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(count),
                        ctypes.c_int(norb), ctypes.c_int(nelec))
    return addrs

def sub_addrs(norb, nelec, orbital_indices, sub_nelec=0):
    '''The addresses of the determinants which include the specified orbital
    indices. The size of the returned addresses is equal to the number of
    determinants of (norb, nelec) system.
    '''
    assert norb < 64
    if sub_nelec == 0:
        strs = make_strings(orbital_indices, nelec)
        return strs2addr(norb, nelec, strs)
    else:
        strs = make_strings(range(norb), nelec)
        counts = numpy.zeros(len(strs), dtype=int)
        for i in orbital_indices:
            counts += (strs & (1 << i)) != 0
        sub_strs = strs[counts == sub_nelec]
        return strs2addr(norb, nelec, sub_strs)

def tn_strs(norb, nelec, n):
    '''Generate strings for Tn amplitudes.  Eg n=1 (T1) has nvir*nocc strings,
    n=2 (T2) has nvir*(nvir-1)/2 * nocc*(nocc-1)/2 strings.
    '''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')

    if nelec < n or norb-nelec < n:
        return numpy.zeros(0, dtype=int)
    occs_allow = numpy.asarray(make_strings(range(nelec), n)[::-1])
    virs_allow = numpy.asarray(make_strings(range(nelec,norb), n))
    hf_str = int('1'*nelec, 2)
    tns = (hf_str | virs_allow.reshape(-1,1)) ^ occs_allow
    return tns.ravel()

if __name__ == '__main__':
    print([bin(i) for i in make_strings(range(2,5), 2)])
    print(make_strings(range(4), 2))
    #print(gen_linkstr_index(range(6), 3))
#    index = gen_linkstr_index(range(8), 4)
#    idx16 = index[:16]
#    print(idx16[:,:,2])
