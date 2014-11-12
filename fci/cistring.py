#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import ctypes
import math
import numpy
import pyscf.lib

_alib = os.path.join(os.path.dirname(pyscf.lib.__file__), 'libmcscf.so')
libfci = ctypes.CDLL(_alib)

# refer to ci.rdm3.gen_strings
def gen_strings4orblist(orb_list, nelec, ordering=True):
    if nelec == 0:
        return []
    def gen_str_iter(orb_list, nelec):
        if nelec == 1:
            res = [(1<<i) for i in orb_list]
            res.reverse()
        elif nelec >= len(orb_list):
            n = 0
            for i in orb_list:
                n = n | (1<<i)
            res = [n]
        else:
            restorb = orb_list[1:]
            res = gen_str_iter(restorb, nelec)
            for n in gen_str_iter(restorb, nelec-1):
                res.append(n | (1<<orb_list[0]))
        return res
    if ordering:
        orb_list = sorted(orb_list, reverse=True)
    else:
        orb_list = reversed(orb_list)
    strings = gen_str_iter(orb_list, nelec)
    assert(strings.__len__() == num_strings(len(orb_list),nelec))
    return strings

def num_strings(n, m):
    return math.factorial(n) \
            / (math.factorial(n-m)*math.factorial(m))

# Return an mapping-index for each string
# For given string str0, index[str0] is (nocc+nocc*nvir) x 4 array.
# The first nocc rows [i(:occ),i(:occ),str0,sign] are occupied-occupied
# excitations, which do not change the string. The next nocc*nvir rows
# [a(:vir),i(:occ),str1,sign] are occupied-virtual exciations, starting from
# str0, annihilating i, creating a, to get str1.
def gen_linkstr_index_o0(orb_list, nelec, strs=None):
    if strs is None:
        strs = gen_strings4orblist(orb_list, nelec)
    strdic = dict(zip(strs,range(strs.__len__())))
    def pump1e(str0):
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1<<i):
                occ.append(i)
            else:
                vir.append(i)
        pumpmap = []
        for i in occ:
            pumpmap.append((i, i, strdic[str0], 1))
        for i in occ:
            for a in vir:
                str1 = str0 ^ (1<<i) | (1<<a)
                pumpmap.append((a, i, strdic[str1], parity(str0,str1)))
        return pumpmap

    t = [pump1e(s) for s in strs]
    return numpy.array(t, dtype=numpy.int32)

def gen_linkstr_index(orb_list, nocc, strs=None):
    if strs is None:
        strs = gen_strings4orblist(orb_list, nocc)
    strs = numpy.array(strs)
    norb = len(orb_list)
    nvir = norb - nocc
    na = num_strings(norb, nocc)
    link_index = numpy.empty((na,nocc*nvir+nocc,4), dtype=numpy.int32)
    libfci.FCIlinkstr_index(link_index.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(na),
                            ctypes.c_int(nocc),
                            strs.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(0))
    return link_index

# compress the a, i index, to fit the symmetry of integrals
def reform_linkstr_index(link_index):
    link_new = numpy.empty_like(link_index)
    for k, tab in enumerate(link_index):
        for j, (a, i, str1, sign) in enumerate(tab):
            if a > i:
                ai = a*(a+1)/2+i
            else:
                ai = i*(i+1)/2+a
            link_new[k,j] = (ai,str1,sign,0)
    return link_new

# p^+ q|0> where p > q, link_index [pq, *, str1, sign] 
def gen_linkstr_index_trilidx(orb_list, nocc, strs=None):
    if strs is None:
        strs = gen_strings4orblist(orb_list, nocc)
    strs = numpy.array(strs)
    norb = len(orb_list)
    nvir = norb - nocc
    na = num_strings(norb, nocc)
    link_index = numpy.empty((na,nocc*nvir+nocc,4), dtype=numpy.int32)
    libfci.FCIlinkstr_index(link_index.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(na),
                            ctypes.c_int(nocc),
                            strs.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(1))
    return link_index

def parity(string0, string1):
    ss = string1 - string0
    def count_bit1(n):
        # see Hamming weight problem and K&R C program
        return bin(n).count('1')
    if ss > 0:
        # string1&ss gives the number of 1s between two strings
        return (-1) ** (count_bit1(string1&ss))
    elif ss == 0:
        return 1
    else:
        return (-1) ** (count_bit1(string0&(-ss)))

def addr2str_o0(norb, nelec, addr):
    assert(num_strings(norb, nelec) > addr)
    if addr == 0:
        return (1<<nelec) - 1   # ..0011..11
    else:
        for i in reversed(range(norb)):
            addrcum = num_strings(i, nelec)
            if addrcum <= addr:
                return (1<<i) | addr2str_o0(i, nelec-1, addr-addrcum)
def addr2str_o1(norb, nelec, addr):
    assert(num_strings(norb, nelec) > addr)
    str1 = 0
    nelec_left = nelec
    for norb_left in reversed(range(norb)):
        addrcum = num_strings(norb_left, nelec_left)
        if nelec_left == 0:
            break
        elif addr == 0:
            str1 |= (1<<nelec_left) - 1
            break
        elif addrcum <= addr:
            str1 |= 1<<norb_left
            addr -= addrcum
            nelec_left -= 1
    return str1
def addr2str(norb, nelec, addr):
    return addr2str_o1(norb, nelec, addr)

def str2addr_o0(norb, nelec, string):
    if norb <= nelec or nelec == 0:
        return 0
    elif (1<<(norb-1)) & string:  # remove the first bit
        return num_strings(norb-1, nelec) \
                + str2addr_o0(norb-1, nelec-1, string^(1<<(norb-1)))
    else:
        return str2addr_o0(norb-1, nelec, string)
def str2addr_o1(norb, nelec, string):
    #TODO: assert norb > first-bit-in-string, nelec == num-1-in-string
    addr = 0
    nelec_left = nelec
    for norb_left in reversed(range(norb)):
        if nelec_left == 0 or norb_left < nelec_left:
            break
        elif (1<<norb_left) & string:
            addr += num_strings(norb_left, nelec_left)
            nelec_left -= 1
    return addr
def str2addr(norb, nelec, string):
    if isinstance(string, str):
        string = int(string, 2)
    libfci.FCIstr2addr.restype = ctypes.c_int
    return libfci.FCIstr2addr(ctypes.c_int(norb), ctypes.c_int(nelec),
                              ctypes.c_ulong(string))

if __name__ == '__main__':
    #print(gen_strings4orblist(range(4), 2))
    #print(gen_linkstr_index(range(6), 3))
#    index = gen_linkstr_index(range(8), 4)
#    idx16 = index[:16]
#    print(idx16[:,:,2])
    tab1 = gen_linkstr_index_o0(range(8), 4)
    tab2 = gen_linkstr_index(range(8), 4)
    print(abs(tab1 - tab2).sum())

    print(addr2str_o0(6, 3, 7) - addr2str(6, 3, 7))
    print(addr2str_o0(6, 3, 8) - addr2str(6, 3, 8))
    print(addr2str_o0(7, 4, 9) - addr2str(7, 4, 9))

    print(str2addr(6, 3, addr2str(6, 3, 7)) - 7)
    print(str2addr(6, 3, addr2str(6, 3, 8)) - 8)
    print(str2addr(7, 4, addr2str(7, 4, 9)) - 9)
