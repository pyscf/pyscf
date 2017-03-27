#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import ctypes
import math
import numpy
from pyscf import lib

libfci = lib.load_library('libfci')

def gen_strings4orblist(orb_list, nelec):
    '''Generate string from the given orbital list.

    Returns:
        list of int64.  One int64 element represents one string in binary format.
        The binary format takes the convention that the one bit stands for one
        orbital, bit-1 means occupied and bit-0 means unoccupied.  The lowest
        (right-most) bit corresponds to the lowest orbital in the orb_list.

    Exampels:

    >>> [bin(x) for x in gen_strings4orblist((0,1,2,3),2)]
    [0b11, 0b101, 0b110, 0b1001, 0b1010, 0b1100]
    >>> [bin(x) for x in gen_strings4orblist((3,1,0,2),2)]
    [0b1010, 0b1001, 0b11, 0b1100, 0b110, 0b101]
    '''
    assert(nelec >= 0)
    if nelec == 0:
        return [0]
    elif nelec > len(orb_list):
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
    strings = gen_str_iter(orb_list[::-1], nelec)
    assert(strings.__len__() == num_strings(len(orb_list),nelec))
    return numpy.asarray(strings, dtype=numpy.int64)

def num_strings(n, m):
    if m < 0 or m > n:
        return 0
    else:
        return math.factorial(n) // (math.factorial(n-m)*math.factorial(m))

def gen_linkstr_index_o0(orb_list, nelec, strs=None):
    if strs is None:
        strs = gen_strings4orblist(orb_list, nelec)
    strdic = dict(zip(strs,range(strs.__len__())))
    def propgate1e(str0):
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1<<i):
                occ.append(i)
            else:
                vir.append(i)
        linktab = []
        for i in occ:
            linktab.append((i, i, strdic[str0], 1))
        for i in occ:
            for a in vir:
                str1 = str0 ^ (1<<i) | (1<<a)
                # [cre, des, target_address, parity]
                linktab.append((a, i, strdic[str1], cre_des_sign(a, i, str0)))
        return linktab

    t = [propgate1e(s) for s in strs]
    return numpy.array(t, dtype=numpy.int32)

# return [cre, des, target_address, parity]
def gen_linkstr_index(orb_list, nocc, strs=None, tril=False):
    '''Look up table, for the strings relationship in terms of a
    creation-annihilating operator pair.

    For given string str0, index[str0] is (nocc+nocc*nvir) x 4 array.
    The first nocc rows [i(:occ),i(:occ),str0,sign] are occupied-occupied
    excitations, which do not change the string. The next nocc*nvir rows
    [a(:vir),i(:occ),str1,sign] are occupied-virtual exciations, starting from
    str0, annihilating i, creating a, to get str1.
    '''
    if strs is None:
        strs = gen_strings4orblist(orb_list, nocc)
    strs = numpy.array(strs, dtype=numpy.uint64)
    assert(all(strs[:-1] < strs[1:]))
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
    index, to match the 4-fold symmetry of integrals.
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
    ai = a*(a+1)//2 + i
    ia = i*(i+1)//2 + a
    link_new[:,:,0][a>i ] = ai[a>i ]
    link_new[:,:,0][a<=i] = ia[a<=i]
    link_new[:,:,1] = 0
    return link_new

def gen_linkstr_index_trilidx(orb_list, nocc, strs=None):
    r'''Generate linkstr_index with the assumption that :math:`p^+ q|0\rangle`
    where :math:`p > q`.
    So the resultant link_index has the structure ``[pq, *, str1, sign]``.
    It is identical to a call to ``reform_linkstr_index(gen_linkstr_index(...))``.
    '''
    return gen_linkstr_index(orb_list, nocc, strs, True)

# return [cre, des, target_address, parity]
def gen_cre_str_index_o0(orb_list, nelec):
    cre_strs = gen_strings4orblist(orb_list, nelec+1)
    credic = dict(zip(cre_strs,range(cre_strs.__len__())))
    def progate1e(str0):
        linktab = []
        for i in orb_list:
            if not str0 & (1<<i):
                str1 = str0 | (1<<i)
                linktab.append((i, 0, credic[str1], cre_sign(i, str0)))
        return linktab

    t = [progate1e(s) for s in gen_strings4orblist(orb_list, nelec)]
    return numpy.array(t, dtype=numpy.int32)
def gen_cre_str_index_o1(orb_list, nelec):
    norb = len(orb_list)
    assert(nelec < norb)
    strs = gen_strings4orblist(orb_list, nelec)
    strs = numpy.array(strs, dtype=numpy.int64)
    na = strs.shape[0]
    link_index = numpy.empty((len(strs),norb-nelec,4), dtype=numpy.int32)
    libfci.FCIcre_str_index(link_index.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(na),
                            ctypes.c_int(nelec),
                            strs.ctypes.data_as(ctypes.c_void_p))
    return link_index
def gen_cre_str_index(orb_list, nelec):
    '''linkstr_index to map between N electron string to N+1 electron string.
    It maps the given string to the address of the string which is generated by
    the creation operator.

    For given string str0, index[str0] is nvir x 4 array.  Each entry
    [i(cre),--,str1,sign] means starting from str0, creating i, to get str1.
    '''
    return gen_cre_str_index_o1(orb_list, nelec)

# return [cre, des, target_address, parity]
def gen_des_str_index_o0(orb_list, nelec):
    des_strs = gen_strings4orblist(orb_list, nelec-1)
    desdic = dict(zip(des_strs,range(des_strs.__len__())))
    def progate1e(str0):
        linktab = []
        for i in orb_list:
            if str0 & (1<<i):
                str1 = str0 ^ (1<<i)
                linktab.append((0, i, desdic[str1], des_sign(i, str0)))
        return linktab

    t = [progate1e(s) for s in gen_strings4orblist(orb_list, nelec)]
    return numpy.array(t, dtype=numpy.int32)
def gen_des_str_index_o1(orb_list, nelec):
    assert(nelec > 0)
    strs = gen_strings4orblist(orb_list, nelec)
    strs = numpy.array(strs, dtype=numpy.int64)
    norb = len(orb_list)
    na = strs.shape[0]
    link_index = numpy.empty((len(strs),nelec,4), dtype=numpy.int32)
    libfci.FCIdes_str_index(link_index.ctypes.data_as(ctypes.c_void_p),
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
    '''
    return gen_des_str_index_o1(orb_list, nelec)



# Determine the sign of  p^+ q |string0>
def cre_des_sign(p, q, string0):
    if p == q:
        return 1
    else:
        if (string0 & (1<<p)) or (not (string0 & (1<<q))):
            return 0
        elif p > q:
            mask = (1 << p) - (1 << (q+1))
        else:
            mask = (1 << q) - (1 << (p+1))
        return (-1) ** bin(string0 & mask).count('1')

# Determine the sign of  p^+ |string0>
def cre_sign(p, string0):
    if (string0 & (1<<p)):
        return 0
    else:
        return (-1) ** bin(string0>>(p+1)).count('1')

# Determine the sign of  p |string0>
def des_sign(p, string0):
    if (not (string0 & (1<<p))):
        return 0
    else:
        return (-1) ** bin(string0>>(p+1)).count('1')

# Determine the sign of  string1 = p^+ q |string0>
def parity(string0, string1):
    sys.stderr.write('Function cistring.parity is deprecated\n')
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
    if addr == 0 or nelec == norb or nelec == 0:
        return (1<<nelec) - 1   # ..0011..11
    else:
        for i in reversed(range(norb)):
            addrcum = num_strings(i, nelec)
            if addrcum <= addr:
                return (1<<i) | addr2str_o0(i, nelec-1, addr-addrcum)
def addr2str_o1(norb, nelec, addr):
    assert(num_strings(norb, nelec) > addr)
    if addr == 0 or nelec == norb or nelec == 0:
        return (1<<nelec) - 1   # ..0011..11
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
    '''Convert CI determinant address to string'''
    return addr2str_o1(norb, nelec, addr)

#def str2addr_o0(norb, nelec, string):
#    if norb <= nelec or nelec == 0:
#        return 0
#    elif (1<<(norb-1)) & string:  # remove the first bit
#        return num_strings(norb-1, nelec) \
#                + str2addr_o0(norb-1, nelec-1, string^(1<<(norb-1)))
#    else:
#        return str2addr_o0(norb-1, nelec, string)
#def str2addr_o1(norb, nelec, string):
#    #TODO: assert norb > first-bit-in-string, nelec == num-1-in-string
#    addr = 0
#    nelec_left = nelec
#    for norb_left in reversed(range(norb)):
#        if nelec_left == 0 or norb_left < nelec_left:
#            break
#        elif (1<<norb_left) & string:
#            addr += num_strings(norb_left, nelec_left)
#            nelec_left -= 1
#    return addr
def str2addr(norb, nelec, string):
    '''Convert the string to the CI determinant address'''
    if isinstance(string, str):
        assert(string.count('1') == nelec)
        string = int(string, 2)
    else:
        assert(bin(string).count('1') == nelec)
    libfci.FCIstr2addr.restype = ctypes.c_int
    return libfci.FCIstr2addr(ctypes.c_int(norb), ctypes.c_int(nelec),
                              ctypes.c_ulonglong(string))

def tn_strs(norb, nelec, n):
    '''Generate strings for Tn amplitudes.  Eg n=1 (T1) has nvir*nocc strings,
    n=2 (T2) has nvir*(nvir-1)/2 * nocc*(nocc-1)/2 strings.
    '''
    if nelec < n or norb-nelec < n:
        return numpy.zeros(0, dtype=int)
    occs_allow = numpy.asarray(gen_strings4orblist(range(nelec), n)[::-1])
    virs_allow = numpy.asarray(gen_strings4orblist(range(nelec,norb), n))
    hf_str = int('1'*nelec, 2)
    tns = (hf_str | virs_allow.reshape(-1,1)) ^ occs_allow
    return tns.ravel()

if __name__ == '__main__':
    #print([bin(i) for i in gen_strings4orblist(range(2,5), 2)])
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

    tab1 = gen_cre_str_index_o0(range(8), 4)
    tab2 = gen_cre_str_index_o1(range(8), 4)
    print(abs(tab1 - tab2).sum())
    tab1 = gen_des_str_index_o0(range(8), 4)
    tab2 = gen_des_str_index_o1(range(8), 4)
    print(abs(tab1 - tab2).sum())
