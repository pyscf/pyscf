#!/usr/bin/env python
#
# File: cistring.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import math
import numpy
from pyscf.lib import _mcscf

# refer to ci.rdm3.gen_strings
def gen_strings4orblist(orb_list, nelec, ordering=True):
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

# index[str0] is [a(:vir),i(:occ),str1,sign]
# starting from str0, annihilating i, creating a, to get str1
#def gen_linkstr_index(orb_list, nelec, strs=None):
#    if strs is None:
#        strs = gen_strings4orblist(orb_list, nelec)
#    strdic = dict(zip(strs,range(strs.__len__())))
#    def pump1e(str0):
#        occ = []
#        vir = []
#        for i in orb_list:
#            if str0 & (1<<i):
#                occ.append(i)
#            else:
#                vir.append(i)
#        pumpmap = []
#        for i in occ:
#            pumpmap.append((i, i, strdic[str0], 1))
#        for i in occ:
#            for a in vir:
#                str1 = str0 ^ (1<<i) | (1<<a)
#                pumpmap.append((a, i, strdic[str1], parity(str0,str1)))
#        return pumpmap
#
#    t = [pump1e(s) for s in strs]
#    return numpy.array(t, dtype=numpy.int32)
def gen_linkstr_index(orb_list, nelec, strs=None):
    if strs is None:
        strs = gen_strings4orblist(orb_list, nelec)
    return _mcscf.gen_linkstr_index(orb_list, nelec, strs)

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

