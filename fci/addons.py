#!/usr/bin/env python

import sys
import numpy
import cistring

def large_ci(ci, norb, nelec, tol=.1):
    if isinstance(nelec, int):
        neleca = nelecb = nelec/2
    else:
        neleca, nelecb = nelec
    idx = numpy.argwhere(abs(ci) > tol)
    res = []
    for i,j in idx:
        res.append((ci[i,j], \
                    bin(cistring.addr2str(norb, neleca, i)), \
                    bin(cistring.addr2str(norb, nelecb, j))))
    return res

def initguess_triplet(norb, nelec, binstring):
    if isinstance(nelec, int):
        neleca = nelecb = nelec/2
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    addr = cistring.str2addr(norb, neleca, int(binstring,2))
    ci0 = numpy.zeros((na,nb))
    ci0[addr,0] = numpy.sqrt(.5)
    ci0[0,addr] =-numpy.sqrt(.5)
    return ci0
