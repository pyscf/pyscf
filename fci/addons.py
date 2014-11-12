#!/usr/bin/env python

import sys
import numpy
import cistring

def large_ci(ci, ncas, nelecas, tol=.1):
    idx = numpy.argwhere(abs(ci) > tol)
    res = []
    for i,j in idx:
        res.append((ci[i,j], \
                    bin(cistring.addr2str(ncas, nelecas/2, i)), \
                    bin(cistring.addr2str(ncas, nelecas/2, j))))
    return res

def initguess_triplet(ncas, nelec, binstring):
    na = cistring.num_strings(ncas, nelec/2)
    addr = cistring.str2addr(ncas, nelec/2, int(binstring,2))
    ci0 = numpy.zeros((na,na))
    ci0[addr,0] = numpy.sqrt(.5)
    ci0[0,addr] =-numpy.sqrt(.5)
    return ci0
