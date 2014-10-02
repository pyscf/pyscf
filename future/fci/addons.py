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
