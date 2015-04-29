#!/usr/bin/env python

from functools import reduce
import numpy
from pyscf import gto
from pyscf.lib import logger

def mo_map(mol1, mo1, mol2, mo2, base=1, tol=.5):
    s = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    s = reduce(numpy.dot, (mo1.T, s, mo2))
    idx = numpy.argwhere(abs(s) > tol)
    for i,j in idx:
        logger.info(mol1, '<mo-1|mo-2>  %d  %d  %12.8f',
                    i+base, j+base, s[i,j])
    return idx, s

def mo_1to1map(s):
    ''' given <i|j>, [similar-j-to-i for i in <bra|]
    '''
    s1 = abs(s)
    like_input = []
    for i in range(s1.shape[0]):
        k = numpy.argmax(s1[i])
        like_input.append(k)
        s1[:,k] = 0
    return like_input
