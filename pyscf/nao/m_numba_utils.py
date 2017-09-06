from __future__ import division
import numpy as np
import numba as nb

"""
    numba functions to performs some basics operations
"""

@nb.jit(nopython=True)
def triu_indices_numba(ind, dim):

    count = 0
    for i in range(dim):
        for j in range(i, dim):
            ind[i, j] = count
            count += 1

@nb.jit(nopython=True)
def fill_triu(mat, ind, triu, s1, f1, s2, f2):
    
    for i1 in range(s1,f1):
        for i2 in range(s2,f2):
            if ind[i1, i2] >= 0:
                triu[ind[i1, i2]] = mat[i1-s1,i2-s2]
